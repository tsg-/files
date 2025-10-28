#!/bin/bash
set -euo pipefail

#==============================================================================
# vLLM + LMCache S3 Benchmark Runner
#------------------------------------------------------------------------------
# For each document length in DOC_LENGTHS:
#   1. Start vLLM (clean state) inside the container.
#   2. Run the benchmark once.
#   3. Restart vLLM.
#   4. Run the benchmark again.
#   5. Clear the S3 cache prefix.
# All artefacts (benchmark logs, CSVs, plots, vLLM logs) are stored under a
# dedicated directory per data point.
#==============================================================================

#------------------------------------------------------------------------------
# Configuration
#------------------------------------------------------------------------------
readonly CONTAINER_NAME="hpu-tip"
readonly MODEL="Qwen/Qwen3-32B"
readonly PORT=8000
#readonly DOC_LENGTHS=(1536 3072 6144 9216 12288 13312)
#readonly DOC_LENGTHS=(9216 12288 13312)
readonly DOC_LENGTHS=(8192 16384 32768 65536 98304 129024)
readonly NUM_DOCUMENTS=1
readonly OUTPUT_LEN=100
readonly REPEAT_COUNT=1
readonly MAX_INFLIGHT=1
readonly BENCHMARK_SCRIPT="${HOME}/LMCache/benchmarks/long_doc_qa/long_doc_qa.py"
readonly LMCACHE_CONFIG_FILE="lmcache-s3-nixl.yaml"

readonly AWS_PROFILE="lmcache"
readonly S3_BUCKET="lmcache"
readonly S3_PREFIX="test-g3"

VLLM_STARTUP_TIMEOUT=600    # seconds
readonly HEALTH_CHECK_INTERVAL=2     # seconds

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
readonly RESULTS_ROOT="s3_benchmark_results_${TIMESTAMP}"
readonly SUMMARY_FILE="${RESULTS_ROOT}/summary.txt"

mkdir -p "${RESULTS_ROOT}"

#------------------------------------------------------------------------------
# Logging helpers
#------------------------------------------------------------------------------
log_info()   { printf '[INFO] %s\n' "$*"; }
log_warn()   { printf '[WARN] %s\n' "$*"; }
log_error()  { printf '[ERROR] %s\n' "$*"; }
log_success(){ printf '[OK] %s\n' "$*"; }

#------------------------------------------------------------------------------
# Utility helpers
#------------------------------------------------------------------------------
fail() {
    log_error "$1"
    exit 1
}

docker_exec() {
    docker exec "${CONTAINER_NAME}" bash -lc "$1"
}

ensure_container_running() {
    docker ps --format '{{.Names}}' | grep -qx "${CONTAINER_NAME}" \
        || fail "Container '${CONTAINER_NAME}' is not running"
}

#------------------------------------------------------------------------------
# vLLM lifecycle helpers
#------------------------------------------------------------------------------
clear_s3_cache() {
    local bucket="lmcache"
    local profile="lmcache"

    log_info "Clearing S3 cache prefix s3://${S3_BUCKET}"
    #aws s3 rm "s3://${S3_BUCKET}/${S3_PREFIX}" --recursive --profile "${AWS_PROFILE}" --quiet || true
    aws s3 rm "s3://${bucket}/"  --recursive --profile "${profile}" --quiet --exclude "*" --include "vllm_*" --dryrun
}

stop_vllm() {
    local reason=${1:-""}
    [ -n "${reason}" ] && log_info "Stopping vLLM (${reason})"

    if ! docker_exec "pgrep -f 'vllm serve'" >/dev/null 2>&1; then
        log_info "No vLLM process found in container"
        return
    fi

    docker_exec "pkill -INT -f 'vllm serve'" || true

    local waited=0
    while docker_exec "pgrep -f 'vllm serve'" >/dev/null 2>&1; do
        if [ "${waited}" -ge 30 ]; then
            log_warn "vLLM still running after SIGINT; sending SIGKILL"
            docker_exec "pkill -9 -f 'vllm serve'" || true
            break
        fi
        sleep 1
        waited=$((waited + 1))
    done

    docker_exec "pkill -9 -f 'multiprocessing.spawn'" >/dev/null 2>&1 || true
    docker_exec "pkill -9 -f 'multiprocessing.resource_tracker'" >/dev/null 2>&1 || true
    docker_exec "pkill -9 python3" >/dev/null 2>&1 || true
    docker_exec "pkill -9 python" >/dev/null 2>&1 || true
}

wait_for_vllm() {
    local remote_log=$1
    local elapsed=0

    log_info "Waiting for vLLM to finish startup..."
    while [ "${elapsed}" -lt "${VLLM_STARTUP_TIMEOUT}" ]; do
        if docker_exec "[ -f '${remote_log}' ]" >/dev/null 2>&1; then
            if docker_exec "grep -q 'Application startup complete' '${remote_log}'" >/dev/null 2>&1; then
                log_success "vLLM is ready"
                return 0
            fi
        fi
        sleep "${HEALTH_CHECK_INTERVAL}"
        elapsed=$((elapsed + HEALTH_CHECK_INTERVAL))
    done

    log_error "vLLM did not become ready within ${VLLM_STARTUP_TIMEOUT}s"
    docker_exec "tail -n 50 '${remote_log}'" || true
    return 1
}

start_vllm() {
    local context_len=$1
    local run_label=$2
    local remote_log="/tmp/vllm_${run_label}.log"

    stop_vllm "pre-start cleanup"
    docker_exec "rm -f '${remote_log}'" || true

    log_info "Starting vLLM with max-model-len=${context_len}"
    docker exec -d "${CONTAINER_NAME}" bash -lc " \
PYTHONPATH=$PYTHONPATH:/root/vllm:/root/vllm-gaudi:/root/LMCache \
HABANA_VISIBLE_DEVICES=all \
NO_CUDA_EXT=1 \
PT_HPU_GPU_MIGRATION=1 \
HF_TOKEN=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 \
HUGGING_FACE_HUB_TOKEN=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 \
LMCACHE_CONFIG_FILE=/root/lmcache-s3-nixl.yaml \
LMCACHE_USE_EXPERIMENTAL=True \
PYTHONHASHSEED=67 \
AWS_PROFILE='lmcache' \
VLLM_USE_V1=1 \
VLLM_SKIP_WARMUP=True \
PT_HPU_LAZY_MODE=0 \
PT_HPU_GPU_MIGRATION=1 \
VLLM_USE_V1=1 \
VLLM_SKIP_WARMUP=True \
VLLM_EXPONENTIAL_BUCKETING=False \
VLLM_PROMPT_USE_FUSEDSDPA=1 \
PT_HPU_SDPA_QKV_SLICE_MODE_FWD=1 \
PT_HPU_QKV_SLICE_SEQ_LEN_THLD=131072 \
PT_HPU_SDPA_BR_FACTOR=2048 \
PT_HPU_SDPA_BC_FACTOR=2048 \
vllm serve ${MODEL} \
        --gpu-memory-utilization 0.55 \
        --rope-scaling '{\"rope_type\":\"yarn\",\"factor\":4.0,\"original_max_position_embeddings\":32768}' \
        --max-model-len ${context_len} \
        --kv-transfer-config '{\"kv_connector\":\"LMCacheConnectorV1\",\"kv_role\":\"kv_both\"}' \
        --tensor-parallel-size 1 \
        --port ${PORT} \
        > '${remote_log}' 2>&1"

    wait_for_vllm "${remote_log}"
}

collect_vllm_log() {
    local remote_log=$1
    local destination=$2

    if docker_exec "[ -f '${remote_log}' ]" >/dev/null 2>&1; then
        docker cp "${CONTAINER_NAME}:${remote_log}" "${destination}" >/dev/null 2>&1 || \
            log_warn "Failed to copy vLLM log ${remote_log}"
    else
        log_warn "vLLM log ${remote_log} not found in container"
    fi
}

#------------------------------------------------------------------------------
# Benchmark execution
#------------------------------------------------------------------------------
#python3 LMCache/benchmarks/long_doc_qa/long_doc_qa.py --num-documents 1         --document-length 129024 --output-len 100 --repeat-count 1         --repeat-mode random         --model Qwen/Qwen3-32B         --port 8000         --max-inflight-requests 1
run_benchmark_once() {
    local doc_length=$1
    local run_num=$2
    local doc_dir=$3

    local run_label="doc${doc_length}_run${run_num}"
    local run_dir="${doc_dir}/run${run_num}"
    local remote_log="/tmp/vllm_${run_label}.log"

    mkdir -p "${run_dir}/results"

    log_info "----- Document length ${doc_length} tokens (run ${run_num}) -----"

    start_vllm 131072 "${run_label}"

    local benchmark_log="${run_dir}/benchmark.log"
    local output_file="${run_dir}/results/ttft_${doc_length}.out"

    local success=0
    if python3 "${BENCHMARK_SCRIPT}" \
        --model "${MODEL}" \
        --port "${PORT}" \
        --num-documents "${NUM_DOCUMENTS}" \
        --document-length "${doc_length}" \
        --output-len "${OUTPUT_LEN}" \
        --repeat-count "${REPEAT_COUNT}" \
        --repeat-mode random \
        --max-inflight-requests "${MAX_INFLIGHT}" \
        --output "${output_file}" \
        2>&1 | tee "${benchmark_log}"
    then
        success=1
        log_success "Benchmark completed for doc length ${doc_length}, run ${run_num}"
    else
        log_error "Benchmark failed for doc length ${doc_length}, run ${run_num} (first attempt)"
        # Retry once
        if python3 "${BENCHMARK_SCRIPT}" \
            --model "${MODEL}" \
            --port "${PORT}" \
            --num-documents "${NUM_DOCUMENTS}" \
            --document-length "${doc_length}" \
            --output-len "${OUTPUT_LEN}" \
            --repeat-count "${REPEAT_COUNT}" \
            --repeat-mode random \
            --max-inflight-requests "${MAX_INFLIGHT}" \
            --output "${output_file}" \
            2>&1 | tee -a "${benchmark_log}"
        then
            success=1
            log_success "Benchmark succeeded on retry for doc length ${doc_length}, run ${run_num}"
        else
            log_error "Benchmark retry failed for doc length ${doc_length}, run ${run_num}"
        fi
    fi

    local artefacts=(warmup_round.csv query_round.csv warmup_round.png query_round.png query_round_stats.json warmup_round_stats.json)
    for artefact in "${artefacts[@]}"; do
        if [ -f "${artefact}" ]; then
            mv "${artefact}" "${run_dir}/${artefact}" || log_warn "Failed to move ${artefact}"
        fi
    done

    collect_vllm_log "${remote_log}" "${run_dir}/vllm_server.log"
    stop_vllm "post-run ${run_label}"
    docker_exec "rm -f '${remote_log}'" || true

    local warmup_csv="${run_dir}/warmup_round.csv"
    local warmup_ttft=""
    if [ -f "${warmup_csv}" ]; then
        warmup_ttft=$(python3 - "${warmup_csv}" <<'PY'
import csv, sys
fname = sys.argv[1]
try:
    with open(fname, newline="") as fh:
        reader = csv.DictReader(fh)
        values = [float(row["ttft"]) for row in reader if row.get("ttft") not in (None, "", "nan")]
    if values:
        print(f"{sum(values)/len(values):.3f}", end="")
except Exception:
    pass
PY
)
    fi

    cat >> "${SUMMARY_FILE}" <<EOF
Document length: ${doc_length}, run: ${run_num}
  Results dir: ${run_dir}
  Benchmark status: $([ "${success}" -eq 1 ] && echo "success" || echo "failed")
  Warmup TTFT: ${warmup_ttft:-N/A}s
EOF
}

#------------------------------------------------------------------------------
# Main driver
#------------------------------------------------------------------------------
main() {
    ensure_container_running
    clear_s3_cache

    cat > "${SUMMARY_FILE}" <<EOF
vLLM + LMCache S3 Benchmark
Timestamp: ${TIMESTAMP}
Model: ${MODEL}
Document lengths: ${DOC_LENGTHS[*]}
Output length: ${OUTPUT_LEN}
Num documents: ${NUM_DOCUMENTS}
Max inflight requests: ${MAX_INFLIGHT}
Repeat count: ${REPEAT_COUNT}
Results root: ${RESULTS_ROOT}

EOF

    for doc_length in "${DOC_LENGTHS[@]}"; do
        local doc_dir="${RESULTS_ROOT}/doclen_${doc_length}"
        mkdir -p "${doc_dir}"

        run_benchmark_once "${doc_length}" 1 "${doc_dir}"
        sleep 5
        run_benchmark_once "${doc_length}" 2 "${doc_dir}"

        clear_s3_cache
        printf '\n' >> "${SUMMARY_FILE}"
    done

    log_success "Benchmark complete. Results written to ${RESULTS_ROOT}"
    log_info "Summary available at ${SUMMARY_FILE}"
}

main "$@"
