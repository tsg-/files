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
#readonly MODEL="Qwen/Qwen3-32B"
readonly MODEL="meta-llama/Llama-3.1-70B"
readonly PORT=8000
readonly TP_SIZE="${1:-2}"
CHAT_TEMPLATE_PATH="${CHAT_TEMPLATE_PATH:-}"
ROPE_SCALING_ARG=""

# Auto-select chat template for Llama 3.1-family models if user didn't provide one
if [[ -z "${CHAT_TEMPLATE_PATH}" ]]; then
    shopt -s nocasematch
    if [[ "${MODEL}" == *"llama-3.1"* || "${MODEL}" == *"llama3.1"* ]]; then
        CHAT_TEMPLATE_PATH="/root/vllm/examples/tool_chat_template_llama3.1_json.jinja"
    fi
    shopt -u nocasematch
fi

shopt -s nocasematch
if [[ "${MODEL}" == Qwen* ]]; then
    ROPE_SCALING_ARG="        --rope-scaling '{\"rope_type\":\"yarn\",\"factor\":4.0,\"original_max_position_embeddings\":32768}' "
fi
shopt -u nocasematch
#readonly DOC_LENGTHS=(1536 3072 6144 9216 12288 13312)
#readonly DOC_LENGTHS=(9216 12288 13312)
readonly DOC_LENGTHS=(129152)
readonly NUM_DOCUMENTS=1
readonly OUTPUT_LEN=100
readonly REPEAT_COUNT=1
readonly MAX_INFLIGHT=1
readonly BENCHMARK_SCRIPT="${HOME}/LMCache/benchmarks/long_doc_qa/long_doc_qa.py"
readonly AWS_PROFILE="lmcache"
readonly S3_BUCKET="lmcache"
readonly S3_PREFIX="test-g3"

readonly SCENARIO_NAMES=("lmcache_cpu_cold" "lmcache_cpu_hot" "lmcache_s3_warm" "lmcache_s3_cached" "lmcache_s3_nixl_warm" "lmcache_s3_nixl_cached")
readonly SCENARIO_KV_CONFIGS=(
    '{"kv_connector":"LMCacheConnectorV1","kv_role":"kv_both"}'
    '{"kv_connector":"LMCacheConnectorV1","kv_role":"kv_both"}'
    '{"kv_connector":"LMCacheConnectorV1","kv_role":"kv_both"}'
    '{"kv_connector":"LMCacheConnectorV1","kv_role":"kv_both"}'
    '{"kv_connector":"LMCacheConnectorV1","kv_role":"kv_both"}'
    '{"kv_connector":"LMCacheConnectorV1","kv_role":"kv_both"}'
)
readonly SCENARIO_LMCACHE_CONFIGS=(
    "/root/lmcache-cpu.yaml"
    "/root/lmcache-cpu.yaml"
    "/root/lmcache-s3.yaml"
    "/root/lmcache-s3.yaml"
    "/root/lmcache-s3-nixl.yaml"
    "/root/lmcache-s3-nixl.yaml"
)
readonly SCENARIO_RUN_COUNTS=(1 1 1 1 1 1)
readonly SCENARIO_CLEAR_BEFORE=("true" "false" "false" "false" "false" "false")
readonly SCENARIO_START_FLAGS=("true" "false" "true" "true" "true" "true")
readonly SCENARIO_STOP_FLAGS=("false" "true" "true" "true" "true" "true")
readonly SCENARIO_SESSION_LABELS=(
    "lmcache_cpu_session"
    "lmcache_cpu_session"
    "lmcache_s3_warm_session"
    "lmcache_s3_cached_session"
    "lmcache_s3_nixl_warm_session"
    "lmcache_s3_nixl_cached_session"
)

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

build_vllm_command() {
    local context_len=$1
    local kv_transfer_config=$2
    local lmcache_config=$3

    cat <<EOF
export PYTHONPATH=/opt/habanalabs/qual/diag_tool:/opt/habanalabs/qual/diag_tool/automation::/root/vllm:/root/vllm-gaudi:/root/LMCache
export HF_HOME=/root/.cache/huggingface
export HF_HUB_CACHE=/root/.cache/huggingface/hub
export HUGGINGFACE_HUB_CACHE=/root/.cache/huggingface/hub
export HF_TOKEN=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
export HUGGING_FACE_HUB_TOKEN=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
EOF

    if [[ -n "${lmcache_config}" ]]; then
        cat <<EOF
export LMCACHE_LOG_LEVEL=INFO
export LMCACHE_CONFIG_FILE=${lmcache_config}
export LMCACHE_USE_EXPERIMENTAL=True
export PYTHONHASHSEED=67
export AWS_PROFILE='lmcache'
EOF
    else
        cat <<'EOF'
unset LMCACHE_CONFIG_FILE
EOF
    fi

    cat <<'EOF'
export HABANA_VISIBLE_DEVICES=all
export NO_CUDA_EXT=1
export PT_HPU_LAZY_MODE=0
export PT_HPU_GPU_MIGRATION=1
export VLLM_USE_V1=1
export VLLM_SKIP_WARMUP=True
export VLLM_EXPONENTIAL_BUCKETING=False
export VLLM_CONTIGUOUS_PA=1
export VLLM_PROMPT_USE_FUSEDSDPA=1
export VLLM_FSDPA_WITH_GATHERED_CACHE=1
export LOG_LEVEL_PT_STATS=0
export VLLM_DEBUG_FSDPA=0
export PT_HPU_SDPA_QKV_SLICE_MODE_FWD=1
export PT_HPU_QKV_SLICE_SEQ_LEN_THLD=16384
export PT_HPU_SDPA_BR_FACTOR=2048
export PT_HPU_SDPA_BC_FACTOR=2048

EOF

cat <<EOF
vllm serve ${MODEL} \
        --gpu-memory-utilization 0.5 \
        --max-model-len ${context_len} \
        --tensor-parallel-size ${TP_SIZE} \
$( printf "${ROPE_SCALING_ARG}" )\
$( if [[ -n "${kv_transfer_config}" ]]; then
    printf "        --kv-transfer-config '%s' " "${kv_transfer_config}"
fi )$( if [[ -n "${CHAT_TEMPLATE_PATH}" ]]; then
    printf "        --chat-template '%s' " "${CHAT_TEMPLATE_PATH}"
fi )        --no-enable-prefix-caching \
        --max-num-batched-tokens 140000 \
        --max-num-seqs 1 \
        --port ${PORT}
EOF
}

#------------------------------------------------------------------------------
# vLLM lifecycle helpers
#------------------------------------------------------------------------------
clear_s3_cache() {
    local bucket="lmcacheg3"
    local profile="lmcache"

    log_info "Clearing S3 cache prefix s3://${S3_BUCKET}"
    #aws s3 rm "s3://${S3_BUCKET}/${S3_PREFIX}" --recursive --profile "${AWS_PROFILE}" --quiet || true
    #aws s3 rm "s3://${bucket}/"  --recursive --profile "${profile}" --quiet --exclude "*" --include "vllm_*" --dryrun
    aws s3 rb "s3://${bucket}/"  --profile "${profile}" --force
    aws s3 mb "s3://${bucket}/"  --profile "${profile}"
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
    local kv_transfer_config=$3
    local lmcache_config=$4
    local remote_log="/tmp/vllm_${run_label}.log"
    local vllm_cmd
    local attempt=1

    while true; do
        stop_vllm "pre-start cleanup (attempt ${attempt})"
        docker_exec "rm -f '${remote_log}'" || true

        vllm_cmd=$(build_vllm_command "${context_len}" "${kv_transfer_config}" "${lmcache_config}")

        log_info "Starting vLLM with max-model-len=${context_len} (attempt ${attempt})"
        log_info "Waiting for vLLM to finish startup (this may take a few minutes)..."
        docker exec -d "${CONTAINER_NAME}" bash -lc "
remote_log='${remote_log}'
cat <<CMD > /tmp/vllm_launch.sh
${vllm_cmd}
CMD
printf '[INFO] vLLM launch command (max-model-len=${context_len})\n\n' > \"\${remote_log}\"
cat /tmp/vllm_launch.sh >> \"\${remote_log}\"
bash /tmp/vllm_launch.sh >> \"\${remote_log}\" 2>&1
rm -f /tmp/vllm_launch.sh
"

        local elapsed=0
        local device_acquire_error=0
        local startup_complete=0
        while [ "${elapsed}" -lt "${VLLM_STARTUP_TIMEOUT}" ]; do
            if docker_exec "[ -f '${remote_log}' ]" >/dev/null 2>&1; then
                if docker_exec "grep -qi 'Device acquire failed' '${remote_log}'" >/dev/null 2>&1; then
                    device_acquire_error=1
                    break
                fi
                if docker_exec "grep -q 'Application startup complete' '${remote_log}'" >/dev/null 2>&1; then
                    startup_complete=1
                    break
                fi
            fi
            sleep "${HEALTH_CHECK_INTERVAL}"
            elapsed=$((elapsed + HEALTH_CHECK_INTERVAL))
        done

        if [ "${startup_complete}" -eq 1 ]; then
            CURRENT_REMOTE_LOG="${remote_log}"
            CURRENT_SESSION_LABEL="${run_label}"
            break
        fi

        if [ "${device_acquire_error}" -eq 1 ]; then
            log_warn "vLLM startup failed due to device acquire error (attempt ${attempt}); resetting HPU and retrying"
            stop_vllm "device acquire failure cleanup"
            docker_exec "if [[ -x /root/hpu_reset.sh ]]; then /root/hpu_reset.sh; else echo 'hpu_reset.sh missing'; fi" || \
                log_warn "hpu_reset.sh failed during retry"
            if (( attempt >= 3 )); then
                fail "vLLM failed to start after ${attempt} attempts (device acquire failed). See ${remote_log}"
            fi
            ((attempt++))
            sleep 10
            continue
        fi

        log_error "vLLM did not become ready within ${VLLM_STARTUP_TIMEOUT}s; see ${remote_log}"
        fail "vLLM failed to start; see ${remote_log}"
    done
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
CURRENT_REMOTE_LOG=""
CURRENT_SESSION_LABEL=""

run_benchmark_once() {
    local doc_length=$1
    local run_num=$2
    local doc_dir=$3
    local scenario_name=$4
    local kv_transfer_config=$5
    local lmcache_config=$6
    local start_server=${7:-"true"}
    local stop_server=${8:-"true"}
    local session_label=${9:-""}

    local run_label="${scenario_name}_doc${doc_length}_run${run_num}"
    local run_dir="${doc_dir}/run${run_num}"

    mkdir -p "${run_dir}/results"

    log_info "----- Scenario ${scenario_name}: Document length ${doc_length} tokens (run ${run_num}) -----"

    if [[ "${start_server}" == "true" ]]; then
        docker_exec "if [[ -x /root/hpu_reset.sh ]]; then /root/hpu_reset.sh; else echo 'hpu_reset.sh missing'; fi" || \
            log_warn "hpu_reset.sh failed, continuing"
        local session="${session_label:-${scenario_name}_session}"
        CURRENT_SESSION_LABEL="${session}"
        start_vllm 131072 "${session}" "${kv_transfer_config}" "${lmcache_config}"
    fi

    local remote_log="${CURRENT_REMOTE_LOG}"
    if [[ -z "${remote_log}" ]]; then
        fail "Attempting to run benchmark without active vLLM server"
    fi

    local benchmark_log="${run_dir}/benchmark.log"
    local output_file="${run_dir}/results/ttft_${doc_length}.out"
    log_info "Running long_doc_qa benchmark (scenario=${scenario_name}, run=${run_num})..."

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

    if [[ "${stop_server}" == "true" ]]; then
        stop_vllm "post-run session ${CURRENT_SESSION_LABEL}"
        docker_exec "rm -f '${remote_log}'" || true
        CURRENT_REMOTE_LOG=""
        CURRENT_SESSION_LABEL=""
    fi

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
Document length: ${doc_length}, run: ${run_num}, scenario: ${scenario_name}
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

    cat > "${SUMMARY_FILE}" <<EOF
vLLM + LMCache S3 Benchmark
Timestamp: ${TIMESTAMP}
Model: ${MODEL}
TP size: ${TP_SIZE}
Document lengths: ${DOC_LENGTHS[*]}
Output length: ${OUTPUT_LEN}
Num documents: ${NUM_DOCUMENTS}
Max inflight requests: ${MAX_INFLIGHT}
Repeat count: ${REPEAT_COUNT}
Results root: ${RESULTS_ROOT}
Scenarios: ${SCENARIO_NAMES[*]}

EOF

    for idx in "${!SCENARIO_NAMES[@]}"; do
        local scenario_name="${SCENARIO_NAMES[$idx]}"
        local kv_transfer_config="${SCENARIO_KV_CONFIGS[$idx]}"
        local lmcache_config="${SCENARIO_LMCACHE_CONFIGS[$idx]}"
        local run_count="${SCENARIO_RUN_COUNTS[$idx]}"
        local clear_before="${SCENARIO_CLEAR_BEFORE[$idx]}"
        local start_flag="${SCENARIO_START_FLAGS[$idx]}"
        local stop_flag="${SCENARIO_STOP_FLAGS[$idx]}"
        local session_label="${SCENARIO_SESSION_LABELS[$idx]}"

        if [[ "${clear_before}" == "true" ]]; then
            clear_s3_cache
        fi

        printf 'Scenario: %s (kv_transfer=%s, lmcache_config=%s)\n' \
            "${scenario_name}" "${kv_transfer_config:-none}" "${lmcache_config:-none}" >> "${SUMMARY_FILE}"
        printf '\n' >> "${SUMMARY_FILE}"

        for doc_length in "${DOC_LENGTHS[@]}"; do
            local doc_dir="${RESULTS_ROOT}/${scenario_name}/doclen_${doc_length}"
            mkdir -p "${doc_dir}"

            for ((run_id = 1; run_id <= run_count; run_id++)); do
                run_benchmark_once "${doc_length}" "${run_id}" "${doc_dir}" "${scenario_name}" "${kv_transfer_config}" "${lmcache_config}" "${start_flag}" "${stop_flag}" "${session_label}"
                if (( run_id < run_count )); then
                    sleep 5
                fi
            done

            printf '\n' >> "${SUMMARY_FILE}"
        done
    done

    log_success "Benchmark complete. Results written to ${RESULTS_ROOT}"
    log_info "Summary available at ${SUMMARY_FILE}"
}

main "$@"
