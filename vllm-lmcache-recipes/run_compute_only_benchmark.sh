#!/bin/bash
set -euo pipefail  # Exit on error, undefined variables, and pipe failures

#==============================================================================
# vLLM + LMCache Benchmark Automation Script
#
# This script automates benchmarking of vLLM with LMCache across multiple
# context lengths, capturing comprehensive metrics and statistics.
#==============================================================================

#------------------------------------------------------------------------------
# Configuration
#------------------------------------------------------------------------------
CONTAINER_NAME="vdg2"
# Use Hugging Face model ID or local path inside container
# Example HF: "Qwen/Qwen3-32B" or "meta-llama/Llama-3.1-70B-Instruct"
# Example local: "/root/.cache/huggingface/hub/models--meta-llama--Llama-3.1-70B/snapshots/349b2ddb53ce8f2849a6c168a81980ab25258dac"
MODEL="meta-llama/Llama-3.1-70B"
readonly PORT=8000
CONTEXT_LENGTHS=(8192 16384 32768 65536 98304 130900)
OUTPUT_LEN=100
readonly REPEAT_COUNT=2
readonly MAX_INFLIGHT=1
readonly BENCHMARK_SCRIPT="${HOME}/LMCache/benchmarks/long_doc_qa/long_doc_qa.py"

# Timeouts and retries
readonly VLLM_STARTUP_TIMEOUT=600  # seconds (10 minutes)
readonly VLLM_SHUTDOWN_WAIT=5      # seconds
readonly HEALTH_CHECK_INTERVAL=2   # seconds

#------------------------------------------------------------------------------
# Output Setup
#------------------------------------------------------------------------------
readonly TIMESTAMP=$(date +%Y%m%d_%H%M%S)
readonly OUTPUT_DIR="benchmark_results_${TIMESTAMP}"
readonly SUMMARY_FILE="${OUTPUT_DIR}/summary.txt"

mkdir -p "${OUTPUT_DIR}"

#------------------------------------------------------------------------------
# Colors for output
#------------------------------------------------------------------------------
readonly RED='\033[0;31m'
readonly GREEN='\033[0;32m'
readonly YELLOW='\033[1;33m'
readonly BLUE='\033[0;34m'
readonly NC='\033[0m' # No Color

#------------------------------------------------------------------------------
# Logging functions
#------------------------------------------------------------------------------
log_info() {
    echo -e "${BLUE}[INFO]${NC} $*"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $*"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $*"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $*"
}

#------------------------------------------------------------------------------
# Collect hardware information
#------------------------------------------------------------------------------
collect_hardware_info() {
    local hw_info=""

    # Get CPU info
    if command -v lscpu &> /dev/null; then
        local cpu_model=$(lscpu | grep "Model name" | cut -d: -f2 | xargs)
        local cpu_cores=$(lscpu | grep "^CPU(s):" | cut -d: -f2 | xargs)
        hw_info="${hw_info}CPU: ${cpu_model} (${cpu_cores} cores)\n"
    fi

    # Get memory info
    if [ -f /proc/meminfo ]; then
        local total_mem=$(grep MemTotal /proc/meminfo | awk '{printf "%.1f GB", $2/1024/1024}')
        hw_info="${hw_info}RAM: ${total_mem}\n"
    fi

    # Get GPU/accelerator info from container (summarized)
    local gpu_count=$(docker exec "${CONTAINER_NAME}" bash -c '
        if command -v hl-smi &> /dev/null; then
            hl-smi -L 2>/dev/null | grep "AIP (accel" | wc -l
        elif command -v nvidia-smi &> /dev/null; then
            nvidia-smi --list-gpus 2>/dev/null | wc -l
        else
            echo "0"
        fi
    ' 2>/dev/null | tr -d ' ')

    if [ "${gpu_count}" -gt 0 ]; then
        local gpu_name=$(docker exec "${CONTAINER_NAME}" bash -c '
            if command -v hl-smi &> /dev/null; then
                hl-smi -Q name.0 -L 2>/dev/null | grep -oP "HL-\d+\w*" | head -1
            elif command -v nvidia-smi &> /dev/null; then
                nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1
            fi
        ' 2>/dev/null)

        local gpu_mem=$(docker exec "${CONTAINER_NAME}" bash -c '
            if command -v hl-smi &> /dev/null; then
                hl-smi -Q memory.total.0 -L 2>/dev/null | grep -oP "\d+ MB" | head -1
            elif command -v nvidia-smi &> /dev/null; then
                nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null | head -1 | awk "{print \$1\" MB\"}"
            fi
        ' 2>/dev/null)

        hw_info="${hw_info}Accelerator: ${gpu_count}x ${gpu_name} (${gpu_mem} each)\n"
    fi

    # Get OS info
    if [ -f /etc/os-release ]; then
        local os_name=$(grep "^PRETTY_NAME=" /etc/os-release | cut -d= -f2 | tr -d '"')
        hw_info="${hw_info}OS: ${os_name}\n"
    fi

    # Get kernel version
    local kernel=$(uname -r)
    hw_info="${hw_info}Kernel: ${kernel}\n"

    echo -e "${hw_info}"
}

#------------------------------------------------------------------------------
# Initialize summary file
#------------------------------------------------------------------------------
init_summary() {
    cat > "${SUMMARY_FILE}" <<EOF
================================================================================
vLLM + LMCache Benchmark Results
================================================================================
Date: $(date)
Model: ${MODEL}
Port: ${PORT}
Output length: ${OUTPUT_LEN}
Repeat count: ${REPEAT_COUNT}
Max inflight requests: ${MAX_INFLIGHT}
Context lengths: ${CONTEXT_LENGTHS[*]}

Hardware Information:
$(collect_hardware_info)
================================================================================

EOF
}

#------------------------------------------------------------------------------
# Cleanup any existing vLLM instances
#------------------------------------------------------------------------------
cleanup_existing_vllm() {
    log_info "Checking for existing vLLM instances..."

    local vllm_pids=$(docker exec "${CONTAINER_NAME}" pgrep -f "vllm serve" 2>/dev/null || echo "")

    if [ -n "${vllm_pids}" ]; then
        log_warning "Found existing vLLM process(es): ${vllm_pids}"
        log_info "Cleaning up existing vLLM instances..."
        docker exec "${CONTAINER_NAME}" pkill -9 -f "vllm serve" 2>/dev/null || true
        sleep 2
        log_success "Cleanup complete"
    else
        log_info "No existing vLLM instances found"
    fi

    # Kill any multiprocessing workers (leftover from previous vLLM runs)
    log_info "Checking for multiprocessing workers..."
    docker exec "${CONTAINER_NAME}" pkill -9 -f "multiprocessing.spawn" 2>/dev/null || true
    docker exec "${CONTAINER_NAME}" pkill -9 -f "multiprocessing.resource_tracker" 2>/dev/null || true

    # Kill any Python processes to free GPU memory
    log_info "Killing Python processes to free GPU memory..."
    docker exec "${CONTAINER_NAME}" pkill -9 python3 2>/dev/null || true
    sleep 2

    # Reset AMD GPUs if available
    if docker exec "${CONTAINER_NAME}" command -v rocm-smi &>/dev/null; then
        log_info "Resetting AMD GPUs..."
        docker exec "${CONTAINER_NAME}" bash -c 'for i in {0..7}; do rocm-smi --gpureset --device $i 2>/dev/null || true; done' || true
        sleep 1
    fi

    # Clean up any leftover temp log files
    docker exec "${CONTAINER_NAME}" bash -c 'rm -f /tmp/vllm_*.log' 2>/dev/null || true
}

#------------------------------------------------------------------------------
# Check prerequisites
#------------------------------------------------------------------------------
check_prerequisites() {
    log_info "Checking prerequisites..."

    # Check if docker container is running
    if ! docker ps --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
        log_error "Docker container '${CONTAINER_NAME}' is not running"
        exit 1
    fi

    # Check if benchmark script exists
    if [ ! -f "${BENCHMARK_SCRIPT}" ]; then
        log_error "Benchmark script not found: ${BENCHMARK_SCRIPT}"
        exit 1
    fi

    # Check if python3 is available
    if ! command -v python3 &> /dev/null; then
        log_error "python3 is required but not found"
        exit 1
    fi

    # Check if curl is available
    if ! command -v curl &> /dev/null; then
        log_error "curl is required but not found"
        exit 1
    fi

    log_success "Prerequisites check passed"
}

#------------------------------------------------------------------------------
# Wait for vLLM to be fully ready (process running + server started)
#------------------------------------------------------------------------------
wait_for_vllm() {
    local ctx_len=$1
    local max_attempts=$((VLLM_STARTUP_TIMEOUT / HEALTH_CHECK_INTERVAL))
    local attempt=0
    local log_file="/tmp/vllm_${ctx_len}.log"

    log_info "Waiting for vLLM to be ready..."

    while [ $attempt -lt $max_attempts ]; do
        # Check if vLLM process is running with full command line
        local vllm_process=$(docker exec "${CONTAINER_NAME}" pgrep -af "vllm serve" 2>/dev/null)
        if [ -z "${vllm_process}" ]; then
            sleep ${HEALTH_CHECK_INTERVAL}
            attempt=$((attempt + 1))
            echo -n "."
            continue
        fi

        # Process is running, check logs for successful startup
        if docker exec "${CONTAINER_NAME}" test -f "${log_file}" 2>/dev/null; then
            # Look for the application startup complete message
            if docker exec "${CONTAINER_NAME}" bash -c "grep -q 'Application startup complete' '${log_file}'" 2>/dev/null; then
                echo ""
                log_success "vLLM is ready (PID: ${vllm_process%%[[:space:]]*})!"
                return 0
            fi
        fi

        sleep ${HEALTH_CHECK_INTERVAL}
        attempt=$((attempt + 1))
        echo -n "."
    done

    echo ""
    log_error "vLLM failed to start after ${VLLM_STARTUP_TIMEOUT} seconds"

    # Show diagnostics
    if docker exec "${CONTAINER_NAME}" test -f "${log_file}" 2>/dev/null; then
        log_info "Last 20 lines of vLLM log:"
        docker exec "${CONTAINER_NAME}" tail -20 "${log_file}" 2>/dev/null || true

        # Check for specific errors
        if docker exec "${CONTAINER_NAME}" grep -q "ValueError: math domain error" "${log_file}" 2>/dev/null; then
            log_error "Context length may be too large for available memory"
            log_info "Consider reducing max-model-len or increasing GPU memory"
        fi
    fi

    return 1
}

#------------------------------------------------------------------------------
# Determine tensor parallel size based on model
#------------------------------------------------------------------------------
get_tensor_parallel_size() {
    local model=$1

    # Extract model size from name (e.g., "70B", "32B", "8B")
    if [[ "${model}" =~ ([0-9]+)B ]]; then
        local model_size="${BASH_REMATCH[1]}"

        if [ "${model_size}" -ge 70 ]; then
            echo "4"  # 70B+ models need at least 4 GPUs
        elif [ "${model_size}" -ge 30 ]; then
            echo "2"  # 30B+ models need at least 2 GPUs
        else
            echo "1"  # Smaller models can fit on 1 GPU
        fi
    else
        # Default to 1 if we can't determine size
        echo "1"
    fi
}

#------------------------------------------------------------------------------
# Start vLLM server in container
#------------------------------------------------------------------------------
start_vllm() {
    local ctx_len=$1
    local result_dir="${OUTPUT_DIR}/ctx_${ctx_len}"

    mkdir -p "${result_dir}"

    # Determine tensor parallel size based on model
    local tp_size=$(get_tensor_parallel_size "${MODEL}")

    log_info "Starting vLLM with max-model-len: ${ctx_len}, tensor-parallel-size: ${tp_size}"

    # Build rope-scaling argument only for Qwen models
    local rope_scaling=""
    if [[ "${MODEL}" =~ [Qq]wen ]]; then
        rope_scaling='--rope-scaling '"'"'{"rope_type":"yarn","factor":4.0,"original_max_position_embeddings":32768}'"'"
    fi

    # Build chat-template argument for Llama models (required for transformers v4.44+)
    # Using a simple template that works with the OpenAI API format
    local chat_template=""
    if [[ "${MODEL}" =~ [Ll]lama ]]; then
        chat_template='--chat-template '"'"'{{- bos_token }}
{%- for message in messages %}
    {%- if message["role"] == "system" %}
        {{- "<|start_header_id|>system<|end_header_id|>\n\n" + message["content"] + "<|eot_id|>" }}
    {%- elif message["role"] == "user" %}
        {{- "<|start_header_id|>user<|end_header_id|>\n\n" + message["content"] + "<|eot_id|>" }}
    {%- elif message["role"] == "assistant" %}
        {{- "<|start_header_id|>assistant<|end_header_id|>\n\n" + message["content"] + "<|eot_id|>" }}
    {%- endif %}
{%- endfor %}
{%- if add_generation_prompt %}
    {{- "<|start_header_id|>assistant<|end_header_id|>\n\n" }}
{%- endif %}'"'"
    fi

    # Start vLLM in background and redirect logs
    docker exec -d "${CONTAINER_NAME}" bash -c "
        PT_HPU_GPU_MIGRATION=1 VLLM_USE_V1=1 VLLM_SKIP_WARMUP=true PT_HPUGRAPH_DISABLE_TENSOR_CACHE=false \
        vllm serve ${MODEL} \
            ${rope_scaling} \
            ${chat_template} \
            --max-model-len ${ctx_len} \
            --gpu-memory-utilization 0.85 \
            --tensor-parallel-size ${tp_size} \
            --kv-transfer-config '{\"kv_connector\":\"LMCacheConnectorV1\",\"kv_role\":\"kv_both\"}' \
            --port ${PORT} > /tmp/vllm_${ctx_len}.log 2>&1
    "

    wait_for_vllm "${ctx_len}"
}

#------------------------------------------------------------------------------
# Stop vLLM server and capture logs
#------------------------------------------------------------------------------
stop_vllm() {
    local ctx_len=$1
    local result_dir="${OUTPUT_DIR}/ctx_${ctx_len}"

    log_info "Stopping vLLM..."

    # Copy server logs from container
    docker exec "${CONTAINER_NAME}" cat "/tmp/vllm_${ctx_len}.log" > "${result_dir}/vllm_server.log" 2>/dev/null || true

    # Kill vLLM process
    docker exec "${CONTAINER_NAME}" pkill -f "vllm serve" || true
    docker exec "${CONTAINER_NAME}" rm -f "/tmp/vllm_${ctx_len}.log" || true

    sleep ${VLLM_SHUTDOWN_WAIT}
    log_success "vLLM stopped"
}

#------------------------------------------------------------------------------
# Capture Prometheus metrics
#------------------------------------------------------------------------------
capture_metrics() {
    local ctx_len=$1
    local result_dir=$2
    local phase=$3  # "before" or "after"

    local metrics_file="${result_dir}/lmcache_metrics_${phase}.txt"
    local all_metrics_file="${result_dir}/all_metrics_${phase}.txt"

    # Save all metrics for debugging
    curl -sf "http://localhost:${PORT}/metrics" > "${all_metrics_file}" 2>/dev/null || true

    # Extract cache-related metrics (LMCache or vLLM prefix cache)
    grep -E "lmcache|prefix_cache|cache_usage" "${all_metrics_file}" > "${metrics_file}" 2>/dev/null || true

    if [ -s "${metrics_file}" ]; then
        log_success "Captured cache metrics (${phase})"
    elif [ "${phase}" = "after" ]; then
        # Only warn if metrics are missing after the benchmark
        log_warning "No cache metrics available (${phase})"
        log_info "Check ${all_metrics_file} for all available metrics"
    fi
}

#------------------------------------------------------------------------------
# Extract and summarize cache statistics from CSV
#------------------------------------------------------------------------------
extract_csv_stats() {
    local result_dir=$1
    local csv_file="${result_dir}/query_round.csv"

    [ ! -f "${csv_file}" ] && return

    # Count hits and misses (ensure we get numeric values)
    local total_requests
    total_requests=$(tail -n +2 "${csv_file}" 2>/dev/null | wc -l 2>/dev/null | awk '{print $1}' 2>/dev/null)
    # Validate and default to 0 if not a number
    if ! [[ "${total_requests}" =~ ^[0-9]+$ ]]; then
        total_requests=0
    fi

    local cache_misses
    cache_misses=$(tail -n +2 "${csv_file}" 2>/dev/null | grep -c "True" 2>/dev/null || echo "0")
    # Validate and default to 0 if not a number
    if ! [[ "${cache_misses}" =~ ^[0-9]+$ ]]; then
        cache_misses=0
    fi

    # Safe arithmetic - both variables are guaranteed to be numbers now
    local cache_hits=$((total_requests - cache_misses))

    {
        echo "Cache Statistics (from CSV):"
        echo "  Total requests: ${total_requests}"
        echo "  Cache hits: ${cache_hits}"
        echo "  Cache misses: ${cache_misses}"

        if [ "${total_requests}" -gt 0 ]; then
            local hit_rate=$(awk "BEGIN {printf \"%.2f\", (${cache_hits}/${total_requests})*100}")
            echo "  Cache hit rate: ${hit_rate}%"
        fi
    } >> "${SUMMARY_FILE}"

    # Calculate TTFT statistics using Python
    python3 - "${csv_file}" >> "${SUMMARY_FILE}" 2>/dev/null <<'PYTHON' || true
import sys
import pandas as pd

try:
    df = pd.read_csv(sys.argv[1])
    if 'is_miss' in df.columns and 'ttft' in df.columns:
        hits = df[df['is_miss'] == False]
        misses = df[df['is_miss'] == True]

        if not hits.empty:
            print(f"  Avg TTFT (cache hits): {hits['ttft'].mean():.3f}s")
        if not misses.empty:
            print(f"  Avg TTFT (cache misses): {misses['ttft'].mean():.3f}s")
        if not hits.empty and not misses.empty:
            speedup = misses['ttft'].mean() / hits['ttft'].mean()
            print(f"  Cache speedup: {speedup:.2f}x")
except Exception:
    pass
PYTHON
}

#------------------------------------------------------------------------------
# Extract and summarize Prometheus metrics
#------------------------------------------------------------------------------
extract_prometheus_stats() {
    local result_dir=$1
    local metrics_file="${result_dir}/lmcache_metrics_after.txt"

    [ ! -f "${metrics_file}" ] && return

    echo "" >> "${SUMMARY_FILE}"
    echo "Cache Metrics (Prometheus):" >> "${SUMMARY_FILE}"

    python3 - "${metrics_file}" >> "${SUMMARY_FILE}" 2>/dev/null <<'PYTHON' || true
import sys
import re

try:
    with open(sys.argv[1], 'r') as f:
        lines = f.readlines()

    metrics = {}
    for line in lines:
        if line.startswith('#') or not line.strip():
            continue

        match = re.match(r'([a-zA-Z_:][a-zA-Z0-9_:]*(?:\{[^}]*\})?\s+)([0-9.eE+-]+)', line)
        if match:
            metric_name = match.group(1).strip()
            value = float(match.group(2))
            base_name = metric_name.split('{')[0]
            metrics.setdefault(base_name, []).append((metric_name, value))

    # Key metrics to display (both LMCache and vLLM prefix cache)
    key_metrics = [
        ('vllm:lmcache_hit_rate', 'rate'),
        ('vllm:lmcache_retrieve_hit_rate', 'rate'),
        ('vllm:lmcache_num_tokens_hit', 'count'),
        ('vllm:lmcache_num_tokens_req', 'count'),
        ('vllm:gpu_prefix_cache_queries_total', 'count'),
        ('vllm:gpu_prefix_cache_hit_tokens_total', 'count'),
        ('vllm:gpu_cache_usage_perc', 'percent'),
        ('vllm:cpu_cache_usage_perc', 'percent'),
    ]

    for key, metric_type in key_metrics:
        if key in metrics:
            for metric_name, value in metrics[key]:
                display_name = metric_name.replace('vllm:lmcache_', '').replace('vllm:', '').replace('_', ' ').title()
                if metric_type == 'bytes':
                    print(f"  {display_name}: {value / (1024*1024):.2f} MB")
                elif metric_type == 'rate':
                    print(f"  {display_name}: {value * 100:.2f}%")
                elif metric_type == 'percent':
                    print(f"  {display_name}: {value * 100:.2f}%")
                else:
                    print(f"  {display_name}: {value:.0f}")
except Exception as e:
    print(f"  Error parsing metrics: {e}")
PYTHON
}

#------------------------------------------------------------------------------
# Extract server log cache stats
#------------------------------------------------------------------------------
extract_server_log_stats() {
    local result_dir=$1
    local log_file="${result_dir}/vllm_server.log"

    [ ! -f "${log_file}" ] && return

    echo "" >> "${SUMMARY_FILE}"
    echo "vLLM Server Log (Cache-related):" >> "${SUMMARY_FILE}"

    grep -i "cache.*hit\|cache.*miss\|lmcache" "${log_file}" 2>/dev/null | tail -20 >> "${SUMMARY_FILE}" || \
        echo "  No cache-related logs found" >> "${SUMMARY_FILE}"
}

#------------------------------------------------------------------------------
# Run benchmark for a specific context length
#------------------------------------------------------------------------------
run_benchmark() {
    local doc_len=$1
    local ctx_len=$2
    local result_dir="${OUTPUT_DIR}/ctx_${ctx_len}"

    mkdir -p "${result_dir}"

    log_info "Running benchmark with document length: ${doc_len}, context length: ${ctx_len}"

    # Capture metrics before
    capture_metrics "${ctx_len}" "${result_dir}" "before"

    # Run benchmark
    local log_file="${result_dir}/benchmark.log"
    if ! python3 "${BENCHMARK_SCRIPT}" \
        --num-documents 1 \
        --document-length "${doc_len}" \
        --output-len "${OUTPUT_LEN}" \
        --repeat-count "${REPEAT_COUNT}" \
        --repeat-mode random \
        --model "${MODEL}" \
        --port "${PORT}" \
        --max-inflight-requests "${MAX_INFLIGHT}" 2>&1 | tee "${log_file}"; then

        log_error "Benchmark failed for context length ${ctx_len}"
        echo "=== Context Length: ${ctx_len} ===" >> "${SUMMARY_FILE}"
        echo "FAILED" >> "${SUMMARY_FILE}"
        echo "" >> "${SUMMARY_FILE}"
        return 1
    fi

    # Capture metrics after
    capture_metrics "${ctx_len}" "${result_dir}" "after"

    # Move generated files
    for file in warmup_round.csv query_round.csv warmup_round.png query_round.png; do
        [ -f "${file}" ] && mv "${file}" "${result_dir}/"
    done

    log_success "Benchmark completed for context length ${ctx_len}"

    # Generate summary
    {
        echo "=== Context Length: ${ctx_len} ==="
        grep -E "Warmup round mean TTFT|Warmup round time|Query round mean TTFT|Query round time|Query round prompt count|Actual.*gain|Actual latency" "${log_file}" 2>/dev/null || true
    } >> "${SUMMARY_FILE}"

    # Calculate and add throughput metrics
    if [ -f "${log_file}" ]; then
        python3 - "${log_file}" "${doc_len}" "${OUTPUT_LEN}" "${REPEAT_COUNT}" >> "${SUMMARY_FILE}" 2>/dev/null <<'PYTHON' || true
import sys
import re

try:
    log_file = sys.argv[1]
    doc_len = int(sys.argv[2])
    output_len = int(sys.argv[3])
    repeat_count = int(sys.argv[4])

    with open(log_file, 'r') as f:
        content = f.read()

    # Extract times
    warmup_time_match = re.search(r'Warmup round time:\s+([\d.]+)s', content)
    query_time_match = re.search(r'Query round time:\s+([\d.]+)s', content)
    query_count_match = re.search(r'Query round prompt count:\s+(\d+)', content)

    if warmup_time_match and query_time_match and query_count_match:
        warmup_time = float(warmup_time_match.group(1))
        query_time = float(query_time_match.group(1))
        query_count = int(query_count_match.group(1))

        # Calculate throughput
        # Warmup: processing doc_len input tokens + output_len output tokens (once)
        warmup_total_tokens = doc_len + output_len
        warmup_throughput = warmup_total_tokens / warmup_time if warmup_time > 0 else 0

        # Query: processing multiple requests with same token counts
        query_total_tokens = query_count * (doc_len + output_len)
        query_throughput = query_total_tokens / query_time if query_time > 0 else 0

        # Output token generation throughput (decode phase)
        warmup_decode_throughput = output_len / warmup_time if warmup_time > 0 else 0
        query_decode_throughput = (query_count * output_len) / query_time if query_time > 0 else 0

        print("\nThroughput Metrics:")
        print(f"  Warmup total throughput: {warmup_throughput:.2f} tokens/s")
        print(f"  Warmup decode throughput: {warmup_decode_throughput:.2f} tokens/s")
        print(f"  Query total throughput: {query_throughput:.2f} tokens/s")
        print(f"  Query decode throughput: {query_decode_throughput:.2f} tokens/s")
        print(f"  Throughput improvement: {query_throughput / warmup_throughput:.2f}x" if warmup_throughput > 0 else "")

except Exception as e:
    pass
PYTHON
    fi

    extract_csv_stats "${result_dir}"
    extract_prometheus_stats "${result_dir}"
    extract_server_log_stats "${result_dir}"
    echo "" >> "${SUMMARY_FILE}"

    return 0
}

#------------------------------------------------------------------------------
# Main execution
#------------------------------------------------------------------------------
main() {
    echo "========================================="
    echo "vLLM + LMCache Benchmark Automation"
    echo "========================================="
    echo "Model: ${MODEL}"
    echo "Context lengths: ${CONTEXT_LENGTHS[*]}"
    echo "Output directory: ${OUTPUT_DIR}"
    echo "========================================="
    echo ""

    check_prerequisites
    cleanup_existing_vllm
    init_summary

    local total=${#CONTEXT_LENGTHS[@]}
    local current=0

    for ctx_len in "${CONTEXT_LENGTHS[@]}"; do
        current=$((current + 1))
        echo ""
        echo "========================================="
        echo "[$current/$total] Processing context length: ${ctx_len}"
        echo "========================================="

        if ! start_vllm "${ctx_len}"; then
            log_error "Skipping context length ${ctx_len} due to startup failure"
            echo "=== Context Length: ${ctx_len} ===" >> "${SUMMARY_FILE}"
            echo "SKIPPED - vLLM startup failed" >> "${SUMMARY_FILE}"
            echo "" >> "${SUMMARY_FILE}"
            continue
        fi

        # Calculate document length: ctx_len - output_len - prompt_overhead
        # Reserve OUTPUT_LEN tokens for generation and ~100 tokens for prompt formatting
        local doc_len=$((ctx_len - OUTPUT_LEN - 100))

        run_benchmark "${doc_len}" "${ctx_len}"
        stop_vllm "${ctx_len}"
    done

    echo ""
    echo "========================================="
    log_success "All benchmarks completed!"
    echo "Results saved to: ${OUTPUT_DIR}"
    echo "Summary file: ${SUMMARY_FILE}"
    echo "========================================="
    echo ""
    echo "Summary of results:"
    cat "${SUMMARY_FILE}"
}

# Run main function
main "$@"
