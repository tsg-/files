# LMSYS-Chat-1M Benchmark Flow for vLLM + LMCache

This directory contains tools for benchmarking vLLM with LMCache using the LMSYS-Chat-1M dataset, which contains real-world multi-turn conversations.

## Overview

The LMSYS-Chat-1M dataset provides 1 million real conversations from Chatbot Arena, making it ideal for:
- Testing KV cache effectiveness in multi-turn scenarios
- Measuring prefix caching benefits with real user interaction patterns
- Benchmarking LMCache with realistic workloads

## Files

- **`lmsys_chat_replay.py`** - Main benchmark script that replays conversations to vLLM
- **`analyze_results.py`** - Generate visualizations and HTML reports from benchmark results
- **`run_lmsys_benchmark.sh`** - End-to-end automation script
- **`requirements_lmsys.txt`** - Python dependencies

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements_lmsys.txt
```

### 2. Authenticate with HuggingFace

The LMSYS-Chat-1M dataset requires accepting a license agreement:

1. Visit: https://huggingface.co/datasets/lmsys/lmsys-chat-1m
2. Click "Agree and access repository"
3. Login with HuggingFace CLI:

```bash
huggingface-cli login
```

### 3. Start vLLM with LMCache

```bash
export MODEL_PATH="meta-llama/Meta-Llama-3.1-8B-Instruct"

vllm serve $MODEL_PATH \
  --port 8000 \
  --enable-prefix-caching \
  --kv-transfer-config '{"kv_connector":"LMCacheConnectorV1", "kv_role":"kv_both"}' \
  --disable-log-requests \
  --served-model-name default
```

### 4. Run Benchmark

```bash
python lmsys_chat_replay.py \
  --vllm-endpoint http://localhost:8000/v1 \
  --num-conversations 100 \
  --min-turns 3 \
  --output results.json
```

### 5. Analyze Results

```bash
python analyze_results.py results.json \
  --output-dir ./analysis \
  --html-report report.html
```

Open `report.html` in your browser to view the interactive report.

## Automated End-to-End Flow

Run the complete benchmark automatically:

```bash
chmod +x run_lmsys_benchmark.sh
./run_lmsys_benchmark.sh
```

This will:
1. Install dependencies
2. Check HuggingFace authentication
3. Guide you through starting vLLM
4. Run the benchmark
5. Generate analysis reports

## Configuration Options

### Benchmark Parameters

```bash
python lmsys_chat_replay.py \
  --vllm-endpoint http://localhost:8000/v1 \    # vLLM API URL
  --model-name default \                         # Model name for requests
  --num-conversations 100 \                      # Number of conversations to test
  --min-turns 3 \                                # Minimum turns per conversation
  --max-turns 10 \                               # Maximum turns (optional)
  --max-concurrent 5 \                           # Concurrent conversations
  --output results.json                          # Output file
```

### Filter by Conversation Characteristics

The script automatically filters for:
- English language conversations
- Multi-turn dialogues (configurable minimum)
- Conversations with sufficient context for KV cache benefits

## Understanding the Results

### Key Metrics

1. **Cache Hit Rate**: Percentage of tokens loaded from KV cache vs recomputed
   - Higher is better
   - Should increase with conversation length

2. **TTFT (Time to First Token)**: Latency before first output token
   - Lower is better
   - Should improve for subsequent turns with cache hits

3. **TPOT (Time per Output Token)**: Average time per generated token
   - Lower is better
   - Indicates generation throughput

### Expected Patterns

- **First Turn**: No cache benefit (cold start)
- **Subsequent Turns**: Increasing cache hit rates
- **Longer Conversations**: Higher overall efficiency gains

### Visualization Examples

The analysis script generates:

1. **Cache Hit Rate by Turn Count**: Shows how cache effectiveness scales
2. **Latency Distribution**: Compares first turn vs subsequent turns
3. **Token Efficiency**: Correlates cache hits with input length

## Example Output

```
==========================================
BENCHMARK RESULTS
==========================================
Total Conversations:     100
Total Requests:          347
Total Tokens:            145,823
Avg TTFT (ms):           78.45
Avg TPOT (ms):           25.31
Cache Hit Rate:          42.5%
==========================================
```

## Comparison: With vs Without KV Caching

To measure the impact of KV caching:

### Baseline (No Caching)

```bash
vllm serve $MODEL_PATH --disable-prefix-caching --port 8000
python lmsys_chat_replay.py --output baseline.json
```

### With LMCache

```bash
vllm serve $MODEL_PATH \
  --enable-prefix-caching \
  --kv-transfer-config '{"kv_connector":"LMCacheConnectorV1", "kv_role":"kv_both"}' \
  --port 8000

python lmsys_chat_replay.py --output lmcache.json
```

### Compare Results

```python
import json

with open('baseline.json') as f:
    baseline = json.load(f)
with open('lmcache.json') as f:
    lmcache = json.load(f)

ttft_improvement = (1 - lmcache['summary']['avg_ttft_ms'] / baseline['summary']['avg_ttft_ms']) * 100
print(f"TTFT Improvement: {ttft_improvement:.1f}%")
print(f"Cache Hit Rate: {lmcache['summary']['cache_hit_rate']:.1%}")
```

## Troubleshooting

### Dataset Download Issues

If streaming from HuggingFace is slow, download the dataset locally:

```bash
# Download dataset
huggingface-cli download lmsys/lmsys-chat-1m --repo-type dataset --local-dir ./lmsys-data

# Use local path
python lmsys_chat_replay.py --dataset-path ./lmsys-data
```

### vLLM Connection Errors

Ensure vLLM is running and accessible:

```bash
curl http://localhost:8000/health
curl http://localhost:8000/v1/models
```

### Memory Issues

For large-scale benchmarks, reduce concurrency:

```bash
python lmsys_chat_replay.py --max-concurrent 1
```

## Advanced Usage

### Custom Conversation Filtering

Modify `LMSysDatasetLoader.load_dataset()` to filter by:
- Specific models
- Language preferences
- Token length ranges
- Topic categories (via conversation content analysis)

### Integration with LMCache Metrics

For detailed LMCache statistics, query the internal API:

```python
import requests

# Get LMCache stats
response = requests.get("http://localhost:8000/lmcache/stats")
stats = response.json()
print(f"Total cache hits: {stats['cache_hits']}")
print(f"Total cache misses: {stats['cache_misses']}")
```

## Citation

If you use this benchmark in your research:

```bibtex
@misc{zheng2023lmsyschat1m,
    title={LMSYS-Chat-1M: A Large-Scale Real-World LLM Conversation Dataset}, 
    author={Lianmin Zheng and Wei-Lin Chiang and Ying Sheng and others},
    year={2023},
    eprint={2309.11998},
    archivePrefix={arXiv}
}
```

## License

This benchmark code is provided under the Apache 2.0 license. The LMSYS-Chat-1M dataset has its own license agreement that must be accepted separately.
