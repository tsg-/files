#!/bin/bash
# End-to-End Flow for vLLM + LMCache with LMSYS-Chat-1M Dataset
# This script demonstrates the complete workflow from setup to benchmarking

set -e  # Exit on error

echo "=========================================="
echo "vLLM + LMCache LMSYS-Chat-1M Benchmark"
echo "=========================================="
echo ""

# Configuration
MODEL_PATH=${MODEL_PATH:-"meta-llama/Meta-Llama-3.1-8B-Instruct"}
VLLM_PORT=${VLLM_PORT:-8000}
NUM_CONVERSATIONS=${NUM_CONVERSATIONS:-100}
MIN_TURNS=${MIN_TURNS:-3}
MAX_CONCURRENT=${MAX_CONCURRENT:-5}

# Step 1: Install dependencies
echo "Step 1: Installing dependencies..."
pip install -q -r requirements_lmsys.txt
echo "✓ Dependencies installed"
echo ""

# Step 2: Authenticate with HuggingFace
echo "Step 2: HuggingFace Authentication"
echo "The LMSYS-Chat-1M dataset requires accepting a license agreement."
echo "Please visit: https://huggingface.co/datasets/lmsys/lmsys-chat-1m"
echo ""
read -p "Have you accepted the license and logged in with 'huggingface-cli login'? (y/n) " -n 1 -r
echo ""
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Please run: huggingface-cli login"
    echo "Then re-run this script."
    exit 1
fi
echo "✓ Authentication confirmed"
echo ""

# Step 3: Start vLLM server with LMCache
echo "Step 3: Starting vLLM server with LMCache..."
echo "Command:"
echo "  vllm serve $MODEL_PATH \\"
echo "    --port $VLLM_PORT \\"
echo "    --enable-prefix-caching \\"
echo "    --kv-transfer-config '{\"kv_connector\":\"LMCacheConnectorV1\", \"kv_role\":\"kv_both\"}' \\"
echo "    --disable-log-requests \\"
echo "    --served-model-name default"
echo ""
echo "NOTE: Start the vLLM server in a separate terminal, then press Enter to continue..."
read -p "Press Enter when vLLM server is running..."
echo ""

# Step 4: Wait for vLLM to be ready
echo "Step 4: Checking vLLM server availability..."
MAX_RETRIES=30
RETRY_COUNT=0
while [ $RETRY_COUNT -lt $MAX_RETRIES ]; do
    if curl -s "http://localhost:$VLLM_PORT/health" > /dev/null 2>&1; then
        echo "✓ vLLM server is ready"
        break
    fi
    RETRY_COUNT=$((RETRY_COUNT + 1))
    echo "  Waiting for vLLM server... ($RETRY_COUNT/$MAX_RETRIES)"
    sleep 2
done

if [ $RETRY_COUNT -eq $MAX_RETRIES ]; then
    echo "✗ vLLM server did not become available"
    exit 1
fi
echo ""

# Step 5: Run baseline benchmark (without KV caching)
echo "Step 5: Running BASELINE benchmark (without KV caching)..."
echo "This will take several minutes..."
python lmsys_chat_replay.py \
    --vllm-endpoint "http://localhost:$VLLM_PORT/v1" \
    --model-name default \
    --num-conversations $NUM_CONVERSATIONS \
    --min-turns $MIN_TURNS \
    --max-concurrent $MAX_CONCURRENT \
    --output lmsys_baseline_results.json

echo "✓ Baseline benchmark complete"
echo ""

# Step 6: Analyze results
echo "Step 6: Analyzing results..."
echo "Results saved to:"
echo "  - lmsys_baseline_results.json"
echo ""

# Step 7: Generate comparison report
echo "Step 7: Generating comparison report..."
cat > generate_report.py << 'EOF'
import json
import sys

def load_results(filepath):
    with open(filepath) as f:
        return json.load(f)

def print_comparison(baseline_file):
    baseline = load_results(baseline_file)
    
    print("=" * 80)
    print("LMSYS-Chat-1M BENCHMARK SUMMARY")
    print("=" * 80)
    print()
    print(f"Conversations Analyzed:  {baseline['summary']['total_conversations']}")
    print(f"Total Requests:          {baseline['summary']['total_requests']}")
    print(f"Total Tokens:            {baseline['summary']['total_tokens']:,}")
    print()
    print("LATENCY METRICS:")
    print(f"  Avg TTFT (ms):         {baseline['summary']['avg_ttft_ms']:.2f}")
    print(f"  Avg TPOT (ms):         {baseline['summary']['avg_tpot_ms']:.2f}")
    print()
    print("KV CACHE METRICS:")
    print(f"  Cache Hit Rate:        {baseline['summary']['cache_hit_rate']:.2%}")
    print()
    print("=" * 80)
    print()
    
    # Multi-turn analysis
    print("MULTI-TURN CONVERSATION ANALYSIS:")
    print("-" * 80)
    
    turn_stats = {}
    for conv in baseline['conversations']:
        num_turns = conv['num_turns']
        if num_turns not in turn_stats:
            turn_stats[num_turns] = {
                'count': 0,
                'total_tokens': 0,
                'avg_cache_hit_rate': []
            }
        
        turn_stats[num_turns]['count'] += 1
        turn_stats[num_turns]['total_tokens'] += conv['total_input_tokens']
        
        hit_rate = conv['cache_hit_tokens'] / max(conv['total_input_tokens'], 1)
        turn_stats[num_turns]['avg_cache_hit_rate'].append(hit_rate)
    
    print(f"{'Turns':<10} {'Count':<10} {'Avg Tokens':<15} {'Avg Cache Hit Rate':<20}")
    print("-" * 80)
    
    for turns in sorted(turn_stats.keys()):
        stats = turn_stats[turns]
        avg_tokens = stats['total_tokens'] / stats['count']
        avg_hit_rate = sum(stats['avg_cache_hit_rate']) / len(stats['avg_cache_hit_rate'])
        
        print(f"{turns:<10} {stats['count']:<10} {avg_tokens:<15.0f} {avg_hit_rate:<20.2%}")
    
    print("-" * 80)
    print()

if __name__ == "__main__":
    print_comparison("lmsys_baseline_results.json")
EOF

python generate_report.py
rm generate_report.py
echo ""

echo "=========================================="
echo "BENCHMARK COMPLETE!"
echo "=========================================="
echo ""
echo "Next Steps:"
echo "1. Review the results above"
echo "2. Compare with/without prefix caching enabled"
echo "3. Test with different LMCACHE_CONFIG_FILE settings"
echo "4. Analyze cache hit rates by conversation length"
echo ""
echo "To visualize results, use the analyze_results.py script"
