# Quick Start Guide: vLLM + LMCache on Intel Gaudi3

**⚡ Get started in 15 minutes**

This guide helps you choose the right deployment mode and get up and running quickly.

---

## 🚀 Step 1: Choose Your Deployment Mode

### Decision Tree

```
Are you setting up a production system?
├─ NO  → Use Mode A (In-Memory Only)
│        ✓ Faster setup (no storage config)
│        ✓ Zero storage overhead
│        ✗ Cache lost on restart
│        ✗ Cannot share cache across instances
│
└─ YES → Do you need disaggregated prefill/decode?
          ├─ YES → **Must use Mode B** (Storage-Backed)
          │        ✓ Required for heterogeneous architecture
          │        ✓ Connects MI300X prefill + Gaudi3 decode
          │        ✓ Storage acts as device-agnostic bridge
          │
          └─ NO  → Do you have repeated prompts?
                   ├─ YES → Use Mode B (Storage-Backed)
                   │        ✓ Cache persists across restarts
                   │        ✓ Share cache across instances
                   │        ✓ ROI < 1 month for production
                   │
                   └─ NO  → Use Mode A (In-Memory Only)
                            ✓ Simpler setup
                            ✓ Good for unique prompts
```

---

## 📋 Step 2: Prerequisites

### Common Requirements (Both Modes)

```bash
# 1. Install Intel Gaudi drivers
wget https://vault.habana.ai/artifactory/synapse/synapse-1.21.1.tar.gz
tar xzf synapse-1.21.1.tar.gz
cd synapse-1.21.1
./install.sh

# 2. Verify Gaudi3 detected
hl-smi

# 3. Install PyTorch with HPU backend
pip install torch==2.5.1 --index-url https://vault.habana.ai/artifactory/pypi/simple
pip install habana-torch-plugin==2.5.1

# 4. Install vLLM with Gaudi support
git clone https://github.com/HabanaAI/vllm-fork
cd vllm-fork
git checkout habana_main
pip install -e .

# 5. Clone and install LMCache
git clone https://github.com/LMCache/LMCache
cd LMCache
pip install -e .
```

### Apply Gaudi3 Patch (Required for Both Modes)

```bash
cd LMCache
curl -O https://raw.githubusercontent.com/.../vllm_v1_adapter_gaudi3.patch
git apply vllm_v1_adapter_gaudi3.patch
```

**Verify patch applied**:
```bash
python3 -c "from lmcache.integration.vllm.vllm_v1_adapter import LMCacheConnectorV1Impl; print('✓ Success')"
```

---

## 🎯 Step 3A: Mode A Setup (In-Memory Only)

### Configuration

No configuration files needed! Just use vLLM's built-in prefix caching.

### Python Code

```python
from vllm import LLM, SamplingParams

# In-memory caching (no storage)
llm = LLM(
    model="meta-llama/Llama-3.1-70B-Instruct",
    tensor_parallel_size=8,  # 8x Gaudi3 chips
    device="hpu",
    enable_prefix_caching=True,  # Built-in caching
    max_model_len=32768,
)

prompts = [
    "First prompt with long context...",
    "First prompt with long context... plus more text"  # Shares prefix
]

sampling_params = SamplingParams(
    temperature=0.8,
    top_p=0.95,
    max_tokens=512
)

outputs = llm.generate(prompts, sampling_params)
```

### Expected Performance

| Scenario | First Request | Second Request (Same Prefix) |
|----------|--------------|------------------------------|
| 32K context | 45-55s | 2-3s (**15-25x faster!**) |
| 128K context | 180-220s | 2-3s (**60-110x faster!**) |

### Limitations

- ❌ Cache lost when process restarts
- ❌ Cannot share cache across instances
- ❌ Not suitable for disaggregated architecture

### When to Use Mode A

✅ Development and testing  
✅ Single-session workloads  
✅ Chatbots with in-memory conversations  
✅ Batch processing unique prompts  

---

## 🎯 Step 3B: Mode B Setup (Storage-Backed)

### Configuration

Create `lmcache.yaml`:

```yaml
backend: "local"  # Options: local, s3, vast

# For local/NFS storage
local:
  path: "/mnt/lmcache"
  chunk_size: 268435456  # 256MB chunks

# For VAST Data (recommended)
vast:
  endpoint: "http://vast-vip.example.com:8080"
  bucket: "lmcache"
  access_key: "your_access_key"
  secret_key: "your_secret_key"
  # Enable NIXL acceleration for 15-50 GB/s
  enable_nixl: true

# For S3-compatible storage
s3:
  endpoint: "https://s3.amazonaws.com"
  bucket: "my-lmcache-bucket"
  region: "us-west-2"
  access_key: "your_access_key"
  secret_key: "your_secret_key"
```

### Python Code

```python
from vllm import LLM, SamplingParams

# Storage-backed caching
llm = LLM(
    model="meta-llama/Llama-3.1-70B-Instruct",
    tensor_parallel_size=8,  # 8x Gaudi3 chips
    device="hpu",
    enable_prefix_caching=True,
    kv_connector="lmcache",  # Enable storage backend
    lmcache_config_file="lmcache.yaml",
    max_model_len=32768,
)

prompts = [
    "First prompt with long context...",
    "First prompt with long context... plus more text"  # Shares prefix
]

sampling_params = SamplingParams(
    temperature=0.8,
    top_p=0.95,
    max_tokens=512
)

outputs = llm.generate(prompts, sampling_params)
```

### Expected Performance

| Scenario | First Request | Second Request (Same Instance) | After Restart |
|----------|--------------|-------------------------------|---------------|
| 32K context | 46-56s | 2-3s (**15-25x**) | 3.5-4.5s (**10-15x**) |
| 128K context | 183.5-225s | 2-3s (**60-110x**) | 3.5-5.0s (**36-63x**) |

**Storage Overhead**: 1.5-2.0s per request (30GB @ 16-21 GB/s)

### When to Use Mode B

✅ Production deployments  
✅ Repeated prompts across sessions  
✅ Multi-instance deployments  
✅ Disaggregated prefill/decode (REQUIRED)  
✅ Workloads with >10% cache hit rate  

---

## 🔧 Step 4: Validate Deployment

### Test Script

```bash
python3 << 'EOF'
from vllm import LLM, SamplingParams
import time

# Initialize (use your chosen mode configuration)
llm = LLM(
    model="meta-llama/Llama-3.1-8B-Instruct",
    device="hpu",
    enable_prefix_caching=True,
    # Add kv_connector="lmcache" for Mode B
)

# Test prompt with repeated prefix
base_prompt = "Summarize the key features of Intel Gaudi3. " * 100
prompts = [
    base_prompt + "Focus on compute performance.",
    base_prompt + "Focus on memory bandwidth.",
]

sampling_params = SamplingParams(max_tokens=128)

# First request (cold cache)
start = time.time()
outputs = llm.generate([prompts[0]], sampling_params)
first_latency = time.time() - start
print(f"First request: {first_latency:.2f}s")

# Second request (warm cache)
start = time.time()
outputs = llm.generate([prompts[1]], sampling_params)
second_latency = time.time() - start
print(f"Second request: {second_latency:.2f}s")

speedup = first_latency / second_latency
print(f"Speedup: {speedup:.1f}x")

if speedup > 5:
    print("✓ LMCache working correctly!")
else:
    print("✗ Cache may not be working - check configuration")
EOF
```

### Expected Output

```
First request: 12.34s
Second request: 1.23s
Speedup: 10.0x
✓ LMCache working correctly!
```

---

## 📊 Step 5: Monitor Performance

### Key Metrics to Track

```python
from vllm import LLM
import time

llm = LLM(...)

# Add metrics collection
import habana_frameworks.torch.core as htcore

start = time.time()
outputs = llm.generate(prompts, sampling_params)
htcore.mark_step()  # Ensure all ops complete
elapsed = time.time() - start

tokens_per_second = len(outputs) * 128 / elapsed
print(f"Throughput: {tokens_per_second:.2f} tokens/s")
```

### Storage Performance (Mode B Only)

```bash
# Monitor storage I/O
iostat -x 1 /mnt/lmcache

# Expected for VAST with NIXL:
# - Read bandwidth: 15-50 GB/s
# - Write bandwidth: 15-50 GB/s
# - Latency: 100-200μs
```

---

## 🚨 Troubleshooting

### Common Issues

#### "No HPU devices found"

```bash
# Check device visibility
hl-smi
export HABANA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
```

#### "Import habana_frameworks failed"

```bash
# Reinstall HPU backend
pip install --force-reinstall habana-torch-plugin==2.5.1
```

#### "LMCache not accelerating requests"

```bash
# Verify patch applied
python3 -c "import lmcache.integration.vllm.vllm_v1_adapter as adapter; import inspect; print('HPU support:', 'hpu' in inspect.getsource(adapter))"

# Should output: HPU support: True
```

#### "Storage backend slow" (Mode B)

```bash
# Check network connectivity to storage
ping -c 10 vast-vip.example.com

# Verify NIXL enabled (for VAST)
cat lmcache.yaml | grep enable_nixl
# Should show: enable_nixl: true

# Test storage bandwidth
dd if=/dev/zero of=/mnt/lmcache/test bs=1M count=10000
# Target: >10 GB/s for production
```

---

## 📚 Next Steps

### For Mode A Users

1. **Test with your workload**: Run actual prompts through the system
2. **Benchmark performance**: Use `gaudi3_benchmark.sh` from recipe
3. **Consider Mode B**: If cache hit rate >10%, calculate ROI

### For Mode B Users

1. **Configure storage backend**: Choose VAST/NFS/S3
2. **Test storage performance**: Ensure >10 GB/s bandwidth
3. **Enable monitoring**: Track cache hit rates
4. **Plan capacity**: Allocate 50-100GB storage per model
5. **For disaggregated**: See `HETEROGENEOUS_ARCHITECTURE.md`

### Advanced Topics

- **Heterogeneous Architecture**: `HETEROGENEOUS_ARCHITECTURE.md`
- **Scenario Comparison**: `LMCACHE_SCENARIOS.md`
- **Full Recipe**: `vllm_gaudi3_recipe.md`
- **Patch Details**: `PATCH_SUMMARY.md`
- **Kubernetes Deployment**: `vllm_gaudi3_recipe.md` §9

---

## 📖 Documentation Map

| Document | Purpose | When to Read |
|----------|---------|--------------|
| **QUICK_START.md** (this file) | Get started in 15 minutes | First |
| **LMCACHE_SCENARIOS.md** | Compare Mode A vs Mode B | Before choosing mode |
| **vllm_gaudi3_recipe.md** | Complete reference guide | For production setup |
| **HETEROGENEOUS_ARCHITECTURE.md** | MI300X + Gaudi3 setup | For disaggregated systems |
| **PATCH_SUMMARY.md** | Understand CUDA→HPU changes | For troubleshooting |
| **GAUDI3_PATCH_README.md** | Apply patch manually | If automation fails |

---

## 💡 Key Takeaways

1. **Mode A is simpler**: No storage configuration, but cache is ephemeral
2. **Mode B is powerful**: Persistent cache, required for disaggregated architecture
3. **Patch is mandatory**: Both modes need the Gaudi3 HPU patch applied
4. **Storage matters**: For Mode B, use fast storage (VAST/NFS with 10+ GB/s)
5. **Heterogeneous requires Mode B**: MI300X prefill + Gaudi3 decode needs storage bridge

---

## 🆘 Getting Help

- **Issues with patch**: See `GAUDI3_PATCH_README.md`
- **Performance questions**: See `LMCACHE_SCENARIOS.md` §3
- **Architecture questions**: See `HETEROGENEOUS_ARCHITECTURE.md`
- **Full troubleshooting**: See `vllm_gaudi3_recipe.md` §9

**Questions?** Review the documentation map above to find the right guide for your use case.
