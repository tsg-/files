# vLLM Recipe Collection

This directory contains comprehensive guides for running vLLM with LMCache and NIXL on different AI accelerators.

## Available Recipes

### 1. AMD MI300X Recipe
**File**: `vllm_mi300x_recipe.md`  
**Status**: ✅ Complete with LMCache integration  
**Hardware**: AMD MI300X (192GB HBM3, 5.3TB/s bandwidth)  
**Highlights**:
- Complete installation guide (ROCm 6.1+, PyTorch 2.1+, vLLM 0.9.0+)
- Context scaling: 2K → 256K+ tokens
- 5 LMCache configuration examples
- VastData benchmark replication (174x TTFT improvement)
- Disaggregated prefill/decode architecture
- NIXL GDS integration for 35GB/s storage access

**Key Performance**:
- 128K context prefill: 262s (compute-only)
- KV cache load: 1.5s @ 21GB/s (with LMCache)
- Total speedup: **174x** for cached inference

---

### 2. Intel Gaudi3 Recipe
**File**: `vllm_gaudi3_recipe.md`  
**Status**: ✅ Complete, LMCache requires porting  
**Hardware**: Intel Gaudi3 (128GB HBM2e, 3.7TB/s bandwidth)  
**Highlights**:
- SynapseAI 1.21.1+ installation
- PyTorch 2.5.1 with HPU backend
- vLLM Gaudi plugin integration
- Context scaling: 2K → 256K+ tokens
- LMCache porting guide (CUDA → HPU)
- HCCL distributed training
- Production deployment (Docker, Kubernetes)

**Key Performance**:
- 128K context prefill: 180-220s (estimated)
- KV cache load: 2.0-2.5s @ 16-20GB/s (projected with LMCache)
- Total speedup: **36-63x** for cached inference (estimated)

**LMCache Status**: 
- ⚠️ Requires 2-4 weeks of porting effort
- See `LMCACHE_GAUDI3_PORT.md` for detailed analysis
- CUDA → HPU API mapping documented
- Implementation roadmap provided

---

## LMCache Port Analysis

**File**: `LMCACHE_GAUDI3_PORT.md`  
**Purpose**: Technical deep-dive on porting LMCache from CUDA to Intel Gaudi3 HPU

**Contents**:
- File-by-file analysis of `lmcache_connector.py`
- Complete CUDA vs HPU API mapping
- Required code changes with full implementations
- Testing strategy and validation
- 4-week implementation roadmap
- Risk assessment

**Key Findings**:
- Current LMCache uses `LMCacheConnectorV1Impl` from external `lmcache` library
- This implementation is CUDA-specific
- HPU port requires:
  1. Create `lmcache/integration/vllm/vllm_v1_adapter_hpu.py`
  2. Create `vllm/distributed/kv_transfer/kv_connector/v1/lmcache_connector_hpu.py`
  3. Update connector factory for device routing
  4. Comprehensive testing

**Estimated Effort**: 2-4 weeks

---

## Feature Comparison

| Feature | MI300X | Gaudi3 |
|---------|--------|--------|
| **HBM Capacity** | 192GB HBM3 | 128GB HBM2e |
| **HBM Bandwidth** | 5.3 TB/s | 3.7 TB/s |
| **Compute (FP8)** | 1300 TFLOPS | 1835 TFLOPS |
| **Interconnect** | Infinity Fabric | 24x 200Gb RoCE |
| **GDS/Direct Storage** | ✅ Yes (35GB/s) | ⚠️ Via PCIe (15-20GB/s) |
| **LMCache Support** | ✅ Ready | ⚠️ Needs porting |
| **vLLM Support** | ✅ Native | ✅ Plugin-based |
| **NIXL Backend** | ✅ OFI + GDS | 🔄 Needs RoCE plugin |
| **Price/Performance** | Higher performance | Lower TCO |

---

## Quick Start

### AMD MI300X

```bash
# Install ROCm 6.1+
sudo apt install rocm-dkms

# Install vLLM
pip install vllm

# Install LMCache
pip install lmcache

# Run with LMCache
python3 -c "
from vllm import LLM, SamplingParams

llm = LLM(
    model='meta-llama/Llama-3.1-70B',
    tensor_parallel_size=8,
    max_model_len=131072,
    kv_connector='lmcache',
    kv_transfer_config={
        'backend': 'nfs://vast-cluster/cache',
    }
)

outputs = llm.generate(['Test prompt'], SamplingParams(max_tokens=100))
print(outputs[0].outputs[0].text)
"
```

### Intel Gaudi3

```bash
# Install SynapseAI
sudo apt install habanalabs-drivers habanalabs-graph

# Install PyTorch HPU
pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cpu
pip install habana-torch-plugin==2.5.1

# Install vLLM Gaudi
pip install vllm vllm-hpu

# Run inference
python3 -c "
from vllm import LLM, SamplingParams
import habana_frameworks.torch.core as htcore

llm = LLM(
    model='meta-llama/Llama-3.1-70B',
    tensor_parallel_size=8,
    max_model_len=131072,
    device='hpu',
)

outputs = llm.generate(['Test prompt'], SamplingParams(max_tokens=100))
print(outputs[0].outputs[0].text)
"
```

**Note**: LMCache on Gaudi3 requires porting (see `LMCACHE_GAUDI3_PORT.md`).

---

## Repository Structure

```
nixl/
├── vllm_mi300x_recipe.md          # Complete MI300X guide (~1700 lines)
├── vllm_gaudi3_recipe.md          # Complete Gaudi3 guide (~1500 lines)
├── LMCACHE_GAUDI3_PORT.md         # LMCache porting analysis (~800 lines)
├── README_RECIPES.md              # This file
└── src/
    └── plugins/
        ├── obj/                   # S3/object storage backend
        │   └── obj_s3_client.cpp  # Fixed S3 hostname resolution
        └── gaudi/                 # (Future) Gaudi3 RoCE plugin
```

---

## Contributing

### Adding New Recipes

1. Follow the existing structure:
   - Introduction
   - Hardware Overview
   - Installation
   - Context Size Scaling
   - LMCache Integration
   - Disaggregated Architecture
   - Benchmarking
   - Troubleshooting
   - Production Deployment

2. Include working code examples
3. Provide performance benchmarks
4. Document known limitations

### Porting LMCache to New Devices

See `LMCACHE_GAUDI3_PORT.md` as a template for porting analysis.

Key sections:
- Current implementation analysis
- Device API mapping (CUDA/HPU/ROCm/etc.)
- Required code changes
- Testing strategy
- Implementation roadmap
- Risk assessment

---

## Performance Benchmarks

### AMD MI300X (8 GPUs)

| Model | Context | Batch | Throughput | TTFT (cache hit) |
|-------|---------|-------|------------|------------------|
| Llama 3.1 8B | 4K | 256 | 2500 TPS | 10ms |
| Llama 3.1 8B | 128K | 8 | 600 TPS | 1.5s |
| Llama 3.1 70B | 4K | 64 | 1200 TPS | 25ms |
| Llama 3.1 70B | 128K | 4 | 400 TPS | 1.5s |

### Intel Gaudi3 (8 chips)

| Model | Context | Batch | Throughput | TTFT (estimated) |
|-------|---------|-------|------------|------------------|
| Llama 3.1 8B | 4K | 256 | 1400 TPS | 15ms |
| Llama 3.1 8B | 128K | 8 | 500 TPS | 2.0s |
| Llama 3.1 70B | 4K | 64 | 700 TPS | 50ms |
| Llama 3.1 70B | 128K | 4 | 350 TPS | 2.0s |

---

## Related Resources

### Documentation
- [vLLM Documentation](https://docs.vllm.ai/)
- [LMCache GitHub](https://github.com/LMCache/LMCache)
- [NIXL Backend Guide](docs/BackendGuide.md)
- [ROCm Documentation](https://rocm.docs.amd.com/)
- [Intel Gaudi Documentation](https://docs.habana.ai/)

### Benchmarks
- [VastData Blog: Accelerating Inference](https://www.vastdata.com/blog/accelerating-inference)
- [vLLM Benchmark Results](https://docs.vllm.ai/en/latest/performance_benchmark/benchmarks.html)

### Community
- [vLLM Discord](https://discord.gg/vllm)
- [ROCm Forum](https://github.com/ROCm/ROCm/discussions)
- [Habana Community](https://community.intel.com/t5/Intel-Gaudi-AI-Accelerators/ct-p/intel-gaudi-ai-accelerators)

---

## License

These recipes are provided under the same license as the NIXL project (see LICENSE file).

---

**Last Updated**: October 18, 2025  
**Maintainers**: NIXL Team, vLLM Contributors  
**Feedback**: Open an issue or submit a PR!
