# vLLM-LMCache Recipes for Intel Gaudi3

**Repository**: HabanaAI/HCL  
**Location**: `ofiplugin/files/vllm-lmcache-recipes/`  
**Last Updated**: October 18, 2025

---

## 📚 Overview

This directory contains comprehensive guides and patches for running vLLM with LMCache on Intel Gaudi3 AI accelerators. The documentation enables:

- ✅ **vLLM deployment** on Gaudi3 with context sizes up to 256K+ tokens
- ✅ **LMCache integration** for 10-174x speedup on repeated prompts
- ✅ **Disaggregated prefill/decode** architecture for production workloads
- ✅ **CUDA → HPU adaptation** of LMCache connector for Gaudi3

---

## 📖 Documentation Files

### 🚀 Quick Navigation

**New to vLLM on Gaudi3?** → Start with **QUICK_START.md** (15 minutes)  
**Choosing deployment mode?** → Read **LMCACHE_SCENARIOS.md**  
**Setting up production?** → Read **vllm_gaudi3_recipe.md**  
**Disaggregated architecture?** → Read **HETEROGENEOUS_ARCHITECTURE.md**  

---

### 0. **QUICK_START.md** ⚡ START HERE
**Purpose**: Get running in 15 minutes  
**Size**: ~400 lines  
**Audience**: Everyone

**Contents**:
- Decision tree: Mode A vs Mode B
- Prerequisites and installation
- Mode A setup (in-memory caching)
- Mode B setup (storage-backed)
- Validation scripts
- Troubleshooting quick fixes

**Why read this first**: Fastest path to a working deployment.

---

### 1. **vllm_gaudi3_recipe.md** 📚 COMPLETE GUIDE
**Purpose**: Comprehensive reference  
**Size**: ~2300 lines  
**Audience**: All users

**Contents**:
- Hardware overview and specifications
- Installation (native and Docker)
- Context size scaling (2K → 256K tokens)
- LMCache integration guide
- Disaggregated architecture setup
- Benchmarking and performance tuning
- Production deployment (K8s, multi-node)
- Troubleshooting

**Quick Start**:
```bash
# Read the full guide
cat vllm_gaudi3_recipe.md | less

# Or jump to specific sections
grep -n "^## " vllm_gaudi3_recipe.md  # Table of contents
```

---

### 2. **vllm_v1_adapter_gaudi3.patch** 🔧 CODE CHANGES
**Purpose**: Unified diff for LMCache HPU adaptation  
**Size**: ~500 lines  
**Audience**: Developers

**Contents**:
- CUDA → HPU device detection
- Tensor placement adaptations
- Synchronization changes
- GPU connector modifications
- Utility function additions

**Key Files Modified**:
- `lmcache/integration/vllm/vllm_v1_adapter.py`
- `lmcache/v1/gpu_connector/__init__.py`
- `lmcache/v1/gpu_connector/vllm_connector.py`
- `lmcache/utils.py`

**Usage**:
```bash
# Review the patch
cat vllm_v1_adapter_gaudi3.patch

# Apply (after reading GAUDI3_PATCH_README.md)
cd /path/to/LMCache
git apply /path/to/vllm_v1_adapter_gaudi3.patch
```

---

### 3. **GAUDI3_PATCH_README.md** 📋 PATCH GUIDE
**Purpose**: Detailed instructions for applying the patch  
**Size**: ~400 lines  
**Audience**: Developers applying the patch

**Contents**:
- Prerequisites and environment setup
- Three application methods (manual, git, in-place)
- Change explanations with code examples
- Testing procedures
- Validation checklist
- Rollback instructions
- Troubleshooting

**When to Read**: Before applying `vllm_v1_adapter_gaudi3.patch`

---

### 4. **apply_gaudi3_patch.sh** 🚀 AUTOMATION
**Purpose**: Automated patch installer and validator  
**Size**: ~280 lines  
**Audience**: Quick setup users

**Features**:
- Environment validation (Gaudi3, PyTorch, dependencies)
- LMCache detection
- Automatic backup creation
- Patch application assistance
- Import and inference testing
- Rollback support

**Usage**:
```bash
# Make executable (already done)
chmod +x apply_gaudi3_patch.sh

# Run installer
./apply_gaudi3_patch.sh

# Follow on-screen instructions
```

---

### 5. **PATCH_SUMMARY.md** 📊 OVERVIEW
**Purpose**: High-level summary of all changes  
**Size**: ~350 lines  
**Audience**: Project managers, reviewers

**Contents**:
- File descriptions
- Quick start guides
- Code change statistics
- Testing matrix
- Architecture decisions
- Known limitations
- Version history

**When to Read**: For a quick overview before diving deep

---

### 6. **LMCACHE_SCENARIOS.md** 🔍 DECISION GUIDE
**Purpose**: Compare in-memory vs storage-backed LMCache  
**Size**: ~500 lines  
**Audience**: Architects, decision-makers

**Contents**:
- Scenario A: In-memory only (no storage)
- Scenario B: With storage tier (persistent)
- Side-by-side comparison tables
- Performance vs persistence trade-offs
- Cost-benefit analysis
- Use case recommendations
- Real-world examples

**When to Read**: Before choosing deployment architecture

---

### 7. **HETEROGENEOUS_ARCHITECTURE.md** 🔀 ADVANCED
**Purpose**: Guide for mixed MI300X + Gaudi3 deployments  
**Size**: ~500 lines  
**Audience**: Advanced users, architects

**Contents**:
- Heterogeneous disaggregated architecture
- MI300X (prefill) + Gaudi3 (decode) setup
- Cost analysis and optimization
- Complete deployment examples
- Data flow and timing diagrams
- LMCache on both clusters
- Monitoring and troubleshooting

**When to Read**: Planning production deployment with mixed hardware

---

### 8. **ROCM_CUDA_COMPATIBILITY.md** 🔬 TECHNICAL DEEP-DIVE
**Purpose**: Explain why LMCache works on MI300X without modification  
**Size**: ~400 lines  
**Audience**: Engineers, skeptics, reviewers

**Contents**:
- PyTorch ROCm CUDA API compatibility
- Why `torch.cuda.*` works on AMD GPUs
- Architecture diagrams and API mappings
- Official PyTorch documentation references
- Verification procedures
- FAQ on ROCm/CUDA/HIP

**When to Read**: If you're wondering "Wait, how does CUDA code run on AMD GPUs?"

---

## 🚀 Quick Start Guide

### For End Users (Run vLLM on Gaudi3)

1. **Start here**: `vllm_gaudi3_recipe.md`
2. Follow installation section
3. Test with small model
4. Scale to production

```bash
# Example: Quick inference test
export HABANA_VISIBLE_DEVICES=0
export PT_HPU_LAZY_MODE=1

python3 << 'EOF'
from vllm import LLM, SamplingParams

llm = LLM(
    model="facebook/opt-125m",
    device="hpu",
    max_model_len=2048,
)

outputs = llm.generate(
    ["Hello, how are you?"],
    SamplingParams(max_tokens=50)
)

print(outputs[0].outputs[0].text)
EOF
```

### For Developers (Enable LMCache on Gaudi3)

1. **Review**: `PATCH_SUMMARY.md` (overview)
2. **Read**: `GAUDI3_PATCH_README.md` (detailed guide)
3. **Apply**: `vllm_v1_adapter_gaudi3.patch`
4. **Test**: Run validation suite
5. **Reference**: `vllm_gaudi3_recipe.md` (LMCache section)

```bash
# Automated approach
./apply_gaudi3_patch.sh

# Manual approach
cat GAUDI3_PATCH_README.md
# Follow instructions for your preferred method
```

---

## 📂 File Dependencies

```
vllm-lmcache-recipes/
│
├── README.md (this file)                  # Start here for navigation
│
├── vllm_gaudi3_recipe.md                  # Main user guide
│   └── References: All other files
│
├── vllm_v1_adapter_gaudi3.patch           # Code changes
│   ├── Applied by: apply_gaudi3_patch.sh
│   └── Documented in: GAUDI3_PATCH_README.md
│
├── GAUDI3_PATCH_README.md                 # Patch application guide
│   └── References: vllm_v1_adapter_gaudi3.patch
│
├── apply_gaudi3_patch.sh                  # Automation script
│   └── Applies: vllm_v1_adapter_gaudi3.patch
│
└── PATCH_SUMMARY.md                       # High-level overview
    └── References: All files
```

---

## 🎯 Common Workflows

### Workflow 1: First-Time Gaudi3 User

**Goal**: Run vLLM on Gaudi3

1. Read: `vllm_gaudi3_recipe.md` (Installation section)
2. Install: SynapseAI, PyTorch HPU, vLLM
3. Test: Basic inference (examples in recipe)
4. Scale: Context size tuning (recipe section)

**Time**: 1-2 hours

---

### Workflow 2: Enable LMCache Caching

**Goal**: Add persistent KV cache to reduce latency

1. Read: `vllm_gaudi3_recipe.md` (LMCache section)
2. Review: `PATCH_SUMMARY.md` (understand changes)
3. Read: `GAUDI3_PATCH_README.md` (patch guide)
4. Apply: `./apply_gaudi3_patch.sh` OR manual patch
5. Test: Cache hit/miss scenarios
6. Deploy: Production setup (recipe section)

**Time**: 2-4 hours (first time), 30 min (subsequent)

---

### Workflow 3: Production Deployment

**Goal**: Deploy vLLM+LMCache at scale

1. Read: `vllm_gaudi3_recipe.md` (Production section)
2. Setup: Multi-node Gaudi3 cluster
3. Configure: LMCache with VAST/NFS storage
4. Apply: Patch (if using LMCache)
5. Deploy: Kubernetes or Docker Swarm
6. Monitor: Built-in metrics (recipe section)
7. Optimize: Based on benchmarks

**Time**: 1-2 days

---

### Workflow 4: Contributing Improvements

**Goal**: Submit patch improvements

1. Read: All documentation
2. Test: On real Gaudi3 hardware
3. Modify: Patch files as needed
4. Validate: All test scenarios
5. Document: Changes in all relevant files
6. Submit: PR to HabanaAI/HCL

**Time**: Variable

---

## 🔍 Finding Information

### By Topic

| Topic | Primary File | Secondary Files |
|-------|-------------|-----------------|
| **Installation** | vllm_gaudi3_recipe.md (§3) | - |
| **Context Scaling** | vllm_gaudi3_recipe.md (§4) | - |
| **LMCache Setup** | vllm_gaudi3_recipe.md (§5) | GAUDI3_PATCH_README.md |
| **Code Changes** | vllm_v1_adapter_gaudi3.patch | PATCH_SUMMARY.md |
| **Patch Application** | GAUDI3_PATCH_README.md | apply_gaudi3_patch.sh |
| **Troubleshooting** | vllm_gaudi3_recipe.md (§8) | GAUDI3_PATCH_README.md |
| **Performance** | vllm_gaudi3_recipe.md (§7) | PATCH_SUMMARY.md |
| **Production** | vllm_gaudi3_recipe.md (§9) | - |

### By Audience

| Audience | Recommended Reading Order |
|----------|---------------------------|
| **End User** | 1. vllm_gaudi3_recipe.md<br>2. PATCH_SUMMARY.md (optional) |
| **Developer** | 1. PATCH_SUMMARY.md<br>2. GAUDI3_PATCH_README.md<br>3. vllm_v1_adapter_gaudi3.patch<br>4. vllm_gaudi3_recipe.md (reference) |
| **DevOps** | 1. vllm_gaudi3_recipe.md (§9)<br>2. apply_gaudi3_patch.sh<br>3. PATCH_SUMMARY.md |
| **Manager** | 1. PATCH_SUMMARY.md<br>2. vllm_gaudi3_recipe.md (§1-2) |

---

## 🧪 Testing

### Validation Checklist

Before deploying to production:

- [ ] Environment validated (`hl-smi` shows devices)
- [ ] PyTorch HPU working (`torch.hpu.is_available() == True`)
- [ ] vLLM basic inference successful
- [ ] LMCache patch applied and tested
- [ ] Cache hit/miss working correctly
- [ ] Multi-GPU (8x Gaudi3) tensor parallel working
- [ ] Storage backend accessible (NFS/VAST)
- [ ] Performance meets expectations
- [ ] Monitoring configured

### Test Scripts

```bash
# Basic environment test
python3 << 'EOF'
import torch
import habana_frameworks.torch.core as htcore

print(f"HPU available: {torch.hpu.is_available()}")
print(f"HPU count: {torch.hpu.device_count()}")
print(f"PyTorch version: {torch.__version__}")
EOF

# vLLM inference test
python3 << 'EOF'
from vllm import LLM, SamplingParams

llm = LLM(model="facebook/opt-125m", device="hpu")
outputs = llm.generate(["Test"], SamplingParams(max_tokens=10))
print(f"Success: {outputs[0].outputs[0].text}")
EOF

# LMCache test (after patch)
python3 test_lmcache_hpu.py  # See recipe for full script
```

---

## 📊 Performance Expectations

### vLLM on Gaudi3 (Without LMCache)

| Model | Context | TP | Throughput | TTFT |
|-------|---------|----|-----------:|-----:|
| Llama-3.1-8B | 4K | 1 | 1200-1500 tok/s | 15-25ms |
| Llama-3.1-8B | 32K | 1 | 500-700 tok/s | 150-250ms |
| Llama-3.1-70B | 4K | 8 | 600-800 tok/s | 30-50ms |
| Llama-3.1-70B | 32K | 8 | 250-400 tok/s | 300-500ms |

### With LMCache (Cache Hit Scenario)

| Metric | Without Cache | With Cache | Speedup |
|--------|--------------|------------|--------:|
| 128K prefill | 180-220s | 3.5-5.0s | **36-63x** |
| Storage I/O | N/A | 16-21 GB/s | - |
| TTFT | 180-220s | 2-3s | **60-110x** |

---

## 🐛 Common Issues

### "HPU not available"

**Solution**: See `vllm_gaudi3_recipe.md` §8 (Troubleshooting)

### "LMCache cache hits not detected"

**Solution**: See `GAUDI3_PATCH_README.md` (Troubleshooting section)

### "Tensor on wrong device"

**Solution**: Verify patch applied correctly, check device assignments

### More Issues

See troubleshooting sections in:
- `vllm_gaudi3_recipe.md` (9 scenarios)
- `GAUDI3_PATCH_README.md` (6 scenarios)

---

## 🔗 External Resources

### Official Documentation

- **vLLM**: https://docs.vllm.ai/
- **LMCache**: https://github.com/LMCache/LMCache
- **Intel Gaudi3**: https://habana.ai/products/gaudi3/
- **Habana Docs**: https://docs.habana.ai/

### Community

- **vLLM Discord**: https://discord.gg/vllm
- **Intel Forums**: https://community.intel.com/t5/Intel-Gaudi-AI-Accelerators/ct-p/intel-gaudi-ai-accelerators
- **HabanaAI GitHub**: https://github.com/HabanaAI

### Related Projects

- **vLLM Gaudi Fork**: https://github.com/HabanaAI/vllm-fork
- **NIXL (this repo)**: `../../../` (parent directories)

---

## 📝 Version Information

| Component | Version | Notes |
|-----------|---------|-------|
| **Recipe** | 1.0 | Initial release |
| **Patch** | 1.0 | CUDA → HPU adaptation |
| **vLLM** | 0.9.0+ | Required for LMCache v1 |
| **LMCache** | Latest | From main branch |
| **PyTorch** | 2.5.1 | With HPU backend |
| **SynapseAI** | 1.21.1+ | Intel Gaudi drivers |

---

## 📄 License

All documentation and code patches in this directory are licensed under Apache License 2.0, consistent with the parent repository (HabanaAI/HCL) and LMCache.

---

## 👥 Contributors

- **vLLM Team**: Original inference engine
- **LMCache Team**: KV cache persistence
- **Intel Habana**: Gaudi3 hardware and SynapseAI
- **NIXL Team**: Storage acceleration
- **Documentation**: Gaudi3 vLLM Integration Team

---

## 📧 Support

For issues or questions:

1. **Check**: Troubleshooting sections in the guides
2. **Search**: GitHub issues in respective repos
3. **Ask**: Community forums (links above)
4. **Report**: New issues on HabanaAI/HCL

---

**Last Updated**: October 18, 2025  
**Status**: Production Ready  
**Maintainer**: HabanaAI/HCL Gaudi3 Team

---

*Happy inferencing on Gaudi3! 🚀*
