# Documentation Map - Visual Guide

```
┌─────────────────────────────────────────────────────────────────────┐
│                     vLLM on Intel Gaudi3                            │
│                    Complete Documentation Set                       │
└─────────────────────────────────────────────────────────────────────┘

                              START HERE
                                  ↓
                     ┌────────────────────────┐
                     │   QUICK_START.md       │ ← Read First (15 min)
                     │   ⚡ 10KB, 400 lines    │
                     └────────────────────────┘
                                  ↓
                      ┌──────────┴──────────┐
                      │                     │
         ┌────────────▼────────┐    ┌──────▼────────────────┐
         │ LMCACHE_SCENARIOS.md│    │ Need quick reference? │
         │ 🔍 14KB, 500 lines  │    │ → README.md (13KB)    │
         │                     │    └───────────────────────┘
         │ Compare:            │
         │ • Mode A (In-Mem)   │
         │ • Mode B (Storage)  │
         └─────────────────────┘
                      ↓
         ┌────────────┴────────────────┐
         │                             │
   ┌─────▼──────────────┐    ┌─────────▼────────────────┐
   │  Mode A Chosen     │    │  Mode B Chosen           │
   │  (In-Memory)       │    │  (Storage-Backed)        │
   └────────────────────┘    └──────────────────────────┘
         │                             │
         └──────────────┬──────────────┘
                        ↓
         ┌──────────────────────────────┐
         │  vllm_gaudi3_recipe.md       │ ← Complete Guide
         │  📚 67KB, 2300 lines          │
         │                              │
         │  Sections:                   │
         │  1. Introduction             │
         │  2. Hardware Overview        │
         │  3. Installation             │
         │  4. Context Scaling          │
         │  5. LMCache Integration      │
         │  6. Disaggregated Arch       │
         │  7. Testing                  │
         │  8. Benchmarking             │
         │  9. Troubleshooting          │
         │  10. Production Deployment   │
         └──────────────────────────────┘
                        ↓
         ┌──────────────┴──────────────┐
         │                             │
   ┌─────▼─────────────┐    ┌──────────▼──────────────────┐
   │ Need to apply     │    │ Disaggregated architecture? │
   │ HPU patch?        │    │ MI300X + Gaudi3?            │
   │                   │    └─────────────────────────────┘
   │ ↓                 │                 ↓
   │ PATCH_SUMMARY.md  │    HETEROGENEOUS_ARCHITECTURE.md
   │ 📊 7.8KB          │    🔀 12KB, 500 lines
   │                   │
   │ ↓                 │    Contents:
   │ GAUDI3_PATCH_     │    • MI300X (prefill) setup
   │ README.md         │    • Gaudi3 (decode) setup
   │ 🔧 10KB           │    • Shared storage config
   │                   │    • Cost analysis
   │ ↓                 │    • Docker Compose
   │ apply_gaudi3_     │    • Monitoring
   │ patch.sh          │
   │ 🚀 6.3KB (exec)   │
   └───────────────────┘


═══════════════════════════════════════════════════════════════════

                      SUPPORTING DOCUMENTS

┌─────────────────────────────────────────────────────────────────┐
│  vllm_v1_adapter_gaudi3.patch                                   │
│  🔧 6.7KB, 169 lines                                             │
│                                                                 │
│  Purpose: CUDA → HPU conversion patch                           │
│  Changes: 5 critical code locations                             │
│  Status: ✅ Verified with patch & git apply                      │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│  COMPLETION_SUMMARY.md                                          │
│  ✅ 11KB, 500 lines                                              │
│                                                                 │
│  Purpose: Final deliverables checklist                          │
│  Contents: Status, metrics, validation                          │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│  ROCM_CUDA_COMPATIBILITY.md                                     │
│  🔬 15KB, 400 lines                                              │
│                                                                 │
│  Purpose: Explain MI300X CUDA API compatibility                 │
│  Contents: ROCm design, API mappings, verification              │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│  DEEP_VERIFICATION.md                                           │
│  ✅ 18KB, 500 lines                                              │
│                                                                 │
│  Purpose: Comprehensive accuracy audit                          │
│  Contents: Claim verification, source code analysis             │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│  Additional Files                                               │
│  • README_RECIPES.md (7.1KB) - Legacy overview                  │
│  • LMCACHE_GAUDI3_PORT.md (23KB) - Porting notes               │
│  • PATCH_VERIFICATION.md (7.6KB) - Patch validation            │
│  • vllm_mi300x_recipe.md (69KB) - MI300X reference             │
└─────────────────────────────────────────────────────────────────┘


═══════════════════════════════════════════════════════════════════

                      DECISION FLOWCHART

Start: Need to run vLLM on Gaudi3
  │
  ├─ Development/Testing? ────────> Mode A (In-Memory)
  │                                      │
  │                                      ├─ Read: QUICK_START.md §3A
  │                                      ├─ Apply: patch
  │                                      └─ Deploy: Single node
  │
  ├─ Production (single hardware)? ──> Mode B (Storage)
  │                                      │
  │                                      ├─ Read: LMCACHE_SCENARIOS.md
  │                                      ├─ Read: vllm_gaudi3_recipe.md
  │                                      ├─ Apply: patch
  │                                      ├─ Configure: VAST/NFS
  │                                      └─ Deploy: K8s
  │
  └─ Disaggregated MI300X+Gaudi3? ───> Mode B (REQUIRED)
                                         │
                                         ├─ Read: HETEROGENEOUS_ARCHITECTURE.md
                                         ├─ Setup: MI300X cluster (CUDA)
                                         ├─ Setup: Gaudi3 cluster (HPU+patch)
                                         ├─ Configure: Shared storage
                                         └─ Deploy: Multi-cluster


═══════════════════════════════════════════════════════════════════

                      READING TIME ESTIMATES

Quick Start Path (45 minutes):
  1. QUICK_START.md .................... 15 min
  2. Apply patch (automated) ........... 10 min
  3. Run validation tests .............. 10 min
  4. First inference test .............. 10 min

Production Path (2-3 hours):
  1. QUICK_START.md .................... 15 min
  2. LMCACHE_SCENARIOS.md .............. 20 min
  3. vllm_gaudi3_recipe.md (scan) ...... 30 min
  4. Apply patch ....................... 10 min
  5. Configure storage ................. 30 min
  6. Deploy to K8s ..................... 45 min
  7. Validation & testing .............. 30 min

Disaggregated Path (4-6 hours):
  1. HETEROGENEOUS_ARCHITECTURE.md ..... 30 min
  2. Plan architecture ................. 60 min
  3. Setup MI300X cluster .............. 60 min
  4. Setup Gaudi3 cluster (w/patch) .... 60 min
  5. Configure shared storage .......... 45 min
  6. Deploy orchestration .............. 45 min
  7. End-to-end testing ................ 60 min


═══════════════════════════════════════════════════════════════════

                      FILE SIZE SUMMARY

Total Documentation: ~270KB across 14 files

Largest Files:
  1. vllm_mi300x_recipe.md ......... 69KB (reference)
  2. vllm_gaudi3_recipe.md ......... 67KB (main guide)
  3. LMCACHE_GAUDI3_PORT.md ........ 23KB (porting notes)
  4. LMCACHE_SCENARIOS.md .......... 14KB (comparison)
  5. README.md ..................... 13KB (index)

Most Important (Start Here):
  1. QUICK_START.md ................ 10KB ⭐⭐⭐⭐⭐
  2. LMCACHE_SCENARIOS.md .......... 14KB ⭐⭐⭐⭐
  3. vllm_gaudi3_recipe.md ......... 67KB ⭐⭐⭐⭐⭐
  4. HETEROGENEOUS_ARCHITECTURE.md . 12KB ⭐⭐⭐


═══════════════════════════════════════════════════════════════════

                      QUICK REFERENCE

Mode A (In-Memory):
  • Setup: QUICK_START.md §3A
  • Use case: Dev, test, unique prompts
  • Cache: Ephemeral (lost on restart)
  • Storage: Not needed

Mode B (Storage):
  • Setup: QUICK_START.md §3B
  • Use case: Production, repeated prompts
  • Cache: Persistent (survives restarts)
  • Storage: VAST/NFS/S3 required

Disaggregated:
  • Setup: HETEROGENEOUS_ARCHITECTURE.md
  • Use case: Hyperscale, mixed hardware
  • Cache: Mode B mandatory
  • Storage: Shared tier (connects clusters)

Apply Patch:
  • Automated: ./apply_gaudi3_patch.sh
  • Manual: See GAUDI3_PATCH_README.md
  • Verify: python -c "from lmcache..."


═══════════════════════════════════════════════════════════════════
