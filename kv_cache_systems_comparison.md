# KV Cache Management Systems Comparison
## LMCache vs llm-d-kv-cache vs NVIDIA Dynamo KVBM

**Date:** January 13, 2026  
**Analysis:** Comparing three approaches to distributed KV cache management for LLM inference  
**Updated:** Aligned with actual codebase implementation (see [CODE_REVIEW_vs_COMPARISON.md](CODE_REVIEW_vs_COMPARISON.md) for detailed verification)

---

## Executive Summary

These are **three distinct approaches** to the same problem: managing KV cache across distributed LLM inference workloads.

| System | Layer | Primary Focus | Origin |
|--------|-------|---------------|--------|
| **LMCache** | Storage Extension | Cross-session KV reuse, multi-tier caching | University of Chicago (2024) |
| **llm-d-kv-cache** | Routing/Orchestration | Cache-aware request routing across vLLM fleet | Red Hat + Partners (2025) |
| **NVIDIA Dynamo KVBM** | Infrastructure Framework | Full-stack distributed inference with memory management | NVIDIA (2025) |

**Key Insight:** They can work together, not compete:
- **LMCache** provides the storage layer (where KV blocks live)
- **llm-d-kv-cache** provides the routing logic (which pod to send requests to)
- **Dynamo KVBM** provides the orchestration framework (how to manage the entire system)

In fact, **Dynamo uses LMCache** as its storage backend, and **llm-d uses LMCache** as default KV cache layer.

---

## 1. Adoption and Maturity Assessment

### 1.1 LMCache: Production-Validated Storage Layer (Mature)

**Project Status:** Mature (1.5+ years, production-proven)  
**Launch:** June 2024 (University of Chicago research)  
**Governance:** PyTorch Foundation (announced October 2025)  
**Community:**
- 5,000+ GitHub stars (August 2025 milestone)
- Bi-weekly office hours
- Company: Tensormesh (founded by LMCache team)

**Production Adopters:**

**Cloud Providers:**
- **GMI Cloud:** Production deployment (blog post)
- **Google Cloud:** Official integration
- **CoreWeave:** Production use with Cohere

**Infrastructure Partners:**
- **Redis:** Official KV cache backend integration
- **WEKA:** Storage partner (270 GB/s throughput)
- **VAST Data:** Deep collaboration with NVIDIA NIXL
- **PliOps:** Storage offloading partner

**Enterprise Use Cases:**
- **Financial company:** Custom pinning API for frequent documents (unnamed, NDA)
- **Agent company:** KV compression and cross-node transfer (unnamed, NDA)
- **Cohere:** Production RAG workloads with CoreWeave

**Framework Integration:**
- **vLLM:** Native KVConnector support
- **vLLM Production Stack:** Default KV cache layer
- **SGLang:** Integration in progress
- **NVIDIA Dynamo:** Integrated as storage backend (September 2025)
- **llm-d:** Default KV cache layer
- **KServe:** Native support (PR merged)

**Hardware Support:**
- **NVIDIA:** Production (CUDA, cuFile, GPUDirect Storage)
- **AMD:** In review (ROCm support PR)
- **Huawei Ascend:** In review (NPU-specific kernels)
- **ARM:** In review (ARM64 architecture)

**Maturity Indicators:**
- **Battle-tested:** 3-10x TTFT improvements in production
- **Release cadence:** Rapid iteration (v0.3.6 → v0.3.9 in months)
- **Enterprise features:** Plugin framework, observability, multi-tier caching
- **Academic backing:** Multiple SIGCOMM/EuroSys papers
- **Commercial support:** Tensormesh company formation

**Key Metrics:**
- Block size: 256 tokens (vs vLLM's 16)
- Storage tiers: GPU, CPU, Disk, S3, Remote nodes
- Performance: 17x TTFT reduction (IBM+Supermicro benchmark)

### 1.2 llm-d-kv-cache: Emerging Routing & Orchestration Layer (Early Adoption)

**Project Status:** Early Adoption (8 months old, active development)  
**Launch:** May 20, 2025 at Red Hat Summit  
**Governance:** Community-led, Red Hat sponsored  
**Community:**
- 94 GitHub stars (January 2026)
- Bi-weekly office hours
- Active SIG participation
**Code Maturity:** Core components fully implemented; roadmap items (Q1-Q2 2026) in progress

**Founding Contributors:**
- Red Hat (lead)
- Google Cloud
- IBM Research
- NVIDIA
- CoreWeave

**Partners:**
- AMD, Cisco, Hugging Face, Intel, Lambda Labs, Mistral AI

**Production Deployments:**
- **Red Hat OpenShift AI:** Native integration
- **IBM Cloud:** OpenShift deployments
- **Google GKE:** Beta (Inference Gateway integration)
- **CoreWeave:** Testing/validation

**Framework Integration:**
- **vLLM:** Event-driven integration (ZMQ, KVEvents)
- **LMCache:** Default storage layer (optional Redis/Valkey backends)

**Maturity Indicators:**
- **Release cadence:** Quarterly (v0.3 Oct 2025, v0.4 Dec 2025)
- **Hardware validation:** NVIDIA (production), Intel XPU (CI), Google TPU (CI), AMD (testing)
- **Benchmarks:** 87% cache hit rate, 88% TTFT improvement
- **Enterprise readiness:** Helm charts, K8s operators, observability

**Key Differentiators:**
- **Layer:** Routing/orchestration (external to vLLM, gRPC-based)
- **Kubernetes-deployable:** Runs well on Kubernetes; K8s-native features (CRDs, operators) planned for v1.0
- **Vendor-neutral:** Works with NVIDIA, AMD, Intel GPUs (multi-vendor validated)
- **Index-agnostic:** Pluggable index backends (Redis, Valkey, in-memory, cost-aware memory)
- **Storage-independent:** Does NOT manage KV block storage; integrates with LMCache or vLLM native storage

### 1.3 NVIDIA Dynamo KVBM: Full-Stack Framework (Production-Ready)

**Project Status:** Production-Ready (announced GTC 2025)  
**Launch:** March 2025 (GTC, general availability)  
**Governance:** NVIDIA-led open source (Apache 2.0)  
**Community:**
- Active Discord server
- Office Hours (playlist on YouTube)
- Rapid OSS releases (v0.7 December 2025)

**Commercial Offering:**
- **NVIDIA AI Enterprise:** Enterprise support (future release)
- **NVIDIA NIM:** Integration for fast deployment
- **Dynamo-Triton:** Current enterprise variant

**Production Adopters:**

**Named Users:**
- **Baseten:** 2x faster inference (October 2025 blog)
- **Run:ai:** Integration for gang scheduling (v2.23)
- **XConn Technologies + MemVerge:** CXL memory pooling demo (OCP Summit 2025)

**Cloud/Infrastructure Partners:**
- **VAST Data:** Deep NIXL/KVBM collaboration
- **AWS:** EFA support (v0.2+)
- **Storage vendors:** Targeting NetApp, Pure Storage

**Framework Integration:**
- **TensorRT-LLM:** Native (v1.1.0rc5)
- **vLLM:** Full support
- **SGLang:** Full support
- **LMCache:** Integrated as storage backend (September 2025)

**Hardware Support:**
- **NVIDIA:** Full support (H100, Blackwell GB200, GH200)
- **Primary target:** NVIDIA GPUs with NVLink, InfiniBand, Spectrum-X

**Maturity Indicators:**
- **Benchmarks:** 30x throughput increase (DeepSeek-R1 671B on GB200)
- **Benchmarks:** 2x throughput increase (Llama 70B on Hopper)
- **MLPerf:** Record-setting (Blackwell + Dynamo)
- **Release cadence:** Rapid (v0.7 in December 2025)
- **Enterprise readiness:** AI Enterprise support coming, Kubernetes operator, etcd/NATS coordination

**Key Features:**
- **Full-stack:** Frontend router + backend workers + KVBM
- **Disaggregated P/D:** Separate prefill and decode stages
- **Dynamic scheduling:** GPU orchestration based on demand
- **Multi-tier KVBM:** GPU (G1), CPU (G2), SSD/pooled (G3), S3/object (G4)
- **Built in Rust:** Core performance, Python for extensibility

**Ecosystem Position:**
- **NVIDIA-centric:** Strongest on NVIDIA hardware
- **Open but optimized:** Works elsewhere, shines with NVLink/InfiniBand
- **Commercial path:** AI Enterprise support provides enterprise adoption runway

---

## 2. Architecture and Design Philosophy

### 2.1 LMCache: Storage-Centric "Redis for KV Cache"

**Design Philosophy:** Content-addressable, multi-tier cache storage with cross-session reuse

**Architecture:**
```
┌─────────────────────────────────────────────────────────┐
│              vLLM Instance (Primary Engine)              │
│                                                           │
│  ┌───────────────────────────────────────────────────┐  │
│  │ vLLM KVConnector Interface                        │  │
│  │ (put, get, contains)                              │  │
│  └─────────────────┬─────────────────────────────────┘  │
│                    │                                     │
└────────────────────┼─────────────────────────────────────┘
                     │
          ┌──────────▼──────────┐
          │     LMCache Layer    │
          │  (Storage Manager)   │
          └──────────┬───────────┘
                     │
      ┌──────────────┼──────────────┐
      │              │               │
┌─────▼──────┐  ┌───▼────┐  ┌──────▼──────┐
│ GPU Tier   │  │ CPU    │  │ Disk/S3     │
│ (fastest)  │  │ Memory │  │ (largest)   │
└────────────┘  └────────┘  └─────────────┘
```

**Key Components:**
1. **Storage Backends:**
   - LocalCPU (CPU memory on same node)
   - LocalDisk (NVMe, via NIXL + GPUDirect Storage)
   - S3Connector (Object storage, cross-datacenter)
   - RemoteConnector (Other nodes, RDMA via NIXL)

2. **Hash-Based Deduplication:**
   - 256-token blocks (larger than vLLM's 16)
   - Content-addressable (same prompt = same hash = reuse)
   - Global across sessions and users

3. **Zero-Copy Transfers:**
   - GPU→CPU: CUDA pinned memory
   - GPU→Disk: GPUDirect Storage (cuFile, NIXL)
   - GPU→S3: Direct GPU-to-network (NIXL + S3)

4. **Compression:**
   - CacheGen compression (3x reduction)
   - FP8 quantization support

**Workload Fit:**
- ✅ Multi-turn conversations (cross-session reuse)
- ✅ RAG workloads (document caching)
- ✅ Long-context queries (offload to disk/S3)
- ✅ Multi-user shared contexts (deduplicate)

**Limitations:**
- Relies on vLLM/SGLang as primary engine
- No built-in routing (needs llm-d or similar)
- No orchestration (just storage)

### 2.2 llm-d-kv-cache: Routing-Centric "Traffic Cop"

**Design Philosophy:** Cache-aware request routing across distributed vLLM fleet

**Architecture:**
```
┌──────────────────────────────────────────────────────────┐
│                 Inference Scheduler                       │
│                   (Envoy Proxy)                           │
│                                                           │
│  ┌────────────────────────────────────────────────────┐  │
│  │  llm-d KV-Cache Indexer (Routing Brain)           │  │
│  │                                                     │  │
│  │  ┌─────────────┐  ┌─────────────┐  ┌───────────┐ │  │
│  │  │PrefixStore  │  │TokenProcessor│ │ Scorer    │ │  │
│  │  │(LRU cache)  │  │(hash blocks) │ │(rank pods)│ │  │
│  │  └─────────────┘  └─────────────┘  └───────────┘ │  │
│  │                                                     │  │
│  │  ┌──────────────────────────────────────────────┐ │  │
│  │  │     kvblock.Index (Global KV Block Map)      │ │  │
│  │  │     (in-memory, Redis, or Valkey)            │ │  │
│  │  │     block_hash → [pod_ids]                   │ │  │
│  │  └──────────────────────────────────────────────┘ │  │
│  └────────────────────────────────────────────────────┘  │
└───────────────────────┬──────────────────────────────────┘
                        │ Routes to best pod
         ┌──────────────┼──────────────┐
         │              │               │
    ┌────▼────┐    ┌───▼────┐    ┌────▼────┐
    │vLLM Pod1│    │vLLM Pod2│   │vLLM Pod3│
    │(+LMCache)│   │(+LMCache)│  │(+LMCache)│
    └─────┬───┘    └────┬────┘   └────┬────┘
          │             │              │
          └─────────────┼──────────────┘
                        │ KVEvents (ZMQ)
                  ┌─────▼──────┐
                  │Event Pool  │
                  │(updates    │
                  │ Index)     │
                  └────────────┘
```

**Key Components:**
1. **PrefixStore:** Token-level LRU cache (avoids re-tokenization)
2. **TokenProcessor:** Converts prompts to KV block keys (matches vLLM hashing)
3. **kvblock.Index:** Global map of which pods have which blocks
4. **Scorer:** Ranks pods by consecutive block matches
5. **Event Pool:** ZMQ subscriber for vLLM KVEvents (BlockStored/BlockRemoved)

**Pluggable Backends:**
- In-memory (default, fastest, single-node indexer)
- Redis (persistent, distributed)
- Valkey (Redis-compatible + RDMA for ultra-low latency)

**Workload Fit:**
- ✅ Distributed vLLM deployments (Kubernetes)
- ✅ Conversational AI (session affinity routing)
- ✅ High cache hit rate required (routing optimization)
- ✅ Load + cache-aware scheduling

**Limitations:**
- Requires Kubernetes (not standalone)
- Requires vLLM with KVEvents support
- Does not handle storage (delegates to LMCache or vLLM native)
- Routing-only (no orchestration of disaggregated P/D)

### 2.3 NVIDIA Dynamo KVBM: Full-Stack "AI Factory"

**Design Philosophy:** Complete distributed inference orchestration with intelligent memory management

**Architecture:**
```
┌────────────────────────────────────────────────────────────┐
│                  Dynamo Frontend (Rust)                     │
│  - OpenAI-compatible API                                    │
│  - Request routing (KV-aware, prefill vs decode)           │
│  - Dynamic GPU scheduling                                   │
└────────────┬───────────────────────────────────────────────┘
             │
    ┌────────┼────────┐
    │        │         │
┌───▼────┐ ┌▼────────┐ ┌▼───────────┐
│Prefill │ │ Decode  │ │  Decode    │
│Worker  │ │ Worker  │ │  Worker    │
│(TRT-LLM│ │ (vLLM)  │ │  (SGLang)  │
└───┬────┘ └┬────────┘ └┬───────────┘
    │       │            │
    └───────┼────────────┘
            │
    ┌───────▼────────┐
    │  KVBM (Core)   │
    │  (Rust-based)  │
    │                │
    │  ┌──────────┐  │
    │  │G1: Device│  │  ← GPU HBM (hot, fastest)
    │  │  Memory  │  │
    │  └──────────┘  │
    │  ┌──────────┐  │
    │  │G2: CPU   │  │  ← CPU memory (warm, staging)
    │  │  Memory  │  │
    │  └──────────┘  │
    │  ┌──────────┐  │
    │  │G3: Local/│  │  ← NVMe, pooled SSD (cold cache)
    │  │   Pooled │  │     CXL memory pools
    │  │   SSD    │  │     VAST, WEKA, NetApp
    │  └──────────┘  │
    │  ┌──────────┐  │
    │  │G4: Remote│  │  ← S3, object storage (archival)
    │  │  Storage │  │
    │  └──────────┘  │
    └────────────────┘
           │
    ┌──────▼──────┐
    │   NIXL      │  ← GPU-to-Storage transfers
    │ (RDMA/GDS)  │     (GPUDirect, InfiniBand)
    └─────────────┘
```

**Key Components:**
1. **Frontend (Rust):** Request routing, scheduling, OpenAI-compatible API
2. **Grove API:** Kubernetes-native orchestration (declarative multi-node deployment)
3. **KVBM (KV Block Manager):**
   - Four-tier memory hierarchy (G1-G4)
   - Ownership-driven lifecycle management
   - Asynchronous eviction, latency-critical retrieval
   - Deduplication by sequence hash

4. **NIXL Integration:** GPU-to-storage transfers (RDMA, GPUDirect Storage)
5. **LMCache Integration:** Optional storage backend for G3/G4 tiers
6. **Coordination:** etcd (cluster state), NATS (messaging)

**Disaggregated Serving:**
- Prefill workers: Handle prompt processing
- Decode workers: Handle token generation
- Independent scaling based on workload mix

**Workload Fit:**
- ✅ Datacenter-scale deployments (1000s of GPUs)
- ✅ Reasoning models (DeepSeek-R1, high throughput)
- ✅ Heterogeneous hardware (mixed GPU types, CXL pools)
- ✅ SLA-driven routing (latency vs throughput optimization)

**Limitations:**
- NVIDIA-optimized (best on NVLink, InfiniBand, Spectrum-X)
- Complex setup (etcd, NATS, Grove, KVBM, multiple workers)
- Python 3.12 only for KVBM (Ubuntu 24.04)
- Rapid API churn (v0.7 in December 2025, interfaces evolving)

---

## 3. Feature Comparison Matrix

| Feature | LMCache | llm-d-kv-cache | Dynamo KVBM |
|---------|---------|----------------|-------------|
| **Primary Role** | Storage Layer | Routing Layer | Full Infrastructure |
| **Scope** | Single-instance storage | Multi-instance routing | Datacenter orchestration |
| **Deployment Model** | vLLM plugin | External Kubernetes service | Standalone framework |
| **KV Cache Storage** ||||
| GPU memory | Via vLLM | N/A (routing only) | G1 tier (managed) |
| CPU memory | Yes (LocalCPU) | N/A | G2 tier (managed) |
| Local NVMe/SSD | Yes (NIXL/GDS) | N/A | G3 tier (managed) |
| S3/Object storage | Yes (S3Connector) | N/A | G4 tier (managed) |
| Remote nodes (RDMA) | Yes (RemoteConnector) | N/A | Cross-worker (NIXL) |
| **Routing & Scheduling** ||||
| Cache-aware routing | No (storage only) | Yes (primary feature) | Yes (built-in) |
| Score-based pod selection | No | Yes (LongestPrefix scoring) | Yes (disaggregated) |
| Pluggable scoring strategies | No | Yes (extensible) | Yes |
| Prefill/Decode split | No | No | Yes (disaggregated) |
| Dynamic GPU scheduling | No | No | Yes |
| SLA-driven placement | No | No | Yes (latency/throughput) |
| **Integration** ||||
| vLLM | Yes (KVConnector) | Yes (KVEvents) | Yes (full support) |
| TensorRT-LLM | No | No | Yes (native) |
| SGLang | In progress | No | Yes (full support) |
| Kubernetes-native | No (runs in pods) | Yes (requires K8s) | Yes (Grove API) |
| **Storage Backends** ||||
| In-memory/local | Yes | Yes (index only) | Yes (G1/G2 tiers) |
| Redis | Yes | Yes (index backend) | No (etcd for state) |
| Valkey (RDMA) | No | Yes (index backend) | No |
| VAST/WEKA/NetApp | Via NIXL | N/A | Yes (G3 tier targets) |
| CXL memory pools | No | N/A | Yes (G3 tier, demo'd) |
| **Hardware Support** ||||
| NVIDIA GPUs | Yes (production) | Yes (production) | Yes (primary target) |
| AMD GPUs | In review (PR) | Yes (tested) | Limited (focus NVIDIA) |
| Intel Gaudi/XPU | In review (PR) | Yes (CI validated) | Unknown |
| Google TPU | No | Yes (CI validated) | No |
| ARM architecture | In review (PR) | Unknown | No (x86 only) |
| **Advanced Features** ||||
| Compression | Yes (CacheGen 3x) | No (index only) | No |
| Cross-session reuse | Yes (storage) | Depends on backend | No (per-request) |
| Content deduplication | Yes (hash-based) | Via index (depends on backend) | Yes (KVBM hash) |
| Cross-node caching | No (single node) | Planned Q1-Q2 2026 | Yes (RDMA/NIXL) |
| Multi-model serving | Via vLLM | Via vLLM | Yes (native) |
| **Observability** ||||
| Metrics | Plugin framework | KVEvents tracking | KVBM metrics/events |
| Dashboard | No (bring your own) | No (bring your own) | Prometheus/Grafana |
| Debugging tools | lmcache_frontend | Event stream | KVBM debug hooks |
| **Maturity** ||||
| Project age | 1.5 years | 8 months | 10 months |
| Production deployments | 10+ named users | 4 founding contributors (Red Hat, Google, IBM, NVIDIA, CoreWeave) | 2+ named users |
| Core implementation | Fully mature (v0.3.x) | Fully implemented, roadmap in progress | Production-ready (v0.7+) |
| API stability | Stable | Pre-v1.0 (changes expected) | Pre-v1.0 (rapid evolution) |
| Enterprise support | Tensormesh (commercial) | Red Hat (OpenShift AI), community | NVIDIA AI Enterprise (coming) |
| Community size | 5,000+ stars | 94 stars (newer) | Active Discord |
| **Performance Claims** ||||
| TTFT improvement | 3-10x (cross-session reuse) | 88% TTFT improvement (87% cache hit rate) | 30x (DeepSeek-R1 reasoning) |
| Key metric | Storage efficiency | Routing decision quality | System throughput |
| Block size | 256 tokens | Configurable (via vLLM) | Configurable (16-32) |
| Latency focus | Storage tier latency | Scoring latency <10ms | Disaggregated serving |

---

## 4. Integration Patterns

### 4.1 LMCache Standalone (with vLLM)

**Use case:** Single vLLM deployment with multi-tier caching

```yaml
# vLLM + LMCache deployment
apiVersion: apps/v1
kind: Deployment
spec:
  template:
    spec:
      containers:
      - name: vllm
        image: vllm/vllm:latest
        args:
        - --model=meta-llama/Llama-3.1-70B-Instruct
        - --kv-connector-backend=lmcache
        - --kv-offloading-backend=lmcache
        - --kv-offloading-size=100  # GiB to offload
        env:
        - name: LMCACHE_LOCAL_CPU_SIZE
          value: "50G"  # CPU tier
        - name: LMCACHE_LOCAL_DISK_PATH
          value: "/data/kv-cache"  # Disk tier
        - name: LMCACHE_S3_BUCKET
          value: "s3://my-kv-cache"  # S3 tier
        volumeMounts:
        - name: nvme
          mountPath: /data/kv-cache
```

**Benefits:**
- Cross-session KV reuse
- Multi-tier caching (GPU → CPU → Disk → S3)
- Works with single vLLM instance

**Limitations:**
- No cross-pod routing
- No disaggregated P/D
- Storage-only solution

### 4.2 llm-d-kv-cache + vLLM + LMCache (Recommended Kubernetes Stack)

**Use case:** Distributed vLLM fleet with cache-aware routing and multi-tier storage

```yaml
# llm-d Helm chart (simplified)
vllm:
  replicaCount: 5
  args:
  - --enable-prefix-caching
  - --kv-connector-backend=lmcache  # Storage layer
  - --kv-offloading-backend=lmcache

inferenceScheduler:
  enabled: true
  kvcache:
    enabled: true
    backend: valkey  # or redis, in-memory
    scoring: consecutive-prefix

lmcache:
  enabled: true
  localCPU:
    size: 50G
  localDisk:
    enabled: true
    path: /data/kv-cache
```

**Data flow:**
```
1. Client request → llm-d Scheduler (external routing decision)
2. Scheduler calls llm-d-kv-cache Indexer via gRPC: GetPodScores(prompt, pods)
3. Indexer pipeline:
   - Tokenizer → PrefixStore (check for cached tokens)
   - TokenProcessor → Convert tokens to KV block keys
   - kvblock.Index → Lookup which pods have those blocks
   - Scorer → Rank pods by consecutive block matches (LongestPrefix strategy)
4. Indexer returns scores: {"pod-1": 95.0, "pod-2": 42.0, "pod-3": 0.0}
5. Scheduler routes to pod-1 (highest score)
6. Pod-1 (vLLM + LMCache): 
   - Checks local GPU cache (vLLM)
   - Checks LMCache (CPU/Disk/S3 tiers)
   - Loads missing blocks from remote KV storage
7. Pod-1 emits KVEvents (ZMQ) → kvevents.Pool processes → Index updates
```

**Benefits:**
- 87% cache hit rate (vs 18% with round-robin load balancing)
- Cross-session reuse via LMCache storage
- Intelligent cache-aware routing via llm-d-kv-cache
- Kubernetes-deployable (standard Helm charts)
- Vendor-neutral (NVIDIA, AMD, Intel GPUs validated)
- Multi-tier storage (GPU → CPU → Disk → S3)

**Limitations:**
- Requires Kubernetes cluster
- Requires vLLM with KVEvents support (ZMQ endpoint)
- Does not manage KV block storage (delegates to LMCache or vLLM native)
- Routing-only (no disaggregated prefill/decode like Dynamo)
- Index backends must be managed separately (Redis/Valkey if distributed)

### 4.3 Dynamo KVBM + LMCache (Full-Stack)

**Use case:** Datacenter-scale disaggregated inference

```bash
# Dynamo with LMCache storage backend
dynamo-frontend \
  --store-kv etcd \
  --kvbm-enable \
  --kvbm-lmcache-backend \
  --kvbm-g3-tier vast  # or weka, netapp

dynamo-worker \
  --engine vllm \
  --model meta-llama/Llama-3.1-70B-Instruct \
  --role prefill \
  --kvbm-enable

dynamo-worker \
  --engine vllm \
  --model meta-llama/Llama-3.1-70B-Instruct \
  --role decode \
  --kvbm-enable
```

**Data flow:**
```
1. Client request → Dynamo Frontend
2. Frontend: Routes to Prefill Worker (KV-aware)
3. Prefill Worker: Processes prompt, writes KV to KVBM
4. KVBM: Stores in G1 (GPU), evicts to G2 (CPU) → G3 (VAST/LMCache)
5. Frontend: Routes decode requests to Decode Worker
6. Decode Worker: Reads KV from KVBM (G1/G2/G3 as needed)
7. KVBM: Async fetches from G3 (LMCache) if not in G1/G2
```

**Benefits:**
- Datacenter-scale orchestration
- Disaggregated P/D (throughput optimization)
- Four-tier memory hierarchy
- NVIDIA ecosystem integration

**Limitations:**
- NVIDIA-optimized (less portable)
- Complex setup (etcd, NATS, multiple workers)
- Evolving APIs (v0.7, pre-v1.0)

---

## 5. Hardware and Platform Support

### 5.1 LMCache: Expanding Multi-Vendor

| Platform | Status | Notes |
|----------|--------|-------|
| NVIDIA GPUs | ✅ Production | CUDA, cuFile, GPUDirect Storage |
| AMD MI300X | 🔄 In Review | ROCm support PR active |
| Intel Gaudi | 🔄 In Review | NPU-specific kernels PR active |
| Huawei Ascend | 🔄 In Review | torch_npu backend integration |
| ARM64 | 🔄 In Review | Architecture support PR |
| Google TPU | ❌ Not Planned | vLLM doesn't support TPU + LMCache |

**Operating Systems:**
- Linux: ✅ Production
- Windows: ❌ Not supported
- MacOS: ❌ Not supported

**Storage Targets:**
- Local NVMe: ✅ Production (NIXL/GDS)
- Network filesystems: ✅ Production (NFS, Lustre, GPFS via NIXL)
- Object storage: ✅ Production (S3, compatible)
- VAST Data: ✅ Production (collaboration)
- WEKA: ✅ Production (270 GB/s)
- Redis: ✅ Production (remote backend)

### 5.2 llm-d-kv-cache: Accelerator-Agnostic Index Layer

**Note:** llm-d-kv-cache is hardware-agnostic (it's a routing/index layer, not a compute framework). Support depends on vLLM's backend.

| Platform | Status | Notes |
|----------|--------|-------|
| NVIDIA GPUs | ✅ Production | Full support via vLLM backend |
| AMD MI300X | ✅ Tested | Validated via vLLM ROCm |
| Intel Gaudi/XPU | ✅ CI Validated | Validated via vLLM Intel backend |
| Google TPU | ✅ CI Validated | Validated via vLLM JAX backend |
| ARM64 | ✅ Compatible | No GPU compute, but indexer runs anywhere |
| x86-64 | ✅ Primary | Standard deployment architecture |

**Kubernetes Distributions:**
- Red Hat OpenShift: ✅ Production (native integration, OpenShift AI)
- Google GKE: ✅ Compatible (standard deployment)
- AWS EKS: ✅ Compatible (standard Helm charts)
- Azure AKS: ✅ Compatible (standard Helm charts)
- On-premises K8s: ✅ Compatible (tested)
- Kind/Minikube: ✅ Development (examples provided)

**Storage Backends (for Index):**
- In-memory: ✅ Production (default)
- Redis: ✅ Production (distributed)
- Valkey: ✅ Production (RDMA-enabled)

### 5.3 NVIDIA Dynamo KVBM: NVIDIA-Optimized

| Platform | Status | Notes |
|----------|--------|-------|
| NVIDIA H100 | ✅ Production | Primary target |
| NVIDIA GH200 | ✅ Production | Grace-Hopper |
| NVIDIA GB200 | ✅ Production | Blackwell (30x claims) |
| NVIDIA Spectrum-X | ✅ Production | Networking fabric |
| AMD GPUs | ⚠️ Limited | Focus is NVIDIA |
| Intel Gaudi | ❓ Unknown | Not documented |
| Google TPU | ❌ Not Supported | NVIDIA-centric |

**Operating Systems:**
- Ubuntu 24.04: ✅ Production (Python 3.12 for KVBM)
- Other Linux: ⚠️ Limited (Python 3.11 TensorRT-LLM issues)

**Networking:**
- NVLink: ✅ Production (intra-node GPU transfers)
- InfiniBand: ✅ Production (inter-node, RDMA)
- RoCE: ✅ Production (Ethernet RDMA)
- AWS EFA: ✅ Supported (v0.2+)
- CXL Memory: ✅ Demo'd (XConn + MemVerge)

**Storage Targets (G3 Tier):**
- Local NVMe: ✅ Production (NIXL/GDS)
- VAST Data: ✅ Collaboration (deep integration)
- WEKA: ✅ Target (storage partner)
- NetApp: ✅ Target (future)
- LMCache: ✅ Integrated (September 2025)

---

## 6. Performance Benchmarks

### 6.1 LMCache Performance Data

**ShareGPT Multi-Turn Conversations:**
- Workload: 200 users, 5+ rounds each, 2 rounds context
- TTFT reduction: 3-10x improvement
- ITL reduction: Significant (exact numbers vary by model)

**Document Analysis (Chat-Bot):**
- Input: 20K tokens per query
- Output: 100 tokens per query
- TTFT: Minimal (cache hit = skip prefill)
- Concurrency: High throughput under load

**IBM + Supermicro Tiered Caching:**
- Setup: Gaudi3, tiered L1/L2/L3 cache
- Result: 17x TTFT reduction
- Collaboration: Intel NIXL + LMCache

**Cohere + CoreWeave RAG:**
- Workload: Enterprise RAG with document pinning
- Storage: VAST Data via LMCache
- Result: "Breaking the Memory Barrier" (exact numbers NDA)

### 6.2 llm-d-kv-cache Performance Data

**Red Hat Blog Benchmarks:**
- Cache hit rate: 87% (vs 18% round-robin)
- TTFT improvement: 88%
- Setup: 5 vLLM pods, conversational workload

**Detailed Metrics (from blog):**
- P50 TTFT: 0.22s (cache-aware) vs 1.8s (round-robin)
- P95 TTFT: 0.45s (cache-aware) vs 3.2s (round-robin)
- Throughput: 1200 tok/s (cache-aware) vs 850 tok/s (round-robin)

**Setup:**
- Model: Llama-3.1-8B
- Workload: Multi-turn conversations
- Infrastructure: Kubernetes, 5 replicas

### 6.3 NVIDIA Dynamo KVBM Performance Data

**MLPerf Inference Benchmarks:**
- Blackwell + Dynamo: Record-setting results
- Date: September 2025 announcement

**DeepSeek-R1 671B on GB200 NVL72:**
- Throughput increase: **30x** (tokens/sec/GPU)
- Setup: Disaggregated serving
- Date: GTC 2025 announcement

**Llama 70B on Hopper:**
- Throughput increase: **2x**
- Setup: Compared to vanilla vLLM
- Date: GTC 2025 announcement

**Baseten Production:**
- Inference speedup: **2x**
- Date: October 2025 blog post
- Customer: Named production user

**VAST Data Collaboration:**
- Prefill time improvement: **10x** (large-context)
- Setup: NIXL + GPUDirect Storage + KVBM
- Storage: VAST Data platform

### 6.4 Performance Comparison Notes

**Important caveats:**
1. **Different baselines:** 
   - LMCache: vs no caching
   - llm-d: vs round-robin load balancing
   - Dynamo: vs vanilla vLLM or other frameworks

2. **Different workloads:**
   - LMCache: Cross-session, long-context
   - llm-d: Distributed fleet, cache locality
   - Dynamo: Reasoning models, disaggregated P/D

3. **Different hardware:**
   - LMCache: H100, Gaudi3, MI300X (varies)
   - llm-d: Generic GPU clusters
   - Dynamo: Blackwell GB200 (flagship), Hopper

**Conclusion:** All three deliver significant improvements, but in different scenarios. Not directly comparable.

---

## 7. Use Case Recommendations

### 7.1 Decision Matrix

| Use Case | Best Choice | Why |
|----------|-------------|-----|
| **Single vLLM instance, long contexts** | LMCache | Multi-tier caching, S3 offload |
| **RAG with document caching** | LMCache | Content-addressable, cross-session reuse |
| **Kubernetes multi-replica vLLM** | llm-d-kv-cache + LMCache | Cache-aware routing + storage |
| **Conversational AI (multi-turn)** | llm-d-kv-cache + LMCache | Session affinity + cross-session reuse |
| **Red Hat OpenShift AI** | llm-d-kv-cache + LMCache | Native integration, supported path |
| **Datacenter-scale (1000s GPUs)** | Dynamo KVBM (+LMCache) | Full orchestration, disaggregated P/D |
| **Reasoning models (DeepSeek-R1)** | Dynamo KVBM | Optimized for high-throughput reasoning |
| **NVIDIA-only infrastructure** | Dynamo KVBM | Best hardware integration |
| **Multi-vendor GPUs (NVIDIA+AMD+Intel)** | llm-d-kv-cache + LMCache | Vendor-neutral, validated on all |
| **Prototype/Research** | LMCache standalone | Simple, well-documented, flexible |
| **Enterprise with NVIDIA AI Enterprise** | Dynamo KVBM (future) | Commercial support coming |

### 7.2 Adoption Recommendations by Organization

**Startups / Small Teams:**
- **Start with:** LMCache + vLLM (single deployment)
- **Add later:** llm-d-kv-cache when scaling to 5+ replicas
- **Skip:** Dynamo (too complex for small scale)

**Mid-Sized Companies:**
- **Kubernetes-based:** llm-d-kv-cache + LMCache (cache-aware routing)
- **On-premises + cloud:** LMCache (portable across environments)
- **Evaluate:** Dynamo if NVIDIA-only and planning large scale

**Enterprises:**
- **Red Hat customers:** llm-d-kv-cache + LMCache (supported via OpenShift AI)
- **NVIDIA customers:** Dynamo KVBM (AI Enterprise support coming)
- **Multi-vendor:** llm-d-kv-cache + LMCache (vendor-neutral)

**Cloud Providers / GPU Clouds:**
- **General offering:** LMCache (works with vLLM, SGLang)
- **Managed K8s:** llm-d-kv-cache (differentiation for distributed workloads)
- **NVIDIA-optimized tier:** Dynamo KVBM (premium offering)

---

## 8. Integration and Compatibility

### 8.1 Can They Work Together?

**Yes! They're complementary:**

```
┌─────────────────────────────────────────────────────────┐
│              Application / LLM Service                   │
└───────────────────────┬─────────────────────────────────┘
                        │
┌───────────────────────▼─────────────────────────────────┐
│       llm-d-kv-cache (Routing Layer)                    │
│       "Which pod should serve this request?"            │
└───────────────────────┬─────────────────────────────────┘
                        │
        ┌───────────────┼───────────────┐
        │               │               │
┌───────▼──────┐ ┌──────▼──────┐ ┌─────▼───────┐
│ vLLM Pod 1   │ │ vLLM Pod 2  │ │ vLLM Pod 3  │
│              │ │             │ │             │
│ ┌──────────┐ │ │ ┌─────────┐ │ │ ┌─────────┐ │
│ │ LMCache  │ │ │ │ LMCache │ │ │ │ LMCache │ │
│ │(Storage) │ │ │ │(Storage)│ │ │ │(Storage)│ │
│ └──────────┘ │ │ └─────────┘ │ │ └─────────┘ │
└──────────────┘ └─────────────┘ └─────────────┘
       │                │                │
       └────────────────┼────────────────┘
                        │
              ┌─────────▼──────────┐
              │ Shared KV Storage  │
              │ (Redis, S3, VAST)  │
              └────────────────────┘
```

**Or:**

```
┌─────────────────────────────────────────────────────────┐
│          Dynamo Frontend (Orchestration)                 │
└───────────────────────┬─────────────────────────────────┘
                        │
        ┌───────────────┼───────────────┐
        │               │               │
┌───────▼──────┐ ┌──────▼──────┐ ┌─────▼───────┐
│Prefill Worker│ │Decode Worker│ │Decode Worker│
│ (vLLM/TRT)   │ │  (vLLM)     │ │  (SGLang)   │
└──────┬───────┘ └──────┬──────┘ └──────┬──────┘
       │                │                │
       └────────────────┼────────────────┘
                        │
              ┌─────────▼──────────┐
              │   Dynamo KVBM      │
              │  (Memory Manager)  │
              └─────────┬──────────┘
                        │
              ┌─────────▼──────────┐
              │   LMCache Backend  │
              │  (G3/G4 Storage)   │
              └────────────────────┘
```

### 8.2 Verified Integrations

| Integration | Status | Notes |
|-------------|--------|-------|
| **LMCache + vLLM** | ✅ Production | Native KVConnector |
| **LMCache + SGLang** | ✅ In Progress | Paged attention compatible |
| **LMCache + Dynamo** | ✅ Production | Integrated Sept 2025 |
| **llm-d + vLLM** | ✅ Production | Event-driven (ZMQ) |
| **llm-d + LMCache** | ✅ Production | Default storage layer |
| **Dynamo + vLLM** | ✅ Production | Full support |
| **Dynamo + TRT-LLM** | ✅ Production | Native support |
| **Dynamo + SGLang** | ✅ Production | Full support |
| **Dynamo + LMCache** | ✅ Production | G3/G4 backend |

**Not Compatible:**
- llm-d + Dynamo: Overlap (both provide routing/orchestration)
- llm-d + TRT-LLM: llm-d only supports vLLM events

---

## 9. Maturity and Risk Assessment

### 9.1 LMCache: Low Risk, Proven

**Strengths:**
- 1.5+ years in production
- Multiple named users (GMI Cloud, Cohere, Google Cloud)
- PyTorch Foundation (governance stability)
- Academic backing (SIGCOMM, EuroSys papers)
- Commercial support (Tensormesh)

**Risks:**
- API evolution (v0.3.x rapid releases)
- Hardware support still expanding (AMD/Intel/ARM in PR)
- Relies on vLLM/SGLang as primary engine

**Mitigation:**
- Stick to stable releases (v0.3.9+)
- Use production-validated backends (CUDA, LocalCPU, S3)
- Commercial support available via Tensormesh

### 9.2 llm-d-kv-cache: Medium Risk, Early Adoption (Roadmap in Progress)

**Strengths:**
- Strong industrial backing (Red Hat, Google, IBM, NVIDIA, CoreWeave founding contributors)
- Core components fully implemented and tested
- Vendor-neutral (NVIDIA, AMD, Intel GPUs all validated)
- Red Hat enterprise support path (OpenShift AI)
- Extensible architecture (pluggable backends, scoring strategies)
- Clear roadmap with concrete Q1-Q2 2026 deliverables

**Risks:**
- Young project (8 months old, but with 5-person founding team)
- Limited public production deployments (4 founding contributors only)
- API not yet stable (pre-v1.0, breaking changes possible)
- Smaller community adoption (94 GitHub stars)
- Roadmap items (cross-node caching, advanced scoring) not yet delivered

**Current Implementation Status:**
- ✅ Core indexer, scoring, event processing: COMPLETE
- ⚠️ K8s-native features (CRDs, operators): PLANNED for v1.0
- 🔄 Cross-node KV sharing: Q1-Q2 2026
- 🔄 Advanced scoring (ML-based): Q1-Q2 2026
- 🔄 Storage backend integrations (WEKA, VAST): Q1-Q2 2026

**Mitigation:**
- Red Hat customers: Use OpenShift AI integration (supported path)
- Non-Red Hat K8s: Stable for current features, plan for breaking changes in v1.0
- Early adopters: Report issues; community responsive
- Wait for v1.0 (target: Q2 2026) for API stability and K8s operators

### 9.3 NVIDIA Dynamo KVBM: Medium-High Risk, Cutting Edge

**Strengths:**
- NVIDIA backing (AI Enterprise support coming)
- Record-setting benchmarks (30x on GB200)
- Full-stack solution (less integration burden)
- Active development (v0.7 in Dec 2025)

**Risks:**
- NVIDIA-centric (less portable)
- Complex setup (etcd, NATS, Grove, multiple workers)
- Rapid API churn (v0.7, interfaces evolving)
- Python version constraints (3.12 only for KVBM)
- Limited public references (2 named users)

**Mitigation:**
- NVIDIA customers: Wait for AI Enterprise inclusion
- Others: Prototype but expect breaking changes
- Use Dynamo-Triton (enterprise variant) for production

### 9.4 Risk Comparison Matrix

| Dimension | LMCache | llm-d-kv-cache | Dynamo KVBM |
|-----------|---------|----------------|-------------|
| **API Stability** | Medium (evolving) | Medium (pre-v1.0) | Medium-High (rapid) |
| **Production Validation** | High (10+ users) | Medium (4 contributors) | Medium (2+ users) |
| **Community Support** | High (5K stars) | Low (94 stars) | Medium (Discord) |
| **Vendor Lock-in** | Low (multi-vendor) | Low (vendor-neutral) | Medium-High (NVIDIA) |
| **Complexity** | Low (plugin) | Medium (K8s) | High (full-stack) |
| **Documentation** | Good | Good | Good |
| **Enterprise Support** | Yes (Tensormesh) | Yes (Red Hat) | Coming (AI Enterprise) |
| **Breaking Changes Risk** | Medium | Medium | High |
| **Overall Risk** | **Low** | **Medium** | **Medium-High** |

---

## 10. Future Roadmap and Trends

### 10.1 LMCache Roadmap

**Confirmed (2025-2026):**
- Hardware expansion: AMD (ROCm), Intel (Gaudi), Huawei (Ascend), ARM64
- PyTorch Foundation membership (announced October 2025)
- Plugin framework enhancements (observability, management)
- Tensormesh commercial offerings (enterprise features)

**Likely (2026+):**
- Tighter Dynamo integration (default G3/G4 backend)
- More storage partners (NetApp, Pure Storage)
- Multi-framework support (beyond vLLM/SGLang)

### 10.2 llm-d-kv-cache Roadmap (Officially Tracked)

**Currently Implemented (v0.3-0.4):**
- ✅ Token-level LRU caching (PrefixStore)
- ✅ KV block indexing (kvblock.Index with 4 backends)
- ✅ Pod scoring (LongestPrefixMatch strategy)
- ✅ ZMQ event processing (BlockStored, BlockRemoved)
- ✅ gRPC service for remote indexing
- ✅ Prometheus metrics
- ✅ Examples (offline, online, gRPC service)

**Confirmed (Q1-Q2 2026):**
- Cross-node KV cache sharing (E/W caching with global indexing)
- Integration with WEKA, VAST, NetApp (distributed storage backends)
- Advanced scoring algorithms (ML-based pod selection framework)
- Kubernetes native features (CRDs, operators)
- v1.0 release (API stability freeze)

**Likely (2026+):**
- Broader enterprise adoption (Red Hat channel expansion)
- More storage vendor integrations
- Cost-optimized placement algorithms
- Enhanced observability (tracing, custom dashboards)

### 10.3 NVIDIA Dynamo KVBM Roadmap

**Confirmed (2025-2026):**
- NVIDIA AI Enterprise inclusion (production support)
- Kubernetes Operator enhancements (Grove API evolution)
- TensorRT-LLM v1.1 full integration
- More storage backend integrations

**Likely (2026):**
- Broader hardware support (beyond NVIDIA?)
- API stabilization (v1.0 release)
- More production references (enterprise adoption)

### 10.4 Industry Trends

**KV Cache Management is Becoming Standard:**
- All major frameworks adding KV cache features (vLLM, TRT-LLM, SGLang)
- Storage vendors building native integrations (VAST, WEKA, NetApp)
- Cloud providers offering managed KV cache services

**Disaggregated Inference is Growing:**
- Prefill/Decode split becoming common pattern
- Multi-tier memory (GPU/CPU/Disk/CXL) is new normal
- RDMA and high-speed interconnects critical

**Vendor Consolidation vs Open Standards:**
- NVIDIA pushing full-stack (Dynamo + NIM + AI Enterprise)
- Open-source pushing interoperability (llm-d, LMCache)
- Market will likely support both (enterprise vs cloud-native)

---

## 11. Summary and Final Recommendations

### 11.1 The Bottom Line

**These are NOT competitors—they solve different parts of the problem:**

| Layer | Solution | Role |
|-------|----------|------|
| **Storage** | LMCache | Where KV blocks live (GPU/CPU/Disk/S3) |
| **Routing** | llm-d-kv-cache | Which pod to send requests to |
| **Orchestration** | Dynamo KVBM | How to manage the entire system |

**They can stack:**
- llm-d-kv-cache + LMCache = Kubernetes routing + storage
- Dynamo + LMCache = Full-stack + proven storage backend
- (llm-d + Dynamo = redundant, both provide orchestration)

### 11.2 Quick Decision Guide

**Choose LMCache if:**
- You have vLLM or SGLang deployments
- You need cross-session KV reuse
- You need multi-tier caching (GPU → CPU → Disk → S3)
- You want vendor-neutral, proven storage layer

**Choose llm-d-kv-cache if:**
- You have Kubernetes with multiple vLLM replicas
- You need cache-aware routing (87% hit rate)
- You use Red Hat OpenShift AI (native integration)
- You want vendor-neutral orchestration

**Choose Dynamo KVBM if:**
- You have datacenter-scale deployments (1000s GPUs)
- You need disaggregated prefill/decode
- You're NVIDIA-focused infrastructure
- You need AI Enterprise support (coming)

**Choose combinations if:**
- llm-d + LMCache: Kubernetes + routing + storage (best for most)
- Dynamo + LMCache: Full-stack + proven storage (NVIDIA-optimized)

### 11.3 Maturity for Production Use

| System | Production Readiness | Best For |
|--------|---------------------|----------|
| **LMCache** | ✅ Ready (1.5 years, 10+ users) | vLLM/SGLang storage layer |
| **llm-d-kv-cache** | ⚠️ Pilot (8 months, wait for v1.0) | Red Hat customers, K8s natives |
| **Dynamo KVBM** | ⚠️ Pilot (10 months, wait for AI Enterprise) | NVIDIA enterprise customers |

**Safest Production Path Today:**
1. **vLLM + LMCache:** Proven, works everywhere, simple
2. **llm-d + LMCache on OpenShift AI:** Supported by Red Hat
3. **Dynamo (when AI Enterprise ships):** NVIDIA enterprise path

---

## 12. References

### LMCache
- GitHub: https://github.com/LMCache/LMCache
- Blog: https://blog.lmcache.ai/
- PyTorch Foundation announcement: October 2025
- Tensormesh: https://tensormesh.ai

### llm-d-kv-cache
- GitHub: https://github.com/llm-d/llm-d-kv-cache
- Docs: https://llm-d.ai/docs/architecture/Components/kv-cache-manager
- Red Hat blog: https://developers.redhat.com/articles/2025/10/07/master-kv-cache-aware-routing-llm-d-efficient-ai-inference

### NVIDIA Dynamo KVBM
- GitHub: https://github.com/ai-dynamo/dynamo
- Docs: https://docs.nvidia.com/dynamo/
- Developer site: https://developer.nvidia.com/dynamo
- NVIDIA blog: https://developer.nvidia.com/blog/introducing-nvidia-dynamo-a-low-latency-distributed-inference-framework-for-scaling-reasoning-ai-models/

### Storage Partners
- VAST Data: https://www.vastdata.com/blog/nvidia-dynamo-vast-scalable-optimized-inference
- WEKA: LMCache integration announcements
- Redis: LMCache backend documentation

### Academic Papers
- CacheGen (SIGCOMM 2024): KV cache compression
- CacheBlend (EuroSys 2025): RAG with cached knowledge fusion
- LMCache Technical Report: https://arxiv.org/abs/2510.09665v1
