# Distributed Inference Stack — Layered Architecture v2
## Projects: Dynamo (NVIDIA) + llm-d (Community)

> **v2 — corrected after deep code scan of NIXL plugins, KVBM sub-crates,
> llm-d-kv-cache, vLLM connector APIs, canonical vllm-project/vllm CCL analysis,
> Intel IO Direct stack (gds-linux, gds-liburing, gds-liburing-cufilewrapper),
> openucx ZE transport layer, and gds-linux 11-patch kernel series analysis.
> Section 9 (GPU Comms SW team code changes) + Section 10 (validation plan) added.
> Full document review pass completed: risks, building blocks, interface catalog,
> validation coverage, and upstreaming/delivery tracks all updated.**
>
> Key corrections over v1:
> 1. NIXL is a **multi-plugin subsystem** with its own 3-sub-layer structure (not one box)
> 2. KVBM spans **4 sub-crates** with distinct hardware dependencies
> 3. llm-d-kv-cache spans **two separate layers** (L3 indexer + L5 fs-backend)
> 4. vLLM's **internal API surface** (KVConnectorBase_V1, OffloadingSpec, NixlConnector)
>    is an explicit interface layer — version-pinned differently per project
> 5. **Level-Zero (Intel GPU) is NOT present** in either codebase
> 6. **PdConnector** = composite of DynamoConnector + vLLM's own NixlConnector
> 7. **cuFile / GDS** loaded via dlopen at runtime in both NIXL and llmd-fs-backend
> 8. **dynamo-memory** crate is the unified Rust wrapper for both cudarc and nixl-sys
> 9. **L5/L6 order corrected** (v2 had them swapped): KVBM/storage connectors are L5
>    (they *create* nixlAgents and call into NIXL); NIXL transport library is L6.
>    vLLM (L4) also creates nixlAgents directly via NixlConnector — explicit layer-skip noted.
> 10. **CCL libraries (NCCL/RCCL/HCCL/OneCCL) are L6b** — transport middleware peer to NIXL,
>     NOT hardware (L10). Both consumed by L4 DeviceCommunicator.
> 11. **NIXL has two use cases**: KV cache P/D transfer (NixlConnector) AND MoE EP dispatch
>     (NixlEPAll2AllManager). Spec-decode and EPLB weight rebalancing do NOT use NIXL.
> 12. **Dynamo core is transport-free** — no NIXL or CCL calls. KVBM and vLLM workers are
>     the transport consumers. llm-d Go sidecar is orchestration-only (zero NIXL/CCL deps).
> 13. **KVBM NCCL usage**: `TransferMode::Replicated` only — own ncclComm for KV block
>     replication. Default mode is `Sharded` (no NCCL). Separate from vLLM's TP comm.
> 14. **GPU Comms SW team code-change analysis** (Section 9): NIXL core is
>     already platform-neutral (`nixl_mem_t = VRAM_SEG`); Intel memory registration lives
>     in plugins (area 1). Code work beyond the 3 stated areas spans **5 categories, 17 items**:
>     NIXL libfabric/UCX Intel XPU gaps (A1–A5), KVBM Rust platform abstraction (B1–B6),
>     NIXL agent config/versioning (C1–C3), vLLM XPU EP path (D1–D2),
>     and cross-cutting items (E1–E2). XpuCommunicator has no NIXL EP backend wired — gap confirmed.

---

## 0. High-Level Stack Overview

Two stacks, same bottom half. Read this before the detailed diagram.

```
╔══════════════════════════════════════════════════════════════════════════════════════╗
║  DYNAMO  (NVIDIA)                  ║  llm-d  (Community)                            ║
╠══════════════════════════════════════════════════════════════════════════════════════╣
║  L0   Client         HTTP / OpenAI-compatible API consumers                         ║
╠══════════════════════════════════════════════════════════════════════════════════════╣
║  L1   API Gateway    Dynamo Frontend (Python)   │  llm-d IGW (Envoy + Gateway API)  ║
╠══════════════════════════════════════════════════════════════════════════════════════╣
║  L2   Routing &      Dynamo Router              │  llm-d EPP (Go) + PD-Sidecar      ║
║       Disagg.        (Python+Rust KV-overlap)   │  (inference-scheduler)             ║
╠══════════════════════════════════════════════════════════════════════════════════════╣
║  L3   KV Cache       Dynamo KV Router           │  llm-d KV Cache Indexer (Go)       ║
║       Coordination   (RadixTree, Velo events)   │  + ZMQ KV Events Pool              ║
╠══════════════════════════════════════════════════════════════════════════════════════╣
║  L4   Inference      ╔══════════════════════════════════════════════════════╗        ║
║       Engine         ║  vLLM  (shared by both)                             ║        ║
║                      ║  • Tensor/Pipeline Parallel workers                  ║        ║
║                      ║  • NCCL/RCCL/OneCCL (L6b) for TP/PP/DP/EP collectives       ║        ║
║                      ║  • KV connector plugin (DynamoConnector / llmd-fs)  ║        ║
║                      ╚══════════════════════════════════════════════════════╝        ║
╠══════════════════════════════════════════════════════════════════════════════════════╣
║  L5   KV Block       Dynamo KVBM               │  llmd-fs-backend (Python+CUDA)     ║
║       Management     (4 Rust crates)            │  vLLM OffloadingSpec plugin        ║
║       & Storage      creates nixlAgents ─────────────────────────────────► L6       ║
╠══════════════════════════════════════════════════════════════════════════════════════╣
║  L6   Transport      ╔══════════════════════════════════════════════════════╗        ║
║       Abstraction    ║  NIXL  (C++17, 14 plugins, shared by both)          ║        ║
║                      ║  nixlAgent API · UCX · LIBFABRIC · cuda_gds · OBJ  ║        ║
║                      ║  ← consumed by L5 KVBM AND directly by L4 vLLM     ║        ║
║                      ╚══════════════════════════════════════════════════════╝        ║
║  L6b  Collective     NCCL (libnccl.so.2) · RCCL (librccl.so.1)                     ║
║       Comms (CCL)    HCCL (habana_frameworks) · OneCCL (oneccl_bindings)            ║
║                      ← consumed by L4 DeviceCommunicator (same level as NIXL)       ║
║                      ← also consumed by L5 KVBM for TransferMode::Replicated        ║
╠══════════════════════════════════════════════════════════════════════════════════════╣
║  L7   Rust Runtime   dynamo-memory · dynamo-runtime · dynamo-llm · dynamo-tokens    ║
║       Foundation     Python↔Rust FFI (Maturin/PyO3)                                 ║
╠══════════════════════════════════════════════════════════════════════════════════════╣
║  L8   Control Plane  Dynamo Planner (Python)    │  llm-d WVA (Go)                   ║
║       & Autoscaling  cost model, KPI targets    │  VariantAutoscaling CRD            ║
╠══════════════════════════════════════════════════════════════════════════════════════╣
║  L9   Orchestration  Dynamo K8s Operator (Go)   │  llm-d Gateway API + CRDs         ║
║                      DGD/DWM CRDs, PodClique    │  InferencePool, HPA/KEDA           ║
╠══════════════════════════════════════════════════════════════════════════════════════╣
║  L10  Hardware       GPU (NVIDIA·AMD·Gaudi·XPU) · RDMA Network · NVMe/Object Store  ║
║                      CUDA/ROCm/HabanaDriver/OneAPI · UCX · RDMA libs · libcufile     ║
╚══════════════════════════════════════════════════════════════════════════════════════╝
```

### Key architectural observations

| # | Observation |
|---|---|
| 1 | **vLLM (L4) and NIXL (L6) are shared infrastructure** — both projects build on top of them |
| 2 | **Two separate communication planes**: CCL libs (NCCL/RCCL/HCCL/OneCCL at L6b) for activations/weights within a worker (TP/PP/DP/EP); NIXL (L6) for KV cache block transfer between P/D workers |
| 3 | **P/D disaggregation is handled at different layers**: routing decision at L2, KV transfer coordination at L3/L4, actual data movement at L5→L6 |
| 4 | **L5 sits between L4 and L6**: KVBM creates nixlAgents (consuming L6), while exposing a vLLM connector (consumed by L4). vLLM also reaches L6 directly via NixlConnector (layer skip) |
| 5 | **L7 (Rust runtime) is Dynamo-only** — llm-d is pure Go+Python with no Rust layer |
| 6 | **FNV block hash (dynamo-tokens Rust / kvblock Go)** is the critical shared contract crossing the L3/L4 boundary — must be byte-identical |
| 7 | **llm-d-kv-cache spans L3 and L5** — the Go indexer lives at L3; the Python+CUDA fs-backend lives at L5 |

---

## 1. Layered Architecture Diagram — Detailed

```
╔══════════════════════════════════════════════════════════════════════════════════════╗
║  L0 — CLIENT                                                                        ║
║  HTTP/OpenAI-compat consumers (curl, Python SDK, etc.)                              ║
╚═══════════════════════════╤════════════════════════════════════════════════════════╝
                            │  HTTP/REST  OpenAI API
╔═══════════════════════════▼════════════════════════════════════════════════════════╗
║  L1 — INGRESS / API GATEWAY                                                        ║
║  ┌────────────────────────────────┐  ┌─────────────────────────────────────────┐   ║
║  │  Dynamo: Frontend (Python)     │  │  llm-d: IGW (Envoy + Gateway API)       │   ║
║  │  • OpenAI HTTP server :8000    │  │  • Envoy data-plane :80/:443            │   ║
║  │  • Tokenize / preprocess       │  │  • ext-proc filter → EPP callback       │   ║
║  │  • Detokenize / Prometheus     │  │  • HTTPRoute, GatewayClass, TLS         │   ║
║  └────────────────────────────────┘  └─────────────────────────────────────────┘   ║
╚═══════════════════════════╤════════════════════════════════════════════════════════╝
          [I2a] Velo RPC (TCP/HTTP2/NATS)  |  [I2b] gRPC ext-proc (Envoy→EPP)
╔═══════════════════════════▼════════════════════════════════════════════════════════╗
║  L2 — ROUTING / DISAGGREGATION ORCHESTRATION                                       ║
║  ┌──────────────────────────────────────┐  ┌──────────────────────────────────┐    ║
║  │  Dynamo: Router + Global Router      │  │  llm-d: EPP (Endpoint Picker)   │    ║
║  │  Python + Rust FFI (dynamo_kv_router)│  │  • Filters: label, label-select  │    ║
║  │  • KV overlap cost function          │  │  • Scorers: precise-prefix-cache,│    ║
║  │  • Disaggregated P→D orchestration   │  │    load-aware, session-affinity  │    ║
║  │  • Worker failure migration          │  │  Go (llm-d-inference-scheduler)  │    ║
║  │  • Multi-router state sync           │  ├──────────────────────────────────┤    ║
║  │                                      │  │  llm-d: PD-Sidecar               │    ║
║  │                                      │  │  • P/D disaggregation proxy      │    ║
║  │                                      │  │  • KV transfer coordination      │    ║
║  │                                      │  │  Go                              │    ║
║  └──────────────────────────────────────┘  └──────────────────────────────────┘    ║
╚═══╤═════════════════════════════════════════════════╤══════════════════════════════╝
    │ [I3a] Rust RadixTree API (in-process)            │ [I3b] gRPC GetPodScores
    │ [I3c] NATS/ZMQ KvStored/KvRemoved events         │ [I3d] ZMQ BlockStored events
╔═══▼═════════════════════════════════════════════════▼══════════════════════════════╗
║  L3 — KV CACHE COORDINATION                                                        ║
║  ┌──────────────────────────────────────┐  ┌──────────────────────────────────┐    ║
║  │  Dynamo: KV Router (Rust)            │  │  llm-d: KV Cache Indexer (Go)    │    ║
║  │  lib/kv-router                       │  │  pkg/kvcache + pkg/kvblock        │    ║
║  │  • RadixTree: block hash → worker    │  │  • Two-level LRU (GPU+CPU tiers) │    ║
║  │  • Overlap scoring                   │  │  • FNV-64a block key hashing     │    ║
║  │  • dynamo-tokens (shared hash lib)   │  │  • Backends: LRU / Redis /Valkey │    ║
║  ├──────────────────────────────────────┤  ├──────────────────────────────────┤    ║
║  │  Dynamo: Velo Events (Rust)          │  │  llm-d: KV Events Pool (Go)      │    ║
║  │  lib/velo-events                     │  │  pkg/kvevents                    │    ║
║  │  • NATS JetStream / ZMQ pub/sub      │  │  • ZMQ subscriber (go-zeromq)    │    ║
║  │  • KvStored/KvRemoved/Free topics    │  │  • Sharded pool (FNV shard key)  │    ║
║  │  • Sequence tracking multi-router    │  │  • Engine adapter (vLLM parser)  │    ║
║  │                                      │  ├──────────────────────────────────┤    ║
║  │                                      │  │  llm-d: Tokenization Service     │    ║
║  │                                      │  │  pkg/tokenization                │    ║
║  │                                      │  │  • gRPC TokenizationService      │    ║
║  │                                      │  │  • Chat template rendering       │    ║
║  └──────────────────────────────────────┘  └──────────────────────────────────┘    ║
║  NOTE: llm-d-inference-scheduler embeds llm-d-kv-cache v0.7.1 as a Go module       ║
║  (indexer can run in-process with EPP, or as a separate gRPC sidecar service)      ║
╚═══╤════════════════════════════════════════════════════════════════════════════╤═══╝
    │ [I4a] Velo RPC + disaggregated_params                                     │
    │ [I4b] vLLM HTTP API + PD-Sidecar injection headers                        │
╔═══▼════════════════════════════════════════════════════════════════════════════▼═══╗
║  L4 — INFERENCE ENGINE LAYER                                                       ║
║  ┌─────────────────────────────────────────────────────────────────────────────┐   ║
║  │  vLLM  (Python — primary engine for both projects)                          │   ║
║  │                                                                             │   ║
║  │  ┌──────────────────────────────┐    ┌──────────────────────────────────┐  │   ║
║  │  │  Dynamo vLLM wrappers        │    │  Disaggregated Prefill Workers   │  │   ║
║  │  │  components/dynamo/vllm/     │    │  ← compute prompt KV cache       │  │   ║
║  │  │  • instrumented_scheduler.py │    │  → publish KV events (ZMQ/NATS)  │  │   ║
║  │  │  • publisher.py (KV events)  │    └──────────────┬───────────────────┘  │   ║
║  │  │  • args.py (connector config)│                   │  [I5] NIXL GPU↔GPU   │   ║
║  │  │                              │    ┌──────────────▼───────────────────┐  │   ║
║  │  │  Dynamo KV Connectors:       │    │  Disaggregated Decode Workers    │  │   ║
║  │  │  DynamoConnector             │    │  ← receive KV via NIXL transfer  │  │   ║
║  │  │    extends KVConnectorBase_V1│    │  → autoregressive token gen      │  │   ║
║  │  │  PdConnector (composite):    │    └──────────────────────────────────┘  │   ║
║  │  │    = DynamoConnector         │                                          │   ║
║  │  │    + NixlConnector(vLLM own) │                                          │   ║
║  │  └──────────────────────────────┘                                          │   ║
║  │  Also: SGLang backend (ZMQ transport) · TRT-LLM backend                   │   ║
║  │                                                                             │   ║
║  │  ══════ COLLECTIVE COMMUNICATION (NCCL / RCCL / HCCL / OneCCL) ══════════  │   ║
║  │  Source: ~/src/vllm (canonical vllm-project/vllm main)                    │   ║
║  │  vLLM parallel_state.py: init_process_group(backend=<platform-selected>)  │   ║
║  │  4 process groups — distinct collective operations per group:              │   ║
║  │                                                                             │   ║
║  │  TP group — Tensor Parallel (hottest path; fires every attention+MLP)     │   ║
║  │    AllReduce      — merge partial sums after column-parallel linear        │   ║
║  │    AllReduceV     — SymmMem variant (NVLink, H100+)                        │   ║
║  │    AllGather      — reconstruct full activations (sequence parallel)       │   ║
║  │    AllGatherV     — variable-size (MoE token gather across DP)             │   ║
║  │    ReduceScatter  — shard activations for sequence-parallel input          │   ║
║  │    ReduceScatterV — variable-size reduce-scatter                           │   ║
║  │    Broadcast      — scheduler metadata; sampler dist (TP rank 0 → all)    │   ║
║  │    P2P send/recv  — PP stage activation passing reuses TP pynccl comm     │   ║
║  │                                                                             │   ║
║  │    AllReduce priority (tried in order; falls back to next):                │   ║
║  │      1. SymmMemAllReduce    NVLink symmetric memory (CUDA, H100+)          │   ║
║  │      2. QuickAllReduce      Quantized SHM (ROCm MI300 only; Q8/Q6/Q4)     │   ║
║  │      3. FlashInferAllReduce FlashInfer allreduce_fusion kernel             │   ║
║  │      4. CustomAllreduce     CUDA SHM (intra-node; skips for large tensors) │   ║
║  │      5. pynccl              NCCL/RCCL (cross-node; always available)       │   ║
║  │                                                                             │   ║
║  │  PP group — Pipeline Parallel (inter-stage, per micro-batch)               │   ║
║  │    P2P send/recv  — hidden states stage-to-stage via pynccl.send/recv     │   ║
║  │    broadcast_tensor_dict — final logits: last PP stage → all PP ranks     │   ║
║  │    ⚠ CPU metadata (scheduling signals) uses GLOO; tensors use NCCL        │   ║
║  │                                                                             │   ║
║  │  DP group — Data Parallel (multi-engine disaggregated inference)           │   ║
║  │    AllReduce — coordinate_batch_across_dp: batch token-count sync/step    │   ║
║  │    AllReduce — DP parallel config version sync (max/min)                   │   ║
║  │    AllGather — MoE: gather hidden states across DP ranks for EP dispatch  │   ║
║  │                                                                             │   ║
║  │  EP group — Expert Parallel / MoE all-to-all (pluggable backends):        │   ║
║  │    NaiveAll2AllManager         AllReduce-based (debug/test only)           │   ║
║  │    AgRsAll2AllManager          AllGather + ReduceScatter                   │   ║
║  │    DeepEPHTAll2AllManager      DeepEP high-throughput (NVIDIA)             │   ║
║  │    DeepEPLLAll2AllManager      DeepEP low-latency (NVIDIA)                 │   ║
║  │    NixlEPAll2AllManager        NIXL-based EP ← NIXL used here too! [note] │   ║
║  │    FlashInferNVLinkTwoSided    FlashInfer NVLink two-sided kernels          │   ║
║  │    FlashInferNVLinkOneSided    FlashInfer NVLink one-sided kernels          │   ║
║  │    MoriAll2AllManager          AMD ROCm MoRI kernels                       │   ║
║  │    batch_isend_irecv  — EPLB expert weight redistribution (pynccl P2P)    │   ║
║  │    ⚠ [note] NixlEPAll2AllManager: NIXL used for elastic EP dispatch,      │   ║
║  │      separate from the KV-cache NIXL path — two distinct NIXL use cases   │   ║
║  │                                                                             │   ║
║  │  DeviceCommunicator (platform-dispatched at init_process_group time):      │   ║
║  │  • CudaCommunicator  [NVIDIA GPU]  pynccl → libnccl.so.2                  │   ║
║  │  • CudaCommunicator  [AMD ROCm]    pynccl → librccl.so.1  (same class)    │   ║
║  │  • XpuCommunicator   [Intel XPU]   OneCCL → backend='xccl'                │   ║
║  │  • CpuCommunicator   [CPU]         OneCCL → backend='ccl'                 │   ║
║  │  (HpuCommunicator / HCCL is Gaudi-specific, added by vllm-gaudi fork)     │   ║
║  │                                                                             │   ║
║  │  [CCL USE: P2P KV Transfer] Alternative to NIXL, not used by Dynamo/llm-d │   ║
║  │  P2pNcclConnector: libnccl direct (own dedicated ncclComm) + ZMQ metadata │   ║
║  │  ⚠ Mutually exclusive with NixlConnector — only one KV backend active     │   ║
║  │                                                                             │   ║
║  │  ═══════ vLLM v1 INTERNAL API SURFACE  (critical shared interface) ══════  │   ║
║  │  [I6a] KVConnectorBase_V1   vllm.distributed.kv_transfer.kv_connector.v1  │   ║
║  │         ← DynamoConnector implements this                                  │   ║
║  │  [I6b] NixlConnector        vllm.distributed.kv_transfer.kv_connector.v1  │   ║
║  │         ← vLLM-native, used inside PdConnector                             │   ║
║  │         ← NixlHandshakePayload: out-of-band handshake for decode workers   │   ║
║  │  [I6c] OffloadingSpec       vllm.v1.kv_offload.spec                        │   ║
║  │         ← llmd_fs_backend.SharedStorageOffloadingSpec implements this       │   ║
║  │  [I6d] OffloadingHandler    vllm.v1.kv_offload.worker.worker               │   ║
║  │         ← BaseStorageOffloadingHandler implements this                     │   ║
║  │  [I6e] SchedulerOutput, Request  vllm.v1.core  consumed by KVBM           │   ║
║  │  [I6f] KVCacheBlocks, BlockHash  vllm.v1.core.kv_cache_utils               │   ║
║  │                                                                             │   ║
║  │  VERSION PINS:  Dynamo → vllm==0.19.0    llm-d → vllm==0.18.1             │   ║
║  │  ⚠ Minor version gap: I6 APIs must be validated across both versions       │   ║
║  └─────────────────────────────────────────────────────────────────────────────┘   ║
╚═══╤════════════════════════════════════════════════════════════════════════════╤═══╝
    │ [I7a] nixl-sys = "=0.10.1" Rust FFI → libnixl.so  (KVBM-physical path)   │
    │ [I7b] nixl._api Python pybind11  ←  NixlConnector / PdConnector at L4     │
    │        ⚠ LAYER SKIP: vLLM (L4) creates nixlAgents DIRECTLY via            │
    │          NixlConnector/PdConnector, bypassing L5 KVBM entirely            │
    │ [I8]  vLLM OffloadingSpec / KVConnectorBase_V1 plugin load (--kv-xfer-cfg)│
╔═══▼════════════════════════════════════════════════════════════════════════════▼═══╗
║  L5 — KV BLOCK MANAGEMENT + STORAGE CONNECTORS                                     ║
║  (sits between inference engine and transport; creates nixlAgents)                  ║
║                                                                                    ║
║  ── A. Dynamo KVBM (4 sub-crates, bottom→top dependency order) ──────────────────  ║
║                                                                                    ║
║  kvbm-common   (Rust)                                                              ║
║  Pure types: BlockId, SequenceHash                                                 ║
║  dep: dynamo-tokens  (shared FNV block hash lib — critical cross-project contract) ║
║                                                                                    ║
║  kvbm-logical  (Rust)                                                              ║
║  Block lifecycle FSM: Reset → Complete → Registered → Inactive                     ║
║  Eviction backends: LRU | MultiLRU | TinyLFU frequency tracking                   ║
║  Block registry, active/inactive pool, events pubsub                               ║
║  dep: kvbm-common  tokio  lru  prometheus                                          ║
║                                                                                    ║
║  kvbm-kernels  (Rust + CUDA)                                                       ║
║  CUDA kernels: layout conversions NHD↔HND↔Universal, vectorized copy,             ║
║  SM-based batch copy, fused permute+copy                                           ║
║  dep: cudarc = "=0.19.3"  (Rust CUDA binding)   ← I9                             ║
║                                                                                    ║
║  kvbm-physical (Rust)                                                              ║
║  Physical layout (block_id → MemRegion); NIXL memory registration (RDMA metadata) ║
║  Two transfer executors:                                                           ║
║    cuda.rs  → GPU↔CPU via cudaMemcpyAsync  (cudarc, no NIXL needed)               ║
║    nixl.rs  → GPU↔remote: creates NixlAgent, registers memory, posts xfer req     ║
║               uses dynamo-memory::NixlAgent wrapper → nixl-sys → L6 NIXL core     ║
║  dep: kvbm-kernels  dynamo-memory { cudarc + nixl-sys }   ← I7a, I9              ║
║  NOTE: nixl-sys = "=0.10.1" exact-pinned — NIXL update requires coordinated bump  ║
║                                                                                    ║
║  bindings/kvbm  (Rust+Python, PyO3/Maturin)                                       ║
║  DynamoConnector  — extends vLLM KVConnectorBase_V1  ← I6a                        ║
║  PdConnector      — MultiConnector(DynamoConnector, NixlConnector)                 ║
║                     NixlConnector half creates its OWN nixlAgent at L4 directly   ║
║  KvConnectorWorker / KvConnectorLeader  — scheduler+worker split                   ║
║  dep: dynamo-llm  dynamo-runtime  kvbm-physical  kvbm-logical                      ║
║                                                                                    ║
║  ── B. llm-d: llmd-fs-backend  (Python + C++/CUDA extension) ───────────────────  ║
║  ⚠ Does NOT use NIXL — shared filesystem (NFS/PVC) is the transfer medium         ║
║    GPU↔storage via CUDA cudaMemcpyAsync or optional cuFile (GDS via dlopen)       ║
║                                                                                    ║
║  SharedStorageOffloadingSpec  → implements vLLM OffloadingSpec    ← I6c            ║
║  BaseStorageOffloadingHandler → implements vLLM OffloadingHandler ← I6d            ║
║  StorageOffloadEngine (C++)   → CUDA thread pool, NUMA staging buffers, I/O       ║
║  tensor_copier.cu             → cudaMemcpyAsync (D↔H) or CUDA kernels             ║
║  gds_file_io.cpp              → cuFile via dlopen  ← I10 (runtime only, optional) ║
║  dep (build): torch==2.10.0  libnuma  libdl                                        ║
║  dep (runtime): vllm==0.18.1  (version-locked)                                    ║
║                                                                                    ║
║  pvc_evictor  (Python, multiprocess)                                               ║
║  Disk-space daemon: Crawler + Activator + Deleter  (hysteresis 85%→70%)           ║
╚═══╤════════════════════════════════════════════════════════════════════════════════╝
    │ [I11] KVBM-physical calls into L6 NIXL via dynamo-memory::NixlAgent
    │       nixl-sys = "=0.10.1" Rust FFI → libnixl.so
    │ [I9]  cudarc = "=0.19.3" Rust → libcuda.so / libcudart.so  (KVBM-kernels)
    │ [I10] llmd-fs-backend: dlopen("libcufile.so") at runtime (optional GDS path)
╔═══▼════════════════════════════════════════════════════════════════════════════════╗
║  L6 — NIXL: TRANSPORT ABSTRACTION + PLUGIN SUBSYSTEM                               ║
║  (consumed by L5 KVBM-physical AND directly by L4 vLLM NixlConnector)             ║
║                                                                                    ║
║  ┌─────────────────────────────────────────────────────────────────────────────┐   ║
║  │  NIXL Core  (C++17)                                                         │   ║
║  │  • nixlAgent — public API: registerMem, createXferReq, postXfer, checkXfer │   ║
║  │    Instantiated by: kvbm-physical (L5) via Rust nixl-sys                   │   ║
║  │                     NixlConnector (L4) via Python nixl._api                 │   ║
║  │  • nixlPluginManager — singleton; loads .so plugins via dlopen at runtime  │   ║
║  │  • nixlBackendEngine — abstract base class all 14 plugins implement         │   ║
║  │  • Transfer model: one-sided async RDMA (initiator-driven, non-blocking)   │   ║
║  │  • Memory segments: DRAM_SEG, VRAM_SEG, FILE_SEG, BLK_SEG, OBJ_SEG        │   ║
║  │  • Metadata exchange: ETCD (shared w/ Dynamo svc disc) or side-channel     │   ║
║  ├──────────────────────────┬──────────────────────────────────────────────────┤   ║
║  │  Python bindings         │  Rust bindings                                  │   ║
║  │  nixl._api  (pybind11)   │  nixl-sys = "=0.10.1"                          │   ║
║  │  used by: NixlConnector  │  → dynamo-memory::NixlAgent → kvbm-physical     │   ║
║  └──────────────────────────┴──────────────────────────────────────────────────┘   ║
║                                                                                    ║
║  NIXL TRANSPORT PLUGINS (dynamic .so; multiple can be active simultaneously):      ║
║                                                                                    ║
║  ── Network / Compute Fabric ────────────────────────────────────────────────────  ║
║  UCX          RDMA + intra-node GPU↔CPU (IB/RoCEv2/iWARP)                         ║
║               dep: libucx ≥1.20  RDMA drivers  CUDA+GDRCopy (optional)            ║
║                                                                                    ║
║  LIBFABRIC    Multi-rail RDMA, topology-aware (AWS EFA, Gaudi, multi-NIC)          ║
║               dep: libfabric ≥1.21  hwloc ≥2.10  libnuma                          ║
║                                                                                    ║
║  OFI          SHM (intra-node) + Verbs (inter-node); RoCEv2, Gaudi HMEM           ║
║               dep: libfabric ≥1.16                                                 ║
║                                                                                    ║
║  UCCL         P2P GPU RDMA (inter-node), flexible software transport               ║
║               dep: libuccl_p2p                                                     ║
║                                                                                    ║
║  GPUNETIO     GPU-kernel-initiated RDMA (DOCA GPUNetIO, BlueField/ConnectX)        ║
║               dep: DOCA GPUNetIO SDK  CUDA kernels                                 ║
║                                                                                    ║
║  ── Storage ─────────────────────────────────────────────────────────────────────  ║
║  cuda_gds     GPUDirect Storage: VRAM↔FILE without CPU staging                     ║
║               dep: CUDA  libcufile (GDS driver)  cufile.json config               ║
║                                                                                    ║
║  gds_mt       Multi-threaded GDS                                                   ║
║               dep: CUDA  libcufile  Taskflow C++ lib                               ║
║                                                                                    ║
║  POSIX        CPU file I/O: io_uring / Linux AIO / POSIX AIO                      ║
║               dep: liburing -or- libaio-dev  (seccomp for io_uring in containers) ║
║                                                                                    ║
║  OBJ          S3 / S3-CRT / S3-accel (RDMA-accelerated)                           ║
║               dep: AWS SDK                                                         ║
║                                                                                    ║
║  AZURE_BLOB   Azure Blob Storage                                                   ║
║               dep: Azure C++ SDK                                                   ║
║                                                                                    ║
║  HF3FS        3FS filesystem (DeepSeek AI)                                         ║
║               dep: hf3fs_usrbio.so  libhf3fs_api_shared.so                        ║
║                                                                                    ║
║  ── Specialized ─────────────────────────────────────────────────────────────────  ║
║  MOONCAKE     Disaggregated KV cache store (TCP/RDMA/CXL/NVMe-oF)                  ║
║               dep: Mooncake Transfer Engine shared lib                             ║
║                                                                                    ║
║  GUSLI        Experimental                                                         ║
║                                                                                    ║
║  Plugin selection by NIXL core:                                                    ║
║    VRAM↔VRAM (remote)  →  UCX or LIBFABRIC (RDMA)                                 ║
║    VRAM↔FILE (local)   →  cuda_gds (if GDS driver present) else POSIX fallback    ║
║    DRAM↔DRAM (local)   →  UCX SHM or OFI SHM                                      ║
║    DRAM↔OBJ            →  OBJ / AZURE_BLOB                                        ║
╚═══╤════════════════════════════════════════════════════════════════════════════════╝
    │ [I12] dynamo-memory: NixlCompatible trait + NixlMemory type erasure
    │       bridges cudarc (GPU alloc) and nixl-sys (RDMA registration)
╔═══▼════════════════════════════════════════════════════════════════════════════════╗
║  L7 — RUST RUNTIME FOUNDATION + FFI BRIDGES                                        ║
║  ┌───────────────────────────────┐  ┌────────────────────────────────────────────┐ ║
║  │  dynamo-memory  (Rust)        │  │  Dynamo Runtime  lib/runtime/  (Rust)      │ ║
║  │  Unified GPU+NIXL memory mgmt │  │  • Pipeline: Source→Operator→Sink         │ ║
║  │  • cudarc wrapper             │  │  • Service engine + registry               │ ║
║  │  • nixl-sys wrapper (=0.10.1) │  │  • Component lifecycle, health checks      │ ║
║  │  • NixlCompatible trait       │  │  • Distributed runtime coordination        │ ║
║  │  • NixlMemory type erasure    │  │  • Prometheus metrics registry             │ ║
║  │  • NUMA-aware placement       │  ├────────────────────────────────────────────┤ ║
║  │                               │  │  Dynamo LLM  lib/llm/  (Rust)             │ ║
║  │  dynamo-tokens  (Rust)        │  │  • Backend trait (inference API)           │ ║
║  │  Shared block hash library    │  │  • Worker discovery                        │ ║
║  │  FNV-64a, chunk-aligned keys  │  │  • KServe gRPC (ModelStreamInfer)          │ ║
║  │  ⚠ Used by kvbm-common AND   │  │  • Request migration on failure            │ ║
║  │    llm-d kvblock (Go FNV-64a) │  │  • LLM protocols (req/resp types)          │ ║
║  │    Must stay byte-identical   │  └────────────────────────────────────────────┘ ║
║  └───────────────────────────────┘                                                 ║
║                                                                                    ║
║  Python ↔ Rust FFI  (Maturin / PyO3)  [I13]                                       ║
║  Exposed Python modules:                                                           ║
║    dynamo_runtime · dynamo_llm · dynamo_kv_router                                 ║
║    prometheus_metrics · kserve_grpc_client · http_client                          ║
║    planner_coord · kvbm._core  (DynamoConnector, KvConnectorWorker/Leader)        ║
║  ⚠ CRITICAL: FFI ABI is not independently versioned.                              ║
║    Any Rust type change cascades to Frontend (L1) + Router (L2) + Planner (L8)   ║
║    simultaneously — these cannot be updated independently.                         ║
╚══════════════════════════════════════════════════════════════════════════════════╝

╔══════════════════════════════════════════════════════════════════════════════════════╗
║  L8 — CONTROL PLANE / AUTOSCALING                                                   ║
║  ┌──────────────────────────────────┐  ┌────────────────────────────────────────┐   ║
║  │  Dynamo: Planner (Python)        │  │  llm-d: WVA (Workload Variant          │   ║
║  │  • Scrapes Prometheus /metrics   │  │  Autoscaler, Go)                       │   ║
║  │  • Load predictors:              │  │  • Collector: Prometheus scrape        │   ║
║  │    ARIMA / Kalman / Prophet      │  │  • Saturation Analyzer:                │   ║
║  │  • Throughput or load-based      │  │    KV util + queue depth + slack       │   ║
║  │  • KubernetesConnector (DGD CRD) │  │  • Actuator: updates replica count    │   ║
║  │  • VirtualConnector (Slurm/ext)  │  │  • CRD owner: VariantAutoscaling       │   ║
║  ├──────────────────────────────────┤  │  • Integrates with HPA / KEDA          │   ║
║  │  Dynamo: Profiler (Python)       │  └────────────────────────────────────────┘   ║
║  │  • Performance sweeps            │                                               ║
║  │  • Interpolation surfaces for    │                                               ║
║  │    throughput, TTFT, ITL         │                                               ║
║  └──────────────────────────────────┘                                               ║
╚══════════════════════════════════════════════════════════════════════════════════════╝

╔══════════════════════════════════════════════════════════════════════════════════════╗
║  L9 — KUBERNETES / DEPLOYMENT ORCHESTRATION                                         ║
║  ┌──────────────────────────────────┐  ┌────────────────────────────────────────┐   ║
║  │  Dynamo: K8s Operator (Go)       │  │  llm-d: Gateway API + CRDs             │   ║
║  │  CRDs:                           │  │  • InferencePool (GIE)                 │   ║
║  │  • DynamoGraphDeployment (DGD)   │  │  • GatewayClass / HTTPRoute            │   ║
║  │  • DynamoWorkerMetadata (DWM)    │  │  • VariantAutoscaling (llmd.ai/v1a1)   │   ║
║  │  • DynamoCheckpoint              │  │  • HPA / KEDA scalers                  │   ║
║  │  • PodClique / PodCliqueSet      │  │  Helm: workload-variant-autoscaler     │   ║
║  │  • DGD Scaling Adapter           │  │        kv-cache chart                  │   ║
║  │  Discovery:                      │  └────────────────────────────────────────┘   ║
║  │  • K8s: EndpointSlices + DWM CRD │                                               ║
║  │  • Bare-metal: etcd or file-based│                                               ║
║  └──────────────────────────────────┘                                               ║
╚══════════════════════════════════════════════════════════════════════════════════════╝

╔══════════════════════════════════════════════════════════════════════════════════════╗
║  L6b — COLLECTIVE COMMUNICATION LIBRARIES  (peer to L6 NIXL; both consumed by L4)  ║
║  These are transport middleware, NOT hardware. Same conceptual level as NIXL.        ║
║    NCCL   (libnccl.so.2)           NVIDIA GPU — activations, weights, KV P2P        ║
║    RCCL   (librccl.so.1)           AMD GPU/ROCm — same pynccl code path [I22]       ║
║    HCCL   (habana_frameworks)      Intel Gaudi/HPU — vllm-gaudi fork only [I19]     ║
║    OneCCL (oneccl_bindings_for_pytorch) Intel XPU/CPU [I20]                         ║
║  ⚠ NCCL/RCCL also used in KVBM (L5) for TransferMode::Replicated (own ncclComm)    ║
║                                                                                      ║
║  L10 — HARDWARE INFRASTRUCTURE                                                      ║
║  GPU: NVIDIA (HBM, NVLink, NVSwitch) · AMD (XGMI) · Intel GPU · Gaudi              ║
║  Network: InfiniBand, RoCEv2, AWS EFA, iWARP, PCIe P2P                             ║
║  NIC-compute: DOCA GPUNetIO (BlueField/ConnectX GPU-kernel-initiated RDMA)          ║
║  Storage: GPU HBM → Host DRAM → Local NVMe → NFS/PVC/Ceph → S3/Azure/3FS          ║
║  SW: CUDA/ROCm/OneAPI drivers, UCX, RDMA drivers, libcufile (NVIDIA GDS) / Intel IO Direct (cuFile ZE adapter), GPUDirect (GDR) ║
║  ⚠ Level-Zero (Intel GPU) NOT present in the Dynamo or llm-d codebases as of scan date.
     Level Zero IS used by the Intel IO Direct stack (gds-liburing-cufilewrapper).          ║
╚══════════════════════════════════════════════════════════════════════════════════════╝
```

---

## 2. Critical Interfaces — Complete Table

| ID   | Interface                              | Protocol / Mechanism                                    | Layers        |
|------|----------------------------------------|---------------------------------------------------------|---------------|
| I2a  | Frontend → Dynamo Router               | Velo RPC (TCP/HTTP2/NATS), PreprocessedRequest          | L1 → L2       |
| I2b  | Envoy → EPP                            | gRPC ext-proc (Envoy ExternalProcessor API)             | L1 → L2       |
| I3a  | Router → KV Router (Dynamo)            | In-process Rust RadixTree API                           | L2 ↔ L3       |
| I3b  | EPP → KV Indexer (llm-d)               | gRPC IndexerService.GetPodScores proto                  | L2 → L3       |
| I3c  | vLLM workers → Velo Events             | NATS JetStream / ZMQ (KvStored/KvRemoved/Free)          | L4 → L3       |
| I3d  | vLLM workers → KV Events Pool          | ZMQ topic `kv@<pod>@<model>`, msgpack-encoded blocks    | L4 → L3       |
| I4a  | Router → Workers (Dynamo)              | Velo RPC + disaggregated_params struct                  | L2 → L4       |
| I4b  | PD-Sidecar → vLLM API                  | HTTP + KV transfer injection headers                    | L2 → L4       |
| I5   | Prefill ↔ Decode (P/D xfer)            | NIXL VRAM→VRAM (UCX/NVLink/PCIe) + NixlHandshakePayload| L4 ↔ L4       |
| I6a  | DynamoConnector ↔ vLLM                 | KVConnectorBase_V1  (vllm.distributed.kv_transfer.v1)   | L4 internal   |
| I6b  | NixlConnector ↔ vLLM                   | vLLM-native NixlConnector + NixlHandshakePayload        | L4 internal   |
| I6c  | llmd-fs-backend ↔ vLLM                 | OffloadingSpec  (vllm.v1.kv_offload.spec)               | L4/L5 internal|
| I6d  | llmd-fs-backend handler ↔ vLLM         | OffloadingHandler (vllm.v1.kv_offload.worker.worker)    | L5 ↔ L4       |
| I6e  | KVBM leader/worker ↔ vLLM              | SchedulerOutput, Request, KVCacheBlocks (vllm.v1.core)  | L5 ↔ L4       |
| I6f  | Block identity contract                | FNV-64a hash, chunk-aligned tokens — must be byte-identical across vLLM, Dynamo (dynamo-tokens Rust), llm-d (kvblock Go) | L3↔L4 |
| I7a  | kvbm-physical → NIXL                   | nixl-sys = "=0.10.1" Rust FFI → libnixl.so              | L5 → L6       |
| I7b  | NixlConnector / PdConnector → NIXL     | nixl._api Python (pybind11) → libnixl.so  ⚠ LAYER SKIP  | L4 → L6       |
| I8   | llmd-fs-backend vLLM plugin load       | --kv-transfer-config JSON (OffloadingConnector)         | L5 → L4       |
| I9   | KVBM CUDA kernels → GPU               | cudarc = "=0.19.3" Rust → libcuda.so / libcudart.so     | L5 → L10      |
| I10  | llmd-fs-backend GDS path               | dlopen("libcufile.so") at runtime — optional; resolves to NVIDIA GDS or Intel IO Direct cuFile ZE adapter | L5 → L10      |
| I11  | NIXL plugins → system libs             | Dynamic link at plugin-load time (per-plugin deps)      | L6 → L10      |
| I12  | dynamo-memory crate                    | NixlCompatible trait bridges cudarc + nixl-sys          | L5/L7 ↔ L6    |
| I13  | Python ↔ Rust FFI                      | Maturin/PyO3 — dynamo_runtime, dynamo_llm, kvbm._core   | L7 ↔ L1/L2/L8 |
| I14  | Planner/WVA → metrics                  | Prometheus HTTP scrape /metrics (label schema contract) | L8 → L4/L1    |
| I15  | Planner/WVA → K8s                      | K8s API patch DGD CRD / update HPA metrics              | L8 → L9       |
| I16  | Operator ↔ components                  | EndpointSlices + DynamoWorkerMetadata CRD               | L9 ↔ L1–L4    |
| I17  | NIXL metadata / ETCD                   | Shared etcd with Dynamo service discovery               | L6 cross-cut  |
| I18  | TP/PP/DP/EP collectives (NVIDIA GPU)   | NCCL via pynccl — AllReduce, AllGather, ReduceScatter, P2P send/recv, batch_isend_irecv | L4 → L6b |
| I19  | TP/PP/DP/EP collectives (Intel Gaudi)  | HCCL via habana_frameworks.torch.distributed (vllm-gaudi fork only)     | L4 → L6b |
| I20  | TP/PP/DP/EP collectives (Intel XPU)    | OneCCL via torch-ccl (oneccl_bindings_for_pytorch)                      | L4 → L6b |
| I21  | P2P KV transfer alt path (NCCL-based)  | P2pNcclConnector: libnccl direct + ZMQ metadata (mutually exclusive with NIXL) | L4 ↔ L4 |
| I22  | TP/PP/DP/EP collectives (AMD GPU/ROCm) | RCCL via pynccl (librccl.so.1) — same CudaCommunicator class as I18    | L4 → L6b |

---

## 3. NIXL Internal 3-Sub-Layer Structure

```
consumers (kvbm-physical, NixlConnector, PdConnector, KV benchmarks)
      │
  NIXL Core API  (nixlAgent C++ / nixl._api Python / nixl-sys Rust)
      │  registerMem · createXferReq · postXfer · checkXfer · getNotifs
      │
  nixlPluginManager  (singleton, dlopen-based runtime plugin loader)
      │  loadBackendPlugin() · discoverPluginsFromDir() · getAvailPlugins()
      │
  nixlBackendEngine  (abstract: supportsRemote/Local/Notif · prepXfer · postXfer)
      │
  ┌───────────────────────────┬───────────────────────────┬──────────────────────┐
  │ Network / Fabric          │ Storage                   │ Specialized          │
  │ UCX (libucx, RDMA)        │ cuda_gds (libcufile)       │ UCCL (libuccl_p2p)   │
  │ LIBFABRIC (libfabric,     │ gds_mt (libcufile,        │ GPUNETIO (DOCA SDK)  │
  │   hwloc, libnuma)         │   Taskflow)               │ HF3FS (hf3fs_usrbio) │
  │ OFI (libfabric)           │ POSIX (libaio/liburing)   │ MOONCAKE (engine .so)│
  │                           │ OBJ (AWS SDK)             │ GUSLI                │
  │                           │ AZURE_BLOB (Azure SDK)    │                      │
  └───────────────────────────┴───────────────────────────┴──────────────────────┘
      │                             │                              │
  RDMA / Fabric HW              GPU + Storage HW              Specialized HW
  (IB, RoCE, EFA)               (GPU HBM, NVMe, SSD)          (BlueField NICs)
```

---

## 4. KVBM Internal Dependency Chain

```
vLLM (Python)
  └─ --kv-transfer-config JSON
       └─ DynamoConnector  extends KVConnectorBase_V1       [I6a]
       └─ PdConnector  =  DynamoConnector + NixlConnector   [I6b]
              │
       KvConnectorWorker / KvConnectorLeader  (PyO3, Rust)
              │
       kvbm-logical  ─── kvbm-common ─── dynamo-tokens (FNV hash)
              │
       kvbm-physical
         ├── kvbm-kernels  ─── cudarc = "=0.19.3"           [I9]
         │        └── CUDA Runtime (libcuda.so)
         └── dynamo-memory
               ├── cudarc wrapper  (GPU alloc, D2H/H2D copy)
               └── nixl-sys = "=0.10.1"                     [I7a]
                      └── NIXL Core (libnixl.so)
                               └── UCX / cuda_gds / POSIX plugins
```

---

## 5. llm-d-kv-cache: Correct Layer Placement

The repo spans **two separate architectural layers**:

```
L3 — Cache Coordination  (Go service, no GPU deps)
  pkg/kvcache       KV Cache Indexer (two-level LRU or Redis)
  pkg/kvevents      ZMQ subscriber + sharded FNV worker pool
  pkg/kvblock       FNV-64a block key computation, LRU index update
  pkg/tokenization  gRPC TokenizationService (Go + Python templates)
  api/indexerpb     IndexerService.proto  → gRPC scoring endpoint
  api/tokenizerpb   TokenizationService.proto
  go.mod dep:       go-zeromq/zmq4  hashicorp/lru  go-redis  grpc  k8s client-go

  COUPLING: llm-d-inference-scheduler imports llm-d-kv-cache v0.7.1 as Go module
  → scorer can call indexer in-process (zero-hop) OR via gRPC sidecar

L5 — KV Block Storage Tier  (Python + CUDA C++ extension)
  kv_connectors/llmd_fs_backend/
  SharedStorageOffloadingSpec   → vLLM OffloadingSpec      [I6c]
  BaseStorageOffloadingHandler  → vLLM OffloadingHandler   [I6d]
  StorageOffloadEngine (C++)    → CUDA thread pool + io
  tensor_copier.cu              → cudaMemcpyAsync or kernel
  gds_file_io.cpp               → cuFile via dlopen        [I10]
  dep (build):   torch==2.10.0  libnuma  libdl
  dep (runtime): vllm==0.18.1
  NO NIXL — shared filesystem (NFS/PVC) is the transfer medium
```

---

## 6. Common Building Blocks Shared Between Projects

| Building Block        | Dynamo                                | llm-d                                 | Alignment Required                          |
|-----------------------|---------------------------------------|---------------------------------------|---------------------------------------------|
| **vLLM**              | v0.19.0, KVConnectorBase_V1           | v0.18.1, OffloadingSpec               | Minor version gap; I6 API compat check      |
| **NIXL**              | v1.0.1; nixl-sys=0.10.1 (Rust+C++)    | NIXL v2 (PD-Sidecar, Python API)      | NIXL API version lock across both           |
| **FNV block hash**    | dynamo-tokens crate (Rust FNV-64a)    | pkg/kvblock (Go FNV-64a)              | **Must be byte-identical** — no drift ever  |
| **ZMQ KV events**     | Velo Events (Rust, NATS/ZMQ)          | go-zeromq/zmq4                        | Event schema: topic format + msgpack fields |
| **CUDA**              | cudarc=0.19.3 (Rust)                  | torch CUDA ext (llmd-fs-backend)      | Different paths; same GPU runtime           |
| **cuFile / GDS**      | NIXL cuda_gds plugin (dlopen)         | llmd-fs-backend gds_file_io (dlopen)  | Both optional; both runtime dlopen          |
| **Prometheus**        | Frontend /metrics                     | vLLM pod /metrics                     | Label names must match for WVA/Planner      |
| **ETCD**              | Service discovery + NIXL metadata     | Not used by llm-d (K8s-native)        | Namespace isolation if NIXL uses etcd       |
| **Kubernetes**        | Dynamo Operator, Helm                 | Gateway API CRDs, Helm                | CRD schemas are independent                 |
| **NCCL**              | L6b — TP/PP/DP/EP collectives (NVIDIA GPU, via vLLM workers) | Via vLLM workers (same as Dynamo) | libnccl version pin (cuMem, NVLink compat) |
| **RCCL**              | L6b — TP/PP/DP/EP collectives (AMD GPU/ROCm, via vLLM workers) | Via vLLM workers (same as Dynamo) | Same pynccl path; librccl.so.1 version pin  |
| **HCCL**              | L6b — Gaudi/HPU (vllm-gaudi fork only; not in canonical vLLM) | Not applicable                    | backend='hccl' in init_process_group        |
| **OneCCL**            | L6b — Intel XPU/CPU (via vLLM XpuCommunicator/CpuCommunicator) | Via vLLM workers (same as Dynamo) | torch-ccl package version lock              |
| **Intel IO Direct**   | NIXL cuda_gds plugin via `dlopen("libcufile.so")` — optional   | llmd-fs-backend `gds_file_io.cpp` via `dlopen` | Both are runtime-optional; cuFile ZE adapter (`gds-liburing-cufilewrapper`) replaces NVIDIA `libcufile.so` for Intel XPU; binary name compatibility required |

---

## 7a. NCCL / OneCCL / HCCL Placement in the Architecture

> Source verified against `~/src/vllm` (canonical vllm-project/vllm, main branch).
> Key files: `device_communicators/cuda_communicator.py`, `device_communicators/all2all.py`,
> `communication_op.py`, `parallel_state.py`, `utils/nccl.py`, `v1/worker/dp_utils.py`

```
COLLECTIVE COMMUNICATION LIBRARIES — WHERE THEY FIT
====================================================

These libraries (L6b) are consumed by L4 (Inference Engine). They are transport middleware,
not hardware — they sit at the same conceptual level as NIXL (L6).
They are NOT part of the KV cache P/D transfer path (that is NIXL's domain).
Exception: NixlEPAll2AllManager uses NIXL for MoE EP dispatch — see USE CASE 4.

─────────────────────────────────────────────────────────────────────────────
USE CASE 1: Tensor Parallel  (hottest path — fires every attention + MLP layer)
─────────────────────────────────────────────────────────────────────────────

  vLLM model forward pass (every layer, every micro-batch)
        │
  communication_op.py (thin wrappers that call into GroupCoordinator):
    tensor_model_parallel_all_reduce()         ← after each attention/MLP output
    tensor_model_parallel_all_gather()         ← reconstruct sequence-parallel activations
    tensor_model_parallel_reduce_scatter()     ← shard for sequence-parallel input
    tensor_model_parallel_gather()             ← collect on rank 0 (non-distributed path)
    broadcast_tensor_dict()                    ← scheduler metadata, sampler inputs
        │
  GroupCoordinator.all_reduce(input_)  →  CudaCommunicator.all_reduce()
        │
  AllReduce priority chain (tried in order; falls through if disabled/unavailable):
    1. SymmMemAllReduce   torch.ops.vllm.all_reduce_symmetric_with_copy
                         NVLink symmetric memory (CUDA only, H100+, VLLM_ALLREDUCE_USE_SYMM_MEM)
    2. QuickAllReduce     Quantized SHM (ROCm MI300 only; Q8/Q6/Q4; XGMI P2P)
    3. FlashInferAllReduce  flashinfer.comm.allreduce_fusion kernel
                           (NVIDIA; VLLM_ALLREDUCE_USE_FLASHINFER)
    4. CustomAllreduce    CUDA SHM fast path (intra-node only; skips for large tensors)
    5. pynccl             NCCL/RCCL — always available; used cross-node

  AllGather / ReduceScatter / AllGatherV / ReduceScatterV:
    All route through pynccl (no SHM fast paths; variable-size variants also supported)

─────────────────────────────────────────────────────────────────────────────
USE CASE 2: Pipeline Parallel  (inter-stage, per micro-batch)
─────────────────────────────────────────────────────────────────────────────

  gpu_model_runner.py (simplified):
    if not pp_group.is_last_rank:
        pp_group.send_tensor_dict(hidden_states)   ← pynccl.send() under the hood
    broadcasted = pp_group.broadcast_tensor_dict(  ← final logits: last rank → all PP
        model_output, src=last_pp_rank)

  ⚠ CPU control metadata uses GLOO group (cpu_group in GroupCoordinator)
  ⚠ Tensor data uses NCCL/device group (device_group in GroupCoordinator)

─────────────────────────────────────────────────────────────────────────────
USE CASE 3: Data Parallel  (DP batch coordination)
─────────────────────────────────────────────────────────────────────────────

  dp_utils.py::coordinate_batch_across_dp():
    AllReduce on dp_group  — sync num_tokens (batch size) across DP replicas each step
    AllReduce on dp_group  — parallel config version sync (max/min for consistency)
    AllGatherV on dp_group — MoE: gather hidden states from all DP ranks for EP dispatch

─────────────────────────────────────────────────────────────────────────────
USE CASE 4: Expert Parallel (MoE All-to-All — pluggable backends)
─────────────────────────────────────────────────────────────────────────────

  CudaCommunicator.dispatch() / .combine() → self.all2all_manager.*
    (backend selected by VLLM_MLA_ALL2ALL_BACKEND env var or model config)

  Backend options (cuda_communicator.py, all2all.py):
    naive                       NaiveAll2AllManager      AllReduce-based (debug/test)
    allgather_reducescatter     AgRsAll2AllManager        AllGather + ReduceScatter
    deepep_high_throughput      DeepEPHTAll2AllManager    DeepEP kernels (NVIDIA)
    deepep_low_latency          DeepEPLLAll2AllManager    DeepEP kernels (NVIDIA)
    nixl_ep                     NixlEPAll2AllManager      NIXL elastic EP dispatch ◄─┐
    flashinfer_nvlink_two_sided FlashInferNVLinkTwoSided  FlashInfer NVLink 2-sided  │
    flashinfer_nvlink_one_sided FlashInferNVLinkOneSided  FlashInfer NVLink 1-sided  │
    mori                        MoriAll2AllManager         AMD ROCm MoRI kernels      │
                                                                                       │
  ⚠ NixlEPAll2AllManager uses nixl_ep Python package for MoE expert dispatch         │
    (SEPARATE from the KV-cache NixlConnector at L4 — two distinct NIXL consumers)   ◄┘

  EPLB expert weight redistribution (rebalance_execute.py):
    pynccl.batch_isend_irecv()  — move expert weight tensors between EP ranks at runtime

─────────────────────────────────────────────────────────────────────────────
USE CASE 5: P2P KV Transfer via NCCL (alternative to NIXL; not used by Dynamo/llm-d)
─────────────────────────────────────────────────────────────────────────────

  vLLM P2pNcclConnector  (kv_transfer/kv_connector/v1/p2p/)
        │
  P2pNcclEngine:
    ├── ZMQ for control plane (tensor IDs, worker addresses, sizes)
    └── pynccl_wrapper (NCCLLibrary ctypes) → own dedicated ncclComm
        ├── NCCL env tuning: NCCL_MAX/MIN_NCHANNELS, NCCL_CUMEM_ENABLE=1
        └── send/recv KV blocks Prefill→Decode (cross-node)

  ⚠ MUTUALLY EXCLUSIVE with NixlConnector/DynamoConnector.
    Dynamo and llm-d both use NIXL for KV transfer; P2pNcclConnector is vLLM's
    own fallback for deployments where NIXL is unavailable.

─────────────────────────────────────────────────────────────────────────────
LAYER MAP SUMMARY
─────────────────────────────────────────────────────────────────────────────

  L4 — vLLM workers call collective ops every forward pass
       DeviceCommunicator abstraction hides which .so is loaded
       ↓
  L6b — CCL libraries (transport middleware, same conceptual level as NIXL):
        NVIDIA GPU:  libnccl.so.2    NVLink/PCIe/InfiniBand  [I18]
        AMD GPU:     librccl.so.1    XGMI/PCIe               [I22]  ← same CudaCommunicator
        Intel XPU:   liboneccl.so    RDMA/PCIe               [I20]
        CPU:         liboneccl.so    shared-memory/TCP        [I20]
        Intel Gaudi: libhabanalabs-hccl.so  Gaudi RDMA        [I19]  ← vllm-gaudi fork only
       ↓
  L10 — Hardware + low-level drivers (CUDA/ROCm/OneAPI drivers, RDMA drivers, NICs)
    NCCL/CCL = activations, weight shards, MoE routing (within logical worker boundary)
    NIXL      = KV cache blocks (between Prefill/Decode workers, cross-worker/cross-node)
    Exception: NixlEPAll2AllManager bridges both — NIXL for MoE EP token dispatch.

  Dynamo snapshot.py:  sets NCCL_SOCKET_IFNAME=lo, NCCL_CUMEM_ENABLE=0,
  NCCL_P2P_DISABLE=0 during checkpoint — confirms NCCL live during worker saves.
```

```
Phase 1 — freeze FIRST (unblocks everything):
  I6f  FNV block hash spec (chunk size, algorithm, byte order — cross-language)
  I6   vLLM connector API version alignment (which vLLM, which sub-APIs)
  I3d  ZMQ KV event schema (topic format, msgpack field names)
  I7a  nixl-sys exact version lock

Phase 2 — freeze to enable most workstreams:
  I4a  Velo RPC disaggregated_params struct
  I3b  IndexerService gRPC proto
  I5   NixlHandshakePayload schema (P/D GPU xfer handshake)
  I13  FFI module API surface (Rust→Python exposed types)

Phase 3 — freeze for E2E validation:
  I14  Prometheus metrics label schema
  I16  K8s CRD schemas
```

**Workstream independence map:**

| Workstream | Layers | Gating interfaces | Independent from |
|------------|--------|-------------------|-----------------|
| A — Inference engine | L4 | I4a, I4b, I6, I3c, I3d | L1, L3, L5, L6, L6b, L7, L8, L9 |
| B — NIXL fabric | L6 | I11, I7a | ALL upper layers |
| B2 — CCL libraries | L6b | I18, I19, I20, I22 | L5, L6, L7, L8, L9 |
| C — KVBM block manager | L5 | I6a, I7a, I9 | L1, L2, L3, L8, L9 |
| D — llm-d KV indexer | L3 | I3b, I3d | L1, L5, L6, L6b, L7, L8, L9 |
| E — llmd-fs-backend | L5 | I6c, I6d, I10 | L1, L2, L3, L6, L6b, L7, L8, L9 |
| F — Autoscaling | L8 | I14, I15 | L3, L4, L5, L6, L6b, L7 |
| G — K8s platform | L9 | I16 | L3, L4, L5, L6, L6b |
| H — Rust runtime foundation | L7 | I13 (freeze first) | L5, L6, L9 — but blocks L1, L2, L8 |

---

## 7b. Direct NIXL / CCL Usage — Dynamo Core vs KVBM vs llm-d

> Source verified against `~/src/dynamo` and `~/src/llm-d`.

```
WHO CALLS NIXL OR CCL DIRECTLY?
================================

┌──────────────────────────────┬──────────────────────────────────┬─────────────────────────────────┐
│ Component                    │ NIXL                             │ CCL (NCCL/RCCL/etc.)            │
├──────────────────────────────┼──────────────────────────────────┼─────────────────────────────────┤
│ DYNAMO CORE                  │                                  │                                 │
│  lib/runtime                 │ ✗ None                           │ ✗ None                          │
│  velo-common/events/         │ ✗ None                           │ ✗ None                          │
│  velo-transports             │ ✗ None                           │ ✗ None                          │
│  kv-router                   │ ✗ None                           │ ✗ None                          │
│  tokens / protocols / config │ ✗ None                           │ ✗ None                          │
│  dynamo/components (Python)  │ ⚙ Config only — passes           │ ✗ None                          │
│                              │   --kv-transfer-config to vLLM;  │                                 │
│                              │   reads connector type string     │                                 │
│  dynamo.nixl_connect (Python)│ ✓ Direct — wraps nixl._api for   │ ✗ None                          │
│  (lib/bindings/python)       │   RDMA transfers in multimodal   │                                 │
│                              │   encode workers (not KV cache)  │                                 │
├──────────────────────────────┼──────────────────────────────────┼─────────────────────────────────┤
│ KVBM                         │                                  │                                 │
│  kvbm-common/logical/kernels │ ✗ None                           │ ✗ None                          │
│  kvbm-physical               │ ⚙ Metadata only — serializes     │ ✗ None                          │
│                              │   NixlMetadata into block        │                                 │
│                              │   descriptors; no agent calls    │                                 │
│  dynamo-memory crate         │ ✓ Direct — owns NixlAgent        │ ✗ None                          │
│                              │   lifecycle: new(), add_backend()│                                 │
│                              │   NixlBackendConfig from env     │                                 │
│  block_manager/worker.rs     │ ✓ Direct — calls                 │ ✗ None                          │
│                              │   dynamo_memory::NixlAgent::new()│                                 │
│                              │   per KVBM worker                │                                 │
│  block_manager/transfer.rs   │ ✓ Direct — nixl_sys::Nixl-       │ ✓ Direct — cudarc::nccl         │
│                              │   Descriptor for block RDMA xfer │   TransferMode::Replicated only │
│  block_manager/nccl_         │ ✗ None                           │ ✓ Direct — ncclGetUniqueId,     │
│    bootstrap.rs              │                                  │   ncclCommInitRank; own ncclComm│
│                              │ DEFAULT: Sharded (no NCCL)       │   Default mode = Sharded = off  │
├──────────────────────────────┼──────────────────────────────────┼─────────────────────────────────┤
│ vLLM WORKERS                 │                                  │                                 │
│  (in any Dynamo deployment)  │ ✓ Via NixlConnector /            │ ✓ Via CudaCommunicator/pynccl   │
│                              │   DynamoConnector (KV cache P/D) │   TP/PP/DP/EP collectives       │
├──────────────────────────────┼──────────────────────────────────┼─────────────────────────────────┤
│ llm-d SIDECAR (Go)           │ ⚙ Orchestrates only — routes     │ ✗ None                          │
│  connector_nixlv2.go         │   HTTP between P/D vLLM pods;    │                                 │
│                              │   extracts kv_transfer_params    │                                 │
│                              │   from prefiller JSON response;  │                                 │
│                              │   NO direct nixl._api calls      │                                 │
│ vLLM WORKERS                 │                                  │                                 │
│  (in any llm-d deployment)   │ ✓ Via NixlConnector              │ ✓ Via CudaCommunicator/pynccl   │
└──────────────────────────────┴──────────────────────────────────┴─────────────────────────────────┘

KEY DISTINCTIONS
════════════════

Dynamo core = pure orchestration (Velo RPC, ETCD, routing, scheduling).
  No transport library dependencies. Only knows connector *names* as config strings.

KVBM = the transport-aware layer.
  NIXL: dynamo-memory owns the NixlAgent; kvbm-physical attaches block metadata.
  NCCL: block_manager/nccl_bootstrap.rs creates its OWN ncclComm (separate from
        vLLM's TP comm) for TransferMode::Replicated — replicates KV blocks across
        KVBM worker ranks. Default mode is Sharded (no NCCL in KVBM).

dynamo.nixl_connect (Python) = second NIXL consumer in Dynamo, independent of KVBM.
  Used by multimodal encode workers to move tensor data via NIXL RDMA.
  NOT related to KV cache transfer path.

llm-d = orchestration only at the Go layer.
  The sidecar proxy coordinates which vLLM pods talk; actual NIXL GPU transfers
  happen inside the vLLM workers on both the prefill and decode sides.
  llm-d itself has zero NIXL or CCL library dependencies.
```

---

## 8. Key Risks (Corrected)

| Risk | Root Cause | Impact | Mitigation |
|------|-----------|--------|-----------|
| **vLLM version skew** | Dynamo=0.19.0, llm-d=0.18.1; I6 APIs differ | Connectors incompatible across versions | Agree shared vLLM base; track v1 API changelog jointly |
| **FNV block hash drift** | Three independent implementations: Rust (dynamo-tokens), Go (kvblock), Python (vLLM) | 0% cache hit rate on any mismatch | Single cross-language spec + byte-level test vectors |
| **nixl-sys exact pin** | = "=0.10.1" exact pin in dynamo-memory | NIXL update requires coordinated kvbm-physical + dynamo-memory bump | NIXL semantic versioning + deprecation window policy |
| **Maturin FFI cascade** | Rust type changes in dynamo_runtime/dynamo_llm | Frontend + Router + Planner must all update simultaneously | Additive-only FFI changes; version the exposed PyO3 types |
| **NIXL plugin load failure** | dlopen at runtime; missing .so = silent failure | Wrong transfer path or crash | sanity_check.py validation on pod startup; expose plugin list as metric |
| **GDS silently disabled** | libcufile loaded via dlopen in both NIXL and llmd-fs-backend | GPU→File bypasses CPU staging; missing driver → silent POSIX fallback | Explicit GDS capability check at init; expose `gds_enabled` gauge |
| **ZMQ event gaps** | KV events are best-effort (no ack) | Stale KV index → degraded routing quality (not correctness) | Gap-tolerant indexer; periodic AllBlocksCleared flush from vLLM |
| **ETCD shared by NIXL + svc disc** | Both Dynamo service discovery and NIXL metadata can use same etcd | etcd overload degrades both planes simultaneously | Separate etcd namespaces; rate-limit NIXL metadata writes |
| **PdConnector version coupling** | PdConnector = DynamoConnector + vLLM's NixlConnector; both vLLM-version-specific | Update to new vLLM may break either half independently | Integration test matrix: KVBM connector × NixlConnector × vLLM version |
| **gds-linux not upstream** | 11-patch series in `tsg-/gds-linux:dmabuf-rw-xfs_tsg` is a custom development branch; P01–P11 not yet submitted upstream | Intel IO Direct requires custom kernel build; cannot deploy on standard distro kernels; upstream acceptance in block/nvme/io_uring/XFS trees takes 1–3 kernel release cycles | DKMS fallback (KU-4) for near-term deployments; parallel RHEL/Ubuntu engagement (DI-1, DI-2); P05–P09 already co-authored with io_uring maintainer — accelerates merge path |
| **libcufile.so name conflict** | Intel IO Direct cuFile shim uses `libcufile.so.1` binary name; NVIDIA GDS also ships `libcufile.so.1` | Installing both on the same system creates undefined symbol resolution; LD_LIBRARY_PATH dependency is brittle | Distro package name `libcufile-intel` vs `libcufile` (NVIDIA); legal/OSPO approval required (OS-2, DI-3) |

---

## 9. GPU Communication Libraries SW Team: Code Changes Beyond Integration

> **Team:** GPU Communication Libraries SW
> **Owns:** OneCCL (L6b) · NIXL (L6) · UCX · libfabric · Intel IO Direct
>   (vendor-neutral GPU Direct Storage solution)
> **Layers:** L5 (KV Block Management & Storage) + L6 (Transport Abstraction) + L6b (Collective Comms)
>
> Three areas the team **already owns** — intentionally excluded below:
> 1. **NIC-specific plugin work** — NIXL UCX plugin (`UCS_MEMORY_TYPE_ZE` for Intel XPU),
>    libfabric provider config (`FI_HMEM_ZE`), UCX HMEM Level Zero wiring
> 2. **Intel IO Direct** — the two-component GPU-direct-to-storage stack under `~/src/gds/`:
>    - **`gds-liburing`**: liburing fork adding `io_uring_register_buffers_dmabuf()` — kernel I/O layer
>    - **`gds-liburing-cufilewrapper`**: cuFile API compatibility shim (`libcufile.so`) implemented
>      with Level Zero (`ZE_EXTERNAL_MEMORY_TYPE_FLAG_DMA_BUF` → dmabuf fd) + `gds-liburing`
>    Requires Linux kernel 6.17+ with dmabuf-rw patches; replaces NVIDIA `cuda_gds` / `libcufile.so`
> 3. **Level Zero / OneCCL (vLLM application layer)** — `XpuCommunicator` integration,
>    `oneccl_bindings_for_pytorch` TP/PP/DP collectives at L4

```
UPSTREAM NIXL BUILD SYSTEM — Key Facts (verified from ~/src/nixl)
═══════════════════════════════════════════════════════════════════
nixl_mem_t (src/api/cpp/nixl_types.h):
  enum nixl_mem_t {DRAM_SEG, VRAM_SEG, BLK_SEG, OBJ_SEG, FILE_SEG}
  Platform-neutral. VRAM_SEG covers any GPU vendor.

Plugin build gating (src/plugins/meson.build):
  UCX plugin   — gated on ucx_dep.found() ONLY.
                 cuda_dep is listed in dependencies but is a no-op when
                 not found (Meson ignores not-found optional deps).
                 → UCX plugin builds WITHOUT CUDA on Intel hardware.
  LIBFABRIC    — gated on libfabric_dep.found(). cuda_dep is optional
                 (compile flag -DHAVE_CUDA added only if found).
                 → libfabric plugin builds WITHOUT CUDA.
  GDS / GDS_MT — explicitly gated on cuda_dep.found() and not disabled.
                 → Replaced by Intel IO Direct (area 2). ✓
  cuda_gds     — CUDA hard-required. Same — area 2. ✓

UCX GPU Device API (meson.build:307):
  gated on: ucx_dep.found() AND cuda_dep.found() AND nvcc_prog.found()
  Feature: ucp_device_remote_mem_list_create — GPU-side UCX ops (CUDA+NVCC)
  → NOT available for Intel XPU. This is a GAP — see item A1 below.

CONSEQUENCE: NIXL plugin code changes for Intel are within area 1.
The code change burden outside area 1/2/3 is in:
  • NIXL build system (UCX GPU Device API gap)
  • KVBM Rust crates (all CUDA-specific)
  • vLLM XPU EP path (NixlEPAll2AllManager not wired for XPU)

VERDICT
════════
Integration work:  ~70%  (plugin wiring, driver config, perf tuning, testing)
Code changes:      ~30%  (5 categories, 17 items — listed below)
```

---

### 9.1  Category A — NIXL: Intel XPU Code Gaps (Build System + Plugins)

> **UCX ownership boundary:** The NIXL UCX plugin (`~/src/nixl/src/plugins/ucx/`) is a thin
> wrapper around **openucx** (`~/src/ucx`). Item A1 involves changes at the **openucx layer**
> — specifically `src/uct/ze/` (ZE copy transport: `ze_copy_ep.c`, `ze_copy_iface.c`,
> `ze_copy_md.c`), `src/ucm/ze/` (ZE memory management: `zemem.c`, `ze_alloc.c`), and
> `src/uct/ib/mlx5/` (IB HMEM ZE wiring for RDMA with XPU memory buffers).
> These are **upstream openucx contributions**; NIXL's plugin wrapper picks them up once
> the UCX library exposes the ZE GPU Device API. Item A5 is a diagnostic fix in the NIXL
> plugin only.

```
VERIFIED FROM ~/src/nixl AND ~/src/ucx SOURCE TREES:

openucx ZE layer (~/src/ucx):
  src/uct/ze/copy/  — ze_copy_ep.c, ze_copy_iface.c, ze_copy_md.c (ZE UCT transport)
  src/uct/ze/base/  — ze_base.c/h (ZE device context)
  src/ucm/ze/       — zemem.c/h (ZE memory hooks), ze_alloc.c (ZE allocator)
  src/tools/perf/ze/— ZE perf alloc helpers
  → ZE UCT transport exists but IB HMEM ZE wiring in src/uct/ib/mlx5/ needs validation
    and potential extension for GPU-registered RDMA buffers over IB/RoCE.

UCX plugin (src/plugins/ucx/ucx_utils.cpp):
  memReg() calls ucp_mem_map() → ucp_mem_query() to verify VRAM type.
  UCX internally handles ze_ipc via UCS_MEMORY_TYPE_ZE transport —
  NIXL UCX plugin does NOT call zeMemGetIpcHandle directly.
  GAP: error message on VRAM-detected-as-HOST is CUDA-specific text:
       "UCX is likely not configured with CUDA support"
       → Should say "CUDA or Level Zero" on XPU.

libfabric backend (src/utils/libfabric/libfabric_rail_manager.cpp:195):
  Runtime selection code:
    if   (getNumNvidiaAccel() > 0) → FI_HMEM_CUDA
    elif (getNumAwsAccel()   > 0) → FI_HMEM_NEURON
    else                          → FI_HMEM_SYSTEM   ← Intel XPU lands HERE
  getSupportedMems() adds VRAM_SEG ONLY when runtime_ == FI_HMEM_CUDA.
  → On Intel XPU: libfabric backend reports NO VRAM_SEG support.
  → All KV cache transfers silently fall back to DRAM_SEG (host staging).
  THIS IS A CRITICAL GAP: not a NIC provider config, it is topology
  detection + runtime branch logic inside the libfabric backend itself.

nixlLibfabricTopology (src/utils/libfabric/libfabric_topology.cpp):
  Only has isNvidiaAccel() and getNumAwsAccel() methods.
  No isIntelAccel() / getNumIntelAccel() exists.
```

| # | File | Change Required | Why Not Area 1 |
|---|---|---|---|
| A1 | `meson.build:307` + `src/api/gpu/ucx/` + **openucx `src/uct/ze/` + `src/uct/ib/mlx5/`** | **Extend UCX GPU Device API to Intel XPU — two layers.** (1) **openucx (`~/src/ucx`)**: verify and extend ZE UCT transport (`uct/ze/copy/`, `uct/ze/base/`) and UCM ZE memory hooks (`ucm/ze/`) for GPU-registered IB RDMA buffers over RoCE/IB; patch `uct/ib/mlx5/` HMEM ZE wiring to ensure `UCS_MEMORY_TYPE_ZE` is registered for IB MRs — these changes go upstream to openucx. (2) **NIXL build gate** (`meson.build:307`): add Level Zero + DPC++ detection branch alongside `cuda_dep AND nvcc_prog` to enable `nixl_device_ze.hpp` / DPC++ equivalent of `nixl_device.cuh` for GPU-side UCX ops. | Area 1 covers NIC-side UCX memory registration (`ucp_mem_map` → `UCS_MEMORY_TYPE_ZE`). Item A1 is the GPU-side device kernel interface (zero-copy GPU-initiated UCX) — openucx UCT ZE transport + NIXL device API gate are separate components from the NIC plugin config. |
| A2 | `src/utils/libfabric/libfabric_topology.{cpp,h}` | **Add Intel XPU accelerator detection.** Add `isIntelAccel(hwloc_obj_t)` and `getNumIntelAccel()` alongside existing `isNvidiaAccel()` and `getNumAwsAccel()`. Uses hwloc PCI vendor ID for Intel GPU (vendor `0x8086`, class `0x0302` / `0x0380`). | topology.cpp is a shared utility, not a NIC provider. Area 1 covers the OFI provider wiring (`FI_HMEM_ZE` flag in `fi_info`), not accelerator topology discovery. |
| A3 | `src/utils/libfabric/libfabric_rail_manager.cpp:195` | **Add `FI_HMEM_ZE` runtime branch.** Add `else if (topology->getNumIntelAccel() > 0) runtime_ = FI_HMEM_ZE;` after the NEURON branch. Without this, Intel XPU falls to `FI_HMEM_SYSTEM` and `getSupportedMems()` never returns `VRAM_SEG`. | This is the libfabric backend runtime selection logic — not the OFI provider hint passed to `fi_getinfo()`. Area 1 covers the NIC-level `FI_HMEM_ZE` provider configuration; this is the per-transfer memory-type dispatch inside the backend. |
| A4 | `src/plugins/libfabric/libfabric_backend.cpp:305–325` | **Add `FI_HMEM_ZE` initialization block.** Currently only `FI_HMEM_CUDA` has a context-management init block (`#ifdef HAVE_CUDA`). For ZE: add `#ifdef HAVE_LEVEL_ZERO / if (runtime_ == FI_HMEM_ZE)` block that initializes `ze_context_handle_t` and sets `fi_mr_attr.iface = FI_HMEM_ZE`. Also fix the VRAM_SEG registration path at line 705–765 to handle ZE alongside CUDA. | This is the NIXL libfabric backend C++ initialization — not the OFI provider selection (area 1). The provider is already selected by the libfabric framework; this code sets up the NIXL-side ZE context used for memory registration. |
| A5 | `src/plugins/ucx/ucx_utils.cpp:591–594` | **Fix CUDA-specific error message in `memReg`.** The error `"UCX is likely not configured with CUDA support"` is fired for any `VRAM_SEG` registered as HOST by UCX, including ZE memory. Should be platform-neutral: `"UCX is not configured with GPU memory support (CUDA/ZE)"`. Minor but causes misdiagnosis on Intel. | Diagnostic accuracy — not a NIC plugin change. |

### 9.2  Category B — KVBM Rust: Intel Platform Abstraction (L5)

> KVBM's Rust crates (`dynamo-memory`, `kvbm-physical`, `kvbm-kernels`, `block_manager`)
> have hard `cudarc` (CUDA) dependencies. These are the **memory management and async DMA
> layer in Rust**, distinct from NIC plugins (area 1), Intel IO Direct storage (area 2),
> and OneCCL vLLM-level collectives (area 3).

| # | File / Crate | Change Required | Why Not Area 1 / 2 / 3 |
|---|---|---|---|
| B1 | `dynamo-memory/src/device.rs` | **Level Zero device memory context.** Replaces `cudarc::driver::CudaContext` + `cudaMalloc`/`cudaFree`. Intel equivalent: `ze_context_handle_t` + `zeMemAllocDevice`/`zeMemFree`. Need a `GpuContext` trait or `#[cfg(feature = "cuda"/"level-zero")]` impls. | Memory allocation, not collectives (OneCCL), not NIC transport, not storage. |
| B2 | `dynamo-memory/src/pinned.rs` | **Level Zero pinned host memory.** Replaces `cudarc::driver::result::malloc_host` + `CU_MEMHOSTALLOC_WRITECOMBINED`. Intel: `zeMemAllocHost` with `ZE_HOST_MEM_ALLOC_FLAG_BIAS_WRITE_COMBINED`. Same trait abstraction as B1. | CPU-side staging buffer allocation — not covered by any of the three stated areas. |
| B3 | `kvbm-physical/src/transfer/executor/cuda.rs` | **Level Zero / SYCL async DMA executor.** Replaces `cudaMemcpyBatchAsync` (CUDA 12.9+) + `CudaStream`. Intel path: `zeCommandListAppendMemoryCopy` + `ze_command_queue` OR SYCL queue copy. `TransferPreferences::Automatic` must detect Intel GPU and route to `nixl.rs` executor or new Level Zero executor — not the CUDA one. | H2D/D2H/D2D copy path in KVBM, completely separate from NIC plugins, storage, and collectives. |
| B4 | `kvbm-kernels/cuda/` + `build.rs` | **SYCL/DPC++ port of CUDA kernels.** `memcpy_batch.cu`, `vectorized_copy.cu`, `permute_kernels.cu` compiled by `nvcc`. For Intel: DPC++ (`icpx`) equivalents using `sycl::` or `oneapi::mkl`. `build.rs` needs Intel SYCL toolchain detection + fallback stub path. | Device-side memory manipulation kernels — no relation to NIC, storage, or collectives. |
| B5 | `block_manager/distributed/nccl_bootstrap.rs` | **OneCCL for `TransferMode::Replicated` in Rust.** Uses `cudarc::nccl` (CUDA-only). For Intel: replace with OneCCL point-to-point send/recv OR disable replicated mode for Intel with a compile-time/runtime guard. Note: this is KVBM's **own** comm object, separate from vLLM's TP communicator — `TransferMode::Sharded` is the default and avoids this path entirely. | Area 3 ("OneCCL application changes") refers to the vLLM Python layer. This is Rust FFI to the CCL library inside KVBM itself — a different file, a different binding layer. |
| B6 | `kvbm-physical/src/layout/builder.rs:349` | **Remove CUDA-only device guard.** `bail!("tensor at index {} is not on a CUDA device")` blocks ALL block registration on non-CUDA hardware. Replace with platform-agnostic device presence check. | One-line critical blocker for all non-NVIDIA operation; must be a code change. |

### 9.3  Category C — NIXL Agent Config, Versioning & Metadata

| # | File / Module | Change Required | Why Not Integration Only |
|---|---|---|---|
| C1 | `dynamo-memory/src/nixl/agent.rs` | **`NixlBackendConfig` fast-fail on platform mismatch.** Currently `add_backend_with_params` bails on unknown plugin name but does NOT enforce backend-platform consistency (e.g., requesting UCX_CUDA backend on an Intel XPU node where the CUDA HMEM provider is absent). Add a platform-consistency check that fails loudly with actionable error rather than a runtime NIXL plugin-load error. | Config validation code change — cannot be resolved by tuning environment variables alone. |
| C2 | `nixl-sys` crate (exact pin `=0.10.1`) | **Version uplift CI gate for Intel backend landing.** When NIXL ships Level Zero memory support or Intel IO Direct plugin, Dynamo's exact-pin requires coordinated bump in `dynamo-memory/Cargo.toml` + `kvbm-physical/Cargo.toml`. Need an automated CI test that verifies NIXL ABI stability before accepting pin changes. | Code process change — not achievable through configuration. Absent this gate, Intel backend delivery can silently break Dynamo's KVBM at any pin bump. |
| C3 | NIXL peer-discovery metadata / ETCD | **Namespace isolation: NIXL peer-discovery vs Dynamo service-discovery.** Both use the same etcd instance with no code enforcing separation today. Add a namespace prefix parameter to `NixlAgent::new()` init call + to Dynamo runtime's etcd client. Both are Rust code changes. | No env-var or config file can fix this — the parameter does not exist today. |

### 9.4  Category D — vLLM XPU EP Path for MoE All-to-All (L4 ↔ L6 / L6b)

> Verified against canonical `vllm-project/vllm` (not a fork).
> `XpuCommunicator` (Intel XPU device communicator) supports only two EP backends:
> `NaiveAll2AllManager` and `AgRsAll2AllManager` (AllGather + ReduceScatter fallback).
> `NixlEPAll2AllManager` (the NIXL-based backend this team owns) is NOT wired for XPU.

| # | File / Module | Change Required | Why Not Area 3 |
|---|---|---|---|
| D1 | `vllm/distributed/device_communicators/xpu_communicator.py` | **Wire `NixlEPAll2AllManager` for XPU.** Add `nixl` as a valid `all2all_backend` choice in `XpuCommunicator.__init__`. NIXL's `nixl_mem_t` is already `VRAM_SEG` (platform-neutral), so if the UCX/libfabric plugin handles Intel memory registration correctly (area 1), the EP manager should function. This wiring is missing and must be added. | Area 3 covers OneCCL TP/PP AllReduce/AllGather. MoE EP token dispatch uses NIXL, not OneCCL — and this team owns `NixlEPAll2AllManager`. Adding it to XpuCommunicator is code, not config. |
| D2 | `vllm/distributed/all2all.py` | **Validate `NixlEPAll2AllManager` on Intel backend end-to-end.** `NixlEPAll2AllManager` calls `nixl._api.init_agent()`, `register_memory()`, `make_prepped_xfer()`. Verify these work with Intel UCX/libfabric plugin for `VRAM_SEG` memory on XPU. May require XPU-specific buffer alignment or `ze_fence` synchronization in the NIXL transfer completion path. | The EP manager code may need targeted changes for XPU. Not covered by area 1 (plugin only) or area 3 (collectives only). |

### 9.5  Category E — Cross-Cutting: vLLM Connector API + Multi-Agent Scoping

| # | File / Module | Change Required | Note |
|---|---|---|---|
| E1 | `bindings/kvbm` PyO3 + `KVConnectorBase_V1` | **vLLM connector API version tracking.** Dynamo (v0.19.0) and llm-d (v0.18.1) pin to different vLLM versions. When vLLM promotes `KVConnectorBase_V1` from experimental, `DynamoConnector` and `PdConnector` PyO3 bindings need updates. Ongoing. | Versioned interface contract; requires engineering process, not just testing. |
| E2 | `block_manager/distributed/worker.rs` | **Multi-NixlAgent namespace scoping.** Currently one `NixlAgent("kvbm-worker-{id}")` per KVBM worker. If both NIXL-KV-transfer AND `NixlEPAll2AllManager` are active on the same GPU node, agent name conflicts arise. Need scoped naming convention (e.g., `"kvbm-kv-{id}"` vs `"kvbm-ep-{id}"`). | Design-level code change; required when Intel XPU runs both KV P/D transfer and MoE EP dispatch. |

---

### 9.6  Summary

```
INTEGRATION ONLY — no code change required beyond 3 stated areas:
  • Configure DYN_KVBM_NIXL_BACKEND_UCX / OFI / Intel-IO-Direct env vars
  • Tune NIXL plugin parameters (UCX transport mode, libfabric provider, rail count)
    — libfabric plugin already builds without CUDA (verified: ~/src/nixl meson)
    — UCX plugin already builds without CUDA (cuda_dep is optional no-op)
  • OFI provider-level FI_HMEM_ZE wiring (fi_getinfo hint flags) — area 1 ✓
  • UCX UCS_MEMORY_TYPE_ZE wiring (UCX transport config) — area 1 ✓
  • Deploy Intel IO Direct drivers + set GDS-equivalent tuning knobs — area 2 ✓
  • Configure vLLM XpuCommunicator OneCCL TP/PP (already in place) — area 3 ✓
  • ETCD topology planning (after namespace isolation code is in place — C3)
  • llm-d Go sidecar: zero NIXL/CCL deps — pure HTTP orchestration; no changes needed

REQUIRES CODE CHANGES beyond the 3 stated areas:
  ┌─ NIXL Intel XPU gaps (verified from ~/src/nixl) ───────────────────────────────────┐
  │  A1  meson.build:307 + openucx src/uct/ze/ + src/uct/ib/mlx5/ — UCX GPU Device API  │
  │                                            for XPU (openucx upstream + NIXL build gate) │
  │  A2  libfabric_topology.{cpp,h}          — Add isIntelAccel() / getNumIntelAccel() │
  │  A3  libfabric_rail_manager.cpp:195      — Add FI_HMEM_ZE runtime branch            │
  │                                            (CRITICAL: without this, no VRAM_SEG)    │
  │  A4  libfabric_backend.cpp:305–325       — FI_HMEM_ZE init block + ZE mr attr      │
  │  A5  ucx_utils.cpp:591–594              — Fix CUDA-specific error message           │
  ├─ KVBM Rust platform abstraction (verified from ~/src/dynamo) ──────────────────────┤
  │  B1  dynamo-memory/device.rs     — Level Zero device memory (ze_context_handle_t)  │
  │  B2  dynamo-memory/pinned.rs     — Level Zero pinned host memory (zeMemAllocHost)  │
  │  B3  kvbm-physical executor      — Level Zero / SYCL async DMA (replace CUDA one)  │
  │  B4  kvbm-kernels/cuda/          — SYCL/DPC++ port of device-side copy kernels     │
  │  B5  nccl_bootstrap.rs           — OneCCL P2P for TransferMode::Replicated (Rust)  │
  │  B6  layout/builder.rs:349       — Remove CUDA-only device guard (1-line blocker)  │
  ├─ NIXL agent config & versioning ──────────────────────────────────────────────────┤
  │  C1  NixlBackendConfig           — Platform-consistency fast-fail validation        │
  │  C2  nixl-sys pin =0.10.1        — Coordinated uplift CI gate for Intel backend     │
  │  C3  ETCD namespace isolation    — Code change in NixlAgent init + Dynamo runtime   │
  ├─ vLLM XPU EP path (verified from ~/src/vllm) ─────────────────────────────────────┤
  │  D1  XpuCommunicator             — Wire NixlEPAll2AllManager as valid XPU backend   │
  │  D2  NixlEPAll2AllManager        — Validate + fix for XPU buffer/sync requirements  │
  ├─ Cross-cutting ────────────────────────────────────────────────────────────────────┤
  │  E1  KVConnectorBase_V1          — vLLM connector API version tracking (ongoing)    │
  │  E2  NixlAgent naming            — Scoped names for KV-transfer vs EP-dispatch      │
  └────────────────────────────────────────────────────────────────────────────────────┘

MOST CRITICAL (blocks all XPU KV transfer via libfabric):
  A2 + A3 + A4 — libfabric backend has no Intel XPU runtime path.
  Without A3, getSupportedMems() never returns VRAM_SEG on XPU.

HEAVIEST WORK: B1–B4 (KVBM Rust abstraction, ~45% of code change effort)
               A2–A4 (libfabric ZE path, ~25%)
CRITICAL BLOCKERS: A3, B6, D1
```

---

## 10. GPU Comms SW Team: Functional and Performance Validation Plan

> **Scope:** Component-level (unit + micro-benchmark) **and** end-to-end stack validation
> for NIXL, OneCCL, Intel IO Direct, UCX, libfabric, and their integration into
> Dynamo (KVBM) and vLLM (XpuCommunicator, NixlEPAll2AllManager).
> Each component has a dedicated validation phase before stack integration.

---

### 10.1  NIXL — Component Functional Validation

```
Test infrastructure (~/src/nixl):
  test/nixl/nixl_test.cpp         core agent + descriptor functional tests
  test/unit/                      per-plugin unit tests
  benchmark/nixlbench/            nixl_worker.cpp, payload-sweep benchmarks
```

| ID | What to Verify | Pass Criteria | Platform |
|---|---|---|---|
| NF-1 | Agent init, UCX, XPU: `addBackend` selects `UCS_MEMORY_TYPE_ZE` | No plugin-load error; backend listed | Intel XPU |
| NF-2 | Agent init, libfabric, XPU: `getSupportedMems` returns `VRAM_SEG` (needs A2+A3) | `VRAM_SEG` present | Intel XPU |
| NF-3 | VRAM_SEG registration, UCX: `ucp_mem_query` returns ZE type not HOST | No VRAM-as-HOST error | Intel XPU |
| NF-4 | VRAM_SEG registration, libfabric: `fi_mr_reg` with `FI_HMEM_ZE` (needs A4) | MR key returned, `NIXL_OK` | Intel XPU |
| NF-5 | DRAM→VRAM transfer, UCX: `makeXfer+postXfer+waitXfer` CPU→XPU | Byte-correct; no SYSTEM-path fallback | Intel XPU |
| NF-6 | VRAM→VRAM transfer, libfabric: cross-node XPU→XPU with `FI_HMEM_ZE` | Byte-correct | 2× XPU nodes |
| NF-7 | POSIX plugin baseline: `FILE_SEG` register + read/write NVMe | Functional | Any |
| NF-8 | Intel IO Direct plugin: GPU-direct-to-storage, no CPU staging | `gds_enabled=true`; data correct | XPU + NVMe |
| NF-9 | OBJ/S3 plugin: `makeXfer` to S3-compatible endpoint (MinIO) | Object retrievable | Any |
| NF-10 | Multi-agent same node: KV-transfer + EP-dispatch agents coexist (needs E2) | No namespace collision | Intel XPU |
| NF-11 | Cross-vendor P/D: NVIDIA prefill → Intel XPU decode via `NixlConnector` | Output correct vs single-GPU reference | Mixed |
| NF-12 | ETCD namespace isolation: NIXL + Dynamo svc-disc on same ETCD (needs C3) | Zero key collisions | Any |
| NF-13 | `NixlEPAll2AllManager` baseline on NVIDIA GPU: MoE EP dispatch with `nixl_ep` backend | Dispatch completes; output matches `naive` backend reference | NVIDIA GPU (baseline before XPU) |

---

### 10.2  NIXL — Performance Benchmarks

```
Tool: ~/src/nixl/benchmark/nixlbench/  (nixl_worker.cpp, initiator/target roles)
```

| ID | Metric | Target | Notes |
|---|---|---|---|
| NP-1 | Unidirectional BW, UCX, XPU→XPU, 1 KB–1 GB | ≥ 80 % NIC line rate | ZE HMEM over RoCE/IB/EFA |
| NP-2 | Unidirectional BW, libfabric, XPU→XPU, 1 KB–1 GB | ≥ 80 % NIC line rate | After A2–A4 changes |
| NP-3 | ZE-BW vs CUDA-BW, same payload, same topology | > 0.95× ratio | Regression gate |
| NP-4 | Round-trip latency, UCX, 1 KB / 4 KB / 64 KB | < 2× CUDA baseline | Ping-pong |
| NP-5 | Multi-rail BW scaling: 1, 2, 4 rails | Linear ± 10 % | libfabric rail manager |
| NP-6 | Intel IO Direct vs POSIX sequential BW (component level) | ≥ 2× POSIX | GPU-direct NVMe; full detail in §10.6a IP-1/IP-2 |
| NP-7 | NIXL KV cache P/D end-to-end BW | ≥ 40 GB/s per node pair | nixlbench worker |
| NP-8 | Multi-agent BW overhead (2 agents vs 1) | < 5 % degradation | Needs E2 |

---

### 10.3  OneCCL — Component Functional Validation

```
Test infrastructure: ~/src/oneCCL/tests/functional/
  allreduce_test.cpp, allgather_test.cpp, alltoall_test.cpp,
  reduce_scatter_test.cpp, broadcast_test.cpp, transport.cpp
```

| ID | What to Verify | Pass Criteria |
|---|---|---|
| CF-1 | AllReduce TP, XPU: `ccl::allreduce` fp16/bf16, `world_size` = 2, 4, 8 | Numerically correct vs `torch.distributed` reference |
| CF-2 | AllGather PP activations: variable `recv_counts` on XPU | Correct concatenated output |
| CF-3 | ReduceScatter EP: sum reduction on XPU | Numerically correct |
| CF-4 | Send/Recv P2P, KVBM replicated (B5): `ccl::send+recv` for 4 MB KV block | Byte-identical at receiver | ⚠ `TransferMode::Sharded` is default — this tests the optional `::Replicated` mode only |
| CF-5 | Mixed-precision: bf16, fp16, int8 on XPU | No dtype artifacts |
| CF-6 | Process-group isolation: two `ccl::communicator` on same node | No cross-contamination |
| CF-7 | Fault tolerance: rank exits mid-collective | Graceful error, no hang |
| CF-8 | XPU TP + CPU DP simultaneously | Both collectives complete correctly |
| CF-9 | Broadcast on XPU: `ccl::broadcast` for scheduler metadata (4 KB) | Byte-correct at all ranks | Used by vLLM every step for `broadcast_tensor_dict` |
| CF-10 | PP P2P send/recv on XPU: `ccl::send` + `ccl::recv` for pipeline-stage activations | Tensors received in order; no hang at stage boundary | Validates OneCCL as backend for PP inter-stage transfer |

---

### 10.4  OneCCL — Performance Benchmarks

| ID | Metric | Target | Notes |
|---|---|---|---|
| CP-1 | AllReduce in-place BW at 1 MB / 16 MB / 256 MB, world_size = 8 | ≥ 80 % of NCCL, same topology | `oneccl_bindings_for_pytorch` |
| CP-2 | AllReduce latency at 1 KB | < 1.5× NCCL | Small-message regime |
| CP-3 | AllGather BW, world_size = 8 | ≥ 80 % of NCCL | |
| CP-4 | AllToAll BW, MoE workload sizes | ≥ 80 % of NCCL EP baseline | |
| CP-5 | vLLM TP time-per-token, TP = 4 / TP = 8, bf16 | ≤ 105 % of NCCL baseline | Via `XpuCommunicator` |
| CP-6 | KVBM replicated P2P, 4 MB KV block | < 500 µs per block | OneCCL P2P, item B5 |

---

### 10.5  Intel IO Direct — Component Functional Validation

> **Full stack:**
> ```
> NIXL cuda_gds plugin  →  dlopen("libcufile.so")
>                              ↓
>   ~/src/gds/gds-liburing-cufilewrapper/src/cufile_iouring.c   (cuFile API shim)
>     ├─ Level Zero: zeMemAllocDevice + ZE_EXTERNAL_MEMORY_TYPE_FLAG_DMA_BUF → dmabuf fd
>     └─ ~/src/gds/gds-liburing/src/register.c: io_uring_register_buffers_dmabuf()
>                              ↓
>   Linux kernel (github.com/tsg-/gds-linux, branch dmabuf-rw-xfs_tsg)
>     11-patch series — NOT yet upstream; requires custom kernel build
>                              ↓
>   NVMe  (GPU-direct, no CPU staging)
> ```
>
> **Kernel patch series (`dmabuf-rw-xfs_tsg` branch — 11 patches, bottom to top):**
> ```
> P01  block: introduce dma_token backed bio type
>        (Pavel Begunkov) Repurposes bi_io_vec space for pre-mapped DMA bios;
>        drivers that implement dma_map blk-mq op receive dma_token bios directly.
>
> P02  block: add infra to handle dmabuf tokens
>        (Pavel Begunkov) Block layer plumbing for dmabuf_token lifecycle.
>
> P03  nvme-pci: add support for dmabuf registration
>        (Pavel Begunkov) Implements ->dma_map in nvme-pci: nvme_dma_map (sg table +
>        DMA list), nvme_dma_token. Handles ->move_notify via fence + queue quiesce + remap.
>
> P04  nvme mapping  (Pavel Begunkov) — additional nvme mapping work
>
> P05  io_uring/rsrc: add imu flags
>        (Pavel Begunkov) Replaces is_kbuf with flags field in io_mapped_ubuf.
>
> P06  io_uring/rsrc: extended reg buffer registration
>        (Pavel Begunkov, suggested by Tushar Gohad) Adds struct io_uring_reg_buffer
>        as extended structure via io_uring_rsrc_update2 flag; adds target_fd + dmabuf_fd.
>
> P07  io_uring/rsrc: add dmabuf-backed buffer registration
>        (Pavel Begunkov, suggested by Tushar Gohad, Vishal Verma, David Wei)
>        Registers dmabuf-backed io_uring buffers; takes target_fd + dmabuf_fd;
>        retains target file for per-request access control.
>
> P08  io_uring/rsrc: implement dmabuf regbuf import
>        (Pavel Begunkov, suggested by David Wei, Vishal Verma, Tushar Gohad)
>        Opt-in import of dmabuf registered buffers per-request; validates that
>        the request's file matches the registered target file.
>
> P09  io_uring/rw: enable dma registered buffers
>        (Pavel Begunkov, suggested by Tushar Gohad) Wires dmabuf registered
>        buffers into the io_uring read-write submission path.
>
> P10  Implement dma_map for xfs
>        (Tushar Gohad) XFS filesystem dma_map file operation callback:
>        xfs_dma_token, xfs_verify_dma_extents(), xfs_invalidate_range(),
>        xfs_file_dma_map(). Page cache coherency + extent layout validation.
>
> P11  dmabuf, io_uring debug  (Tushar Gohad) — debug/instrumentation work
> ```
>
> **Status:** In-development patch series. P01–P09 authored by Pavel Begunkov (io_uring
> maintainer). P10 (XFS dma_map) and P11 by Tushar Gohad. Not yet submitted upstream.
> Tests require a custom kernel build from this branch.
>
> **Three components require independent validation before integration:**
> - **`gds-linux` kernel** — P01–P11 must be tested at block/nvme/io_uring/xfs layer
> - **`gds-liburing`** — dmabuf-aware liburing; test infra: `examples/dmabuf-read.c`
> - **`gds-liburing-cufilewrapper`** — cuFile shim; test infra: `tests/test_wrapper.c`,
>   `tests/test_batch_io.c`, `tests/test_multiple_buffers.c`, `tests/test_udmabuf_sealed.c`

#### 10.5a  `gds-linux` Kernel Patches — Functional Validation

| ID | Patch(es) | What to Verify | Pass Criteria |
|---|---|---|---|
| KP-1 | P01+P02 | `dma_token` bio: bio with `BIO_DMA_TOKEN` flag carries `dma_token` without `bio_vec`; block layer routes correctly | No OOPS; `dma_token` pointer valid in driver's `dma_map` callback |
| KP-2 | P03+P04 | NVMe `->dma_map`: `nvme_dma_map` created for Intel XPU dmabuf fd; `sg_table` + DMA list built; `nvme_dma_token` returned | DMA map succeeds; `dma_get_sgtable` populated |
| KP-3 | P03 | `->move_notify` path: fence signalled after in-flight NVMe I/Os; queue quiesced; mapping recreated | No data corruption; fence fires; queue resumes |
| KP-4 | P05+P06 | `io_uring_rsrc_update2` with `IORING_RSRC_F_EXTENDED_UPDATE` flag: `struct io_uring_reg_buffer` parsed; `target_fd` + `dmabuf_fd` extracted | Syscall returns 0; slot populated |
| KP-5 | P07 | dmabuf buffer registration: `IORING_REGISTER_BUFFERS2` with dmabuf fd + target_fd (NVMe block device) | Buffer slot registered; target file retained in `io_mapped_ubuf` |
| KP-6 | P08 | dmabuf regbuf import: request with mismatched file rejected; matching file accepted | Mismatch → `EACCES`; match → import succeeds |
| KP-7 | P09 | `io_uring` READ_FIXED / WRITE_FIXED with dmabuf registered buffer from NVMe | Data correct; no OOPS; `IORING_OP_READ_FIXED` completes |
| KP-8 | P10 | XFS `->dma_map`: `xfs_verify_dma_extents()` validates contiguous layout; `xfs_file_dma_map()` called by io_uring on XFS file | Extent validation passes for well-laid-out file; page cache invalidated |
| KP-9 | P10 | XFS page-cache coherency: `xfs_invalidate_range()` before DMA write; stale cache pages evicted | `fincore` shows zero cached pages after DMA write |
| KP-10 | P01–P11 | End-to-end kernel path: io_uring READ_FIXED from XFS file on NVMe into Level Zero dmabuf | Data correct in GPU VRAM; no CPU staging; `perf mem` shows no CPU reads |

#### 10.5b  `gds-liburing` — dmabuf Buffer Registration Layer

| ID | What to Verify | Pass Criteria |
|---|---|---|
| GL-1 | `io_uring_register_buffers_dmabuf()` succeeds with a valid dmabuf fd (needs KP-5) | `IORING_REGISTER_BUFFERS2` returns 0; buffer slot registered |
| GL-2 | `IORING_OP_READ_FIXED` NVMe→dmabuf buffer (`examples/dmabuf-read.c`) | Data byte-correct; no CPU-copy in `perf` |
| GL-3 | `IORING_OP_WRITE_FIXED` dmabuf buffer→NVMe | Data correct on disk; verified with `O_DIRECT` re-read |
| GL-4 | Buffer unregister + re-register cycle: no fd leak | `lsof` count stable; `IORING_UNREGISTER_BUFFERS` returns 0 |
| GL-5 | Kernel version gate: graceful failure without P06–P09 patches | Descriptive error; no kernel oops |

#### 10.5c  `gds-liburing-cufilewrapper` — cuFile API Shim

| ID | What to Verify | Pass Criteria |
|---|---|---|
| IF-1 | `cuFileDriverOpen()` with Level Zero context; no `libcuda.so` loaded | `CU_FILE_SUCCESS`; `ldd` shows no NVIDIA libs |
| IF-2 | ZE dmabuf export: `zeMemAllocDevice` with `ZE_EXTERNAL_MEMORY_TYPE_FLAG_DMA_BUF`; `cuFileBufRegister()` succeeds | `dmabuf_fd >= 0`; `registered_idx` valid |
| IF-3 | `cuFileWrite()` GPU VRAM→NVMe, no CPU staging (`test_wrapper.c`) | No CPU BW spike; `strace` shows `io_uring_enter` only |
| IF-4 | `cuFileRead()` NVMe→GPU VRAM, no CPU staging | Data byte-correct on device; no `pread` in strace |
| IF-5 | Batch I/O: `cuFileBatchIOSubmit()` with `nr=4,8,16` concurrent SQEs (`test_batch_io.c`) | All completions received; data correct |
| IF-6 | Multi-buffer: register N buffers, interleaved read/write (`test_multiple_buffers.c`) | No slot collision; all transfers correct |
| IF-7 | `udmabuf` sealed path (`test_udmabuf_sealed.c`): dmabuf via `/dev/udmabuf` + `memfd` | Byte-correct; tests pass |
| IF-8 | Graceful fallback: `cuFileDriverOpen()` without kernel dmabuf-rw patches | Returns `CU_FILE_PLATFORM_NOT_SUPPORTED`; no crash |

#### 10.5d  NIXL Integration — cuFile ZE Adapter via dlopen

| ID | What to Verify | Pass Criteria |
|---|---|---|
| IF-9 | NIXL `cuda_gds` plugin: `dlopen("libcufile.so")` resolves to `gds-liburing-cufilewrapper`; `gds_enabled=true` | Plugin listed in `NIXL_DEBUG`; no NVIDIA GDS plugin loaded |
| IF-10 | NIXL E2E: `makeXfer(FILE_SEG)` + `postXfer` + `waitXfer` via cuFile ZE adapter | Data byte-correct end-to-end; `NIXL_OK` |
| IF-11 | NIXL fallback: wrapper library absent → NIXL falls back to POSIX plugin | No crash; `gds_enabled=false` in NIXL log |
| IF-12 | `llmd-fs-backend`: KV offload + reload via NIXL + cuFile ZE adapter | Prefix-cache hit after evict+reload cycle |
| IF-13 | Concurrent: NIXL UCX P/D transfer + NIXL IO Direct KV offload simultaneously | No interference; both complete with `NIXL_OK` |

---

### 10.6  Intel IO Direct — Performance Benchmarks

> **Test tooling (`~/src/gds/gds-liburing-cufilewrapper/benchmarks/`):**
> - `bench_gpu_nvme_p2p_dmabuf` — sequential R/W BW, configurable size
> - `bench_gpu_nvme_cpu_bouncebuf_naive` — CPU bounce-buffer baseline (compat mode)
> - `bench_baseline_gpu_memorybw_noIO` — GPU VRAM BW ceiling (upper bound)
> - `bench_gpu_nvme_cufile_wrapperoverhead` — API overhead (register/deregister cycle)
> - `bench_gpu_nvme_diagnostic_writeread` — write vs read asymmetry
> - `bench_gpu_nvme_p2p_dmabuf_batchwr` — async batched writes
> - `scripts/run_all_benchmarks.sh` — sweep 2 MB → 1024 MB across all paths
>
> **Known benchmark gaps (from `benchmarks/COVERAGE.md`):**
> small-I/O latency/IOPS (< 2 MB), queue-depth sweep, true `cuFileBatchIO` nr-sweep,
> random I/O, multi-threaded concurrency — all added as IP-4 through IP-8 below.

#### 10.6a  `gds-liburing-cufilewrapper` Benchmarks

| ID | Metric | Target | Tool |
|---|---|---|---|
| IP-1 | Sequential write BW, XPU VRAM→NVMe, 2 MB–1 GB | ≥ 2× `pwrite()` + `cudaMemcpy` (CPU bounce) | `bench_gpu_nvme_p2p_dmabuf` vs `bench_gpu_nvme_cpu_bouncebuf_naive` |
| IP-2 | Sequential read BW, NVMe→XPU VRAM, 2 MB–1 GB | ≥ 2× `pread()` + `cudaMemcpy` baseline | Same |
| IP-3 | API overhead: `cuFileDriverOpen` + `cuFileBufRegister` + `cuFileHandleRegister` + deregister | < 500 µs total | `bench_gpu_nvme_cufile_wrapperoverhead` |
| IP-4 | Small-I/O latency: 4 K / 8 K / 16 K / 32 K / 128 K / 512 K single transfer | < 200 µs at 64 K | New test — gap from COVERAGE.md |
| IP-5 | Queue depth sweep: io_uring depth 1, 4, 16, 32, 128 at 4 MB transfer size | BW peaks at depth ≤ 32 for NVMe | New test — gap from COVERAGE.md |
| IP-6 | `cuFileBatchIO` nr sweep: `nr=1,2,4,8,16,32` through `cuFileBatchIOSubmit` | Linear BW scaling to saturation | New test — gap from COVERAGE.md |
| IP-7 | Random I/O: 4 K random reads, 10 K ops | ≥ 150 K IOPS | New test — gap from COVERAGE.md |
| IP-8 | Multi-threaded: N threads × per-thread io_uring, shared NVMe | BW ≥ 0.85× N × single-thread BW | New test — gap from COVERAGE.md |

#### 10.6b  NIXL E2E Storage Performance

| ID | Metric | Target | Notes |
|---|---|---|---|
| IP-9 | NIXL KV offload sustained: evict + reload throughput | > 20 GB/s | `llmd-fs-backend` + cuFile ZE adapter |
| IP-10 | NIXL IO Direct vs POSIX: same KV block, compare BW | ≥ 2× POSIX | Controlled comparison |

---

---

## 10.10  Intel IO Direct — Upstreaming, Open Source, and Delivery

> This section covers the ecosystem work required to productize and deliver the Intel IO Direct
> stack beyond the development prototype. These are parallel tracks to the functional and
> performance validation work in Sections 10.5 and 10.6.

#### Kernel Upstreaming (gds-linux patches)

| ID | Work Item | Owner Hint | Notes |
|---|---|---|---|
| KU-1 | **NVMe DMA-BUF kernel upstreaming** — submit P01–P04 (`block/` dma_token + `nvme-pci` dma_map) to Linux block + NVMe maintainers (Keith Busch, Jens Axboe) | Kernel team | P03 `->move_notify` fence/quiesce logic will need careful review; NVMe queue quiesce path is latency-sensitive |
| KU-2 | **io_uring patches upstreaming** — submit P05–P09 to Pavel Begunkov / io_uring tree | Begunkov (external) + Intel kernel team | P06–P08 already Suggested-by Tushar Gohad; coordinate with Begunkov for submission timeline |
| KU-3 | **XFS dma_map upstreaming** — submit P10 to XFS maintainers (Darrick Wong, Chandan Babu) | Intel kernel team (Tushar Gohad) | Requires `xfs_verify_dma_extents()` review; interaction with XFS direct I/O and reflink |
| KU-4 | **DKMS fallback route** — package P01–P10 as an out-of-tree DKMS module for distributions that cannot ship a full custom kernel; target kernel LTS baseline (6.12 LTS) | Intel kernel team | Needed if upstream acceptance is delayed > 1 release cycle; DKMS allows deployment on standard distro kernels without recompile |

#### liburing Upstreaming (gds-liburing patches)

| ID | Work Item | Owner Hint | Notes |
|---|---|---|---|
| LU-1 | **`io_uring_register_buffers_dmabuf()` API upstreaming** — submit `gds-liburing/src/register.c` addition to liburing upstream (Axboe/Begunkov) | Intel team + Begunkov | API depends on kernel P06–P07 being upstream first; coordinate submission after KU-2 |
| LU-2 | **liburing packaging** — once upstream, ensure `gds-liburing` changes land in `liburing` distribution packages (RPM/DEB) so cuFile wrapper can link against distro liburing | Intel + distro partners | Required before distro integration (DI-1, DI-2) |

#### Open Source, Legal, and Naming

| ID | Work Item | Owner Hint | Notes |
|---|---|---|---|
| OS-1 | **Open Source PDT approval** — initiate Intel Open Source Program Office (OSPO) Product Delivery Team review for `gds-liburing-cufilewrapper` project | OSPO + engineering lead | Required before any public repo, binary, or package release; includes export control review |
| OS-2 | **Project naming** — confirm "Intel IO Direct" as the product/brand name; resolve `libcufile.so` compatibility shim naming (binary delivers as `libcufile.so.1` — verify no trademark conflict with NVIDIA cuFile) | Legal + OSPO | NVIDIA cuFile is a proprietary trademark; the shim replaces the binary interface but the name may require clearance or aliasing |
| OS-3 | **License selection** — choose OSS license for `gds-liburing-cufilewrapper` (MIT per current `LICENSE` file); confirm compatible with NIXL (Apache 2.0), liburing (MIT/LGPL2), Level Zero SDK (MIT) | Legal + OSPO | MIT chosen in current repo; verify LGPL2 liburing dependency does not impose copyleft on the shim |
| OS-4 | **Delivery method** — decide: (a) standalone `libcufile.so` package, (b) bundled in NIXL wheel/package, (c) distro native package. Each has different OSPO and security review requirements | Product management + OSPO | Option (b) simplest for AI stack consumers; option (c) best for broad adoption; may pursue both in parallel |

#### Distro Integration

| ID | Work Item | Owner Hint | Notes |
|---|---|---|---|
| DI-1 | **Red Hat / RHEL integration** — engage Red Hat kernel team to include P01–P10 kernel patches in RHEL kernel or as RHEL module; package `libcufile` (Intel IO Direct) and updated `liburing` as RPMs | Intel + Red Hat | RHEL is primary enterprise Linux target for data center AI workloads; DKMS route (KU-4) as fallback |
| DI-2 | **Canonical / Ubuntu integration** — engage Canonical kernel team for Ubuntu 26.04 LTS inclusion of dmabuf-rw patches; package `libcufile-intel` and `liburing` (updated) as DEBs in Ubuntu universe | Intel + Canonical | Ubuntu is primary cloud/developer target; align with Ubuntu kernel freeze schedule |
| DI-3 | **Package naming in distros** — coordinate `libcufile` package name with NVIDIA (who ships `libcufile` for GDS); may need `libcufile-intel` or `intel-io-direct` package name in distro repos | Intel + distro partners | Naming collision risk in RPM/DEB repos if both `libcufile` (NVIDIA) and `libcufile` (Intel IO Direct) are installable |
| DI-4 | **DKMS fallback package** — if distro kernel inclusion is delayed, publish DKMS package (`intel-io-direct-dkms`) for RHEL 9.x / Ubuntu 22.04/24.04 that back-ports P01–P10 against LTS kernels | Intel kernel team | Provides day-1 deployment path on existing enterprise installs; requires per-kernel-version CI |

---

### 10.7  UCX (openucx Layer) + libfabric — Component Functional Validation

> **openucx test infrastructure (`~/src/ucx`):**
> ```
> test/gtest/                   C++ unit tests for UCT / UCP layers
> src/tools/perf/ucx_perftest   RDMA BW + latency benchmark (--memory-type ze)
> src/tools/perf/ze/            ZE-specific perf alloc helpers (ze_alloc.c)
> ucx_info -d <transport>       Transport capability query
> ```

#### 10.7a  openucx — ZE Transport Functional Validation

| ID | What to Verify | Pass Criteria |
|---|---|---|
| UF-1a | ZE UCT transport load: `ucx_info -d ze_copy` lists ZE copy interface | `ze_copy` in transport list; no CUDA dep |
| UF-1b | UCM ZE memory hook: `ucm/ze/zemem.c` — `zeMemAllocDevice` intercepted; ZE type detected | `UCS_MEMORY_TYPE_ZE` returned by `ucs_mem_type_detect` |
| UF-1c | IB HMEM ZE wiring: `ucp_ep_create` with ZE VRAM buffer, IB transport selected | `UCS_MEMORY_TYPE_ZE` in `ucp_mem_query`; no CPU-bounce |
| UF-1d | `ucx_perftest --memory-type ze` PUT: endpoint create + RDMA PUT with ZE memory | Transfer completes; `UCS_OK` |
| UF-1e | ZE IPC: `ze_copy` transport cross-process ZE VRAM sharing via IPC handle | Remote access succeeds; memory type preserved |

#### 10.7b  openucx — ZE Transport Performance

| ID | Metric | Target | Tool |
|---|---|---|---|
| UP-1 | PUT BW, ZE memory, 1 KB–1 GB | ≥ 80 % NIC line rate | `ucx_perftest -t put_bw --memory-type ze` |
| UP-2 | PUT latency at 1 KB, ZE memory | < 2× CUDA baseline, same topology | `ucx_perftest -t put_lat --memory-type ze` |
| UP-3 | ZE intra-node copy BW (PCIe, ze_copy transport) | ≥ 90 % of `zeCommandListAppendMemoryCopy` baseline | `src/tools/perf/ze/` |
| UP-4 | ZE vs CUDA RDMA BW ratio, same IB HCA | > 0.95× | Regression gate |

#### 10.7c  libfabric — ZE Provider Functional Validation

| ID | What to Verify | Pass Criteria |
|---|---|---|
| UF-3 | `fi_getinfo` with `FI_HMEM_ZE` hint returns provider | Provider with `FI_HMEM_ZE` capability present |
| UF-4 | libfabric RDMA Write ZE: `fi_write` XPU→remote XPU | Byte-correct; no host-memory bounce |
| UF-5 | Rail selection: `nixlLibfabricRailManager` maps Intel XPU to NUMA-closest NIC | Correct affinity visible in `NIXL_DEBUG` log |
| UF-6 | Multi-rail striping: transfer > `striping_threshold` uses N rails | BW scales with rail count |

---

### 10.8  End-to-End Stack Validation

| ID | Layers Exercised | Metrics |
|---|---|---|
| EE-1 | Dynamo P/D, Intel XPU, full L0–L7 | TTFT, tokens/sec, KV cache hit rate |
| EE-2 | llm-d P/D, Intel XPU, L0–L6 | Same as EE-1 via llm-d EPP routing |
| EE-3 | Mixed: NVIDIA prefill → Intel XPU decode, L4+L5+L6 | KV transfer BW; output correctness vs single-platform |
| EE-4 | MoE EP dispatch XPU via `NixlEPAll2AllManager`, L4+L6 (needs D1+D2) | Time-per-token vs OneCCL AgRs fallback |
| EE-5 | KV offload + reload, L5+L6+Intel IO Direct | Cache hit rate after evict; reload latency |
| EE-6 | TP=8 AllReduce XPU, L4+L6b, OneCCL | Iteration time vs NCCL/NVIDIA baseline |
| EE-7 | Concurrent P/D + TP, all layers | No deadlock; throughput ≥ 90 % of serial sum |

---

### 10.9  Validation Sequencing and Gate Dependencies

```
PHASE 1 — Component validation (all parallel, no stack dependency)
  openucx ZE transport         UF-1a to UF-1e, UP-1 to UP-4
                                ← openucx upstream changes (A1)
  libfabric ZE tests           UF-3 to UF-6
                                ← BLOCKED by A2 + A3 + A4
  OneCCL collectives           CF-1 to CF-8
  gds-linux kernel patches     KP-1 to KP-10
                                ← P01–P09 by Pavel Begunkov (io_uring maintainer)
                                ← P10 XFS dma_map + P11 debug by Tushar Gohad
                                ← custom kernel build required; NOT yet upstream
  gds-liburing dmabuf layer    GL-1 to GL-5   ← BLOCKED by KP-1 to KP-9 (kernel P05–P09)
  gds-liburing-cufilewrapper   IF-1 to IF-8, IP-1 to IP-8 (standalone, no NIXL yet)
                                ← BLOCKED by GL-1 to GL-5 green
  NixlEPAll2AllManager baseline NF-13 (NVIDIA GPU only — no XPU dep)

PHASE 2 — NIXL agent integration (after Phase 1 green)
  NIXL + UCX on XPU            NF-1, NF-3, NF-5   ← needs UF-1a–1e green
  NIXL + libfabric on XPU      NF-2, NF-4, NF-6
  NIXL + cuFile ZE adapter     IF-9, IF-10, IF-11  ← NIXL dlopen + E2E storage
  Multi-agent scoping          NF-10               ← needs E2
  ETCD namespace isolation     NF-12               ← needs C3

PHASE 3 — KVBM integration (after Phase 2 green)
  KVBM on Intel XPU                                ← BLOCKED by B1–B6; B6 is day-0 gate
  KVBM + NIXL P/D transfer     NF-11
  KVBM + IO Direct KV offload  IF-12, IF-13, IP-9, IP-10
  KVBM replicated + OneCCL     CF-4                ← needs B5

PHASE 4 — vLLM XPU integration (after Phases 1–3 green)
  XpuCommunicator + OneCCL     CP-5, CF-9, CF-10
  NixlEPAll2AllManager XPU                         ← needs D1 + D2; gated on NF-13 baseline

PHASE 5 — End-to-end + Performance (after Phase 4 green)
  EE-1 through EE-7
  Performance sweeps NP-*, CP-*, UP-* in parallel with EE tests

CRITICAL GATE ITEMS
  KP-1 to KP-9 (gds-linux kernel P01–P09)          → gates entire Intel IO Direct stack
  KP-8–KP-9 (P10 XFS dma_map)                  → gates XFS-backed IO Direct path
  A1   (openucx ZE transport / IB HMEM ZE)      → gates all UCX ZE tests (Phase 1)
  A3   (FI_HMEM_ZE runtime branch in NIXL)      → gates libfabric ZE test suite
  GL-1 (gds-liburing dmabuf registration)       → gates all cuFile wrapper tests
  B6   (CUDA device-guard removal in KVBM)      → gates all KVBM Phase 3 work
  D1   (XpuCommunicator EP wiring in vLLM)      → gates EE-4 MoE EP end-to-end test
```
