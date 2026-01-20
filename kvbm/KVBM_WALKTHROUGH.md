# KVBM (KV Block Manager) Architecture Walkthrough

**Context:** This document explains the KVBM architecture and code structure in response to GitHub issue #3577 "How to Understand KVBM?"

---

## System Overview

KVBM is Dynamo's **distributed KV-cache block management system**. It decouples memory/storage management from inference runtimes (vLLM, TensorRT-LLM, SGLang), enabling **GPU ↔ CPU ↔ Disk ↔ Remote** tiering with asynchronous block offload/onboard.

### Key Design Principles
- **Separation of Concerns:** Memory management is independent of inference logic.
- **Distributed Architecture:** Multi-worker setup with a Leader coordinating block allocation/transfer.
- **Async Offload:** Blocks move between tiers (GPU→CPU→Disk) without re-computation.
- **Runtime-Agnostic:** Works with multiple inference engines via lightweight connectors.

---

## Block Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                    INFERENCE RUNTIME LAYER                      │
│  (vLLM / TensorRT-LLM / SGLang via KV Connector)               │
│  - Calls KVBM for KV-cache allocation/lookup                   │
└────────────────┬────────────────────────────────────────────────┘
                 │
                 │ PyO3 Bindings
                 ▼
┌─────────────────────────────────────────────────────────────────┐
│                   KVBM RUST BINDING LAYER                       │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │ lib.rs (PyModule Init)                                   │  │
│  │ - Tokio runtime initialization                           │  │
│  │ - Cancellation token management                          │  │
│  │ - OTEL logging setup                                     │  │
│  │ - Module registration (BlockManager, Workers, Leader)    │  │
│  └──────────────────────────────────────────────────────────┘  │
│                                                                  │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │ block_manager.rs (Core Logic)                            │  │
│  │                                                          │  │
│  │  BlockManager {                                         │  │
│  │    ├── inner: KvBlockManager                           │  │
│  │    │   └── Manages device/host/disk layouts           │  │
│  │    ├── _drt: DistributedRuntime                        │  │
│  │    │   └── Shared tokio runtime & cancellation        │  │
│  │    └── _controller: BlockManagerClient                │  │
│  │        └── Lifecycle & component integration           │  │
│  │  }                                                      │  │
│  │                                                          │  │
│  │  BlockManagerBuilder {                                 │  │
│  │    ├── worker_id, page_size                            │  │
│  │    ├── leader: KvbmLeader                              │  │
│  │    ├── kvbm_metrics (optional)                         │  │
│  │    └── consolidator_config (optional)                  │  │
│  │  }                                                      │  │
│  └──────────────────────────────────────────────────────────┘  │
│                                                                  │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │ Submodules                                               │  │
│  │ - distributed/ : KvbmLeader, KvbmWorker                 │  │
│  │ - controller/  : BlockManagerClient, BlockPoolStatus    │  │
│  │ - cache_stats/ : Metrics & statistics                   │  │
│  └──────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
                 │
                 │ Async Calls (tokio)
                 ▼
┌─────────────────────────────────────────────────────────────────┐
│               DYNAMO-LLM BLOCK MANAGER (Rust)                   │
│  KvBlockManager<Logical<DistributedLeaderWorkerResources>>     │
│                                                                  │
│  ┌────────────────────────────────────────────────────────┐   │
│  │ Layout System (Device/Host/Disk)                       │   │
│  │ - DeviceLayout (GPU memory)                            │   │
│  │ - HostLayout (CPU pinned memory + offload filter)      │   │
│  │ - DiskLayout (SSD cache with frequency filtering)      │   │
│  └────────────────────────────────────────────────────────┘   │
│                                                                  │
│  ┌────────────────────────────────────────────────────────┐   │
│  │ Logical Resource Layer                                 │   │
│  │ - DistributedLeaderWorkerResources                     │   │
│  │   └── Translates global block IDs to physical addrs   │   │
│  │       Format: {worker_id}.{block_set_id}.{block_id}   │   │
│  └────────────────────────────────────────────────────────┘   │
│                                                                  │
│  ┌────────────────────────────────────────────────────────┐   │
│  │ Offload Components                                     │   │
│  │ - FrequencyFilter (CPU→Disk offload selection)         │   │
│  │   └── Toggled via DYN_KVBM_DISABLE_DISK_OFFLOAD_FILTER│   │
│  │ - KvEventConsolidator (metrics aggregation)           │   │
│  └────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
                 │
                 │ NIXL Transfer API + Direct I/O
                 ▼
┌─────────────────────────────────────────────────────────────────┐
│                  PHYSICAL MEMORY TIERS                          │
│                                                                  │
│  GPU Memory                                                     │
│  └─ Fastest, limited (~40-80GB per GPU)                        │
│                                                                  │
│  CPU Pinned Memory (Host RAM)                                  │
│  └─ DYN_KVBM_CPU_CACHE_GB (configured, ~100GB typical)        │
│                                                                  │
│  Disk / SSD Cache                                              │
│  └─ DYN_KVBM_DISK_CACHE_GB (optional)                          │
│  └─ Path: DYN_KVBM_DISK_CACHE_DIR (default: /tmp/)            │
│  └─ Uses O_DIRECT for GDS alignment (if enabled)              │
│                                                                  │
│  Remote (Network Storage / Cloud)                              │
│  └─ Via distributed worker connections                        │
└─────────────────────────────────────────────────────────────────┘
```

---

## Component Details

### 1. **Python→Rust Bridge (PyO3)**

**File:** `src/lib.rs`

**Purpose:** Exposes Rust KVBM as a Python module named `_core`.

**Key Functions:**
- `init_pyo3_tokio_rt()` → Initializes a multi-threaded Tokio runtime once per process.
  - Worker threads: `RuntimeConfig.num_worker_threads` (defaults to CPU count).
  - Max blocking threads: `RuntimeConfig.max_blocking_threads`.
  - All async features enabled.

- `get_current_tokio_handle()` → Returns the shared Tokio runtime handle for async work.

- `get_current_cancel_token()` → Returns a shared `CancellationToken` to gracefully shut down async tasks.

- `to_pyerr<E>()` → Converts Rust errors to Python exceptions.

- `extract_distributed_runtime_from_obj()` → Extracts a weak reference to `DistributedRuntime` from a Python capsule, allowing Python to hold a handle to the shared runtime.

**Environment Variable:**
- `OTEL_EXPORT_ENABLED` → If `"true"`, initialize OpenTelemetry logging with OTLP batch exporter.

---

### 2. **Block Manager Binding (block_manager.rs)**

**File:** `src/block_manager.rs`

**Purpose:** Main interface between Python and the Dynamo KV block manager.

#### **BlockManager Class**

```rust
#[pyclass]
pub struct BlockManager {
    inner: VllmBlockManager,          // The actual block manager
    _drt: Option<Arc<DistributedRuntime>>,  // Shared runtime
    _controller: Option<VllmController>,    // Lifecycle controller
}
```

**Constructor Signature:**
```python
BlockManager(
    worker_id: int,
    leader: Optional[KvbmLeader] = None,
    page_size: int = 32,
    num_device_blocks: Optional[int] = None,
    disable_device_pool: bool = False
)
```

**Workflow:**

1. **Configuration Building:**
   - Create `KvBlockManagerConfig` with runtime settings.
   - Add model config: `num_layers=1`, `outer_dim=1`, `page_size`, `inner_dim=1`.
   - These values are placeholders; actual dimensions are managed by the inference runtime.

2. **Leader Acquisition:**
   - If `leader` is provided:
     - Extract the inner leader and `DistributedRuntime` from the capsule.
     - Query leader for block counts: `num_device_blocks()`, `num_host_blocks()`, `num_disk_blocks()`.
     - Configure layouts accordingly.
   - If no leader:
     - Currently panics (leadership is required for full functionality).

3. **Layout Configuration:**
   - **Device Layout:** Maps to GPU memory if `!disable_device_pool`.
   - **Host Layout:** Maps to CPU pinned memory with optional disk offload filter.
     - Filter controlled by `DYN_KVBM_DISABLE_DISK_OFFLOAD_FILTER` env var.
     - When enabled, `FrequencyFilter` selects frequently-accessed blocks for retention on CPU.
   - **Disk Layout:** Maps to SSD cache if `num_disk_blocks() > 0`.

4. **Async Initialization:**
   - Tokio runtime calls `KvBlockManager::new()` with layouts and `DistributedLeaderWorkerResources`.
   - Returns the fully initialized `BlockManager` Python object.

**Methods:**
- `block_size() -> usize` → Returns the size (in bytes) of a single block.
- `init_controller(component: Component) -> PyResult<()>` → Attaches a component (namespace, name, runtime connection) for lifecycle tracking.

---

#### **BlockManagerBuilder Class**

A fluent API for Rust code to build `BlockManager` asynchronously.

```rust
pub struct BlockManagerBuilder {
    worker_id: u64,
    leader: Option<KvbmLeader>,
    page_size: usize,
    disable_device_pool: bool,
    kvbm_metrics: Option<KvbmMetrics>,
    consolidator_config: Option<(String, Option<String>, EventSource)>,
}
```

**Methods:**
- `new()` → Create a builder with defaults.
- `.worker_id(id)` → Set worker ID.
- `.leader(leader)` → Set the leader.
- `.page_size(ps)` → Set page size.
- `.disable_device_pool(yes)` → Disable GPU pool.
- `.kvbm_metrics(metrics)` → Attach metrics.
- `.consolidator_config(engine_ep, output_ep, source)` → Configure event consolidation.
- `async fn build()` → Async build that returns a populated `BlockManager`.

---

#### **FrequencyFilter**

**Purpose:** Dynamically selects which blocks to offload from CPU to disk.

**Configuration:**
```rust
FrequencyFilter::new(
    access_count_threshold: 2,              // Min accesses before retaining
    eviction_window: Duration::from_secs(600),  // 10-minute window
    max_blocks: 1_000_000,                  // Upper limit on tracked blocks
    cancel_token: CancellationToken,        // For graceful shutdown
    runtime: tokio::runtime::Handle,        // For spawning background work
)?
```

**Behavior:**
- Tracks block access frequency over a 10-minute sliding window.
- Blocks accessed < 2 times are eligible for disk offload.
- Prevents excessive SSD write amplification (important for enterprise drives like KIOXIA CM7).
- **Disabled by:** `DYN_KVBM_DISABLE_DISK_OFFLOAD_FILTER=true` (for testing or if SSD write limits are not a concern).

---

### 3. **Distributed Components (distributed/)**

**File:** `src/block_manager/distributed.rs` (implied, not shown in detail here)

**Key Classes:**
- **`KvbmLeader`** → Holds leader-side resources (block counts, allocator).
  - Method: `dissolve()` → Returns `(InnerLeader, Arc<DistributedRuntime>)`.
  - Queries: `num_device_blocks()`, `num_host_blocks()`, `num_disk_blocks()`.

- **`KvbmWorker`** → Represents a worker process in the distributed system.
  - Communicates with the leader to request/release blocks.
  - Handles remote block access (worker_id → physical address mapping).

---

### 4. **Controller Components (controller/)**

**File:** `src/block_manager/controller.rs` (implied)

**Key Classes:**
- **`BlockManagerClient`** → Interfaces with the block manager controller.
  - Methods for triggering resets, querying pool status, managing ownership transfers.

- **`BlockPoolStatus`** → Returns current utilization metrics.
  - Free blocks, used blocks, allocated blocks per tier.

- **`ResetBlocksResponse`** → Result of a block reset operation.

---

### 5. **Metrics & Statistics (cache_stats/)**

**File:** `src/block_manager/cache_stats.rs` (implied)

**Purpose:** Collects KV-cache performance metrics:
- Hit/miss rates per tier.
- Offload/onboard counts.
- Latency distributions.

**Configuration:**
- Enabled via `DYN_KVBM_METRICS=true`.
- Metrics endpoint port: `DYN_KVBM_METRICS_PORT` (default: 6880).
- Provides Prometheus-compatible `/metrics` endpoint.

---

## Data Flow Example: Block Access

### Scenario: Inference Engine Needs a KV Block

```
1. Python (vLLM):
   "I need block 42 from layout B"
   └─ Call KVBM's get_block() or similar
   
2. Rust BlockManager (Logical Layer):
   Lookup block 42 in DistributedLeaderWorkerResources
   └─ Resolve global ID {worker_id}.{block_set_id}.{block_id} → physical address
   
3. Check Block Location:
   a) Is it in GPU memory? → Return pointer, access immediately.
   b) Is it in CPU pinned memory? → Return pointer, access immediately.
   c) Is it on disk? → Trigger async onboard via NIXL/GDS.
   d) Is it on remote worker? → Issue network transfer via NIXL.
   
4. Async Transfer (if needed):
   FrequencyFilter checks: "Should this block stay on CPU after offload?"
   └─ If frequently accessed → Keep on CPU.
   └─ If rarely accessed → Eligible for disk offload.
   
   NIXL (Network I/O eXtensible Library):
   └─ P2P DMA transfer (GPU↔GPU or GPU↔CPU↔Disk).
   └─ Leverages GDS (GPU DirectStorage) for disk I/O acceleration.
   
5. Python (vLLM) Resumes:
   Block is now in requested location → Inference continues.
```

---

## Environment Configuration

| Variable | Default | Purpose |
|----------|---------|---------|
| `DYN_KVBM_CPU_CACHE_GB` | **required** | CPU pinned memory size |
| `DYN_KVBM_DISK_CACHE_GB` | optional | SSD cache size |
| `DYN_KVBM_DISK_CACHE_DIR` | `/tmp/` | SSD cache directory |
| `DYN_KVBM_DISK_ZEROFILL_FALLBACK` | `false` | Fallback for filesystems without `fallocate()` |
| `DYN_KVBM_DISK_DISABLE_O_DIRECT` | `false` | Disable O_DIRECT (for compatibility) |
| `DYN_KVBM_LEADER_WORKER_INIT_TIMEOUT_SECS` | 120 | Timeout for leader↔worker sync |
| `DYN_KVBM_METRICS` | `false` | Enable metrics endpoint |
| `DYN_KVBM_METRICS_PORT` | 6880 | Prometheus metrics port |
| `DYN_KVBM_DISABLE_DISK_OFFLOAD_FILTER` | `false` | Disable frequency-based offload selection |
| `OTEL_EXPORT_ENABLED` | `false` | Enable OpenTelemetry logging |

---

## Addressing GitHub Issue #3577

### Question: "What are the different memory tiers?"

**Answer:**
1. **GPU Memory** (Device): Fastest tier, limited to ~40–80 GB per GPU.
2. **CPU Pinned Memory** (Host): Medium tier, configured via `DYN_KVBM_CPU_CACHE_GB`.
3. **Disk / SSD** (Storage): Slowest tier, optional, configured via `DYN_KVBM_DISK_CACHE_GB`.
4. **Remote / Network** (Distributed): Blocks on other workers accessible via distributed leader.

### Question: "Are data, metadata, and extra metadata transmitted via different channels?"

**Answer:**
KVBM uses a **unified NIXL transport layer** that abstracts the underlying I/O mechanism:
- **Single-Server Transfers:** NVIDIA peer-to-peer DMA (GPU↔CPU, CPU↔SSD via GDS).
- **Cross-Server Transfers:** RDMA or TCP over Ethernet (managed by NIXL).
- **Metadata:** Exchanged once at startup (`SerializedLayout`), then block lists sent via the same channel.

### Question: "What is the global block identification scheme?"

**Answer:**
KVBM uses a **3-tuple global block ID:**
```
{worker_id}.{block_set_id}.{block_id}

worker_id      : Which worker owns this block (0, 1, 2, ... N).
block_set_id   : Which layout this block belongs to (e.g., device vs. host).
block_id       : Index within the layout.
```

Metadata (`SerializedLayout`) maps block_set_id to physical buffer addresses and NIXL handles.

---

## Initialization Sequence

```
1. Python Application Starts
   └─ import dynamo._core (triggers __init__ in lib.rs)
   
2. Tokio Runtime Initialized
   └─ init_pyo3_tokio_rt() called once (OnceLock guards it).
   └─ Multi-threaded runtime with cancellation token.
   
3. OTEL Logging Initialized (if enabled)
   └─ Spawns OTLP batch exporter background thread.
   
4. Python Creates BlockManager
   └─ Python: bm = BlockManager(worker_id=0, leader=kvbm_leader, ...)
   └─ Rust: BlockManager::new() called
   
5. Layouts Configured
   └─ Device, Host, Disk layouts created based on leader's counts.
   └─ FrequencyFilter attached to HostLayout if disk_blocks > 0.
   
6. KvBlockManager Initialized
   └─ Async call to KvBlockManager::new() via Tokio runtime.
   └─ DistributedLeaderWorkerResources created.
   
7. Python Receives Initialized BlockManager
   └─ Ready for inference engine to call get_block(), allocate(), etc.
```

---

## Summary

**KVBM is a smart, distributed memory manager that:**
- Decouples inference logic from memory management.
- Supports multi-tier caching (GPU → CPU → Disk → Remote).
- Asynchronously moves blocks between tiers to optimize throughput.
- Protects SSD longevity via frequency-based offload selection.
- Provides a clean Python API backed by fast Rust/Tokio internals.

**Key Takeaway:** The Rust block_manager.rs acts as a **configuration and lifecycle manager**, while the Dynamo-LLM backend handles the actual block transfers. The leader/worker distributed architecture allows horizontal scaling across multiple machines.
