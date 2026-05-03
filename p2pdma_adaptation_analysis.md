# Adapting gds-liburing-cufilewrapper to use P2PDMA (hipFile-style)

Analysis of what it would take to adapt the Intel Arc GPU libcufile wrapper to use kernel p2pdma similar to AMD's hipFile approach.

**Date**: 2026-05-03

---

## Flow Diagrams

### 1. AMD hipFile Approach (Current Production)

```
┌─────────────────────────────────────────────────────────────────┐
│ Application Layer                                               │
│ ┌─────────────────────────────────────────────────────────────┐ │
│ │ hipFileRead(fh, gpu_ptr, size, file_offset, buffer_offset)  │ │
│ └──────────────────────────┬──────────────────────────────────┘ │
└────────────────────────────┼────────────────────────────────────┘
                             │
┌────────────────────────────▼────────────────────────────────────┐
│ hipFile Library (Userspace)                                     │
│ ┌──────────────────────────────────────────────────────────────┐│
│ │ Backend Scoring System                                       ││
│ │  • Fastpath: Check alignment, GPU memory type, fs support    ││
│ │  • Fallback: Always available (bounce buffer)                ││
│ └───┬──────────────────────────────────────────────────┬───────┘│
│     │ Score=100 (P2P eligible)      Score=0 (fallback) │        │
│ ┌───▼──────────────────────────┐  ┌──────────────────▼────────┐ │
│ │ Fastpath Backend             │  │ Fallback Backend          │ │
│ │  1. Validate params          │  │  1. Allocate bounce buf   │ │
│ │  2. Create hipAmdFileHandle  │  │  2. hipMemcpy GPU↔Host    │ │
│ │  3. Call hipAmdFileRead()    │  │  3. pread/pwrite file     │ │
│ └───┬──────────────────────────┘  └───────────────────────────┘ │
└─────┼───────────────────────────────────────────────────────────┘
      │
┌─────▼───────────────────────────────────────────────────────────┐
│ HIP Runtime (ROCm)                                              │
│ ┌──────────────────────────────────────────────────────────────┐│
│ │ hipAmdFileRead/Write (NOT public API)                        ││
│ │  • Obtained via hipGetProcAddress()                          ││
│ │  • Dynamic function lookup at runtime                        ││
│ └───┬──────────────────────────────────────────────────────────┘│
└─────┼───────────────────────────────────────────────────────────┘
      │
┌─────▼───────────────────────────────────────────────────────────┐
│ ROCR Runtime → HSA → libhsakmt (Thunk)                          │
└─────┬───────────────────────────────────────────────────────────┘
      │
┌─────▼───────────────────────────────────────────────────────────┐
│ Linux Kernel - KFD Driver (amdgpu/amdkfd)                       │
│ ┌──────────────────────────────────────────────────────────────┐│
│ │ kfd_ioctl_ais() - "AMD Infinity Storage" IOCTL               ││
│ │  ↓                                                           ││
│ │ kfd_ais_rw_file()                                            ││
│ │  1. Validate pcie_p2pdma_distance >= 0                       ││
│ │  2. Use kernel P2PDMA subsystem (CONFIG_PCI_P2PDMA)          ││
│ │  3. Setup DMA mapping: GPU BAR ↔ NVMe                        ││
│ │  4. Execute P2P DMA transfer                                 ││
│ └──────────────────────────────────────────────────────────────┘│
└─────┬───────────────────────────────────────────────────────────┘
      │
┌─────▼───────────────────────────────────────────────────────────┐
│ Hardware: AMD GPU ←──PCIe P2P──→ NVMe SSD                       │
│  Direct BAR-to-BAR transfer, no CPU/system memory involved      │
└─────────────────────────────────────────────────────────────────┘

Key Characteristics:
  ✓ Single IOCTL per operation (kfd_ioctl_ais)
  ✓ Kernel manages P2PDMA mapping per-operation
  ✓ Transparent fallback to bounce buffer
  ✓ No io_uring, no dmabuf export
  ✓ GPU driver owns entire I/O path
```

---

### 2. Current libze/io_uring/dmabuf Approach (Intel Arc Implementation)

```
┌─────────────────────────────────────────────────────────────────┐
│ Application Layer                                               │
│ ┌─────────────────────────────────────────────────────────────┐ │
│ │ cuFileRead(fh, gpu_ptr, size, file_offset, buffer_offset)  │ │
│ └──────────────────────────┬──────────────────────────────────┘ │
└────────────────────────────┼────────────────────────────────────┘
                             │
┌────────────────────────────▼────────────────────────────────────┐
│ libcufile.so (Wrapper - Userspace)                              │
│ ┌──────────────────────────────────────────────────────────────┐│
│ │ issue_io_op()                                                ││
│ │  1. Validate alignment (512-byte sectors)                    ││
│ │  2. Check if dmabuf registered for this buffer+file pair     ││
│ └───┬─────────────────────────────────┬────────────────────────┘│
│     │ dmabuf available                │ dmabuf failed           │
│ ┌───▼─────────────────────────────┐  ┌▼───────────────────────┐│
│ │ Direct Path                     │  │ Fallback Path          ││
│ │  1. Get io_uring SQE            │  │  1. zeMemAllocHost()   ││
│ │  2. io_uring_prep_read_fixed()  │  │  2. Level Zero memcpy  ││
│ │     - buf_idx = registered slot │  │     (GPU ↔ Host)       ││
│ │  3. io_uring_submit_and_wait()  │  │  3. io_uring read/write││
│ │  4. Process CQE completion      │  │     (Host ↔ File)      ││
│ └───┬─────────────────────────────┘  └────────────────────────┘│
└─────┼──────────────────────────────────────────────────────────┘
      │
┌─────▼──────────────────────────────────────────────────────────┐
│ Level Zero (Intel GPU Driver - Userspace)                       │
│ ┌──────────────────────────────────────────────────────────────┐│
│ │ Buffer Registration Phase (one-time setup):                  ││
│ │  1. zeMemAllocDevice() - Allocate GPU memory                 ││
│ │  2. zeMemGetAllocProperties() - Export dmabuf FD             ││
│ │  3. Store dmabuf_fd in buffer tracking                       ││
│ └──────────────────────────────────────────────────────────────┘│
└─────┬──────────────────────────────────────────────────────────┘
      │ dmabuf_fd
┌─────▼──────────────────────────────────────────────────────────┐
│ liburing (io_uring Userspace Library)                           │
│ ┌──────────────────────────────────────────────────────────────┐│
│ │ register_dmabuf_buffer() - One-time per buffer+file pair:    ││
│ │                                                              ││
│ │  struct io_uring_regbuf_desc desc = {                        ││
│ │      .type      = IO_REGBUF_TYPE_DMABUF,                     ││
│ │      .target_fd = nvme_fd,        // File to read/write      ││
│ │      .dmabuf_fd = dmabuf_fd       // GPU memory dmabuf       ││
│ │  };                                                          ││
│ │                                                              ││
│ │  io_uring_register_buffers_update() with EXTENDED flag       ││
│ └───┬──────────────────────────────────────────────────────────┘│
└─────┼──────────────────────────────────────────────────────────┘
      │ IORING_REGISTER_BUFFERS_UPDATE
┌─────▼──────────────────────────────────────────────────────────┐
│ Linux Kernel - io_uring Core                                    │
│ ┌──────────────────────────────────────────────────────────────┐│
│ │ io_register_dmabuf_buffer()                                  ││
│ │  1. Get dmabuf from FD                                       ││
│ │  2. Call file->f_op->create_dmabuf_token()                   ││
│ │  3. Block device creates io_dmabuf_token + io_dmabuf_map     ││
│ │  4. Call dma_buf_map_attachment() - creates long-lived map   ││
│ │  5. Store mapping in io_uring buffer registry               ││
│ └──────────────────────────────────────────────────────────────┘│
│                                                                 │
│ ┌──────────────────────────────────────────────────────────────┐│
│ │ I/O Path (per operation):                                    ││
│ │  1. IORING_OP_READ_FIXED / WRITE_FIXED                       ││
│ │  2. Lookup registered buffer by index                        ││
│ │  3. Use ITER_DMABUF_MAP iterator                             ││
│ │  4. NVMe driver builds PRP list from dmabuf mapping          ││
│ │  5. Execute DMA transfer                                     ││
│ └───┬──────────────────────────────────────────────────────────┘│
└─────┼──────────────────────────────────────────────────────────┘
      │
┌─────▼──────────────────────────────────────────────────────────┐
│ Hardware: Intel Arc GPU ←──PCIe P2P (via dmabuf)──→ NVMe SSD   │
│  Direct BAR-to-BAR transfer, no CPU/system memory involved      │
└─────────────────────────────────────────────────────────────────┘

Key Characteristics:
  ✓ Long-lived dmabuf mappings (amortized overhead)
  ✓ io_uring async I/O framework
  ✓ Generic dmabuf mechanism (vendor-agnostic in theory)
  ✓ Requires kernel 6.17+ with io_uring dmabuf patches
  ✓ Two-phase: (1) register dmabuf, (2) submit I/O
```

---

### 3. Proposed libze/io_uring/p2pdma Approach (hipFile-style for Intel Arc)

```
┌─────────────────────────────────────────────────────────────────┐
│ Application Layer                                               │
│ ┌─────────────────────────────────────────────────────────────┐ │
│ │ cuFileRead(fh, gpu_ptr, size, file_offset, buffer_offset)  │ │
│ └──────────────────────────────┬────────────────────────────────┘│
└────────────────────────────────┼────────────────────────────────┘
                                 │
┌────────────────────────────────▼────────────────────────────────┐
│ libcufile.so (Wrapper - Userspace)                              │
│ ┌──────────────────────────────────────────────────────────────┐│
│ │ Backend Scoring System (NEW - modeled after hipFile)         ││
│ │  • Fastpath: Check alignment, GPU memory, p2pdma_distance    ││
│ │  • Fallback: Bounce buffer via io_uring + Level Zero memcpy ││
│ └───┬──────────────────────────────────────────────────┬───────┘│
│     │ Score=100 (P2P eligible)      Score=0 (fallback) │        │
│ ┌───▼──────────────────────────┐  ┌──────────────────▼────────┐│
│ │ Fastpath Backend (NEW)       │  │ Fallback Backend (EXISTS) ││
│ │  1. Validate params          │  │  1. Allocate bounce buf   ││
│ │  2. Create zeFileHandle      │  │  2. Level Zero memcpy     ││
│ │  3. Call zeAmdFileRead()     │  │     (GPU ↔ Host)          ││
│ │     OR                       │  │  3. io_uring read/write   ││
│ │     Use p2pdma-specific IOCTL│  │     (Host ↔ File)         ││
│ └───┬──────────────────────────┘  └───────────────────────────┘│
└─────┼──────────────────────────────────────────────────────────┘
      │
      │ ┌──────────────────────────────────────────────────────┐
      │ │ OPTION A: Level Zero Extension (preferred)           │
      │ └──────────────────────────────────────────────────────┘
      │
┌─────▼──────────────────────────────────────────────────────────┐
│ Level Zero Extensions (NEW API - requires Intel driver changes) │
│ ┌──────────────────────────────────────────────────────────────┐│
│ │ zeFileHandleCreate(device, fd, &file_handle)                 ││
│ │  • Registers file descriptor with GPU driver                 ││
│ │  • Driver validates p2pdma support                           ││
│ │                                                              ││
│ │ zeFileRead(file_handle, device_ptr, size, offset, stream)    ││
│ │  • Direct GPU-to-file I/O via driver IOCTL                   ││
│ │  • Synchronous or async (stream-based)                       ││
│ └───┬──────────────────────────────────────────────────────────┘│
└─────┼──────────────────────────────────────────────────────────┘
      │
      │      OR
      │
      │ ┌──────────────────────────────────────────────────────┐
      │ │ OPTION B: Direct i915 IOCTL (less portable)          │
      │ └──────────────────────────────────────────────────────┘
      │
┌─────▼──────────────────────────────────────────────────────────┐
│ i915 Kernel Driver (Intel GPU - NEW IOCTL needed)               │
│ ┌──────────────────────────────────────────────────────────────┐│
│ │ DRM_IOCTL_I915_GEM_FILE_IO (NEW)                             ││
│ │  1. Validate pci_p2pdma_distance_many() >= 0                 ││
│ │  2. Get GPU memory object from handle                        ││
│ │  3. Use kernel P2PDMA subsystem (CONFIG_PCI_P2PDMA)          ││
│ │  4. Setup DMA mapping: Intel GPU BAR ↔ NVMe                  ││
│ │  5. Execute P2P DMA transfer                                 ││
│ │  6. Return bytes transferred or error                        ││
│ └──────────────────────────────────────────────────────────────┘│
└─────┬──────────────────────────────────────────────────────────┘
      │
┌─────▼──────────────────────────────────────────────────────────┐
│ Hardware: Intel Arc GPU ←──PCIe P2P (p2pdma)──→ NVMe SSD       │
│  Direct BAR-to-BAR transfer, no CPU/system memory involved      │
└─────────────────────────────────────────────────────────────────┘

Key Characteristics:
  ✓ Similar architecture to hipFile
  ✓ Single IOCTL per operation (or Level Zero call)
  ✓ Kernel manages P2PDMA mapping
  ✓ No dmabuf export required
  ✓ No io_uring dependency for P2P path
  ✓ Transparent fallback to bounce buffer
  ✓ Requires Intel driver changes (significant effort)
```

---

### 4. Hybrid Approach: io_uring with p2pdma Registration (Alternative)

```
┌─────────────────────────────────────────────────────────────────┐
│ Application Layer                                               │
│ ┌─────────────────────────────────────────────────────────────┐ │
│ │ cuFileRead(fh, gpu_ptr, size, file_offset, buffer_offset)  │ │
│ └──────────────────────────┬──────────────────────────────────┘ │
└────────────────────────────┼────────────────────────────────────┘
                             │
┌────────────────────────────▼────────────────────────────────────┐
│ libcufile.so (Wrapper - Userspace)                              │
│ ┌──────────────────────────────────────────────────────────────┐│
│ │ issue_io_op() - MODIFIED                                     ││
│ │  1. Validate alignment (4096-byte for p2pdma)                ││
│ │  2. Check if p2pdma buffer registered                        ││
│ └───┬─────────────────────────────────┬────────────────────────┘│
│     │ p2pdma available                │ fallback               │
│ ┌───▼─────────────────────────────┐  ┌▼───────────────────────┐│
│ │ Direct Path (MODIFIED)          │  │ Fallback Path (SAME)   ││
│ │  1. Get io_uring SQE            │  │  1. zeMemAllocHost()   ││
│ │  2. io_uring_prep_read_fixed()  │  │  2. Level Zero memcpy  ││
│ │     - buf_idx points to p2pdma  │  │  3. io_uring I/O       ││
│ │  3. io_uring_submit_and_wait()  │  └────────────────────────┘│
│ └───┬─────────────────────────────┘                            │
└─────┼──────────────────────────────────────────────────────────┘
      │
┌─────▼──────────────────────────────────────────────────────────┐
│ liburing (NEW p2pdma registration API)                          │
│ ┌──────────────────────────────────────────────────────────────┐│
│ │ register_p2pdma_buffer() - NEW FUNCTION:                     ││
│ │                                                              ││
│ │  struct io_uring_regbuf_desc desc = {                        ││
│ │      .type      = IO_REGBUF_TYPE_P2PDMA,  // NEW TYPE        ││
│ │      .target_fd = nvme_fd,                                   ││
│ │      .gpu_dev   = ze_device_handle,       // GPU device info ││
│ │      .gpu_ptr   = device_ptr              // GPU BAR address ││
│ │  };                                                          ││
│ │                                                              ││
│ │  io_uring_register_buffers_update()                          ││
│ └───┬──────────────────────────────────────────────────────────┘│
└─────┼──────────────────────────────────────────────────────────┘
      │
┌─────▼──────────────────────────────────────────────────────────┐
│ Linux Kernel - io_uring Core (MODIFIED)                         │
│ ┌──────────────────────────────────────────────────────────────┐│
│ │ io_register_p2pdma_buffer() - NEW HANDLER                    ││
│ │  1. Resolve GPU device from handle                           ││
│ │  2. Validate pci_p2pdma_distance() >= 0                      ││
│ │  3. Call pci_p2pdma_map_sg() to create mapping               ││
│ │  4. Store p2pdma pages in io_uring buffer registry           ││
│ └──────────────────────────────────────────────────────────────┘│
│                                                                 │
│ ┌──────────────────────────────────────────────────────────────┐│
│ │ I/O Path (per operation):                                    ││
│ │  1. IORING_OP_READ_FIXED / WRITE_FIXED                       ││
│ │  2. Lookup registered p2pdma buffer                          ││
│ │  3. Use ITER_P2PDMA iterator (NEW)                           ││
│ │  4. NVMe driver builds PRP list from p2pdma pages            ││
│ │  5. Execute DMA transfer                                     ││
│ └───┬──────────────────────────────────────────────────────────┘│
└─────┼──────────────────────────────────────────────────────────┘
      │
┌─────▼──────────────────────────────────────────────────────────┐
│ Hardware: Intel Arc GPU ←──PCIe P2P (p2pdma)──→ NVMe SSD       │
│  Direct BAR-to-BAR transfer, no CPU/system memory involved      │
└─────────────────────────────────────────────────────────────────┘

Key Characteristics:
  ✓ Keeps io_uring async framework
  ✓ Uses kernel p2pdma instead of dmabuf
  ✓ No dmabuf export overhead
  ✓ Requires io_uring kernel changes (new registration type)
  ✓ More kernel-generic than Option 3
  ✓ Medium implementation effort
```

---

## Comparison Matrix

| Aspect | Current (dmabuf) | Proposed (Option A) | Hybrid (Option B) |
|--------|------------------|---------------------|-------------------|
| **Kernel Changes** | Minimal (use existing io_uring dmabuf) | Major (new i915 IOCTL + Level Zero API) | Medium (new io_uring p2pdma type) |
| **Driver Changes** | None | Intel GPU driver + Level Zero | None (uses existing p2pdma) |
| **Userspace Complexity** | Medium (dmabuf export) | Low (direct calls) | Medium (same as dmabuf) |
| **Architecture** | io_uring-centric | GPU driver-centric (like hipFile) | io_uring-centric with p2pdma |
| **Async Framework** | io_uring built-in | Custom or Level Zero streams | io_uring built-in |
| **Portability** | Vendor-agnostic (dmabuf) | Intel-specific | Vendor-agnostic (p2pdma) |
| **Mapping Lifecycle** | Long-lived (registration) | Per-op or cached | Long-lived (registration) |
| **Kernel Version** | 6.17+ (dmabuf patches) | Any with p2pdma | Custom (p2pdma patches) |
| **Implementation Effort** | Low (already done) | Very High | High |
| **Performance** | Excellent (proven) | Excellent (hipFile model) | Excellent (p2pdma native) |
| **Fallback** | CPU bounce buffer | CPU bounce buffer | CPU bounce buffer |

---

## Required Changes by Approach

### Option A: GPU Driver-Centric (hipFile Model)

**Intel Driver Changes:**
1. Add new i915 IOCTL: `DRM_IOCTL_I915_GEM_FILE_IO`
2. Implement file handle registration in driver
3. Add p2pdma validation and mapping logic
4. Handle synchronous and async I/O operations
5. Integrate with kernel p2pdma subsystem

**Level Zero Changes:**
1. New API extension: `ZE_extension_file_io`
2. Functions:
   - `zeFileHandleCreate(device, fd, flags, &handle)`
   - `zeFileHandleDestroy(handle)`
   - `zeFileRead(handle, device_ptr, size, offset, event)`
   - `zeFileWrite(handle, device_ptr, size, offset, event)`
3. Add p2pdma distance checking
4. Expose file I/O capabilities in device properties

**Userspace Library Changes:**
1. Remove dmabuf export logic
2. Add backend scoring system (like hipFile)
3. Implement fastpath using Level Zero file I/O
4. Keep fallback path (already exists)
5. Add configuration for p2pdma enable/disable
6. Update buffer registration to skip dmabuf export

**Estimated Effort:** 6-12 months (requires Intel collaboration)

---

### Option B: Hybrid io_uring + p2pdma

**Kernel Changes:**
1. Add `IO_REGBUF_TYPE_P2PDMA` registration type
2. Implement `io_register_p2pdma_buffer()` handler
3. Create `ITER_P2PDMA` iterator type
4. Modify block layer to handle p2pdma buffers
5. Add GPU device resolution from userspace handle

**liburing Changes:**
1. Add `io_uring_regbuf_desc_p2pdma` structure
2. New helper: `io_uring_register_p2pdma_buffer()`
3. Update documentation

**Userspace Library Changes:**
1. Replace dmabuf export with GPU BAR address lookup
2. Use new p2pdma registration API
3. Increase alignment from 512B to 4KB (p2pdma requirement)
4. Add p2pdma distance validation
5. Keep fallback path (already exists)

**Estimated Effort:** 3-6 months (kernel upstreaming is slow)

---

### Option C: Keep Current dmabuf Approach (No Changes)

**Advantages:**
- Already implemented and working
- Vendor-agnostic (works with any GPU supporting dmabuf)
- Uses mainline io_uring dmabuf patches
- No Intel-specific dependencies
- Proven performance

**Disadvantages:**
- Requires kernel 6.17+ with dmabuf patches
- dmabuf export overhead (minor)
- Two-phase registration (dmabuf + io_uring)

**Recommendation:** This is the most pragmatic choice unless:
1. You need compatibility with older kernels (pre-6.17)
2. You want to avoid dmabuf layer
3. Intel provides Level Zero file I/O extensions

---

## Detailed Change Analysis

### What Would Change in the Codebase

#### For Option A (GPU Driver-Centric):

**File: `src/cufile_iouring.c`**

**Changes to `cuFileBufRegister()` (lines 870-950):**
```c
// REMOVE: dmabuf export logic
- ze_ipc_mem_handle_t ipc_handle;
- zeMemGetIpcHandle(g_state.ze_context, dev_ptr, &ipc_handle);
- buf->dmabuf_fd = ipc_handle.fd;

// ADD: Just store device pointer, no export needed
buf->dev_ptr = dev_ptr;
buf->size = size;
buf->ze_allocated = 1;
```

**Changes to `issue_io_op()` (lines 1212-1347):**
```c
// REMOVE: dmabuf registration
- register_dmabuf(&buf->dmabuf_fd, fd, &buf->registered_idx);

// REMOVE: io_uring fixed buffer I/O
- io_uring_prep_read_fixed(sqe, fd, ...);

// ADD: Level Zero file I/O call
ze_file_handle_t fh;
zeFileHandleCreate(g_state.ze_device, fd, 0, &fh);

ze_file_desc_t desc = {
    .stype = ZE_STRUCTURE_TYPE_FILE_DESC,
    .device_ptr = buf->dev_ptr,
    .size = size,
    .file_offset = file_offset,
    .buffer_offset = buffer_offset
};

if (is_read) {
    zeFileRead(fh, &desc, NULL);  // NULL = synchronous
} else {
    zeFileWrite(fh, &desc, NULL);
}

zeFileHandleDestroy(fh);
```

**New Files:**
- `src/backend_scoring.c` - Backend selection logic
- `src/level_zero_fileio.c` - Wrapper for Level Zero file I/O
- `src/p2pdma_check.c` - p2pdma distance validation

**Remove:**
- All dmabuf-specific code (~200 lines)
- io_uring buffer registration for P2P path
- `register_dmabuf_buffer()` function

**Keep:**
- io_uring for fallback path (CPU bounce buffer)
- Batch API (can use Level Zero events instead of io_uring)
- Global state management
- Alignment validation (change to 4KB)

---

#### For Option B (Hybrid io_uring + p2pdma):

**File: `src/cufile_iouring.c`**

**Changes to `cuFileBufRegister()` (lines 870-950):**
```c
// MODIFY: Get GPU BAR address instead of dmabuf
- zeMemGetIpcHandle(g_state.ze_context, dev_ptr, &ipc_handle);
- buf->dmabuf_fd = ipc_handle.fd;

+ ze_memory_allocation_properties_t props;
+ zeMemGetAllocProperties(g_state.ze_context, dev_ptr, &props, NULL);
+ buf->gpu_bar_address = props.paddr;  // Physical BAR address
```

**Changes to `register_dmabuf_buffer()` (lines 1124-1138):**
```c
// Rename to register_p2pdma_buffer()
static int register_p2pdma_buffer(void *gpu_ptr, uintptr_t bar_addr,
                                   int target_fd, int *reg_idx) {
    struct io_uring_regbuf_desc_p2pdma p2p_buf = {
        .type      = IO_REGBUF_TYPE_P2PDMA,
        .target_fd = target_fd,
        .gpu_dev   = g_state.ze_device,    // Need to pass device info
        .gpu_bar   = bar_addr,
        .size      = size
    };

    // Use new p2pdma registration API
    return io_uring_register_p2pdma_buffer(&g_state.ring, &p2p_buf, reg_idx);
}
```

**Changes to alignment validation (lines 1062-1100):**
```c
// Change from 512B to 4KB
- #define SECTOR_SIZE 512
+ #define P2PDMA_ALIGN 4096

- if (file_offset % SECTOR_SIZE != 0 || size % SECTOR_SIZE != 0)
+ if (file_offset % P2PDMA_ALIGN != 0 || size % P2PDMA_ALIGN != 0)
```

**Keep:**
- All io_uring logic (SQE/CQE handling)
- Batch API
- Async write tracking
- Fallback path

**New Dependencies:**
- Updated liburing with p2pdma support
- Kernel with io_uring p2pdma patches

---

## Recommendation

**Best Approach: Keep Current dmabuf Implementation (Option C)**

**Reasons:**
1. **Already Working**: Proven implementation with good performance
2. **Vendor Agnostic**: Works with any GPU supporting dmabuf (AMD, Intel, NVIDIA)
3. **Mainline Path**: io_uring dmabuf patches are being upstreamed
4. **Low Risk**: No dependency on Intel-specific changes
5. **Future Proof**: If io_uring adds p2pdma support, migration is straightforward

**When to Consider Alternatives:**
- **Option A** if Intel provides Level Zero file I/O extension
- **Option B** if io_uring adds p2pdma registration type to mainline
- Both options require significant kernel/driver development effort

**Incremental Improvements to Current Implementation:**
1. Add p2pdma distance checking before dmabuf registration
2. Improve fallback detection (avoid failed registration attempts)
3. Add statistics to track P2P vs fallback usage
4. Optimize batch operations for better CQE handling
5. Add small I/O optimizations (avoid P2P for < 64KB transfers)

---

## Summary

The current dmabuf approach is well-designed and follows the direction of mainline Linux kernel development. While hipFile's p2pdma approach works well for AMD, replicating it for Intel Arc would require substantial driver changes that may not align with Intel's roadmap.

**Key Insight**: The dmabuf abstraction provides vendor independence at the cost of a thin extra layer. This is a good trade-off for a cuFile wrapper aiming for portability.

If the goal is maximum performance similarity to hipFile, focus should be on:
1. Optimizing the existing dmabuf path
2. Improving fallback heuristics
3. Adding backend scoring to avoid failed P2P attempts
4. Collaborating with Intel on potential Level Zero extensions

The architectural difference (io_uring vs GPU-driver-IOCTL) is less important than ensuring reliable P2P DMA when conditions are met and graceful fallback otherwise.
