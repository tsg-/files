# MI300X + Gaudi3 + LMCache Configuration Guide

## Important Notes on UCX Transport for Heterogeneous Hardware

This setup uses **heterogeneous accelerators** (AMD MI300X + Intel Gaudi3), which require special attention to UCX transport configuration:

- **AMD MI300X (ROCm)**: Does NOT support CUDA transports (`cuda_copy`, `cuda_ipc`)
- **Intel Gaudi3 (Habana)**: Does NOT support CUDA transports
- **Solution**: Use hardware-agnostic UCX transports that work across different accelerators

### Correct UCX Transport Settings

**For AMD MI300X (Prefill Node with ROCm):**
```bash
# Use ROCm-specific transport for GPU direct
UCX_TLS=rc,rocm,sm,self         # RDMA + ROCm GPU + shared memory + loopback
# or for TCP fallback:
UCX_TLS=tcp,rocm,sm,self        # TCP + ROCm GPU + shared memory + loopback
```

**For Intel Gaudi3 (Decode Node with Habana):**
```bash
# Use Gaudi-specific transport for GPU direct (requires ucx-gaudi)
UCX_TLS=rc,gaudi,sm,self        # RDMA + Gaudi GPU + shared memory + loopback
# or for TCP fallback:
UCX_TLS=tcp,gaudi,sm,self       # TCP + Gaudi GPU + shared memory + loopback
```

### UCX Transport Breakdown

| Transport | Purpose | Works With |
|-----------|---------|------------|
| **rc** | RDMA Reliable Connection (InfiniBand) | All hardware |
| **dc** | RDMA Dynamically Connected (InfiniBand) | All hardware |
| **tcp** | TCP/IP networking | All hardware |
| **sm** | Shared memory (intra-node) | All hardware |
| **self** | Loopback (same process) | All hardware |
| **rocm** | AMD GPU direct transport | ✅ AMD ROCm only |
| **gaudi** | Intel Gaudi GPU direct transport | ✅ Intel Gaudi only |
| **cuda_copy** | NVIDIA GPU-direct copy | ❌ NVIDIA only |
| **cuda_ipc** | NVIDIA GPU IPC | ❌ NVIDIA only |

---

## Architecture Overview

This guide explains how to configure LMCache with NIXL for a disaggregated prefill-decode setup using AMD MI300X (prefill) and Intel Gaudi3 (decode), with optional Ceph S3 for persistent KV cache storage.

### Architecture Layers

There are **three distinct layers** working together:

```
┌─────────────────────────────────────────────────────────────┐
│ Layer 1: vLLM KV Transfer API                              │
│          --kv-transfer-config (JSON)                        │
│          Tells vLLM: "use NixlConnector for disaggregation"│
└─────────────────────────────────────────────────────────────┘
                           │
┌─────────────────────────────────────────────────────────────┐
│ Layer 2: LMCache (cache management + transfer logic)       │
│          LMCACHE_ENABLE_PD=True                            │
│          LMCACHE_TRANSFER_CHANNEL=nixl                     │
│          Implements the actual cache transfer via NIXL      │
└─────────────────────────────────────────────────────────────┘
                           │
┌─────────────────────────────────────────────────────────────┐
│ Layer 3: Transport & Storage (physical layer)              │
│  - UCX/NIXL: High-speed P→D transfer (real-time)          │
│  - S3/Ceph: Persistent cache storage (optional)            │
└─────────────────────────────────────────────────────────────┘
```

## Configuration Comparison: With S3 vs Without S3

### Architecture 1: NIXL Only (No S3)

**Use case:** Single-session inference, no cross-request cache sharing

```
Prefill (MI300X)          NIXL (RDMA)          Decode (Gaudi3)
─────────────────         ───────────          ───────────────
┌─────────────┐                                ┌─────────────┐
│ GPU (192GB) │───────────────────────────────>│ GPU (128GB) │
│ CPU (20GB)  │    50-100 GB/s, 1-10ms         │ CPU (15GB)  │
│ Disk (100GB)│                                │ Disk (50GB) │
└─────────────┘                                └─────────────┘

Cache scope: Request-only
Persistence: None (cache lost after request)
```

### Architecture 2: NIXL + S3 (Full Stack)

**Use case:** Multi-session, cross-node cache sharing, production workloads

```
Prefill (MI300X)          NIXL (RDMA)          Decode (Gaudi3)
─────────────────         ───────────          ───────────────
┌─────────────┐                                ┌─────────────┐
│ GPU (192GB) │───────────────────────────────>│ GPU (128GB) │
│ CPU (20GB)  │    50-100 GB/s, 1-10ms         │ CPU (15GB)  │
│ Disk (100GB)│                                │ Disk (50GB) │
└──────┬──────┘                                └──────┬──────┘
       │                                              │
       └──────────────────┬───────────────────────────┘
                          │
                  ┌───────▼────────┐
                  │ S3/Ceph (∞)    │
                  │ L2 - Persistent│
                  │ 10-100ms       │
                  └────────────────┘
                         ▲
                         │
            Shared across all nodes
            Persists across requests
```

**Cache scope:** Global (all nodes, all requests)
**Persistence:** Durable storage for prefix reuse

## Configuration 1: NIXL Only (Without S3)

### Prefill Node (MI300X with ROCm) - NIXL Only

```bash
docker run -d \
  --name vllm-prefill-mi300x \
  --device=/dev/kfd --device=/dev/dri \
  --group-add video \
  --net=host \
  --ipc=host \
  -e HSA_FORCE_FINE_GRAIN_PCIE=1 \
  -e ROCM_HOME=/opt/rocm \
  -e HIP_VISIBLE_DEVICES=0,1,2,3 \
  \
  # ===== UCX Transport for RDMA (ROCm with GPU direct) ===== #
  -e UCX_TLS=rc,rocm,sm,self \
  \
  # ===== NIXL Side Channel ===== #
  -e VLLM_NIXL_SIDE_CHANNEL_HOST=<prefill-node-ip> \
  -e VLLM_NIXL_SIDE_CHANNEL_PORT=50051 \
  \
  # ===== LMCache Local Tiers Only (No S3) ===== #
  -e LMCACHE_LOCAL_CPU=True \
  -e LMCACHE_MAX_LOCAL_CPU_SIZE=20.0 \
  -e LMCACHE_LOCAL_DISK=/tmp/lmcache \
  -e LMCACHE_MAX_LOCAL_DISK_SIZE=100.0 \
  -e LMCACHE_CACHE_POLICY=LRU \
  \
  # ===== LMCache Prefill-Decode Disaggregation ===== #
  -e LMCACHE_ENABLE_PD=True \
  -e LMCACHE_PD_ROLE=sender \
  -e LMCACHE_TRANSFER_CHANNEL=nixl \
  \
  # ===== NO S3 Configuration ===== #
  # LMCACHE_REMOTE_URL not set - no remote storage
  \
  # ===== LMCache Performance Tuning ===== #
  -e LMCACHE_CHUNK_SIZE=256 \
  -e LMCACHE_SAVE_DECODE_CACHE=False \
  \
  vllm-prefill-mi300x:latest \
  vllm serve meta-llama/Llama-3.1-70b \
    --tensor-parallel-size 4 \
    --max-num-seqs 512 \
    --max-model-len 8192 \
    --gpu-memory-utilization 0.90 \
    --enable-chunked-prefill \
    --enable-prefix-caching \
    --kv-transfer-config '{"kv_connector":"NixlConnector","kv_role":"kv_producer","kv_buffer_device":"cpu"}' \
    --port 8000 \
    --trust-remote-code
```

### Decode Node (Gaudi3 with Habana) - NIXL Only

```bash
docker run -d \
  --name vllm-decode-gaudi3 \
  --runtime=habana \
  -e HABANA_VISIBLE_DEVICES=all \
  -e PT_HPU_LAZY_MODE=1 \
  -e OMPI_MCA_btl_vader_single_copy_mechanism=none \
  --cap-add=sys_nice \
  --net=host \
  --ipc=host \
  \
  # ===== UCX Transport for RDMA (Gaudi with GPU direct) ===== #
  -e UCX_TLS=rc,gaudi,sm,self \
  \
  # ===== NIXL Side Channel - connects to prefill ===== #
  -e VLLM_NIXL_SIDE_CHANNEL_HOST=<prefill-node-ip> \
  -e VLLM_NIXL_SIDE_CHANNEL_PORT=50051 \
  \
  # ===== LMCache Local Tiers Only (No S3) ===== #
  -e LMCACHE_LOCAL_CPU=True \
  -e LMCACHE_MAX_LOCAL_CPU_SIZE=15.0 \
  -e LMCACHE_LOCAL_DISK=/tmp/lmcache \
  -e LMCACHE_MAX_LOCAL_DISK_SIZE=50.0 \
  -e LMCACHE_CACHE_POLICY=LRU \
  \
  # ===== LMCache Prefill-Decode Disaggregation ===== #
  -e LMCACHE_ENABLE_PD=True \
  -e LMCACHE_PD_ROLE=receiver \
  -e LMCACHE_TRANSFER_CHANNEL=nixl \
  \
  # ===== NO S3 Configuration ===== #
  # LMCACHE_REMOTE_URL not set - no remote storage
  \
  # ===== LMCache Performance Tuning ===== #
  -e LMCACHE_CHUNK_SIZE=256 \
  \
  vllm-decode-gaudi3:latest \
  vllm serve meta-llama/Llama-3.1-70b \
    --tensor-parallel-size 4 \
    --max-num-seqs 256 \
    --max-model-len 8192 \
    --gpu-memory-utilization 0.85 \
    --kv-transfer-config '{"kv_connector":"NixlConnector","kv_role":"kv_consumer","kv_buffer_device":"cpu"}' \
    --port 8001 \
    --trust-remote-code
```

## Configuration 2: NIXL + S3 (Full Stack)

### Prefill Node (MI300X with ROCm) - NIXL + S3

```bash
docker run -d \
  --name vllm-prefill-mi300x \
  --device=/dev/kfd --device=/dev/dri \
  --group-add video \
  --net=host \
  --ipc=host \
  -e HSA_FORCE_FINE_GRAIN_PCIE=1 \
  -e ROCM_HOME=/opt/rocm \
  -e HIP_VISIBLE_DEVICES=0,1,2,3 \
  \
  # ===== UCX Transport for RDMA (ROCm with GPU direct) ===== #
  -e UCX_TLS=rc,rocm,sm,self \
  \
  # ===== NIXL Side Channel ===== #
  -e VLLM_NIXL_SIDE_CHANNEL_HOST=<prefill-node-ip> \
  -e VLLM_NIXL_SIDE_CHANNEL_PORT=50051 \
  \
  # ===== LMCache Local Cache Tiers (L1) ===== #
  -e LMCACHE_LOCAL_CPU=True \
  -e LMCACHE_MAX_LOCAL_CPU_SIZE=20.0 \
  -e LMCACHE_LOCAL_DISK=/tmp/lmcache \
  -e LMCACHE_MAX_LOCAL_DISK_SIZE=100.0 \
  -e LMCACHE_CACHE_POLICY=LRU \
  \
  # ===== LMCache Prefill-Decode Disaggregation ===== #
  -e LMCACHE_ENABLE_PD=True \
  -e LMCACHE_PD_ROLE=sender \
  -e LMCACHE_TRANSFER_CHANNEL=nixl \
  \
  # ===== LMCache Remote Persistent Storage (L2) ===== #
  -e LMCACHE_REMOTE_URL=s3://s3.cephlab.com/nixl/ \
  -e LMCACHE_S3_ENDPOINT_URL=http://s3.cephlab.com \
  -e AWS_ACCESS_KEY_ID=215QPBKECPTMTQ1MLZYW \
  -e AWS_SECRET_ACCESS_KEY=bvzEjZ4ePrisBSj3NOhB85FstcBOgy1nQeUBPoUc \
  -e AWS_DEFAULT_REGION=default \
  \
  # ===== LMCache Performance Tuning ===== #
  -e LMCACHE_CHUNK_SIZE=256 \
  -e LMCACHE_SAVE_DECODE_CACHE=False \
  \
  vllm-prefill-mi300x:latest \
  vllm serve meta-llama/Llama-3.1-70b \
    --tensor-parallel-size 4 \
    --max-num-seqs 512 \
    --max-model-len 8192 \
    --gpu-memory-utilization 0.90 \
    --enable-chunked-prefill \
    --enable-prefix-caching \
    --kv-transfer-config '{"kv_connector":"NixlConnector","kv_role":"kv_producer","kv_buffer_device":"cpu"}' \
    --port 8000 \
    --trust-remote-code
```

### Decode Node (Gaudi3 with Habana) - NIXL + S3

```bash
docker run -d \
  --name vllm-decode-gaudi3 \
  --runtime=habana \
  -e HABANA_VISIBLE_DEVICES=all \
  -e PT_HPU_LAZY_MODE=1 \
  -e OMPI_MCA_btl_vader_single_copy_mechanism=none \
  --cap-add=sys_nice \
  --net=host \
  --ipc=host \
  \
  # ===== UCX Transport for RDMA (Gaudi with GPU direct) ===== #
  -e UCX_TLS=rc,gaudi,sm,self \
  \
  # ===== NIXL Side Channel - connects to prefill ===== #
  -e VLLM_NIXL_SIDE_CHANNEL_HOST=<prefill-node-ip> \
  -e VLLM_NIXL_SIDE_CHANNEL_PORT=50051 \
  \
  # ===== LMCache Local Cache Tiers (L1) ===== #
  -e LMCACHE_LOCAL_CPU=True \
  -e LMCACHE_MAX_LOCAL_CPU_SIZE=15.0 \
  -e LMCACHE_LOCAL_DISK=/tmp/lmcache \
  -e LMCACHE_MAX_LOCAL_DISK_SIZE=50.0 \
  -e LMCACHE_CACHE_POLICY=LRU \
  \
  # ===== LMCache Prefill-Decode Disaggregation ===== #
  -e LMCACHE_ENABLE_PD=True \
  -e LMCACHE_PD_ROLE=receiver \
  -e LMCACHE_TRANSFER_CHANNEL=nixl \
  \
  # ===== LMCache Remote Persistent Storage (L2) ===== #
  # IMPORTANT: Same S3 bucket as prefill node!
  -e LMCACHE_REMOTE_URL=s3://s3.cephlab.com/nixl/ \
  -e LMCACHE_S3_ENDPOINT_URL=http://s3.cephlab.com \
  -e AWS_ACCESS_KEY_ID=215QPBKECPTMTQ1MLZYW \
  -e AWS_SECRET_ACCESS_KEY=bvzEjZ4ePrisBSj3NOhB85FstcBOgy1nQeUBPoUc \
  -e AWS_DEFAULT_REGION=default \
  \
  # ===== LMCache Performance Tuning ===== #
  -e LMCACHE_CHUNK_SIZE=256 \
  \
  vllm-decode-gaudi3:latest \
  vllm serve meta-llama/Llama-3.1-70b \
    --tensor-parallel-size 4 \
    --max-num-seqs 256 \
    --max-model-len 8192 \
    --gpu-memory-utilization 0.85 \
    --kv-transfer-config '{"kv_connector":"NixlConnector","kv_role":"kv_consumer","kv_buffer_device":"cpu"}' \
    --port 8001 \
    --trust-remote-code
```

## Key Configuration Changes for Heterogeneous Setup

### 1. UCX Transport - Use GPU-Specific Transports

❌ **Don't use:** `UCX_TLS=rc,cuda_copy,cuda_ipc`
✅ **Use for MI300X:** `UCX_TLS=rc,rocm,sm,self`
✅ **Use for Gaudi3:** `UCX_TLS=rc,gaudi,sm,self`

**Reason:** `cuda_copy` and `cuda_ipc` are NVIDIA-specific. Use `rocm` for AMD and `gaudi` for Intel Gaudi.

### 2. KV Buffer Device - CPU instead of CUDA

❌ **Don't use:** `"kv_buffer_device":"cuda"`
✅ **Use instead:** `"kv_buffer_device":"cpu"`

**Reason:** In heterogeneous setups, CPU buffers provide compatibility across different accelerator types.

### 3. Actual Ceph S3 Configuration

```bash
LMCACHE_REMOTE_URL=s3://s3.cephlab.com/nixl/
LMCACHE_S3_ENDPOINT_URL=http://s3.cephlab.com
AWS_ACCESS_KEY_ID=215QPBKECPTMTQ1MLZYW
AWS_SECRET_ACCESS_KEY=bvzEjZ4ePrisBSj3NOhB85FstcBOgy1nQeUBPoUc
AWS_DEFAULT_REGION=default
```

## UCX Transport Configuration Reference

### For RDMA (InfiniBand) - With GPU Direct
```bash
UCX_TLS=rc,rocm,sm,self         # AMD MI300X (Prefill)
# rc = reliable connection (RDMA)
# rocm = ROCm GPU direct transport
# sm = shared memory (intra-node)
# self = loopback (same process)

UCX_TLS=rc,gaudi,sm,self        # Intel Gaudi3 (Decode)
# rc = reliable connection (RDMA)
# gaudi = Gaudi GPU direct transport (requires ucx-gaudi)
# sm = shared memory (intra-node)
# self = loopback (same process)
```

### For TCP Fallback (no RDMA available)
```bash
UCX_TLS=tcp,sm,self             # TCP instead of RDMA
```

### For RoCE (RDMA over Ethernet) - Heterogeneous Hardware
```bash
UCX_TLS=rc,sm,self              # Same as InfiniBand
# Ensure RoCE is configured on NICs
```

### ❌ DO NOT USE for ROCm/Habana
```bash
UCX_TLS=rc,cuda_copy,cuda_ipc   # NVIDIA ONLY - will fail!
```

## Verification Steps

### Verify UCX Configuration

```bash
# Check UCX transport on prefill node
docker exec vllm-prefill-mi300x env | grep UCX_TLS
# Expected: UCX_TLS=rc,rocm,sm,self

# Check UCX transport on decode node
docker exec vllm-decode-gaudi3 env | grep UCX_TLS
# Expected: UCX_TLS=rc,gaudi,sm,self

# Verify no CUDA transports are configured
docker exec vllm-prefill-mi300x env | grep UCX_TLS | grep -q cuda && echo "ERROR: CUDA transport found!" || echo "OK: No CUDA transport"
docker exec vllm-decode-gaudi3 env | grep UCX_TLS | grep -q cuda && echo "ERROR: CUDA transport found!" || echo "OK: No CUDA transport"
```

### Verify NIXL Connection

```bash
# Check NIXL logs on both nodes
docker logs vllm-prefill-mi300x | grep -i nixl
docker logs vllm-decode-gaudi3 | grep -i nixl

# Look for successful connection messages
# Should NOT see errors about CUDA or unsupported transports
```

### Verify S3 Configuration

```bash
# Test S3 access
aws s3 ls s3://nixl/ --endpoint-url http://s3.cephlab.com

# Verify credentials
docker exec vllm-prefill-mi300x env | grep AWS_ACCESS_KEY_ID
```

## Troubleshooting Heterogeneous Setup

### Issue: UCX transport errors with CUDA

**Symptom:**
```
ERROR: cuda_copy transport not available
ERROR: cuda_ipc not supported on this platform
```

**Solution:**
```bash
# Change from:
UCX_TLS=rc,cuda_copy,cuda_ipc

# To:
UCX_TLS=rc,sm,self
```

### Issue: KV buffer device errors

**Symptom:**
```
ERROR: CUDA device not available
```

**Solution:**
```bash
# Change kv_buffer_device from "cuda" to "cpu":
--kv-transfer-config '{"kv_connector":"NixlConnector","kv_role":"kv_producer","kv_buffer_device":"cpu"}'
```

### Issue: RDMA not working

**Solution:**
```bash
# Fall back to TCP:
UCX_TLS=tcp,sm,self

# Verify RDMA devices
ibstat
rdma link
```

## Performance Expectations

With correct UCX configuration (no CUDA transports):

- **RDMA throughput:** 50-100 GB/s between nodes
- **TCP throughput:** 1-10 GB/s between nodes
- **Shared memory (intra-node):** 100+ GB/s
- **NIXL transfer latency:** 1-10ms (RDMA), 10-50ms (TCP)

## References

- [vLLM Documentation](https://docs.vllm.ai/)
- [LMCache Documentation](https://docs.lmcache.ai/)
- [UCX Documentation](https://openucx.readthedocs.io/)
- [ROCm Documentation](https://rocm.docs.amd.com/)
- [Habana Gaudi Documentation](https://docs.habana.ai/)
