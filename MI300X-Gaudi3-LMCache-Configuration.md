# MI300X + Gaudi3 + LMCache Configuration Guide

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

### When to Use
- ✅ Testing/development
- ✅ Single-session inference
- ✅ No need for prefix cache sharing across requests
- ✅ Want simplest setup
- ❌ NOT for production with repeated prefixes

### Prefill Node (MI300X) - NIXL Only

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
  # ===== UCX Transport for RDMA ===== #
  -e UCX_TLS=rc,cuda_copy,cuda_ipc \
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
    --kv-transfer-config '{"kv_connector":"NixlConnector","kv_role":"kv_producer","kv_buffer_device":"cuda"}' \
    --port 8000 \
    --trust-remote-code
```

### Decode Node (Gaudi3) - NIXL Only

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
  # ===== UCX Transport for RDMA ===== #
  -e UCX_TLS=rc,cuda_copy,cuda_ipc \
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
    --kv-transfer-config '{"kv_connector":"NixlConnector","kv_role":"kv_consumer","kv_buffer_device":"cuda"}' \
    --port 8001 \
    --trust-remote-code
```

### Request Flow (NIXL Only)

```
Single Request Flow:
────────────────────

1. Request arrives at Prefill
   ↓
2. Prefill checks LOCAL cache only
   GPU cache → CPU cache → Disk cache
   ↓
3. MISS - Compute prefill
   ↓
4. Store in local tiers
   GPU ← CPU ← Disk
   ↓
5. Transfer KV cache to Decode via NIXL
   (RDMA, 50-100 GB/s, 1-10ms)
   ↓
6. Decode receives, stores in local tiers
   ↓
7. Decode generates tokens
   ↓
8. Request completes - KV cache may be evicted from local tiers
```

**Second Request with Same Prefix:**
```
1. Request arrives at Prefill
   ↓
2. Check local cache
   - If still in GPU/CPU/Disk: HIT (fast)
   - If evicted: MISS (recompute)
   ↓
3. If MISS: Full prefill computation again
   ↓
4. Transfer to Decode via NIXL
   ↓
5. Decode generates tokens
```

**Key Limitation:** No cache persistence across:
- Different prefill nodes
- Different decode nodes
- After local cache eviction
- Server restarts

---

## Configuration 2: NIXL + S3 (Full Stack)

### When to Use
- ✅ Production workloads
- ✅ High prefix reuse across requests
- ✅ Multiple prefill/decode nodes
- ✅ Long-term cache persistence
- ✅ Cross-node cache sharing

### Prefill Node (MI300X) - NIXL + S3

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
  # ===== UCX Transport for RDMA ===== #
  -e UCX_TLS=rc,cuda_copy,cuda_ipc \
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
    --kv-transfer-config '{"kv_connector":"NixlConnector","kv_role":"kv_producer","kv_buffer_device":"cuda"}' \
    --port 8000 \
    --trust-remote-code
```

### Decode Node (Gaudi3) - NIXL + S3

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
  # ===== UCX Transport for RDMA ===== #
  -e UCX_TLS=rc,cuda_copy,cuda_ipc \
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
    --kv-transfer-config '{"kv_connector":"NixlConnector","kv_role":"kv_consumer","kv_buffer_device":"cuda"}' \
    --port 8001 \
    --trust-remote-code
```

### Request Flow Examples (NIXL + S3)

#### Scenario 1: First Request (Cold Cache)

```
1. Request arrives at Prefill
   ↓
2. Prefill checks LMCache hierarchy
   GPU cache → CPU cache → Disk cache → S3
   ↓
3. MISS everywhere - Compute prefill
   ↓
4. Store in local LMCache tiers
   GPU ← CPU ← Disk ← S3 (async save for future)
   ↓
5. Transfer KV cache to Decode via NIXL
   (RDMA, 50-100 GB/s, 1-10ms)
   ↓
6. Decode receives, stores in local tiers
   ↓
7. Decode generates tokens
```

#### Scenario 2: Second Request (Warm S3 Cache, Different Node)

```
1. Request arrives at Prefill Node 2 (different from first request)
   ↓
2. Prefill checks LMCache
   GPU miss → CPU miss → Disk miss → S3 HIT!
   ↓
3. Load from S3 → Disk → CPU → GPU
   (10-100ms latency, but NO prefill computation!)
   ↓
4. Skip prefill computation - huge time savings
   ↓
5. Transfer KV cache to Decode via NIXL
   ↓
6. Decode checks local cache
   - If cached locally: use local (fastest)
   - If miss: use incoming NIXL data
   ↓
7. Decode generates tokens
```

#### Scenario 3: Concurrent Request (Decode Node Already Has Cache)

```
1. Request arrives at Prefill
   ↓
2. Prefill loads from local tiers (GPU/CPU hit - fast)
   ↓
3. Prefill initiates NIXL transfer
   ↓
4. Decode receives request
   ↓
5. Decode checks local cache FIRST
   GPU HIT! (from previous request)
   ↓
6. Decode can use local cache, ignore/drop NIXL data
   OR merge if partial prefix match
   ↓
7. Decode generates tokens using local cache
```

#### Scenario 4: After Server Restart

```
1. Prefill/Decode nodes restart (local cache lost)
   ↓
2. New request arrives
   ↓
3. Prefill checks local cache: MISS (empty after restart)
   ↓
4. Prefill checks S3: HIT! (persistent storage survived restart)
   ↓
5. Load from S3 → restore to local tiers
   ↓
6. Continue with NIXL transfer to Decode
   ↓
7. Cache survived restart - no recomputation needed!
```

## Configuration Comparison Summary

| Feature | NIXL Only | NIXL + S3 |
|---------|-----------|-----------|
| **Setup Complexity** | Simple | Moderate |
| **Cache Scope** | Request-only | Global (all nodes) |
| **Persistence** | ❌ None | ✅ Durable |
| **Cross-Node Sharing** | ❌ No | ✅ Yes |
| **Survives Restart** | ❌ No | ✅ Yes |
| **Prefix Cache Hit Rate** | Low (~10-20%) | High (60-90%) |
| **Best For** | Testing, single-session | Production, multi-node |
| **S3 Costs** | None | Storage + API costs |
| **Cache Latency** | 1-10ms (NIXL) | 1-10ms (NIXL) + 10-100ms (S3) |
| **Compute Savings** | Minimal | 3-10× (LMCache claims) |

## LMCache Configuration Differences

### NIXL Only

```bash
# What's configured:
LMCACHE_ENABLE_PD=True              # Enable prefill-decode disagg
LMCACHE_PD_ROLE=sender/receiver     # Role in disaggregation
LMCACHE_TRANSFER_CHANNEL=nixl       # Use NIXL for transfer
LMCACHE_LOCAL_CPU=True              # Local CPU tier
LMCACHE_LOCAL_DISK=/tmp/lmcache     # Local disk tier

# What's NOT configured:
# LMCACHE_REMOTE_URL=...            # No S3 - omit this line
# AWS credentials                    # Not needed
```

**Cache hierarchy:** GPU → CPU → Disk (request-scoped only)

### NIXL + S3

```bash
# Everything from NIXL Only, PLUS:
LMCACHE_REMOTE_URL=s3://s3.cephlab.com/nixl/  # S3 backend
LMCACHE_S3_ENDPOINT_URL=http://s3.cephlab.com # Ceph S3 endpoint
AWS_ACCESS_KEY_ID=215QPBKECPTMTQ1MLZYW        # S3 credentials
AWS_SECRET_ACCESS_KEY=bvzEjZ4ePrisBSj3NOhB... # S3 credentials
AWS_DEFAULT_REGION=default                     # S3 region
```

**Cache hierarchy:** GPU → CPU → Disk → **S3** (globally shared, persistent)

## vLLM Configuration (Same for Both)

The vLLM configuration is **identical** whether using S3 or not:

```bash
# vLLM only knows about NIXL connector
--kv-transfer-config '{"kv_connector":"NixlConnector","kv_role":"kv_producer","kv_buffer_device":"cuda"}'
```

vLLM delegates to LMCache, which then:
- **Without S3:** Uses only local tiers
- **With S3:** Uses local tiers + S3 remote tier

## UCX Transport Configuration

For **RDMA** (InfiniBand):
```bash
UCX_TLS=rc,cuda_copy,cuda_ipc   # rc = reliable connection
# or
UCX_TLS=dc,cuda_copy,cuda_ipc   # dc = dynamically connected
```

For **TCP** (no RDMA):
```bash
UCX_TLS=tcp,cuda_copy,cuda_ipc
```

For **RoCE** (RDMA over Ethernet):
```bash
UCX_TLS=rc,cuda_copy,cuda_ipc
# Ensure RoCE is configured on NICs
```

## Performance Comparison

### NIXL Only - Expected Performance

**First Request:**
- Prefill: Full computation (e.g., 2 seconds for 8K tokens)
- NIXL transfer: 1-10ms
- Total TTFT: ~2-3 seconds

**Second Request (same prefix):**
- If local cache HIT: ~500ms-1s (faster)
- If local cache evicted: ~2-3 seconds (same as first - NO BENEFIT)

**Cache Hit Rate:** 10-20% (only if cached locally and not evicted)

### NIXL + S3 - Expected Performance

**First Request:**
- Prefill: Full computation (e.g., 2 seconds)
- NIXL transfer: 1-10ms
- S3 save (async): 50-200ms background
- Total TTFT: ~2-3 seconds

**Second Request (same prefix, different node or after eviction):**
- S3 fetch: 50-200ms
- Skip prefill computation: 0ms (HUGE SAVINGS)
- NIXL transfer: 1-10ms
- Total TTFT: ~200-500ms (4-10× faster!)

**Cache Hit Rate:** 60-90% (production workloads, chatbots even higher)

**Compute Savings:**
- 3-10× reduction in prefill time (LMCache benchmark)
- Each 10% cache hit increase = ~8-10% compute cost reduction

## Configuration Checklist

### For NIXL Only Setup

**vLLM Layer:**
- [ ] `--kv-transfer-config` with `NixlConnector` on both nodes
- [ ] `kv_role` set to `kv_producer` (prefill) and `kv_consumer` (decode)
- [ ] `--enable-prefix-caching` on both nodes

**LMCache PD Layer:**
- [ ] `LMCACHE_ENABLE_PD=True` on both nodes
- [ ] `LMCACHE_PD_ROLE=sender` on prefill
- [ ] `LMCACHE_PD_ROLE=receiver` on decode
- [ ] `LMCACHE_TRANSFER_CHANNEL=nixl` on both nodes

**LMCache Storage Layer (Local Only):**
- [ ] `LMCACHE_LOCAL_CPU=True` on both nodes
- [ ] `LMCACHE_LOCAL_DISK` configured on both nodes
- [ ] `LMCACHE_REMOTE_URL` NOT set (no S3)

**NIXL Transport Layer:**
- [ ] `UCX_TLS` configured for RDMA or TCP
- [ ] `VLLM_NIXL_SIDE_CHANNEL_HOST` and `_PORT` set
- [ ] Network connectivity verified

### For NIXL + S3 Setup

**All of the above, PLUS:**

**LMCache Storage Layer (with S3):**
- [ ] `LMCACHE_REMOTE_URL=s3://...` on both nodes (same bucket)
- [ ] `LMCACHE_S3_ENDPOINT_URL` on both nodes
- [ ] AWS credentials configured on both nodes
- [ ] S3 bucket created and accessible

**S3 Backend:**
- [ ] Ceph S3 gateway configured
- [ ] Bucket permissions verified
- [ ] Network connectivity to S3 endpoint verified
- [ ] Optional: S3 lifecycle policies configured

## When to Choose Each Configuration

### Choose NIXL Only If:
- ✅ Testing or development environment
- ✅ Single-session workloads (e.g., batch processing)
- ✅ No repeated prefix patterns
- ✅ Want to avoid S3 costs and complexity
- ✅ Don't care about cache persistence

### Choose NIXL + S3 If:
- ✅ Production deployment
- ✅ Interactive applications (chatbots, Q&A)
- ✅ High prefix reuse (system prompts, common questions)
- ✅ Multiple prefill/decode nodes
- ✅ Want to maximize cache hit rates
- ✅ Need cache to survive restarts
- ✅ ROI justifies S3 storage costs

**Rule of Thumb:** If you expect >30% of requests to share prefixes, S3 pays for itself in compute savings.

## Verification Steps

### Verify NIXL Only Setup

```bash
# 1. Check NIXL connection
docker logs vllm-prefill-mi300x | grep -i nixl
docker logs vllm-decode-gaudi3 | grep -i nixl

# 2. Send test request
curl -X POST http://<prefill-node-ip>:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"meta-llama/Llama-3.1-70b","prompt":"Test","max_tokens":10}'

# 3. Check logs for NIXL transfer
docker logs vllm-prefill-mi300x | grep -i "transfer\|nixl"

# 4. Send second request with same prefix
# Should see faster response if still in local cache
```

### Verify NIXL + S3 Setup

```bash
# 1. All NIXL checks from above

# 2. Check S3 bucket access
aws s3 ls s3://nixl/ \
  --endpoint-url http://s3.cephlab.com

# 3. Send test request
curl -X POST http://<prefill-node-ip>:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"meta-llama/Llama-3.1-70b","prompt":"Test","max_tokens":10}'

# 4. Verify S3 writes (should see new objects)
aws s3 ls s3://nixl/ --recursive \
  --endpoint-url http://s3.cephlab.com

# 5. Send second request with same prefix
# Should see cache hit from S3 in logs

# 6. Check LMCache stats (if monitoring enabled)
curl http://localhost:8080/stats
```

## Troubleshooting

### NIXL Connection Issues

```bash
# Verify UCX transport
docker exec vllm-prefill-mi300x env | grep UCX_TLS

# Check RDMA devices
ibstat
rdma link

# Test RDMA connectivity
ib_write_bw <prefill-node-ip>  # Run on decode node
```

### S3 Access Issues

```bash
# Verify credentials
aws s3 ls s3://nixl \
  --endpoint-url http://s3.cephlab.com

# Check bucket permissions
aws s3api get-bucket-policy \
  --bucket nixl \
  --endpoint-url http://s3.cephlab.com

# Monitor S3 operations
docker logs vllm-prefill-mi300x | grep -i "s3\|remote"
```

### Low Cache Hit Rate (with S3)

```bash
# Check if S3 writes are happening
aws s3 ls s3://nixl/ --recursive \
  --endpoint-url http://s3.cephlab.com

# Check LMCache logs
docker logs vllm-prefill-mi300x | grep -i "cache hit\|cache miss"

# Verify chunk size isn't too large
docker exec vllm-prefill-mi300x env | grep LMCACHE_CHUNK_SIZE
```

## Key Takeaways

1. **NIXL is always used** for real-time prefill→decode KV transfer (both configurations)
2. **S3 is optional** - adds persistent, cross-node cache sharing
3. **vLLM config is the same** - only LMCache env vars differ
4. **NIXL Only:** Simple, request-scoped, low cache hit rates
5. **NIXL + S3:** Production-ready, global cache, high hit rates, compute savings

## References

- [vLLM Documentation](https://docs.vllm.ai/)
- [LMCache Documentation](https://docs.lmcache.ai/)
- [LMCache Configuration Reference](https://docs.lmcache.ai/api_reference/configurations.html)
- [NIXL Connector Documentation](https://docs.vllm.ai/en/stable/features/nixl_connector_usage.html)
- [Mooncake Paper (FAST '25 Best Paper)](https://www.usenix.org/conference/fast25/presentation/qin)
- [UCX Documentation](https://openucx.readthedocs.io/)
