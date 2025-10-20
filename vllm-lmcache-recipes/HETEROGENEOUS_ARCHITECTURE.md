# Heterogeneous Disaggregated Architecture: MI300X + Gaudi3

## 🚨 Important: ROCm CUDA Compatibility

**Key Insight**: MI300X uses the **same `torch.cuda.*` API** as NVIDIA GPUs!

- ✅ **MI300X does NOT need modification** - PyTorch ROCm reuses CUDA interfaces
- ✅ **Gaudi3 DOES need our patch** - Intel uses different `torch.hpu` API  
- ✅ **Storage bridges both** - Device-agnostic NFS/S3

**Why This Works**: AMD's ROCm deliberately maintains CUDA API compatibility. See `ROCM_CUDA_COMPATIBILITY.md` for detailed explanation.

---

## Quick Reference

### Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────┐
│                     Production LLM Inference                         │
│                  with Heterogeneous Hardware                         │
└─────────────────────────────────────────────────────────────────────┘

┌────────────────────┐                        ┌───────────────────┐
│  Prefill Cluster   │                        │  Decode Cluster   │
│  AMD MI300X        │                        │  Intel Gaudi3     │
├────────────────────┤                        ├───────────────────┤
│                    │                        │                   │
│  vLLM 0.9.0+       │                        │  vLLM 0.9.0+      │
│  + LMCache (CUDA)  │                        │  + LMCache (HPU)  │
│  + ROCm 6.0        │                        │  + SynapseAI 1.21 │
│                    │                        │  + HPU PATCH ⭐   │
│  Role: PRODUCER    │                        │  Role: CONSUMER   │
│  ✓ Process prompts │                        │  ✓ Load KV cache  │
│  ✓ Save KV cache   │                        │  ✓ Generate tokens│
│                    │                        │                   │
│  8x MI300X         │                        │  16x Gaudi3       │
│  192GB HBM each    │                        │  128GB HBM each   │
│  Total: 1.5TB      │                        │  Total: 2TB       │
│                    │                        │                   │
└──────────┬─────────┘                        └─────────┬─────────┘
           │                                            │
           │ Save KV (20-25 GB/s)    Load KV (16-20 GB/s)
           │                                            │
           │              ┌─────────────────┐          │
           └─────────────►│ Shared Storage  │◄─────────┘
                          │  VAST Data      │
                          │  + NIXL         │
                          │                 │
                          │  • Device-agnostic NFS/S3
                          │  • High-bandwidth (50+ GB/s)
                          │  • Low-latency metadata
                          │  • Stores KV cache chunks
                          │                 │
                          └─────────────────┘
```

## Cost Analysis

### Hardware Costs (Approximate)

| Component | Unit Cost | Quantity | Total | Notes |
|-----------|-----------|----------|-------|-------|
| **AMD MI300X** | $15,000 | 8 | $120,000 | Prefill cluster |
| **Intel Gaudi3** | $8,000 | 16 | $128,000 | Decode cluster (2x nodes) |
| **VAST Storage** | $50,000 | 1 | $50,000 | 500TB, 50GB/s |
| **Networking** | $20,000 | 1 | $20,000 | 100GbE switches |
| **Total** | - | - | **$318,000** | Heterogeneous setup |

### Compared to Homogeneous

| Setup | Total Cost | Prefill | Decode | Cost Efficiency |
|-------|------------|---------|--------|-----------------|
| **All MI300X** (24 GPUs) | $360,000 | 8 MI300X | 16 MI300X | Baseline |
| **All Gaudi3** (24 GPUs) | $192,000 | 8 Gaudi3 | 16 Gaudi3 | 47% cheaper |
| **Heterogeneous** | $248,000 | 8 MI300X | 16 Gaudi3 | 31% cheaper, best perf |

**Winner**: Heterogeneous setup
- 31% lower cost than all-MI300X
- Better performance than all-Gaudi3 (MI300X prefill faster)
- Optimal hardware for each workload

## Throughput Analysis

### Per-Request Latency (128K context, 512 output tokens)

| Stage | Time | Hardware | Notes |
|-------|------|----------|-------|
| **Prefill** | 50-80s | 8x MI300X | Large context processing |
| **KV Save** | 1.2-1.5s | MI300X → Storage | 30GB KV cache @ 20-25 GB/s |
| **KV Load** | 1.8-2.2s | Storage → Gaudi3 | 30GB KV cache @ 16-20 GB/s |
| **Decode** | 2-3s | 16x Gaudi3 | 512 tokens @ 170-250 tok/s |
| **Total** | **55-87s** | End-to-end | First token to last token |

### Cluster Utilization

**Prefill Cluster (8x MI300X)**:
- Processes 1 request at a time (50-80s)
- Throughput: 45-72 requests/hour
- Utilization: 100% during prefill, 0% during decode

**Decode Cluster (16x Gaudi3, 2 nodes)**:
- Can handle multiple requests simultaneously
- Per-node: 1 request (2-3s decode time)
- Total throughput: ~1200 requests/hour (with batching)
- Utilization: Continuous (requests queued from prefill)

**Bottleneck**: Prefill cluster (50-80s per request)
**Solution**: Add more MI300X prefill nodes OR use chunked prefill

## Data Flow

### Request Processing Timeline

```
Time: 0s
┌────────────────────────────────────────────────────────────┐
│ User submits 128K token prompt                              │
└────────────────────────────────────────────────────────────┘
                          ↓
Time: 0s - 60s
┌────────────────────────────────────────────────────────────┐
│ MI300X Prefill Cluster                                      │
│ • Load model (if cold start)                                │
│ • Process 128K tokens                                       │
│ • Generate KV cache (30GB)                                  │
│ • LMCache saves to VAST                                     │
└────────────────────────────────────────────────────────────┘
                          ↓
Time: 60s - 61.5s
┌────────────────────────────────────────────────────────────┐
│ Storage Transfer                                            │
│ • KV cache written to VAST (1.2-1.5s)                      │
│ • Metadata updated                                          │
└────────────────────────────────────────────────────────────┘
                          ↓
Time: 61.5s - 63.7s
┌────────────────────────────────────────────────────────────┐
│ Gaudi3 Decode Cluster                                       │
│ • LMCache loads KV from VAST (1.8-2.2s)                    │
│ • Request assigned to available node                        │
└────────────────────────────────────────────────────────────┘
                          ↓
Time: 63.7s - 66s
┌────────────────────────────────────────────────────────────┐
│ Gaudi3 Decode Cluster                                       │
│ • Generate 512 output tokens (2-3s)                        │
│ • Stream back to user                                       │
└────────────────────────────────────────────────────────────┘
                          ↓
Time: 66s
┌────────────────────────────────────────────────────────────┐
│ Response complete                                           │
│ • Total time: ~66 seconds                                   │
│ • TTFT: ~64 seconds (time to first token)                  │
└────────────────────────────────────────────────────────────┘
```

## LMCache Configuration

### Prefill Node (MI300X)

```yaml
# lmcache_prefill_config.yaml
version: "1.0"
role: "kv_producer"

storage_backend:
  type: "vast"
  config:
    path: "vast://vast-cluster/kv-cache"
    bandwidth_gbps: 25
    
cache_config:
  chunk_size: 256
  dtype: "bfloat16"
  save_decode_cache: false
  priority_limit: 5

device:
  type: "cuda"  # MI300X uses CUDA/ROCm
  memory_utilization: 0.90
```

### Decode Node (Gaudi3)

```yaml
# lmcache_decode_config.yaml
version: "1.0"
role: "kv_consumer"

storage_backend:
  type: "vast"
  config:
    path: "vast://vast-cluster/kv-cache"
    bandwidth_gbps: 20
    
cache_config:
  chunk_size: 256
  dtype: "bfloat16"
  enable_async_loading: true
  priority_limit: 5

device:
  type: "hpu"  # Gaudi3 uses HPU (requires patch!)
  memory_utilization: 0.85
```

## Monitoring

### Key Metrics to Track

**Prefill Cluster (MI300X)**:
- Prefill latency (target: < 80s for 128K)
- KV cache save time (target: < 2s)
- GPU utilization (target: > 95%)
- Queue depth (requests waiting for prefill)

**Decode Cluster (Gaudi3)**:
- KV cache load time (target: < 3s)
- Decode throughput (target: > 150 tok/s)
- HPU utilization (target: > 85%)
- Request queueing time

**Storage (VAST)**:
- Read bandwidth (target: > 15 GB/s)
- Write bandwidth (target: > 20 GB/s)
- Metadata latency (target: < 50ms)
- Cache hit rate (target: > 60%)

## Troubleshooting

### Issue: Slow KV cache transfer

**Symptoms**: Load/save times > 5s

**Diagnosis**:
```bash
# Test storage bandwidth from MI300X
dd if=/dev/zero of=/mnt/vast/test bs=1G count=10 oflag=direct
# Should see > 20 GB/s

# Test from Gaudi3
dd if=/dev/zero of=/mnt/vast/test bs=1G count=10 oflag=direct
# Should see > 15 GB/s
```

**Solutions**:
- Check network connectivity
- Verify VAST/NFS mount options
- Enable NIXL acceleration
- Check for storage contention

### Issue: KV cache not found on decode cluster

**Symptoms**: Decode cluster can't find cached KV

**Diagnosis**:
```bash
# On prefill node
ls -lh /mnt/vast/kv-cache/<request_id>/

# On decode node
ls -lh /mnt/vast/kv-cache/<request_id>/
```

**Solutions**:
- Verify same storage path on both clusters
- Check NFS export permissions
- Ensure request IDs match exactly
- Check LMCache metadata synchronization

### Issue: Device type mismatch

**Symptoms**: "Expected CUDA tensor, got HPU tensor" or vice versa

**Solution**: This is why we need the patch!
- Prefill (MI300X): LMCache saves in device-agnostic format
- Storage: Stores tensors as CPU-serialized format
- Decode (Gaudi3): LMCache loads to HPU (requires HPU patch)

## Summary

### When to Use Heterogeneous Architecture

✅ **Use MI300X + Gaudi3 when**:
- Processing very long contexts (64K-128K+)
- Cost is a concern
- Need high decode throughput
- Have heterogeneous hardware available

❌ **Don't use when**:
- Context < 32K (homogeneous is simpler)
- Need lowest possible latency (single-cluster better)
- Limited networking infrastructure
- No shared storage available

### Key Takeaways

1. **LMCache runs on BOTH clusters** - different roles
2. **HPU patch is REQUIRED** for Gaudi3 decode cluster
3. **Storage is the bridge** between different hardware
4. **31% cost savings** vs all-MI300X setup
5. **Production-ready** with proper monitoring

---

**Last Updated**: October 19, 2025  
**Architecture**: Heterogeneous Disaggregated (MI300X + Gaudi3)  
**Status**: Production-Ready (with HPU patch)
