# vLLM Prefill-Decode Disaggregation Recipe with Heterogeneous Hardware

## Architecture Overview

This recipe sets up a **heterogeneous disaggregated prefill-decode architecture** using:
- **vLLM**: Serving engine for LLM inference
- **LMCache**: KV cache management system
- **NIXL**: High-performance transport layer for KV cache transfer
- **Ceph Storage**: Persistent storage backend for KV cache (S3 API or NFS over RDMA)
- **AMD MI300X**: Prefill nodes (memory-bandwidth optimized)
- **Intel Gaudi3**: Decode nodes (compute-efficient, cost-optimized)

### KVCache-Centric Disaggregated Architecture

This system implements a **KVCache-centric disaggregated architecture** that separates prefill and decoding operations around a centralized distributed KV cache system.

```
┌─────────────────────────────────────────────────────────────────────────────────────────┐
│                    KVCache-Centric Disaggregated Architecture                           │
│                          (MI300X Prefill + Gaudi3 Decode)                               │
└─────────────────────────────────────────────────────────────────────────────────────────┘

┌──────────────────────┐                                    ┌──────────────────────┐
│  KVCache-centric     │                                    │  Prefill Stage       │
│     Conductor        │                                    │  Optimization        │
│  (Global Scheduler)  │                                    │                      │
│                      │                                    │  Goal: Maximize      │
│ ┌──────────────────┐ │        ╔════════════════════╗      │   Cache Reuse        │
│ │ Cache-aware      │ │        ║  PREFILL POOL      ║      │                      │
│ │ Prefill Scheduler│◄┼───────►║  (AMD MI300X)      ║      │  Constraints:        │
│ └──────────────────┘ │        ║                    ║      │  • TTFT SLO          │
│                      │        ║ ┌────────────────┐ ║      │  • MFU > threshold   │
│ ┌──────────────────┐ │        ║ │ GPU/HBM3       │ ║      │  • KVCache < DRAM    │
│ │ KVCache Balance  │ │        ║ │ 192GB          │ ║      │                      │
│ │ Scheduler        │◄┼────┐   ║ │ - Paged Cache  │ ║      └──────────────────────┘
│ └──────────────────┘ │    │   ║ │ - Local Chunked│ ║
│                      │    │   ║ │   Prefill Sched│ ║
│ ┌──────────────────┐ │    │   ║ └────────┬───────┘ ║
│ │ Load-balance     │ │    │   ║          │         ║
│ │ Decoding         │◄┼───┐│   ║ ┌────────▼───────┐ ║
│ │ Scheduler        │ │   ││   ║ │ CPU/DRAM/SSD   │ ║
│ └──────────────────┘ │   ││   ║ │ Distributed    │ ║
│                      │   ││   ║ │ KVCache Pool   │ ║
└──────────────────────┘   ││   ║ └────────┬───────┘ ║
                           ││   ╚══════════╪═════════╝
                           ││              │
                           ││   ┌──────────▼─────────┐
                           ││   │  KVCache Transfer  │
                           │└──►│      Engine        │◄─── RDMA/NIXL
                           │    │  (Ceph NFS/RDMA)   │     High-speed
                           │    └──────────┬─────────┘     Transfer
                           │               │
                           │    ╔══════════╪═════════╗
                           │    ║ ┌────────▼───────┐ ║
                           │    ║ │ CPU/DRAM/SSD   │ ║
                           │    ║ │ Distributed    │ ║
                           │    ║ │ KVCache Pool   │ ║
                           │    ║ └────────┬───────┘ ║
                           │    ║          │         ║     ┌──────────────────────┐
                           │    ║ ┌────────▼───────┐ ║     │  Decode Stage        │
                           │    ║ │ GPU/HBM2e      │ ║     │  Optimization        │
                           │    ║ │ 128GB          │ ║     │                      │
                           └───►║ │ - Paged Cache  │ ║     │  Goal: Maximize      │
                                ║ │ - Local Decode │ ║     │   Throughput         │
                                ║ │   Scheduler    │ ║     │                      │
                                ║ └────────────────┘ ║     │  Constraints:        │
                                ║                    ║     │  • TBT SLO           │
                                ║  DECODING POOL     ║     │  • KVCache < VRAM    │
                                ║  (Intel Gaudi3)    ║     │                      │
                                ╚════════════════════╝     └──────────────────────┘

Data Flow:
  ━━━►  High-priority request path (RDMA/NIXL)
  ────► Control/scheduling signals

Key Innovation: "KVCaching is the key enabler for Efficient Decode"
  • Reduce from quadratic complexity to linear, move bottleneck from FLOPs to memory bw/capacity
  • Harness underutilized CPU, DRAM, SSD resources for distributed KV cache
  • Enable efficient near-GPU prefix caching across heterogeneous hardware
  • Significantly enhance global cache capacity and inter-node transfer bandwidth
  • Process 100B+ tokens daily across thousands of nodes (Mooncake production scale)
```

### Why Heterogeneous Hardware?

This architecture leverages the strengths of different hardware for different workloads:

**Prefill Phase (AMD MI300X):**
- **Memory-bound workload**: Prefill requires loading and processing long input sequences
- **MI300X advantages**:
  - 192GB HBM3 memory (2-2.4× larger than typical GPUs)
  - 5.3 TB/s memory bandwidth (excellent for attention operations)
  - Can handle longer context windows (128K+ tokens)
  - Better batch processing of large prompts
- **Cost justification**: Higher cost per node justified by massive throughput gains

**Decode Phase (Intel Gaudi3):**
- **Compute-bound workload**: Decode generates one token at a time with smaller KV cache access
- **Gaudi3 advantages**:
  - 128GB HBM2e sufficient for most decode scenarios
  - Excellent compute efficiency for autoregressive generation
  - Lower power consumption per token
  - Better cost per token for decode workloads
- **Cost optimization**: More cost-effective than premium GPUs for decode-only tasks

```
┌──────────────────────────────────┐          ┌──────────────────────────────────┐
│    Prefill Node (AMD MI300X)     │          │    Decode Node (Intel Gaudi3)    │
│  192GB HBM3, 5.3TB/s bandwidth   │          │   128GB HBM2e, 24 Tensor cores   │
│  ┌────────────────────────────┐  │          │  ┌────────────────────────────┐  │
│  │  vLLM (Producer, ROCm)     │  │          │  │  vLLM (Consumer, SynapseAI)│  │
│  └────────────┬───────────────┘  │          │  └────────────┬───────────────┘  │
│               │                  │          │               │                  │
│  ┌────────────▼───────────────┐  │◄─ NIXL ─►│  ┌────────────▼───────────────┐  │
│  │  LMCache L1 (MI300X mem)   │  │   RDMA   │  │  LMCache L1 (Gaudi3 mem)   │  │
│  │  Capacity: ~180GB usable   │  │          │  │  Capacity: ~120GB usable   │  │
│  └────────────┬───────────────┘  │          │  └────────────┬───────────────┘  │
│               │                  │          │               │                  │
│  ┌────────────▼───────────────┐  │          │  ┌────────────▼───────────────┐  │
│  │   LMCache L1 (CPU RAM)     │  │          │  │   LMCache L1 (CPU RAM)     │  │
│  └────────────┬───────────────┘  │          │  └────────────┬───────────────┘  │
│               │                  │          │               │                  │
│  ┌────────────▼───────────────┐  │          │  ┌────────────▼───────────────┐  │
│  │  Local Disk (NVMe, opt.)   │  │          │  │  Local Disk (NVMe, opt.)   │  │
│  └────────────┬───────────────┘  │          │  └────────────┬───────────────┘  │
└───────────────┼──────────────────┘          └───────────────┼──────────────────┘
                │                                             │
                └────────────────────┬────────────────────────┘
                                     │
                     ┌───────────────▼────────────────┐
                     │  Ceph Storage (L2 - Remote)    │
                     │  ┌──────────────────────────┐  │
                     │  │ Option A: S3 API         │  │
                     │  └──────────────────────────┘  │
                     │  ┌──────────────────────────┐  │
                     │  │ Option B: NFS/RDMA       │  │
                     │  └──────────────────────────┘  │
                     └────────────────────────────────┘

Cache Hierarchy:
  L1: Device memory (MI300X: ~180GB, Gaudi3: ~120GB)
  L1: CPU memory (fast, ~10-100GB, LRU eviction)
  L1: Local disk (medium, ~TB, optional)
  L2: Remote storage (Ceph S3 or NFS/RDMA, unlimited, persistent)
      - S3 API: Simple, object storage, 10-100ms latency
      - NFS/RDMA: High performance, filesystem, 1-5ms latency

Hardware Specifications:
  Prefill (MI300X):  8 GCDs, 192GB HBM3, 5.3TB/s, ROCm stack
  Decode (Gaudi3):   24 Tensor cores, 128GB HBM2e, SynapseAI stack
```

## Prerequisites

### Hardware Requirements

**Prefill Nodes:**
- **AMD MI300X** accelerators (192GB HBM3)
- Minimum: 1x MI300X per node
- Recommended: 4-8x MI300X for tensor parallelism
- Host: 256GB+ RAM, 2TB+ NVMe SSD
- RDMA-capable NICs (for NIXL and NFS/RDMA)

**Decode Nodes:**
- **Intel Gaudi3** accelerators (128GB HBM2e)
- Minimum: 1x Gaudi3 per node
- Recommended: 4-8x Gaudi3 for tensor parallelism
- Host: 128GB+ RAM, 1TB+ NVMe SSD
- RDMA-capable NICs (for NIXL and NFS/RDMA)

### Software Requirements

**Prefill Nodes (MI300X):**
- Docker with AMD ROCm support
- ROCm 6.0+ (included in vLLM Docker image)
- Python 3.10+
- Ceph cluster with S3 gateway OR CephFS with NFS-Ganesha

**Decode Nodes (Gaudi3):**
- Docker with habanalabs-container-runtime
- Intel Gaudi Software Suite (SynapseAI) 1.18+
- Python 3.10+
- Ceph cluster access (same as prefill nodes)

### Network Requirements

- **High-bandwidth network**: 100Gbps+ between prefill and decode nodes
- **RDMA support**: InfiniBand or RoCE (recommended for NIXL and NFS/RDMA)
- **Low latency**: <10μs for best NIXL performance
- **Connectivity**: All nodes must reach Ceph storage backend

## Step 1: Setup Ceph Storage Backend

You can use either **S3 API** (object storage) or **NFS over RDMA** (filesystem) to access Ceph storage. NFS over RDMA provides lower latency and higher throughput for KV cache operations.

### Option A: Ceph S3 Backend

#### 1.1 Configure Ceph S3 Access

```bash
# Set environment variables for S3 access
export S3_ENDPOINT_URL="https://your-ceph-s3-endpoint.com"
export AWS_ACCESS_KEY_ID="your-access-key"
export AWS_SECRET_ACCESS_KEY="your-secret-key"
export S3_BUCKET_NAME="vllm-kv-cache"
```

#### 1.2 Create S3 Bucket for KV Cache

```bash
# Install AWS CLI or s3cmd
pip install awscli

# Configure AWS CLI for Ceph
aws configure set aws_access_key_id $AWS_ACCESS_KEY_ID
aws configure set aws_secret_access_key $AWS_SECRET_ACCESS_KEY
aws configure set default.region us-east-1

# Create bucket
aws s3 mb s3://${S3_BUCKET_NAME} --endpoint-url ${S3_ENDPOINT_URL}
```

### Option B: CephFS with NFS over RDMA (Recommended for Performance)

NFS over RDMA provides significantly better performance than S3 API for KV cache workloads:
- **Lower latency**: 1-5ms vs 10-100ms for S3
- **Higher throughput**: 10-50 GB/s vs 100 MB-1 GB/s for S3
- **Better IOPS**: Direct filesystem access without REST API overhead

#### 1.1 Prerequisites for NFS over RDMA

On Ceph cluster:
- CephFS filesystem configured
- NFS-Ganesha v4.0+ installed with RDMA support
- RDMA-capable NICs (Mellanox ConnectX-5 or newer)

On compute nodes:
- RDMA drivers installed (MLNX_OFED or inbox drivers)
- NFS client with RDMA support

#### 1.2 Configure NFS-Ganesha on Ceph

On your Ceph cluster, configure NFS-Ganesha to export CephFS:

```bash
# Create CephFS filesystem (if not already exists)
ceph fs volume create cephfs

# Deploy NFS-Ganesha service
ceph nfs cluster create nfs-ganesha

# Create NFS export for KV cache
ceph nfs export create cephfs \
  --cluster-id nfs-ganesha \
  --pseudo-path /kv-cache \
  --path /volumes/kv-cache \
  --client-addr 192.168.1.0/24

# Enable RDMA transport (if RDMA NICs available)
ceph config set nfs-ganesha.nfs-ganesha NFS_CORE_PARAM RDMA_CLIENT_SUPPORT=true
```

#### 1.3 Mount NFS with RDMA on Compute Nodes

On each prefill/decode node:

```bash
# Install NFS client with RDMA support
apt-get update && apt-get install -y nfs-common

# Load RDMA modules
modprobe rdma_cm ib_core mlx5_core mlx5_ib

# Create mount point
mkdir -p /mnt/ceph-nfs-kv-cache

# Mount with RDMA transport
mount -t nfs -o \
  rdma,port=20049,vers=4.2,proto=rdma \
  nfs-ganesha-server-ip:/kv-cache \
  /mnt/ceph-nfs-kv-cache

# Verify RDMA mount
mount | grep rdma
# Should show: nfs-ganesha-server-ip:/kv-cache on /mnt/ceph-nfs-kv-cache type nfs4 (rdma,...)

# Set environment variable for LMCache
export CEPH_NFS_PATH="/mnt/ceph-nfs-kv-cache"
```

#### 1.4 Persistent NFS Mount (Optional)

Add to `/etc/fstab` for automatic mounting on boot:

```bash
cat >> /etc/fstab <<EOF
nfs-ganesha-server-ip:/kv-cache  /mnt/ceph-nfs-kv-cache  nfs4  rdma,port=20049,vers=4.2,proto=rdma,_netdev  0 0
EOF
```

#### 1.5 Performance Tuning for NFS over RDMA

Optimize NFS mount options for KV cache workloads:

```bash
# Remount with performance tunings
mount -t nfs -o \
  rdma,port=20049,vers=4.2,proto=rdma,\
  rsize=1048576,wsize=1048576,\
  timeo=600,retrans=2,\
  hard,intr,\
  noatime,nodiratime,\
  async \
  nfs-ganesha-server-ip:/kv-cache \
  /mnt/ceph-nfs-kv-cache
```

**Mount options explained:**
- `rdma,proto=rdma`: Enable RDMA transport
- `rsize=1048576,wsize=1048576`: 1MB read/write buffer sizes
- `hard,intr`: Ensure reliability, allow interrupts
- `noatime,nodiratime`: Disable access time updates for performance
- `async`: Asynchronous writes (higher throughput, slightly lower durability)

#### 1.6 Verify NFS over RDMA Performance

Test RDMA connectivity and performance:

```bash
# Check RDMA devices
ibstat
rdma link

# Test sequential write performance
dd if=/dev/zero of=/mnt/ceph-nfs-kv-cache/test.dat bs=1M count=1000 oflag=direct
# Expected: >5 GB/s with RDMA

# Test sequential read performance
dd if=/mnt/ceph-nfs-kv-cache/test.dat of=/dev/null bs=1M iflag=direct
# Expected: >10 GB/s with RDMA

# Test random IOPS
fio --name=randrw --ioengine=libaio --rw=randrw --bs=4k --direct=1 \
    --size=1G --numjobs=4 --runtime=60 --group_reporting \
    --directory=/mnt/ceph-nfs-kv-cache
# Expected: >50k IOPS with RDMA

# Clean up
rm /mnt/ceph-nfs-kv-cache/test.dat
```

## Step 2: Install vLLM with Docker

### 2.1 Pull vLLM Docker Images

**For Prefill Nodes (AMD MI300X with ROCm):**
```bash
# vLLM with ROCm support for MI300X
docker pull rocm/vllm:latest
# OR build from source with ROCm 6.0+
```

**For Decode Nodes (Intel Gaudi3):**
```bash
# Habana base image with PyTorch
docker pull vault.habana.ai/gaudi-docker/1.18.0/ubuntu22.04/habanalabs/pytorch-installer-2.5.0:latest
```

### 2.2 Build Custom Images with LMCache and NIXL

**Prefill Dockerfile (MI300X + ROCm):**

```dockerfile
# Dockerfile.vllm-prefill-mi300x
FROM rocm/vllm:latest

# Install LMCache
RUN pip install lmcache

# Install NIXL connector (if available for ROCm)
RUN pip install nixl-connector || echo "NIXL may require manual setup for ROCm"

# Install S3 dependencies
RUN pip install boto3 botocore

# ROCm-specific optimizations
ENV HSA_FORCE_FINE_GRAIN_PCIE=1
ENV ROCM_HOME=/opt/rocm
ENV HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

WORKDIR /workspace
```

**Decode Dockerfile (Gaudi3 + SynapseAI):**

```dockerfile
# Dockerfile.vllm-decode-gaudi3
FROM vault.habana.ai/gaudi-docker/1.18.0/ubuntu22.04/habanalabs/pytorch-installer-2.5.0:latest

# Install vLLM with Gaudi support
RUN pip install vllm --extra-index-url https://download.habana.ai/

# Install LMCache
RUN pip install lmcache

# Install NIXL connector
RUN pip install nixl-connector

# Install S3 dependencies
RUN pip install boto3 botocore

# Gaudi-specific environment
ENV HABANA_VISIBLE_DEVICES=all
ENV PT_HPU_LAZY_MODE=1

WORKDIR /workspace
```

### 2.3 Build the Images

**Build prefill image:**
```bash
docker build -f Dockerfile.vllm-prefill-mi300x -t vllm-prefill-mi300x:latest .
```

**Build decode image:**
```bash
docker build -f Dockerfile.vllm-decode-gaudi3 -t vllm-decode-gaudi3:latest .
```

### 2.4 Verify Hardware Access

**Verify MI300X access:**
```bash
docker run --rm --device=/dev/kfd --device=/dev/dri \
  --group-add video --cap-add=SYS_PTRACE --security-opt seccomp=unconfined \
  vllm-prefill-mi300x:latest rocm-smi
```

**Verify Gaudi3 access:**
```bash
docker run --rm --runtime=habana -e HABANA_VISIBLE_DEVICES=all \
  vllm-decode-gaudi3:latest hl-smi
```

## Step 3: Configure Prefill Node (AMD MI300X)

### 3.1 Create Prefill Configuration

Create `prefill_config.json` for MI300X:

```json
{
  "model": "meta-llama/Llama-3.1-70b",
  "tensor_parallel_size": 4,
  "pipeline_parallel_size": 1,
  "max_num_seqs": 512,
  "max_model_len": 8192,
  "gpu_memory_utilization": 0.90,
  "enable_chunked_prefill": true,
  "enable_prefix_caching": true,
  "kv_connector": "NixlConnector",
  "kv_role": "producer",
  "kv_rank": 0,
  "kv_parallel_size": 1,
  "nixl_transport": "rdma",
  "nixl_port": 50051,
  "kv_buffer_size": "20GB",
  "kv_cache_backend": "s3",
  "s3_endpoint_url": "${S3_ENDPOINT_URL}",
  "s3_bucket_name": "${S3_BUCKET_NAME}",
  "s3_prefix": "kv-cache/"
}
```

**Configuration Notes for MI300X:**
- `max_num_seqs: 512`: Higher batch size leverages 192GB memory
- `max_model_len: 8192`: Longer context enabled by large memory bandwidth
- `gpu_memory_utilization: 0.90`: Conservative due to 192GB capacity
- `kv_buffer_size: 20GB`: Larger buffer for high-throughput KV transfer

### 3.2 Launch Prefill Instance (MI300X)

```bash
# Start prefill node on MI300X
docker run -d \
  --name vllm-prefill-mi300x \
  --device=/dev/kfd \
  --device=/dev/dri \
  --group-add video \
  --cap-add=SYS_PTRACE \
  --security-opt seccomp=unconfined \
  --net=host \
  --ipc=host \
  -e HSA_FORCE_FINE_GRAIN_PCIE=1 \
  -e ROCM_HOME=/opt/rocm \
  -e HIP_VISIBLE_DEVICES=0,1,2,3 \
  -e AWS_ACCESS_KEY_ID=${AWS_ACCESS_KEY_ID} \
  -e AWS_SECRET_ACCESS_KEY=${AWS_SECRET_ACCESS_KEY} \
  -e S3_ENDPOINT_URL=${S3_ENDPOINT_URL} \
  -e S3_BUCKET_NAME=${S3_BUCKET_NAME} \
  -v $(pwd)/prefill_config.json:/workspace/config.json \
  vllm-prefill-mi300x:latest \
  python -m vllm.entrypoints.openai.api_server \
    --model meta-llama/Llama-3.1-70b \
    --tensor-parallel-size 4 \
    --max-num-seqs 512 \
    --max-model-len 8192 \
    --gpu-memory-utilization 0.90 \
    --enable-chunked-prefill \
    --enable-prefix-caching \
    --kv-connector NixlConnector \
    --kv-role producer \
    --kv-rank 0 \
    --kv-parallel-size 1 \
    --nixl-transport rdma \
    --nixl-port 50051 \
    --nixl-proxy-host prefill-node-ip \
    --kv-buffer-size 20GB \
    --port 8000 \
    --trust-remote-code
```

**MI300X-Specific Flags:**
- `--device=/dev/kfd --device=/dev/dri`: ROCm device access
- `--group-add video`: Required for ROCm
- `HSA_FORCE_FINE_GRAIN_PCIE=1`: Optimize PCIe transfers
- `HIP_VISIBLE_DEVICES`: Control which MI300X GPUs are visible

## Step 4: Configure Decode Node (Intel Gaudi3)

### 4.1 Create Decode Configuration

Create `decode_config.json` for Gaudi3:

```json
{
  "model": "meta-llama/Llama-3.1-70b",
  "tensor_parallel_size": 4,
  "pipeline_parallel_size": 1,
  "max_num_seqs": 256,
  "max_model_len": 8192,
  "gpu_memory_utilization": 0.85,
  "kv_connector": "NixlConnector",
  "kv_role": "consumer",
  "kv_rank": 0,
  "kv_parallel_size": 1,
  "nixl_transport": "rdma",
  "nixl_port": 50052,
  "nixl_proxy_host": "prefill-node-ip:50051",
  "kv_buffer_size": "15GB",
  "kv_cache_backend": "s3",
  "s3_endpoint_url": "${S3_ENDPOINT_URL}",
  "s3_bucket_name": "${S3_BUCKET_NAME}",
  "s3_prefix": "kv-cache/"
}
```

**Configuration Notes for Gaudi3:**
- `max_num_seqs: 256`: Moderate batch size for decode workload
- `max_model_len: 8192`: Match prefill context length
- `gpu_memory_utilization: 0.85`: Conservative for 128GB Gaudi3
- `kv_buffer_size: 15GB`: Sufficient for decode throughput

### 4.2 Launch Decode Instance (Gaudi3)

```bash
# Start decode node on Gaudi3
docker run -d \
  --name vllm-decode-gaudi3 \
  --runtime=habana \
  -e HABANA_VISIBLE_DEVICES=all \
  -e PT_HPU_LAZY_MODE=1 \
  -e OMPI_MCA_btl_vader_single_copy_mechanism=none \
  --cap-add=sys_nice \
  --net=host \
  --ipc=host \
  -e AWS_ACCESS_KEY_ID=${AWS_ACCESS_KEY_ID} \
  -e AWS_SECRET_ACCESS_KEY=${AWS_SECRET_ACCESS_KEY} \
  -e S3_ENDPOINT_URL=${S3_ENDPOINT_URL} \
  -e S3_BUCKET_NAME=${S3_BUCKET_NAME} \
  -v $(pwd)/decode_config.json:/workspace/config.json \
  vllm-decode-gaudi3:latest \
  python -m vllm.entrypoints.openai.api_server \
    --model meta-llama/Llama-3.1-70b \
    --tensor-parallel-size 4 \
    --max-num-seqs 256 \
    --max-model-len 8192 \
    --gpu-memory-utilization 0.85 \
    --kv-connector NixlConnector \
    --kv-role consumer \
    --kv-rank 0 \
    --kv-parallel-size 1 \
    --nixl-transport rdma \
    --nixl-port 50052 \
    --nixl-proxy-host prefill-node-ip:50051 \
    --kv-buffer-size 15GB \
    --port 8001 \
    --trust-remote-code
```

**Gaudi3-Specific Flags:**
- `--runtime=habana`: Habana container runtime
- `PT_HPU_LAZY_MODE=1`: Enable lazy mode for better performance
- `HABANA_VISIBLE_DEVICES=all`: Expose all Gaudi3 accelerators
- `OMPI_MCA_btl_vader_single_copy_mechanism=none`: Optimize MPI for Gaudi

## Step 5: Configure LMCache with S3 Backend

### 5.1 LMCache Configuration via Environment Variables

LMCache can be configured using environment variables (recommended) or YAML config files. Below are the key configuration options:

#### Essential Configuration

Add these environment variables to both prefill and decode Docker commands:

```bash
# Core cache settings
-e LMCACHE_CHUNK_SIZE=256 \
-e LMCACHE_LOCAL_CPU=True \
-e LMCACHE_MAX_LOCAL_CPU_SIZE=10.0 \
-e LMCACHE_LOCAL_DISK=/tmp/lmcache \
-e LMCACHE_MAX_LOCAL_DISK_SIZE=50.0 \
-e LMCACHE_CACHE_POLICY=LRU \

# Remote backend configuration (choose one)

# Option A: S3 backend
-e LMCACHE_REMOTE_URL=s3://${S3_BUCKET_NAME}/kv-cache/ \
-e LMCACHE_S3_ENDPOINT_URL=${S3_ENDPOINT_URL} \
-e AWS_ACCESS_KEY_ID=${AWS_ACCESS_KEY_ID} \
-e AWS_SECRET_ACCESS_KEY=${AWS_SECRET_ACCESS_KEY} \

# Option B: NFS over RDMA backend (higher performance)
-e LMCACHE_REMOTE_URL=file:///mnt/ceph-nfs-kv-cache \
-v /mnt/ceph-nfs-kv-cache:/mnt/ceph-nfs-kv-cache:rw \  # Mount NFS path into container

# Performance optimizations
-e LMCACHE_SAVE_DECODE_CACHE=False \
-e LMCACHE_USE_LAYERWISE=False \

# Disaggregated prefill (NIXL)
-e LMCACHE_ENABLE_PD=True \
-e LMCACHE_PD_ROLE=sender \    # or "receiver" for decode nodes
-e LMCACHE_TRANSFER_CHANNEL=nixl \
```

**Note**: When using NFS over RDMA, ensure the NFS mount is accessible from within the Docker container by using `-v` to bind mount the path.

#### Advanced Configuration Options

```bash
# Blending configuration (for cache blending optimization)
-e LMCACHE_ENABLE_BLENDING=False \
-e LMCACHE_BLEND_RECOMPUTE_RATIOS=0.15 \

# P2P sharing (for multi-node cache sharing)
-e LMCACHE_ENABLE_P2P=True \
-e LMCACHE_LOOKUP_URL=http://lookup-server:8080 \

# Priority-based caching
-e LMCACHE_PRIORITY_LIMIT=100 \

# Internal API server (for runtime monitoring)
-e LMCACHE_INTERNAL_API_SERVER_ENABLED=True \

# NUMA-aware memory allocation
-e LMCACHE_NUMA_MODE=True \
```

### 5.2 Alternative: YAML Configuration File

Create `lmcache_config.yaml` (optional, if not using environment variables):

```yaml
# Core cache configuration
chunk_size: 256
local_cpu: true
max_local_cpu_size: 10.0  # GB
local_disk: /tmp/lmcache
max_local_disk_size: 50.0  # GB
cache_policy: LRU  # Options: LRU, LFU, FIFO

# Remote storage (S3)
remote_url: s3://vllm-kv-cache/kv-cache/

# Performance settings
save_decode_cache: false
use_layerwise: false

# Disaggregated prefill
enable_pd: true
pd_role: sender  # or "receiver"
transfer_channel: nixl

# Advanced features
enable_blending: false
blend_recompute_ratios: 0.15
enable_p2p: false
priority_limit: 100

# Monitoring
internal_api_server_enabled: true
```

Mount the config file:
```bash
-v $(pwd)/lmcache_config.yaml:/workspace/lmcache_config.yaml \
-e LMCACHE_CONFIG_PATH=/workspace/lmcache_config.yaml
```

### 5.3 Cache Tier Management

LMCache automatically manages cache across multiple tiers:

**Tier Hierarchy:**
1. **GPU Memory (L1)**: Fastest, managed by vLLM
2. **CPU Memory (L1)**: Intermediate tier, acts as prefetch cache
   - LRU eviction when full
   - Configurable via `LMCACHE_MAX_LOCAL_CPU_SIZE`
3. **Local Disk (L1)**: Optional local cache tier
   - Larger capacity than CPU memory
   - Configurable via `LMCACHE_MAX_LOCAL_DISK_SIZE`
4. **Remote Storage (L2)**: S3/shared storage, persistent
   - Unlimited capacity
   - Highest latency but durable

**Promotion/Demotion Flow:**
- **GPU → CPU**: Automatic when GPU cache fills
- **CPU → Disk**: LRU eviction when CPU cache reaches max size
- **Disk → S3**: Background offloading for persistence
- **S3 → CPU → GPU**: Automatic prefetch when cache hit detected

## Step 6: Setup Load Balancer (Optional)

For multiple prefill/decode instances, configure a load balancer:

### 6.1 Using NGINX

Create `nginx.conf`:

```nginx
upstream prefill_backend {
    least_conn;
    server prefill-node-1:8000;
    server prefill-node-2:8000;
}

upstream decode_backend {
    least_conn;
    server decode-node-1:8001;
    server decode-node-2:8001;
}

server {
    listen 80;
    server_name vllm-lb;

    location /v1/completions {
        proxy_pass http://prefill_backend;
        proxy_http_version 1.1;
        proxy_set_header Connection "";
        proxy_buffering off;
        proxy_read_timeout 3600s;
    }

    location /v1/chat/completions {
        proxy_pass http://decode_backend;
        proxy_http_version 1.1;
        proxy_set_header Connection "";
        proxy_buffering off;
        proxy_read_timeout 3600s;
    }
}
```

## Step 7: Verification and Testing

### 7.1 Check Prefill Node

```bash
# Check prefill node health
curl http://prefill-node-ip:8000/health

# Check NIXL connection
docker logs vllm-prefill | grep -i nixl
```

### 7.2 Check Decode Node

```bash
# Check decode node health
curl http://decode-node-ip:8001/health

# Check NIXL connection
docker logs vllm-decode | grep -i nixl
```

### 7.3 Verify S3 Backend

```bash
# Check if KV cache is being stored in S3
aws s3 ls s3://${S3_BUCKET_NAME}/kv-cache/ --endpoint-url ${S3_ENDPOINT_URL}
```

### 7.4 Test End-to-End

```bash
# Send test request
curl -X POST http://prefill-node-ip:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "meta-llama/Llama-3-70b-hf",
    "prompt": "Explain quantum computing in simple terms:",
    "max_tokens": 100,
    "temperature": 0.7
  }'
```

## Step 8: Multi-Instance Deployment

### 8.1 Deploy Multiple Prefill Instances

```bash
# Launch prefill instance 1
docker run -d --name vllm-prefill-1 \
  --runtime=habana \
  -e HABANA_VISIBLE_DEVICES=0,1,2,3 \
  --net=host \
  -e AWS_ACCESS_KEY_ID=${AWS_ACCESS_KEY_ID} \
  -e AWS_SECRET_ACCESS_KEY=${AWS_SECRET_ACCESS_KEY} \
  -e S3_ENDPOINT_URL=${S3_ENDPOINT_URL} \
  vllm-disagg:latest \
  python -m vllm.entrypoints.openai.api_server \
    --model meta-llama/Llama-3-70b-hf \
    --tensor-parallel-size 4 \
    --kv-connector NixlConnector \
    --kv-role producer \
    --kv-rank 0 \
    --nixl-port 50051 \
    --port 8000

# Launch prefill instance 2
docker run -d --name vllm-prefill-2 \
  --runtime=habana \
  -e HABANA_VISIBLE_DEVICES=4,5,6,7 \
  --net=host \
  -e AWS_ACCESS_KEY_ID=${AWS_ACCESS_KEY_ID} \
  -e AWS_SECRET_ACCESS_KEY=${AWS_SECRET_ACCESS_KEY} \
  -e S3_ENDPOINT_URL=${S3_ENDPOINT_URL} \
  vllm-disagg:latest \
  python -m vllm.entrypoints.openai.api_server \
    --model meta-llama/Llama-3-70b-hf \
    --tensor-parallel-size 4 \
    --kv-connector NixlConnector \
    --kv-role producer \
    --kv-rank 1 \
    --nixl-port 50053 \
    --port 8002
```

### 8.2 Deploy Multiple Decode Instances

```bash
# Launch decode instance 1
docker run -d --name vllm-decode-1 \
  --runtime=habana \
  -e HABANA_VISIBLE_DEVICES=0,1,2,3 \
  --net=host \
  -e AWS_ACCESS_KEY_ID=${AWS_ACCESS_KEY_ID} \
  -e AWS_SECRET_ACCESS_KEY=${AWS_SECRET_ACCESS_KEY} \
  -e S3_ENDPOINT_URL=${S3_ENDPOINT_URL} \
  vllm-disagg:latest \
  python -m vllm.entrypoints.openai.api_server \
    --model meta-llama/Llama-3-70b-hf \
    --tensor-parallel-size 4 \
    --kv-connector NixlConnector \
    --kv-role consumer \
    --kv-rank 0 \
    --nixl-proxy-host prefill-node-ip:50051,prefill-node-ip:50053 \
    --port 8001

# Launch decode instance 2
docker run -d --name vllm-decode-2 \
  --runtime=habana \
  -e HABANA_VISIBLE_DEVICES=4,5,6,7 \
  --net=host \
  -e AWS_ACCESS_KEY_ID=${AWS_ACCESS_KEY_ID} \
  -e AWS_SECRET_ACCESS_KEY=${AWS_SECRET_ACCESS_KEY} \
  -e S3_ENDPOINT_URL=${S3_ENDPOINT_URL} \
  vllm-disagg:latest \
  python -m vllm.entrypoints.openai.api_server \
    --model meta-llama/Llama-3-70b-hf \
    --tensor-parallel-size 4 \
    --kv-connector NixlConnector \
    --kv-role consumer \
    --kv-rank 1 \
    --nixl-proxy-host prefill-node-ip:50051,prefill-node-ip:50053 \
    --port 8003
```

## Monitoring and Troubleshooting

### Monitor Docker Logs

```bash
# Prefill logs
docker logs -f vllm-prefill

# Decode logs
docker logs -f vllm-decode

# Filter for LMCache events
docker logs vllm-prefill 2>&1 | grep -i "lmcache\|cache hit\|cache miss"
```

### Monitor LMCache Internal API

If `LMCACHE_INTERNAL_API_SERVER_ENABLED=True`, you can query cache statistics:

```bash
# Get cache statistics
curl http://localhost:8080/stats

# Get cache health
curl http://localhost:8080/health

# Inspect cache contents
curl http://localhost:8080/cache/list

# Get specific cache entry
curl http://localhost:8080/cache/get?key=<cache_key>
```

### Monitor Cache Hit Rates

```bash
# Monitor vLLM metrics (if Prometheus is enabled)
curl http://localhost:8000/metrics | grep cache

# Expected metrics:
# - vllm:cache_hit_rate
# - vllm:cache_miss_count
# - vllm:prefill_time_saved
# - lmcache:gpu_cache_utilization
# - lmcache:cpu_cache_utilization
# - lmcache:s3_cache_size
```

### Monitor S3 Storage

```bash
# Check S3 usage
aws s3 ls s3://${S3_BUCKET_NAME}/kv-cache/ --recursive --human-readable --summarize --endpoint-url ${S3_ENDPOINT_URL}

# Monitor S3 request metrics (if CloudWatch/Ceph monitoring enabled)
# - GetObject count (cache reads)
# - PutObject count (cache writes)
# - Transfer bandwidth
```

### Monitor NIXL Performance

```bash
# Check NIXL transfer stats
docker exec vllm-prefill cat /proc/net/nixl/stats
docker exec vllm-decode cat /proc/net/nixl/stats

# Expected output:
# - Bytes transferred
# - Transfer rate (GB/s)
# - Connection status
# - Error count
```

### Monitor Cache Tier Utilization

```bash
# GPU memory (via nvidia-smi or hl-smi for Gaudi)
nvidia-smi  # or hl-smi

# CPU memory usage by LMCache
docker exec vllm-prefill ps aux | grep vllm
docker stats vllm-prefill

# Disk usage (if local disk cache enabled)
docker exec vllm-prefill df -h /tmp/lmcache
```

### Performance Metrics to Track

1. **Time to First Token (TTFT)**
   - Target: <500ms for cached prefixes
   - Without cache: 1-5 seconds for long contexts

2. **Cache Hit Rate**
   - Target: >60% for production workloads
   - Varies by use case (chatbots: 70-90%, general: 40-60%)

3. **Prefill Time Savings**
   - LMCache claims 3-10x reduction
   - Measure: prefill_time_with_cache / prefill_time_without_cache

4. **Network Bandwidth (NIXL)**
   - Target: >50 GB/s with RDMA
   - Monitor saturation during peak load

### Common Issues

1. **NIXL Connection Failed**
   - Check network connectivity: `ping prefill-node-ip`
   - Verify firewall rules allow RDMA/TCP on ports 50051-50053
   - Check NIXL logs: `docker logs vllm-prefill 2>&1 | grep -i nixl`

2. **S3 Access Denied**
   - Verify AWS credentials: `aws s3 ls s3://${S3_BUCKET_NAME} --endpoint-url ${S3_ENDPOINT_URL}`
   - Check bucket permissions and IAM policies
   - Confirm endpoint URL is correct

3. **KV Cache Miss (Low Hit Rate)**
   - Check LMCache configuration: chunk_size might be too large/small
   - Verify S3 backend is reachable
   - Check if cache warming is needed for cold start
   - Review cache eviction policy (LRU vs LFU vs FIFO)

4. **High Latency**
   - Monitor NIXL buffer size: increase `--kv-buffer-size` if network is underutilized
   - Check network bandwidth: use `iperf3` between nodes
   - Review CPU cache size: increase `LMCACHE_MAX_LOCAL_CPU_SIZE` if evicting too frequently
   - Consider using RDMA instead of TCP for NIXL transport

5. **Out of Memory (OOM)**
   - Reduce `gpu_memory_utilization` from 0.9 to 0.8
   - Decrease `LMCACHE_MAX_LOCAL_CPU_SIZE`
   - Enable CPU offloading if not already enabled
   - Check for memory leaks in logs

6. **Slow S3 Operations**
   - Use multipart upload for large objects
   - Consider using Ceph with SSD-backed pools
   - Enable compression: `LMCACHE_ENABLE_COMPRESSION=True`
   - Evaluate network latency to S3 endpoint

## Performance Tuning

### 1. NIXL Transport Optimization

```bash
# Use RDMA for low-latency networks
--nixl-transport rdma

# Use TCP for general networks
--nixl-transport tcp

# Adjust buffer size based on workload
--kv-buffer-size 20GB
```

### 2. LMCache Configuration

```yaml
cache:
  chunk_size: 512  # Larger chunks for better throughput
  max_cache_size: 200GB  # Adjust based on available memory
  eviction_policy: lru  # or 'lfu' for frequency-based eviction
  compression: true  # Enable compression for storage efficiency
```

### 3. S3 Backend Optimization

```bash
# Use multipart upload for large KV caches
export AWS_CLI_MULTIPART_THRESHOLD=100MB
export AWS_CLI_MULTIPART_CHUNKSIZE=50MB

# Enable S3 transfer acceleration (if supported by Ceph)
aws s3api put-bucket-accelerate-configuration \
  --bucket ${S3_BUCKET_NAME} \
  --accelerate-configuration Status=Enabled \
  --endpoint-url ${S3_ENDPOINT_URL}
```

### 4. Object Storage Lifecycle Management

Managing "unbounded" cache space in object storage requires different strategies than traditional filesystems:

#### Strategy 1: Metadata Store + Background Cleanup (Recommended)

Use Redis/etcd to track cache metadata and enforce size limits:

```python
# Example metadata tracking
# In Redis:
{
  "cache_key_hash": {
    "s3_object": "s3://bucket/cache/obj_12345",
    "size": 4096000,
    "last_access": 1234567890,
    "hit_count": 42
  }
}
```

When cache limit is reached:
1. Query metadata store for LRU candidates
2. Delete objects from S3 (async, fire-and-forget)
3. Update metadata atomically

#### Strategy 2: Time-Based Expiration with S3 Lifecycle

Configure S3 lifecycle policies for automatic cleanup:

```bash
# Create lifecycle policy
cat > lifecycle_policy.json <<EOF
{
  "Rules": [
    {
      "Id": "DeleteOldCache",
      "Status": "Enabled",
      "Prefix": "kv-cache/",
      "Expiration": {
        "Days": 7
      }
    }
  ]
}
EOF

# Apply lifecycle policy
aws s3api put-bucket-lifecycle-configuration \
  --bucket ${S3_BUCKET_NAME} \
  --lifecycle-configuration file://lifecycle_policy.json \
  --endpoint-url ${S3_ENDPOINT_URL}
```

**Pros**: Automatic cleanup, no manual intervention
**Cons**: Can't react to space pressure immediately

#### Strategy 3: Hybrid Approach

Use local disk tier with strict LRU eviction, and S3 as write-only cold tier:

```
GPU → CPU → Local Disk (fixed size, strict LRU) → S3 (write-only, no eviction)
```

Let S3 lifecycle rules handle eventual cleanup of cold data.

## Alternative Storage Backends

While this recipe focuses on Ceph S3, LMCache supports multiple storage backends:

### CPU RAM (Fastest Local Tier)

```bash
-e LMCACHE_LOCAL_CPU=True \
-e LMCACHE_MAX_LOCAL_CPU_SIZE=20.0 \
```

**Use case**: Fast local cache tier, automatic LRU eviction

### Local Disk/Filesystem

```bash
-e LMCACHE_LOCAL_DISK=/mnt/nvme/lmcache \
-e LMCACHE_MAX_LOCAL_DISK_SIZE=500.0 \
```

**Use case**: Larger local cache capacity, NVMe/SSD backed

### CephFS via NFS over RDMA

```bash
# Mount NFS with RDMA first (see Step 1, Option B)
# Then configure LMCache to use the mount
-e LMCACHE_REMOTE_URL=file:///mnt/ceph-nfs-kv-cache \
-v /mnt/ceph-nfs-kv-cache:/mnt/ceph-nfs-kv-cache:rw \
```

**Use case**: Shared distributed filesystem with RDMA performance
- **Latency**: 1-5ms (much lower than S3)
- **Throughput**: 10-50 GB/s (10-50× faster than S3)
- **IOPS**: 50k+ (excellent for small random reads/writes)
- **Best for**: Large deployments needing shared cache with low latency
- **Requirements**: RDMA-capable network (InfiniBand or RoCE)

**Advantages over S3:**
- Direct filesystem access (no REST API overhead)
- Better for small object operations (KV cache chunks)
- POSIX semantics enable atomic operations
- Lower CPU overhead with RDMA bypass

**Comparison with Mooncake:**
- NFS over RDMA: Simpler to deploy, mature, POSIX-compliant
- Mooncake: Purpose-built for KV cache, higher throughput (87-190 GB/s), better cache hit rates

### Redis

```bash
-e LMCACHE_REMOTE_URL=redis://redis-server:6379/lmcache
```

**Use case**: Fast distributed cache with atomic operations, good for metadata tracking

### Mooncake Storage

```bash
-e LMCACHE_REMOTE_URL=mooncake://mooncake-server:50051/lmcache
```

**Use case**: High-performance distributed KV cache storage (up to 190 GB/s on 8x400 Gbps RoCE)
- Significantly faster than S3 for KV cache operations
- See [Mooncake paper](https://www.usenix.org/conference/fast25/presentation/qin) (FAST '25 Best Paper)
- Achieves 2.36× higher cache hit rate vs local cache

### InfiniStore

```bash
-e LMCACHE_REMOTE_URL=infinistore://infinistore-server:50052/lmcache
```

**Use case**: RDMA-based distributed storage for ultra-low latency

### ValKey

```bash
-e LMCACHE_REMOTE_URL=valkey://valkey-server:6379/lmcache
```

**Use case**: Redis-compatible open-source alternative

### Backend Comparison

| Backend | Latency | Throughput | Capacity | Persistence | Best For |
|---------|---------|------------|----------|-------------|----------|
| CPU RAM | ~μs | 100+ GB/s | 10-100 GB | No | Hot cache tier |
| Local Disk | ~ms | 1-10 GB/s | 1-10 TB | Yes | Medium tier |
| **NFS/RDMA (Ceph)** | **1-5 ms** | **10-50 GB/s** | **Unlimited** | **Yes** | **Shared cache, RDMA networks** |
| Redis | 1-5 ms | 1 GB/s | 100 GB-1 TB | Optional | Distributed, fast |
| Mooncake | 5-10 ms | 87-190 GB/s | Unlimited | Yes | Large-scale, high perf |
| S3/Ceph | 10-100 ms | 100 MB-1 GB/s | Unlimited | Yes | Cold storage, durable |
| InfiniStore | 1-5 ms | 50-100 GB/s | 1-100 TB | Yes | RDMA networks |

**Performance Tiers Summary:**
- **Tier 0 (Local)**: CPU RAM, Local Disk - lowest latency, limited capacity
- **Tier 1 (Fast Distributed)**: NFS/RDMA, InfiniStore, Redis - good balance, 1-5ms latency
- **Tier 2 (High Throughput)**: Mooncake - purpose-built for KV cache, highest throughput
- **Tier 3 (Cold Storage)**: S3/Ceph - highest capacity, highest latency, lowest cost/GB

## Best Practices

### 1. Sizing Cache Tiers

**GPU Memory:**
- Leave 10-20% headroom: use `--gpu-memory-utilization 0.8-0.9`
- Larger models need more GPU memory for weights

**CPU Memory:**
- Start with 10-20 GB per instance: `LMCACHE_MAX_LOCAL_CPU_SIZE=10.0`
- Monitor eviction rate; increase if too frequent
- Scale with number of concurrent requests

**Local Disk:**
- Use NVMe SSDs for best performance
- Size based on expected working set: 50-500 GB typical
- Enable only if you have fast local storage

**Remote Storage (S3):**
- Effectively unlimited, but monitor costs
- Use lifecycle policies to clean up old cache entries
- Consider tiering to cheaper storage classes after 30+ days

### 2. Choosing Storage Backend

**Small deployment (1-4 nodes):**
- Use CPU + Local Disk + S3
- Simple setup, good cost/performance
- Upgrade to NFS/RDMA if you have RDMA network

**Medium deployment (4-16 nodes):**
- **Recommended**: CPU + Disk + NFS/RDMA (Ceph) + S3 (cold storage)
- Alternative: CPU + Disk + Redis + S3
- NFS/RDMA provides better performance for shared cache with POSIX semantics

**Large deployment (16+ nodes):**
- **Option 1**: CPU + Disk + Mooncake + S3 (cold storage) - highest throughput
- **Option 2**: CPU + Disk + NFS/RDMA (Ceph) + S3 - simpler, POSIX-compliant
- **Option 3**: CPU + Disk + InfiniStore + S3 - RDMA-based distributed storage
- Choose based on: Mooncake for max throughput, NFS/RDMA for simplicity and maturity

**Decision factors:**
- **Have RDMA network?** Consider NFS/RDMA or Mooncake over S3
- **Need max throughput (>50 GB/s)?** Use Mooncake
- **Want POSIX filesystem semantics?** Use NFS/RDMA
- **Need simple object storage?** Use S3
- **Budget-constrained?** Use S3 with lifecycle policies

### 3. Optimizing Cache Hit Rate

**Workload-specific tuning:**
- **Chatbots**: Enable P2P sharing, longer cache retention (7+ days)
- **Document QA**: Larger chunk_size (512-1024), save system prompts
- **Code completion**: Smaller chunk_size (128-256), frequent updates

**Configuration tips:**
```bash
# For chatbot workloads (high prefix reuse)
-e LMCACHE_CHUNK_SIZE=256 \
-e LMCACHE_CACHE_POLICY=LFU \  # Frequency-based
-e LMCACHE_ENABLE_P2P=True \

# For document QA (long contexts)
-e LMCACHE_CHUNK_SIZE=512 \
-e LMCACHE_SAVE_DECODE_CACHE=False \  # Only cache prefill
-e LMCACHE_USE_LAYERWISE=True \  # Pipeline transfers

# For mixed workloads
-e LMCACHE_CACHE_POLICY=LRU \  # Balanced approach
-e LMCACHE_PRIORITY_LIMIT=50 \  # Only cache high-priority requests
```

### 4. Production Deployment Checklist

**Before going live:**
- [ ] Enable monitoring: `LMCACHE_INTERNAL_API_SERVER_ENABLED=True`
- [ ] Configure S3 lifecycle policies for automatic cleanup
- [ ] Set appropriate cache size limits to avoid OOM
- [ ] Test NIXL connectivity between all prefill/decode pairs
- [ ] Implement health checks and automated restarts
- [ ] Set up alerting for cache hit rate < 40%
- [ ] Enable compression for S3 backend: saves 30-50% storage
- [ ] Configure backup/restore procedures for critical cache data
- [ ] Document disaster recovery procedures
- [ ] Load test with production-like traffic patterns

**Security considerations:**
- Use IAM roles instead of hardcoded AWS credentials
- Encrypt S3 buckets at rest (SSE-S3 or SSE-KMS)
- Use TLS for NIXL transport in untrusted networks
- Restrict S3 bucket access with least-privilege policies
- Enable S3 versioning for critical cache data
- Audit S3 access logs regularly

### 5. Cost Optimization

**Reduce S3 costs:**
- Enable compression: 30-50% savings
- Use lifecycle policies: transition to cheaper storage classes
- Set `LMCACHE_PRIORITY_LIMIT` to only cache high-value requests
- Monitor and delete stale cache entries

**Compute cost savings:**
- LMCache reduces prefill compute by 3-10×
- Enables higher batch sizes and better GPU utilization
- Can reduce number of prefill instances by 2-3×

**Network costs:**
- Use RDMA to reduce CPU overhead
- Co-locate prefill/decode nodes to minimize inter-AZ traffic
- Consider Mooncake for high-throughput, cost-effective KV transfer

## Key Insights from Production Deployments

This section summarizes learnings from Mooncake (Moonshot AI's Kimi chatbot platform), which won Best Paper at FAST 2025 and processes **100+ billion tokens daily** across thousands of nodes.

### 1. KVCache-Centric Architecture Principles

**Trade Storage for Computation:**
- Principle: Store more KV cache to avoid redundant prefill computation
- Impact: 59-498% increase in effective request capacity
- Key insight: Underutilized CPU, DRAM, SSD, and NIC resources can be aggregated into a distributed KV cache pool

**Multi-Tier Storage Strategy:**
- Extend cache across device (GPU), host (CPU/DRAM), and remote storage (distributed)
- Use RadixAttention with multi-tier KV cache storage
- Implement intelligent promotion/demotion between tiers based on access patterns

### 2. Topology-Aware Path Selection

**Challenge:**
In distributed KV cache systems, different NICs have varying performance characteristics and network paths.

**Solution:**
- Generate and broadcast topology matrices across the cluster
- Categorize NICs into "preferred" and "secondary" lists for each memory type
- Route cache traffic based on topology to minimize latency and maximize bandwidth

**Benefits:**
- Optimizes network utilization across heterogeneous NICs
- Reduces tail latency by avoiding congested paths
- Better load balancing across available network interfaces

### 3. Distributed Memory Pooling

**Concept:**
Aggregate underutilized memory from multiple nodes into a unified distributed storage service.

**Implementation:**
- Pool DRAM and SSD resources from GPU cluster nodes
- Create shared memory pool accessible via high-speed network
- Enable elastic scaling of cache capacity without adding dedicated storage nodes

**Results:**
- Maximizes resource efficiency across the cluster
- Reduces TCO by utilizing existing hardware
- Enables cache sharing across prefill and decode instances

### 4. KVCache-Centric Scheduling

**Scheduling Objectives (in order):**
1. **Maximize KV cache reuse**: Prioritize requests that can reuse existing cache entries
2. **Balance workloads**: Distribute requests evenly across prefill nodes
3. **Guarantee TTFT SLO**: Ensure Time-to-First-Token meets service level objectives

**Scheduling Strategy:**
- Track KV cache locations and access patterns
- Route requests to nodes with relevant cached prefixes
- Implement cache-aware load balancing (not just round-robin)

**Impact:**
- 2.36× higher cache hit rate vs. local-only cache
- 48% reduction in prefill computation time
- Better compliance with latency SLOs

### 5. Production Deployment Lessons

**Scale:**
- Operational across thousands of nodes
- Processing 100+ billion tokens daily
- Supporting Kimi chatbot (leading Chinese LLM service)

**Key Metrics:**
- Request capacity increase: 59-498% vs. baselines
- Cache hit rate improvement: 2.36× vs. local cache
- Prefill time savings: Up to 48%

**Critical Success Factors:**
1. **Network is bottleneck**: High-bandwidth, low-latency network (RDMA) is essential
2. **Cache placement matters**: Intelligent scheduling based on cache locality is crucial
3. **Multi-tier is necessary**: Single-tier cache insufficient for production scale
4. **Topology awareness**: Network topology significantly impacts performance
5. **Shared cache wins**: Distributed cache pool outperforms per-node caches

### 6. Architectural Recommendations

Based on Mooncake's production experience:

**For medium-scale deployments (4-16 nodes):**
- Start with CPU + Local Disk + NFS/RDMA
- Implement cache-aware request routing
- Monitor cache hit rates and adjust chunk sizes

**For large-scale deployments (16+ nodes):**
- Deploy dedicated distributed KV cache layer (Mooncake, InfiniStore, or NFS/RDMA cluster)
- Implement topology-aware routing
- Use KVCache-centric scheduling (not just load balancing)
- Pool underutilized resources (CPU, DRAM, SSD)

**Performance expectations:**
- Target cache hit rate: >60% for production workloads
- Chatbots can achieve 70-90% hit rates with proper tuning
- Each 10% increase in hit rate translates to ~8-10% reduction in compute costs

## Reference Architecture

### Small-Scale Deployment (1 Prefill + 1 Decode)

**Prefill Node (AMD MI300X):**
- 1x MI300X (192GB HBM3, 8 GCDs)
- Host: 256GB DDR5 RAM, 2TB NVMe SSD
- ROCm 6.0+
- 1x 100Gbps RDMA NIC

**Decode Node (Intel Gaudi3):**
- 4x Gaudi3 (128GB HBM2e each, total 512GB)
- Host: 128GB DDR5 RAM, 1TB NVMe SSD
- SynapseAI 1.18+
- 1x 100Gbps RDMA NIC

**Shared Infrastructure:**
- Network: 100Gbps RDMA (InfiniBand or RoCE)
- Storage: Ceph S3 or NFS/RDMA with 10TB capacity
- Use case: Development, small production workloads

**Performance Expectations:**
- Prefill throughput: 50K-100K tokens/sec (MI300X)
- Decode throughput: 10K-15K tokens/sec (4x Gaudi3)
- TTFT: <500ms with cache, 1-3s without cache

### Medium-Scale Deployment (4 Prefill + 8 Decode)

**Prefill Cluster (AMD MI300X):**
- 4 nodes, each with 2x MI300X (384GB HBM3 per node)
- Total prefill capacity: 8x MI300X
- Host per node: 512GB RAM, 4TB NVMe SSD
- 2x 200Gbps RDMA NICs per node

**Decode Cluster (Intel Gaudi3):**
- 8 nodes, each with 4x Gaudi3 (512GB HBM2e per node)
- Total decode capacity: 32x Gaudi3
- Host per node: 256GB RAM, 2TB NVMe SSD
- 2x 200Gbps RDMA NICs per node

**Shared Infrastructure:**
- Network: 200Gbps RDMA fabric
- Storage: Ceph NFS/RDMA or Mooncake with 50-100TB capacity
- Load Balancer: NGINX or HAProxy
- Orchestration: Kubernetes recommended

**Cost Optimization:**
- Ratio: 1 MI300X prefill : 4 Gaudi3 decode (balanced for most workloads)
- MI300X handles memory-intensive prefill
- Gaudi3 provides cost-effective decode scaling

**Performance Expectations:**
- Aggregate prefill: 400K-800K tokens/sec
- Aggregate decode: 80K-120K tokens/sec
- Concurrent users: 1K-5K
- Cache hit rate target: >60%

### Large-Scale Deployment (16+ Prefill + 32+ Decode)

**Prefill Cluster (AMD MI300X):**
- 16+ nodes, each with 4x MI300X (768GB HBM3 per node)
- Total prefill capacity: 64+ MI300X
- Host per node: 1TB RAM, 8TB NVMe SSD
- 4x 400Gbps InfiniBand NICs per node

**Decode Cluster (Intel Gaudi3):**
- 32+ nodes, each with 8x Gaudi3 (1TB HBM2e per node)
- Total decode capacity: 256+ Gaudi3
- Host per node: 512GB RAM, 4TB NVMe SSD
- 2x 400Gbps InfiniBand NICs per node

**Shared Infrastructure:**
- Network: 400Gbps InfiniBand fabric with RDMA
- Storage: Mooncake or NFS/RDMA cluster with 200TB-1PB capacity
- Distributed cache pool: Aggregate underutilized DRAM/SSD
- Load Balancer: Multiple NGINX instances with DNS round-robin
- Orchestration: Kubernetes with custom KV-cache-aware scheduling
- Monitoring: Prometheus + Grafana for metrics

**Advanced Features:**
- Topology-aware routing (Mooncake-style)
- KVCache-centric scheduling
- Distributed memory pooling
- Auto-scaling based on request load

**Performance Expectations:**
- Aggregate prefill: 3M-6M tokens/sec
- Aggregate decode: 500K-1M tokens/sec
- Concurrent users: 50K-100K+
- Cache hit rate target: 70-90% (with proper tuning)
- SLO: 95th percentile TTFT <1s

**Cost Analysis (Approximate):**
- MI300X nodes: Higher capex, justified by prefill throughput
- Gaudi3 nodes: Lower opex for decode workload
- Overall TCO: 30-40% lower than homogeneous premium GPU setup
- Power efficiency: Gaudi3's lower TDP reduces power costs for decode

## vLLM Cache Lifecycle Policies

### Current Implementation (GPU Cache)

#### Basic LRU Policy

vLLM implements a reference-counting LRU eviction policy for GPU cache:

**Eviction Trigger:**
- Activates when no free KV blocks remain

**Eviction Criteria (in order):**
1. **First**: Only evict blocks with `reference_count == 0` (not in use by any request)
2. **Second**: Among zero-ref blocks, evict Least Recently Used (LRU)
3. **Tertiary**: Prefer blocks at the end of longest prefix

#### Block Management

- Cache divided into fixed-size "KV blocks"
- Each block uniquely identified by `hash(prefix_tokens + block_tokens)`
- Enables sharing blocks across requests with common prefixes
- Hash-based lookup allows efficient prefix cache reuse

### Proposed/In-Progress Enhancements

#### 1. Frequency and Cost Aware Eviction ([RFC #23641](https://github.com/vllm-project/vllm/issues/23641))

Instead of pure LRU, track retention benefit:

```python
retention_benefit = freq * compute_cost
compute_cost = cost_factor * size^alpha  # alpha ≈ 2
```

**Benefits:**
- Evict block with lowest retention benefit
- Considers both access frequency and recomputation cost
- Better for workloads with varying prefix sizes
- Avoids evicting expensive-to-recompute prefixes

#### 2. Cross-Engine KV Cache Offloading ([RFC #14724](https://github.com/vllm-project/vllm/issues/14724))

**Two-tier architecture:**
- **L1 Cache**: Local GPU/CPU memory (fast)
- **L2 Cache**: Remote storage (disk, network, S3)

**Offloading modes:**
- `ALL`: Offload everything to remote cache
- `HOT`: Offload only frequently accessed entries
- `EVICTED`: Offload only when evicted from L1

**Supported eviction policies:**
- **LRU** (Least Recently Used) - default, balanced
- **FIFO** (First In First Out) - simple, predictable
- **S3FIFO** (Scan-resistant) - better for mixed workloads with one-time scans

#### 3. CPU Offloading ([RFC #16144](https://github.com/vllm-project/vllm/issues/16144))

**Simplified policy:**
- Initial implementation: round-robin for simplicity
- Planned enhancement: LRU policy for better cache efficiency
- Enables larger effective cache capacity using host memory

### Related RFCs and Resources

- [Frequency and Cost Aware Eviction Policy](https://github.com/vllm-project/vllm/issues/23641)
- [Cross-Engine KV Cache Offloading](https://github.com/vllm-project/vllm/issues/14724)
- [CPU Offloading (V1)](https://github.com/vllm-project/vllm/issues/16144)
- [New eviction strategy for prefix cache indexer (aibrix)](https://github.com/vllm-project/aibrix/issues/892)

## Additional Resources

### Documentation
- [vLLM Documentation](https://docs.vllm.ai/)
- [LMCache Documentation](https://docs.lmcache.ai/)
- [LMCache GitHub Repository](https://github.com/LMCache/LMCache)
- [LMCache Configuration Reference](https://docs.lmcache.ai/api_reference/configurations.html)
- [NIXL Connector Documentation](https://docs.vllm.ai/en/stable/features/nixl_connector_usage.html)
- [Ceph S3 API Documentation](https://docs.ceph.com/en/latest/radosgw/s3/)
- [CephFS Documentation](https://docs.ceph.com/en/latest/cephfs/)
- [NFS-Ganesha with Ceph](https://docs.ceph.com/en/latest/cephfs/nfs/)
- [NFS over RDMA (NFS/RDMA) Guide](https://www.kernel.org/doc/Documentation/filesystems/nfs/nfs-rdma.txt)

### Research Papers
- [Mooncake: KVCache-centric Architecture for LLM Serving (FAST '25, Best Paper)](https://www.usenix.org/conference/fast25/presentation/qin)
- [LMCache Technical Report](https://lmcache.ai/tech_report.pdf)
- [Epic: Efficient Position-Independent Context Caching](https://arxiv.org/html/2410.15332v1)
- [vLLM Automatic Prefix Caching Design](https://docs.vllm.ai/en/stable/design/prefix_caching.html)

### Community and Support
- [vLLM GitHub Issues](https://github.com/vllm-project/vllm/issues)
- [LMCache GitHub Issues](https://github.com/LMCache/LMCache/issues)
- [vLLM Discord Community](https://discord.gg/vllm)

### Related Tools and Integrations
- [Mooncake KV Cache Storage](https://github.com/kvcache-ai/Mooncake)
- [NIXL for Intel Gaudi](https://docs.habana.ai/)
- [vLLM Examples with LMCache](https://docs.vllm.ai/en/latest/examples/others/lmcache.html)

