# LMCache S3 Support - Quick Reference

**Generated**: October 22, 2025  
**Patch File**: `lmcache-s3-complete.patch` (71 KB)  
**Commits**: 6 S3-related changes from LMCache upstream  

---

## Quick Start

### Apply the Patch

```bash
cd /Users/tgohad/src/ofiplugin/files/LMCache-fork
git checkout dev
git apply /Users/tgohad/src/ofiplugin/files/lmcache-s3-complete.patch
git add .
git commit -m "Port S3 connector from LMCache upstream (c0988e4..fe20ff0)"
```

### Install Dependencies

```bash
pip install -e ".[s3]"
# or
pip install boto3>=1.28.0 awscrt>=0.20.0
```

### Test S3 Connector

```python
from lmcache.v1.storage_backend.connector.s3_connector import S3Connector

connector = S3Connector(bucket="test-bucket", prefix="kv-cache/")
connector.put("test-key", b"test-data")
data = connector.get("test-key")
assert data == b"test-data"
print("✅ S3 connector works!")
```

---

## What's Included

1. **S3 Connector** (`lmcache/v1/storage_backend/connector/s3_connector.py`)
   - Sync and async S3 operations
   - Memory leak fixes (2 separate fixes)
   - Semaphore deadlock fix
   - ~850 lines

2. **S3 Adapter** (`lmcache/v1/storage_backend/connector/s3_adapter.py`)
   - AWS SDK wrapper with awscrt
   - Async upload/download
   - ~200 lines

3. **Documentation** (`docs/source/kv_cache/storage_backends/s3.rst`)
   - Configuration guide
   - Examples
   - ~100 lines

4. **Examples** (`examples/kv_cache_reuse/remote_backends/s3/`)
   - Sample configs
   - Usage patterns
   - ~150 lines

5. **Dependencies**
   - `awscrt>=0.20.0` (added to common.txt)
   - `boto3>=1.28.0` (added to s3 extras)

---

## Commits Applied

| Commit | Date | Description |
|--------|------|-------------|
| c0988e4 | Aug 18 | Initial S3 connector (#1374) |
| d9d3151 | Aug 19 | Add awscrt to common requirements (#1383) |
| 72e2018 | Aug 20 | Improve S3 (#1402) |
| b7cee81 | Aug 25 | Fix double semaphore acquire (#1427) |
| eb272a8 | Aug 29 | Fix mem leak (#1461) |
| 525618e | Sep 2 | Fix mem leak in S3 connector (#1495) |
| fe20ff0 | Sep 18 | Async S3 support (#1614) |

---

## Key Features

✅ AWS S3 remote storage backend  
✅ Async I/O for better performance  
✅ Memory leak fixes (critical for production)  
✅ Deadlock prevention  
✅ MinIO / S3-compatible storage support  
✅ Configurable via YAML or Python API  

---

## Configuration Example

```yaml
# lmcache-s3.yaml
chunk_size: 256
local_device: "cpu"
remote_serde: "cachegen"

storage_backend:
  type: "remote"
  connector:
    type: "s3"
    bucket: "lmcache-kv-cache"
    prefix: "models/qwen3-32b/"
    region: "us-west-2"
```

---

## Next Steps

1. **Review the full guide**: `LMCACHE_S3_PORTING_GUIDE.md`
2. **Apply the patch**: See Quick Start above
3. **Test on Gaudi3**: Combine with HPU support
4. **Submit PR**: To LMCache-fork repository

---

## Files Location

- **Complete Patch**: `/Users/tgohad/src/ofiplugin/files/lmcache-s3-complete.patch`
- **Individual Patches**: `/tmp/lmcache-s3-patches/`
- **Porting Guide**: `/Users/tgohad/src/ofiplugin/files/LMCACHE_S3_PORTING_GUIDE.md`
- **This Summary**: `/Users/tgohad/src/ofiplugin/files/LMCACHE_S3_SUMMARY.md`

---

## Support

For detailed instructions, troubleshooting, and architecture details, see:
**`LMCACHE_S3_PORTING_GUIDE.md`** (comprehensive 400+ line guide)
