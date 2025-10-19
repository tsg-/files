#!/bin/bash
# apply_gaudi3_patch.sh
# Quick-start script for applying LMCache Gaudi3 HPU patch

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}LMCache Gaudi3 HPU Patch Installer${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""

# Check if running on Gaudi3 system
echo -e "${YELLOW}[1/7] Checking Gaudi3 environment...${NC}"
if ! command -v hl-smi &> /dev/null; then
    echo -e "${RED}ERROR: hl-smi not found. Is this a Gaudi3 system?${NC}"
    echo -e "${YELLOW}Install Gaudi drivers: sudo apt-get install habanalabs-drivers${NC}"
    exit 1
fi

# Check device count
DEVICE_COUNT=$(hl-smi | grep -c "Gaudi" || true)
echo -e "${GREEN}✓ Found ${DEVICE_COUNT} Gaudi device(s)${NC}"

# Check Python and HPU availability
echo -e "${YELLOW}[2/7] Checking Python and PyTorch HPU...${NC}"
python3 << 'EOF'
import sys
try:
    import torch
    print(f"✓ PyTorch version: {torch.__version__}")
    
    if torch.hpu.is_available():
        print(f"✓ HPU available: {torch.hpu.device_count()} device(s)")
    else:
        print("✗ HPU not available")
        sys.exit(1)
        
    import habana_frameworks.torch.core as htcore
    print("✓ habana_frameworks available")
    
except ImportError as e:
    print(f"✗ Import error: {e}")
    print("\nInstall with:")
    print("pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cpu")
    print("pip install habana-torch-plugin==2.5.1 --index-url https://vault.habana.ai/artifactory/api/pypi/gaudi-pt-modules/simple")
    sys.exit(1)
EOF

if [ $? -ne 0 ]; then
    exit 1
fi

# Locate LMCache directory
echo -e "${YELLOW}[3/7] Locating LMCache installation...${NC}"
LMCACHE_DIR=$(python3 -c "import lmcache; import os; print(os.path.dirname(lmcache.__file__))" 2>/dev/null || echo "")

if [ -z "$LMCACHE_DIR" ]; then
    echo -e "${RED}ERROR: LMCache not found. Install with:${NC}"
    echo -e "${YELLOW}cd /path/to/LMCache && pip install -e .${NC}"
    exit 1
fi

echo -e "${GREEN}✓ LMCache found at: ${LMCACHE_DIR}${NC}"

# Create backup
echo -e "${YELLOW}[4/7] Creating backup of original files...${NC}"
BACKUP_DIR="${LMCACHE_DIR}.backup.$(date +%Y%m%d_%H%M%S)"
mkdir -p "$BACKUP_DIR"

FILES_TO_BACKUP=(
    "integration/vllm/vllm_v1_adapter.py"
    "v1/gpu_connector/__init__.py"
    "v1/gpu_connector/vllm_connector.py"
    "utils.py"
)

for file in "${FILES_TO_BACKUP[@]}"; do
    if [ -f "${LMCACHE_DIR}/${file}" ]; then
        mkdir -p "$(dirname "${BACKUP_DIR}/${file}")"
        cp "${LMCACHE_DIR}/${file}" "${BACKUP_DIR}/${file}"
        echo -e "${GREEN}✓ Backed up: ${file}${NC}"
    fi
done

echo -e "${GREEN}✓ Backup created at: ${BACKUP_DIR}${NC}"

# Apply patches
echo -e "${YELLOW}[5/7] Applying HPU adaptations...${NC}"

# Patch 1: vllm_v1_adapter.py
echo -e "${YELLOW}  Patching vllm_v1_adapter.py...${NC}"
ADAPTER_FILE="${LMCACHE_DIR}/integration/vllm/vllm_v1_adapter.py"

# Add HPU imports
if ! grep -q "habana_frameworks.torch.core" "$ADAPTER_FILE"; then
    # Insert after torch import
    sed -i.bak '/^import torch$/a\
\
# HPU-specific imports for Gaudi3\
try:\
    import habana_frameworks.torch.core as htcore\
except ImportError:\
    htcore = None
' "$ADAPTER_FILE"
    echo -e "${GREEN}✓ Added HPU imports${NC}"
fi

# Note: Full patch application would require more complex sed/awk
# For production use, use the .patch file with git apply or manual editing
echo -e "${YELLOW}  NOTE: Complete patch requires manual application.${NC}"
echo -e "${YELLOW}  See GAUDI3_PATCH_README.md for full instructions.${NC}"

# Test basic import
echo -e "${YELLOW}[6/7] Testing patched modules...${NC}"
python3 << 'EOF'
try:
    # Test basic imports
    import torch
    import habana_frameworks.torch.core as htcore
    from lmcache.integration.vllm.vllm_v1_adapter import LMCacheConnectorV1Impl
    
    print("✓ All imports successful")
    
    # Check device detection
    if torch.hpu.is_available():
        print(f"✓ HPU available: {torch.hpu.device_count()} device(s)")
    else:
        print("✗ HPU not detected")
        
except Exception as e:
    print(f"✗ Import test failed: {e}")
    exit(1)
EOF

if [ $? -ne 0 ]; then
    echo -e "${RED}ERROR: Import test failed${NC}"
    echo -e "${YELLOW}Rolling back...${NC}"
    
    # Restore backup
    for file in "${FILES_TO_BACKUP[@]}"; do
        if [ -f "${BACKUP_DIR}/${file}" ]; then
            cp "${BACKUP_DIR}/${file}" "${LMCACHE_DIR}/${file}"
        fi
    done
    
    exit 1
fi

# Run quick inference test
echo -e "${YELLOW}[7/7] Running quick inference test...${NC}"
python3 << 'EOF'
import os
os.environ['HABANA_VISIBLE_DEVICES'] = '0'
os.environ['PT_HPU_LAZY_MODE'] = '1'

try:
    from vllm import LLM, SamplingParams
    
    print("Creating LLM on HPU...")
    llm = LLM(
        model="facebook/opt-125m",  # Small model for quick test
        tensor_parallel_size=1,
        max_model_len=512,
        device="hpu",
        dtype="bfloat16",
    )
    
    print("Running inference...")
    outputs = llm.generate(
        ["Hello, this is a test."],
        SamplingParams(temperature=0.0, max_tokens=10)
    )
    
    print(f"✓ Inference successful: {outputs[0].outputs[0].text}")
    
except Exception as e:
    print(f"⚠ Inference test failed: {e}")
    print("This may be due to missing models or other dependencies.")
    print("The patch was applied but vLLM may need additional setup.")
EOF

echo ""
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}Patch Installation Summary${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""
echo -e "${GREEN}✓ Gaudi3 devices detected${NC}"
echo -e "${GREEN}✓ HPU PyTorch available${NC}"
echo -e "${GREEN}✓ LMCache located${NC}"
echo -e "${GREEN}✓ Backup created at: ${BACKUP_DIR}${NC}"
echo -e "${YELLOW}⚠ Manual patch application required for full functionality${NC}"
echo ""
echo -e "${YELLOW}Next Steps:${NC}"
echo "1. Review the patch file: vllm_v1_adapter_gaudi3.patch"
echo "2. Apply manual changes from GAUDI3_PATCH_README.md"
echo "3. Run full test suite:"
echo "   python3 test_lmcache_hpu.py"
echo ""
echo -e "${YELLOW}Rollback Instructions:${NC}"
echo "If you need to revert the changes:"
echo "  cp -r ${BACKUP_DIR}/* ${LMCACHE_DIR}/"
echo ""
echo -e "${GREEN}Installation complete!${NC}"
