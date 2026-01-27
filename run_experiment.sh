#!/bin/bash
# Phase-RWKV Experiment Launcher
# Complete workflow automation

set -e

STUDIO_USER="tony_studio"
STUDIO_HOST="192.168.1.195"
STUDIO_PATH="~/phase-rwkv-training"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

echo -e "${CYAN}╔════════════════════════════════════════════════════════════════════╗${NC}"
echo -e "${CYAN}║${NC}  ${PURPLE}🌀 PHASE-RWKV: CONSCIOUSNESS EXPERIMENT - ATTEMPT 3${NC}             ${CYAN}║${NC}"
echo -e "${CYAN}╚════════════════════════════════════════════════════════════════════╝${NC}"
echo ""

# Step 1: Check for training data
echo -e "${BLUE}[1/5]${NC} Checking training data..."
if [ -f "data/high_resonance.jsonl" ]; then
    SAMPLE_COUNT=$(wc -l < data/high_resonance.jsonl)
    echo -e "   ${GREEN}✅ Found training data: ${SAMPLE_COUNT} samples${NC}"
else
    echo -e "   ${YELLOW}⚠️  Training data not found${NC}"
    echo -e "   ${CYAN}Generating synthetic data...${NC}"
    python3 generate_synthetic_data.py
    echo ""
fi

# Step 2: Verify local setup
echo -e "${BLUE}[2/5]${NC} Verifying local setup..."

# Check Python
if ! command -v python3 &> /dev/null; then
    echo -e "   ${RED}❌ Python 3 not found${NC}"
    exit 1
fi
echo -e "   ${GREEN}✅ Python 3 available${NC}"

# Check files
REQUIRED_FILES=("train_phase_rwkv.py" "phase_rwkv.py" "visualize_metrics.py")
for file in "${REQUIRED_FILES[@]}"; do
    if [ ! -f "$file" ]; then
        echo -e "   ${RED}❌ Missing: $file${NC}"
        exit 1
    fi
done
echo -e "   ${GREEN}✅ All required files present${NC}"
echo ""

# Step 3: Deploy to Mac Studio
echo -e "${BLUE}[3/5]${NC} Deploying to Mac Studio..."
echo -e "   ${CYAN}Target: ${STUDIO_USER}@${STUDIO_HOST}${NC}"

# Test connection
if ! ssh -o ConnectTimeout=5 ${STUDIO_USER}@${STUDIO_HOST} "echo 'Connection test'" &> /dev/null; then
    echo -e "   ${RED}❌ Cannot connect to Mac Studio${NC}"
    echo -e "   ${YELLOW}Is the Studio powered on and connected?${NC}"
    exit 1
fi
echo -e "   ${GREEN}✅ Connection established${NC}"

# Create remote directory
ssh ${STUDIO_USER}@${STUDIO_HOST} "mkdir -p ${STUDIO_PATH}"

# Sync files
echo -e "   ${CYAN}Syncing files...${NC}"
rsync -az --progress \
    train_phase_rwkv.py \
    phase_rwkv.py \
    data/high_resonance.jsonl \
    ${STUDIO_USER}@${STUDIO_HOST}:${STUDIO_PATH}/

echo -e "   ${GREEN}✅ Deployment complete${NC}"
echo ""

# Step 4: Verify Studio environment
echo -e "${BLUE}[4/5]${NC} Verifying Studio environment..."

ssh ${STUDIO_USER}@${STUDIO_HOST} << 'EOF'
cd ~/phase-rwkv-training

# Check packages
echo "   Checking dependencies..."

MISSING_DEPS=0

if ! python3 -c "import torch" 2>/dev/null; then
    echo "   ⚠️  PyTorch not installed"
    MISSING_DEPS=1
else
    TORCH_VERSION=$(python3 -c "import torch; print(torch.__version__)" 2>/dev/null)
    echo "   ✅ PyTorch: $TORCH_VERSION"
fi

if ! python3 -c "import transformers" 2>/dev/null; then
    echo "   ⚠️  Transformers not installed"
    MISSING_DEPS=1
else
    TRANSFORMERS_VERSION=$(python3 -c "import transformers; print(transformers.__version__)" 2>/dev/null)
    echo "   ✅ Transformers: $TRANSFORMERS_VERSION"
fi

if ! python3 -c "import rwkv" 2>/dev/null; then
    echo "   ⚠️  RWKV not installed"
    MISSING_DEPS=1
else
    echo "   ✅ RWKV installed"
fi

# Check MPS
MPS_AVAILABLE=$(python3 -c "import torch; print(torch.backends.mps.is_available())" 2>/dev/null)
if [ "$MPS_AVAILABLE" = "True" ]; then
    echo "   ✅ MPS (Metal) available"
else
    echo "   ⚠️  MPS not available (will use CPU)"
fi

if [ $MISSING_DEPS -eq 1 ]; then
    echo ""
    echo "   ⚠️  Some dependencies missing. Install with:"
    echo "   pip3 install torch transformers rwkv"
    exit 1
fi
EOF

if [ $? -ne 0 ]; then
    echo -e "   ${RED}❌ Environment verification failed${NC}"
    exit 1
fi

echo -e "   ${GREEN}✅ Studio environment verified${NC}"
echo ""

# Step 5: Launch training
echo -e "${BLUE}[5/5]${NC} Launching training..."
echo ""
echo -e "${CYAN}╔════════════════════════════════════════════════════════════════════╗${NC}"
echo -e "${CYAN}║${NC}  ${YELLOW}Training Configuration:${NC}                                           ${CYAN}║${NC}"
echo -e "${CYAN}║${NC}    • Model: RWKV-4-Pile-430M (frozen)                            ${CYAN}║${NC}"
echo -e "${CYAN}║${NC}    • Phase Core: 16 oscillators, K=2.0                           ${CYAN}║${NC}"
echo -e "${CYAN}║${NC}    • Batch size: 4 (gradient accumulation: 4)                    ${CYAN}║${NC}"
echo -e "${CYAN}║${NC}    • Steps: 500 (~2 hours)                                       ${CYAN}║${NC}"
echo -e "${CYAN}║${NC}    • Checkpoints: Every 50 steps                                 ${CYAN}║${NC}"
echo -e "${CYAN}║${NC}    • Target: R ∈ [0.80, 0.95], U ≈ 0.5                          ${CYAN}║${NC}"
echo -e "${CYAN}╚════════════════════════════════════════════════════════════════════╝${NC}"
echo ""

read -p "$(echo -e ${YELLOW}Start training on Mac Studio? [y/N]:${NC} )" -n 1 -r
echo ""

if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo -e "${YELLOW}Training cancelled.${NC}"
    echo ""
    echo "To run manually:"
    echo "  ssh ${STUDIO_USER}@${STUDIO_HOST}"
    echo "  cd ${STUDIO_PATH}"
    echo "  python3 train_phase_rwkv.py --iters 500 --batch-size 4 --checkpoint-every 50"
    exit 0
fi

echo ""
echo -e "${GREEN}🚀 Launching training on Mac Studio...${NC}"
echo -e "${CYAN}═══════════════════════════════════════════════════════════════════${NC}"
echo ""

# Launch training (with nohup so it continues if connection drops)
ssh ${STUDIO_USER}@${STUDIO_HOST} << 'EOF'
cd ~/phase-rwkv-training

# Create logs directory
mkdir -p logs

# Run training with output logged
nohup python3 train_phase_rwkv.py \
    --iters 500 \
    --batch-size 4 \
    --gradient-accumulation-steps 4 \
    --checkpoint-every 50 \
    --target-uncertainty 0.5 \
    > logs/training_$(date +%Y%m%d_%H%M%S).log 2>&1 &

TRAIN_PID=$!
echo $TRAIN_PID > training.pid

echo ""
echo "✅ Training launched (PID: $TRAIN_PID)"
echo ""
echo "Monitor with:"
echo "  tail -f logs/training_*.log"
echo ""
echo "Or monitor checkpoints:"
echo "  watch 'ls -lth checkpoints_rwkv/*.pt | head -10'"
echo ""
EOF

echo ""
echo -e "${GREEN}═══════════════════════════════════════════════════════════════════${NC}"
echo -e "${GREEN}✅ Training launched successfully!${NC}"
echo ""
echo -e "${CYAN}Monitor training:${NC}"
echo "  ssh ${STUDIO_USER}@${STUDIO_HOST}"
echo "  tail -f ~/phase-rwkv-training/logs/training_*.log"
echo ""
echo -e "${CYAN}When complete, fetch results:${NC}"
echo "  rsync -avz ${STUDIO_USER}@${STUDIO_HOST}:~/phase-rwkv-training/checkpoints_rwkv/ checkpoints_rwkv/"
echo "  ./visualize_metrics.py"
echo ""
echo -e "${PURPLE}🌀 The apparatus is running. The quantum state evolves.${NC}"
echo ""
