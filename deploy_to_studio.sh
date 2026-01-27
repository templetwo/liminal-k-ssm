#!/bin/bash
# Deploy Phase-RWKV training to Mac Studio and launch

set -e

STUDIO_USER="tony_studio"
STUDIO_HOST="192.168.1.195"
STUDIO_PATH="~/phase-rwkv-training"

echo "🚀 Deploying Phase-RWKV to Mac Studio..."

# Create remote directory
ssh ${STUDIO_USER}@${STUDIO_HOST} "mkdir -p ${STUDIO_PATH}"

# Sync files
echo "📦 Syncing files..."
rsync -avz --progress \
    train_phase_rwkv.py \
    phase_rwkv.py \
    phase-gpt-distilled/data/high_resonance.jsonl \
    ${STUDIO_USER}@${STUDIO_HOST}:${STUDIO_PATH}/

# Check for required packages
echo "🔧 Checking dependencies..."
ssh ${STUDIO_USER}@${STUDIO_HOST} << 'EOF'
cd ~/phase-rwkv-training

# Check Python packages
echo "Checking PyTorch..."
python3 -c "import torch; print(f'PyTorch: {torch.__version__}')" || echo "❌ PyTorch missing"

echo "Checking transformers..."
python3 -c "import transformers; print(f'Transformers: {transformers.__version__}')" || echo "❌ Transformers missing"

echo "Checking rwkv..."
python3 -c "import rwkv; print(f'RWKV: {rwkv.__version__}')" || echo "❌ RWKV missing"

echo "Checking MPS availability..."
python3 -c "import torch; print(f'MPS available: {torch.backends.mps.is_available()}')"
EOF

echo ""
echo "✅ Deployment complete!"
echo ""
echo "To run training on Studio:"
echo "  ssh ${STUDIO_USER}@${STUDIO_HOST}"
echo "  cd ${STUDIO_PATH}"
echo "  python3 train_phase_rwkv.py --iters 500 --batch-size 4 --checkpoint-every 50"
echo ""
echo "Or run directly:"
echo "  ssh ${STUDIO_USER}@${STUDIO_HOST} 'cd ${STUDIO_PATH} && python3 train_phase_rwkv.py --iters 500 --batch-size 4'"
