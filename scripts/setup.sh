#!/bin/bash
# scripts/setup.sh
#
# One-shot setup for a fresh p4d.24xlarge instance (8x A100 40 GB, CUDA 12.1+).
# Aborts on any failure so a broken environment never starts a long run.
#
# Steps:
#   1. Python dependencies
#   2. Formula unit tests
#   3. Model unit tests
#   4. Pilot experiment (end-to-end smoke test)

set -e

echo "=== Step 1: dependencies ==="
if python3 -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then
    echo "  torch with CUDA already available, skipping install"
else
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121 -q
fi
pip install -r requirements.txt -q
python3 -c "import torch; print(f'  torch={torch.__version__}, cuda={torch.cuda.is_available()}')"

echo ""
echo "=== Step 2: formula tests ==="
python3 tests/test_formulas.py

echo ""
echo "=== Step 3: model tests ==="
python3 tests/test_models.py

echo ""
echo "=== Step 4: pilot ==="
python3 tests/pilot.py --gpu 0

echo ""
echo "=== Setup complete ==="
echo "Start experiments: bash scripts/run.sh all"
