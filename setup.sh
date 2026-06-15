#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────
#  tcm-cpi-fusion  —  one-command environment setup
#  No conda needed. Uses Python venv + pip only.
#  Usage:  bash setup.sh
# ─────────────────────────────────────────────────────────
set -e

PY=${PYTHON:-python3}
VENV=".venv"

echo "==> Checking Python version..."
$PY -c "import sys; assert sys.version_info >= (3,10), 'Need Python 3.10+'" \
  || { echo "ERROR: Python 3.10+ required. Install from https://python.org"; exit 1; }

echo "==> Creating virtual environment in $VENV/ ..."
$PY -m venv $VENV

echo "==> Activating..."
source $VENV/bin/activate

echo "==> Upgrading pip + wheel..."
pip install --upgrade pip wheel setuptools -q

echo "==> Installing PyTorch (CPU build — no CUDA locally)..."
pip install torch==2.1.0 torchvision --index-url https://download.pytorch.org/whl/cpu -q

echo "==> Installing PyTorch Geometric (CPU)..."
pip install torch-geometric -q
pip install torch-scatter torch-sparse torch-cluster \
  -f https://data.pyg.org/whl/torch-2.1.0+cpu.html -q

echo "==> Installing all other dependencies..."
pip install -r requirements.txt -q

echo "==> Installing this package in editable mode..."
pip install -e . -q

echo ""
echo "✅  Setup complete!"
echo ""
echo "   Activate your environment any time with:"
echo "     source .venv/bin/activate"
echo ""
echo "   Then run the pipeline:"
echo "     python scripts/01_download_data.py"
echo "     python scripts/02_build_dataset.py"
echo "     python scripts/03_precompute_embeddings.py"
echo "     python scripts/04_run_baseline.py"
echo "     python scripts/05_train_model.py   ← run this one on Colab (GPU)"
