#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
#  tcm-cpi-fusion — environment setup (Linux / macOS / Git Bash on Windows)
#  No conda needed. Pure Python venv + pip.
#
#  Usage:
#    Linux/macOS:  bash setup.sh
#    Windows:      run setup_windows.bat  instead
# ─────────────────────────────────────────────────────────────────────────────
set -e

PY=${PYTHON:-python3}
VENV=".venv"

echo "==> Checking Python version..."
$PY -c "import sys; assert sys.version_info >= (3,10), 'Need Python 3.10+'"

echo "==> Creating virtual environment..."
$PY -m venv $VENV
source $VENV/bin/activate

echo "==> Upgrading pip..."
python -m pip install --upgrade pip wheel setuptools -q

echo "==> Installing PyTorch (CPU)..."
pip install torch==2.1.0 --index-url https://download.pytorch.org/whl/cpu -q

echo "==> Installing PyTorch Geometric..."
pip install torch-geometric -q

echo "==> Installing remaining dependencies..."
pip install -r requirements.txt -q

echo "==> Installing project in editable mode..."
pip install -e . -q

echo ""
echo "✅  Done!  Activate with:  source .venv/bin/activate"
