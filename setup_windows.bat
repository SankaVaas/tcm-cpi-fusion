@echo off
REM ─────────────────────────────────────────────────────────────────────────
REM  tcm-cpi-fusion — Windows setup (no conda, no admin rights needed)
REM  Run from the repo root:  setup_windows.bat
REM ─────────────────────────────────────────────────────────────────────────

echo =^> Creating virtual environment...
python -m venv .venv
call .venv\Scripts\activate.bat

echo =^> Upgrading pip...
python -m pip install --upgrade pip wheel setuptools

echo =^> Installing PyTorch (CPU, Windows)...
pip install torch==2.1.0 --index-url https://download.pytorch.org/whl/cpu

echo =^> Installing PyTorch Geometric (Windows-safe, no torch-scatter)...
pip install torch-geometric

REM Note: torch-scatter/sparse have broken Windows DLLs for many builds.
REM PyG works fine without them for GCNConv + global_mean_pool (which we use).
REM DO NOT install torch-scatter on Windows.

echo =^> Installing all other dependencies...
pip install -r requirements.txt

echo =^> Installing project...
pip install -e .

echo.
echo Done! Activate any time with:
echo   .venv\Scripts\activate
