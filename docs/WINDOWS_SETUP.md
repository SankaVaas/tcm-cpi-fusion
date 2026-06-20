# Windows Setup Guide

## Known Windows Issues & Fixes

### 1. `torch-scatter` DLL crash (`code 0xc0000139`)

**Do not install `torch-scatter` on Windows.** The pre-built wheels have broken DLLs
for most Python/torch version combinations. Our code only uses `GCNConv` and
`global_mean_pool` from PyG, which work fine **without** `torch-scatter`.

**Fix:** `setup_windows.bat` already skips it. If you installed it manually, remove it:
```bat
pip uninstall torch-scatter torch-sparse torch-cluster -y
```

### 2. `No module named 'esm'`

`fair-esm` doesn't install cleanly from PyPI on Windows. Install from source:
```bat
pip install git+https://github.com/facebookresearch/esm.git
```
Or skip it entirely — ESM embeddings are only needed for `scripts/03_precompute_embeddings.py`,
which is better run on Colab anyway.

### 3. `pip install --upgrade pip` fails in venv

Windows sometimes blocks pip self-upgrade inside venv. Run it as:
```bat
python -m pip install --upgrade pip
```
(not `pip install --upgrade pip`)

## Recommended Windows Workflow

```bat
REM 1. Setup
setup_windows.bat

REM 2. Activate
.venv\Scripts\activate

REM 3. Run tests (should all pass without esm/torch-scatter)
pytest tests/ -v

REM 4. Data pipeline (local CPU)
python scripts\01_download_data.py
python scripts\02_build_dataset.py
python scripts\04_run_baseline.py

REM 5. ESM embeddings + model training → do these on Google Colab
REM    Upload your data/processed/splits/ folder to Colab
REM    Run scripts/03 and 05 there
```

## Verified working on Windows with

- Python 3.11.x
- torch 2.1.0 (CPU wheel)
- torch-geometric (latest, no scatter/sparse)
- rdkit, scikit-learn, pandas, transformers
