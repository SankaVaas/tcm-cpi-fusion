# ─────────────────────────────────────────────────────────
#  Paste this into the FIRST cell of any Colab notebook.
#  Installs all dependencies on a fresh Colab T4 runtime.
# ─────────────────────────────────────────────────────────
import subprocess, sys

def run(cmd): subprocess.run(cmd, shell=True, check=True)

# Clone your repo (replace with your actual GitHub URL once pushed)
# run("git clone https://github.com/YOUR_USERNAME/tcm-cpi-fusion.git")
# %cd tcm-cpi-fusion

# PyTorch (Colab usually has this, but pin the version)
run("pip install torch==2.1.0 torchvision --quiet")

# PyTorch Geometric (GPU wheels for Colab T4 — CUDA 11.8)
run("pip install torch-geometric --quiet")
run("pip install torch-scatter torch-sparse torch-cluster "
    "-f https://data.pyg.org/whl/torch-2.1.0+cu118.html --quiet")

# Everything else
run("pip install transformers datasets rdkit deepchem pubchempy "
    "fair-esm pandas numpy scipy scikit-learn meeko vina "
    "matplotlib seaborn plotly wandb tqdm pyyaml requests joblib --quiet")

# Install the project itself
run("pip install -e . --quiet")

print("✅ Colab environment ready!")
