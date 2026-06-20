# TCM-CPI-Fusion 🧬

**Dual-stream ChemBERTa + GCN for Traditional Chinese Medicine Compound–Protein Interaction Prediction**

> PhD research project · Computational Chemistry & Medical AI  
> Platform: **Google Colab T4** (training) + **Local CPU** (preprocessing, docking, inference)

---

## Research Overview

Predicts binding between TCM bioactive compounds and disease-related protein targets using a
dual-stream deep learning model combining:

- **ChemBERTa** — SMILES-based molecular transformer (84M params)
- **Graph Convolutional Network** — molecular topology encoder
- **ESM-2** — protein language model (8M, CPU-feasible)

Top predictions are validated with CPU-native **AutoDock Vina** docking.

**Novelty**: most TCM-AI studies use single-modality encoders; this is the first dual-stream
ChemBERTa+GCN fusion trained on TCMSP-enriched CPI data, with docking validation.

---

## Quick Start

### 1. Clone & install (no conda needed — pure Python venv)

```bash
git clone https://github.com/YOUR_USERNAME/tcm-cpi-fusion
cd tcm-cpi-fusion
bash setup.sh          # creates .venv/, installs everything
source .venv/bin/activate
```

> **On Colab**: paste the contents of `notebooks/colab_setup.py` into your first cell instead.

### 2. Five-step pipeline

```bash
# Step 1 — Download BindingDB (auto) + manually export TCMSP CSVs
python scripts/01_download_data.py

# Step 2 — Parse + merge → train/val/test splits  (~15 min, local CPU)
python scripts/02_build_dataset.py

# Step 3 — Fetch UniProt sequences + ESM-2 embeddings  (Colab GPU recommended)
python scripts/03_precompute_embeddings.py

# Step 4 — Random Forest baseline  (~10 min, local CPU)
python scripts/04_run_baseline.py

# Step 5 — Train DualStreamCPI  (~2 hrs on Colab T4)
python scripts/05_train_model.py
```

### 3. Run tests

```bash
pytest tests/ -v
```

---

## Repository Structure

```
tcm-cpi-fusion/
├── data/
│   ├── raw/           tcmsp/ bindingdb/ chembl/ uniprot/ pdb/
│   └── processed/     graphs/ embeddings/ splits/ docking/
├── src/
│   ├── data/          download.py  tcmsp_parser.py  bindingdb_parser.py  dataset.py
│   ├── features/      mol_graph.py  protein_embed.py  fingerprints.py
│   ├── models/        components.py  dual_stream.py  baseline.py
│   ├── training/      trainer.py  losses.py
│   ├── evaluation/    metrics.py  explainability.py
│   ├── docking/       pdb_fetcher.py  vina_runner.py
│   └── utils/         seed.py  io.py  pubchem.py  uniprot.py
├── notebooks/
│   ├── 01_data_exploration/   01_tcmsp_eda  02_bindingdb_eda
│   ├── 02_baseline/           03_baseline_rf
│   ├── 03_model_training/     04_dual_stream_colab
│   └── 04_analysis/           05_results  06_gradcam  07_docking
├── scripts/           01_download → 05_train (sequential pipeline)
├── experiments/
│   ├── configs/       default.yaml  baseline.yaml
│   ├── checkpoints/   (gitignored)
│   └── results/       JSON metrics per experiment
├── tests/             pytest unit tests
└── docs/              DATA_SOURCES.md  ARCHITECTURE.md
```

---

## Data Sources (all instant-access, no approval needed)

| Source | Records | How |
|--------|---------|-----|
| TCMSP | 29,384 compounds | Manual CSV export (5 min) |
| BindingDB | ~2.8M interactions | Auto-downloaded by script 01 |
| UniProt REST | Any protein | On-demand API |
| RCSB PDB | Any structure | On-demand API |
| ChemBERTa weights | HuggingFace Hub | Auto via `transformers` |
| ESM-2 weights | GitHub releases | Auto via `fair-esm` |

---

## Expected Results

| Model | ROC-AUC | PR-AUC | EF@1% |
|-------|---------|--------|-------|
| RF Baseline | ~0.78 | ~0.55 | ~8×  |
| **DualStreamCPI** | **~0.88** | **~0.72** | **~15×** |

---

## Compute Split

| Task | Where | Time |
|------|-------|------|
| Data preprocessing | Local CPU | ~20 min |
| ESM-2 embedding (8M) | Local CPU | ~2 hrs |
| RF Baseline | Local CPU | ~10 min |
| DualStreamCPI training | Colab T4 | ~2 hrs |
| AutoDock Vina (100 hits) | Local CPU | ~3 hrs |
| Inference on new compounds | Local CPU | <1 sec/mol |

---

## License

MIT. Data source licenses vary — see [docs/DATA_SOURCES.md](docs/DATA_SOURCES.md).
