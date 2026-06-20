# Data Sources

All datasets below are **freely accessible without approval or waiting**.

| Dataset | URL | License | Size |
|---------|-----|---------|------|
| TCMSP | https://tcmsp-e.com | ODbL | 29,384 compounds, 3,311 targets |
| BindingDB | https://www.bindingdb.org/bind/downloads | CC BY 3.0 | ~2.8M interactions |
| ChEMBL | https://chembl.ebi.ac.uk | CC BY-SA 3.0 | ~2.4M compounds |
| UniProt REST | https://rest.uniprot.org | CC BY 4.0 | 250M+ entries |
| RCSB PDB | https://rcsb.org | Open | 220k+ structures |
| PubChem | via pubchempy | Public Domain | 116M+ compounds |
| ESM-2 (8M) | https://github.com/facebookresearch/esm | MIT | 8M params |
| ChemBERTa-zinc | https://huggingface.co/seyonec/ChemBERTa-zinc-base-v1 | Apache 2.0 | 84M params |

---

## TCMSP Manual Download (one-time, ~5 min)

1. Go to https://tcmsp-e.com/tcmsp.php
2. **Browse Molecules** → click **Download** (CSV) → save as:
   ```
   data/raw/tcmsp/herb_compounds.csv
   ```
3. **Browse Targets** → Download → save as:
   ```
   data/raw/tcmsp/compound_targets.csv
   ```

---

## BindingDB (auto-downloaded by script 01)

Approximately 2 GB compressed. Script `01_download_data.py` handles this automatically.

---

## UniProt & PDB

Fetched on-demand via REST API in scripts `03` and docking module. No download needed.
