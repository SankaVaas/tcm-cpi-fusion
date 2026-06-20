# Model Architecture: DualStreamCPI

## Overview

```
   TCM Compound SMILES
          │
   ┌──────┴────────────────────┐
   │                           │
ChemBERTa-zinc            GCN Encoder
seyonec/ChemBERTa-zinc    GCNConv(6→128→256)
[B, 768]                  + global_mean_pool
   │                      [B, 256]
   └──────────┬───────────────┘
              │   concat
   ┌──────────┴───────────┐
   │   Protein Encoder    │
   │ ESM-2 (precomputed)  │
   │ mean-pool → Linear   │
   │ [B, 256]             │
   └──────────┬───────────┘
              │  [B, 1280]
         ┌────┴─────┐
         │ MLP Head │
         │ 1280→512 │ BN+ReLU+Drop
         │  512→128 │ ReLU+Drop
         │  128→1   │
         └────┬─────┘
              │
        Binding score (logit)
```

## Components

| Component | Model | Dim | Trainable |
|-----------|-------|-----|-----------|
| ChemBERTa | seyonec/ChemBERTa-zinc-base-v1 | 768 | Yes (fine-tuned) |
| GCN | GCNConv ×2 + global_mean_pool | 256 | Yes |
| Protein encoder | ESM-2 8M (pre-computed) + Linear | 256 | Linear only |
| Fusion head | Linear ×3 + BN + Dropout | 1→scalar | Yes |

## Training Details

- **Loss**: BCEWithLogitsLoss (pos_weight=5.0 for class imbalance)
- **Optimiser**: AdamW (lr=2e-4, weight_decay=1e-5)
- **Scheduler**: CosineAnnealingLR
- **Batch size**: 32
- **Epochs**: 50

## Baseline Comparison

| Model | Input | ROC-AUC (target) |
|-------|-------|-----------------|
| Random Forest | Morgan FP (2048-bit) | ~0.78 |
| DualStreamCPI (ours) | SMILES + graph + protein | ~0.88 |
