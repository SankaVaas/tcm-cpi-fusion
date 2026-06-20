"""Training loop for DualStreamCPI."""
import torch
import torch.nn as nn
from tqdm import tqdm
from sklearn.metrics import roc_auc_score

try:
    import wandb
    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False


class CPITrainer:
    def __init__(self, model, optimizer, scheduler=None,
                 device: str = "cuda", use_wandb: bool = False):
        self.model     = model.to(device)
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device    = device
        self.use_wandb = use_wandb and HAS_WANDB
        self.criterion = nn.BCEWithLogitsLoss()

    def _move(self, batch):
        tokens, graph, prot, labels = batch
        tokens = {k: v.to(self.device) for k, v in tokens.items()}
        graph  = graph.to(self.device)
        prot   = prot.to(self.device)
        labels = labels.to(self.device)
        return tokens, graph, prot, labels

    def train_epoch(self, loader) -> float:
        self.model.train()
        total_loss = 0.0
        for batch in tqdm(loader, desc="train", leave=False):
            tokens, graph, prot, labels = self._move(batch)
            self.optimizer.zero_grad()
            logits = self.model(tokens, graph, prot)
            loss   = self.criterion(logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            total_loss += loss.item()
        return total_loss / len(loader)

    @torch.no_grad()
    def evaluate(self, loader) -> dict:
        self.model.eval()
        all_logits, all_labels = [], []
        for batch in loader:
            tokens, graph, prot, labels = self._move(batch)
            logits = self.model(tokens, graph, prot)
            all_logits.extend(logits.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())
        auc = roc_auc_score(all_labels, all_logits)
        return {"roc_auc": auc}

    def fit(self, train_loader, val_loader, epochs: int = 30):
        best_auc, best_state = 0.0, None
        for ep in range(1, epochs + 1):
            loss    = self.train_epoch(train_loader)
            metrics = self.evaluate(val_loader)
            print(f"Epoch {ep:03d} | loss={loss:.4f} | val_auc={metrics['roc_auc']:.4f}")
            if self.use_wandb:
                wandb.log({"epoch": ep, "train_loss": loss, **metrics})
            if metrics["roc_auc"] > best_auc:
                best_auc   = metrics["roc_auc"]
                best_state = {k: v.clone() for k, v in self.model.state_dict().items()}
            if self.scheduler:
                self.scheduler.step()
        print(f"Best val ROC-AUC: {best_auc:.4f}")
        self.model.load_state_dict(best_state)
        return self.model
