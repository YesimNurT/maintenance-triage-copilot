"""Training loop for the sequence baseline: seeded, early-stopped on a train hold-out."""

import time
from collections.abc import Callable, Sequence

import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from torch import nn
from torch.utils.data import DataLoader, Dataset

from mtc.models.sequence_data import normalise


def pick_device(name: str) -> torch.device:
    """``auto`` picks Apple MPS, then CUDA, then CPU; any other name is used as given."""
    if name != "auto":
        return torch.device(name)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)


class SequenceDataset(Dataset):
    """Normalises one window at a time, so the full array can stay on disk."""

    def __init__(
        self,
        sequences: np.ndarray,
        rows: Sequence[int],
        targets: Sequence[int],
        mean: np.ndarray,
        std: np.ndarray,
        channels: Sequence[int],
    ):
        self.sequences, self.rows = sequences, np.asarray(rows)
        self.targets = np.asarray(targets, dtype=np.float32)
        self.channels = list(channels)
        self.mean, self.std = mean[self.channels], std[self.channels]

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, i: int) -> tuple[torch.Tensor, torch.Tensor]:
        window = np.asarray(self.sequences[self.rows[i]])[:, self.channels]
        x = normalise(window, self.mean, self.std).T  # [channels, time]
        return torch.from_numpy(np.ascontiguousarray(x)), torch.tensor(self.targets[i])


@torch.no_grad()
def predict(model: nn.Module, loader: DataLoader, device: torch.device) -> np.ndarray:
    """Probability of class 1 for every sequence, in loader order."""
    model.eval()
    scores = [torch.sigmoid(model(x.to(device))).cpu().numpy() for x, _ in loader]
    return np.concatenate(scores)


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    holdout_loader: DataLoader,
    holdout_targets: np.ndarray,
    epochs: int,
    learning_rate: float,
    patience: int,
    device: torch.device,
    log: Callable[[str], None] = print,
) -> list[dict[str, float]]:
    """Train with Adam and BCE; keep the weights of the best hold-out AUC.

    Stops after ``patience`` epochs without improvement. Returns one record per epoch.
    """
    model.to(device)
    optimiser = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_fn = nn.BCEWithLogitsLoss()
    best_auc, best_state, stale, history = -1.0, None, 0, []
    for epoch in range(1, epochs + 1):
        start = time.perf_counter()
        model.train()
        total, n = 0.0, 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimiser.zero_grad()
            loss = loss_fn(model(x), y)
            loss.backward()
            optimiser.step()
            total, n = total + loss.item() * len(y), n + len(y)
        auc = float(roc_auc_score(holdout_targets, predict(model, holdout_loader, device)))
        seconds = time.perf_counter() - start
        history.append({"epoch": epoch, "loss": total / n, "holdout_auc": auc, "seconds": seconds})
        log(f"epoch {epoch:2d}  loss {total / n:.4f}  holdout auc {auc:.3f}  {seconds:.0f} s")
        if auc > best_auc:
            best_auc, stale = auc, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            stale += 1
            if stale >= patience:
                break
    model.load_state_dict(best_state)
    return history
