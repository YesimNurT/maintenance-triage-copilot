"""F2.3: InceptionTime on 1 Hz windows, per channel set (needs build_sequences).

    uv run python scripts/f2_sequence_baseline.py --channels without_oil
    uv run python scripts/f2_sequence_baseline.py --channels all
    uv run python scripts/f2_sequence_baseline.py --channels oil_only

Early stopping uses a hold-out taken from train; metrics are on val. Test is not read.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from mtc.config import Settings, get_settings
from mtc.models.inception import InceptionTime
from mtc.models.sequence_data import CHANNEL_SETS, channel_indices, fit_channel_stats
from mtc.models.signal_check import auc_by_group, bootstrap_auc_ci, evaluate
from mtc.models.train import SequenceDataset, pick_device, predict, set_seed, train_model

N_BOOT = 1000
MIN_AUC = 0.55  # docs/DECISIONS.md, 2026-10-04


def run(settings: Settings, channels: str) -> Path:
    set_seed(settings.random_seed)
    sequences = np.load(settings.processed_dir / "sequences.npy", mmap_mode="r")
    index = pd.read_parquet(settings.processed_dir / "sequences_index.parquet")
    train_rows = np.flatnonzero(index["split"] == "train")
    val_rows = np.flatnonzero(index["split"] == "val")
    y = index["y"].to_numpy()
    fit_rows, holdout_rows = train_test_split(
        train_rows,
        test_size=settings.seq_holdout_share,
        stratify=y[train_rows],
        random_state=settings.random_seed,
    )
    mean, std = fit_channel_stats(sequences, fit_rows)
    selected = channel_indices(channels)

    def loader(rows: np.ndarray, shuffle: bool) -> DataLoader:
        dataset = SequenceDataset(sequences, rows, y[rows], mean, std, selected)
        return DataLoader(dataset, batch_size=settings.seq_batch_size, shuffle=shuffle)

    device = pick_device(settings.device)
    model = InceptionTime(len(selected), settings.seq_filters, settings.seq_depth)
    print(f"{channels}: {len(selected)} channels, fit {len(fit_rows)}, holdout "
          f"{len(holdout_rows)}, val {len(val_rows)}, device {device}")
    history = train_model(
        model,
        loader(fit_rows, shuffle=True),
        loader(holdout_rows, shuffle=False),
        y[holdout_rows],
        settings.seq_epochs,
        settings.seq_learning_rate,
        settings.seq_patience,
        device,
    )

    score = predict(model, loader(val_rows, shuffle=False), device)
    y_val = y[val_rows]
    metrics = evaluate(y_val, score)
    metrics["auc_ci95"] = bootstrap_auc_ci(y_val, score, N_BOOT, settings.random_seed)
    report = {
        "channels": CHANNEL_SETS[channels],
        "n_fit": int(len(fit_rows)),
        "n_holdout": int(len(holdout_rows)),
        "n_val": int(len(val_rows)),
        "majority_accuracy": float(max(y_val.mean(), 1 - y_val.mean())),
        "val": metrics,
        "val_auc_by_label": auc_by_group(y_val, score, index["label"].iloc[val_rows]),
        "signal_present": bool(metrics["auc_ci95"][0] > MIN_AUC),
        "history": history,
    }
    low, high = metrics["auc_ci95"]
    print(f"val auc {metrics['auc']:.3f} [{low:.3f}, {high:.3f}]  acc {metrics['accuracy']:.3f}  "
          f"majority {report['majority_accuracy']:.3f}  signal_present {report['signal_present']}")

    out = settings.results_dir / "f2" / f"sequence_{channels}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))
    print(f"written {out}")
    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--channels", choices=sorted(CHANNEL_SETS), default="without_oil")
    run(get_settings(), parser.parse_args().channels)
