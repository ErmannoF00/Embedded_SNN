"""Supervised ALIF training with held-out validation and final untouched test.

The test partition is evaluated exactly once, after the final epoch. Training
never uses test/validation examples to fit normalization statistics.
"""
import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, f1_score

from .dataset import load_labeled_windows, synthetic_demo
from .features import extract_features
from .model import AdaptiveLIF
from .splitting import split_indices


def run_training(args):
    if args.demo:
        windows, labels, groups = synthetic_demo(seed=args.seed)
        origin = "synthetic oscillatory demo (not real EEG)"
    else:
        windows, labels, groups = load_labeled_windows(args.data)
        origin = str(args.data)
    train_idx, val_idx, test_idx = split_indices(labels, groups, args.seed)
    raw = extract_features(windows)
    mean = raw[train_idx].mean(axis=0, dtype=np.float32)
    std = np.maximum(raw[train_idx].std(axis=0, dtype=np.float32), np.float32(1e-6))
    features = ((raw - mean) / std).astype(np.float32)

    torch.set_num_threads(1)
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    model = AdaptiveLIF()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    train_x = torch.from_numpy(features[train_idx])
    val_x = torch.from_numpy(features[val_idx])
    test_x = torch.from_numpy(features[test_idx])
    train_y = torch.from_numpy(labels[train_idx])
    val_y = torch.from_numpy(labels[val_idx])
    batch_loss = float("nan")

    for epoch in range(args.epochs):
        model.train()
        for batch in torch.randperm(len(train_idx)).split(args.batch_size):
            optimizer.zero_grad(set_to_none=True)
            scores, _ = model(train_x[batch])
            loss = torch.nn.functional.cross_entropy(scores, train_y[batch])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            batch_loss = loss.item()
        if (epoch + 1) % 5 == 0 or epoch == 0:
            model.eval()
            with torch.no_grad():
                scores, spikes = model(val_x)
                val_accuracy = (scores.argmax(1) == val_y).float().mean().item()
                mean_spikes = spikes.mean().item()
            print(f"epoch={epoch+1:3d} loss={batch_loss:.4f} "
                  f"validation_accuracy={val_accuracy:.3f} spikes/neuron={mean_spikes:.2f}")

    # A test score is reported only once. No epoch/hyperparameter selection by test.
    model.eval()
    with torch.no_grad():
        scores, spikes = model(test_x)
        prediction = scores.argmax(dim=1).numpy()
    true = labels[test_idx]
    metrics = {
        "dataset": origin, "seed": args.seed, "epochs": args.epochs,
        "split": {"train": len(train_idx), "validation": len(val_idx), "test": len(test_idx)},
        "group_disjoint": groups is not None,
        "test_accuracy": float((prediction == true).mean()),
        "test_balanced_accuracy": float(balanced_accuracy_score(true, prediction)),
        "test_macro_f1": float(f1_score(true, prediction, average="macro", zero_division=0)),
        "confusion_matrix_rows_true": confusion_matrix(true, prediction, labels=[0, 1]).tolist(),
        "mean_hidden_spikes_per_neuron": float(spikes.mean().item()),
        "interpretation": "Synthetic integration test; NOT EEG performance" if args.demo else
            "Held-out windows; EEG generalization depends on independent subject/recording groups",
    }
    args.output.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "mean": torch.from_numpy(mean.copy()),
                "std": torch.from_numpy(std.copy()), "origin": origin, "seed": args.seed},
               args.output / "checkpoint.pt")
    np.savez_compressed(args.output / "holdout.npz",
                        raw_windows=windows[test_idx],
                        normalized_features=features[test_idx],
                        labels=true, python_scores=scores.numpy())
    (args.output / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    print(json.dumps(metrics, indent=2))
    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    origin = parser.add_mutually_exclusive_group(required=True)
    origin.add_argument("--demo", action="store_true", help="Labeled synthetic signals")
    origin.add_argument("--data", type=Path, help="Real labeled NPZ: windows, labels, groups")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=0.003)
    parser.add_argument("--seed", type=int, default=41)
    parser.add_argument("--output", type=Path, default=Path("artifacts"))
    run_training(parser.parse_args())
