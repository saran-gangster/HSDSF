"""Generate evaluation plots for a trained anomaly model.

Produces:
- Reconstruction error distribution (benign vs trojan)
- ROC curve
- Precision-Recall curve

Uses artifacts produced by phase4/train_anomaly.py.
"""

import argparse
import json
import pickle
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import (
    auc,
    average_precision_score,
    precision_recall_curve,
    roc_curve,
)
from torch.utils.data import DataLoader

from phase4.train_anomaly import (
    LSTMAutoencoder,
    build_windows,
    is_trojan_label,
    load_runs,
    sanitize_numeric,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Plot evaluation metrics for Phase 4 model")
    p.add_argument("--artifacts-dir", type=Path, default=Path("phase4/artifacts"))
    p.add_argument("--runs-dir", type=Path, default=None)
    p.add_argument("--output-dir", type=Path, default=None)
    p.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    p.add_argument("--batch-size", type=int, default=256)
    return p.parse_args()


def load_artifacts(artifacts_dir: Path) -> Tuple[Dict, List[str], float, object]:
    config = json.loads((artifacts_dir / "config.json").read_text())
    features = json.loads((artifacts_dir / "features.json").read_text())
    threshold = json.loads((artifacts_dir / "threshold.json").read_text())["threshold"]
    with open(artifacts_dir / "scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    return config, features, float(threshold), scaler


def reconstruction_errors(model: torch.nn.Module, windows: np.ndarray, device: torch.device, batch_size: int) -> np.ndarray:
    model.eval()
    loader = DataLoader(torch.from_numpy(windows), batch_size=batch_size, shuffle=False)
    criterion = torch.nn.MSELoss(reduction="none")
    errors: List[float] = []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            recon = model(batch)
            loss = criterion(recon, batch).mean(dim=(1, 2))
            errors.extend(loss.detach().cpu().numpy().tolist())
    return np.asarray(errors, dtype=np.float32)


def label_targets(
    labels: Sequence[str],
    trojan_flags: np.ndarray,
    anomaly_mode: str,
    train_labels: Sequence[str],
) -> np.ndarray:
    if anomaly_mode == "trojan":
        return np.asarray([is_trojan_label(lbl) or bool(t) for lbl, t in zip(labels, trojan_flags)], dtype=int)
    return np.asarray([lbl not in train_labels or bool(t) for lbl, t in zip(labels, trojan_flags)], dtype=int)


def main() -> None:
    args = parse_args()

    config, features, threshold, scaler = load_artifacts(args.artifacts_dir)

    runs_dir = Path(config["runs_dir"]) if args.runs_dir is None else args.runs_dir
    output_dir = (args.artifacts_dir / "plots") if args.output_dir is None else args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(
        "cuda" if args.device == "cuda" or (args.device == "auto" and torch.cuda.is_available()) else "cpu"
    )

    df = load_runs(runs_dir)
    df = sanitize_numeric(df, features)

    window_size = int(config["window_size"])
    window_stride = int(config["window_stride"])

    windows, labels, run_ids, trojan_flags = build_windows(df, features, window_size=window_size, stride=window_stride)

    # Normalize with saved scaler
    windows_scaled = windows.reshape(-1, windows.shape[2])
    windows_scaled = scaler.transform(windows_scaled)
    windows_scaled = windows_scaled.reshape(windows.shape).astype(np.float32)

    # Load model
    model = LSTMAutoencoder(
        input_dim=len(features),
        hidden_dim=int(config.get("hidden_dim", 64)),
        latent_dim=int(config.get("latent_dim", 32)),
    )
    state = torch.load(args.artifacts_dir / "model.pt", map_location="cpu")
    model.load_state_dict(state)
    model.to(device)

    errors = reconstruction_errors(model, windows_scaled, device=device, batch_size=args.batch_size)

    anomaly_mode = config.get("anomaly_mode", "nontrain")
    train_labels = config.get("train_labels", ["idle", "normal"])
    benign_eval_labels = config.get("benign_eval_labels", train_labels)

    y_true = label_targets(labels, trojan_flags, anomaly_mode=anomaly_mode, train_labels=train_labels)

    # Define benign set for the distribution plot
    benign_mask = np.asarray(
        [(lbl in benign_eval_labels) and (not is_trojan_label(lbl)) and (not bool(t)) for lbl, t in zip(labels, trojan_flags)],
        dtype=bool,
    )
    trojan_mask = y_true.astype(bool)

    benign_errors = errors[benign_mask]
    trojan_errors = errors[trojan_mask]

    # 1) Reconstruction error distribution
    plt.figure(figsize=(8, 5))
    bins = 60
    plt.hist(benign_errors, bins=bins, alpha=0.6, density=True, label=f"Benign ({len(benign_errors)})")
    plt.hist(trojan_errors, bins=bins, alpha=0.6, density=True, label=f"Trojan/anomaly ({len(trojan_errors)})")
    plt.axvline(threshold, color="k", linestyle="--", linewidth=1.5, label=f"Threshold = {threshold:.3f}")
    plt.title("Reconstruction Error Distribution")
    plt.xlabel("Reconstruction MSE")
    plt.ylabel("Density")
    plt.legend()
    plt.tight_layout()
    out1 = output_dir / "reconstruction_error_distribution.png"
    plt.savefig(out1, dpi=160)
    plt.close()

    # 2) ROC
    if len(np.unique(y_true)) > 1:
        fpr, tpr, _ = roc_curve(y_true, errors)
        roc_auc = auc(fpr, tpr)
        plt.figure(figsize=(6, 6))
        plt.plot(fpr, tpr, label=f"ROC AUC = {roc_auc:.3f}")
        plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
        plt.title("ROC Curve")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.legend(loc="lower right")
        plt.tight_layout()
        out2 = output_dir / "roc_curve.png"
        plt.savefig(out2, dpi=160)
        plt.close()

        # 3) Precision-Recall
        precision, recall, _ = precision_recall_curve(y_true, errors)
        ap = average_precision_score(y_true, errors)
        plt.figure(figsize=(6, 6))
        plt.plot(recall, precision, label=f"AP = {ap:.3f}")
        plt.title("Precision-Recall Curve")
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.legend(loc="lower left")
        plt.tight_layout()
        out3 = output_dir / "pr_curve.png"
        plt.savefig(out3, dpi=160)
        plt.close()
    else:
        print("Skipping ROC/PR: y_true has only one class")

    print(f"Wrote: {out1}")
    if len(np.unique(y_true)) > 1:
        print(f"Wrote: {output_dir / 'roc_curve.png'}")
        print(f"Wrote: {output_dir / 'pr_curve.png'}")


if __name__ == "__main__":
    main()
