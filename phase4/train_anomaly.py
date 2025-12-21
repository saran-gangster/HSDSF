"""Train an LSTM autoencoder for telemetry anomaly detection.

This script consumes Phase 3 telemetry runs, builds sliding windows,
trains an unsupervised reconstruction model on benign classes, and
exports artifacts (model, scaler, threshold, ONNX optional).
"""

import argparse
import json
import math
import pickle
import random
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, Dataset


def is_trojan_label(label: str) -> bool:
    return str(label).startswith("trojan")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class LSTMAutoencoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 64, latent_dim: int = 32):
        super().__init__()
        self.encoder = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.to_latent = nn.Linear(hidden_dim, latent_dim)
        self.from_latent = nn.Linear(latent_dim, hidden_dim)
        self.decoder = nn.LSTM(hidden_dim, hidden_dim, batch_first=True)
        self.output = nn.Linear(hidden_dim, input_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        enc_out, _ = self.encoder(x)
        latent = self.to_latent(enc_out[:, -1, :])
        dec_seed = self.from_latent(latent).unsqueeze(1)
        dec_in = dec_seed.repeat(1, x.size(1), 1)
        dec_out, _ = self.decoder(dec_in)
        return self.output(dec_out)


def read_trojan_intervals(path: Path) -> List[Tuple[float, float]]:
    if not path.exists():
        return []
    intervals = []
    for line in path.read_text().splitlines():
        parts = line.strip().split(",")
        if len(parts) != 2:
            continue
        try:
            start = float(parts[0])
            end = float(parts[1])
            intervals.append((start, end))
        except ValueError:
            continue
    return intervals


def mark_trojan_active(df: pd.DataFrame, intervals: Sequence[Tuple[float, float]]) -> pd.Series:
    if not intervals:
        return pd.Series(False, index=df.index)
    mask = np.zeros(len(df), dtype=bool)
    t = df["t_wall"].to_numpy()
    for start, end in intervals:
        mask |= (t >= start) & (t <= end)
    return pd.Series(mask, index=df.index)


def load_runs(runs_dir: Path) -> pd.DataFrame:
    frames = []
    for run_dir in sorted(p for p in runs_dir.iterdir() if p.is_dir()):
        telemetry_path = run_dir / "telemetry.csv"
        if not telemetry_path.exists():
            continue
        df = pd.read_csv(telemetry_path)
        df["run_id"] = run_dir.name
        intervals = read_trojan_intervals(run_dir / "trojan_intervals.csv")
        df["trojan_active"] = mark_trojan_active(df, intervals)
        frames.append(df)
    if not frames:
        raise FileNotFoundError(f"No runs found under {runs_dir}")
    return pd.concat(frames, ignore_index=True)


def select_feature_columns(df: pd.DataFrame) -> List[str]:
    drop_cols = {"label", "run_id", "t_wall", "ts_unix", "trojan_active"}
    candidates = [c for c in df.columns if c not in drop_cols]
    numeric = [c for c in candidates if pd.api.types.is_numeric_dtype(df[c])]
    return numeric


def sanitize_numeric(df: pd.DataFrame, features: Sequence[str]) -> pd.DataFrame:
    """Replace inf/NaN, forward/back fill, then zero-fill remaining gaps."""
    clean = df.copy()
    clean[features] = clean[features].replace([np.inf, -np.inf], np.nan)
    clean[features] = clean[features].ffill().bfill().fillna(0.0)
    return clean


def build_windows(
    df: pd.DataFrame,
    features: Sequence[str],
    window_size: int,
    stride: int,
) -> Tuple[np.ndarray, List[str], List[str], np.ndarray]:
    windows: List[np.ndarray] = []
    labels: List[str] = []
    run_ids: List[str] = []
    trojan_flags: List[bool] = []
    for run_id, group in df.groupby("run_id"):
        g = group.sort_values("t_wall")
        x = g[features].to_numpy(dtype=np.float32)
        trojan = g["trojan_active"].to_numpy(dtype=bool)
        label = g["label"].iloc[0]
        for start in range(0, len(g) - window_size + 1, stride):
            end = start + window_size
            window = x[start:end]
            trojan_window = trojan[start:end].any()
            windows.append(window)
            labels.append(label)
            run_ids.append(run_id)
            trojan_flags.append(bool(trojan_window))
    if not windows:
        raise ValueError("No windows generated; check window_size/stride vs run lengths")
    return (
        np.stack(windows),
        labels,
        run_ids,
        np.asarray(trojan_flags, dtype=bool),
    )


class WindowDataset(Dataset):
    def __init__(self, windows: np.ndarray):
        self.windows = torch.from_numpy(windows)

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, idx: int) -> torch.Tensor:
        return self.windows[idx]


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    epochs: int,
    lr: float,
) -> Tuple[List[float], List[float]]:
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    train_losses: List[float] = []
    val_losses: List[float] = []
    for epoch in range(epochs):
        model.train()
        total = 0.0
        batches = 0
        for batch in train_loader:
            batch = batch.to(device)
            recon = model(batch)
            loss = criterion(recon, batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total += loss.item()
            batches += 1
        train_loss = total / max(1, batches)
        train_losses.append(train_loss)

        model.eval()
        vtotal = 0.0
        vbatches = 0
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                recon = model(batch)
                loss = criterion(recon, batch)
                vtotal += loss.item()
                vbatches += 1
        val_loss = vtotal / max(1, vbatches)
        val_losses.append(val_loss)
        print(f"Epoch {epoch + 1}/{epochs} train={train_loss:.4f} val={val_loss:.4f}")
    return train_losses, val_losses


def split_benign_runs(
    df: pd.DataFrame,
    train_labels: Sequence[str],
    val_fraction: float,
    test_fraction: float,
    seed: int,
) -> Tuple[List[str], List[str], List[str]]:
    """Split benign runs into train/val/test by run_id.

    Trojan runs (any trojan_active=True) are excluded from benign splits.
    """
    run_meta = (
        df.groupby("run_id")
        .agg(label=("label", "first"), trojan_any=("trojan_active", "any"))
        .reset_index()
    )
    benign_runs = run_meta[(run_meta["label"].isin(train_labels)) & (~run_meta["trojan_any"])][
        "run_id"
    ].tolist()
    if len(benign_runs) < 3:
        raise ValueError(
            f"Need at least 3 benign runs for train/val/test split; found {len(benign_runs)}"
        )

    rng = np.random.default_rng(seed)
    rng.shuffle(benign_runs)

    n = len(benign_runs)
    n_val = max(1, int(round(n * val_fraction)))
    n_test = max(1, int(round(n * test_fraction)))
    if n_val + n_test >= n:
        # Ensure at least one train run remains.
        n_test = max(1, n - n_val - 1)
        if n_val + n_test >= n:
            n_val = max(1, n - n_test - 1)

    val_runs = benign_runs[:n_val]
    test_runs = benign_runs[n_val : n_val + n_test]
    train_runs = benign_runs[n_val + n_test :]
    return train_runs, val_runs, test_runs


def reconstruction_errors(model: nn.Module, loader: DataLoader, device: torch.device) -> np.ndarray:
    criterion = nn.MSELoss(reduction="none")
    errors: List[float] = []
    model.eval()
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            recon = model(batch)
            loss = criterion(recon, batch).mean(dim=(1, 2))
            errors.extend(loss.cpu().numpy().tolist())
    return np.asarray(errors, dtype=np.float32)


def save_artifacts(
    output_dir: Path,
    model: nn.Module,
    scaler: StandardScaler,
    features: Sequence[str],
    threshold: float,
    config: Dict,
    export_onnx: bool,
    window_size: int,
    onnx_opset: int,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), output_dir / "model.pt")
    with open(output_dir / "scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)
    (output_dir / "features.json").write_text(json.dumps(list(features), indent=2))
    (output_dir / "threshold.json").write_text(json.dumps({"threshold": threshold}, indent=2))
    (output_dir / "config.json").write_text(json.dumps(config, indent=2))
    if export_onnx:
        dummy = torch.zeros(1, window_size, len(features))
        try:
            torch.onnx.export(
                model,
                dummy,
                output_dir / "model.onnx",
                input_names=["input"],
                output_names=["reconstruction"],
                opset_version=onnx_opset,
            )
        except ModuleNotFoundError as exc:
            print("ONNX export skipped: install onnx and onnxscript (pip install onnx onnxscript)")
            print(f"Reason: {exc}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train anomaly model on Phase 3 telemetry")
    parser.add_argument("--runs-dir", type=Path, default=Path("phase3/data/runs"))
    parser.add_argument("--output-dir", type=Path, default=Path("phase4/artifacts"))
    parser.add_argument("--window-size", type=int, default=50, help="Samples per window (dt ~0.1s → 5s)")
    parser.add_argument("--window-stride", type=int, default=5, help="Stride between windows in samples")
    parser.add_argument("--train-labels", nargs="*", default=["idle", "normal"], help="Labels treated as benign for training")
    parser.add_argument("--val-fraction", type=float, default=0.2, help="Fraction of train windows used for validation")
    parser.add_argument("--test-fraction", type=float, default=0.2, help="Fraction of benign runs held out for testing")
    parser.add_argument(
        "--split-by",
        choices=["run", "window"],
        default="run",
        help="How to split benign data for train/val: by run (recommended) or by window",
    )
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--latent-dim", type=int, default=32)
    parser.add_argument("--threshold-percentile", type=float, default=99.0, help="Percentile of train errors used as threshold")
    parser.add_argument(
        "--threshold-source",
        choices=["train", "val"],
        default="train",
        help="Which benign split to use for threshold calibration",
    )
    parser.add_argument(
        "--anomaly-mode",
        choices=["nontrain", "trojan"],
        default="nontrain",
        help=(
            "Define what counts as anomalous in evaluation: "
            "'nontrain' = label not in --train-labels OR trojan_active; "
            "'trojan' = trojan label/intervals only"
        ),
    )
    parser.add_argument("--export-onnx", action="store_true")
    parser.add_argument("--onnx-opset", type=int, default=18)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = torch.device("cuda" if args.device == "cuda" or (args.device == "auto" and torch.cuda.is_available()) else "cpu")

    df = load_runs(args.runs_dir)
    feature_cols = select_feature_columns(df)
    df = sanitize_numeric(df, feature_cols)

    windows, labels, run_ids, trojan_flags = build_windows(
        df, feature_cols, window_size=args.window_size, stride=args.window_stride
    )

    benign_mask = np.array([lbl in args.train_labels and not trojan for lbl, trojan in zip(labels, trojan_flags)], dtype=bool)
    if benign_mask.sum() == 0:
        raise ValueError("No benign windows found for training; check train-labels")

    scaler = StandardScaler()
    train_flat = windows[benign_mask].reshape(-1, windows.shape[2])
    if np.isnan(train_flat).any():
        raise ValueError("Training data contains NaN after sanitization; check inputs")
    scaler.fit(train_flat)
    windows_scaled = windows.reshape(-1, windows.shape[2])
    windows_scaled = scaler.transform(windows_scaled)
    windows_scaled = windows_scaled.reshape(windows.shape)
    if np.isnan(windows_scaled).any():
        raise ValueError("Scaled windows contain NaN; aborting")

    if args.split_by == "window":
        benign_indices = np.nonzero(benign_mask)[0]
        np.random.shuffle(benign_indices)
        val_size = max(1, int(len(benign_indices) * args.val_fraction))
        val_indices = benign_indices[:val_size]
        train_indices = benign_indices[val_size:]
        train_set = set(train_indices.tolist())
        val_set = set(val_indices.tolist())
        test_indices = np.array([i for i in range(len(windows_scaled)) if i not in train_set and i not in val_set])
    else:
        train_runs, val_runs, test_runs = split_benign_runs(
            df=df,
            train_labels=args.train_labels,
            val_fraction=args.val_fraction,
            test_fraction=args.test_fraction,
            seed=args.seed,
        )
        train_set = set(train_runs)
        val_set = set(val_runs)
        test_set = set(test_runs)

        train_indices = np.array([i for i, rid in enumerate(run_ids) if rid in train_set and benign_mask[i]])
        val_indices = np.array([i for i, rid in enumerate(run_ids) if rid in val_set and benign_mask[i]])

        # Test includes: (a) benign test runs + (b) all non-benign runs (trojan labels or any trojan_active windows)
        test_indices = np.array(
            [
                i
                for i, (rid, lbl, trojan_any) in enumerate(zip(run_ids, labels, trojan_flags))
                if (rid in test_set) or (lbl not in args.train_labels) or trojan_any
            ]
        )

    train_ds = WindowDataset(windows_scaled[train_indices])
    val_ds = WindowDataset(windows_scaled[val_indices])
    test_ds = WindowDataset(windows_scaled[test_indices])

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)

    model = LSTMAutoencoder(len(feature_cols), hidden_dim=args.hidden_dim, latent_dim=args.latent_dim).to(device)

    train_losses, val_losses = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        epochs=args.epochs,
        lr=args.lr,
    )

    train_errors = reconstruction_errors(model, train_loader, device)
    val_errors = reconstruction_errors(model, val_loader, device)
    test_errors = reconstruction_errors(model, test_loader, device)

    calib_errors = train_errors if args.threshold_source == "train" else val_errors
    threshold = float(np.percentile(calib_errors, args.threshold_percentile))

    test_labels = [labels[i] for i in test_indices]
    test_trojan_flags = trojan_flags[test_indices]
    if args.anomaly_mode == "trojan":
        y_true = np.array(
            [is_trojan_label(lbl) or bool(trojan) for lbl, trojan in zip(test_labels, test_trojan_flags)],
            dtype=int,
        )
    else:
        y_true = np.array(
            [lbl not in args.train_labels or bool(trojan) for lbl, trojan in zip(test_labels, test_trojan_flags)],
            dtype=int,
        )
    y_pred = (test_errors >= threshold).astype(int)

    report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    try:
        auc = float(roc_auc_score(y_true, test_errors)) if len(np.unique(y_true)) > 1 else math.nan
    except Exception:
        auc = math.nan

    label_counts_series = pd.Series(test_labels).value_counts()
    test_label_counts = {str(k): int(v) for k, v in label_counts_series.items()}

    metrics = {
        "train_loss_last": train_losses[-1],
        "val_loss_last": val_losses[-1],
        "threshold": threshold,
        "threshold_percentile": args.threshold_percentile,
        "threshold_source": args.threshold_source,
        "anomaly_mode": args.anomaly_mode,
        "classification_report": report,
        "auc": auc,
        "num_train_windows": int(len(train_ds)),
        "num_val_windows": int(len(val_ds)),
        "num_test_windows": int(len(test_ds)),
        "test_label_counts": test_label_counts,
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))

    config = {
        "runs_dir": str(args.runs_dir),
        "window_size": args.window_size,
        "window_stride": args.window_stride,
        "train_labels": args.train_labels,
        "val_fraction": args.val_fraction,
        "test_fraction": args.test_fraction,
        "split_by": args.split_by,
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "lr": args.lr,
        "hidden_dim": args.hidden_dim,
        "latent_dim": args.latent_dim,
        "threshold_percentile": args.threshold_percentile,
        "threshold_source": args.threshold_source,
        "anomaly_mode": args.anomaly_mode,
        "onnx_opset": args.onnx_opset,
        "device": str(device),
        "seed": args.seed,
    }

    save_artifacts(
        output_dir=args.output_dir,
        model=model.cpu(),
        scaler=scaler,
        features=feature_cols,
        threshold=threshold,
        config=config,
        export_onnx=args.export_onnx,
        window_size=args.window_size,
        onnx_opset=args.onnx_opset,
    )

    print("Training complete")
    print(f"Artifacts written to {args.output_dir}")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
