#!/usr/bin/env python3
"""Train dynamic telemetry classifier ensemble (Colab/T4 recommended).

Trains a deep ensemble of TCN models for temporal classification.
Outputs predictions with uncertainty estimation.

Usage:
    python dynamic/train_dynamic.py \
        --processed-dir data/fusionbench_sim/processed/unseen_workload \
        --out-dir models/dynamic/unseen_workload \
        --model tcn --n-ensemble 5 --epochs 20
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from dynamic.models.tcn import TCN, SimpleCNN


# ============================================================================
# Legacy MLP (for backwards compatibility)
# ============================================================================

def _agg_features(X: np.ndarray) -> np.ndarray:
    """Aggregate window [N,T,F] into [N, 4F] using mean/std/min/max."""
    mean = X.mean(axis=1)
    std = X.std(axis=1)
    mn = X.min(axis=1)
    mx = X.max(axis=1)
    return np.concatenate([mean, std, mn, mx], axis=1).astype(np.float32)


class MLP(nn.Module):
    def __init__(self, d_in: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


# ============================================================================
# Data loading
# ============================================================================

def _load_npz(path: Path) -> Dict[str, np.ndarray]:
    data = np.load(path, allow_pickle=True)
    return {k: data[k] for k in data.files}


# ============================================================================
# Training utilities
# ============================================================================

def train_single_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_X: torch.Tensor,
    val_y: torch.Tensor,
    device: torch.device,
    epochs: int,
    lr: float,
    patience: int = 10,
    pos_weight: float = 1.0,
) -> Tuple[nn.Module, float]:
    """Train a single model with early stopping, class weighting, and LR scheduling."""
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    # Class weighting for imbalanced data
    pw = torch.tensor([pos_weight], device=device)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pw)
    
    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', patience=3, factor=0.5, min_lr=1e-6
    )
    
    best_val_loss = float("inf")
    best_state = None
    no_improve = 0
    
    for epoch in range(epochs):
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = loss_fn(logits, yb)
            loss.backward()
            # Gradient clipping for stable TCN training
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
        
        # Validation
        model.eval()
        with torch.no_grad():
            val_logits = model(val_X.to(device))
            val_loss = loss_fn(val_logits, val_y.to(device)).item()
        
        # Update LR scheduler
        scheduler.step(val_loss)
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                break
    
    if best_state is not None:
        model.load_state_dict(best_state)
    
    return model, best_val_loss


def predict_ensemble(
    models: List[nn.Module],
    X: torch.Tensor,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Get ensemble predictions, uncertainty, and logits.
    
    Returns:
        p: Mean probability [N]
        u: Uncertainty (std of probabilities) [N]
        logits: Mean logits [N] (for calibration)
    """
    all_logits = []
    all_probs = []
    
    X_device = X.to(device)
    
    for model in models:
        model = model.to(device)  # Ensure model is on same device as data
        model.eval()
        with torch.no_grad():
            logits = model(X_device).cpu().numpy()
            probs = 1.0 / (1.0 + np.exp(-logits))
            all_logits.append(logits)
            all_probs.append(probs)
    
    all_probs = np.array(all_probs)
    all_logits = np.array(all_logits)
    
    p = all_probs.mean(axis=0)
    u = all_probs.std(axis=0)
    mean_logits = all_logits.mean(axis=0)
    
    return p, u, mean_logits


def main() -> int:
    ap = argparse.ArgumentParser(description="Train dynamic expert ensemble")
    ap.add_argument("--processed-dir", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--model", type=str, default="tcn", choices=["tcn", "cnn", "mlp"])
    ap.add_argument("--n-ensemble", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--hidden-size", type=int, default=128)
    ap.add_argument("--patience", type=int, default=10)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Load data
    train = _load_npz(args.processed_dir / "windows_train.npz")
    val = _load_npz(args.processed_dir / "windows_val.npz")

    X_train = train["X"].astype(np.float32)
    y_train = train["y"].astype(np.float32)
    X_val = val["X"].astype(np.float32)
    y_val = val["y"].astype(np.float32)

    print(f"Train: {X_train.shape}, Val: {X_val.shape}")
    print(f"Train labels: {y_train.sum():.0f} / {len(y_train)} positive")
    print(f"Val labels: {y_val.sum():.0f} / {len(y_val)} positive")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Determine input dimensions
    n_train, seq_len, n_features = X_train.shape

    # Prepare data for MLP (aggregated) vs TCN/CNN (sequential)
    if args.model == "mlp":
        X_train_tensor = torch.from_numpy(_agg_features(X_train))
        X_val_tensor = torch.from_numpy(_agg_features(X_val))
        input_size = X_train_tensor.shape[1]
    else:
        X_train_tensor = torch.from_numpy(X_train)
        X_val_tensor = torch.from_numpy(X_val)
        input_size = n_features

    y_train_tensor = torch.from_numpy(y_train)
    y_val_tensor = torch.from_numpy(y_val)

    train_ds = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)

    # Train ensemble
    models: List[nn.Module] = []
    val_losses = []
    
    for i in range(args.n_ensemble):
        print(f"\nTraining ensemble member {i+1}/{args.n_ensemble}")
        torch.manual_seed(args.seed + i)
        
        if args.model == "tcn":
            model = TCN(input_size, hidden_size=args.hidden_size)
        elif args.model == "cnn":
            model = SimpleCNN(input_size, hidden_size=args.hidden_size)
        else:
            model = MLP(input_size)
        
        model = model.to(device)
        
        # Compute class weight for imbalanced data
        pos_ratio = y_train.mean()
        pos_weight = (1 - pos_ratio) / max(pos_ratio, 1e-6)
        
        model, val_loss = train_single_model(
            model, train_loader, X_val_tensor, y_val_tensor,
            device, args.epochs, args.lr, args.patience, pos_weight
        )
        models.append(model.cpu())
        val_losses.append(val_loss)
        print(f"  Val loss: {val_loss:.4f}")
        print(f"  LR final, pos_weight: {pos_weight:.2f}")

    # Evaluate ensemble
    p_val, u_val, logits_val = predict_ensemble(models, X_val_tensor, device)
    
    if len(np.unique(y_val)) > 1:
        val_auc = roc_auc_score(y_val, p_val)
        val_pr = average_precision_score(y_val, p_val)
    else:
        val_auc = float("nan")
        val_pr = float("nan")

    print(f"\nEnsemble Val AUC: {val_auc:.4f}, PR-AUC: {val_pr:.4f}")
    print(f"Uncertainty range: [{u_val.min():.4f}, {u_val.max():.4f}]")

    # Save outputs
    args.out_dir.mkdir(parents=True, exist_ok=True)
    
    # Save each model
    for i, model in enumerate(models):
        torch.save(model.state_dict(), args.out_dir / f"model_{i}.pt")
    
    # Save validation predictions (for calibration)
    np.savez(
        args.out_dir / "val_predictions.npz",
        p=p_val, u=u_val, logits=logits_val, y=y_val,
    )
    
    # Predict on train and test if available
    for split_name, X_data, y_data in [
        ("train", X_train, y_train),
        ("val", X_val, y_val),
    ]:
        if args.model == "mlp":
            X_tensor = torch.from_numpy(_agg_features(X_data))
        else:
            X_tensor = torch.from_numpy(X_data)
        
        p, u, logits = predict_ensemble(models, X_tensor, device)
        np.savez(
            args.out_dir / f"{split_name}_predictions.npz",
            p=p, u=u, logits=logits, y=y_data,
        )

    # Try test set if exists
    test_path = args.processed_dir / "windows_test.npz"
    if test_path.exists():
        test = _load_npz(test_path)
        X_test = test["X"].astype(np.float32)
        y_test = test["y"].astype(np.float32)
        
        if args.model == "mlp":
            X_test_tensor = torch.from_numpy(_agg_features(X_test))
        else:
            X_test_tensor = torch.from_numpy(X_test)
        
        p_test, u_test, logits_test = predict_ensemble(models, X_test_tensor, device)
        np.savez(
            args.out_dir / "test_predictions.npz",
            p=p_test, u=u_test, logits=logits_test, y=y_test,
        )
        
        if len(np.unique(y_test)) > 1:
            test_auc = roc_auc_score(y_test, p_test)
            test_pr = average_precision_score(y_test, p_test)
            print(f"Test AUC: {test_auc:.4f}, PR-AUC: {test_pr:.4f}")

    # Save metadata
    meta = {
        "model": args.model,
        "n_ensemble": args.n_ensemble,
        "input_size": int(input_size),
        "seq_len": int(seq_len) if args.model != "mlp" else None,
        "hidden_size": args.hidden_size,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "val_auc": float(val_auc),
        "val_pr_auc": float(val_pr),
        "val_losses": [float(v) for v in val_losses],
        "processed_dir": str(args.processed_dir),
        "seed": args.seed,
    }
    with (args.out_dir / "meta.json").open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print(f"\nSaved {args.n_ensemble} models to {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
