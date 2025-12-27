#!/usr/bin/env python3
"""Train UGF fusion gate.

Stagewise training:
1. Load frozen static expert predictions (p_s, u_s per binary)
2. Load frozen dynamic ensemble predictions (p_d, u_d per window)
3. Train gate with BCE loss on window labels
4. Save gate model

Usage:
    python fusion/train_fusion.py \
        --processed-dir data/fusionbench_sim/processed/unseen_workload \
        --static-dir models/static \
        --dynamic-dir models/dynamic/unseen_workload \
        --out-dir models/fusion/unseen_workload
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from fusion.gate_model import UGFGate, UGFFusion


def _load_npz(path: Path) -> Dict[str, np.ndarray]:
    data = np.load(path, allow_pickle=True)
    return {k: data[k] for k in data.files}


def _load_static_predictions(
    static_dir: Path,
    window_binary_ids: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Load static predictions and map to window-level by binary_id."""
    # Try calibrated first, fall back to uncalibrated
    calib_path = static_dir / "static_predictions_calibrated.parquet"
    if calib_path.exists():
        df = pd.read_parquet(calib_path)
        p_col = "p_s_calibrated"
    else:
        df = pd.read_parquet(static_dir / "static_predictions.parquet")
        p_col = "p_s"
    
    # Build mapping
    binary_to_p = dict(zip(df["binary_id"], df[p_col]))
    binary_to_u = dict(zip(df["binary_id"], df["u_s"]))
    
    # Map to windows
    p_s = np.array([binary_to_p.get(bid, 0.5) for bid in window_binary_ids])
    u_s = np.array([binary_to_u.get(bid, 0.25) for bid in window_binary_ids])
    
    return p_s.astype(np.float32), u_s.astype(np.float32)


def _load_dynamic_predictions(
    dynamic_dir: Path,
    split: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load dynamic predictions for a split."""
    # Try calibrated first
    calib_path = dynamic_dir / f"{split}_predictions_calibrated.npz"
    if calib_path.exists():
        data = _load_npz(calib_path)
        p_d = data["p_calibrated"]
    else:
        data = _load_npz(dynamic_dir / f"{split}_predictions.npz")
        p_d = data["p"]
    
    return p_d.astype(np.float32), data["u"].astype(np.float32), data["y"].astype(np.float32)


def train_gate(
    p_s_train: np.ndarray,
    p_d_train: np.ndarray,
    u_s_train: np.ndarray,
    u_d_train: np.ndarray,
    y_train: np.ndarray,
    p_s_val: np.ndarray,
    p_d_val: np.ndarray,
    u_s_val: np.ndarray,
    u_d_val: np.ndarray,
    y_val: np.ndarray,
    epochs: int = 50,
    batch_size: int = 256,
    lr: float = 1e-3,
    patience: int = 10,
    device: torch.device = torch.device("cpu"),
) -> Tuple[UGFFusion, float]:
    """Train the UGF gate model."""
    
    # Create model with enhanced gate (predictions + uncertainties as input)
    gate = UGFGate(n_meta_features=0, hidden_size=64, use_predictions=True)
    model = UGFFusion(gate).to(device)
    
    # Prepare data
    train_ds = TensorDataset(
        torch.from_numpy(p_s_train),
        torch.from_numpy(p_d_train),
        torch.from_numpy(u_s_train),
        torch.from_numpy(u_d_train),
        torch.from_numpy(y_train),
    )
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.BCELoss()
    
    best_val_loss = float("inf")
    best_state = None
    no_improve = 0
    
    # Validation tensors
    val_p_s = torch.from_numpy(p_s_val).to(device)
    val_p_d = torch.from_numpy(p_d_val).to(device)
    val_u_s = torch.from_numpy(u_s_val).to(device)
    val_u_d = torch.from_numpy(u_d_val).to(device)
    val_y = torch.from_numpy(y_val).to(device)
    
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        n_batches = 0
        
        for p_s, p_d, u_s, u_d, y in train_loader:
            p_s, p_d = p_s.to(device), p_d.to(device)
            u_s, u_d = u_s.to(device), u_d.to(device)
            y = y.to(device)
            
            optimizer.zero_grad(set_to_none=True)
            
            # Get both prediction and gate values
            p_f, g = model.forward_with_gate(p_s, p_d, u_s, u_d)
            
            # Main BCE loss
            bce_loss = loss_fn(p_f, y)
            
            # Entropy regularization to prevent gate collapse
            # Encourages gate to explore both experts
            eps = 1e-8
            gate_entropy = -(g * torch.log(g + eps) + (1 - g) * torch.log(1 - g + eps)).mean()
            
            # Combined loss (maximize entropy = minimize negative entropy)
            # Increased weight from 0.1 to 0.3 to prevent gate collapse
            loss = bce_loss - 0.3 * gate_entropy
            
            loss.backward()
            optimizer.step()
            
            epoch_loss += bce_loss.item()
            n_batches += 1
        
        # Validation
        model.eval()
        with torch.no_grad():
            val_p_f = model(val_p_s, val_p_d, val_u_s, val_u_d)
            val_loss = loss_fn(val_p_f, val_y).item()
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break
    
    if best_state is not None:
        model.load_state_dict(best_state)
    
    return model.cpu(), best_val_loss


def main() -> int:
    ap = argparse.ArgumentParser(description="Train UGF fusion gate")
    ap.add_argument("--processed-dir", type=Path, required=True,
                    help="Preprocessed data directory with window info")
    ap.add_argument("--static-dir", type=Path, required=True,
                    help="Static expert output directory")
    ap.add_argument("--dynamic-dir", type=Path, required=True,
                    help="Dynamic expert output directory")
    ap.add_argument("--out-dir", type=Path, required=True,
                    help="Output directory for fusion model")
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--patience", type=int, default=10)
    ap.add_argument("--seed", type=int, default=1337)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load window metadata for binary_id mapping
    train_meta = _load_npz(args.processed_dir / "windows_train.npz")
    val_meta = _load_npz(args.processed_dir / "windows_val.npz")
    
    train_binary_ids = train_meta.get("binary_id", np.array(["unknown"] * len(train_meta["y"])))
    val_binary_ids = val_meta.get("binary_id", np.array(["unknown"] * len(val_meta["y"])))

    # Load static predictions (mapped to windows)
    p_s_train, u_s_train = _load_static_predictions(args.static_dir, train_binary_ids)
    p_s_val, u_s_val = _load_static_predictions(args.static_dir, val_binary_ids)
    
    print(f"Static train: p_s mean={p_s_train.mean():.3f}, u_s mean={u_s_train.mean():.3f}")
    print(f"Static val: p_s mean={p_s_val.mean():.3f}, u_s mean={u_s_val.mean():.3f}")

    # Load dynamic predictions
    p_d_train, u_d_train, y_train = _load_dynamic_predictions(args.dynamic_dir, "train")
    p_d_val, u_d_val, y_val = _load_dynamic_predictions(args.dynamic_dir, "val")
    
    print(f"Dynamic train: p_d mean={p_d_train.mean():.3f}, u_d mean={u_d_train.mean():.3f}")
    print(f"Dynamic val: p_d mean={p_d_val.mean():.3f}, u_d mean={u_d_val.mean():.3f}")
    print(f"Train labels: {y_train.sum():.0f}/{len(y_train)} positive")
    print(f"Val labels: {y_val.sum():.0f}/{len(y_val)} positive")

    # Train gate
    model, val_loss = train_gate(
        p_s_train, p_d_train, u_s_train, u_d_train, y_train,
        p_s_val, p_d_val, u_s_val, u_d_val, y_val,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        patience=args.patience,
        device=device,
    )

    # Evaluate on validation
    model.eval()
    with torch.no_grad():
        p_f_val, g_val = model.forward_with_gate(
            torch.from_numpy(p_s_val),
            torch.from_numpy(p_d_val),
            torch.from_numpy(u_s_val),
            torch.from_numpy(u_d_val),
        )
        p_f_val = p_f_val.numpy()
        g_val = g_val.numpy()
    
    # Compute metrics
    from sklearn.metrics import average_precision_score, roc_auc_score
    if len(np.unique(y_val)) > 1:
        val_auc = roc_auc_score(y_val, p_f_val)
        val_pr = average_precision_score(y_val, p_f_val)
        dyn_auc = roc_auc_score(y_val, p_d_val)
        dyn_pr = average_precision_score(y_val, p_d_val)
    else:
        val_auc = val_pr = dyn_auc = dyn_pr = float("nan")
    
    print(f"\nVal UGF AUC: {val_auc:.4f}, PR-AUC: {val_pr:.4f}")
    print(f"Val Dyn AUC: {dyn_auc:.4f}, PR-AUC: {dyn_pr:.4f}")
    print(f"Gate mean: {g_val.mean():.4f}, std: {g_val.std():.4f}")

    # Save model and predictions
    args.out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), args.out_dir / "fusion_model.pt")
    
    np.savez(
        args.out_dir / "val_predictions.npz",
        p_f=p_f_val,
        g=g_val,
        p_s=p_s_val,
        p_d=p_d_val,
        u_s=u_s_val,
        u_d=u_d_val,
        y=y_val,
    )

    # Save metadata
    meta = {
        "val_auc": float(val_auc),
        "val_pr_auc": float(val_pr),
        "dyn_val_auc": float(dyn_auc),
        "dyn_val_pr_auc": float(dyn_pr),
        "val_loss": float(val_loss),
        "gate_mean": float(g_val.mean()),
        "gate_std": float(g_val.std()),
        "epochs": args.epochs,
        "seed": args.seed,
        "processed_dir": str(args.processed_dir),
        "static_dir": str(args.static_dir),
        "dynamic_dir": str(args.dynamic_dir),
    }
    with (args.out_dir / "meta.json").open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print(f"\nSaved fusion model to {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
