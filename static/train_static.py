#!/usr/bin/env python3
"""Train static expert ensemble with uncertainty estimation.

Trains a deep ensemble of gradient boosting models on static binary features.
Outputs predictions (p_s) and uncertainty (u_s) for all binaries.

This can run locally (no GPU required) since static features are small.
"""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import train_test_split


def _load_binary_labels(runs_dir: Path, binaries_df: pd.DataFrame) -> pd.DataFrame:
    """Load binary-level labels from run metadata.
    
    A binary is labeled positive if ANY of its runs had trojan_family != 'none'.
    """
    binary_labels = {}
    
    for run_dir in sorted(runs_dir.glob("run_*")):
        meta_path = run_dir / "meta.json"
        if not meta_path.exists():
            continue
        with meta_path.open("r", encoding="utf-8") as f:
            meta = json.load(f)
        
        binary_id = meta.get("binary_id", "")
        trojan_family = meta.get("trojan_family", "none")
        
        if binary_id:
            # Binary is positive if any run has trojan
            if trojan_family != "none":
                binary_labels[binary_id] = 1
            elif binary_id not in binary_labels:
                binary_labels[binary_id] = 0
    
    # Merge with features
    df = binaries_df.copy()
    df["label"] = df["binary_id"].map(binary_labels).fillna(0).astype(int)
    return df


def train_ensemble(
    X: np.ndarray,
    y: np.ndarray,
    n_estimators: int = 5,
    seed: int = 1337,
) -> List[GradientBoostingClassifier]:
    """Train a deep ensemble of gradient boosting classifiers."""
    ensemble = []
    for i in range(n_estimators):
        # Each member has a different seed for bootstrapped training
        model = GradientBoostingClassifier(
            n_estimators=100,
            max_depth=4,
            learning_rate=0.1,
            subsample=0.8,
            random_state=seed + i,
        )
        model.fit(X, y)
        ensemble.append(model)
    return ensemble


def predict_with_uncertainty(
    ensemble: List[GradientBoostingClassifier],
    X: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Get ensemble predictions and uncertainty.
    
    Returns:
        p: Mean probability across ensemble members
        u: Uncertainty (std of probabilities across members)
    """
    probs = np.array([m.predict_proba(X)[:, 1] for m in ensemble])
    p = probs.mean(axis=0)
    u = probs.std(axis=0)
    return p, u


def main() -> int:
    ap = argparse.ArgumentParser(description="Train static expert ensemble")
    ap.add_argument("--features", type=Path, required=True,
                    help="Path to static_features.parquet")
    ap.add_argument("--runs-dir", type=Path, required=True,
                    help="Path to runs directory for labels")
    ap.add_argument("--out-dir", type=Path, required=True,
                    help="Output directory for model and predictions")
    ap.add_argument("--n-ensemble", type=int, default=5,
                    help="Number of ensemble members")
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--val-split", type=float, default=0.2)
    args = ap.parse_args()

    # Load features
    df = pd.read_parquet(args.features)
    print(f"Loaded {len(df)} binaries with {len(df.columns)} columns")

    # Load labels
    df = _load_binary_labels(args.runs_dir, df)
    print(f"Labels: {df['label'].value_counts().to_dict()}")

    # Prepare features
    feature_cols = [c for c in df.columns if c not in ("binary_id", "label")]
    X = df[feature_cols].values.astype(np.float32)
    y = df["label"].values.astype(np.int64)

    # Train/val split
    X_train, X_val, y_train, y_val, idx_train, idx_val = train_test_split(
        X, y, np.arange(len(y)),
        test_size=args.val_split,
        random_state=args.seed,
        stratify=y if len(np.unique(y)) > 1 else None,
    )

    print(f"Training on {len(X_train)} samples, validating on {len(X_val)}")

    # Train ensemble
    ensemble = train_ensemble(X_train, y_train, n_estimators=args.n_ensemble, seed=args.seed)

    # Predict on all data
    p_all, u_all = predict_with_uncertainty(ensemble, X)
    p_val, u_val = predict_with_uncertainty(ensemble, X_val)

    # Metrics
    if len(np.unique(y_val)) > 1:
        val_auc = roc_auc_score(y_val, p_val)
        val_pr = average_precision_score(y_val, p_val)
    else:
        val_auc = float("nan")
        val_pr = float("nan")

    print(f"Validation AUC: {val_auc:.4f}, PR-AUC: {val_pr:.4f}")
    print(f"Uncertainty range: [{u_all.min():.4f}, {u_all.max():.4f}]")

    # Save outputs
    args.out_dir.mkdir(parents=True, exist_ok=True)
    
    # Save ensemble
    with (args.out_dir / "ensemble.pkl").open("wb") as f:
        pickle.dump(ensemble, f)
    
    # Save predictions per binary
    predictions_df = pd.DataFrame({
        "binary_id": df["binary_id"].values,
        "p_s": p_all,
        "u_s": u_all,
        "label": y,
    })
    predictions_df.to_parquet(args.out_dir / "static_predictions.parquet", index=False)
    
    # Save metadata
    meta = {
        "model": "gradient_boosting_ensemble",
        "n_ensemble": args.n_ensemble,
        "n_features": len(feature_cols),
        "feature_columns": feature_cols,
        "val_auc": float(val_auc),
        "val_pr_auc": float(val_pr),
        "n_train": len(X_train),
        "n_val": len(X_val),
        "seed": args.seed,
    }
    with (args.out_dir / "meta.json").open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    
    print(f"Saved ensemble and predictions to {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
