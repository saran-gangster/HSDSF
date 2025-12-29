#!/usr/bin/env python3
"""Calibrate dynamic expert using temperature scaling.

Applies post-hoc calibration to the dynamic expert ensemble predictions.
Reference: Guo et al., "On Calibration of Modern Neural Networks", ICML 2017.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Tuple

import numpy as np
from scipy.optimize import minimize_scalar


def _ece(y_true: np.ndarray, p: np.ndarray, n_bins: int = 15) -> float:
    """Compute Expected Calibration Error."""
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        if i < n_bins - 1:
            mask = (p >= lo) & (p < hi)
        else:
            mask = (p >= lo) & (p <= hi)
        if not np.any(mask):
            continue
        acc = float(np.mean(y_true[mask]))
        conf = float(np.mean(p[mask]))
        ece += float(np.mean(mask)) * abs(acc - conf)
    return float(ece)


def _calibrate_temperature(
    logits: np.ndarray,
    y_true: np.ndarray,
) -> Tuple[float, float, float]:
    """Find optimal temperature via bounded optimization.
    
    Returns:
        temperature: optimal temperature
        ece_before: ECE before calibration
        ece_after: ECE after calibration
    """
    p_before = 1.0 / (1.0 + np.exp(-logits))
    ece_before = _ece(y_true, p_before)
    
    def objective(T: float) -> float:
        p = 1.0 / (1.0 + np.exp(-logits / T))
        return _ece(y_true, p)
    
    result = minimize_scalar(objective, bounds=(0.1, 10.0), method="bounded")
    temperature = float(result.x)
    
    p_after = 1.0 / (1.0 + np.exp(-logits / temperature))
    ece_after = _ece(y_true, p_after)
    
    return temperature, ece_before, ece_after


def main() -> int:
    ap = argparse.ArgumentParser(description="Calibrate dynamic expert")
    ap.add_argument("--model-dir", type=Path, required=True,
                    help="Path to dynamic model directory")
    args = ap.parse_args()

    # Load validation predictions
    val_path = args.model_dir / "val_predictions.npz"
    if not val_path.exists():
        print(f"Validation predictions not found: {val_path}")
        return 1
    
    data = np.load(val_path)
    logits = data["logits"]
    y_soft = data["y"]
    y = (y_soft >= 0.5).astype(np.float32)  # Convert soft to binary for calibration
    
    # Find optimal temperature
    temperature, ece_before, ece_after = _calibrate_temperature(logits, y)
    
    print(f"Temperature: {temperature:.4f}")
    print(f"ECE before: {ece_before:.4f}")
    print(f"ECE after: {ece_after:.4f}")
    
    # Apply calibration to all saved predictions
    for pred_file in args.model_dir.glob("*_predictions.npz"):
        data = np.load(pred_file)
        logits = data["logits"]
        p_calibrated = 1.0 / (1.0 + np.exp(-logits / temperature))
        
        # Save calibrated version
        calibrated_path = pred_file.with_name(
            pred_file.stem.replace("_predictions", "_predictions_calibrated") + ".npz"
        )
        np.savez(
            calibrated_path,
            p=data["p"],
            u=data["u"],
            logits=logits,
            y=data["y"],
            p_calibrated=p_calibrated,
        )
        print(f"Saved calibrated: {calibrated_path.name}")
    
    # Save calibration metadata
    calib_meta = {
        "temperature": temperature,
        "ece_before": ece_before,
        "ece_after": ece_after,
        "n_samples": len(y),
    }
    with (args.model_dir / "calibration.json").open("w", encoding="utf-8") as f:
        json.dump(calib_meta, f, indent=2)
    
    print(f"\nCalibration complete. Temperature={temperature:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
