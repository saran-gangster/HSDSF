#!/usr/bin/env python3
"""Calibrate static expert using temperature scaling.

Applies post-hoc calibration to the static expert predictions.
Reference: Guo et al., "On Calibration of Modern Neural Networks", ICML 2017.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
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
    """Find optimal temperature via grid search.
    
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
    ap = argparse.ArgumentParser(description="Calibrate static expert")
    ap.add_argument("--predictions", type=Path, required=True,
                    help="Path to static_predictions.parquet")
    ap.add_argument("--out-dir", type=Path, required=True,
                    help="Output directory (same as training dir)")
    args = ap.parse_args()

    # Load predictions
    df = pd.read_parquet(args.predictions)
    p = df["p_s"].values
    y = df["label"].values
    
    # Convert probabilities to logits for temperature scaling
    # Clip to avoid log(0)
    p_clipped = np.clip(p, 1e-7, 1 - 1e-7)
    logits = np.log(p_clipped / (1 - p_clipped))
    
    # Find optimal temperature
    temperature, ece_before, ece_after = _calibrate_temperature(logits, y)
    
    print(f"Temperature: {temperature:.4f}")
    print(f"ECE before: {ece_before:.4f}")
    print(f"ECE after: {ece_after:.4f}")
    
    # Apply calibration
    p_calibrated = 1.0 / (1.0 + np.exp(-logits / temperature))
    
    # Save calibrated predictions
    df["p_s_calibrated"] = p_calibrated
    args.out_dir.mkdir(parents=True, exist_ok=True)
    df.to_parquet(args.out_dir / "static_predictions_calibrated.parquet", index=False)
    
    # Save calibration metadata
    calib_meta = {
        "temperature": temperature,
        "ece_before": ece_before,
        "ece_after": ece_after,
        "n_samples": len(y),
    }
    with (args.out_dir / "calibration.json").open("w", encoding="utf-8") as f:
        json.dump(calib_meta, f, indent=2)
    
    print(f"Saved calibrated predictions to {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
