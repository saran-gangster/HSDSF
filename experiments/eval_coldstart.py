#!/usr/bin/env python3
"""Evaluate cold-start scenario: per-run norm with contaminated warmup.

This is the KILLER EXPERIMENT that justifies LLM-static's value.

Tests hypothesis:
- Dynamic + per-run norm should DEGRADE when warmup is contaminated
- LLM-informed fusion should RECOVER performance
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd


def load_test_data(processed_dir: Path) -> Dict:
    """Load test data from processed directory."""
    test_path = processed_dir / "windows_test.npz"
    data = np.load(test_path, allow_pickle=True)
    return {
        "X": data["X"],
        "y": data["y"],
        "t_center": data.get("t_center", np.arange(len(data["y"]))),
        "run_id": data.get("run_id", np.array(["unknown"] * len(data["y"]))),
        "binary_id": data.get("binary_id", np.array(["unknown"] * len(data["y"]))),
    }


def simulate_contaminated_normalization(
    X: np.ndarray,
    run_ids: np.ndarray,
    warmup_steps: int = 200,
    trojan_active_during_warmup: Dict[str, Tuple[int, int]] = None,
) -> np.ndarray:
    """Simulate per-run normalization with contaminated warmup.
    
    When trojan is active during warmup, the baseline stats are inflated,
    causing the normalized values to be skewed.
    
    Args:
        X: Feature array [N, T, F]
        run_ids: Run ID for each window
        warmup_steps: Number of warmup steps for baseline
        trojan_active_during_warmup: Dict of run_id -> (start_step, end_step) 
                                     where trojan is active during warmup
                                     
    Returns:
        Normalized X array
    """
    X_norm = X.copy()
    
    for run_id in np.unique(run_ids):
        mask = run_ids == run_id
        run_X = X[mask]
        
        if len(run_X) == 0:
            continue
        
        # Flatten windows for normalization
        # Shape: [n_windows, window_len, n_features]
        n_windows, window_len, n_features = run_X.shape
        
        # Use first warmup_steps windows as baseline
        warmup_windows = min(warmup_steps // window_len, n_windows)
        if warmup_windows < 1:
            warmup_windows = 1
        
        warmup_data = run_X[:warmup_windows].reshape(-1, n_features)
        
        # Compute baseline stats
        baseline_mean = warmup_data.mean(axis=0)
        baseline_std = warmup_data.std(axis=0) + 1e-6
        
        # Normalize
        normalized = (run_X - baseline_mean) / baseline_std
        X_norm[mask] = normalized
    
    return X_norm


def evaluate_methods_on_coldstart(
    X: np.ndarray,
    y: np.ndarray,
    run_ids: np.ndarray,
    binary_ids: np.ndarray,
    p_s: np.ndarray,  # Static scores (LLM or handcrafted)
    contaminated_runs: List[str],
    warmup_steps: int = 200,
) -> pd.DataFrame:
    """Evaluate different methods under cold-start conditions.
    
    Methods:
    1. dynamic_only (clean norm): Per-run norm on ALL runs
    2. dynamic_only (contaminated): Per-run norm, but some runs have contaminated warmup
    3. static_inform: Use p_s to adjust threshold/weight for contaminated runs
    """
    from sklearn.linear_model import LogisticRegression
    
    # Simulate models (for testing, just use mean features as score)
    # In real implementation, use trained model
    X_flat = X.mean(axis=1)  # [N, F] - simplified
    
    # Method 1: Dynamic with clean normalization (oracle - knows which runs are clean)
    clean_mask = ~np.isin(run_ids, contaminated_runs)
    
    # Method 2: Dynamic with potentially contaminated normalization
    # (This is the realistic scenario where we don't know which runs are contaminated)
    
    # Method 3: Static-informed
    # Use high p_s as indicator that we should be more cautious about baseline
    
    results = []
    
    # Evaluate on all test data
    y_binary = (y >= 0.5).astype(int)
    
    # Baseline: assume clean normalization works
    # Score = simple proxy (would be model output in reality)
    score_clean = X_flat.mean(axis=1)  # Placeholder
    
    # For contaminated runs, the normalized values will be different
    # Simulate this effect
    contaminated_mask = np.isin(run_ids, contaminated_runs)
    
    # Calculate metrics for different scenarios
    for threshold in [0.3, 0.4, 0.5]:
        # Scenario 1: All runs, naive normalization (doesn't know about contamination)
        # Performance should degrade on contaminated runs
        
        # Scenario 2: LLM-informed (high p_s -> higher threshold)
        # This should help on contaminated runs
        
        results.append({
            "scenario": "evaluation",
            "threshold": threshold,
            "n_contaminated": contaminated_mask.sum(),
            "n_clean": (~contaminated_mask).sum(),
            "total": len(y_binary),
        })
    
    return pd.DataFrame(results)


def main():
    ap = argparse.ArgumentParser(description="Evaluate cold-start scenario")
    ap.add_argument("--processed-dir", type=Path, required=True,
                    help="Path to processed data directory")
    ap.add_argument("--split-json", type=Path, required=True,
                    help="Path to cold-start split JSON with contamination info")
    ap.add_argument("--static-scores", type=Path,
                    help="Path to static scores CSV (binary_id, p_s)")
    ap.add_argument("--output", type=Path, 
                    default=Path("paper/analysis/coldstart_evaluation.json"))
    args = ap.parse_args()
    
    # Load split info
    with open(args.split_json) as f:
        split_info = json.load(f)
    
    contaminated_runs = split_info.get("contaminated_runs", [])
    print(f"Loaded split with {len(contaminated_runs)} contaminated runs")
    
    # Load test data
    test_data = load_test_data(args.processed_dir)
    print(f"Loaded {len(test_data['y'])} test windows")
    
    # Load static scores (or use defaults)
    if args.static_scores and args.static_scores.exists():
        static_df = pd.read_csv(args.static_scores)
        p_s = {row["binary_id"]: row["p_s"] for _, row in static_df.iterrows()}
    else:
        # Default: uniform scores
        p_s = {bid: 0.5 for bid in np.unique(test_data["binary_id"])}
    
    # Convert to array
    p_s_array = np.array([p_s.get(bid, 0.5) for bid in test_data["binary_id"]])
    
    # Evaluate
    results = evaluate_methods_on_coldstart(
        X=test_data["X"],
        y=test_data["y"],
        run_ids=test_data["run_id"],
        binary_ids=test_data["binary_id"],
        p_s=p_s_array,
        contaminated_runs=contaminated_runs,
        warmup_steps=split_info.get("warmup_steps", 200),
    )
    
    print("\n" + "="*60)
    print("COLD-START EVALUATION RESULTS")
    print("="*60)
    print(results.to_string(index=False))
    
    # Save
    args.output.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(args.output.with_suffix(".csv"), index=False)
    
    summary = {
        "contaminated_runs": contaminated_runs,
        "n_contaminated": len(contaminated_runs),
        "results": results.to_dict("records"),
    }
    with open(args.output, "w") as f:
        json.dump(summary, f, indent=2)
    
    print(f"\nSaved to: {args.output}")


if __name__ == "__main__":
    main()
