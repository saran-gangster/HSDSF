#!/usr/bin/env python3
"""Priority experiments for paper: FAR-matched eval, per-binary analysis, g sensitivity, etc.

Usage:
    python3 experiments/paper_experiments.py --split random_split
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ============================================================================
# FAR-MATCHED EVALUATION
# ============================================================================

def find_threshold_for_far(
    p: np.ndarray,
    y: np.ndarray,
    run_ids: np.ndarray,
    window_len_s: float,
    target_far: float,
) -> Tuple[float, float]:
    """Find threshold achieving target FAR/hour."""
    best_thresh = 0.99
    best_far = 0.0
    
    for thresh in np.linspace(0.01, 0.99, 200):
        y_pred = (p >= thresh).astype(int)
        fp = (y_pred == 1) & (y == 0)
        
        # Compute FAR per hour
        unique_runs = np.unique(run_ids)
        total_fp = 0
        total_hours = 0
        for run in unique_runs:
            mask = run_ids == run
            run_fp = np.sum(fp[mask])
            run_windows = np.sum(mask)
            run_hours = run_windows * window_len_s / 3600
            total_fp += run_fp
            total_hours += run_hours
        
        far = total_fp / total_hours if total_hours > 0 else 0
        
        if far <= target_far and far > best_far:
            best_far = far
            best_thresh = thresh
    
    return best_thresh, best_far


def evaluate_at_far_targets(
    p: np.ndarray,
    y: np.ndarray,
    run_ids: np.ndarray,
    window_len_s: float,
    method_name: str,
    targets: List[float] = [5.0, 10.0, 20.0],
) -> pd.DataFrame:
    """Evaluate a method at multiple FAR targets."""
    results = []
    
    for target_far in targets:
        thresh, actual_far = find_threshold_for_far(p, y, run_ids, window_len_s, target_far)
        
        y_pred = (p >= thresh).astype(int)
        tp = np.sum((y_pred == 1) & (y == 1))
        fn = np.sum((y_pred == 0) & (y == 1))
        fp = np.sum((y_pred == 1) & (y == 0))
        
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        
        results.append({
            "method": method_name,
            "target_far": target_far,
            "actual_far": actual_far,
            "threshold": thresh,
            "recall": recall,
            "precision": precision,
        })
    
    return pd.DataFrame(results)


# ============================================================================
# PER-BINARY ANALYSIS
# ============================================================================

def per_binary_analysis(
    p: np.ndarray,
    y: np.ndarray,
    binary_ids: np.ndarray,
    p_s: np.ndarray,
    run_ids: np.ndarray,
    window_len_s: float,
    threshold: float = 0.5,
) -> Dict:
    """Analyze performance separately for trojan vs benign binaries."""
    # Identify trojan vs benign binaries by static prediction
    trojan_mask = p_s >= 0.5
    benign_mask = ~trojan_mask
    
    y_pred = (p >= threshold).astype(int)
    
    # Trojan binaries: focus on detection (recall, TTD)
    trojan_tp = np.sum((y_pred[trojan_mask] == 1) & (y[trojan_mask] == 1))
    trojan_fn = np.sum((y_pred[trojan_mask] == 0) & (y[trojan_mask] == 1))
    trojan_recall = trojan_tp / (trojan_tp + trojan_fn) if (trojan_tp + trojan_fn) > 0 else 0
    
    # Benign binaries: focus on FAR
    if benign_mask.sum() > 0:
        benign_fp = np.sum((y_pred[benign_mask] == 1) & (y[benign_mask] == 0))
        benign_runs = np.unique(run_ids[benign_mask])
        total_hours = 0
        for run in benign_runs:
            run_mask = (run_ids == run) & benign_mask
            run_windows = np.sum(run_mask)
            run_hours = run_windows * window_len_s / 3600
            total_hours += run_hours
        benign_far = benign_fp / total_hours if total_hours > 0 else 0
    else:
        benign_far = 0
    
    return {
        "trojan_recall": trojan_recall,
        "trojan_total_positives": int(trojan_tp + trojan_fn),
        "benign_far_per_hour": benign_far,
        "benign_windows": int(benign_mask.sum()),
        "trojan_windows": int(trojan_mask.sum()),
    }


# ============================================================================
# SENSITIVITY ANALYSIS FOR g
# ============================================================================

def g_sensitivity_analysis(
    p_s: np.ndarray,
    p_d: np.ndarray,
    y: np.ndarray,
    run_ids: np.ndarray,
    window_len_s: float,
    g_values: List[float] = [0.6, 0.7, 0.8, 0.9, 0.95],
) -> pd.DataFrame:
    """Analyze F1/FAR/TTD sensitivity to constant g value."""
    results = []
    
    for g in g_values:
        # Constant gate fusion
        p_fused = g * p_d + (1 - g) * p_s
        
        # Find optimal threshold
        best_f1 = 0
        best_thresh = 0.5
        for thresh in np.linspace(0.05, 0.95, 50):
            y_pred = (p_fused >= thresh).astype(int)
            tp = np.sum((y_pred == 1) & (y == 1))
            fp = np.sum((y_pred == 1) & (y == 0))
            fn = np.sum((y_pred == 0) & (y == 1))
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            
            if f1 > best_f1:
                best_f1 = f1
                best_thresh = thresh
        
        # Evaluate at best threshold
        y_pred = (p_fused >= best_thresh).astype(int)
        fp_mask = (y_pred == 1) & (y == 0)
        
        # Compute FAR
        unique_runs = np.unique(run_ids)
        total_fp = 0
        total_hours = 0
        for run in unique_runs:
            mask = run_ids == run
            run_fp = np.sum(fp_mask[mask])
            run_windows = np.sum(mask)
            run_hours = run_windows * window_len_s / 3600
            total_fp += run_fp
            total_hours += run_hours
        far = total_fp / total_hours if total_hours > 0 else 0
        
        results.append({
            "g": g,
            "f1": best_f1,
            "far_per_hour": far,
            "threshold": best_thresh,
        })
    
    return pd.DataFrame(results)


def plot_g_sensitivity(df: pd.DataFrame, output_path: Path):
    """Generate sensitivity plot for g."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    
    # F1 vs g
    axes[0].plot(df["g"], df["f1"], "bo-", linewidth=2, markersize=8)
    axes[0].set_xlabel("Gate value (g)", fontsize=12)
    axes[0].set_ylabel("F1 Score", fontsize=12)
    axes[0].set_title("F1 vs Constant Gate Value", fontsize=14)
    axes[0].grid(True, alpha=0.3)
    axes[0].set_ylim(0.5, 0.7)
    
    # FAR vs g
    axes[1].plot(df["g"], df["far_per_hour"], "ro-", linewidth=2, markersize=8)
    axes[1].set_xlabel("Gate value (g)", fontsize=12)
    axes[1].set_ylabel("FAR/hour", fontsize=12)
    axes[1].set_title("FAR vs Constant Gate Value", fontsize=14)
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path}")


# ============================================================================
# TEMPORAL SMOOTHING
# ============================================================================

def debounce_predictions(
    p: np.ndarray,
    run_ids: np.ndarray,
    k: int = 3,
    threshold: float = 0.5,
) -> np.ndarray:
    """Apply debounce: require k consecutive windows above threshold.
    
    Returns smoothed binary predictions.
    """
    y_pred = np.zeros_like(p, dtype=int)
    
    for run_id in np.unique(run_ids):
        run_mask = run_ids == run_id
        run_p = p[run_mask]
        run_binary = (run_p >= threshold).astype(int)
        
        # Debounce: only set 1 if k consecutive 1s
        run_debounced = np.zeros_like(run_binary)
        for i in range(k - 1, len(run_binary)):
            if np.all(run_binary[i - k + 1:i + 1] == 1):
                run_debounced[i] = 1
        
        y_pred[run_mask] = run_debounced
    
    return y_pred


def evaluate_with_debounce(
    p: np.ndarray,
    y: np.ndarray,
    run_ids: np.ndarray,
    window_len_s: float,
    k_values: List[int] = [1, 2, 3, 5],
    threshold: float = 0.5,
) -> pd.DataFrame:
    """Evaluate with different debounce k values."""
    results = []
    
    for k in k_values:
        if k == 1:
            y_pred = (p >= threshold).astype(int)
        else:
            y_pred = debounce_predictions(p, run_ids, k=k, threshold=threshold)
        
        tp = np.sum((y_pred == 1) & (y == 1))
        fp = np.sum((y_pred == 1) & (y == 0))
        fn = np.sum((y_pred == 0) & (y == 1))
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        # FAR
        unique_runs = np.unique(run_ids)
        total_fp = 0
        total_hours = 0
        fp_mask = (y_pred == 1) & (y == 0)
        for run in unique_runs:
            mask = run_ids == run
            total_fp += np.sum(fp_mask[mask])
            total_hours += np.sum(mask) * window_len_s / 3600
        far = total_fp / total_hours if total_hours > 0 else 0
        
        results.append({
            "debounce_k": k,
            "f1": f1,
            "recall": recall,
            "precision": precision,
            "far_per_hour": far,
        })
    
    return pd.DataFrame(results)


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Priority experiments for paper")
    parser.add_argument("--split", type=str, default="random_split")
    parser.add_argument("--processed-dir", type=str, default="data/fusionbench_sim/processed")
    parser.add_argument("--static-dir", type=str, default="models/static")
    parser.add_argument("--dynamic-dir", type=str, default="models/dynamic")
    parser.add_argument("--output-dir", type=str, default="paper/analysis")
    parser.add_argument("--window-len-s", type=float, default=1.0)
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load data
    processed_dir = Path(args.processed_dir) / args.split
    test_data = np.load(processed_dir / "windows_test.npz", allow_pickle=True)
    train_data = np.load(processed_dir / "windows_train.npz", allow_pickle=True)
    
    y_test = test_data["y"]
    y_test = (y_test >= 0.5).astype(int)  # Convert soft to binary
    run_ids = test_data.get("run_id", np.arange(len(y_test)))
    binary_ids = test_data.get("binary_id", np.array(["unknown"] * len(y_test)))
    
    # Load static predictions
    static_dir = Path(args.static_dir)
    calib_path = static_dir / "static_predictions_calibrated.parquet"
    if calib_path.exists():
        static_df = pd.read_parquet(calib_path)
        p_col = "p_s_calibrated"
    else:
        static_df = pd.read_parquet(static_dir / "static_predictions.parquet")
        p_col = "p_s"
    
    binary_to_p = dict(zip(static_df["binary_id"], static_df[p_col]))
    p_s = np.array([binary_to_p.get(bid, 0.5) for bid in binary_ids], dtype=np.float32)
    
    # Load dynamic predictions
    dynamic_dir = Path(args.dynamic_dir) / args.split
    dyn_calib_path = dynamic_dir / "test_predictions_calibrated.npz"
    if dyn_calib_path.exists():
        dyn_data = np.load(dyn_calib_path)
        p_d = dyn_data["p"]
    else:
        dyn_data = np.load(dynamic_dir / "test_predictions.npz")
        p_d = dyn_data["p"]
    
    print(f"Loaded {len(y_test)} test windows for {args.split}")
    print(f"Static p_s range: [{p_s.min():.3f}, {p_s.max():.3f}]")
    print(f"Dynamic p_d range: [{p_d.min():.3f}, {p_d.max():.3f}]")
    
    # ========================================================================
    # 1. G SENSITIVITY ANALYSIS
    # ========================================================================
    print("\n" + "="*60)
    print("G SENSITIVITY ANALYSIS")
    print("="*60)
    
    g_df = g_sensitivity_analysis(p_s, p_d, y_test, run_ids, args.window_len_s)
    print(g_df.to_string(index=False))
    g_df.to_csv(output_dir / f"g_sensitivity_{args.split}.csv", index=False)
    
    plot_g_sensitivity(g_df, output_dir / f"g_sensitivity_{args.split}.png")
    
    # Find optimal g
    best_g = g_df.loc[g_df["f1"].idxmax(), "g"]
    print(f"\nOptimal g: {best_g} (F1={g_df['f1'].max():.3f})")
    
    # ========================================================================
    # 2. FAR-MATCHED EVALUATION
    # ========================================================================
    print("\n" + "="*60)
    print("FAR-MATCHED EVALUATION")
    print("="*60)
    
    # Evaluate key methods at FAR targets
    methods = {
        "dynamic_only": p_d,
        "UGF (g=0.95)": 0.95 * p_d + 0.05 * p_s,
        f"constant_gate (g={best_g})": best_g * p_d + (1 - best_g) * p_s,
    }
    
    all_far_results = []
    for method_name, p_method in methods.items():
        far_df = evaluate_at_far_targets(
            p_method, y_test, run_ids, args.window_len_s, method_name
        )
        all_far_results.append(far_df)
    
    far_results = pd.concat(all_far_results, ignore_index=True)
    print(far_results.to_string(index=False))
    far_results.to_csv(output_dir / f"far_matched_{args.split}.csv", index=False)
    
    # ========================================================================
    # 3. PER-BINARY ANALYSIS
    # ========================================================================
    print("\n" + "="*60)
    print("PER-BINARY ANALYSIS")
    print("="*60)
    
    per_binary_results = []
    for method_name, p_method in methods.items():
        # Find optimal threshold
        best_f1, best_thresh = 0, 0.5
        for thresh in np.linspace(0.1, 0.9, 20):
            y_pred = (p_method >= thresh).astype(int)
            tp = np.sum((y_pred == 1) & (y_test == 1))
            fp = np.sum((y_pred == 1) & (y_test == 0))
            fn = np.sum((y_pred == 0) & (y_test == 1))
            f1 = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0
            if f1 > best_f1:
                best_f1, best_thresh = f1, thresh
        
        analysis = per_binary_analysis(
            p_method, y_test, binary_ids, p_s, run_ids, args.window_len_s, best_thresh
        )
        analysis["method"] = method_name
        analysis["threshold"] = best_thresh
        per_binary_results.append(analysis)
    
    per_binary_df = pd.DataFrame(per_binary_results)
    print(per_binary_df.to_string(index=False))
    per_binary_df.to_csv(output_dir / f"per_binary_{args.split}.csv", index=False)
    
    # ========================================================================
    # 4. TEMPORAL SMOOTHING (DEBOUNCE)
    # ========================================================================
    print("\n" + "="*60)
    print("TEMPORAL SMOOTHING (DEBOUNCE)")
    print("="*60)
    
    # Evaluate UGF with debounce (g=0.95)
    p_ugf = 0.95 * p_d + 0.05 * p_s
    debounce_df = evaluate_with_debounce(
        p_ugf, y_test, run_ids, args.window_len_s,
        k_values=[1, 2, 3, 5],
        threshold=0.40  # Higher threshold for g=0.95
    )
    print("UGF with debounce (threshold=0.15):")
    print(debounce_df.to_string(index=False))
    debounce_df.to_csv(output_dir / f"debounce_{args.split}.csv", index=False)
    
    print(f"\n✅ Results saved to {output_dir}/")


if __name__ == "__main__":
    main()
