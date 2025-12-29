#!/usr/bin/env python3
"""Evaluate all fusion methods on test set.

Computes comprehensive metrics for UGF and all baselines,
with bootstrap confidence intervals over runs.

Usage:
    python fusion/eval_fusion.py \
        --processed-dir data/fusionbench_sim/processed/unseen_workload \
        --static-dir models/static \
        --dynamic-dir models/dynamic/unseen_workload \
        --fusion-dir models/fusion/unseen_workload \
        --out-dir results/unseen_workload
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch

from evaluation.events import Interval, load_intervals_csv
from evaluation.metrics import summarize_run_metrics
from fusion.baselines import BASELINES, FusionResult
from fusion.gate_model import UGFFusion, UGFGate


def find_optimal_threshold(
    y_true: np.ndarray,
    p: np.ndarray,
    thresholds: np.ndarray = None,
) -> Tuple[float, float]:
    """Find threshold that maximizes F1 score.
    
    Returns:
        best_threshold: Optimal threshold value
        best_f1: F1 score at optimal threshold
    """
    if thresholds is None:
        thresholds = np.linspace(0.1, 0.9, 17)
    
    best_f1 = 0.0
    best_threshold = 0.5
    
    for thresh in thresholds:
        y_pred = (p >= thresh).astype(int)
        tp = np.sum((y_pred == 1) & (y_true == 1))
        fp = np.sum((y_pred == 1) & (y_true == 0))
        fn = np.sum((y_pred == 0) & (y_true == 1))
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = thresh
    
    return best_threshold, best_f1


def find_threshold_at_target_far(
    y_true: np.ndarray,
    p: np.ndarray,
    run_ids: np.ndarray,
    window_len_s: float,
    target_far_per_hour: float,
    thresholds: np.ndarray = None,
) -> Tuple[float, float]:
    """Find threshold that achieves target FAR/hour.
    
    ML Expert Round 3: "Pick thresholds that achieve the same FAR/hour and
    compare TTD + missed-event rate. FAR/hour and TTD at a fixed FAR are
    usually the operational truth."
    
    Returns:
        threshold: Threshold achieving target FAR (or closest below)
        actual_far: Achieved FAR at that threshold
    """
    if thresholds is None:
        thresholds = np.linspace(0.01, 0.99, 200)
    
    # Compute FAR at each threshold
    # FAR = false_positives_per_hour across runs
    best_threshold = thresholds[-1]  # Start with highest (lowest FAR)
    best_far_diff = float('inf')
    
    for thresh in sorted(thresholds, reverse=True):  # High to low thresh
        y_pred = (p >= thresh).astype(int)
        
        # False positives: predicted 1, actual 0
        fp = (y_pred == 1) & (y_true == 0)
        
        # Compute FAR per hour
        unique_runs = np.unique(run_ids)
        total_fp = 0
        total_hours = 0
        for run in unique_runs:
            run_mask = run_ids == run
            run_fp = np.sum(fp[run_mask])
            run_windows = np.sum(run_mask)
            run_hours = run_windows * window_len_s / 3600
            total_fp += run_fp
            total_hours += run_hours
        
        far_per_hour = total_fp / total_hours if total_hours > 0 else 0
        
        # Find threshold closest to target without exceeding
        if far_per_hour <= target_far_per_hour:
            far_diff = target_far_per_hour - far_per_hour
            if far_diff < best_far_diff:
                best_far_diff = far_diff
                best_threshold = thresh
                best_far = far_per_hour
    
    return best_threshold, best_far if best_far_diff < float('inf') else 0.0


def _load_npz(path: Path) -> Dict[str, np.ndarray]:
    data = np.load(path, allow_pickle=True)
    return {k: data[k] for k in data.files}


def _load_static_predictions(
    static_dir: Path,
    binary_ids: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Load static predictions and map to window-level by binary_id."""
    calib_path = static_dir / "static_predictions_calibrated.parquet"
    if calib_path.exists():
        df = pd.read_parquet(calib_path)
        p_col = "p_s_calibrated"
    else:
        df = pd.read_parquet(static_dir / "static_predictions.parquet")
        p_col = "p_s"
    
    binary_to_p = dict(zip(df["binary_id"], df[p_col]))
    binary_to_u = dict(zip(df["binary_id"], df["u_s"]))
    
    p_s = np.array([binary_to_p.get(bid, 0.5) for bid in binary_ids])
    u_s = np.array([binary_to_u.get(bid, 0.25) for bid in binary_ids])
    
    return p_s.astype(np.float32), u_s.astype(np.float32)


def _load_dynamic_predictions(
    dynamic_dir: Path,
    split: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load dynamic predictions for a split."""
    calib_path = dynamic_dir / f"{split}_predictions_calibrated.npz"
    if calib_path.exists():
        data = _load_npz(calib_path)
        p_d = data["p_calibrated"]
    else:
        data = _load_npz(dynamic_dir / f"{split}_predictions.npz")
        p_d = data["p"]
    
    return p_d.astype(np.float32), data["u"].astype(np.float32), data["y"].astype(np.float32)


def _load_fusion_model(fusion_dir: Path) -> UGFFusion:
    """Load trained fusion model."""
    gate = UGFGate(n_meta_features=0, hidden_size=64, use_predictions=True)
    model = UGFFusion(gate)
    model.load_state_dict(torch.load(fusion_dir / "fusion_model.pt", map_location="cpu"))
    model.eval()
    return model


def evaluate_method(
    method_name: str,
    p: np.ndarray,
    y: np.ndarray,
    t_centers: np.ndarray,
    run_ids: np.ndarray,
    runs_dir: Path,
    window_len_s: float,
    threshold: float = 0.5,
) -> Dict[str, float]:
    """Evaluate a fusion method and return metrics."""
    # Aggregate metrics across runs
    all_metrics: Dict[str, List[float]] = {
        "far_per_hour": [],
        "ttd_median_s": [],
        "ttd_p90_s": [],
        "event_f1": [],
        "pr_auc": [],
        "ece": [],
    }
    
    unique_runs = np.unique(run_ids)
    for run_id in unique_runs:
        mask = run_ids == run_id
        if not np.any(mask):
            continue
        
        run_p = p[mask]
        run_y = y[mask]
        run_t = t_centers[mask]
        
        # Load ground truth intervals
        intervals_path = runs_dir / run_id / "intervals.csv"
        intervals = load_intervals_csv(intervals_path)
        
        # Estimate run duration
        run_duration_s = float(run_t.max() - run_t.min()) + window_len_s
        
        # Compute metrics
        metrics = summarize_run_metrics(
            y_true=run_y.astype(int).tolist(),
            p=run_p.tolist(),
            t_centers=run_t.tolist(),
            true_intervals=intervals,
            run_duration_s=run_duration_s,
            window_len_s=window_len_s,
            threshold=threshold,
        )
        
        for k, v in metrics.items():
            if not np.isnan(v):
                all_metrics[k].append(v)
    
    # Aggregate (mean)
    result = {"method": method_name}
    for k, vals in all_metrics.items():
        if vals:
            result[k] = float(np.mean(vals))
            result[f"{k}_std"] = float(np.std(vals))
        else:
            result[k] = float("nan")
            result[f"{k}_std"] = float("nan")
    
    return result


def main() -> int:
    ap = argparse.ArgumentParser(description="Evaluate fusion methods")
    ap.add_argument("--processed-dir", type=Path, required=True)
    ap.add_argument("--static-dir", type=Path, required=True)
    ap.add_argument("--dynamic-dir", type=Path, required=True)
    ap.add_argument("--fusion-dir", type=Path, required=True)
    ap.add_argument("--runs-dir", type=Path, default=Path("data/fusionbench_sim/runs"))
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--threshold", type=float, default=0.3)  # Lower default for imbalanced data
    ap.add_argument("--window-len-s", type=float, default=5.0)
    ap.add_argument("--sweep-thresholds", action="store_true", 
                    help="Sweep thresholds to find optimal per method")
    args = ap.parse_args()

    # Load test data
    test_path = args.processed_dir / "windows_test.npz"
    if not test_path.exists():
        print(f"Test data not found: {test_path}")
        return 1
    
    test_data = _load_npz(test_path)
    y_test_soft = test_data["y"].astype(np.float32)
    y_test = (y_test_soft >= 0.5).astype(np.float32)  # Convert soft to binary for evaluation
    t_centers = test_data.get("t_center", np.arange(len(y_test)))
    run_ids = test_data.get("run_id", np.array(["run_000001"] * len(y_test)))
    binary_ids = test_data.get("binary_id", np.array(["unknown"] * len(y_test)))
    
    print(f"Test set: {len(y_test)} windows, {len(np.unique(run_ids))} runs")

    # Load predictions
    p_s, u_s = _load_static_predictions(args.static_dir, binary_ids)
    p_d, u_d, _ = _load_dynamic_predictions(args.dynamic_dir, "test")
    
    print(f"Static: p_s mean={p_s.mean():.3f}, u_s mean={u_s.mean():.3f}")
    print(f"Dynamic: p_d mean={p_d.mean():.3f}, u_d mean={u_d.mean():.3f}")

    # Load training data for learned baselines
    train_data = _load_npz(args.processed_dir / "windows_train.npz")
    train_binary_ids = train_data.get("binary_id", np.array(["unknown"] * len(train_data["y"])))
    p_s_train, u_s_train = _load_static_predictions(args.static_dir, train_binary_ids)
    p_d_train, u_d_train, y_train_soft = _load_dynamic_predictions(args.dynamic_dir, "train")
    y_train = (y_train_soft >= 0.5).astype(np.float32)  # Convert soft to binary

    results: List[Dict[str, float]] = []

    # Evaluate baselines
    baseline_methods = [
        ("static_only", {"p_s": p_s}),
        ("dynamic_only", {"p_d": p_d}),
        ("late_fusion_avg", {"p_s": p_s, "p_d": p_d}),
        ("late_fusion_learned", {
            "p_s": p_s, "p_d": p_d,
            "y_train": y_train, "p_s_train": p_s_train, "p_d_train": p_d_train,
        }),
        ("product_of_experts", {"p_s": p_s, "p_d": p_d}),
        ("logit_add", {"p_s": p_s, "p_d": p_d}),
        ("heuristic_gate", {"p_s": p_s, "p_d": p_d, "u_d": u_d, "alpha": 5.0}),
        ("heuristic_both_gate", {"p_s": p_s, "p_d": p_d, "u_s": u_s, "u_d": u_d, "alpha": 5.0}),
        # Hierarchical fusion methods
        ("hierarchical", {"p_s": p_s, "p_d": p_d, "u_s": u_s, "u_d": u_d}),
        ("hierarchical_veto", {"p_s": p_s, "p_d": p_d, "u_s": u_s, "u_d": u_d, "cap_threshold": 0.5}),
        # Ablation methods (ML Expert Round 2)
        ("constant_gate", {"p_s": p_s, "p_d": p_d, "g_value": 0.80}),
        ("shuffle_static", {"p_s": p_s, "p_d": p_d, "u_s": u_s, "u_d": u_d, "seed": 42}),
        ("remove_static_pathway", {"p_s": p_s, "p_d": p_d, "u_s": u_s, "u_d": u_d, "g_value": 0.80}),
        # ML Expert recommended methods
        ("soft_veto", {"p_s": p_s, "p_d": p_d, "tau": 0.5, "steepness": 10.0}),
        ("logit_stacking", {
            "p_s": p_s, "p_d": p_d, "u_d": u_d,
            "y_train": y_train, "p_s_train": p_s_train, "p_d_train": p_d_train,
            "u_d_train": u_d_train,
        }),
        # ML Expert Round 3: piecewise-constant gate (interpretable 2-param fusion)
        ("piecewise_gate", {"p_s": p_s, "p_d": p_d, "tau_s": 0.5, "g_trojan": 0.80, "g_benign": 0.95}),
    ]
    
    for method_name, kwargs in baseline_methods:
        print(f"\nEvaluating {method_name}...")
        baseline_fn = BASELINES[method_name]
        result = baseline_fn(**kwargs)
        
        # Find optimal threshold if sweep enabled
        threshold_to_use = args.threshold
        if args.sweep_thresholds:
            opt_thresh, opt_f1 = find_optimal_threshold(y_test, result.p)
            threshold_to_use = opt_thresh
            print(f"  Optimal threshold: {opt_thresh:.2f} (F1={opt_f1:.3f})")
        
        metrics = evaluate_method(
            method_name=method_name,
            p=result.p,
            y=y_test,
            t_centers=t_centers,
            run_ids=run_ids,
            runs_dir=args.runs_dir,
            window_len_s=args.window_len_s,
            threshold=threshold_to_use,
        )
        metrics["threshold"] = threshold_to_use
        results.append(metrics)
        print(f"  FAR/h: {metrics['far_per_hour']:.2f}, TTD: {metrics['ttd_median_s']:.1f}s, F1: {metrics['event_f1']:.3f}")
        # Debug: show score distribution to verify PR-AUC is computed on correct scores
        print(f"  Score stats: mean={result.p.mean():.4f}, std={result.p.std():.4f}, min={result.p.min():.4f}, max={result.p.max():.4f}")

    # Evaluate UGF (only if fusion model exists)
    fusion_model_path = args.fusion_dir / "fusion_model.pt"
    if fusion_model_path.exists():
        print(f"\nEvaluating UGF...")
        model = _load_fusion_model(args.fusion_dir)
        with torch.no_grad():
            p_f, g = model.forward_with_gate(
                torch.from_numpy(p_s),
                torch.from_numpy(p_d),
                torch.from_numpy(u_s),
                torch.from_numpy(u_d),
            )
            p_f = p_f.numpy()
            g = g.numpy()
        
        # Find optimal threshold if sweep enabled
        threshold_to_use = args.threshold
        if args.sweep_thresholds:
            opt_thresh, opt_f1 = find_optimal_threshold(y_test, p_f)
            threshold_to_use = opt_thresh
            print(f"  Optimal threshold: {opt_thresh:.2f} (F1={opt_f1:.3f})")
        
        metrics = evaluate_method(
            method_name="UGF",
            p=p_f,
            y=y_test,
            t_centers=t_centers,
            run_ids=run_ids,
            runs_dir=args.runs_dir,
            window_len_s=args.window_len_s,
            threshold=threshold_to_use,
        )
        metrics["gate_mean"] = float(g.mean())
        metrics["gate_std"] = float(g.std())
        metrics["threshold"] = threshold_to_use
        results.append(metrics)
        print(f"  FAR/h: {metrics['far_per_hour']:.2f}, TTD: {metrics['ttd_median_s']:.1f}s, F1: {metrics['event_f1']:.3f}")
        print(f"  Gate mean: {g.mean():.3f}, std: {g.std():.3f}")
    else:
        print(f"\nSkipping UGF evaluation (no fusion model at {fusion_model_path})")
        print("  Note: Hierarchical fusion methods don't require training.")

    # Save results
    args.out_dir.mkdir(parents=True, exist_ok=True)
    
    df = pd.DataFrame(results)
    df.to_csv(args.out_dir / "results.csv", index=False)
    
    with (args.out_dir / "results.json").open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, default=float)
    
    # Print summary table
    print("\n" + "=" * 80)
    print("RESULTS SUMMARY")
    print("=" * 80)
    print(df[["method", "far_per_hour", "ttd_median_s", "event_f1", "pr_auc", "ece"]].to_string(index=False))
    
    print(f"\nResults saved to {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
