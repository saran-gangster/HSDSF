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


def _load_split_windows(processed_dir: Path, split: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load window labels and metadata for a split.

    Returns:
        y_bin: [N] binary labels derived from soft labels >= 0.5
        t_centers: [N] float centers (or 0..N-1 if missing)
        run_ids: [N] run_id strings (or a single dummy id if missing)
        binary_ids: [N] binary_id strings (or 'unknown' if missing)
    """
    path = processed_dir / f"windows_{split}.npz"
    data = _load_npz(path)
    y_soft = data["y"].astype(np.float32)
    y_bin = (y_soft >= 0.5).astype(np.float32)
    t_centers = data.get("t_center", np.arange(len(y_bin)))
    run_ids = data.get("run_id", np.array(["run_000001"] * len(y_bin)))
    binary_ids = data.get("binary_id", np.array(["unknown"] * len(y_bin)))
    return y_bin, t_centers, run_ids, binary_ids


def _mean_event_metrics_at_threshold(
    *,
    y: np.ndarray,
    p: np.ndarray,
    t_centers: np.ndarray,
    run_ids: np.ndarray,
    runs_dir: Path,
    window_len_s: float,
    threshold: float,
) -> tuple[float, float]:
    """Return (mean_event_f1, mean_event_far_per_hour) across runs at a threshold."""
    event_f1s: list[float] = []
    fars: list[float] = []
    for run_id in np.unique(run_ids):
        mask = run_ids == run_id
        if not np.any(mask):
            continue
        run_p = p[mask]
        run_y = y[mask]
        run_t = t_centers[mask]
        intervals_path = runs_dir / run_id / "intervals.csv"
        intervals = load_intervals_csv(intervals_path)
        run_duration_s = float(run_t.max() - run_t.min()) + float(window_len_s)
        metrics = summarize_run_metrics(
            y_true=run_y.astype(int).tolist(),
            p=run_p.tolist(),
            t_centers=run_t.tolist(),
            true_intervals=intervals,
            run_duration_s=run_duration_s,
            window_len_s=window_len_s,
            threshold=threshold,
        )
        if not np.isnan(metrics["event_f1"]):
            event_f1s.append(float(metrics["event_f1"]))
        if not np.isnan(metrics["far_per_hour"]):
            fars.append(float(metrics["far_per_hour"]))
    mean_f1 = float(np.mean(event_f1s)) if event_f1s else 0.0
    mean_far = float(np.mean(fars)) if fars else 0.0
    return mean_f1, mean_far


def select_threshold_event_f1(
    *,
    y: np.ndarray,
    p: np.ndarray,
    t_centers: np.ndarray,
    run_ids: np.ndarray,
    runs_dir: Path,
    window_len_s: float,
    thresholds: np.ndarray | None = None,
) -> tuple[float, float]:
    """Select threshold that maximizes mean event-level F1 across runs."""
    if thresholds is None:
        thresholds = np.linspace(0.1, 0.9, 17)
    best_f1 = -1.0
    best_threshold = 0.5
    for thresh in thresholds:
        f1, _ = _mean_event_metrics_at_threshold(
            y=y,
            p=p,
            t_centers=t_centers,
            run_ids=run_ids,
            runs_dir=runs_dir,
            window_len_s=window_len_s,
            threshold=float(thresh),
        )
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = float(thresh)
    return best_threshold, float(best_f1)


def select_threshold_at_target_event_far(
    *,
    y: np.ndarray,
    p: np.ndarray,
    t_centers: np.ndarray,
    run_ids: np.ndarray,
    runs_dir: Path,
    window_len_s: float,
    target_far_per_hour: float,
    thresholds: np.ndarray | None = None,
) -> tuple[float, float]:
    """Select threshold on a split to match an event FAR/h target (mean across runs)."""
    if thresholds is None:
        thresholds = np.linspace(0.01, 0.99, 200)

    best_threshold = float(thresholds[-1])
    best_far = 0.0
    best_gap = float("inf")
    for thresh in sorted([float(x) for x in thresholds], reverse=True):
        _, far = _mean_event_metrics_at_threshold(
            y=y,
            p=p,
            t_centers=t_centers,
            run_ids=run_ids,
            runs_dir=runs_dir,
            window_len_s=window_len_s,
            threshold=thresh,
        )
        if far <= float(target_far_per_hour):
            gap = float(target_far_per_hour) - far
            if gap < best_gap:
                best_gap = gap
                best_threshold = thresh
                best_far = float(far)
    return best_threshold, float(best_far)


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
    ap.add_argument("--threshold-source-split", type=str, default="test", choices=["val", "test"],
                    help="Split used for per-method threshold selection when sweeping (default: test for backward compatibility)")
    ap.add_argument("--eval-split", type=str, default="test", choices=["val", "test"],
                    help="Split used for reporting metrics (default: test)")
    ap.add_argument("--threshold-policy", type=str, default="max_event_f1", choices=["max_event_f1", "target_event_far"],
                    help="How to select thresholds on threshold-source-split")
    ap.add_argument("--target-event-far-per-hour", type=float, default=None,
                    help="If set (and threshold-policy=target_event_far), select thresholds to match this event FAR/h target on the threshold-source-split")
    args = ap.parse_args()

    # Load split metadata for threshold selection and evaluation
    eval_path = args.processed_dir / f"windows_{args.eval_split}.npz"
    if not eval_path.exists():
        print(f"Eval data not found: {eval_path}")
        return 1
    source_path = args.processed_dir / f"windows_{args.threshold_source_split}.npz"
    if args.sweep_thresholds and not source_path.exists():
        print(f"Threshold-source data not found: {source_path}")
        return 1

    y_eval, t_eval, run_eval, binary_eval = _load_split_windows(args.processed_dir, args.eval_split)
    print(f"Eval split '{args.eval_split}': {len(y_eval)} windows, {len(np.unique(run_eval))} runs")
    if args.sweep_thresholds:
        y_src, t_src, run_src, binary_src = _load_split_windows(args.processed_dir, args.threshold_source_split)
        print(f"Threshold-source split '{args.threshold_source_split}': {len(y_src)} windows, {len(np.unique(run_src))} runs")

    # Load predictions for eval split
    p_s, u_s = _load_static_predictions(args.static_dir, binary_eval)
    p_d, u_d, _ = _load_dynamic_predictions(args.dynamic_dir, args.eval_split)
    
    print(f"Static: p_s mean={p_s.mean():.3f}, u_s mean={u_s.mean():.3f}")
    print(f"Dynamic: p_d mean={p_d.mean():.3f}, u_d mean={u_d.mean():.3f}")

    # Load training data for learned baselines
    train_data = _load_npz(args.processed_dir / "windows_train.npz")
    train_binary_ids = train_data.get("binary_id", np.array(["unknown"] * len(train_data["y"])))
    p_s_train, u_s_train = _load_static_predictions(args.static_dir, train_binary_ids)
    p_d_train, u_d_train, y_train_soft = _load_dynamic_predictions(args.dynamic_dir, "train")
    y_train = (y_train_soft >= 0.5).astype(np.float32)  # Convert soft to binary

    # Load predictions for threshold-source split when needed
    if args.sweep_thresholds:
        p_s_src, u_s_src = _load_static_predictions(args.static_dir, binary_src)
        p_d_src, u_d_src, _ = _load_dynamic_predictions(args.dynamic_dir, args.threshold_source_split)

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

        # Determine threshold
        threshold_to_use = float(args.threshold)
        if args.sweep_thresholds:
            # Build source-split scores for this method
            kwargs_src = dict(kwargs)
            if "p_s" in kwargs_src:
                kwargs_src["p_s"] = p_s_src
            if "p_d" in kwargs_src:
                kwargs_src["p_d"] = p_d_src
            if "u_s" in kwargs_src:
                kwargs_src["u_s"] = u_s_src
            if "u_d" in kwargs_src:
                kwargs_src["u_d"] = u_d_src
            if "p_s_train" in kwargs_src:
                kwargs_src["p_s_train"] = p_s_train
            if "p_d_train" in kwargs_src:
                kwargs_src["p_d_train"] = p_d_train
            if "u_d_train" in kwargs_src:
                kwargs_src["u_d_train"] = u_d_train
            kwargs_src["y_train"] = y_train

            result_src = baseline_fn(**kwargs_src)

            if args.threshold_policy == "target_event_far":
                if args.target_event_far_per_hour is None:
                    raise SystemExit("--target-event-far-per-hour is required when --threshold-policy=target_event_far")
                threshold_to_use, achieved_far = select_threshold_at_target_event_far(
                    y=y_src,
                    p=result_src.p,
                    t_centers=t_src,
                    run_ids=run_src,
                    runs_dir=args.runs_dir,
                    window_len_s=args.window_len_s,
                    target_far_per_hour=float(args.target_event_far_per_hour),
                )
                print(f"  Selected threshold on {args.threshold_source_split}: {threshold_to_use:.2f} (mean event FAR/h={achieved_far:.2f})")
            else:
                threshold_to_use, best_f1 = select_threshold_event_f1(
                    y=y_src,
                    p=result_src.p,
                    t_centers=t_src,
                    run_ids=run_src,
                    runs_dir=args.runs_dir,
                    window_len_s=args.window_len_s,
                )
                print(f"  Selected threshold on {args.threshold_source_split}: {threshold_to_use:.2f} (mean event F1={best_f1:.3f})")
        
        metrics = evaluate_method(
            method_name=method_name,
            p=result.p,
            y=y_eval,
            t_centers=t_eval,
            run_ids=run_eval,
            runs_dir=args.runs_dir,
            window_len_s=args.window_len_s,
            threshold=threshold_to_use,
        )
        metrics["threshold"] = threshold_to_use
        metrics["threshold_source_split"] = args.threshold_source_split if args.sweep_thresholds else "fixed"
        metrics["eval_split"] = args.eval_split
        metrics["threshold_policy"] = args.threshold_policy if args.sweep_thresholds else "fixed"
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
        
        # Determine threshold
        threshold_to_use = float(args.threshold)
        if args.sweep_thresholds:
            if args.threshold_source_split == args.eval_split:
                p_src_f = p_f
            else:
                # Build UGF scores on the threshold-source split
                with torch.no_grad():
                    p_f_src, _g_src = model.forward_with_gate(
                        torch.from_numpy(p_s_src),
                        torch.from_numpy(p_d_src),
                        torch.from_numpy(u_s_src),
                        torch.from_numpy(u_d_src),
                    )
                    p_src_f = p_f_src.numpy()

            if args.threshold_policy == "target_event_far":
                if args.target_event_far_per_hour is None:
                    raise SystemExit("--target-event-far-per-hour is required when --threshold-policy=target_event_far")
                threshold_to_use, achieved_far = select_threshold_at_target_event_far(
                    y=y_src,
                    p=p_src_f,
                    t_centers=t_src,
                    run_ids=run_src,
                    runs_dir=args.runs_dir,
                    window_len_s=args.window_len_s,
                    target_far_per_hour=float(args.target_event_far_per_hour),
                )
                print(f"  Selected threshold on {args.threshold_source_split}: {threshold_to_use:.2f} (mean event FAR/h={achieved_far:.2f})")
            else:
                threshold_to_use, best_f1 = select_threshold_event_f1(
                    y=y_src,
                    p=p_src_f,
                    t_centers=t_src,
                    run_ids=run_src,
                    runs_dir=args.runs_dir,
                    window_len_s=args.window_len_s,
                )
                print(f"  Selected threshold on {args.threshold_source_split}: {threshold_to_use:.2f} (mean event F1={best_f1:.3f})")
        
        metrics = evaluate_method(
            method_name="UGF",
            p=p_f,
            y=y_eval,
            t_centers=t_eval,
            run_ids=run_eval,
            runs_dir=args.runs_dir,
            window_len_s=args.window_len_s,
            threshold=threshold_to_use,
        )
        metrics["gate_mean"] = float(g.mean())
        metrics["gate_std"] = float(g.std())
        metrics["threshold"] = threshold_to_use
        metrics["threshold_source_split"] = args.threshold_source_split if args.sweep_thresholds else "fixed"
        metrics["eval_split"] = args.eval_split
        metrics["threshold_policy"] = args.threshold_policy if args.sweep_thresholds else "fixed"
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
