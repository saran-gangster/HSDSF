from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
from sklearn.metrics import average_precision_score

from evaluation.events import Interval, match_events_iou, time_to_detect, windows_to_events


@dataclass(frozen=True)
class Metrics:
    far_per_hour: float
    ttd_median_s: float
    ttd_p90_s: float
    event_f1: float
    pr_auc: float
    ece: float


def _ece_binary(y_true: np.ndarray, p: np.ndarray, *, n_bins: int = 15) -> float:
    y_true = y_true.astype(np.float64)
    p = np.clip(p.astype(np.float64), 1e-9, 1.0 - 1e-9)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        mask = (p >= lo) & (p < hi) if i < n_bins - 1 else (p >= lo) & (p <= hi)
        if not np.any(mask):
            continue
        acc = float(np.mean(y_true[mask]))
        conf = float(np.mean(p[mask]))
        ece += float(np.mean(mask)) * abs(acc - conf)
    return float(ece)


def _events_from_window_probs(
    t_centers: Sequence[float],
    p: Sequence[float],
    *,
    threshold: float,
    window_len_s: float,
) -> List[Interval]:
    y_pred = [1 if float(x) >= float(threshold) else 0 for x in p]
    return windows_to_events(t_centers, y_pred, window_len_s=window_len_s)


def far_per_hour(
    *,
    pred_events: Sequence[Interval],
    duration_s: float,
    true_intervals: Sequence[Interval],
) -> float:
    """False alarms per hour, excluding time covered by true intervals."""
    if duration_s <= 0:
        return 0.0
    true_time = sum(max(0.0, it.end_s - it.start_s) for it in true_intervals)
    benign_time_s = max(1e-9, float(duration_s) - float(true_time))

    # count predicted events that do not overlap any true interval
    fp = 0
    for p in pred_events:
        overlaps = any(max(0.0, min(p.end_s, t.end_s) - max(p.start_s, t.start_s)) > 0 for t in true_intervals)
        if not overlaps:
            fp += 1
    return float(fp) / (benign_time_s / 3600.0)


def event_f1(tp: int, fp: int, fn: int) -> float:
    denom = (2 * tp + fp + fn)
    return float(2 * tp / denom) if denom > 0 else 0.0


def summarize_run_metrics(
    *,
    y_true: Sequence[int],
    p: Sequence[float],
    t_centers: Sequence[float],
    true_intervals: Sequence[Interval],
    run_duration_s: float,
    window_len_s: float,
    threshold: float,
    iou_threshold: float = 0.1,
) -> Dict[str, float]:
    y_true_np = np.asarray(y_true, dtype=np.int64)
    p_np = np.asarray(p, dtype=np.float64)
    pr_auc = float(average_precision_score(y_true_np, p_np)) if len(np.unique(y_true_np)) > 1 else float("nan")
    ece = _ece_binary(y_true_np, p_np)

    pred_events = _events_from_window_probs(t_centers, p, threshold=threshold, window_len_s=window_len_s)
    tp, fp, fn = match_events_iou(pred_events, true_intervals, iou_threshold=iou_threshold)
    ef1 = event_f1(tp, fp, fn)
    far = far_per_hour(pred_events=pred_events, duration_s=run_duration_s, true_intervals=true_intervals)
    delays = time_to_detect(true_intervals, pred_events)
    if delays:
        ttd_med = float(np.median(delays))
        ttd_p90 = float(np.percentile(delays, 90))
    else:
        ttd_med = float("nan")
        ttd_p90 = float("nan")

    return {
        "far_per_hour": float(far),
        "ttd_median_s": float(ttd_med),
        "ttd_p90_s": float(ttd_p90),
        "event_f1": float(ef1),
        "pr_auc": float(pr_auc),
        "ece": float(ece),
    }
