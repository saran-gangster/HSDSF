from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple


@dataclass(frozen=True)
class Interval:
    start_s: float
    end_s: float


def load_intervals_csv(path: Path) -> List[Interval]:
    if not path.exists():
        return []
    out: List[Interval] = []
    with path.open(newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            try:
                s = float(row["t_start_sim_s"])
                e = float(row["t_end_sim_s"])
            except Exception:
                continue
            if e > s:
                out.append(Interval(s, e))
    return out


def interval_overlap(a: Interval, b: Interval) -> float:
    s = max(a.start_s, b.start_s)
    e = min(a.end_s, b.end_s)
    return max(0.0, e - s)


def window_label_soft(
    *,
    t_start: float,
    t_end: float,
    intervals: Sequence[Interval],
) -> float:
    """Compute soft label: fraction of window overlapping with trojan intervals.
    
    Returns a value in [0, 1] representing total overlap fraction.
    This is better for training as it handles boundary windows smoothly.
    """
    if t_end <= t_start:
        return 0.0
    w = Interval(t_start, t_end)
    wlen = t_end - t_start
    
    # Sum overlap with all intervals (clamped to window length)
    total_overlap = 0.0
    for it in intervals:
        total_overlap += interval_overlap(w, it)
    
    # Clamp to [0, 1] in case of overlapping intervals
    return min(1.0, total_overlap / wlen)


def window_label(
    *,
    t_start: float,
    t_end: float,
    intervals: Sequence[Interval],
    overlap_threshold: float = 0.5,
) -> int:
    """Binary label: 1 if overlap fraction >= threshold, else 0."""
    overlap_frac = window_label_soft(t_start=t_start, t_end=t_end, intervals=intervals)
    return 1 if overlap_frac >= overlap_threshold else 0


def windows_to_events(
    t_centers: Sequence[float],
    y_pred: Sequence[int],
    *,
    window_len_s: float,
) -> List[Interval]:
    """Merge contiguous positive windows into predicted intervals."""
    assert len(t_centers) == len(y_pred)
    if not t_centers:
        return []

    half = 0.5 * float(window_len_s)
    events: List[Interval] = []
    cur_s: float | None = None
    cur_e: float | None = None
    for t, y in zip(t_centers, y_pred):
        if int(y) == 1:
            s = float(t) - half
            e = float(t) + half
            if cur_s is None:
                cur_s, cur_e = s, e
            else:
                # if overlapping/touching, merge
                if s <= float(cur_e):
                    cur_e = max(float(cur_e), e)
                else:
                    events.append(Interval(float(cur_s), float(cur_e)))
                    cur_s, cur_e = s, e
        else:
            if cur_s is not None:
                events.append(Interval(float(cur_s), float(cur_e)))
                cur_s = cur_e = None
    if cur_s is not None:
        events.append(Interval(float(cur_s), float(cur_e)))
    return events


def match_events_iou(
    pred: Sequence[Interval],
    true: Sequence[Interval],
    *,
    iou_threshold: float = 0.1,
) -> Tuple[int, int, int]:
    """Return (tp, fp, fn) matching by maximum IoU greedy assignment."""
    if not pred and not true:
        return 0, 0, 0
    used_true = [False] * len(true)
    tp = 0
    fp = 0

    def iou(a: Interval, b: Interval) -> float:
        inter = interval_overlap(a, b)
        union = (a.end_s - a.start_s) + (b.end_s - b.start_s) - inter
        return inter / union if union > 0 else 0.0

    for p in pred:
        best_j = -1
        best = 0.0
        for j, t in enumerate(true):
            if used_true[j]:
                continue
            v = iou(p, t)
            if v > best:
                best = v
                best_j = j
        if best_j >= 0 and best >= iou_threshold:
            used_true[best_j] = True
            tp += 1
        else:
            fp += 1
    fn = sum(1 for u in used_true if not u)
    return tp, fp, fn


def time_to_detect(true: Sequence[Interval], pred: Sequence[Interval]) -> List[float]:
    """For each true interval, compute delay to first overlapping predicted event."""
    delays: List[float] = []
    for t in true:
        start = float(t.start_s)
        best: float | None = None
        for p in pred:
            if interval_overlap(t, p) > 0.0 and p.end_s >= start:
                best = float(p.start_s) - start if float(p.start_s) >= start else 0.0
                break
        if best is not None:
            delays.append(best)
    return delays


def time_to_detect_emission(
    true: Sequence[Interval],
    *,
    t_centers: Sequence[float],
    y_pred: Sequence[int],
    window_len_s: float,
) -> List[float]:
    """Causal TTD using emission times.

    We treat each window score as being emitted at the window end time:
        t_emit = t_center + 0.5 * window_len_s

    For each true interval starting at t0, we find the first emitted positive window
    whose window interval overlaps the true interval, and compute:
        max(0, t_emit - t0)

    This avoids backdating detection earlier than the score emission time.
    """
    assert len(t_centers) == len(y_pred)
    half = 0.5 * float(window_len_s)

    # Precompute emitted positives (emit time + window interval)
    positives: List[tuple[float, Interval]] = []
    for tc, yp in zip(t_centers, y_pred):
        if int(yp) != 1:
            continue
        tc_f = float(tc)
        w = Interval(tc_f - half, tc_f + half)
        t_emit = tc_f + half
        positives.append((t_emit, w))

    positives.sort(key=lambda x: x[0])

    delays: List[float] = []
    for it in true:
        t0 = float(it.start_s)
        best: float | None = None
        for t_emit, w in positives:
            if t_emit < t0:
                # emitted before the event starts; cannot count for causal detection
                continue
            if interval_overlap(it, w) > 0.0:
                best = max(0.0, float(t_emit) - t0)
                break
        if best is not None:
            delays.append(best)
    return delays
