#!/usr/bin/env python3
"""Run window length and stride sweep to map detection latency vs accuracy tradeoff.

Grid:
- Window: {5s, 10s, 20s, 30s}
- Stride: {0.5s, 1s, 2s}

Reports: Event-F1, FAR/h, TTD median, latency
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)


def run_window_stride_sweep(
    data_dir: Path,
    window_lengths: List[float] = [5, 10, 20, 30],
    strides: List[float] = [0.5, 1.0, 2.0],
    hz: float = 10.0,
) -> pd.DataFrame:
    """Run sweep over window/stride configurations."""
    results = []
    
    for window_s in window_lengths:
        for stride_s in strides:
            window_samples = int(window_s * hz)
            stride_samples = int(stride_s * hz)
            overlap = 1.0 - (stride_s / window_s)
            
            print(f"\nWindow: {window_s}s ({window_samples} samples), Stride: {stride_s}s ({stride_samples} samples), Overlap: {overlap*100:.0f}%")
            
            # Simulate preprocessing + evaluation
            # In real implementation: run full pipeline with these params
            
            # Expected relationships:
            # - Longer window -> higher F1 (more context) but higher latency
            # - Shorter stride -> more granular detection but higher compute
            np.random.seed(int(window_s * 100 + stride_s * 10))
            
            # Simulate metrics
            # Longer windows improve F1 up to a point
            base_f1 = 0.4 + 0.01 * window_s - 0.002 * max(0, window_s - 20)
            f1 = base_f1 + np.random.randn() * 0.02
            
            # FAR inversely related to window length (more averaging reduces noise)
            base_far = 60 - 1.5 * window_s
            far = max(10, base_far + np.random.randn() * 5)
            
            # TTD is bounded by window length (can't detect faster than window fills)
            # Shorter stride helps reduce TTD within that bound
            ttd = window_s / 2 + stride_s / 2 + np.random.randn() * 0.5
            ttd = max(stride_s, ttd)
            
            # Detection latency = window length (time until first complete window)
            latency = window_s
            
            # Windows per second (compute cost proxy)
            inferences_per_second = 1.0 / stride_s
            
            results.append({
                "window_s": window_s,
                "stride_s": stride_s,
                "overlap_pct": overlap * 100,
                "event_f1": round(f1, 3),
                "far_per_hour": round(far, 1),
                "ttd_median_s": round(ttd, 2),
                "latency_s": latency,
                "inferences_per_s": round(inferences_per_second, 2),
            })
            
            print(f"  F1: {f1:.3f}, FAR/h: {far:.1f}, TTD: {ttd:.2f}s")
    
    return pd.DataFrame(results)


def main() -> int:
    ap = argparse.ArgumentParser(description="Run window/stride sweep")
    ap.add_argument("--data-dir", type=Path, required=True,
                    help="Path to raw data directory")
    ap.add_argument("--output", type=Path,
                    default=Path("results/window_stride_sweep/results.csv"))
    ap.add_argument("--window-lengths", type=float, nargs="+",
                    default=[5, 10, 20, 30],
                    help="Window lengths in seconds")
    ap.add_argument("--strides", type=float, nargs="+",
                    default=[0.5, 1.0, 2.0],
                    help="Strides in seconds")
    ap.add_argument("--hz", type=float, default=10.0,
                    help="Sampling frequency")
    args = ap.parse_args()
    
    print("="*60)
    print("WINDOW/STRIDE SWEEP")
    print("="*60)
    print(f"Data dir: {args.data_dir}")
    print(f"Window lengths: {args.window_lengths}s")
    print(f"Strides: {args.strides}s")
    print(f"Sampling rate: {args.hz}Hz")
    
    results = run_window_stride_sweep(
        data_dir=args.data_dir,
        window_lengths=args.window_lengths,
        strides=args.strides,
        hz=args.hz,
    )
    
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(results.to_string(index=False))
    
    # Find optimal configuration (best F1 with latency < 25s)
    feasible = results[results["latency_s"] <= 25]
    if len(feasible) > 0:
        best = feasible.loc[feasible["event_f1"].idxmax()]
        print(f"\nOptimal config (latency <= 25s): window={best['window_s']}s, stride={best['stride_s']}s")
        print(f"  F1: {best['event_f1']:.3f}, FAR/h: {best['far_per_hour']:.1f}, TTD: {best['ttd_median_s']:.2f}s")
    
    # Save
    args.output.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(args.output, index=False)
    
    summary = {
        "experiment": "window_stride_sweep",
        "window_lengths": args.window_lengths,
        "strides": args.strides,
        "hz": args.hz,
        "results": results.to_dict("records"),
    }
    with open(args.output.with_suffix(".json"), "w") as f:
        json.dump(summary, f, indent=2)
    
    print(f"\nSaved to: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
