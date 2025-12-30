#!/usr/bin/env python3
"""Create contaminated warmup split for cold-start experiment.

This creates a split where trojan activates during the warmup period,
testing whether per-run normalization degrades and LLM-static helps.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List
import random

import numpy as np
import pandas as pd


def contaminate_warmup(
    runs_dir: Path,
    run_id: str,
    warmup_steps: int = 200,
    contamination_start: int = 0,
    contamination_duration_steps: int = 50,
) -> Dict:
    """Modify intervals to have trojan active during warmup.
    
    Returns metadata about the contamination.
    """
    intervals_path = runs_dir / run_id / "intervals.csv"
    
    if not intervals_path.exists():
        return {"run_id": run_id, "contaminated": False, "reason": "no intervals file"}
    
    # Read existing intervals (file has headers)
    try:
        intervals_df = pd.read_csv(intervals_path)
    except pd.errors.EmptyDataError:
        return {"run_id": run_id, "contaminated": False, "reason": "empty intervals file"}
    
    # Check for trojan intervals (non-empty with actual data)
    if len(intervals_df) == 0 or intervals_df.empty:
        return {"run_id": run_id, "contaminated": False, "reason": "no trojan intervals"}
    
    # The file has columns: t_start_sim_s, t_end_sim_s, trojan_variant, ...
    if "t_start_sim_s" not in intervals_df.columns:
        return {"run_id": run_id, "contaminated": False, "reason": "wrong column format"}
    
    # Add a warmup contamination interval
    # Assume 10 Hz sampling (0.1s per step)
    dt = 0.1  # seconds per step
    contamination_start_s = contamination_start * dt
    contamination_end_s = (contamination_start + contamination_duration_steps) * dt
    
    # Create new interval that overlaps warmup
    new_interval = pd.DataFrame({
        "t_start_sim_s": [contamination_start_s],
        "t_end_sim_s": [contamination_end_s],
    })
    
    # Combine with existing intervals (only keep time columns)
    existing_times = intervals_df[["t_start_sim_s", "t_end_sim_s"]].copy()
    contaminated_df = pd.concat([new_interval, existing_times], ignore_index=True)
    contaminated_df = contaminated_df.sort_values("t_start_sim_s").reset_index(drop=True)
    
    return {
        "run_id": run_id,
        "contaminated": True,
        "contamination_start_s": contamination_start_s,
        "contamination_end_s": contamination_end_s,
        "original_intervals": len(intervals_df),
        "new_intervals": len(contaminated_df),
        "intervals": contaminated_df.to_dict("records"),
    }


def create_coldstart_split(
    base_split_path: Path,
    runs_dir: Path,
    output_path: Path,
    contamination_fraction: float = 0.5,
    warmup_steps: int = 200,
    seed: int = 42,
) -> None:
    """Create a cold-start split where some runs have contaminated warmup.
    
    Args:
        base_split_path: Path to original split JSON
        runs_dir: Path to runs directory
        output_path: Path to save new split JSON
        contamination_fraction: Fraction of test runs to contaminate
        warmup_steps: Number of warmup steps (for reference)
        seed: Random seed
    """
    random.seed(seed)
    
    with open(base_split_path) as f:
        base_split = json.load(f)
    
    # Get test runs that have intervals (trojan runs)
    test_runs = base_split.get("test_runs", [])
    trojan_test_runs = []
    
    for run_id in test_runs:
        intervals_path = runs_dir / run_id / "intervals.csv"
        if intervals_path.exists():
            intervals_df = pd.read_csv(intervals_path)
            # Check for actual trojan intervals (t_start_sim_s column with data)
            if "t_start_sim_s" in intervals_df.columns and len(intervals_df) > 0:
                trojan_test_runs.append(run_id)
    
    # Select runs to contaminate
    n_contaminate = int(len(trojan_test_runs) * contamination_fraction)
    contaminated_runs = random.sample(trojan_test_runs, n_contaminate)
    
    # Create contamination info
    contamination_info = []
    for run_id in contaminated_runs:
        info = contaminate_warmup(
            runs_dir, run_id, warmup_steps,
            contamination_start=50,  # Start at step 50
            contamination_duration_steps=100,  # Run for 100 steps (10 seconds)
        )
        contamination_info.append(info)
    
    # Create new split
    coldstart_split = {
        "train_runs": base_split.get("train_runs", []),
        "val_runs": base_split.get("val_runs", []),
        "test_runs": test_runs,
        "notes": f"Cold-start split: {n_contaminate}/{len(trojan_test_runs)} trojan test runs have warmup contamination",
        "contaminated_runs": contaminated_runs,
        "contamination_info": contamination_info,
        "warmup_steps": warmup_steps,
    }
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(coldstart_split, f, indent=2)
    
    print(f"Created cold-start split: {output_path}")
    print(f"  Total test runs: {len(test_runs)}")
    print(f"  Trojan test runs: {len(trojan_test_runs)}")
    print(f"  Contaminated runs: {n_contaminate}")
    print(f"  Contamination: steps 50-150 (5-15 seconds)")


def main():
    ap = argparse.ArgumentParser(description="Create cold-start split")
    ap.add_argument("--base-split", type=Path, required=True,
                    help="Path to base split JSON (e.g., random_split.json)")
    ap.add_argument("--runs-dir", type=Path, default=Path("data/fusionbench_sim/runs"))
    ap.add_argument("--output", type=Path, default=Path("data/fusionbench_sim/splits/coldstart.json"))
    ap.add_argument("--contamination-fraction", type=float, default=0.5)
    ap.add_argument("--warmup-steps", type=int, default=200)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    
    create_coldstart_split(
        base_split_path=args.base_split,
        runs_dir=args.runs_dir,
        output_path=args.output,
        contamination_fraction=args.contamination_fraction,
        warmup_steps=args.warmup_steps,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
