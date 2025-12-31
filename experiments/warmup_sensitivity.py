#!/usr/bin/env python3
"""Warmup Sensitivity Experiment.

Tests performance degradation as warmup_steps varies from 50 to 400.
This is a key systems-style plot for the paper.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd


def run_cmd(cmd: List[str], cwd: Path) -> int:
    """Run command and return exit code."""
    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=cwd)
    return result.returncode


def run_experiment(
    warmup_steps: int,
    split_path: Path,
    runs_dir: Path,
    base_out_dir: Path,
    repo_root: Path,
) -> Dict[str, float]:
    """Run preprocessing, training, and evaluation for one warmup setting."""
    
    suffix = f"warmup_{warmup_steps}"
    processed_dir = base_out_dir / f"processed_{suffix}"
    model_dir = base_out_dir / f"dynamic_{suffix}"
    results_dir = base_out_dir / f"results_{suffix}"
    fusion_dir = base_out_dir / f"fusion_{suffix}"
    
    # Create directories
    processed_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)
    fusion_dir.mkdir(parents=True, exist_ok=True)
    
    # Step 1: Preprocess with per-run norm and specific warmup
    cmd = [
        sys.executable, "dynamic/preprocess.py",
        "--split", str(split_path),
        "--out-dir", str(processed_dir),
        "--per-run-norm",
        "--warmup-steps", str(warmup_steps),
    ]
    if run_cmd(cmd, repo_root) != 0:
        return {"warmup_steps": warmup_steps, "error": "preprocess failed"}
    
    # Step 2: Train dynamic model (minimal settings for speed)
    cmd = [
        sys.executable, "dynamic/train_dynamic.py",
        "--processed-dir", str(processed_dir),
        "--out-dir", str(model_dir),
        "--model", "tcn",
        "--n-ensemble", "1",
        "--epochs", "20",
        "--batch-size", "128",
    ]
    if run_cmd(cmd, repo_root) != 0:
        return {"warmup_steps": warmup_steps, "error": "training failed"}
    
    # Step 3: Calibrate
    cmd = [
        sys.executable, "dynamic/calibrate_dynamic.py",
        "--model-dir", str(model_dir),
    ]
    run_cmd(cmd, repo_root)  # Non-fatal if fails
    
    # Step 4: Evaluate (dynamic_only is sufficient for this experiment)
    cmd = [
        sys.executable, "fusion/eval_fusion.py",
        "--processed-dir", str(processed_dir),
        "--static-dir", str(base_out_dir / "static"),
        "--dynamic-dir", str(model_dir),
        "--fusion-dir", str(fusion_dir),
        "--runs-dir", str(runs_dir),
        "--out-dir", str(results_dir),
        "--sweep-thresholds",
    ]
    if run_cmd(cmd, repo_root) != 0:
        return {"warmup_steps": warmup_steps, "error": "evaluation failed"}
    
    # Read results
    results_file = results_dir / "results.json"
    if not results_file.exists():
        return {"warmup_steps": warmup_steps, "error": "no results file"}
    
    with open(results_file) as f:
        results = json.load(f)
    
    # Extract dynamic_only metrics
    for method in results:
        if method["method"] == "dynamic_only":
            return {
                "warmup_steps": warmup_steps,
                "event_f1": method["event_f1"],
                "far_per_hour": method["far_per_hour"],
                "ttd_median_s": method["ttd_median_s"],
                "pr_auc": method["pr_auc"],
            }
    
    return {"warmup_steps": warmup_steps, "error": "dynamic_only not found"}


def main():
    ap = argparse.ArgumentParser(description="Warmup sensitivity experiment")
    ap.add_argument("--split", type=Path, required=True,
                    help="Path to split JSON")
    ap.add_argument("--runs-dir", type=Path, default=Path("data/fusionbench_sim/runs"))
    ap.add_argument("--out-dir", type=Path, default=Path("experiments/warmup_sensitivity"))
    ap.add_argument("--warmup-values", type=int, nargs="+", default=[50, 100, 200, 400])
    ap.add_argument("--static-dir", type=Path, default=Path("models/static"),
                    help="Path to static model (will be copied to out-dir)")
    args = ap.parse_args()
    
    repo_root = Path(__file__).parent.parent.absolute()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    
    # Copy/link static model to output dir
    static_dest = args.out_dir / "static"
    if not static_dest.exists():
        import shutil
        shutil.copytree(args.static_dir, static_dest)
    
    results = []
    for warmup in args.warmup_values:
        print(f"\n{'='*60}")
        print(f"Warmup Steps: {warmup}")
        print(f"{'='*60}")
        
        result = run_experiment(
            warmup_steps=warmup,
            split_path=args.split,
            runs_dir=args.runs_dir,
            base_out_dir=args.out_dir,
            repo_root=repo_root,
        )
        results.append(result)
        print(f"Result: {result}")
    
    # Save results
    results_df = pd.DataFrame(results)
    results_df.to_csv(args.out_dir / "warmup_sensitivity.csv", index=False)
    
    # Print summary
    print("\n" + "="*60)
    print("WARMUP SENSITIVITY RESULTS")
    print("="*60)
    print(results_df.to_string(index=False))
    
    # Save JSON too
    with open(args.out_dir / "warmup_sensitivity.json", "w") as f:
        json.dump(results, f, indent=2)
    
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
