#!/usr/bin/env python3
"""Generate FusionBench-Sim dataset with RANDOMIZED activation timing.

This addresses the validity threat that deterministic periodic patterns
(period=40s, duration=10s) may allow the model to learn schedule rather than
actual trojan signatures.

Randomization:
- Random start offset (uniform 0 to period)
- Random inter-arrival times (exponential with mean = period)
- Random durations (uniform 0.5x to 1.5x of base duration)
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path
from typing import List

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from experiments.genlib import (
    generate_run,
    sample_device_instance,
    write_intervals_csv_randomized,
)
from simulator.jetson_sim import SimConfig


WORKLOAD_FAMILIES = [
    "inference_periodic",
    "streaming_steady_gpu",
    "memory_bound",
    "network_bursty",
    "cv_heavy",
]

TROJAN_FAMILIES = [
    "none",
    "mempressure",
    "compute",
    "cv",
    "io",
]


def _write_binaries_csv(path: Path, binary_ids: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["binary_id", "binary_path"])
        for bid in sorted(set(binary_ids)):
            w.writerow([bid, ""])


def main() -> int:
    ap = argparse.ArgumentParser(description="Generate FusionBench-Sim with randomized timing")
    ap.add_argument("--out-runs", type=Path, default=Path("data/fusionbench_sim_random_timing/runs"))
    ap.add_argument("--out-binaries", type=Path, default=Path("data/fusionbench_sim_random_timing/binaries/binaries.csv"))

    ap.add_argument("--seed", type=int, default=2024)  # Different seed from original
    ap.add_argument("--hz", type=float, default=20.0)
    ap.add_argument("--duration-s", type=float, default=180.0)

    ap.add_argument("--n-binaries", type=int, default=6)
    ap.add_argument("--runs-per-binary", type=int, default=6)
    ap.add_argument("--start-run", type=int, default=1)

    ap.add_argument("--include-benign-confounders", action="store_true")
    args = ap.parse_args()

    args.out_runs.mkdir(parents=True, exist_ok=True)
    binary_ids = [f"sim_binary_{i:04d}" for i in range(1, int(args.n_binaries) + 1)]

    run_configs = []
    run_idx = int(args.start_run)
    
    for b_i, binary_id in enumerate(binary_ids):
        for r_i in range(int(args.runs_per_binary)):
            run_id = f"run_{run_idx:06d}"

            workload_family = WORKLOAD_FAMILIES[(b_i + r_i) % len(WORKLOAD_FAMILIES)]
            trojan_family = TROJAN_FAMILIES[(r_i) % len(TROJAN_FAMILIES)]
            power_mode = "MAXN" if ((b_i + r_i) % 2 == 0) else "30W"
            ambient_c = 24.0 if ((b_i + r_i) % 3 != 0) else 35.0

            if trojan_family == "none":
                trojan_on_s = 0.0
                trojan_style = "mempressure"
            else:
                trojan_on_s = 10.0
                trojan_style = trojan_family

            device_instance = sample_device_instance(args.seed, run_id)
            cfg = SimConfig(
                seed=int(args.seed) + run_idx,
                hz=float(args.hz),
                ambient_c=float(ambient_c),
                power_mode=str(power_mode),
                input_voltage_v=19.0,
                jetson_clocks=False,
                fast=True,
                emit_unix_from_sim=True,
                start_unix_s=1700000000.0,
                telemetry_jitter_ms=0.0,
                workload_family=str(workload_family),
                workload_variant="v1",
                trojan_family=str(trojan_family),
                trojan_variant="v1",
                trojan_style=str(trojan_style),
                trojan_strength=1.0,
                trojan_period_s=40.0,
                trojan_on_s=float(trojan_on_s),
                device_instance=device_instance,
                benign_throttle_event_rate_hz=0.05 if args.include_benign_confounders else 0.0,
                perf_multiplex_event_rate_hz=0.05 if args.include_benign_confounders else 0.0,
            )
            # Store randomization seed for intervals
            rng_seed = int(args.seed) + run_idx + 10000
            run_configs.append((str(args.out_runs), run_id, binary_id, cfg, float(args.duration_s), rng_seed))
            run_idx += 1

    # Generate runs with randomized timing
    from concurrent.futures import ThreadPoolExecutor
    n_workers = 8
    
    def _generate_single(config_tuple):
        out_root, run_id, binary_id, cfg, duration_s, rng_seed = config_tuple
        # Generate with deterministic telemetry (simulator)
        run_dir = generate_run(out_root=out_root, run_id=run_id, binary_id=binary_id, cfg=cfg, duration_s=duration_s)
        # Overwrite intervals.csv with randomized version
        write_intervals_csv_randomized(run_dir, cfg, duration_s, rng_seed)
        return run_id
    
    print(f"Generating {len(run_configs)} runs with RANDOMIZED activation timing...")
    with ThreadPoolExecutor(max_workers=n_workers) as executor:
        created = list(executor.map(_generate_single, run_configs))

    _write_binaries_csv(args.out_binaries, binary_ids)
    print(f"Wrote {len(created)} runs under {args.out_runs}")
    print(f"Wrote binaries index: {args.out_binaries}")
    print("\nNOTE: Activation timing is now STOCHASTIC (random offset, exponential inter-arrival, variable duration)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
