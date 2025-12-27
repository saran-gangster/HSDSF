#!/usr/bin/env python3
"""Generate deterministic simulator runs in-process.

Writes HSDSF-FusionBench-Sim layout:
    data/fusionbench_sim/runs/<run_id>/{telemetry.csv, meta.json, intervals.csv}

This avoids HTTP polling and supports fast deterministic generation.
"""

import argparse
import os
import sys

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from experiments.genlib import generate_run, sample_device_instance
from simulator.jetson_sim import SimConfig


def main() -> None:
    p = argparse.ArgumentParser(description="Generate deterministic simulator runs (HSDSF-FusionBench-Sim)")
    p.add_argument("--out-root", type=str, default="data/fusionbench_sim/runs")
    p.add_argument("--run-id", type=str, default="run_000001")
    p.add_argument("--binary-id", type=str, default="sim_binary_0001")
    p.add_argument("--duration-s", type=float, default=120.0)

    p.add_argument("--seed", type=int, default=1337)
    p.add_argument("--hz", type=float, default=20.0)
    p.add_argument("--start-unix-s", type=float, default=1700000000.0)

    p.add_argument("--power-mode", type=str, choices=["30W", "MAXN"], default="30W")
    p.add_argument("--ambient-c", type=float, default=24.0)
    p.add_argument("--input-voltage-v", type=float, default=19.0)
    p.add_argument("--jetson-clocks", action="store_true")

    p.add_argument("--workload-family", type=str, default="inference_periodic")
    p.add_argument("--workload-variant", type=str, default="v1")

    p.add_argument("--trojan-family", type=str, default="sim_trojan")
    p.add_argument("--trojan-variant", type=str, default="v1")
    p.add_argument("--trojan-style", type=str, choices=["mempressure", "compute", "cv", "io"], default="mempressure")
    p.add_argument("--trojan-strength", type=float, default=1.0)
    p.add_argument("--trojan-period-s", type=float, default=40.0)
    p.add_argument("--trojan-on-s", type=float, default=10.0)
    p.add_argument("--device-instance", choices=["auto", "none"], default="auto")

    args = p.parse_args()

    device_instance = None
    if args.device_instance == "auto":
        device_instance = sample_device_instance(args.seed, args.run_id)

    cfg = SimConfig(
        seed=args.seed,
        hz=args.hz,
        ambient_c=args.ambient_c,
        power_mode=args.power_mode,
        input_voltage_v=args.input_voltage_v,
        jetson_clocks=bool(args.jetson_clocks),
        fast=True,
        emit_unix_from_sim=True,
        start_unix_s=args.start_unix_s,
        telemetry_jitter_ms=0.0,
        workload_family=args.workload_family,
        workload_variant=args.workload_variant,
        trojan_family=args.trojan_family,
        trojan_variant=args.trojan_variant,
        trojan_style=args.trojan_style,
        trojan_strength=args.trojan_strength,
        trojan_period_s=args.trojan_period_s,
        trojan_on_s=args.trojan_on_s,
        device_instance=device_instance,
    )

    run_dir = generate_run(out_root=args.out_root, run_id=args.run_id, binary_id=args.binary_id, cfg=cfg, duration_s=args.duration_s)

    print(f"Wrote run: {run_dir}")


if __name__ == "__main__":
    main()
