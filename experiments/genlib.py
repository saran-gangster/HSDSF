from __future__ import annotations

import csv
import hashlib
import json
import os
import random
from dataclasses import asdict
from typing import Any, Dict, List, Optional

from simulator.jetson_sim import SensorRealism, SimConfig, XavierTDGModel


PERF_KEYS = {
    "cycles": "delta_cycles",
    "instructions": "delta_instructions",
    "cache-misses": "delta_cache_misses",
    "context-switches": "delta_context_switches",
    "page-faults": "delta_page_faults",
    "LLC-load-misses": "delta_llc_load_misses",
    "branch-misses": "delta_branch_misses",
}


def _safe_mkdir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def sample_device_instance(seed: int, run_id: str) -> Dict[str, float]:
    token = f"{run_id}:{seed}"
    h = hashlib.sha256(token.encode("utf-8")).hexdigest()
    rng = random.Random(int(h[:16], 16))
    return {
        "theta_scale": rng.uniform(0.9, 1.18),
        "tau_scale": rng.uniform(0.85, 1.20),
        "fan_gain": rng.uniform(0.85, 1.15),
        "bias_cpu_c": 0.8 + rng.uniform(-0.6, 0.6),
        "bias_gpu_c": -0.5 + rng.uniform(-0.6, 0.4),
        "bias_aux_c": 1.2 + rng.uniform(-0.5, 0.7),
        "bias_ttp_c": rng.uniform(-0.4, 0.4),
        "leak_scale_cpu": rng.uniform(0.9, 1.12),
        "leak_scale_gpu": rng.uniform(0.9, 1.12),
        "leak_scale_soc": rng.uniform(0.92, 1.10),
        "leak_scale_cv": rng.uniform(0.9, 1.15),
        "leak_scale_vddrq": rng.uniform(0.9, 1.15),
    }


def write_intervals_csv(run_dir: str, cfg: SimConfig, duration_s: float) -> None:
    path = os.path.join(run_dir, "intervals.csv")
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "t_start_sim_s",
                "t_end_sim_s",
                "trojan_variant",
                "trojan_family",
                "strength",
                "period_s",
                "on_s",
            ]
        )

        if duration_s <= 0:
            return

        period = max(1e-6, float(cfg.trojan_period_s))
        on_s = max(0.0, min(float(cfg.trojan_on_s), period))

        t = 0.0
        while t < duration_s:
            t_start = t
            t_end = min(duration_s, t + on_s)
            if on_s > 0.0:
                w.writerow(
                    [
                        f"{t_start:.6f}",
                        f"{t_end:.6f}",
                        cfg.trojan_variant,
                        cfg.trojan_family,
                        f"{float(cfg.trojan_strength):.6f}",
                        f"{period:.6f}",
                        f"{on_s:.6f}",
                    ]
                )
            t += period


def generate_run(
    *,
    out_root: str,
    run_id: str,
    binary_id: str,
    cfg: SimConfig,
    duration_s: float,
) -> str:
    run_dir = os.path.join(out_root, run_id)
    _safe_mkdir(run_dir)

    meta = {
        "run_id": run_id,
        "binary_id": binary_id,
        "workload_family": cfg.workload_family,
        "workload_variant": cfg.workload_variant,
        "trojan_family": cfg.trojan_family,
        "trojan_variant": cfg.trojan_variant,
        "mode": "trojan" if cfg.trojan_on_s > 0 else "normal",
        "seed": cfg.seed,
        "simulator": {"impl": "simulator/jetson_sim.py"},
        "sim_config": asdict(cfg),
        "device_instance": dict(cfg.device_instance or {}),
    }
    with open(os.path.join(run_dir, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, sort_keys=True)

    write_intervals_csv(run_dir, cfg, duration_s)

    model = XavierTDGModel(cfg)
    telemetry_path = os.path.join(run_dir, "telemetry.csv")

    header: List[str] = [
        # time
        "t_unix_s",
        "t_sim_s",
        # identity
        "run_id",
        "binary_id",
        "workload_family",
        "workload_variant",
        "trojan_family",
        "trojan_variant",
        "trojan_active",
        "mode",
        # meta/regime
        "power_mode",
        "ambient_c",
        "input_voltage_v",
        "jetson_clocks",
        # util
        "cpu_util_avg",
        "cpu_util_max",
        "gpu_util",
        "emc_util",
        "cv_util",
        # freq
        "cpu_freq_mhz",
        "gpu_freq_mhz",
        "emc_freq_mhz",
        # temps
        "temp_cpu_c",
        "temp_gpu_c",
        "temp_aux_c",
        "temp_ttp_c",
        "temp_board_c",
        "fan_est_temp_c",
        # fan
        "fan_pwm",
        "fan_rpm",
        "hysteresis_state",
        # power
        "p_sys5v_mw",
        "p_cpu_mw",
        "p_gpu_mw",
        "p_soc_mw",
        "p_cv_mw",
        "p_vddrq_mw",
        # perf deltas
        *list(PERF_KEYS.values()),
        # masks
        "mask_temp_cpu",
        "mask_temp_gpu",
        "mask_temp_aux",
        "mask_temp_ttp",
        "mask_temp_board",
        "mask_fan_est_temp",
        "mask_power_cpu",
        "mask_power_gpu",
        "mask_power_soc",
        "mask_power_cv",
        "mask_power_vddrq",
        "mask_power_sys5v",
        "mask_power_in",
        # perf masks
        *[f"mask_perf_{k.replace('-', '_').lower()}" for k in PERF_KEYS.keys()],
    ]

    dt = 1.0 / max(1e-6, float(cfg.hz))
    steps = int(round(float(duration_s) / dt))

    realism = SensorRealism(cfg, rng_seed=int(cfg.seed) + 5150)
    prev_totals: Dict[str, float] = {}

    with open(telemetry_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=header)
        w.writeheader()

        for _ in range(steps):
            t_unix_s = float(cfg.start_unix_s) + float(model.t_sim_s) + dt
            st = model.step_dt(dt=dt, t_unix_s=t_unix_s)
            snap = realism.apply(asdict(st), t_sim_s=float(st.t_sim_s))

            cpu_util = snap["cpu_util"]
            cpu_avg = sum(cpu_util) / len(cpu_util)
            cpu_max = max(cpu_util)

            row: Dict[str, Any] = {
                "t_unix_s": f"{float(snap['t_unix_s']):.6f}",
                "t_sim_s": f"{float(snap['t_sim_s']):.6f}",
                "run_id": run_id,
                "binary_id": binary_id,
                "workload_family": cfg.workload_family,
                "workload_variant": cfg.workload_variant,
                "trojan_family": cfg.trojan_family,
                "trojan_variant": cfg.trojan_variant,
                "trojan_active": "1" if bool(snap["trojan_active"]) else "0",
                "mode": str(snap["mode"]),
                "power_mode": str(cfg.power_mode),
                "ambient_c": f"{float(cfg.ambient_c):.6f}",
                "input_voltage_v": f"{float(cfg.input_voltage_v):.6f}",
                "jetson_clocks": "1" if bool(cfg.jetson_clocks) else "0",
                "cpu_util_avg": f"{float(cpu_avg):.6f}",
                "cpu_util_max": f"{float(cpu_max):.6f}",
                "gpu_util": f"{float(snap['gpu_util']):.6f}",
                "emc_util": f"{float(snap['emc_util']):.6f}",
                "cv_util": f"{float(snap['cv_util']):.6f}",
                "cpu_freq_mhz": str(int(snap["cpu_freq_mhz"][0])),
                "gpu_freq_mhz": str(int(snap["gpu_freq_mhz"])),
                "emc_freq_mhz": str(int(snap["emc_freq_mhz"])),
                "temp_cpu_c": f"{float(snap['temp_cpu_c']):.6f}",
                "temp_gpu_c": f"{float(snap['temp_gpu_c']):.6f}",
                "temp_aux_c": f"{float(snap['temp_aux_c']):.6f}",
                "temp_ttp_c": f"{float(snap['temp_ttp_c']):.6f}",
                "temp_board_c": f"{float(snap['temp_board_c']):.6f}",
                "fan_est_temp_c": f"{float(snap['fan_est_temp_c']):.6f}",
                "fan_pwm": str(int(snap["fan_pwm"])),
                "fan_rpm": str(int(snap["fan_rpm"])),
                "hysteresis_state": str(int(snap["hysteresis_state"])),
                "p_sys5v_mw": f"{float(snap['p_sys5v_mw']):.6f}",
                "p_cpu_mw": f"{float(snap['p_cpu_mw']):.6f}",
                "p_gpu_mw": f"{float(snap['p_gpu_mw']):.6f}",
                "p_soc_mw": f"{float(snap['p_soc_mw']):.6f}",
                "p_cv_mw": f"{float(snap['p_cv_mw']):.6f}",
                "p_vddrq_mw": f"{float(snap['p_vddrq_mw']):.6f}",
                "mask_temp_cpu": str(int(snap.get("mask_temp_cpu", 1))),
                "mask_temp_gpu": str(int(snap.get("mask_temp_gpu", 1))),
                "mask_temp_aux": str(int(snap.get("mask_temp_aux", 1))),
                "mask_temp_ttp": str(int(snap.get("mask_temp_ttp", 1))),
                "mask_temp_board": str(int(snap.get("mask_temp_board", 1))),
                "mask_fan_est_temp": str(int(snap.get("mask_fan_est_temp", 1))),
                "mask_power_cpu": str(int(snap.get("mask_power_cpu", 1))),
                "mask_power_gpu": str(int(snap.get("mask_power_gpu", 1))),
                "mask_power_soc": str(int(snap.get("mask_power_soc", 1))),
                "mask_power_cv": str(int(snap.get("mask_power_cv", 1))),
                "mask_power_vddrq": str(int(snap.get("mask_power_vddrq", 1))),
                "mask_power_sys5v": str(int(snap.get("mask_power_sys5v", 1))),
                "mask_power_in": str(int(snap.get("mask_power_in", 1))),
            }

            totals = snap["counters_total"]
            for total_key, out_key in PERF_KEYS.items():
                cur = float(totals.get(total_key, 0.0))
                prev = float(prev_totals.get(total_key, cur))
                delta = max(0.0, cur - prev) if prev_totals else 0.0
                row[out_key] = f"{delta:.6f}"
                prev_totals[total_key] = cur

            perf_masks = snap.get("perf_masks", {})
            for total_key in PERF_KEYS.keys():
                col = f"mask_perf_{total_key.replace('-', '_').lower()}"
                row[col] = str(int(perf_masks.get(total_key, 1)))

            w.writerow(row)

    return run_dir
