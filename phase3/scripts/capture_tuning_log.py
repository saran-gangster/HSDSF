#!/usr/bin/env python3
"""
Capture calibration/tuning logs from the Jetson Xavier simulator via HTTP API.

It produces a single text file with tegrastats-like lines plus:
- computed component power (sum of rails excluding SYS5V)
- a few phase headers + the config that was applied

Assumes the simulator is already running (simd) on 127.0.0.1.

Usage:
  python3 capture_tuning_log.py --out tuning_capture.txt
  python3 capture_tuning_log.py --port 45215 --out tuning_capture.txt
"""

from __future__ import annotations

import argparse
import json
import time
import urllib.request
from typing import Any, Dict, Optional


def http_get_json(port: int, path: str, timeout_s: float = 2.0) -> Dict[str, Any]:
    url = f"http://127.0.0.1:{port}{path}"
    with urllib.request.urlopen(url, timeout=timeout_s) as r:
        return json.loads(r.read().decode("utf-8"))


def http_post_json(port: int, path: str, payload: Dict[str, Any], timeout_s: float = 2.0) -> Dict[str, Any]:
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}{path}",
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout_s) as r:
        return json.loads(r.read().decode("utf-8"))


def clamp(x: float, lo: float, hi: float) -> float:
    return lo if x < lo else hi if x > hi else x


def fmt_tegrastats_line(st: Dict[str, Any]) -> str:
    used = float(st["ram_used_mb"])
    total = int(st["ram_total_mb"])
    free = max(0.0, total - used)
    lfb_blocks = int(clamp(free / 256.0, 100, 900))
    lfb_mb = 4

    cpu_parts = []
    for u, f in zip(st["cpu_util"], st["cpu_freq_mhz"]):
        cpu_parts.append(f"{u:.0f}%@{int(f)}")
    cpu_str = ", ".join(cpu_parts)

    line = (
        f"RAM {used:.0f}/{total}MB (lfb {lfb_blocks}x{lfb_mb}MB) "
        f"CPU [{cpu_str}] "
        f"EMC {st['emc_util']:.0f}%@{int(st['emc_freq_mhz'])} "
        f"GR3D {st['gpu_util']:.0f}%@{int(st['gpu_freq_mhz'])} "
        f"CV {st['cv_util']:.0f}% "
        f"TTP@{st['temp_ttp_c']:.1f}C CPU@{st['temp_cpu_c']:.1f}C GPU@{st['temp_gpu_c']:.1f}C AUX@{st['temp_aux_c']:.1f}C "
        f"FAN_EST@{st['fan_est_temp_c']:.1f}C FAN {int(st['fan_rpm'])}RPM PWM {int(st['fan_pwm'])} HS {int(st['hysteresis_state'])} "
        f"VDD_SYS5V {st['p_sys5v_mw']:.0f}mW VDD_CPU {st['p_cpu_mw']:.0f}mW VDD_GPU {st['p_gpu_mw']:.0f}mW "
        f"VDD_SOC {st['p_soc_mw']:.0f}mW VDD_CV {st['p_cv_mw']:.0f}mW VDD_VDDRQ {st['p_vddrq_mw']:.0f}mW "
        f"MODE {st['mode']} TROJAN={'1' if st.get('trojan_active') else '0'}"
    )
    if st.get("sim_error"):
        line += f" ERROR='{st['sim_error']}'"
    return line


def comp_power_mw(st: Dict[str, Any]) -> float:
    # component power as seen by rails (INA-averaged), excluding SYS5V
    return float(st["p_cpu_mw"]) + float(st["p_gpu_mw"]) + float(st["p_soc_mw"]) + float(st["p_cv_mw"]) + float(st["p_vddrq_mw"])


def write_header(f, title: str) -> None:
    f.write("\n")
    f.write("#" * 90 + "\n")
    f.write(f"# {title}\n")
    f.write("#" * 90 + "\n")


def apply_config(f, port: int, cfg: Dict[str, Any], note: Optional[str] = None) -> None:
    write_header(f, "APPLY CONFIG")
    if note:
        f.write(f"# note: {note}\n")
    f.write("# payload:\n")
    f.write(json.dumps(cfg, indent=2, sort_keys=True) + "\n")
    try:
        resp = http_post_json(port, "/v1/config", cfg)
        f.write("# response:\n")
        f.write(json.dumps(resp, indent=2, sort_keys=True) + "\n")
    except Exception as e:
        f.write(f"# ERROR posting config: {e}\n")
        raise


def set_mode(f, port: int, mode: str) -> None:
    write_header(f, f"SET MODE = {mode}")
    resp = http_post_json(port, "/v1/control", {"mode": mode})
    f.write(json.dumps(resp, indent=2, sort_keys=True) + "\n")


def collect_lines(
    f,
    port: int,
    phase_name: str,
    n_lines: int,
    interval_s: float,
) -> None:
    write_header(f, f"COLLECT {n_lines} LINES: {phase_name}")
    next_t = time.time()

    fan_on_seen = 0
    cap_seen = 0

    for i in range(n_lines):
        # pace sampling
        now = time.time()
        if now < next_t:
            time.sleep(next_t - now)
        next_t += interval_s

        st = http_get_json(port, "/v1/telemetry")

        # If sim died, still write the last line and stop collecting this phase.
        line = fmt_tegrastats_line(st)
        pcomp = comp_power_mw(st)
        psys = float(st["p_sys5v_mw"])
        # Efficiency-ish ratio visible in telemetry
        ratio = (psys / pcomp) if pcomp > 1e-9 else 0.0

        # very rough indicator: if we intentionally set a low cap, seeing pcomp hovering near it suggests cap engagement
        # (not perfect, but useful)
        # We'll mark "CAPLIKELY" if pcomp is in the 85–105% band of 10W or 12W during our cap phase.
        cap_likely = (8500.0 <= pcomp <= 10500.0) or (10200.0 <= pcomp <= 12600.0)

        if int(st.get("fan_pwm", 0)) > 0:
            fan_on_seen += 1
        if cap_likely:
            cap_seen += 1

        suffix = (
            f" | P_COMP {pcomp:.0f}mW SYS5V/COMP {ratio:.3f}"
            + (" CAPLIKELY" if cap_likely else "")
        )

        f.write(line + suffix + "\n")

        if st.get("sim_error"):
            f.write("# Simulator reported sim_error; stopping capture.\n")
            break

    f.write(f"# phase summary: fan_on_lines={fan_on_seen}, cap_likely_lines={cap_seen}\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=45215)
    ap.add_argument("--out", type=str, required=True)
    ap.add_argument("--interval-ms", type=int, default=500)

    # How many lines per phase
    ap.add_argument("--baseline-lines", type=int, default=60)
    ap.add_argument("--fan-lines", type=int, default=120)
    ap.add_argument("--cap-lines", type=int, default=140)
    ap.add_argument("--cooldown-lines", type=int, default=160)

    args = ap.parse_args()
    interval_s = max(0.05, args.interval_ms / 1000.0)

    # Sanity check: simulator reachable
    st0 = http_get_json(args.port, "/v1/telemetry")

    with open(args.out, "w", encoding="utf-8") as f:
        write_header(f, "SIMULATOR TELEMETRY CAPTURE FOR TUNING")
        f.write(f"# port={args.port}\n")
        f.write(f"# interval_s={interval_s}\n")
        f.write("# initial /v1/telemetry snapshot:\n")
        f.write(json.dumps(st0, indent=2, sort_keys=True) + "\n")

        # Phase 0: Baseline (normal, default-ish environment)
        apply_config(
            f,
            args.port,
            {
                "hz": 20.0,
                "ambient_c": 24.0,
                "power_mode": "30W",
                "power_cap_w": 30.0,
                # Work around DVFS index rounding by making rate very fast
                "cpu_step_rate_per_s": 80.0,
                "gpu_step_rate_per_s": 80.0,
                "emc_step_rate_per_s": 80.0,
            },
            note="Baseline. DVFS rates bumped so frequencies can actually move.",
        )
        set_mode(f, args.port, "normal")
        collect_lines(f, args.port, "BASELINE normal (fan should be off)", args.baseline_lines, interval_s)

        # Phase 1: Fan step exercise (force FAN on by raising ambient and load a bit)
        # Goal: cross fan-est 50/63/72/81 thresholds and see HS/PWM change.
        apply_config(
            f,
            args.port,
            {
                "ambient_c": 58.0,
                "power_mode": "30W",
                "power_cap_w": 30.0,
                "frame_gpu_busy_ms": 18.0,
                "frame_cpu_busy_ms": 7.0,
                "bg_spike_rate_hz": 0.9,
                "bg_spike_ms": 90.0,
                "bg_spike_cpu_pct": 30.0,
                # Keep DVFS responsive
                "cpu_step_rate_per_s": 120.0,
                "gpu_step_rate_per_s": 120.0,
                "emc_step_rate_per_s": 120.0,
                # Make fan impact visible (you can change these later; we just need transitions)
                "fan_cooling_min_gain": 0.55,
                "fan_cooling_max_gain": 1.05,
            },
            note="Fan exercise: raise ambient and workload so fan steps trigger.",
        )
        set_mode(f, args.port, "normal")
        collect_lines(f, args.port, "FAN EXERCISE normal (expect HS/PWM transitions)", args.fan_lines, interval_s)

        # Phase 2: Power-cap shaping exercise (force cap to engage)
        # We set a low cap on purpose so exponent differences show up.
        apply_config(
            f,
            args.port,
            {
                "ambient_c": 45.0,
                "power_mode": "MAXN",
                "power_cap_w": 10.0,  # deliberately low to force cap behavior
                "frame_gpu_busy_ms": 28.0,
                "frame_cpu_busy_ms": 12.0,
                "bg_spike_rate_hz": 1.2,
                "bg_spike_ms": 120.0,
                "bg_spike_cpu_pct": 40.0,
                # Make trojan always ON for a clean segment
                "trojan_period_s": 10.0,
                "trojan_on_s": 10.0,
                "trojan_style": "mempressure",
                "trojan_strength": 2.5,
                # DVFS responsive
                "cpu_step_rate_per_s": 140.0,
                "gpu_step_rate_per_s": 140.0,
                "emc_step_rate_per_s": 140.0,
            },
            note="Cap exercise: low power_cap_w + heavy load + trojan always-on so power-cap exponents matter.",
        )
        set_mode(f, args.port, "trojan")
        collect_lines(f, args.port, "CAP EXERCISE trojan (expect CAPLIKELY tags, rail reshaping)", args.cap_lines, interval_s)

        # Phase 3: AUX memory / cooldown (hot -> idle and drop ambient)
        apply_config(
            f,
            args.port,
            {
                "ambient_c": 24.0,
                "power_mode": "30W",
                "power_cap_w": 30.0,
                # keep DVFS responsive even in cooldown
                "cpu_step_rate_per_s": 80.0,
                "gpu_step_rate_per_s": 80.0,
                "emc_step_rate_per_s": 80.0,
            },
            note="Cooldown: drop ambient and go idle. AUX should lag/cool slowly if AUX memory is active.",
        )
        set_mode(f, args.port, "idle")
        collect_lines(f, args.port, "COOLDOWN idle (look for AUX slow tail vs CPU/GPU)", args.cooldown_lines, interval_s)

        write_header(f, "DONE")
    print(f"Wrote capture to: {args.out}")


if __name__ == "__main__":
    main()