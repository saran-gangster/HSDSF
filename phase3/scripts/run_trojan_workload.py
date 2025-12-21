import argparse
import json
import socket
import time
import urllib.request
from pathlib import Path

import numpy as np


def set_simulator_mode(port: int, mode: str):
    """Set simulator workload mode via HTTP API"""
    data = json.dumps({"mode": mode}).encode('utf-8')
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}/v1/control",
        data=data,
        headers={'Content-Type': 'application/json'},
        method='POST',
    )
    with urllib.request.urlopen(req, timeout=2.0) as r:
        return json.loads(r.read().decode('utf-8'))


def set_simulator_config(port: int, payload: dict):
    data = json.dumps(payload).encode('utf-8')
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}/v1/config",
        data=data,
        headers={'Content-Type': 'application/json'},
        method='POST',
    )
    with urllib.request.urlopen(req, timeout=2.0) as r:
        return json.loads(r.read().decode('utf-8'))


def baseline_chunk(dt: float = 0.1):
    x = np.random.randn(1024).astype(np.float32)
    _ = np.sin(x) + 0.01 * x
    time.sleep(dt)


def trojan_compute(active_s: float):
    a = np.random.randn(256, 256).astype(np.float32)
    b = np.random.randn(256, 256).astype(np.float32)
    t0 = time.monotonic()
    while time.monotonic() - t0 < active_s:
        _ = a @ b


def trojan_memory(active_s: float, n_mb: int = 256):
    n = (n_mb * 1024 * 1024) // 4
    src = np.random.randint(0, 255, size=(n,), dtype=np.uint8)
    dst = np.empty_like(src)
    t0 = time.monotonic()
    while time.monotonic() - t0 < active_s:
        np.copyto(dst, src)


def trojan_io(active_s: float, pps: int = 50000):
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    payload = b"x" * 256
    addr = ("127.0.0.1", 9999)
    t0 = time.monotonic()
    interval = 1.0 / max(1, pps)
    next_t = time.monotonic()
    while time.monotonic() - t0 < active_s:
        s.sendto(payload, addr)
        next_t += interval
        while time.monotonic() < next_t:
            pass


def run_trojan_pattern(duration_s: float, period_s: float, active_s: float, variant: str, intervals_out: Path, with_baseline: bool, mem_mb: int, pps: int, simulator_mode: bool = False, simulator_port: int = 45215):
    intervals_out.parent.mkdir(parents=True, exist_ok=True)
    
    if simulator_mode:
        # Use simulator's built-in trojan pattern
        sim_style = "mempressure" if variant == "memory" else ("io" if variant == "io" else "compute")
        set_simulator_config(simulator_port, {
            "trojan_style": sim_style,
            "trojan_period_s": period_s,
            "trojan_on_s": active_s,
        })
        set_simulator_mode(simulator_port, "trojan")
        t0 = time.monotonic()
        
        # Poll simulator to record actual intervals
        intervals = []
        prev_active = False
        interval_start = None
        
        while time.monotonic() - t0 < duration_s:
            try:
                url = f"http://127.0.0.1:{simulator_port}/v1/telemetry"
                with urllib.request.urlopen(url, timeout=1.0) as resp:
                    data = json.loads(resp.read().decode("utf-8"))
                active = data.get("trojan_active", False)
                
                if active and not prev_active:
                    interval_start = time.monotonic() - t0
                elif not active and prev_active and interval_start is not None:
                    intervals.append((interval_start, time.monotonic() - t0))
                    interval_start = None
                prev_active = active
            except Exception:
                pass
            time.sleep(0.1)
        
        with intervals_out.open("w") as f:
            for a, b in intervals:
                f.write(f"{a:.3f},{b:.3f}\n")
        return
    
    # Original implementation for real hardware
    t0 = time.monotonic()
    next_on = t0 + period_s
    intervals = []

    while time.monotonic() - t0 < duration_s:
        now = time.monotonic()
        if now >= next_on:
            on_start = now
            if variant == "compute":
                trojan_compute(active_s)
            elif variant == "memory":
                trojan_memory(active_s, n_mb=mem_mb)
            elif variant == "io":
                trojan_io(active_s, pps=pps)
            else:
                raise ValueError("unknown variant")
            on_end = time.monotonic()
            intervals.append((on_start - t0, on_end - t0))
            next_on += period_s
        else:
            if with_baseline:
                baseline_chunk(0.1)
            else:
                time.sleep(0.05)

    with intervals_out.open("w") as f:
        for a, b in intervals:
            f.write(f"{a:.3f},{b:.3f}\n")


def main():
    ap = argparse.ArgumentParser(description="Trojan workload generator")
    ap.add_argument("--duration", type=float, default=120.0)
    ap.add_argument("--period", type=float, default=10.0)
    ap.add_argument("--active", type=float, default=2.0)
    ap.add_argument("--variant", choices=["compute", "memory", "io"], required=True)
    ap.add_argument("--intervals-out", required=True)
    ap.add_argument("--with-baseline", action="store_true")
    ap.add_argument("--memory-mb", type=int, default=256)
    ap.add_argument("--pps", type=int, default=50000, help="Packets per second for IO variant")
    ap.add_argument("--simulator-mode", action="store_true", help="Drive simulator via HTTP API")
    ap.add_argument("--simulator-port", type=int, default=45215)
    args = ap.parse_args()

    run_trojan_pattern(
        args.duration,
        args.period,
        args.active,
        args.variant,
        Path(args.intervals_out),
        args.with_baseline,
        mem_mb=args.memory_mb,
        pps=args.pps,
        simulator_mode=args.simulator_mode,
        simulator_port=args.simulator_port,
    )


if __name__ == "__main__":
    main()
