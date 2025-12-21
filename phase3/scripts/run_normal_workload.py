import argparse
import json
import time
import urllib.request

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


def cpu_inference_like(duration_s: float, batch: int = 64, dim: int = 512):
    t0 = time.monotonic()
    x = np.random.randn(batch, dim).astype(np.float32)
    w = np.random.randn(dim, dim).astype(np.float32)
    while time.monotonic() - t0 < duration_s:
        y = x @ w
        y = np.tanh(y)
        x = y


def main():
    ap = argparse.ArgumentParser(description="Baseline normal workload")
    ap.add_argument("--duration", type=float, default=120.0)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--dim", type=int, default=512)
    ap.add_argument("--simulator-mode", action="store_true", help="Drive simulator via HTTP API")
    ap.add_argument("--simulator-port", type=int, default=45215)
    args = ap.parse_args()
    
    if args.simulator_mode:
        # Set simulator to normal mode and sleep
        set_simulator_mode(args.simulator_port, "normal")
        time.sleep(args.duration)
    else:
        cpu_inference_like(args.duration, args.batch, args.dim)


if __name__ == "__main__":
    main()
