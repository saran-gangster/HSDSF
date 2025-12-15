import argparse
import time

import numpy as np


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
    args = ap.parse_args()
    cpu_inference_like(args.duration, args.batch, args.dim)


if __name__ == "__main__":
    main()
