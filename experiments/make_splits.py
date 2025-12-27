#!/usr/bin/env python3
"""Create deterministic run-level splits for HSDSF-FusionBench-Sim.

Produces split manifests under data/fusionbench_sim/splits/*.json.

Split types (per plan):
- unseen_workload: hold out a workload_family
- unseen_trojan: hold out a trojan_family (including "none")
- unseen_regime: hold out power_mode or ambient bucket
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)


@dataclass(frozen=True)
class RunMeta:
    run_id: str
    workload_family: str
    trojan_family: str
    power_mode: str
    ambient_c: float


def _load_run_meta(run_dir: Path) -> RunMeta:
    meta = json.loads((run_dir / "meta.json").read_text(encoding="utf-8"))
    cfg = meta.get("sim_config", {})
    return RunMeta(
        run_id=str(meta.get("run_id") or run_dir.name),
        workload_family=str(meta.get("workload_family") or cfg.get("workload_family") or "unknown"),
        trojan_family=str(meta.get("trojan_family") or cfg.get("trojan_family") or "unknown"),
        power_mode=str(meta.get("power_mode") or cfg.get("power_mode") or "unknown"),
        ambient_c=float(meta.get("ambient_c") or cfg.get("ambient_c") or 0.0),
    )


def _write_manifest(path: Path, train: List[str], val: List[str], test: List[str], notes: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "train_runs": train,
        "val_runs": val,
        "test_runs": test,
        "notes": notes,
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _stable_split(ids: Sequence[str], *, val_frac: float, seed: int) -> Tuple[List[str], List[str]]:
    # deterministic split without randomness from runtime hash: sort, then take stride-based selection
    ids = sorted(ids)
    if not ids:
        return [], []
    n_val = max(1, int(round(len(ids) * float(val_frac)))) if len(ids) >= 3 else max(0, len(ids) // 3)
    # simple LCG-style indexing to spread samples
    step = (seed % (len(ids) - 1) + 1) if len(ids) > 1 else 1
    val_idx = set()
    i = seed % len(ids)
    while len(val_idx) < n_val and len(val_idx) < len(ids):
        val_idx.add(i)
        i = (i + step) % len(ids)
    train = [rid for j, rid in enumerate(ids) if j not in val_idx]
    val = [rid for j, rid in enumerate(ids) if j in val_idx]
    return train, val


def main() -> int:
    ap = argparse.ArgumentParser(description="Create run-level split manifests")
    ap.add_argument("--runs-dir", type=Path, default=Path("data/fusionbench_sim/runs"))
    ap.add_argument("--out-dir", type=Path, default=Path("data/fusionbench_sim/splits"))
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--val-frac", type=float, default=0.2)
    ap.add_argument("--holdout-workload", type=str, default="cv_heavy")
    ap.add_argument("--holdout-trojan", type=str, default="compute")
    ap.add_argument("--holdout-power-mode", type=str, default="MAXN")
    ap.add_argument("--holdout-ambient-ge", type=float, default=30.0)
    args = ap.parse_args()

    run_dirs = sorted([p for p in args.runs_dir.iterdir() if p.is_dir() and (p / "meta.json").exists()])
    if not run_dirs:
        raise SystemExit(f"No runs found under {args.runs_dir}")

    metas = [_load_run_meta(p) for p in run_dirs]

    # unseen workload
    test = [m.run_id for m in metas if m.workload_family == args.holdout_workload]
    rest = [m.run_id for m in metas if m.workload_family != args.holdout_workload]
    train, val = _stable_split(rest, val_frac=args.val_frac, seed=args.seed)
    _write_manifest(
        args.out_dir / "unseen_workload.json",
        train=train,
        val=val,
        test=sorted(test),
        notes=f"unseen_workload={args.holdout_workload}",
    )

    # unseen trojan
    test = [m.run_id for m in metas if m.trojan_family == args.holdout_trojan]
    rest = [m.run_id for m in metas if m.trojan_family != args.holdout_trojan]
    train, val = _stable_split(rest, val_frac=args.val_frac, seed=args.seed + 17)
    _write_manifest(
        args.out_dir / "unseen_trojan.json",
        train=train,
        val=val,
        test=sorted(test),
        notes=f"unseen_trojan={args.holdout_trojan}",
    )

    # unseen regime
    # primary: hold out a power mode; secondary: hold out hot ambient bucket if requested
    test = [m.run_id for m in metas if m.power_mode == args.holdout_power_mode or m.ambient_c >= args.holdout_ambient_ge]
    rest = [m.run_id for m in metas if m.run_id not in set(test)]
    train, val = _stable_split(rest, val_frac=args.val_frac, seed=args.seed + 31)
    _write_manifest(
        args.out_dir / "unseen_regime.json",
        train=train,
        val=val,
        test=sorted(test),
        notes=f"unseen_regime=power_mode:{args.holdout_power_mode}|ambient_ge:{args.holdout_ambient_ge}",
    )

    print(f"Wrote splits under: {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
