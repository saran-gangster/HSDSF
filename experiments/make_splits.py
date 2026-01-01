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

# Force unbuffered output
def log(msg: str) -> None:
    print(msg, flush=True)

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)


@dataclass(frozen=True)
class RunMeta:
    run_id: str
    binary_id: str
    workload_family: str
    trojan_family: str
    power_mode: str
    ambient_c: float


def _load_run_meta(run_dir: Path) -> RunMeta:
    meta = json.loads((run_dir / "meta.json").read_text(encoding="utf-8"))
    cfg = meta.get("sim_config", {})
    return RunMeta(
        run_id=str(meta.get("run_id") or run_dir.name),
        binary_id=str(meta.get("binary_id") or "unknown"),
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
    """Deterministic train/val split using seeded random shuffle."""
    import random
    ids = sorted(ids)  # Ensure deterministic ordering
    if not ids:
        return [], []
    
    rng = random.Random(seed)
    shuffled = ids[:]
    rng.shuffle(shuffled)
    
    n_val = max(1, int(round(len(ids) * float(val_frac))))
    val = sorted(shuffled[:n_val])
    train = sorted(shuffled[n_val:])
    return train, val


def main() -> int:
    ap = argparse.ArgumentParser(description="Create run-level split manifests")
    ap.add_argument("--runs-dir", type=Path, default=Path("data/fusionbench_sim/runs"))
    ap.add_argument("--out-dir", type=Path, default=Path("data/fusionbench_sim/splits"))
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--train-frac", type=float, default=0.6)
    ap.add_argument("--val-frac", type=float, default=0.2)
    ap.add_argument("--use-random-splits", action="store_true",
                    help="Also create a random (non-holdout) split for baseline comparison")
    ap.add_argument("--holdout-workload", type=str, default="cv_heavy")
    ap.add_argument("--holdout-trojan", type=str, default="compute")
    ap.add_argument("--holdout-power-mode", type=str, default="MAXN")
    ap.add_argument("--holdout-ambient-ge", type=float, default=30.0)
    args = ap.parse_args()

    log(f"Scanning {args.runs_dir}...")
    run_dirs = sorted([p for p in args.runs_dir.iterdir() if p.is_dir() and (p / "meta.json").exists()])
    log(f"Found {len(run_dirs)} run directories")
    if not run_dirs:
        raise SystemExit(f"No runs found under {args.runs_dir}")

    # Simple sequential load (ThreadPool can deadlock on some systems)
    log("Loading metadata...")
    metas = [_load_run_meta(p) for p in run_dirs]
    log(f"Loaded {len(metas)} metadata files")
    all_ids = [m.run_id for m in metas]

    # Random split (stratified by label distribution)
    if args.use_random_splits:
        log("Creating random split...")
        import random
        rng = random.Random(args.seed)
        shuffled = all_ids[:]
        rng.shuffle(shuffled)
        n = len(shuffled)
        n_train = int(n * args.train_frac)
        n_val = int(n * args.val_frac)
        train = sorted(shuffled[:n_train])
        val = sorted(shuffled[n_train:n_train + n_val])
        test = sorted(shuffled[n_train + n_val:])
        log(f"  Writing random_split.json ({len(train)}/{len(val)}/{len(test)})...")
        _write_manifest(
            args.out_dir / "random_split.json",
            train=train,
            val=val,
            test=test,
            notes=f"random_split: {len(train)}/{len(val)}/{len(test)} train/val/test",
        )
        log("  Done.")

    log("Creating unseen_workload split...")

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

    # Binary-disjoint split (realistic supply-chain scenario)
    log("Creating binary_disjoint split...")
    binary_to_runs: Dict[str, List[str]] = {}
    for m in metas:
        if m.binary_id not in binary_to_runs:
            binary_to_runs[m.binary_id] = []
        binary_to_runs[m.binary_id].append(m.run_id)
    
    all_binaries = sorted(binary_to_runs.keys())
    n_binaries = len(all_binaries)
    
    # Split binaries: 60% train, 20% val, 20% test
    import random
    rng = random.Random(args.seed + 42)
    shuffled_binaries = all_binaries[:]
    rng.shuffle(shuffled_binaries)
    
    n_train_b = max(1, int(n_binaries * args.train_frac))
    n_val_b = max(1, int(n_binaries * args.val_frac))
    
    train_binaries = set(shuffled_binaries[:n_train_b])
    val_binaries = set(shuffled_binaries[n_train_b:n_train_b + n_val_b])
    test_binaries = set(shuffled_binaries[n_train_b + n_val_b:])
    
    train = sorted([r for b in train_binaries for r in binary_to_runs[b]])
    val = sorted([r for b in val_binaries for r in binary_to_runs[b]])
    test = sorted([r for b in test_binaries for r in binary_to_runs[b]])
    
    _write_manifest(
        args.out_dir / "binary_disjoint.json",
        train=train,
        val=val,
        test=test,
        notes=f"binary_disjoint: {len(train_binaries)}/{len(val_binaries)}/{len(test_binaries)} binaries, {len(train)}/{len(val)}/{len(test)} runs",
    )
    log(f"  Binary split: {len(train_binaries)} train, {len(val_binaries)} val, {len(test_binaries)} test binaries")

    log(f"Wrote splits under: {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
