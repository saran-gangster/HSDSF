#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from evaluation.events import load_intervals_csv, window_label, window_label_soft


IDENTITY_COLS = {
    "run_id",
    "binary_id",
    "workload_family",
    "workload_variant",
    "trojan_family",
    "trojan_variant",
    "mode",
    "power_mode",
}


def _load_manifest(path: Path) -> Dict[str, List[str]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return {
        "train_runs": [str(x) for x in data.get("train_runs", [])],
        "val_runs": [str(x) for x in data.get("val_runs", [])],
        "test_runs": [str(x) for x in data.get("test_runs", [])],
        "notes": str(data.get("notes", "")),
    }


def _load_run_df(run_dir: Path) -> pd.DataFrame:
    df = pd.read_csv(run_dir / "telemetry.csv")
    df["run_id"] = str(run_dir.name)
    return df


def _select_numeric_features(df: pd.DataFrame) -> List[str]:
    # keep numeric columns excluding time/labels and identity
    drop = set(IDENTITY_COLS) | {"t_unix_s", "t_sim_s", "trojan_active", "jetson_clocks", "input_voltage_v", "ambient_c"}
    candidates = [c for c in df.columns if c not in drop]
    numeric = [c for c in candidates if pd.api.types.is_numeric_dtype(df[c])]
    # keep masks (they're numeric in CSV) and telemetry numeric
    return sorted(numeric)


def _split_mask_features(features: Sequence[str]) -> tuple[list[str], list[str]]:
    mask = [f for f in features if str(f).startswith("mask_")]
    other = [f for f in features if f not in set(mask)]
    return sorted(mask), sorted(other)


def _add_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    eps = 1e-9
    # IPC and miss rates
    if {"delta_instructions", "delta_cycles"}.issubset(out.columns):
        out["ipc"] = out["delta_instructions"] / np.maximum(out["delta_cycles"], eps)
    if {"delta_cache_misses", "delta_instructions"}.issubset(out.columns):
        out["miss_rate"] = out["delta_cache_misses"] / np.maximum(out["delta_instructions"], eps)
    # power ratios
    if {"p_gpu_mw", "p_sys5v_mw"}.issubset(out.columns):
        out["p_gpu_ratio"] = out["p_gpu_mw"] / np.maximum(out["p_sys5v_mw"], eps)
    if {"p_cpu_mw", "p_sys5v_mw"}.issubset(out.columns):
        out["p_cpu_ratio"] = out["p_cpu_mw"] / np.maximum(out["p_sys5v_mw"], eps)
    # throttling-ish indicators (frequency ratios)
    if {"cpu_freq_mhz", "gpu_freq_mhz"}.issubset(out.columns):
        out["cpu_gpu_freq_ratio"] = out["cpu_freq_mhz"] / np.maximum(out["gpu_freq_mhz"], 1.0)
    return out


def normalize_per_run(
    df: pd.DataFrame,
    features: List[str],
    warmup_steps: int = 200,
) -> pd.DataFrame:
    """Normalize each feature by run warmup baseline.
    
    ML Expert recommendation for unseen_regime fix:
    "Normalize each channel by a run baseline (warm-up mean/var) or rolling mean/var.
    This often gives the single biggest jump for power/thermal telemetry because
    it removes absolute-level shifts."
    
    Args:
        df: DataFrame with run_id column
        features: List of feature columns to normalize
        warmup_steps: Number of initial steps to use as baseline
    
    Returns:
        DataFrame with normalized features (original columns replaced)
    """
    out = df.copy()
    
    for run_id in out["run_id"].unique():
        run_mask = out["run_id"] == run_id
        run_df = out.loc[run_mask, features]
        
        # Compute baseline from warmup period
        warmup = run_df.iloc[:warmup_steps]
        baseline_mean = warmup.mean()
        baseline_std = warmup.std() + 1e-6
        
        # Normalize
        normalized = (run_df - baseline_mean) / baseline_std
        out.loc[run_mask, features] = normalized.values
    
    return out


def _resample(df: pd.DataFrame, *, hz: float) -> pd.DataFrame:
    """Resample to a fixed cadence using t_sim_s as the index."""
    if hz <= 0:
        return df
    step = 1.0 / float(hz)
    df = df.sort_values("t_sim_s").reset_index(drop=True)
    t0 = float(df["t_sim_s"].iloc[0])
    t1 = float(df["t_sim_s"].iloc[-1])
    grid = np.arange(t0, t1 + 1e-9, step)

    # numeric columns get linear interpolation; categorical are forward-filled
    num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    cat_cols = [c for c in df.columns if c not in num_cols]

    out = pd.DataFrame({"t_sim_s": grid})
    for c in num_cols:
        if c == "t_sim_s":
            continue
        out[c] = np.interp(grid, df["t_sim_s"].to_numpy(), df[c].to_numpy())
    for c in cat_cols:
        # align by nearest previous sample
        idx = np.searchsorted(df["t_sim_s"].to_numpy(), grid, side="right") - 1
        idx = np.clip(idx, 0, len(df) - 1)
        out[c] = df[c].to_numpy()[idx]
    return out


def _windowize_run(
    df: pd.DataFrame,
    *,
    intervals_path: Path,
    features: Sequence[str],
    window_len_s: float,
    stride_s: float,
    overlap_threshold: float,
    use_soft_labels: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Extract windows from a run with soft or hard labels.
    
    Args:
        use_soft_labels: If True, return continuous labels [0,1] based on
            overlap fraction. This improves training on boundary windows.
    """
    intervals = load_intervals_csv(intervals_path)
    t = df["t_sim_s"].to_numpy(dtype=np.float64)
    x = df[list(features)].to_numpy(dtype=np.float32)

    dt = float(np.median(np.diff(t))) if len(t) >= 3 else 0.0
    if dt <= 0:
        raise ValueError("Cannot infer cadence from t_sim_s")

    win_n = int(round(window_len_s / dt))
    stride_n = max(1, int(round(stride_s / dt)))
    if win_n < 2:
        raise ValueError("window_len_s too small for cadence")

    xs: List[np.ndarray] = []
    ys: List[float] = []  # Changed to float for soft labels
    t_centers: List[float] = []
    for start in range(0, len(t) - win_n + 1, stride_n):
        end = start + win_n
        t_start = float(t[start])
        t_end = float(t[end - 1])
        
        if use_soft_labels:
            # Soft label: continuous overlap fraction [0, 1]
            yc = window_label_soft(t_start=t_start, t_end=t_end, intervals=intervals)
        else:
            # Hard label: binary based on threshold
            yc = float(window_label(t_start=t_start, t_end=t_end, intervals=intervals, overlap_threshold=overlap_threshold))
        
        xs.append(x[start:end])
        ys.append(yc)
        t_centers.append(0.5 * (t_start + t_end))
    
    return np.stack(xs, axis=0), np.asarray(ys, dtype=np.float32), np.asarray(t_centers, dtype=np.float64)


def _save_npz(path: Path, **arrays: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays)


def main() -> int:
    ap = argparse.ArgumentParser(description="Preprocess FusionBench-Sim telemetry into fixed windows")
    ap.add_argument("--runs-dir", type=Path, default=Path("data/fusionbench_sim/runs"))
    ap.add_argument("--split", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, default=Path("data/fusionbench_sim/processed"))

    ap.add_argument("--resample-hz", type=float, default=20.0)
    ap.add_argument("--window-len-s", type=float, default=10.0)
    ap.add_argument("--stride-s", type=float, default=1.0)
    ap.add_argument("--overlap-threshold", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=1337)
    args = ap.parse_args()

    manifest = _load_manifest(args.split)
    split_name = args.split.stem
    split_out = args.out_dir / split_name
    split_out.mkdir(parents=True, exist_ok=True)

    # load runs and build features list from train
    train_frames: List[pd.DataFrame] = []
    for rid in manifest["train_runs"]:
        df = _load_run_df(args.runs_dir / rid)
        df = _resample(df, hz=args.resample_hz)
        df = _add_derived_features(df)
        train_frames.append(df)
    if not train_frames:
        raise SystemExit("Split has no train runs")

    train_concat = pd.concat(train_frames, ignore_index=True)
    features = _select_numeric_features(train_concat)
    # ensure masks are included (even if interpreted as floats)
    features = sorted(set(features) | {c for c in train_concat.columns if str(c).startswith("mask_")})

    mask_features, scaled_features = _split_mask_features(features)

    # build scaler from train windows (flatten time)
    scaler = StandardScaler()
    # fit on raw points (not windows) to keep it stable
    if scaled_features:
        scaler.fit(
            train_concat[list(scaled_features)]
            .replace([np.inf, -np.inf], np.nan)
            .fillna(0.0)
            .to_numpy(dtype=np.float64)
        )

    def process_split(kind: str, run_ids: Sequence[str]) -> None:
        from concurrent.futures import ThreadPoolExecutor
        
        def process_single_run(rid: str):
            """Process a single run and return results."""
            run_dir = args.runs_dir / rid
            df = _load_run_df(run_dir)
            df = _resample(df, hz=args.resample_hz)
            df = _add_derived_features(df)
            if scaled_features:
                df[list(scaled_features)] = scaler.transform(
                    df[list(scaled_features)].replace([np.inf, -np.inf], np.nan).fillna(0.0).to_numpy(dtype=np.float64)
                ).astype(np.float32)
            # masks stay in {0,1} as float32
            for mf in mask_features:
                if mf in df.columns:
                    df[mf] = df[mf].astype(np.float32)

            x, y, t_centers = _windowize_run(
                df,
                intervals_path=run_dir / "intervals.csv",
                features=features,
                window_len_s=args.window_len_s,
                stride_s=args.stride_s,
                overlap_threshold=args.overlap_threshold,
            )
            binary_id = str(df["binary_id"].iloc[0])
            return x, y, t_centers, rid, binary_id
        
        # Process runs in parallel
        with ThreadPoolExecutor(max_workers=8) as executor:
            results = list(executor.map(process_single_run, run_ids))
        
        # Collect results
        Xs = [r[0] for r in results]
        Ys = [r[1] for r in results]
        Ts = [r[2] for r in results]
        R = []
        B = []
        for x, y, t_centers, rid, binary_id in results:
            R.extend([rid] * len(y))
            B.extend([binary_id] * len(y))

        X = np.concatenate(Xs, axis=0) if Xs else np.zeros((0, 0, 0), dtype=np.float32)
        Y = np.concatenate(Ys, axis=0) if Ys else np.zeros((0,), dtype=np.int64)
        T = np.concatenate(Ts, axis=0) if Ts else np.zeros((0,), dtype=np.float64)
        _save_npz(
            split_out / f"windows_{kind}.npz",
            X=X,
            y=Y,
            t_center=T,
            run_id=np.asarray(R, dtype=object),
            binary_id=np.asarray(B, dtype=object),
            features=np.asarray(list(features), dtype=object),
            window_len_s=float(args.window_len_s),
            stride_s=float(args.stride_s),
            resample_hz=float(args.resample_hz),
            notes=str(manifest.get("notes", "")),
        )

    process_split("train", manifest["train_runs"])
    process_split("val", manifest["val_runs"])
    process_split("test", manifest["test_runs"])

    # save scaler
    scaler_path = split_out / "scaler.json"
    scaler_payload = {
        "mean": scaler.mean_.tolist() if scaled_features else [],
        "scale": scaler.scale_.tolist() if scaled_features else [],
        "features": list(features),
        "scaled_features": list(scaled_features),
        "mask_features": list(mask_features),
    }
    scaler_path.write_text(json.dumps(scaler_payload, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote: {split_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
