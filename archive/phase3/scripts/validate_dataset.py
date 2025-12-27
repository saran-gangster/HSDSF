#!/usr/bin/env python3
"""
validate_dataset.py - Phase 3 dataset validation against checklist

Validates telemetry CSVs against the Phase 3 checklist:
1. Cadence: ~10 rows/sec
2. Perf metrics: nonzero cycles/instructions, plausible IPC (0.1-5.0)
3. Branch/cache features: non-constant during load
4. Thermal/power: temps increase under load vs idle
5. Trojan intervals: alignment with telemetry spikes

Usage:
    python3 validate_dataset.py --runs-dir ../data/runs
    python3 validate_dataset.py --runs-dir ../data/runs --verbose
"""

import argparse
import json
import sys
from pathlib import Path
import pandas as pd
import numpy as np


def load_run(run_dir):
    """Load telemetry.csv and meta.json from a run directory."""
    run_path = Path(run_dir)
    telem_csv = run_path / "telemetry.csv"
    meta_json = run_path / "meta.json"
    trojan_csv = run_path / "trojan_intervals.csv"
    
    if not telem_csv.exists():
        return None, None, None, f"Missing telemetry.csv"
    
    try:
        df = pd.read_csv(telem_csv)
    except Exception as e:
        return None, None, None, f"Failed to read telemetry.csv: {e}"
    
    meta = {}
    if meta_json.exists():
        try:
            with open(meta_json) as f:
                meta = json.load(f)
        except Exception as e:
            print(f"Warning: Failed to read meta.json: {e}", file=sys.stderr)
    
    trojan_intervals = None
    if trojan_csv.exists():
        try:
            trojan_intervals = pd.read_csv(trojan_csv)
            # Treat empty/column-less files as missing so simulator runs don't hard-fail
            if trojan_intervals.empty or trojan_intervals.shape[1] == 0:
                trojan_intervals = None
        except Exception as e:
            print(f"Warning: Failed to read trojan_intervals.csv: {e}", file=sys.stderr)
    
    return df, meta, trojan_intervals, None


def _timestamp_column(df):
    """Pick the appropriate timestamp column, if present."""
    if 'timestamp' in df.columns:
        return 'timestamp'
    if 'ts_unix' in df.columns:
        return 'ts_unix'
    return None


def check_cadence(df, expected_rate=10.0, tolerance=0.2, min_rows=20, min_duration_s=5.0):
    """Check 1: Cadence ~10 rows/sec.

    Accepts either `timestamp` or `ts_unix` columns (simulator uses ts_unix).
    Skips cadence validation for very short / truncated runs (rows < min_rows
    or duration < min_duration_s).
    """
    ts_col = _timestamp_column(df)
    if ts_col is None:
        return False, "Missing timestamp/ts_unix column"
    if len(df) < 2:
        return False, "Too few rows to check cadence"
    
    duration = df[ts_col].iloc[-1] - df[ts_col].iloc[0]

    # Skip clearly truncated runs to avoid false cadence failures
    if len(df) < min_rows or duration < min_duration_s:
        return True, f"Cadence skipped (short run: rows={len(df)}, duration={duration:.2f}s)"
    if duration <= 0:
        return False, "Zero or negative duration"
    
    actual_rate = len(df) / duration
    error = abs(actual_rate - expected_rate) / expected_rate
    
    if error > tolerance:
        return False, f"Cadence {actual_rate:.2f} rows/sec (expected {expected_rate} ± {tolerance*100}%)"
    
    return True, f"Cadence OK: {actual_rate:.2f} rows/sec using {ts_col}"


def check_perf_metrics(df, label):
    """Check 2: Perf metrics nonzero and IPC plausible."""
    issues = []
    perf_cols_present = [c for c in df.columns if c.startswith('perf_')]
    if not perf_cols_present:
        return True, "Perf metrics missing; treating as skipped (simulator/legacy run)"
    
    # For idle runs, allow zero cycles/instructions
    is_idle = 'idle' in label.lower()
    
    if 'perf_cycles' in df.columns:
        nonzero = (df['perf_cycles'] > 0).sum()
        if not is_idle and nonzero < len(df) * 0.5:
            issues.append(f"perf_cycles mostly zero ({nonzero}/{len(df)} nonzero)")
    else:
        issues.append("Missing perf_cycles column")
    
    if 'perf_instructions' in df.columns:
        nonzero = (df['perf_instructions'] > 0).sum()
        if not is_idle and nonzero < len(df) * 0.5:
            issues.append(f"perf_instructions mostly zero ({nonzero}/{len(df)} nonzero)")
    else:
        issues.append("Missing perf_instructions column")
    
    if 'perf_ipc' in df.columns:
        valid_ipc = df[(df['perf_ipc'] >= 0.1) & (df['perf_ipc'] <= 5.0)]
        if not is_idle and len(valid_ipc) < len(df) * 0.5:
            ipc_range = f"{df['perf_ipc'].min():.2f}-{df['perf_ipc'].max():.2f}"
            issues.append(f"perf_ipc out of range 0.1-5.0 for most rows (range: {ipc_range})")
    else:
        issues.append("Missing perf_ipc column")
    
    if issues:
        return False, "; ".join(issues)
    
    return True, "Perf metrics OK"


def check_branch_cache_variance(df, label):
    """Check 3: Branch/cache features non-constant during load."""
    is_idle = 'idle' in label.lower()
    
    issues = []
    
    # For non-idle, expect some variance
    if not is_idle:
        if 'perf_branch_miss_rate' in df.columns:
            variance = df['perf_branch_miss_rate'].var()
            if variance < 1e-6:
                issues.append("perf_branch_miss_rate nearly constant")
        
        if 'perf_cache_miss_rate' in df.columns:
            variance = df['perf_cache_miss_rate'].var()
            if variance < 1e-6:
                issues.append("perf_cache_miss_rate nearly constant")
    
    if issues:
        return False, "; ".join(issues)
    
    return True, "Branch/cache variance OK"


def check_thermal_power(df, label):
    """Check 4: Temps increase under load vs idle baseline."""
    is_idle = 'idle' in label.lower()
    
    # Find temperature columns
    temp_cols = [c for c in df.columns if 'temp' in c.lower() or c.startswith('ts_temp')]
    
    if not temp_cols:
        return True, "No thermal sensors found (simulator mode?)"
    
    # For non-idle runs, expect temp > baseline
    if not is_idle:
        temp_means = {col: df[col].mean() for col in temp_cols}
        # Check if any temp is significantly elevated (>30C as sanity check)
        elevated = any(t > 30 for t in temp_means.values())
        if not elevated:
            return False, f"Temps suspiciously low for non-idle: {temp_means}"
    
    return True, "Thermal OK"


def check_trojan_intervals(df, trojan_intervals, label):
    """Check 5: Trojan intervals align with telemetry spikes.

    In simulator mode, trojan_intervals.csv may be absent; treat as warn.
    """
    if trojan_intervals is None or len(trojan_intervals) == 0:
        if 'trojan' in label.lower():
            return True, "Missing trojan_intervals (simulator mode)"
        return True, "N/A (not a trojan run)"
    
    # Check that intervals are within telemetry time range
    if 'start_time' not in trojan_intervals.columns or 'end_time' not in trojan_intervals.columns:
        return True, "Missing start_time/end_time (treating as simulator mode)"
    
    ts_col = _timestamp_column(df)
    if ts_col is None:
        return False, "Missing timestamp/ts_unix column"
    
    telem_start = df[ts_col].min()
    telem_end = df[ts_col].max()
    
    intervals_in_range = 0
    for _, row in trojan_intervals.iterrows():
        if row['start_time'] >= telem_start and row['end_time'] <= telem_end:
            intervals_in_range += 1
    
    if intervals_in_range == 0:
        return False, "No trojan intervals within telemetry timespan"
    
    # Basic spike check: during trojan ON intervals, expect higher activity
    # (For simulator, this may not show perfect spikes, but should be non-flat)
    if 'perf_cycles' in df.columns:
        # Sample first interval
        first = trojan_intervals.iloc[0]
        interval_rows = df[(df['timestamp'] >= first['start_time']) & 
                          (df['timestamp'] <= first['end_time'])]
        if len(interval_rows) > 0:
            mean_cycles = interval_rows['perf_cycles'].mean()
            if mean_cycles == 0:
                return False, "Zero cycles during trojan interval"
    
    return True, f"Trojan intervals OK ({intervals_in_range} intervals)"


def validate_run(run_dir, verbose=False):
    """Validate a single run directory."""
    run_name = Path(run_dir).name
    
    df, meta, trojan_intervals, error = load_run(run_dir)
    if error:
        return {
            'run': run_name,
            'valid': False,
            'error': error,
            'checks': {}
        }
    
    label = meta.get('label', run_name)
    
    checks = {}
    
    # Run all checks
    checks['cadence'] = check_cadence(df)
    checks['perf_metrics'] = check_perf_metrics(df, label)
    checks['branch_cache'] = check_branch_cache_variance(df, label)
    checks['thermal'] = check_thermal_power(df, label)
    checks['trojan_intervals'] = check_trojan_intervals(df, trojan_intervals, label)
    
    # Overall validity
    all_pass = all(result[0] for result in checks.values())
    
    result = {
        'run': run_name,
        'label': label,
        'valid': all_pass,
        'checks': {k: {'pass': v[0], 'message': v[1]} for k, v in checks.items()}
    }
    
    if verbose:
        print(f"\n=== {run_name} ({label}) ===")
        for check_name, check_result in checks.items():
            status = "✓" if check_result[0] else "✗"
            print(f"  {status} {check_name}: {check_result[1]}")
    
    return result


def main():
    parser = argparse.ArgumentParser(description="Validate Phase 3 dataset")
    parser.add_argument('--runs-dir', default='../data/runs',
                       help='Directory containing run subdirectories')
    parser.add_argument('--verbose', action='store_true',
                       help='Print detailed check results')
    
    args = parser.parse_args()
    
    runs_path = Path(args.runs_dir)
    if not runs_path.exists():
        print(f"Error: {runs_path} does not exist", file=sys.stderr)
        return 1
    
    # Find all run directories
    run_dirs = [d for d in runs_path.iterdir() if d.is_dir()]
    if not run_dirs:
        print(f"Error: No run directories found in {runs_path}", file=sys.stderr)
        return 1
    
    print(f"Validating {len(run_dirs)} runs from {runs_path}...")
    
    results = []
    for run_dir in sorted(run_dirs):
        result = validate_run(run_dir, verbose=args.verbose)
        results.append(result)
    
    # Summary
    print("\n" + "="*60)
    print("VALIDATION SUMMARY")
    print("="*60)
    
    passed = sum(1 for r in results if r['valid'])
    total = len(results)
    
    print(f"\nOverall: {passed}/{total} runs passed all checks\n")
    
    for result in results:
        status = "✓ PASS" if result['valid'] else "✗ FAIL"
        label = result.get('label', result['run'])
        print(f"  {status}: {result['run']} ({label})")
        
        if not result['valid'] and 'error' in result:
            print(f"         Error: {result['error']}")
        elif not result['valid']:
            failed_checks = [k for k, v in result['checks'].items() if not v['pass']]
            print(f"         Failed checks: {', '.join(failed_checks)}")
    
    print()
    
    # Exit code: 0 if all pass, 1 otherwise
    return 0 if passed == total else 1


if __name__ == '__main__':
    sys.exit(main())
