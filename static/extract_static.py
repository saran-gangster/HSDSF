#!/usr/bin/env python3
"""Extract static features from binaries listed in binaries.csv.

For simulator binaries (no real ELF files), uses deterministic pseudo-features
keyed by binary_id hash. Outputs a parquet file with all feature columns.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import pandas as pd

from static.features.deterministic_elf import extract_features_batch, FEATURE_COLUMNS


def main() -> int:
    ap = argparse.ArgumentParser(description="Extract static features")
    ap.add_argument("--binaries-csv", type=Path, required=True,
                    help="Path to binaries.csv with binary_id column")
    ap.add_argument("--out", type=Path, required=True,
                    help="Output parquet file path")
    args = ap.parse_args()

    # Read binary IDs
    binary_ids = []
    with args.binaries_csv.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            bid = row.get("binary_id", "").strip()
            if bid:
                binary_ids.append(bid)

    if not binary_ids:
        print("No binary IDs found in CSV")
        return 1

    print(f"Extracting features for {len(binary_ids)} binaries...")
    
    # Extract features using deterministic generator
    features = extract_features_batch(binary_ids)
    
    # Convert to DataFrame and save
    df = pd.DataFrame(features)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    
    # Try parquet, fall back to CSV
    try:
        df.to_parquet(args.out, index=False)
        print(f"Wrote {len(df)} rows × {len(df.columns)} columns to {args.out}")
    except ImportError:
        csv_path = args.out.with_suffix(".csv")
        df.to_csv(csv_path, index=False)
        print(f"Wrote {len(df)} rows × {len(df.columns)} columns to {csv_path} (parquet unavailable)")

    print(f"Feature columns: {FEATURE_COLUMNS}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
