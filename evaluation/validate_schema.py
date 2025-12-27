#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import glob
import os
import sys
from pathlib import Path

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from evaluation.schema import validate_header, validate_rows


def main() -> int:
    ap = argparse.ArgumentParser(description="Validate HSDSF canonical telemetry CSV schema")
    ap.add_argument("path", type=str, help="telemetry.csv path or glob")
    ap.add_argument("--max-rows", type=int, default=100)  # Reduced for speed
    ap.add_argument("--sample", type=int, default=0, help="Only validate N random files (0=all)")
    args = ap.parse_args()

    matches = sorted(glob.glob(args.path))
    if not matches:
        print(f"No files matched: {args.path}", file=sys.stderr)
        return 2

    # Sample files if requested
    if args.sample > 0 and len(matches) > args.sample:
        import random
        random.seed(1337)
        matches = random.sample(matches, args.sample)
        print(f"Validating {len(matches)} sampled files...")

    def validate_single(p: str) -> tuple:
        path = Path(p)
        with path.open(newline="", encoding="utf-8") as f:
            r = csv.DictReader(f)
            if not r.fieldnames:
                return (p, ["missing header"], False)
            issues = []
            issues.extend(validate_header(r.fieldnames))
            issues.extend(validate_rows(r, max_rows=args.max_rows))
            return (p, issues, len(issues) == 0)

    # Parallel validation
    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=16) as executor:
        results = list(executor.map(validate_single, matches))

    ok = True
    for p, issues, is_ok in results:
        if not is_ok:
            ok = False
            print(f"{p}: {len(issues)} issue(s)", file=sys.stderr)
            for iss in issues[:5]:  # Limit output
                if hasattr(iss, 'kind'):
                    loc = ""
                    if iss.row_index is not None:
                        loc += f" row={iss.row_index}"
                    if iss.column is not None:
                        loc += f" col={iss.column}"
                    print(f"  - [{iss.kind}]{loc}: {iss.message}", file=sys.stderr)
                else:
                    print(f"  - {iss}", file=sys.stderr)
        else:
            print(f"{p}: OK")

    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
