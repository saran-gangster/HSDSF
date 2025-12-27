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
    ap.add_argument("--max-rows", type=int, default=2000)
    args = ap.parse_args()

    matches = sorted(glob.glob(args.path))
    if not matches:
        print(f"No files matched: {args.path}", file=sys.stderr)
        return 2

    ok = True
    for p in matches:
        path = Path(p)
        with path.open(newline="", encoding="utf-8") as f:
            r = csv.DictReader(f)
            if not r.fieldnames:
                print(f"{p}: missing header", file=sys.stderr)
                ok = False
                continue
            issues = []
            issues.extend(validate_header(r.fieldnames))
            issues.extend(validate_rows(r, max_rows=args.max_rows))
            if issues:
                ok = False
                print(f"{p}: {len(issues)} issue(s)", file=sys.stderr)
                for iss in issues[:50]:
                    loc = ""
                    if iss.row_index is not None:
                        loc += f" row={iss.row_index}"
                    if iss.column is not None:
                        loc += f" col={iss.column}"
                    print(f"  - [{iss.kind}]{loc}: {iss.message}", file=sys.stderr)
            else:
                print(f"{p}: OK")

    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
