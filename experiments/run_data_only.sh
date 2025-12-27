#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY=${PY:-"$REPO_ROOT/.venv/bin/python"}

# 1) Generate a small benchmark locally (no training)
"$PY" experiments/generate_fusionbench_sim.py \
  --n-binaries 6 \
  --runs-per-binary 6 \
  --duration-s 180 \
  --include-benign-confounders

# 2) Validate schema (quick)
"$PY" evaluation/validate_schema.py "data/fusionbench_sim/runs/run_*/telemetry.csv" --max-rows 500

# 3) Create split manifests
"$PY" experiments/make_splits.py \
  --holdout-workload cv_heavy \
  --holdout-trojan compute \
  --holdout-power-mode MAXN \
  --holdout-ambient-ge 30

# 4) Preprocess (windowing + normalization) per split
"$PY" dynamic/preprocess.py --split data/fusionbench_sim/splits/unseen_workload.json
"$PY" dynamic/preprocess.py --split data/fusionbench_sim/splits/unseen_trojan.json
"$PY" dynamic/preprocess.py --split data/fusionbench_sim/splits/unseen_regime.json

echo "Data generation complete. Processed NPZ files are under data/fusionbench_sim/processed/."
