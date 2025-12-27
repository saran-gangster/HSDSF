#!/usr/bin/env bash
# Enhanced configuration for HSDSF pipeline with 120 runs
# Use this after clearing old data: rm -rf data/fusionbench_sim/runs/*

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/..&" && pwd)"
cd "$REPO_ROOT"

# Auto-detect Python
if [ -f "$REPO_ROOT/.venv/bin/python" ]; then
    PY="$REPO_ROOT/.venv/bin/python"
elif command -v python3 &> /dev/null; then
    PY="python3"
else
    PY="python"
fi

echo "Generating 120 diverse runs (20 binaries × 6 runs)..."
"$PY" experiments/generate_fusionbench_sim.py \
    --n-binaries 20 \
    --runs-per-binary 6 \
    --duration-s 180 \
    --add-diversity \
    --include-benign-confounders

echo "Generation complete!"
echo "To run full pipeline: bash experiments/run_all.sh"
