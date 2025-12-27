#!/usr/bin/env bash
# ============================================================================
# HSDSF-FusionBench: Master Reproduction Script
# ============================================================================
# This script generates all data, trains all models, evaluates all methods,
# and produces paper-ready tables and figures.
#
# Prerequisites:
# - Python 3.10+ with venv at .venv/
# - Required packages: numpy, pandas, torch, scikit-learn, matplotlib
#
# For GPU training (dynamic/fusion), use Colab with the provided notebooks.
# This script runs CPU-capable steps locally.
# ============================================================================

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

# Auto-detect Python: prefer .venv, fall back to system python3
if [ -f "$REPO_ROOT/.venv/bin/python" ]; then
    PY="$REPO_ROOT/.venv/bin/python"
elif command -v python3 &> /dev/null; then
    PY="python3"
else
    PY="python"
fi
PY="${PY_OVERRIDE:-$PY}"  # Allow override via PY_OVERRIDE env var

DATA_DIR="data/fusionbench_sim"
MODELS_DIR="models"
RESULTS_DIR="results"
PAPER_DIR="paper"

echo "========================================"
echo "HSDSF-FusionBench Reproduction Pipeline"
echo "========================================"
echo "Repo root: $REPO_ROOT"
echo "Python: $PY"
echo ""

# ============================================================================
# Phase 1: Data Generation (runs locally)
# ============================================================================
echo "[Phase 1] Data Generation"
echo "-------------------------"

if [ ! -d "$DATA_DIR/runs/run_000001" ]; then
    echo "Generating benchmark runs..."
    "$PY" experiments/generate_fusionbench_sim.py \
        --n-binaries 6 \
        --runs-per-binary 6 \
        --duration-s 180 \
        --include-benign-confounders
else
    echo "Benchmark runs already exist, skipping generation."
fi

# Validate schema
echo "Validating telemetry schema..."
"$PY" evaluation/validate_schema.py "$DATA_DIR/runs/run_*/telemetry.csv" --max-rows 500

# Create splits
echo "Creating split manifests..."
"$PY" experiments/make_splits.py \
    --holdout-workload cv_heavy \
    --holdout-trojan compute \
    --holdout-power-mode MAXN \
    --holdout-ambient-ge 30

echo ""

# ============================================================================
# Phase 2: Preprocessing (runs locally)
# ============================================================================
echo "[Phase 2] Preprocessing"
echo "-----------------------"

for split in unseen_workload unseen_trojan unseen_regime; do
    if [ ! -f "$DATA_DIR/processed/$split/windows_train.npz" ]; then
        echo "Preprocessing $split..."
        "$PY" dynamic/preprocess.py --split "$DATA_DIR/splits/$split.json"
    else
        echo "Preprocessed data for $split already exists."
    fi
done

echo ""

# ============================================================================
# Phase 3: Static Expert (runs locally - no GPU needed)
# ============================================================================
echo "[Phase 3] Static Expert"
echo "-----------------------"

STATIC_DIR="$MODELS_DIR/static"

# Extract static features
if [ ! -f "$DATA_DIR/binaries/static_features.parquet" ]; then
    echo "Extracting static features..."
    "$PY" static/extract_static.py \
        --binaries-csv "$DATA_DIR/binaries/binaries.csv" \
        --out "$DATA_DIR/binaries/static_features.parquet"
else
    echo "Static features already extracted."
fi

# Train static ensemble
if [ ! -f "$STATIC_DIR/ensemble.pkl" ]; then
    echo "Training static expert ensemble..."
    "$PY" static/train_static.py \
        --features "$DATA_DIR/binaries/static_features.parquet" \
        --runs-dir "$DATA_DIR/runs" \
        --out-dir "$STATIC_DIR" \
        --n-ensemble 5
else
    echo "Static ensemble already trained."
fi

# Calibrate static expert
if [ ! -f "$STATIC_DIR/static_predictions_calibrated.parquet" ]; then
    echo "Calibrating static expert..."
    "$PY" static/calibrate_static.py \
        --predictions "$STATIC_DIR/static_predictions.parquet" \
        --out-dir "$STATIC_DIR"
else
    echo "Static expert already calibrated."
fi

echo ""

# ============================================================================
# Phase 4: Dynamic Expert (requires GPU - run on Colab)
# ============================================================================
echo "[Phase 4] Dynamic Expert"
echo "------------------------"
echo "NOTE: Dynamic training requires GPU. Run on Colab:"
echo ""
echo "For each split (unseen_workload, unseen_trojan, unseen_regime):"
echo "  python dynamic/train_dynamic.py \\"
echo "      --processed-dir data/fusionbench_sim/processed/{split} \\"
echo "      --out-dir models/dynamic/{split} \\"
echo "      --model tcn --n-ensemble 5 --epochs 20"
echo ""
echo "  python dynamic/calibrate_dynamic.py \\"
echo "      --model-dir models/dynamic/{split}"
echo ""

# Check if dynamic models exist
for split in unseen_workload unseen_trojan unseen_regime; do
    DYN_DIR="$MODELS_DIR/dynamic/$split"
    if [ -f "$DYN_DIR/model_0.pt" ]; then
        echo "Dynamic model for $split: FOUND"
    else
        echo "Dynamic model for $split: NOT FOUND (run on Colab)"
    fi
done

echo ""

# ============================================================================
# Phase 5: Fusion (requires GPU - run on Colab after dynamic)
# ============================================================================
echo "[Phase 5] Fusion Training"
echo "-------------------------"
echo "NOTE: Fusion training requires dynamic models. Run on Colab after Phase 4:"
echo ""
echo "For each split:"
echo "  python fusion/train_fusion.py \\"
echo "      --processed-dir data/fusionbench_sim/processed/{split} \\"
echo "      --static-dir models/static \\"
echo "      --dynamic-dir models/dynamic/{split} \\"
echo "      --out-dir models/fusion/{split}"
echo ""

# Check if fusion models exist
for split in unseen_workload unseen_trojan unseen_regime; do
    FUS_DIR="$MODELS_DIR/fusion/$split"
    if [ -f "$FUS_DIR/fusion_model.pt" ]; then
        echo "Fusion model for $split: FOUND"
    else
        echo "Fusion model for $split: NOT FOUND (run on Colab)"
    fi
done

echo ""

# ============================================================================
# Phase 6: Evaluation (runs after training, CPU OK)
# ============================================================================
echo "[Phase 6] Evaluation"
echo "--------------------"

for split in unseen_workload unseen_trojan unseen_regime; do
    DYN_DIR="$MODELS_DIR/dynamic/$split"
    FUS_DIR="$MODELS_DIR/fusion/$split"
    RES_DIR="$RESULTS_DIR/$split"
    
    if [ -f "$FUS_DIR/fusion_model.pt" ]; then
        if [ ! -f "$RES_DIR/results.csv" ]; then
            echo "Evaluating $split..."
            "$PY" fusion/eval_fusion.py \
                --processed-dir "$DATA_DIR/processed/$split" \
                --static-dir "$STATIC_DIR" \
                --dynamic-dir "$DYN_DIR" \
                --fusion-dir "$FUS_DIR" \
                --out-dir "$RES_DIR"
        else
            echo "Results for $split already exist."
        fi
    else
        echo "Skipping evaluation for $split (no fusion model)."
    fi
done

echo ""

# ============================================================================
# Phase 7: Generate Figures (runs after evaluation)
# ============================================================================
echo "[Phase 7] Figure Generation"
echo "---------------------------"

FIG_DIR="$PAPER_DIR/figures"
mkdir -p "$FIG_DIR"

for split in unseen_workload unseen_trojan unseen_regime; do
    RES_DIR="$RESULTS_DIR/$split"
    FUS_DIR="$MODELS_DIR/fusion/$split"
    
    if [ -f "$RES_DIR/results.csv" ]; then
        echo "Generating figures for $split..."
        "$PY" evaluation/plots.py \
            --results-dir "$RES_DIR" \
            --fusion-dir "$FUS_DIR" \
            --out-dir "$FIG_DIR" \
            --split-name "$split"
    else
        echo "Skipping figures for $split (no results)."
    fi
done

echo ""

# ============================================================================
# Summary
# ============================================================================
echo "========================================"
echo "Pipeline Complete"
echo "========================================"
echo ""
echo "Results:  $RESULTS_DIR/"
echo "Figures:  $FIG_DIR/"
echo ""
echo "To complete GPU training steps, follow the Colab instructions above."
echo ""
