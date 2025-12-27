#!/usr/bin/env bash
# ============================================================================
# HSDSF-FusionBench: Master Reproduction Script
# ============================================================================
# This script generates all data, trains all models, evaluates all methods,
# and produces paper-ready tables and figures.
#
# Prerequisites:
# - Python 3.10+
# - Required packages: numpy, pandas, torch, scikit-learn, matplotlib
#
# Run on Google Colab with T4 GPU for full pipeline execution.
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

# Set PYTHONPATH so Python can find project modules
export PYTHONPATH="$REPO_ROOT:${PYTHONPATH:-}"

echo "========================================"
echo "HSDSF-FusionBench Reproduction Pipeline"
echo "========================================"
echo "Repo root: $REPO_ROOT"
echo "Python: $PY"
echo ""

# ============================================================================
# Phase 1: Data Generation
# ============================================================================
echo "[Phase 1] Data Generation"
echo "-------------------------"

if [ ! -d "$DATA_DIR/runs/run_000001" ]; then
    echo "Generating benchmark runs (120 runs)..."
    "$PY" experiments/generate_fusionbench_sim.py \
        --n-binaries 12 \
        --runs-per-binary 10 \
        --duration-s 120 \
        --include-benign-confounders
else
    echo "Benchmark runs already exist, skipping generation."
fi

# Validate schema (sample only for speed)
echo "Validating telemetry schema (sample)..."
"$PY" evaluation/validate_schema.py "$DATA_DIR/runs/run_*/telemetry.csv" --max-rows 100 --sample 10

# Create splits (use random splits for better generalization)
echo "Creating split manifests..."
"$PY" experiments/make_splits.py \
    --use-random-splits \
    --train-frac 0.6 \
    --val-frac 0.2 \
    --holdout-workload cv_heavy \
    --holdout-trojan compute \
    --holdout-power-mode MAXN \
    --holdout-ambient-ge 30

echo ""

# ============================================================================
# Phase 2: Preprocessing
# ============================================================================
echo "[Phase 2] Preprocessing"
echo "-----------------------"

for split in random_split unseen_workload unseen_trojan unseen_regime; do
    if [ ! -f "$DATA_DIR/processed/$split/windows_train.npz" ]; then
        echo "Preprocessing $split (background)..."
        "$PY" dynamic/preprocess.py --split "$DATA_DIR/splits/$split.json" &
    else
        echo "Preprocessed data for $split already exists."
    fi
done
# Wait for all preprocessing to complete
wait
echo "All preprocessing complete."

echo ""

# ============================================================================
# Phase 3: Static Expert (no GPU needed)
# ============================================================================
echo "[Phase 3] Static Expert"
echo "-----------------------"

STATIC_DIR="$MODELS_DIR/static"

# Extract static features
if [ ! -f "$DATA_DIR/binaries/static_features.parquet" ] && [ ! -f "$DATA_DIR/binaries/static_features.csv" ]; then
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
    # Use parquet if available, else CSV
    if [ -f "$DATA_DIR/binaries/static_features.parquet" ]; then
        FEATURES_FILE="$DATA_DIR/binaries/static_features.parquet"
    else
        FEATURES_FILE="$DATA_DIR/binaries/static_features.csv"
    fi
    "$PY" static/train_static.py \
        --features "$FEATURES_FILE" \
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
# Phase 4: Dynamic Expert (GPU training)
# ============================================================================
echo "[Phase 4] Dynamic Expert (GPU)"
echo "------------------------------"

for split in random_split unseen_workload unseen_trojan unseen_regime; do
    DYN_DIR="$MODELS_DIR/dynamic/$split"
    
    if [ ! -f "$DYN_DIR/model_0.pt" ]; then
        echo "Training dynamic expert for $split..."
        "$PY" dynamic/train_dynamic.py \
            --processed-dir "$DATA_DIR/processed/$split" \
            --out-dir "$DYN_DIR" \
            --model tcn \
            --n-ensemble 5 \
            --epochs 30 \
            --batch-size 128
        
        echo "Calibrating dynamic expert for $split..."
        "$PY" dynamic/calibrate_dynamic.py \
            --model-dir "$DYN_DIR"
    else
        echo "Dynamic model for $split already exists."
    fi
done

echo ""

# ============================================================================
# Phase 5: Fusion Training (GPU)
# ============================================================================
echo "[Phase 5] Fusion Training (GPU)"
echo "--------------------------------"

for split in random_split unseen_workload unseen_trojan unseen_regime; do
    DYN_DIR="$MODELS_DIR/dynamic/$split"
    FUS_DIR="$MODELS_DIR/fusion/$split"
    
    if [ ! -f "$FUS_DIR/fusion_model.pt" ]; then
        if [ -f "$DYN_DIR/model_0.pt" ]; then
            echo "Training fusion gate for $split..."
            "$PY" fusion/train_fusion.py \
                --processed-dir "$DATA_DIR/processed/$split" \
                --static-dir "$STATIC_DIR" \
                --dynamic-dir "$DYN_DIR" \
                --out-dir "$FUS_DIR" \
                --epochs 30 \
                --patience 8
        else
            echo "Skipping fusion for $split (no dynamic model)."
        fi
    else
        echo "Fusion model for $split already exists."
    fi
done

echo ""

# ============================================================================
# Phase 6: Evaluation
# ============================================================================
echo "[Phase 6] Evaluation"
echo "--------------------"

for split in random_split unseen_workload unseen_trojan unseen_regime; do
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
                --out-dir "$RES_DIR" \
                --sweep-thresholds
        else
            echo "Results for $split already exist."
        fi
    else
        echo "Skipping evaluation for $split (no fusion model)."
    fi
done

echo ""

# ============================================================================
# Phase 7: Generate Figures
# ============================================================================
echo "[Phase 7] Figure Generation"
echo "---------------------------"

FIG_DIR="$PAPER_DIR/figures"
mkdir -p "$FIG_DIR"

for split in random_split unseen_workload unseen_trojan unseen_regime; do
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
echo "Pipeline Complete!"
echo "========================================"
echo ""
echo "Results:  $RESULTS_DIR/"
echo "Figures:  $FIG_DIR/"
echo ""
echo "Summary of outputs:"
for split in random_split unseen_workload unseen_trojan unseen_regime; do
    if [ -f "$RESULTS_DIR/$split/results.csv" ]; then
        echo "  ✓ $split: results available"
    else
        echo "  ✗ $split: no results"
    fi
done
echo ""
