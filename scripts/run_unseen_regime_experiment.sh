#!/bin/bash
# Run unseen_regime experiment with per-run normalization
# This script preprocesses, trains, and evaluates the dynamic model
# with per-run normalization to mitigate domain shift

set -e

echo "========================================="
echo "Unseen Regime Per-Run Normalization Experiment"
echo "========================================="

# Step 1: Preprocess with per-run normalization
echo ""
echo "Step 1: Preprocessing unseen_regime with per-run normalization..."
echo "========================================="
python -m dynamic.preprocess \
    --split data/fusionbench_sim/splits/unseen_regime.json \
    --per-run-norm \
    --warmup-steps 200

echo "Output: data/fusionbench_sim/processed/unseen_regime_perrun/"

# Step 2: Train dynamic expert on normalized data
echo ""
echo "Step 2: Training dynamic expert on per-run normalized data..."
echo "========================================="
python -m dynamic.train_dynamic \
    --processed-dir data/fusionbench_sim/processed/unseen_regime_perrun \
    --out-dir models/dynamic_unseen_regime_perrun \
    --epochs 30

# Step 3: Calibrate dynamic expert
echo ""
echo "Step 3: Calibrating dynamic expert..."
echo "========================================="
python -m dynamic.calibrate_dynamic \
    --model-dir models/dynamic_unseen_regime_perrun

# Step 4: Evaluate fusion
echo ""
echo "Step 4: Evaluating fusion methods..."
echo "========================================="
python -m fusion.eval_fusion \
    --processed-dir data/fusionbench_sim/processed/unseen_regime_perrun \
    --static-dir models/static \
    --dynamic-dir models/dynamic_unseen_regime_perrun \
    --fusion-dir models/fusion_unseen_regime_perrun \
    --out-dir paper/analysis/unseen_regime_perrun \
    --runs-dir data/fusionbench_sim/runs \
    --sweep-thresholds

echo ""
echo "========================================="
echo "Experiment Complete!"
echo "========================================="
echo "Results saved to: paper/analysis/unseen_regime_perrun/"
