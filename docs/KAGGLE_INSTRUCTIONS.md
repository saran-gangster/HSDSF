# HSDSF Cold-Start Experiment — Kaggle GPU Instructions

## Step 1: Create Kaggle Notebook

1. Go to https://www.kaggle.com/code
2. Click "New Notebook"
3. Enable GPU: Settings → Accelerator → **GPU T4 x2** (or P100)

---

## Step 2: Upload Your Project

**Option A: From GitHub (Recommended)**
```python
!git clone https://github.com/YOUR_USERNAME/HSDSF.git
%cd HSDSF
```

**Option B: Upload ZIP**
1. Zip your HSDSF folder locally
2. Upload to Kaggle as Dataset
3. Then:
```python
!unzip /kaggle/input/hsdsf-project/hsdsf.zip -d /kaggle/working/
%cd /kaggle/working/HSDSF
```

---

## Step 3: Install Dependencies

```python
!pip install torch numpy pandas scikit-learn matplotlib tqdm -q
```

---

## Step 4: Generate Data (if not already uploaded)

```python
import os
os.environ['PYTHONPATH'] = '/kaggle/working/HSDSF'

# Generate 80 runs
!python experiments/generate_fusionbench_sim.py \
    --n-binaries 10 \
    --runs-per-binary 8 \
    --duration-s 120 \
    --include-benign-confounders

# Create splits
!python experiments/make_splits.py \
    --use-random-splits \
    --train-frac 0.6 \
    --val-frac 0.2

# Create cold-start split
!python experiments/create_coldstart_split.py \
    --base-split data/fusionbench_sim/splits/random_split.json \
    --runs-dir data/fusionbench_sim/runs \
    --output data/fusionbench_sim/splits/coldstart.json \
    --contamination-fraction 0.5
```

---

## Step 5: Preprocess Data

```python
# Preprocess for random_split (without per-run norm)
!python dynamic/preprocess.py \
    --split data/fusionbench_sim/splits/random_split.json

# Preprocess with per-run normalization
!python dynamic/preprocess.py \
    --split data/fusionbench_sim/splits/random_split.json \
    --per-run-norm \
    --warmup-steps 200
```

---

## Step 6: Train Dynamic Model

```python
# Train on standard split (GPU should be fast!)
!python dynamic/train_dynamic.py \
    --processed-dir data/fusionbench_sim/processed/random_split \
    --out-dir models/dynamic_random \
    --model tcn \
    --n-ensemble 1 \
    --epochs 30 \
    --batch-size 128 \
    --lr 5e-4

# Calibrate
!python dynamic/calibrate_dynamic.py \
    --model-dir models/dynamic_random
```

---

## Step 7: Train Static Model

```python
# Extract static features
!python static/extract_static.py \
    --binaries-csv data/fusionbench_sim/binaries/binaries.csv \
    --out data/fusionbench_sim/binaries/static_features.parquet

# Train static ensemble
!python static/train_static.py \
    --features data/fusionbench_sim/binaries/static_features.parquet \
    --runs-dir data/fusionbench_sim/runs \
    --out-dir models/static \
    --n-ensemble 5

# Calibrate static
!python static/calibrate_static.py \
    --predictions models/static/static_predictions.parquet \
    --out-dir models/static
```

---

## Step 8: Evaluate Fusion (The Key Comparison!)

```python
# Standard evaluation (without per-run norm)
!python fusion/eval_fusion.py \
    --processed-dir data/fusionbench_sim/processed/random_split \
    --static-dir models/static \
    --dynamic-dir models/dynamic_random \
    --fusion-dir models/fusion_random \
    --out-dir results/random_split \
    --runs-dir data/fusionbench_sim/runs \
    --sweep-thresholds

# Evaluation WITH per-run normalization
!python fusion/eval_fusion.py \
    --processed-dir data/fusionbench_sim/processed/random_split_perrun \
    --static-dir models/static \
    --dynamic-dir models/dynamic_random_perrun \
    --fusion-dir models/fusion_random_perrun \
    --out-dir results/random_split_perrun \
    --runs-dir data/fusionbench_sim/runs \
    --sweep-thresholds
```

---

## Step 9: Cold-Start Experiment (The Killer Result!)

This is where we show LLM-static helps when per-run normalization fails:

```python
# Read cold-start split to see which runs are contaminated
import json
with open('data/fusionbench_sim/splits/coldstart.json') as f:
    coldstart = json.load(f)
print(f"Contaminated runs: {coldstart['contaminated_runs']}")

# The cold-start experiment compares:
# 1. Dynamic + perrun norm (should degrade on contaminated runs)
# 2. Static-informed fusion (should help recover)

# Run evaluation and compare results
# Look at per-run F1 for contaminated vs clean runs
```

---

## Step 10: Download Results

```python
# Zip results for download
!zip -r results.zip results/ models/

# Download link will appear
from IPython.display import FileLink
FileLink('results.zip')
```

---

## Expected Results

| Scenario | Dynamic F1 | Fusion F1 | Notes |
|----------|------------|-----------|-------|
| Clean warmup | ~0.50 | ~0.50 | Similar |
| Contaminated warmup | **↓ degrades** | **↑ helps** | LLM value! |

---

## Time Estimates (on Kaggle GPU)

| Step | Time |
|------|------|
| Data generation | ~5 min |
| Preprocessing | ~2 min |
| Dynamic training (30 epochs) | ~10 min |
| Static training | ~2 min |
| Evaluation | ~3 min |
| **Total** | **~25 min** |
