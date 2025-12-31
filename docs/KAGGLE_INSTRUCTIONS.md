# HSDSF Cold-Start Experiment — Kaggle GPU Instructions (v2)

> **Updated**: Larger dataset (200 runs, 20 binaries) for proper cold-start hypothesis testing

## Step 1: Create Kaggle Notebook

1. Go to https://www.kaggle.com/code
2. Click "New Notebook"
3. Enable GPU: Settings → Accelerator → **GPU T4 x2**

---

## Step 2: Upload Your Project

```python
!git clone https://github.com/YOUR_USERNAME/HSDSF.git
%cd HSDSF

# Set PYTHONPATH
import os
os.environ['PYTHONPATH'] = '/kaggle/working/HSDSF'
```

---

## Step 3: Install Dependencies

```python
!pip install torch numpy pandas scikit-learn matplotlib tqdm pyarrow -q
```

---

## Step 4: Generate LARGER Dataset (200 runs)

```python
# Generate 200 runs (20 binaries × 10 runs each)
# This gives enough samples for proper static training!
!python experiments/generate_fusionbench_sim.py \
    --n-binaries 20 \
    --runs-per-binary 10 \
    --duration-s 120 \
    --include-benign-confounders

# Create splits with proper train/val/test distribution
!python experiments/make_splits.py \
    --use-random-splits \
    --train-frac 0.6 \
    --val-frac 0.2

# Create cold-start split (contaminated warmup)
!python experiments/create_coldstart_split.py \
    --base-split data/fusionbench_sim/splits/random_split.json \
    --runs-dir data/fusionbench_sim/runs \
    --output data/fusionbench_sim/splits/coldstart.json \
    --contamination-fraction 0.5
```

---

## Step 5: Preprocess Data (Both Standard and Per-Run Norm)

```python
# Standard preprocessing (without per-run norm)
!python dynamic/preprocess.py \
    --split data/fusionbench_sim/splits/random_split.json

# WITH per-run normalization (the key comparison!)
!python dynamic/preprocess.py \
    --split data/fusionbench_sim/splits/random_split.json \
    --per-run-norm \
    --warmup-steps 200
```

---

## Step 6: Train Static Model (Now with 20 binaries!)

```python
# Extract static features (20 binaries = ~16 train, ~4 val)
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

## Step 7: Train Dynamic Model (Two Versions)

```python
# Version 1: Standard (without per-run norm)
!python dynamic/train_dynamic.py \
    --processed-dir data/fusionbench_sim/processed/random_split \
    --out-dir models/dynamic_standard \
    --model tcn \
    --n-ensemble 1 \
    --epochs 30 \
    --batch-size 128

!python dynamic/calibrate_dynamic.py \
    --model-dir models/dynamic_standard

# Version 2: WITH per-run normalization
!python dynamic/train_dynamic.py \
    --processed-dir data/fusionbench_sim/processed/random_split_perrun \
    --out-dir models/dynamic_perrun \
    --model tcn \
    --n-ensemble 1 \
    --epochs 30 \
    --batch-size 128

!python dynamic/calibrate_dynamic.py \
    --model-dir models/dynamic_perrun
```

---

## Step 8: Evaluate Both Versions

```python
# Create fusion directories
!mkdir -p models/fusion_standard models/fusion_perrun

# Evaluate STANDARD (without per-run norm)
!python fusion/eval_fusion.py \
    --processed-dir data/fusionbench_sim/processed/random_split \
    --static-dir models/static \
    --dynamic-dir models/dynamic_standard \
    --fusion-dir models/fusion_standard \
    --out-dir results/standard \
    --runs-dir data/fusionbench_sim/runs \
    --sweep-thresholds

# Evaluate WITH per-run normalization
!python fusion/eval_fusion.py \
    --processed-dir data/fusionbench_sim/processed/random_split_perrun \
    --static-dir models/static \
    --dynamic-dir models/dynamic_perrun \
    --fusion-dir models/fusion_perrun \
    --out-dir results/perrun \
    --runs-dir data/fusionbench_sim/runs \
    --sweep-thresholds
```

---

## Step 9: Cold-Start Analysis (The Key Result!)

```python
import json
import pandas as pd

# Load cold-start split info
with open('data/fusionbench_sim/splits/coldstart.json') as f:
    coldstart = json.load(f)

contaminated = set(coldstart['contaminated_runs'])
print(f"Contaminated runs ({len(contaminated)}): {sorted(contaminated)[:5]}...")

# Compare performance on contaminated vs clean runs
# This is where we expect:
# - Standard: Degrades on contaminated runs
# - Per-run norm: Also degrades (warmup poisoned)
# - Static-fusion: Should help recover!

# Load per-run results if available in results.json
for split in ['standard', 'perrun']:
    results_path = f'results/{split}/results.json'
    try:
        with open(results_path) as f:
            results = json.load(f)
        print(f"\n=== {split.upper()} ===")
        for method in ['dynamic_only', 'constant_gate', 'hierarchical_veto']:
            if method in results:
                print(f"{method}: F1={results[method].get('event_f1', 'N/A'):.3f}")
    except:
        print(f"No results for {split}")
```

---

## Step 10: Download Results

```python
!zip -r results_v2.zip results/ models/ data/fusionbench_sim/splits/

from IPython.display import FileLink
FileLink('results_v2.zip')
```

---

## Expected Results (Cold-Start Hypothesis)

### Scenario 1: Clean Warmup
| Condition | Dynamic F1 | Dynamic+Static F1 |
|-----------|------------|-------------------|
| Standard | ~0.50 | ~0.50 |
| Per-run norm | **~0.60** | ~0.55 |

### Scenario 2: Contaminated Warmup
| Condition | Dynamic F1 | Dynamic+Static F1 |
|-----------|------------|-------------------|
| Standard | ~0.50 | ~0.50 |
| Per-run norm | **↓ degrades** | **↑ static helps** |

**The hypothesis**: When per-run normalization fails (contaminated warmup), static-informed fusion should recover performance.

---

## Time Estimates (Kaggle GPU)

| Step | Time |
|------|------|
| Data generation (200 runs) | ~8 min |
| Preprocessing (both) | ~4 min |
| Static training | ~2 min |
| Dynamic training (x2) | ~20 min |
| Evaluation (x2) | ~6 min |
| **Total** | **~40 min** |

---

## Key Differences from v1

| Aspect | v1 | v2 |
|--------|----|----|
| Binaries | 10 | **20** |
| Runs | 80 | **200** |
| Static train samples | 8 | **~16** |
| Per-run norm comparison | ❌ | **✅** |
| Cold-start analysis | ❌ | **✅** |

---

## Step 11: Warmup Sensitivity Experiment (Key Paper Figure!)

This tests how performance varies with warmup period length.

```python
# Quick version: Test 4 warmup values
for warmup in [50, 100, 200, 400]:
    print(f"\n{'='*50}")
    print(f"Testing warmup_steps={warmup}")
    print(f"{'='*50}")
    
    out_dir = f"data/fusionbench_sim/processed/warmup_{warmup}"
    model_dir = f"models/dynamic_warmup_{warmup}"
    
    # Preprocess with this warmup
    !python dynamic/preprocess.py \
        --split data/fusionbench_sim/splits/random_split.json \
        --out-dir {out_dir} \
        --per-run-norm \
        --warmup-steps {warmup}
    
    # Train (minimal epochs for speed)
    !python dynamic/train_dynamic.py \
        --processed-dir {out_dir} \
        --out-dir {model_dir} \
        --model tcn --n-ensemble 1 --epochs 15 --batch-size 128
    
    # Calibrate
    !python dynamic/calibrate_dynamic.py --model-dir {model_dir}
    
    # Evaluate
    !mkdir -p models/fusion_warmup_{warmup} results/warmup_{warmup}
    !python fusion/eval_fusion.py \
        --processed-dir {out_dir} \
        --static-dir models/static \
        --dynamic-dir {model_dir} \
        --fusion-dir models/fusion_warmup_{warmup} \
        --out-dir results/warmup_{warmup} \
        --runs-dir data/fusionbench_sim/runs \
        --sweep-thresholds
```

### Collect Results

```python
import json
import pandas as pd

# Collect warmup sensitivity results
warmup_results = []
for warmup in [50, 100, 200, 400]:
    try:
        with open(f'results/warmup_{warmup}/results.json') as f:
            data = json.load(f)
        for method in data:
            if method['method'] == 'dynamic_only':
                warmup_results.append({
                    'warmup_steps': warmup,
                    'warmup_seconds': warmup * 0.1,  # 10 Hz sampling
                    'event_f1': method['event_f1'],
                    'far_per_hour': method['far_per_hour'],
                    'ttd_median_s': method['ttd_median_s'],
                })
                break
    except Exception as e:
        print(f"Error for warmup={warmup}: {e}")

df = pd.DataFrame(warmup_results)
print("\nWARMUP SENSITIVITY RESULTS")
print("="*60)
print(df.to_string(index=False))

# Save for paper
df.to_csv('results/warmup_sensitivity.csv', index=False)
```

### Expected Results

| Warmup Steps | Warmup (s) | Expected F1 | Notes |
|--------------|------------|-------------|-------|
| 50 | 5s | ~0.50 | Too short |
| 100 | 10s | ~0.55 | Marginal |
| 200 | 20s | ~0.60 | Sweet spot |
| 400 | 40s | ~0.60 | Plateau |

### Time Estimate
~30 min (4 warmup values × ~7 min each)

---

## Step 12: Download All Results

```python
# Include warmup sensitivity results
!zip -r results_complete.zip results/ models/ data/fusionbench_sim/splits/

from IPython.display import FileLink
FileLink('results_complete.zip')
```
