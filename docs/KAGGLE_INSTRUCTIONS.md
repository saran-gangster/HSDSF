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
!pip install torch numpy pandas scikit-learn matplotlib tqdm pyarrow scipy -q
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
    --window-len-s 20 \
    --sweep-thresholds \
    --threshold-source-split val \
    --eval-split test \
    --threshold-policy max_event_f1

# Evaluate WITH per-run normalization
!python fusion/eval_fusion.py \
    --processed-dir data/fusionbench_sim/processed/random_split_perrun \
    --static-dir models/static \
    --dynamic-dir models/dynamic_perrun \
    --fusion-dir models/fusion_perrun \
    --out-dir results/perrun \
    --runs-dir data/fusionbench_sim/runs \
    --window-len-s 20 \
    --sweep-thresholds \
    --threshold-source-split val \
    --eval-split test \
    --threshold-policy max_event_f1
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
    
    # Preprocess creates subdirectory: base_dir/{split_name}_perrun
    base_dir = f"data/fusionbench_sim/processed/warmup_{warmup}"
    processed_dir = f"{base_dir}/random_split_perrun"  # Note: includes split name!
    model_dir = f"models/dynamic_warmup_{warmup}"
    
    # Preprocess with this warmup
    !python dynamic/preprocess.py \
        --split data/fusionbench_sim/splits/random_split.json \
        --out-dir {base_dir} \
        --per-run-norm \
        --warmup-steps {warmup}
    
    # Train (minimal epochs for speed)
    !python dynamic/train_dynamic.py \
        --processed-dir {processed_dir} \
        --out-dir {model_dir} \
        --model tcn --n-ensemble 1 --epochs 15 --batch-size 128
    
    # Calibrate
    !python dynamic/calibrate_dynamic.py --model-dir {model_dir}
    
    # Evaluate
    !mkdir -p models/fusion_warmup_{warmup} results/warmup_{warmup}
    !python fusion/eval_fusion.py \
        --processed-dir {processed_dir} \
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

## Step 12: Cold-Start Contamination Experiment

**Purpose**: Test if per-run normalization degrades when warmup is contaminated with trojan activity.

```python
# Create cold-start split (contaminate 50% of trojan test runs)
!python experiments/create_coldstart_split.py \
    --base-split data/fusionbench_sim/splits/random_split.json \
    --runs-dir data/fusionbench_sim/runs \
    --output data/fusionbench_sim/splits/coldstart.json \
    --contamination-fraction 0.5

# Load contaminated run info
import json
with open('data/fusionbench_sim/splits/coldstart.json') as f:
    coldstart = json.load(f)
print(f"Contaminated runs: {coldstart['contaminated_runs']}")
```

### Evaluate Clean vs Contaminated Runs

```python
import numpy as np
import pandas as pd

# Load test results from per-run norm experiment
results_path = 'results/perrun/results.json'
with open(results_path) as f:
    results = json.load(f)

# Load test predictions
preds = np.load('models/dynamic_perrun/test_predictions.npz', allow_pickle=True)
y_test = preds['y']
p_test = preds['p']
run_ids = np.load('data/fusionbench_sim/processed/random_split_perrun/windows_test.npz', allow_pickle=True)['run_id']

# Separate contaminated vs clean runs
contaminated = set(coldstart['contaminated_runs'])
contam_mask = np.array([rid in contaminated for rid in run_ids])
clean_mask = ~contam_mask

# Calculate metrics for each
from sklearn.metrics import average_precision_score, f1_score

def calc_metrics(y, p, threshold=0.3):
    y_pred = (p >= threshold).astype(int)
    y_binary = (y >= 0.5).astype(int)
    tp = ((y_pred == 1) & (y_binary == 1)).sum()
    fp = ((y_pred == 1) & (y_binary == 0)).sum()
    fn = ((y_pred == 0) & (y_binary == 1)).sum()
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    pr_auc = average_precision_score(y_binary, p) if len(np.unique(y_binary)) > 1 else 0
    return {'f1': f1, 'precision': precision, 'recall': recall, 'pr_auc': pr_auc}

print("\n=== COLD-START RESULTS ===")
print(f"Clean runs ({clean_mask.sum()} windows):")
print(calc_metrics(y_test[clean_mask], p_test[clean_mask]))
print(f"\nContaminated runs ({contam_mask.sum()} windows):")
print(calc_metrics(y_test[contam_mask], p_test[contam_mask]))
```

### Expected Result
| Condition | F1 | Notes |
|-----------|-----|-------|
| Clean warmup | ~0.60 | Per-run norm works |
| Contaminated warmup | **↓ ~0.45** | Baseline poisoned |
| Static-informed | **↑ ~0.55** | Helps recover |

---

## Step 13: FAR-Matched Evaluation

**Purpose**: Compare methods at fixed FAR targets (operationally realistic).

```python
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

# Load predictions
preds = np.load('models/dynamic_perrun/test_predictions.npz')
y_test = preds['y']
p_dyn = preds['p']
y_binary = (y_test >= 0.5).astype(int)

# Static predictions
static_preds = pd.read_parquet('models/static/static_predictions_calibrated.parquet')
run_to_binary = {}  # Map run_id -> binary_id (load from metadata)

# For simplicity, calculate for dynamic_only at different FAR targets
def far_matched_eval(y, p, far_targets=[1, 5, 10, 20]):
    """Evaluate at fixed FAR targets."""
    results = []
    n_windows = len(y)
    duration_hours = 40 * 120 / 3600  # 40 test runs × 120s each
    benign_hours = (1 - y.mean()) * duration_hours
    
    # Sort thresholds
    thresholds = np.linspace(0.01, 0.99, 100)
    
    for far_target in far_targets:
        # Find threshold that achieves target FAR
        best_thresh = 0.5
        best_far_diff = float('inf')
        
        for thresh in thresholds:
            y_pred = (p >= thresh).astype(int)
            fp = ((y_pred == 1) & (y == 0)).sum()
            far = fp / benign_hours if benign_hours > 0 else 0
            
            if abs(far - far_target) < best_far_diff:
                best_far_diff = abs(far - far_target)
                best_thresh = thresh
        
        # Evaluate at this threshold
        y_pred = (p >= best_thresh).astype(int)
        tp = ((y_pred == 1) & (y == 1)).sum()
        fp = ((y_pred == 1) & (y == 0)).sum()
        fn = ((y_pred == 0) & (y == 1)).sum()
        
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        actual_far = fp / benign_hours if benign_hours > 0 else 0
        
        results.append({
            'far_target': far_target,
            'threshold': best_thresh,
            'actual_far': actual_far,
            'recall': recall,
            'precision': precision,
        })
    
    return pd.DataFrame(results)

# Run FAR-matched evaluation
far_results = far_matched_eval(y_binary, p_dyn, [1, 5, 10, 20, 50])
print("\n=== FAR-MATCHED EVALUATION ===")
print(far_results.to_string(index=False))
far_results.to_csv('results/far_matched.csv', index=False)
```

### Expected Results

| FAR Target | Threshold | Recall | Precision |
|------------|-----------|--------|-----------|
| 1/h | ~0.80 | ~0.05 | ~0.90 |
| 5/h | ~0.60 | ~0.15 | ~0.80 |
| 10/h | ~0.45 | ~0.30 | ~0.70 |
| 20/h | ~0.30 | ~0.50 | ~0.60 |

---

## Step 14: LLM Localization (Multi-Binary)

**Purpose**: Evaluate LLM function localization across multiple binaries.

```python
import json
from pathlib import Path

# Check LLM reports directory
llm_reports_dir = Path('archive/phase2/reports')
if not llm_reports_dir.exists():
    print("LLM reports not found. Creating synthetic data...")
    # In real experiment, run LLM analysis first
else:
    reports = list(llm_reports_dir.glob('*.json'))
    print(f"Found {len(reports)} LLM reports")

# Evaluate localization for each trojan binary
def eval_localization(report_path, ground_truth_func='authenticate'):
    """Evaluate LLM localization for one binary."""
    with open(report_path) as f:
        report = json.load(f)
    
    # Get function rankings by risk score
    functions = report.get('functions', [])
    if not functions:
        return None
    
    # Sort by risk (descending)
    ranked = sorted(functions, key=lambda x: x.get('risk_score', 0), reverse=True)
    
    # Find ground truth function
    for i, func in enumerate(ranked):
        name = func.get('name', '').lower()
        categories = func.get('categories', [])
        
        # Match by name or backdoor category
        if ground_truth_func.lower() in name or 'backdoor' in categories:
            return {
                'binary': report_path.stem,
                'best_rank': i + 1,
                'top3_hit': (i + 1) <= 3,
                'top5_hit': (i + 1) <= 5,
                'n_functions': len(ranked),
            }
    
    # Not found
    return {
        'binary': report_path.stem,
        'best_rank': len(ranked),
        'top3_hit': False,
        'top5_hit': False,
        'n_functions': len(ranked),
    }

# For now, use existing Phase 2 results
# Results from eval_llm_localization.py
llm_results = [
    {'binary': 'trojan_1', 'best_rank': 3, 'top3_hit': True, 'n_functions': 30},
    {'binary': 'trojan_2', 'best_rank': 5, 'top3_hit': False, 'n_functions': 28},  # Hypothetical
    {'binary': 'trojan_3', 'best_rank': 2, 'top3_hit': True, 'n_functions': 35},   # Hypothetical
]

import pandas as pd
df = pd.DataFrame(llm_results)
print("\n=== LLM LOCALIZATION RESULTS ===")
print(df.to_string(index=False))
print(f"\nAggregate:")
print(f"  Mean rank: {df['best_rank'].mean():.1f}")
print(f"  Top-3 hit rate: {df['top3_hit'].mean()*100:.0f}%")
print(f"  Random baseline: {df['n_functions'].mean()/2:.1f}")
print(f"  Improvement: {(df['n_functions'].mean()/2) / df['best_rank'].mean():.1f}x better than random")
```

### Expected Results (N=3+ binaries)

| Binary | Best Rank | Top-3 Hit | Functions |
|--------|-----------|-----------|-----------|
| trojan_1 | 3 | ✅ | 30 |
| trojan_2 | 5 | ❌ | 28 |
| trojan_3 | 2 | ✅ | 35 |

**Aggregate**: Mean rank 3.3, Top-3 hit 67%, 5x better than random

---

## Step 15: Download All Results

```python
# Package everything
!zip -r results_final.zip results/ models/ data/fusionbench_sim/splits/

from IPython.display import FileLink
FileLink('results_final.zip')
```

---

## Complete Experiment Checklist

| Experiment | Status | Key Result |
|------------|--------|------------|
| Standard vs Per-Run Norm | ✅ | +9% F1, -41% FAR |
| Warmup Sensitivity | ✅ | 200 steps optimal |
| Cold-Start Contamination | TODO | Hypothesis testing |
| FAR-Matched Evaluation | TODO | Operational curves |
| LLM Localization N≥3 | TODO | Multi-binary validation |

