# Cold-Start & FAR-Matched Experiment Results

**Date**: January 1, 2026  
**Purpose**: Validate cold-start hypothesis and operational FAR-matched evaluation

---

## 1. Cold-Start Contamination Results

### Experiment Setup
| Parameter | Value |
|-----------|-------|
| Total test runs | 40 |
| Trojan test runs | 34 |
| **Contaminated runs** | **17 (50%)** |
| Contamination period | Steps 50-150 (5-15 seconds) |
| Clean runs windows | 2553 |
| Contaminated runs windows | 1887 |

### Results

| Condition | F1 | Precision | Recall | PR-AUC |
|-----------|-----|-----------|--------|--------|
| Clean warmup | 0.320 | 0.199 | 0.826 | 0.312 |
| **Contaminated warmup** | **0.408** | **0.272** | **0.819** | **0.501** |

### ⚠️ Surprising Finding: Contaminated Runs Perform BETTER!

**Expected hypothesis**: Contaminated warmup → degraded performance (baseline poisoned)  
**Actual result**: Contaminated runs show **+27% higher F1** (0.41 vs 0.32)

### Analysis: Why This Happened

1. **More positive windows in contaminated runs**
   - Trojan active during warmup (steps 50-150) + normal activation
   - Higher positive rate → easier detection task

2. **Label distribution effect**
   - Contaminated runs: More positive windows per run
   - PR-AUC much higher (0.50 vs 0.31) suggests better class separation

3. **The "contamination" is actually providing more positive training signal**
   - Not truly a cold-start scenario in the evaluation sense
   - Would need to evaluate ONLY during non-warmup period

### Revised Interpretation

The experiment shows that **more trojan activity** (even during warmup) makes detection **easier**, not harder. This suggests:

1. Per-run normalization is robust to warmup contamination
2. The "cold-start" problem may be less severe than hypothesized
3. OR the evaluation methodology needs refinement

### Paper Statement (Honest)

> "We tested per-run normalization under warmup contamination (trojan active during baseline period). Surprisingly, performance improved (+27% F1) rather than degraded, likely because contaminated runs contain more positive signal. This suggests per-run normalization is more robust than hypothesized, or that our evaluation methodology requires refinement to isolate the cold-start effect."

---

## 2. FAR-Matched Evaluation Results

### Results Table

| FAR Target (1/h) | Threshold | Actual FAR | Recall | Precision |
|------------------|-----------|------------|--------|-----------|
| 1 | 0.772 | 0.0 | 0.0% | — |
| 5 | 0.762 | 3.7 | 3.2% | 87.1% |
| 10 | 0.733 | 12.1 | 8.9% | 85.4% |
| 20 | 0.683 | 20.4 | 14.2% | 84.6% |
| 50 | 0.554 | 50.1 | 19.9% | 75.8% |

### Key Insights

1. **Very Low FAR (1/h) is impractical**
   - Threshold too high → zero detections
   - No recall at 1 FAR/h target

2. **5-20 FAR/h is the operational sweet spot**
   - 5 FAR/h: 3.2% recall, 87% precision
   - 20 FAR/h: 14.2% recall, 85% precision

3. **High FAR (50/h) provides ~20% recall**
   - 1 in 5 true events detected
   - Still operationally challenging

4. **Precision remains high (75-87%)**
   - When we do alert, we're usually correct
   - Useful for high-confidence alerts

### Recall-FAR Trade-off

```
FAR/h:   1    5   10   20   50
Recall:  0%  3%   9%  14%  20%
```

**Interpretation**: Doubling FAR roughly adds 5-10% recall.

### Paper Statement

> "At operationally realistic FAR targets (5-20/h), window-level recall ranges from 3% to 14%. Precision remains high (85-87%), indicating that when the detector alerts, it is usually correct. This recall-FAR trade-off is typical for rare-event detection and highlights the value of fusion methods that can improve recall at fixed FAR."

---

## 3. Combined Insights

### What We Learned

1. **Per-run normalization is robust to contamination** (or our test isn't capturing the right effect)
2. **Low FAR requires accepting low recall** — fundamental trade-off
3. **Precision is stable** — alerts are trustworthy

### Implications for Deployment

| Scenario | Recommended FAR | Expected Recall |
|----------|-----------------|-----------------|
| High-security (minimize false alarms) | 5/h | ~3% |
| Balanced | 20/h | ~14% |
| High-sensitivity (catch more) | 50/h | ~20% |

### Future Work

1. **Refine cold-start evaluation**
   - Evaluate only on non-contaminated time windows
   - Control for positive rate differences

2. **Event-level FAR matching**
   - Current is window-level
   - Event-level may show different trade-offs

---

## 4. Paper-Ready Quotes

### Cold-Start
> "We tested per-run normalization robustness by contaminating the warmup period with trojan activity. Contrary to our hypothesis, performance improved (+27% F1), suggesting either robust normalization or confounding from increased positive rates in contaminated runs."

### FAR-Matched
> "At FAR=20/h, the detector achieves 14.2% window-level recall with 84.6% precision. This recall-FAR trade-off characterizes the operational design space for deployment."

### Combined
> "Our experiments reveal that purely dynamic detection faces fundamental recall limitations at operationally acceptable FAR levels, motivating the use of static fusion for operating-point control."
