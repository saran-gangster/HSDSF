# HSDSF-FusionBench: Complete Experiment Walkthrough

**Date**: December 29, 2024  
**Purpose**: Reference document for paper writing

---

## 1. Problem Statement

**Goal**: Detect hardware-based trojans on edge devices (NVIDIA Jetson) using thermal/power telemetry fusion.

**Challenge**: Combine two fundamentally different signals:
- **Static Expert**: Analyzes binary metadata → P(Capability) — "Is this binary a trojan?"
- **Dynamic Expert**: Analyzes runtime telemetry → P(Activation) — "Is the trojan active NOW?"

---

## 2. Architecture

### 2.1 Static Expert
- Input: ELF file features (deterministic pseudo-features in simulation)
- Model: Deep Ensemble (5 members) for uncertainty quantification
- Output: `p_s` (capability probability), `u_s` (uncertainty)
- **Binary-level AUC: 1.0** (perfect separation)

### 2.2 Dynamic Expert  
- Input: Windowed telemetry (temperature, power, utilization)
- Model: TCN + Deep Ensemble with Focal Loss
- Output: `p_d` (activation probability), `u_d` (uncertainty)
- **Window-level AUC: 0.71-0.73** (except unseen_regime: 0.60)

### 2.3 Uncertainty-Gated Fusion (UGF)
```
p_fusion = g(u_s, u_d, p_s, p_d) × p_d + (1 - g) × p_s
```
- Gate learns to route between experts based on uncertainties and predictions

---

## 3. Dataset & Splits

| Split | Description | Train/Val/Test |
|-------|-------------|----------------|
| random_split | Random 60/20/20 | 120/40/40 runs |
| unseen_workload | Test workloads not in train | 120/40/40 runs |
| unseen_trojan | Test trojan families not in train | 120/40/40 runs |
| unseen_regime | Test power mode (MAXN + high temp) | 66/0/134 runs |

**Binary distribution**: 70% trojans (14/20), 30% benign (6/20)  
**Window-level positive rate**: ~18%

---

## 4. Initial Results (Before ML Expert Consultation)

### 4.1 The Problem: Static Expert Failure at Window Level
```
Static AUC (binary-level): 1.0
Static F1 (window-level): 0.000
```
**Root cause**: Static predicts P(Capability), not P(Activation). A trojan binary is always "capable" but only occasionally "active."

### 4.2 Gate Collapse
```
Gate mean: ~0.80
Gate favors static (g < 0.5): 0%
```
The gate learned to ignore static expert entirely.

---

## 5. ML Expert Guidance (Round 1)

### 5.1 Core Diagnosis
> "Fusing two predictors estimating *different random variables* (capability vs activation) with an additive gate is fundamentally incorrect."

### 5.2 Recommended Fix: Hierarchical Factorization
```
P(Active) = P(Capability) × P(Active | Capability)
p_active = p_cap × p_act_given_cap
```

### 5.3 Additional Recommendations
- Use soft labels (continuous 0-1) instead of binary
- Train dynamic only on trojan binaries
- Add temporal smoothing (debounce)
- Focus on FAR/hour and TTD, not just F1

---

## 6. Implementation: Hierarchical Fusion

```python
def hierarchical(p_s, p_d):
    return p_s * p_d  # Multiplicative, not additive
```

### 6.1 Results: Hierarchical Underperformed

| Split | Dynamic F1 | Hierarchical F1 | Δ |
|-------|------------|-----------------|---|
| random_split | 0.564 | 0.399 | **-0.165** |
| unseen_workload | 0.501 | 0.426 | **-0.075** |
| unseen_trojan | 0.642 | 0.469 | **-0.173** |

**Why?** With 70% trojans, `p_s ≈ 0.995` for most samples, so multiplication barely changes anything.

---

## 7. Surprising Finding: UGF Actually Works!

| Split | Dynamic F1 | UGF F1 | Δ |
|-------|------------|--------|---|
| random_split | 0.564 | **0.646** | **+0.082** |
| unseen_workload | 0.501 | **0.544** | **+0.042** |
| unseen_trojan | 0.642 | 0.622 | -0.020 |
| unseen_regime | 0.296 | 0.064 | -0.232 |

UGF beats dynamic on 2/4 splits despite gate "collapse" to ~0.80!

---

## 8. ML Expert Guidance (Round 2): Ablation Studies

### 8.1 Recommended Ablations
1. **constant_gate**: Fixed g=0.80, no learned routing
2. **shuffle_static**: Randomly permute p_s to break signal
3. **remove_static_pathway**: g×p_d only, no static
4. **logit_stacking**: Learned logit-space combination

### 8.2 Ablation Results (random_split)

| Method | F1 | FAR/h | Interpretation |
|--------|-----|-------|----------------|
| UGF | **0.646** | 44.7 | Best |
| constant_gate | **0.645** | 54.5 | ≈ UGF! |
| dynamic_only | 0.564 | 76.5 | Baseline |
| remove_static | 0.564 | 76.5 | = dynamic |
| shuffle_static | 0.278 | **312** | Crashed! |

### 8.3 Key Insights

1. **constant_gate ≈ UGF** → Gain is from global rescaling, NOT learned routing
2. **remove_static = dynamic** → Gate scaling alone doesn't help
3. **shuffle_static crashed** → Static IS providing real signal (binary identity)

---

## 9. ML Expert Guidance (Round 3): The Truth

> "UGF is (mostly) a constant mixture model in disguise. Static is genuinely valuable, but almost entirely as binary identity/capability information."

### 9.1 What UGF Actually Does
```
p_fusion = 0.80 × p_d + 0.20 × p_s
         = 0.80 × p_d + 0.20 × 0.995  (trojans)
         = 0.80 × p_d + 0.199         (boosts marginal positives)
         
         = 0.80 × p_d + 0.20 × 0.168  (benign)
         = 0.80 × p_d + 0.034         (suppresses)
```

The 20% static contribution boosts trojan-binary windows and suppresses benign-binary windows — simple but effective!

### 9.2 Exception: unseen_workload
```
constant_gate F1: 0.459
UGF F1: 0.544
Δ: +0.085 (significant)
```
On this split, the learned gate IS doing something useful. Routing matters under workload shift.

---

## 10. Final Method Comparison

### 10.1 Best Methods by Split

| Split | Best Method | F1 | FAR/h | TTD |
|-------|-------------|-----|-------|-----|
| random_split | **UGF** | 0.646 | 44.7 | 0.3s |
| unseen_workload | **heuristic_gate** | 0.547 | 74.0 | 1.1s |
| unseen_trojan | **remove_static** | 0.650 | 66.7 | 0.9s |
| unseen_regime | **dynamic_only** | 0.296 | 30.7 | 0.8s |

### 10.2 All Methods (random_split)

| Method | F1 | FAR/h | Notes |
|--------|-----|-------|-------|
| UGF | **0.646** | 44.7 | Best overall |
| constant_gate | 0.645 | 54.5 | Simpler, nearly equivalent |
| piecewise_gate | 0.641 | 57.7 | 2-param version |
| dynamic_only | 0.564 | 76.5 | Base detector |
| late_fusion_learned | 0.560 | 75.2 | Learned weights |
| heuristic_gate | 0.546 | 75.1 | u_d-based routing |
| logit_stacking | 0.539 | 80.2 | Underperformed |
| late_fusion_avg | 0.447 | 46.3 | Simple average |
| hierarchical | 0.399 | 60.9 | Multiplicative |
| soft_veto | 0.403 | 59.8 | Sigmoid suppression |
| shuffle_static | 0.278 | 312 | Ablation (broken) |

---

## 11. The unseen_regime Problem

All methods fail on unseen_regime (best F1 = 0.296).

**Root cause**: Dynamic model not invariant to regime changes (MAXN + high temp).

**Recommended fixes** (not yet implemented):
1. Per-run normalization (subtract warmup baseline)
2. Group DRO across regimes
3. Simulator domain randomization

---

## 12. Conclusions for Paper

### 12.1 Main Findings

1. **UGF improves over dynamic-only by +8% F1 on in-distribution data**, primarily via static-informed score shifting

2. **The learned gate is mostly a constant mixture** — constant_gate (g=0.80) achieves nearly identical performance

3. **Static expert provides binary-level signal** that boosts trojan windows and suppresses benign windows, but cannot distinguish active vs inactive within trojan binaries

4. **Hierarchical (multiplicative) fusion underperforms** because with 70% trojans, p_s ≈ 0.995 for most samples

5. **Generalization varies by split** — no single fusion method dominates all scenarios

### 12.2 Honest Paper Narrative

> "Our Uncertainty-Gated Fusion (UGF) achieves a +8% F1 improvement over dynamic-only detection on in-distribution data. Ablation studies reveal this gain primarily comes from static-informed score shifting rather than learned per-window routing. A simple constant mixture (0.8×dynamic + 0.2×static) is a strong baseline. The learned gate provides additional benefit under certain distribution shifts (unseen workloads) but not others (unseen operating regimes)."

### 12.3 Key Design Lessons

- **Semantic alignment matters**: Don't additively fuse P(Capability) with P(Activation)
- **Ablations are essential**: constant_gate test revealed the true mechanism
- **Simple baselines first**: Many complex methods underperform simple mixing
- **FAR-matched comparison**: F1 at optimal threshold can be misleading

---

## 13. Files Reference

| Component | Key Files |
|-----------|-----------|
| Static Expert | [static/train_static.py](file:///home/saran-gangster/Desktop/Implementations%20&%20Projects/HSDSF/static/train_static.py), [static/calibrate_static.py](file:///home/saran-gangster/Desktop/Implementations%20&%20Projects/HSDSF/static/calibrate_static.py) |
| Dynamic Expert | [dynamic/train_dynamic.py](file:///home/saran-gangster/Desktop/Implementations%20&%20Projects/HSDSF/dynamic/train_dynamic.py), [dynamic/calibrate_dynamic.py](file:///home/saran-gangster/Desktop/Implementations%20&%20Projects/HSDSF/dynamic/calibrate_dynamic.py) |
| Fusion | [fusion/baselines.py](file:///home/saran-gangster/Desktop/Implementations%20&%20Projects/HSDSF/fusion/baselines.py), [fusion/train_fusion.py](file:///home/saran-gangster/Desktop/Implementations%20&%20Projects/HSDSF/fusion/train_fusion.py), [fusion/eval_fusion.py](file:///home/saran-gangster/Desktop/Implementations%20&%20Projects/HSDSF/fusion/eval_fusion.py) |
| Evaluation | [evaluation/events.py](file:///home/saran-gangster/Desktop/Implementations%20&%20Projects/HSDSF/evaluation/events.py), [evaluation/metrics.py](file:///home/saran-gangster/Desktop/Implementations%20&%20Projects/HSDSF/evaluation/metrics.py) |
| Pipeline | [experiments/run_all.sh](file:///home/saran-gangster/Desktop/Implementations%20&%20Projects/HSDSF/experiments/run_all.sh) |
| Documentation | [docs/ML_EXPERT_REPORT.md](file:///home/saran-gangster/Desktop/Implementations%20&%20Projects/HSDSF/docs/ML_EXPERT_REPORT.md), [docs/ML_EXPERT_FOLLOWUP.md](file:///home/saran-gangster/Desktop/Implementations%20&%20Projects/HSDSF/docs/ML_EXPERT_FOLLOWUP.md) |

---

## 14. Quantitative Summary Table (for paper)

| Method | random | unseen_wl | unseen_troj | unseen_reg |
|--------|--------|-----------|-------------|------------|
| static_only | 0.000 | 0.000 | 0.000 | 0.000 |
| dynamic_only | 0.564 | 0.501 | 0.642 | **0.296** |
| UGF | **0.646** | **0.544** | 0.622 | 0.064 |
| constant_gate | 0.645 | 0.459 | 0.626 | 0.065 |
| hierarchical | 0.399 | 0.426 | 0.469 | 0.240 |
| late_fusion_avg | 0.447 | 0.462 | 0.643 | 0.062 |
| product_of_experts | 0.442 | 0.472 | 0.575 | 0.190 |

**Bold** = best for that split

---

# ML Expert Guidance (Round 4): Paper Strategy

## 15. How to Frame UGF Credibly

**Recommended framing**: Lead with operational improvement, then show mechanism via ablations.

### Paper Claim Structure

1. **Claim (what you achieved):**
   > "Static–dynamic fusion improves real-time trojan *activation* detection, delivering +0.082 absolute F1, −42% FAR/h, and −0.5s TTD on in-distribution evaluation."

2. **Evidence (why it happens):**
   > "Ablations show the gain does *not* come from per-window routing; the gate converges to an approximately constant mixture where static acts as a binary-identity-dependent score shift."

3. **Engineering takeaway (what to deploy):**
   > "A constant-mixture baseline matches the learned gate on most splits; the learned gate only helps under certain workload shifts."

### Recommended One-Paragraph Summary

> "UGF improves detection primarily by injecting static capability information as a calibrated score shift, reducing FAR and improving time-to-detect. Surprisingly, learned gating rarely performs fine-grained routing in our setting; a constant mixture reproduces most gains. We document this with targeted ablations and show when adaptivity matters (unseen workload shift) and when all methods fail (unseen regime)."

---

## 16. Priority Experiments to Strengthen Paper

### Priority 1: FAR-Matched Evaluation (HIGHEST)
- Choose FAR targets: 1/h, 5/h, 20/h
- For each method, pick threshold on validation to hit FAR target
- Report **event recall + TTD distribution** on test at those thresholds
- Makes the work feel "deployment-real" rather than "ML-metric driven"

### Priority 2: Temporal Smoothing Integration
- Add k-of-n debounce or EMA postprocessor to final decision layer
- Evaluate same FAR-matched protocol
- Most effective way to reduce FAR without killing TTD

### Priority 3: Per-Binary Breakdown
- FAR/h on **benign binaries only** (static should shine here)
- Detection/TTD on **trojan binaries only**
- Makes "static score shifting" mechanism concrete and defensible

### Priority 4: Sensitivity Analysis for g
- One figure: F1 / FAR / TTD vs g ∈ {0.6, 0.7, 0.8, 0.9, 0.95}
- Reinforces method stability and tunability

### Priority 5: unseen_regime Mitigation
- Implement per-run normalization
- Even if it barely helps, shows "we tried the obvious fix"

### Priority 6: Simulator Limitations
- Domain randomization study (robustness across parameter shifts)
- Or small hardware sanity-check trace

---

## 17. Final Paper Structure Recommendation

1. **Main method**: Present UGF as the intended design
2. **Main result**: Fusion improves FAR/TTD materially  
3. **Key scientific contribution**: Ablation-driven mechanism discovery → constant mixture is near-optimal
4. **Practical takeaway**: Recommend constant mixing as default; keep learned gate for workload shift
5. **Limitations**: unseen_regime remains hard; provide at least one mitigation attempt

### The "Selling" Sentence

> "We show that static capability signals can be exploited to materially reduce false alarms and detection latency in real-time trojan activation detection; extensive ablations reveal that most gains arise from a simple, stable constant-mixture fusion, while adaptive gating helps only under specific workload shifts."

---

## 18. Implementation Checklist

### Must Do (Before Paper Submission)
- [ ] FAR-matched evaluation at 5/h, 10/h, 20/h targets
- [ ] Per-binary analysis (trojan vs benign breakdown)
- [ ] Sensitivity plot for constant g
- [ ] unseen_regime with per-run normalization attempt

### Should Do (Strengthens Paper)
- [ ] Temporal smoothing (debounce) in final results
- [ ] Tune g on validation, report tuned value

### Nice to Have
- [ ] Domain randomization study
- [ ] Small real hardware trace

---

## 19. Key Metric Movements to Highlight

| Metric | Dynamic | UGF | Improvement |
|--------|---------|-----|-------------|
| F1 | 0.564 | 0.646 | +14.5% relative |
| FAR/h | 76.5 | 44.7 | **−42%** |
| TTD | 0.8s | 0.3s | **−0.5s** |

All three improvements are practically meaningful for real-time deployment.

---

## 20. Caveats to Acknowledge

1. **Absolute FAR/h still high** — present results at fixed FAR targets
2. **Simulation-only** — acknowledge and mitigate with domain randomization
3. **unseen_regime failure** — include in main table, show mitigation attempt
4. **random_split may be optimistic** — emphasize generalization splits