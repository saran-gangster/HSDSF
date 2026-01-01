# HSDSF: Hybrid Static-Dynamic Fusion for Firmware Trojan Detection on Edge Devices

---

## Abstract

Firmware-level trojans embedded in edge device software pose a significant security threat, capable of exfiltrating data or establishing persistence while evading traditional static analysis. We present **HSDSF**, a hybrid static-dynamic fusion system for **online detection** of trojan activation using thermal and power telemetry from NVIDIA Jetson-class devices.

Our key contribution is **per-run baseline normalization**, which reduces false alarm rates by 41% (50.6 → 30.0 FAR/h) while improving Event-F1 by 9% (0.55 → 0.60). Per-run warmup normalization is the dominant contributor; fusion primarily shapes operating points by conditioning on binary context. Warmup sensitivity analysis shows 53% Event-F1 improvement from 5→20 seconds, with diminishing returns beyond.

At operationally realistic FAR targets (20/h), our system achieves 14% recall with 85% precision. In a **pilot study (N=3 binaries)**, LLM-based function localization achieves 4.8× better ranking than random, suggesting promise for analyst triage.

---

## 1. Introduction

Edge devices running machine learning inference—from autonomous vehicles to industrial controllers—are increasingly targeted by supply chain attacks. Firmware-level trojans can remain dormant during testing, only activating under specific runtime conditions to exfiltrate data or manipulate outputs.

**The challenge**: Traditional static analysis can identify *capability* (is this binary potentially malicious?) but cannot determine *activation* (is the trojan currently executing?). Conversely, runtime monitoring can detect anomalous behavior but struggles with high false alarm rates due to legitimate workload variations.

**Our approach**: We propose **HSDSF (Hybrid Static-Dynamic Fusion)**, which combines:
1. **Static analysis** of firmware binaries (handcrafted features + LLM-based function localization)
2. **Dynamic monitoring** of thermal/power telemetry using temporal convolutional networks
3. **Fusion** that combines dynamic activation scores with a static capability prior for FAR/recall tradeoff control

**Key insight**: We discover that **per-run baseline normalization** is the dominant factor in detection performance, reducing false alarms by 41% while improving F1 by 9%. This simple preprocessing step outperforms complex fusion architectures.

### Contributions

1. **Per-run normalization as the key intervention**: We identify and validate that computing feature z-scores using a 20-second warmup baseline is the most effective technique for regime-robust detection.

2. **Systematic evaluation framework**: We introduce FusionBench-Sim, a simulated benchmark with 200 runs across 20 binaries, enabling reproducible comparison of fusion strategies.

3. **LLM-based function localization (pilot)**: In a pilot study (N=3), we demonstrate that LLM static analysis achieves 4.8× better localization than random, with estimated 79% inspection reduction under our labeling assumption (one trojan function per binary).

4. **Extensive ablation studies**: Through targeted experiments (shuffle_static, constant_gate, warmup sensitivity), we reveal the true mechanisms behind fusion gains.

---

## 2. Threat Model

### 2.1 Attacker Capabilities

We consider an attacker who:
- **Embeds trojan logic** in firmware binaries deployed to edge devices
- **Triggers activation** via specific inputs, timing, or environmental conditions
- **Performs malicious actions** including compute spikes (cryptomining), memory pressure (cache side-channels), I/O beacons (data exfiltration), or credential bypass (backdoor access)

The attacker **cannot**:
- Modify the telemetry sampling infrastructure
- Disable power or thermal sensors
- Fully eliminate thermal/power effects of trojan activation without sacrificing payload effectiveness

**Worst-case assumption**: The attacker knows the defender uses per-run normalization and may attempt to contaminate the warmup period.

### 2.2 Defender Observables

The defender monitors:

| Sensor | Sampling Rate | Description |
|--------|---------------|-------------|
| CPU temperature | 10 Hz | Per-core thermal sensors |
| GPU temperature | 10 Hz | Discrete GPU thermal |
| Power consumption | 10 Hz | VDD_IN, VDD_SYS rails |
| CPU/GPU utilization | 10 Hz | Load metrics |
| Memory utilization | 10 Hz | DRAM bandwidth |
| Fan speed | 10 Hz | Cooling feedback |

### 2.3 Deployment Assumptions

| Condition | Assumption |
|-----------|------------|
| Clean warmup | First 20 seconds of each run is likely benign |
| Action on detection | Alert only (kill/quarantine is future work) |
| Latency tolerance | Detection within 20-30 seconds acceptable |

---

## 3. System Design

### 3.1 Overview

HSDSF consists of three components:

```
                    ┌─────────────────┐
                    │  Binary File    │
                    └────────┬────────┘
                             │
              ┌──────────────┴──────────────┐
              ▼                             ▼
    ┌─────────────────┐           ┌─────────────────┐
    │  Static Expert  │           │  Dynamic Expert │
    │  (Capability)   │           │  (Activation)   │
    └────────┬────────┘           └────────┬────────┘
             │ p_s, u_s                    │ p_d, u_d
             └──────────────┬──────────────┘
                            ▼
                  ┌─────────────────┐
                  │ Fusion Combiner │
                  │  p = g·p_d +    │
                  │    (1-g)·p_s    │
                  └────────┬────────┘
                           ▼
                      Detection
```

### 3.2 Static Expert

**Input**: ELF binary file  
**Output**: Capability probability `p_s`, uncertainty `u_s`

We extract 22 features from each binary:
- Section sizes (text, data, rodata, bss)
- Symbol counts (imports, exports, dynamic libs)
- Entropy measures (per-section and overall)
- Suspicious patterns (URLs, IPs, hardcoded paths)
- Security features (PIE, stack canary, stripped)

A random forest ensemble (5 members) predicts trojan capability with temperature-scaled calibration.

**Binary-level AUC: 1.00** (perfect separation in simulation)

### 3.3 Dynamic Expert

**Input**: Windowed telemetry [N × 200 × 55]  
**Output**: Activation probability `p_d`, uncertainty `u_d`

We use a Temporal Convolutional Network (TCN) [11] with:
- 3 residual blocks, kernel size 7
- Focal loss for class imbalance (18% positive rate)
- Temperature calibration post-training

**Key preprocessing**: Per-run baseline normalization (applied before windowing)
```python
# Normalize each feature by warmup baseline (per-feature, causal)
warmup = run_features[:200]           # First 20 seconds [200 × F]
mean = warmup.mean(axis=0)            # [F] per-feature mean
std = warmup.std(axis=0) + 1e-6       # [F] per-feature std
normalized = (run_features - mean) / std
```

This per-run, per-feature z-score normalization is inspired by **instance normalization** in time-series [8], removing absolute-level shifts from ambient temperature, power mode, and workload variations. Unlike RevIN, we use a **causal warmup baseline** (first 20s only) rather than full-sequence statistics, ensuring the method is applicable online. When warmup is contaminated, see Section 6.3.

### 3.4 Fusion Methods

We evaluate multiple fusion strategies:

| Method | Formula | Notes |
|--------|---------|-------|
| dynamic_only | p = p_d | Baseline |
| constant_gate | p = g·p_d + (1-g)·p_s | g selected on val from {0.6, 0.7, 0.8, 0.9} |
| piecewise_gate | p = g(τ)·p_d + (1-g)·p_s | Heuristic: g=0.80 if p_s≥0.5, else 0.95 |
| hierarchical | p = p_s · p_d | Multiplicative |
| shuffle_static | (ablation) | Breaks binary-run correspondence |

**Training protocol**: Static ensemble and dynamic TCN are trained independently. For `constant_gate`, we select g on validation (g=0.8 performed best). `piecewise_gate` uses fixed heuristic thresholds. Only `late_fusion_learned` trains a logistic regression.

---

## 4. Experimental Setup

### 4.1 FusionBench-Sim Dataset

| Parameter | Value |
|-----------|-------|
| Binaries | 20 (14 trojan, 6 benign) |
| Runs | 200 (10 per binary) |
| Duration | 120 seconds per run |
| Sampling | 10 Hz (after resampling) |
| **Window length** | **20 seconds (200 samples)** |
| **Window stride** | **1 second (10 samples)** |
| **Overlap** | **95%** |
| **Label type** | Soft (continuous overlap fraction [0,1]) |
| **Label binarization** | Threshold 0.5 for window predictions |
| Positive rate | ~18% of windows |

**Split**: 120 train / 40 validation / 40 test runs. **Split integrity**: The split unit is `run_id`—no windows from the same run appear in both train and test.

**Causal windowing**: At each second t, we score the trailing window [t−20s, t], producing one decision per second after an initial 20s warmup. Windows are strictly causal (no future information).

**Label vs event truth**: Soft labels (overlap fraction) are used only for training the dynamic model. For evaluation, ground-truth events are constructed directly from simulator activation timestamps, and predicted events are formed by thresholding and merging contiguous positive windows.

### 4.2 Trojan Activation Model

Simulated trojans activate with deterministic periodic patterns:
- **Period**: 40 seconds (time between activation starts)
- **Active duration**: 10 seconds per activation
- **Styles**: `compute` (CPU spike), `mempressure` (memory thrashing), `cv` (GPU compute), `io` (I/O beacon)
- **Strength**: Multiplicative amplitude on injected telemetry deltas; `strength=1.0` corresponds to the simulator’s nominal trojan amplitude (e.g., +30°C temperature, +50% utilization)

Ground truth intervals are generated directly from activation timestamps (no jitter). Future work includes stochastic patterns and adversarial timing.

### 4.3 Evaluation Metrics

- **Event-F1**: F1 score on merged detection events (IoU ≥ 0.1)
- **FAR/h**: False alarm events per benign hour
- **TTD**: Time-to-detect (median, p90)
- **PR-AUC**: Precision-recall area under curve

**Evaluation Contract** (applies to all results unless noted):
- Event matching: greedy IoU ≥ 0.1, one-to-one
- FAR denominator: benign time only (excluding trojan-active intervals)
- Default threshold: 0.5 on predicted probability
- Warmup windows **included** in evaluation (20s warmup, then scoring begins)
- All metrics are **event-level** unless explicitly marked "window-level"

---

## 5. Results

### 5.1 Main Result: Per-Run Normalization

**Table 1: Standard vs Per-Run Normalization** (threshold=0.5, Event-F1, piecewise_gate fusion)

| Method | Standard F1 | Per-Run F1 | Δ F1 | Standard FAR | Per-Run FAR | Δ FAR |
|--------|-------------|------------|------|--------------|-------------|-------|
| dynamic_only | 0.551 | **0.599** | +8.7% | 50.6 | **30.0** | **-40.7%** |
| constant_gate | 0.614 | 0.599 | -2.4% | 52.4 | 23.4 | -55.3% |
| piecewise_gate | 0.577 | **0.624** | +8.1% | 65.4 | **24.2** | **-63.0%** |

**Key finding**: Per-run normalization reduces FAR by 41% while improving Event-F1 by 9%. This single preprocessing step is the most impactful intervention.

### 5.2 Warmup Sensitivity Analysis

We varied the warmup period from 5 to 40 seconds (using `piecewise_gate` fusion with per-run normalization):

| Warmup (s) | Steps | Event-F1 | Δ from 5s |
|------------|-------|----------|-----------|
| 5 | 50 | 0.408 | — |
| 10 | 100 | 0.556 | +36% |
| **20** | **200** | **0.626** | **+53%** |
| 40 | 400 | 0.627 | +54% |

**Finding**: Performance improves dramatically from 5→20 seconds (+53% Event-F1), then plateaus. **20 seconds is the optimal baseline period.**

### 5.3 Ablation Studies

**Table 2: Ablation Results (Per-Run Normalized)**

| Method | F1 | FAR/h | Interpretation |
|--------|-----|-------|----------------|
| dynamic_only | 0.599 | 30.0 | Baseline |
| constant_gate (g=0.8) | 0.599 | 23.4 | Simple mixing matches learned |
| piecewise_gate | **0.624** | **24.2** | Best overall |
| shuffle_static | 0.232 | **441.5** | 15× FAR explosion |
| remove_static_pathway | 0.599 | 30.0 | = dynamic (confirms static needed) |

**Key ablation**: `shuffle_static` destroys the correspondence between a run and its binary-level static score. The resulting FAR explosion (30 → 441/h, 15×) indicates that binary-specific static context is a meaningful conditioning signal for controlling false alarms.

### 5.4 FAR-Matched Evaluation

Operating at fixed FAR targets:

| FAR Target | Threshold | Actual FAR | Recall | Precision | TTD median |
|------------|-----------|------------|--------|-----------|------------|
| 5/h | 0.762 | 3.7 | 3.2% | 87.1% | 0.8s |
| 10/h | 0.733 | 12.1 | 8.9% | 85.4% | 0.7s |
| **20/h** | **0.683** | **20.4** | **14.2%** | **84.6%** | **0.7s** |
| 50/h | 0.554 | 50.1 | 19.9% | 75.8% | 0.6s |

**Interpretation**: At operationally realistic 20 FAR/h, we detect 14% of trojan-active windows with 85% precision. Precision remains high across all operating points.

**Note**: This table uses **window-level** recall/precision at validation-tuned thresholds, which differs from the Event-F1 metric in Tables 1-2 (event-level at threshold=0.5).

**Methodology**: For each FAR target, we select the detection threshold on the validation set to meet the FAR constraint, then report test recall/precision at that fixed threshold. Thresholds are not tuned on test data.

### 5.5 LLM Function Localization (Pilot Study, N=3)

**Table 3: LLM Localization Results (N=3 binaries, one ground-truth trojan function per binary)**

| Binary | Best Rank | Top-3 Hit | N Functions | Expected Random Rank |
|--------|-----------|-----------|-------------|----------------------|
| trojan_1 | 3 | ✓ | 30 | 15.5 |
| trojan_2 | 5 | ✗ | 28 | 14.5 |
| trojan_3 | 2 | ✓ | 35 | 18.0 |

**Aggregate Metrics**:
- Mean rank: 3.3 (vs 16.0 random expected across varying function counts)
- Mean Reciprocal Rank (MRR): 0.40
- Top-3 hit rate: 67%
- **Improvement: 4.8× better than random**

**Work saved**: Using LLM-guided triage, an analyst inspects 3.3 functions on average to find the backdoor, vs 16.0 with random inspection—a **79% reduction in inspection effort**.

---

## 6. Discussion

### 6.1 Why Per-Run Normalization Works

Per-run normalization addresses **domain shift** between training and deployment:
- Different ambient temperatures
- Different power modes (30W vs MAXN)
- Different baseline workloads

By computing z-scores relative to each run's warmup period, we make the model invariant to these shifts.

### 6.2 The Role of Static Analysis

Static analysis provides:
1. **FAR control**: Suppresses false alarms on known-benign binaries
2. **Operating-point flexibility**: Allows trading off recall vs FAR
3. **Localization**: LLM identifies specific suspicious functions

Static cannot distinguish active vs inactive trojans within a trojan binary—this requires dynamic monitoring.

### 6.3 Cold-Start Considerations

We tested per-run normalization with **contaminated warmup** (trojan active during baseline period, steps 50-150 of warmup):

| Condition | F1 | Precision | Recall | PR-AUC |
|-----------|-----|-----------|--------|--------|
| Clean warmup | 0.320 | 0.199 | 0.826 | 0.312 |
| Contaminated | 0.408 | 0.272 | 0.819 | 0.501 |

**Surprising result**: Contaminated runs show +27% F1. This is **not** evidence of robustness—rather, contamination injects more positive signal into the run, making detection easier. The higher PR-AUC (0.50 vs 0.31) confirms better class separation due to increased positive rate, not normalization robustness.

**Implication**: Per-run normalization does not degrade catastrophically under warmup contamination, but our evaluation conflates "more positives" with "better detection." To isolate the cold-start effect, future work will compute metrics **excluding warmup windows** and match post-warmup positive rates between clean and contaminated runs.

### 6.4 Limitations

1. **Simulator-only evaluation**: Real hardware validation is future work
2. **Deterministic activation timing**: Trojans activate at fixed periodic intervals; randomized timing is a validity threat (see Future Work)
3. **Binary-shared splits**: Test runs may share binaries with training; binary-disjoint generalization untested
4. **Single platform**: Results specific to Jetson-class devices
5. **Alert-only**: Response actions (kill, quarantine) not implemented

---

## 7. Related Work

**Firmware Trojan Detection**: Prior work on hardware trojans focuses on gate-level detection during manufacturing [1,2]. We address firmware-level trojans (malicious software payloads targeting hardware behavior) at runtime.

**Anomaly Detection for Security**: Host-based intrusion detection systems monitor system calls [3]. We monitor lower-level thermal/power signals resistant to userspace tampering.

**Sensor-Based Side Channels**: Power analysis for crypto extraction [4] and cache timing attacks [5] exploit similar signals. We defend against, rather than exploit, information leakage.

**LLM for Security**: Recent work applies LLMs to vulnerability detection [6,7]. We extend this to function-level trojan localization.

---

## 8. Conclusion

We presented HSDSF, a hybrid static-dynamic fusion system for detecting hardware trojan activation on edge devices. Our key findings:

1. **Per-run baseline normalization** is the single most effective intervention, reducing FAR by 41% and improving F1 by 9%.

2. **Warmup period of 20 seconds** is optimal, providing 53% F1 improvement over 5 seconds.

3. **LLM-based localization** achieves 4.6× better function ranking than random, enabling efficient analyst triage.

4. **Simple fusion suffices**: A constant mixture (g=0.8) matches learned gating, confirming that static contributes global score shifting for FAR control rather than per-window routing.

**Future work** includes real hardware validation, response action integration, and extension to other embedded platforms.

---

## Acknowledgments

[To be added]

---

## References

[1] Tehranipoor, M., and Koushanfar, F. "A Survey of Hardware Trojan Taxonomy and Detection." IEEE Design & Test, 2010.

[2] Bhunia, S., et al. "Hardware Trojan Attacks: Threat Analysis and Countermeasures." Proceedings of the IEEE, 2014.

[3] Sommer, R., and Paxson, V. "Outside the Closed World: On Using Machine Learning for Network Intrusion Detection." IEEE S&P, 2010.

[4] Kocher, P., et al. "Differential Power Analysis." CRYPTO, 1999.

[5] Liu, F., et al. "Last-Level Cache Side-Channel Attacks are Practical." IEEE S&P, 2015.

[6] Chen, M., et al. "Evaluating Large Language Models for Program Repair." ICSE, 2023.

[7] Pearce, H., et al. "Examining Zero-Shot Vulnerability Repair with Large Language Models." IEEE S&P, 2023.

[8] Kim, T., et al. "Reversible Instance Normalization for Accurate Time-Series Forecasting against Distribution Shift." ICLR, 2021.

[9] Guo, C., et al. "On Calibration of Modern Neural Networks." ICML, 2017.

[10] Lin, T., et al. "Focal Loss for Dense Object Detection." ICCV, 2017.

[11] Bai, S., et al. "An Empirical Evaluation of Generic Convolutional and Recurrent Networks for Sequence Modeling." arXiv:1803.01271, 2018.

---

## Appendix A: Evaluation Methodology

### A.1 Event Construction

We convert per-window predictions into events by:
1. Thresholding: ŷ_t = 𝟙[p_t ≥ θ]
2. Merging: Adjacent positive windows within one window length are merged
3. Matching: Greedy IoU matching (threshold 0.1) between predicted and true events

### A.2 False Alarm Rate

```
FAR/h = FP_events / (T_benign / 3600)
```

where T_benign is total time minus true trojan-active time.

### A.3 Time-to-Detect

For each true event, TTD is the delay from true event start to first overlapping prediction start.

---

## Appendix B: LLM Prompting Details

**Model**: Cerebras zai-glm-4.6  
**Temperature**: 0.7  
**Max tokens**: 4096

**Prompt structure**:
```
Analyze this function for potential security issues:
[FUNCTION CODE]

Respond in JSON format:
{
  "name": "function_name",
  "risk_score": 0.0-1.0,
  "categories": ["backdoor", "anti-analysis", ...],
  "findings": ["description of suspicious behavior", ...]
}
```

**Input format**: Decompiled C pseudocode from Ghidra (function names and objdump headers included).

**Risk categories**:
- `backdoor`: Hardcoded credentials, authentication bypass logic
- `anti-analysis`: Debug checks, anti-tampering, TLS/encryption usage
- `privilege_escalation`: Syscall invocations, capability manipulation
- `data_exfiltration`: Network I/O, file operations on sensitive paths

**Parse success rate**: >95%

---

## Appendix C: Model Architectures

### C.1 Static Ensemble
- 5 Random Forest classifiers
- 100 trees each
- Max depth: 10
- Temperature calibration: T = 1.58

### C.2 Dynamic TCN
- Input: [batch, 200, 55]
- 3 residual blocks
- Kernel size: 7
- Hidden dim: 64
- Dropout: 0.2
- Focal loss: α=0.25, γ=2
- Temperature calibration: T = 0.44

### C.3 Fusion Combiner
- Input: [p_s, p_d, u_s, u_d]
- 2-layer MLP (16, 8, 1)
- Sigmoid output
- Trained end-to-end with dynamic

---

## Appendix D: Telemetry Features

The dynamic expert uses 55 features extracted from Jetson telemetry:

**Raw Sensors (49 features)**:
- Temperature: 6 thermal zones (CPU, GPU, board zones)
- Power: 4 rails (VDD_IN, VDD_SYS, VDD_GPU, VDD_CPU) × current/voltage/power = 12
- Utilization: CPU (6 cores), GPU (1), memory (1) = 8
- Frequency: CPU, GPU, EMC = 3
- Performance counters: instructions, cycles, cache misses, branch misses, etc. = 16
- Fan speed, clock masks = 4

**Derived Features (6 features)**:
- IPC (instructions per cycle)
- Cache miss rate
- GPU power ratio (p_gpu / p_sys)
- CPU power ratio (p_cpu / p_sys)
- CPU/GPU frequency ratio

---

## Appendix E: Runtime Overhead (Estimated)

| Component | Value |
|-----------|-------|
| TCN parameters | ~50K |
| Inference latency | <10ms per window (estimated) |
| Telemetry collection | <1% CPU overhead |
| Window processing | 1s stride = 1 inference/second |
| End-to-end detection latency | 20s warmup + window ≈ 20s |

*Note: Inference measured on cloud GPU as reference; Jetson measurements are future work. After initial warmup, the system emits one score per second.*
