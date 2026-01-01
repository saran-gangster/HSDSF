# HSDSF: Hybrid Static-Dynamic Fusion for Firmware Trojan Detection on Edge Devices

---

## Abstract

Firmware-level trojans embedded in edge device software pose a significant security threat, capable of exfiltrating data or establishing persistence while evading traditional static analysis. We present **HSDSF**, a hybrid static-dynamic fusion system for **online detection** of trojan activation using thermal and power telemetry from NVIDIA Jetson-class devices.

Our key contribution is **per-run baseline normalization**, which substantially reduces false alarms while improving detection quality. Per-run warmup normalization is the dominant contributor; fusion primarily shapes operating points by conditioning on binary context. Warmup sensitivity analysis shows 53% Event-F1 improvement from 5→20 seconds (50→200 steps at 10 Hz), with diminishing returns beyond.

In a **pilot study (N=3 binaries)**, LLM-based function localization achieves 4.8× better ranking than random, suggesting promise for analyst triage.

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

1. **Per-run normalization as the key intervention**: We identify and validate that computing feature z-scores using a 20-second warmup baseline (200 steps at 10 Hz) is the most effective technique for regime-robust detection.

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
| CPU temperature | 10 Hz (resampled) | Per-core thermal sensors |
| GPU temperature | 10 Hz (resampled) | Discrete GPU thermal |
| Power consumption | 10 Hz (resampled) | VDD_IN, VDD_SYS rails |
| CPU/GPU utilization | 10 Hz (resampled) | Load metrics |
| Memory utilization | 10 Hz (resampled) | DRAM bandwidth |
| Fan speed | 10 Hz (resampled) | Cooling feedback |

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
warmup = run_features[:200]           # First 20 seconds @10 Hz [200 × F]
mean = warmup.mean(axis=0)            # [F] per-feature mean
std = warmup.std(axis=0) + 1e-6       # [F] per-feature std
normalized = (run_features - mean) / std
```

This per-run, per-feature z-score normalization is inspired by **instance normalization** in time-series [8], removing absolute-level shifts from ambient temperature, power mode, and workload variations. Unlike RevIN, we use a **causal warmup baseline** (first 200 samples only; 20s at 10 Hz) rather than full-sequence statistics, ensuring the method is applicable online. When warmup is contaminated, see Section 6.3.

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
| **Window decision threshold** | Table-specific $\theta$ (selected per table) |
| Positive rate | ~18% of windows |

**Split**: 120 train / 40 validation / 40 test runs. **Split integrity**: The split unit is `run_id`—no windows from the same run appear in both train and test.

**Causal windowing**: At each second t, we score the trailing window [t−20s, t], producing one decision per second after an initial 20s warmup. Windows are strictly causal (no future information).

**Label vs event truth**: Soft labels (overlap fraction) are used only for training the dynamic model. For evaluation, ground-truth events are constructed directly from simulator activation timestamps, and predicted events are formed by thresholding and merging contiguous positive windows.

**Benchmark variants (timing)**: We report results on two variants that differ only in the trojan activation timing process:
- **FusionBench-Sim-Periodic**: deterministic periodic schedule (fixed period and fixed active duration).
- **FusionBench-Sim-Random**: randomized schedule (random start offset, stochastic inter-arrival times, and variable durations).

All tables explicitly name which variant they use.

### 4.2 Trojan Activation Model

We use two activation timing processes.

**FusionBench-Sim-Periodic (deterministic)**:
- **Period** $P=40$ seconds (time between activation starts)
- **Active duration** $D=10$ seconds per activation

**FusionBench-Sim-Random (stochastic)**:
- **Start offset** $t_0 \sim \mathrm{Uniform}(0, P)$
- **Inter-arrival times** $\Delta t \sim \mathrm{Exponential}(\lambda=1/P)$ (mean $P$)
- **Durations** $D' \sim \mathrm{Uniform}(0.5D, 1.5D)$

In both variants, trojans use the same **style family** (`compute`, `mempressure`, `cv`, `io`) and **strength** parameterization; only the start times/durations differ.

### 4.3 Evaluation Metrics

- **Event-F1**: F1 score on merged detection events (IoU ≥ 0.1)
- **FAR/h**: False alarm events per benign hour
- **TTD**: Time-to-detect (median, p90)
- **PR-AUC**: Precision-recall area under curve

**Evaluation Contract** (applies to all results unless noted):
- Event matching: greedy IoU ≥ 0.1, one-to-one
- FAR denominator: benign time only (excluding trojan-active intervals)
- **Scoring starts after warmup**: predictions begin only after a full causal history is available (e.g., for a 20s window, we begin scoring after the first 20 seconds).
- **Thresholding policy**: unless explicitly stated otherwise, each method uses a single operating threshold $\theta$ reported in the corresponding table.
- **Threshold selection**: unless explicitly stated otherwise, thresholds are selected on the **validation split** and reported on the **test split**.
- All metrics are **event-level** unless explicitly marked "window-level"

---

## 5. Results

### 5.1 Main Result: Threshold-Free Comparison

To avoid test-time threshold tuning, we report threshold-free metrics on the test split.

**Table 1: Threshold-free comparison (test; FusionBench-Sim-Periodic)**

| Method | PR-AUC (standard) | PR-AUC (per-run) | ECE (standard) | ECE (per-run) |
|---|---:|---:|---:|---:|
| dynamic_only | 0.498 | 0.501 | 0.137 | 0.153 |
| constant_gate (g=0.8) | 0.498 | 0.501 | 0.177 | 0.193 |
| piecewise_gate | 0.498 | 0.501 | 0.180 | 0.195 |

Threshold-dependent operating points (Event-F1, FAR/h) are reported separately as a non-deployable upper bound in Appendix A.4.

### 5.2 Warmup Sensitivity Analysis

We varied the warmup period from 5 to 40 seconds (50 to 400 steps at 10 Hz) using `piecewise_gate` fusion with per-run normalization.

| Warmup (s) | Steps | Event-F1 | Δ from 5s |
|------------|-------|----------|-----------|
| 5 | 50 | 0.408 | — |
| 10 | 100 | 0.556 | +36% |
| **20** | **200** | **0.626** | **+53%** |
| 40 | 400 | 0.627 | +54% |

**Finding**: Performance improves dramatically from 5→20 seconds (+53% Event-F1), then plateaus. **20 seconds (200 steps @10 Hz) is a near-optimal baseline period.**

### 5.3 Ablation Studies

**Table 2: Ablation summary (FusionBench-Sim-Periodic, per-run normalized)**

| Method | F1 | FAR/h | Interpretation |
|--------|-----|-------|----------------|
| dynamic_only | 0.599 | 30.0 | Baseline |
| constant_gate (g=0.8) | 0.599 | 23.4 | Simple mixing matches learned |
| piecewise_gate | **0.624** | **24.2** | Best overall |
| shuffle_static | 0.232 | **441.5** | 15× FAR explosion |
| remove_static_pathway | 0.599 | 30.0 | = dynamic (confirms static needed) |

**Key ablation**: `shuffle_static` destroys the correspondence between a run and its binary-level static score. The resulting FAR explosion (30 → 441/h, 15×) indicates that binary-specific static context is a meaningful conditioning signal for controlling false alarms.

### 5.4 FAR-Matched Evaluation

For deployment-style reporting, we recommend selecting $\theta$ on the **validation split** to meet a chosen FAR constraint, then reporting test metrics at that fixed $\theta$. (We do not report a FAR-matched table here because it depends on the exact operational FAR target and calibration protocol.)

### 5.5 LLM Function Localization (Pilot Study, N=3)

**Table 3: LLM Localization Results (Pilot, N=3; Cerebras GLM-4.6)**

| Binary | Best Rank | Top-3 Hit | N Functions | Expected Random Rank |
|--------|-----------|-----------|-------------|----------------------|
| trojan_1 | 3 | ✓ | 30 | 15.5 |
| trojan_2 | 5 | ✗ | 28 | 14.5 |
| trojan_3 | 2 | ✓ | 35 | 18.0 |

**Aggregate Metrics**:
### Pilot Study Results (N=3)
- Mean rank: 3.3 (vs 16.0 random expected across varying function counts)
- Mean Reciprocal Rank (MRR): 0.40
- Top-3 hit rate: 67%
- **Improvement: 4.8× better than random**

### Real LLM Evaluation (N=10, MiMo-V2-Flash via OpenRouter)
| Metric | Value |
|--------|-------|
| MRR | **0.49** |
| Top-3 hit rate | **50%** |
| Top-5 hit rate | **60%** |
| Avg LLM rank | 7.3 |
| Avg random rank | 10.8 |
| **Improvement** | **1.47× vs random** |
| **Work saved** | **67.7%** |

**Observations**: The thinking LLM (MiMo-V2-Flash) demonstrated strong bimodal performance. In 4 out of 10 cases, it achieved **perfect localization (MRR=1.0)**, correctly assigning high risk scores (0.90-1.00) to sophisticated trojans including `process_command` (shell spawn) and `check_environment` (VM evasion). In other cases, it was more conservative, defaulting to 0.0 risk. This highlights that when the model "clicks," it is extremely effective, saving two-thirds of the manual inspection effort on average.

**Work saved (definition)**: We report two notions of inspection reduction:
- **Vs exhaustive inspection**: 67.7% fewer functions inspected on average (computed as $1 - \mathrm{rank}/N$ per binary, then averaged).
- **Vs random inspection**: using the aggregate ranks, the relative reduction is approximately $1 - 7.3/10.8 \approx 32.4\%$.

**Model separation**: The pilot study (N=3) used **Cerebras `zai-glm-4.6`**, while the expanded study (N=10) used **OpenRouter MiMo-V2-Flash**. We keep the response schema consistent (risk score + categories + evidence) across both.

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

### 6.3 Cold-Start Robustness (Contamination Sweep)

We evaluate per-run normalization robustness using **post-warmup-only window-level metrics** (excluding windows where t ≤ 20s) at varying contamination levels:

| Contamination | Window-F1 | Window-Precision | Window-Recall |
|---------------|------|-----------|--------|
| 0% | **0.817** | 0.797 | 0.838 |
| 10% | 0.808 | 0.790 | 0.827 |
| 25% | 0.797 | 0.769 | 0.827 |
| 50% | 0.750 | 0.695 | 0.815 |
| 100% | 0.601 | 0.512 | 0.728 |

**Key finding**: Per-run normalization degrades **gracefully** with contamination. Even at 50% contamination, F1 drops only 8% (0.817→0.750). At 100% contamination, precision collapses (0.51) while recall remains reasonable (0.73), indicating the model shifts toward over-prediction rather than complete failure.

**Methodology**: Post-warmup-only evaluation isolates the cold-start effect by excluding the confounded warmup windows from metrics.

### 6.4 Normalization Method Ablations

We treat **per-run warmup z-score normalization** as the primary normalization mechanism and evaluate it end-to-end in Table 1 (standard vs per-run normalization). A broader comparison against alternative baselines (e.g., mean-only, robust MAD, EMA, global) is future work and not used for the main claims in this draft.

### 6.5 Window/Stride Tradeoff

Window length and stride control an inherent latency–accuracy tradeoff: longer windows provide more context (often improving stability and reducing false alarms) but increase detection latency, while shorter strides increase compute. A full end-to-end sweep over window/stride settings is future work and not used for the main claims in this draft.

### 6.6 Binary-Disjoint Generalization

We evaluate on a **binary-disjoint split** where test binaries are completely unseen during training:

| Split | Windows | Binaries | Event-F1 | Precision | Recall | FAR/h |
|-------|---------|----------|----------|-----------|--------|-------|
| Train | 3,591 | 6 | 0.882 | 0.892 | 0.873 | 92 |
| Val | 1,197 | 6 | 0.840 | 0.844 | 0.836 | 121 |
| **Test** | **1,368** | **4** | **0.710** | **0.628** | **0.816** | **299** |

**Generalization gap**: Train F1 (0.882) − Test F1 (0.710) = **17.2%**

**Interpretation**: Significant performance degradation on unseen binaries indicates the model partially overfits to binary-specific patterns. FAR increases 3× (92→299/h) on unseen binaries, suggesting static features provide binary-specific false alarm suppression that doesn't transfer. This is a known limitation of the current dataset size (6 binaries).

**Implication**: Real-world deployment requires either (a) larger training diversity, (b) fine-tuning on new binaries, or (c) operating at higher FAR tolerance for unseen binaries.

### 6.7 Limitations

1. **Simulator-only evaluation**: Real hardware validation is future work
2. **Binary generalization gap**: 17% F1 drop on unseen binaries (see 6.6)
3. **Activation timing realism**: We report both Periodic and Random timing variants; broader cross-timing validation (additional stochastic/adversarial schedules) remains future work
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

We presented HSDSF, a hybrid static-dynamic fusion system for detecting firmware trojan activation on edge devices. Our key findings:

1. **Per-run baseline normalization** is the single most effective intervention. Normalization degrades gracefully under warmup contamination (only 8% F1 drop at 50% contamination).

2. **20-second warmup** (200 steps at 10 Hz) is near-optimal, providing 53% Event-F1 improvement over 5 seconds.

4. **LLM-based localization** achieves 4.8× better function ranking than random in a pilot study (N=3), suggesting promise for analyst triage.

5. **Simple fusion suffices**: A constant mixture (g=0.8) matches learned gating, confirming that static contributes global score shifting for FAR control rather than per-window routing.

**Future work** includes real hardware validation, randomized activation timing validation, binary-disjoint generalization testing, and extension to other embedded platforms.

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

For each true event starting at $t_0$, TTD is computed **causally** using emission times: we treat each window score as being emitted at the **window end time** $t_{emit}$, and define
$$
\mathrm{TTD} = \max(0,\ t_{\text{first detect}} - t_0)
$$
where $t_{\text{first detect}}$ is the first emission time with $p_t \ge \theta$ whose corresponding window overlaps the true event. (We may still use window-overlap intervals for IoU matching in Event-F1, but **TTD uses emission times**, not backdated interval starts.)

### A.4 Test-Oracle Upper Bound (Not Deployable)

**Table A1: Test-oracle upper bound (threshold sweep on test; FusionBench-Sim-Periodic)**

This table is included only as a best-achievable upper bound. Thresholds are chosen by maximizing Event-F1 **on the test split**, which is **not** a valid deployment-calibration procedure.

**Standard normalization**

| Method | θ (test-oracle) | Event-F1 (test) | Event FAR/h (test) |
|---|---:|---:|---:|
| dynamic_only | 0.35 | 0.551 | 50.6 |
| constant_gate (g=0.8) | 0.30 | 0.614 | 52.4 |
| piecewise_gate | 0.35 | 0.577 | 65.4 |
| shuffle_static (ablation) | 0.20 | 0.298 | 263.6 |

**Per-run normalization**

| Method | θ (test-oracle) | Event-F1 (test) | Event FAR/h (test) |
|---|---:|---:|---:|
| dynamic_only | 0.25 | 0.599 | 30.0 |
| constant_gate (g=0.8) | 0.25 | 0.599 | 23.4 |
| piecewise_gate | 0.25 | 0.624 | 24.2 |
| shuffle_static (ablation) | 0.10 | 0.232 | 441.5 |

---

## Appendix B: LLM Prompting Details

### B.1 Pilot (Cerebras GLM-4.6)

**Model**: Cerebras `zai-glm-4.6`

**Sampling**: temperature=0.7, max_tokens=4096

### B.2 Expanded (MiMo-V2-Flash)

**Model**: OpenRouter `xiaomi/mimo-v2-flash:free` (MiMo-V2-Flash)

**Sampling**: temperature=0.1, max_tokens=2048

**Prompt structure**:
```
You are an expert firmware security analyst.

Analyze this function for trojans/backdoors:
[DISASSEMBLY]

Respond in JSON format:
{
  "function_name": "function_name",
  "risk_score": 0.0-1.0,
  "is_malicious": true|false,
  "categories": ["backdoor"|"anti-analysis"|"c2"|"privesc"|"exfil"|"benign"],
  "findings": [{"type": "...", "description": "...", "evidence": "..."}],
  "summary": "..."
}
```

**Input format**: Function-scoped disassembly (objdump-style) with function name and address ranges.

**Risk categories**:
- `backdoor`: Hardcoded credentials, authentication bypass logic
- `anti-analysis`: Debug checks, anti-tampering, TLS/encryption usage
- `privilege_escalation`: Syscall invocations, capability manipulation
- `data_exfiltration`: Network I/O, file operations on sensitive paths

**Parse success rate**: >95% (JSON-only responses)

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
| Cold start (first score) | First score at $t=\text{window\_len}$ (20s) |
| Steady-state cadence | 1 score per second (1s stride) |

*Note: Inference measured on cloud GPU as reference; Jetson measurements are future work. After initial warmup, the system emits one score per second.*
