# Project Plan: Hybrid Static–Dynamic Security Framework on NVIDIA Jetson AGX Xavier

## Project status (simulator-first UGF upgrade)

- Legacy PoC phases have been moved under `archive/`.
- The current end-to-end target plan is in `New_plan.md` (UGF / FusionBench-Sim).
- New scaffolded modules live at repo root: `simulator/`, `data/`, `static/`, `dynamic/`, `fusion/`, `evaluation/`, `experiments/`, `docs/`, `paper/`.

Quick start (generate one deterministic simulator run + validate schema):

```bash
".venv/bin/python" experiments/sim_generate_runs.py --run-id run_000001 --binary-id sim_binary_0001 --duration-s 10
".venv/bin/python" -m unittest simulator.tests.test_schema_validate
```

## 1. Objective

Develop a minimum viable **hybrid static–dynamic security framework** to detect hardware/firmware supply‑chain anomalies on an NVIDIA Jetson AGX Xavier–based edge device.

The PoC will:

1. **Static (off-device)**  
   Use large language models (LLMs) to analyze firmware-style binaries (disassembly, strings, metadata) and flag suspicious logic patterns / potential backdoors.

2. **Dynamic (on-device)**  
   Use Jetson’s on-chip **performance monitoring counters (PMCs)** and **power/thermal telemetry** to profile “known-good” vs “Trojan-like” runtime behavior, and run an anomaly detection model in **real time** on Jetson using **TensorRT (GPU-accelerated)**.

Total estimated effort: **~52 engineering hours** (range 50–55 hours depending on tooling friction).

---

## 2. Assumptions

- Hardware: 1× NVIDIA Jetson AGX Xavier dev kit (with JetPack), 1× Ubuntu laptop (host).
- LLM access: Internet LLM API (OpenAI or similar).
- Scope intentionally **excludes** external power meters / logic analyzers and production hardening.

---

## 3. High-Level Deliverables

1. **Static Analysis Component**
   - Python toolchain to:
     - Disassemble and extract metadata from binaries.
     - Chunk and feed context to an LLM.
     - Aggregate LLM responses into a human-readable security report.
   - Demonstration on:
     - A **clean** synthetic firmware-style binary.
     - A **Trojan-modified** version with a known backdoor.

2. **Dynamic Telemetry & Dataset**
   - Telemetry collector on Jetson aggregating:
     - CPU PMCs (via `perf` or equivalent).
     - `tegrastats` (GPU utilization, memory, etc.).
     - Thermal and power readings exposed via `/sys`.
   - Labeled datasets for:
     - Idle system.
     - Normal workload.
     - Trojan workload.

3. **On-Device Anomaly Detection**
   - Trained **time-series anomaly detection model** (e.g., 1D CNN or LSTM autoencoder).
   - Model exported to ONNX and compiled to a **TensorRT engine**.
   - Real-time monitoring script on Jetson:
     - Consumes live telemetry.
     - Maintains sliding windows.
     - Computes anomaly scores and raises alerts.

4. **Final Report**
   - Architecture overview.
   - Experimental setup and results.
   - Example detection runs (normal vs Trojan).
   - Limitations and next steps.

---

## 4. Work Breakdown & Timeline (Hour-Wise)

Estimated total: **~52 hours**.

### Phase 1 – Project Setup & Threat Model (≈6 hours)

**Goal:** Establish clear scope, threat model, and set up both devices.

| Task | Description | Est. Hours | Outcomes |
|------|-------------|------------|----------|
| 1.1 | Define PoC scope & threat model | 2 h | 1–2 page spec: assets to analyze (binaries), definition of “Trojan-like” behavior, success criteria (e.g., detection of synthetic backdoor + runtime anomalies). |
| 1.2 | Environment setup: Jetson & laptop | 3 h | Jetson with JetPack, `perf`, `tegrastats`, Python deps (PyTorch, TensorRT tools). Laptop with `objdump`, `strings`, LLM API client. |
| 1.3 | Quick ramp-up: Jetson + perf basics | 1 h | Hands-on familiarity with `tegrastats`, basic `perf stat` usage, confirming PMCs are accessible. |

**Phase 1 Deliverable:**  
Short written threat model + fully prepared dev environment.

---

### Phase 2 – Static LLM-Based Firmware/Binary Analysis (≈12 hours)

**Goal:** Pipeline that uses an LLM to flag suspicious logic in firmware-like binaries.

| Task | Description | Est. Hours | Outcomes |
|------|-------------|------------|----------|
| 2.1 | Collect & prepare binaries | 2 h | Folder with: 1–2 Jetson system binaries + 1 synthetic firmware-like binary; disassembly (`objdump -d`), `strings`, symbol tables (`nm`, `readelf`). |
| 2.2 | Create clean vs Trojan firmware-like binaries | 4 h | `firmware_clean` and `firmware_trojan` ELF binaries, with known backdoor logic (e.g., master password, magic command). Ground-truth description of backdoor location. |
| 2.3 | Build LLM static analysis pipeline | 4 h | `analyze_binary.py`: generates disassembly/metadata, chunks text, sends to LLM with a structured prompt, and aggregates results into JSON/markdown report with suspicious segments and confidence scores. |
| 2.4 | Run static analysis & validate | 2 h | LLM-based reports for `firmware_clean` and `firmware_trojan` (and optionally 1 real system binary). Evidence that the Trojan segments are highlighted more strongly than the clean version. |

**Phase 2 Deliverable:**  
Reusable Python static-analysis tool + sample reports showing differentiation between clean vs backdoored binaries.

---

### Phase 3 – Dynamic Telemetry Collection & Dataset Creation (≈16 hours)

**Goal:** Collect rich, labeled runtime data for normal vs Trojan behavior.

| Task | Description | Est. Hours | Outcomes |
|------|-------------|------------|----------|
| 3.1 | Instrument PMCs & sensors | 4 h | `collect_telemetry.py`: periodically samples CPU PMCs (via `perf` or similar), `tegrastats`, and `/sys` thermal/power sensors into a unified, timestamped CSV/JSON format. |
| 3.2 | Implement known-good workloads | 3 h | `run_normal_inference.py`: e.g., TensorRT/PyTorch inference loop; optionally a simple control/IPC loop. Stable, repeatable “normal” workload scripts. |
| 3.3 | Implement Trojan workload | 4 h | `run_trojan_workload.py`: same base workload as normal but with hidden periodic extra compute (e.g., matrix multiplications, synthetic crypto-like loop). Triggered on a deterministic schedule for easy labeling. |
| 3.4 | Collect labeled telemetry datasets | 5 h | 30–60 minutes of data across 3 scenarios: idle, normal, Trojan. Each run saved with scenario labels and known Trojan activation intervals for supervised evaluation. |

**Phase 3 Deliverable:**  
Labeled telemetry dataset capturing PMCs + power/thermal behavior for idle, normal, and Trojan conditions.

---

### Phase 4 – Anomaly Model Training & On-Device Deployment (≈14 hours)

**Goal:** Train a small anomaly detection model and deploy it as a real-time TensorRT engine on Jetson.

| Task | Description | Est. Hours | Outcomes |
|------|-------------|------------|----------|
| 4.1 | Preprocess data & feature engineering | 3 h | Preprocessing script: loads telemetry logs, aligns/resamples at fixed interval, selects features, normalizes (stores mean/std), creates sliding windows (e.g., 5–10s). Produces `X_train`, `X_test`, `y_test`. |
| 4.2 | Train compact anomaly detection model | 4 h | PyTorch model (e.g., 1D CNN or LSTM autoencoder). Trained on normal windows only; evaluated via reconstruction error to distinguish Trojan vs normal. Basic metrics (ROC/AUC or separation of error distributions). |
| 4.3 | Convert to TensorRT & integrate | 5 h | Model exported to ONNX and compiled to TensorRT engine. `rt_monitor.py` on Jetson: ingests live telemetry, maintains sliding window, normalizes, runs inference via TensorRT, and logs anomaly scores/alerts. |
| 4.4 | End-to-end functional test & tuning | 2 h | Live tests in three modes: idle, normal, Trojan. Adjustment of anomaly thresholds and possibly window size to reduce false positives and catch Trojan behavior reliably. |

**Phase 4 Deliverable:**  
On-device real-time anomaly detection service, GPU-accelerated via TensorRT, with demonstrated detection of Trojan-like runtime behavior.

---

### Phase 5 – Documentation & Final Reporting (≈4 hours)

**Goal:** Produce a concise, shareable report and summarize results.

| Task | Description | Est. Hours | Outcomes |
|------|-------------|------------|----------|
| 5.1 | Document architecture & results | 3 h | Internal report / slide deck summarizing architecture, data flows, model design, static and dynamic results (including confusion matrix or basic metrics), and screenshots/log excerpts. |
| 5.2 | Review & polish | 1 h | Incorporate feedback, clarify limitations, and outline next steps (e.g., richer Trojans, more features, extended testbed). |

**Phase 5 Deliverable:**  
Final project report suitable for internal review and planning of next-stage work.

---

## 5. Overall Timeline & Resource Summary

- **Total estimated effort:** ~**52 hours** (realistic range 50–55 hours).
- If allocated **~20 hours/week** of engineer time:
  - Expected duration: **about 2.5–3 weeks**.
- Critical path:
  - Phases 1 → 2 → 3 → 4 are sequential.
  - Phase 5 can overlap slightly with late Phase 4 (documentation as results come in).

---

## 6. What This PoC Demonstrates

By the end of this plan, you will have:

1. **Static-Analysis Capability**
   - An LLM-assisted tool that can highlight suspicious logic in firmware-like binaries based on disassembly and metadata.
   - Demonstrated ability to identify a known, synthetic backdoor.

2. **Dynamic Behavioral Fingerprinting**
   - A telemetry pipeline leveraging Jetson’s PMCs and thermal/power signals to characterize runtime behavior.
   - Labeled datasets of normal vs Trojan-like operation.

3. **On-Device Real-Time Detection**
   - A compact anomaly detection model deployed with TensorRT on Jetson AGX Xavier.
   - Live detection of Trojan-like anomalies with measurable performance (false positives/negatives).

4. **Foundation for Further Work**
   - Clear architecture, source code, and data that can be extended to more realistic firmware, richer hardware events, or more advanced models.

