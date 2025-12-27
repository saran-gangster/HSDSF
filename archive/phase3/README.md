# Phase 3: Telemetry + Workloads + Experiment Harness

This phase adds a runnable data-collection stack for supply-chain anomaly detection on Jetson-class devices. It mirrors the plan in planforphase3.md and is structured so most pieces can be exercised on a non-Jetson host (perf-only, fake tegrastats) before moving to hardware.

## Layout

- configs/perf_events.yaml: default perf stat event list; override at runtime if needed.
- data/runs/: per-run outputs (telemetry.csv, trojan_intervals.csv, meta.json).
- scripts/collect_telemetry.py: unified collector for perf + tegrastats + /sys sensors with zero-order hold alignment.
- scripts/perf_reader.py, scripts/tegrastats_reader.py, scripts/sys_sensors.py: parsers and helpers.
- scripts/run_normal_workload.py: baseline inference-like CPU loop.
- scripts/run_trojan_workload.py: compute/memory/IO trojans with interval logging and optional baseline overlay.
- scripts/run_experiment.py: orchestrator that spins up the collector and launches a scenario.
- scripts/setup_jetson.sh: automated setup for Jetson hardware.
- scripts/setup_linux.sh: automated setup for generic Linux systems.
- scripts/run_tests.sh: run smoke tests to validate installation.
- scripts/collect_dataset.sh: automated full dataset collection protocol.
- scripts/fake_tegrastats.sh: simulated tegrastats for non-Jetson testing.

## Automated Setup and Execution

### Jetson Setup (Recommended)

1) Run the automated setup script:
```bash
cd phase3/scripts
bash setup_jetson.sh
```

This will:
- Install Python dependencies
- Install perf tools
- Verify tegrastats availability
- Optionally configure performance modes (nvpmodel, jetson_clocks)

2) Run smoke tests:
```bash
bash run_tests.sh
```

3) Collect full dataset (90-120 minutes):
```bash
bash collect_dataset.sh
```

### Linux Setup (Non-Jetson)

1) Run the Linux setup script:
```bash
cd phase3/scripts
bash setup_linux.sh
```

This will:
- Install Python dependencies
- Install perf tools
- Create fake_tegrastats.sh for testing

2) Run smoke tests with fake tegrastats:
```bash
bash run_tests.sh --fake-tegrastats
```

Or without tegrastats:
```bash
bash run_tests.sh --no-tegrastats
```

### Windows (Limited Support)

On Windows, perf/tegrastats are unavailable. Use WSL (Windows Subsystem for Linux) and follow the Linux setup above, or manually run experiments with both --no-perf and --no-tegrastats flags.

## Manual Quickstart

### Host without Jetson

1) Install deps:
```bash
pip install -r phase3/requirements.txt
```

2) Run a short smoke test (perf only, skip tegrastats):
```bash
python phase3/scripts/run_experiment.py --label idle --duration 30 --no-tegrastats
python phase3/scripts/run_experiment.py --label normal --duration 30 --no-tegrastats
python phase3/scripts/run_experiment.py --label trojan_compute --trojan-variant compute --duration 30 --no-tegrastats
```

### Jetson target

1) Setup:
```bash
sudo apt-get update
sudo apt-get install python3-pip linux-tools-common linux-tools-generic linux-tools-$(uname -r)
pip3 install -r phase3/requirements.txt
sudo tegrastats --interval 1000 --count 1
sudo nvpmodel -m 0
sudo jetson_clocks
```

2) Run experiments:
```bash
python3 phase3/scripts/run_experiment.py --label idle --duration 60
python3 phase3/scripts/run_experiment.py --label normal --duration 60
python3 phase3/scripts/run_experiment.py --label trojan_compute --trojan-variant compute --duration 60
```

Outputs land under phase3/data/runs/<timestamp_label>/.

## Dataset Collection Protocol (from plan)

The collect_dataset.sh script automates this protocol:

- Idle: 3 × 10 minutes
- Normal: 3 × 10 minutes
- Trojans (solo): 2 × 10 minutes each of compute, memory, io
- Optional: overlay variants with --trojan-overlay

Total: 11 runs, ~90-120 minutes

Run with: `bash phase3/scripts/collect_dataset.sh`

Options:
- `--duration SECONDS`: Override default 600s per run
- `--no-tegrastats`: Skip tegrastats (for non-Jetson)
- `--fake-tegrastats`: Use simulated tegrastats
- `--save-raw`: Save raw perf/tegrastats streams
- `--dry-run`: Preview commands without execution

## Notes

- Sampling: collector aligns all sources to dt (default 0.1 s) and holds last seen values for slower feeds.
- Derived metrics: perf_ipc, perf_branch_miss_rate, perf_cache_miss_rate are computed when inputs are available.
- Perf tuning: adjust events via --perf-events or configs/perf_events.yaml.
- Safety: raw perf/tegrastats streams can be saved with --save-raw for debugging.
