# Phase 3: Jetson Deployment Guide

**Target Hardware:** NVIDIA Jetson AGX Xavier (or compatible Jetson platform)  
**Estimated Total Time:** 2–3 hours (including setup + full dataset collection)

This guide provides a step-by-step checklist for deploying Phase 3 on your Jetson device when hardware arrives.

---

## Prerequisites

- NVIDIA Jetson AGX Xavier running JetPack (L4T)
- Internet connection for package installation
- sudo/root access
- This repository copied to the Jetson device

---

## Step-by-Step Deployment Protocol

### 1. Transfer Repository to Jetson

```bash
# Option A: Clone from GitHub
git clone https://github.com/saran-gangster/HSDSF.git
cd HSDSF

# Option B: Transfer via scp from your dev machine
# (from your dev machine)
scp -r /path/to/HSDSF jetson@<jetson-ip>:~/
# (then on Jetson)
cd ~/HSDSF
```

---

### 2. Automated Setup (Recommended)

```bash
cd phase3/scripts
bash setup_jetson.sh
```

**What this does:**
- Updates package lists
- Installs Python 3 and pip (if not present)
- Installs perf tools (`linux-tools-common`, `linux-tools-generic`, kernel-specific tools)
- Installs Python dependencies from `phase3/requirements.txt` (numpy, pyyaml)
- Verifies `tegrastats` availability
- Verifies `perf` availability
- Checks perf event permissions
- Optionally configures Jetson for maximum performance (`nvpmodel -m 0`, `jetson_clocks`)

**Expected output:** Setup completion message with next steps.

---

### 3. Configure Performance Mode (Critical for Consistent Data)

If not done during setup, run manually:

```bash
# Set to MAXN mode (maximum performance)
sudo nvpmodel -m 0
sudo nvpmodel -q  # verify current mode

# Lock clocks to maximum
sudo jetson_clocks
sudo jetson_clocks --show  # verify clock settings
```

**Why:** Reduces thermal throttling and frequency scaling noise in telemetry data.

---

### 4. Configure Perf Permissions (if needed)

Check current permission level:

```bash
cat /proc/sys/kernel/perf_event_paranoid
```

If the value is > 1, lower it for non-root perf access:

```bash
echo 1 | sudo tee /proc/sys/kernel/perf_event_paranoid
```

**Note:** This setting resets on reboot. For permanent change, add to `/etc/sysctl.conf`:
```bash
echo "kernel.perf_event_paranoid = 1" | sudo tee -a /etc/sysctl.conf
```

---

### 5. Run Smoke Tests (Validation - 90 seconds)

```bash
cd phase3/scripts
bash run_tests.sh
```

**What this does:**
- Runs 3 test scenarios (idle, normal, trojan_compute) for 30 seconds each
- Validates that telemetry collection works
- Checks for expected data columns

**Expected output:**
```
==========================================
Phase 3 Smoke Tests
==========================================
Test duration: 30s per scenario

Test 1/3: Idle scenario...
Run completed: .../data/runs/YYYYMMDD_HHMMSS_idle

Test 2/3: Normal workload...
Run completed: .../data/runs/YYYYMMDD_HHMMSS_normal

Test 3/3: Trojan compute workload...
Run completed: .../data/runs/YYYYMMDD_HHMMSS_trojan_compute

==========================================
Smoke Tests Complete!
==========================================

Recent test outputs:
  - data/runs/.../
    Rows: ~280-300 (expected ~300)
    ✓ perf_cycles found
    ✓ perf_ipc found
    ✓ ts_gpu_util found
    ✓ ts_ram_used_mb found
    ✓ temp_* sensors found
    ✓ meta.json found
```

---

### 6. Validate Smoke Test Output (Critical)

Inspect the most recent test run:

```bash
cd phase3

# Check telemetry data structure
head -20 data/runs/$(ls -t data/runs | head -1)/telemetry.csv

# Check metadata
cat data/runs/$(ls -t data/runs | head -1)/meta.json
```

**Verify the following columns exist:**

**Perf metrics (performance counters):**
- `perf_cycles`
- `perf_instructions`
- `perf_ipc` (derived: instructions/cycles)
- `perf_branch_misses`
- `perf_branch_miss_rate` (derived)
- `perf_cache_misses`
- `perf_cache_miss_rate` (derived)

**tegrastats metrics (Jetson-specific):**
- `ts_gpu_util` (GPU utilization %)
- `ts_ram_used_mb` (RAM usage in MB)
- `ts_emc_util` (memory controller utilization %)
- `ts_cpu_util_*` (per-core CPU utilization)
- `ts_power_*` (power rail measurements if available)

**System sensors (/sys/class/thermal):**
- `temp_*` (thermal zone temperatures in Celsius)

**Metadata:**
- `ts_unix` (Unix timestamp)
- `t_wall` (wall-clock time from experiment start)
- `label` (scenario label)

**Expected data properties:**
- **Sampling rate:** ~10 rows/sec (due to `--dt 0.1`)
- **Row count:** ~280-300 rows for 30-second runs
- **Nonzero values:** `perf_cycles`, `perf_instructions` should be nonzero during active workloads
- **IPC range:** `perf_ipc` typically 0.1–5.0 depending on workload
- **Temperature increase:** Temps should rise slightly under load vs idle

---

### 7. Collect Full Dataset (90–120 minutes)

Once smoke tests pass, collect the complete dataset:

```bash
cd phase3/scripts
bash collect_dataset.sh
```

**What this does:**
- **Idle:** 3 runs × 10 minutes = 30 minutes
- **Normal:** 3 runs × 10 minutes = 30 minutes
- **Trojan Compute:** 2 runs × 10 minutes = 20 minutes
- **Trojan Memory:** 2 runs × 10 minutes = 20 minutes
- **Trojan I/O:** 2 runs × 10 minutes = 20 minutes

**Total:** 11 runs, ~120 minutes

**Expected output:**
```
==========================================
Phase 3 Full Dataset Collection
==========================================
Duration per run: 600s (10.0 min)
...

Run 1/11: idle
Start time: ...
Run completed: .../data/runs/YYYYMMDD_HHMMSS_idle
End time: ...

...

Run 11/11: trojan_io (io)
...

==========================================
Dataset Collection Complete!
==========================================
Total runs: 11
Data location: phase3/data/runs/
```

---

### 8. Post-Collection Validation

After dataset collection completes, validate the output:

```bash
cd phase3

# Count total runs
ls data/runs/ | wc -l
# Expected: 11 (or more if you ran smoke tests)

# Check sizes
du -sh data/runs/*

# Inspect a trojan run for interval data
cat data/runs/$(ls -t data/runs | grep trojan_compute | head -1)/trojan_intervals.csv
```

**Expected per-run outputs:**
- `telemetry.csv`: ~6000 rows (10 min × 10 rows/sec)
- `meta.json`: experiment metadata
- `trojan_intervals.csv`: trojan activation timestamps (trojan runs only)
- Optional `*.perf.log`, `*.tegrastats.log` (if `--save-raw` was used)

---

### 9. Validation Checklist

Go through each validation point from the plan:

#### ✓ Sampling Cadence
- **Expected:** ~10 rows/sec
- **Check:** `wc -l data/runs/<run>/telemetry.csv` should show ~6000 rows for 10-minute runs

#### ✓ Perf Metrics (Active Workloads)
- **perf_cycles, perf_instructions:** Should be nonzero and increasing
- **perf_ipc:** Should be in range 0.1–5.0 (varies by workload type)

#### ✓ Branch & Cache Features
- **perf_branch_misses, perf_branch_miss_rate:** Non-constant during load
- **perf_cache_miss_rate:** Higher for memory trojan vs normal

#### ✓ tegrastats Metrics
- **ts_gpu_util:** Should spike for GPU-heavy workloads (if using TensorRT/GPU inference)
- **ts_emc_util:** Should differ between scenarios
- **ts_ram_used_mb:** Should vary between idle/normal/trojan

#### ✓ Thermals & Power
- **temp_*:** Temperatures should increase under load vs idle
- **ts_power_*:** Power consumption should correlate with workload (if available)

#### ✓ Trojan Intervals Alignment
Compare `trojan_intervals.csv` timestamps with spikes in `telemetry.csv`:
- **Compute Trojan:** Look for spikes in `perf_cycles`, `perf_instructions`, CPU util
- **Memory Trojan:** Look for spikes in `ts_emc_util`, `perf_cache_miss_rate`, lower IPC
- **I/O Trojan:** Look for spikes in `perf_context_switches`, `perf_page_faults`

---

### 10. Manual Run Examples (Optional)

If you need to re-run specific scenarios or adjust parameters:

```bash
cd phase3

# Single idle run (60 seconds)
python3 scripts/run_experiment.py --label idle --duration 60

# Single normal workload run
python3 scripts/run_experiment.py --label normal --duration 60

# Trojan variants
python3 scripts/run_experiment.py --label trojan_compute --trojan-variant compute --duration 60
python3 scripts/run_experiment.py --label trojan_memory --trojan-variant memory --duration 60
python3 scripts/run_experiment.py --label trojan_io --trojan-variant io --duration 60

# Trojan with baseline overlay (alternating trojan + normal)
python3 scripts/run_experiment.py --label trojan_compute --trojan-variant compute --trojan-overlay --duration 60

# Custom parameters
python3 scripts/run_experiment.py \
  --label trojan_compute \
  --trojan-variant compute \
  --duration 300 \
  --period 10.0 \
  --active 2.0 \
  --perf-ms 100 \
  --tegrastats-ms 500 \
  --save-raw
```

**Key parameters:**
- `--label`: Scenario name (idle | normal | trojan_compute | trojan_memory | trojan_io)
- `--trojan-variant`: Trojan type (compute | memory | io)
- `--duration`: Total experiment duration in seconds
- `--period`: Trojan activation period (default 10.0s)
- `--active`: Active window within each period (default 2.0s)
- `--perf-ms`: perf stat sampling interval (default 100ms)
- `--tegrastats-ms`: tegrastats sampling interval (default 500ms)
- `--save-raw`: Save raw perf/tegrastats streams for debugging

---

## Troubleshooting

### Issue: `perf` not found for kernel

**Symptom:**
```
WARNING: perf not found for kernel X.Y.Z-N
```

**Solution:**
```bash
# Install kernel-specific tools
KERNEL=$(uname -r)
sudo apt-get install linux-tools-${KERNEL}

# If package not available, try generic
sudo apt-get install linux-tools-generic
```

---

### Issue: `tegrastats` command not found

**Symptom:**
```
tegrastats: command not found
```

**Solution:**
- `tegrastats` is JetPack-specific. Verify you're running on a Jetson device.
- Check if it's in `/usr/bin/tegrastats`
- If missing, reinstall JetPack or nvidia-l4t-tools:
  ```bash
  sudo apt-get install nvidia-l4t-tools
  ```

---

### Issue: perf permission denied

**Symptom:**
```
perf_event_open failed: Permission denied
```

**Solution:**
```bash
echo 1 | sudo tee /proc/sys/kernel/perf_event_paranoid
```

Or run experiments with `sudo`:
```bash
sudo python3 scripts/run_experiment.py --label idle --duration 60
```

---

### Issue: Thermal throttling during runs

**Symptom:**
- Clock frequencies drop during long runs
- Inconsistent performance metrics

**Solution:**
```bash
# Re-apply jetson_clocks
sudo jetson_clocks

# Monitor temps in real-time
sudo tegrastats --interval 1000

# Consider adding cooling (fan, heatsink) if temps exceed 80°C consistently
```

---

### Issue: Low disk space

**Symptom:**
```
No space left on device
```

**Solution:**
- Each 10-minute run generates ~1-2 MB of telemetry data
- 11 runs ≈ 15-20 MB total
- Clear old runs if needed:
  ```bash
  rm -rf phase3/data/runs/2024*  # example: delete old dates
  ```

---

## Quick Reference: Complete Command Sequence

```bash
# 1. Transfer repo to Jetson
git clone https://github.com/saran-gangster/HSDSF.git
cd HSDSF

# 2. Setup
cd phase3/scripts
bash setup_jetson.sh

# 3. Configure performance
sudo nvpmodel -m 0
sudo jetson_clocks
echo 1 | sudo tee /proc/sys/kernel/perf_event_paranoid

# 4. Smoke test (90 seconds)
bash run_tests.sh

# 5. Validate smoke test output
head -20 ../data/runs/$(ls -t ../data/runs | head -1)/telemetry.csv
cat ../data/runs/$(ls -t ../data/runs | head -1)/meta.json

# 6. Collect full dataset (90-120 minutes)
bash collect_dataset.sh

# 7. Validate dataset
ls ../data/runs/ | wc -l  # should show 11+ runs
du -sh ../data/runs/*
```

---

## Expected Timeline

| Phase | Duration | Description |
|-------|----------|-------------|
| 1. Transfer & Setup | 10-15 min | Clone repo, run setup_jetson.sh |
| 2. Performance Config | 2-3 min | nvpmodel, jetson_clocks, perf permissions |
| 3. Smoke Tests | 2 min | 3 × 30-second validation runs |
| 4. Smoke Validation | 5 min | Inspect outputs, verify columns |
| 5. Full Dataset Collection | 120 min | 11 × 10-minute runs (automated) |
| 6. Post-Collection Validation | 10 min | Check row counts, intervals, metrics |
| **TOTAL** | **~2.5 hours** | End-to-end deployment + collection |

---

## Data Output Summary

After successful collection, you should have:

```
phase3/data/runs/
├── YYYYMMDD_HHMMSS_idle/                    (3 runs)
│   ├── telemetry.csv                        (~6000 rows each)
│   └── meta.json
├── YYYYMMDD_HHMMSS_normal/                  (3 runs)
│   ├── telemetry.csv
│   └── meta.json
├── YYYYMMDD_HHMMSS_trojan_compute/          (2 runs)
│   ├── telemetry.csv
│   ├── trojan_intervals.csv
│   └── meta.json
├── YYYYMMDD_HHMMSS_trojan_memory/           (2 runs)
│   ├── telemetry.csv
│   ├── trojan_intervals.csv
│   └── meta.json
└── YYYYMMDD_HHMMSS_trojan_io/               (2 runs)
    ├── telemetry.csv
    ├── trojan_intervals.csv
    └── meta.json
```

**Total:** 11 runs, ~66,000 telemetry rows, ~15-20 MB

---

## Next Steps After Data Collection

1. **Transfer data to dev machine** for Phase 4 analysis:
   ```bash
   # From dev machine
   scp -r jetson@<jetson-ip>:~/HSDSF/phase3/data/runs /path/to/local/HSDSF/phase3/data/
   ```

2. **Proceed to Phase 4** (Anomaly Detection Model):
   - Feature engineering from telemetry
   - Train isolation forest / autoencoder
   - Evaluate detection performance

3. **Archive raw data** (recommended):
   ```bash
   cd phase3
   tar -czf data_$(date +%Y%m%d).tar.gz data/runs/
   ```

---

**Document Version:** 1.0  
**Last Updated:** December 17, 2025  
**Maintained by:** HSDSF Phase 3 Team
