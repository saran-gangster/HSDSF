---

## 0. Phase 3 Goals (Definition of “Done”)

By the end of Phase 3 you should have:

1. A **unified telemetry collector** that logs, at a fixed base rate:
   - CPU **PMCs** via `perf`: cycles, instructions, branches, branch-misses, cache events, etc.
   - **tegrastats** metrics: GPU/EMC utilization, RAM, CPU utilization, temps, power rails.
   - **/sys sensors**: additional thermal zones and on-board power rail readings (if accessible).
2. A small set of **workload scripts**:
   - `normal` workload: “known-good” inference‑like loop (CPU now; TensorRT later).
   - `trojan` workloads:
     - **compute Trojan** (ALU-heavy),
     - **memory Trojan** (DRAM/cache-heavy),
     - **I/O Trojan** (network/syscall/interrupt-heavy).
   - Trojans can run either **alone** or as a **parasitic overlay** on the normal workload.
3. A **reproducible experiment harness** that:
   - Creates one directory per run.
   - Runs:
     - `idle`
     - `normal`
     - `trojan_compute`, `trojan_memory`, `trojan_io` (solo and/or overlay).
   - Captures:
     - `telemetry.csv` (time‑aligned signals),
     - `trojan_intervals.csv` (for Trojan scenarios; ON/OFF intervals),
     - `meta.json` (environment, JetPack/nvpmodel/clocks, sampling configs).
4. A **dataset protocol**: how many runs, durations, labels, and a validation checklist.

Everything below is designed so you can write and test most code **without** the Jetson.

---

## 1. High-Level Architecture

We’ll structure Phase 3 into three logical components:

1. **Telemetry Layer**  
   Multi-source collector that:
   - Spawns `perf stat` and `tegrastats` as background processes.
   - Periodically samples `/sys` sensor files.
   - Maintains a shared “latest value” state per source.
   - Emits one merged row every **base interval** (e.g., 100 ms) using **zero-order hold** to upsample slow signals.

2. **Workload Layer**  
   - Standalone scripts for `normal` and `trojan` workloads.
   - Trojans run in **bursts** (e.g., 2s ON every 10s) and log their ON intervals.

3. **Experiment Orchestration Layer**  
   - A single script `run_experiment.py` that:
     - Creates a run directory.
     - Starts the collector.
     - Launches workload(s).
     - Stops collector, writes metadata.

Data flows:

```text
+---------------------+   +--------------------+
| run_experiment.py   |   |   Workloads        |
|   - config, label   |-->| normal / trojan_*  |
+-----------+---------+   +---------+----------+
            |                           |
            v                           |
   +--------+---------------------------+--------+
   |          Telemetry Collector                |
   |  perf  +  tegrastats  +  /sys sensors      |
   +--------+---------------------------+--------+
            |
            v
   per-run dir: telemetry.csv, trojan_intervals.csv, meta.json
```

---

## 2. Repository Layout

Create a repo like:

```text
phase3/
  README.md
  requirements.txt
  scripts/
    collect_telemetry.py
    perf_reader.py
    tegrastats_reader.py
    sys_sensors.py
    run_normal_workload.py
    run_trojan_workload.py
    run_experiment.py
  configs/
    perf_events.yaml          # optional; per-board tailoring later
  data/
    runs/                     # each experiment creates a subdir here
```

This keeps telemetry, workloads, and orchestration decoupled and testable.

---

## 3. Telemetry Design

### 3.1 Sampling & Synchronization Strategy

- Choose a **base sampling rate**: `dt = 0.1s` (10 Hz) for a good fidelity/overhead balance.
- `perf stat` supports `-I <ms>` to emit event counts at a given period (use 100 ms).
- `tegrastats` supports `--interval <ms>`; set 500 ms (2 Hz) initially.
- `/sys` sensors are directly read every 100 ms.

**Alignment method (per reviewer feedback):**

- Internal **master clock** uses `time.monotonic()`.
- Every `dt`, do:
  - Snapshot last values from:
    - perf thread,
    - tegrastats thread,
    - /sys immediate read.
  - Compute derived features (e.g., IPC, miss rates).
  - Write one row with `t_wall` (seconds since run start) and `ts_unix` (absolute time).
- For slower sources (tegrastats, some /sys entries), we hold the **last observed value** between updates (**zero-order hold**).

This ensures all signals are aligned to a single, regular time grid, which is exactly what your preprocessing in Phase 4 will expect.

---

### 3.2 Telemetry Schema

Each row in `telemetry.csv` will look like:

- **Core time columns**:
  - `t_wall`: float, seconds since start of collection (monotonic).
  - `ts_unix`: float, Unix timestamp.

- **Scenario labels**:
  - `label`: scenario label (`idle`, `normal`, `trojan_compute`, etc.).
  - (Per-sample Trojan flag is **not** written here; instead we log **intervals** separately for clarity.)

- **perf-derived columns** (prefix `perf_`):
  - Raw counts per interval (you can treat as rates if interval is fixed):
    - `perf_cycles`,
    - `perf_instructions`,
    - `perf_branches`,
    - `perf_branch_misses`,
    - `perf_cache_references`,
    - `perf_cache_misses`,
    - `perf_context_switches`,
    - `perf_cpu_migrations`,
    - `perf_page_faults`.
  - Derived:
    - `perf_ipc` = instructions / cycles.
    - `perf_branch_miss_rate` = branch_misses / branches.
    - `perf_cache_miss_rate` = cache_misses / cache_references.

- **tegrastats-derived columns** (prefix `ts_`):
  - `ts_ram_used_mb`, `ts_ram_total_mb`
  - `ts_gpu_util` (GR3D_FREQ %)
  - `ts_emc_util` (EMC_FREQ %)
  - `ts_cpu_util_sum` (sum of core utilizations, 0–(100 * num_cores))
  - `ts_temp_<name>_c` (temps reported directly by tegrastats)
  - `ts_pwr_<rail>_mw` (per-rail powers like `ts_pwr_pom_5v_in_mw`)

- **/sys-derived columns**:
  - `temp_<zone_type>_c` for each discovered thermal zone
  - Optional: `pwr_<rail>_mw` if accessible via INA3221 etc.

Values may be blank or NaN if not supported; Phase 4 can select a stable subset of features.

---

## 4. Telemetry Implementation

### 4.1 Perf Reader

We use `perf stat -I 100 -x ,` to get CSV-like lines:

```bash
LC_ALL=C perf stat -a -I 100 -x , \
  -e cycles,instructions,branches,branch-misses,cache-references,cache-misses,context-switches,cpu-migrations,page-faults \
  -- sleep 999999
```

Typical line format (simplified, may vary slightly):

```text
0.100,123456, ,cycles,100000000,100.00,0.00
```

So:

- `time` = `parts[0]`
- `value` = `parts[1]`
- `event` = `parts[3]`

We will confirm on hardware (and adjust indices if required). The parser is defensive and ignores malformed lines.

```python
# scripts/perf_reader.py
from dataclasses import dataclass
from typing import Dict, Optional

@dataclass
class PerfSample:
    t: float                # seconds since perf start (as reported by perf)
    values: Dict[str, float]

def parse_perf_stat_line(line: str) -> Optional[PerfSample]:
    parts = [p.strip() for p in line.split(",")]
    # Expecting at least time, value, unit, event
    if len(parts) < 4:
        return None

    try:
        t = float(parts[0])
    except ValueError:
        return None

    raw_val = parts[1]
    event = parts[3]  # event name
    if not event:
        return None

    # Ignore non-counted
    if raw_val in ("<not counted>", "<not supported>", ""):
        return PerfSample(t=t, values={event: float("nan")})

    raw_val = raw_val.replace(" ", "")
    try:
        val = float(raw_val.replace(",", ""))  # strip thousand separators just in case
    except ValueError:
        return None

    # Normalize key to snake_case (no dashes)
    ev_key = event.replace("-", "_")
    return PerfSample(t=t, values={ev_key: val})
```

### 4.2 tegrastats Reader

We parse common fields via regex; missing ones are simply omitted.

```python
# scripts/tegrastats_reader.py
import re
from typing import Dict

RE_RAM = re.compile(r"RAM\s+(\d+)/(\d+)MB")
RE_CPU = re.compile(r"CPU\s+\[(.*?)\]")
RE_GPU = re.compile(r"GR3D_FREQ\s+(\d+)%")
RE_EMC = re.compile(r"EMC_FREQ\s+(\d+)%")
RE_TEMP = re.compile(r"([A-Za-z0-9_]+)@(-?\d+(?:\.\d+)?)C")
RE_PWR = re.compile(r"(POM_[A-Za-z0-9_]+)\s+(\d+)mW")

def parse_tegrastats_line(line: str) -> Dict[str, float]:
    out: Dict[str, float] = {}

    m = RE_RAM.search(line)
    if m:
        out["ts_ram_used_mb"] = float(m.group(1))
        out["ts_ram_total_mb"] = float(m.group(2))

    m = RE_GPU.search(line)
    if m:
        out["ts_gpu_util"] = float(m.group(1))

    m = RE_EMC.search(line)
    if m:
        out["ts_emc_util"] = float(m.group(1))

    m = RE_CPU.search(line)
    if m:
        items = [s.strip() for s in m.group(1).split(",")]
        util_sum = 0.0
        for it in items:
            um = re.match(r"(\d+)%@(\d+)", it)
            if um:
                util_sum += float(um.group(1))
        out["ts_cpu_util_sum"] = util_sum

    for name, temp in RE_TEMP.findall(line):
        key = f"ts_temp_{name.lower()}_c"
        out[key] = float(temp)

    for rail, mw in RE_PWR.findall(line):
        key = f"ts_pwr_{rail.lower()}_mw"
        out[key] = float(mw)

    return out
```

### 4.3 /sys Sensors

We discover thermal zones dynamically:

```python
# scripts/sys_sensors.py
from pathlib import Path
from typing import Dict, List, Tuple

def discover_thermal_zones() -> List[Tuple[str, Path]]:
    zones = []
    base = Path("/sys/class/thermal")
    if not base.exists():
        return zones
    for zdir in base.glob("thermal_zone*"):
        tfile = zdir / "type"
        tempfile = zdir / "temp"
        if tfile.exists() and tempfile.exists():
            ztype = tfile.read_text().strip()
            zones.append((ztype, tempfile))
    return zones

def read_thermal_c(zones: List[Tuple[str, Path]]) -> Dict[str, float]:
    out = {}
    for ztype, tempfile in zones:
        try:
            raw = tempfile.read_text().strip()
            val = float(raw)
        except Exception:
            continue
        c = val / 1000.0 if val > 200 else val   # assume millidegree if large
        key = f"temp_{ztype.lower()}_c".replace(" ", "_")
        out[key] = c
    return out
```

Power rails via `/sys` vary a lot between Jetson revisions; you can add a second discovery function later that searches for INA3221 or iio devices. For Phase 3, tegrastats power metrics are usually sufficient.

### 4.4 Central Collector

The collector:

- Starts `perf` and `tegrastats` subprocesses.
- Spawns threads that continuously read their outputs and update `latest` dict.
- Every `dt` seconds:
  - Reads `/sys` temps.
  - Snapshots latest perf + tegrastats.
  - Computes derived metrics.
  - Writes a CSV row.

```python
# scripts/collect_telemetry.py
import argparse, csv, json, os, signal, subprocess, threading, time
from pathlib import Path
from typing import Dict, Any, Optional

from perf_reader import parse_perf_stat_line, PerfSample
from tegrastats_reader import parse_tegrastats_line
from sys_sensors import discover_thermal_zones, read_thermal_c

def start_subprocess(cmd: list[str]) -> subprocess.Popen:
    return subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
        universal_newlines=True,
        preexec_fn=os.setsid
    )

def perf_reader_thread(stream, latest: Dict[str, Any], lock: threading.Lock):
    for line in iter(stream.readline, ""):
        sample: Optional[PerfSample] = parse_perf_stat_line(line)
        if not sample:
            continue
        with lock:
            latest["perf_t"] = sample.t
            for k, v in sample.values.items():
                latest[f"perf_{k}"] = v

def tegrastats_reader_thread(stream, latest: Dict[str, Any], lock: threading.Lock):
    for line in iter(stream.readline, ""):
        parsed = parse_tegrastats_line(line)
        if not parsed:
            continue
        now = time.monotonic()
        with lock:
            latest["ts_t_wall"] = now
            latest.update(parsed)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--label", required=True)   # idle|normal|trojan_...
    ap.add_argument("--dt", type=float, default=0.1)
    ap.add_argument("--perf-ms", type=int, default=100)
    ap.add_argument("--tegrastats-ms", type=int, default=500)
    ap.add_argument("--perf-mode", choices=["system", "pid"], default="system")
    ap.add_argument("--pid", type=int, default=None)
    args = ap.parse_args()

    outpath = Path(args.out)
    outpath.parent.mkdir(parents=True, exist_ok=True)

    thermal_zones = discover_thermal_zones()

    perf_events = ",".join([
        "cycles",
        "instructions",
        "branches",
        "branch-misses",
        "cache-references",
        "cache-misses",
        "context-switches",
        "cpu-migrations",
        "page-faults",
    ])

    if args.perf_mode == "system":
        perf_target = "-a"
    else:
        if args.pid is None:
            raise SystemExit("perf-mode pid requires --pid")
        perf_target = f"-p {args.pid}"

    perf_cmd = ["bash", "-lc", " ".join([
        "LC_ALL=C",
        "perf stat",
        perf_target,
        f"-I {args.perf_ms}",
        "-x ,",
        f"-e {perf_events}",
        "-- sleep 999999"
    ])]

    tegra_cmd = ["tegrastats", "--interval", str(args.tegrastats_ms)]

    latest: Dict[str, Any] = {}
    lock = threading.Lock()

    perf_p = start_subprocess(perf_cmd)
    tegra_p = start_subprocess(tegra_cmd)

    t_perf = threading.Thread(
        target=perf_reader_thread,
        args=(perf_p.stderr, latest, lock),
        daemon=True,
    )
    t_tegra = threading.Thread(
        target=tegrastats_reader_thread,
        args=(tegra_p.stdout, latest, lock),
        daemon=True,
    )
    t_perf.start()
    t_tegra.start()

    t0 = time.monotonic()
    next_t = t0
    rows = 0
    fieldnames = None

    def shutdown():
        for p in (perf_p, tegra_p):
            try:
                os.killpg(os.getpgid(p.pid), signal.SIGTERM)
            except Exception:
                pass

    try:
        with outpath.open("w", newline="") as f:
            writer = None
            while True:
                now = time.monotonic()
                if now < next_t:
                    time.sleep(min(0.01, next_t - now))
                    continue

                t_wall = now - t0
                sys_vals = read_thermal_c(thermal_zones)

                with lock:
                    snap = dict(latest)

                # Derived metrics
                cyc = snap.get("perf_cycles")
                ins = snap.get("perf_instructions")
                if isinstance(cyc, (int, float)) and cyc and isinstance(ins, (int, float)):
                    snap["perf_ipc"] = ins / cyc

                br = snap.get("perf_branches")
                brm = snap.get("perf_branch_misses")
                if isinstance(br, (int, float)) and br and isinstance(brm, (int, float)):
                    snap["perf_branch_miss_rate"] = brm / br

                cr = snap.get("perf_cache_references")
                cm = snap.get("perf_cache_misses")
                if isinstance(cr, (int, float)) and cr and isinstance(cm, (int, float)):
                    snap["perf_cache_miss_rate"] = cm / cr

                row = {
                    "t_wall": t_wall,
                    "ts_unix": time.time(),
                    "label": args.label,
                }
                row.update(sys_vals)
                row.update(snap)

                if writer is None:
                    fieldnames = sorted(row.keys())
                    writer = csv.DictWriter(f, fieldnames=fieldnames)
                    writer.writeheader()

                for k in fieldnames:
                    row.setdefault(k, "")
                writer.writerow(row)
                rows += 1
                next_t += args.dt
    except KeyboardInterrupt:
        pass
    finally:
        shutdown()
        meta = {
            "rows": rows,
            "dt": args.dt,
            "label": args.label,
            "perf_ms": args.perf_ms,
            "tegrastats_ms": args.tegrastats_ms,
            "perf_mode": args.perf_mode,
        }
        outpath.with_suffix(".meta.json").write_text(json.dumps(meta, indent=2))

if __name__ == "__main__":
    main()
```

For non‑Jetson testing: you can add a `--no-tegrastats` flag and skip `tegrastats` spawn, or provide a fake script that prints sample lines.

---

## 5. Workloads

### 5.1 Normal Workload

Short-term (no Jetson): a CPU inference-like loop. Later you replace this with TensorRT/PyTorch inference on GPU.

```python
# scripts/run_normal_workload.py
import argparse, time
import numpy as np

def cpu_inference_like(duration_s: float, batch: int = 64, dim: int = 512):
    t0 = time.monotonic()
    x = np.random.randn(batch, dim).astype(np.float32)
    w = np.random.randn(dim, dim).astype(np.float32)
    while time.monotonic() - t0 < duration_s:
        y = x @ w
        y = np.tanh(y)
        x = y

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--duration", type=float, default=120.0)
    args = ap.parse_args()
    cpu_inference_like(args.duration)

if __name__ == "__main__":
    main()
```

On Jetson, you can:

- Either swap this out to run a real TensorRT engine.
- Or add a `--use-gpu` flag to pick a PyTorch CUDA matmul workload.

### 5.2 Trojan Workloads

We want:

- Clear **ON intervals** for labeling.
- Three **variants**: compute, memory, I/O.
- Ability to run:
  - **Solo** (just Trojan).
  - **Overlay**: Trojan runs periodically on top of `normal` workload in the same process or in a separate process (noisy neighbor).

We’ll implement the trojans with an internal scheduler and dump ON intervals:

```python
# scripts/run_trojan_workload.py
import argparse, socket, time
import numpy as np

def baseline_chunk(dt: float = 0.1):
    # trivial baseline work, so Trojan stands out
    x = np.random.randn(1024).astype(np.float32)
    y = np.sin(x) + 0.01 * x
    time.sleep(dt)

def trojan_compute(active_s: float):
    a = np.random.randn(256, 256).astype(np.float32)
    b = np.random.randn(256, 256).astype(np.float32)
    t0 = time.monotonic()
    while time.monotonic() - t0 < active_s:
        _ = a @ b

def trojan_memory(active_s: float, n_mb: int = 256):
    n = (n_mb * 1024 * 1024) // 4
    src = np.random.randint(0, 255, size=(n,), dtype=np.uint8)
    dst = np.empty_like(src)
    t0 = time.monotonic()
    while time.monotonic() - t0 < active_s:
        np.copyto(dst, src)

def trojan_io(active_s: float, pps: int = 50000):
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    payload = b"x" * 256
    addr = ("127.0.0.1", 9999)
    t0 = time.monotonic()
    interval = 1.0 / max(1, pps)
    next_t = time.monotonic()
    while time.monotonic() - t0 < active_s:
        s.sendto(payload, addr)
        next_t += interval
        while time.monotonic() < next_t:
            pass

def run_trojan_pattern(duration_s: float, period_s: float, active_s: float,
                       variant: str, intervals_out: str, with_baseline: bool):
    t0 = time.monotonic()
    next_on = t0 + period_s  # delay first activation to capture baseline
    intervals = []

    while time.monotonic() - t0 < duration_s:
        now = time.monotonic()
        if now >= next_on:
            on_start = now
            if variant == "compute":
                trojan_compute(active_s)
            elif variant == "memory":
                trojan_memory(active_s)
            elif variant == "io":
                trojan_io(active_s)
            else:
                raise ValueError("unknown variant")
            on_end = time.monotonic()
            intervals.append((on_start - t0, on_end - t0))
            next_on += period_s
        else:
            if with_baseline:
                baseline_chunk(0.1)
            else:
                time.sleep(0.05)

    with open(intervals_out, "w") as f:
        for a, b in intervals:
            f.write(f"{a:.3f},{b:.3f}\n")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--duration", type=float, default=120.0)
    ap.add_argument("--period", type=float, default=10.0)
    ap.add_argument("--active", type=float, default=2.0)
    ap.add_argument("--variant", choices=["compute", "memory", "io"], required=True)
    ap.add_argument("--intervals-out", required=True)
    ap.add_argument("--with-baseline", action="store_true",
                    help="run light baseline work between Trojan bursts")
    args = ap.parse_args()

    run_trojan_pattern(args.duration, args.period, args.active,
                       args.variant, args.intervals_out, args.with_baseline)

if __name__ == "__main__":
    main()
```

This script is:

- **Self-labeled**: outputs `trojan_intervals.csv` for each run.
- Adjustable: `period` and `active` can be tuned to change Trojan duty cycle.
- Capable of overlaying a baseline if `--with-baseline` is used.

If later you want the Trojan as a true “noisy neighbor” on top of the *normal* workload, you can:

- Run `run_normal_workload.py` in one process.
- Run `run_trojan_workload.py --with-baseline false` in another process.
- Collect system-wide perf; the Trojan will appear as contention.

---

## 6. Experiment Orchestration

We now need a single script that:

- Creates a **run directory**.
- Starts `collect_telemetry.py` with the appropriate label.
- Runs the chosen scenario:
  - `idle`: just sleep.
  - `normal`: run normal workload.
  - `trojan_*`: run Trojan script (possibly overlay mode).
- Stops collector.
- Writes **meta.json**.

```python
# scripts/run_experiment.py
import argparse, json, os, subprocess, time
from pathlib import Path

def sh(cmd: str) -> str:
    try:
        return subprocess.check_output(["bash", "-lc", cmd], text=True).strip()
    except Exception:
        return ""

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", required=True,  # idle|normal|trojan_compute...
                    help="Scenario label")
    ap.add_argument("--duration", type=float, default=600.0)
    ap.add_argument("--outdir", default="data/runs")
    ap.add_argument("--trojan-variant", choices=["compute", "memory", "io"])
    ap.add_argument("--trojan-overlay", action="store_true",
                    help="Overlay Trojan over light baseline (within same process)")
    args = ap.parse_args()

    run_id = time.strftime("%Y%m%d_%H%M%S")
    run_dir = Path(args.outdir) / f"{run_id}_{args.label}"
    run_dir.mkdir(parents=True, exist_ok=True)

    telemetry_path = run_dir / "telemetry.csv"
    intervals_path = run_dir / "trojan_intervals.csv"

    # Start telemetry collector
    collector = subprocess.Popen([
        "python3", "scripts/collect_telemetry.py",
        "--out", str(telemetry_path),
        "--label", args.label,
        "--dt", "0.1",
        "--perf-ms", "100",
        "--tegrastats-ms", "500",
        "--perf-mode", "system",
    ])

    time.sleep(1.0)  # collector warm-up

    try:
        if args.label == "idle":
            time.sleep(args.duration)
        elif args.label == "normal":
            subprocess.check_call([
                "python3", "scripts/run_normal_workload.py",
                "--duration", str(args.duration),
            ])
        elif args.label.startswith("trojan_"):
            variant = args.trojan_variant or args.label.split("_", 1)[1]
            # Trojan-only; baseline inside Trojan script if overlay requested
            subprocess.check_call([
                "python3", "scripts/run_trojan_workload.py",
                "--duration", str(args.duration),
                "--period", "10",
                "--active", "2",
                "--variant", variant,
                "--intervals-out", str(intervals_path),
            ] + (["--with-baseline"] if args.trojan_overlay else []))
        else:
            raise SystemExit(f"Unknown label {args.label}")
    finally:
        collector.terminate()
        try:
            collector.wait(timeout=10)
        except subprocess.TimeoutExpired:
            collector.kill()

    # Environment metadata
    meta = {
        "label": args.label,
        "duration_s": args.duration,
        "uname": sh("uname -a"),
        "jetpack": sh("cat /etc/nv_tegra_release 2>/dev/null || true"),
        "nvpmodel": sh("sudo -n nvpmodel -q 2>/dev/null || true"),
        "jetson_clocks": sh("sudo -n jetson_clocks --show 2>/dev/null || true"),
    }
    (run_dir / "meta.json").write_text(json.dumps(meta, indent=2))

    print(f"Run completed: {run_dir}")

if __name__ == "__main__":
    main()
```

---

## 7. How to Use This Plan

### 7.1 Now (No Jetson yet)

On your laptop:

1. Install dependencies:

```bash
pip install numpy
sudo apt-get install linux-tools-common linux-tools-generic linux-tools-$(uname -r)
```

2. `perf` will work on your CPU; `tegrastats` will not. For now:
   - Either add a `--no-tegrastats` flag to `collect_telemetry.py` and skip that subprocess.
   - Or create a fake `tegrastats` script in `scripts/` that prints sample lines at 1–2 Hz and modify the command to call that when not on Jetson.

3. Test:
   - `python3 scripts/run_experiment.py --label idle --duration 30`
   - `python3 scripts/run_experiment.py --label normal --duration 30`
   - `python3 scripts/run_experiment.py --label trojan_compute --trojan-variant compute --duration 30`

4. Inspect CSV outputs to verify:
   - Roughly 10 rows/sec.
   - Perf counters non-empty.
   - Derived IPC & miss rates present.

This validates your parsing/synchronization logic before hardware access.

### 7.2 On Jetson (When Available)

1. Install requirements:

```bash
sudo apt-get update
sudo apt-get install python3-pip linux-tools-common linux-tools-generic linux-tools-$(uname -r)
pip3 install numpy
sudo tegrastats --interval 1000 --count 1   # verify available
```

2. Set a stable performance state (to reduce noise):

```bash
sudo nvpmodel -m 0         # MAXN or appropriate mode for your board
sudo jetson_clocks
```

3. Collect an initial smoke-test set:

```bash
python3 scripts/run_experiment.py --label idle --duration 60
python3 scripts/run_experiment.py --label normal --duration 60
python3 scripts/run_experiment.py --label trojan_compute --trojan-variant compute --duration 60
```

4. Inspect:

- Confirm all key features present:
  - `perf_cycles`, `perf_instructions`, `perf_ipc`.
  - `ts_gpu_util`, `ts_ram_used_mb`, some temps/power.
  - `temp_*` from `/sys`.

5. If perf complains about permissions, either run experiments with `sudo` or lower `perf_event_paranoid`:

```bash
echo 1 | sudo tee /proc/sys/kernel/perf_event_paranoid
```

### 7.3 Full Dataset Collection Protocol

After validation, collect longer runs:

- **Idle**: 3 × 10 minutes
- **Normal**: 3 × 10 minutes
- **Trojan (solo)**:
  - 2 × 10 minutes each of `compute`, `memory`, `io`
- (Optionally) **Trojan overlay** (with `--trojan-overlay`) for 1–2 runs per variant.

Commands (examples):

```bash
# Idle
python3 scripts/run_experiment.py --label idle --duration 600

# Normal
python3 scripts/run_experiment.py --label normal --duration 600

# Trojan variants
python3 scripts/run_experiment.py --label trojan_compute --trojan-variant compute --duration 600
python3 scripts/run_experiment.py --label trojan_memory  --trojan-variant memory  --duration 600
python3 scripts/run_experiment.py --label trojan_io      --trojan-variant io      --duration 600
```

Total: ~90–120 minutes of telemetry, consistent with your plan.

---

## 8. Validation Checklist

After the first real run on Jetson:

1. **Cadence**: ~10 rows/sec in `telemetry.csv` for each run.
2. **Perf metrics**:
   - `perf_cycles` and `perf_instructions` nonzero for active workloads.
   - `perf_ipc` in a plausible range (0.1–5.0, depending on load).
3. **Branch & cache features**:
   - `perf_branch_misses` and `perf_branch_miss_rate` non-constant during load.
   - `perf_cache_miss_rate` higher for memory Trojan vs normal.
4. **tegrastats**:
   - `ts_gpu_util` spikes for GPU-heavy normal workload (once you switch to TensorRT / GPU).
   - `ts_emc_util` and RAM usage differ between scenarios.
5. **Thermals / power**:
   - Temps increase under load vs idle.
   - If power rails are present, they should correlate with load.
6. **Trojan intervals**:
   - Check `trojan_intervals.csv` vs spikes:
     - Compute Trojan: cycles/instructions/CPU util up, possibly IPC changes.
     - Memory Trojan: EMC util and cache miss rate up, IPC down.
     - I/O Trojan: context-switches/page-faults up.

---

## 9. Planned Refinements (Optional but Easy)

Once the basic plan is working, you can improve:

- **Event configuration via YAML** (`configs/perf_events.yaml`) and runtime discovery (`perf list`) to adapt to different Jetson variants.
- **Parquet output** for more efficient Phase‑4 ingestion.
- **Per-PID perf mode** by:
  - Launching workload, capturing its PID, and starting `collect_telemetry.py --perf-mode pid --pid <PID>`.
- Adding a `--save-raw` option to `collect_telemetry.py` that logs raw `perf` and `tegrastats` lines in sidecar files for debugging.

---

