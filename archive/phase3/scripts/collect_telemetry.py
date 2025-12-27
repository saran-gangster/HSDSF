import argparse
import csv
import json
import os
import signal
import subprocess
import sys
import threading
import time
import urllib.request
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import yaml

from perf_reader import PerfSample, parse_perf_stat_line
from tegrastats_reader import parse_tegrastats_line
from sys_sensors import discover_thermal_zones, read_thermal_c


def _platform_group_kwargs() -> Dict[str, Any]:
    if os.name == "nt":
        create_flag = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0x00000200)
        return {"creationflags": create_flag}
    return {"preexec_fn": os.setsid}


def start_subprocess(cmd: List[str], *, capture_stdout: bool, capture_stderr: bool) -> subprocess.Popen:
    kwargs: Dict[str, Any] = {
        "stdout": subprocess.PIPE if capture_stdout else None,
        "stderr": subprocess.PIPE if capture_stderr else None,
        "text": True,
        "bufsize": 1,
        "universal_newlines": True,
    }
    kwargs.update(_platform_group_kwargs())
    return subprocess.Popen(cmd, **kwargs)


def load_perf_events(explicit: Optional[List[str]], config_path: Optional[Path]) -> List[str]:
    if explicit:
        return explicit
    if config_path and config_path.exists():
        cfg = yaml.safe_load(config_path.read_text()) or {}
        if isinstance(cfg, dict) and isinstance(cfg.get("perf_events"), list):
            return [str(e) for e in cfg["perf_events"]]
    return [
        "cycles",
        "instructions",
        "branches",
        "branch-misses",
        "cache-references",
        "cache-misses",
        "context-switches",
        "cpu-migrations",
        "page-faults",
    ]


def perf_reader_thread(stream, latest: Dict[str, Any], lock: threading.Lock, stop: threading.Event, raw_fp=None):
    for line in iter(stream.readline, ""):
        if stop.is_set():
            break
        if raw_fp:
            raw_fp.write(line)
        sample: Optional[PerfSample] = parse_perf_stat_line(line)
        if not sample:
            continue
        with lock:
            latest["perf_t"] = sample.t
            for k, v in sample.values.items():
                latest[f"perf_{k}"] = v


def simulator_perf_reader_thread(port: int, latest: Dict[str, Any], lock: threading.Lock, stop: threading.Event, interval_s: float):
    """Poll simulator HTTP API for perf counters"""
    prev_counters = {}
    while not stop.is_set():
        try:
            url = f"http://127.0.0.1:{port}/v1/telemetry"
            with urllib.request.urlopen(url, timeout=2.0) as resp:
                data = json.loads(resp.read().decode("utf-8"))
            
            counters = data.get("counters_total", {})
            # Compute deltas (perf stat -I style)
            with lock:
                for ev, total in counters.items():
                    # Normalize hyphens to underscores for CSV compatibility
                    ev_normalized = ev.replace('-', '_')
                    if ev in prev_counters:
                        latest[f"perf_{ev_normalized}"] = total - prev_counters[ev]
                    prev_counters[ev] = total
        except Exception:
            pass
        time.sleep(interval_s)


def tegrastats_reader_thread(stream, latest: Dict[str, Any], lock: threading.Lock, stop: threading.Event, raw_fp=None):
    for line in iter(stream.readline, ""):
        if stop.is_set():
            break
        if raw_fp:
            raw_fp.write(line)
        parsed = parse_tegrastats_line(line)
        if not parsed:
            continue
        now = time.monotonic()
        with lock:
            latest["ts_t_wall"] = now
            latest.update(parsed)


def maybe_open(path: Optional[Path]):
    return path.open("w") if path else None


def main(argv: Optional[Iterable[str]] = None):
    script_dir = Path(__file__).resolve().parent
    default_cfg = script_dir.parent / "configs" / "perf_events.yaml"

    ap = argparse.ArgumentParser(description="Collect perf/tegrastats/sys telemetry")
    ap.add_argument("--out", required=True, help="Output CSV path")
    ap.add_argument("--label", required=True, help="Scenario label")
    ap.add_argument("--dt", type=float, default=0.1, help="Base sampling period in seconds")
    ap.add_argument("--perf-ms", type=int, default=100, help="perf stat interval in ms")
    ap.add_argument("--tegrastats-ms", type=int, default=500, help="tegrastats interval in ms")
    ap.add_argument("--perf-mode", choices=["system", "pid"], default="system")
    ap.add_argument("--pid", type=int, default=None, help="PID to monitor when perf-mode=pid")
    ap.add_argument("--perf-events", nargs="*", help="Override perf events list")
    ap.add_argument("--perf-events-file", type=Path, default=default_cfg, help="YAML file with perf_events list")
    ap.add_argument("--no-perf", action="store_true", help="Skip launching perf stat")
    ap.add_argument("--no-tegrastats", action="store_true", help="Skip launching tegrastats")
    ap.add_argument("--tegrastats-cmd", type=str, default=None, help="Alternate tegrastats command for testing")
    ap.add_argument("--save-raw", action="store_true", help="Save raw perf/tegrastats streams next to CSV")
    ap.add_argument("--stop-after", type=float, default=None, help="Optional hard stop after N seconds")
    ap.add_argument("--simulator-mode", action="store_true", help="Use simulator HTTP API instead of real perf")
    ap.add_argument("--simulator-port", type=int, default=45215, help="Simulator HTTP port")
    ap.add_argument("--sysfs-root", type=Path, default=None, help="Override sysfs root (use simulator mock sysfs)")
    args = ap.parse_args(list(argv) if argv is not None else None)

    if args.simulator_mode and args.sysfs_root is None:
        args.sysfs_root = script_dir / "mock_sysfs"

    outpath = Path(args.out)
    outpath.parent.mkdir(parents=True, exist_ok=True)

    perf_events = load_perf_events(args.perf_events, args.perf_events_file)

    perf_target = "-a" if args.perf_mode == "system" else f"-p {args.pid}" if args.pid else None
    if args.perf_mode == "pid" and args.pid is None:
        raise SystemExit("perf-mode pid requires --pid")

    raw_perf_path = outpath.with_suffix(".perf.log") if args.save_raw else None
    raw_tegrastats_path = outpath.with_suffix(".tegrastats.log") if args.save_raw else None

    thermal_zones = discover_thermal_zones(args.sysfs_root)

    latest: Dict[str, Any] = {}
    lock = threading.Lock()
    stop_event = threading.Event()

    perf_p = None
    tegra_p = None
    perf_thread = None
    tegra_thread = None

    perf_cmd: List[str] = []
    if not args.no_perf and not args.simulator_mode and perf_target:
        perf_events_str = ",".join(perf_events)
        perf_cmd = [
            "bash",
            "-lc",
            " ".join([
                "LC_ALL=C",
                "perf stat",
                perf_target,
                f"-I {args.perf_ms}",
                "-x ,",
                f"-e {perf_events_str}",
                "-- sleep 999999",
            ]),
        ]

    tegrastats_cmd = None
    if not args.no_tegrastats:
        tegrastats_cmd = args.tegrastats_cmd.split() if args.tegrastats_cmd else ["tegrastats", "--interval", str(args.tegrastats_ms)]

    try:
        perf_fp = maybe_open(raw_perf_path)
        tegra_fp = maybe_open(raw_tegrastats_path)

        if args.simulator_mode:
            # Use simulator HTTP API for perf counters
            perf_thread = threading.Thread(
                target=simulator_perf_reader_thread,
                args=(args.simulator_port, latest, lock, stop_event, args.perf_ms / 1000.0),
                daemon=True,
            )
            perf_thread.start()
        elif perf_cmd:
            try:
                perf_p = start_subprocess(perf_cmd, capture_stdout=False, capture_stderr=True)
                perf_thread = threading.Thread(
                    target=perf_reader_thread,
                    args=(perf_p.stderr, latest, lock, stop_event, perf_fp),
                    daemon=True,
                )
                perf_thread.start()
            except FileNotFoundError:
                perf_p = None

        if tegrastats_cmd:
            try:
                tegra_p = start_subprocess(tegrastats_cmd, capture_stdout=True, capture_stderr=False)
                tegra_thread = threading.Thread(
                    target=tegrastats_reader_thread,
                    args=(tegra_p.stdout, latest, lock, stop_event, tegra_fp),
                    daemon=True,
                )
                tegra_thread.start()
            except FileNotFoundError:
                tegra_p = None

        t0 = time.monotonic()
        next_t = t0
        rows = 0
        fieldnames: Optional[List[str]] = None

        def shutdown():
            stop_event.set()
            for p in (perf_p, tegra_p):
                if not p:
                    continue
                try:
                    if os.name == "nt":
                        p.terminate()
                    else:
                        os.killpg(os.getpgid(p.pid), signal.SIGTERM)
                except Exception:
                    try:
                        p.terminate()
                    except Exception:
                        pass

        with outpath.open("w", newline="") as f:
            writer: Optional[csv.DictWriter] = None
            warmup_rows = 0  # Collect a few rows before writing header
            header_written = False
            
            while True:
                now = time.monotonic()
                if args.stop_after and (now - t0) >= args.stop_after:
                    break
                if now < next_t:
                    time.sleep(min(0.01, next_t - now))
                    continue

                t_wall = now - t0
                sys_vals = read_thermal_c(thermal_zones)

                with lock:
                    snap = dict(latest)

                cyc = snap.get("perf_cycles")
                ins = snap.get("perf_instructions")
                if isinstance(cyc, (int, float)) and cyc and isinstance(ins, (int, float)):
                    snap["perf_ipc"] = ins / cyc

                # Use branch_instructions (from simulator) or branches (from real perf)
                br = snap.get("perf_branch_instructions") or snap.get("perf_branches")
                brm = snap.get("perf_branch_misses")
                if isinstance(br, (int, float)) and br and isinstance(brm, (int, float)):
                    snap["perf_branch_miss_rate"] = brm / br

                cr = snap.get("perf_cache_references")
                cm = snap.get("perf_cache_misses")
                if isinstance(cr, (int, float)) and cr and isinstance(cm, (int, float)):
                    snap["perf_cache_miss_rate"] = cm / cr

                row: Dict[str, Any] = {
                    "t_wall": t_wall,
                    "ts_unix": time.time(),
                    "label": args.label,
                }
                row.update(sys_vals)
                row.update(snap)

                # Wait for at least 3 warmup iterations to let all data sources populate
                if not header_written:
                    warmup_rows += 1
                    if warmup_rows < 3:
                        next_t += args.dt
                        continue
                    # Now write header with full field set
                    fieldnames = sorted(row.keys())
                    writer = csv.DictWriter(f, fieldnames=fieldnames)
                    writer.writeheader()
                    header_written = True

                for k in fieldnames:
                    row.setdefault(k, "")
                writer.writerow(row)
                rows += 1
                next_t += args.dt
    except KeyboardInterrupt:
        pass
    finally:
        stop_event.set()
        for p in (perf_p, tegra_p):
            if not p:
                continue
            try:
                if os.name == "nt":
                    p.terminate()
                else:
                    os.killpg(os.getpgid(p.pid), signal.SIGTERM)
            except Exception:
                try:
                    p.terminate()
                except Exception:
                    pass

        for th in (perf_thread, tegra_thread):
            if th:
                th.join(timeout=2)

        meta = {
            "rows": rows if "rows" in locals() else 0,
            "dt": args.dt,
            "label": args.label,
            "perf_ms": args.perf_ms,
            "tegrastats_ms": args.tegrastats_ms,
            "perf_mode": args.perf_mode,
            "perf_events": perf_events,
            "no_perf": args.no_perf,
            "no_tegrastats": args.no_tegrastats,
            "stop_after": args.stop_after,
            "platform": sys.platform,
        }
        outpath.with_suffix(".meta.json").write_text(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
