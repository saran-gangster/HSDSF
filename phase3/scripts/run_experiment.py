import argparse
import json
import platform
import subprocess
import sys
import time
from pathlib import Path


def sh(cmd: str) -> str:
    try:
        return subprocess.check_output(cmd, text=True, shell=True, stderr=subprocess.STDOUT).strip()
    except Exception:
        return ""


def main():
    script_dir = Path(__file__).resolve().parent
    root = script_dir.parent

    ap = argparse.ArgumentParser(description="Run a telemetry experiment")
    ap.add_argument("--label", required=True, help="idle|normal|trojan_compute|trojan_memory|trojan_io")
    ap.add_argument("--duration", type=float, default=600.0)
    ap.add_argument("--outdir", type=Path, default=root / "data" / "runs")
    ap.add_argument("--trojan-variant", choices=["compute", "memory", "io"], help="Override trojan variant")
    ap.add_argument("--trojan-overlay", action="store_true", help="Add light baseline between trojan bursts")
    ap.add_argument("--period", type=float, default=10.0, help="Trojan activation period")
    ap.add_argument("--active", type=float, default=2.0, help="Trojan active window inside each period")
    ap.add_argument("--memory-mb", type=int, default=256, help="Memory footprint for memory trojan")
    ap.add_argument("--pps", type=int, default=50000, help="Packets per second for IO trojan")
    ap.add_argument("--dt", type=float, default=0.1)
    ap.add_argument("--perf-ms", type=int, default=100)
    ap.add_argument("--tegrastats-ms", type=int, default=500)
    ap.add_argument("--no-perf", action="store_true")
    ap.add_argument("--no-tegrastats", action="store_true")
    ap.add_argument("--tegrastats-cmd", type=str, default=None)
    ap.add_argument("--save-raw", action="store_true")
    ap.add_argument("--simulator-mode", action="store_true", help="Use simulator instead of real hardware")
    ap.add_argument("--simulator-port", type=int, default=45215, help="Simulator HTTP port")
    args = ap.parse_args()

    run_id = time.strftime("%Y%m%d_%H%M%S")
    run_dir = args.outdir / f"{run_id}_{args.label}"
    run_dir.mkdir(parents=True, exist_ok=True)

    telemetry_path = run_dir / "telemetry.csv"
    intervals_path = run_dir / "trojan_intervals.csv"

    collector_cmd = [
        sys.executable,
        str(script_dir / "collect_telemetry.py"),
        "--out",
        str(telemetry_path),
        "--label",
        args.label,
        "--dt",
        str(args.dt),
        "--perf-ms",
        str(args.perf_ms),
        "--tegrastats-ms",
        str(args.tegrastats_ms),
    ]

    if args.simulator_mode:
        collector_cmd.extend(["--simulator-mode", "--simulator-port", str(args.simulator_port)])
        collector_cmd.extend(["--sysfs-root", str(script_dir / "mock_sysfs")])
        # Use fake_tegrastats for simulator
        if not args.no_tegrastats:
            tegra_script = str(script_dir / "fake_tegrastats.sh")
            collector_cmd.extend(["--tegrastats-cmd", f"TEGRA_SIM_PORT={args.simulator_port} {tegra_script} --interval {args.tegrastats_ms}"])
    
    if args.no_perf:
        collector_cmd.append("--no-perf")
    if args.no_tegrastats:
        collector_cmd.append("--no-tegrastats")
    if args.tegrastats_cmd:
        collector_cmd.extend(["--tegrastats-cmd", args.tegrastats_cmd])
    if args.save_raw:
        collector_cmd.append("--save-raw")

    collector = subprocess.Popen(collector_cmd)
    time.sleep(1.0)

    normal_cmd = [
        sys.executable,
        str(script_dir / "run_normal_workload.py"),
        "--duration",
        str(args.duration),
    ]
    if args.simulator_mode:
        normal_cmd.extend(["--simulator-mode", "--simulator-port", str(args.simulator_port)])

    trojan_cmd_base = [
        sys.executable,
        str(script_dir / "run_trojan_workload.py"),
        "--duration",
        str(args.duration),
        "--period",
        str(args.period),
        "--active",
        str(args.active),
        "--intervals-out",
        str(intervals_path),
    ]
    if args.trojan_overlay:
        trojan_cmd_base.append("--with-baseline")
    if args.simulator_mode:
        trojan_cmd_base.extend(["--simulator-mode", "--simulator-port", str(args.simulator_port)])

    try:
        if args.label == "idle":
            time.sleep(args.duration)
        elif args.label == "normal":
            subprocess.check_call(normal_cmd)
        elif args.label.startswith("trojan_"):
            variant = args.trojan_variant or args.label.split("_", 1)[1]
            trojan_cmd = trojan_cmd_base + ["--variant", variant, "--memory-mb", str(args.memory_mb), "--pps", str(args.pps)]
            subprocess.check_call(trojan_cmd)
        else:
            raise SystemExit(f"Unknown label {args.label}")
    finally:
        collector.terminate()
        try:
            collector.wait(timeout=10)
        except subprocess.TimeoutExpired:
            collector.kill()

    meta = {
        "label": args.label,
        "duration_s": args.duration,
        "dt": args.dt,
        "perf_ms": args.perf_ms,
        "tegrastats_ms": args.tegrastats_ms,
        "no_perf": args.no_perf,
        "no_tegrastats": args.no_tegrastats,
        "trojan_variant": args.trojan_variant,
        "trojan_overlay": args.trojan_overlay,
        "period": args.period,
        "active": args.active,
        "memory_mb": args.memory_mb,
        "pps": args.pps,
        "run_dir": str(run_dir),
        "platform": platform.platform(),
        "uname": sh("uname -a"),
        "jetpack": sh("cat /etc/nv_tegra_release 2>/dev/null || true"),
        "nvpmodel": sh("sudo -n nvpmodel -q 2>/dev/null || true"),
        "jetson_clocks": sh("sudo -n jetson_clocks --show 2>/dev/null || true"),
    }
    (run_dir / "meta.json").write_text(json.dumps(meta, indent=2))

    print(f"Run completed: {run_dir}")


if __name__ == "__main__":
    main()
