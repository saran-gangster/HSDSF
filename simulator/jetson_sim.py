#!/usr/bin/env python3
"""
Jetson AGX Xavier TDG-aligned telemetry simulator (stdlib only).

Implements, per user-provided TDG requirement list:
- Thermal Transfer Plate (TTP) node with limit 80°C
- AUX sensor with slow dynamics correlated with EMC and SoC/CV power
- Junction-to-plate resistance θjp: balanced 0.35 °C/W, unbalanced 0.65 °C/W
- Junction-to-board secondary path θjb: 5.5 °C/W
- INA3221 topology: 0-0040 (GPU/CPU/SOC), 0-0041 (CV/VDDRQ/SYS5V)
- INA3221 sysfs exposure uses 512-sample rolling average (LPF)
- thermal-fan-est virtual sensor: (3*CPU + 3*GPU + 4*AUX)/10
- Fan step table + hysteresis: OFF/77/120/160/255 with specified down-trip points
- Thermal throttling and reset thresholds vary by power mode: 30W vs MaxN
- Reset when T > max limit; hard stop when T > 105°C
- Input voltage efficiency scalar for SYS5V power:
    P_total = P_components * (1.0 + 0.015*(Vin - 9.0))

Plus “realism fixes” (configurable):
- θjp continuous blending based on power concentration (instead of hard 60% rule)
- fan PWM affects cooling (theta_heatsink_eff modulated by PWM)
- board node tracks TTP (not a magic power fraction)
- rail-weighted power cap shaping (GPU/CV hit harder, SoC less)
- AUX thermal memory (AUX cools down grudgingly)
- sensor bias offsets (CPU +0.8°C, GPU -0.5°C, AUX +1.2°C by default)
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import signal
import threading
import time
from collections import deque
from dataclasses import dataclass, asdict, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Deque, Dict, List, Optional, Set, Tuple

DEFAULT_PORT = 45215

# Shape-only (not claiming exact Xavier BSP values)
N_CPU = 8
RAM_TOTAL_MB = 31928
SWAP_TOTAL_MB = 0

CPU_FREQ_STEPS_MHZ = [345, 768, 1190, 1420, 1650, 1900, 2200]
GPU_FREQ_STEPS_MHZ = [318, 522, 675, 900, 1100]
EMC_FREQ_STEPS_MHZ = [800, 1066, 1331, 1600, 1866, 2133]

SUPPORTED_EVENTS = [
    "cycles",
    "instructions",
    "cache-references",
    "cache-misses",
    "branch-instructions",
    "branch-misses",
    "context-switches",
    "cpu-migrations",
    "page-faults",
    "L1-dcache-loads",
    "L1-dcache-load-misses",
    "LLC-loads",
    "LLC-load-misses",
]

# TDG thresholds from the user-provided requirement list
THERM_THRESH = {
    "30W": {
        "CPU": (90.0, 95.5),
        "GPU": (92.5, 98.0),
        "AUX": (89.0, 94.5),
    },
    "MAXN": {
        "CPU": (86.0, 91.5),
        "GPU": (88.0, 93.5),
        "AUX": (82.0, 87.5),
    },
}
HARD_TRIP_C = 105.0


# ---------------------------- utilities ----------------------------

def now_s() -> float:
    return time.time()

def clamp(x: float, lo: float, hi: float) -> float:
    return lo if x < lo else hi if x > hi else x

def _safe_mkdir_for(path: str) -> None:
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)

def ensure_file(path: str, content: str) -> None:
    """Write only when changed to reduce IO churn."""
    _safe_mkdir_for(path)
    try:
        with open(path, "r", encoding="utf-8") as f:
            old = f.read()
        if old == content:
            return
    except FileNotFoundError:
        pass
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)

def choose_step_idx(steps: List[int], x01: float, max_idx: int, min_idx: int = 0) -> int:
    x01 = clamp(x01, 0.0, 1.0)
    idx = int(round(x01 * (len(steps) - 1)))
    idx = max(min_idx, idx)
    idx = min(max_idx, idx)
    return idx

def fmt_commas_int(x: float) -> str:
    return f"{int(round(x)):,}"


# ---------------------------- rolling average (INA LPF) ----------------------------

class RollingAvg:
    """O(1) rolling average over last N samples."""
    def __init__(self, n: int):
        if n <= 0:
            raise ValueError("n must be > 0")
        self.n = n
        self.q: Deque[float] = deque()
        self.s = 0.0

    def push(self, x: float) -> None:
        if len(self.q) == self.n:
            self.s -= self.q.popleft()
        self.q.append(x)
        self.s += x

    def avg(self) -> float:
        return self.s / len(self.q) if self.q else 0.0


# ---------------------------- config/state ----------------------------

@dataclass
class SimConfig:
    seed: int = 1337
    hz: float = 20.0
    ambient_c: float = 24.0

    # Determinism / offline generation
    fast: bool = False
    emit_unix_from_sim: bool = True
    start_unix_s: float = 1700000000.0

    # Mode for offline collection ("idle", "normal", "trojan")
    mode: str = "normal"

    # Dataset identity
    workload_family: str = "inference_periodic"
    workload_variant: str = "v1"
    trojan_family: str = "sim_trojan"
    trojan_variant: str = "v1"

    # Power mode affects thermal thresholds
    power_mode: str = "30W"  # "30W" or "MAXN"
    power_cap_w: float = 30.0  # soft cap used for shaping

    # Input voltage and inefficiency
    input_voltage_v: float = 19.0

    # Trojan schedule (seconds)
    trojan_period_s: float = 40.0
    trojan_on_s: float = 10.0
    trojan_style: str = "mempressure"  # "mempressure"|"compute"|"cv"|"io"
    trojan_strength: float = 1.0

    # INA averaging window
    ina_window_samples: int = 512
    ina_internal_hz: float = 200.0  # INA conversion cadence (decoupled from sim tick)

    # Thermal network constants
    theta_jp_balanced_c_per_w: float = 0.35
    theta_jp_unbalanced_c_per_w: float = 0.65
    theta_jb_c_per_w: float = 5.5
    ttp_max_c: float = 80.0

    # Heatsink effective theta in TTP equation:
    # TTP = Tamb + Ptotal * theta_heatsink_eff
    theta_heatsink_eff_c_per_w: float = 1.15

    # Thermal inertia (TTP and board are heavy masses; junctions are lighter)
    # Calibrated from step-response fits (heat-up + cooldown phases)
    tau_ttp_s: float = 16.5     # avg of heat-up 13.5s and cooldown 19.5s
    tau_board_s: float = 320.0  # Backside stiffener has BIG inertia (calibration delta too small)

    # NOTE: per-cluster CPU taus (4 Carmel clusters)
    tau_cpu_s: float = 20.9     # avg of heat-up 16.3s and cooldown 25.4s
    tau_cpu_cluster_s: float = 20.4  # avg of heat-up 16.4s and cooldown 24.4s
    tau_gpu_s: float = 20.6     # avg of heat-up 16.8s and cooldown 24.4s
    # Two-lump AUX: CV (DLA/PVA/CV) and DRAM (VDDRQ)
    tau_aux_s: float = 25.7     # avg of heat-up 19.6s and cooldown 31.8s
    tau_aux_cv_s: float = 23.3  # avg of heat-up 16.8s and cooldown 29.8s
    tau_aux_ddr_s: float = 37.2 # cooldown only (heat-up delta too small)


    # Workload shaping
    frame_rate_hz: float = 30.0
    frame_gpu_busy_ms: float = 8.0
    frame_cpu_busy_ms: float = 3.2
    bg_spike_rate_hz: float = 0.35
    bg_spike_ms: float = 45.0
    bg_spike_cpu_pct: float = 18.0

    # DVFS inertia + caps
    cpu_max_step_idx: int = len(CPU_FREQ_STEPS_MHZ) - 1
    gpu_max_step_idx: int = len(GPU_FREQ_STEPS_MHZ) - 1
    emc_max_step_idx: int = len(EMC_FREQ_STEPS_MHZ) - 1
    cpu_step_rate_per_s: float = 3.5
    gpu_step_rate_per_s: float = 4.0
    emc_step_rate_per_s: float = 3.0

    # Throttle behavior pacing
    throttle_step_period_s: float = 0.5
    restore_step_period_s: float = 1.0

    # Noise
    util_noise_sigma: float = 1.4
    sensor_noise_sigma: float = 0.25

    # ---------------- realism fixes ----------------

    # (1) θjp blending: continuously move between balanced and unbalanced
    # based on concentration of power in CPU/GPU/CV.
    theta_jp_blend: bool = True
    theta_jp_blend_knee: float = 0.33
    theta_jp_blend_span: float = 0.33

    # (2) Fan affects cooling by modulating effective heatsink theta.
    # cooling_gain = min + (max-min)*(pwm/255); theta_eff = theta_heatsink_eff / cooling_gain
    fan_cooling_min_gain: float = 0.50
    fan_cooling_max_gain: float = 0.95

    # (3) Board node tracks TTP rather than power magic-number
    board_ttp_coupling: float = 0.48

    # (4) Rail-weighted power cap shaping (GPU/CV get hit harder; SoC less).
    power_cap_exp_cpu: float = 1.10
    power_cap_exp_gpu: float = 1.40
    power_cap_exp_cv: float = 1.60
    power_cap_exp_soc: float = 0.60
    power_cap_exp_vddrq: float = 1.20

    # (5) AUX thermal memory
    # AUX retains heat noticeably after load drops
    # Broke into CV and DDR memory behaviour
    tau_aux_memory_s: float = 260.0  # legacy fallback
    tau_aux_cv_memory_s: float = 70.0   # calibrated CV lump decay is faster
    tau_aux_ddr_memory_s: float = 110.0 # calibrated DDR lump decay (3x CV tau)
    aux_cv_memory_hold_c: float = 2.5
    aux_ddr_memory_hold_c: float = 4.5
    aux_cv_weight: float = 0.60  # reported AUX = w*CV + (1-w)*DDRQ


    # (6) Sensor bias (°C): affects fan control + throttling + sysfs reported temps.
    bias_cpu_c: float = 0.8
    bias_gpu_c: float = -0.5
    bias_aux_c: float = 1.2
    bias_ttp_c: float = 0.0

    # Measurement realism knobs
    telemetry_jitter_ms: float = 5.0
    temp_cpu_update_ms: float = 100.0
    temp_gpu_update_ms: float = 100.0
    temp_aux_update_ms: float = 250.0
    temp_ttp_update_ms: float = 250.0
    temp_board_update_ms: float = 1000.0
    power_update_ms: float = 120.0

    # Control surfaces
    jetson_clocks: bool = False

    # Domain randomization / device instance overrides
    device_instance: Optional[Dict[str, float]] = None
    leak_scale_cpu: float = 1.0
    leak_scale_gpu: float = 1.0
    leak_scale_soc: float = 1.0
    leak_scale_cv: float = 1.0
    leak_scale_vddrq: float = 1.0

    # Sensor dropout realism
    sensor_dropout_prob: float = 0.0   # approx per-second probability
    sensor_dropout_duration_s: float = 2.0
    sensor_dropout_groups: Tuple[str, ...] = ("temps", "power", "fan")

    # Benign confounder events
    benign_throttle_event_rate_hz: float = 0.0
    benign_throttle_event_duration_s: float = 8.0
    benign_throttle_temp_delta_c: float = 6.0
    benign_throttle_fan_gain: float = 0.7

    perf_multiplex_event_rate_hz: float = 0.0
    perf_multiplex_event_duration_s: float = 5.0
    perf_multiplex_visible: Tuple[str, ...] = ("cycles", "instructions", "cache-misses")

    def __post_init__(self) -> None:
        if self.device_instance:
            self._apply_device_instance(dict(self.device_instance))

    def _apply_device_instance(self, inst: Dict[str, float]) -> None:
        theta_scale = float(inst.get("theta_scale", 1.0))
        if theta_scale != 1.0:
            self.theta_heatsink_eff_c_per_w *= theta_scale
            self.theta_jp_balanced_c_per_w *= theta_scale
            self.theta_jp_unbalanced_c_per_w *= theta_scale
            self.theta_jb_c_per_w *= theta_scale

        tau_scale = float(inst.get("tau_scale", 1.0))
        if tau_scale != 1.0:
            tau_fields = [
                "tau_ttp_s", "tau_board_s", "tau_cpu_s", "tau_cpu_cluster_s",
                "tau_gpu_s", "tau_aux_s", "tau_aux_cv_s", "tau_aux_ddr_s",
                "tau_aux_memory_s", "tau_aux_cv_memory_s", "tau_aux_ddr_memory_s",
            ]
            for name in tau_fields:
                setattr(self, name, getattr(self, name) * tau_scale)

        if "bias_cpu_c" in inst:
            self.bias_cpu_c = float(inst["bias_cpu_c"])
        if "bias_gpu_c" in inst:
            self.bias_gpu_c = float(inst["bias_gpu_c"])
        if "bias_aux_c" in inst:
            self.bias_aux_c = float(inst["bias_aux_c"])
        if "bias_ttp_c" in inst:
            self.bias_ttp_c = float(inst["bias_ttp_c"])

        fan_gain = float(inst.get("fan_gain", 1.0))
        if fan_gain != 1.0:
            self.fan_cooling_min_gain *= fan_gain
            self.fan_cooling_max_gain *= fan_gain
            self.fan_cooling_max_gain = clamp(self.fan_cooling_max_gain, 0.05, 1.5)

        self.leak_scale_cpu *= float(inst.get("leak_scale_cpu", 1.0))
        self.leak_scale_gpu *= float(inst.get("leak_scale_gpu", 1.0))
        self.leak_scale_soc *= float(inst.get("leak_scale_soc", 1.0))
        self.leak_scale_cv *= float(inst.get("leak_scale_cv", 1.0))
        self.leak_scale_vddrq *= float(inst.get("leak_scale_vddrq", 1.0))

        self.device_instance = {k: float(v) for k, v in inst.items()}


@dataclass
class SimState:
    t_unix_s: float
    t_sim_s: float
    mode: str
    trojan_active: bool
    trojan_phase_s: float

    # Util (%)
    cpu_util: List[float]
    gpu_util: float
    emc_util: float
    cv_util: float

    # Frequencies
    cpu_freq_mhz: List[int]
    gpu_freq_mhz: int
    emc_freq_mhz: int

    # Memory
    ram_used_mb: float
    ram_total_mb: int
    swap_used_mb: float
    swap_total_mb: int

    # Thermal sensors (reported temps)
    temp_cpu_c: float
    temp_cpu_clusters_c: List[float]
    temp_gpu_c: float
    temp_aux_c: float
    temp_aux_cv_c: float
    temp_aux_ddr_c: float
    temp_ttp_c: float
    temp_board_c: float

    # Virtual fan estimate
    fan_est_temp_c: float

    # Fan hysteresis state
    hysteresis_state: int  # 0..4 corresponding to PWM [0,77,120,160,255]
    fan_pwm: int
    fan_rpm: int

    # Powers (mW) - exposed via INA averaged values
    p_cpu_mw: float      # VDD_CPU
    p_gpu_mw: float      # VDD_GPU
    p_soc_mw: float      # VDD_SOC
    p_cv_mw: float       # VDD_CV
    p_vddrq_mw: float    # VDD_VDDRQ
    p_sys5v_mw: float    # VDD_SYS5V (total module input)

    # Back-compat alias (many tools expect "input power")
    p_in_mw: float

    # Perf totals
    counters_total: Dict[str, float]

    # Missingness / staleness masks (1=fresh update, 0=stale/reused)
    mask_temp_cpu: int = 1
    mask_temp_gpu: int = 1
    mask_temp_aux: int = 1
    mask_temp_ttp: int = 1
    mask_temp_board: int = 1
    mask_fan_est_temp: int = 1
    mask_power_cpu: int = 1
    mask_power_gpu: int = 1
    mask_power_soc: int = 1
    mask_power_cv: int = 1
    mask_power_vddrq: int = 1
    mask_power_sys5v: int = 1
    mask_power_in: int = 1
    perf_masks: Dict[str, int] = field(default_factory=dict)


class ThermalReset(Exception):
    pass

class ThermalHardStop(Exception):
    pass


class SensorRealism:
    """Applies sensor staleness, masks, and dropout faults."""

    SENSOR_PERIODS_MS = {
        "temp_cpu_c": lambda cfg: cfg.temp_cpu_update_ms,
        "temp_gpu_c": lambda cfg: cfg.temp_gpu_update_ms,
        "temp_aux_c": lambda cfg: cfg.temp_aux_update_ms,
        "temp_ttp_c": lambda cfg: cfg.temp_ttp_update_ms,
        "temp_board_c": lambda cfg: cfg.temp_board_update_ms,
        "fan_est_temp_c": lambda cfg: cfg.temp_aux_update_ms,
    }

    SENSOR_MASKS = {
        "temp_cpu_c": "mask_temp_cpu",
        "temp_gpu_c": "mask_temp_gpu",
        "temp_aux_c": "mask_temp_aux",
        "temp_ttp_c": "mask_temp_ttp",
        "temp_board_c": "mask_temp_board",
        "fan_est_temp_c": "mask_fan_est_temp",
    }

    POWER_KEYS = [
        "p_cpu_mw",
        "p_gpu_mw",
        "p_soc_mw",
        "p_cv_mw",
        "p_vddrq_mw",
        "p_sys5v_mw",
        "p_in_mw",
    ]

    POWER_MASKS = {
        "p_cpu_mw": "mask_power_cpu",
        "p_gpu_mw": "mask_power_gpu",
        "p_soc_mw": "mask_power_soc",
        "p_cv_mw": "mask_power_cv",
        "p_vddrq_mw": "mask_power_vddrq",
        "p_sys5v_mw": "mask_power_sys5v",
        "p_in_mw": "mask_power_in",
    }

    DROPOUT_GROUPS_MAP = {
        "temps": ["temp_cpu_c", "temp_gpu_c", "temp_aux_c", "temp_ttp_c", "temp_board_c"],
        "fan": ["fan_est_temp_c"],
        "power": POWER_KEYS,
    }

    def __init__(self, cfg: SimConfig, rng_seed: Optional[int] = None):
        self.cfg = cfg
        self.sensor_cache: Dict[str, float] = {}
        self.sensor_last: Dict[str, float] = {}
        self.power_cache: Dict[str, float] = {}
        self.power_last: Optional[float] = None
        self.dropout_rng = random.Random(rng_seed if rng_seed is not None else int(cfg.seed) + 42421)
        self.active_dropouts: Dict[str, float] = {}
        self._last_t: Optional[float] = None
        self._groups_signature: Tuple[str, ...] = tuple(cfg.sensor_dropout_groups)
        self.dropout_groups: Dict[str, List[str]] = self._build_dropout_group_map()

    def _build_dropout_group_map(self) -> Dict[str, List[str]]:
        groups: Dict[str, List[str]] = {}
        for name in self.cfg.sensor_dropout_groups:
            keys = [k for k in self.DROPOUT_GROUPS_MAP.get(name, [])]
            if keys:
                groups[name] = keys
        return groups

    def _refresh_groups_if_needed(self) -> None:
        sig = tuple(self.cfg.sensor_dropout_groups)
        if sig != self._groups_signature:
            self._groups_signature = sig
            self.dropout_groups = self._build_dropout_group_map()
            self.active_dropouts.clear()

    def apply(self, snap: Dict[str, Any], t_sim_s: float) -> Dict[str, Any]:
        self._refresh_groups_if_needed()
        snap = self._apply_staleness(snap, t_sim_s)
        snap = self._apply_dropouts(snap, t_sim_s)
        return snap

    def _apply_staleness(self, snap: Dict[str, Any], t_sim_s: float) -> Dict[str, Any]:
        now = float(t_sim_s)
        for key, period_fn in self.SENSOR_PERIODS_MS.items():
            period_ms = period_fn(self.cfg)
            period_s = max(1e-6, float(period_ms) / 1000.0)
            last = self.sensor_last.get(key)
            should_refresh = (last is None) or (now - last >= period_s)
            if should_refresh:
                if key in snap:
                    self.sensor_cache[key] = snap[key]
                self.sensor_last[key] = now
                snap[self.SENSOR_MASKS[key]] = 1
            else:
                cached = self.sensor_cache.get(key)
                if cached is not None:
                    snap[key] = cached
                    snap[self.SENSOR_MASKS[key]] = 0
                else:
                    snap[self.SENSOR_MASKS[key]] = 1

        power_period_s = max(1e-6, float(self.cfg.power_update_ms) / 1000.0)
        should_refresh_power = (self.power_last is None) or (now - self.power_last >= power_period_s)
        if should_refresh_power:
            for key in self.POWER_KEYS:
                if key in snap:
                    self.power_cache[key] = snap[key]
                mask = self.POWER_MASKS[key]
                snap[mask] = 1
            self.power_last = now
        else:
            for key in self.POWER_KEYS:
                cached = self.power_cache.get(key)
                if cached is not None:
                    snap[key] = cached
                snap[self.POWER_MASKS[key]] = 0
        return snap

    def _apply_dropouts(self, snap: Dict[str, Any], t_sim_s: float) -> Dict[str, Any]:
        self._update_dropout_timers(t_sim_s)
        if not self.active_dropouts:
            return snap

        for group, remaining in self.active_dropouts.items():
            keys = self.dropout_groups.get(group, [])
            for key in keys:
                if key in self.POWER_MASKS:
                    mask_key = self.POWER_MASKS[key]
                    cache = self.power_cache
                else:
                    mask_key = self.SENSOR_MASKS.get(key)
                    cache = self.sensor_cache
                if mask_key:
                    snap[mask_key] = 0
                # Robust policy: use cache if available, else set to NaN
                # This prevents mask=0 but fresh value inconsistencies
                if key in cache:
                    snap[key] = cache[key]
                else:
                    snap[key] = float("nan")
        return snap

    def _update_dropout_timers(self, t_sim_s: float) -> None:
        now = float(t_sim_s)
        if self._last_t is None:
            self._last_t = now
        dt = max(0.0, now - (self._last_t or now))
        self._last_t = now

        expired = [g for g, remaining in self.active_dropouts.items() if remaining - dt <= 0.0]
        for g in expired:
            self.active_dropouts.pop(g, None)
        for g in list(self.active_dropouts.keys()):
            self.active_dropouts[g] = max(0.0, self.active_dropouts[g] - dt)

        rate = max(0.0, float(self.cfg.sensor_dropout_prob))
        if rate <= 0.0 or not self.dropout_groups:
            return

        available = [g for g in self.dropout_groups if g not in self.active_dropouts]
        if not available or dt <= 0.0:
            return

        trigger_p = clamp(rate * dt, 0.0, 0.95)
        if self.dropout_rng.random() < trigger_p:
            group = self.dropout_rng.choice(available)
            duration = max(1e-6, float(self.cfg.sensor_dropout_duration_s))
            self.active_dropouts[group] = duration


# ---------------------------- model ----------------------------

class XavierTDGModel:
    FAN_PWM_TABLE = [0, 77, 120, 160, 255]

    def __init__(self, cfg: SimConfig):
        self.cfg = cfg
        self.rng = random.Random(cfg.seed)
        self._perf_visible_signature: Tuple[str, ...] = tuple(cfg.perf_multiplex_visible)
        visible = {e for e in cfg.perf_multiplex_visible if e in SUPPORTED_EVENTS}
        self._perf_visible: Set[str] = visible if visible else set(SUPPORTED_EVENTS)
        self._benign_throttle_left_s = 0.0
        self._perf_multiplex_left_s = 0.0
        self._current_perf_masks: Dict[str, int] = {e: 1 for e in SUPPORTED_EVENTS}

        self.mode = "idle"
        # Deterministic sim time (advanced only by provided dt)
        self.t_sim_s = 0.0
        # Back-compat wall-clock stepping support (used by HTTP daemon mode)
        self._last_step_unix_s = now_s()

        self.trojan_phase_s = 0.0
        self.trojan_active = False

        # Workload state
        self.bg_spike_left_s = 0.0

        # Utilization state
        self.cpu_util = [2.0] * N_CPU
        self.gpu_util = 1.0
        self.emc_util = 3.0
        self.cv_util = 0.5

        # DVFS indices
        self.cpu_step_idx = 2
        self.gpu_step_idx = 0
        self.emc_step_idx = 0

        # Thermal throttling caps (start at mode caps)
        self.cap_cpu_idx = cfg.cpu_max_step_idx
        self.cap_gpu_idx = cfg.gpu_max_step_idx
        self.cap_emc_idx = cfg.emc_max_step_idx
        self._throttle_timer_s = 0.0
        self._restore_timer_s = 0.0

        # Memory
        self.ram_used_mb = 1800.0
        self._ram_wander = 0.0

        # Thermal nodes (internal "true" temps, without bias)
        amb = cfg.ambient_c
        self.temp_ttp_c = amb + 6.0
        self.temp_board_c = amb + 2.0
        # Per-cluster CPU temps (4 clusters of 2 cores each)
        self.temp_cpu_clusters = [amb + 10.0 for _ in range(4)]
        self.temp_cpu_c = max(self.temp_cpu_clusters)
        self.temp_gpu_c = amb + 8.0
        # Two-lump AUX internals
        self.temp_aux_cv_c = amb + 7.0   # CV / DLA / PVA
        self.temp_aux_ddr_c = amb + 7.0  # DRAM hot spot
        # memory state for each lump
        self.aux_cv_mem_c = self.temp_aux_cv_c
        self.aux_ddr_mem_c = self.temp_aux_ddr_c
        # reported AUX
        self.temp_aux_c = max(self.temp_aux_cv_c, self.temp_aux_ddr_c)

        # Fan
        self.hysteresis_state = 0
        self.fan_pwm = 0
        self.fan_rpm = 0
        self.fan_est_temp_c = (3*self._meas_cpu_c() + 3*self._meas_gpu_c() + 4*self._meas_aux_c()) / 10.0

        # INA rolling averages
        n = int(cfg.ina_window_samples)
        self.ina_vdd_gpu = RollingAvg(n)
        self.ina_vdd_cpu = RollingAvg(n)
        self.ina_vdd_soc = RollingAvg(n)
        self.ina_vdd_cv = RollingAvg(n)
        self.ina_vddrq = RollingAvg(n)
        self.ina_sys5v = RollingAvg(n)

        # Initialize with small baseline powers (mW)
        self.p_cpu_raw = 700.0
        self.p_gpu_raw = 400.0
        self.p_soc_raw = 1200.0
        self.p_cv_raw = 120.0
        self.p_vddrq_raw = 250.0
        self.p_sys5v_raw = 2500.0

        for _ in range(n):
            self.ina_vdd_gpu.push(self.p_gpu_raw)
            self.ina_vdd_cpu.push(self.p_cpu_raw)
            self.ina_vdd_soc.push(self.p_soc_raw)
            self.ina_vdd_cv.push(self.p_cv_raw)
            self.ina_vddrq.push(self.p_vddrq_raw)
            self.ina_sys5v.push(self.p_sys5v_raw)

        self.counters_total = {e: 0.0 for e in SUPPORTED_EVENTS}
        # convenience: cluster -> core indices (2 cores per cluster)
        self._cluster_core_idx = [[0,1],[2,3],[4,5],[6,7]]

    # ---------------- measured temps (bias) ----------------

    def _meas_ttp_c(self) -> float:
        return self.temp_ttp_c + float(self.cfg.bias_ttp_c)

    def _meas_cpu_c(self) -> float:
        # reported CPU sensor is the hottest cluster (Tj)
        self.temp_cpu_c = max(self.temp_cpu_clusters)
        return self.temp_cpu_c + float(self.cfg.bias_cpu_c)

    def _meas_gpu_c(self) -> float:
        return self.temp_gpu_c + float(self.cfg.bias_gpu_c)

    def _meas_aux_c(self) -> float:
        return self.temp_aux_c + float(self.cfg.bias_aux_c)

    # ---------------- mode/scheduling ----------------

    def set_mode(self, mode: str) -> None:
        if mode not in ("idle", "normal", "trojan"):
            raise ValueError("mode must be idle|normal|trojan")
        self.mode = mode
        if mode != "trojan":
            self.trojan_active = False
            self.trojan_phase_s = 0.0

    def _trojan_schedule(self, dt: float) -> None:
        if self.mode != "trojan":
            self.trojan_active = False
            self.trojan_phase_s = 0.0
            return
        self.trojan_phase_s += dt
        period = max(1e-3, self.cfg.trojan_period_s)
        on = clamp(self.cfg.trojan_on_s, 0.0, period)
        phase = (self.trojan_phase_s % period)
        self.trojan_active = (phase < on)

    def _frame_wave(self, t_sim: float) -> Tuple[float, float]:
        fr = max(1e-3, self.cfg.frame_rate_hz)
        phase = (t_sim * fr) % 1.0
        gpu_duty = clamp((self.cfg.frame_gpu_busy_ms / 1000.0) * fr, 0.0, 0.98)
        cpu_duty = clamp((self.cfg.frame_cpu_busy_ms / 1000.0) * fr, 0.0, 0.98)
        gpu_on = 1.0 if phase < gpu_duty else 0.0
        cpu_on = 1.0 if phase < cpu_duty else 0.0
        gpu_on = clamp(gpu_on + self.rng.gauss(0.0, 0.12), 0.0, 1.0)
        cpu_on = clamp(cpu_on + self.rng.gauss(0.0, 0.10), 0.0, 1.0)
        return cpu_on, gpu_on

    def _maybe_start_bg_spike(self, dt: float) -> None:
        if self.bg_spike_left_s > 0.0:
            return
        p = clamp(self.cfg.bg_spike_rate_hz * dt, 0.0, 0.8)
        if self.rng.random() < p:
            self.bg_spike_left_s = self.cfg.bg_spike_ms / 1000.0

    def _update_benign_events(self, dt: float) -> None:
        self._benign_throttle_left_s = max(0.0, self._benign_throttle_left_s - dt)
        self._perf_multiplex_left_s = max(0.0, self._perf_multiplex_left_s - dt)

        benign_context = (self.mode != "trojan") or (self.mode == "trojan" and not self.trojan_active)
        if not benign_context:
            return

        thr_rate = max(0.0, float(self.cfg.benign_throttle_event_rate_hz))
        if self._benign_throttle_left_s <= 0.0 and thr_rate > 0.0:
            if self.rng.random() < clamp(thr_rate * dt, 0.0, 0.95):
                self._benign_throttle_left_s = max(1e-6, float(self.cfg.benign_throttle_event_duration_s))

        perf_rate = max(0.0, float(self.cfg.perf_multiplex_event_rate_hz))
        if self._perf_multiplex_left_s <= 0.0 and perf_rate > 0.0:
            if self.rng.random() < clamp(perf_rate * dt, 0.0, 0.95):
                self._perf_multiplex_left_s = max(1e-6, float(self.cfg.perf_multiplex_event_duration_s))

    def _apply_util_dynamics(self, dt: float, t_sim: float) -> None:
        """Update utilization targets.

        Workload families must change correlation structure (not only offsets).
        """
        fam = str(self.cfg.workload_family or "inference_periodic")

        if self.mode == "idle":
            cpu_t, gpu_t, emc_t, cv_t = (4.0, 1.0, 4.0, 0.5)
        else:
            # Baseline demand
            cpu_t, gpu_t, emc_t, cv_t = (18.0, 28.0, 20.0, 6.0)

            if fam == "streaming_steady_gpu":
                # Steady GPU, smoother EMC coupling, less bursty CPU
                cpu_b01, gpu_b01 = self._frame_wave(t_sim)
                cpu_t += 10.0 * cpu_b01
                gpu_t += 55.0 * (0.75 + 0.25 * gpu_b01)
                emc_t += 16.0 * (0.15 * cpu_b01 + 0.85 * gpu_b01)
                cv_t += 3.0 * (0.1 * cpu_b01 + 0.9 * gpu_b01)
            elif fam == "memory_bound":
                # EMC dominates, GPU moderate, CPU moderate; more cache pressure
                cpu_b01, gpu_b01 = self._frame_wave(t_sim)
                cpu_t += 14.0 * cpu_b01
                gpu_t += 26.0 * gpu_b01
                emc_t += 40.0 * (0.55 * cpu_b01 + 0.45 * gpu_b01)
                cv_t += 2.0 * gpu_b01
            elif fam == "network_bursty":
                # CPU burstiness + context/page-fault spikes (handled in perf), moderate EMC
                cpu_b01, gpu_b01 = self._frame_wave(t_sim)
                cpu_t += 30.0 * cpu_b01
                gpu_t += 14.0 * gpu_b01
                emc_t += 14.0 * (0.75 * cpu_b01 + 0.25 * gpu_b01)
                cv_t += 1.5 * cpu_b01
            elif fam == "cv_heavy":
                # CV rail dominates, AUX correlation stronger, GPU modest
                cpu_b01, gpu_b01 = self._frame_wave(t_sim)
                cpu_t += 8.0 * cpu_b01
                gpu_t += 10.0 * gpu_b01
                emc_t += 18.0 * (0.25 * cpu_b01 + 0.75 * gpu_b01)
                cv_t += 28.0 * (0.3 * cpu_b01 + 0.7 * gpu_b01)
            else:
                # inference_periodic (default)
                cpu_b01, gpu_b01 = self._frame_wave(t_sim)
                cpu_t += 22.0 * cpu_b01
                gpu_t += 44.0 * gpu_b01
                emc_t += 18.0 * (0.35 * cpu_b01 + 0.65 * gpu_b01)
                cv_t += 6.0 * (0.2 * cpu_b01 + 0.8 * gpu_b01)

        self._maybe_start_bg_spike(dt)
        if self.bg_spike_left_s > 0.0:
            cpu_t += self.cfg.bg_spike_cpu_pct
            emc_t += 6.0
            self.bg_spike_left_s = max(0.0, self.bg_spike_left_s - dt)

        # Trojan styles
        if self.mode == "trojan" and self.trojan_active:
            s = clamp(self.cfg.trojan_strength, 0.1, 3.0)
            if self.cfg.trojan_style == "compute":
                cpu_t += 16.0 * s
                gpu_t += 10.0 * s
                emc_t += 10.0 * s
                cv_t += 6.0 * s
            elif self.cfg.trojan_style == "cv":
                cv_t += 38.0 * s
                emc_t += 16.0 * s
                cpu_t += 4.0 * s
                gpu_t += 2.0 * s
            elif self.cfg.trojan_style == "io":
                cpu_t += 14.0 * s
                gpu_t += 1.0 * s
                emc_t += 8.0 * s
                cv_t += 1.0 * s
            else:  # mempressure
                cpu_t += 5.0 * s
                gpu_t += 2.5 * s
                emc_t += 22.0 * s
                cv_t += 7.0 * s

        # Demand/capacity feedback: lower freq -> higher util at same demand
        cpu_f = CPU_FREQ_STEPS_MHZ[self.cpu_step_idx]
        gpu_f = GPU_FREQ_STEPS_MHZ[self.gpu_step_idx]
        emc_f = EMC_FREQ_STEPS_MHZ[self.emc_step_idx]

        cpu_scale = max(CPU_FREQ_STEPS_MHZ) / max(1e-6, cpu_f)
        gpu_scale = max(GPU_FREQ_STEPS_MHZ) / max(1e-6, gpu_f)
        emc_scale = max(EMC_FREQ_STEPS_MHZ) / max(1e-6, emc_f)

        cpu_t *= cpu_scale ** 0.7
        gpu_t *= gpu_scale ** 0.7
        emc_t *= emc_scale ** 0.7
        cv_t  *= emc_scale ** 0.4

        alpha = clamp(dt * 2.2, 0.03, 0.35)

        avg_cpu = sum(self.cpu_util) / len(self.cpu_util)
        avg_cpu = (1 - alpha) * avg_cpu + alpha * cpu_t + self.rng.gauss(0.0, self.cfg.util_noise_sigma)
        base = clamp(avg_cpu, 0.0, 100.0)

        new = []
        for i in range(N_CPU):
            jitter = self.rng.gauss(0.0, 2.8)
            bias = 2.5 if i < 2 else (1.0 if i < 4 else 0.0)
            new.append(clamp(base + bias + jitter, 0.0, 100.0))
        self.cpu_util = new

        self.gpu_util = clamp((1 - alpha) * self.gpu_util + alpha * gpu_t + self.rng.gauss(0.0, 2.2), 0.0, 100.0)
        self.emc_util = clamp((1 - alpha) * self.emc_util + alpha * emc_t + self.rng.gauss(0.0, 1.8), 0.0, 100.0)
        self.cv_util = clamp((1 - alpha) * self.cv_util + alpha * cv_t + self.rng.gauss(0.0, 1.4), 0.0, 100.0)

        # RAM correlated with GPU/EMC and some wander
        ram_target = 1650.0 if self.mode == "idle" else (2350.0 + 4.5 * self.gpu_util + 2.0 * self.emc_util + 1.2 * self.cv_util)
        self._ram_wander = clamp(0.98 * self._ram_wander + self.rng.gauss(0.0, 6.0), -180.0, 220.0)
        if self.rng.random() < clamp(0.015 * dt, 0.0, 0.05):
            self._ram_wander -= self.rng.uniform(60.0, 140.0)
        ram_target += self._ram_wander
        alpha_ram = clamp(dt * 0.25, 0.01, 0.08)
        self.ram_used_mb = clamp((1 - alpha_ram) * self.ram_used_mb + alpha_ram * ram_target, 800.0, RAM_TOTAL_MB - 650.0)

    # ---------------- DVFS governor ----------------

    def _rate_limit(self, cur: float, target: float, rate_per_s: float, dt: float) -> float:
        delta = target - cur
        max_delta = rate_per_s * dt
        if abs(delta) <= max_delta:
            return target
        return cur + math.copysign(max_delta, delta)

    def _governor(self, dt: float) -> None:
        if self.cfg.jetson_clocks:
            self.cpu_step_idx = int(clamp(self.cap_cpu_idx, 0, self.cfg.cpu_max_step_idx))
            self.gpu_step_idx = int(clamp(self.cap_gpu_idx, 0, self.cfg.gpu_max_step_idx))
            self.emc_step_idx = int(clamp(self.cap_emc_idx, 0, self.cfg.emc_max_step_idx))
            return

        avg_cpu_u01 = (sum(self.cpu_util) / len(self.cpu_util)) / 100.0
        gpu_u01 = self.gpu_util / 100.0
        emc_u01 = self.emc_util / 100.0

        cpu_demand = clamp(0.08 + 1.15 * avg_cpu_u01, 0.0, 1.0)
        gpu_demand = clamp(0.05 + 1.10 * gpu_u01, 0.0, 1.0)
        emc_demand = clamp(0.10 + 1.05 * emc_u01, 0.0, 1.0)

        cpu_cap = int(clamp(self.cap_cpu_idx, 0, self.cfg.cpu_max_step_idx))
        gpu_cap = int(clamp(self.cap_gpu_idx, 0, self.cfg.gpu_max_step_idx))
        emc_cap = int(clamp(self.cap_emc_idx, 0, self.cfg.emc_max_step_idx))

        cpu_tgt = choose_step_idx(CPU_FREQ_STEPS_MHZ, cpu_demand, max_idx=cpu_cap)
        gpu_tgt = choose_step_idx(GPU_FREQ_STEPS_MHZ, gpu_demand, max_idx=gpu_cap)
        emc_tgt = choose_step_idx(EMC_FREQ_STEPS_MHZ, emc_demand, max_idx=emc_cap)

        self.cpu_step_idx = int(round(self._rate_limit(self.cpu_step_idx, cpu_tgt, self.cfg.cpu_step_rate_per_s, dt)))
        self.gpu_step_idx = int(round(self._rate_limit(self.gpu_step_idx, gpu_tgt, self.cfg.gpu_step_rate_per_s, dt)))
        self.emc_step_idx = int(round(self._rate_limit(self.emc_step_idx, emc_tgt, self.cfg.emc_step_rate_per_s, dt)))

        self.cpu_step_idx = int(clamp(self.cpu_step_idx, 0, cpu_cap))
        self.gpu_step_idx = int(clamp(self.gpu_step_idx, 0, gpu_cap))
        self.emc_step_idx = int(clamp(self.emc_step_idx, 0, emc_cap))

    # ---------------- fan step/hysteresis ----------------

    def _update_fan_state(self) -> None:
        """
        Implements fan state machine + hysteresis.
        Uses thermal-fan-est = (3*CPU + 3*GPU + 4*AUX)/10.

        Realism tweak: fan estimate uses biased sensor temps (so fan behavior matches reported temps).
        """
        t_cpu = self._meas_cpu_c()
        t_gpu = self._meas_gpu_c()
        t_aux = self._meas_aux_c()
        self.fan_est_temp_c = (3.0 * t_cpu + 3.0 * t_gpu + 4.0 * t_aux) / 10.0

        t = self.fan_est_temp_c
        s = self.hysteresis_state

        # Upward transitions
        if t >= 81.0:
            s = 4
        elif t >= 72.0:
            s = max(s, 3)
        elif t >= 63.0:
            s = max(s, 2)
        elif t >= 50.0:
            s = max(s, 1)

        # Downward transitions
        if s == 4 and t < 73.0:
            s = 3
        if s == 3 and t < 64.0:
            s = 2
        if s == 2 and t < 55.0:
            s = 1
        if s == 1 and t < 32.0:
            s = 0

        self.hysteresis_state = int(clamp(s, 0, 4))
        self.fan_pwm = int(self.FAN_PWM_TABLE[self.hysteresis_state])

        if self.fan_pwm <= 0:
            self.fan_rpm = 0
        else:
            self.fan_rpm = int(max(0.0, 900.0 + self.fan_pwm * 18.0 + self.rng.gauss(0.0, 35.0)))

    # ---------------- power model ----------------

    def _voltage_ratio(self, f_mhz: float, f_max_mhz: float) -> float:
        r = clamp(f_mhz / max(1e-6, f_max_mhz), 0.0, 1.0)
        return 0.75 + 0.35 * (r ** 0.7)

    def _compute_raw_rails_mw(self) -> Tuple[float, float, float, float, float, float]:
        """
        Returns raw (instantaneous) powers in mW:
          (VDD_GPU, VDD_CPU, VDD_SOC, VDD_CV, VDD_VDDRQ, VDD_SYS5V)
        """
        max_cpu = max(CPU_FREQ_STEPS_MHZ)
        max_gpu = max(GPU_FREQ_STEPS_MHZ)
        max_emc = max(EMC_FREQ_STEPS_MHZ)

        avg_cpu_u = (sum(self.cpu_util) / len(self.cpu_util)) / 100.0
        gpu_u = self.gpu_util / 100.0
        emc_u = self.emc_util / 100.0
        cv_u = self.cv_util / 100.0

        cpu_f = CPU_FREQ_STEPS_MHZ[self.cpu_step_idx]
        gpu_f = GPU_FREQ_STEPS_MHZ[self.gpu_step_idx]
        emc_f = EMC_FREQ_STEPS_MHZ[self.emc_step_idx]

        v_cpu = self._voltage_ratio(cpu_f, max_cpu)
        v_gpu = self._voltage_ratio(gpu_f, max_gpu)
        v_emc = self._voltage_ratio(emc_f, max_emc)

        leak_cpu = (240.0 + 5.5 * max(0.0, self.temp_cpu_c - self.cfg.ambient_c)) * float(self.cfg.leak_scale_cpu)
        leak_gpu = (170.0 + 4.0 * max(0.0, self.temp_gpu_c - self.cfg.ambient_c)) * float(self.cfg.leak_scale_gpu)
        leak_soc = (520.0 + 2.8 * max(0.0, self.temp_ttp_c - self.cfg.ambient_c)) * float(self.cfg.leak_scale_soc)
        leak_cv  = (90.0 + 2.0 * max(0.0, self.temp_aux_c - self.cfg.ambient_c)) * float(self.cfg.leak_scale_cv)

        cpu_dyn = 7200.0 * (v_cpu ** 2) * (cpu_f / max_cpu) * clamp(avg_cpu_u, 0.0, 1.0)
        gpu_dyn = 9800.0 * (v_gpu ** 2) * (gpu_f / max_gpu) * clamp(gpu_u, 0.0, 1.0)
        ddr_dyn = 2200.0 * (v_emc ** 2) * (emc_f / max_emc) * clamp(emc_u, 0.0, 1.0)
        cv_dyn  = 2600.0 * (0.65 + 0.35 * (emc_f / max_emc)) * clamp(cv_u, 0.0, 1.0)

        soc_dyn = 2100.0 * (
            0.45 * clamp(emc_u, 0.0, 1.0)
            + 0.20 * clamp(gpu_u, 0.0, 1.0)
            + 0.18 * clamp(avg_cpu_u, 0.0, 1.0)
            + 0.12 * clamp(cv_u, 0.0, 1.0)
        )

        p_cpu = leak_cpu + cpu_dyn
        p_gpu = leak_gpu + gpu_dyn
        p_soc = leak_soc + soc_dyn
        p_cv  = leak_cv + cv_dyn
        p_vddrq = (120.0 + ddr_dyn) * float(self.cfg.leak_scale_vddrq)

        # Soft cap with rail-weighted compression (realism tweak)
        total_components_w = (p_cpu + p_gpu + p_soc + p_cv + p_vddrq) / 1000.0
        if total_components_w > self.cfg.power_cap_w:
            scale = self.cfg.power_cap_w / max(1e-6, total_components_w)

            def cap_dyn(p: float, base: float, exp: float) -> float:
                dyn = max(0.0, p - base)
                return base + dyn * (scale ** float(exp))

            p_cpu   = cap_dyn(p_cpu,   leak_cpu, self.cfg.power_cap_exp_cpu)
            p_gpu   = cap_dyn(p_gpu,   leak_gpu, self.cfg.power_cap_exp_gpu)
            p_cv    = cap_dyn(p_cv,    leak_cv,  self.cfg.power_cap_exp_cv)
            p_soc   = cap_dyn(p_soc,   leak_soc, self.cfg.power_cap_exp_soc)
            p_vddrq = cap_dyn(p_vddrq, 120.0,    self.cfg.power_cap_exp_vddrq)

        # Add small instantaneous noise (will be hidden by INA averaging)
        p_cpu += self.rng.gauss(0.0, 18.0)
        p_gpu += self.rng.gauss(0.0, 20.0)
        p_soc += self.rng.gauss(0.0, 22.0)
        p_cv  += self.rng.gauss(0.0, 10.0)
        p_vddrq += self.rng.gauss(0.0, 10.0)

        p_cpu = max(0.0, p_cpu)
        p_gpu = max(0.0, p_gpu)
        p_soc = max(0.0, p_soc)
        p_cv  = max(0.0, p_cv)
        p_vddrq = max(0.0, p_vddrq)

        # Input voltage inefficiency scalar (exact formula)
        vin = float(self.cfg.input_voltage_v)
        ineff = 1.0 + (0.015 * (vin - 9.0))
        p_sys5v = (p_cpu + p_gpu + p_soc + p_cv + p_vddrq) * ineff

        return (p_gpu, p_cpu, p_soc, p_cv, p_vddrq, p_sys5v)

    def _update_ina_averages(self) -> None:
        p_gpu, p_cpu, p_soc, p_cv, p_vddrq, p_sys5v = self._compute_raw_rails_mw()

        self.p_gpu_raw = p_gpu
        self.p_cpu_raw = p_cpu
        self.p_soc_raw = p_soc
        self.p_cv_raw = p_cv
        self.p_vddrq_raw = p_vddrq
        self.p_sys5v_raw = p_sys5v

        # approximate internal INA sampling cadence, decoupled from telemetry tick
        k = max(1, int(round(float(self.cfg.ina_internal_hz) / max(1e-6, self.cfg.hz))))
        for _ in range(k):
            ng = self.rng.gauss
            self.ina_vdd_gpu.push(max(0.0, p_gpu   + ng(0.0, 5.0)))
            self.ina_vdd_cpu.push(max(0.0, p_cpu   + ng(0.0, 5.0)))
            self.ina_vdd_soc.push(max(0.0, p_soc   + ng(0.0, 6.0)))
            self.ina_vdd_cv.push (max(0.0, p_cv    + ng(0.0, 3.0)))
            self.ina_vddrq.push  (max(0.0, p_vddrq + ng(0.0, 3.0)))
            self.ina_sys5v.push  (max(0.0, p_sys5v + ng(0.0, 8.0)))

    # ---------------- thermal network ----------------

    def _theta_jp_per_domain(self) -> Tuple[float, float, float]:
        """
        θjp balanced vs unbalanced.
        Realism tweak: optionally blend continuously between balanced and unbalanced
        based on concentration (dominance) of CPU/GPU/CV power fraction, and apply
        the blended theta to ALL domains.
        """
        p_cpu_w = self.p_cpu_raw / 1000.0
        p_gpu_w = self.p_gpu_raw / 1000.0
        p_cv_w  = self.p_cv_raw / 1000.0
        tot = p_cpu_w + p_gpu_w + p_cv_w
        b = self.cfg.theta_jp_balanced_c_per_w
        u = self.cfg.theta_jp_unbalanced_c_per_w

        if tot <= 1e-6:
            return (b, b, b)

        fr_cpu = p_cpu_w / tot
        fr_gpu = p_gpu_w / tot
        fr_cv  = p_cv_w / tot
        dom = max(fr_cpu, fr_gpu, fr_cv)

        if self.cfg.theta_jp_blend:
            knee = max(1e-6, float(self.cfg.theta_jp_blend_knee))
            span = max(1e-6, float(self.cfg.theta_jp_blend_span))
            x = clamp((dom - knee) / span, 0.0, 1.0)
            theta = b + (u - b) * x
            return (theta, theta, theta)

        # fallback: original heuristic
        if dom > 0.60:
            if fr_cpu == dom:
                return (u, b, b)
            if fr_gpu == dom:
                return (b, u, b)
            return (b, b, u)
        return (b, b, b)

    def _first_order_to_target(self, cur: float, target: float, tau: float, dt: float) -> float:
        tau = max(1e-6, tau)
        a = clamp(dt / tau, 0.0, 1.0)
        return cur + a * (target - cur)

    def _thermal_step(self, dt: float) -> None:
        """
        TTP = Tamb + Ptotal * theta_heatsink_eff
        Realism tweak: theta is modulated by fan PWM (cooling gain).
        Board: tracks TTP slowly via coupling factor.
        Junctions: parallel θjp and θjb to TTP and board.
        AUX: includes thermal memory.
        """
        amb = self.cfg.ambient_c
        if self._benign_throttle_left_s > 0.0:
            amb += float(self.cfg.benign_throttle_temp_delta_c)

        p_total_w = (self.p_cpu_raw + self.p_gpu_raw + self.p_soc_raw + self.p_cv_raw + self.p_vddrq_raw) / 1000.0

        # Fan-cooling coupling (uses previous-step fan_pwm -> natural lag)
        pwm01 = clamp(self.fan_pwm / 255.0, 0.0, 1.0)
        cg_min = float(self.cfg.fan_cooling_min_gain)
        cg_max = float(self.cfg.fan_cooling_max_gain)
        cooling_gain = clamp(cg_min + (cg_max - cg_min) * pwm01, 1e-3, 10.0)
        if self._benign_throttle_left_s > 0.0:
            throttle_gain = clamp(float(self.cfg.benign_throttle_fan_gain), 0.05, 1.0)
            cooling_gain *= throttle_gain
        theta_eff = self.cfg.theta_heatsink_eff_c_per_w / cooling_gain

        ttp_target = amb + p_total_w * theta_eff
        self.temp_ttp_c = self._first_order_to_target(self.temp_ttp_c, ttp_target, self.cfg.tau_ttp_s, dt)
        self.temp_ttp_c += self.rng.gauss(0.0, self.cfg.sensor_noise_sigma * 0.03)

        # Board tracks TTP, not power (avoids magic number)
        board_target = amb + float(self.cfg.board_ttp_coupling) * (self.temp_ttp_c - amb)
        self.temp_board_c = self._first_order_to_target(self.temp_board_c, board_target, self.cfg.tau_board_s, dt)
        self.temp_board_c += self.rng.gauss(0.0, self.cfg.sensor_noise_sigma * 0.02)

        theta_cpu, theta_gpu, theta_aux = self._theta_jp_per_domain()
        theta_jb = self.cfg.theta_jb_c_per_w

        def junction_target(p_w: float, t_plate: float, t_board: float, r_jp: float, r_jb: float) -> float:
            # P = (Tj-Tplate)/Rjp + (Tj-Tboard)/Rjb
            g1 = 1.0 / max(1e-9, r_jp)
            g2 = 1.0 / max(1e-9, r_jb)
            return (p_w + t_plate * g1 + t_board * g2) / (g1 + g2)

        # CPU: distribute p_cpu across 4 clusters according to util
        p_cpu_w = self.p_cpu_raw / 1000.0
        # compute cluster utilization fractions
        cluster_utils = []
        for cidx in self._cluster_core_idx:
            s = sum(self.cpu_util[i] for i in cidx) / (2.0 * 100.0)  # 0..1 fraction per cluster
            cluster_utils.append(max(1e-6, s))
        tot_c_frac = sum(cluster_utils)
        cluster_fracs = [c / tot_c_frac for c in cluster_utils]

        new_cluster_temps = []
        for ci, frac in enumerate(cluster_fracs):
            p_cl_w = p_cpu_w * frac
            cl_tgt = junction_target(p_cl_w, self.temp_ttp_c, self.temp_board_c, theta_cpu, theta_jb)
            # per-cluster tau
            self.temp_cpu_clusters[ci] = self._first_order_to_target(
                self.temp_cpu_clusters[ci], cl_tgt, self.cfg.tau_cpu_cluster_s, dt
            )
            self.temp_cpu_clusters[ci] += self.rng.gauss(0.0, self.cfg.sensor_noise_sigma * 0.04)
            new_cluster_temps.append(self.temp_cpu_clusters[ci])
        # effective CPU Tj is hottest cluster
        self.temp_cpu_c = max(new_cluster_temps)

        # GPU
        p_gpu_w = self.p_gpu_raw / 1000.0
        gpu_tgt = junction_target(p_gpu_w, self.temp_ttp_c, self.temp_board_c, theta_gpu, theta_jb)
        self.temp_gpu_c = self._first_order_to_target(self.temp_gpu_c, gpu_tgt, self.cfg.tau_gpu_s, dt)
        self.temp_gpu_c += self.rng.gauss(0.0, self.cfg.sensor_noise_sigma * 0.04)

        # Two-lump AUX model: CV lump and DDR lump (separate tau and memory)
        p_cv_w = self.p_cv_raw / 1000.0
        p_ddr_w = self.p_vddrq_raw / 1000.0

        cv_tgt = junction_target(p_cv_w, self.temp_ttp_c, self.temp_board_c, theta_aux, theta_jb)
        ddr_tgt = junction_target(p_ddr_w, self.temp_ttp_c, self.temp_board_c, theta_aux, theta_jb)

        # CV memory behavior
        if cv_tgt > self.aux_cv_mem_c:
            self.aux_cv_mem_c = cv_tgt
        else:
            self.aux_cv_mem_c = self._first_order_to_target(self.aux_cv_mem_c, cv_tgt, self.cfg.tau_aux_cv_memory_s, dt)
        cv_tgt_eff = max(cv_tgt, self.aux_cv_mem_c - float(self.cfg.aux_cv_memory_hold_c))
        self.temp_aux_cv_c = self._first_order_to_target(self.temp_aux_cv_c, cv_tgt_eff, self.cfg.tau_aux_cv_s, dt)
        self.temp_aux_cv_c += self.rng.gauss(0.0, self.cfg.sensor_noise_sigma * 0.03)

        # DDR memory behavior (retains more heat)
        if ddr_tgt > self.aux_ddr_mem_c:
            self.aux_ddr_mem_c = ddr_tgt
        else:
            self.aux_ddr_mem_c = self._first_order_to_target(self.aux_ddr_mem_c, ddr_tgt, self.cfg.tau_aux_ddr_memory_s, dt)
        ddr_tgt_eff = max(ddr_tgt, self.aux_ddr_mem_c - float(self.cfg.aux_ddr_memory_hold_c))
        self.temp_aux_ddr_c = self._first_order_to_target(self.temp_aux_ddr_c, ddr_tgt_eff, self.cfg.tau_aux_ddr_s, dt)
        self.temp_aux_ddr_c += self.rng.gauss(0.0, self.cfg.sensor_noise_sigma * 0.03)

        # reported AUX = weighted combination (configurable)
        w = clamp(float(self.cfg.aux_cv_weight), 0.0, 1.0)
        self.temp_aux_c = w * self.temp_aux_cv_c + (1.0 - w) * self.temp_aux_ddr_c

    # ---------------- throttling + reset/hard stop ----------------

    def _enforce_protection(self, dt: float) -> None:
        """
        Uses BIASED (measured) temps to decide throttling/reset/hard trip.
        """
        t_cpu = self._meas_cpu_c()
        t_gpu = self._meas_gpu_c()
        t_aux = self._meas_aux_c()
        t_ttp = self._meas_ttp_c()

        pm = self.cfg.power_mode.upper()
        if pm not in THERM_THRESH:
            pm = "30W"
        (cpu_rec, cpu_max) = THERM_THRESH[pm]["CPU"]
        (gpu_rec, gpu_max) = THERM_THRESH[pm]["GPU"]
        (aux_rec, aux_max) = THERM_THRESH[pm]["AUX"]

        if (t_cpu > HARD_TRIP_C) or (t_gpu > HARD_TRIP_C) or (t_aux > HARD_TRIP_C):
            raise ThermalHardStop(f"Hard stop: exceeded {HARD_TRIP_C}°C")

        if (t_cpu > cpu_max) or (t_gpu > gpu_max) or (t_aux > aux_max):
            raise ThermalReset(
                f"Thermal reset: CPU {t_cpu:.1f} (>{cpu_max}), "
                f"GPU {t_gpu:.1f} (>{gpu_max}), AUX {t_aux:.1f} (>{aux_max})"
            )

        throttle_cpu = t_cpu > cpu_rec
        throttle_gpu = t_gpu > gpu_rec
        throttle_aux = t_aux > aux_rec
        throttle_ttp = t_ttp > self.cfg.ttp_max_c

        any_throttle = throttle_cpu or throttle_gpu or throttle_aux or throttle_ttp

        if any_throttle:
            self._throttle_timer_s -= dt
            if self._throttle_timer_s <= 0.0:
                self._throttle_timer_s = self.cfg.throttle_step_period_s

                if throttle_cpu or throttle_ttp:
                    self.cap_cpu_idx = max(0, self.cap_cpu_idx - 1)
                if throttle_gpu or throttle_ttp:
                    self.cap_gpu_idx = max(0, self.cap_gpu_idx - 1)

                if throttle_aux:
                    self.cap_emc_idx = max(0, self.cap_emc_idx - 1)
                    self.cap_cpu_idx = max(0, self.cap_cpu_idx - 1)
                    self.cap_gpu_idx = max(0, self.cap_gpu_idx - 1)
        else:
            self._restore_timer_s -= dt
            if self._restore_timer_s <= 0.0:
                self._restore_timer_s = self.cfg.restore_step_period_s
                self.cap_cpu_idx = min(self.cfg.cpu_max_step_idx, self.cap_cpu_idx + 1)
                self.cap_gpu_idx = min(self.cfg.gpu_max_step_idx, self.cap_gpu_idx + 1)
                self.cap_emc_idx = min(self.cfg.emc_max_step_idx, self.cap_emc_idx + 1)

    # ---------------- perf counters ----------------

    def _refresh_perf_visible_if_needed(self) -> None:
        """Recompute _perf_visible if config.perf_multiplex_visible changed."""
        sig = tuple(self.cfg.perf_multiplex_visible)
        if sig != self._perf_visible_signature:
            self._perf_visible_signature = sig
            visible = {e for e in self.cfg.perf_multiplex_visible if e in SUPPORTED_EVENTS}
            self._perf_visible = visible if visible else set(SUPPORTED_EVENTS)

    def _update_perf(self, dt: float) -> None:
        self._refresh_perf_visible_if_needed()
        total_cycles = 0.0
        cpu_f = CPU_FREQ_STEPS_MHZ[self.cpu_step_idx]
        for u in self.cpu_util:
            total_cycles += (cpu_f * 1e6) * dt * (u / 100.0)

        emc = clamp(self.emc_util / 100.0, 0.0, 1.0)
        gpu = clamp(self.gpu_util / 100.0, 0.0, 1.0)
        cv  = clamp(self.cv_util / 100.0, 0.0, 1.0)

        ipc = 2.05 - 0.75 * emc - 0.08 * gpu - 0.04 * cv
        if self.mode == "trojan" and self.trojan_active and self.cfg.trojan_style in ("mempressure", "cv"):
            ipc -= 0.18 * clamp(self.cfg.trojan_strength, 0.5, 2.0)
        ipc = clamp(ipc + self.rng.gauss(0.0, 0.06), 0.55, 2.25)

        instructions = total_cycles * ipc
        cache_refs = instructions * (0.30 + 0.30 * emc)
        miss_rate = 0.028 + 0.11 * emc + 0.02 * gpu + 0.02 * cv
        if self.mode == "trojan" and self.trojan_active:
            miss_rate += 0.03 * clamp(self.cfg.trojan_strength, 0.5, 2.0)
        cache_misses = cache_refs * clamp(miss_rate + self.rng.gauss(0.0, 0.006), 0.01, 0.35)

        branch_ins = instructions * 0.17
        br_miss_rate = 0.010 + 0.012 * emc
        if self.mode == "trojan" and self.trojan_active and self.cfg.trojan_style == "compute":
            br_miss_rate += 0.006 * clamp(self.cfg.trojan_strength, 0.5, 2.0)
        branch_misses = branch_ins * clamp(br_miss_rate + self.rng.gauss(0.0, 0.0018), 0.001, 0.09)

        avg_cpu = sum(self.cpu_util) / len(self.cpu_util)
        ctx = dt * max(0.0, 140.0 + 6.5 * avg_cpu + self.rng.gauss(0.0, 25.0))
        mig = dt * max(0.0, 2.0 + 0.11 * avg_cpu + self.rng.gauss(0.0, 0.5))
        faults = dt * max(0.0, 110.0 + 2.4 * emc * 100.0 + self.rng.gauss(0.0, 18.0))

        if self.mode == "trojan" and self.trojan_active and self.cfg.trojan_style == "io":
            scale = clamp(self.cfg.trojan_strength, 0.5, 2.0)
            ctx *= 2.5 * scale
            faults *= 1.6 * scale
            mig *= 1.3 * scale
            ipc = max(0.55, ipc - 0.20)

        l1_loads = instructions * 0.22
        l1_miss = l1_loads * clamp(0.024 + 0.13 * emc + (0.02 if (self.mode == "trojan" and self.trojan_active) else 0.0), 0.01, 0.45)
        llc_loads = instructions * (0.028 + 0.060 * emc)
        llc_miss = llc_loads * clamp(0.07 + 0.22 * emc + (0.06 if (self.mode == "trojan" and self.trojan_active) else 0.0), 0.02, 0.85)

        deltas = {
            "cycles": total_cycles,
            "instructions": instructions,
            "cache-references": cache_refs,
            "cache-misses": cache_misses,
            "branch-instructions": branch_ins,
            "branch-misses": branch_misses,
            "context-switches": ctx,
            "cpu-migrations": mig,
            "page-faults": faults,
            "L1-dcache-loads": l1_loads,
            "L1-dcache-load-misses": l1_miss,
            "LLC-loads": llc_loads,
            "LLC-load-misses": llc_miss,
        }
        if self._perf_multiplex_left_s > 0.0:
            visible = self._perf_visible
        else:
            visible = set(SUPPORTED_EVENTS)
        if not visible:
            visible = set(SUPPORTED_EVENTS)
        self._current_perf_masks = {e: (1 if e in visible else 0) for e in SUPPORTED_EVENTS}

        for k, v in deltas.items():
            if self._current_perf_masks.get(k, 1):
                self.counters_total[k] += max(0.0, v)

    # ---------------- main step ----------------

    def step_dt(self, dt: float, t_unix_s: float) -> SimState:
        dt = clamp(float(dt), 1.0 / max(1e-6, self.cfg.hz), 0.75)
        self.t_sim_s += dt
        t_sim = self.t_sim_s

        self._trojan_schedule(dt)
        self._update_benign_events(dt)
        self._apply_util_dynamics(dt, t_sim)
        self._governor(dt)

        self._update_ina_averages()
        self._thermal_step(dt)

        # Fan update after thermal step (fan uses current biased temps)
        self._update_fan_state()

        self._enforce_protection(dt)
        self._update_perf(dt)

        # INA-averaged exposed powers
        p_gpu = self.ina_vdd_gpu.avg()
        p_cpu = self.ina_vdd_cpu.avg()
        p_soc = self.ina_vdd_soc.avg()
        p_cv = self.ina_vdd_cv.avg()
        p_vddrq = self.ina_vddrq.avg()
        p_sys5v = self.ina_sys5v.avg()

        cpu_f = CPU_FREQ_STEPS_MHZ[self.cpu_step_idx]
        gpu_f = GPU_FREQ_STEPS_MHZ[self.gpu_step_idx]
        emc_f = EMC_FREQ_STEPS_MHZ[self.emc_step_idx]

        # Report biased temps (what tools see)
        return SimState(
            t_unix_s=float(t_unix_s),
            t_sim_s=float(t_sim),
            mode=self.mode,
            trojan_active=bool(self.trojan_active),
            trojan_phase_s=float(self.trojan_phase_s),

            cpu_util=[float(x) for x in self.cpu_util],
            gpu_util=float(self.gpu_util),
            emc_util=float(self.emc_util),
            cv_util=float(self.cv_util),

            cpu_freq_mhz=[cpu_f] * N_CPU,
            gpu_freq_mhz=int(gpu_f),
            emc_freq_mhz=int(emc_f),

            ram_used_mb=float(self.ram_used_mb),
            ram_total_mb=RAM_TOTAL_MB,
            swap_used_mb=0.0,
            swap_total_mb=SWAP_TOTAL_MB,

            temp_cpu_c=float(self._meas_cpu_c()),
            temp_cpu_clusters_c=[float(x) for x in self.temp_cpu_clusters],
            temp_gpu_c=float(self._meas_gpu_c()),
            temp_aux_c=float(self._meas_aux_c()),
            temp_aux_cv_c=float(self.temp_aux_cv_c),
            temp_aux_ddr_c=float(self.temp_aux_ddr_c),
            temp_ttp_c=float(self._meas_ttp_c()),
            temp_board_c=float(self.temp_board_c),

            fan_est_temp_c=float(self.fan_est_temp_c),

            hysteresis_state=int(self.hysteresis_state),
            fan_pwm=int(self.fan_pwm),
            fan_rpm=int(self.fan_rpm),

            p_cpu_mw=float(p_cpu),
            p_gpu_mw=float(p_gpu),
            p_soc_mw=float(p_soc),
            p_cv_mw=float(p_cv),
            p_vddrq_mw=float(p_vddrq),
            p_sys5v_mw=float(p_sys5v),
            p_in_mw=float(p_sys5v),

            counters_total={k: float(v) for k, v in self.counters_total.items()},
            perf_masks=dict(self._current_perf_masks),
        )

    def step(self) -> SimState:
        """Back-compat wall-clock step used by daemon mode.

        For deterministic/offline generation use step_dt().
        """
        t_unix_s = now_s()
        dt = t_unix_s - self._last_step_unix_s
        self._last_step_unix_s = t_unix_s
        dt = clamp(dt, 1.0 / max(1e-6, self.cfg.hz), 0.75)
        return self.step_dt(dt=dt, t_unix_s=t_unix_s)


# ---------------------------- sysfs mock ----------------------------

def write_sysfs(sysfs_root: str, st: SimState) -> None:
    """
    Exposes INA3221 topology and thermal sensors.
    Units:
      - thermal_zone*/temp: millidegree C
      - in_power*_input: mW
      - hwmon fan: RPM and PWM (0-255)
    """
    if not sysfs_root:
        return

    zones = [
        ("thermal_zone0", "CPU-therm", st.temp_cpu_c),
        ("thermal_zone1", "GPU-therm", st.temp_gpu_c),
        ("thermal_zone2", "AUX-therm", st.temp_aux_c),
        ("thermal_zone3", "TTP-therm", st.temp_ttp_c),
        ("thermal_zone4", "thermal-fan-est", st.fan_est_temp_c),
    ]
    for zname, ztype, temp_c in zones:
        base = os.path.join(sysfs_root, "devices/virtual/thermal", zname)
        ensure_file(os.path.join(base, "type"), f"{ztype}\n")
        ensure_file(os.path.join(base, "temp"), f"{int(round(temp_c * 1000.0))}\n")

    class_thermal = os.path.join(sysfs_root, "class/thermal")
    for zname, ztype, temp_c in zones:
        zdir = os.path.join(class_thermal, zname)
        ensure_file(os.path.join(zdir, "type"), f"{ztype}\n")
        ensure_file(os.path.join(zdir, "temp"), f"{int(round(temp_c * 1000.0))}\n")

    # INA3221 0-0040
    ina40 = os.path.join(sysfs_root, "bus/i2c/drivers/ina3221x", "0-0040", "iio_device")
    ensure_file(os.path.join(ina40, "name"), "ina3221x-0x40\n")
    ensure_file(os.path.join(ina40, "in_power0_label"), "VDD_GPU\n")
    ensure_file(os.path.join(ina40, "in_power1_label"), "VDD_CPU\n")
    ensure_file(os.path.join(ina40, "in_power2_label"), "VDD_SOC\n")
    ensure_file(os.path.join(ina40, "in_power0_input"), f"{int(round(st.p_gpu_mw))}\n")
    ensure_file(os.path.join(ina40, "in_power1_input"), f"{int(round(st.p_cpu_mw))}\n")
    ensure_file(os.path.join(ina40, "in_power2_input"), f"{int(round(st.p_soc_mw))}\n")

    # INA3221 0-0041
    ina41 = os.path.join(sysfs_root, "bus/i2c/drivers/ina3221x", "0-0041", "iio_device")
    ensure_file(os.path.join(ina41, "name"), "ina3221x-0x41\n")
    ensure_file(os.path.join(ina41, "in_power0_label"), "VDD_CV\n")
    ensure_file(os.path.join(ina41, "in_power1_label"), "VDD_VDDRQ\n")
    ensure_file(os.path.join(ina41, "in_power2_label"), "VDD_SYS5V\n")
    ensure_file(os.path.join(ina41, "in_power0_input"), f"{int(round(st.p_cv_mw))}\n")
    ensure_file(os.path.join(ina41, "in_power1_input"), f"{int(round(st.p_vddrq_mw))}\n")
    ensure_file(os.path.join(ina41, "in_power2_input"), f"{int(round(st.p_sys5v_mw))}\n")

    # Fan hwmon
    hwmon = os.path.join(sysfs_root, "class/hwmon/hwmon0")
    ensure_file(os.path.join(hwmon, "name"), "pwm-fan\n")
    ensure_file(os.path.join(hwmon, "pwm1"), f"{st.fan_pwm}\n")
    ensure_file(os.path.join(hwmon, "fan1_input"), f"{st.fan_rpm}\n")
    ensure_file(os.path.join(hwmon, "pwm1_enable"), "1\n")
    ensure_file(os.path.join(hwmon, "pwm1_max"), "255\n")

    # cpufreq: scaling_cur_freq in kHz
    for i in range(N_CPU):
        cpu_base = os.path.join(sysfs_root, "devices/system/cpu", f"cpu{i}", "cpufreq")
        ensure_file(os.path.join(cpu_base, "scaling_cur_freq"), f"{int(st.cpu_freq_mhz[i] * 1000)}\n")

    # devfreq: GPU/EMC cur_freq in Hz
    devfreq_gpu = os.path.join(sysfs_root, "devices/virtual/devfreq", "gpu")
    devfreq_emc = os.path.join(sysfs_root, "devices/virtual/devfreq", "emc")
    ensure_file(os.path.join(devfreq_gpu, "cur_freq"), f"{int(st.gpu_freq_mhz * 1_000_000)}\n")
    ensure_file(os.path.join(devfreq_emc, "cur_freq"), f"{int(st.emc_freq_mhz * 1_000_000)}\n")

    # GPU load (0..1000 scaled integer)
    gpu_load_path = os.path.join(sysfs_root, "devices/gpu.0")
    ensure_file(os.path.join(gpu_load_path, "load"), f"{int(round(clamp(st.gpu_util, 0, 100) * 10))}\n")


# ---------------------------- power mode caps helper ----------------------------

def apply_power_mode_caps(cfg: SimConfig, model: XavierTDGModel) -> None:
    """Apply DVFS and power cap defaults based on cfg.power_mode.
    
    This ensures consistent behavior between daemon and offline generation.
    30W mode: conservative caps for lower power draw
    MAXN mode: higher caps for maximum performance
    """
    mode_caps = {
        "30W": {"power_cap_w": 30.0, "cpu_max_step_idx": 5, "gpu_max_step_idx": 3, "emc_max_step_idx": 4},
        "MAXN": {"power_cap_w": 50.0, "cpu_max_step_idx": 6, "gpu_max_step_idx": 4, "emc_max_step_idx": 5},
    }
    pm = str(cfg.power_mode).upper()
    caps = mode_caps.get(pm)
    if not caps:
        return
    
    # Apply defaults to config
    cfg.power_cap_w = float(caps["power_cap_w"])
    cfg.cpu_max_step_idx = min(int(caps["cpu_max_step_idx"]), len(CPU_FREQ_STEPS_MHZ) - 1)
    cfg.gpu_max_step_idx = min(int(caps["gpu_max_step_idx"]), len(GPU_FREQ_STEPS_MHZ) - 1)
    cfg.emc_max_step_idx = min(int(caps["emc_max_step_idx"]), len(EMC_FREQ_STEPS_MHZ) - 1)
    
    # Clamp model's throttle caps to config limits
    model.cap_cpu_idx = min(model.cap_cpu_idx, cfg.cpu_max_step_idx)
    model.cap_gpu_idx = min(model.cap_gpu_idx, cfg.gpu_max_step_idx)
    model.cap_emc_idx = min(model.cap_emc_idx, cfg.emc_max_step_idx)


# ---------------------------- server ----------------------------

class SimServer:
    def __init__(self, cfg: SimConfig, sysfs_root: str):
        self.cfg = cfg
        self.sysfs_root = sysfs_root
        self.model = XavierTDGModel(cfg)
        # Apply power mode caps (30W vs MAXN) before setting mode
        apply_power_mode_caps(cfg, self.model)
        # Apply mode from config before priming (avoids silent idle runs)
        self.model.set_mode(cfg.mode)
        self.lock = threading.Lock()
        # Seeded jitter RNG (avoid global random for reproducibility)
        self._jitter_rng = random.Random(int(cfg.seed) + 99991)
        # Prime with a single deterministic step
        t_unix_s0 = float(cfg.start_unix_s) if cfg.emit_unix_from_sim else now_s()
        self.latest: SimState = self.model.step_dt(dt=1.0 / max(1e-6, cfg.hz), t_unix_s=t_unix_s0)
        self.exposed_state: SimState = self.latest
        self.stop_evt = threading.Event()
        self.last_error: Optional[str] = None

        self.sensor_realism = SensorRealism(cfg, rng_seed=int(cfg.seed) + 3331)

    def _apply_power_mode_caps(self, updates: Dict[str, Any]) -> None:
        mode_caps = {
            "30W": {"power_cap_w": 30.0, "cpu_max_step_idx": 5, "gpu_max_step_idx": 3, "emc_max_step_idx": 4},
            "MAXN": {"power_cap_w": 50.0, "cpu_max_step_idx": 6, "gpu_max_step_idx": 4, "emc_max_step_idx": 5},
        }
        pm = str(self.cfg.power_mode).upper()
        caps = mode_caps.get(pm)
        if not caps:
            return
        if "power_cap_w" not in updates:
            self.cfg.power_cap_w = caps["power_cap_w"]
        if "cpu_max_step_idx" not in updates:
            self.cfg.cpu_max_step_idx = min(caps["cpu_max_step_idx"], len(CPU_FREQ_STEPS_MHZ) - 1)
        if "gpu_max_step_idx" not in updates:
            self.cfg.gpu_max_step_idx = min(caps["gpu_max_step_idx"], len(GPU_FREQ_STEPS_MHZ) - 1)
        if "emc_max_step_idx" not in updates:
            self.cfg.emc_max_step_idx = min(caps["emc_max_step_idx"], len(EMC_FREQ_STEPS_MHZ) - 1)
        self.model.cap_cpu_idx = min(self.model.cap_cpu_idx, self.cfg.cpu_max_step_idx)
        self.model.cap_gpu_idx = min(self.model.cap_gpu_idx, self.cfg.gpu_max_step_idx)
        self.model.cap_emc_idx = min(self.model.cap_emc_idx, self.cfg.emc_max_step_idx)

    def set_mode(self, mode: str) -> None:
        with self.lock:
            self.model.set_mode(mode)

    def set_config(self, **kwargs) -> None:
        with self.lock:
            for k, v in kwargs.items():
                if not hasattr(self.cfg, k):
                    raise ValueError(f"unknown config key: {k}")
                setattr(self.cfg, k, v)
            if "power_mode" in kwargs:
                self._apply_power_mode_caps(kwargs)

    def run_loop(self) -> None:
        while not self.stop_evt.is_set():
            hz = max(1e-6, float(self.cfg.hz))
            period = 1.0 / hz
            t0 = now_s()
            try:
                with self.lock:
                    if bool(self.cfg.fast):
                        dt = period
                        if bool(self.cfg.emit_unix_from_sim):
                            t_unix_s = float(self.cfg.start_unix_s) + float(self.model.t_sim_s) + dt
                        else:
                            t_unix_s = now_s()
                        raw_state = self.model.step_dt(dt=dt, t_unix_s=t_unix_s)
                    else:
                        # Wall-clock driven
                        raw_state = self.model.step()

                    exposed = self.sensor_realism.apply(asdict(raw_state), t_sim_s=float(raw_state.t_sim_s))
                    self.exposed_state = SimState(**exposed)
                    self.latest = self.exposed_state
                    write_sysfs(self.sysfs_root, self.exposed_state)
            except (ThermalReset, ThermalHardStop) as e:
                self.last_error = str(e)
                self.stop_evt.set()
            if not bool(self.cfg.fast):
                elapsed = now_s() - t0
                time.sleep(max(0.0, period - elapsed))

    def snapshot(self) -> Dict:
        with self.lock:
            d = asdict(self.exposed_state)
            # Include simulator regime/meta fields for HTTP guard and CSV consistency
            d["sim_fast_mode"] = bool(self.cfg.fast)
            d["sim_hz"] = float(self.cfg.hz)
            d["sim_seed"] = int(self.cfg.seed)
            # Regime fields (source of truth for collectors)
            d["sim_power_mode"] = str(self.cfg.power_mode)
            d["sim_ambient_c"] = float(self.cfg.ambient_c)
            d["sim_input_voltage_v"] = float(self.cfg.input_voltage_v)
            d["sim_jetson_clocks"] = bool(self.cfg.jetson_clocks)
            d["sim_workload_family"] = str(self.cfg.workload_family)
            d["sim_workload_variant"] = str(self.cfg.workload_variant)
            d["sim_trojan_family"] = str(self.cfg.trojan_family)
            d["sim_trojan_variant"] = str(self.cfg.trojan_variant)
        if self.last_error:
            d["sim_error"] = self.last_error
        return d


def make_handler(sim: SimServer):
    class Handler(BaseHTTPRequestHandler):
        def _json(self, code: int, obj: Dict) -> None:
            jitter_s = max(0.0, float(sim.cfg.telemetry_jitter_ms) / 1000.0)
            if jitter_s > 0:
                time.sleep(sim._jitter_rng.uniform(0.0, jitter_s))
            body = json.dumps(obj).encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path.startswith("/v1/telemetry"):
                self._json(200, sim.snapshot())
                return
            if self.path.startswith("/v1/events"):
                self._json(200, {"supported_events": SUPPORTED_EVENTS})
                return
            self._json(404, {"error": "not found"})

        def do_POST(self):
            length = int(self.headers.get("Content-Length", "0"))
            raw = self.rfile.read(length) if length > 0 else b"{}"
            try:
                payload = json.loads(raw.decode("utf-8"))
            except Exception:
                self._json(400, {"error": "invalid json"})
                return

            if self.path.startswith("/v1/control"):
                mode = payload.get("mode")
                if mode:
                    try:
                        sim.set_mode(mode)
                    except Exception as e:
                        self._json(400, {"error": str(e)})
                        return
                self._json(200, {"ok": True, "mode": mode})
                return

            if self.path.startswith("/v1/config"):
                allowed = {
                    # core
                    "hz", "ambient_c", "power_mode", "power_cap_w", "input_voltage_v",
                    "fast",  # allow toggling fast mode
                    # schedule
                    "trojan_period_s", "trojan_on_s", "trojan_style", "trojan_strength",
                    # INA
                    "ina_window_samples", "ina_internal_hz",
                    # thermal constants
                    "theta_jp_balanced_c_per_w", "theta_jp_unbalanced_c_per_w", "theta_jb_c_per_w",
                    "ttp_max_c", "theta_heatsink_eff_c_per_w",
                    "tau_ttp_s", "tau_board_s", "tau_cpu_s", "tau_gpu_s", "tau_aux_s",
                    # workload
                    "frame_rate_hz", "frame_gpu_busy_ms", "frame_cpu_busy_ms",
                    "bg_spike_rate_hz", "bg_spike_ms", "bg_spike_cpu_pct",
                    # dvfs
                    "cpu_max_step_idx", "gpu_max_step_idx", "emc_max_step_idx",
                    "cpu_step_rate_per_s", "gpu_step_rate_per_s", "emc_step_rate_per_s",
                    # throttle pacing
                    "throttle_step_period_s", "restore_step_period_s",

                    # realism fixes
                    "theta_jp_blend", "theta_jp_blend_knee", "theta_jp_blend_span",
                    "fan_cooling_min_gain", "fan_cooling_max_gain",
                    "board_ttp_coupling",
                    "power_cap_exp_cpu", "power_cap_exp_gpu", "power_cap_exp_cv",
                    "power_cap_exp_soc", "power_cap_exp_vddrq",
                    "tau_aux_memory_s",
                    "tau_aux_cv_memory_s", "tau_aux_ddr_memory_s",
                    "aux_cv_memory_hold_c", "aux_ddr_memory_hold_c",
                    "aux_cv_weight",
                    "bias_cpu_c", "bias_gpu_c", "bias_aux_c", "bias_ttp_c",
                    "telemetry_jitter_ms", "temp_cpu_update_ms", "temp_gpu_update_ms",
                    "temp_aux_update_ms", "temp_ttp_update_ms", "temp_board_update_ms",
                    "power_update_ms", "jetson_clocks",
                    # dropout + confounders
                    "sensor_dropout_prob", "sensor_dropout_duration_s", "sensor_dropout_groups",
                    "benign_throttle_event_rate_hz", "benign_throttle_event_duration_s",
                    "benign_throttle_temp_delta_c", "benign_throttle_fan_gain",
                    "perf_multiplex_event_rate_hz", "perf_multiplex_event_duration_s", "perf_multiplex_visible",
                }
                updates = {k: payload[k] for k in payload.keys() if k in allowed}
                try:
                    sim.set_config(**updates)
                except Exception as e:
                    self._json(400, {"error": str(e)})
                    return
                self._json(200, {"ok": True, "updated": updates})
                return

            self._json(404, {"error": "not found"})

        def log_message(self, fmt, *args):
            return

    return Handler


# ---------------------------- HTTP helpers ----------------------------

def http_get_json(port: int, path: str) -> Dict:
    import urllib.request
    url = f"http://127.0.0.1:{port}{path}"
    with urllib.request.urlopen(url, timeout=2.0) as r:
        return json.loads(r.read().decode("utf-8"))

def http_post_json(port: int, path: str, payload: Dict) -> Dict:
    import urllib.request
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}{path}",
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=2.0) as r:
        return json.loads(r.read().decode("utf-8"))


# ---------------------------- tegrastats-like ----------------------------

def fmt_tegrastats_line(st: Dict) -> str:
    used = float(st["ram_used_mb"])
    total = int(st["ram_total_mb"])
    free = max(0.0, total - used)
    lfb_blocks = int(clamp(free / 256.0, 100, 900))
    lfb_mb = 4

    cpu_parts = []
    for u, f in zip(st["cpu_util"], st["cpu_freq_mhz"]):
        cpu_parts.append(f"{u:.0f}%@{int(f)}")
    cpu_str = ", ".join(cpu_parts)

    line = (
        f"RAM {used:.0f}/{total}MB (lfb {lfb_blocks}x{lfb_mb}MB) "
        f"CPU [{cpu_str}] "
        f"EMC {st['emc_util']:.0f}%@{int(st['emc_freq_mhz'])} "
        f"GR3D {st['gpu_util']:.0f}%@{int(st['gpu_freq_mhz'])} "
        f"CV {st['cv_util']:.0f}% "
        f"TTP@{st['temp_ttp_c']:.1f}C CPU@{st['temp_cpu_c']:.1f}C GPU@{st['temp_gpu_c']:.1f}C AUX@{st['temp_aux_c']:.1f}C "
        f"FAN_EST@{st['fan_est_temp_c']:.1f}C FAN {int(st['fan_rpm'])}RPM PWM {int(st['fan_pwm'])} HS {int(st['hysteresis_state'])} "
        f"VDD_SYS5V {st['p_sys5v_mw']:.0f}mW VDD_CPU {st['p_cpu_mw']:.0f}mW VDD_GPU {st['p_gpu_mw']:.0f}mW "
        f"VDD_SOC {st['p_soc_mw']:.0f}mW VDD_CV {st['p_cv_mw']:.0f}mW VDD_VDDRQ {st['p_vddrq_mw']:.0f}mW "
        f"MODE {st['mode']} TROJAN={'1' if st.get('trojan_active') else '0'}"
    )
    if st.get("sim_error"):
        line += f" ERROR='{st['sim_error']}'"
    return line


# ---------------------------- calibration helpers ----------------------------

def _mean_window(vals: List[float], frac: float) -> float:
    n = max(1, int(len(vals) * frac))
    return sum(vals[:n]) / n if vals else float("nan")


def _linear_regression(xs: List[float], ys: List[float]) -> Tuple[float, float]:
    """Return slope, r2 for simple linear regression."""
    if not xs or not ys or len(xs) != len(ys):
        return float("nan"), float("nan")
    n = len(xs)
    mean_x = sum(xs) / n
    mean_y = sum(ys) / n
    num = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
    den = sum((x - mean_x) ** 2 for x in xs)
    if den == 0:
        return float("nan"), float("nan")
    slope = num / den
    ss_tot = sum((y - mean_y) ** 2 for y in ys)
    ss_res = sum((y - (slope * x + (mean_y - slope * mean_x))) ** 2 for x, y in zip(xs, ys))
    r2 = 1.0 - (ss_res / ss_tot) if ss_tot != 0 else float("nan")
    return slope, r2


def estimate_tau_from_step(times: List[float], values: List[float], min_delta: float = 0.5) -> Tuple[Optional[float], Dict[str, float]]:
    """Estimate first-order tau from a single monotonic step using log fit."""
    if len(times) < 6 or len(values) < 6:
        return None, {"reason": "too few samples"}

    t0 = float(times[0])
    base = _mean_window(values, 0.1)
    final = _mean_window(values[::-1], 0.1)  # mean of last 10%
    delta = final - base
    if math.isnan(base) or math.isnan(final) or abs(delta) < min_delta:
        return None, {"reason": "delta too small", "delta": delta}

    xs: List[float] = []
    ys: List[float] = []
    for ti, vi in zip(times, values):
        err = vi - final
        if abs(err) < 1e-6:
            continue
        xs.append(float(ti) - t0)
        ys.append(math.log(abs(err)))

    slope, r2 = _linear_regression(xs, ys)
    if math.isnan(slope) or slope >= 0:
        return None, {"reason": "slope non-negative", "slope": slope, "r2": r2}

    tau = -1.0 / slope
    return tau, {"r2": r2, "delta": delta, "base": base, "final": final}


def _extract_aligned_time_values(
    rows: List[Dict[str, str]], t_col: str, v_col: str
) -> Tuple[List[float], List[float]]:
    """Build aligned (times, values) arrays, only including rows with valid data."""
    times: List[float] = []
    values: List[float] = []
    for row in rows:
        t_raw = row.get(t_col)
        v_raw = row.get(v_col)
        if t_raw in (None, "") or v_raw in (None, ""):
            continue
        try:
            t = float(t_raw)
            v = float(v_raw)
            times.append(t)
            values.append(v)
        except (ValueError, TypeError):
            continue
    return times, values


def calibrate_from_csv(path: str, min_delta: float = 0.5) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        r = csv.DictReader(f)
        rows = list(r)
    if not rows:
        raise SystemExit("calibrate: CSV is empty")

    t_col = "t_sim_s" if "t_sim_s" in rows[0] else "t_unix_s"

    sensors = {
        "tau_ttp_s": "temp_ttp_c",
        "tau_board_s": "temp_board_c",
        "tau_cpu_s": "temp_cpu_c",
        "tau_gpu_s": "temp_gpu_c",
        "tau_aux_s": "temp_aux_c",
        "tau_aux_cv_s": "temp_aux_cv_c",
        "tau_aux_ddr_s": "temp_aux_ddr_c",
    }

    # per-cluster CPU taus; will aggregate later
    cluster_cols = [
        "temp_cpu_cluster0_c",
        "temp_cpu_cluster1_c",
        "temp_cpu_cluster2_c",
        "temp_cpu_cluster3_c",
    ]

    results: Dict[str, Any] = {"taus": {}, "debug": {}}

    for key, col in sensors.items():
        if col not in rows[0]:
            results["debug"][key] = {"reason": f"column {col} missing"}
            continue
        # Build aligned arrays (handles dropout/missingness correctly)
        time_vals, vals = _extract_aligned_time_values(rows, t_col, col)
        if len(time_vals) < 6:
            results["debug"][key] = {"reason": "too few valid samples", "n_valid": len(time_vals)}
            continue
        tau, meta = estimate_tau_from_step(time_vals, vals, min_delta=min_delta)
        if tau is not None:
            results["taus"][key] = tau
        results["debug"][key] = meta

    cluster_taus = []
    for i, col in enumerate(cluster_cols):
        if col not in rows[0]:
            continue
        time_vals, vals = _extract_aligned_time_values(rows, t_col, col)
        if len(time_vals) < 6:
            results["debug"][f"tau_cpu_cluster{i}_s"] = {"reason": "too few valid samples", "n_valid": len(time_vals)}
            continue
        tau, meta = estimate_tau_from_step(time_vals, vals, min_delta=min_delta)
        results["debug"][f"tau_cpu_cluster{i}_s"] = meta
        if tau is not None:
            cluster_taus.append(tau)

    if cluster_taus:
        results["taus"]["tau_cpu_cluster_s"] = sum(cluster_taus) / len(cluster_taus)
        # keep legacy tau_cpu_s suggestion in sync with clusters if not already set
        if "tau_cpu_s" not in results["taus"]:
            results["taus"]["tau_cpu_s"] = results["taus"]["tau_cpu_cluster_s"]

    return results


def cmd_calibrate(args: argparse.Namespace) -> None:
    res = calibrate_from_csv(args.csv, min_delta=args.min_delta_c)
    print("Suggested taus (seconds):")
    for k, v in sorted(res["taus"].items()):
        print(f"  {k}: {v:.2f}")
    if args.out_json:
        payload = {"suggested": res["taus"], "debug": res["debug"], "source_csv": args.csv}
        with open(args.out_json, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, sort_keys=True)
        print(f"Wrote {args.out_json}")


# ---------------------------- commands ----------------------------

def cmd_simd(args: argparse.Namespace) -> None:
    cfg = SimConfig(
        seed=args.seed,
        hz=args.hz,
        ambient_c=args.ambient_c,
        power_mode=args.power_mode,
        power_cap_w=args.power_cap_w,
        input_voltage_v=args.input_voltage_v,
        trojan_period_s=args.trojan_period_s,
        trojan_on_s=args.trojan_on_s,
        trojan_style=args.trojan_style,
        trojan_strength=args.trojan_strength,
        theta_heatsink_eff_c_per_w=args.theta_heatsink_eff_c_per_w,
    )
    sim = SimServer(cfg, sysfs_root=args.sysfs_root)
    sim._apply_power_mode_caps({})

    def handle_sig(*_):
        sim.stop_evt.set()

    signal.signal(signal.SIGINT, handle_sig)
    signal.signal(signal.SIGTERM, handle_sig)

    th = threading.Thread(target=sim.run_loop, daemon=True)
    th.start()

    server = ThreadingHTTPServer(("127.0.0.1", args.port), make_handler(sim))
    server.timeout = 0.5

    write_sysfs(args.sysfs_root, sim.latest)

    while not sim.stop_evt.is_set():
        server.handle_request()

def cmd_control(args: argparse.Namespace) -> None:
    out = http_post_json(args.port, "/v1/control", {"mode": args.mode})
    print(json.dumps(out, indent=2))

def cmd_tegrastats(args: argparse.Namespace) -> None:
    interval_s = max(0.05, args.interval_ms / 1000.0)
    try:
        while True:
            st = http_get_json(args.port, "/v1/telemetry")
            print(fmt_tegrastats_line(st))
            time.sleep(interval_s)
    except KeyboardInterrupt:
        return

def parse_event_list(s: str) -> List[str]:
    events = [x.strip() for x in s.split(",") if x.strip()]
    for e in events:
        if e not in SUPPORTED_EVENTS:
            raise SystemExit(f"Unsupported event: {e}\nSupported: {', '.join(SUPPORTED_EVENTS)}")
    return events

def cmd_perf(args: argparse.Namespace) -> None:
    if args.perf_cmd == "list":
        for e in SUPPORTED_EVENTS:
            print(e)
        return

    if args.perf_cmd == "stat":
        events = parse_event_list(args.events)
        duration_s = float(args.duration_s)
        if duration_s <= 0.0:
            raise SystemExit("--duration-s must be > 0")

        st0 = http_get_json(args.port, "/v1/telemetry")
        t0 = st0["t_unix_s"]
        c0 = st0["counters_total"]

        time.sleep(duration_s)

        st1 = http_get_json(args.port, "/v1/telemetry")
        t1 = st1["t_unix_s"]
        c1 = st1["counters_total"]

        dt = max(1e-6, float(t1) - float(t0))

        print(" Performance counter stats for 'jetson-sim':\n")
        d_cycles = c1["cycles"] - c0["cycles"]
        d_insn = c1["instructions"] - c0["instructions"]
        ipc = (d_insn / d_cycles) if d_cycles > 0 else 0.0

        for e in events:
            d = max(0.0, c1[e] - c0[e])
            if e == "instructions":
                print(f"   {fmt_commas_int(d):>15}      {e:<20}  # {ipc:0.2f}  insn per cycle")
            else:
                print(f"   {fmt_commas_int(d):>15}      {e}")
        print(f"\n   {dt:0.6f} seconds time elapsed\n")
        return

    raise SystemExit("perf: expected 'list' or 'stat'")


def _collect_header(perf_events: List[Tuple[str, str]], mask_cols: List[str], perf_mask_cols: List[str]) -> List[str]:
    header = [
        "t_unix_s", "t_sim_s",
        "run_id", "binary_id", "workload_family", "workload_variant",
        "trojan_family", "trojan_variant", "trojan_active", "mode",
        "trojan_phase_s",
        "power_mode", "ambient_c", "input_voltage_v", "jetson_clocks",
        "cpu_util_avg", "cpu_util_max", "gpu_util", "emc_util", "cv_util",
        "cpu_freq_mhz", "gpu_freq_mhz", "emc_freq_mhz",
        "temp_cpu_c", "temp_gpu_c", "temp_aux_c", "temp_ttp_c", "temp_board_c",
        "fan_est_temp_c", "hysteresis_state", "fan_pwm", "fan_rpm",
        "p_sys5v_mw", "p_cpu_mw", "p_gpu_mw", "p_soc_mw", "p_cv_mw", "p_vddrq_mw",
    ]
    header += [out_key for _, out_key in perf_events]
    header += mask_cols
    header += perf_mask_cols
    return header


def _build_collect_row(
    snapshot: Dict[str, Any],
    args: argparse.Namespace,
    perf_events: List[Tuple[str, str]],
    mask_cols: List[str],
    perf_mask_cols: List[str],
    prev_totals: Dict[str, float],
) -> List[Any]:
    cpu_util = snapshot.get("cpu_util") or []
    cpu_avg = (sum(cpu_util) / len(cpu_util)) if cpu_util else 0.0
    cpu_max = max(cpu_util) if cpu_util else 0.0

    cpu_freqs = snapshot.get("cpu_freq_mhz") or []
    cpu_freq = cpu_freqs[0] if cpu_freqs else float("nan")

    # Prefer snapshot sim_* fields (source of truth) over args
    power_mode = snapshot.get("sim_power_mode", args.power_mode)
    ambient_c = snapshot.get("sim_ambient_c", args.ambient_c)
    input_voltage_v = snapshot.get("sim_input_voltage_v", args.input_voltage_v)
    jetson_clocks = 1 if snapshot.get("sim_jetson_clocks", args.jetson_clocks) else 0
    workload_family = snapshot.get("sim_workload_family", args.workload_family)
    workload_variant = snapshot.get("sim_workload_variant", args.workload_variant)
    trojan_family = snapshot.get("sim_trojan_family", args.trojan_family)
    trojan_variant = snapshot.get("sim_trojan_variant", args.trojan_variant)

    row = [
        snapshot.get("t_unix_s"), snapshot.get("t_sim_s"),
        args.run_id, args.binary_id, workload_family, workload_variant,
        trojan_family, trojan_variant,
        int(bool(snapshot.get("trojan_active"))), snapshot.get("mode"),
        snapshot.get("trojan_phase_s"),
        power_mode, f"{float(ambient_c):.6f}", f"{float(input_voltage_v):.6f}",
        jetson_clocks,
        cpu_avg, cpu_max, snapshot.get("gpu_util"), snapshot.get("emc_util"), snapshot.get("cv_util"),
        cpu_freq, snapshot.get("gpu_freq_mhz"), snapshot.get("emc_freq_mhz"),
        snapshot.get("temp_cpu_c", float("nan")), snapshot.get("temp_gpu_c", float("nan")),
        snapshot.get("temp_aux_c", float("nan")), snapshot.get("temp_ttp_c", float("nan")),
        snapshot.get("temp_board_c", float("nan")),
        snapshot.get("fan_est_temp_c", float("nan")), snapshot.get("hysteresis_state"),
        snapshot.get("fan_pwm"), snapshot.get("fan_rpm"),
        snapshot.get("p_sys5v_mw"), snapshot.get("p_cpu_mw"), snapshot.get("p_gpu_mw"),
        snapshot.get("p_soc_mw"), snapshot.get("p_cv_mw"), snapshot.get("p_vddrq_mw"),
    ]

    counters = snapshot.get("counters_total", {})
    deltas: List[float] = []
    for event_name, _ in perf_events:
        total = float(counters.get(event_name, 0.0))
        prev = prev_totals.get(event_name)
        delta = (total - prev) if prev is not None else 0.0
        prev_totals[event_name] = total
        deltas.append(delta)
    row.extend(deltas)

    for col in mask_cols:
        row.append(int(snapshot.get(col, 1)))

    perf_masks = snapshot.get("perf_masks", {})
    for event_name, _ in perf_events:
        mask_key = f"mask_perf_{event_name.replace('-', '_').lower()}"
        row.append(int(perf_masks.get(event_name, 1)))

    return row


def _load_sim_config_from_file(path: str) -> SimConfig:
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    if not isinstance(payload, dict):
        raise SystemExit("offline sim config must be a JSON object")
    return SimConfig(**payload)


def _collect_offline(
    cfg: SimConfig,
    args: argparse.Namespace,
    duration_s: float,
    writer: csv.writer,
    perf_events: List[Tuple[str, str]],
    mask_cols: List[str],
    perf_mask_cols: List[str],
) -> None:
    model = XavierTDGModel(cfg)
    # Apply power mode caps (30W vs MAXN) before setting mode
    apply_power_mode_caps(cfg, model)
    # CRITICAL: Set mode from config so workload families and Trojan schedules activate
    model.set_mode(cfg.mode)
    realism = SensorRealism(cfg, rng_seed=int(cfg.seed) + 5150)
    dt = 1.0 / max(1e-6, float(cfg.hz))
    # Fix step count rounding drift: use exact tick count
    steps = max(1, int(duration_s * float(cfg.hz)))
    prev_totals: Dict[str, float] = {}
    for _ in range(steps):
        t_unix_s = float(cfg.start_unix_s) + float(model.t_sim_s) + dt
        st = model.step_dt(dt=dt, t_unix_s=t_unix_s)
        snap = realism.apply(asdict(st), t_sim_s=float(st.t_sim_s))
        # Add sim_* metadata fields from cfg (source of truth for offline)
        snap["sim_power_mode"] = str(cfg.power_mode)
        snap["sim_ambient_c"] = float(cfg.ambient_c)
        snap["sim_input_voltage_v"] = float(cfg.input_voltage_v)
        snap["sim_jetson_clocks"] = bool(cfg.jetson_clocks)
        snap["sim_workload_family"] = str(cfg.workload_family)
        snap["sim_workload_variant"] = str(cfg.workload_variant)
        snap["sim_trojan_family"] = str(cfg.trojan_family)
        snap["sim_trojan_variant"] = str(cfg.trojan_variant)
        snap["sim_fast_mode"] = bool(cfg.fast)
        snap["sim_hz"] = float(cfg.hz)
        snap["sim_seed"] = int(cfg.seed)
        row = _build_collect_row(snap, args, perf_events, mask_cols, perf_mask_cols, prev_totals)
        writer.writerow(row)

def cmd_collect(args: argparse.Namespace) -> None:
    interval_s = max(0.05, args.interval_ms / 1000.0)
    duration_s = float(args.duration_s)
    if duration_s <= 0:
        raise SystemExit("--duration-s must be > 0")

    perf_events = [
        ("cycles", "delta_cycles"),
        ("instructions", "delta_instructions"),
        ("cache-misses", "delta_cache_misses"),
        ("context-switches", "delta_context_switches"),
        ("page-faults", "delta_page_faults"),
        ("LLC-load-misses", "delta_llc_load_misses"),
        ("branch-misses", "delta_branch_misses"),
    ]

    mask_cols = [
        "mask_temp_cpu", "mask_temp_gpu", "mask_temp_aux", "mask_temp_ttp", "mask_temp_board",
        "mask_fan_est_temp",
        "mask_power_cpu", "mask_power_gpu", "mask_power_soc", "mask_power_cv",
        "mask_power_vddrq", "mask_power_sys5v", "mask_power_in",
    ]
    perf_mask_cols = [f"mask_perf_{name.replace('-', '_').lower()}" for name, _ in perf_events]

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(_collect_header(perf_events, mask_cols, perf_mask_cols))

        if args.offline_sim_config:
            cfg = _load_sim_config_from_file(args.offline_sim_config)
            _collect_offline(cfg, args, duration_s, w, perf_events, mask_cols, perf_mask_cols)
            return

        events_info = http_get_json(args.port, "/v1/events")
        supported = set(events_info.get("supported_events", []))
        for name, _ in perf_events:
            if name not in supported:
                raise SystemExit(f"Simulator missing event {name}")

        t_end = now_s() + duration_s
        prev_totals: Dict[str, float] = {}
        checked_fast = False

        while now_s() < t_end:
            snapshot = http_get_json(args.port, "/v1/telemetry")
            if not checked_fast:
                if snapshot.get("sim_fast_mode") and not args.allow_fast_http:
                    raise SystemExit(
                        "collect: target simulator is running in fast mode; use --offline-sim-config or rerun without fast mode"
                    )
                checked_fast = True

            row = _build_collect_row(snapshot, args, perf_events, mask_cols, perf_mask_cols, prev_totals)
            w.writerow(row)
            f.flush()
            time.sleep(interval_s)

def main() -> None:
    p = argparse.ArgumentParser(description="Jetson AGX Xavier TDG-aligned telemetry simulator (stdlib only)")
    sub = p.add_subparsers(dest="cmd", required=True)

    ps = sub.add_parser("simd", help="run simulator daemon (HTTP + sysfs mock)")
    ps.add_argument("--port", type=int, default=DEFAULT_PORT)
    ps.add_argument("--sysfs-root", type=str, default="./mock_sysfs")
    ps.add_argument("--hz", type=float, default=20.0)
    ps.add_argument("--seed", type=int, default=1337)
    ps.add_argument("--ambient-c", type=float, default=24.0)
    ps.add_argument("--power-mode", type=str, choices=["30W", "MAXN"], default="30W")
    ps.add_argument("--power-cap-w", type=float, default=30.0)
    ps.add_argument("--input-voltage-v", type=float, default=19.0)
    ps.add_argument("--trojan-period-s", type=float, default=40.0)
    ps.add_argument("--trojan-on-s", type=float, default=10.0)
    ps.add_argument("--trojan-style", type=str, choices=["mempressure", "compute", "cv", "io"], default="mempressure")
    ps.add_argument("--trojan-strength", type=float, default=1.0)
    ps.add_argument("--theta-heatsink-eff-c-per-w", type=float, default=1.15)
    ps.set_defaults(func=cmd_simd)

    pc = sub.add_parser("control", help="set simulator mode")
    pc.add_argument("--port", type=int, default=DEFAULT_PORT)
    pc.add_argument("--mode", type=str, choices=["idle", "normal", "trojan"], required=True)
    pc.set_defaults(func=cmd_control)

    pt = sub.add_parser("tegrastats", help="print tegrastats-like lines from simulator")
    pt.add_argument("--port", type=int, default=DEFAULT_PORT)
    pt.add_argument("--interval-ms", type=int, default=500)
    pt.set_defaults(func=cmd_tegrastats)

    pp = sub.add_parser("perf", help="perf-like interface (simulated)")
    ppsub = pp.add_subparsers(dest="perf_cmd", required=True)

    ppl = ppsub.add_parser("list", help="list supported perf events")
    ppl.add_argument("--port", type=int, default=DEFAULT_PORT)
    ppl.set_defaults(func=cmd_perf)

    pps = ppsub.add_parser("stat", help="perf stat -e ... --duration-s ...")
    pps.add_argument("--port", type=int, default=DEFAULT_PORT)
    pps.add_argument("-e", "--events", type=str, default="cycles,instructions,cache-misses,branch-misses")
    pps.add_argument("--duration-s", type=float, default=1.0)
    pps.set_defaults(func=cmd_perf)

    pr = sub.add_parser("collect", help="collect telemetry into CSV")
    pr.add_argument("--port", type=int, default=DEFAULT_PORT)
    pr.add_argument("--out", type=str, required=True)
    pr.add_argument("--run-id", type=str, default="run_collect")
    pr.add_argument("--binary-id", type=str, default="sim_binary_collect")
    pr.add_argument("--workload-family", type=str, default="unknown")
    pr.add_argument("--workload-variant", type=str, default="v0")
    pr.add_argument("--trojan-family", type=str, default="sim_trojan")
    pr.add_argument("--trojan-variant", type=str, default="v0")
    pr.add_argument("--power-mode", type=str, choices=["30W", "MAXN"], default="30W")
    pr.add_argument("--ambient-c", type=float, default=24.0)
    pr.add_argument("--input-voltage-v", type=float, default=19.0)
    pr.add_argument("--jetson-clocks", action="store_true")
    pr.add_argument("--offline-sim-config", type=str, help="Path to JSON describing SimConfig for offline deterministic collection")
    pr.add_argument("--allow-fast-http", action="store_true", help="Allow HTTP polling when simulator fast mode is enabled")
    pr.add_argument("--duration-s", type=float, default=120.0)
    pr.add_argument("--interval-ms", type=int, default=200)
    pr.set_defaults(func=cmd_collect)

    pcali = sub.add_parser("calibrate", help="fit first-order taus from telemetry CSV")
    pcali.add_argument("--csv", type=str, required=True, help="CSV produced by collect")
    pcali.add_argument("--min-delta-c", type=float, default=0.5, help="minimum temp delta to attempt a fit")
    pcali.add_argument("--out-json", type=str, help="optional path to write suggestions + debug")
    pcali.set_defaults(func=cmd_calibrate)

    args = p.parse_args()
    args.func(args)

if __name__ == "__main__":
    main()