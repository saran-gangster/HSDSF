from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Set


REQUIRED_COLUMNS: Set[str] = {
    # time
    "t_unix_s",
    "t_sim_s",
    # identity / labels
    "run_id",
    "binary_id",
    "workload_family",
    "workload_variant",
    "trojan_family",
    "trojan_variant",
    "trojan_active",
    "mode",
    # regime/meta
    "power_mode",
    "ambient_c",
    "input_voltage_v",
    "jetson_clocks",
    # util
    "cpu_util_avg",
    "cpu_util_max",
    "gpu_util",
    "emc_util",
    "cv_util",
    # freq
    "cpu_freq_mhz",
    "gpu_freq_mhz",
    "emc_freq_mhz",
    # temps
    "temp_cpu_c",
    "temp_gpu_c",
    "temp_aux_c",
    "temp_ttp_c",
    "temp_board_c",
    "fan_est_temp_c",
    # fan
    "fan_pwm",
    "fan_rpm",
    "hysteresis_state",
    # power
    "p_sys5v_mw",
    "p_cpu_mw",
    "p_gpu_mw",
    "p_soc_mw",
    "p_cv_mw",
    "p_vddrq_mw",
    # perf deltas
    "delta_cycles",
    "delta_instructions",
    "delta_cache_misses",
    "delta_context_switches",
    "delta_page_faults",
    "delta_llc_load_misses",
    "delta_branch_misses",
    # masks
    "mask_temp_cpu",
    "mask_temp_gpu",
    "mask_temp_aux",
    "mask_temp_ttp",
    "mask_temp_board",
    "mask_fan_est_temp",
    "mask_power_cpu",
    "mask_power_gpu",
    "mask_power_soc",
    "mask_power_cv",
    "mask_power_vddrq",
    "mask_power_sys5v",
    "mask_power_in",
    # perf masks
    "mask_perf_cycles",
    "mask_perf_instructions",
    "mask_perf_cache_misses",
    "mask_perf_context_switches",
    "mask_perf_page_faults",
    "mask_perf_llc_load_misses",
    "mask_perf_branch_misses",
}


@dataclass(frozen=True)
class SchemaIssue:
    kind: str
    message: str
    row_index: Optional[int] = None
    column: Optional[str] = None


def _safe_float(x: object) -> float:
    try:
        return float(x)  # type: ignore[arg-type]
    except Exception:
        return float("nan")


def validate_header(fieldnames: Sequence[str]) -> List[SchemaIssue]:
    present = set(fieldnames)
    missing = sorted(REQUIRED_COLUMNS - present)
    issues: List[SchemaIssue] = []
    if missing:
        issues.append(SchemaIssue(kind="missing_columns", message=f"Missing required columns: {missing}"))
    return issues


def validate_rows(
    rows: Iterable[Dict[str, str]],
    *,
    max_rows: int = 2000,
) -> List[SchemaIssue]:
    issues: List[SchemaIssue] = []
    for i, row in enumerate(rows):
        if i >= max_rows:
            break

        # ranges: util
        for k in ["cpu_util_avg", "cpu_util_max", "gpu_util", "emc_util", "cv_util"]:
            v = _safe_float(row.get(k))
            if v == v:  # not NaN
                if not (0.0 <= v <= 100.0):
                    issues.append(SchemaIssue(kind="range", message=f"{k} out of range: {v}", row_index=i, column=k))

        # ranges: temps
        for k in [
            "temp_cpu_c",
            "temp_gpu_c",
            "temp_aux_c",
            "temp_ttp_c",
            "temp_board_c",
            "fan_est_temp_c",
        ]:
            v = _safe_float(row.get(k))
            if v == v:
                if not (-10.0 <= v <= 120.0):
                    issues.append(SchemaIssue(kind="range", message=f"{k} out of range: {v}", row_index=i, column=k))

        # ranges: power
        p = _safe_float(row.get("p_sys5v_mw"))
        if p == p and p < 0.0:
            issues.append(SchemaIssue(kind="range", message=f"p_sys5v_mw negative: {p}", row_index=i, column="p_sys5v_mw"))

        # masks are binary
        for mk, mv in row.items():
            if not mk.startswith("mask_"):
                continue
            if mv not in {"0", "1", "0.0", "1.0"}:
                issues.append(SchemaIssue(kind="mask", message=f"{mk} not binary: {mv}", row_index=i, column=mk))

    return issues
