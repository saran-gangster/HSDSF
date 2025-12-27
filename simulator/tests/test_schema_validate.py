import csv
import glob
import math
import os
import unittest


REQUIRED_COLUMNS = {
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
    "mask_perf_cycles",
    "mask_perf_instructions",
    "mask_perf_cache_misses",
    "mask_perf_context_switches",
    "mask_perf_page_faults",
    "mask_perf_llc_load_misses",
    "mask_perf_branch_misses",
}


def _to_float(x: str) -> float:
    try:
        return float(x)
    except Exception:
        return float("nan")


class TestTelemetrySchemaValidate(unittest.TestCase):
    def test_first_available_run_csv_validates(self):
        repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
        runs_glob = os.path.join(repo_root, "data", "fusionbench_sim", "runs", "run_*", "telemetry.csv")
        matches = sorted(glob.glob(runs_glob))
        if not matches:
            self.skipTest("No generated runs found under data/fusionbench_sim/runs; generate a run first")

        path = matches[0]
        with open(path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            self.assertIsNotNone(reader.fieldnames)
            fieldnames = set(reader.fieldnames or [])
            missing = sorted(REQUIRED_COLUMNS - fieldnames)
            self.assertFalse(missing, f"Missing required columns: {missing}")

            for i, row in enumerate(reader):
                # only scan a prefix for speed
                if i >= 500:
                    break

                # ranges
                for k in ["cpu_util_avg", "cpu_util_max", "gpu_util", "emc_util", "cv_util"]:
                    v = _to_float(row[k])
                    if not math.isnan(v):
                        self.assertGreaterEqual(v, 0.0, k)
                        self.assertLessEqual(v, 100.0, k)

                for k in ["temp_cpu_c", "temp_gpu_c", "temp_aux_c", "temp_ttp_c", "temp_board_c", "fan_est_temp_c"]:
                    v = _to_float(row[k])
                    if not math.isnan(v):
                        self.assertGreaterEqual(v, -10.0, k)
                        self.assertLessEqual(v, 120.0, k)

                self.assertGreaterEqual(_to_float(row["p_sys5v_mw"]), 0.0)

                # masks are binary
                mask_cols = [c for c in row.keys() if c.startswith("mask_")]
                for mk in mask_cols:
                    mv = row[mk]
                    self.assertIn(mv, {"0", "1", "0.0", "1.0"}, mk)


if __name__ == "__main__":
    unittest.main()
