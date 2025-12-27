# HSDSF Telemetry Schema (Canonical Contract)

This schema is the contract between simulator and downstream preprocessing/training.

## Required Columns

### Time
- `t_unix_s` (float): UNIX seconds
- `t_sim_s` (float): simulation seconds since run start (real runs may set `t_sim_s = t_unix_s - t0`)

### Identity / labels
- `run_id` (string)
- `binary_id` (string)
- `workload_family` (string)
- `workload_variant` (string)
- `trojan_family` (string)
- `trojan_variant` (string)
- `trojan_active` (int 0/1)
- `mode` (string: `idle` | `normal` | `trojan`)

### Regime / meta
- `power_mode` (string: `30W` | `MAXN`)
- `ambient_c` (float)
- `input_voltage_v` (float)
- `jetson_clocks` (int 0/1)

### Telemetry numeric
Util (%):
- `cpu_util_avg`, `cpu_util_max`, `gpu_util`, `emc_util`, `cv_util`

Freq:
- `cpu_freq_mhz`, `gpu_freq_mhz`, `emc_freq_mhz`

Temps (°C):
- `temp_cpu_c`, `temp_gpu_c`, `temp_aux_c`, `temp_ttp_c`, `temp_board_c`, `fan_est_temp_c`

Fan:
- `fan_pwm`, `fan_rpm`, `hysteresis_state`

Power (mW):
- `p_sys5v_mw`, `p_cpu_mw`, `p_gpu_mw`, `p_soc_mw`, `p_cv_mw`, `p_vddrq_mw`

Perf deltas (per sampling interval):
- `delta_cycles`, `delta_instructions`, `delta_cache_misses`, `delta_context_switches`, `delta_page_faults`, `delta_llc_load_misses`, `delta_branch_misses`

### Missingness / staleness masks
Definition: `1 = fresh update`, `0 = stale/reused`.

Temp masks:
- `mask_temp_cpu`, `mask_temp_gpu`, `mask_temp_aux`, `mask_temp_ttp`, `mask_temp_board`, `mask_fan_est_temp`

Power masks:
- `mask_power_cpu`, `mask_power_gpu`, `mask_power_soc`, `mask_power_cv`, `mask_power_vddrq`, `mask_power_sys5v`, `mask_power_in`

Perf masks (for multiplexing events):
- `mask_perf_cycles`, `mask_perf_instructions`, `mask_perf_cache_misses`, `mask_perf_context_switches`, `mask_perf_page_faults`, `mask_perf_llc_load_misses`, `mask_perf_branch_misses`

Notes:
- Simulator may reuse cached values for realism; masks must expose that reuse.
- Sensor dropout events reuse cached values and force the corresponding mask to `0`.
- Perf multiplexing hides some counters; hidden counters keep their totals frozen and expose mask `0`.
