#!/bin/bash
# Simulator-backed tegrastats replacement for Phase 3 data collection
# Polls jetson_sim.py HTTP API and formats output like tegrastats
# Usage: ./fake_tegrastats.sh --interval <milliseconds>

PORT="${TEGRA_SIM_PORT:-45215}"
INTERVAL_MS=500

# Parse arguments (tegrastats compatibility)
while [[ $# -gt 0 ]]; do
    case $1 in
        --interval)
            INTERVAL_MS="$2"
            shift 2
            ;;
        *)
            shift
            ;;
    esac
done

INTERVAL_S=$(echo "scale=3; $INTERVAL_MS / 1000" | bc)

echo "Simulator tegrastats running on port ${PORT} with interval: ${INTERVAL_MS}ms" >&2

while true; do
    # Poll simulator API
    DATA=$(curl -s "http://127.0.0.1:${PORT}/v1/telemetry" 2>/dev/null)
    if [ $? -ne 0 ] || [ -z "$DATA" ]; then
        sleep "$INTERVAL_S"
        continue
    fi

    # Extract fields using grep/sed for portability
    RAM_USED=$(echo "$DATA" | grep -o '"ram_used_mb":[0-9.]*' | cut -d: -f2 | head -1)
    RAM_TOTAL=$(echo "$DATA" | grep -o '"ram_total_mb":[0-9]*' | cut -d: -f2 | head -1)
    GPU_UTIL=$(echo "$DATA" | grep -o '"gpu_util":[0-9.]*' | cut -d: -f2 | head -1)
    EMC_UTIL=$(echo "$DATA" | grep -o '"emc_util":[0-9.]*' | cut -d: -f2 | head -1)
    
    # CPU utils sum from array
    CPU_UTILS=$(echo "$DATA" | grep -o '"cpu_util":\[[^]]*\]' | sed 's/.*\[\([^]]*\)\].*/\1/' | tr -d ' ')
    CPU_SUM=0
    for u in $(echo "$CPU_UTILS" | tr ',' ' '); do
        CPU_SUM=$(echo "$CPU_SUM + $u" | bc)
    done
    
    TEMP_CPU=$(echo "$DATA" | grep -o '"temp_cpu_c":[0-9.]*' | cut -d: -f2 | head -1)
    TEMP_GPU=$(echo "$DATA" | grep -o '"temp_gpu_c":[0-9.]*' | cut -d: -f2 | head -1)
    TEMP_AUX=$(echo "$DATA" | grep -o '"temp_aux_c":[0-9.]*' | cut -d: -f2 | head -1)
    TEMP_TTP=$(echo "$DATA" | grep -o '"temp_ttp_c":[0-9.]*' | cut -d: -f2 | head -1)
    
    P_SYS=$(echo "$DATA" | grep -o '"p_sys5v_mw":[0-9.]*' | cut -d: -f2 | head -1)
    P_CPU=$(echo "$DATA" | grep -o '"p_cpu_mw":[0-9.]*' | cut -d: -f2 | head -1)
    P_GPU=$(echo "$DATA" | grep -o '"p_gpu_mw":[0-9.]*' | cut -d: -f2 | head -1)
    P_SOC=$(echo "$DATA" | grep -o '"p_soc_mw":[0-9.]*' | cut -d: -f2 | head -1)
    P_CV=$(echo "$DATA" | grep -o '"p_cv_mw":[0-9.]*' | cut -d: -f2 | head -1)
    
    GPU_FREQ=$(echo "$DATA" | grep -o '"gpu_freq_mhz":[0-9]*' | cut -d: -f2 | head -1)
    EMC_FREQ=$(echo "$DATA" | grep -o '"emc_freq_mhz":[0-9]*' | cut -d: -f2 | head -1)
    
    # Format like tegrastats output
    printf "RAM %d/%dMB (lfb 512x4MB) CPU [" "${RAM_USED%%.*}" "${RAM_TOTAL}"
    # Individual CPU utils
    first=1
    for u in $(echo "$CPU_UTILS" | tr ',' ' '); do
        [ $first -eq 0 ] && printf ","
        printf "%d%%@1420" "${u%%.*}"
        first=0
    done
    printf "] EMC_FREQ %d%%@%d GR3D_FREQ %d%%@%d " \
        "${EMC_UTIL%%.*}" "${EMC_FREQ}" "${GPU_UTIL%%.*}" "${GPU_FREQ}"
    printf "CPU@%.1fC GPU@%.1fC AUX@%.1fC TTP@%.1fC " \
        "${TEMP_CPU}" "${TEMP_GPU}" "${TEMP_AUX}" "${TEMP_TTP}"
    printf "POM_5V_IN %dmW VDD_CPU %dmW VDD_GPU %dmW VDD_SOC %dmW VDD_CV %dmW\n" \
        "${P_SYS%%.*}" "${P_CPU%%.*}" "${P_GPU%%.*}" "${P_SOC%%.*}" "${P_CV%%.*}"
    
    sleep "$INTERVAL_S"
done
