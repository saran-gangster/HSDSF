#!/bin/bash
# Fake tegrastats for non-Jetson testing
# Prints sample tegrastats-like output at specified interval
# Usage: ./fake_tegrastats.sh --interval <milliseconds>

INTERVAL_MS=500

# Parse arguments (minimal tegrastats compatibility)
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

echo "Fake tegrastats running with interval: ${INTERVAL_MS}ms" >&2

while true; do
    # Generate realistic-looking random values
    RAM=$((RANDOM % 4096 + 2048))
    CPU1=$((RANDOM % 100))
    CPU2=$((RANDOM % 100))
    CPU3=$((RANDOM % 100))
    CPU4=$((RANDOM % 100))
    GPU=$((RANDOM % 100))
    EMC=$((RANDOM % 100))
    TEMP_PLL=$((RANDOM % 20 + 40))
    TEMP_CPU=$((RANDOM % 20 + 45))
    TEMP_GPU=$((RANDOM % 25 + 40))
    TEMP_AO=$((RANDOM % 15 + 35))
    PWR=$((RANDOM % 5000 + 2000))
    
    # Format similar to real tegrastats output
    echo "RAM ${RAM}/8192MB (lfb 512x4MB) CPU [${CPU1}%@1420,${CPU2}%@1420,${CPU3}%@1420,${CPU4}%@1420] EMC_FREQ ${EMC}%@1600 GR3D_FREQ ${GPU}%@1300 PLL@${TEMP_PLL}C CPU@${TEMP_CPU}C PMIC@${TEMP_CPU}C GPU@${TEMP_GPU}C AO@${TEMP_AO}C thermal@${TEMP_CPU}C POM_5V_IN ${PWR}/${PWR}mW POM_5V_GPU ${PWR}/${PWR}mW"
    
    sleep "$INTERVAL_S"
done
