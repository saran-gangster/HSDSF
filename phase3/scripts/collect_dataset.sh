#!/bin/bash
# Full dataset collection protocol for Phase 3
# Implements the dataset collection plan from planforphase3.md section 7.3
#
# Total runtime: ~90-120 minutes
# - Idle: 3 × 10 minutes = 30 min
# - Normal: 3 × 10 minutes = 30 min
# - Trojan compute: 2 × 10 minutes = 20 min
# - Trojan memory: 2 × 10 minutes = 20 min
# - Trojan io: 2 × 10 minutes = 20 min

set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PHASE3_ROOT="$(dirname "$SCRIPT_DIR")"

# Default parameters
DURATION=600  # 10 minutes
NO_TEGRASTATS=""
FAKE_TEGRASTATS=""
SAVE_RAW=""
DRY_RUN=false

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --no-tegrastats)
            NO_TEGRASTATS="--no-tegrastats"
            shift
            ;;
        --fake-tegrastats)
            FAKE_TEGRASTATS="--tegrastats-cmd ${SCRIPT_DIR}/fake_tegrastats.sh"
            shift
            ;;
        --save-raw)
            SAVE_RAW="--save-raw"
            shift
            ;;
        --duration)
            DURATION="$2"
            shift 2
            ;;
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --no-tegrastats       Skip tegrastats collection (for non-Jetson)"
            echo "  --fake-tegrastats     Use fake tegrastats script for testing"
            echo "  --save-raw            Save raw perf/tegrastats streams"
            echo "  --duration SECONDS    Duration per run (default: 600 = 10 min)"
            echo "  --dry-run             Print commands without executing"
            exit 1
            ;;
    esac
done

cd "$PHASE3_ROOT"

echo "=========================================="
echo "Phase 3 Full Dataset Collection"
echo "=========================================="
echo "Duration per run: ${DURATION}s ($(echo "scale=1; $DURATION / 60" | bc) min)"
echo "Tegrastats: ${NO_TEGRASTATS:-enabled} ${FAKE_TEGRASTATS}"
echo "Save raw streams: ${SAVE_RAW:-disabled}"
echo "Dry run: $DRY_RUN"
echo ""

TOTAL_RUNS=11
CURRENT_RUN=0

# Helper function to run experiment
run_exp() {
    local label=$1
    local variant=$2
    local overlay=$3
    
    CURRENT_RUN=$((CURRENT_RUN + 1))
    
    echo ""
    echo "=========================================="
    echo "Run ${CURRENT_RUN}/${TOTAL_RUNS}: ${label}${variant:+ ($variant)}${overlay:+ with overlay}"
    echo "=========================================="
    echo "Start time: $(date)"
    
    CMD=(python3 scripts/run_experiment.py
        --label "$label"
        --duration "$DURATION"
        $NO_TEGRASTATS
        $FAKE_TEGRASTATS
        $SAVE_RAW)
    
    if [ -n "$variant" ]; then
        CMD+=(--trojan-variant "$variant")
    fi
    
    if [ "$overlay" = "true" ]; then
        CMD+=(--trojan-overlay)
    fi
    
    if [ "$DRY_RUN" = true ]; then
        echo "[DRY RUN] Would execute: ${CMD[*]}"
    else
        "${CMD[@]}"
        echo "End time: $(date)"
    fi
}

# === IDLE SCENARIOS (3 runs) ===
echo ""
echo "======================================"
echo "PHASE 1: IDLE Scenarios (3 runs)"
echo "======================================"
run_exp "idle" "" ""
run_exp "idle" "" ""
run_exp "idle" "" ""

# === NORMAL WORKLOAD (3 runs) ===
echo ""
echo "======================================"
echo "PHASE 2: NORMAL Workload (3 runs)"
echo "======================================"
run_exp "normal" "" ""
run_exp "normal" "" ""
run_exp "normal" "" ""

# === TROJAN COMPUTE (2 runs) ===
echo ""
echo "======================================"
echo "PHASE 3: TROJAN COMPUTE (2 runs)"
echo "======================================"
run_exp "trojan_compute" "compute" ""
run_exp "trojan_compute" "compute" ""

# === TROJAN MEMORY (2 runs) ===
echo ""
echo "======================================"
echo "PHASE 4: TROJAN MEMORY (2 runs)"
echo "======================================"
run_exp "trojan_memory" "memory" ""
run_exp "trojan_memory" "memory" ""

# === TROJAN I/O (1 run, last optional run for 11 total) ===
echo ""
echo "======================================"
echo "PHASE 5: TROJAN I/O (1 run)"
echo "======================================"
run_exp "trojan_io" "io" ""

echo ""
echo "=========================================="
echo "Dataset Collection Complete!"
echo "=========================================="
echo "Total runs: ${CURRENT_RUN}"
echo "End time: $(date)"
echo ""

# Summary
if [ "$DRY_RUN" = false ]; then
    echo "Output directory: data/runs/"
    echo ""
    echo "Dataset summary:"
    for label in idle normal trojan_compute trojan_memory trojan_io; do
        COUNT=$(find data/runs -type d -name "*_${label}" | wc -l)
        echo "  ${label}: ${COUNT} runs"
    done
    echo ""
    
    # Disk usage
    TOTAL_SIZE=$(du -sh data/runs 2>/dev/null | cut -f1 || echo "unknown")
    echo "Total size: ${TOTAL_SIZE}"
    echo ""
    
    echo "Next steps:"
    echo "1. Validate dataset: Check for expected row counts (~$(echo "$DURATION * 10" | bc) rows per run)"
    echo "2. Inspect samples: head -20 data/runs/\$(ls -t data/runs | head -1)/telemetry.csv"
    echo "3. Proceed to Phase 4: Data preprocessing and model training"
fi

echo ""
