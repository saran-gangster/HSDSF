#!/bin/bash
# Run Phase 3 smoke tests
# Quick validation of telemetry collection and workloads

set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PHASE3_ROOT="$(dirname "$SCRIPT_DIR")"

# Parse arguments
NO_TEGRASTATS=""
FAKE_TEGRASTATS=""
DURATION=30

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
        --duration)
            DURATION="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 [--no-tegrastats] [--fake-tegrastats] [--duration SECONDS]"
            exit 1
            ;;
    esac
done

cd "$PHASE3_ROOT"

echo "=========================================="
echo "Phase 3 Smoke Tests"
echo "=========================================="
echo "Test duration: ${DURATION}s per scenario"
echo "Tegrastats: ${NO_TEGRASTATS:-enabled} ${FAKE_TEGRASTATS}"
echo ""

# Test 1: Idle
echo "Test 1/3: Idle scenario..."
python3 scripts/run_experiment.py \
    --label idle \
    --duration "$DURATION" \
    $NO_TEGRASTATS \
    $FAKE_TEGRASTATS

echo ""

# Test 2: Normal workload
echo "Test 2/3: Normal workload..."
python3 scripts/run_experiment.py \
    --label normal \
    --duration "$DURATION" \
    $NO_TEGRASTATS \
    $FAKE_TEGRASTATS

echo ""

# Test 3: Trojan compute
echo "Test 3/3: Trojan compute workload..."
python3 scripts/run_experiment.py \
    --label trojan_compute \
    --trojan-variant compute \
    --duration "$DURATION" \
    $NO_TEGRASTATS \
    $FAKE_TEGRASTATS

echo ""
echo "=========================================="
echo "Smoke Tests Complete!"
echo "=========================================="
echo ""

# Find the most recent run directories
echo "Recent test outputs:"
ls -td data/runs/*/ | head -3 | while read -r dir; do
    echo "  - $dir"
    
    # Quick validation
    TELEMETRY="${dir}telemetry.csv"
    META="${dir}meta.json"
    
    if [ -f "$TELEMETRY" ]; then
        ROW_COUNT=$(wc -l < "$TELEMETRY")
        EXPECTED=$((DURATION * 10))  # ~10 Hz
        echo "    Rows: ${ROW_COUNT} (expected ~${EXPECTED})"
        
        # Check for key columns
        HEADER=$(head -1 "$TELEMETRY")
        if echo "$HEADER" | grep -q "perf_cycles"; then
            echo "    ✓ perf_cycles found"
        else
            echo "    ✗ perf_cycles missing"
        fi
        
        if echo "$HEADER" | grep -q "perf_ipc"; then
            echo "    ✓ perf_ipc found"
        else
            echo "    ✗ perf_ipc missing"
        fi
    else
        echo "    ✗ telemetry.csv not found"
    fi
    
    if [ -f "$META" ]; then
        echo "    ✓ meta.json found"
    else
        echo "    ✗ meta.json missing"
    fi
    
    echo ""
done

echo ""
echo "To inspect telemetry data, try:"
echo "  head -20 data/runs/\$(ls -t data/runs | head -1)/telemetry.csv"
echo "  cat data/runs/\$(ls -t data/runs | head -1)/meta.json"
echo ""
