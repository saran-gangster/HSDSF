#!/bin/bash
# Setup script for generic Linux systems (non-Jetson)
# This script sets up the environment for testing Phase 3 components without Jetson hardware

set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PHASE3_ROOT="$(dirname "$SCRIPT_DIR")"

echo "=========================================="
echo "Phase 3 Setup for Linux (non-Jetson)"
echo "=========================================="
echo ""

# Update package lists
echo "1. Updating package lists..."
sudo apt-get update -qq

# Install Python and pip if needed
echo "2. Installing Python3 and pip..."
sudo apt-get install -y python3 python3-pip

# Install perf tools
echo "3. Installing perf tools..."
sudo apt-get install -y linux-tools-common linux-tools-generic
KERNEL_VERSION=$(uname -r)
sudo apt-get install -y linux-tools-${KERNEL_VERSION} || echo "Note: Exact kernel tools may not be available"

# Install Python dependencies
echo "4. Installing Python dependencies from requirements.txt..."
pip3 install --user -r "${PHASE3_ROOT}/requirements.txt"

# Verify perf is available
echo "5. Verifying perf availability..."
if command -v perf &> /dev/null; then
    echo "   perf found"
    perf --version
else
    echo "   WARNING: perf not found in PATH"
fi

# Check perf permissions
echo "6. Checking perf event permissions..."
PARANOID_LEVEL=$(cat /proc/sys/kernel/perf_event_paranoid 2>/dev/null || echo "unknown")
echo "   Current perf_event_paranoid level: ${PARANOID_LEVEL}"
if [ "$PARANOID_LEVEL" -gt 1 ] 2>/dev/null; then
    echo "   To allow non-root perf access, run:"
    echo "   echo 1 | sudo tee /proc/sys/kernel/perf_event_paranoid"
fi

# Create fake tegrastats script for testing
FAKE_TEGRASTATS="${SCRIPT_DIR}/fake_tegrastats.sh"
if [ ! -f "$FAKE_TEGRASTATS" ]; then
    echo ""
    echo "7. Creating fake tegrastats script for testing..."
    cat > "$FAKE_TEGRASTATS" << 'EOF'
#!/bin/bash
# Fake tegrastats for non-Jetson testing
# Prints sample tegrastats-like output at specified interval

INTERVAL_MS=${2:-500}
INTERVAL_S=$(echo "scale=3; $INTERVAL_MS / 1000" | bc)

while true; do
    RAM=$((RANDOM % 4096 + 2048))
    CPU1=$((RANDOM % 100))
    CPU2=$((RANDOM % 100))
    CPU3=$((RANDOM % 100))
    CPU4=$((RANDOM % 100))
    GPU=$((RANDOM % 100))
    EMC=$((RANDOM % 100))
    TEMP=$((RANDOM % 20 + 40))
    PWR=$((RANDOM % 5000 + 2000))
    
    echo "RAM ${RAM}/8192MB CPU [${CPU1}%@1420,${CPU2}%@1420,${CPU3}%@1420,${CPU4}%@1420] EMC_FREQ ${EMC}% GR3D_FREQ ${GPU}% PLL@${TEMP}C CPU@${TEMP}C GPU@${TEMP}C AO@${TEMP}C POM_5V_IN ${PWR}mW"
    
    sleep "$INTERVAL_S"
done
EOF
    chmod +x "$FAKE_TEGRASTATS"
    echo "   Created: ${FAKE_TEGRASTATS}"
fi

echo ""
echo "=========================================="
echo "Setup Complete!"
echo "=========================================="
echo ""
echo "Note: tegrastats is not available on non-Jetson systems."
echo "Use --no-tegrastats flag or --tegrastats-cmd ${SCRIPT_DIR}/fake_tegrastats.sh for testing."
echo ""
echo "Next steps:"
echo "1. Run smoke tests: bash ${SCRIPT_DIR}/run_tests.sh --no-tegrastats"
echo ""
echo "For manual testing, try:"
echo "  cd ${PHASE3_ROOT}"
echo "  python3 scripts/run_experiment.py --label idle --duration 30 --no-tegrastats"
echo ""
