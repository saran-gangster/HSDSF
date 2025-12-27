#!/bin/bash
# Setup script for NVIDIA Jetson hardware
# Run this script on the Jetson device to install dependencies and configure for optimal data collection

set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PHASE3_ROOT="$(dirname "$SCRIPT_DIR")"

echo "=========================================="
echo "Phase 3 Setup for NVIDIA Jetson"
echo "=========================================="
echo ""

# Check if running on Jetson
if [ ! -f /etc/nv_tegra_release ]; then
    echo "WARNING: /etc/nv_tegra_release not found. This may not be a Jetson device."
    echo "Continuing anyway..."
    echo ""
fi

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

# Verify tegrastats is available
echo "5. Verifying tegrastats availability..."
if command -v tegrastats &> /dev/null; then
    echo "   tegrastats found"
    sudo tegrastats --interval 1000 --count 1 > /dev/null 2>&1 || echo "   Warning: tegrastats test run failed"
else
    echo "   WARNING: tegrastats not found in PATH"
fi

# Verify perf is available
echo "6. Verifying perf availability..."
if command -v perf &> /dev/null; then
    echo "   perf found"
    perf --version
else
    echo "   WARNING: perf not found in PATH"
fi

# Check perf permissions
echo "7. Checking perf event permissions..."
PARANOID_LEVEL=$(cat /proc/sys/kernel/perf_event_paranoid 2>/dev/null || echo "unknown")
echo "   Current perf_event_paranoid level: ${PARANOID_LEVEL}"
if [ "$PARANOID_LEVEL" -gt 1 ] 2>/dev/null; then
    echo "   To allow non-root perf access, run:"
    echo "   echo 1 | sudo tee /proc/sys/kernel/perf_event_paranoid"
fi

# Configure Jetson for performance (optional, requires sudo)
echo ""
echo "8. Jetson Performance Configuration (optional)"
echo "   Would you like to configure the Jetson for maximum performance? (y/n)"
read -r CONFIGURE_PERF

if [[ "$CONFIGURE_PERF" =~ ^[Yy]$ ]]; then
    echo "   Checking nvpmodel..."
    if command -v nvpmodel &> /dev/null; then
        echo "   Setting nvpmodel to mode 0 (MAXN)..."
        sudo nvpmodel -m 0 || echo "   Warning: nvpmodel command failed"
        sudo nvpmodel -q
    else
        echo "   nvpmodel not found, skipping"
    fi

    echo "   Checking jetson_clocks..."
    if command -v jetson_clocks &> /dev/null; then
        echo "   Enabling jetson_clocks..."
        sudo jetson_clocks || echo "   Warning: jetson_clocks command failed"
        sudo jetson_clocks --show
    else
        echo "   jetson_clocks not found, skipping"
    fi
else
    echo "   Skipping performance configuration"
fi

echo ""
echo "=========================================="
echo "Setup Complete!"
echo "=========================================="
echo ""
echo "Next steps:"
echo "1. Run smoke tests: bash ${SCRIPT_DIR}/run_tests.sh"
echo "2. Collect full dataset: bash ${SCRIPT_DIR}/collect_dataset.sh"
echo ""
echo "For manual testing, try:"
echo "  cd ${PHASE3_ROOT}"
echo "  python3 scripts/run_experiment.py --label idle --duration 30"
echo ""
