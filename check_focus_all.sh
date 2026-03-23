#!/bin/bash
# check_focus_all.sh
# Runs focus_check on the main machine, dosa-0, and dosa-1, then reports
# aggregated pass/fail. Run this before starting any recording session.
#
# Usage:
#   ./check_focus_all.sh                     # basic check
#   ./check_focus_all.sh --save-frames       # also save PNG per camera to /tmp/focus_check_frames/
#   ./check_focus_all.sh --threshold 150     # stricter sharpness threshold
#
# Requirements:
#   - focus_check binary built:  cd build && make focus_check
#   - SSH key-based auth to dosa-0 and dosa-1 (no password prompt)

set -euo pipefail

DOSA0="192.168.1.100"
DOSA1="192.168.1.110"
ORANGE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FOCUS_CHECK="$ORANGE_DIR/build/focus_check"
EXTRA_ARGS=("$@")

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'  # no colour

overall_pass=0

# --------------------------------------------------------------------------
run_local() {
    echo -e "${YELLOW}===== LOCAL (main machine) =====${NC}"
    if [ ! -x "$FOCUS_CHECK" ]; then
        echo -e "${RED}ERROR: $FOCUS_CHECK not found or not executable.${NC}"
        echo "Build it with:  cd $ORANGE_DIR/build && make focus_check"
        return 1
    fi
    cd "$ORANGE_DIR"
    "$FOCUS_CHECK" "${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}"
}

run_remote() {
    local label=$1
    local ip=$2
    echo -e "${YELLOW}===== $label ($ip) =====${NC}"
    if ! ssh -o ConnectTimeout=5 -o BatchMode=yes "$ip" true 2>/dev/null; then
        echo -e "${RED}ERROR: Cannot SSH to $ip. Check connectivity and SSH keys.${NC}"
        return 1
    fi
    ssh -o ConnectTimeout=10 "$ip" \
        "cd $ORANGE_DIR && $FOCUS_CHECK ${EXTRA_ARGS[*]+"${EXTRA_ARGS[*]}"}"
}

# --------------------------------------------------------------------------
echo ""
echo "Orange focus pre-check — $(date)"
echo "Threshold: ${EXTRA_ARGS[*]:-default (100)}"
echo ""

run_local        || overall_pass=1
echo ""
run_remote "dosa-0" "$DOSA0" || overall_pass=1
echo ""
run_remote "dosa-1" "$DOSA1" || overall_pass=1
echo ""

echo "============================================================"
if [ "$overall_pass" -eq 0 ]; then
    echo -e "${GREEN}[PASS] All machines: cameras are in focus. Safe to record.${NC}"
else
    echo -e "${RED}[FAIL] One or more cameras need focus adjustment before recording.${NC}"
    echo ""
    echo "  Common fixes:"
    echo "  1. If you see 'focus NOT applied': update 'focus' in"
    echo "     config/<serial>.json to a value inside the reported range."
    echo "  2. If sharpness is low but focus was applied: use the orange"
    echo "     GUI Focus slider to find a sharper position, then update"
    echo "     the JSON config with that value."
    echo "  3. Re-run this script after any config change."
fi
echo ""

exit "$overall_pass"
