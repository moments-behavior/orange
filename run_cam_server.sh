#!/usr/bin/env bash
set -euo pipefail

# Path to the binary (relative to this script)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BINARY="$SCRIPT_DIR/release/cam_server"

# Derive the short host name
HOST_RAW="$(hostname -s 2>/dev/null || cat /etc/hostname)"
HOST="${HOST_RAW%%.*}"  # strip any domain

# Normalize common variants (e.g., "waffle0" -> "waffle-0")
if [[ "$HOST" =~ ^waffle([0-9]+)$ ]]; then
  HOST="waffle-${BASH_REMATCH[1]}"
fi

# Validate the derived server name (adjust the pattern as needed)
PATTERN='^waffle-[0-9]+$'
if [[ ! "$HOST" =~ $PATTERN ]]; then
  echo "Error: derived server name '$HOST' doesn't match pattern: $PATTERN" >&2
  echo "Tip: override via env var: CAM_SERVER_NAME=waffle-0 $0" >&2
  exit 1
fi

# Allow manual override via env var if needed
SERVER="${CAM_SERVER_NAME:-$HOST}"

echo "Detected server name: $SERVER"

# Check binary
if [[ ! -x "$BINARY" ]]; then
  echo "Error: '$BINARY' not found or not executable." >&2
  exit 1
fi

# Run with sudo only if not already root
if [[ $EUID -ne 0 ]]; then
  exec sudo "$BINARY" "$SERVER"
else
  exec "$BINARY" "$SERVER"
fi

