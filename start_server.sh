#!/usr/bin/env bash
set -euo pipefail

# Path to the binary (relative to this script)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BINARY="$SCRIPT_DIR/release/cam_server"

# Server name: $CAM_SERVER_NAME if set, else the host's short hostname.
HOST_RAW="$(hostname -s 2>/dev/null || cat /etc/hostname)"
SERVER="${CAM_SERVER_NAME:-${HOST_RAW%%.*}}"

echo "Starting cam_server as: $SERVER"

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
