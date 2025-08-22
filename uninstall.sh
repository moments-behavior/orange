#!/usr/bin/env bash
set -euo pipefail

APP_NAME="red"

YES=0
DRYRUN=0
PREFIX="$HOME/.local"   # matches install.sh default

usage() {
  cat <<EOF
Usage: $0 [OPTIONS] [PREFIX]

Remove user-local installation of $APP_NAME.

Options:
  -y, --yes       Do not prompt for confirmation
  -n, --dry-run   Show what would be removed without deleting
  -h, --help      Show this help

Arguments:
  PREFIX          Install prefix used earlier (default: \$HOME/.local)
EOF
}

# ---- Parse args
while [[ $# -gt 0 ]]; do
  case "$1" in
    -y|--yes) YES=1; shift ;;
    -n|--dry-run) DRYRUN=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) PREFIX="$1"; shift ;;
  esac
done

BINDIR="$PREFIX/bin"
APPDIR="$PREFIX/opt/$APP_NAME"
DESKTOP_FILE="$HOME/.local/share/applications/$APP_NAME.desktop"
ICON_BASE="$HOME/.local/share/icons/hicolor"
ICON_256="$ICON_BASE/256x256/apps/$APP_NAME.png"

targets=(
  "$APPDIR"
  "$BINDIR/$APP_NAME"
  "$DESKTOP_FILE"
  "$ICON_256"
)

echo "[*] Uninstalling $APP_NAME from prefix: $PREFIX"
echo "    This will remove:"
for t in "${targets[@]}"; do
  echo "      - $t"
done

if [[ $YES -eq 0 ]]; then
  read -r -p "Proceed? [y/N] " ans
  [[ "${ans:-}" =~ ^[Yy]$ ]] || { echo "Aborted."; exit 1; }
fi

remove_path() {
  local p="$1"
  if [[ -e "$p" || -L "$p" ]]; then
    echo "rm -rf \"$p\""
    [[ $DRYRUN -eq 0 ]] && rm -rf "$p"
  else
    echo "(not found) $p"
  fi
}

for t in "${targets[@]}"; do
  remove_path "$t"
done

# Refresh caches (if not dry-run and tools present)
if [[ $DRYRUN -eq 0 ]]; then
  command -v update-desktop-database >/dev/null 2>&1 && \
    update-desktop-database "$HOME/.local/share/applications" || true

  command -v gtk-update-icon-cache >/dev/null 2>&1 && \
    gtk-update-icon-cache -f "$ICON_BASE" || true
fi

echo "✔ Uninstall complete."

