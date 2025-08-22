#!/usr/bin/env bash
set -euo pipefail

APP_NAME="orange"

# --- Locations (run this script from your project root) ---
SRC_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BIN_SRC="$SRC_DIR/release/$APP_NAME"
FONTS_SRC="$SRC_DIR/fonts"
ICON_SRC="$SRC_DIR/icon.png"

# Optional prefix (default to ~/.local)
PREFIX="${1:-"$HOME/.local"}"

# Install targets
BINDIR="$PREFIX/bin"
APPDIR="$PREFIX/opt/$APP_NAME"                 # app bundle dir (binary + fonts + icon)
DESKTOP_DIR="$HOME/.local/share/applications"  # user desktop entries

echo "[*] Installing $APP_NAME into: $PREFIX"
echo "    - Binary:  $BIN_SRC"
echo "    - Fonts:   $FONTS_SRC (if exists)"
echo "    - Icon:    $ICON_SRC  (if exists)"

# --- Sanity checks ---
if [[ ! -f "$BIN_SRC" ]]; then
  echo "ERROR: $BIN_SRC not found. Build your app or check the path." >&2
  exit 1
fi

mkdir -p "$APPDIR" "$BINDIR" "$DESKTOP_DIR"

# --- Copy binary into appdir (not directly into bin to keep fonts nearby) ---
install -m 755 "$BIN_SRC" "$APPDIR/$APP_NAME"

# --- Copy fonts (if present) ---
if [[ -d "$FONTS_SRC" ]]; then
  mkdir -p "$APPDIR/fonts"
  if command -v rsync >/dev/null 2>&1; then
    rsync -a --delete "$FONTS_SRC"/ "$APPDIR/fonts"/
  else
    # -a preserves perms; '/.' copies contents not the dir itself
    cp -a "$FONTS_SRC"/. "$APPDIR/fonts"/
  fi
else
  echo "WARN: fonts/ not found; continuing without it."
fi

# --- Install icon into appdir and reference it by absolute path in .desktop ---
if [[ -f "$ICON_SRC" ]]; then
  cp -f "$ICON_SRC" "$APPDIR/icon.png"
else
  echo "WARN: icon.png not found; launcher will have a generic icon."
fi

# --- Wrapper in ~/.local/bin so the app runs from its own dir (finds fonts/) ---
WRAPPER="$BINDIR/$APP_NAME"
cat > "$WRAPPER" <<EOF
#!/usr/bin/env bash
APPDIR="$APPDIR"
cd "\$APPDIR"
exec "\$APPDIR/$APP_NAME" "\$@"
EOF
chmod +x "$WRAPPER"

# --- Desktop entry (launcher) ---
DESKTOP_FILE="$DESKTOP_DIR/$APP_NAME.desktop"
ICON_PATH="$APPDIR/$APP_NAME.png"   # absolute path avoids theme cache issues

cat > "$DESKTOP_FILE" <<EOF
[Desktop Entry]
Type=Application
Name=Orange
Comment=Orange Application
TryExec=$BINDIR/$APP_NAME
Exec=$BINDIR/$APP_NAME %F
Icon=$ICON_PATH
Terminal=false
Categories=Graphics;Utility;
# Uncomment if you set your window class to "orange" (helps dock show the right icon)
# StartupWMClass=orange
EOF

# --- Refresh desktop database (icon cache not needed for absolute icon path) ---
if command -v update-desktop-database >/dev/null 2>&1; then
  update-desktop-database "$DESKTOP_DIR" || true
fi

echo
echo "✔ Install complete."
echo "➤ Run: $APP_NAME"
echo "   (Make sure '$BINDIR' is in your PATH. If not, add this to your shell rc:)"
echo "     export PATH=\"\$HOME/.local/bin:\$PATH\""
