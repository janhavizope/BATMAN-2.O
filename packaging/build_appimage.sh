#!/usr/bin/env bash
# Build BATMAN-x86_64.AppImage from the current code (Linux x86_64).
#
# Run from the project root, with the dependencies installed
# (pip install -r requirements.txt pyinstaller) and the Git LFS score files
# pulled (git lfs pull):
#
#     bash packaging/build_appimage.sh
#
# Steps: PyInstaller bundles the app and its data into dist/BATMAN/, this
# script lays that out as an AppDir (launcher, .desktop file, icon), and
# appimagetool packs the AppDir into one executable file.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
BUILD="$ROOT/build/appimage"
APPDIR="$BUILD/BATMAN.AppDir"
OUT="$ROOT/dist/BATMAN-x86_64.AppImage"
APPIMAGETOOL="$BUILD/appimagetool-x86_64.AppImage"
# Pinned appimagetool release (https://github.com/AppImage/appimagetool/releases)
APPIMAGETOOL_URL="https://github.com/AppImage/appimagetool/releases/download/1.9.1/appimagetool-x86_64.AppImage"

# Refuse to package Git LFS pointer files instead of the real scores
if head -c 40 data/elliptic_plus/wallet_scores.parquet | grep -q "git-lfs"; then
  echo "data/elliptic_plus/*.parquet are Git LFS pointers. Run: git lfs install && git lfs pull" >&2
  exit 1
fi

echo "==> PyInstaller"
pyinstaller --noconfirm --clean --distpath dist --workpath "$BUILD/pyinstaller" packaging/batman.spec

echo "==> AppDir"
rm -rf "$APPDIR"
mkdir -p "$APPDIR/usr/lib"
cp -a dist/BATMAN "$APPDIR/usr/lib/BATMAN"
cat > "$APPDIR/AppRun" <<'EOF'
#!/bin/sh
HERE="$(dirname "$(readlink -f "$0")")"
exec "$HERE/usr/lib/BATMAN/BATMAN" "$@"
EOF
chmod +x "$APPDIR/AppRun"
cat > "$APPDIR/batman.desktop" <<'EOF'
[Desktop Entry]
Type=Application
Name=BATMAN
Comment=Bitcoin Anomaly Traffic & Monitoring Analysis Network
Exec=BATMAN
Icon=batman
Categories=Utility;Finance;
Terminal=false
EOF
# Square 256x256 icon cut from the app's background image
python - "$APPDIR/batman.png" <<'EOF'
import sys
from PIL import Image
image = Image.open("resources/images/bg.png").convert("RGBA")
side = min(image.size)
left, top = (image.width - side) // 2, (image.height - side) // 2
image.crop((left, top, left + side, top + side)).resize((256, 256)).save(sys.argv[1])
EOF

echo "==> appimagetool"
if [ ! -x "$APPIMAGETOOL" ]; then
  curl -fsSL -o "$APPIMAGETOOL" "$APPIMAGETOOL_URL"
  chmod +x "$APPIMAGETOOL"
fi
# --appimage-extract-and-run: works without FUSE (e.g. in CI containers)
ARCH=x86_64 "$APPIMAGETOOL" --appimage-extract-and-run "$APPDIR" "$OUT"
ls -lh "$OUT"
