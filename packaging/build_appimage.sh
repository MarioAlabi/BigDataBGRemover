#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
APP_DIR="$ROOT_DIR/AppDir"
OUT_IMAGE="$ROOT_DIR/BigDataBGRemover-x86_64.AppImage"
TOOL="${APPIMAGETOOL:-/tmp/appimagetool-x86_64.AppImage}"

cd "$ROOT_DIR"
cargo build --release

rm -rf "$APP_DIR"
mkdir -p "$APP_DIR/usr/bin"
cp "$ROOT_DIR/target/release/bigdata-bg-remover" "$APP_DIR/usr/bin/bigdata-bg-remover"
cp "$ROOT_DIR/packaging/bigdata-bg-remover.desktop" "$APP_DIR/bigdata-bg-remover.desktop"
cp "$ROOT_DIR/packaging/bigdata-bg-remover.svg" "$APP_DIR/bigdata-bg-remover.svg"
cp "$ROOT_DIR/packaging/bigdata-bg-remover.svg" "$APP_DIR/.DirIcon"

cat > "$APP_DIR/AppRun" <<'EOF'
#!/usr/bin/env bash
HERE="$(dirname "$(readlink -f "${0}")")"
exec "$HERE/usr/bin/bigdata-bg-remover" "$@"
EOF
chmod +x "$APP_DIR/AppRun"

ARCH=x86_64 "$TOOL" --appimage-extract-and-run "$APP_DIR" "$OUT_IMAGE"
chmod +x "$OUT_IMAGE"
echo "AppImage creado en: $OUT_IMAGE"
