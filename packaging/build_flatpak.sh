#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
APP_ID="io.github.marioalabi.BigDataBGRemover"
RUNTIME_VER="25.08"
BUILD_DIR="$ROOT_DIR/flatpak-build"
REPO_DIR="$ROOT_DIR/flatpak-repo"
OUT_BUNDLE="$ROOT_DIR/releases/BigDataBGRemover-x86_64.flatpak"

cd "$ROOT_DIR"
cargo build --release
mkdir -p "$ROOT_DIR/releases"

rm -rf "$BUILD_DIR" "$REPO_DIR"
flatpak build-init "$BUILD_DIR" "$APP_ID" org.freedesktop.Sdk org.freedesktop.Platform "$RUNTIME_VER"

install -Dm755 "$ROOT_DIR/target/release/bigdata-bg-remover" "$BUILD_DIR/files/bin/bigdata-bg-remover"
install -Dm644 "$ROOT_DIR/packaging/bigdata-bg-remover.desktop" "$BUILD_DIR/files/share/applications/${APP_ID}.desktop"
install -Dm644 "$ROOT_DIR/packaging/bigdata-bg-remover.svg" "$BUILD_DIR/files/share/icons/hicolor/scalable/apps/${APP_ID}.svg"

# Update Icon= and Exec= in the installed desktop file to match the Flatpak App ID
sed -i "s/^Icon=.*/Icon=${APP_ID}/" "$BUILD_DIR/files/share/applications/${APP_ID}.desktop"
sed -i "s/^Exec=.*/Exec=bigdata-bg-remover/" "$BUILD_DIR/files/share/applications/${APP_ID}.desktop"

flatpak build-finish "$BUILD_DIR" \
  --command=bigdata-bg-remover \
  --share=ipc \
  --share=network \
  --socket=x11 \
  --socket=wayland \
  --device=dri \
  --filesystem=xdg-pictures \
  --filesystem=xdg-download

flatpak build-export "$REPO_DIR" "$BUILD_DIR"
flatpak build-bundle "$REPO_DIR" "$OUT_BUNDLE" "$APP_ID"

rm -rf "$BUILD_DIR" "$REPO_DIR"
echo "Flatpak bundle creado en: $OUT_BUNDLE"
