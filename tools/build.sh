#!/usr/bin/env bash
# Build DiaBloS standalone app and macOS DMG installer.
#
# Usage:
#   source ~/.venvs/diablos-arm64/bin/activate && ./tools/build.sh      # arm64
#   source ~/.venvs/evaluate-diablos/bin/activate && ./tools/build.sh   # x86_64
#
# Steps:
#   1. Sync _BLOCK_MODULES in block_loader.py with blocks/ directory
#   2. Run PyInstaller to produce the app bundle
#   3. Package into a DMG with Applications shortcut
#
# The DMG is named by architecture: DiaBloS-arm64.dmg or DiaBloS-x86_64.dmg
set -euo pipefail
cd "$(dirname "$0")/.."

# Detect architecture from the active Python
ARCH=$(python -c "import platform; print(platform.machine())")
DMG_NAME="DiaBloS-${ARCH}.dmg"

echo "==> Architecture: ${ARCH}"
echo "==> Syncing block registry..."
python tools/sync_block_registry.py

echo "==> Building with PyInstaller..."
rm -rf dist/DiaBloS dist/DiaBloS.app build/diablos
pyinstaller --noconfirm diablos.spec

echo "==> Creating DMG installer..."
STAGING=$(mktemp -d)
cp -R dist/DiaBloS.app "$STAGING/"
ln -s /Applications "$STAGING/Applications"
rm -f "dist/${DMG_NAME}"
hdiutil create -volname "DiaBloS" -srcfolder "$STAGING" -ov -format UDZO "dist/${DMG_NAME}"
rm -rf "$STAGING"

echo "==> Done."
echo "    App:       dist/DiaBloS.app"
echo "    Installer: dist/${DMG_NAME}"
