#!/bin/bash
# Builds Untie.app with PyInstaller (run from the repository root, after tools/export_models.py).
set -euo pipefail
test -f untie/models/meta.json
python -m PyInstaller --noconfirm --clean --windowed \
  --name Untie \
  --icon packaging/icon.png \
  --osx-bundle-identifier in.soundbrewery.untie \
  --paths . \
  --add-data "untie/index.html:untie" \
  --add-data "untie/models:untie/models" \
  --hidden-import untie.app \
  --hidden-import untie.engine \
  --hidden-import untie.models \
  --hidden-import untie.selftest \
  --hidden-import untie.audio_io \
  --hidden-import scipy.cluster.hierarchy \
  --hidden-import untie.aaf_export \
  --hidden-import untie.pitch \
  --hidden-import untie.voice_profile \
  --hidden-import scipy.optimize \
  --hidden-import scipy.signal \
  --collect-all aaf2 \
  --exclude-module speechbrain \
  --exclude-module torchaudio \
  --exclude-module huggingface_hub \
  --exclude-module matplotlib \
  --exclude-module tkinter \
  --exclude-module IPython \
  ${EXTRA_PYI_ARGS:-} \
  packaging/launch.py
# make the window title / menu say Un-tie
/usr/libexec/PlistBuddy -c "Set :CFBundleName Un-tie" dist/Untie.app/Contents/Info.plist || true
/usr/libexec/PlistBuddy -c "Add :CFBundleDisplayName string Un-tie" dist/Untie.app/Contents/Info.plist || true
/usr/libexec/PlistBuddy -c "Add :NSHighResolutionCapable bool true" dist/Untie.app/Contents/Info.plist || true
