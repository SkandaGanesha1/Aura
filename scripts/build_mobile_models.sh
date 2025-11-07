#!/usr/bin/env bash
set -euo pipefail

# Copies compiled ExecuTorch artifacts into the Android project assets directory.
# Usage:
#   bash build_mobile_models.sh --compiled-dir ../models/compiled --android-dir ../android

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

COMPILED_DIR="${PROJECT_ROOT}/models/compiled"
ANDROID_DIR="${PROJECT_ROOT}/android"
ASSETS_DIR="${ANDROID_DIR}/app/src/main/assets/models"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --compiled-dir)
      COMPILED_DIR="$(realpath "$2")"
      shift 2
      ;;
    --android-dir)
      ANDROID_DIR="$(realpath "$2")"
      shift 2
      ;;
    *)
      echo "Unknown argument: $1" >&2
      exit 1
      ;;
  esac
done

ASSETS_DIR="${ANDROID_DIR}/app/src/main/assets/models"
mkdir -p "${ASSETS_DIR}"

if [[ ! -d "${COMPILED_DIR}" ]]; then
  echo "Compiled model directory not found: ${COMPILED_DIR}" >&2
  exit 1
fi

echo "Copying compiled models from ${COMPILED_DIR} to ${ASSETS_DIR}"
rsync -av --delete "${COMPILED_DIR}/" "${ASSETS_DIR}/"

echo "Model assets synced successfully."
