#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUTPUT_DIR="${ROOT_DIR}/demos"
MPLCONFIGDIR="${ROOT_DIR}/.matplotlib"

mkdir -p "${OUTPUT_DIR}" "${MPLCONFIGDIR}"

for env_id in StackCube-v1 PushT-v1; do
  echo "Downloading ${env_id}..."
  MPLCONFIGDIR="${MPLCONFIGDIR}" python -m mani_skill.utils.download_demo "${env_id}" -o "${OUTPUT_DIR}"
done

echo "Done. Files:"
find "${OUTPUT_DIR}" \( -iname "*.h5" -o -iname "*.hdf5" -o -iname "*.json" \) -print
