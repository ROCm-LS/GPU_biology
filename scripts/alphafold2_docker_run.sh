#!/bin/bash
# Long-running AlphaFold2 container: GPU, /cache, /work.
#
# ROCm stack version (embedded in default image tag): set ALPHAFOLD2_ROCM_VERSION (default 7.2.3).
# Default image: alphafold2-amd-gpu:v2.3.2_rocm${ALPHAFOLD2_ROCM_VERSION}
# Override full image: ALPHAFOLD2_IMAGE=...
# Override name: ALPHAFOLD2_CONTAINER_NAME
# Paths: ALPHAFOLD2_CACHE_DIR, ALPHAFOLD2_WORK_DIR, ALPHAFOLD2_DATABASE_DIR (-> /work/databases),
#        COLABFOLD_MSA_DIR (-> /colabfold_work), MYSCRATCH
# Default / Pawsey images often have no PyMOL; Docker root: python -m pip install pymol-open-source-whl
# (Singularity/Setonix: non-root — see scripts/README.md PyMOL section).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=docker_rocm_common.sh
source "${SCRIPT_DIR}/docker_rocm_common.sh"

ALPHAFOLD2_ROCM_VERSION="${ALPHAFOLD2_ROCM_VERSION:-7.2.3}"
CONTAINER_NAME="${ALPHAFOLD2_CONTAINER_NAME:-${USER}_alphafold2_rocm${ALPHAFOLD2_ROCM_VERSION}}"
IMAGE="${ALPHAFOLD2_IMAGE:-alphafold2-amd-gpu:v2.3.2_rocm${ALPHAFOLD2_ROCM_VERSION}}"
CACHE_DIR="${ALPHAFOLD2_CACHE_DIR:-${MYSCRATCH:-$HOME}/alphafold_cache}"
WORK_DIR="${ALPHAFOLD2_WORK_DIR:-${HOME}/alphafold_work}"
DATABASE_DIR="${ALPHAFOLD2_DATABASE_DIR:-${MYSCRATCH:-$HOME}/databases}"
MSA_DIR="${COLABFOLD_MSA_DIR:-${HOME}/colabfold_work}"

setup_docker_rocm_dev_args
mkdir -p "${CACHE_DIR}" "${WORK_DIR}" "${DATABASE_DIR}" "${MSA_DIR}"

HIP_DEVICES="$(_discover_hip_visible_devices)"
echo "Using HIP_VISIBLE_DEVICES=${HIP_DEVICES} (compute GPUs from rocm-smi)"
echo "ROCm version (image tag): ${ALPHAFOLD2_ROCM_VERSION}  image: ${IMAGE}  container: ${CONTAINER_NAME}"
echo "Cache:     ${CACHE_DIR} -> /cache"
echo "Work:      ${WORK_DIR} -> /work"
echo "Databases: ${DATABASE_DIR} -> /work/databases  (--data_dir; full or minimal tree)"
echo "MSA:       ${MSA_DIR} -> /colabfold_work  (ColabFold .a3m; set COLABFOLD_MSA_DIR to match colabfold container)"

docker run -d \
  --name "${CONTAINER_NAME}" \
  "${DOCKER_DEV_ARGS[@]}" \
  --group-add video \
  --shm-size=64g \
  -e XDG_CACHE_HOME=/cache \
  -e MPLCONFIGDIR=/cache \
  -e CACHE_DIR=/cache \
  -e JAX_PLATFORMS=rocm,cpu \
  -e "HIP_VISIBLE_DEVICES=${HIP_DEVICES}" \
  -v "${CACHE_DIR}:/cache" \
  -v "${WORK_DIR}:/work" \
  -v "${DATABASE_DIR}:/work/databases" \
  -v "${MSA_DIR}:/colabfold_work" \
  "${IMAGE}" \
  tail -f /dev/null
