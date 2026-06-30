#!/bin/bash
# Long-running ColabFold container: GPU, /cache, /work.
#
# ROCm stack version (image tag suffix): set COLABFOLD_ROCM_VERSION (default 6.2.4).
# Default image: colabfold-amd-gpu:rocm${COLABFOLD_ROCM_VERSION}
# Override full image: COLABFOLD_IMAGE=quay.io/pawsey/colabfold:rocm6.2.4
# Published Pawsey images have no PyMOL; as root in Docker: python -m pip install pymol-open-source-whl
# (Singularity/Setonix: non-root — use two-container scripts; see scripts/README.md PyMOL section).
# Override name: COLABFOLD_CONTAINER_NAME
# Paths: COLABFOLD_CACHE_DIR, COLABFOLD_WORK_DIR, MYSCRATCH
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=docker_rocm_common.sh
source "${SCRIPT_DIR}/docker_rocm_common.sh"

COLABFOLD_ROCM_VERSION="${COLABFOLD_ROCM_VERSION:-6.2.4}"
CONTAINER_NAME="${COLABFOLD_CONTAINER_NAME:-${USER}_colabfold_rocm_${COLABFOLD_ROCM_VERSION}}"
IMAGE="${COLABFOLD_IMAGE:-colabfold-amd-gpu:rocm${COLABFOLD_ROCM_VERSION}}"
CACHE_DIR="${COLABFOLD_CACHE_DIR:-${MYSCRATCH:-$HOME}/colabfold_cache}"
WORK_DIR="${COLABFOLD_WORK_DIR:-${HOME}/colabfold_work}"

setup_docker_rocm_dev_args
mkdir -p "${CACHE_DIR}" "${WORK_DIR}"

HIP_DEVICES="$(_discover_hip_visible_devices)"
echo "Using HIP_VISIBLE_DEVICES=${HIP_DEVICES} (compute GPUs from rocm-smi)"
echo "ROCm version (image tag): ${COLABFOLD_ROCM_VERSION}  image: ${IMAGE}  container: ${CONTAINER_NAME}"
echo "Cache: ${CACHE_DIR} -> /cache"
echo "Work:  ${WORK_DIR} -> /work"

# Match ColabFold Dockerfiles: JAX_PLATFORMS=rocm for 6.x, rocm,cpu for 7.x.
if [[ "${COLABFOLD_ROCM_VERSION}" == 7.* ]]; then
  JAX_PLATFORMS_VALUE="rocm,cpu"
else
  JAX_PLATFORMS_VALUE="rocm"
fi

docker run -d \
  --name "${CONTAINER_NAME}" \
  "${DOCKER_DEV_ARGS[@]}" \
  --group-add video \
  --shm-size=64g \
  -e XDG_CACHE_HOME=/cache \
  -e MPLCONFIGDIR=/cache \
  -e CACHE_DIR=/cache \
  -e "JAX_PLATFORMS=${JAX_PLATFORMS_VALUE}" \
  -e "HIP_VISIBLE_DEVICES=${HIP_DEVICES}" \
  -v "${CACHE_DIR}:/cache" \
  -v "${WORK_DIR}:/work" \
  "${IMAGE}" \
  tail -f /dev/null
