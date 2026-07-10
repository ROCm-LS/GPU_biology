#!/usr/bin/env bash
# Build ROCm 6.2.4 MPICH base for alphafold2/rocm6.2.4 and colabfold/rocm6.2.4.
# Run from anywhere. Build context: this directory (setonix_containers/).
# Tag matches quay.io/pawsey/rocm-mpich-base:rocm6.2.4-mpich3.4.3-ubuntu24.04 (without registry prefix).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

LOCAL_TAG="rocm-mpich-base:rocm6.2.4-mpich3.4.3-ubuntu24.04"
QUAY_TAG="quay.io/pawsey/rocm-mpich-base:rocm6.2.4-mpich3.4.3-ubuntu24.04"

docker build \
  --build-arg ROCM_VERSION=6.2.4 \
  --build-arg OS_VERSION=24.04 \
  -t "${LOCAL_TAG}" \
  -f rocm-mpich-base/buildrocm-mpich-base.dockerfile \
  .

echo "Built ${LOCAL_TAG}"
echo "App Dockerfiles use ${QUAY_TAG}; either pull that or: docker tag ${LOCAL_TAG} ${QUAY_TAG}"
