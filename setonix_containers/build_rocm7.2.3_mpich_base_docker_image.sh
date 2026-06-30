#!/usr/bin/env bash
# Build ROCm 7.2.3 MPICH base for alphafold2/rocm7.2.3 and colabfold/rocm7.2.3.
# Run from anywhere. Build context: this directory (setonix_containers/).
# See setonix_containers/rocm7.2.3-mpich-base/README.md and alphafold2/rocm7.2.3/README.md.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

LOCAL_TAG="rocm-mpich-base:rocm7.2.3-mpich3.4.3-ubuntu24.04"
QUAY_TAG="quay.io/pawsey/rocm-mpich-base:rocm7.2.3-mpich3.4.3-ubuntu24.04"

docker build \
  --build-arg ROCM_VERSION=7.2.3 \
  --build-arg OS_VERSION=24.04 \
  -t "${LOCAL_TAG}" \
  -f rocm7.2.3-mpich-base/buildrocm-mpich-base.dockerfile \
  .

echo "Built ${LOCAL_TAG}"
echo "When published on Quay: docker pull ${QUAY_TAG} && docker tag ${QUAY_TAG} ${LOCAL_TAG}"
