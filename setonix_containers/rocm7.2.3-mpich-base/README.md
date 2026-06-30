# ROCm 7.2.3 MPICH base

Builds **`rocm-mpich-base:rocm7.2.3-mpich3.4.3-ubuntu24.04`** for **`alphafold2/rocm7.2.3/`** and **`colabfold/rocm7.2.3/`**.

## Pull or build

```bash
LOCAL=rocm-mpich-base:rocm7.2.3-mpich3.4.3-ubuntu24.04
QUAY=quay.io/pawsey/rocm-mpich-base:rocm7.2.3-mpich3.4.3-ubuntu24.04

# When published on Quay:
docker pull "$QUAY" && docker tag "$QUAY" "$LOCAL"

# Or build from GPU_biology repo root:
bash setonix_containers/build_rocm7.2.3_mpich_base_docker_image.sh
```

Manual build (same as the script):

```bash
docker build \
  --build-arg ROCM_VERSION=7.2.3 \
  --build-arg OS_VERSION=24.04 \
  -t rocm-mpich-base:rocm7.2.3-mpich3.4.3-ubuntu24.04 \
  -f setonix_containers/rocm7.2.3-mpich-base/buildrocm-mpich-base.dockerfile \
  setonix_containers
```

- **Context:** **`setonix_containers`** (needs **`lustrempich-base/`** vendored there).
- **Ignore** **`setonix_containers/mpi/`** — accidental duplicate of the parent tree.

App images: **`alphafold2/rocm7.2.3/README.md`**, **`colabfold/rocm7.2.3/README.md`**.

## Build arguments

| Argument | Default | Notes |
|----------|---------|--------|
| `OS_VERSION` | `24.04` | |
| `ROCM_VERSION` | `6.0.2` | Pass **`7.2.3`** |
| `LINUX_KERNEL` | `6.8.0-31` | |
| `MPICH_VERSION` | `3.4.3` | |

Provides: libfabric, Lustre client, Lustre-aware MPICH, ROCm, RCCL, OSU micro-benchmarks.
