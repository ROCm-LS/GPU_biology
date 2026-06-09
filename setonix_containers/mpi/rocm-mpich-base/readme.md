# ROCm + MPICH base (vendored recipe)

`buildrocm-mpich-base.dockerfile` is a **vendored** Setonix-style recipe (**libfabric + Lustre client + MPICH + ROCm + aws-ofi-rccl + OSU**), aligned with the Pawsey Supercomputing Centre images published on [quay.io/pawsey](https://quay.io/pawsey).

## Prerequisites

The Dockerfile **`COPY lustrempich-base/csel.patch`** … — you need **`lustrempich-base/`** next to **`rocm-mpich-base/`** under **`setonix_containers/mpi/`**. Obtain that directory from your Pawsey or site distribution if it is missing from GPU_biology.

## Build arguments (see Dockerfile `ARG`)

Examples: `OS_VERSION` (default **24.04** in this copy), `LINUX_KERNEL`, `ROCM_VERSION`, `MPI4PY_VERSION`, etc.

## Example: ROCm 7.2.3 (matches ColabFold / AlphaFold `FROM` in this repo)

Run from the **GPU_biology** repository root so `-f` and the build context resolve:

```bash
docker build --no-cache \
  --build-arg ROCM_VERSION=7.2.3 \
  --build-arg OS_VERSION=24.04 \
  -t quay.io/pawsey/rocm-mpich-base:rocm7.2.3-mpich3.4.3-ubuntu24.04 \
  -f setonix_containers/mpi/rocm-mpich-base/buildrocm-mpich-base.dockerfile \
  setonix_containers/mpi
```

- **Build context** must be **`setonix_containers/mpi`** (not `…/rocm-mpich-base` only), so `COPY lustrempich-base/...` and `COPY rocm-mpich-base/...` work.
- Drop **`--no-cache`** when you want layer cache.

Then build **`colabfold/rocm7.2.3/`** or **`alphafold2/rocm7.2.3/`** as documented in those folders’ `README.md` files.

## Older example (different tag)

```text
docker build --build-arg ROCM_VERSION=6.1 \
  -t mpich-luster-rocm-base:3.4.3_ubuntu22.04-rocm6.1 \
  -f setonix_containers/mpi/rocm-mpich-base/buildrocm-mpich-base.dockerfile \
  setonix_containers/mpi
```

This container provides: libfabric, Lustre client, Lustre-aware MPICH, ROCm, RCCL (aws-ofi-rccl), OSU micro-benchmarks.
