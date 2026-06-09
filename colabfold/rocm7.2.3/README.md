# ColabFold on ROCm 7.2.3 (Docker)

Image based on `quay.io/pawsey/rocm-mpich-base:rocm7.2.3-mpich3.4.3-ubuntu24.04` with ColabFold 1.5.5 and JAX ROCm 0.7.1.

**PyMOL** is **optional** (same **`INSTALL_PYMOL`** pattern as **`alphafold2/rocm7.2.3/`**). Default **`INSTALL_PYMOL=1`** installs **pymol-open-source==3.1.0a0** plus OpenGL-related libraries for **`scripts/split_and_fold_segments_colabfold_single_container.py`**. Set **`INSTALL_PYMOL=0`** for a smaller ColabFold-only image and use **`split_and_fold_segments_colabfold.py`** + a separate PyMOL container (or extend the image).

## Pawsey ROCm MPICH base (optional)

This image **`FROM quay.io/pawsey/rocm-mpich-base:rocm7.2.3-mpich3.4.3-ubuntu24.04`**. To rebuild that base locally, use the vendored recipe under **`setonix_containers/mpi/rocm-mpich-base/buildrocm-mpich-base.dockerfile`** (Setonix-oriented MPI + ROCm layout aligned with images published on [quay.io/pawsey](https://quay.io/pawsey)).

**Prerequisite:** `COPY` in that recipe expects **`setonix_containers/mpi/lustrempich-base/`** (e.g. `csel.patch`, `ch4r_init.patch`). That directory is **not** shipped in GPU_biology — add it from your Pawsey or site-provided materials before building.

From the **GPU_biology** repository root:

```bash
docker build --no-cache \
  --build-arg ROCM_VERSION=7.2.3 \
  --build-arg OS_VERSION=24.04 \
  -t quay.io/pawsey/rocm-mpich-base:rocm7.2.3-mpich3.4.3-ubuntu24.04 \
  -f setonix_containers/mpi/rocm-mpich-base/buildrocm-mpich-base.dockerfile \
  setonix_containers/mpi
```

- **`-f`** path is relative to the directory you run `docker build` from (here: repo root).
- **Final argument** is the **build context** and must be **`setonix_containers/mpi`**, not `…/rocm-mpich-base` alone, so `COPY lustrempich-base/...` and `COPY rocm-mpich-base/...` resolve.
- Omit **`--no-cache`** for incremental rebuilds when iterating.

See also **`setonix_containers/mpi/rocm-mpich-base/readme.md`**.

## Build

Run **`docker build` from the repository root** (or any directory) and pass this folder as the **build context** so `COPY` finds `environment.yml`, `requirements-pip.txt`, and `patch_colabfold_batch.py`.

```bash
# From GPU_biology repo root (default: includes PyMOL)
docker build -f colabfold/rocm7.2.3/Dockerfile -t colabfold-rocm:7.2.3 colabfold/rocm7.2.3/
```

ColabFold without PyMOL (two-container workflow):

```bash
docker build -f colabfold/rocm7.2.3/Dockerfile -t colabfold-rocm:7.2.3-nopymol \
  --build-arg INSTALL_PYMOL=0 \
  colabfold/rocm7.2.3/
```

Optional: enable BuildKit for better layer caching.

```bash
DOCKER_BUILDKIT=1 docker build -f colabfold/rocm7.2.3/Dockerfile -t colabfold-rocm:7.2.3 colabfold/rocm7.2.3/
```

## Build arguments

| Argument | Default | Meaning |
|----------|---------|---------|
| `INSTALL_PYMOL` | `1` | Set to `0` to skip **pymol-open-source** and graphics `apt` packages (smaller image; no single-container PyMOL stitch). |
| `DOWNLOAD_WEIGHTS` | `0` | Set to `1` to run `colabfold.download` at build time (large). If `0`, download once at runtime with `/cache` mounted (see Dockerfile comments). |
| `MINIFORGE_VERSION` | `25.3.0-3` | Miniforge installer pin. |
| `ROCM_JAX_TAG` | `rocm-jax-v0.7.1` | ROCm JAX release tag (wheel URLs). |
| `JAX_ROCM_VER` | `0.7.1` | JAX / jaxlib version pin. |

Example with weights baked into the image:

```bash
docker build -f colabfold/rocm7.2.3/Dockerfile -t colabfold-rocm:7.2.3-weights \
  --build-arg DOWNLOAD_WEIGHTS=1 \
  colabfold/rocm7.2.3/
```

## Run (host helper)

Long-running container with `/work` and cache mounts: **`scripts/colabfold_docker_run.sh`** (override `COLABFOLD_ROCM_VERSION`, `COLABFOLD_IMAGE`, `COLABFOLD_CONTAINER_NAME`, etc. as documented in that script).

## Related docs

- Root **`README.md`** — repo layout and tiling/stitch scripts.
- **`scripts/README.md`** — host script examples; **ROCm 7.2.3** note for `GPU_BIOLOGY_FORCE_ROCM_732_JAX` on single-container ColabFold.
