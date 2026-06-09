# AlphaFold2 on ROCm 7.2.3 (Docker)

Image based on `quay.io/pawsey/rocm-mpich-base:rocm7.2.3-mpich3.4.3-ubuntu24.04` with AlphaFold **v2.3.2** at `/app/alphafold`, OpenMM-HIP, and JAX ROCm **0.7.1** (see `Dockerfile` for XLA env aligned with ColabFold ROCm 7.2.3).

**PyMOL** is **optional**: the default image is suitable for **`run_alphafold.py`** and the two-container host script **`scripts/split_and_fold_segments_alphafold2.py`**. Set **`INSTALL_PYMOL=1`** for **`scripts/split_and_fold_segments_alphafold2_single_container.py`** — same wheel and OpenGL-related **`apt`** packages as **`colabfold/rocm7.2.3/Dockerfile`** when **`INSTALL_PYMOL=1`** (the ColabFold 7.2.3 default).

## Pawsey ROCm MPICH base (optional)

This image **`FROM quay.io/pawsey/rocm-mpich-base:rocm7.2.3-mpich3.4.3-ubuntu24.04`**. To rebuild that base locally, use **`setonix_containers/mpi/rocm-mpich-base/buildrocm-mpich-base.dockerfile`** (Setonix-oriented MPI + ROCm layout aligned with images on [quay.io/pawsey](https://quay.io/pawsey)).

**Prerequisite:** add **`lustrempich-base/`** under **`setonix_containers/mpi/`** (patch files are not bundled in GPU_biology — obtain them from your Pawsey or site distribution).

From the **GPU_biology** repository root:

```bash
docker build --no-cache \
  --build-arg ROCM_VERSION=7.2.3 \
  --build-arg OS_VERSION=24.04 \
  -t quay.io/pawsey/rocm-mpich-base:rocm7.2.3-mpich3.4.3-ubuntu24.04 \
  -f setonix_containers/mpi/rocm-mpich-base/buildrocm-mpich-base.dockerfile \
  setonix_containers/mpi
```

Use **`setonix_containers/mpi`** as the build context (final path). See **`setonix_containers/mpi/rocm-mpich-base/readme.md`**.

## Build

Run **`docker build` from the repository root** and pass this folder as the **build context** so `COPY` finds `config.py`, `amber_minimize.py`, and `requirements.txt`.

```bash
# From GPU_biology repo root — inference / two-container stitch (no PyMOL)
docker build -f alphafold2/rocm7.2.3/Dockerfile \
  -t alphafold2-amd-gpu:v2.3.2_rocm7.2.3 \
  alphafold2/rocm7.2.3/
```

Default tag matches **`scripts/alphafold2_docker_run.sh`** (`ALPHAFOLD2_ROCM_VERSION` default **7.2.3**, or set **`ALPHAFOLD2_IMAGE`** explicitly).

**Single-container** stitch (install PyMOL in the image):

```bash
docker build -f alphafold2/rocm7.2.3/Dockerfile \
  -t alphafold2-amd-gpu:v2.3.2_rocm7.2.3_pymol \
  --build-arg INSTALL_PYMOL=1 \
  alphafold2/rocm7.2.3/
```

Optional: `DOCKER_BUILDKIT=1` for layer caching.

## Build arguments

| Argument | Default | Meaning |
|----------|---------|---------|
| `INSTALL_PYMOL` | `0` | Set to `1` to install **pymol-open-source==3.1.0a0** and graphics libraries for in-image PyMOL (single-container tiling/stitch). |

## Run (host helper)

Long-running container with `/work` and cache mounts: **`scripts/alphafold2_docker_run.sh`**. Set **`ALPHAFOLD2_ROCM_VERSION`** or **`ALPHAFOLD2_IMAGE`** to the tag you built (with or without `_pymol`).

## Persistence / bind mounts

The **`Dockerfile`** footer comments describe bind-mounting a host clone over `/app/alphafold` if you maintain patches outside the image.

## Related docs

- Root **`README.md`** — AlphaFold2 database layouts (full vs minimal) and customer workflow.
- **`scripts/README.md`** — host tiling + fold + stitch examples.
