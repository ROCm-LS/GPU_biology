# AlphaFold2 on ROCm 7.2.3 (Docker)

Image based on `rocm-mpich-base:rocm7.2.3-mpich3.4.3-ubuntu24.04` with AlphaFold **v2.3.2** at `/app/alphafold`, OpenMM-HIP, and JAX ROCm **0.7.1** (see `Dockerfile` for XLA env aligned with ColabFold ROCm 7.2.3).

**PyMOL** is **optional**: the default image is suitable for **`run_alphafold.py`** and the two-container host script **`scripts/split_and_fold_segments_alphafold2.py`**. Set **`INSTALL_PYMOL=1`** for **`scripts/split_and_fold_segments_alphafold2_single_container.py`** — same wheel and OpenGL-related **`apt`** packages as **`colabfold/rocm7.2.3/Dockerfile`** when **`INSTALL_PYMOL=1`** (the ColabFold 7.2.3 default).

## ROCm MPICH base (pull or build)

AlphaFold and ColabFold ROCm **7.2.3** images share one base tag. The **`Dockerfile`** accepts **`BASE_IMAGE`** (default: local name below). **Application images in this repo are not published** — build them locally after the base is available.

```bash
# Names used below (same tag, with or without registry prefix)
LOCAL_BASE=rocm-mpich-base:rocm7.2.3-mpich3.4.3-ubuntu24.04
QUAY_BASE=quay.io/pawsey/rocm-mpich-base:rocm7.2.3-mpich3.4.3-ubuntu24.04
```

**Option A — pull** (when [quay.io/pawsey](https://quay.io/pawsey) publishes ROCm 7.2.3, e.g. after your PR merges):

```bash
docker pull "$QUAY_BASE"
docker tag "$QUAY_BASE" "$LOCAL_BASE"   # matches Dockerfile default BASE_IMAGE
```

Or skip the tag and pass the registry name when building AlphaFold/ColabFold:

```bash
docker build --build-arg BASE_IMAGE="$QUAY_BASE" -f alphafold2/rocm7.2.3/Dockerfile ...
```

**Option B — build locally** (until the image is on Quay, or to test a recipe change):

```bash
bash setonix_containers/build_rocm7.2.3_mpich_base_docker_image.sh
```

Or from the **GPU_biology** repository root:

```bash
docker build --no-cache \
  --build-arg ROCM_VERSION=7.2.3 \
  --build-arg OS_VERSION=24.04 \
  -t "$LOCAL_BASE" \
  -f setonix_containers/rocm7.2.3-mpich-base/buildrocm-mpich-base.dockerfile \
  setonix_containers
```

- **Build context** must be **`setonix_containers`** so `COPY lustrempich-base/...` resolves (`setonix_containers/lustrempich-base/` is vendored in this repo).
- Ignore **`setonix_containers/mpi/`** — an accidental duplicate of the parent tree (do not use as build context); canonical paths are under **`setonix_containers/`** directly.
- Omit **`--no-cache`** for incremental rebuilds.
- Recipe details: **`setonix_containers/rocm7.2.3-mpich-base/README.md`**.

## Build

Run **`docker build` from the repository root** and pass this folder as the **build context** so `COPY` finds `config.py`, `amber_minimize.py`, and `requirements.txt`. Ensure **`$LOCAL_BASE`** exists (Option A or B above).

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
| `BASE_IMAGE` | `rocm-mpich-base:rocm7.2.3-mpich3.4.3-ubuntu24.04` | ROCm MPICH base image (`docker pull` + tag, or local build). Use `quay.io/pawsey/...` after publish without retagging. |
| `INSTALL_PYMOL` | `0` | Set to `1` to install **pymol-open-source==3.1.0a0** and graphics libraries for in-image PyMOL (single-container tiling/stitch). |

## Run (host helper)

Long-running container: **`scripts/alphafold2_docker_run.sh`** (`/work`, **`/work/databases`**, `/colabfold_work`, `/cache`). Set **`ALPHAFOLD2_DATABASE_DIR`**, **`ALPHAFOLD2_ROCM_VERSION`**, or **`ALPHAFOLD2_IMAGE`** as needed.

## Persistence / bind mounts

The **`Dockerfile`** footer comments describe bind-mounting a host clone over `/app/alphafold` if you maintain patches outside the image.

## Related docs

- Root **`README.md`** — AlphaFold2 database layouts (full vs minimal) and customer workflow.
- **`alphafold2/scripts/README.md`** — optional minimal-DB workflow (scripts live on host / `/work`, not in the image): ColabFold `.a3m` → precomputed MSAs, `run_af2.sh`.
- **`scripts/README.md`** — host tiling + fold + stitch examples.
- **`colabfold/rocm7.2.3/README.md`** — same base pull/build pattern for ColabFold.
