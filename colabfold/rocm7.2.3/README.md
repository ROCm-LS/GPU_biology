# ColabFold on ROCm 7.2.3 (Docker)

Image based on `rocm-mpich-base:rocm7.2.3-mpich3.4.3-ubuntu24.04` with ColabFold 1.5.5 and JAX ROCm 0.7.1.

**PyMOL** is **optional** (same **`INSTALL_PYMOL`** pattern as **`alphafold2/rocm7.2.3/`**). Default **`INSTALL_PYMOL=1`** installs **pymol-open-source==3.1.0a0** plus OpenGL-related libraries for **`scripts/split_and_fold_segments_colabfold_single_container.py`**. Set **`INSTALL_PYMOL=0`** for a smaller ColabFold-only image and use **`split_and_fold_segments_colabfold.py`** + a separate PyMOL container (or extend the image).

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
docker build --build-arg BASE_IMAGE="$QUAY_BASE" -f colabfold/rocm7.2.3/Dockerfile ...
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

- **`-f`** path is relative to the directory you run `docker build` from (here: repo root).
- **Build context** must be **`setonix_containers`** so `COPY lustrempich-base/...` resolves (`setonix_containers/lustrempich-base/` is vendored in this repo).
- Ignore **`setonix_containers/mpi/`** — an accidental duplicate of the parent tree (do not use as build context); canonical paths are under **`setonix_containers/`** directly.
- Omit **`--no-cache`** for incremental rebuilds.
- Recipe details: **`setonix_containers/rocm7.2.3-mpich-base/README.md`**.

## Build

Run **`docker build` from the repository root** (or any directory) and pass this folder as the **build context** so `COPY` finds `environment.yml`, `requirements-pip.txt`, and `patch_colabfold_batch.py`. Ensure **`$LOCAL_BASE`** exists (Option A or B above).

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
| `BASE_IMAGE` | `rocm-mpich-base:rocm7.2.3-mpich3.4.3-ubuntu24.04` | ROCm MPICH base image (`docker pull` + tag, or local build). Use `quay.io/pawsey/...` after publish without retagging. |
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

## Databases and cache

ColabFold does **not** use AlphaFold’s `--data_dir`. Model **params** (and optional local MMseqs2 DBs) live under the ColabFold data directory — by default **`/cache/colabfold`** when `XDG_CACHE_HOME=/cache` (as in **`scripts/colabfold_docker_run.sh`**). Override with **`colabfold_batch --data /path`**.

```bash
docker exec <colabfold_container> python3 -m colabfold.download
```

Without **`--data`**, `colabfold_batch` uses that default cache path for **weights** (`params/`). The **output folder** argument is results only.

For **FASTA** input, default MSA generation uses the **public ColabFold/MMseqs API** (`--host-url` default), not local files under `/cache`. Pass **`.a3m` / `.a2m`** to skip MSA search. Use **`colabfold_batch … --msa-only`** to write **`.a3m`** files without structure prediction (handy before AlphaFold2 fold or to split MSA and GPU jobs). For **MSA in ColabFold → fold in AlphaFold2** with a minimal AF2 tree, see **`alphafold2/scripts/README.md`**.

## Run (host helper)

Long-running container with `/work`, **`/colabfold_work`** (ColabFold MSA / batch output), and cache mounts: **`scripts/colabfold_docker_run.sh`**. Set **`COLABFOLD_MSA_DIR`** on the host (default **`${HOME}/colabfold_work`**); use the **same** value when starting **`scripts/alphafold2_docker_run.sh`** so AlphaFold2 can read `.a3m` files at `/colabfold_work`.

Example MSA-only run inside the ColabFold container:

```bash
colabfold_batch /work/query.fasta /colabfold_work/run1 --msa-only
# → /colabfold_work/run1/query_output/*.a3m (exact names depend on ColabFold)
```

Override `COLABFOLD_ROCM_VERSION`, `COLABFOLD_IMAGE`, `COLABFOLD_CONTAINER_NAME`, `COLABFOLD_WORK_DIR`, `COLABFOLD_MSA_DIR`, etc. as documented in that script.

## Related docs

- Root **`README.md`** — repo layout and tiling/stitch scripts.
- **`scripts/README.md`** — host script examples; **ROCm 7.2.3** note for `GPU_BIOLOGY_FORCE_ROCM_732_JAX` on single-container ColabFold.
- **`alphafold2/rocm7.2.3/README.md`** — same base pull/build pattern for AlphaFold2.
- **`alphafold2/scripts/README.md`** — ColabFold `.a3m` → AlphaFold2 precomputed MSAs.
