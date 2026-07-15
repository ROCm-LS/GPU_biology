# Scripts (`scripts/`)

**TL;DR:** [QUICKSTART.md](../QUICKSTART.md) — Setonix / Singularity commands without reading this file.

Run from anywhere; use absolute paths or `cd` to your project tree first.

This directory has **two variants** of the long-sequence split / fold / stitch pipelines (same idea for **ColabFold** and **AlphaFold2**):

| | **Host orchestrator** | **In-container** |
|---|----------------------|------------------|
| **Script** | `split_and_fold_segments_colabfold.py`, `split_and_fold_segments_alphafold2.py` | `split_and_fold_segments_colabfold_single_container.py`, `split_and_fold_segments_alphafold2_single_container.py` |
| **Runs on** | **Host** (login node, workstation, or job driver outside the fold image) | **Inside** the ColabFold or AlphaFold2 container (`docker exec`, interactive shell, or a wrapper that execs into the image) |
| **Fold step** | Launches **`colabfold_batch`** or **`run_alphafold.py`** in the fold container via Docker / Singularity | Calls fold tools **in-process** in the same Python interpreter |
| **Stitch step** | Launches a **second** PyMOL container (or `.sif`) via **`split_fold_stitch/`** | **`import pymol`** in the same process — PyMOL must be in the fold image |
| **Typical use** | HPC (**Setonix**): published fold `.sif` without PyMOL + separate PyMOL `.sif` | Docker with root: install PyMOL into the fold container, or build with **`INSTALL_PYMOL=1`** (7.2.3) |

The host scripts use **`split_fold_stitch/container.py`** to wire fold and PyMOL containers. The single-container scripts duplicate tiling and stitch logic so they stand alone and do **not** import that orchestration layer.

**AlphaFold2 only:** the host script accepts **FASTA** or **A2M / A3M** (ColabFold MSAs; optional **`--colabfold-a3m`** with FASTA). A3M inputs are sliced per chunk and converted to **`msas/*.sto`** inside the AF2 container before folding. See **`alphafold2/scripts/README.md`**.

## Prerequisites

- Docker (or Singularity/Apptainer) with ROCm GPU devices passed through as in the `*_docker_run.sh` scripts. On Setonix: `module load singularity/3.11.4-nompi` (see [QUICKSTART.md](../QUICKSTART.md)).
- For **AlphaFold2**: **`/work`** for inputs/outputs; **`ALPHAFOLD2_DATABASE_DIR`** → **`/work/databases`** (`--data_dir`); **`COLABFOLD_MSA_DIR`** → **`/colabfold_work`**. See root **`README.md`** *Database setups* and **`alphafold2/scripts/README.md`**. For ColabFold **`/cache`**, see **`colabfold/rocm7.2.3/README.md`**.

## PyMOL and split / fold / stitch

Long-sequence tiling needs **PyMOL** for the stitch step. Inference-only runs (`colabfold_batch`, `run_alphafold.py`) do not.

**Published Pawsey ROCm 6.2.4 images** (e.g. **`quay.io/pawsey/colabfold:rocm6.2.4`** and matching **AlphaFold2** tags on [quay.io/pawsey](https://quay.io/pawsey)) ship **without** PyMOL. GPU_biology **`colabfold/rocm6.2.4/`** and **`alphafold2/rocm6.2.4/`** recipes match that layout. ROCm **7.2.3** Dockerfiles can include PyMOL at **build** time via **`INSTALL_PYMOL`** (ColabFold default on; AlphaFold2 default off) — see **`colabfold/rocm7.2.3/README.md`** and **`alphafold2/rocm7.2.3/README.md`**.

| Workflow | Host script | Where PyMOL lives |
|----------|-------------|-------------------|
| **Two-container** (fold + PyMOL) | `split_and_fold_segments_colabfold.py`, `split_and_fold_segments_alphafold2.py` | Second container or `.sif` (default Docker image `jysgro/pymol:deb12-2.5.0_sc`; override `--pymol-image` / `--pymol-sif`) |
| **Single-container** (fold + stitch in one image) | `split_and_fold_segments_*_single_container.py` | Same image as ColabFold / AlphaFold2 — PyMOL must already be installed |

**Docker — add PyMOL to a running fold container** (works for published 6.2.4 images and any fold image built without PyMOL). Long-running containers from **`colabfold_docker_run.sh`** / **`alphafold2_docker_run.sh`** run as **root**, so you can install into the image filesystem and then run a single-container script with `docker exec`.

**Why root:** PyMOL is installed into the container’s system Python (`pip` / `apt`). That requires **root** inside the image. On **Setonix**, **Singularity / Apptainer** runs the fold `.sif` as a **non-root** user, so `pip install` cannot modify the image (writes fail or do not land where `import pymol` expects). Use the **two-container** host scripts there, or bake PyMOL into the image before `singularity build`.

**Quick install** (as root in Docker):

```bash
CONTAINER=<colabfold_or_alphafold2_container_name>
docker exec -u root "${CONTAINER}" bash -lc \
  'python -m pip install --no-cache-dir pymol-open-source-whl && python -c "import pymol; print(\"PyMOL OK\")"'
```

**If `import pymol` fails** (missing OpenGL libraries), install the same graphics **`apt`** stack as **`INSTALL_PYMOL=1`** in **`colabfold/rocm7.2.3/Dockerfile`**, then retry pip:

```bash
docker exec -u root "${CONTAINER}" bash -lc '
  apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglu1-mesa libglew2.2 libpng16-16t64 libfreetype6 libxml2 libglib2.0-0t64 \
  && python -m pip install --no-cache-dir pymol-open-source-whl \
  && python -c "import pymol; print(\"PyMOL OK\")"
'
```

Alternatively, match the 7.2.3 Dockerfile pin exactly: `python -m pip install pymol-open-source==3.1.0a0` (after the same `apt` packages). Package names are for **Ubuntu 24.04** (Pawsey 6.2.4 base).

Then, for example:

```text
docker exec -w /work -e HIP_VISIBLE_DEVICES=0 "${CONTAINER}" \
  python3 /path/to/GPU_biology/scripts/split_and_fold_segments_colabfold_single_container.py \
  /work/query.fa --max-chunk-aa 400 -- --num-recycle 3
```

Installs are **lost** if you remove the container. To keep them, `docker commit` the container to a new tag or rebuild from a Dockerfile with the same `apt` / `pip` block (or **`INSTALL_PYMOL=1`** on 7.2.3).

**Singularity / Apptainer:** a converted `.sif` runs as a **non-root** user on HPC systems such as **Setonix** — you cannot install PyMOL into the image at runtime with `pip` or `apt`. Use the **two-container** host scripts with a separate PyMOL `.sif`, or convert a Docker image that already includes PyMOL (build with **`INSTALL_PYMOL=1`**, or install with **`docker exec -u root`** as above then export to `.sif`).

## Long-sequence pipelines

### Host orchestrators (run on the host)

These scripts stay on the **host**. They tile the sequence, call **AlphaFold2** or **ColabFold** in one container, then call **PyMOL** in another (via **`split_fold_stitch/`**).

#### ColabFold + PyMOL

```text
python3 /path/to/GPU_biology/scripts/split_and_fold_segments_colabfold.py QUERY.fasta \
  --work-dir /path/to/project \
  --colabfold-cache /path/to/colabfold_cache \
  -- --<any colabfold_batch flags>
```

### AlphaFold2 + PyMOL (FASTA or A3M; same script, full or minimal DB)

Flags after `--` must match your site: **`full_dbs`** + complete paths (customer), or **`reduced_dbs`** + a small `databases` tree + real **pdb70** (internal). Bootstrap a minimal tree with **`alphafold2/scripts/create_dummy_reduced_databases.sh`**, then download pdb70 into `…/pdb70/pdb70/`. See root **`README.md`**.

```text
python3 /path/to/GPU_biology/scripts/split_and_fold_segments_alphafold2.py QUERY.fa \
  --work-dir /path/to/project \
  --alphafold2-container-name <running_af2_container> \
  -- --data_dir=/work/databases --model_preset=monomer --db_preset=full_dbs \
  <…remaining flags required by your AlphaFold2 image…>
```

- **`--work-dir`**: common parent of input FASTA/A3M, chunk files, and `--af2-output-base` (default `<work-dir>/af2_predictions`).
- **`--colabfold-a3m`**: with FASTA input, column-slice a full-query ColabFold `.a3m` per chunk (same as A3M primary input).
- **`--`**: everything after is forwarded to **`run_alphafold.py`** unchanged.

Environment overrides for the docker helper scripts: see `alphafold2_docker_run.sh` / `colabfold_docker_run.sh` headers (`ALPHAFOLD2_ROCM_VERSION`, `COLABFOLD_ROCM_VERSION`, `ALPHAFOLD2_IMAGE`, `COLABFOLD_IMAGE`, `ALPHAFOLD2_SCRIPTS_DIR`, `ALPHAFOLD2_MOUNT_SCRIPTS`, `MYSCRATCH`, etc.).

### In-container (run inside the fold image)

Copy or bind-mount this repo (or at least the script) into the ColabFold / AlphaFold2 container, then invoke with **`python3`** there — not from the host orchestration path above.

These scripts bundle tiling, folding, and PyMOL merge logic in one file. They do **not** use `split_fold_stitch/` or a second PyMOL container. PyMOL must be in the fold image — see **[PyMOL and split / fold / stitch](#pymol-and-split--fold--stitch)** (build **`INSTALL_PYMOL`**, Docker runtime install, or use the two-container scripts instead).

### ColabFold + PyMOL (one image)

```text
python3 /path/to/GPU_biology/scripts/split_and_fold_segments_colabfold_single_container.py QUERY.fa \
  --max-chunk-aa 400 -- --num-recycle 3
```

On **ROCm 7.2.3**, ColabFold single-container and dual-container orchestration add JAX/XLA settings (including `--xla_gpu_enable_triton_gemm=false`) when the fold image path or environment looks like 7.2.3 (e.g. `rocm7.2.3` in the `.sif` name or `ROCM_PATH` contains `/rocm-7.2.3`). Set `GPU_BIOLOGY_FORCE_ROCM_732_JAX=1` or `0` to force that workaround on or off. The AlphaFold2 single-container script still uses only minimal `XLA_FLAGS` unless you set that override.

### AlphaFold2 + PyMOL (one image)

```text
python3 /gpu_biology/scripts/split_and_fold_segments_alphafold2_single_container.py QUERY.fa \
  --output-dir-base /work/run1 --data-dir /work/databases \
  -- --model_preset=monomer …
```

``alphafold2_docker_run.sh`` bind-mounts the full repo read-only at ``/gpu_biology``
(``alphafold2/scripts`` is also available at ``/work/af2_scripts``).

**A3M / A2M input** (e.g. MSAs from ColabFold): same script; it converts each chunk’s alignment via **`alphafold2/scripts/convert_colabfold_a3m_to_sto.py`** and runs with **`--use_precomputed_msas=true`** by default. See **`alphafold2/scripts/README.md`**.
