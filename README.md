# GPU_biology

ROCm-oriented **Dockerfiles** and small **host scripts** for structural biology tools on AMD GPUs (e.g. MI250X). This repo is meant to be shared as a **minimal, self-contained baseline**: build or pull images, bind-mount a working directory, run.

## Layout

| Path | Purpose |
|------|---------|
| `alphafold2/rocm7.2.3/` | AlphaFold2 on ROCm 7.2.3 — see **[alphafold2/rocm7.2.3/README.md](alphafold2/rocm7.2.3/README.md)** for `docker build` (optional **`INSTALL_PYMOL=1`** for single-container stitch). |
| `alphafold2/scripts/` | AlphaFold2 helpers: **minimal `reduced_dbs` tree**, **ColabFold A3M → precomputed MSAs**, example **`run_af2.sh`** — see **[alphafold2/scripts/README.md](alphafold2/scripts/README.md)** |
| `colabfold/rocm7.2.3/` | ColabFold on ROCm 7.2.3 — optional **PyMOL** (`INSTALL_PYMOL`, default on) — see **[colabfold/rocm7.2.3/README.md](colabfold/rocm7.2.3/README.md)**. |
| `colabfold/rocm6.2.4/` | ColabFold only (matches published Pawsey image; no PyMOL — see [PyMOL](#pymol-for-long-sequence-stitch)) |
| `scripts/` | Split / fold / stitch entrypoints — **host orchestrators** and **in-container** variants (see below) |
| `examples/` | Optional sample inputs / HPC snippets (not required for the core flow) |

## Scripts (`scripts/`)

Each of **ColabFold** and **AlphaFold2** has two pipeline scripts:

| Variant | Scripts | Where it runs |
|---------|---------|---------------|
| **Host orchestrator** | `split_and_fold_segments_colabfold.py`, `split_and_fold_segments_alphafold2.py` | **Host** — launches fold (ColabFold or AlphaFold2) in one container and PyMOL in another via `split_fold_stitch/` |
| **In-container** | `split_and_fold_segments_colabfold_single_container.py`, `split_and_fold_segments_alphafold2_single_container.py` | **Inside** the fold container — tiling, fold, and PyMOL stitch in one Python process |

**Long-sequence tiling (optional):**

- **ColabFold** — `split_and_fold_segments_colabfold.py` (**host**): FASTA or A2M/A3M → `colabfold_batch` (GPU container) → PyMOL stitch (separate container). Uses `split_fold_stitch/` helpers.
- **AlphaFold2** — `split_and_fold_segments_alphafold2.py` (**host**): **FASTA only** → `run_alphafold.py` (AlphaFold2 container) → PyMOL stitch. You pass all `run_alphafold.py` flags after `--`. The **database layout** can be either a full install or a **minimal stub tree** (see [AlphaFold2 database setups](#alphafold2-database-setups-same-scripts-both-sites) below); the host scripts are the same in both cases.

**In-container** (tiling + fold + PyMOL stitch in one process; run **inside** the fold image):

- **`split_and_fold_segments_colabfold_single_container.py`** — **ColabFold + PyMOL** in one image; build **`colabfold/rocm7.2.3/`** with default **`INSTALL_PYMOL=1`** (see **`colabfold/rocm7.2.3/README.md`**). **`INSTALL_PYMOL=0`** or **`colabfold/rocm6.2.4/`** → use the **two-container** flow below or extend the image.
- **`split_and_fold_segments_alphafold2_single_container.py`** — AlphaFold2 + PyMOL in one process. Use an image built with **`INSTALL_PYMOL=1`** (see **`alphafold2/rocm7.2.3/README.md`**); the default Dockerfile build has no PyMOL (use **`split_and_fold_segments_alphafold2.py`** + a PyMOL container, or extend the image).

For **host-orchestrated** runs, use `split_and_fold_segments_colabfold.py` and `split_and_fold_segments_alphafold2.py` on the **host**. For **in-container** runs, use the `*_single_container.py` scripts inside the fold image. Details: `scripts/README.md`.

### PyMOL for long-sequence stitch

Published **Pawsey ROCm 6.2.4** ColabFold / AlphaFold2 images ([quay.io/pawsey](https://quay.io/pawsey)) do **not** include PyMOL. That is expected for folding only.

- **Singularity / Apptainer (typical on Setonix):** use the **two-container** host scripts — fold `.sif` + a separate PyMOL `.sif` — or rebuild the fold image with PyMOL baked in before `singularity build`. Runtime `pip install` inside the fold `.sif` fails because the process is **not root**.
- **Docker with root in the container:** as **root**, run `python -m pip install pymol-open-source-whl` (or the full `apt` + pip stack in **`scripts/README.md`**) via **`docker exec`**, then run the **single-container** scripts inside that container.
- **ROCm 7.2.3 local builds:** optional PyMOL at image build time — **`colabfold/rocm7.2.3/README.md`** (default on) and **`alphafold2/rocm7.2.3/README.md`** (`INSTALL_PYMOL=1`).

**Containers (long-running `docker run … tail -f`):**

- `colabfold_docker_run.sh` — ColabFold image, `/work` + cache mounts, GPU discovery.
- `alphafold2_docker_run.sh` — AlphaFold2 image, same pattern.

**Shared:** `docker_rocm_common.sh`, `rocm_compute_devices.py` (HIP / device checks).

See `scripts/README.md` for paths, env overrides, and one-line examples.

## Database setups (ColabFold and AlphaFold2)

Use the **same** host scripts and **`/work`** bind-mount pattern on your cluster and at a customer. **ColabFold** keeps its DBs under **`/cache`**; **AlphaFold2** uses **`--data_dir`** (often `/work/databases`). Both support **reduced** layouts — ColabFold’s default MSA stack is smaller than AlphaFold **`full_dbs`**; AlphaFold **`reduced_dbs` + precomputed MSAs** can be smaller still on disk. Details: **`colabfold/rocm7.2.3/README.md`** (cache) and **`alphafold2/scripts/README.md`** (MSA handoff).

### ColabFold — cache under `/cache`

- Start a container with **`scripts/colabfold_docker_run.sh`** (`COLABFOLD_CACHE_DIR` → `/cache`, `XDG_CACHE_HOME=/cache`).
- **Model params:** `docker exec <container> python3 -m colabfold.download` (see **`colabfold/rocm7.2.3/README.md`**).
- **FASTA input:** default MSA search uses the public ColabFold/MMseqs API unless you configure local search; **`.a3m` / `.a2m` input** skips MSA search (params still needed to fold in ColabFold).

### AlphaFold2 — `data_dir` under `/work`

### A) Full databases (typical customer / production)

- Host a **complete** AlphaFold database tree (genetic search DBs, templates, mmcif, etc.) as required by your AlphaFold version ([upstream docs](https://github.com/google-deepmind/alphafold)).
- Mount it under **`/work/databases`** (or any path you pass to `--data_dir`).
- Typical flags include `--db_preset=full_dbs`, `--model_preset=monomer` (or multimer), plus all database path arguments your image expects.

### B) Minimal tree / reduced preset (internal cluster, avoid multi‑TB copies)

For environments where you **do not** mirror the full archive but still want **identical host scripts** as in (A):

- **Bootstrap the tree (dummy genetic DBs + mmcif shell, real PDB70 slot):** from the repo run  
  **`alphafold2/scripts/create_dummy_reduced_databases.sh [OUTPUT_DIR]`**  
  (default: `./minimal_af2_databases`). It creates tiny placeholder FASTAs for **UniRef90**, **MGnify**, and **small_bfd**, plus **`pdb_mmcif/`** (`obsolete.dat` and empty `mmcif_files/`). It creates **`pdb70/pdb70/`** as an **empty directory** with a **`README_PDB70.txt`** — you must fill that inner directory with the **real PDB70** download from the AlphaFold image (see below).
- Use **`--db_preset=reduced_dbs`** (or the equivalent your `run_alphafold.py` supports) so AlphaFold expects a smaller set of inputs.
- Mount the resulting directory as **`/work/databases`** (or pass its host path as `--data_dir`). The generated **`README_GPU_biology_minimal_dbs.txt`** inside the tree lists example **`run_alphafold.py`** path flags.
- **PDB70 (templates, real only):** download **pdb70** into **`…/pdb70/pdb70/`** using the **data download utilities shipped with AlphaFold inside the image**. The clone in `alphafold2/rocm7.2.3` is at **`/app/alphafold`**; helpers live under **`/app/alphafold/scripts`** (exact script names depend on the AlphaFold tag—compare with [AlphaFold `scripts/`](https://github.com/google-deepmind/alphafold/tree/main/scripts)). Inspect what your image ships, then run the downloader that populates **pdb70** into your mounted tree, for example:

  ```bash
  docker exec -w /app/alphafold <alphafold2_container_name> bash -lc 'ls scripts'
  # follow AlphaFold’s README for your version to fetch pdb70 (and any other small deps)
  ```

  Point **`--pdb70_database_path`** (and related flags) at the inner **`…/pdb70/pdb70`** directory once it contains the real database files.

- **Optional — skip large genetic DBs:** if you provide **precomputed MSAs** (`--use_precomputed_msas=true` and paths AlphaFold expects), you can avoid hosting UniRef/BFD-sized data while still using the same tiling/stitch scripts. That is independent of this repo; flags are standard AlphaFold.

### C) ColabFold MSAs → AlphaFold2 fold (smallest combined footprint)

Use ColabFold for **MSA generation** (reduced **`/cache`** as above), then AlphaFold2 for **prediction** without AF2 genetic DBs:

1. Minimal **`--data_dir`** as in (B) (`create_dummy_reduced_databases.sh` + real **pdb70** only — no large UniRef/MGnify/BFD).
2. Run **`colabfold_batch`** on FASTA (or pass **`.a3m`** to skip ColabFold search); take the output **`.a3m`** (under `…_output/`).
3. Convert with **`alphafold2/scripts/convert_colabfold_a3m_to_sto.py`** into `{output_dir}/{fasta_stem}/msas/`.
4. Run **`run_alphafold.py`** with **`--db_preset=reduced_dbs`** and **`--use_precomputed_msas=true`**, or **`alphafold2/scripts/run_af2.sh`** (`COLABFOLD_A3M=…`).

Full steps, flags, and tiling with **`.a3m` input**: **[alphafold2/scripts/README.md](alphafold2/scripts/README.md)**.

**Summary:** customer runs may use full DB mirrors on either tool; internal cluster runs use **ColabFold `/cache`** (optionally `SKIP_TEMPLATES`, `mmseqs2_uniref`) plus **AlphaFold `reduced_dbs`** with ColabFold-derived `.sto` files. Host tiling scripts are unchanged.

## Customer workflow (minimal)

1. Build images (see **`colabfold/rocm7.2.3/README.md`** and **`alphafold2/rocm7.2.3/README.md`** for sample `docker build` commands and build args) or use your registry tags.
2. Put **inputs, outputs, and (for AlphaFold2) the database directory** under one host tree so it can be mounted as **`/work`** in both AlphaFold2 and PyMOL containers (see [AlphaFold2 database setups](#alphafold2-database-setups-same-scripts-both-sites) for full vs minimal DB layout).
3. Start long-running containers with that mount (see `scripts/alphafold2_docker_run.sh` / `colabfold_docker_run.sh`).
4. Run the tiling + fold + stitch scripts; for AlphaFold2, pass the appropriate **`run_alphafold.py`** flags after `--` for your site (full or minimal DB case).

## VRAM monitoring

1. Submit the job to Slurm and note the node.
2. Log in to the node.
3. Run `bash vram_monitoring.sh <job_id>` (adjust the sleep interval inside the script if you want a different sampling rate).
