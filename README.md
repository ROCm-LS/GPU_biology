# GPU_biology

ROCm-oriented **Dockerfiles** and small **host scripts** for structural biology tools on AMD GPUs (e.g. MI250X). This repo is meant to be shared as a **minimal, self-contained baseline**: build or pull images, bind-mount a working directory, run.

## Layout

| Path | Purpose |
|------|---------|
| `alphafold2/rocm7.2.3/` | AlphaFold2 on ROCm 7.2.3 — see **[alphafold2/rocm7.2.3/README.md](alphafold2/rocm7.2.3/README.md)** for `docker build` (optional **`INSTALL_PYMOL=1`** for single-container stitch). |
| `alphafold2/scripts/` | Host helper: **minimal dummy DB layout** for `reduced_dbs` (see below) |
| `colabfold/rocm7.2.3/` | ColabFold on ROCm 7.2.3 — optional **PyMOL** (`INSTALL_PYMOL`, default on) — see **[colabfold/rocm7.2.3/README.md](colabfold/rocm7.2.3/README.md)**. |
| `colabfold/rocm6.2.4/` | ColabFold only (no PyMOL in image; see below) |
| `scripts/` | Host entrypoints (see below) |
| `examples/` | Optional sample inputs / HPC snippets (not required for the core flow) |

## Host scripts (`scripts/`)

**Long-sequence tiling (optional):**

- **ColabFold** — `split_and_fold_segments_colabfold.py`: FASTA or A2M/A3M → `colabfold_batch` (GPU container) → PyMOL stitch (separate container). Uses `split_fold_stitch/` helpers.
- **AlphaFold2** — `split_and_fold_segments_alphafold2.py`: **FASTA only** → `run_alphafold.py` (AlphaFold2 container) → PyMOL stitch. You pass all `run_alphafold.py` flags after `--`. The **database layout** can be either a full install or a **minimal stub tree** (see [AlphaFold2 database setups](#alphafold2-database-setups-same-scripts-both-sites) below); the host scripts are the same in both cases.

**Single-container (run inside one fold image; tiling + fold + PyMOL stitch in-process):**

- **`split_and_fold_segments_colabfold_single_container.py`** — **ColabFold + PyMOL** in one image; build **`colabfold/rocm7.2.3/`** with default **`INSTALL_PYMOL=1`** (see **`colabfold/rocm7.2.3/README.md`**). **`INSTALL_PYMOL=0`** or **`colabfold/rocm6.2.4/`** → use the **two-container** flow below or extend the image.
- **`split_and_fold_segments_alphafold2_single_container.py`** — AlphaFold2 + PyMOL in one process. Use an image built with **`INSTALL_PYMOL=1`** (see **`alphafold2/rocm7.2.3/README.md`**); the default Dockerfile build has no PyMOL (use **`split_and_fold_segments_alphafold2.py`** + a PyMOL container, or extend the image).

For **host-orchestrated** two-container runs (separate ColabFold or AlphaFold2 container + PyMOL container), use `split_and_fold_segments_colabfold.py` and `split_and_fold_segments_alphafold2.py` instead. See `scripts/README.md`.

**Containers (long-running `docker run … tail -f`):**

- `colabfold_docker_run.sh` — ColabFold image, `/work` + cache mounts, GPU discovery.
- `alphafold2_docker_run.sh` — AlphaFold2 image, same pattern.

**Shared:** `docker_rocm_common.sh`, `rocm_compute_devices.py` (HIP / device checks).

See `scripts/README.md` for paths, env overrides, and one-line examples.

## AlphaFold2 database setups (same scripts, both sites)

Use the **same** `scripts/split_and_fold_segments_alphafold2.py`, container images, and **`/work` bind-mount idea** on your cluster and at a customer. Only the **`run_alphafold.py` arguments after `--`** and the contents of **`--data_dir`** differ.

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

**Summary:** customer runs (A) with `full_dbs` and a real data mirror; your cluster runs (B) with `reduced_dbs`, a small `databases` tree, pdb70 from the image’s scripts, and placeholders or precomputed MSAs as needed. **`split_and_fold_segments_alphafold2.py` does not change** between the two.

## Customer workflow (minimal)

1. Build images (see **`colabfold/rocm7.2.3/README.md`** and **`alphafold2/rocm7.2.3/README.md`** for sample `docker build` commands and build args) or use your registry tags.
2. Put **inputs, outputs, and (for AlphaFold2) the database directory** under one host tree so it can be mounted as **`/work`** in both AlphaFold2 and PyMOL containers (see [AlphaFold2 database setups](#alphafold2-database-setups-same-scripts-both-sites) for full vs minimal DB layout).
3. Start long-running containers with that mount (see `scripts/alphafold2_docker_run.sh` / `colabfold_docker_run.sh`).
4. Run the tiling + fold + stitch scripts; for AlphaFold2, pass the appropriate **`run_alphafold.py`** flags after `--` for your site (full or minimal DB case).

## VRAM monitoring

1. Submit the job to Slurm and note the node.
2. Log in to the node.
3. Run `bash vram_monitoring.sh <job_id>` (adjust the sleep interval inside the script if you want a different sampling rate).
