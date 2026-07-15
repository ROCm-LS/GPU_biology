# Quick start — Setonix / Pawsey (TL;DR)

Copy-paste outline to **reproduce** long-sequence split–fold–stitch on Setonix.
Singularity module used for verified runs: **`singularity/3.11.4-nompi`**.

Full docs: [README.md](README.md), [scripts/README.md](scripts/README.md), [scripts/VALIDATION.md](scripts/VALIDATION.md) (experiments → reports).

---

## 0. Overview

| Item | Notes |
|------|--------|
| Problem | Sequences ~3000+ aa OOM on a single GPU fold |
| Solution | Split → fold chunks → stitch with PyMOL (pLDDT / RMSD anchors) |
| Host scripts | `scripts/split_and_fold_segments_colabfold.py`, `scripts/split_and_fold_segments_alphafold2.py` |
| Setonix pattern | Host Python + **fold `.sif`** + **PyMOL `.sif`** (two containers) |
| Test input | `examples/inputs/3013aa.fasta` (2 segments: 1–3000, 2001–3013) |

---

## 1. Scratch layout

`MYSCRATCH` is set on Setonix (e.g. `/scratch/pawsey0007/$USER`).

```text
${MYSCRATCH}/
  GPU_biology/                         # clone this repo
  containers/                          # .sif files
  cache/                               # ColabFold params / cache
  *.tar                                # docker-archive exports (7.2.3 only)
```

Site-wide AlphaFold DB (read-only):

```text
/scratch/references/alphafold_feb2024/databases
```

Every run:

```bash
module load singularity/3.11.4-nompi
cd ${MYSCRATCH}/GPU_biology
```

**Interactive SLURM allocation**

```bash
srun -A ${PAWSEY_PROJECT}-gpu --partition gpu-highmem --pty -N 1 -n 1  --nodes=1 --time=10:00:00 --gres=gpu:2 /bin/bash
```

---

## 2. Container images (`.sif`)

Two sources:

| ROCm | Source | On Setonix |
|------|--------|------------|
| **6.2.4** | Pawsey [quay.io/pawsey](https://quay.io/pawsey) | `singularity pull` or copy existing `.sif` → `${MYSCRATCH}/containers/` |
| **7.2.3** | Docker built in AMD dev env | `docker save` → `.tar` → copy to `${MYSCRATCH}` → `singularity build` |

Target files (all under `${MYSCRATCH}/containers/`):

| File | How |
|------|-----|
| `colabfold_rocm6.2.4.sif` | §2.1 — `singularity pull` from quay.io/pawsey |
| `alphafold2_rocm6.2.4.sif` | §2.1 — `singularity pull` from quay.io/pawsey |
| `colabfold_rocm7.2.3_nopymol.sif` | §2.2 — `singularity build` from docker-archive `.tar` |
| `alphafold2_rocm7.2.3_nopymol.sif` | §2.2 — `singularity build` from docker-archive `.tar` |
| `pymol.sif` | §2.1 — `singularity pull` |

### 2.1 ROCm 6.2.4 — Pawsey images

```bash
module load singularity/3.11.4-nompi
mkdir -p ${MYSCRATCH}/containers
cd ${MYSCRATCH}/containers

singularity pull --name colabfold_rocm6.2.4.sif \
  docker://quay.io/pawsey/colabfold:rocm6.2.4

singularity pull --name alphafold2_rocm6.2.4.sif \
  docker://quay.io/pawsey/alphafold2-amd-gpu:v2.3.2_rocm6.2.4

singularity pull --name pymol.sif \
  docker://jysgro/pymol:deb12-2.5.0_sc
```

### 2.2 ROCm 7.2.3 — AMD Docker → Setonix `.sif`

**On AMD build machine** — export Docker image to tar:

```bash
# [FILL IN] docker save commands used before copying to Setonix, e.g.:
# docker save -o alphafold2_rocm7.2.3-nopymol.tar <image>:<tag>
# docker save -o colabfold_rocm7.2.3-nopymol.tar <image>:<tag>
# docker save -o pymol.tar <image>:<tag>
```

Copy `.tar` files to `${MYSCRATCH}/` (scp, rsync, etc.) — **`[FILL IN]`** transfer command if needed.

**On Setonix** — convert to `.sif`:

```bash
module load singularity/3.11.4-nompi
mkdir -p ${MYSCRATCH}/containers

# AlphaFold2 (no PyMOL) — verified
singularity build \
  ${MYSCRATCH}/containers/alphafold2_rocm7.2.3_nopymol.sif \
  docker-archive:${MYSCRATCH}/alphafold2_rocm7.2.3-nopymol.tar

# ColabFold (no PyMOL)
singularity build \
  ${MYSCRATCH}/containers/colabfold_rocm7.2.3_nopymol.sif \
  docker-archive:${MYSCRATCH}/colabfold_rocm7.2.3-nopymol.tar

# PyMOL — [FILL IN]
# singularity build \
#   ${MYSCRATCH}/containers/pymol.sif \
#   docker-archive:${MYSCRATCH}/pymol.tar
```

**`[FILL IN]`** — optional: build with PyMOL (`alphafold_rocm7.2.3.sif`, full colabfold image, etc.)

---

## 3. One-time setup

### 3.1 Clone repo

```bash
cd ${MYSCRATCH}
# [FILL IN] git clone / copy command for GPU_biology
```

### 3.2 ColabFold cache (params under `${MYSCRATCH}/cache`)

```bash
# [FILL IN] one-time colabfold.download inside colabfold .sif, e.g.:
# singularity exec --bind ${MYSCRATCH}/cache:/cache \
#   ${MYSCRATCH}/containers/colabfold_rocm6.2.4.sif \
#   python3 -m colabfold.download
```

---

## 4. Run commands — 3013 aa

Run on a **GPU node** after §1 module load. Logs via `tee` are optional but recommended.

### 4.1 ColabFold — ROCm 6.2.4

```bash
python3.11 scripts/split_and_fold_segments_colabfold.py \
  examples/inputs/3013aa.fasta \
  --runtime singularity \
  --colabfold-sif ../containers/colabfold_rocm6.2.4.sif \
  --pymol-sif ../containers/pymol.sif \
  --colabfold-cache ${MYSCRATCH}/cache \
  --work-dir colabfold_work_rocm6.2.4 \
  -- --disable-unified-memory \
  2>&1 | tee colabfold_work_rocm6.2.4/3013aa.colabfold.rocm6.2.4.log
```

### 4.2 ColabFold — ROCm 7.2.3

```bash
python3.11 scripts/split_and_fold_segments_colabfold.py \
  examples/inputs/3013aa.fasta \
  --runtime singularity \
  --colabfold-sif ${MYSCRATCH}/containers/colabfold_rocm7.2.3_nopymol.sif \
  --pymol-sif ${MYSCRATCH}/containers/pymol.sif \
  --colabfold-cache ${MYSCRATCH}/cache \
  --work-dir colabfold_work_rocm7.2.3 \
  -- --disable-unified-memory \
  2>&1 | tee colabfold_work_rocm7.2.3/3013aa.colabfold.rocm7.2.3.log
```

### 4.3 AlphaFold2 — ROCm 6.2.4

CPU MSA/search (`jackhmmer`, `hhsearch`) first; GPU during JAX inference.

```bash
python3.11 scripts/split_and_fold_segments_alphafold2.py \
  examples/inputs/3013aa.fasta \
  --runtime singularity \
  --alphafold2-sif ${MYSCRATCH}/containers/alphafold2_rocm6.2.4.sif \
  --pymol-sif ${MYSCRATCH}/containers/pymol.sif \
  --data-dir /scratch/references/alphafold_feb2024/databases \
  --work-dir alphafold2_work_rocm6.2.4 \
  --no-use-precomputed-msas \
  2>&1 | tee alphafold2_work_rocm6.2.4/3013aa.alphafold2.rocm6.2.4.log
```

**`[FILL IN]`** — extra `run_alphafold.py` flags after `--` if needed:

```bash
#  ... -- --model_preset=monomer --db_preset=full_dbs --max_template_date=...
```

### 4.4 AlphaFold2 — ROCm 7.2.3

```bash
python3.11 scripts/split_and_fold_segments_alphafold2.py \
  examples/inputs/3013aa.fasta \
  --runtime singularity \
  --alphafold2-sif ${MYSCRATCH}/containers/alphafold2_rocm7.2.3_nopymol.sif \
  --pymol-sif ${MYSCRATCH}/containers/pymol.sif \
  --data-dir /scratch/references/alphafold_feb2024/databases \
  --work-dir alphafold2_work_rocm7.2.3 \
  --no-use-precomputed-msas \
  2>&1 | tee alphafold2_work_rocm7.2.3/3013aa.alphafold2.rocm7.2.3.log
```

### 4.5 Other sequences / variants

Run additional targets in the **same** `--work-dir` to build a multi-target tree for validation
(see [scripts/VALIDATION.md](scripts/VALIDATION.md)):

```bash
# 5005 aa (same work-dir as 3013aa)
python3.11 scripts/split_and_fold_segments_alphafold2.py \
  examples/inputs/5005aa.fasta \
  --runtime singularity \
  --alphafold2-sif ${MYSCRATCH}/containers/alphafold2_rocm7.2.3_nopymol.sif \
  --pymol-sif ${MYSCRATCH}/containers/pymol.sif \
  --data-dir /scratch/references/alphafold_feb2024/databases \
  --work-dir alphafold2_work_rocm7.2.3 \
  --no-use-precomputed-msas \
  2>&1 | tee alphafold2_work_rocm7.2.3/5005aa.alphafold2.rocm7.2.3.log

# 1IH7.7 (A3M input)
python3.11 scripts/split_and_fold_segments_alphafold2.py \
  examples/inputs/1IH7.7.a3m \
  --runtime singularity \
  --alphafold2-sif ${MYSCRATCH}/containers/alphafold2_rocm7.2.3_nopymol.sif \
  --pymol-sif ${MYSCRATCH}/containers/pymol.sif \
  --data-dir /scratch/references/alphafold_feb2024/databases \
  --work-dir alphafold2_work_rocm7.2.3 \
  2>&1 | tee alphafold2_work_rocm7.2.3/1IH7.7.alphafold2.rocm7.2.3.log
```

Re-stitch only (fold already done): add `--skip-alphafold` (or `--skip-colabfold`).

Balanced tiling for 3013 aa: use a **separate** `--work-dir` and `--plan-mode balanced`.

---

## 5. Expected outputs

Under each `--work-dir`:

| Artifact | ColabFold | AlphaFold2 |
|----------|-----------|------------|
| Chunk FASTAs | `3013aa_part_0_1-3000.fa`, `3013aa_part_1_2001-3013.fa` | same |
| Per-chunk PDB | `3013aa_part_*_output/*_rank_001*.pdb` | `af2_predictions/3013aa_part_*/ranked_0.pdb` |
| Stitched | `3013aa_stitched_plddt.pdb`, `3013aa_stitched_rmsd.pdb` | same |
| Plans | `.split_fold_stitch/3013aa_stitch_plddt.json`, etc. (one file per target) | same |

---

## 6. Validation reports (local)

After runs finish, transfer stitched PDBs and part files from `--work-dir` (not large
`.pkl` files), then generate a markdown report on a machine with PyMOL installed.

**Full guide:** [scripts/VALIDATION.md](scripts/VALIDATION.md)

```bash
# On local machine (example paths)
python3 scripts/generate_split_stitch_validation_report.py \
  --no-colabfold-ref \
  --af2-output-root /home/sajandhy/af2_output \
  --candidate-dir alphafold2:/path/to/alphafold2_work_rocm7.2.3 \
  --output reports/alphafold2_split_stitch_validation_report.md
```

---

## 7. Quick checks

```bash
# [FILL IN] GPU visible on compute node
ls -l /dev/kfd
rocm-smi

# Singularity + images
which singularity
ls -lh ${MYSCRATCH}/containers/*.sif

# After run
ls -lh ${WORK_DIR}/3013aa_stitched_*.pdb
```

---

## 8. Troubleshooting

| Symptom | Fix |
|---------|-----|
| Singularity not found | `module load singularity/3.11.4-nompi` |
| `/dev/kfd` missing | Request GPUs in Slurm |
| Requires fold `.sif` | Pass `--colabfold-sif` or `--alphafold2-sif` (absolute path) |
| Stitch fails | `--pymol-sif` required; fold images have no PyMOL on Setonix |
| GPU idle during jackhmmer | Normal for AlphaFold2 until MSA/template stage finishes |

**`[FILL IN]`** — any Pawsey-specific issues you hit (OOM on chunk 0, PyMOL bind paths, etc.)
