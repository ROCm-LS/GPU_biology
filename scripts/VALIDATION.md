# Split-fold-stitch validation and reporting

How to **run split-fold-stitch experiments** (Setonix / MI250X), collect outputs under a
`--work-dir`, and **generate comparison reports** against full-length reference folds.

**Related docs**

- [QUICKSTART.md](../QUICKSTART.md) — Setonix container setup and copy-paste fold commands
- [scripts/README.md](README.md) — pipeline architecture, PyMOL, host vs single-container scripts

---

## 1. Workflow overview

```text
Setonix (MI250X)                         Local machine (or MI355X)
─────────────────                        ─────────────────────────
split_and_fold_segments_*.py             generate_split_stitch_validation_report.py
  --work-dir <run>/          rsync  →      --candidate-dir LABEL:<run>/
  INPUT.fasta                              --reference-root / --af2-output-root
       │                                              │
       ▼                                              ▼
  <run>/3013aa_stitched_plddt.pdb            reports/*.md
```

1. **Experiment** — fold segments and stitch on a memory-limited GPU; all artifacts live under
   one `--work-dir` per run configuration.
2. **Transfer** — copy stitched PDBs, part inputs, and plan JSON (not multi-GB `.pkl` files).
3. **References** — full-length monolithic folds on high-memory hardware (separate trees).
4. **Report** — PyMOL Cα superposition of stitched vs reference; markdown summary.

---

## 2. Validation targets

The report scripts know three targets (see `stitch_compare_core.TARGET_CASES`):

| Target | Input | Length | Default tiling |
|--------|-------|--------|----------------|
| `3013aa` | `3013aa.fasta` | 3013 | 2 segments: 1–3000, 2001–3013 |
| `5005aa` | `5005aa.fasta` | 5005 | 2 segments: 1–3000, 2001–5005 |
| `1IH7.7` | `1IH7.7.a3m` | 903 | 5 × 300 aa windows, 100 aa overlap |

Inputs live in `examples/inputs/`.

---

## 3. Running experiments (`--work-dir`)

Use the **host orchestrator** scripts on Setonix (fold `.sif` + PyMOL `.sif`). See
[QUICKSTART.md §4](../QUICKSTART.md#4-run-commands--3013-aa) for full Singularity examples.

### 3.1 One target, one work directory

Pick a **stable work directory name** per backend and ROCm version, e.g.:

- `colabfold_work_rocm7.2.3`
- `alphafold2_work_rocm7.2.3`

**ColabFold**

```bash
python3 scripts/split_and_fold_segments_colabfold.py \
  examples/inputs/3013aa.fasta \
  --runtime singularity \
  --colabfold-sif ${MYSCRATCH}/containers/colabfold_rocm7.2.3_nopymol.sif \
  --pymol-sif ${MYSCRATCH}/containers/pymol.sif \
  --colabfold-cache ${MYSCRATCH}/cache \
  --work-dir colabfold_work_rocm7.2.3 \
  --stitch-modes both \
  --validate-adjacent-segments \
  2>&1 | tee colabfold_work_rocm7.2.3/3013aa.colabfold.rocm7.2.3.log
```

**AlphaFold2**

```bash
python3 scripts/split_and_fold_segments_alphafold2.py \
  examples/inputs/3013aa.fasta \
  --runtime singularity \
  --alphafold2-sif ${MYSCRATCH}/containers/alphafold2_rocm7.2.3_nopymol.sif \
  --pymol-sif ${MYSCRATCH}/containers/pymol.sif \
  --data-dir /scratch/references/alphafold_feb2024/databases \
  --work-dir alphafold2_work_rocm7.2.3 \
  --no-use-precomputed-msas \
  --stitch-modes both \
  --validate-adjacent-segments \
  2>&1 | tee alphafold2_work_rocm7.2.3/3013aa.alphafold2.rocm7.2.3.log
```

Repeat with `5005aa.fasta` and `1IH7.7.a3m` in the **same** `--work-dir` to build a
multi-target candidate tree for one report.

### 3.2 Useful flags

| Flag | Purpose |
|------|---------|
| `--work-dir DIR` | Root for chunk files, stitched PDBs, `.split_fold_stitch/`, logs |
| `--stitch-modes both` | Write `*_stitched_plddt.pdb` and `*_stitched_rmsd.pdb` (default: `both`) |
| `--plan-mode default` | Fixed ~3000 aa windows (default) |
| `--plan-mode balanced` | Shorter first segment when one chunk would dominate (e.g. 3013 aa) |
| `--validate-adjacent-segments` | Pre-stitch overlap checks in PyMOL |
| `--skip-colabfold` / `--skip-alphafold` | Re-stitch only (fold outputs must already exist) |
| `--max-chunk-aa N` | Cap segment width (default 3012) |

### 3.3 Comparing tiling plans (default vs balanced)

Use **separate work directories** and **distinct candidate labels** in the report:

```bash
# Default plan
python3 scripts/split_and_fold_segments_alphafold2.py \
  examples/inputs/3013aa.fasta \
  --work-dir alphafold2_work_rocm7.2.3 \
  --plan-mode default \
  ...

# Balanced plan (separate directory)
python3 scripts/split_and_fold_segments_alphafold2.py \
  examples/inputs/3013aa.fasta \
  --work-dir alphafold2_work_rocm7.2.3_balanced \
  --plan-mode balanced \
  ...
```

Do not mix default and balanced part FASTAs in one directory; reporting discovers tiling from
part filenames and may need label hints (`alphafold2-default` vs `alphafold2-balanced`).

---

## 4. Work directory layout (what reporting needs)

Under each `--work-dir`:

| Artifact | Required for report? | ColabFold path | AlphaFold2 path |
|----------|----------------------|----------------|-----------------|
| Stitched PDBs | **Yes** | `{base}_stitched_plddt.pdb`, `{base}_stitched_rmsd.pdb` | same |
| Part inputs | **Yes** (tiling) | `{base}_part_{i}_{start}-{end}.fa` | `{base}_part_{i}_{start}-{end}.fasta` |
| Plan JSON | Optional | `.split_fold_stitch/{base}_stitch_plddt.json` | same |
| Run log | Optional (segment scores) | `{base}.colabfold.*.log` | not parsed yet |
| Per-chunk PDBs | No (for report) | `{base}_part_*_output/*_rank_001*.pdb` | `af2_predictions/{base}_part_*/ranked_0.pdb` |
| `features.pkl` | **No** — skip transfer | — | ~GB per segment |

Example after running all three targets in one AF2 work dir:

```text
alphafold2_work_rocm7.2.3/
  3013aa.fasta
  3013aa_part_0_1-3000.fasta
  3013aa_part_1_2001-3013.fasta
  3013aa_stitched_plddt.pdb
  3013aa_stitched_rmsd.pdb
  5005aa.fasta
  5005aa_part_*
  5005aa_stitched_*.pdb
  1IH7.7.a3m
  1IH7.7_part_*
  1IH7.7_stitched_*.pdb
  .split_fold_stitch/
    3013aa_stitch_plddt.json
    5005aa_stitch_plddt.json
    1IH7.7_stitch_plddt.json
```

Plan files are **per target** (`{base}_stitch_*.json`), not overwritten when you stitch the
next sequence in the same work dir.

---

## 5. Transferring results off Setonix

Copy the **minimal validation set** — stitched PDBs, part FASTAs/A3Ms, and plan JSON. Omit
`af2_predictions/**/features.pkl` and other large intermediates.

**Filtered rsync (recommended)**

```bash
REMOTE=user@setonix:/scratch/.../GPU_biology/alphafold2_work_rocm7.2.3
LOCAL=/path/to/GPU_biology/Pawsey/MI250X/alphafold2_work_rocm7.2.3

rsync -av \
  --include='*/' \
  --include='*_stitched_*.pdb' \
  --include='*_part_*' \
  --include='*.fasta' --include='*.fa' --include='*.a3m' \
  --include='.split_fold_stitch/***' \
  --include='*.log' \
  --exclude='*' \
  "${REMOTE}/" "${LOCAL}/"
```

**Tar pipe (fast for a few targets)**

```bash
ssh user@setonix 'cd .../alphafold2_work_rocm7.2.3 && tar czf - \
  *_stitched_*.pdb *_part_* .split_fold_stitch *.fasta *.a3m' \
  | tar xzf - -C "${LOCAL}"
```

---

## 6. Full-length reference directories

Reports compare stitched candidates to **monolithic** full-length folds (different machine).

### ColabFold references (`--reference-root`)

Default in scripts: `/home/sajandhy/colabfold_work`

```text
colabfold_work/
  3013aa_output/*_rank_001*.pdb
  5005aa_output/*_rank_001*.pdb
  1IH7.7_output/*_rank_001*.pdb
  3013aa.log          # optional; rank pLDDT/pTM in report
```

### AlphaFold2 references (`--af2-output-root`)

Default: `/home/sajandhy/af2_output`

```text
af2_output/
  3013aa/ranked_0.pdb
  5005aa/ranked_0.pdb
  1IH7/ranked_0.pdb   # note: directory is 1IH7, target name 1IH7.7
```

---

## 7. Local setup for reporting

Reporting uses the **PyMOL Python API** (same logic as the stitch worker).

```bash
python3 -m venv ~/.venv
source ~/.venv/bin/activate
pip install pymol-open-source

cd /path/to/GPU_biology
python3 scripts/generate_split_stitch_validation_report.py --help
```

Alternatively run via a PyMOL container with the repo bind-mounted.

---

## 8. Generate comparison reports

### 8.1 Combined ColabFold + AlphaFold2 report

```bash
python3 scripts/generate_split_stitch_validation_report.py \
  --reference-root /home/sajandhy/colabfold_work \
  --af2-output-root /home/sajandhy/af2_output \
  --candidate-dir colabfold:/path/to/colabfold_work_rocm7.2.3 \
  --candidate-dir alphafold2:/path/to/alphafold2_work_rocm7.2.3 \
  --stitch-suffix both \
  --output reports/split_stitch_validation_report.md
```

### 8.2 AlphaFold2 only

```bash
python3 scripts/generate_split_stitch_validation_report.py \
  --no-colabfold-ref \
  --af2-output-root /home/sajandhy/af2_output \
  --candidate-dir alphafold2-default:/path/to/alphafold2_work_rocm7.2.3 \
  --candidate-dir alphafold2-balanced:/path/to/alphafold2_work_rocm7.2.3_balanced \
  --stitch-suffix both \
  --output reports/alphafold2_split_stitch_validation_report.md
```

### 8.3 ColabFold only (shorthand `--work-dir`)

`--work-dir` is sugar for `--candidate-dir colabfold:PATH`:

```bash
python3 scripts/generate_split_stitch_validation_report.py \
  --work-dir /path/to/colabfold_work_rocm7.2.3 \
  --no-af2-ref \
  --reference-root /home/sajandhy/colabfold_work \
  --output reports/colabfold_split_stitch_validation_report.md
```

### 8.4 Report CLI reference

| Flag | Meaning |
|------|---------|
| `--candidate-dir LABEL:PATH` | Stitched work tree; repeat for multiple runs. Label appears in the report table. Use prefixes `colabfold` or `alphafold2` (e.g. `alphafold2-default`). |
| `--work-dir PATH` | Shorthand for one ColabFold candidate |
| `--stitch-suffix both` | Compare `plddt`, `rmsd`, or `both` stitched variants |
| `--reference-root` | ColabFold full-length root |
| `--af2-output-root` | AlphaFold2 full-length root |
| `--no-colabfold-ref` / `--no-af2-ref` | Skip one reference backend |
| `--output` | Markdown report path (default `reports/split_stitch_validation_report.md`) |
| `--platform`, `--container` | Labels in the report header |

The generator scans each candidate dir for known targets (`3013aa`, `5005aa`, `1IH7.7`) that
have `{target}_stitched_{plddt,rmsd}.pdb`. Missing targets are skipped silently.

---

## 9. Single-target check (`compare_stitched_to_reference.py`)

Quick RMSD without generating a full report:

```bash
# From explicit PDB path
pymol -cq scripts/compare_stitched_to_reference.py -- \
  --reference-backend alphafold2 \
  --af2-output-root /home/sajandhy/af2_output \
  --target 3013aa \
  --candidate /path/to/alphafold2_work_rocm7.2.3/3013aa_stitched_plddt.pdb

# From work directory (discovers tiling from part files / plan JSON)
pymol -cq scripts/compare_stitched_to_reference.py -- \
  --reference-backend alphafold2 \
  --af2-output-root /home/sajandhy/af2_output \
  --target 3013aa \
  --candidate-dir /path/to/alphafold2_work_rocm7.2.3 \
  --stitch-suffix both
```

---

## 10. Troubleshooting

| Problem | Likely cause | Fix |
|---------|--------------|-----|
| `No comparisons produced` | No `{target}_stitched_*.pdb` in candidate dir | Re-run stitch or fix rsync includes |
| Wrong tiling / PyMOL overlap error | Mixed part files from different plans in one dir | Separate work dirs; use `alphafold2-default` / `alphafold2-balanced` labels |
| Missing reference | Path or subdir mismatch (e.g. `1IH7` vs `1IH7.7`) | Check `--af2-output-root` layout (§6) |
| `Import pymol` fails | PyMOL not in active Python env | `pip install pymol-open-source` or use PyMOL container |
| Segment scores `n/a` in report | ColabFold log missing or AF2 backend | Only ColabFold logs are parsed for per-segment pLDDT today |
| Plan JSON shows wrong target | Old unprefixed `stitch_plddt.json` from before prefix change | Re-stitch with updated scripts; prefer `{base}_stitch_*.json` |

---

## 11. End-to-end checklist

**On Setonix**

- [ ] `module load singularity/3.11.4-nompi`
- [ ] Run each target (or batch) with `--work-dir <run>/`
- [ ] Confirm `{base}_stitched_plddt.pdb` exists for each target
- [ ] Optional: `--validate-adjacent-segments` for junction RMSD in logs

**Transfer**

- [ ] Rsync stitched PDBs + part files + `.split_fold_stitch/` (not `.pkl`)

**Local**

- [ ] Full-length references in `colabfold_work` and/or `af2_output`
- [ ] PyMOL available in Python env
- [ ] `generate_split_stitch_validation_report.py` with `--candidate-dir` pointing at work dir(s)
- [ ] Review `reports/*.md`
