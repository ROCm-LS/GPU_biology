# Host scripts

Run from anywhere; use absolute paths or `cd` to your project tree first.

## Prerequisites

- Docker (or Singularity/Apptainer) with ROCm GPU devices passed through as in the `*_docker_run.sh` scripts.
- For **AlphaFold2**: a host directory mounted at **`/work`** in both the AlphaFold2 and PyMOL images, with inputs/outputs and a **`databases`** tree under it. That tree can be **full** (customer) or **minimal + reduced_dbs** (internal cluster); see the root **`README.md`** section *AlphaFold2 database setups*.

## Long-sequence pipelines

### ColabFold + PyMOL

```text
python3 /path/to/GPU_biology/scripts/split_and_fold_segments_colabfold.py QUERY.fa \
  --work-dir /path/to/project \
  --colabfold-cache /path/to/colabfold_cache \
  -- --<any colabfold_batch flags>
```

### AlphaFold2 + PyMOL (FASTA; same script, full or minimal DB)

Flags after `--` must match your site: **`full_dbs`** + complete paths (customer), or **`reduced_dbs`** + a small `databases` tree + real **pdb70** (internal). Bootstrap a minimal tree with **`alphafold2/scripts/create_dummy_reduced_databases.sh`**, then download pdb70 into `…/pdb70/pdb70/`. See root **`README.md`**.

```text
python3 /path/to/GPU_biology/scripts/split_and_fold_segments_alphafold2.py QUERY.fa \
  --work-dir /path/to/project \
  --alphafold2-container-name <running_af2_container> \
  -- --data_dir=/work/databases --model_preset=monomer --db_preset=full_dbs \
  <…remaining flags required by your AlphaFold2 image…>
```

- **`--work-dir`**: common parent of input FASTA, chunk FASTAs, and `--af2-output-base` (default `<work-dir>/af2_predictions`).
- **`--`**: everything after is forwarded to **`run_alphafold.py`** unchanged.

Environment overrides for the docker helper scripts: see `alphafold2_docker_run.sh` / `colabfold_docker_run.sh` headers (`ALPHAFOLD2_ROCM_VERSION`, `COLABFOLD_ROCM_VERSION`, `ALPHAFOLD2_IMAGE`, `COLABFOLD_IMAGE`, `MYSCRATCH`, etc.).

## Single-container pipelines (inside the fold image)

These scripts bundle tiling, folding, and PyMOL merge logic in one file. They do **not** use `split_fold_stitch/` or a second PyMOL container.

**ColabFold:** see **`colabfold/rocm7.2.3/README.md`** — **`INSTALL_PYMOL=1`** (default) installs **pymol-open-source** for single-container stitch; **`INSTALL_PYMOL=0`** matches a ColabFold-only image (use **`split_and_fold_segments_colabfold.py`**). **`colabfold/rocm6.2.4/`** has no PyMOL unless you extend it.

**AlphaFold2:** see **`alphafold2/rocm7.2.3/README.md`** — default image has no PyMOL; **`--build-arg INSTALL_PYMOL=1`** uses the same PyMOL wheel and graphics **`apt`** stack as ColabFold 7.2.3 with **`INSTALL_PYMOL=1`**, for **`split_and_fold_segments_alphafold2_single_container.py`**.

### ColabFold + PyMOL (one image)

```text
python3 /path/to/GPU_biology/scripts/split_and_fold_segments_colabfold_single_container.py QUERY.fa \
  --max-chunk-aa 400 -- --num-recycle 3
```

On **ROCm 7.2.3**, that script alone adds JAX/XLA settings (including `--xla_gpu_enable_triton_gemm=false`) when the environment looks like 7.2.3 (e.g. `ROCM_PATH` contains `/rocm-7.2.3`). Set `GPU_BIOLOGY_FORCE_ROCM_732_JAX=1` or `0` to force that workaround on or off. Dual-container orchestration and the AlphaFold2 single-container script use only minimal `XLA_FLAGS` (`--xla_gpu_autotune_level=0`).

### AlphaFold2 + PyMOL (one image)

```text
python3 /path/to/GPU_biology/scripts/split_and_fold_segments_alphafold2_single_container.py QUERY.fa \
  --output-dir-base /work/run1 --data-dir /work/databases \
  -- --model_preset=monomer …
```
