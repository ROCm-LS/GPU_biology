# AlphaFold2 host helpers

Scripts for an **optional** **`reduced_dbs`** workflow: avoid **Jackhmmer** over full UniRef/MGnify/BFD archives by supplying **precomputed MSAs** to `run_alphafold.py` (often after ColabFold MSA generation).

**Not in Docker images by design.** Published Pawsey AlphaFold2 / ColabFold images (and the GPU_biology app Dockerfiles) ship inference only — `run_alphafold.py`, `colabfold_batch`, params/cache mounts. They do **not** include `run_af2.sh`, `convert_colabfold_a3m_to_sto.py`, or `create_dummy_reduced_databases.sh`. You only need these when using the **minimal database + ColabFold MSA handoff** path below. For **`full_dbs`** with genetic search inside AlphaFold, call **`run_alphafold.py`** with your site’s flags and ignore this folder.

Deliver helpers via **bind-mounted `/work`** (or `docker cp` / copy on the host) — see [Getting these scripts into the AlphaFold2 container](#getting-these-scripts-into-the-alphafold2-container).

## When to use this

| Goal | Approach |
|------|----------|
| Full genetic search inside AlphaFold | `--db_preset=full_dbs`, real database tree, `--use_precomputed_msas=false` |
| Minimal DB tree + **MSA from ColabFold** | `--db_preset=reduced_dbs`, dummy genetic DBs from `create_dummy_reduced_databases.sh`, real **pdb70**, `--use_precomputed_msas=true` |
| Tiled long sequence + ColabFold MSAs | `scripts/split_and_fold_segments_alphafold2_single_container.py` with **`.a3m` / `.a2m` input** (calls the converter per chunk) |

ColabFold is often faster or more convenient for **MSA generation**; AlphaFold2 is then used for **structure prediction** with the same alignment, without hosting multi‑terabyte genetic databases.

## ColabFold vs AlphaFold2 — what to host

Both tools support **reduced** setups (see root **`README.md`** and **`colabfold/rocm7.2.3/README.md`**). ColabFold’s default local MSA stack is **already smaller than AlphaFold `full_dbs`** (UniRef30 + env via MMseqs2, not UniRef90 + MGnify + BFD). AlphaFold2’s **`reduced_dbs` + precomputed MSAs** can be **smaller still** on disk: placeholder genetic FASTAs, real **pdb70** only, and `.sto` files from ColabFold.

| Stage | Tool | Minimal storage |
|-------|------|-----------------|
| MSA from FASTA | ColabFold | `colabfold_batch … --msa-only` (API MSA, no fold); or full `colabfold_batch` and use the `.a3m` from output |
| MSA already known | ColabFold | Input `.a3m` — no search DBs; params in `/cache` if folding in ColabFold |
| Structure | AlphaFold2 | `create_dummy_reduced_databases.sh` + pdb70 + `.sto` from **`convert_colabfold_a3m_to_sto.py`** |

## Layout AlphaFold expects (`--use_precomputed_msas`)

For each FASTA stem `QUERY` and `--output_dir=OUT`, AlphaFold reads:

```text
OUT/QUERY/msas/uniref90_hits.sto
OUT/QUERY/msas/mgnify_hits.sto
OUT/QUERY/msas/small_bfd_hits.sto
```

With `--db_preset=reduced_dbs`, the three `.sto` files may contain the **same** MSA (the converter writes identical content to all three). You still need a **minimal `--data_dir`** (placeholders + real pdb70) — see **`create_dummy_reduced_databases.sh`** and root **`README.md`**.

If any `.sto` is missing, AlphaFold runs **Jackhmmer** and requires real genetic FASTAs at `--uniref90_database_path`, etc.

## `convert_colabfold_a3m_to_sto.py`

Converts a **ColabFold-style `.a3m`** (e.g. from `colabfold_batch` output `…_output/query.a3m`) into the three Stockholm files above.

To generate that `.a3m` from FASTA **without** ColabFold structure prediction, run **`colabfold_batch … --msa-only`** (uses the public MSA API by default; writes alignments under the output directory, then exits). A second `colabfold_batch` on the same paths can fold later if you stay in ColabFold; for AlphaFold2, take the `.a3m` and convert as below.

- Strips ColabFold/mmseqs comment lines (`#…`).
- Parses with AlphaFold’s `alphafold.data.parsers` — run **inside the AlphaFold2 image** (or any env with the `alphafold` package on `PYTHONPATH`).

```bash
# Inside AlphaFold2 container (paths under /work are typical bind mounts)
python3 /path/to/GPU_biology/alphafold2/scripts/convert_colabfold_a3m_to_sto.py \
  /colabfold_work/run1/query_output/query.a3m \
  /work/af2_out/query/msas
```

Arguments:

1. **`a3m_path`** — input `.a3m`
2. **`output_dir`** — directory that will contain the three `*_hits.sto` files (created if needed)

## `run_af2.sh` — example single-sequence run

Wrapper around `run_alphafold.py` with **`reduced_dbs`**, **`--use_precomputed_msas=true`**, and optional auto-conversion from ColabFold.

Environment variables:

| Variable | Default | Meaning |
|----------|---------|---------|
| `FASTA` | `/work/inputs/10aa.fasta` | Input FASTA |
| `OUTPUT_DIR` | `/work/af2_output` | `--output_dir` |
| `ALPHAFOLD_DATA_DIR` | `/work/databases` | Minimal database tree |
| `USE_PRECOMPUTED_MSAS` | `true` | Set `false` to run Jackhmmer (needs real genetic DBs) |
| `COLABFOLD_A3M` | *(unset)* | If set, runs `convert_colabfold_a3m_to_sto.py` into `OUTPUT_DIR/<fasta_stem>/msas/` before fold |

Example — MSA from ColabFold, fold with AlphaFold2:

```bash
export FASTA=/work/inputs/query.fasta
export OUTPUT_DIR=/work/af2_out
export ALPHAFOLD_DATA_DIR=/work/databases
export COLABFOLD_A3M=/colabfold_work/run1/query_output/query.a3m

bash /work/af2_scripts/run_af2.sh
```

### Getting these scripts into the AlphaFold2 container

GPU_biology **does not `COPY` these into app images** — same as published Pawsey tags. The image already has **`/app/alphafold/run_alphafold.py`** and the **`alphafold`** package; add repo helpers only when you use the minimal-DB workflow.

| Approach | When to use |
|----------|-------------|
| **Copy on host** into `$ALPHAFOLD2_WORK_DIR/af2_scripts/` | Persistent under bind-mounted **`/work`** (recommended) |
| **`docker cp`** | Quick local test into a running container — see below |
| **Extra bind-mount** | Dev: `-v …/GPU_biology/alphafold2/scripts:/work/af2_scripts:ro` on `docker run` |

`create_dummy_reduced_databases.sh` runs on the **host** (or any shell); point the output tree at **`/work/databases`** in the container.

**`docker cp` into a running AlphaFold2 container** (after `scripts/alphafold2_docker_run.sh`):

```bash
REPO=/path/to/GPU_biology
CONTAINER="${ALPHAFOLD2_CONTAINER_NAME:-${USER}_alphafold2_rocm7.2.3}"

docker exec "${CONTAINER}" mkdir -p /work/af2_scripts
docker cp "${REPO}/alphafold2/scripts/run_af2.sh" \
  "${CONTAINER}:/work/af2_scripts/run_af2.sh"
docker cp "${REPO}/alphafold2/scripts/convert_colabfold_a3m_to_sto.py" \
  "${CONTAINER}:/work/af2_scripts/convert_colabfold_a3m_to_sto.py"
docker exec "${CONTAINER}" chmod +x /work/af2_scripts/run_af2.sh

# quick check
docker exec "${CONTAINER}" ls -la /work/af2_scripts
docker exec "${CONTAINER}" python3 /work/af2_scripts/convert_colabfold_a3m_to_sto.py --help
```

Because `/work` is bind-mounted, those files also appear on the host at **`$ALPHAFOLD2_WORK_DIR/af2_scripts/`** (default `~/alphafold_work/af2_scripts/`).

Keep helpers in a **subdir** of `/work` (e.g. `/work/af2_scripts/`), not mixed into `databases/`. Inputs, outputs, and `create_dummy_reduced_databases.sh` output stay at `/work/inputs`, `/work/af2_out`, `/work/databases`, etc.

`run_af2.sh` calls **`${ALPHAFOLD_HOME:-/app/alphafold}/run_alphafold.py`** and finds **`convert_colabfold_a3m_to_sto.py`** next to itself. You only need the **two** helper files in the same directory.

Or skip `run_af2.sh` and run the converter plus `run_alphafold.py` directly:

```bash
python3 /work/af2_scripts/convert_colabfold_a3m_to_sto.py /work/colab.a3m /work/af2_out/query/msas
python3 /app/alphafold/run_alphafold.py --fasta_paths=… --use_precomputed_msas=true …
```

Equivalent flags (see script for full list):

```text
--model_preset=monomer
--db_preset=reduced_dbs
--use_precomputed_msas=true
--max_template_date=1900-01-01
--data_dir=… --pdb70_database_path=…/pdb70/pdb70
```

## End-to-end workflow (ColabFold MSA → AlphaFold2 fold)

```mermaid
flowchart LR
  A[FASTA] --> B[colabfold_batch]
  B --> C["*.a3m in *_output/"]
  C --> D[convert_colabfold_a3m_to_sto.py]
  D --> E["OUT/stem/msas/*.sto"]
  F[Minimal data_dir + pdb70] --> G[run_alphafold.py]
  A --> G
  E --> G
  G --> H[ranked_*.pdb]
```

1. Bootstrap **`/work/databases`**: `create_dummy_reduced_databases.sh`, then download **pdb70** (root **`README.md`**).
2. Run **ColabFold** on the query FASTA — e.g. `colabfold_batch /work/query.fasta /colabfold_work/run1 --msa-only` (GPU not required for MSA-only; params/cache under `/cache` as usual). Note the output `.a3m` (under `/colabfold_work/run1…_output/`). Use the same host path for **`COLABFOLD_MSA_DIR`** in both **`colabfold_docker_run.sh`** and **`alphafold2_docker_run.sh`** so AlphaFold2 sees `/colabfold_work`.
3. **Convert** A3M → `.sto` (converter or `COLABFOLD_A3M=… run_af2.sh`).
4. **Fold** with `run_alphafold.py` / `run_af2.sh` and `--use_precomputed_msas=true`.

## Tiling + stitch with A3M input

**`scripts/split_and_fold_segments_alphafold2_single_container.py`** accepts **FASTA, A2M, or A3M**. For MSA inputs it slices alignments per chunk, writes chunk FASTAs, and calls **`convert_colabfold_a3m_to_sto.py`** into each chunk’s `msas/` directory before `run_alphafold.py`. Defaults match `run_af2.sh` (`reduced_dbs`, precomputed MSAs).

```bash
python3 scripts/split_and_fold_segments_alphafold2_single_container.py \
  /work/long_query.a3m \
  --output-dir-base /work/af2_chunks \
  --data-dir /work/databases
```

The two-container host script **`split_and_fold_segments_alphafold2.py`** is **FASTA-only**; pre-convert MSAs per chunk or use the single-container script for A3M.

## `create_dummy_reduced_databases.sh`

Creates placeholder genetic DB files and mmcif layout for **`--db_preset=reduced_dbs`**. Does **not** replace pdb70 or real MSAs. See generated **`README_GPU_biology_minimal_dbs.txt`** in the output tree.

## Related docs

- Root **`README.md`** — database layouts (full vs minimal) and customer workflow.
- **`alphafold2/rocm7.2.3/README.md`** — build and run the AlphaFold2 container.
- **`colabfold/rocm7.2.3/README.md`** — ColabFold cache and `colabfold_batch` defaults.
- **`scripts/README.md`** — host tiling/stitch entrypoints.
