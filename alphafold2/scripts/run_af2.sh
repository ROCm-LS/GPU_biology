#!/usr/bin/env bash
# Example local run. Database tree (--data_dir and paths below) defaults to /work/databases
# (persistent). Override with ALPHAFOLD_DATA_DIR=/path/to/tree. Do not keep large real DBs only
# under /tmp. See scripts/download_all_data.sh for layout.
# pdb70 HHsearch prefix must be .../pdb70/pdb70 (see docker/run_docker.py).
#
# With --use_precomputed_msas, AlphaFold reads ONLY:
#   ${OUTPUT_DIR}/${FASTA_STEM}/msas/uniref90_hits.sto
#   ${OUTPUT_DIR}/${FASTA_STEM}/msas/mgnify_hits.sto
#   ${OUTPUT_DIR}/${FASTA_STEM}/msas/small_bfd_hits.sto
# If ANY are missing, Jackhmmer runs and needs real *_database_path files.
#
# Optional: set COLABFOLD_A3M=/path/to/query.a3m to auto-fill msas/ via convert_colabfold_a3m_to_sto.py

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ALPHAFOLD_HOME="${ALPHAFOLD_HOME:-/app/alphafold}"
RUN_ALPHAFOLD="${ALPHAFOLD_HOME}/run_alphafold.py"

FASTA="${FASTA:-/work/inputs/10aa.fasta}"
OUTPUT_DIR="${OUTPUT_DIR:-/work/af2_output}"
DATA_DIR="${ALPHAFOLD_DATA_DIR:-/work/databases}"
USE_PRECOMPUTED_MSAS="${USE_PRECOMPUTED_MSAS:-true}"

# Must match run_alphafold.py: pathlib.Path(fasta_path).stem (strips one suffix: .fa, .fasta, …).
FASTA_BASE="$(basename "${FASTA}")"
FASTA_STEM="${FASTA_BASE%.fasta}"
FASTA_STEM="${FASTA_STEM%.FASTA}"
FASTA_STEM="${FASTA_STEM%.fa}"
FASTA_STEM="${FASTA_STEM%.FA}"
MSA_DIR="${OUTPUT_DIR}/${FASTA_STEM}/msas"

if [[ "${USE_PRECOMPUTED_MSAS}" == "true" ]]; then
  mkdir -p "${MSA_DIR}"
  if [[ -n "${COLABFOLD_A3M:-}" ]]; then
    python3 "${SCRIPT_DIR}/convert_colabfold_a3m_to_sto.py" "${COLABFOLD_A3M}" "${MSA_DIR}"
  fi
  _missing=()
  for f in uniref90_hits.sto mgnify_hits.sto small_bfd_hits.sto; do
    [[ -f "${MSA_DIR}/${f}" ]] || _missing+=("${MSA_DIR}/${f}")
  done
  if [[ "${#_missing[@]}" -gt 0 ]]; then
    echo "Precomputed MSAs incomplete (--use_precomputed_msas). Missing:" >&2
    printf '  %s\n' "${_missing[@]}" >&2
    echo "Jackhmmer would run and require a real UniRef90 FASTA at --uniref90_database_path." >&2
    echo "Fix: populate the directory above (e.g. COLABFOLD_A3M=/colabfold_work/run1/${FASTA_STEM}_output/*.a3m bash $0) or download DBs and set USE_PRECOMPUTED_MSAS=false." >&2
    exit 1
  fi
fi

if [[ ! -f "${RUN_ALPHAFOLD}" ]]; then
  echo "run_alphafold.py not found at ${RUN_ALPHAFOLD}. Set ALPHAFOLD_HOME (default /app/alphafold)." >&2
  exit 1
fi

PRECOMPUTED_FLAG=(--use_precomputed_msas=true)
if [[ "${USE_PRECOMPUTED_MSAS}" != "true" ]]; then
  PRECOMPUTED_FLAG=(--use_precomputed_msas=false)
fi

python3 "${RUN_ALPHAFOLD}" \
  --fasta_paths="${FASTA}" \
  --output_dir="${OUTPUT_DIR}" \
  --model_preset=monomer \
  --db_preset=reduced_dbs \
  --max_template_date=1900-01-01 \
  --use_gpu_relax=false \
  --data_dir="${DATA_DIR}" \
  --uniref90_database_path="${DATA_DIR}/uniref90/uniref90.fasta" \
  --mgnify_database_path="${DATA_DIR}/mgnify/mgy_clusters_2022_05.fa" \
  --small_bfd_database_path="${DATA_DIR}/small_bfd/bfd-first_non_consensus_sequences.fasta" \
  --pdb70_database_path="${DATA_DIR}/pdb70/pdb70" \
  --template_mmcif_dir="${DATA_DIR}/pdb_mmcif/mmcif_files" \
  --obsolete_pdbs_path="${DATA_DIR}/pdb_mmcif/obsolete.dat" \
  "${PRECOMPUTED_FLAG[@]}"
