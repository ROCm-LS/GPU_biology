#!/usr/bin/env bash
# Create a minimal directory tree for AlphaFold2 --db_preset=reduced_dbs on clusters
# where full genetic/template archives are not mirrored.
#
# This script writes **tiny placeholder FASTA** files for UniRef90, MGnify, and small_bfd
# (enough for path checks; not useful for real MSA search — use precomputed MSAs or
# expect searches to be degenerate). It creates **pdb_mmcif** layout and an **empty
# pdb70 directory** with instructions only.
#
# **pdb70** is NOT generated here: populate it with the real PDB70 database using the
# download helpers under /app/alphafold/scripts inside your AlphaFold container, then
# point --pdb70_database_path at that directory (see README.md).
#
# Usage:
#   ./create_dummy_reduced_databases.sh [OUTPUT_DIR]
# Default OUTPUT_DIR: ./minimal_af2_databases

set -euo pipefail

ROOT="${1:-./minimal_af2_databases}"
mkdir -p "$ROOT"
ROOT="$(cd "$ROOT" && pwd)"

echo "-> Creating reduced_dbs layout under: $ROOT"

mkdir -p "${ROOT}/uniref90" \
         "${ROOT}/mgnify" \
         "${ROOT}/small_bfd" \
         "${ROOT}/pdb_mmcif/mmcif_files" \
         "${ROOT}/pdb70/pdb70"

# Minimal valid FASTA (single short sequence) — placeholders only.
cat >"${ROOT}/uniref90/uniref90.fasta" <<'EOF'
>placeholder_uniref90
AAAA
EOF

cat >"${ROOT}/mgnify/mgy_clusters_2022_05.fa" <<'EOF'
>placeholder_mgnify
CCCC
EOF

cat >"${ROOT}/small_bfd/bfd-first_non_consensus_sequences.fasta" <<'EOF'
>placeholder_small_bfd
GGGG
EOF

# AlphaFold expects obsolete.dat (can be empty for “no obsolete entries” style use).
touch "${ROOT}/pdb_mmcif/obsolete.dat"

# pdb70: real database only — inner pdb70/ is where official download scripts place files.
cat >"${ROOT}/pdb70/README_PDB70.txt" <<'EOF'
This layout expects the **real PDB70** database inside the inner `pdb70/` directory
(i.e. .../databases/pdb70/pdb70/ with mmseqs2-style files from the official download).

From a running AlphaFold container (clone at /app/alphafold in GPU_biology alphafold2/rocm7.2.3 image):

  docker exec -w /app/alphafold <container_name> bash -lc 'ls scripts'

Use the download script(s) documented for your AlphaFold tag to fetch PDB70 **into**
this inner directory (see AlphaFold README). Then pass:

  --pdb70_database_path=.../databases/pdb70/pdb70

…using the same parent path you passed to --data_dir / .../databases.
EOF

cat >"${ROOT}/README_GPU_biology_minimal_dbs.txt" <<EOF
Minimal AlphaFold2 "reduced_dbs" layout (GPU_biology).

- uniref90/, mgnify/, small_bfd/: **dummy FASTA placeholders** — not for production MSA.
  Use --use_precomputed_msas=true and real .sto/.a3m under your run if you skip real DBs.
  ColabFold .a3m → .sto: alphafold2/scripts/convert_colabfold_a3m_to_sto.py (see alphafold2/scripts/README.md).
- pdb_mmcif/: obsolete.dat present; mmcif_files/ is empty — add mmCIFs if you enable templates
  beyond PDB70, or rely on PDB70 only per your flags.
- pdb70/pdb70/: **empty inner dir** — populate with the real PDB70 download (see pdb70/README_PDB70.txt).

Example run_alphafold.py flags (adjust paths to your mount, e.g. /work/databases):

  --db_preset=reduced_dbs \\
  --data_dir=${ROOT} \\
  --uniref90_database_path=${ROOT}/uniref90/uniref90.fasta \\
  --mgnify_database_path=${ROOT}/mgnify/mgy_clusters_2022_05.fa \\
  --small_bfd_database_path=${ROOT}/small_bfd/bfd-first_non_consensus_sequences.fasta \\
  --pdb70_database_path=${ROOT}/pdb70/pdb70 \\
  --template_mmcif_dir=${ROOT}/pdb_mmcif/mmcif_files \\
  --obsolete_pdbs_path=${ROOT}/pdb_mmcif/obsolete.dat

After downloading PDB70, the inner .../pdb70/pdb70 directory should contain the
database files produced by AlphaFold’s download scripts (not committed in GPU_biology).
EOF

echo "-> Done."
echo "   Next: populate ${ROOT}/pdb70/pdb70/ with the real PDB70 database (see pdb70/README_PDB70.txt)."
