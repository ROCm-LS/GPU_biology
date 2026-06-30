#!/usr/bin/env python3
# Copyright 2021 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Convert ColabFold A3M to AlphaFold precomputed Stockholm MSAs (reduced_dbs).

Writes three identical files expected by run_alphafold.py with
--use_precomputed_msas and --db_preset=reduced_dbs:
  uniref90_hits.sto, mgnify_hits.sto, small_bfd_hits.sto

Example:
  python3 convert_colabfold_a3m_to_sto.py \\
    /work/colabfold_work/10aa_output/_10_AAs.a3m \\
    /work/af2_out/10aa/msas
"""

from __future__ import annotations

import argparse
from pathlib import Path

from alphafold.data import parsers

OUTPUT_NAMES = (
    'uniref90_hits.sto',
    'mgnify_hits.sto',
    'small_bfd_hits.sto',
)


def _strip_colabfold_a3m_prefix(text: str) -> str:
  """Drop ColabFold/mmseqs header lines (e.g. '#123\\t1') before FASTA parsing."""
  kept = []
  for line in text.splitlines():
    stripped = line.strip()
    if stripped.startswith('#'):
      continue
    kept.append(line)
  return '\n'.join(kept)


def a3m_to_af2_sto_string(a3m_path: str) -> str:
  """Returns Stockholm-like lines parse_stockholm() can read (first seq = query)."""
  with open(a3m_path, encoding='utf-8') as f:
    raw = f.read()
  msa = parsers.parse_a3m(_strip_colabfold_a3m_prefix(raw))

  lines = []
  counts: dict[str, int] = {}
  align_lines: list[str] = []
  for i, (seq, desc) in enumerate(zip(msa.sequences, msa.descriptions)):
    raw = (desc.split()[0] if desc else f'seq{i}').strip()
    if raw not in counts:
      counts[raw] = 0
      name = raw
    else:
      counts[raw] += 1
      name = f'{raw}_{counts[raw]}'
    # Single space (not tab): deduplicate_stockholm_msa/_keep_line use partition(' ').
    align_lines.append(f'{name} {seq}')

  width = len(msa.sequences[0]) if msa.sequences else 0
  # Chunk terminator expected by remove_empty_columns_from_stockholm_msa (same width as alignments).
  rf_line = f'#=GC RF {"." * width}'
  parts = ['# STOCKHOLM 1.0'] + align_lines + [rf_line, '//']
  return '\n'.join(parts) + '\n'


def main() -> None:
  parser = argparse.ArgumentParser(
      description=(
          'Convert a ColabFold-style .a3m into AlphaFold precomputed .sto files.'
      ))
  parser.add_argument(
      'a3m_path',
      type=Path,
      help='Input ColabFold .a3m path.')
  parser.add_argument(
      'output_dir',
      type=Path,
      help=(
          'Directory to write uniref90_hits.sto, mgnify_hits.sto, '
          'small_bfd_hits.sto (e.g. {output_dir}/{fasta_stem}/msas).'))
  args = parser.parse_args()

  a3m_path = args.a3m_path.expanduser().resolve()
  out_dir = args.output_dir.expanduser().resolve()

  if not a3m_path.is_file():
    raise SystemExit(f'Not a file: {a3m_path}')
  if a3m_path.suffix.lower() != '.a3m':
    raise SystemExit(f'Expected a .a3m file, got: {a3m_path}')

  out_dir.mkdir(parents=True, exist_ok=True)
  sto = a3m_to_af2_sto_string(str(a3m_path))
  for fname in OUTPUT_NAMES:
    (out_dir / fname).write_text(sto, encoding='utf-8')
  print(f'Wrote {len(OUTPUT_NAMES)} files to {out_dir}')


if __name__ == '__main__':
  main()
