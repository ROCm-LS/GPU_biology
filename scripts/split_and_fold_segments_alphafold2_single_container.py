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

r"""
Split, fold (``run_alphafold.py``), and stitch long sequences **inside one AlphaFold2
Docker image** that also has ``pymol-open-source`` available (often installed with root).

**Runs inside the fold container** (e.g. ``docker exec -w /work <af2_container> python3 …``).
For a **host** that launches separate AlphaFold2 and PyMOL containers, use
``split_and_fold_segments_alphafold2.py`` in this directory instead.

Same workflow idea as ``split_and_fold_segments_colabfold_single_container.py``, but
calling ``run_alphafold.py`` instead of ``colabfold_batch``. Duplicate tiling / PyMOL
logic vs the dual-container host scripts is intentional (standalone).

Requires: PyMOL Python API, AlphaFold deps, hhsearch/jackhmmer on PATH when not using
precomputed MSAs.

Example:
  python3 split_and_fold_segments_alphafold2_single_container.py query.fa \\
    --output-dir-base /work/af2_chunks/run1 --data-dir /work/databases \\
    -- --use_precomputed_msas=true

Tokens after a bare ``--`` are forwarded to ``run_alphafold.py``.
"""

from __future__ import annotations

import argparse
import collections
import glob
import os
import pathlib
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed

from pymol import cmd, stored

from alphafold.data import parsers as af_parsers

# -----------------------------------------------------------------------------
# Tie-breaks for pLDDT (primary) vs RMSD (secondary) window selection
# -----------------------------------------------------------------------------
_PDDT_TIE: float = 1e-4
_RMSD_TIE: float = 1e-6

# Tiling (same defaults as ColabFold companion script)
WINDOW_SIZE = 3000
OVERLAP = 1000
MAX_CHUNK_AA = 3012
MIN_OVERLAP = OVERLAP
JUNCTION_ALIGN_W = 200
ANCHOR_SLIDE = 50


def _parse_hip_visible_devices() -> list[int]:
  raw = os.environ.get('HIP_VISIBLE_DEVICES', '')
  if not raw.strip():
    return []
  out: list[int] = []
  for part in raw.split(','):
    p = part.strip()
    if not p:
      continue
    try:
      out.append(int(p))
    except ValueError:
      pass
  return out


def _discover_gpu_ids() -> list[int]:
  try:
    out = subprocess.check_output(
        'rocm-smi -i|grep GUID|wc -l',
        shell=True,
        text=True,
        stderr=subprocess.STDOUT,
        timeout=30,
    )
  except (FileNotFoundError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
    return [0]
  try:
    n = int(out.strip())
  except ValueError:
    return [0]
  if n > 0:
    return list(range(n))
  return [0]


def _init_gpu_ids() -> list[int]:
  if 'HIP_VISIBLE_DEVICES' in os.environ:
    return _parse_hip_visible_devices()
  return _discover_gpu_ids()


GPU_IDS = _init_gpu_ids()
if not GPU_IDS:
  print(
      'Warning: no GPU indices (empty HIP_VISIBLE_DEVICES?); using [0].',
      file=sys.stderr,
  )
  GPU_IDS = [0]

SCRIPT_PATH = pathlib.Path(__file__).resolve()
REPO_ROOT = SCRIPT_PATH.parent
if str(REPO_ROOT) not in sys.path:
  sys.path.insert(0, str(REPO_ROOT))

from split_fold_stitch.tiling import plan_tiling, print_chunk_plan as print_tiling_plan


def _tiling_window_overlap(max_chunk_aa: int) -> tuple[int, int]:
  w = min(WINDOW_SIZE, max_chunk_aa)
  o = min(
      OVERLAP,
      w - 1,
      max(ANCHOR_SLIDE + 1, w // 3),
  )
  o = min(o, w - 1)
  if o < 1 and w > 1:
    o = min(50, w - 1)
  return w, o


def _is_msa_file(path: str) -> bool:
  return os.path.splitext(path)[1].lower() in ('.a3m', '.a2m')


def _chunk_file_extension(input_path: str) -> str:
  ext = os.path.splitext(input_path)[1]
  return ext if ext else '.fa'


def chunk_stem(base: str, part_index: int, s: int, e: int) -> str:
  return f'{base}_part_{part_index}_{s + 1}-{e}'


def _read_fasta_like_sequence(lines: list[str]) -> tuple[str, str]:
  if not lines:
    raise ValueError('Empty input file.')
  header = lines[0].rstrip('\n\r')
  body = ''.join(l.strip() for l in lines[1:])
  return header, body


def _read_a3m_query_only(lines: list[str]) -> tuple[str, str]:
  if not lines:
    raise ValueError('Empty input file.')
  raw = [ln.rstrip('\n\r') for ln in lines]
  i = 0
  while i < len(raw) and (not raw[i].strip() or raw[i].lstrip().startswith('#')):
    i += 1
  while i < len(raw) and not raw[i].lstrip().startswith('>'):
    i += 1
  if i >= len(raw):
    raise ValueError("A3M: no '>' record found.")
  header = raw[i]
  parts: list[str] = []
  j = i + 1
  while j < len(raw) and not raw[j].lstrip().startswith('>'):
    parts.append(raw[j].strip().replace(' ', ''))
    j += 1
  q = ''.join(parts)
  if not q:
    raise ValueError('A3M: no query sequence under first header; check format.')
  return header, q


def read_sequence_input(path: str) -> tuple[str, str]:
  with open(path, 'r') as f:
    lines = f.readlines()
  ext = os.path.splitext(path)[1].lower()
  if ext in ('.a3m', '.a2m'):
    return _read_a3m_query_only(lines)
  return _read_fasta_like_sequence(lines)


def parse_a3m_file(path: str) -> list[tuple[str, str]]:
  with open(path, 'r') as f:
    lines = f.readlines()
  raw = [ln.rstrip('\n\r') for ln in lines]
  i = 0
  while i < len(raw) and (not raw[i].strip() or raw[i].lstrip().startswith('#')):
    i += 1
  records: list[tuple[str, str]] = []
  while i < len(raw):
    if not raw[i].strip():
      i += 1
      continue
    if not raw[i].lstrip().startswith('>'):
      i += 1
      continue
    h = raw[i]
    i += 1
    parts: list[str] = []
    while i < len(raw) and not raw[i].lstrip().startswith('>'):
      parts.append(raw[i].strip().replace(' ', ''))
      i += 1
    body = ''.join(parts)
    if body:
      records.append((h, body))
  if not records:
    raise ValueError('A3M: no sequence records in file.')
  return records


def iter_a3m_match_blocks(seq: str):
  i = 0
  n = len(seq)
  while i < n:
    if seq[i].islower():
      i += 1
      continue
    j = i + 1
    while j < n and seq[j].islower():
      j += 1
    yield seq[i:j]
    i = j


def a3m_match_state_count_with_check(records: list[tuple[str, str]]) -> int:
  b0 = list(iter_a3m_match_blocks(records[0][1]))
  m = len(b0)
  for idx, (_h, s) in enumerate(records[1:], start=1):
    bm = list(iter_a3m_match_blocks(s))
    if len(bm) != m:
      raise ValueError(
          f'A3M: row 0 has {m} match states, row {idx} has {len(bm)}; '
          'inconsistent MSA (cannot take column-matched segments).'
      )
  return m


def write_a3m_match_slice(
    records: list[tuple[str, str]],
    s: int,
    e: int,
    n_match: int,
    out_path: str,
    part_index: int,
) -> None:
  if s < 0 or e > n_match or s >= e:
    raise ValueError(f'match slice [{s}, {e}) out of range for n_match={n_match!r}.')
  with open(out_path, 'w') as f:
    for h, srow in records:
      bl = list(iter_a3m_match_blocks(srow))
      if len(bl) != n_match:
        raise ValueError('internal: row length mismatch in write_a3m_match_slice')
      sub = ''.join(bl[s:e])
      hline = h.rstrip()
      if f'_p{part_index}' not in hline:
        hline = f'{hline}_p{part_index}'
      f.write(f'{hline}\n{sub}\n')


def get_chunks(total_len: int, *, max_chunk_aa: int = MAX_CHUNK_AA) -> list[tuple[int, int]]:
  if total_len <= 0:
    return []
  w, o = _tiling_window_overlap(max_chunk_aa)
  if w <= o and total_len > max_chunk_aa:
    raise ValueError(
        f'tiling: window {w} must exceed overlap {o}; try a larger --max-chunk-aa'
    )
  chunks: list[tuple[int, int]] = []
  start = 0
  while start < total_len:
    remaining = total_len - start
    if remaining <= max_chunk_aa:
      chunks.append((start, total_len))
      break
    end = start + w
    chunks.append((start, end))
    start = end - o
  return chunks


def validate_chunk_plan(
    chunks: list[tuple[int, int]],
    total_len: int,
    *,
    max_chunk_aa: int = MAX_CHUNK_AA,
    min_adjacent_overlap: int = MIN_OVERLAP,
) -> None:
  if not chunks:
    if total_len == 0:
      return
    raise ValueError('No chunks for non-empty sequence.')
  s0, _e0 = chunks[0]
  if s0 != 0:
    raise ValueError('First chunk should start at residue 0 (0-based).')
  s_last, e_last = chunks[-1]
  if e_last != total_len:
    raise ValueError('Last chunk does not end at L.')
  for i, (a, b) in enumerate(chunks):
    n_aa = b - a
    if n_aa > max_chunk_aa:
      raise ValueError(
          f'Segment {i} has length {n_aa}, must be <= {max_chunk_aa} a.a. '
          f'(0-based [{a}, {b})).'
      )
    if n_aa <= 0:
      raise ValueError(f'Empty segment {i}.')
  for i in range(1, len(chunks)):
    s_prev, e_prev = chunks[i - 1]
    s_i, e_i = chunks[i]
    overlap_len = e_prev - s_i
    if s_i < s_prev:
      raise ValueError('Chunk start indices went backwards; invalid tiling.')
    if s_i > e_prev:
      raise ValueError(
          f'Gap between segment {i - 1} and {i} (0-based: prev end {e_prev}, '
          f'next start {s_i}).'
      )
    if s_i < e_prev and overlap_len < min_adjacent_overlap:
      raise ValueError(
          f'Overlap between chunk {i - 1} and {i} is {overlap_len} a.a.; '
          f'need at least {min_adjacent_overlap} (tiling / OVERLAP / --max-chunk-aa).'
      )


def print_chunk_plan(chunks: list[tuple[int, int]]) -> None:
  print('Segment plan (1-based residue numbers):')
  for i, (s, e) in enumerate(chunks):
    ovl = ''
    if i > 0:
      _sp, ep = chunks[i - 1]
      if s < ep:
        ovl = f'  (overlap with previous: {ep - s} aa)'
    print(f'  part {i}: {s + 1}-{e}  ({e - s} aa){ovl}')


def alphafold_prediction_dir(output_dir_base: str, fasta_path: str) -> str:
  stem = pathlib.Path(fasta_path).stem
  return os.path.join(output_dir_base, stem)


def alphafold_best_model_pdb(prediction_dir: str) -> str | None:
  """AlphaFold monomer writes ranked_0.pdb (best confidence)."""
  p = os.path.join(prediction_dir, 'ranked_0.pdb')
  if os.path.isfile(p):
    return p
  cand = sorted(glob.glob(os.path.join(prediction_dir, 'ranked_*.pdb')))
  return cand[0] if cand else None


def all_chunk_folds_done_af2(out_dirs: list[str]) -> bool:
  for d in out_dirs:
    if alphafold_best_model_pdb(d) is None:
      return False
  return bool(out_dirs)


def _chunk_overlap_local_in_prev(
    s_prev: int, e_prev: int, s_curr: int, _e_curr: int
) -> tuple[int, int] | None:
  if s_curr >= e_prev:
    return None
  lo = s_curr + 1 - s_prev
  hi = e_prev - s_prev
  if hi < lo or (hi - lo) < (ANCHOR_SLIDE + 1):
    return None
  return (lo, hi)


def _anchor_window_is_better(
    plddt_new: float,
    rmsd_new: float,
    plddt_cur: float,
    rmsd_cur: float,
    primary: str,
) -> bool:
  if primary == 'plddt':
    if plddt_new > plddt_cur + _PDDT_TIE:
      return True
    if abs(plddt_new - plddt_cur) <= _PDDT_TIE and rmsd_new < rmsd_cur - _RMSD_TIE:
      return True
    return False
  if primary == 'rmsd':
    if rmsd_new < rmsd_cur - _RMSD_TIE:
      return True
    if abs(rmsd_new - rmsd_cur) <= _RMSD_TIE and plddt_new > plddt_cur + _PDDT_TIE:
      return True
    return False
  raise ValueError(f"primary must be 'plddt' or 'rmsd', got {primary!r}")


def find_best_anchor_by_pair_silent(
    out_dir_a: str,
    out_dir_b: str,
    s_a: int,
    s_b: int,
    ovl: tuple[int, int],
    primary: str = 'plddt',
) -> int | None:
  f1 = alphafold_best_model_pdb(out_dir_a)
  f2 = alphafold_best_model_pdb(out_dir_b)
  if not f1 or not f2:
    return None
  lo, hi = ovl[0], ovl[1]
  if hi - lo < ANCHOR_SLIDE + 1:
    return None
  na, nb = 'pqa_anch', 'pqb_anch'
  for nm in (na, nb):
    if nm in (cmd.get_names('objects') or []):
      try:
        cmd.delete(nm)
      except Exception:
        pass
  best_r: int | None = None
  best_p = 0.0
  best_rmsd = float('inf')
  try:
    cmd.load(f1, na)
    cmd.load(f2, nb)
    cmd.alter(na, f'resv += {s_a}')
    cmd.alter(nb, f'resv += {s_b}')
    cmd.sort()
    for r in range(lo, hi - ANCHOR_SLIDE):
      g0 = s_a + r
      ca = f'resi {g0}-{g0+ANCHOR_SLIDE} and name CA'
      m1 = cmd.get_model(f'{na} and {ca}')
      if not m1.atom:
        continue
      plddt = sum(a.b for a in m1.atom) / len(m1.atom)
      try:
        sr = cmd.super(f'{nb} and {ca}', f'{na} and {ca}', cycles=0)
      except Exception:
        continue
      if not sr[1] or int(sr[1]) == 0:
        continue
      rmsd = float(sr[0])
      if best_r is None:
        best_r, best_p, best_rmsd = r, plddt, rmsd
      elif _anchor_window_is_better(plddt, rmsd, best_p, best_rmsd, primary):
        best_r, best_p, best_rmsd = r, plddt, rmsd
  finally:
    for nm in (na, nb):
      if nm in (cmd.get_names('objects') or []):
        try:
          cmd.delete(nm)
        except Exception:
          pass
  return best_r


def validate_overlap_pair(
    out_dir_a: str,
    out_dir_b: str,
    s_a: int,
    e_a: int,
    s_b: int,
    e_b: int,
    pair_index: int,
    anchor_primary: str = 'plddt',
) -> None:
  f1 = alphafold_best_model_pdb(out_dir_a)
  f2 = alphafold_best_model_pdb(out_dir_b)
  if not f1 or not f2:
    print(
        f'--- Pair {pair_index - 1} vs {pair_index}: '
        f'missing ranked_*.pdb under prediction dir ---'
    )
    return

  ovl_local = _chunk_overlap_local_in_prev(s_a, e_a, s_b, e_b)
  pbe, g_anchor0 = None, None
  if ovl_local is not None:
    pbe = find_best_anchor_by_pair_silent(
        out_dir_a, out_dir_b, s_a, s_b, ovl_local, primary=anchor_primary
    )
    if pbe is not None:
      g_anchor0 = s_a + pbe

  has_overlap_1b = s_b < e_a
  o1, o2 = s_b + 1, e_a
  ovl_lbl = (
      f'overlap 1-based {o1}–{o2}'
      if has_overlap_1b and o1 <= o2
      else 'abutting (no overlap window)'
  )

  cmd.reinitialize()
  cmd.load(f1, 'frag1')
  cmd.load(f2, 'frag2')
  cmd.alter('frag1', f'resv += {s_a}')
  cmd.alter('frag2', f'resv += {s_b}')
  cmd.sort()
  other = 'rmsd' if anchor_primary == 'plddt' else 'plddt'
  print(
      f'--- Validation: part {pair_index - 1} vs {pair_index} ({ovl_lbl}) | '
      f'anchor: primary {anchor_primary}, secondary {other} ---'
  )

  anchor_plddt = 0.0
  if has_overlap_1b and g_anchor0 is not None:
    an = f'resi {g_anchor0}-{g_anchor0 + ANCHOR_SLIDE} and name CA'
    m_anchor1 = cmd.get_model(f'frag1 and {an}')
    if m_anchor1.atom:
      anchor_plddt = sum((a.b for a in m_anchor1.atom)) / len(m_anchor1.atom)

  overlap_plddt = 0.0
  ov_ca = f'resi {o1}-{o2} and name CA' if (has_overlap_1b and o1 <= o2) else 'none'
  if has_overlap_1b and o1 <= o2:
    m_ov1 = cmd.get_model(f'frag1 and {ov_ca}')
    if m_ov1.atom:
      overlap_plddt = sum(a.b for a in m_ov1.atom) / len(m_ov1.atom)

  if has_overlap_1b and o1 <= o2 and g_anchor0 is not None:
    an = f'resi {g_anchor0}-{g_anchor0 + ANCHOR_SLIDE} and name CA'
    super_res = cmd.super(f'frag2 and {an}', f'frag1 and {an}', cycles=0)
  elif has_overlap_1b and o1 <= o2 and ovl_local is not None:
    super_res = cmd.super(f'frag2 and {ov_ca}', f'frag1 and {ov_ca}', cycles=0)
  else:
    w = min(JUNCTION_ALIGN_W, e_b - s_b, e_a - s_a)
    w = max(1, w)
    super_res = cmd.super(
        f'frag2 and resi {s_b+1}-{s_b+w} and name CA',
        f'frag1 and resi {e_a - w+1}-{e_a} and name CA',
    )

  rmsd, n_al = float(super_res[0]), int(super_res[1])
  print(f'  Best-anchor pLDDT (frag1, 51-mer, if any): {anchor_plddt:.2f} (target: >70)')
  print(f'  Full-overlap pLDDT: {overlap_plddt:.2f}')
  print(f'  RMSD on used selection: {rmsd:.3f} Å  ({n_al} CA)')
  if g_anchor0 is not None and anchor_plddt > 70.0 and rmsd < 2.0:
    print('  High-confidence anchor: OK for merge (per validate_and_analyze).')
  else:
    print('  Anchor thresholds not met or abutting; review if needed.')

  stored.scores = []
  if has_overlap_1b and o1 <= o2 and ov_ca != 'none':
    cmd.iterate(f'frag1 and {ov_ca}', 'stored.scores.append(b)')
  if stored.scores and has_overlap_1b:
    ap = sum(stored.scores) / len(stored.scores)
    print(f'  Avg pLDDT in overlap (iterate, frag1): {ap:.2f}')
  elif has_overlap_1b and o1 <= o2:
    print('  (No pLDDT in overlap; empty selection after renumbering?)')
  if rmsd < 2.0:
    print('  Consistency: PASS (RMSD < 2 Å) — validate.py')
  elif rmsd < 5.0:
    print('  Consistency: WARNING (2–5 Å) — flexible / stitch region?')
  else:
    print('  Consistency: FAIL (RMSD ≥5 Å) — re-check split / overlap')


def validate_all_adjacent_pairs(
    chunks: list[tuple[int, int]],
    out_dirs: list[str],
    anchor_primary: str = 'plddt',
) -> None:
  for i in range(1, len(chunks)):
    s_prev, e_prev = chunks[i - 1]
    s_i, e_i = chunks[i]
    validate_overlap_pair(
        out_dirs[i - 1],
        out_dirs[i],
        s_prev,
        e_prev,
        s_i,
        e_i,
        i,
        anchor_primary=anchor_primary,
    )


def _run_one_alphafold_chunk(
    i: int,
    fasta_path: str,
    output_dir_base: str,
    gpu_id: int,
    run_alphafold_extra: list[str],
) -> tuple[int, int]:
  env = os.environ.copy()
  env['HIP_VISIBLE_DEVICES'] = str(gpu_id)
  env.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE', 'false')
  env.setdefault('XLA_FLAGS', '--xla_gpu_autotune_level=0')

  run_py = str(REPO_ROOT / 'run_alphafold.py')
  cmd_list = [
      sys.executable,
      run_py,
      f'--fasta_paths={fasta_path}',
      f'--output_dir={output_dir_base}',
      *run_alphafold_extra,
  ]
  print(
      f'-> Chunk {i} on GCD {gpu_id}: {" ".join(cmd_list[:4])} ...'
  )
  p = subprocess.run(cmd_list, env=env, cwd=str(REPO_ROOT), check=False)
  return i, p.returncode


def run_parallel_af2(
    chunk_fastas: list[str],
    output_dir_base: str,
    *,
    run_alphafold_extra: list[str],
) -> None:
  n_gpus = max(1, len(GPU_IDS))
  n = len(chunk_fastas)

  def work(i: int, fp: str):
    gpu_id = GPU_IDS[i % n_gpus]
    return _run_one_alphafold_chunk(
        i, fp, output_dir_base, gpu_id, run_alphafold_extra
    )

  with ThreadPoolExecutor(max_workers=n_gpus) as ex:
    futs = [ex.submit(work, i, fp) for i, fp in enumerate(chunk_fastas)]
    for fut in as_completed(futs):
      i, rc = fut.result()
      if rc != 0:
        print(
            f'Warning: run_alphafold.py chunk {i} exited {rc} '
            f'({os.path.basename(chunk_fastas[i])!r}).',
            file=sys.stderr,
        )
  print('-> All run_alphafold.py jobs finished.')


def stitch_results(
    chunks,
    out_dirs,
    final_name: str = 'final_stitched.pdb',
    anchor_primary: str = 'plddt',
) -> None:
  other = 'rmsd' if anchor_primary == 'plddt' else 'plddt'
  print(
      f'-> Structural stitching (anchor: primary {anchor_primary}, secondary {other})...'
  )
  cmd.reinitialize()
  master_obj = 'full_protein'

  for i, (s, e) in enumerate(chunks):
    pdb_path = alphafold_best_model_pdb(out_dirs[i])
    if not pdb_path:
      print(f'Warning: No ranked PDB for chunk {i} in {out_dirs[i]}')
      continue

    chunk_obj = f'c{i}'
    cmd.load(pdb_path, chunk_obj)
    cmd.alter(chunk_obj, f'resv += {s}')
    cmd.sort()

    if i == 0:
      cmd.create(master_obj, chunk_obj)
    else:
      s_prev, e_prev = chunks[i - 1]
      last_global = e_prev
      if s < e_prev:
        ovl = _chunk_overlap_local_in_prev(s_prev, e_prev, s, e)
        pbe = (
            find_best_anchor_by_pair_silent(
                out_dirs[i - 1],
                out_dirs[i],
                s_prev,
                s,
                ovl,
                primary=anchor_primary,
            )
            if ovl
            else None
        )
        g0 = s_prev + pbe if pbe is not None else None
        ov_start, ov_end = s + 1, last_global
        if g0 is not None:
          s_sel = f'resi {g0}-{g0+ANCHOR_SLIDE} and name CA'
          cmd.super(f'{chunk_obj} and {s_sel}', f'{master_obj} and {s_sel}')
        else:
          sel = f'resi {ov_start}-{ov_end}'
          cmd.align(f'{chunk_obj} and {sel}', f'{master_obj} and {sel}')
        new_residues = f'resi {ov_end + 1}-{e}'
      else:
        w = min(JUNCTION_ALIGN_W, e_prev, e - s)
        w = max(1, w)
        lo_m = e_prev - w + 1
        lo_c = s + 1
        cmd.super(
            f'{chunk_obj} and resi {lo_c}-{lo_c + w - 1}',
            f'{master_obj} and resi {lo_m}-{lo_m + w - 1}',
        )
        new_residues = f'resi {s+1}-{e}'
      cmd.create(master_obj, f'{master_obj} or ({chunk_obj} and {new_residues})')

    cmd.delete(chunk_obj)

  names = cmd.get_names('objects') or []
  if master_obj not in names:
    print(
        f'ERROR: No structures loaded; cannot write {final_name!r}.',
        file=sys.stderr,
    )
    return
  cmd.save(final_name, master_obj)
  print(f'-> SUCCESS: {final_name} generated.')


def print_stitch_modes_summary(base: str, modes_done: list[str]) -> None:
  paths = {m: f'{base}_stitched_{m}.pdb' for m in modes_done}
  existing = {m: p for m, p in paths.items() if os.path.isfile(p)}
  if not existing:
    return
  print('--- Stitched model summary ---')
  print('  Modes: ' + ', '.join(sorted(existing)))
  print('  Files: ' + ', '.join(f'{k}={v}' for k, v in sorted(existing.items())))
  for _m, p in existing.items():
    print(f'  -> {p}')


def _strip_a3m_hash_comments(text: str) -> str:
  kept = []
  for line in text.splitlines():
    if line.strip().startswith('#'):
      continue
    kept.append(line)
  return '\n'.join(kept)


def write_query_fasta_from_a3m(a3m_path: str, fasta_path: str) -> None:
  """Write a one-sequence FASTA for ``run_alphafold.py`` (expects FASTA, not A3M paths).

  Uses the first sequence from the A3M (aligned, gaps stripped per AlphaFold parse_a3m).
  """
  with open(a3m_path, encoding='utf-8') as f:
    raw = f.read()
  msa = af_parsers.parse_a3m(_strip_a3m_hash_comments(raw))
  if not msa.sequences:
    raise ValueError(f'No sequences in {a3m_path!r}')
  seq = msa.sequences[0]
  desc = (msa.descriptions[0].split()[0] if msa.descriptions else 'query').strip()
  parent = os.path.dirname(os.path.abspath(fasta_path))
  if parent:
    os.makedirs(parent, exist_ok=True)
  with open(fasta_path, 'w', encoding='utf-8') as out:
    out.write(f'>{desc}\n{seq}\n')


def prepare_chunk_msas_from_a3m(
    chunk_a3m_path: str,
    output_dir_base: str,
    chunk_fasta_stem: str,
) -> None:
  """writes .../{stem}/msas/*.sto via convert_colabfold_a3m_to_sto.py"""
  msa_dir = os.path.join(output_dir_base, chunk_fasta_stem, 'msas')
  os.makedirs(msa_dir, exist_ok=True)
  conv = REPO_ROOT / 'convert_colabfold_a3m_to_sto.py'
  subprocess.run(
      [sys.executable, str(conv), chunk_a3m_path, msa_dir],
      cwd=str(REPO_ROOT),
      check=True,
  )


def build_default_af2_args(data_dir: str) -> list[str]:
  """Same defaults as run_af2.sh (reduced_dbs, monomer)."""
  return [
      '--model_preset=monomer',
      '--db_preset=reduced_dbs',
      '--max_template_date=1900-01-01',
      '--use_gpu_relax=false',
      f'--data_dir={data_dir}',
      f'--uniref90_database_path={data_dir}/uniref90/uniref90.fasta',
      f'--mgnify_database_path={data_dir}/mgnify/mgy_clusters_2022_05.fa',
      f'--small_bfd_database_path={data_dir}/small_bfd/bfd-first_non_consensus_sequences.fasta',
      f'--pdb70_database_path={data_dir}/pdb70/pdb70',
      f'--template_mmcif_dir={data_dir}/pdb_mmcif/mmcif_files',
      f'--obsolete_pdbs_path={data_dir}/pdb_mmcif/obsolete.dat',
      '--use_precomputed_msas=true',
  ]


def main() -> None:
  p = argparse.ArgumentParser(
      description='Split input, run AlphaFold2 per overlapping segment, stitch (PyMOL).',
      formatter_class=argparse.RawDescriptionHelpFormatter,
      epilog=(
          'Pass extra run_alphafold.py flags after a lone -- . '
          'If you omit them, defaults match run_af2.sh (reduced_dbs + precomputed MSAs).'
      ),
  )
  p.add_argument(
      'input',
      help='Input FASTA or A2M/A3M path.',
  )
  p.add_argument(
      '--output-dir-base',
      default=os.environ.get('AF2_CHUNK_OUTPUT_BASE', '/work/af2_chunk_runs'),
      help='AlphaFold --output_dir base (each chunk creates a subdirectory by FASTA stem).',
  )
  p.add_argument(
      '--data-dir',
      default=os.environ.get('ALPHAFOLD_DATA_DIR', '/work/databases'),
      help='--data_dir for databases (params, pdb70, mmcif, ...).',
  )
  p.add_argument(
      '--chunk-work-dir',
      default=None,
      help='Where to write per-chunk FASTA/A3M files (default: <output-dir-base>/_chunks).',
  )
  p.add_argument(
      '--max-chunk-aa',
      type=int,
      default=None,
      metavar='N',
      help='Max residues per segment (default: 3012).',
  )
  p.add_argument(
      '--plan-mode',
      choices=('default', 'balanced'),
      default='default',
      help=(
          'tiling policy: default uses fixed ~3000 aa windows; balanced shrinks the first '
          'window when one segment would dominate the stitched model (e.g. 3013 aa).'
      ),
  )
  p.add_argument(
      '--skip-alphafold',
      action='store_true',
      help='Only plan chunks / write inputs; do not run AlphaFold (for debugging).',
  )
  p.add_argument(
      '--stitch-modes',
      choices=('both', 'plddt', 'rmsd'),
      default='both',
      help='Anchor policy for stitching when multiple segments.',
  )
  p.add_argument(
      '--validate-adjacent-segments',
      action='store_true',
      help='Run overlap validation between consecutive chunks before stitch.',
  )
  try:
    i = sys.argv.index('--', 1)
  except ValueError:
    run_alphafold_extra: list[str] | None = None
  else:
    run_alphafold_extra = sys.argv[i + 1 :]
    del sys.argv[i:]

  args = p.parse_args()
  seq_input = os.path.abspath(args.input)
  output_base = os.path.abspath(args.output_dir_base)
  data_dir = args.data_dir
  chunk_work = args.chunk_work_dir or os.path.join(output_base, '_chunks')
  os.makedirs(chunk_work, exist_ok=True)

  ext = _chunk_file_extension(seq_input)
  base = os.path.splitext(seq_input)[0]
  msa_in = _is_msa_file(seq_input)
  a3m_records: list[tuple[str, str]] | None = None
  a3m_n_match: int | None = None

  if msa_in:
    a3m_records = parse_a3m_file(seq_input)
    a3m_n_match = a3m_match_state_count_with_check(a3m_records)
    _header, seq = a3m_records[0]
    total_len = a3m_n_match
  else:
    _header, seq = read_sequence_input(seq_input)
    total_len = len(seq)

  mca = args.max_chunk_aa if args.max_chunk_aa is not None else MAX_CHUNK_AA
  if mca < ANCHOR_SLIDE * 2:
    raise SystemExit(f'--max-chunk-aa ({mca}) too small; use at least ~{ANCHOR_SLIDE * 2}.')

  chunks, tw, tov, mode_used = plan_tiling(
      total_len, max_chunk_aa=mca, plan_mode=args.plan_mode)
  validate_chunk_plan(
      chunks, total_len, max_chunk_aa=mca, min_adjacent_overlap=tov)
  print_tiling_plan(chunks, plan_mode=mode_used)
  print(
      f'  (tiling: max {mca} aa per segment, window {tw} aa, overlap {tov} aa, '
      f'plan mode {mode_used})')

  one_segment = len(chunks) == 1

  chunk_fastas: list[str] = []
  pred_dirs: list[str] = []

  if run_alphafold_extra is None:
    run_alphafold_extra = build_default_af2_args(data_dir)

  if one_segment:
    stem_single = pathlib.Path(seq_input).stem
    if msa_in and a3m_records is not None:
      prepare_chunk_msas_from_a3m(seq_input, output_base, stem_single)
      fasta_single = os.path.join(chunk_work, f'{stem_single}.fasta')
      write_query_fasta_from_a3m(seq_input, fasta_single)
      chunk_fastas = [fasta_single]
      pred_dirs = [alphafold_prediction_dir(output_base, fasta_single)]
    else:
      chunk_fastas = [seq_input]
      pred_dirs = [alphafold_prediction_dir(output_base, seq_input)]
  else:
    for i, (s, e) in enumerate(chunks):
      stem = chunk_stem(os.path.basename(base), i, s, e)
      fn = os.path.join(chunk_work, f'{stem}{ext}')
      if msa_in and a3m_records is not None and a3m_n_match is not None:
        write_a3m_match_slice(a3m_records, s, e, a3m_n_match, fn, part_index=i)
        prepare_chunk_msas_from_a3m(fn, output_base, pathlib.Path(fn).stem)
        fasta_fn = os.path.join(chunk_work, f'{stem}.fasta')
        write_query_fasta_from_a3m(fn, fasta_fn)
        chunk_fastas.append(fasta_fn)
        pred_dirs.append(alphafold_prediction_dir(output_base, fasta_fn))
      else:
        with open(fn, 'w') as out:
          out.write(f'{_header}_p{i}\n{seq[s:e]}\n')
        chunk_fastas.append(fn)
        pred_dirs.append(alphafold_prediction_dir(output_base, fn))

  if args.skip_alphafold:
    print('-> --skip-alphafold: wrote chunk inputs / MSAs; exiting.')
    return

  if not one_segment:
    run_parallel_af2(chunk_fastas, output_base, run_alphafold_extra=run_alphafold_extra)
    if not all_chunk_folds_done_af2(pred_dirs):
      raise SystemExit(
          'AlphaFold did not produce ranked_*.pdb in every chunk directory. '
          'See errors above.'
      )

  if one_segment:
    env = os.environ.copy()
    env['HIP_VISIBLE_DEVICES'] = str(GPU_IDS[0])
    env.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE', 'false')
    env.setdefault('XLA_FLAGS', '--xla_gpu_autotune_level=0')
    print(f'-> Single segment: prediction dir will be {pred_dirs[0]!r}')
    rc = subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / 'run_alphafold.py'),
            f'--fasta_paths={chunk_fastas[0]}',
            f'--output_dir={output_base}',
            *run_alphafold_extra,
        ],
        cwd=str(REPO_ROOT),
        env=env,
        check=False,
    ).returncode
    if rc != 0:
      raise SystemExit(rc)
    return

  if args.stitch_modes == 'both':
    mode_list = ['plddt', 'rmsd']
  else:
    mode_list = [args.stitch_modes]

  if args.validate_adjacent_segments:
    for mo in mode_list:
      other = 'rmsd' if mo == 'plddt' else 'plddt'
      print(f'\n### Pre-stitch validation (anchor: primary {mo}, secondary {other}) ###\n')
      validate_all_adjacent_pairs(chunks, pred_dirs, anchor_primary=mo)

  for mo in mode_list:
    out_pdb = f'{base}_stitched_{mo}.pdb'
    print(f'\n### Stitch: anchor policy {mo} -> {out_pdb} ###\n')
    stitch_results(chunks, pred_dirs, final_name=out_pdb, anchor_primary=mo)

  print_stitch_modes_summary(base, mode_list)


if __name__ == '__main__':
  main()
