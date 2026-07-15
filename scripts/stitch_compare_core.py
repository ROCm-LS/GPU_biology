"""Shared structure comparison helpers for split-fold-stitch validation."""

from __future__ import annotations

import glob
import json
import os
import re
from dataclasses import dataclass, field
from typing import Literal

from pymol import cmd, stored

ReferenceBackend = Literal['colabfold', 'alphafold2']
CandidateBackend = Literal['colabfold', 'alphafold2']


@dataclass(frozen=True)
class TargetCase:
    name: str
    length: int
    input_file: str
    segments: str
    overlap_desc: str
    overlap_windows: tuple[tuple[int, int], ...]
    ref_chain: str | None = None
    ref_note: str | None = None
    colabfold_ref_glob: str | None = None
    colabfold_ref_log: str | None = None
    af2_subdir: str | None = None


TARGET_CASES: tuple[TargetCase, ...] = (
    TargetCase(
        name='3013aa',
        length=3013,
        input_file='3013aa.fasta',
        segments='2 segments: 1–3000, 2001–3013',
        overlap_desc='1000 aa (resi 2001–3000)',
        overlap_windows=((2001, 3000),),
        colabfold_ref_glob='3013aa_output/*_rank_001*.pdb',
        colabfold_ref_log='3013aa.log',
        af2_subdir='3013aa',
    ),
    TargetCase(
        name='5005aa',
        length=5005,
        input_file='5005aa.fasta',
        segments='2 segments: 1–3000, 2001–5005',
        overlap_desc='1000 aa (resi 2001–3000)',
        overlap_windows=((2001, 3000),),
        colabfold_ref_glob='5005aa_output/*_rank_001*.pdb',
        colabfold_ref_log='5005aa.log',
        af2_subdir='5005aa',
    ),
    TargetCase(
        name='1IH7.7',
        length=903,
        input_file='1IH7.7.a3m',
        segments='5 segments: 300 aa windows, step 200 aa',
        overlap_desc='100 aa per junction (resi 201–300, 401–500, 601–700, 801–900)',
        overlap_windows=((201, 300), (401, 500), (601, 700), (801, 900)),
        colabfold_ref_glob='1IH7.7_output/*_rank_001*.pdb',
        colabfold_ref_log='1IH72.log',
        ref_chain='A',
        ref_note=(
            'ColabFold reference is a 7-chain multimer (7 × 903 aa); comparison uses '
            'chain A only. AlphaFold2 reference is a single-chain monomer (ranked_0). '
            'Split-fold segments used AF2 monomer (ptm).'
        ),
        af2_subdir='1IH7',
    ),
)


@dataclass
class OverlapMetrics:
    lo: int
    hi: int
    rmsd: float
    pairs: int
    span: int
    pair_frac: float


@dataclass
class ComparisonMetrics:
    ref_path: str
    cand_path: str
    length: int
    n_ref: int
    n_cand: int
    global_rmsd: float
    global_pairs: int
    global_pair_frac: float
    ref_plddt: float
    cand_plddt: float
    plddt_delta: float
    overlaps: list[OverlapMetrics] = field(default_factory=list)


@dataclass
class CandidateWorkDir:
    label: CandidateBackend
    path: str
    stitch_variants: str = 'both'  # plddt | rmsd | both


def case_by_name(name: str) -> TargetCase:
    for case in TARGET_CASES:
        if case.name == name:
            return case
    known = ', '.join(c.name for c in TARGET_CASES)
    raise SystemExit(f'Unknown target {name!r}; expected one of: {known}')


def _resolve_one(root: str, pattern: str) -> str:
    matches = sorted(glob.glob(os.path.join(root, pattern)))
    if len(matches) == 1:
        return matches[0]
    if not matches:
        raise FileNotFoundError(f'No match for {os.path.join(root, pattern)!r}')
    raise FileNotFoundError(
        f'Multiple matches for {pattern!r}; pass an explicit path:\n'
        + '\n'.join(f'  {m}' for m in matches)
    )


def resolve_colabfold_reference(reference_root: str, case: TargetCase) -> str:
    if not case.colabfold_ref_glob:
        raise ValueError(f'No ColabFold reference configured for {case.name}')
    return _resolve_one(reference_root, case.colabfold_ref_glob)


def resolve_af2_reference(af2_output_root: str, case: TargetCase, rank: int = 0) -> str:
    if not case.af2_subdir:
        raise ValueError(f'No AlphaFold2 reference configured for {case.name}')
    path = os.path.join(af2_output_root, case.af2_subdir, f'ranked_{rank}.pdb')
    if not os.path.isfile(path):
        raise FileNotFoundError(path)
    return path


def resolve_reference(
    *,
    backend: ReferenceBackend,
    case: TargetCase,
    reference_root: str | None = None,
    af2_output_root: str | None = None,
    explicit_path: str | None = None,
    rank: int = 0,
) -> str:
    if explicit_path:
        if not os.path.isfile(explicit_path):
            raise FileNotFoundError(explicit_path)
        return explicit_path
    if backend == 'colabfold':
        if not reference_root:
            raise ValueError('reference_root is required for ColabFold references')
        return resolve_colabfold_reference(reference_root, case)
    if backend == 'alphafold2':
        if not af2_output_root:
            raise ValueError('af2_output_root is required for AlphaFold2 references')
        return resolve_af2_reference(af2_output_root, case, rank=rank)
    raise ValueError(f'Unknown reference backend {backend!r}')


def stitched_candidate_path(work_dir: str, case: TargetCase, *, suffix: str = 'plddt') -> str:
    return os.path.join(work_dir, f'{case.name}_stitched_{suffix}.pdb')


def stitch_suffixes(mode: str = 'both') -> tuple[str, ...]:
    mode = mode.lower()
    if mode == 'both':
        return ('plddt', 'rmsd')
    if mode in ('plddt', 'rmsd'):
        return (mode,)
    raise ValueError(f"stitch variant must be 'plddt', 'rmsd', or 'both', got {mode!r}")


def find_stitched_candidates(
    work_dir: str,
    case: TargetCase,
    *,
    variants: str = 'both',
) -> list[tuple[str, str]]:
    """Return (suffix, path) for each stitched PDB that exists."""
    found: list[tuple[str, str]] = []
    for suffix in stitch_suffixes(variants):
        path = stitched_candidate_path(work_dir, case, suffix=suffix)
        if os.path.isfile(path):
            found.append((suffix, path))
    return found


def find_stitched_candidate(work_dir: str, case: TargetCase, *, prefer: str = 'plddt') -> str | None:
    found = find_stitched_candidates(work_dir, case, variants='both')
    if not found:
        return None
    for suffix, path in found:
        if suffix == prefer:
            return path
    return found[0][1]


def _mean_bfactor(obj: str, selection: str) -> float:
    stored._bf = []
    cmd.iterate(f'{obj} and {selection} and name CA', 'stored._bf.append(b)')
    if not stored._bf:
        raise RuntimeError(f'No Cα B-factors for {obj} selection {selection!r}')
    return sum(stored._bf) / len(stored._bf)


def _ca_count(obj: str, selection: str = 'all') -> int:
    stored._n = 0
    cmd.iterate(f'{obj} and {selection} and name CA', 'stored._n += 1')
    return int(stored._n)


def _rmsd_after_super(mobile: str, ref: str, selection: str = 'all') -> tuple[float, int]:
    out = cmd.super(
        f'{mobile} and {selection} and name CA',
        f'{ref} and {selection} and name CA',
        cycles=0,
    )
    if not out or not out[1]:
        raise RuntimeError(f'Superposition failed for selection {selection!r}')
    return float(out[0]), int(out[1])


def compare_structures(
    *,
    ref_path: str,
    cand_path: str,
    length: int,
    overlap_windows: tuple[tuple[int, int], ...],
    ref_chain: str | None = None,
) -> ComparisonMetrics:
    ref_obj = 'ref'
    ref_sel = ref_obj if not ref_chain else f'{ref_obj} and chain {ref_chain}'

    cmd.reinitialize()
    cmd.load(ref_path, ref_obj)
    cmd.load(cand_path, 'cand')

    n_ref = _ca_count(ref_sel)
    n_cand = _ca_count('cand')
    g_rmsd, g_pairs = _rmsd_after_super('cand', ref_sel, 'all')
    ref_plddt = _mean_bfactor(ref_sel, 'all')
    cand_plddt = _mean_bfactor('cand', 'all')

    overlaps: list[OverlapMetrics] = []
    for lo, hi in overlap_windows:
        sel = f'resi {lo}-{hi}'
        o_rmsd, o_pairs = _rmsd_after_super('cand', ref_sel, sel)
        span = hi - lo + 1
        overlaps.append(
            OverlapMetrics(
                lo=lo,
                hi=hi,
                rmsd=o_rmsd,
                pairs=o_pairs,
                span=span,
                pair_frac=o_pairs / span,
            )
        )

    return ComparisonMetrics(
        ref_path=ref_path,
        cand_path=cand_path,
        length=length,
        n_ref=n_ref,
        n_cand=n_cand,
        global_rmsd=g_rmsd,
        global_pairs=g_pairs,
        global_pair_frac=g_pairs / length,
        ref_plddt=ref_plddt,
        cand_plddt=cand_plddt,
        plddt_delta=cand_plddt - ref_plddt,
        overlaps=overlaps,
    )


def parse_colabfold_rank_metrics(log_path: str) -> tuple[float | None, float | None]:
    if not os.path.isfile(log_path):
        return None, None
    text = open(log_path, encoding='utf-8').read()
    hits = re.findall(r'rank_001[^\n]*pLDDT=([0-9.]+) pTM=([0-9.]+)', text)
    if not hits:
        return None, None
    return float(hits[-1][0]), float(hits[-1][1])


def parse_af2_rank_plddt(af2_dir: str, rank: int = 0) -> float | None:
    ranking_path = os.path.join(af2_dir, 'ranking_debug.json')
    if not os.path.isfile(ranking_path):
        return None
    data = json.load(open(ranking_path, encoding='utf-8'))
    order = data.get('order') or []
    plddts = data.get('plddts') or {}
    if not order:
        return None
    model_key = order[rank] if rank < len(order) else order[0]
    val = plddts.get(model_key)
    return float(val) if val is not None else None


def parse_stitch_policy_diff(log_path: str) -> float | None:
    if not os.path.isfile(log_path):
        return None
    text = open(log_path, encoding='utf-8').read()
    m = re.search(
        r'Cα RMSD \(after super\) between the two full stitched models: ([0-9.]+)',
        text,
    )
    return float(m.group(1)) if m else None


def parse_segment_rank_metrics(work_dir: str, name: str) -> list[tuple[float, float]]:
    scores: list[tuple[float, float]] = []
    for log_path in sorted(glob.glob(os.path.join(work_dir, f'{name}_part_*_output/log.txt'))):
        text = open(log_path, encoding='utf-8').read()
        hits = re.findall(r'rank_001[^\n]*pLDDT=([0-9.]+) pTM=([0-9.]+)', text)
        if hits:
            scores.append((float(hits[-1][0]), float(hits[-1][1])))
    return scores


def parse_af2_segment_rank_plddt(work_dir: str, name: str) -> list[float]:
    scores: list[float] = []
    for ranking_path in sorted(
        glob.glob(os.path.join(work_dir, 'af2_predictions', f'{name}_part_*', 'ranking_debug.json'))
    ):
        val = parse_af2_rank_plddt(os.path.dirname(ranking_path), rank=0)
        if val is not None:
            scores.append(val)
    return scores


def junction_label(lo: int, hi: int) -> str:
    return f'{lo}–{hi}'


def parse_candidate_dirs(
    specs: list[str],
    *,
    stitch_variants: str = 'both',
) -> list[CandidateWorkDir]:
    out: list[CandidateWorkDir] = []
    for spec in specs:
        if ':' not in spec:
            raise SystemExit(
                f'Invalid --candidate-dir {spec!r}; use LABEL:PATH (e.g. colabfold:/work/run)'
            )
        label, path = spec.split(':', 1)
        label = label.strip().lower()
        path = os.path.abspath(path.strip())
        if label not in ('colabfold', 'alphafold2'):
            raise SystemExit(
                f'Unknown candidate label {label!r}; expected colabfold or alphafold2'
            )
        out.append(
            CandidateWorkDir(label=label, path=path, stitch_variants=stitch_variants)
        )
    return out
