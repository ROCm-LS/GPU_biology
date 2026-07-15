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


_PART_STEM_RE = re.compile(
    r'^(.+)_part_(\d+)_(\d+)-(\d+)\.(?:fasta|fa|a3m|FASTA|FA|A3M)$',
    re.IGNORECASE,
)


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


@dataclass(frozen=True)
class RunTiling:
    """Segment plan discovered from a split-fold work directory."""

    segments: str
    overlap_desc: str
    overlap_windows: tuple[tuple[int, int], ...]
    chunks_1based: tuple[tuple[int, int], ...] = ()
    plan_mode: str | None = None
    source: str = 'default'  # part_files | plan_json | default


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
    label: str
    path: str
    backend: ReferenceBackend
    stitch_variants: str = 'both'  # plddt | rmsd | both


def infer_candidate_backend(label: str) -> ReferenceBackend:
    """Map a display label to ColabFold vs AlphaFold2 segment-score parsing."""
    low = label.strip().lower()
    if low == 'colabfold' or low.startswith('colabfold'):
        return 'colabfold'
    if low == 'alphafold2' or low.startswith('alphafold2') or low.startswith('af2'):
        return 'alphafold2'
    raise SystemExit(
        f'Cannot infer fold backend from candidate label {label!r}; '
        "use a label starting with 'colabfold' or 'alphafold2' (e.g. alphafold2-default)."
    )


def case_by_name(name: str) -> TargetCase:
    for case in TARGET_CASES:
        if case.name == name:
            return case
    known = ', '.join(c.name for c in TARGET_CASES)
    raise SystemExit(f'Unknown target {name!r}; expected one of: {known}')


def _chunks_half_open_from_1based(
    ranges_1based: list[tuple[int, int]],
) -> list[tuple[int, int]]:
    """Convert inclusive 1-based ``(start, end)`` pairs to half-open 0-based chunks."""
    return [(start - 1, end) for start, end in ranges_1based]


def overlap_windows_from_chunks(
    chunks: list[tuple[int, int]],
) -> tuple[tuple[int, int], ...]:
    """Adjacent overlap windows (1-based inclusive) from half-open chunk indices."""
    windows: list[tuple[int, int]] = []
    for i in range(1, len(chunks)):
        _s_prev, e_prev = chunks[i - 1]
        s_i, _e_i = chunks[i]
        if s_i < e_prev:
            windows.append((s_i + 1, e_prev))
    return tuple(windows)


def _format_segments_label(chunks_0: list[tuple[int, int]]) -> str:
    parts = [f'{s + 1}–{e}' for s, e in chunks_0]
    return f'{len(chunks_0)} segments: ' + ', '.join(parts)


def junction_label(lo: int, hi: int) -> str:
    return f'{lo}–{hi}'


def _format_overlap_desc(windows: tuple[tuple[int, int], ...]) -> str:
    if not windows:
        return 'n/a (single segment or abutting chunks)'
    spans = [hi - lo + 1 for lo, hi in windows]
    labels = ', '.join(f'resi {junction_label(lo, hi)}' for lo, hi in windows)
    if len(set(spans)) == 1:
        return f'{spans[0]} aa per junction ({labels})'
    detail = ', '.join(
        f'resi {junction_label(lo, hi)} ({hi - lo + 1} aa)' for lo, hi in windows
    )
    return f'junction overlaps: {detail}'


def _collect_part_range_options(
    work_dir: str,
    case_name: str,
) -> dict[int, set[tuple[int, int]]]:
    """Map part index to all ``(start, end)`` ranges seen in on-disk segment inputs."""
    options: dict[int, set[tuple[int, int]]] = {}
    pattern = os.path.join(work_dir, f'{case_name}_part_*')
    for path in sorted(glob.glob(pattern)):
        match = _PART_STEM_RE.match(os.path.basename(path))
        if not match or match.group(1) != case_name:
            continue
        part_index = int(match.group(2))
        start_1 = int(match.group(3))
        end_1 = int(match.group(4))
        options.setdefault(part_index, set()).add((start_1, end_1))
    return options


def _is_coherent_tiling(ranges_1based: list[tuple[int, int]], *, length: int) -> bool:
    if not ranges_1based or ranges_1based[0][0] != 1 or ranges_1based[-1][1] != length:
        return False
    for i in range(1, len(ranges_1based)):
        prev_s, prev_e = ranges_1based[i - 1]
        cur_s, _cur_e = ranges_1based[i]
        if cur_s <= prev_s or cur_s >= prev_e:
            return False
    return True


def enumerate_coherent_tilings(
    options: dict[int, set[tuple[int, int]]],
    *,
    length: int,
) -> list[list[tuple[int, int]]]:
    """All valid 1-based inclusive tilings implied by on-disk part filename options."""
    if not options:
        return []

    results: list[list[tuple[int, int]]] = []
    seen: set[tuple[tuple[int, int], ...]] = set()
    max_idx = max(options)

    def search(part_idx: int, current: list[tuple[int, int]]) -> None:
        if current and current[-1][1] == length:
            if _is_coherent_tiling(current, length=length):
                key = tuple(current)
                if key not in seen:
                    seen.add(key)
                    results.append(current[:])
            return
        if part_idx > max_idx:
            return
        if part_idx not in options:
            search(part_idx + 1, current)
            return
        for rng in sorted(options[part_idx]):
            if current:
                prev_s, prev_e = current[-1]
                if rng[0] <= prev_s or rng[0] >= prev_e:
                    continue
            search(part_idx + 1, current + [rng])

    search(0, [])
    return results


def discover_chunks_from_part_files(
    work_dir: str,
    case_name: str,
    *,
    length: int | None = None,
    plan_hint: str | None = None,
) -> list[tuple[int, int]] | None:
    """Read 0-based half-open chunks from ``{case}_part_{i}_{a}-{b}.*`` inputs."""
    options = _collect_part_range_options(work_dir, case_name)
    if not options:
        return None

    if length is None:
        case = case_by_name(case_name)
        length = case.length

    tilings = enumerate_coherent_tilings(options, length=length)
    if not tilings:
        return None
    if len(tilings) == 1:
        return _chunks_half_open_from_1based(tilings[0])

    if plan_hint == 'default':
        chosen = min(tilings, key=len)
    elif plan_hint == 'balanced':
        chosen = max(tilings, key=len)
    else:
        chosen = tilings[0]
    return _chunks_half_open_from_1based(chosen)


def infer_plan_hint(label: str) -> str | None:
    """Infer split-fold plan mode from a candidate display label."""
    low = label.strip().lower()
    if 'balanced' in low:
        return 'balanced'
    if 'default' in low:
        return 'default'
    return None


def _plan_matches_case(data: dict, case_name: str) -> bool:
    raw = data.get('base')
    if raw is None:
        return False
    stem = os.path.splitext(os.path.basename(str(raw)))[0]
    return stem == case_name


def load_plan_json_chunks(
    work_dir: str,
    case_name: str,
) -> tuple[list[tuple[int, int]], str | None] | None:
    """Load 0-based chunks from ``.split_fold_stitch/{case}_*.json`` plan files."""
    plan_dir = os.path.join(work_dir, '.split_fold_stitch')
    candidates = (
        f'{case_name}_stitch_plddt.json',
        f'{case_name}_stitch_rmsd.json',
        f'{case_name}_summary.json',
        # Legacy single-target filenames (last run only).
        'stitch_plddt.json',
        'stitch_rmsd.json',
        'summary.json',
    )
    for fname in candidates:
        path = os.path.join(plan_dir, fname)
        if not os.path.isfile(path):
            continue
        with open(path, encoding='utf-8') as fh:
            data = json.load(fh)
        if not _plan_matches_case(data, case_name):
            continue
        chunks = [tuple(pair) for pair in data['chunks']]
        if not chunks:
            continue
        return chunks, data.get('plan_mode')
    return None


def resolve_run_tiling(
    work_dir: str,
    case: TargetCase,
    *,
    plan_hint: str | None = None,
) -> RunTiling:
    """
    Discover tiling for a candidate work tree.

    Prefers on-disk part filenames, then a matching stitch plan JSON, then the
    static ``TargetCase`` defaults. When duplicate part indices disagree,
    ``plan_hint`` (``default`` / ``balanced``, often from the candidate label)
    selects the intended tiling.
    """
    work_dir = os.path.abspath(work_dir)
    chunks = discover_chunks_from_part_files(
        work_dir, case.name, length=case.length, plan_hint=plan_hint
    )
    plan_mode: str | None = None
    source = 'default'

    if chunks:
        source = 'part_files'
        plan = load_plan_json_chunks(work_dir, case.name)
        if plan is not None:
            plan_mode = plan[1]
    else:
        plan = load_plan_json_chunks(work_dir, case.name)
        if plan is not None:
            chunks, plan_mode = plan
            source = 'plan_json'

    if not chunks:
        return RunTiling(
            segments=case.segments,
            overlap_desc=case.overlap_desc,
            overlap_windows=case.overlap_windows,
            source='default',
        )

    windows = overlap_windows_from_chunks(chunks)
    chunks_1based = tuple((s + 1, e) for s, e in chunks)
    segments = _format_segments_label(chunks)
    overlap_desc = _format_overlap_desc(windows)
    if plan_mode and plan_mode != 'default':
        segments = f'{segments} (plan={plan_mode})'

    return RunTiling(
        segments=segments,
        overlap_desc=overlap_desc,
        overlap_windows=windows,
        chunks_1based=chunks_1based,
        plan_mode=plan_mode,
        source=source,
    )


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
        label = label.strip()
        path = os.path.abspath(path.strip())
        backend = infer_candidate_backend(label)
        out.append(
            CandidateWorkDir(
                label=label,
                path=path,
                backend=backend,
                stitch_variants=stitch_variants,
            )
        )
    return out
