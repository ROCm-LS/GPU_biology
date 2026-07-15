#!/usr/bin/env python3
"""Generate split-fold-stitch validation reports vs full-length references.

Compares stitched outputs from MI250X/MI300X (OOM workaround) against monolithic
full-length folds. Supports:

- **ColabFold** references (``--reference-root``)
- **AlphaFold2** references (``--af2-output-root``, default ``/home/sajandhy/af2_output``)
- Multiple stitched candidate trees via ``--candidate-dir LABEL:PATH``

Requires PyMOL (``pip install pymol-open-source`` or container).

Example (ColabFold + AlphaFold2 references, ColabFold + AF2 split-stitch candidates)::

  python3 scripts/generate_split_stitch_validation_report.py \\
    --reference-root /home/sajandhy/colabfold_work \\
    --af2-output-root /home/sajandhy/af2_output \\
    --candidate-dir colabfold:/path/to/colabfold_work_rocm7.2.3 \\
    --candidate-dir alphafold2:/path/to/alphafold2_work_rocm7.2.3 \\
    --output reports/split_stitch_validation_report.md
"""

from __future__ import annotations

import argparse
import datetime as dt
import os
import sys

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)

from stitch_compare_core import (
    TARGET_CASES,
    CandidateWorkDir,
    ComparisonMetrics,
    RunTiling,
    TargetCase,
    compare_structures,
    find_stitched_candidates,
    junction_label,
    parse_af2_rank_plddt,
    parse_af2_segment_rank_plddt,
    parse_colabfold_rank_metrics,
    parse_segment_rank_metrics,
    parse_stitch_policy_diff,
    parse_candidate_dirs,
    resolve_af2_reference,
    resolve_colabfold_reference,
    resolve_run_tiling,
    infer_plan_hint,
)


def _fmt(x: float | None, nd: int = 2) -> str:
    if x is None:
        return 'n/a'
    return f'{x:.{nd}f}'


def _candidate_run_log(work_dir: str, case_name: str, candidate: CandidateWorkDir) -> str | None:
    if candidate.backend == 'colabfold':
        path = os.path.join(work_dir, f'{case_name}.colabfold.rocm7.2.3.log')
        return path if os.path.isfile(path) else None
    return None


def _collect_case_result(
    *,
    case: TargetCase,
    candidate: CandidateWorkDir,
    ref_backend: str,
    ref_path: str,
    ref_rank_plddt: float | None,
    ref_rank_ptm: float | None,
    metrics: ComparisonMetrics,
    stitch_diff: float | None,
    seg_scores_txt: str | None,
    stitch_suffix: str,
    run_tiling: RunTiling,
) -> dict:
    return {
        'case': case,
        'candidate_label': candidate.label,
        'stitch_suffix': stitch_suffix,
        'candidate_dir': candidate.path,
        'ref_backend': ref_backend,
        'ref_path': ref_path,
        'ref_rank_plddt': ref_rank_plddt,
        'ref_rank_ptm': ref_rank_ptm,
        'metrics': metrics,
        'stitch_diff': stitch_diff,
        'seg_scores_txt': seg_scores_txt,
        'run_tiling': run_tiling,
    }


def _segment_scores_text(case: TargetCase, candidate: CandidateWorkDir) -> str | None:
    if candidate.backend == 'colabfold':
        scores = parse_segment_rank_metrics(candidate.path, case.name)
        if not scores:
            return None
        return ', '.join(
            f'seg{i}: pLDDT={p:.1f} pTM={t:.3f}' for i, (p, t) in enumerate(scores)
        )
    scores = parse_af2_segment_rank_plddt(candidate.path, case.name)
    if not scores:
        return None
    return ', '.join(f'seg{i}: pLDDT={p:.1f}' for i, p in enumerate(scores))


def _gather_results(
    *,
    reference_root: str | None,
    af2_output_root: str | None,
    candidates: list[CandidateWorkDir],
) -> list[dict]:
    rows: list[dict] = []
    ref_backends: list[tuple[str, str | None]] = []
    if reference_root:
        ref_backends.append(('colabfold', reference_root))
    if af2_output_root:
        ref_backends.append(('alphafold2', af2_output_root))
    if not ref_backends:
        raise SystemExit('Pass at least one of --reference-root or --af2-output-root.')

    for case in TARGET_CASES:
        for cand in candidates:
            stitched = find_stitched_candidates(
                cand.path, case, variants=cand.stitch_variants
            )
            if not stitched:
                continue
            run_log = _candidate_run_log(cand.path, case.name, cand)
            stitch_diff = parse_stitch_policy_diff(run_log) if run_log else None
            seg_scores_txt = _segment_scores_text(case, cand)
            run_tiling = resolve_run_tiling(
                cand.path, case, plan_hint=infer_plan_hint(cand.label)
            )

            for stitch_suffix, cand_path in stitched:
                for ref_backend, ref_root in ref_backends:
                    try:
                        if ref_backend == 'colabfold':
                            ref_path = resolve_colabfold_reference(ref_root, case)
                            ref_log = os.path.join(ref_root, case.colabfold_ref_log or '')
                            ref_rank_plddt, ref_rank_ptm = parse_colabfold_rank_metrics(ref_log)
                        else:
                            ref_path = resolve_af2_reference(ref_root, case)
                            ref_rank_plddt = parse_af2_rank_plddt(
                                os.path.join(ref_root, case.af2_subdir or case.name),
                                rank=0,
                            )
                            ref_rank_ptm = None
                    except (FileNotFoundError, ValueError):
                        continue

                    metrics = compare_structures(
                        ref_path=ref_path,
                        cand_path=cand_path,
                        length=case.length,
                        overlap_windows=run_tiling.overlap_windows,
                        ref_chain=case.ref_chain if ref_backend == 'colabfold' else None,
                    )
                    rows.append(
                        _collect_case_result(
                            case=case,
                            candidate=cand,
                            stitch_suffix=stitch_suffix,
                            ref_backend=ref_backend,
                            ref_path=ref_path,
                            ref_rank_plddt=ref_rank_plddt,
                            ref_rank_ptm=ref_rank_ptm,
                            metrics=metrics,
                            stitch_diff=stitch_diff,
                            seg_scores_txt=seg_scores_txt,
                            run_tiling=run_tiling,
                        )
                    )
    if not rows:
        raise SystemExit('No comparisons produced; check candidate and reference paths.')
    return rows


def _summary_table(rows: list[dict]) -> list[str]:
    lines = [
        (
            '| Target | Candidate | Reference | Global RMSD (Å) | Paired Cα | '
            'Ref pLDDT | Stitched pLDDT | Δ pLDDT | Stitch Δ (Å) |'
        ),
        '|--------|-----------|-----------|-----------------|-----------|'
        '-----------|------------------|---------|--------------|',
    ]
    for row in rows:
        m = row['metrics']
        case = row['case']
        lines.append(
            f'| {case.name} | {row["candidate_label"]}/{row["stitch_suffix"]} | {row["ref_backend"]} | '
            f'{m.global_rmsd:.2f} | {m.global_pairs}/{case.length} '
            f'({100 * m.global_pair_frac:.0f}%) | '
            f'{m.ref_plddt:.1f} | {m.cand_plddt:.1f} | {m.plddt_delta:+.1f} | '
            f'{_fmt(row["stitch_diff"])} |'
        )
    return lines


def _detail_section(row: dict) -> list[str]:
    case = row['case']
    m = row['metrics']
    tiling: RunTiling = row['run_tiling']
    lines = [
        (
            f'### {case.name} — {row["candidate_label"]}/{row["stitch_suffix"]} stitched vs '
            f'{row["ref_backend"]} full-length'
        ),
        '',
        f'- **Input:** `{case.input_file}`',
        f'- **Tiling:** {tiling.segments}',
        f'- **Overlap:** {tiling.overlap_desc}',
    ]
    if tiling.source != 'default':
        lines.append(f'- **Tiling source:** `{tiling.source}` (from `{row["candidate_dir"]}`)')
    lines.extend(
        [
            f'- **Candidate:** `{m.cand_path}`',
            f'- **Reference:** `{row["ref_path"]}`',
        ]
    )
    if case.ref_note and row['ref_backend'] == 'colabfold':
        lines.append(f'- **Note:** {case.ref_note}')
    if row['ref_rank_plddt'] is not None:
        if row['ref_rank_ptm'] is not None:
            lines.append(
                f'- **Reference rank-001 score:** pLDDT={row["ref_rank_plddt"]:.1f}, '
                f'pTM={row["ref_rank_ptm"]:.3f}'
            )
        else:
            lines.append(
                f'- **Reference ranked_0 score:** pLDDT={row["ref_rank_plddt"]:.1f}'
            )
    if row['seg_scores_txt']:
        lines.append(f'- **Split-fold segment scores:** {row["seg_scores_txt"]}')
    if row['stitch_diff'] is not None:
        lines.append(
            f'- **Stitch-policy consistency (plddt vs rmsd):** {row["stitch_diff"]:.3f} Å'
        )

    lines.extend(
        [
            '',
            '| Metric | Value |',
            '|--------|-------|',
            f'| Cα count (ref / stitched) | {m.n_ref} / {m.n_cand} |',
            (
                f'| Global Cα RMSD | {m.global_rmsd:.3f} Å ({m.global_pairs} pairs, '
                f'{100 * m.global_pair_frac:.1f}% of length) |'
            ),
            (
                f'| Mean pLDDT (Cα B-factor) | ref={m.ref_plddt:.2f}, '
                f'stitched={m.cand_plddt:.2f} (Δ={m.plddt_delta:+.2f}) |'
            ),
            '',
            '**Overlap / junction RMSD vs reference:**',
            '',
            '| Window (resi) | RMSD (Å) | Pairs | Coverage |',
            '|---------------|----------|-------|----------|',
        ]
    )
    for ov in m.overlaps:
        lines.append(
            f'| {junction_label(ov.lo, ov.hi)} | {ov.rmsd:.3f} | {ov.pairs} | '
            f'{100 * ov.pair_frac:.0f}% of {ov.span} |'
        )
    lines.append('')
    return lines


def _interpretation(rows: list[dict]) -> list[str]:
    def _find(name: str, cand: str, ref: str) -> dict | None:
        for row in rows:
            if (
                row['case'].name == name
                and row['candidate_label'] == cand
                and row['ref_backend'] == ref
            ):
                return row
        return None

    bullets: list[str] = []
    ih = _find('1IH7.7', 'colabfold', 'alphafold2')
    if ih:
        m = ih['metrics']
        bullets.append(
            f'- **1IH7.7:** Global RMSD {m.global_rmsd:.1f} Å with '
            f'{100 * m.global_pair_frac:.0f}% Cα pairing vs AlphaFold2; junction windows '
            f'≤ {max(ov.rmsd for ov in m.overlaps):.1f} Å. Strong agreement for this target.'
        )
    aa3013_cf = _find('3013aa', 'colabfold', 'colabfold')
    aa3013_af2 = _find('3013aa', 'colabfold', 'alphafold2')
    if aa3013_cf:
        m = aa3013_cf['metrics']
        bullets.append(
            f'- **3013aa vs ColabFold reference:** RMSD {m.global_rmsd:.1f} Å, '
            f'pLDDT Δ {m.plddt_delta:+.1f}.'
        )
    if aa3013_af2:
        m = aa3013_af2['metrics']
        bullets.append(
            f'- **3013aa vs AlphaFold2 reference:** RMSD {m.global_rmsd:.1f} Å, '
            f'pLDDT Δ {m.plddt_delta:+.1f}.'
        )
    aa5005 = _find('5005aa', 'colabfold', 'alphafold2')
    if aa5005:
        m = aa5005['metrics']
        bullets.append(
            f'- **5005aa vs AlphaFold2 reference:** RMSD {m.global_rmsd:.1f} Å with '
            f'{100 * m.global_pair_frac:.0f}% pairing; stitched pLDDT '
            f'{m.cand_plddt:.1f} vs reference {m.ref_plddt:.1f}.'
        )
    af2_3013_default = _find('3013aa', 'alphafold2-default', 'alphafold2')
    af2_3013_balanced = _find('3013aa', 'alphafold2-balanced', 'alphafold2')
    af2_3013 = af2_3013_default or _find('3013aa', 'alphafold2', 'alphafold2')
    if af2_3013_default and af2_3013_balanced:
        m_def = af2_3013_default['metrics']
        m_bal = af2_3013_balanced['metrics']
        bullets.append(
            f'- **3013aa AF2 default vs balanced (same reference):** default RMSD '
            f'{m_def.global_rmsd:.1f} Å; balanced RMSD {m_bal.global_rmsd:.1f} Å.'
        )
    elif af2_3013:
        m = af2_3013['metrics']
        bullets.append(
            f'- **3013aa AF2 split-stitch (MI250X) vs AF2 full-length:** RMSD '
            f'{m.global_rmsd:.1f} Å — available for cross-backend comparison.'
        )

    if not bullets:
        bullets.append('- See per-target tables above for numeric agreement.')

    return [
        '## Interpretation',
        '',
        '### Operational objective (OOM workaround)',
        '',
        (
            'Split-fold-stitch enables ColabFold and AlphaFold2 inference on MI250X/MI300X '
            'for inputs that OOM in full-length mode. The same sequences can be run '
            'full-length on MI355X or other high-memory systems.'
        ),
        '',
        '### Structural agreement with full-length references',
        '',
        *bullets,
        '',
        '### Notes',
        '',
        (
            '- **Paired Cα fraction** indicates how much of the chain superposes in one '
            'consistent alignment; low fractions mean largely different global folds.\n'
            '- **pLDDT** is per-residue confidence, not global agreement.\n'
            '- ColabFold and AlphaFold2 references may differ from each other as well as '
            'from split-stitch outputs.'
        ),
        '',
        '## Conclusion',
        '',
        (
            'Split-fold-stitch is a practical OOM workaround that produces complete models '
            'with competitive per-residue confidence. Agreement with full-length references '
            'is target-dependent; validate biologically important regions for production use.'
        ),
        '',
    ]


def build_report(
    *,
    work_dirs: list[CandidateWorkDir],
    reference_root: str | None,
    af2_output_root: str | None,
    platform: str,
    container: str,
) -> str:
    rows = _gather_results(
        reference_root=reference_root,
        af2_output_root=af2_output_root,
        candidates=work_dirs,
    )
    today = dt.date.today().isoformat()
    cand_lines = '\n'.join(f'- `{c.label}`: `{c.path}`' for c in work_dirs)
    ref_lines = []
    if reference_root:
        ref_lines.append(f'- **ColabFold:** `{reference_root}`')
    if af2_output_root:
        ref_lines.append(f'- **AlphaFold2:** `{af2_output_root}`')

    lines = [
        '# Split-fold-stitch validation report',
        '',
        f'Generated: {today}',
        '',
        '## Purpose',
        '',
        (
            'Validation of **split-fold-stitch** as an OOM workaround on memory-constrained '
            'GPUs (MI250X/MI300X). Stitched candidates are compared to **full-length '
            'monolithic** references from ColabFold and/or AlphaFold2.'
        ),
        '',
        '## Method summary',
        '',
        '| Parameter | Value |',
        '|-----------|-------|',
        '| Max segment width | 3000 aa (default; 300 aa for 1IH7.7 A3M test) |',
        '| Adjacent overlap | 1000 aa (long targets) or 100 aa (1IH7.7) |',
        '| Stitch anchor | Sliding 51-mer; pLDDT-primary (reported) and RMSD-primary |',
        f'| Split-fold platform | {platform} |',
        f'| Container | {container} |',
        '',
        '### Candidate directories',
        '',
        cand_lines,
        '',
        '### Reference directories',
        '',
        *ref_lines,
        '',
        '## Comparison methodology',
        '',
        (
            'PyMOL Cα `super` (see `scripts/compare_stitched_to_reference.py`). '
            'AlphaFold2 references use `ranked_0.pdb`; ColabFold uses rank-001 PDB.'
        ),
        '',
        '## Results summary',
        '',
        *_summary_table(rows),
        '',
        '## Per-comparison details',
        '',
    ]
    for row in rows:
        lines.extend(_detail_section(row))
    lines.extend(_interpretation(rows))
    lines.extend(
        [
            '## Reproduce',
            '',
            '```bash',
            'python3 scripts/generate_split_stitch_validation_report.py \\',
            '  --reference-root /home/sajandhy/colabfold_work \\',
            '  --af2-output-root /home/sajandhy/af2_output \\',
            '  --candidate-dir colabfold:/path/to/colabfold_work_rocm7.2.3 \\',
            '  --candidate-dir alphafold2:/path/to/alphafold2_work_rocm7.2.3 \\',
            '  --output reports/split_stitch_validation_report.md',
            '```',
            '',
            'Single comparison:',
            '',
            '```bash',
            'pymol -cq scripts/compare_stitched_to_reference.py -- \\',
            '  --reference-backend alphafold2 \\',
            '  --af2-output-root /home/sajandhy/af2_output \\',
            '  --target 3013aa \\',
            '  --candidate /path/to/3013aa_stitched_plddt.pdb',
            '```',
            '',
        ]
    )
    return '\n'.join(lines)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description='Generate split-fold-stitch validation report.')
    p.add_argument(
        '--work-dir',
        help=(
            'Shorthand: add one ColabFold candidate directory (same as '
            '--candidate-dir colabfold:PATH).'
        ),
    )
    p.add_argument(
        '--candidate-dir',
        action='append',
        default=[],
        metavar='LABEL:PATH',
        help=(
            'Stitched candidate tree; repeat for multiple runs. LABEL is shown in the '
            "report (e.g. alphafold2-default, alphafold2-balanced). Backend is inferred "
            "from the label prefix ('colabfold' or 'alphafold2')."
        ),
    )
    p.add_argument(
        '--stitch-suffix',
        choices=('plddt', 'rmsd', 'both'),
        default='both',
        help='Stitched variants from each --candidate-dir (default: both).',
    )
    p.add_argument(
        '--reference-root',
        default='/home/sajandhy/colabfold_work',
        help='ColabFold full-length reference root (omit with --no-colabfold-ref).',
    )
    p.add_argument(
        '--no-colabfold-ref',
        action='store_true',
        help='Skip ColabFold full-length references.',
    )
    p.add_argument(
        '--af2-output-root',
        default='/home/sajandhy/af2_output',
        help='AlphaFold2 full-length reference root (<target>/ranked_0.pdb).',
    )
    p.add_argument(
        '--no-af2-ref',
        action='store_true',
        help='Skip AlphaFold2 full-length references.',
    )
    p.add_argument(
        '--platform',
        default='AMD MI250X / MI300X (Pawsey Setonix)',
        help='Platform label for split-fold runs.',
    )
    p.add_argument(
        '--container',
        default='colabfold_rocm7.2.3_nopymol.sif / alphafold2_rocm7.2.3.sif + pymol.sif',
        help='Container description for split-fold runs.',
    )
    p.add_argument(
        '--output',
        default='reports/split_stitch_validation_report.md',
        help='Output markdown report path.',
    )
    args = p.parse_args(argv)

    candidate_specs = list(args.candidate_dir)
    if args.work_dir:
        candidate_specs.append(f'colabfold:{args.work_dir}')
    if not candidate_specs:
        candidate_specs.append(
            'colabfold:/home/sajandhy/GPU_biology/Pawsey/MI250X/colabfold_work_rocm7.2.3'
        )
        candidate_specs.append(
            'alphafold2:/home/sajandhy/GPU_biology/Pawsey/MI250X/alphafold2_work_rocm7.2.3'
        )

    work_dirs = parse_candidate_dirs(candidate_specs, stitch_variants=args.stitch_suffix)
    reference_root = None if args.no_colabfold_ref else os.path.abspath(args.reference_root)
    af2_output_root = None if args.no_af2_ref else os.path.abspath(args.af2_output_root)

    report = build_report(
        work_dirs=work_dirs,
        reference_root=reference_root,
        af2_output_root=af2_output_root,
        platform=args.platform,
        container=args.container,
    )

    out_path = os.path.abspath(args.output)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, 'w', encoding='utf-8') as fh:
        fh.write(report)
    print(f'Wrote {out_path}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
