#!/usr/bin/env python3
"""Compare stitched (or other) models to a full-length reference PDB.

Uses PyMOL Cα superposition (same logic as split_fold_stitch/pymol_worker.py).

Run headless in a PyMOL container::

  pymol -cq scripts/compare_stitched_to_reference.py -- \\
    --reference /work/3013aa_reference/3013aa_rank_001*.pdb \\
    --candidate /work/colabfold_work/3013aa_stitched_plddt.pdb \\
    --candidate /work/colabfold_work/3013aa_stitched_rmsd.pdb \\
    --overlap-start 2001 --overlap-end 3000

AlphaFold2 full-length reference (auto-resolve from output dir)::

  pymol -cq scripts/compare_stitched_to_reference.py -- \\
    --reference-backend alphafold2 \\
    --af2-output-root /home/sajandhy/af2_output \\
    --target 3013aa \\
    --candidate /work/3013aa_stitched_plddt.pdb \\
    --candidate /work/3013aa_stitched_rmsd.pdb

Or discover both stitched variants from a work directory::

  pymol -cq scripts/compare_stitched_to_reference.py -- \\
    --reference-backend alphafold2 \\
    --target 3013aa \\
    --candidate-dir /work/colabfold_work_rocm7.2.3 \\
    --stitch-suffix both

Or with plain python3 if ``import pymol`` works in the current environment.

Run from the repository root (``~/GPU_biology``) so PyMOL can find ``stitch_compare_core.py``.
Alternatively: ``PYTHONPATH=~/GPU_biology/scripts pymol -cq ...``
"""

from __future__ import annotations

import argparse
import glob
import os
import sys


def _bootstrap_scripts_path() -> None:
    """Ensure ``scripts/`` is on sys.path when PyMOL runs this file.

    Under ``pymol -cq scripts/foo.py``, ``__file__`` is often only the basename,
    so dirname(__file__) does not resolve to ``.../GPU_biology/scripts``.
    """
    candidates: list[str] = []
    file_ref = globals().get('__file__')
    if file_ref:
        candidates.append(os.path.dirname(os.path.abspath(file_ref)))
    if sys.argv:
        candidates.append(os.path.dirname(os.path.abspath(sys.argv[0])))
    candidates.append(os.path.join(os.getcwd(), 'scripts'))
    candidates.append(os.path.join(os.getcwd(), 'pipeline_scripts'))

    seen: set[str] = set()
    for directory in candidates:
        if not directory or directory in seen:
            continue
        seen.add(directory)
        if os.path.isfile(os.path.join(directory, 'stitch_compare_core.py')):
            if directory not in sys.path:
                sys.path.insert(0, directory)
            return

    raise SystemExit(
        'Could not locate stitch_compare_core.py. Run from the repo root '
        '(~/GPU_biology) or set PYTHONPATH=~/GPU_biology/scripts.'
    )


_bootstrap_scripts_path()

from pymol import cmd, stored

from stitch_compare_core import (
    TARGET_CASES,
    case_by_name,
    compare_structures,
    find_stitched_candidates,
    junction_label,
    resolve_reference,
)


def _expand_path(path: str) -> str:
    matches = sorted(glob.glob(path))
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        raise SystemExit(
            f'Glob {path!r} matched multiple files; pass an explicit path:\n'
            + '\n'.join(f'  {m}' for m in matches)
        )
    if os.path.isfile(path):
        return path
    raise SystemExit(f'File not found: {path!r}')


def _print_comparison(
    metrics,
    *,
    label: str,
    overlap_start: int | None,
    overlap_end: int | None,
) -> None:
    print(f'\n=== {label} ===')
    print(f'  reference: {metrics.ref_path}  ({metrics.n_ref} Cα)')
    print(f'  candidate: {metrics.cand_path}  ({metrics.n_cand} Cα)')
    if metrics.n_ref != metrics.n_cand:
        print(
            f'  WARNING: Cα counts differ ({metrics.n_ref} vs {metrics.n_cand}); '
            'RMSD uses matched residues from superposition only.'
        )

    print(
        f'  global Cα RMSD (cand → ref): {metrics.global_rmsd:.3f} Å  '
        f'({metrics.global_pairs} pairs)'
    )

    if overlap_start is not None and overlap_end is not None:
        for ov in metrics.overlaps:
            if ov.lo == overlap_start and ov.hi == overlap_end:
                print(
                    f'  overlap Cα RMSD (resi {overlap_start}-{overlap_end}): '
                    f'{ov.rmsd:.3f} Å  ({ov.pairs} pairs)'
                )
                break
    elif metrics.overlaps:
        for ov in metrics.overlaps:
            print(
                f'  overlap Cα RMSD (resi {junction_label(ov.lo, ov.hi)}): '
                f'{ov.rmsd:.3f} Å  ({ov.pairs} pairs)'
            )

    print(
        f'  mean pLDDT (B-factor, all Cα): ref={metrics.ref_plddt:.2f}  '
        f'cand={metrics.cand_plddt:.2f}'
    )


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description='Superpose candidate PDBs onto a full-length reference; report Cα RMSD.',
    )
    p.add_argument(
        '--reference',
        help='Full-length reference PDB path (glob allowed if unique). '
        'Optional when --target and a reference backend root are provided.',
    )
    p.add_argument(
        '--reference-backend',
        choices=('colabfold', 'alphafold2'),
        default='colabfold',
        help='How to resolve --target into a reference PDB (default: colabfold).',
    )
    p.add_argument(
        '--reference-root',
        help='ColabFold output root (e.g. /home/sajandhy/colabfold_work).',
    )
    p.add_argument(
        '--af2-output-root',
        default='/home/sajandhy/af2_output',
        help='AlphaFold2 output root containing <target>/ranked_0.pdb.',
    )
    p.add_argument(
        '--target',
        choices=[c.name for c in TARGET_CASES],
        help='Known validation target; auto-select overlap windows and reference path.',
    )
    p.add_argument(
        '--reference-chain',
        help='Reference chain ID for multimer PDBs (default: case-specific, often A).',
    )
    p.add_argument(
        '--candidate',
        action='append',
        default=[],
        metavar='PDB',
        help='Stitched PDB to compare (repeat for plddt and rmsd). '
        'Optional if --candidate-dir and --target are set.',
    )
    p.add_argument(
        '--candidate-dir',
        help='Work directory containing <target>_stitched_plddt.pdb and/or '
        '<target>_stitched_rmsd.pdb (requires --target).',
    )
    p.add_argument(
        '--stitch-suffix',
        choices=('plddt', 'rmsd', 'both'),
        default='both',
        help='Which stitched variants to load from --candidate-dir (default: both).',
    )
    p.add_argument(
        '--overlap-start',
        type=int,
        help='1-based start of stitch overlap window.',
    )
    p.add_argument(
        '--overlap-end',
        type=int,
        help='1-based end of stitch overlap window.',
    )
    p.add_argument(
        '--no-overlap-report',
        action='store_true',
        help='Skip per-overlap RMSD (global only).',
    )
    args = p.parse_args(argv)

    case = case_by_name(args.target) if args.target else None
    ref_chain = args.reference_chain
    if case and ref_chain is None:
        ref_chain = case.ref_chain

    if args.reference:
        ref_path = _expand_path(args.reference)
        ref_chain = ref_chain  # may stay None
        if case:
            overlap_windows = case.overlap_windows
            length = case.length
        else:
            overlap_windows = ()
            if not args.no_overlap_report:
                if args.overlap_start is None or args.overlap_end is None:
                    raise SystemExit(
                        'Pass --overlap-start/--overlap-end, --target, or --no-overlap-report.'
                    )
                lo, hi = sorted((args.overlap_start, args.overlap_end))
                overlap_windows = ((lo, hi),)
            cmd.reinitialize()
            cmd.load(ref_path, 'ref')
            stored._n = 0
            ref_sel = 'ref' if not ref_chain else f'ref and chain {ref_chain}'
            cmd.iterate(f'{ref_sel} and name CA', 'stored._n += 1')
            length = int(stored._n)
    else:
        if not case:
            raise SystemExit('Pass --reference or --target with a reference backend root.')
        try:
            ref_path = resolve_reference(
                backend=args.reference_backend,
                case=case,
                reference_root=args.reference_root,
                af2_output_root=args.af2_output_root,
            )
        except (FileNotFoundError, ValueError) as exc:
            raise SystemExit(str(exc)) from exc
        overlap_windows = case.overlap_windows
        length = case.length

    if args.no_overlap_report:
        overlap_windows = ()
    elif case is None and args.overlap_start is not None and args.overlap_end is not None:
        lo, hi = sorted((args.overlap_start, args.overlap_end))
        overlap_windows = ((lo, hi),)

    candidates: list[str] = list(args.candidate)
    if args.candidate_dir:
        if not case:
            raise SystemExit('--candidate-dir requires --target.')
        for _suffix, path in find_stitched_candidates(
            os.path.abspath(args.candidate_dir),
            case,
            variants=args.stitch_suffix,
        ):
            candidates.append(path)
    if not candidates:
        raise SystemExit(
            'Pass at least one --candidate PDB, or --candidate-dir with --target.'
        )

    print(f'Reference: {ref_path}')
    for cand in candidates:
        cand_path = _expand_path(cand)
        metrics = compare_structures(
            ref_path=ref_path,
            cand_path=cand_path,
            length=length,
            overlap_windows=overlap_windows,
            ref_chain=ref_chain,
        )
        _print_comparison(
            metrics,
            label=os.path.basename(cand_path),
            overlap_start=args.overlap_start,
            overlap_end=args.overlap_end,
        )
    return 0


def _entry() -> int:
    if __name__ == '__main__':
        return main()
    if __name__ == 'pymol' and len(sys.argv) > 1:
        return main(sys.argv[1:])
    return 0


if __name__ == '__main__':
    raise SystemExit(_entry())
elif __name__ == 'pymol':
    _entry()
