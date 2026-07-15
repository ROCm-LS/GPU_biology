#!/usr/bin/env python3
"""
ColabFold split-fold-stitch (host orchestrator). For AlphaFold2 use ``split_and_fold_segments_alphafold2.py``.

Split a long FASTA/A3M, fold segments with ColabFold, stitch with PyMOL.

Runs from the **host** and always invokes **ColabFold inside a container** (Docker or
Singularity). It does not run ``colabfold_batch`` on the host Python environment.

  - ColabFold: ``quay.io/pawsey/colabfold:rocm6.2.4`` (Docker) or a .sif (Singularity)
  - PyMOL: ``jysgro/pymol:deb12-2.5.0_sc`` (Docker) or a .sif (Singularity)

Example (Singularity on HPC / Setonix):

  module load singularity/3.11.4-nompi
  python scripts/split_and_fold_segments_colabfold.py query.fa \\
    --runtime singularity \\
    --colabfold-sif /path/to/colabfold_rocm6.2.4.sif \\
    --pymol-sif /path/to/pymol.sif \\
    --work-dir $PWD

Example (Docker):

  python scripts/split_and_fold_segments_colabfold.py query.fa \\
    --runtime docker \\
    --work-dir /home/me/colabfold_work \\
    --colabfold-cache /home/me/colabfold_cache

Re-stitch only (fold outputs already present):

  python scripts/split_and_fold_segments_colabfold.py query.fa --skip-colabfold

Forward extra flags to colabfold_batch after ``--``:

  python scripts/split_and_fold_segments_colabfold.py query.fa -- --num-recycle 3
"""

from __future__ import annotations

import argparse
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed

from rocm_compute_devices import resolve_orchestrator_gpu_ids
from split_fold_stitch.container import (
    ContainerConfig,
    ContainerRunner,
    add_container_cli_args,
    container_config_from_args,
    resolve_work_dir,
)
from split_fold_stitch.plan import build_plan_json, plan_json_path, relativize_plan_paths, write_plan_json
from split_fold_stitch.tiling import (
    colabfold_msa_input_fail_hint,
    prepare_chunk_inputs,
)


GPU_IDS = resolve_orchestrator_gpu_ids()


def _run_one_colabfold_chunk(
    runner: ContainerRunner,
    i: int,
    fasta: str,
    out_dir: str,
    gpu_id: int,
    max_concurrent: int,
    *,
    colabfold_batch_extra: list[str],
) -> tuple[int, int]:
    print(
        f"-> Running chunk {i} on GCD {gpu_id} "
        f"(up to {max_concurrent} colabfold job(s) in parallel)..."
    )
    rc = runner.run_colabfold_batch(
        fasta,
        out_dir,
        gpu_id,
        colabfold_batch_extra,
    )
    return i, rc


def run_parallel_colabfold(
    runner: ContainerRunner,
    chunk_fastas: list[str],
    out_dirs: list[str],
    *,
    msa_input: bool = False,
    colabfold_batch_extra: list[str] | None = None,
) -> list[str]:
    n_gpus = max(1, len(GPU_IDS))
    n = len(chunk_fastas)
    if n != len(out_dirs):
        raise ValueError("chunk_fastas and out_dirs must be the same length")
    cfe = colabfold_batch_extra if colabfold_batch_extra is not None else []

    def work(i: int, fasta: str) -> tuple[int, int]:
        gpu_id = GPU_IDS[i % n_gpus]
        return _run_one_colabfold_chunk(
            runner,
            i,
            fasta,
            out_dirs[i],
            gpu_id,
            n_gpus,
            colabfold_batch_extra=cfe,
        )

    with ThreadPoolExecutor(max_workers=n_gpus) as ex:
        futs = [ex.submit(work, i, fasta) for i, fasta in enumerate(chunk_fastas)]
        for fut in as_completed(futs):
            i, rc = fut.result()
            if rc != 0:
                print(
                    f"Warning: colabfold_batch for chunk {i} exited {rc} "
                    f"({os.path.basename(chunk_fastas[i])!r} -> {out_dirs[i]!r})",
                    file=sys.stderr,
                )
                if msa_input:
                    print("  " + colabfold_msa_input_fail_hint(), file=sys.stderr)
    print("-> All colabfold_batch jobs finished.")
    return out_dirs


def _run_colabfold_stage(
    runner: ContainerRunner,
    seq_input: str,
    msa_in: bool,
    chunk_files: list[str],
    out_dirs: list[str],
    skip_colabfold: bool,
    single_original_input: bool,
    *,
    colabfold_batch_extra: list[str],
) -> list[str]:
    if skip_colabfold:
        if not runner.all_chunk_folds_done(out_dirs):
            raise SystemExit(
                f"--skip-colabfold: missing *_rank_001*.pdb in at least one of {out_dirs!r}"
            )
        print("-> Skipping colabfold_batch (existing output dirs).")
        return out_dirs

    if runner.all_chunk_folds_done(out_dirs):
        print(
            "Note: output dir(s) already contain rank_001 PDBs; colabfold_batch will still run. "
            "Use --skip-colabfold to re-stitch only. Resume behavior depends on your colabfold build.",
        )

    n_gpus = max(1, len(GPU_IDS))
    if single_original_input:
        out_dir0 = out_dirs[0]
        if msa_in:
            print(
                f"-> Single segment within length limit: folding original A2M/A3M "
                f"({os.path.basename(seq_input)!r}) — full MSA preserved, no per-part file."
            )
        else:
            print(
                "-> Single segment within length limit: folding original input "
                "(no per-part copy)."
            )
        _i, rc = _run_one_colabfold_chunk(
            runner,
            0,
            seq_input,
            out_dir0,
            GPU_IDS[0 % n_gpus],
            n_gpus,
            colabfold_batch_extra=colabfold_batch_extra,
        )
        if rc != 0:
            print(
                f"Error: colabfold_batch failed with exit {rc} "
                f"({os.path.basename(seq_input)!r} -> {out_dir0!r})",
                file=sys.stderr,
            )
            if msa_in:
                print("  " + colabfold_msa_input_fail_hint(), file=sys.stderr)
            raise SystemExit(1)
    else:
        run_parallel_colabfold(
            runner,
            chunk_files,
            out_dirs,
            msa_input=msa_in,
            colabfold_batch_extra=colabfold_batch_extra,
        )

    if not runner.all_chunk_folds_done(out_dirs):
        err = (
            "ColabFold did not write a *_rank_001*.pdb in every output directory. "
            "See colabfold_batch errors above."
        )
        if msa_in:
            err = err + " " + colabfold_msa_input_fail_hint()
        raise SystemExit(err)
    return out_dirs


def _run_pymol_stage(
    runner: ContainerRunner,
    base: str,
    chunks: list[tuple[int, int]],
    out_dirs: list[str],
    *,
    stitch_modes: list[str],
    validate_adjacent_segments: bool,
    work_dir: str,
    plan_mode: str = "default",
) -> None:
    plan_dir = os.path.join(work_dir, ".split_fold_stitch")
    os.makedirs(plan_dir, exist_ok=True)

    if validate_adjacent_segments:
        for mo in stitch_modes:
            other = "rmsd" if mo == "plddt" else "plddt"
            print(
                f"\n### Pre-stitch validation (anchor: primary {mo}, secondary {other}) ###\n"
            )
            plan_path = plan_json_path(plan_dir, base, "validate", mo)
            write_plan_json(
                plan_path,
                relativize_plan_paths(
                    build_plan_json(
                        base=base,
                        chunks=chunks,
                        out_dirs=out_dirs,
                        anchor_primary=mo,
                        fold_backend="colabfold",
                        plan_mode=plan_mode,
                    ),
                    work_dir,
                ),
            )
            rc = runner.run_pymol_worker("validate", plan_path)
            if rc != 0:
                raise SystemExit(f"PyMOL validation failed (exit {rc}) for mode {mo!r}.")

    for mo in stitch_modes:
        out_pdb = f"{base}_stitched_{mo}.pdb"
        print(f"\n### Stitch: anchor policy {mo} -> {out_pdb} ###\n")
        plan_path = plan_json_path(plan_dir, base, "stitch", mo)
        write_plan_json(
            plan_path,
            relativize_plan_paths(
                build_plan_json(
                    base=base,
                    chunks=chunks,
                    out_dirs=out_dirs,
                    output_pdb=out_pdb,
                    anchor_primary=mo,
                    fold_backend="colabfold",
                    plan_mode=plan_mode,
                ),
                work_dir,
            ),
        )
        rc = runner.run_pymol_worker("stitch", plan_path)
        if rc != 0:
            raise SystemExit(f"PyMOL stitch failed (exit {rc}) for mode {mo!r}.")

    summary_plan = plan_json_path(plan_dir, base, "summary")
    write_plan_json(
        summary_plan,
        relativize_plan_paths(
            build_plan_json(
                base=base,
                chunks=chunks,
                out_dirs=out_dirs,
                modes=stitch_modes,
                fold_backend="colabfold",
                plan_mode=plan_mode,
            ),
            work_dir,
        ),
    )
    runner.run_pymol_worker("summary", summary_plan)


def main(
    seq_input: str,
    runner: ContainerRunner,
    *,
    stitch_modes: str = "both",
    skip_colabfold: bool = False,
    validate_adjacent_segments: bool = False,
    max_chunk_aa: int | None = None,
    plan_mode: str = "default",
    colabfold_batch_extra: list[str] | None = None,
) -> None:
    seq_input = os.path.abspath(seq_input)
    base, _header, msa_in, chunks, chunk_files, out_dirs, one_segment, plan_mode_used = (
        prepare_chunk_inputs(
            seq_input, max_chunk_aa=max_chunk_aa, plan_mode=plan_mode
        )
    )

    abs_paths = [seq_input, *chunk_files, *out_dirs]
    work_dir = resolve_work_dir(runner.config.work_dir, *abs_paths)
    runner.config.work_dir = work_dir
    runner.work_dir = work_dir
    print(f"-> Work directory (container mount /work): {work_dir}")

    out_dirs = _run_colabfold_stage(
        runner,
        seq_input,
        msa_in,
        chunk_files,
        out_dirs,
        skip_colabfold,
        one_segment,
        colabfold_batch_extra=colabfold_batch_extra or [],
    )

    if one_segment:
        print(
            f"-> Single segment: ColabFold result in {out_dirs[0]!r}. "
            "Skipping adjacent-segment validation, stitch, and merge summary."
        )
        if validate_adjacent_segments:
            print(
                "  (Note: --validate-adjacent-segments is ignored when there is only one segment.)"
            )
        return

    if stitch_modes == "both":
        mode_list: list[str] = ["plddt", "rmsd"]
    else:
        mode_list = [stitch_modes]

    _run_pymol_stage(
        runner,
        base,
        chunks,
        out_dirs,
        stitch_modes=mode_list,
        validate_adjacent_segments=validate_adjacent_segments,
        work_dir=work_dir,
        plan_mode=plan_mode_used,
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description=(
            "Split, fold (ColabFold container), and stitch (PyMOL container) a long sequence."
        ),
        epilog=(
            "To forward argv tokens to colabfold_batch unchanged, add a bare -- after this "
            "script's own options, then the colabfold flags and values, e.g.  "
            "%(prog)s query.fa --max-chunk-aa 400 -- --num-recycle 3"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "input",
        help=(
            "input FASTA or A2M/A3M path. Multi-segment A2M/A3M outputs column-sliced, valid A3M part files; "
            "multi-segment FASTA writes query-only subsequences per part."
        ),
    )
    p.add_argument(
        "--stitch-modes",
        choices=("both", "plddt", "rmsd"),
        default="both",
        help=(
            "anchor policy for overlap stitching: pLDDT- or RMSD-primary 51-mer, or both (two merged PDBs). "
            "With a single segment, ColabFold output is left in <base>_output; stitch, validation, and "
            "summary are not run."
        ),
    )
    p.add_argument(
        "--skip-colabfold",
        action="store_true",
        help=(
            "do not run colabfold_batch; require existing chunk *_output dirs with a rank_001 PDB. "
            "Re-run behavior of colabfold itself is not controlled here; use this to avoid re-folding in this script."
        ),
    )
    p.add_argument(
        "--validate-adjacent-segments",
        action="store_true",
        help=(
            "after folding, for each --stitch-modes anchor, run validate_all_adjacent_pairs on "
            "consecutive segment PDBs in the overlap (no work when there is only one segment)."
        ),
    )
    p.add_argument(
        "--max-chunk-aa",
        type=int,
        default=None,
        metavar="N",
        help=(
            "cap each ColabFold run to N query residues (default: 3012). Use a smaller N to force "
            "multiple segments for long queries or for A2M/A3M when a single run exceeds GPU memory; "
            "overlap is chosen automatically and may be < 1000 aa."
        ),
    )
    p.add_argument(
        "--plan-mode",
        choices=("default", "balanced"),
        default="default",
        help=(
            "tiling policy: default uses fixed ~3000 aa windows; balanced shrinks the first window "
            "when one segment would dominate the stitched model (e.g. 3013 aa just above the "
            "3000 aa OOM cap). Ignored for single-segment inputs."
        ),
    )
    add_container_cli_args(p, container_only=True)

    try:
        i = sys.argv.index("--", 1)
    except ValueError:
        colabfold_batch_extra: list[str] = []
    else:
        colabfold_batch_extra = sys.argv[i + 1 :]
        del sys.argv[i:]

    a = p.parse_args()
    seq_input = os.path.abspath(a.input)
    work_dir = resolve_work_dir(a.work_dir, seq_input)
    cfg = container_config_from_args(a, work_dir, container_only=True)
    runner = ContainerRunner(cfg)

    print(
        f"-> Runtime: {cfg.runtime} | ColabFold: "
        f"{cfg.colabfold_container_name or cfg.colabfold_sif or cfg.colabfold_image} | "
        f"PyMOL: {cfg.pymol_sif or cfg.pymol_image}"
    )

    main(
        seq_input,
        runner,
        stitch_modes=a.stitch_modes,
        skip_colabfold=a.skip_colabfold,
        validate_adjacent_segments=a.validate_adjacent_segments,
        max_chunk_aa=a.max_chunk_aa,
        plan_mode=a.plan_mode,
        colabfold_batch_extra=colabfold_batch_extra,
    )
