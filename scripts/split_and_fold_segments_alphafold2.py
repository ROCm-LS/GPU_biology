#!/usr/bin/env python3
"""
Long FASTA → tiled segments → **AlphaFold2** (one container) → **PyMOL** stitch (another).

**Runs on the host.** This script orchestrates Docker / Singularity containers; it does
not run inside the AlphaFold2 image. For tiling + fold + stitch in one process inside
the fold container, use ``split_and_fold_segments_alphafold2_single_container.py``.

Designed for a **standard AlphaFold2 setup**: local database tree and ``run_alphafold.py``
flags you pass after ``--`` (e.g. ``--data_dir``, ``--model_preset``, ``--db_preset``).
Each run writes ``<--af2-output-base>/<fasta_stem>/ranked_*.pdb``; stitching uses those paths.

**Input:** FASTA only (one query sequence). For A2M/A3M or custom MSA flows, use the
ColabFold orchestrator or run AlphaFold2 yourself per chunk.

**Layout:** match ``--work-dir`` to a host directory that is bind-mounted as ``/work`` in
both containers, with databases and outputs under that tree (see repo ``README.md``).

Example::

  python scripts/split_and_fold_segments_alphafold2.py query.fa \\
    --work-dir /data/af2_project \\
    --alphafold2-container-name my_af2 \\
    -- --data_dir=/work/databases --model_preset=monomer --db_preset=full_dbs

Forward all ``run_alphafold.py`` arguments after a bare ``--``.
"""

from __future__ import annotations

import argparse
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed

from rocm_compute_devices import discover_compute_rocm_gpu_ids
from split_fold_stitch.container import (
    ContainerRunner,
    add_alphafold2_container_cli_args,
    add_container_cli_args,
    container_config_from_args,
    resolve_work_dir,
)
from split_fold_stitch.plan import build_plan_json, relativize_plan_paths, write_plan_json
from split_fold_stitch.tiling import prepare_chunk_inputs


def _parse_hip_visible_devices() -> list[int]:
    raw = os.environ.get("HIP_VISIBLE_DEVICES", "")
    if not raw.strip():
        return []
    out: list[int] = []
    for part in raw.split(","):
        p = part.strip()
        if not p:
            continue
        try:
            out.append(int(p))
        except ValueError:
            pass
    return out


def _init_gpu_ids() -> list[int]:
    if "HIP_VISIBLE_DEVICES" in os.environ:
        return _parse_hip_visible_devices()
    return discover_compute_rocm_gpu_ids()


GPU_IDS = _init_gpu_ids()
if not GPU_IDS:
    print(
        "Warning: no GPU indices (empty HIP_VISIBLE_DEVICES?); using [0].",
        file=sys.stderr,
    )
    GPU_IDS = [0]


def _af2_pred_dirs(output_base: str, chunk_fastas: list[str]) -> list[str]:
    return [
        os.path.join(output_base, os.path.splitext(os.path.basename(fp))[0])
        for fp in chunk_fastas
    ]


def _run_one_af2_chunk(
    runner: ContainerRunner,
    i: int,
    fasta: str,
    output_base: str,
    gpu_id: int,
    max_concurrent: int,
    *,
    run_af_extra: list[str],
) -> tuple[int, int]:
    print(
        f"-> AlphaFold2 chunk {i} on GCD {gpu_id} "
        f"(up to {max_concurrent} job(s) in parallel)..."
    )
    rc = runner.run_alphafold_fasta(fasta, output_base, gpu_id, run_af_extra)
    return i, rc


def run_parallel_af2(
    runner: ContainerRunner,
    chunk_fastas: list[str],
    output_base: str,
    *,
    run_af_extra: list[str],
    gpu_slot: int = 0,
) -> None:
    n_gpus = max(1, len(GPU_IDS))

    def work(i: int, fp: str) -> tuple[int, int]:
        gpu_id = GPU_IDS[(gpu_slot + i) % n_gpus]
        return _run_one_af2_chunk(
            runner, i, fp, output_base, gpu_id, n_gpus, run_af_extra=run_af_extra
        )

    with ThreadPoolExecutor(max_workers=n_gpus) as ex:
        futs = [ex.submit(work, i, fp) for i, fp in enumerate(chunk_fastas)]
        for fut in as_completed(futs):
            i, rc = fut.result()
            if rc != 0:
                print(
                    f"Warning: run_alphafold.py for chunk {i} exited {rc} "
                    f"({os.path.basename(chunk_fastas[i])!r}).",
                    file=sys.stderr,
                )
    print("-> All run_alphafold.py jobs finished.")


def _run_pymol_stage(
    runner: ContainerRunner,
    base: str,
    chunks: list[tuple[int, int]],
    pred_dirs: list[str],
    *,
    stitch_modes: list[str],
    validate_adjacent_segments: bool,
    work_dir: str,
) -> None:
    plan_dir = os.path.join(work_dir, ".split_fold_stitch")
    os.makedirs(plan_dir, exist_ok=True)
    fb = "alphafold2"

    if validate_adjacent_segments:
        for mo in stitch_modes:
            other = "rmsd" if mo == "plddt" else "plddt"
            print(
                f"\n### Pre-stitch validation (anchor: primary {mo}, secondary {other}) ###\n"
            )
            plan_path = os.path.join(plan_dir, f"validate_{mo}.json")
            write_plan_json(
                plan_path,
                relativize_plan_paths(
                    build_plan_json(
                        base=base,
                        chunks=chunks,
                        out_dirs=pred_dirs,
                        anchor_primary=mo,
                        fold_backend=fb,
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
        plan_path = os.path.join(plan_dir, f"stitch_{mo}.json")
        write_plan_json(
            plan_path,
            relativize_plan_paths(
                build_plan_json(
                    base=base,
                    chunks=chunks,
                    out_dirs=pred_dirs,
                    output_pdb=out_pdb,
                    anchor_primary=mo,
                    fold_backend=fb,
                ),
                work_dir,
            ),
        )
        rc = runner.run_pymol_worker("stitch", plan_path)
        if rc != 0:
            raise SystemExit(f"PyMOL stitch failed (exit {rc}) for mode {mo!r}.")

    summary_plan = os.path.join(plan_dir, "summary.json")
    write_plan_json(
        summary_plan,
        relativize_plan_paths(
            build_plan_json(
                base=base,
                chunks=chunks,
                out_dirs=pred_dirs,
                modes=stitch_modes,
                fold_backend=fb,
            ),
            work_dir,
        ),
    )
    runner.run_pymol_worker("summary", summary_plan)


def _run_af2_stage(
    runner: ContainerRunner,
    chunk_fastas: list[str],
    pred_dirs: list[str],
    output_base: str,
    skip_af2: bool,
    one_segment: bool,
    *,
    run_af_extra: list[str],
    gpu_slot: int = 0,
) -> None:
    if skip_af2:
        if not runner.all_chunk_folds_done_af2(pred_dirs):
            raise SystemExit(
                f"--skip-alphafold: missing ranked_*.pdb under at least one of {pred_dirs!r}"
            )
        print("-> Skipping run_alphafold.py (existing prediction dirs).")
        return

    if runner.all_chunk_folds_done_af2(pred_dirs):
        print(
            "Note: prediction dir(s) already contain ranked PDBs; run_alphafold.py will still run. "
            "Use --skip-alphafold to stitch only."
        )

    n_gpus = max(1, len(GPU_IDS))
    if one_segment:
        chunk_i = 0
        gpu_id = GPU_IDS[(gpu_slot + chunk_i) % n_gpus]
        _i, rc = _run_one_af2_chunk(
            runner,
            chunk_i,
            chunk_fastas[0],
            output_base,
            gpu_id,
            n_gpus,
            run_af_extra=run_af_extra,
        )
        if rc != 0:
            raise SystemExit(
                f"run_alphafold.py failed with exit {rc} ({chunk_fastas[0]!r})."
            )
    else:
        run_parallel_af2(
            runner,
            chunk_fastas,
            output_base,
            run_af_extra=run_af_extra,
            gpu_slot=gpu_slot,
        )

    if not runner.all_chunk_folds_done_af2(pred_dirs):
        raise SystemExit(
            "AlphaFold2 did not produce ranked_*.pdb in every prediction directory. "
            "See run_alphafold.py errors above."
        )


def main(
    seq_input: str,
    runner: ContainerRunner,
    *,
    af2_output_base: str,
    stitch_modes: str = "both",
    skip_alphafold: bool = False,
    validate_adjacent_segments: bool = False,
    max_chunk_aa: int | None = None,
    run_alphafold_extra: list[str] | None = None,
    af2_gpu_slot: int = 0,
) -> None:
    seq_input = os.path.abspath(seq_input)
    base, _header, msa_in, chunks, chunk_files, _cf_out_dirs, one_segment = (
        prepare_chunk_inputs(seq_input, max_chunk_aa=max_chunk_aa)
    )
    if msa_in:
        raise SystemExit(
            "This script accepts FASTA only. Use split_and_fold_segments_colabfold.py "
            "for ColabFold (FASTA or A3M), or run run_alphafold.py per chunk with your own MSA setup."
        )

    output_base = os.path.abspath(af2_output_base)
    os.makedirs(output_base, exist_ok=True)
    pred_dirs = _af2_pred_dirs(output_base, chunk_files)

    abs_paths = [seq_input, *chunk_files, output_base, *pred_dirs]
    work_dir = resolve_work_dir(runner.config.work_dir, *abs_paths)
    runner.config.work_dir = work_dir
    runner.work_dir = work_dir
    print(f"-> Work directory (container mount /work): {work_dir}")
    print(f"-> AlphaFold --output_dir base (under /work): {output_base}")

    _run_af2_stage(
        runner,
        chunk_files,
        pred_dirs,
        output_base,
        skip_alphafold,
        one_segment,
        run_af_extra=run_alphafold_extra or [],
        gpu_slot=af2_gpu_slot,
    )

    if one_segment:
        print(
            f"-> Single segment: AlphaFold2 result in {pred_dirs[0]!r}. "
            "Skipping validation, stitch, and summary."
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
        pred_dirs,
        stitch_modes=mode_list,
        validate_adjacent_segments=validate_adjacent_segments,
        work_dir=work_dir,
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description=(
            "Split FASTA, fold segments (AlphaFold2 container), stitch (PyMOL container)."
        ),
        epilog=(
            "Required run_alphafold.py flags (after --) depend on your image; typical full-DB "
            "run: --data_dir=/work/databases --model_preset=monomer --db_preset=full_dbs "
            "plus database path flags as in AlphaFold2 docs."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "input",
        help="input FASTA (single query); long sequences are tiled automatically.",
    )
    p.add_argument(
        "--af2-output-base",
        default=None,
        metavar="DIR",
        help=(
            "host directory passed to run_alphafold.py as --output_dir (default: "
            "<work-dir>/af2_predictions). Predictions appear as DIR/<fasta_stem>/."
        ),
    )
    p.add_argument(
        "--stitch-modes",
        choices=("both", "plddt", "rmsd"),
        default="both",
        help="anchor policy for overlap stitching (ignored for a single segment).",
    )
    p.add_argument(
        "--skip-alphafold",
        action="store_true",
        help=(
            "do not run run_alphafold.py; require existing DIR/<fasta_stem>/ with ranked_*.pdb."
        ),
    )
    p.add_argument(
        "--validate-adjacent-segments",
        action="store_true",
        help="after folding, validate overlap between consecutive segment PDBs.",
    )
    p.add_argument(
        "--max-chunk-aa",
        type=int,
        default=None,
        metavar="N",
        help="max residues per segment (default: 3012, same tiling as ColabFold script).",
    )
    p.add_argument(
        "--af2-gpu-slot",
        type=int,
        default=0,
        metavar="K",
        help=(
            "0-based offset into HIP_VISIBLE_DEVICES / discovered GPU list (modulo GPU count). "
            "Single-segment runs use GPU_IDS[K %% n_gpus] for the sole AlphaFold job. "
            "Multi-chunk runs assign chunk i to GPU_IDS[(K + i) %% n_gpus]."
        ),
    )
    add_container_cli_args(p)
    add_alphafold2_container_cli_args(p)

    try:
        i = sys.argv.index("--", 1)
    except ValueError:
        run_alphafold_extra: list[str] = []
    else:
        run_alphafold_extra = sys.argv[i + 1 :]
        del sys.argv[i:]

    a = p.parse_args()
    seq_input = os.path.abspath(a.input)
    work_dir = resolve_work_dir(a.work_dir, seq_input)
    if a.af2_output_base:
        af2_out = os.path.abspath(a.af2_output_base)
    else:
        af2_out = os.path.join(work_dir, "af2_predictions")
    cfg = container_config_from_args(a, work_dir)
    runner = ContainerRunner(cfg)

    print(
        f"-> Runtime: {cfg.runtime} | AlphaFold2: "
        f"{cfg.alphafold2_container_name or cfg.alphafold2_sif or cfg.alphafold2_image} "
        f"(app {cfg.alphafold2_app_root}) | PyMOL: {cfg.pymol_sif or cfg.pymol_image}"
    )

    main(
        seq_input,
        runner,
        af2_output_base=af2_out,
        stitch_modes=a.stitch_modes,
        skip_alphafold=a.skip_alphafold,
        validate_adjacent_segments=a.validate_adjacent_segments,
        max_chunk_aa=a.max_chunk_aa,
        run_alphafold_extra=run_alphafold_extra,
        af2_gpu_slot=a.af2_gpu_slot,
    )
