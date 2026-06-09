#!/usr/bin/env python3
"""
PyMOL worker for split-fold-stitch: validation and structural merging.

Run headlessly inside a PyMOL container, e.g.:

  pymol -cq split_fold_stitch/pymol_worker.py -- stitch \\
    --plan /work/run_plan.json

Or from the host orchestrator via ``ContainerRunner.run_pymol_worker(...)``.
"""

from __future__ import annotations

import argparse
import glob
import os
import sys

_pkg_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _pkg_root not in sys.path:
    sys.path.insert(0, _pkg_root)

from pymol import cmd, stored

from split_fold_stitch.tiling import ANCHOR_SLIDE, JUNCTION_ALIGN_W

_PDDT_TIE: float = 1e-4
_RMSD_TIE: float = 1e-6


def _best_chunk_pdb(pred_dir: str, fold_backend: str) -> str | None:
    """Pick one structure PDB per chunk directory (ColabFold vs AlphaFold2 layout)."""
    if fold_backend == "alphafold2":
        p0 = os.path.join(pred_dir, "ranked_0.pdb")
        if os.path.isfile(p0):
            return p0
        cand = sorted(glob.glob(os.path.join(pred_dir, "ranked_*.pdb")))
        return cand[0] if cand else None
    if fold_backend != "colabfold":
        raise ValueError(f"Unknown fold_backend {fold_backend!r}")
    g = glob.glob(os.path.join(pred_dir, "*_rank_001*.pdb"))
    return g[0] if g else None


def _chunk_overlap_local_in_prev(
    s_prev: int, e_prev: int, s_curr: int, _e_curr: int
) -> tuple[int, int] | None:
    """1-based local [lo, hi] in the previous segment's ColabFold PDB, or None if no overlap."""
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
    if primary == "plddt":
        if plddt_new > plddt_cur + _PDDT_TIE:
            return True
        if abs(plddt_new - plddt_cur) <= _PDDT_TIE and rmsd_new < rmsd_cur - _RMSD_TIE:
            return True
        return False
    if primary == "rmsd":
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
    primary: str = "plddt",
    *,
    fold_backend: str = "colabfold",
) -> int | None:
    f1 = _best_chunk_pdb(out_dir_a, fold_backend)
    f2 = _best_chunk_pdb(out_dir_b, fold_backend)
    if not f1 or not f2:
        return None
    lo, hi = ovl[0], ovl[1]
    if hi - lo < ANCHOR_SLIDE + 1:
        return None
    na, nb = "pqa_anch", "pqb_anch"
    for nm in (na, nb):
        if nm in (cmd.get_names("objects") or []):
            try:
                cmd.delete(nm)
            except Exception:
                pass
    best_r: int | None = None
    best_p = 0.0
    best_rmsd = float("inf")
    try:
        cmd.load(f1, na)
        cmd.load(f2, nb)
        cmd.alter(na, f"resv += {s_a}")
        cmd.alter(nb, f"resv += {s_b}")
        cmd.sort()
        for r in range(lo, hi - ANCHOR_SLIDE):
            g0 = s_a + r
            ca = f"resi {g0}-{g0+ANCHOR_SLIDE} and name CA"
            m1 = cmd.get_model(f"{na} and {ca}")
            if not m1.atom:
                continue
            plddt = sum(a.b for a in m1.atom) / len(m1.atom)
            try:
                sr = cmd.super(f"{nb} and {ca}", f"{na} and {ca}", cycles=0)
            except Exception:
                continue
            if not sr[1] or int(sr[1]) == 0:
                continue
            rmsd = float(sr[0])
            if best_r is None:
                best_r, best_p, best_rmsd = r, plddt, rmsd
            elif _anchor_window_is_better(
                plddt, rmsd, best_p, best_rmsd, primary
            ):
                best_r, best_p, best_rmsd = r, plddt, rmsd
    finally:
        for nm in (na, nb):
            if nm in (cmd.get_names("objects") or []):
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
    anchor_primary: str = "plddt",
    *,
    fold_backend: str = "colabfold",
) -> None:
    f1p = _best_chunk_pdb(out_dir_a, fold_backend)
    f2p = _best_chunk_pdb(out_dir_b, fold_backend)
    if not f1p or not f2p:
        print(
            f"--- Pair {pair_index - 1} vs {pair_index}: "
            f"missing prediction PDB under chunk dir ---"
        )
        return

    ovl_local = _chunk_overlap_local_in_prev(s_a, e_a, s_b, e_b)
    pbe, g_anchor0 = None, None
    if ovl_local is not None:
        pbe = find_best_anchor_by_pair_silent(
            out_dir_a, out_dir_b, s_a, s_b, ovl_local, primary=anchor_primary,
            fold_backend=fold_backend,
        )
        if pbe is not None:
            g_anchor0 = s_a + pbe

    has_overlap_1b = s_b < e_a
    o1, o2 = s_b + 1, e_a
    ovl_lbl = (
        f"overlap 1-based {o1}–{o2}"
        if has_overlap_1b and o1 <= o2
        else "abutting (no overlap window)"
    )

    cmd.reinitialize()
    cmd.load(f1p, "frag1")
    cmd.load(f2p, "frag2")
    cmd.alter("frag1", f"resv += {s_a}")
    cmd.alter("frag2", f"resv += {s_b}")
    cmd.sort()
    other = "rmsd" if anchor_primary == "plddt" else "plddt"
    print(
        f"--- Validation: part {pair_index - 1} vs {pair_index} ({ovl_lbl}) | "
        f"anchor: primary {anchor_primary}, secondary {other} ---"
    )

    anchor_plddt = 0.0
    if has_overlap_1b and g_anchor0 is not None:
        an = f"resi {g_anchor0}-{g_anchor0 + ANCHOR_SLIDE} and name CA"
        m_anchor1 = cmd.get_model(f"frag1 and {an}")
        if m_anchor1.atom:
            anchor_plddt = sum((a.b for a in m_anchor1.atom)) / len(m_anchor1.atom)

    overlap_plddt = 0.0
    ov_ca = f"resi {o1}-{o2} and name CA" if (has_overlap_1b and o1 <= o2) else "none"
    if has_overlap_1b and o1 <= o2:
        m_ov1 = cmd.get_model(f"frag1 and {ov_ca}")
        if m_ov1.atom:
            overlap_plddt = sum((a.b for a in m_ov1.atom)) / len(m_ov1.atom)

    if has_overlap_1b and o1 <= o2 and g_anchor0 is not None:
        an = f"resi {g_anchor0}-{g_anchor0 + ANCHOR_SLIDE} and name CA"
        super_res = cmd.super(f"frag2 and {an}", f"frag1 and {an}", cycles=0)
    elif has_overlap_1b and o1 <= o2 and ovl_local is not None:
        super_res = cmd.super(
            f"frag2 and {ov_ca}", f"frag1 and {ov_ca}", cycles=0
        )
    else:
        w = min(JUNCTION_ALIGN_W, e_b - s_b, e_a - s_a)
        w = max(1, w)
        super_res = cmd.super(
            f"frag2 and resi {s_b+1}-{s_b+w} and name CA",
            f"frag1 and resi {e_a - w+1}-{e_a} and name CA",
        )

    rmsd, n_al = float(super_res[0]), int(super_res[1])
    print(
        f"  Best-anchor pLDDT (frag1, 51-mer, if any): {anchor_plddt:.2f} (target: >70)"
    )
    print(f"  Full-overlap pLDDT: {overlap_plddt:.2f}")
    print(f"  RMSD on used selection: {rmsd:.3f} Å  ({n_al} CA)")
    if g_anchor0 is not None and anchor_plddt > 70.0 and rmsd < 2.0:
        print("  High-confidence anchor: OK for merge (per validate_and_analyze).")
    else:
        print("  Anchor thresholds not met or abutting; review if needed.")

    stored.scores = []
    if has_overlap_1b and o1 <= o2 and ov_ca != "none":
        cmd.iterate(f"frag1 and {ov_ca}", "stored.scores.append(b)")
    if stored.scores and has_overlap_1b:
        ap = sum(stored.scores) / len(stored.scores)
        print(f"  Avg pLDDT in overlap (iterate, frag1): {ap:.2f}")
    elif has_overlap_1b and o1 <= o2:
        print("  (No pLDDT in overlap; empty selection after renumbering?)")
    if rmsd < 2.0:
        print("  Consistency: PASS (RMSD < 2 Å) — validate.py")
    elif rmsd < 5.0:
        print("  Consistency: WARNING (2–5 Å) — flexible / stitch region?")
    else:
        print("  Consistency: FAIL (RMSD ≥5 Å) — re-check split / overlap")


def validate_all_adjacent_pairs(
    chunks: list[tuple[int, int]],
    out_dirs: list[str],
    anchor_primary: str = "plddt",
    *,
    fold_backend: str = "colabfold",
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
            fold_backend=fold_backend,
        )


def stitch_results(
    chunks: list[tuple[int, int]],
    out_dirs: list[str],
    final_name: str = "final_stitched.pdb",
    anchor_primary: str = "plddt",
    *,
    fold_backend: str = "colabfold",
) -> None:
    other = "rmsd" if anchor_primary == "plddt" else "plddt"
    print(
        f"-> Starting structural stitching (anchor: primary {anchor_primary}, "
        f"secondary {other})..."
    )
    cmd.reinitialize()
    master_obj = "full_protein"

    for i, (s, e) in enumerate(chunks):
        pdb_path = _best_chunk_pdb(out_dirs[i], fold_backend)
        if not pdb_path:
            print(f"Warning: No PDB found for chunk {i}")
            continue

        chunk_obj = f"c{i}"
        cmd.load(pdb_path, chunk_obj)
        cmd.alter(chunk_obj, f"resv += {s}")
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
                        fold_backend=fold_backend,
                    )
                    if ovl
                    else None
                )
                g0 = s_prev + pbe if pbe is not None else None
                ov_start, ov_end = s + 1, last_global
                if g0 is not None:
                    s_sel = f"resi {g0}-{g0+ANCHOR_SLIDE} and name CA"
                    cmd.super(
                        f"{chunk_obj} and {s_sel}", f"{master_obj} and {s_sel}"
                    )
                else:
                    sel = f"resi {ov_start}-{ov_end}"
                    cmd.align(
                        f"{chunk_obj} and {sel}", f"{master_obj} and {sel}"
                    )
                new_residues = f"resi {ov_end + 1}-{e}"
            else:
                w = min(JUNCTION_ALIGN_W, e_prev, e - s)
                w = max(1, w)
                lo_m = e_prev - w + 1
                lo_c = s + 1
                cmd.super(
                    f"{chunk_obj} and resi {lo_c}-{lo_c + w - 1}",
                    f"{master_obj} and resi {lo_m}-{lo_m + w - 1}",
                )
                new_residues = f"resi {s+1}-{e}"
            cmd.create(
                master_obj, f"{master_obj} or ({chunk_obj} and {new_residues})"
            )

        cmd.delete(chunk_obj)

    names = cmd.get_names("objects") or []
    if master_obj not in names:
        print(
            f"ERROR: No structures loaded; cannot write {final_name!r}. "
            "Each chunk directory must contain a prediction PDB "
            "(ColabFold: *_rank_001*.pdb; AlphaFold2: ranked_*.pdb).",
            file=sys.stderr,
        )
        return
    cmd.save(final_name, master_obj)
    print(f"-> SUCCESS: {final_name} generated.")


def rmsd_two_structures_calpha(path_a: str, path_b: str) -> float | None:
    if not os.path.isfile(path_a) or not os.path.isfile(path_b):
        return None
    try:
        cmd.reinitialize()
        cmd.load(path_a, "st_mdl_a")
        cmd.load(path_b, "st_mdl_b")
        out = cmd.super(
            "st_mdl_b and name CA", "st_mdl_a and name CA", cycles=0
        )
        if not out or not out[1] or int(out[1]) == 0:
            return None
        return float(out[0])
    except Exception:
        return None


def print_stitch_modes_summary(
    base: str,
    modes_done: list[str],
) -> None:
    paths = {m: f"{base}_stitched_{m}.pdb" for m in modes_done}
    existing = {m: p for m, p in paths.items() if os.path.isfile(p)}
    if not existing:
        return
    print("--- Stitched model summary ---")
    print("  Modes: " + ", ".join(sorted(existing)))
    print("  Files: " + ", ".join(f"{k}={v}" for k, v in sorted(existing.items())))
    names = list(existing.keys())
    if len(names) == 2:
        a, b = names[0], names[1]
        r = rmsd_two_structures_calpha(existing[a], existing[b])
        if r is not None:
            print(
                f"  Cα RMSD (after super) between the two full stitched models: {r:.3f} Å"
            )
            print(
                "  (Large values mean the two anchor policies yield meaningfully different merges.)"
            )
        else:
            print("  (Could not superpose the two merged PDBs; check structures.)")
    for m, p in existing.items():
        print(f"  -> {m}: {p}")


from split_fold_stitch.plan import (
    build_plan_json,
    chunks_from_plan,
    load_plan_json,
    relativize_plan_paths,
    resolve_plan_paths,
    write_plan_json,
)


def _plan_work_dir(plan_path: str) -> str:
    env = os.environ.get("SPLIT_FOLD_WORK_DIR", "").strip()
    if env:
        return os.path.abspath(env)
    # Default: parent of .split_fold_stitch plan directory.
    return os.path.abspath(os.path.join(os.path.dirname(plan_path), ".."))


def cmd_validate(args: argparse.Namespace) -> int:
    plan = resolve_plan_paths(load_plan_json(args.plan), _plan_work_dir(args.plan))
    chunks = chunks_from_plan(plan)
    fb = plan.get("fold_backend", "colabfold")
    validate_all_adjacent_pairs(
        chunks,
        plan["out_dirs"],
        anchor_primary=plan.get("anchor_primary", "plddt"),
        fold_backend=fb,
    )
    return 0


def cmd_stitch(args: argparse.Namespace) -> int:
    plan = resolve_plan_paths(load_plan_json(args.plan), _plan_work_dir(args.plan))
    chunks = chunks_from_plan(plan)
    fb = plan.get("fold_backend", "colabfold")
    stitch_results(
        chunks,
        plan["out_dirs"],
        final_name=plan["output_pdb"],
        anchor_primary=plan.get("anchor_primary", "plddt"),
        fold_backend=fb,
    )
    return 0


def cmd_summary(args: argparse.Namespace) -> int:
    plan = resolve_plan_paths(load_plan_json(args.plan), _plan_work_dir(args.plan))
    print_stitch_modes_summary(plan["base"], plan["modes"])
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="PyMOL worker for split-fold-stitch.")
    sub = p.add_subparsers(dest="command", required=True)

    p_val = sub.add_parser("validate", help="validate adjacent segment pairs")
    p_val.add_argument("--plan", required=True, help="JSON plan file")
    p_val.set_defaults(func=cmd_validate)

    p_st = sub.add_parser("stitch", help="merge segment PDBs")
    p_st.add_argument("--plan", required=True, help="JSON plan file")
    p_st.set_defaults(func=cmd_stitch)

    p_sum = sub.add_parser("summary", help="print stitched model summary")
    p_sum.add_argument("--plan", required=True, help="JSON plan file")
    p_sum.set_defaults(func=cmd_summary)

    args = p.parse_args(argv)
    return int(args.func(args))


def _entry() -> int:
    if __name__ == "__main__":
        return main()
    # PyMOL `-cq script.py -- args` executes the script with __name__ == "pymol".
    if __name__ == "pymol" and len(sys.argv) > 1:
        return main(sys.argv[1:])
    return 0


if __name__ in ("__main__", "pymol"):
    raise SystemExit(_entry())
