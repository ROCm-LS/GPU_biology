"""JSON plan files exchanged between the host orchestrator and PyMOL worker."""

from __future__ import annotations

import json
import os
from typing import Literal

PlanKind = Literal["validate", "stitch", "summary"]


def plan_json_basename(base: str, kind: PlanKind, mode: str | None = None) -> str:
    """Filename for one plan under ``.split_fold_stitch`` (e.g. ``3013aa_stitch_plddt.json``)."""
    if kind == "summary":
        return f"{base}_summary.json"
    if kind in ("validate", "stitch"):
        if mode not in ("plddt", "rmsd"):
            raise ValueError(f"mode required for kind={kind!r}, got {mode!r}")
        return f"{base}_{kind}_{mode}.json"
    raise ValueError(f"Unknown plan kind {kind!r}")


def plan_json_path(
    plan_dir: str,
    base: str,
    kind: PlanKind,
    mode: str | None = None,
) -> str:
    return os.path.join(plan_dir, plan_json_basename(base, kind, mode))


def build_plan_json(
    *,
    base: str,
    chunks: list[tuple[int, int]],
    out_dirs: list[str],
    output_pdb: str | None = None,
    anchor_primary: str = "plddt",
    modes: list[str] | None = None,
    fold_backend: str = "colabfold",
    plan_mode: str = "default",
) -> dict:
    """
    fold_backend: ``colabfold`` (``*_rank_001*.pdb``) or ``alphafold2`` (``ranked_*.pdb``).
    """
    return {
        "base": base,
        "chunks": [list(c) for c in chunks],
        "out_dirs": out_dirs,
        "output_pdb": output_pdb,
        "anchor_primary": anchor_primary,
        "modes": modes or [],
        "fold_backend": fold_backend,
        "plan_mode": plan_mode,
    }


def write_plan_json(path: str, plan: dict) -> None:
    with open(path, "w") as f:
        json.dump(plan, f, indent=2)


def relativize_plan_paths(plan: dict, work_dir: str) -> dict:
    """Store paths relative to work_dir so container workers can resolve them."""
    root = os.path.abspath(work_dir)

    def rel(p: str | None) -> str | None:
        if p is None:
            return None
        ap = os.path.abspath(p)
        if ap == root:
            return "."
        if ap.startswith(root + os.sep):
            return os.path.relpath(ap, root)
        return p

    out = dict(plan)
    out["base"] = rel(plan["base"]) or plan["base"]
    out["out_dirs"] = [rel(d) or d for d in plan["out_dirs"]]
    if plan.get("output_pdb"):
        out["output_pdb"] = rel(plan["output_pdb"])
    return out


def resolve_plan_paths(plan: dict, work_dir: str) -> dict:
    """Expand relative plan paths to absolute paths on the current filesystem."""
    root = os.path.abspath(work_dir)

    def abs_path(p: str | None) -> str | None:
        if p is None:
            return None
        if os.path.isabs(p):
            return p
        return os.path.join(root, p)

    out = dict(plan)
    out["base"] = abs_path(plan["base"]) or plan["base"]
    out["out_dirs"] = [abs_path(d) or d for d in plan["out_dirs"]]
    if plan.get("output_pdb"):
        out["output_pdb"] = abs_path(plan["output_pdb"])
    return out


def load_plan_json(path: str) -> dict:
    with open(path, "r") as f:
        return json.load(f)


def chunks_from_plan(plan: dict) -> list[tuple[int, int]]:
    return [tuple(pair) for pair in plan["chunks"]]
