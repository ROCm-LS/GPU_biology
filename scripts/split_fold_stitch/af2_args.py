"""Build and merge ``run_alphafold.py`` flag lists from a database root."""

from __future__ import annotations

import os
from typing import Sequence

# Flags that must be set for a typical full-DB or reduced-DB run (absl rejects None).
_DATABASE_PATH_FLAGS = (
    "uniref90_database_path",
    "mgnify_database_path",
    "pdb70_database_path",
    "template_mmcif_dir",
    "obsolete_pdbs_path",
)
_OTHER_REQUIRED_FLAGS = (
    "max_template_date",
    "use_gpu_relax",
    "model_preset",
    "db_preset",
    "data_dir",
)


def _normalize_flag_name(name: str) -> str:
    return name.lstrip("-").replace("-", "_")


def parse_flags(argv: Sequence[str]) -> tuple[dict[str, str], list[str]]:
    """Parse ``--key=value`` and ``--key value`` tokens into a flag dict."""
    flags: dict[str, str] = {}
    passthrough: list[str] = []
    i = 0
    argv_list = list(argv)
    while i < len(argv_list):
        tok = argv_list[i]
        if not tok.startswith("--"):
            passthrough.append(tok)
            i += 1
            continue
        if "=" in tok:
            name, val = tok[2:].split("=", 1)
            flags[_normalize_flag_name(name)] = val
            i += 1
            continue
        name = tok[2:]
        norm = _normalize_flag_name(name)
        if i + 1 < len(argv_list) and not argv_list[i + 1].startswith("--"):
            flags[norm] = argv_list[i + 1]
            i += 2
        else:
            flags[norm] = "true"
            i += 1
    return flags, passthrough


def flags_to_argv(flags: dict[str, str], passthrough: Sequence[str]) -> list[str]:
    out = [f"--{key}={value}" for key, value in flags.items()]
    out.extend(passthrough)
    return out


def build_database_path_flags(data_dir: str, db_preset: str) -> dict[str, str]:
    root = data_dir.rstrip("/")
    flags = {
        "data_dir": root,
        "uniref90_database_path": f"{root}/uniref90/uniref90.fasta",
        "mgnify_database_path": f"{root}/mgnify/mgy_clusters_2022_05.fa",
        "pdb70_database_path": f"{root}/pdb70/pdb70",
        "template_mmcif_dir": f"{root}/pdb_mmcif/mmcif_files",
        "obsolete_pdbs_path": f"{root}/pdb_mmcif/obsolete.dat",
    }
    if db_preset == "full_dbs":
        flags["bfd_database_path"] = (
            f"{root}/bfd/bfd_metaclust_clu_complete_id30_c90_final_seq.sorted_opt"
        )
        # Required by run_alphafold.py when db_preset=full_dbs (HHblits); path matches
        # alphafold/docker/run_docker.py and download_all_databases.sh layout.
        flags["uniref30_database_path"] = f"{root}/uniref30/UniRef30_2021_03"
    else:
        flags["small_bfd_database_path"] = (
            f"{root}/small_bfd/bfd-first_non_consensus_sequences.fasta"
        )
    return flags


def build_default_af2_args(
    data_dir: str,
    *,
    db_preset: str = "reduced_dbs",
    use_precomputed_msas: bool = True,
) -> list[str]:
    """Defaults aligned with ``alphafold2/scripts/run_af2.sh`` (reduced_dbs, monomer)."""
    flags = build_database_path_flags(data_dir, db_preset)
    flags.update(
        {
            "model_preset": "monomer",
            "db_preset": db_preset,
            "max_template_date": "1900-01-01" if use_precomputed_msas else "2023-05-14",
            "use_gpu_relax": "false",
            "use_precomputed_msas": "true" if use_precomputed_msas else "false",
        }
    )
    return flags_to_argv(flags, [])


def _needs_database_defaults(user_flags: dict[str, str]) -> bool:
    for key in _DATABASE_PATH_FLAGS:
        if key not in user_flags:
            return True
    for key in _OTHER_REQUIRED_FLAGS:
        if key not in user_flags:
            return True
    if user_flags.get("db_preset", "reduced_dbs") == "full_dbs":
        if "bfd_database_path" not in user_flags:
            return True
        if "uniref30_database_path" not in user_flags:
            return True
    elif "small_bfd_database_path" not in user_flags:
        return True
    return False


def data_dir_from_argv(argv: Sequence[str]) -> str | None:
    """Return ``--data_dir`` from a resolved ``run_alphafold.py`` argv list."""
    flags, _ = parse_flags(argv)
    return flags.get("data_dir")


def ensure_singularity_host_bind(
    singularity_bind: list[str], host_path: str, work_dir: str
) -> str | None:
    """Add ``host:host`` to *singularity_bind* if *host_path* is outside *work_dir*."""
    host = os.path.abspath(host_path)
    root = os.path.abspath(work_dir)
    if host == root or host.startswith(root + os.sep):
        return None
    spec = f"{host}:{host}"
    if spec in singularity_bind:
        return None
    singularity_bind.append(spec)
    return spec


def resolve_run_alphafold_extra(
    extra: Sequence[str],
    *,
    data_dir: str,
    db_preset: str | None = None,
    use_precomputed_msas: bool | None = None,
) -> list[str]:
    """Fill missing ``run_alphafold.py`` flags from *data_dir*; user *extra* wins on conflict."""
    user_flags, passthrough = parse_flags(extra)
    if not extra and not data_dir:
        return list(extra)

    effective_data_dir = user_flags.get("data_dir") or data_dir
    if not effective_data_dir:
        return list(extra)

    effective_db_preset = user_flags.get("db_preset") or db_preset or "reduced_dbs"

    precomputed = use_precomputed_msas
    if precomputed is None:
        if "use_precomputed_msas" in user_flags:
            precomputed = user_flags["use_precomputed_msas"].lower() in (
                "true",
                "1",
                "yes",
            )
        else:
            precomputed = False

    if not _needs_database_defaults(user_flags):
        return list(extra)

    defaults = build_database_path_flags(effective_data_dir, effective_db_preset)
    defaults.update(
        {
            "model_preset": "monomer",
            "db_preset": effective_db_preset,
            "max_template_date": "1900-01-01" if precomputed else "2023-05-14",
            "use_gpu_relax": "false",
            "use_precomputed_msas": "true" if precomputed else "false",
        }
    )
    merged = {**defaults, **user_flags}
    return flags_to_argv(merged, passthrough)
