"""AlphaFold2 precomputed MSA prep for dual-container orchestration (A3M → .sto)."""

from __future__ import annotations

import os
from typing import Sequence

from split_fold_stitch.container import ContainerRunner, host_to_container
from split_fold_stitch.tiling import (
    _is_msa_file,
    a3m_match_state_count_with_check,
    chunk_stem,
    parse_a3m_file,
    read_sequence_input,
    write_a3m_match_slice,
)

GPU_BIOLOGY_REPO_MOUNT = "/gpu_biology"

MSA_STO_NAMES = (
    "uniref90_hits.sto",
    "mgnify_hits.sto",
    "small_bfd_hits.sto",
)


def resolve_convert_a3m_script_host(scripts_host: str) -> str:
    """Host path to ``convert_colabfold_a3m_to_sto.py`` under the GPU_biology repo."""
    env = os.environ.get("ALPHAFOLD2_CONVERT_A3M_SCRIPT", "").strip()
    if env and os.path.isfile(env):
        return os.path.abspath(env)
    repo_root = os.path.dirname(scripts_host)
    candidate = os.path.join(
        repo_root, "alphafold2", "scripts", "convert_colabfold_a3m_to_sto.py"
    )
    if os.path.isfile(candidate):
        return candidate
    raise FileNotFoundError(
        f"convert_colabfold_a3m_to_sto.py not found at {candidate!r}. "
        "Set ALPHAFOLD2_CONVERT_A3M_SCRIPT or run from a GPU_biology checkout."
    )


def write_query_fasta_from_a3m(a3m_path: str, fasta_path: str) -> None:
    """Write one-sequence FASTA for ``run_alphafold.py`` from the query row in an A3M."""
    header, seq = parse_a3m_file(a3m_path)[0]
    query = "".join(c for c in seq if not c.islower()).replace("-", "")
    if not query:
        raise ValueError(f"A3M query sequence empty after stripping inserts/gaps: {a3m_path!r}")
    name = header.lstrip(">").split()[0] if header.lstrip().startswith(">") else header.split()[0]
    parent = os.path.dirname(os.path.abspath(fasta_path))
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(fasta_path, "w", encoding="utf-8") as out:
        out.write(f">{name}\n{query}\n")


def chunk_msas_ready(output_base: str, chunk_stem: str) -> bool:
    msa_dir = os.path.join(output_base, chunk_stem, "msas")
    return all(os.path.isfile(os.path.join(msa_dir, name)) for name in MSA_STO_NAMES)


def _resolve_a3m_msa_source(
    seq_input: str,
    msa_in: bool,
    colabfold_a3m: str | None,
) -> tuple[list[tuple[str, str]] | None, int | None, str | None]:
    if msa_in:
        records = parse_a3m_file(seq_input)
        n_match = a3m_match_state_count_with_check(records)
        return records, n_match, seq_input
    if colabfold_a3m:
        path = os.path.abspath(colabfold_a3m)
        if not os.path.isfile(path):
            raise FileNotFoundError(f"--colabfold-a3m not found: {path!r}")
        records = parse_a3m_file(path)
        n_match = a3m_match_state_count_with_check(records)
        return records, n_match, path
    return None, None, None


def prepare_dual_container_af2_chunks(
    runner: ContainerRunner,
    *,
    seq_input: str,
    base: str,
    msa_in: bool,
    colabfold_a3m: str | None,
    chunks: Sequence[tuple[int, int]],
    chunk_files: Sequence[str],
    one_segment: bool,
    output_base: str,
) -> list[str]:
    """
    When input is A3M/A2M or FASTA + ``--colabfold-a3m``, slice MSAs per chunk,
    convert each chunk A3M to AlphaFold ``msas/*.sto`` inside the AF2 container,
    and return chunk FASTA paths for ``run_alphafold.py``.
    """
    records, n_match, _full_a3m = _resolve_a3m_msa_source(seq_input, msa_in, colabfold_a3m)
    if records is None or n_match is None:
        return list(chunk_files)

    if not msa_in:
        _header, seq = read_sequence_input(seq_input)
        if len(seq) != n_match:
            raise SystemExit(
                f"FASTA length ({len(seq)}) != ColabFold A3M match columns "
                f"({n_match}); use matching query/MSA files."
            )

    output_base = os.path.abspath(output_base)
    os.makedirs(output_base, exist_ok=True)
    chunk_fastas: list[str] = []

    if one_segment:
        if msa_in:
            a3m_path = os.path.abspath(chunk_files[0])
            stem = os.path.splitext(os.path.basename(a3m_path))[0]
            fasta_path = os.path.join(os.path.dirname(a3m_path), f"{stem}.fasta")
            write_query_fasta_from_a3m(a3m_path, fasta_path)
            _ensure_chunk_sto(runner, a3m_path, output_base, stem)
            chunk_fastas.append(fasta_path)
        else:
            assert colabfold_a3m is not None
            fasta_path = os.path.abspath(chunk_files[0])
            stem = os.path.splitext(os.path.basename(fasta_path))[0]
            a3m_path = os.path.join(os.path.dirname(fasta_path), f"{stem}.a3m")
            write_a3m_match_slice(records, 0, n_match, n_match, a3m_path, part_index=0)
            _ensure_chunk_sto(runner, a3m_path, output_base, stem)
            chunk_fastas.append(fasta_path)
        return chunk_fastas

    base_name = os.path.basename(base)
    for i, (s, e) in enumerate(chunks):
        stem = chunk_stem(base_name, i, s, e)
        if msa_in:
            a3m_path = os.path.abspath(chunk_files[i])
            fasta_path = os.path.join(os.path.dirname(a3m_path), f"{stem}.fasta")
            write_query_fasta_from_a3m(a3m_path, fasta_path)
        else:
            fasta_path = os.path.abspath(chunk_files[i])
            a3m_path = os.path.join(os.path.dirname(fasta_path), f"{stem}.a3m")
            write_a3m_match_slice(records, s, e, n_match, a3m_path, part_index=i)
        _ensure_chunk_sto(runner, a3m_path, output_base, stem)
        chunk_fastas.append(fasta_path)

    return chunk_fastas


def _ensure_chunk_sto(
    runner: ContainerRunner,
    chunk_a3m_host: str,
    output_base: str,
    chunk_stem: str,
) -> None:
    if chunk_msas_ready(output_base, chunk_stem):
        print(f"-> MSAs already present for {chunk_stem!r}; skipping A3M conversion.")
        return
    msa_dir = os.path.join(output_base, chunk_stem, "msas")
    print(f"-> Converting {os.path.basename(chunk_a3m_host)!r} -> {msa_dir}/")
    rc = runner.run_convert_colabfold_a3m_to_sto(chunk_a3m_host, msa_dir)
    if rc != 0:
        raise SystemExit(
            f"convert_colabfold_a3m_to_sto.py failed (exit {rc}) for {chunk_a3m_host!r}."
        )


def input_uses_precomputed_msas(seq_input: str, colabfold_a3m: str | None) -> bool:
    return _is_msa_file(seq_input) or bool(colabfold_a3m)
