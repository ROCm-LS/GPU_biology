"""Pure-Python sequence tiling and chunk file preparation (no PyMOL / ColabFold)."""

from __future__ import annotations

import os

# Tiling: max ~3000 aa per ColabFold run, 1000 aa overlap so consecutive windows share
# the same stretch (1-based: 1–3000 and 2001–5005 for 5005 aa, overlap 2001–3000).
WINDOW_SIZE = 3000
OVERLAP = 1000
MAX_CHUNK_AA = 3012
MIN_OVERLAP = OVERLAP
JUNCTION_ALIGN_W = 200
ANCHOR_SLIDE = 50


def _tiling_window_overlap(max_chunk_aa: int) -> tuple[int, int]:
    """
    Return (window_width, step_overlap) for a given per-segment cap.
    Overlap is reduced automatically when the window is small so stitching stays possible.
    """
    w = min(WINDOW_SIZE, max_chunk_aa)
    o = min(
        OVERLAP,
        w - 1,
        max(ANCHOR_SLIDE + 1, w // 3),
    )
    o = min(o, w - 1)
    if o < 1 and w > 1:
        o = min(50, w - 1)
    return w, o


def _is_msa_file(path: str) -> bool:
    return os.path.splitext(path)[1].lower() in (".a3m", ".a2m")


def colabfold_msa_input_fail_hint() -> str:
    return (
        "A2M/A3M: GPU OOM or XLA issues are often MSA size on one GCD. "
        "Try a shallower a3m (e.g. hhfilter), or colabfold max MSA / template limits "
        "if your build supports them, or use split windows with per-segment FASTA."
    )


def chunk_file_extension(input_path: str) -> str:
    """Use the same suffix as the input (e.g. .fa, .fasta, .a3m); default .fa if none."""
    ext = os.path.splitext(input_path)[1]
    return ext if ext else ".fa"


def chunk_stem(base: str, part_index: int, s: int, e: int) -> str:
    """
    Stem for one segment: {base}_part_{i}_{a}-{b} with 1-based a..b
    (same as print_chunk_plan: half-open [s,e) -> residues s+1 through e).
    """
    return f"{base}_part_{part_index}_{s+1}-{e}"


def _read_fasta_like_sequence(lines: list[str]) -> tuple[str, str]:
    """Multi-line FASTA: first line header, remaining lines concatenated as one sequence."""
    if not lines:
        raise ValueError("Empty input file.")
    header = lines[0].rstrip("\n\r")
    body = "".join(l.strip() for l in lines[1:])
    return header, body


def _read_a3m_query_only(lines: list[str]) -> tuple[str, str]:
    """
    A3M / A2M: the **query** is the first sequence in the file (ColabFold / HH-suite style).
    """
    if not lines:
        raise ValueError("Empty input file.")
    raw = [ln.rstrip("\n\r") for ln in lines]
    i = 0
    while i < len(raw) and (not raw[i].strip() or raw[i].lstrip().startswith("#")):
        i += 1
    while i < len(raw) and not raw[i].lstrip().startswith(">"):
        i += 1
    if i >= len(raw):
        raise ValueError("A3M: no '>' record found.")
    header = raw[i]
    parts: list[str] = []
    j = i + 1
    while j < len(raw) and not raw[j].lstrip().startswith(">"):
        parts.append(raw[j].strip().replace(" ", ""))
        j += 1
    q = "".join(parts)
    if not q:
        raise ValueError("A3M: no query sequence under first header; check format.")
    return header, q


def read_sequence_input(path: str) -> tuple[str, str]:
    """Dispatch on extension: a3m/a2m use query-only read; otherwise FASTA-style join."""
    with open(path, "r") as f:
        lines = f.readlines()
    ext = os.path.splitext(path)[1].lower()
    if ext in (".a3m", ".a2m"):
        return _read_a3m_query_only(lines)
    return _read_fasta_like_sequence(lines)


def parse_a3m_file(path: str) -> list[tuple[str, str]]:
    """All ``>`` records: (header line, one sequence string per line joined)."""
    with open(path, "r") as f:
        lines = f.readlines()
    raw = [ln.rstrip("\n\r") for ln in lines]
    i = 0
    while i < len(raw) and (not raw[i].strip() or raw[i].lstrip().startswith("#")):
        i += 1
    records: list[tuple[str, str]] = []
    while i < len(raw):
        if not raw[i].strip():
            i += 1
            continue
        if not raw[i].lstrip().startswith(">"):
            i += 1
            continue
        h = raw[i]
        i += 1
        parts: list[str] = []
        while i < len(raw) and not raw[i].lstrip().startswith(">"):
            parts.append(raw[i].strip().replace(" ", ""))
            i += 1
        body = "".join(parts)
        if body:
            records.append((h, body))
    if not records:
        raise ValueError("A3M: no sequence records in file.")
    return records


def iter_a3m_match_blocks(seq: str):
    """Yield HH a3m match blocks (match char + following insert lowercase)."""
    i = 0
    n = len(seq)
    while i < n:
        if seq[i].islower():
            i += 1
            continue
        j = i + 1
        while j < n and seq[j].islower():
            j += 1
        yield seq[i:j]
        i = j


def a3m_match_state_count_with_check(records: list[tuple[str, str]]) -> int:
    """All rows must have the same number of match blocks; return that count."""
    b0 = list(iter_a3m_match_blocks(records[0][1]))
    m = len(b0)
    for idx, (_h, s) in enumerate(records[1:], start=1):
        bm = list(iter_a3m_match_blocks(s))
        if len(bm) != m:
            raise ValueError(
                f"A3M: row 0 has {m} match states, row {idx} has {len(bm)}; "
                "inconsistent MSA (cannot take column-matched segments)."
            )
    return m


def write_a3m_match_slice(
    records: list[tuple[str, str]],
    s: int,
    e: int,
    n_match: int,
    out_path: str,
    part_index: int,
) -> None:
    """Write a valid sub-a3m: same match-state range [s, e) for every row."""
    if s < 0 or e > n_match or s >= e:
        raise ValueError(
            f"match slice [{s}, {e}) out of range for n_match={n_match!r}."
        )
    with open(out_path, "w") as f:
        for h, srow in records:
            bl = list(iter_a3m_match_blocks(srow))
            if len(bl) != n_match:
                raise ValueError("internal: row length mismatch in write_a3m_match_slice")
            sub = "".join(bl[s:e])
            hline = h.rstrip()
            if f"_p{part_index}" not in hline:
                hline = f"{hline}_p{part_index}"
            f.write(f"{hline}\n{sub}\n")


def get_chunks(total_len: int, *, max_chunk_aa: int = MAX_CHUNK_AA) -> list[tuple[int, int]]:
    """Overlapping windows covering [0, total_len)."""
    if total_len <= 0:
        return []
    w, o = _tiling_window_overlap(max_chunk_aa)
    if w <= o and total_len > max_chunk_aa:
        raise ValueError(
            f"tiling: window {w} must exceed overlap {o}; try a larger --max-chunk-aa"
        )
    chunks: list[tuple[int, int]] = []
    start = 0
    while start < total_len:
        remaining = total_len - start
        if remaining <= max_chunk_aa:
            chunks.append((start, total_len))
            break
        end = start + w
        chunks.append((start, end))
        start = end - o
    return chunks


def validate_chunk_plan(
    chunks: list[tuple[int, int]],
    total_len: int,
    *,
    max_chunk_aa: int = MAX_CHUNK_AA,
    min_adjacent_overlap: int = MIN_OVERLAP,
) -> None:
    """Check full cover, per-segment length, and enough overlap (no gaps)."""
    if not chunks:
        if total_len == 0:
            return
        raise ValueError("No chunks for non-empty sequence.")
    s0, _e0 = chunks[0]
    if s0 != 0:
        raise ValueError("First chunk should start at residue 0 (0-based).")
    s_last, e_last = chunks[-1]
    if e_last != total_len:
        raise ValueError("Last chunk does not end at L.")
    for i, (a, b) in enumerate(chunks):
        n_aa = b - a
        if n_aa > max_chunk_aa:
            raise ValueError(
                f"Segment {i} has length {n_aa}, must be <= {max_chunk_aa} a.a. "
                f"(0-based [{a}, {b}))."
            )
        if n_aa <= 0:
            raise ValueError(f"Empty segment {i}.")
    for i in range(1, len(chunks)):
        s_prev, e_prev = chunks[i - 1]
        s_i, e_i = chunks[i]
        overlap_len = e_prev - s_i
        if s_i < s_prev:
            raise ValueError("Chunk start indices went backwards; invalid tiling.")
        if s_i > e_prev:
            raise ValueError(
                f"Gap between segment {i - 1} and {i} (0-based: prev end {e_prev}, "
                f"next start {s_i})."
            )
        if s_i < e_prev and overlap_len < min_adjacent_overlap:
            raise ValueError(
                f"Overlap between chunk {i - 1} and {i} is {overlap_len} a.a.; "
                f"need at least {min_adjacent_overlap} (tiling / OVERLAP / --max-chunk-aa)."
            )


def print_chunk_plan(chunks: list[tuple[int, int]]) -> None:
    print("Segment plan (1-based residue numbers):")
    for i, (s, e) in enumerate(chunks):
        ovl = ""
        if i > 0:
            _sp, ep = chunks[i - 1]
            if s < ep:
                ovl = f"  (overlap with previous: {ep - s} aa)"
        print(f"  part {i}: {s + 1}-{e}  ({e - s} aa){ovl}")


def prepare_chunk_inputs(
    seq_input: str,
    *,
    max_chunk_aa: int | None = None,
) -> tuple[
    str,
    str,
    bool,
    list[tuple[int, int]],
    list[str],
    list[str],
    bool,
]:
    """
    Parse input, build tiling plan, write per-segment files when needed.

    Returns:
        base, header, msa_in, chunks, chunk_files, out_dirs, one_segment
    """
    ext = chunk_file_extension(seq_input)
    base = os.path.splitext(seq_input)[0]
    msa_in = _is_msa_file(seq_input)
    a3m_records: list[tuple[str, str]] | None = None
    a3m_n_match: int | None = None
    if msa_in:
        a3m_records = parse_a3m_file(seq_input)
        a3m_n_match = a3m_match_state_count_with_check(a3m_records)
        header, seq = a3m_records[0]
        total_len = a3m_n_match
    else:
        header, seq = read_sequence_input(seq_input)
        total_len = len(seq)

    mca = max_chunk_aa if max_chunk_aa is not None else MAX_CHUNK_AA
    if mca < ANCHOR_SLIDE * 2:
        raise SystemExit(
            f"max-chunk-aa ({mca}) is too small; use at least ~{ANCHOR_SLIDE * 2} for overlap/anchors."
        )
    tw, tov = _tiling_window_overlap(mca)

    chunks = get_chunks(total_len, max_chunk_aa=mca)
    validate_chunk_plan(
        chunks, total_len, max_chunk_aa=mca, min_adjacent_overlap=tov
    )
    print_chunk_plan(chunks)
    if mca != MAX_CHUNK_AA or len(chunks) > 1:
        print(
            f"  (tiling: max {mca} aa per segment, window {tw} aa, adjacent overlap {tov} aa)"
        )
    if msa_in and len(chunks) == 1 and max_chunk_aa is None:
        print(
            "  (By query the input fits a single ColabFold run. For A2M/A3M OOM, "
            "set a smaller per-run cap, e.g.  --max-chunk-aa 300  to split; multi-segment "
            "A2M/A3M write column-matched sub-MSA files per part.)"
        )
    if msa_in and len(chunks) > 1:
        print(
            "  (Multi-segment A2M/A3M: each part file is a valid A3M slice, same MSA row count as input.)"
        )
    one_segment = len(chunks) == 1

    chunk_files: list[str] = []
    out_dirs: list[str] = []
    if one_segment:
        out_dirs = [f"{base}_output"]
        chunk_files = [seq_input]
    else:
        for i, (s, e) in enumerate(chunks):
            stem = chunk_stem(base, i, s, e)
            fn = f"{stem}{ext}"
            if msa_in and a3m_records is not None and a3m_n_match is not None:
                write_a3m_match_slice(
                    a3m_records, s, e, a3m_n_match, fn, part_index=i
                )
            else:
                with open(fn, "w") as f:
                    f.write(f"{header}_p{i}\n{seq[s:e]}\n")
            chunk_files.append(fn)
            out_dirs.append(f"{stem}_output")

    return base, header, msa_in, chunks, chunk_files, out_dirs, one_segment
