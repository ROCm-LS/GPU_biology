#!/usr/bin/env python3
"""
Split, fold (ColabFold), and stitch long sequences **inside one Docker image** that has
both ``colabfold_batch`` and ``pymol-open-source`` (typically installed with root in the image).

For a **host** that launches separate ColabFold and PyMOL containers, use
``split_and_fold_segments_colabfold.py`` in this directory instead.

See the repo ``README.md`` for dual-container vs single-container workflows.
"""

import argparse
import os, subprocess, sys, glob
from concurrent.futures import ThreadPoolExecutor, as_completed
from pymol import cmd, stored

_script_dir = os.path.dirname(os.path.abspath(__file__))
if _script_dir not in sys.path:
    sys.path.insert(0, _script_dir)
from split_fold_stitch.jax_rocm_env import jax_xla_env_for_fold_subprocess
from split_fold_stitch.tiling import plan_tiling, print_chunk_plan as print_tiling_plan

# Tie-breaks for pLDDT (primary) vs RMSD (secondary) window selection
_PDDT_TIE: float = 1e-4
_RMSD_TIE: float = 1e-6

def _parse_hip_visible_devices() -> list[int]:
    """
    Comma-separated integers, e.g. 0,1,2. Invalid tokens are skipped; empty
    or whitespace-only value yields [].
    """
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

def _discover_gpu_ids() -> list[int]:
    """All GPUs: count of GUID lines in `rocm-smi -i` (no HIP set in the parent)."""
    try:
        out = subprocess.check_output(
            "rocm-smi -i|grep GUID|wc -l",
            shell=True,
            text=True,
            stderr=subprocess.STDOUT,
            timeout=30,
        )
    except (FileNotFoundError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return [0]
    try:
        n = int(out.strip())
    except ValueError:
        return [0]
    if n > 0:
        return list(range(n))
    return [0]

def _init_gpu_ids() -> list[int]:
    if "HIP_VISIBLE_DEVICES" in os.environ:
        return _parse_hip_visible_devices()
    return _discover_gpu_ids()

# Honor HIP in the parent environment; else enumerate devices via rocm-smi; else a single GCD
GPU_IDS = _init_gpu_ids()
if not GPU_IDS:
    print(
        "Warning: no GPU indices (empty HIP_VISIBLE_DEVICES?); using [0].",
        file=sys.stderr,
    )
    GPU_IDS = [0]

# Tiling: max ~3000 aa per ColabFold run, 1000 aa overlap so consecutive windows share
# the same stretch (1-based: 1–3000 and 2001–5005 for 5005 aa, overlap 2001–3000).
WINDOW_SIZE = 3000
# Next segment starts at (previous_end - OVERLAP) + 1 in 1-based coords, e.g. after 1–3000
# the next window is 2001–… (overlap 2001–3000).
OVERLAP = 1000
# Hard cap: each segment must be < 3013 a.a. (e.g. last run 1–3000 and next 2001–5005 = 3005 a.a.)
MAX_CHUNK_AA = 3012
MIN_OVERLAP = OVERLAP
JUNCTION_ALIGN_W = 200  # for abutting chunks: last/first W residues to superpose
# Sliding 51-residue pLDDT window (r..r+50) in local numbering; from find_best_anchor.
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


def _colabfold_msa_input_fail_hint() -> str:
    return (
        "A2M/A3M: GPU OOM or XLA issues are often MSA size on one GCD. "
        "Try a shallower a3m (e.g. hhfilter), or colabfold max MSA / template limits "
        "if your build supports them, or use split windows with per-segment FASTA."
    )


def _chunk_file_extension(input_path: str) -> str:
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

    Many a3m files have a leading comment line (``# length ...``) before the first ``>``
    header, and the next line is *another* small ``>...`` (e.g. ``>101``) before the
    long query line. A naive "line0 + line1" read would use ``>101`` (4 characters)
    as the sequence. We skip leading ``#`` lines, then take the first ``>`` record
    and all sequence lines until the next ``>`` (ignoring the MSA block).
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
    """
    All ``>`` records: (header line, one sequence string per line joined).
    Skips the same leading ``#``/blank run as the query reader.
    """
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
    """
    One HH a3m block = one **match** character (A-Z, gap ``-``/``.``) plus
    any following **insert** letters (a–z) until the next match. Lowercase
    is never a new match: it is attached to the **previous** match.
    Orphan leading lowercase is skipped (rare, invalid).
    """
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
    """
    Write a **valid** sub-a3m: same match-state range [s, e) for every row
    (half-open 0-based, same convention as ``chunks``).
    """
    if s < 0 or e > n_match or s >= e:
        raise ValueError(
            f"match slice [{s}, {e}) out of range for n_match={n_match!r}."
        )
    # No leading ``#`` line: AlphaFold's parse_a3m -> parse_fasta does not skip comments, so
    # a # line causes "list index out of range" when appending to sequence before any ``>``.
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
    """
    Overlapping windows covering [0, total_len). The last run may be up to
    ``max_chunk_aa`` residues. Interior windows use ``_tiling_window_overlap``.
    """
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


def find_best_anchor(out_dir: str, overlap_start: int, overlap_end: int) -> int | None:
    """
    Scan the previous chunk's output for the best sliding 51-mer (r..r+50) pLDDT
    in local residue numbering (1-based, same as the standalone find_best_anchor.py).
    """
    cmd.reinitialize()
    matches = glob.glob(os.path.join(out_dir, "*_rank_001*.pdb"))
    if not matches:
        return None
    cmd.load(matches[0], "ref")
    if overlap_end - overlap_start <= ANCHOR_SLIDE + 1:
        cmd.delete("ref")
        return None
    best_avg = 0.0
    best_res = overlap_start
    for r in range(overlap_start, overlap_end - ANCHOR_SLIDE):
        model = cmd.get_model(
            f"ref and resi {r}-{r+ANCHOR_SLIDE} and name CA"
        )
        if not model.atom:
            continue
        scores = [a.b for a in model.atom]
        avg = sum(scores) / len(scores)
        if avg > best_avg:
            best_avg = avg
            best_res = r
    if "ref" in (cmd.get_names("objects") or []):
        try:
            cmd.delete("ref")
        except Exception:
            pass
    print("--- Anchor discovery (previous-chunk local numbering) ---")
    print(
        f"  Best window at residues: {best_res}-{best_res+ANCHOR_SLIDE} "
        f" (avg pLDDT: {best_avg:.2f})"
    )
    return best_res


def find_best_anchor_silent(
    out_dir: str, overlap_start: int, overlap_end: int
) -> int | None:
    """No reinitialize: loads a temp object, scans, deletes. For use inside larger PyMOL flows."""
    matches = glob.glob(os.path.join(out_dir, "*_rank_001*.pdb"))
    if not matches or overlap_end - overlap_start <= ANCHOR_SLIDE + 1:
        return None
    nm = "scan_ref"
    for name in list(cmd.get_names("objects") or []):
        if name == nm:
            try:
                cmd.delete(nm)
            except Exception:
                break
    cmd.load(matches[0], nm)
    best_avg = 0.0
    best_res: int | None = None
    try:
        for r in range(overlap_start, overlap_end - ANCHOR_SLIDE):
            model = cmd.get_model(f"{nm} and resi {r}-{r+ANCHOR_SLIDE} and name CA")
            if not model.atom:
                continue
            scores = [a.b for a in model.atom]
            avg = sum(scores) / len(scores)
            if best_res is None or avg > best_avg:
                best_avg = avg
                best_res = r
    finally:
        try:
            if nm in (cmd.get_names("objects") or []):
                cmd.delete(nm)
        except Exception:
            pass
    return best_res


def _anchor_window_is_better(
    plddt_new: float,
    rmsd_new: float,
    plddt_cur: float,
    rmsd_cur: float,
    primary: str,
) -> bool:
    """
    True if the new 51-mer (pLDDT, RMSD after super) is preferred: primary wins,
    secondary breaks ties. primary is ``plddt`` or ``rmsd``.
    """
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
) -> int | None:
    """
    Slide 51-mer in overlap: score each with mean pLDDT (on frag a) and RMSD after
    superposing frag b -> frag a on the same window. ``primary`` decides first,
    the other is tie-break. Does not cmd.reinitialize (safe inside stitch/validate),
    and removes temporary load objects.
    """
    f1 = glob.glob(os.path.join(out_dir_a, "*_rank_001*.pdb"))
    f2 = glob.glob(os.path.join(out_dir_b, "*_rank_001*.pdb"))
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
        cmd.load(f1[0], na)
        cmd.load(f2[0], nb)
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
) -> None:
    """
    Merged from validate.py + validate_and_analyze.py: pLDDT, RMSD on
    the best 51-mer in overlap (when found), and validate.py-style thresholds.
    """
    f1 = glob.glob(os.path.join(out_dir_a, "*_rank_001*.pdb"))
    f2 = glob.glob(os.path.join(out_dir_b, "*_rank_001*.pdb"))
    if not f1 or not f2:
        print(
            f"--- Pair {pair_index - 1} vs {pair_index}: "
            f"missing rank_001 PDB ---"
        )
        return

    ovl_local = _chunk_overlap_local_in_prev(s_a, e_a, s_b, e_b)
    pbe, g_anchor0 = None, None
    if ovl_local is not None:
        pbe = find_best_anchor_by_pair_silent(
            out_dir_a, out_dir_b, s_a, s_b, ovl_local, primary=anchor_primary
        )
        if pbe is not None:
            g_anchor0 = s_a + pbe

    has_overlap_1b = s_b < e_a
    o1, o2 = s_b + 1, e_a
    ovl_lbl = f"overlap 1-based {o1}–{o2}" if has_overlap_1b and o1 <= o2 else "abutting (no overlap window)"

    cmd.reinitialize()
    cmd.load(f1[0], "frag1")
    cmd.load(f2[0], "frag2")
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
    print(
        f"  Full-overlap pLDDT: {overlap_plddt:.2f}"
    )
    print(
        f"  RMSD on used selection: {rmsd:.3f} Å  ({n_al} CA)"
    )
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
        )



def _run_one_colabfold_chunk(
    i: int,
    fasta: str,
    out_dir: str,
    gpu_id: int,
    max_concurrent: int,
    *,
    colabfold_batch_extra: list[str],
) -> tuple[int, int]:
    os.makedirs(out_dir, exist_ok=True)
    env = os.environ.copy()
    env["HIP_VISIBLE_DEVICES"] = str(gpu_id)
    env.update(
        jax_xla_env_for_fold_subprocess(
            image_hints="",
            enable_rocm_732_triton_gemm_workaround=True,
        )
    )
    print(
        f"-> Running chunk {i} on GCD {gpu_id} (up to {max_concurrent} colabfold job(s) in parallel)..."
    )
    cmd_list = [
        "colabfold_batch",
        *colabfold_batch_extra,
        "--disable-unified-memory",
        fasta,
        out_dir,
    ]
    p = subprocess.run(
        cmd_list,
        env=env,
        check=False,
    )
    return i, p.returncode


def all_chunk_folds_done(out_dirs: list[str]) -> bool:
    """True if every chunk dir has at least one *_rank_001*.pdb (ColabFold result present)."""
    for d in out_dirs:
        if not glob.glob(os.path.join(d, "*_rank_001*.pdb")):
            return False
    return bool(out_dirs)


def run_parallel(
    chunk_fastas: list[str],
    out_dirs: list[str],
    *,
    msa_input: bool = False,
    colabfold_batch_extra: list[str] | None = None,
) -> list[str]:
    """
    At most len(GPU_IDS) colabfold_batch jobs at a time. Extra chunks wait
    in a thread pool; they are not all spawned at once (avoids piling many
    jobs on the same GPU and OOM). With a single GCD, chunks run one after another.
    out_dirs[i] must be the colabfold output path for chunk_fastas[i] (e.g. stem from
    the chunk file + ``_output`` so names stay aligned with segment range).
    If ``msa_input`` (original input was A2M/A3M), non-zero colabfold exits get an
    extra one-line hint about MSA / single-GCD memory.
    """
    n_gpus = max(1, len(GPU_IDS))
    n = len(chunk_fastas)
    if n != len(out_dirs):
        raise ValueError("chunk_fastas and out_dirs must be the same length")
    cfe = colabfold_batch_extra if colabfold_batch_extra is not None else []

    def work(i, fasta):
        gpu_id = GPU_IDS[i % n_gpus]
        return _run_one_colabfold_chunk(
            i, fasta, out_dirs[i], gpu_id, n_gpus, colabfold_batch_extra=cfe
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
                    print("  " + _colabfold_msa_input_fail_hint(), file=sys.stderr)
    print("-> All colabfold_batch jobs finished.")
    return out_dirs

def stitch_results(
    chunks,
    out_dirs,
    final_name: str = "final_stitched.pdb",
    anchor_primary: str = "plddt",
) -> None:
    other = "rmsd" if anchor_primary == "plddt" else "plddt"
    print(
        f"-> Starting structural stitching (anchor: primary {anchor_primary}, "
        f"secondary {other})..."
    )
    cmd.reinitialize()
    master_obj = "full_protein"
    
    for i, (s, e) in enumerate(chunks):
        pdb_match = glob.glob(os.path.join(out_dirs[i], "*_rank_001*.pdb"))
        if not pdb_match:
            print(f"Warning: No PDB found for chunk {i}")
            continue
        
        chunk_obj = f"c{i}"
        cmd.load(pdb_match[0], chunk_obj)
        
        # Shift residue numbering to match global sequence
        cmd.alter(chunk_obj, f"resv += {s}")
        cmd.sort()
        
        if i == 0:
            cmd.create(master_obj, chunk_obj)
        else:
            s_prev, e_prev = chunks[i - 1]
            last_global = e_prev  # 1-based last residue of previous segment
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
                    )
                    if ovl
                    else None
                )
                g0 = s_prev + pbe if pbe is not None else None
                ov_start, ov_end = s + 1, last_global
                if g0 is not None:
                    s_sel = (
                        f"resi {g0}-{g0+ANCHOR_SLIDE} and name CA"
                    )
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
                # Abutting: superpose C-term of model to N-term of this chunk
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
            "Each chunk must have a ColabFold *_rank_001*.pdb in its _output directory.",
            file=sys.stderr,
        )
        return
    cmd.save(final_name, master_obj)
    print(f"-> SUCCESS: {final_name} generated.")


def report_rmsd_vs_stitched_in_overlap(
    stitched_path: str,
    chunks: list[tuple[int, int]],
    out_dirs: list[str],
) -> None:
    """
    Optional: not called from main. After merge, the stitched PDB is one model, so the
    usual pair RMSD in overlap
    is not a property of the file alone. This loads the stitched result plus each
    source rank_001, renumbers to global, and for each 1..N-1 interface reports
    the Cα RMSD after superposing the previous- or current-fragment model onto
    the stitched model on the *overlap* (same for both, reference = stitched).
    That measures how the final merge deviates from each local prediction in the
    join region, not a cross-RMSD between the two unmerged frags.
    """
    if len(chunks) < 2:
        return
    if not os.path.isfile(stitched_path):
        print(
            f"  (Stitched file missing: {stitched_path}; skip overlap RMSD vs stitch.)",
            file=sys.stderr,
        )
        return
    ovl_tmpl = "resi {0}-{1} and name CA"
    print("--- Cα RMSD: each source model vs stitched, after super on overlap ---")
    for i in range(1, len(chunks)):
        s_a, e_a = chunks[i - 1]
        s_b, e_b = chunks[i]
        if s_b >= e_a:
            print(
                f"  interface {i - 1}–{i}: abutting; no overlap RMSD vs stitch."
            )
            continue
        o1, o2 = s_b + 1, e_a
        f_a = glob.glob(
            os.path.join(out_dirs[i - 1], "*_rank_001*.pdb")
        )
        f_b = glob.glob(os.path.join(out_dirs[i], "*_rank_001*.pdb"))
        ovl = ovl_tmpl.format(o1, o2)

        for label, s_off, f_list in (
            (f"part {i - 1} (earlier in sequence)", s_a, f_a),
            (f"part {i}   (later in sequence)", s_b, f_b),
        ):
            if not f_list:
                odir = out_dirs[i - 1] if s_off == s_a else out_dirs[i]
                print(f"  {label}: no rank_001 in {odir}")
                continue
            cmd.reinitialize()
            cmd.load(stitched_path, "S")
            cmd.load(f_list[0], "F")
            cmd.alter("F", f"resv += {s_off}")
            cmd.sort()
            try:
                sr = cmd.super(f"F and {ovl}", f"S and {ovl}", cycles=0)
                r = float(sr[0])
                n = int(sr[1])
            except Exception as e:
                print(
                    f"  interface {i - 1}–{i} {label}: could not super ({e})"
                )
                continue
            print(
                f"  interface {i - 1}–{i}  overlap 1-based {o1}–{o2}  —  "
                f"RMSD to stitched: {r:.3f} Å  ({n} Cα)  ({label})"
            )


def rmsd_two_structures_calpha(path_a: str, path_b: str) -> float | None:
    """Cα RMSD (after super) between two full PDBs; reinitialize first."""
    if not os.path.isfile(path_a) or not os.path.isfile(path_b):
        return None
    try:
        cmd.reinitialize()
        # Avoid object names "A"/"B" — single letters are reserved in PyMOL selections.
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
    """End-of-pipeline: list stitched PDB paths; if two policy outputs exist, compare them."""
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
            print("  (Large values mean the two anchor policies yield meaningfully different merges.)")
        else:
            print("  (Could not superpose the two merged PDBs; check structures.)")
    for m, p in existing.items():
        print(f"  -> {m}: {p}")


def _run_colabfold_stage(
    seq_input: str,
    msa_in: bool,
    chunk_files: list[str],
    out_dirs: list[str],
    skip_colabfold: bool,
    single_original_input: bool,
    *,
    colabfold_batch_extra: list[str],
) -> list[str]:
    """
    Run colabfold_batch for each (chunk file, out dir) pair, or skip if asked.
    ``single_original_input``: one segment, folding ``seq_input`` as-is (no per-part file).
    """
    if skip_colabfold:
        if not all_chunk_folds_done(out_dirs):
            raise SystemExit(
                f"--skip-colabfold: missing *_rank_001*.pdb in at least one of {out_dirs!r}"
            )
        print("-> Skipping colabfold_batch (existing output dirs).")
        return out_dirs
    if all_chunk_folds_done(out_dirs):
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
                print("  " + _colabfold_msa_input_fail_hint(), file=sys.stderr)
            raise SystemExit(1)
    else:
        _ = run_parallel(
            chunk_files,
            out_dirs,
            msa_input=msa_in,
            colabfold_batch_extra=colabfold_batch_extra,
        )
    if not all_chunk_folds_done(out_dirs):
        err = (
            "ColabFold did not write a *_rank_001*.pdb in every output directory. "
            "See colabfold_batch errors above. A frequent cause is a NumPy / TensorFlow "
            "binary mismatch: use TensorFlow 2.16+ with NumPy 2.x (e.g. `pip install "
            "'tensorflow-cpu>=2.16.2'`) so `import tensorflow` works in the same "
            "interpreter as PyMOL/pandas."
        )
        if msa_in:
            err = err + " " + _colabfold_msa_input_fail_hint()
        raise SystemExit(err)
    return out_dirs


def main(
    seq_input: str,
    *,
    stitch_modes: str = "both",
    skip_colabfold: bool = False,
    validate_adjacent_segments: bool = False,
    max_chunk_aa: int | None = None,
    plan_mode: str = "default",
    colabfold_batch_extra: list[str] | None = None,
) -> None:
    ext = _chunk_file_extension(seq_input)
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
    chunks, tw, tov, mode_used = plan_tiling(
        total_len, max_chunk_aa=mca, plan_mode=plan_mode
    )
    validate_chunk_plan(
        chunks, total_len, max_chunk_aa=mca, min_adjacent_overlap=tov
    )
    print_tiling_plan(chunks, plan_mode=mode_used)
    if mca != MAX_CHUNK_AA or len(chunks) > 1 or mode_used != "default":
        print(
            f"  (tiling: max {mca} aa per segment, window {tw} aa, "
            f"adjacent overlap {tov} aa, plan mode {mode_used})"
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

    # 2. ColabFold
    out_dirs = _run_colabfold_stage(
        seq_input=seq_input,
        msa_in=msa_in,
        chunk_files=chunk_files,
        out_dirs=out_dirs,
        skip_colabfold=skip_colabfold,
        single_original_input=one_segment,
        colabfold_batch_extra=colabfold_batch_extra or [],
    )

    if one_segment:
        print(
            f"-> Single segment: ColabFold result in {out_dirs[0]!r}. "
            "Skipping adjacent-segment validation, stitch, and merge summary."
        )
        if validate_adjacent_segments:
            print("  (Note: --validate-adjacent-segments is ignored when there is only one segment.)")
        return

    if stitch_modes == "both":
        mode_list: list[str] = ["plddt", "rmsd"]
    else:
        mode_list = [stitch_modes]

    if validate_adjacent_segments:
        for mo in mode_list:
            other = "rmsd" if mo == "plddt" else "plddt"
            print(
                f"\n### Pre-stitch validation (anchor: primary {mo}, secondary {other}) ###\n"
            )
            validate_all_adjacent_pairs(chunks, out_dirs, anchor_primary=mo)

    for mo in mode_list:
        out_pdb = f"{base}_stitched_{mo}.pdb"
        print(f"\n### Stitch: anchor policy {mo} -> {out_pdb} ###\n")
        stitch_results(chunks, out_dirs, final_name=out_pdb, anchor_primary=mo)

    print_stitch_modes_summary(base, mode_list)

if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="Split, fold, and stitch a long sequence (ColabFold + PyMOL).",
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
            "summary are not run. Summary is only printed for multi-segment runs."
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
            "3000 aa OOM cap)."
        ),
    )
    # Tokens after a standalone `--` are not parsed here; they are passed to colabfold_batch verbatim.
    try:
        i = sys.argv.index("--", 1)
    except ValueError:
        colabfold_batch_extra: list[str] = []
    else:
        colabfold_batch_extra = sys.argv[i + 1 :]
        del sys.argv[i:]

    a = p.parse_args()
    main(
        a.input,
        stitch_modes=a.stitch_modes,
        skip_colabfold=a.skip_colabfold,
        validate_adjacent_segments=a.validate_adjacent_segments,
        max_chunk_aa=a.max_chunk_aa,
        plan_mode=a.plan_mode,
        colabfold_batch_extra=colabfold_batch_extra,
    )
