"""Discover ROCm device indices suitable for JAX/HIP compute (exclude display Radeon)."""

from __future__ import annotations

import os
import re
import subprocess
import sys
from typing import Sequence


def parse_hip_visible_devices(raw: str) -> list[int]:
    """Parse a comma-separated HIP_VISIBLE_DEVICES string (invalid tokens skipped)."""
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


def resolve_orchestrator_gpu_ids() -> list[int]:
    """GPU indices for host split-fold orchestrators.

    Honors ``HIP_VISIBLE_DEVICES`` when set in the environment. When unset, uses
    ROCm indices of **compute** GPUs only (``rocm-smi`` device names: Instinct /
    MI*, gfx90a, …; excludes display Radeon and non-GPU rocminfo agents).

    Does **not** write ``HIP_VISIBLE_DEVICES`` into ``os.environ``; each container
    job sets the device index explicitly (avoids confusing Singularity/Slurm).
    """
    if "HIP_VISIBLE_DEVICES" in os.environ:
        ids = parse_hip_visible_devices(os.environ["HIP_VISIBLE_DEVICES"])
        return ids if ids else [0]
    return discover_compute_rocm_gpu_ids()


def discover_node_gpu_ids(
    rocm_smi_cmd: Sequence[str] = ("rocm-smi", "-i"),
    timeout: float = 30.0,
) -> list[int]:
    """Return ROCm indices of compute GPUs (alias for :func:`discover_compute_rocm_gpu_ids`)."""
    return discover_compute_rocm_gpu_ids(rocm_smi_cmd=rocm_smi_cmd, timeout=timeout)


_COMPUTE_NAME_MARKERS = (
    "INSTINCT",
    "MI355",
    "MI350",
    "MI300",
    "MI250",
    "MI210",
    "MI100",
    "MI50",
    "GFX90A",
    "GFX942",
    "GFX1100",
    "CDNA",
    "CDNA2",
    "CDNA3",
)


def is_compute_rocm_gpu(device_name: str) -> bool:
    """True for Instinct / MI* / gfx90a accelerators; false for display Radeon."""
    n = device_name.upper()
    if "RADEON" in n and "INSTINCT" not in n:
        return False
    return any(token in n for token in _COMPUTE_NAME_MARKERS)


def discover_compute_rocm_gpu_ids(
    rocm_smi_cmd: Sequence[str] = ("rocm-smi", "-i"),
    timeout: float = 30.0,
) -> list[int]:
    """
    Parse ``rocm-smi -i`` and return ROCm indices of compute GPUs only.

    Excludes display Radeon devices and other non-compute agents that may appear
    in ``rocminfo``. Falls back to ``[0]`` if ``rocm-smi`` is unavailable or no
    compute GPU is matched.
    """
    if not rocm_gpu_visible_on_host():
        return [0]

    try:
        out = subprocess.check_output(
            list(rocm_smi_cmd),
            text=True,
            stderr=subprocess.STDOUT,
            timeout=timeout,
        )
    except (FileNotFoundError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return [0]

    ids: list[int] = []
    for line in out.splitlines():
        m = re.search(r"GPU\[(\d+)\].*Device Name:\s*(.+)", line)
        if m and is_compute_rocm_gpu(m.group(2).strip()):
            ids.append(int(m.group(1)))
    return ids if ids else [0]


def format_hip_visible_devices(ids: Sequence[int]) -> str:
    return ",".join(str(i) for i in ids)


def rocm_gpu_visible_on_host() -> bool:
    """True when the host exposes ROCm devices (``/dev/kfd``)."""
    return os.path.exists("/dev/kfd")


def require_rocm_gpu_visible(*, context: str = "fold") -> None:
    """Abort if this host cannot access ROCm GPUs (typical on login nodes)."""
    if rocm_gpu_visible_on_host():
        return
    raise SystemExit(
        "No ROCm GPU is visible on this host (/dev/kfd missing).\n"
        f"Cannot run {context} without a GPU allocation.\n"
        "On Setonix, request a GPU node and load Singularity inside that session, e.g.:\n"
        "  srun --partition=gpu --gpus=1 --cpus-per-task=8 --time=2:00:00 --pty bash\n"
        "  module load singularity/3.11.4-nompi\n"
        "  python3 scripts/split_and_fold_segments_colabfold.py QUERY.fa \\\n"
        "    --runtime singularity --colabfold-sif /path/to/colabfold.sif \\\n"
        "    --pymol-sif /path/to/pymol.sif --work-dir $PWD\n"
        "\n"
        "Or submit a batch job with #SBATCH --partition=gpu and --gpus=…, then run the script "
        "inside the job (not on the login node)."
    )


def main() -> int:
    if "HIP_VISIBLE_DEVICES" in os.environ:
        ids = parse_hip_visible_devices(os.environ["HIP_VISIBLE_DEVICES"])
        print(format_hip_visible_devices(ids if ids else [0]))
    else:
        ids = discover_compute_rocm_gpu_ids()
        print(format_hip_visible_devices(ids))
    return 0


if __name__ == "__main__":
    sys.exit(main())
