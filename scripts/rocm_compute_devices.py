"""Discover ROCm device indices suitable for JAX/HIP compute (exclude display Radeon)."""

from __future__ import annotations

import re
import subprocess
import sys
from typing import Sequence


def is_compute_rocm_gpu(device_name: str) -> bool:
    """True for Instinct / MI* accelerators; false for typical Radeon display GPUs."""
    n = device_name.upper()
    if "RADEON" in n and "INSTINCT" not in n:
        return False
    return any(
        token in n
        for token in ("INSTINCT", "MI300", "MI250", "MI210", "MI100", "MI50")
    )


def discover_compute_rocm_gpu_ids(
    rocm_smi_cmd: Sequence[str] = ("rocm-smi", "-i"),
    timeout: float = 30.0,
) -> list[int]:
    """
    Parse `rocm-smi -i` and return ROCm indices of compute GPUs only.

    Falls back to [0] if rocm-smi is unavailable or no compute GPU is matched.
    """
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


def main() -> int:
    ids = discover_compute_rocm_gpu_ids()
    print(format_hip_visible_devices(ids))
    return 0


if __name__ == "__main__":
    sys.exit(main())
