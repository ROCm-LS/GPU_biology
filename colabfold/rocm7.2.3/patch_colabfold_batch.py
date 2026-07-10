#!/usr/bin/env python3
"""Patch ColabFold batch.py for ROCm/JAX compatibility.

Patch 1: Replace Colab TPU probe with JAX_PLATFORMS-friendly GPU/CPU detection.
Patch 2: Round pad_len up to the next power of 2 so Triton softmax tile reads
         stay within the allocated buffer (JAX 0.7.1 / ROCm 7.2.x uses tile=2048
         for any N where next_power_of_2(N) == 2048; if N < 2048 the extract
         reads past the end of the buffer, causing an aperture violation).
"""

from __future__ import annotations

import re
import site
import sys
from pathlib import Path

# ---------- Patch 1: device detection ----------

DEVICE_REPLACEMENT = """    # check what device is available (JAX 0.6+: use JAX_PLATFORMS; no Colab TPU setup)
    if jax.local_devices()[0].platform == 'cpu':
        logger.info("WARNING: no GPU detected, will be using CPU")
        DEVICE = "cpu"
        use_gpu_relax = False
    else:
        import tensorflow as tf
        tf.get_logger().setLevel(logging.ERROR)
        logger.info('Running on GPU')
        DEVICE = "gpu"
        tf.config.set_visible_devices([], 'GPU')
"""

DEVICE_PATTERN = re.compile(
    r"    # check what device is available\n"
    r"    try:\n"
    r"        # check if TPU is available\n"
    r".*?"
    r"        tf\.config\.set_visible_devices\(\[\], 'GPU'\)\n",
    re.DOTALL,
)

# ---------- Patch 2: round pad_len to next power-of-2 ----------

PAD_OLD = """\
                # decide how much to pad (to avoid recompiling)
                if seq_len > pad_len:
                    if isinstance(recompile_padding, float):
                        pad_len = math.ceil(seq_len * recompile_padding)
                    else:
                        pad_len = seq_len + recompile_padding
                    pad_len = min(pad_len, max_len)
"""

# Round up to next power-of-2 so that Triton softmax tile size (always a
# power-of-2) equals the actual buffer size and never reads out of bounds.
PAD_NEW = """\
                # decide how much to pad (to avoid recompiling)
                if seq_len > pad_len:
                    if isinstance(recompile_padding, float):
                        pad_len = math.ceil(seq_len * recompile_padding)
                    else:
                        pad_len = seq_len + recompile_padding
                    pad_len = min(pad_len, max_len)
                # Round up to the next power of 2 so the Triton softmax tile
                # (next_power_of_2(pad_len)) equals the buffer size and avoids
                # out-of-bounds reads that cause HSA aperture violations on ROCm.
                if pad_len > 1:
                    import math as _math
                    pad_len = 2 ** _math.ceil(_math.log2(pad_len))
"""


def main() -> int:
    batch = Path(site.getsitepackages()[0]) / "colabfold" / "batch.py"
    if not batch.is_file():
        print(f"error: {batch} not found", file=sys.stderr)
        return 1
    text = batch.read_text()
    changed = False

    # Patch 1
    if DEVICE_PATTERN.search(text):
        text = DEVICE_PATTERN.sub(DEVICE_REPLACEMENT, text, count=1)
        print(f"Patched device detection in {batch}")
        changed = True
    elif "no Colab TPU setup" in text:
        print(f"Device detection already patched: {batch}")
    else:
        print(f"error: device block not found in {batch}", file=sys.stderr)
        return 1

    # Patch 2
    if PAD_OLD in text:
        text = text.replace(PAD_OLD, PAD_NEW, 1)
        print(f"Patched pad_len power-of-2 rounding in {batch}")
        changed = True
    elif "Round up to the next power of 2" in text:
        print(f"pad_len rounding already patched: {batch}")
    else:
        print(f"error: pad_len block not found in {batch}", file=sys.stderr)
        return 1

    if changed:
        batch.write_text(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
