"""JAX / XLA environment for single-container ColabFold on ROCm (optional 7.2.3 workaround)."""

from __future__ import annotations

import os

# ROCm 7.2.3 + JAX 0.7.x: Triton GEMM can trigger XLA "Too small divisible part of the
# contracting dimension"; disable that path (match Dockerfiles under */rocm7.2.3/).
_ROCM_732_MARKERS = ("rocm7.2.3", "rocm-7.2.3", "rocm_7.2.3", "/rocm-7.2.3")


def jax_xla_env_for_fold_subprocess(
    *,
    image_hints: str = "",
    enable_rocm_732_triton_gemm_workaround: bool = False,
) -> dict[str, str]:
    """
    Return env keys to merge (e.g. into ``os.environ.copy()``) for fold child processes.

    By default: ``XLA_PYTHON_CLIENT_PREALLOCATE=false`` and
    ``XLA_FLAGS=--xla_gpu_autotune_level=0`` only.

    When ``enable_rocm_732_triton_gemm_workaround`` is **True** (single-container ColabFold
    only), ROCm **7.2.3** additionally gets ``JAX_PLATFORM_NAME``, platform allocator, and
    ``--xla_gpu_enable_triton_gemm=false`` (avoids XLA *Too small divisible part of the
    contracting dimension* on some JAX builds). Other callers must leave this ``False``.

    **ROCm 7.2.3 detection** (only if ``enable_rocm_732_triton_gemm_workaround``)

    - ``GPU_BIOLOGY_FORCE_ROCM_732_JAX=1`` or ``0`` — force that path on/off.
    - Else ``image_hints`` or ``ROCM_PATH`` / ``ROCM_RELEASE`` / ``ROCM_VERSION`` /
      ``ROCM_DIR`` contains a marker such as ``rocm7.2.3`` or ``/rocm-7.2.3``.
    """
    minimal = {
        "XLA_PYTHON_CLIENT_PREALLOCATE": "false",
        "XLA_FLAGS": "--xla_gpu_autotune_level=0",
    }
    if not enable_rocm_732_triton_gemm_workaround:
        return minimal

    override = os.environ.get("GPU_BIOLOGY_FORCE_ROCM_732_JAX", "").strip().lower()
    if override in ("0", "false", "no"):
        use_732 = False
    elif override in ("1", "true", "yes"):
        use_732 = True
    else:
        rocm = "".join(
            os.environ.get(k, "")
            for k in ("ROCM_PATH", "ROCM_RELEASE", "ROCM_VERSION", "ROCM_DIR")
        )
        blob = f"{image_hints} {rocm}".lower()
        use_732 = any(m in blob for m in _ROCM_732_MARKERS)

    if use_732:
        return {
            "JAX_PLATFORM_NAME": "gpu",
            "XLA_PYTHON_CLIENT_ALLOCATOR": "platform",
            "XLA_PYTHON_CLIENT_PREALLOCATE": "false",
            "XLA_FLAGS": (
                "--xla_gpu_autotune_level=0 --xla_gpu_enable_triton_gemm=false"
            ),
        }
    return minimal
