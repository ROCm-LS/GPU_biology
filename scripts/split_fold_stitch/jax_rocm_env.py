"""JAX / XLA environment for ColabFold and AlphaFold2 fold subprocesses on ROCm."""

from __future__ import annotations

import os

# ROCm 7.2.3 + JAX 0.7.x: Triton GEMM can trigger XLA "Too small divisible part of the
# contracting dimension"; disable that path (match Dockerfiles under */rocm7.2.3/).
_ROCM_732_MARKERS = ("rocm7.2.3", "rocm-7.2.3", "rocm_7.2.3", "/rocm-7.2.3")

# Minimal settings: ROCm 6.2.4 ColabFold, dual-container AlphaFold2 (6.2.4 and 7.2.3).
MINIMAL_ROCM_JAX_ENV: dict[str, str] = {
    "XLA_PYTHON_CLIENT_PREALLOCATE": "false",
    "XLA_FLAGS": "--xla_gpu_autotune_level=0",
}

# Extended settings: ROCm 7.2.3 ColabFold (platform allocator + disable Triton GEMM).
COLABFOLD_ROCM_732_JAX_ENV: dict[str, str] = {
    "JAX_PLATFORM_NAME": "gpu",
    "XLA_PYTHON_CLIENT_ALLOCATOR": "platform",
    "XLA_PYTHON_CLIENT_PREALLOCATE": "false",
    "XLA_FLAGS": "--xla_gpu_autotune_level=0 --xla_gpu_enable_triton_gemm=false",
}


def minimal_rocm_jax_env() -> dict[str, str]:
    """JAX/XLA env for ROCm 6.2.4 and for AlphaFold2 dual-container runs."""
    return dict(MINIMAL_ROCM_JAX_ENV)


def _blob_from_image_hints(image_hints: str) -> str:
    rocm = "".join(
        os.environ.get(k, "")
        for k in ("ROCM_PATH", "ROCM_RELEASE", "ROCM_VERSION", "ROCM_DIR")
    )
    return f"{image_hints} {rocm}".lower()


def is_rocm_732_image(image_hints: str = "") -> bool:
    """True when *image_hints* or ROCm env vars indicate ROCm 7.2.3."""
    return any(m in _blob_from_image_hints(image_hints) for m in _ROCM_732_MARKERS)


def _use_colabfold_732_jax_env(image_hints: str) -> bool:
    override = os.environ.get("GPU_BIOLOGY_FORCE_ROCM_732_JAX", "").strip().lower()
    if override in ("0", "false", "no"):
        return False
    if override in ("1", "true", "yes"):
        return True
    return is_rocm_732_image(image_hints)


def colabfold_batch_jax_env(*, image_hints: str = "") -> dict[str, str]:
    """JAX/XLA env for ``colabfold_batch`` inside a container.

    - **ROCm 6.2.4** (``.sif`` / image path contains ``rocm6.2.4``): minimal env.
    - **ROCm 7.2.3** (``rocm7.2.3`` in path or ``GPU_BIOLOGY_FORCE_ROCM_732_JAX=1``):
      platform allocator + ``--xla_gpu_enable_triton_gemm=false``.
    - **Unknown image**: minimal (safe default for 6.2.4).
    """
    if _use_colabfold_732_jax_env(image_hints):
        return dict(COLABFOLD_ROCM_732_JAX_ENV)
    return minimal_rocm_jax_env()


def alphafold_fold_jax_env(*, image_hints: str = "") -> dict[str, str]:
    """JAX/XLA env for ``run_alphafold.py`` in dual-container orchestration.

    Uses minimal settings on both ROCm 6.2.4 and 7.2.3 (see ``scripts/README.md``).
    """
    del image_hints  # reserved for future per-version tuning
    return minimal_rocm_jax_env()


def jax_xla_env_for_fold_subprocess(
    *,
    image_hints: str = "",
    enable_rocm_732_triton_gemm_workaround: bool = False,
) -> dict[str, str]:
    """
    Return env keys for fold child processes (single-container ColabFold).

    When ``enable_rocm_732_triton_gemm_workaround`` is **True**, applies the same
    ROCm-version detection as :func:`colabfold_batch_jax_env`. Otherwise minimal only.
    """
    if not enable_rocm_732_triton_gemm_workaround:
        return minimal_rocm_jax_env()
    return colabfold_batch_jax_env(image_hints=image_hints)
