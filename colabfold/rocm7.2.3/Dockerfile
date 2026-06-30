# Faster rebuilds: minimal mamba solve + pip requirements + optional weight download.
# Optional faster rebuilds: DOCKER_BUILDKIT=1 docker build ... (enables layer cache; no --mount required).
# One-time weights (large): --build-arg DOWNLOAD_WEIGHTS=1
# PyMOL (single-container stitch): default on; --build-arg INSTALL_PYMOL=0 to omit.
# Or after first run: docker exec <container> python3 -m colabfold.download  (use mounted /cache)
FROM rocm-mpich-base:rocm7.2.3-mpich3.4.3-ubuntu24.04
# Optional: rebuild that base from GPU_biology root (see colabfold/rocm7.2.3/README.md).
# Context must be setonix_containers/mpi (vendor lustrempich-base/; not bundled in GPU_biology).
#   docker build --no-cache --build-arg ROCM_VERSION=7.2.3 --build-arg OS_VERSION=24.04 \
#     -t quay.io/pawsey/rocm-mpich-base:rocm7.2.3-mpich3.4.3-ubuntu24.04 \
#     -f setonix_containers/mpi/rocm-mpich-base/buildrocm-mpich-base.dockerfile \
#     setonix_containers/mpi

SHELL ["/bin/bash", "-o", "pipefail", "-c"]

ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update --quiet \
    && apt-get install --no-install-recommends --yes --quiet \
        build-essential \
        cmake \
        git \
        tzdata \
        wget \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get autoremove --yes \
    && apt-get clean

WORKDIR /opt

COPY patch_colabfold_batch.py /opt/colabfold/
ENV PYTHONPATH=/opt/colabfold:${PYTHONPATH}

ENV ROCM_RELEASE=7.2.3
ENV ROCM_PATH=/opt/rocm-${ROCM_RELEASE}
ENV PATH=${ROCM_PATH}/bin:${ROCM_PATH}/llvm/bin:${PATH}
ENV LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:${ROCM_PATH}/lib:/opt/miniforge3/lib
ENV JAX_PLATFORMS=rocm,cpu
ENV LLVM_PATH=${ROCM_PATH}/llvm
ENV HIP_DEVICE_LIB_PATH=${ROCM_PATH}/amdgcn/bitcode
ENV JAX_ROCM_PLUGIN_INTERNAL_BITCODE_PATH=${ROCM_PATH}/amdgcn/bitcode
ENV JAX_ROCM_PLUGIN_INTERNAL_LLD_PATH=${ROCM_PATH}/llvm/bin
# MI300X / ROCm + JAX 0.7: same XLA workarounds as alphafold2/rocm7.2.3 (see config.py fuse_projection_weights).
ENV XLA_PYTHON_CLIENT_PREALLOCATE=false
ENV XLA_PYTHON_CLIENT_ALLOCATOR=platform
ENV JAX_PLATFORM_NAME=gpu
ENV XLA_FLAGS="--xla_gpu_autotune_level=0 --xla_gpu_enable_triton_gemm=false --xla_disable_hlo_passes=custom-kernel-fusion-rewriter"

# Pin Miniforge installer (avoid "latest" busting cache every release).
ARG MINIFORGE_VERSION=25.3.0-3
RUN set -eux; \
    curl -fsSL -o /tmp/mf.sh \
      "https://github.com/conda-forge/miniforge/releases/download/${MINIFORGE_VERSION}/Miniforge3-${MINIFORGE_VERSION}-Linux-$(uname -m).sh"; \
    bash /tmp/mf.sh -b -p /opt/miniforge3 -s; \
    rm -f /tmp/mf.sh
ENV PATH=/opt/miniforge3/bin:${PATH}

# Fast: only python+pip from conda (seconds, not 30+ min "Resolving environment").
COPY environment.yml requirements-pip.txt /opt/colabfold/
RUN mamba install -y --strict-channel-priority --file /opt/colabfold/environment.yml \
    && mamba clean -afy

RUN pip install --no-cache-dir --root-user-action=ignore --break-system-packages \
        -r /opt/colabfold/requirements-pip.txt \
    && mamba install -y -c conda-forge -c bioconda --strict-channel-priority \
        colabfold=1.5.5 --no-deps \
    && mamba clean -afy

# Optional pymol-open-source (default on): OpenGL-related apt + wheel, same idea as
# alphafold2/rocm7.2.3/ INSTALL_PYMOL. Set --build-arg INSTALL_PYMOL=0 for ColabFold-only
# (two-container stitch); INSTALL_PYMOL=1 for scripts/split_and_fold_segments_colabfold_single_container.py.
ARG INSTALL_PYMOL=1
RUN set -eux; \
    if [ "${INSTALL_PYMOL}" = "1" ]; then \
      apt-get update --quiet && \
      apt-get install --no-install-recommends --yes --quiet \
        libgl1 \
        libglu1-mesa \
        libglew2.2 \
        libpng16-16t64 \
        libfreetype6 \
        libxml2 \
        libglib2.0-0t64 \
      && rm -rf /var/lib/apt/lists/* \
      && pip install --no-cache-dir --root-user-action=ignore --break-system-packages \
        pymol-open-source==3.1.0a0 \
      && python3 -c "import pymol; print('PyMOL', getattr(pymol, '__version__', '3.1.0a0'), 'OK')"; \
    else \
      echo "INSTALL_PYMOL=${INSTALL_PYMOL}: skipping pymol-open-source and graphics libs"; \
    fi

RUN PY_SITE="$(python3 -c 'import site; print(site.getsitepackages()[0])')" && \
    mkdir -p "${PY_SITE}/rocm" && \
    ln -sfn "${ROCM_PATH}/lib" "${PY_SITE}/rocm/lib"

ARG ROCM_JAX_TAG=rocm-jax-v0.7.1
ARG JAX_ROCM_VER=0.7.1
RUN pip uninstall -y jax jaxlib jax-rocm60-pjrt jax-rocm60-plugin jax-rocm7-pjrt jax-rocm7-plugin 2>/dev/null || true && \
    rm -rf "$(python3 -c 'import site; print(site.getsitepackages()[0])')/jax_plugins" && \
    pip install --no-cache-dir --root-user-action=ignore --break-system-packages \
        "jax==${JAX_ROCM_VER}" --no-deps && \
    pip install --no-cache-dir --root-user-action=ignore --break-system-packages \
        "https://github.com/ROCm/jax/releases/download/${ROCM_JAX_TAG}/jax_rocm7_pjrt-${JAX_ROCM_VER}-py3-none-manylinux_2_28_x86_64.whl" \
        "https://github.com/ROCm/jax/releases/download/${ROCM_JAX_TAG}/jax_rocm7_plugin-${JAX_ROCM_VER}-cp311-cp311-manylinux_2_28_x86_64.whl" \
        "https://github.com/ROCm/jax/releases/download/${ROCM_JAX_TAG}/jaxlib-${JAX_ROCM_VER}-cp311-cp311-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl"

RUN SITE="$(python3 -c 'import site; print(site.getsitepackages()[0])')" && \
    if [[ -d "${SITE}/alphafold" ]]; then \
      find "${SITE}/alphafold" -type f -name '*.py' -exec sed -i 's/jax\.tree_map(/jax.tree_util.tree_map(/g' {} + ; \
    fi && \
    MP="${SITE}/alphafold/model/mapping.py" && \
    if [[ -f "${MP}" ]]; then \
      sed -i 's/@jax\.util\.wraps(fun, docstr=docstr)/@functools.wraps(fun)/' "${MP}"; \
    fi && \
    AF_CFG="${SITE}/alphafold/model/config.py" && \
    if [[ -f "${AF_CFG}" ]]; then \
      sed -i "s/'fuse_projection_weights': True/'fuse_projection_weights': False/g" "${AF_CFG}"; \
    fi && \
    python3 /opt/colabfold/patch_colabfold_batch.py

# Weights are multi-GB; skip by default. Use cache volume + download once at runtime.
ARG DOWNLOAD_WEIGHTS=0
RUN if [[ "${DOWNLOAD_WEIGHTS}" == "1" ]]; then python3 -m colabfold.download; else \
      echo "Skipping colabfold.download (set --build-arg DOWNLOAD_WEIGHTS=1 or run in container with /cache mounted)"; \
    fi
