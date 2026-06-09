FROM quay.io/pawsey/rocm-mpich-base:rocm7.2.3-mpich3.4.3-ubuntu24.04
# Optional: rebuild that base from GPU_biology root (see alphafold2/rocm7.2.3/README.md).
# Context must be setonix_containers/mpi (vendor lustrempich-base/; not bundled in GPU_biology).
#   docker build --no-cache --build-arg ROCM_VERSION=7.2.3 --build-arg OS_VERSION=24.04 \
#     -t quay.io/pawsey/rocm-mpich-base:rocm7.2.3-mpich3.4.3-ubuntu24.04 \
#     -f setonix_containers/mpi/rocm-mpich-base/buildrocm-mpich-base.dockerfile \
#     setonix_containers/mpi

# Use bash to support string substitution.
SHELL ["/bin/bash", "-o", "pipefail", "-c"]

# ROCm environment
ENV ROCM_RELEASE 7.2.3
ENV ROCM_PATH /opt/rocm-$ROCM_RELEASE
ENV PATH $ROCM_PATH/bin:$ROCM_PATH/llvm/bin:$PATH
ENV LD_LIBRARY_PATH $LD_LIBRARY_PATH:$ROCM_PATH/lib:/opt/miniforge3/lib
ENV JAX_PLATFORMS "rocm,cpu"
# JAX/XLA on ROCm 7.2.3: avoid "Too small divisible part of the contracting dimension"
# (disable Triton GEMM path); align with colabfold/rocm7.2.3/Dockerfile.
ENV XLA_PYTHON_CLIENT_PREALLOCATE=false
ENV XLA_PYTHON_CLIENT_ALLOCATOR=platform
ENV JAX_PLATFORM_NAME=gpu
ENV XLA_FLAGS="--xla_gpu_autotune_level=0 --xla_gpu_enable_triton_gemm=false"
# XLA ROCm: lld on PATH; LLVM for some jaxlib paths.
ENV LLVM_PATH $ROCM_PATH/llvm
# If the JAX ROCm plugin cannot find HIP bitcode, set these to your layout (Pawsey images
# sometimes expose bitcode under Clang’s tree; there may be a symlink at $ROCM_PATH/amdgcn/bitcode).
ENV HIP_DEVICE_LIB_PATH=$ROCM_PATH/lib/llvm/amdgcn/bitcode
ENV JAX_ROCM_PLUGIN_INTERNAL_BITCODE_PATH=$ROCM_PATH/lib/llvm/amdgcn/bitcode
ENV JAX_ROCM_PLUGIN_INTERNAL_LLD_PATH=$ROCM_PATH/llvm/bin

ARG DEBIAN_FRONTEND=noninteractive
RUN set -eux; \
    printf '%s\n' \
      'Acquire::Retries "8";' \
      'Acquire::http::Timeout "30";' \
      'Acquire::https::Timeout "30";' \
      'Acquire::ForceIPv4 "true";' \
      > /etc/apt/apt.conf.d/99-network-retries; \
    find /etc/apt -type f \( -name '*.list' -o -name '*.sources' \) -print0 \
      | xargs -0 sed -ri 's|http://archive.ubuntu.com/ubuntu|https://security.ubuntu.com/ubuntu|g' || true; \
    apt-get update; \
    apt-get install --no-install-recommends --yes \
      build-essential \
      cmake \
      git \
      hmmer \
      kalign \
      tzdata \
      wget \
      clang \
      doxygen; \
    rm -rf /var/lib/apt/lists/*; \
    apt-get autoremove --yes; \
    apt-get clean

WORKDIR /opt

# Clone alphafold
ENV ALPHAFOLD_PATH /app/alphafold
RUN set -eux ; \
    git clone --branch v2.3.2 https://github.com/google-deepmind/alphafold.git $ALPHAFOLD_PATH ; \
    cd $ALPHAFOLD_PATH ; \
    sed -i 's#CUDA#OpenCL#g' alphafold/relax/amber_minimize.py ; \
    sed -i 's#HIP#OpenCL#g' alphafold/relax/amber_minimize.py ; \
    cd $ALPHAFOLD_PATH/alphafold/common ; \
    curl -LO https://git.scicore.unibas.ch/schwede/openstructure/-/raw/7102c63615b64735c4941278d92b554ec94415f8/modules/mol/alg/src/stereo_chemical_props.txt
COPY config.py ${ALPHAFOLD_PATH}/alphafold/model
COPY amber_minimize.py ${ALPHAFOLD_PATH}/alphafold/relax/amber_minimize.py

# Install miniforge (mamba)
RUN set -eux ; \
    curl -L -O "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-$(uname)-$(uname -m).sh" ; \
    bash Miniforge3-$(uname)-$(uname -m).sh -b -p /opt/miniforge3 -s ; \
    rm -rf ./Miniforge3-*
ENV PATH /opt/miniforge3/bin:$PATH

# streamhpc::openmm-hip==8.0.0 is gone from repodata; conda-forge openmm-hip 8.1.1 solves with pdbfixer.
RUN mamba install -y -c conda-forge -c bioconda --strict-channel-priority \
    python=3.11 \
    pdbfixer==1.9 \
    bioconda::hhsuite \
    openmm-hip=8.1.1 \
    && mamba clean -afy

# xla_rocm7 set_rocm_paths() recognizes pip package "rocm" or Miniforge site-packages/rocm/lib -> system ROCm.
RUN PY_SITE="$(/opt/miniforge3/bin/python3 -c 'import site; print(site.getsitepackages()[0])')" && \
    mkdir -p "${PY_SITE}/rocm" && \
    ln -sfn "${ROCM_PATH}/lib" "${PY_SITE}/rocm/lib"

# Install pip packages.
COPY requirements.txt /app/alphafold/requirements.txt
# JAX: ROCm 7 PJRT/plugin + jaxlib from ROCm/jax releases (rocm-jax-v0.7.1, cp311).
# Do not use jax_rocm60_* on ROCm 7. Install order follows AMD JAX-on-ROCm guidance.
ARG ROCM_JAX_TAG=rocm-jax-v0.7.1
ARG JAX_ROCM_VER=0.7.1
RUN pip3 install -r /app/alphafold/requirements.txt --no-cache-dir --root-user-action=ignore --break-system-packages && \
    pip uninstall -y jax jaxlib jax-rocm60-pjrt jax-rocm60-plugin jax-rocm7-pjrt jax-rocm7-plugin 2>/dev/null || true && \
    rm -rf /opt/miniforge3/lib/python3.11/site-packages/jax_plugins && \
    python3 -m pip install --no-cache-dir --root-user-action=ignore --break-system-packages \
        "jax==${JAX_ROCM_VER}" --no-deps && \
    python3 -m pip install --no-cache-dir --root-user-action=ignore --break-system-packages \
        "https://github.com/ROCm/jax/releases/download/${ROCM_JAX_TAG}/jax_rocm7_pjrt-${JAX_ROCM_VER}-py3-none-manylinux_2_28_x86_64.whl" \
        "https://github.com/ROCm/jax/releases/download/${ROCM_JAX_TAG}/jax_rocm7_plugin-${JAX_ROCM_VER}-cp311-cp311-manylinux_2_28_x86_64.whl" \
        "https://github.com/ROCm/jax/releases/download/${ROCM_JAX_TAG}/jaxlib-${JAX_ROCM_VER}-cp311-cp311-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl"

# Fix OpenMM imports in AlphaFold
RUN find /app/alphafold -type f -name "*.py" -exec sed -i \
    -e 's/from simtk\.openmm/from openmm/g' \
    -e 's/from simtk import openmm/import openmm/g' \
    -e 's/import simtk\.openmm/import openmm/g' \
    -e 's/simtk\.openmm\./openmm./g' \
    {} +

RUN sed -i 's/from simtk.openmm.app.internal.pdbstructure import PdbStructure/from openmm.app.internal.pdbstructure import PdbStructure/g' \
    /app/alphafold/alphafold/relax/amber_minimize.py && \
    find /app/alphafold -type f -name "*.py" -exec sed -i \
    's/simtk\.unit/openmm.unit/g' {} +

# JAX 0.6+ / 0.7 vs AlphaFold v2.3.2: tree_map and jax.util.wraps removed.
RUN find /app/alphafold -type f -name "*.py" -exec sed -i \
    's/jax\.tree_map(/jax.tree_util.tree_map(/g' \
    {} + && \
    sed -i 's/@jax\.util\.wraps(fun, docstr=docstr)/@functools.wraps(fun)/' \
    /app/alphafold/alphafold/model/mapping.py

# Optional pymol-open-source (cp311 wheel): same OpenGL-related apt stack as
# colabfold/rocm7.2.3/ for scripts/split_and_fold_segments_alphafold2_single_container.py.
# Default 0 keeps the image smaller for inference-only / two-container workflows.
ARG INSTALL_PYMOL=0
RUN set -eux; \
    if [ "${INSTALL_PYMOL}" = "1" ]; then \
      apt-get update && \
      apt-get install --no-install-recommends --yes \
        libgl1 \
        libglu1-mesa \
        libglew2.2 \
        libpng16-16t64 \
        libfreetype6 \
        libxml2 \
        libglib2.0-0t64 \
      && rm -rf /var/lib/apt/lists/* \
      && pip3 install --no-cache-dir --root-user-action=ignore --break-system-packages \
        pymol-open-source==3.1.0a0 \
      && python3 -c "import pymol; print('PyMOL', getattr(pymol, '__version__', '3.1.0a0'), 'OK')"; \
    else \
      echo "INSTALL_PYMOL=${INSTALL_PYMOL}: skipping pymol-open-source"; \
    fi

WORKDIR /app/alphafold

# ---------------------------------------------------------------------------
# Persistence (edits vs container lifecycle)
#
# Everything under /app/alphafold is fixed at *image build* time (clone + COPY
# above). Recreating the container from the same image does not lose that tree,
# but rebuilding the image or starting from a fresh clone does — unless you
# keep your fork on the host and bind-mount it at run time:
#
#   mkdir -p "$HOME/alphafold-rocm" && cd "$HOME/alphafold-rocm"
#   git clone --branch v2.3.2 https://github.com/google-deepmind/alphafold.git .
#   # apply the same patches as this Dockerfile (or git remote + pull your branch)
#   docker run --rm -it \
#     -v "$HOME/alphafold-rocm:/app/alphafold" \
#     ...your gpu/rocm flags... \
#     your-image:tag
#
# The bind mount replaces /app/alphafold in the container; all edits live on
# the host. Large databases are usually separate mounts (e.g. /work/databases).
# ---------------------------------------------------------------------------
