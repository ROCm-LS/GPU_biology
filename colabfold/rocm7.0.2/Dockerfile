FROM quay.io/pawsey/rocm-mpich-base:rocm7.0.2-mpich3.4.3-ubuntu24.04

# Use bash to support string substitution.
SHELL ["/bin/bash", "-o", "pipefail", "-c"]

ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update --quiet \
    && apt-get install --no-install-recommends --yes --quiet \
        build-essential \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get autoremove --yes \
    && apt-get clean

WORKDIR /opt

# ROCm environment
ENV ROCM_RELEASE 7.0.2
ENV ROCM_PATH /opt/rocm-$ROCM_RELEASE
ENV PATH $ROCM_PATH/bin:$ROCM_PATH/llvm/bin:$PATH
ENV LD_LIBRARY_PATH $LD_LIBRARY_PATH:$ROCM_PATH/lib
ENV JAX_PLATFORMS="rocm,cpu"
ENV PIP_NO_CACHE_DIR=true

# Install miniforge (mamba)
RUN set -eux ; \
    curl -L -O "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-$(uname)-$(uname -m).sh" ; \
    bash Miniforge3-$(uname)-$(uname -m).sh -b -p /opt/miniforge3 -s ; \
    rm -rf ./Miniforge3-*
ENV PATH /opt/miniforge3/bin:$PATH

RUN mamba install -c conda-forge -c bioconda python=3.12.3 kalign2=2.04 hhsuite=3.3.0 mmseqs2=18.8cc5c
RUN pip install colabfold[alphafold,openmm]==1.6.1 && \
    pip uninstall -y jax jaxlib && \
    pip install https://github.com/ROCm/rocm-jax/releases/download/rocm-jax-v0.6.0/jax_rocm7_pjrt-0.6.0-py3-none-manylinux_2_28_x86_64.whl \
    https://github.com/ROCm/rocm-jax/releases/download/rocm-jax-v0.6.0/jax_rocm7_plugin-0.6.0-cp312-cp312-manylinux_2_28_x86_64.whl \
    https://github.com/ROCm/rocm-jax/releases/download/rocm-jax-v0.6.0/jaxlib-0.6.2-cp312-cp312-manylinux2014_x86_64.whl \
    jax==0.6.2 && \
    pip install openmm[hip7]==8.5.2 && \
    mamba clean -afy

RUN sed -i 's/CUDA/HIP/g' /opt/miniforge3/lib/python3.12/site-packages/alphafold/relax/amber_minimize.py

COPY Dockerfile /opt/docker-recipes/
LABEL org.opencontainers.image.authors="Sarah Beecroft <sarah.beecroft@csiro.au>" \
      org.opencontainers.image.description="Colabfold 1.6.1 with ROCm7.0.2 support on Ubuntu 24.04 for AMD GPUs" \
      org.opencontainers.image.source="https://github.com/PawseySC/GPU_biology/tree/main/colabfold/rocm7.0.2" \
      org.opencontainers.image.name="colabfold_1.6.1_rocm7.0.2_ubuntu24.04" \
      Python_version="3.12.3"

ENV XLA_PYTHON_CLIENT_MEM_FRACTION="4.0"
ENV XLA_PYTHON_CLIENT_ALLOCATOR="platform"
ENV TF_FORCE_GPU_ALLOW_GROWTH="true"
ENV TF_FORCE_UNIFIED_MEMORY="0"
ENV XLA_FLAGS='--xla_gpu_enable_command_buffer= --xla_gpu_autotune_level=0'
