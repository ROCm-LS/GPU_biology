FROM quay.io/pawsey/rocm-mpich-base:rocm6.4.1-mpich3.4.3-ubuntu24.04

# ROCm environment
ENV ROCM_RELEASE 6.4.1
ENV ROCM_PATH /opt/rocm-$ROCM_RELEASE

# Update environment
RUN apt-get update && \
    apt install apt-transport-https -y && \
    apt-get install -y \
    build-essential \
    git \
    && \
    apt-get remove -y python3-scipy && \
    rm -rf /var/lib/apt/lists/* && \
    apt-get autoremove -y && \
    apt-get clean

# Install pytorch 2.9.1 with ROCm 6.4 support
RUN pip config set global.break-system-packages true && \
    pip install --no-cache-dir torch==2.9.1 \
    torchvision==0.24.1 \
    torchaudio==2.9.1 \
    --index-url https://download.pytorch.org/whl/rocm6.4 && \
    pip install utils

# Clone boltz release v.2.2.1
WORKDIR /opt
RUN git clone -b v2.2.1 https://github.com/jwohlwend/boltz.git
WORKDIR /opt/boltz

# Custom pyproject.toml
COPY pyproject.toml .
# Install boltz
RUN	pip install -e . --no-cache-dir 

COPY Dockerfile pyproject.toml /opt/docker-recipes/
WORKDIR /opt 
RUN rm -rf /opt/boltz
LABEL org.opencontainers.image.authors="Sarah Beecroft <sarah.beecroft@csiro.au>" \
      org.opencontainers.image.description="Boltz2 v.2.2.1 with ROCm 6.4.1 support on Ubuntu 24.04 for AMD GPUs" \
      org.opencontainers.image.source="https://github.com/sarahbeecroft/boltz2/v2.2.1/rocm6.4.1" \
      org.opencontainers.image.name="boltz2v.2.2.1_rocm6.4.1_ubuntu24.04"