FROM quay.io/pawsey/rocm-mpich-base:rocm7.0.2-mpich3.4.3-ubuntu24.04

LABEL org.opencontainers.image.authors="Sarah Beecroft <sarah.beecroft@csiro.au>" \
      org.opencontainers.image.description="Tiberius v2.0.6 with ROCm7.0.2 support on Ubuntu 24.04 for AMD GPUs" \
      org.opencontainers.image.source="https://github.com/PawseySC/GPU_biology/tree/main/tiberius/v2.0.6" \
      org.opencontainers.image.name="tiberius_v2.0.6_rocm7.0.2_ubuntu24.04"

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
ENV PIP_NO_CACHE_DIR=false

RUN apt-get update --quiet && \
    apt-get install --no-install-recommends --yes --quiet \
    build-essential \
    time \
    git \
    libgsl-dev \
    libboost-all-dev \
    libsuitesparse-dev \
    liblpsolve55-dev \
    libsqlite3-dev \
    libmysql++-dev \
    libboost-iostreams-dev \
    libbamtools-dev \
    samtools \
    libhts-dev \
    cdbfasta \
    libfile-which-perl \
    libparallel-forkmanager-perl \
    libyaml-perl \
    libdbd-mysql-perl \
    bamtools \
    bedtools \
    diamond-aligner \
    seqkit \
    transdecoder \
    minimap2 \
    perl \
    libdb-dev \
    zlib1g-dev \
    ca-certificates \
    hisat2 \
    && \
    rm -rf /var/lib/apt/lists/* && \
    apt-get autoremove -y && \
    apt-get clean

# Install cpanm + Perl modules (URI includes URI::Escape)
RUN curl -fsSL https://cpanmin.us | perl - App::cpanminus && \
    cpanm --notest DB_File && \
    cpanm --notest URI

RUN pip install \
    setuptools \
    wheel \
    "hatchling>=1.26" \
    "packaging>=24.0" \
    "pyfamsa<0.6.0" \
    pyyaml \
    pyBigWig \
    bio \
    scikit-learn \
    biopython \
    bcbio-gff \
    requests && \
    pip install tensorflow-rocm==2.19.1 -f https://repo.radeon.com/rocm/manylinux/rocm-rel-7.0.2/ --upgrade && \
    pip install --no-deps learnMSA

COPY vendor /opt
WORKDIR /opt

RUN cd Augustus && \
    make clean && \
    make && \
    make install

RUN cd Tiberius && \
    pip install . && \
    chmod +x tiberius.py && \
    chmod +x tiberius/*py && \
    mkdir -p /opt/Tiberius/model_weights && \
    chmod -R 777 /opt/Tiberius/model_weights && \
    ln -s /opt/Tiberius/tiberius.py /usr/local/bin/tiberius

RUN cd EvidencePipeline/EvidencePipeline/scripts && \
    chmod +x *py

RUN cd miniprot-boundary-scorer && make
RUN cd miniprot && make

ENV PATH=/usr/local/bin/:${PATH}:/opt/Augustus/bin/:/opt/Tiberius/tiberius/:/opt/Tiberius/:/opt/Tiberius/tiberius/scripts:/opt/EvidencePipeline/EvidencePipeline/scripts/:/opt/miniprothint:/opt/miniprot:/opt/miniprot-boundary-scorer:/opt/hisat2-2.2.1/:/opt/stringtie-3.0.3.Linux_x86_64:/opt/sratoolkit.3.3.0-ubuntu64/bin/:/opt/gffread-0.12.7.Linux_x86_64

RUN mkdir -p /opt/docker-recipes/
COPY AMD_dockerfile prescript.sh /opt/docker-recipes/
