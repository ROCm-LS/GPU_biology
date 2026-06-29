#!/bin/bash -l
# This script is used to download and install dependencies for Tiberius into the vendor directory before you run 'podman build -f Dockerfile'.
# This is simply for convenience, because it's much faster to pull all these from the internet outside of a build context. 
# And compilation will be done inside the container. 
mkdir -p vendor
cd vendor
git clone --branch v2.0.6 https://github.com/Gaius-Augustus/Tiberius.git
git clone https://github.com/Gaius-Augustus/Augustus/
git clone https://github.com/tomasbruna/miniprothint
git clone https://github.com/lh3/miniprot
git clone https://github.com/tomasbruna/miniprot-boundary-scorer
wget https://github.com/gpertea/stringtie/releases/download/v3.0.3/stringtie-3.0.3.Linux_x86_64.tar.gz
tar xzf stringtie-3.0.3.Linux_x86_64.tar.gz
wget https://ftp-trace.ncbi.nlm.nih.gov/sra/sdk/3.3.0/sratoolkit.3.3.0-ubuntu64.tar.gz
tar xzf sratoolkit.3.3.0-ubuntu64.tar.gz
wget -q  https://github.com/gpertea/gffread/releases/download/v0.12.7/gffread-0.12.7.Linux_x86_64.tar.gz
tar xzf gffread-0.12.7.Linux_x86_64.tar.gz
rm *tar.gz
