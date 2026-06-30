#!/bin/bash

docker build \
  --build-arg ROCM_VERSION=7.2.3 \
  -t rocm-mpich-base:rocm7.2.3-mpich3.4.3-ubuntu24.04 \
  -f rocm-7.2.3-mpich-base/buildrocm-mpich-base.dockerfile \
  .
