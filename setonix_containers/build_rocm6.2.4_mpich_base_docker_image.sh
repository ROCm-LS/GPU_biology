#!/bin/bash

docker build \
  --build-arg ROCM_VERSION=6.2.4 \
  -t rocm-mpich-rocm-base:3.4.3_ubuntu24.04-rocm6.2.4 \
  -f rocm-mpich-base/buildrocm-mpich-base.dockerfile \
  .
