# Building the rocm container

This recipe can be built with the following build args:

* `OS_VERSION` (default `24.04`)
* `LINUX_KERNEL` (default `6.8.0-31`)
* `ROCM_VERSION` (default `6.0.2`)
* `MPICH_VERSION` (default `3.4.3`)

Published images use the tag format `quay.io/pawsey/rocm-mpich-base:rocm<ROCM>-mpich<MPICH>-ubuntu<OS>`. Downstream recipes already consume this naming, for example the [ColabFold ROCm 6.2.4 image](https://github.com/SarahBeecroft/amd_porting/blob/e8dd6626a418a55a3cfb05af1efacd8423facb76/colabfold/rocm6.2.4/Dockerfile#L1):

```dockerfile
FROM quay.io/pawsey/rocm-mpich-base:rocm6.2.4-mpich3.4.3-ubuntu24.04
```

To build the same base with ROCm 7.2.3, keep `mpich3.4.3` and `ubuntu24.04` in the tag and only change the ROCm version.

This container provides:

* libfabric
* lustre
* lustre aware mpi build with libfabric
* rocm
* rccl
* osu microbenchmarks that provide tests for checking the mpi works

Example build. Use `setonix/mpi` as the build context (the final `.`) so the dockerfile can `COPY` MPICH patches from `lustrempich-base/`:

```
cd setonix
sudo docker build \
  --build-arg ROCM_VERSION=7.2.3 \
  --build-arg OS_VERSION=24.04 \
  -t quay.io/pawsey/rocm-mpich-base:rocm7.2.3-mpich3.4.3-ubuntu24.04 \
  -f rocm-mpich-base/buildrocm-mpich-base.dockerfile .
```
