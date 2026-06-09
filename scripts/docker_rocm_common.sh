# Shared ROCm GPU checks and HIP_VISIBLE_DEVICES discovery for docker run scripts.
# Source from sibling scripts: source "${SCRIPT_DIR}/docker_rocm_common.sh"

: "${SCRIPT_DIR:?SCRIPT_DIR must be set before sourcing docker_rocm_common.sh}"

ROCM_GPU_HELPER="${SCRIPT_DIR}/rocm_compute_devices.py"

_amd_display_present() {
  lspci -nn 2>/dev/null | grep -iE 'vga|3d|display' | grep -qi '\[1002:' || return 1
  return 0
}

_amdgpu_modprobe_required() {
  [[ -e /dev/kfd ]] && return 1
  _amd_display_present || return 1
  modinfo amdgpu &>/dev/null || return 1
  grep -q '^amdgpu ' /proc/modules 2>/dev/null && return 1
  return 0
}

_discover_hip_visible_devices() {
  if [[ -n "${HIP_VISIBLE_DEVICES:-}" ]]; then
    echo "${HIP_VISIBLE_DEVICES}"
    return
  fi
  if [[ -f "${ROCM_GPU_HELPER}" ]]; then
    python3 "${ROCM_GPU_HELPER}"
    return
  fi
  echo "Warning: ${ROCM_GPU_HELPER} not found; defaulting HIP_VISIBLE_DEVICES=0" >&2
  echo "0"
}

setup_docker_rocm_dev_args() {
  DOCKER_DEV_ARGS=()
  if [[ -e /dev/kfd ]]; then
    DOCKER_DEV_ARGS+=(--device=/dev/kfd)
  else
    if _amdgpu_modprobe_required; then
      echo "ERROR: AMD GPU is present and the amdgpu module is installed, but it is not loaded." >&2
      echo "  /dev/kfd will not appear until the driver is loaded. Run:" >&2
      echo "    sudo modprobe amdgpu" >&2
      echo "  Then re-run this script." >&2
      exit 1
    fi
    echo "Warning: /dev/kfd not found — ROCm GPU acceleration will not be available." >&2
    echo "  Install AMDGPU/ROCm drivers on the host, or use an AMD GPU machine. See:" >&2
    echo "  https://rocm.docs.amd.com/" >&2
  fi
  if [[ -e /dev/dri ]]; then
    DOCKER_DEV_ARGS+=(--device=/dev/dri)
  else
    echo "Warning: /dev/dri not found — no DRM/GPU render nodes exposed to the container." >&2
  fi
}
