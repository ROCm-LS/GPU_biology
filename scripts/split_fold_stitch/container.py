"""Run ColabFold and PyMOL inside Docker, Singularity/Apptainer, or locally on the host."""

from __future__ import annotations

import glob
import os
import shlex
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from typing import Sequence


CONTAINER_WORK_MOUNT = "/work"
CACHE_CONTAINER_MOUNT = "/cache"
SCRIPTS_CONTAINER_MOUNT = "/pipeline_scripts"
DEFAULT_COLABFOLD_IMAGE = "quay.io/pawsey/colabfold:rocm6.2.4"
DEFAULT_ALPHAFOLD2_IMAGE = "alphafold2-amd-gpu:v2.3.2_rocm7.2.3"
DEFAULT_PYMOL_IMAGE = "jysgro/pymol:deb12-2.5.0_sc"


@dataclass
class ContainerConfig:
    """How the host orchestrator invokes ColabFold and PyMOL."""

    runtime: str = "docker"
    work_dir: str = "."
    colabfold_image: str = DEFAULT_COLABFOLD_IMAGE
    pymol_image: str = DEFAULT_PYMOL_IMAGE
    colabfold_sif: str | None = None
    pymol_sif: str | None = None
    cache_dir: str | None = None
    colabfold_container_name: str | None = None
    alphafold2_image: str = DEFAULT_ALPHAFOLD2_IMAGE
    alphafold2_sif: str | None = None
    alphafold2_container_name: str | None = None
    # Path to AlphaFold clone inside the AF2 container (and cwd for local runs).
    alphafold2_app_root: str = "/app/alphafold"
    docker_shm_size: str = "64g"
    singularity_bind: list[str] = field(default_factory=list)
    singularity_rocm: bool = True
    docker_gpu_devices: bool = True
    pymol_command: str = "pymol"


def resolve_work_dir(work_dir: str | None, *paths: str) -> str:
    """Pick a directory to bind-mount as /work (common parent of pipeline paths)."""
    if work_dir:
        return os.path.abspath(work_dir)
    abs_paths = [os.path.abspath(p) for p in paths if p]
    if not abs_paths:
        return os.getcwd()
    common = os.path.commonpath(abs_paths)
    if os.path.isfile(common):
        common = os.path.dirname(common)
    return common


def host_to_container(path: str, work_dir: str) -> str:
    """Map a host path under work_dir to /work/... inside the container."""
    host = os.path.abspath(path)
    root = os.path.abspath(work_dir)
    if host == root:
        return CONTAINER_WORK_MOUNT
    if not host.startswith(root + os.sep) and host != root:
        raise ValueError(
            f"Path {host!r} is not under work directory {root!r}; "
            "set --work-dir to a common parent or move files."
        )
    rel = os.path.relpath(host, root)
    return f"{CONTAINER_WORK_MOUNT}/{rel}" if rel != "." else CONTAINER_WORK_MOUNT


class ContainerRunner:
    def __init__(self, config: ContainerConfig):
        self.config = config
        self.work_dir = os.path.abspath(config.work_dir)
        os.makedirs(self.work_dir, exist_ok=True)
        pkg_dir = os.path.dirname(os.path.abspath(__file__))
        self._scripts_host = os.path.dirname(pkg_dir)
        self._pymol_worker_host = os.path.join(pkg_dir, "pymol_worker.py")

    def _pymol_worker_container_path(self) -> str:
        return f"{SCRIPTS_CONTAINER_MOUNT}/split_fold_stitch/pymol_worker.py"

    def _pymol_pythonpath(self, *, for_container: bool) -> str:
        if for_container:
            return f"{SCRIPTS_CONTAINER_MOUNT}{os.pathsep}{CONTAINER_WORK_MOUNT}"
        return f"{self._scripts_host}{os.pathsep}{self.work_dir}"

    def _scripts_docker_volume_args(self) -> list[str]:
        return ["-v", f"{self._scripts_host}:{SCRIPTS_CONTAINER_MOUNT}:ro"]

    def _scripts_singularity_bind(self) -> str:
        return f"{self._scripts_host}:{SCRIPTS_CONTAINER_MOUNT}:ro"

    def _cache_env(self, *, for_container: bool) -> dict[str, str]:
        """Env vars so colabfold_batch uses the cache mount (not /.cache under /)."""
        if not self.config.cache_dir:
            return {}
        cache_root = (
            CACHE_CONTAINER_MOUNT
            if for_container
            else os.path.abspath(self.config.cache_dir)
        )
        return {
            "XDG_CACHE_HOME": cache_root,
            "MPLCONFIGDIR": cache_root,
            "CACHE_DIR": cache_root,
        }

    def _singularity_binds(self) -> list[str]:
        binds = [f"{self.work_dir}:{CONTAINER_WORK_MOUNT}"]
        if self.config.cache_dir:
            cache = os.path.abspath(self.config.cache_dir)
            os.makedirs(cache, exist_ok=True)
            binds.append(f"{cache}:{CACHE_CONTAINER_MOUNT}")
        binds.extend(self.config.singularity_bind)
        return binds

    def _docker_user_args(self) -> list[str]:
        """Write outputs as the invoking host user (avoids root-owned / permission errors)."""
        if not hasattr(os, "getuid"):
            return []
        return ["-u", f"{os.getuid()}:{os.getgid()}"]

    def _docker_volume_args(self) -> list[str]:
        args = ["-v", f"{self.work_dir}:{CONTAINER_WORK_MOUNT}"]
        if self.config.cache_dir:
            cache = os.path.abspath(self.config.cache_dir)
            os.makedirs(cache, exist_ok=True)
            args.extend(["-v", f"{cache}:{CACHE_CONTAINER_MOUNT}"])
            for k, v in self._cache_env(for_container=True).items():
                args.extend(["-e", f"{k}={v}"])
        return args

    def _docker_gpu_args(self) -> list[str]:
        if not self.config.docker_gpu_devices:
            return []
        return [
            "--device=/dev/kfd",
            "--device=/dev/dri",
            "--group-add",
            "video",
            "--shm-size",
            self.config.docker_shm_size,
        ]

    def _run_subprocess(
        self,
        cmd: Sequence[str],
        *,
        env: dict[str, str] | None = None,
        check: bool = False,
        cwd: str | None = None,
    ) -> subprocess.CompletedProcess[str]:
        merged = os.environ.copy()
        if env:
            merged.update(env)
        print(f"-> exec: {' '.join(shlex.quote(c) for c in cmd)}")
        return subprocess.run(
            list(cmd),
            env=merged,
            check=check,
            text=True,
            cwd=cwd,
        )

    def run_colabfold_batch(
        self,
        fasta_host: str,
        out_dir_host: str,
        gpu_id: int,
        extra_args: Sequence[str],
    ) -> int:
        fasta_host = os.path.abspath(fasta_host)
        out_dir_host = os.path.abspath(out_dir_host)
        os.makedirs(out_dir_host, exist_ok=True)

        runtime = self.config.runtime.lower()
        in_container = runtime != "local"
        if runtime == "local":
            fasta_arg, out_arg = fasta_host, out_dir_host
        else:
            fasta_arg = host_to_container(fasta_host, self.work_dir)
            out_arg = host_to_container(out_dir_host, self.work_dir)

        inner_cmd = [
            "colabfold_batch",
            *extra_args,
            "--disable-unified-memory",
            fasta_arg,
            out_arg,
        ]
        env = {
            "HIP_VISIBLE_DEVICES": str(gpu_id),
            "XLA_PYTHON_CLIENT_PREALLOCATE": "false",
            "XLA_FLAGS": "--xla_gpu_autotune_level=0",
            **self._cache_env(for_container=in_container),
        }
        return self._run_colabfold_inner(inner_cmd, env)

    def _run_colabfold_inner(
        self, inner_cmd: Sequence[str], env: dict[str, str]
    ) -> int:
        runtime = self.config.runtime.lower()
        if runtime == "local":
            return self._run_subprocess(inner_cmd, env=env).returncode

        if runtime == "docker":
            if self.config.colabfold_container_name:
                cmd = [
                    "docker",
                    "exec",
                    "-w",
                    CONTAINER_WORK_MOUNT,
                ]
                for k, v in env.items():
                    cmd.extend(["-e", f"{k}={v}"])
                cmd.append(self.config.colabfold_container_name)
                cmd.extend(inner_cmd)
                return self._run_subprocess(cmd).returncode

            cmd = [
                "docker",
                "run",
                "--rm",
                *self._docker_user_args(),
                *self._docker_gpu_args(),
                *self._docker_volume_args(),
                "-w",
                CONTAINER_WORK_MOUNT,
            ]
            for k, v in env.items():
                cmd.extend(["-e", f"{k}={v}"])
            cmd.append(self.config.colabfold_image)
            cmd.extend(inner_cmd)
            return self._run_subprocess(cmd).returncode

        if runtime in ("singularity", "apptainer"):
            exe = runtime if shutil.which(runtime) else "apptainer"
            sif = self.config.colabfold_sif or self.config.colabfold_image
            if not sif or not os.path.isfile(sif):
                raise FileNotFoundError(
                    f"ColabFold Singularity image not found: {sif!r}. "
                    "Pass --colabfold-sif /path/to/colabfold.sif"
                )
            cmd = [exe, "exec"]
            if self.config.singularity_rocm:
                cmd.append("--rocm")
            for bind in self._singularity_binds():
                cmd.extend(["--bind", bind])
            for k, v in env.items():
                cmd.extend(["--env", f"{k}={v}"])
            cmd.extend([sif, *inner_cmd])
            return self._run_subprocess(cmd).returncode

        raise ValueError(
            f"Unknown runtime {runtime!r}; use docker, singularity, apptainer, or local."
        )

    def run_alphafold_fasta(
        self,
        fasta_host: str,
        output_dir_base_host: str,
        gpu_id: int,
        extra_args: Sequence[str],
    ) -> int:
        """Run ``run_alphafold.py`` for one FASTA; predictions go under ``output_dir_base/<fasta_stem>/``."""
        fasta_host = os.path.abspath(fasta_host)
        output_dir_base_host = os.path.abspath(output_dir_base_host)
        os.makedirs(output_dir_base_host, exist_ok=True)

        runtime = self.config.runtime.lower()
        if runtime == "local":
            fasta_arg = fasta_host
            out_base_arg = output_dir_base_host
        else:
            fasta_arg = host_to_container(fasta_host, self.work_dir)
            out_base_arg = host_to_container(output_dir_base_host, self.work_dir)

        ar = self.config.alphafold2_app_root
        inner_cmd: list[str]
        if runtime == "local":
            run_py = os.path.join(ar, "run_alphafold.py")
            inner_cmd = [
                sys.executable,
                run_py,
                f"--fasta_paths={fasta_arg}",
                f"--output_dir={out_base_arg}",
                *extra_args,
            ]
        else:
            inner_cmd = [
                "python3",
                f"{ar}/run_alphafold.py",
                f"--fasta_paths={fasta_arg}",
                f"--output_dir={out_base_arg}",
                *extra_args,
            ]
        env = {
            "HIP_VISIBLE_DEVICES": str(gpu_id),
            "XLA_PYTHON_CLIENT_PREALLOCATE": "false",
            "XLA_FLAGS": "--xla_gpu_autotune_level=0",
        }
        return self._run_alphafold_inner(inner_cmd, env)

    def _run_alphafold_inner(
        self, inner_cmd: Sequence[str], env: dict[str, str]
    ) -> int:
        runtime = self.config.runtime.lower()
        if runtime == "local":
            return self._run_subprocess(
                inner_cmd,
                env=env,
                cwd=self.config.alphafold2_app_root,
            ).returncode

        ar = self.config.alphafold2_app_root
        if runtime == "docker":
            if self.config.alphafold2_container_name:
                cmd = [
                    "docker",
                    "exec",
                    "-w",
                    ar,
                ]
                for k, v in env.items():
                    cmd.extend(["-e", f"{k}={v}"])
                cmd.append(self.config.alphafold2_container_name)
                cmd.extend(inner_cmd)
                return self._run_subprocess(cmd).returncode

            cmd = [
                "docker",
                "run",
                "--rm",
                *self._docker_user_args(),
                *self._docker_gpu_args(),
                *self._docker_volume_args(),
                "-w",
                ar,
            ]
            for k, v in env.items():
                cmd.extend(["-e", f"{k}={v}"])
            cmd.append(self.config.alphafold2_image)
            cmd.extend(inner_cmd)
            return self._run_subprocess(cmd).returncode

        if runtime in ("singularity", "apptainer"):
            exe = runtime if shutil.which(runtime) else "apptainer"
            sif = self.config.alphafold2_sif or self.config.alphafold2_image
            if not sif or not os.path.isfile(sif):
                raise FileNotFoundError(
                    f"AlphaFold2 Singularity image not found: {sif!r}. "
                    "Pass --alphafold2-sif /path/to/alphafold2.sif"
                )
            cmd = [exe, "exec"]
            if self.config.singularity_rocm:
                cmd.append("--rocm")
            for bind in self._singularity_binds():
                cmd.extend(["--bind", bind])
            for k, v in env.items():
                cmd.extend(["--env", f"{k}={v}"])
            cmd.extend([sif, *inner_cmd])
            return self._run_subprocess(cmd).returncode

        raise ValueError(
            f"Unknown runtime {runtime!r}; use docker, singularity, apptainer, or local."
        )

    def run_pymol_worker(
        self,
        subcommand: str,
        plan_host_path: str,
    ) -> int:
        plan_host_path = os.path.abspath(plan_host_path)
        runtime = self.config.runtime.lower()
        if runtime == "local":
            worker_path = self._pymol_worker_host
            plan_path = plan_host_path
        else:
            worker_path = self._pymol_worker_container_path()
            plan_path = host_to_container(plan_host_path, self.work_dir)
        inner = [
            self.config.pymol_command,
            "-cq",
            worker_path,
            "--",
            subcommand,
            "--plan",
            plan_path,
        ]
        return self._run_pymol_inner(inner)

    def _run_pymol_inner(self, inner_cmd: Sequence[str]) -> int:
        runtime = self.config.runtime.lower()
        if runtime == "local":
            if not shutil.which(self.config.pymol_command.split()[0]):
                raise FileNotFoundError(
                    f"PyMOL not found on PATH ({self.config.pymol_command!r}). "
                    "Install PyMOL or use --runtime docker/singularity with --pymol-image."
                )
            return self._run_subprocess(
                inner_cmd,
                env={
                    "PYTHONPATH": self._pymol_pythonpath(for_container=False),
                    "SPLIT_FOLD_WORK_DIR": self.work_dir,
                },
            ).returncode

        if runtime == "docker":
            cmd = [
                "docker",
                "run",
                "--rm",
                *self._docker_user_args(),
                "-e",
                f"PYTHONPATH={self._pymol_pythonpath(for_container=True)}",
                "-e",
                f"SPLIT_FOLD_WORK_DIR={CONTAINER_WORK_MOUNT}",
                "-v",
                f"{self.work_dir}:{CONTAINER_WORK_MOUNT}",
                *self._scripts_docker_volume_args(),
                "-w",
                CONTAINER_WORK_MOUNT,
                self.config.pymol_image,
                *inner_cmd,
            ]
            return self._run_subprocess(cmd).returncode

        if runtime in ("singularity", "apptainer"):
            exe = runtime if shutil.which(runtime) else "apptainer"
            sif = self.config.pymol_sif or self.config.pymol_image
            if not sif or not os.path.isfile(sif):
                raise FileNotFoundError(
                    f"PyMOL Singularity image not found: {sif!r}. "
                    "Pass --pymol-sif /path/to/pymol.sif"
                )
            cmd = [
                exe,
                "exec",
                "--env",
                f"PYTHONPATH={self._pymol_pythonpath(for_container=True)}",
                "--env",
                f"SPLIT_FOLD_WORK_DIR={CONTAINER_WORK_MOUNT}",
            ]
            for bind in self._singularity_binds():
                cmd.extend(["--bind", bind])
            cmd.extend(["--bind", self._scripts_singularity_bind()])
            cmd.extend([sif, *inner_cmd])
            return self._run_subprocess(cmd).returncode

        raise ValueError(f"Unknown runtime {runtime!r}")

    def all_chunk_folds_done(self, out_dirs: Sequence[str]) -> bool:
        for d in out_dirs:
            if not glob.glob(os.path.join(d, "*_rank_001*.pdb")):
                return False
        return bool(out_dirs)

    def all_chunk_folds_done_af2(self, pred_dirs: Sequence[str]) -> bool:
        """Each path is the AlphaFold prediction directory (contains ``ranked_*.pdb``)."""
        for d in pred_dirs:
            p0 = os.path.join(d, "ranked_0.pdb")
            if os.path.isfile(p0):
                continue
            if not glob.glob(os.path.join(d, "ranked_*.pdb")):
                return False
        return bool(pred_dirs)


def detect_default_runtime() -> str:
    if shutil.which("docker"):
        return "docker"
    if shutil.which("apptainer") or shutil.which("singularity"):
        return "singularity"
    return "local"


def add_container_cli_args(parser) -> None:
    g = parser.add_argument_group("container orchestration")
    g.add_argument(
        "--runtime",
        choices=("docker", "singularity", "apptainer", "local"),
        default=os.environ.get("SPLIT_FOLD_RUNTIME") or detect_default_runtime(),
        help="how to run ColabFold and PyMOL (default: docker if available, else singularity, else local)",
    )
    g.add_argument(
        "--work-dir",
        default=None,
        help="host directory bind-mounted as /work (default: common parent of input paths)",
    )
    g.add_argument(
        "--colabfold-image",
        default=os.environ.get("COLABFOLD_IMAGE", DEFAULT_COLABFOLD_IMAGE),
        help=f"Docker image for colabfold_batch (default: {DEFAULT_COLABFOLD_IMAGE})",
    )
    g.add_argument(
        "--pymol-image",
        default=os.environ.get("PYMOL_IMAGE", DEFAULT_PYMOL_IMAGE),
        help=f"Docker image for PyMOL worker (default: {DEFAULT_PYMOL_IMAGE})",
    )
    g.add_argument(
        "--colabfold-sif",
        default=os.environ.get("COLABFOLD_SIF"),
        help="Singularity/Apptainer image for ColabFold (required for singularity runtime unless converted .sif path)",
    )
    g.add_argument(
        "--pymol-sif",
        default=os.environ.get("PYMOL_SIF"),
        help="Singularity/Apptainer image for PyMOL",
    )
    g.add_argument(
        "--colabfold-cache",
        default=os.environ.get("COLABFOLD_CACHE_DIR"),
        help=(
            "host ColabFold cache directory (mounted at /cache; sets XDG_CACHE_HOME=/cache "
            "so weights load from <cache>/colabfold/params)"
        ),
    )
    g.add_argument(
        "--colabfold-container-name",
        default=os.environ.get("COLABFOLD_CONTAINER_NAME"),
        help="reuse a long-running ColabFold container via docker exec instead of docker run",
    )
    g.add_argument(
        "--singularity-bind",
        action="append",
        default=[],
        help="extra singularity --bind specs (repeatable), e.g. /scratch:/scratch",
    )
    g.add_argument(
        "--no-singularity-rocm",
        action="store_true",
        help="do not pass --rocm to singularity exec (use for CPU-only images)",
    )
    g.add_argument(
        "--no-docker-gpu",
        action="store_true",
        help="omit /dev/kfd and /dev/dri from docker run (CPU-only ColabFold)",
    )


def add_alphafold2_container_cli_args(parser) -> None:
    """CLI for ``split_and_fold_segments_alphafold2.py`` (AlphaFold2 + PyMOL containers)."""
    g = parser.add_argument_group("AlphaFold2 container")
    g.add_argument(
        "--alphafold2-image",
        default=os.environ.get("ALPHAFOLD2_IMAGE", DEFAULT_ALPHAFOLD2_IMAGE),
        help=f"Docker image for run_alphafold.py (default: {DEFAULT_ALPHAFOLD2_IMAGE})",
    )
    g.add_argument(
        "--alphafold2-sif",
        default=os.environ.get("ALPHAFOLD2_SIF"),
        help="Singularity/Apptainer .sif for AlphaFold2 (singularity/apptainer runtime)",
    )
    g.add_argument(
        "--alphafold2-container-name",
        default=os.environ.get("ALPHAFOLD2_CONTAINER_NAME"),
        help="reuse a long-running AlphaFold2 container via docker exec",
    )
    g.add_argument(
        "--alphafold2-app-root",
        default=os.environ.get("ALPHAFOLD2_APP_ROOT", "/app/alphafold"),
        help="path to AlphaFold clone inside AF2 image (and cwd for --runtime local)",
    )


def container_config_from_args(args, work_dir: str) -> ContainerConfig:
    cfg = ContainerConfig(
        runtime=args.runtime,
        work_dir=work_dir,
        colabfold_image=args.colabfold_image,
        pymol_image=args.pymol_image,
        colabfold_sif=args.colabfold_sif,
        pymol_sif=args.pymol_sif,
        cache_dir=args.colabfold_cache,
        colabfold_container_name=args.colabfold_container_name,
        singularity_bind=list(args.singularity_bind or []),
        singularity_rocm=not args.no_singularity_rocm,
        docker_gpu_devices=not args.no_docker_gpu,
    )
    if hasattr(args, "alphafold2_image"):
        cfg.alphafold2_image = args.alphafold2_image
        cfg.alphafold2_sif = args.alphafold2_sif
        cfg.alphafold2_container_name = args.alphafold2_container_name
        cfg.alphafold2_app_root = args.alphafold2_app_root
    return cfg
