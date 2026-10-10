"""Safe, reproducible runtime metadata and atomic JSON result files."""

import importlib
import json
import os
import platform
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any


def _output(command: list[str]) -> str | None:
    """Read one local tool's short version string without using a shell."""
    if shutil.which(command[0]) is None:
        return None
    try:
        proc = subprocess.run(
            command, capture_output=True, text=True, timeout=3, check=False
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    return proc.stdout.strip() or proc.stderr.strip() or None


def environment_metadata() -> dict[str, Any]:
    """Build a non-sensitive environment snapshot for future benchmark records.

    No usernames, home paths, environment values, raw inputs, or secrets are stored.
    """
    try:
        torch = importlib.import_module("torch")

        torch_version = torch.__version__
        cuda_available = torch.cuda.is_available()
        cuda_version = torch.version.cuda
        gpu_name = torch.cuda.get_device_name(0) if cuda_available else None
    except ImportError:
        torch_version = None
        cuda_available = False
        cuda_version = None
        gpu_name = None

    total_memory_bytes: int | None = None
    try:
        total_memory_bytes = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
    except (AttributeError, OSError, ValueError):
        pass

    disk_free_bytes = shutil.disk_usage(Path.cwd()).free

    return {
        "os": platform.system(),
        "architecture": platform.machine(),
        "python": platform.python_version(),
        "torch": torch_version,
        "cuda_available": cuda_available,
        "cuda_version": cuda_version,
        "gpu": gpu_name,
        "cpu_count": os.cpu_count(),
        "memory_bytes": total_memory_bytes,
        "disk_free_bytes": disk_free_bytes,
        "uv_available": shutil.which("uv") is not None,
        "pip_available": _output([sys.executable, "-m", "pip", "--version"]) is not None,
        "docker_available": shutil.which("docker") is not None,
        "ffmpeg_available": shutil.which("ffmpeg") is not None,
        "git_available": shutil.which("git") is not None,
        "git_commit": (_output(["git", "rev-parse", "HEAD"]) or "unavailable").splitlines()[0],
        "internet_access": "not tested (run network check separately)",
    }


def write_result(path: Path, metrics: dict[str, Any], *, config: dict[str, Any]) -> None:
    """Write evidence atomically; metadata is captured automatically.

    Called by future bench/eval scripts, never by README hand-entered values.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": 1,
        "environment": environment_metadata(),
        "config": config,
        "metrics": metrics,
    }
    temp_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as tmp:
            temp_name = tmp.name
            json.dump(payload, tmp, indent=2, sort_keys=True, allow_nan=False)
            tmp.write("\n")
        os.replace(temp_name, path)
    finally:
        if temp_name is not None:
            Path(temp_name).unlink(missing_ok=True)
