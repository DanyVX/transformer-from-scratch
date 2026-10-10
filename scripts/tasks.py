"""Cross-platform task runner: python scripts/tasks.py <task>.

Make is an optional thin wrapper, not a requirement for Windows users.
"""

import argparse
import importlib.util
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def run(*command: str) -> None:
    subprocess.run(command, cwd=ROOT, check=True)


def main() -> None:
    choices = (
        "setup", "lint", "format", "typecheck", "test", "test-fast",
        "bench", "run", "docker-build", "clean",
    )
    parser = argparse.ArgumentParser()
    parser.add_argument("task", choices=choices)
    args = parser.parse_args()
    python = sys.executable
    if args.task == "setup":
        if shutil.which("uv"):
            run("uv", "sync", "--extra", "dev")
        else:
            run(python, "-m", "pip", "install", "-e", ".[dev]")
    elif args.task == "lint":
        run(python, "-m", "ruff", "check", ".")
        run(python, "-m", "ruff", "format", "--check", ".")
    elif args.task == "format":
        run(python, "-m", "ruff", "check", "--fix", ".")
        run(python, "-m", "ruff", "format", ".")
    elif args.task == "typecheck":
        run(python, "-m", "mypy")
    elif args.task == "test":
        run(python, "-m", "pytest", "--cov=tfs", "--cov-report=term-missing")
    elif args.task == "test-fast":
        run(python, "-m", "pytest", "-m", "not slow and not gpu and not network")
    elif args.task == "bench":
        raise SystemExit("No benchmark has been implemented (M1+). No numbers fabricated.")
    elif args.task == "run":
        run(python, "-m", "tfs", "config", "--profile", "tiny")
    elif args.task == "docker-build":
        if not (ROOT / "Dockerfile").is_file():
            raise SystemExit("Dockerfile is not implemented; see docs/LIMITATIONS.md.")
        run("docker", "build", "-t", "transformer-from-scratch", ".")
    elif args.task == "clean":
        for name in (".pytest_cache", ".mypy_cache", ".ruff_cache", "build"):
            shutil.rmtree(ROOT / name, ignore_errors=True)
    else:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
