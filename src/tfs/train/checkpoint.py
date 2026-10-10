"""Atomic, trusted-local checkpoint saving and resume validation."""

import hashlib
import json
import os
import random
import tempfile
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch


def config_fingerprint(config: dict[str, Any]) -> str:
    canonical = json.dumps(config, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(canonical).hexdigest()


def capture_rng() -> dict[str, Any]:
    result: dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        result["cuda"] = torch.cuda.get_rng_state_all()
    return result


def restore_rng(state: dict[str, Any]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if torch.cuda.is_available() and "cuda" in state:
        torch.cuda.set_rng_state_all(state["cuda"])


def save_checkpoint(path: Path, state: dict[str, Any]) -> None:
    """Tempfile on same filesystem + atomic rename; local files only."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", delete=False
        ) as handle:
            temporary = handle.name
        torch.save(state, temporary)
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            Path(temporary).unlink(missing_ok=True)


def load_checkpoint(path: Path, expected_fingerprint: str) -> dict[str, Any]:
    """Read only a locally trusted checkpoint containing optimizer and RNG state."""
    state = torch.load(path, map_location="cpu", weights_only=False)
    if state.get("fingerprint") != expected_fingerprint:
        raise ValueError("Checkpoint configuration mismatch; refusing unsafe resume")
    return cast(dict[str, Any], state)
