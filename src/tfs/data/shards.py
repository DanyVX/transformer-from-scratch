"""Document-separated splits with checksummed memmapped tokens."""

import hashlib
import json
import random
from collections.abc import Callable, Sequence
from pathlib import Path

import numpy as np


def split_documents(
    documents: Sequence[str], validation_fraction: float = 0.2, seed: int = 42
) -> tuple[list[str], list[str]]:
    """Split documents before sampling windows to avoid overlap leakage."""
    if len(documents) < 2:
        raise ValueError("At least two documents are needed for a validation split")
    if not 0 < validation_fraction < 1:
        raise ValueError("validation_fraction must be strictly between zero and one")
    ordered = list(range(len(documents)))
    random.Random(seed).shuffle(ordered)
    val_count = min(len(documents) - 1, max(1, round(len(documents) * validation_fraction)))
    val_ids = set(ordered[:val_count])
    return (
        [doc for index, doc in enumerate(documents) if index not in val_ids],
        [doc for index, doc in enumerate(documents) if index in val_ids],
    )


def save_token_shards(
    directory: Path,
    splits: dict[str, Sequence[str]],
    encode: Callable[[str], Sequence[int]],
    vocab_size: int,
    separator_id: int,
) -> dict[str, object]:
    """Build uint16/32 shards; validate IDs and store exact file checksums."""
    if vocab_size < 1 or separator_id < 0 or separator_id >= vocab_size:
        raise ValueError("Invalid vocab_size or document separator token")
    directory.mkdir(parents=True, exist_ok=True)
    dtype = np.dtype("<u2" if vocab_size <= 65535 else "<u4")
    manifest: dict[str, object] = {
        "schema": 1,
        "dtype": dtype.str,
        "vocab_size": vocab_size,
        "separator_id": separator_id,
        "shards": {},
    }
    for name, docs in splits.items():
        if not name.isidentifier():
            raise ValueError(f"Invalid shard name: {name!r}")
        token_list: list[int] = []
        for document in docs:
            encoded = list(encode(document))
            if any(not isinstance(token, int) or token < 0 or token >= vocab_size for token in encoded):
                raise ValueError("Tokenizer emitted out-of-vocabulary ID")
            token_list.extend(encoded)
            token_list.append(separator_id)
        data = np.array(token_list, dtype=dtype)
        path = directory / f"{name}.bin"
        data.tofile(path)
        checksum = hashlib.sha256(path.read_bytes()).hexdigest()
        shards = manifest["shards"]
        if not isinstance(shards, dict):
            raise AssertionError("Corrupt manifest builder")
        shards[name] = {
            "file": path.name, "tokens": int(data.size), "sha256": checksum,
            "documents": len(docs),
        }
    (directory / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest


def load_shard(directory: Path, name: str, *, verify_checksum: bool = True) -> np.memmap:
    """Refuse truncated/corrupted data, unsafe paths, and missing manifests."""
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("schema") != 1:
        raise ValueError("Unsupported shard manifest version")
    info = manifest["shards"].get(name)
    if info is None:
        raise ValueError(f"Unknown shard: {name}")
    filename = info["file"]
    if Path(filename).name != filename or not filename.endswith(".bin"):
        raise ValueError("Unsafe shard filename")
    path = directory / filename
    dtype = np.dtype(manifest["dtype"])
    expected_bytes = int(info["tokens"]) * dtype.itemsize
    if path.stat().st_size != expected_bytes:
        raise ValueError("Truncated or oversized token shard")
    if verify_checksum and hashlib.sha256(path.read_bytes()).hexdigest() != info["sha256"]:
        raise ValueError("Shard checksum mismatch")
    return np.memmap(path, mode="r", dtype=dtype, shape=(int(info["tokens"]),))


class WindowSampler:
    """Checkpointable random or sequential next-token windows.

    Generator state plus cursor restore deterministic batch order.
    """

    def __init__(
        self, tokens: np.ndarray, block_size: int, batch_size: int,
        seed: int = 42, sequential: bool = False,
    ) -> None:
        if block_size < 1 or batch_size < 1:
            raise ValueError("block_size and batch_size must be positive")
        if len(tokens) <= block_size:
            raise ValueError("Not enough tokens for even one input/target window")
        self.tokens = tokens
        self.block_size = block_size
        self.batch_size = batch_size
        self.rng = np.random.default_rng(seed)
        self.sequential = sequential
        self.cursor = 0

    def next_batch(self) -> tuple[np.ndarray, np.ndarray]:
        """Get X[t] and target[t]=raw[t+1], with no off-by-one shift."""
        upper = len(self.tokens) - self.block_size
        if self.sequential:
            starts = (self.cursor + np.arange(self.batch_size) * self.block_size) % upper
            self.cursor = int((self.cursor + self.batch_size * self.block_size) % upper)
        else:
            starts = self.rng.integers(0, upper, size=self.batch_size)
        x = np.stack([self.tokens[i : i + self.block_size] for i in starts]).astype(np.int64)
        y = np.stack(
            [self.tokens[i + 1 : i + self.block_size + 1] for i in starts]
        ).astype(np.int64)
        return x, y

    def state_dict(self) -> dict[str, object]:
        return {"cursor": self.cursor, "rng": self.rng.bit_generator.state}

    def load_state_dict(self, state: dict[str, object]) -> None:
        self.cursor = int(state["cursor"])
        self.rng.bit_generator.state = state["rng"]
