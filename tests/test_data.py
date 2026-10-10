"""Shard corruption checks and deterministic stream round-trips."""

import numpy as np
import pytest

from tfs.data.shards import WindowSampler, load_shard, save_token_shards, split_documents


def test_split_by_documents_with_fixed_seed() -> None:
    documents = [f"document-{i:04d}" for i in range(20)]
    train, val = split_documents(documents, seed=15)
    assert not set(train).intersection(val)
    assert len(train) + len(val) == len(documents)
    assert split_documents(documents, seed=15) == (train, val)


def test_small_split_is_rejected() -> None:
    with pytest.raises(ValueError):
        split_documents(["only one"])


def test_memmap_round_trip_and_shift(tmp_path) -> None:
    docs = {"train": ["abc", "def"], "val": ["ghi"]}
    save_token_shards(tmp_path, docs, lambda s: list(s.encode()), 512, 511)
    shard = load_shard(tmp_path, "train")
    assert list(shard) == [97, 98, 99, 511, 100, 101, 102, 511]
    sampler = WindowSampler(shard, block_size=3, batch_size=2, seed=12)
    x, y = sampler.next_batch()
    np.testing.assert_array_equal(x[:, 1:], y[:, :-1])


def test_resume_random_windows_exact(tmp_path) -> None:
    data = np.arange(100, dtype=np.uint16)
    first = WindowSampler(data, 8, 3, seed=123)
    first.next_batch()
    state = first.state_dict()
    next_x, next_y = first.next_batch()
    second = WindowSampler(data, 8, 3, seed=123)
    second.load_state_dict(state)
    x, y = second.next_batch()
    np.testing.assert_array_equal(x, next_x)
    np.testing.assert_array_equal(y, next_y)


def test_sequential_sampler() -> None:
    sampler = WindowSampler(np.arange(32), block_size=4, batch_size=2, sequential=True)
    x, y = sampler.next_batch()
    np.testing.assert_array_equal(x[0], [0, 1, 2, 3])
    np.testing.assert_array_equal(y[0], [1, 2, 3, 4])
    np.testing.assert_array_equal(x[1], [4, 5, 6, 7])


def test_truncated_or_changed_shard_detected(tmp_path) -> None:
    save_token_shards(tmp_path, {"train": ["hello"]}, lambda s: list(s.encode()), 512, 511)
    path = tmp_path / "train.bin"
    path.write_bytes(path.read_bytes()[:-1])
    with pytest.raises(ValueError, match="Truncated"):
        load_shard(tmp_path, "train")


def test_large_vocab_uses_uint32(tmp_path) -> None:
    manifest = save_token_shards(
        tmp_path, {"train": ["x"]}, lambda _: [70_000], 100_000, 99_999
    )
    assert manifest["dtype"] == "<u4"
    assert load_shard(tmp_path, "train").dtype.itemsize == 4


def test_bad_token_rejected(tmp_path) -> None:
    with pytest.raises(ValueError, match="out-of-vocabulary"):
        save_token_shards(tmp_path, {"train": ["bad"]}, lambda _: [-1], 512, 511)
