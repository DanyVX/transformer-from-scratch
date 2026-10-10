"""Tests for byte-level BPE, including arbitrary bytes and special injection."""

from pathlib import Path

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from tfs.tokenizer import ByteBPETokenizer
from tfs.tokenizer.bpe import _apply_pair, _pieces


@pytest.fixture(scope="module")
def tokenizer() -> ByteBPETokenizer:
    return ByteBPETokenizer.train(
        [
            "hello world\nhello tokenizer\n",
            "سلام دنیا مرحبا بالعالم",
            "print('hello')\t42\n",
            "<|endoftext|> should be ordinary user text",
            "abababababaa aaaaa 1234567890",
        ],
        vocab_size=320,
    )


@pytest.mark.parametrize(
    "text",
    [
        "",
        " ",
        "  \t\n \r\n",
        " hello ",
        "\x00null\x00",
        "👩🏽‍💻 👨‍👩‍👧‍👦 café e\u0301",
        "سلام دنیا اردو العربية",
        "12345678901234567890",
        "a" * 2000,
        "<|endoftext|>",
        "\n\n\n\t",
    ],
)
def test_text_roundtrip(tokenizer: ByteBPETokenizer, text: str) -> None:
    assert tokenizer.decode(tokenizer.encode(text)) == text


@given(st.binary(max_size=256))
@settings(max_examples=80, deadline=None)
def test_arbitrary_bytes_roundtrip(tokenizer: ByteBPETokenizer, data: bytes) -> None:
    assert tokenizer.decode_bytes(tokenizer.encode_bytes(data)) == data


@given(st.text(alphabet=st.characters(blacklist_categories=["Cs"]), max_size=200))
@settings(max_examples=70, deadline=None)
def test_unicode_roundtrip(tokenizer: ByteBPETokenizer, text: str) -> None:
    assert tokenizer.decode(tokenizer.encode(text), errors="strict") == text


def test_lossless_invalid_utf8(tokenizer: ByteBPETokenizer) -> None:
    raw = b"\xff\xfe\x80hi\x00\xf0\x28\x8c\x28"
    ids = tokenizer.encode(raw)
    assert tokenizer.decode_bytes(ids) == raw
    assert "\ufffd" in tokenizer.decode(ids)


def test_special_token_opt_in(tokenizer: ByteBPETokenizer) -> None:
    marker = "<|endoftext|>"
    text = f"before {marker} after"
    ordinary = tokenizer.encode(text)
    enabled = tokenizer.encode(text, allowed_special={marker})
    assert tokenizer.special_ids[marker] not in ordinary
    assert enabled.count(tokenizer.special_ids[marker]) == 1
    assert tokenizer.decode(enabled) == text
    with pytest.raises(ValueError, match="Unknown allowed"):
        tokenizer.encode(text, allowed_special={"<|unknown|>"})
    with pytest.raises(TypeError, match="requires str"):
        tokenizer.encode(b"test", allowed_special={marker})


def test_deterministic_tie_breaking_and_vocab_arithmetic() -> None:
    first = ByteBPETokenizer.train(["aba"], vocab_size=258, special_tokens=())
    second = ByteBPETokenizer.train(["aba"], vocab_size=258, special_tokens=())
    assert first.merges == second.merges
    assert first.merges[0] == (97, 98)
    assert first.vocab_size == 258
    assert first.decode(first.encode("aba")) == "aba"


def test_training_small_or_empty_corpus() -> None:
    empty = ByteBPETokenizer.train([], vocab_size=1000)
    assert empty.vocab_size == 257
    assert empty.encode("anything") == list(b"anything")
    with pytest.raises(ValueError, match="vocab_size"):
        ByteBPETokenizer.train(["text"], vocab_size=256)


def test_no_cross_pretoken_boundary() -> None:
    model = ByteBPETokenizer.train(["a b"], vocab_size=260, special_tokens=())
    assert all(model.vocab[left] + model.vocab[right] != b"a b" for left, right in model.merges)


def test_heap_encoder_agrees_with_naive_rank_order(
    tokenizer: ByteBPETokenizer,
) -> None:
    samples = ["hellooooo" * 40, "سلام دنیا" * 10, "   \n", "<|endoftext|>"]
    for sample in samples:
        reference: list[int] = []
        for piece in _pieces(sample):
            ids = list(piece)
            for rank, pair in enumerate(tokenizer.merges):
                ids = _apply_pair(ids, pair, 256 + rank)
            reference.extend(ids)
        assert tokenizer.encode(sample) == reference


def test_long_repeated_input(tokenizer: ByteBPETokenizer) -> None:
    data = ("ababaaaaa" * 3000).encode("utf-8")
    assert tokenizer.decode_bytes(tokenizer.encode_bytes(data)) == data


def test_save_load_and_tamper_detection(
    tokenizer: ByteBPETokenizer, tmp_path: Path
) -> None:
    tokenizer.save(tmp_path)
    reloaded = ByteBPETokenizer.load(tmp_path)
    assert reloaded.merges == tokenizer.merges
    assert reloaded.special_ids == tokenizer.special_ids
    assert reloaded.encode("سلام hello 123") == tokenizer.encode("سلام hello 123")

    path = tmp_path / "merges.txt"
    path.write_text("999 999\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Merges file differs"):
        ByteBPETokenizer.load(tmp_path)


def test_invalid_merge_and_unknown_id() -> None:
    with pytest.raises(ValueError, match="Invalid merge"):
        ByteBPETokenizer(merges=[(300, 1)])
    with pytest.raises(ValueError, match="Duplicate special"):
        ByteBPETokenizer(special_tokens=("<eos>", "<eos>"))
    with pytest.raises(ValueError, match="Unknown token ID"):
        ByteBPETokenizer().decode([9999])
