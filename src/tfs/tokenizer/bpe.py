"""Byte-level BPE with deterministic training and heap-based ranked encoding.

The pre-tokenization pattern follows the general GPT-2 regex design; the merge
algorithm, serialization, and special-token policy are implemented here.

No reference tokenization library is imported into production code.
"""

import heapq
import json
from collections import Counter
from collections.abc import Iterable, Iterator, Sequence
from pathlib import Path

import regex

GPT2_PRETOKEN_PATTERN = (
    r"'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"
)
_PATTERN = regex.compile(GPT2_PRETOKEN_PATTERN)


def _pieces(document: str | bytes) -> Iterator[bytes]:
    """Split text without normalization, preserving even invalid UTF-8 bytes."""
    if isinstance(document, bytes):
        text = document.decode("utf-8", errors="surrogateescape")
        errors = "surrogateescape"
    elif isinstance(document, str):
        text = document
        errors = "surrogatepass"
    else:
        raise TypeError("Tokenizer input must be str or bytes")

    for match in _PATTERN.finditer(text):
        yield match.group().encode("utf-8", errors=errors)


def _apply_pair(piece: list[int], pair: tuple[int, int], new_id: int) -> list[int]:
    """Merge all non-overlapping occurrences, scanning from left to right."""
    output: list[int] = []
    i = 0
    while i < len(piece):
        if i + 1 < len(piece) and (piece[i], piece[i + 1]) == pair:
            output.append(new_id)
            i += 2
        else:
            output.append(piece[i])
            i += 1
    return output


class ByteBPETokenizer:
    """A lossless byte tokenizer with explicit special-token opt-in.

    Merge identifiers start at 256. Special-token identifiers come after all
    merges. Training never merges across a GPT-2-style pre-token boundary.
    """

    def __init__(
        self,
        merges: Sequence[tuple[int, int]] = (),
        special_tokens: Sequence[str] = ("<|endoftext|>",),
    ) -> None:
        self.merges = list(merges)
        self.special_tokens = tuple(special_tokens)
        if len(set(self.special_tokens)) != len(self.special_tokens):
            raise ValueError("Duplicate special token")

        self.vocab: list[bytes] = [bytes([i]) for i in range(256)]
        self.merge_ranks: dict[tuple[int, int], int] = {}
        for rank, pair in enumerate(self.merges):
            left, right = pair
            if left < 0 or right < 0 or left >= len(self.vocab) or right >= len(self.vocab):
                raise ValueError(f"Invalid merge at rank {rank}: {pair}")
            if pair in self.merge_ranks:
                raise ValueError(f"Duplicate merge: {pair}")
            self.merge_ranks[pair] = rank
            self.vocab.append(self.vocab[left] + self.vocab[right])

        self.special_ids = {
            token: len(self.vocab) + offset for offset, token in enumerate(self.special_tokens)
        }
        self.id_to_special = {id_: token for token, id_ in self.special_ids.items()}

    @property
    def vocab_size(self) -> int:
        return len(self.vocab) + len(self.special_tokens)

    @classmethod
    def train(
        cls,
        texts: Iterable[str | bytes],
        vocab_size: int,
        special_tokens: Sequence[str] = ("<|endoftext|>",),
    ) -> "ByteBPETokenizer":
        """Train deterministic BPE merges by corpus pair frequency.

        Resolve frequency ties by lexicographically smaller (left_id, right_id).
        M1 uses full pair recounting per merge; streaming/incremental training
        for very large corpora is tracked in docs/LIMITATIONS.md.
        """
        if len(set(special_tokens)) != len(special_tokens):
            raise ValueError("Duplicate special token")
        if vocab_size < 256 + len(special_tokens):
            raise ValueError("vocab_size must hold all 256 bytes and special tokens")

        chunks = [list(part) for text in texts for part in _pieces(text)]
        merges: list[tuple[int, int]] = []
        max_merges = vocab_size - 256 - len(special_tokens)

        for rank in range(max_merges):
            frequencies: Counter[tuple[int, int]] = Counter()
            for chunk in chunks:
                frequencies.update(zip(chunk, chunk[1:], strict=False))
            if not frequencies:
                break
            best = min(frequencies, key=lambda pair: (-frequencies[pair], pair))
            new_id = 256 + rank
            chunks = [_apply_pair(chunk, best, new_id) for chunk in chunks]
            merges.append(best)

        return cls(merges, special_tokens)

    def _encode_piece(self, piece: bytes) -> list[int]:
        """Apply merge rules in rank order via a heap and linked neighbors.

        Each valid merge removes one linked-list node; stale heap entries
        are ignored. This avoids repeatedly rescanning long pieces.
        """
        n = len(piece)
        if n == 0:
            return []
        ids = list(piece)
        prev = [i - 1 for i in range(n)]
        next_ = [i + 1 for i in range(n)]
        next_[-1] = -1
        alive = [True] * n
        heap: list[tuple[int, int, int, int]] = []

        def candidate(index: int) -> None:
            if index < 0 or not alive[index]:
                return
            neighbor = next_[index]
            if neighbor < 0 or not alive[neighbor]:
                return
            pair = (ids[index], ids[neighbor])
            rank = self.merge_ranks.get(pair)
            if rank is not None:
                heapq.heappush(heap, (rank, index, pair[0], pair[1]))

        for index in range(n - 1):
            candidate(index)

        while heap:
            rank, left_index, a, b = heapq.heappop(heap)
            if not alive[left_index]:
                continue
            right_index = next_[left_index]
            if right_index < 0 or not alive[right_index]:
                continue
            if ids[left_index] != a or ids[right_index] != b:
                continue

            ids[left_index] = 256 + rank
            alive[right_index] = False
            next_[left_index] = next_[right_index]
            if next_[left_index] >= 0:
                prev[next_[left_index]] = left_index
            candidate(prev[left_index])
            candidate(left_index)

        output: list[int] = []
        index = 0
        while index >= 0:
            output.append(ids[index])
            index = next_[index]
        return output

    def encode(
        self,
        text: str | bytes,
        *,
        allowed_special: Iterable[str] = (),
    ) -> list[int]:
        """Return token IDs; specials in ordinary user text are never implicit."""
        allowed = set(allowed_special)
        unknown = allowed - self.special_ids.keys()
        if unknown:
            raise ValueError(f"Unknown allowed special token(s): {sorted(unknown)!r}")
        if not allowed:
            return [id_ for part in _pieces(text) for id_ in self._encode_piece(part)]
        if not isinstance(text, str):
            raise TypeError("Special-token recognition requires str input")

        escaped = sorted((regex.escape(item) for item in allowed), key=len, reverse=True)
        splitter = regex.compile("(" + "|".join(escaped) + ")")
        tokens: list[int] = []
        for segment in splitter.split(text):
            if segment in allowed:
                tokens.append(self.special_ids[segment])
            else:
                tokens.extend(self.encode(segment))
        return tokens

    def encode_bytes(self, raw: bytes) -> list[int]:
        """Encode arbitrary bytes, including malformed UTF-8 and NUL bytes."""
        return self.encode(raw)

    def decode_bytes(self, ids: Iterable[int]) -> bytes:
        """Losslessly reconstruct raw bytes. Raises for unknown token IDs."""
        output = bytearray()
        for id_ in ids:
            if 0 <= id_ < len(self.vocab):
                output.extend(self.vocab[id_])
            elif id_ in self.id_to_special:
                output.extend(self.id_to_special[id_].encode("utf-8"))
            else:
                raise ValueError(f"Unknown token ID: {id_}")
        return bytes(output)

    def decode(self, ids: Iterable[int], *, errors: str = "replace") -> str:
        """Decode UTF-8 text, replacing malformed byte sequences by default."""
        return self.decode_bytes(ids).decode("utf-8", errors=errors)

    def save(self, directory: Path) -> None:
        """Write portable JSON metadata and an independently checked merge list."""
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        record = {
            "version": 1,
            "normalization": "none",
            "pretoken_pattern": GPT2_PRETOKEN_PATTERN,
            "merges": [list(pair) for pair in self.merges],
            "special_tokens": list(self.special_tokens),
            "vocab_size": self.vocab_size,
        }
        (directory / "tokenizer.json").write_text(
            json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        (directory / "merges.txt").write_text(
            "".join(f"{left} {right}\n" for left, right in self.merges),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, directory: Path) -> "ByteBPETokenizer":
        """Load and check both files; refuse mismatched or unknown formats."""
        directory = Path(directory)
        record = json.loads((directory / "tokenizer.json").read_text(encoding="utf-8"))
        if record.get("version") != 1 or record.get("normalization") != "none":
            raise ValueError("Unsupported tokenizer format")
        if record.get("pretoken_pattern") != GPT2_PRETOKEN_PATTERN:
            raise ValueError("Pre-tokenizer mismatch")
        merges = [tuple(item) for item in record["merges"]]
        specials = record["special_tokens"]
        model = cls(merges, specials)
        if model.vocab_size != record["vocab_size"]:
            raise ValueError("Vocabulary size mismatch")
        expected = "".join(f"{left} {right}\n" for left, right in model.merges)
        actual = (directory / "merges.txt").read_text(encoding="utf-8")
        if expected != actual:
            raise ValueError("Merges file differs from JSON metadata")
        return model
