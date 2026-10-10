"""Incremental UTF-8 decoding buffers incomplete multibyte BPE fragments."""

import codecs
from collections.abc import Iterable, Iterator

from tfs.tokenizer import ByteBPETokenizer


def stream_text(tokenizer: ByteBPETokenizer, token_ids: Iterable[int]) -> Iterator[str]:
    """Yield decoded fragments without splitting valid multi-byte code points."""
    decoder = codecs.getincrementaldecoder("utf-8")("replace")
    for token_id in token_ids:
        fragment = decoder.decode(tokenizer.decode_bytes([token_id]), final=False)
        if fragment:
            yield fragment
    last = decoder.decode(b"", final=True)
    if last:
        yield last
