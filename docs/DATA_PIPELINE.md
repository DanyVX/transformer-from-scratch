# M4 corpus pipeline

Split at the **document level** with an explicit seed before tokenizing; never select overlapping random windows before splitting. Shards store one separator token after each document, and record dtype, byte size, SHA-256, document count, and tokenizer vocabulary bounds. uint32 is selected when the vocabulary exceeds 65535 entries. The memmap loader validates size and optionally checksum.

A deterministic sampler returns input positions t..t+block-1 and targets t+1..t+block. Its numpy RNG state and sequential cursor can be included in M5 checkpoints. Workers default to zero to avoid Windows spawn differences.

No external corpus has been downloaded or redistributed. The required TinyShakespeare/TinyStories experiments cannot claim data license compliance or quality until their exact license, revision and checksum have been reviewed.
