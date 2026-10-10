# Data card

Datasets currently downloaded or distributed: **none**. M1 tokenizer tests and benchmark use only synthetic text literals embedded in the source, including English, Urdu and short code examples.

M4 will specify corpus version, source, checksum, license, split-by-document strategy, permitted use, and deletion route. Only permissively licensed, synthetic, or consented data can be used. Never commit real private text, identity documents, biometric data, audio, corpus shards, or checkpoints.

## Demo data

The default demo uses hand-authored, synthetic English sentences in \`scripts/demo.py\`; the study uses synthetic English/code literals in \`eval/study.py\`. Synthetic tokens are stored in ignored data/processed/, partitioned by document before tokenization. No user data, raw biometrics, identity documents, or external corpora are used.

External TinyShakespeare/TinyStories evaluation is outstanding pending verification of each download's license, revision, source and SHA-256 checksum. Small synthetic validation windows have unavoidable distribution limitations, even where train and validation documents are split.
