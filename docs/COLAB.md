# Larger runs on Colab/Kaggle

The \`small\` profile is a 10–30M-scale single-GPU architecture; \`gpt2-124m\` is the reference geometry, not an approved training dataset or recipe.

1. Start a Python 3.11/3.12 runtime with an NVIDIA GPU and PyTorch installed.
2. Clone the repository, then \`python -m pip install -e ".[dev,model]"\`.
3. Confirm hardware with \`python -m tfs environment\`.
4. Inspect \`configs/small.yaml\`. Reduce microbatch/context if CUDA memory is insufficient.
5. Supply **licensed and checksummed** text data through \`tfs.data.shards\`; do not commit the corpus or generated model weights.
6. Start with the tested CPU model+sampler/trainer interfaces, enabling CUDA only after verifying device availability. Save checkpoints privately and copy JSON benchmark provenance into the report.
7. For GPT-2 reference verification, supply a legitimately obtained local weight directory to \`scripts/verify_gpt2.py --hf-local-path <directory>\`. The script never downloads weights.

Do not upload credentials, private ID documents, personal audio/video, faces or embeddings to notebooks or public storage. Multi-GPU and mixed-precision training are not validated.
