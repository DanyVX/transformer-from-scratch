# Model card

Model architecture: original GPT-style decoder-only transformer (learned absolute positions, pre-LN, GELU, tied embeddings by default). Three named geometry profiles include tiny 1.27M, small ~17.4M and GPT-2 124M. Experimental variations support RoPE, post-LN, SwiGLU, untied embeddings and unscaled residual projections.

Weights distributed: **none**. The demo trains locally on synthetic ASCII text and saves weights under gitignored checkpoints/. There is no released foundation model, and no claim of deployed accuracy or acceptable fairness characteristics.

Reference loading: \`tfs.hf_compat.gpt2.load_gpt2_weights\` maps an externally supplied HF GPT-2 architecture into the original model; official released pretrained GPT-2 124M logits are **not yet verified**. Do not equate random-architecture alignment with pretrained verification.

Datasets: only synthetic text is used in default tests/demo. External model/dataset licenses, revisions, checksums and permitted uses must be confirmed before downloading. Bias, robustness, safety, harmful generation and multilingual quality are not measured.

Intended use: teaching, transparent implementation and reproducibility research. Not for identity verification, automated decision-making, safety-critical advice, or public deployment.
