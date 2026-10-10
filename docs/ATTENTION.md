# M2 attention decisions

- Explicit QKV projection and scaled softmax in fp32 for numerical stability. The fused PyTorch SDPA backend is provided for comparison, not presented as a custom FlashAttention kernel.
- Offset-aware Boolean causal mask supports full and KV-cached incremental queries. No padding masks: M4 training uses packed fixed-size blocks.
- RoPE rotates adjacent feature pairs, independently in each head, with absolute position offsets.
- Tests cover full vs fused equivalence, future-token perturbations and one-token cached decoding.
- GPU/fp16 kernels are not measured by these CPU tests. Results remain TBD until benchmark scripts run.
