# Architecture (reference milestone)

The model is pre-normalized. For a hidden state `x`, each block computes `x + GQA(RMSNorm(x))`, then `x + SwiGLU(RMSNorm(x))`. Queries have `num_attention_heads` heads; keys and values have `num_kv_heads`, repeated per query group in this initial correctness-oriented implementation. RoPE rotates paired coordinates using `theta^(-2i/d)` frequencies. Attention scores are scaled by `sqrt(head_dim)` and masked so a position can only attend to itself and earlier positions.

The reference `naive` backend materializes attention scores. `sdpa` delegates the attention kernel to PyTorch SDPA. Both use the same projections and positional math. The KV cache stores rotated keys and values using KV-head count, then expands them for attention in this readable reference implementation.

The optional experimental latent attention compresses token state `x_t` into `c_t = W_down x_t`, then derives keys and values as `k_t = W_k c_t` and `v_t = W_v c_t`. The cache can store `c_t` instead of expanded key/value heads. This small implementation is a research sketch, not a faithful reproduction of a production MLA architecture and has no quality or performance results.

MTP auxiliary heads predict additional horizons `t+2` through `t+K+1` alongside the normal next-token head. MoE uses router softmax followed by top-k expert dispatch and a load-balance term; it currently loops over experts and is intended as a correctness reference, not a fast training backend.
