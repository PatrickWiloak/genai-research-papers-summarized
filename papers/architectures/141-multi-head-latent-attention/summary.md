---
title: "DeepSeek-V2: A Strong, Economical, and Efficient Mixture-of-Experts Language Model (Multi-head Latent Attention)"
slug: "141-multi-head-latent-attention"
number: 141
category: "architectures"
authors: "DeepSeek-AI"
published: "May 2024 (arXiv)"
year: 2024
url: "https://arxiv.org/abs/2405.04434"
tags: ["attention", "architecture", "efficiency", "inference-optimization"]
---

# DeepSeek-V2: A Strong, Economical, and Efficient Mixture-of-Experts Language Model (Multi-head Latent Attention)

**Authors:** DeepSeek-AI
**Published:** May 2024 (arXiv)
**Paper:** [arxiv.org/abs/2405.04434](https://arxiv.org/abs/2405.04434)

---

## Why This Paper Matters

When a language model generates text, it stores a **key and a value vector for every past token, in every layer** - the [KV cache](../../../explainers/concepts/kv-cache.md). For long contexts and many simultaneous users, this cache, not the model weights, is often what fills GPU memory and caps throughput. [Grouped-Query Attention](../75-grouped-query-attention/summary.md) shrank it by letting groups of query heads share keys and values, at some cost in quality.

DeepSeek-V2 introduced **Multi-head Latent Attention (MLA)**, which shrinks the cache much further **without** that quality loss. Instead of caching full keys and values, MLA caches one small compressed **latent vector** per token and reconstructs the keys and values from it when needed. In the paper's comparisons MLA not only used far less memory than standard multi-head attention - it *outperformed* it.

- **93.3% smaller KV cache** than DeepSeek's previous 67B dense model.
- **5.76x higher maximum generation throughput** on the same hardware.
- **42.5% lower training cost** per trillion tokens, thanks mainly to its mixture-of-experts design.
- **The attention design of the DeepSeek line**, carried into [DeepSeek-V3](../../language-models/27-deepseek-v3/summary.md) and R1, and adopted by other open models such as Moonshot's Kimi K2.

**The insight:** the keys and values of all attention heads are highly redundant. Project them jointly into a low-dimensional latent space, cache only that, and let the model learn to expand it back - the compression becomes part of what the model learns rather than a lossy afterthought.

---

## The Model: DeepSeek-V2

MLA was introduced inside a full model release:
- **236B total parameters, 21B active per token** - a mixture-of-experts model ("DeepSeekMoE", with many fine-grained experts plus shared experts; see [Mixture of Experts](../37-mixture-of-experts/summary.md)).
- **128K-token context window.**
- **Pretrained on 8.1 trillion tokens.**
- **Training cost:** 172.8K GPU hours per trillion tokens, against 300.6K for DeepSeek 67B.
- **Throughput:** more than 50,000 generated tokens per second on a single node of 8 H800 GPUs, in a deployment using FP8 weights and a 6-bit quantised KV cache.

The rest of this summary focuses on MLA, the part with the widest influence.

---

## The Core Innovation: Cache a Latent, Not the Keys and Values

### Standard multi-head attention
For each token and each layer, cache `n_heads x d_head` numbers for keys and the same for values.

```
MHA cache per token per layer = 2 x n_heads x d_head
DeepSeek-V2 dimensions:          2 x 128 x 128 = 32,768 numbers
```

### MLA: joint low-rank compression
Project the token's hidden state down to a small latent vector `c_KV` (dimension 512 in DeepSeek-V2). Cache **only** `c_KV`. At attention time, up-project it to per-head keys and values.

```
hidden state h  --down-project-->  c_KV (512)   <- this is what gets cached
                                     |
                      up-project to keys and values for all 128 heads
```

### The matrix-absorption trick
Up-projecting the whole cache at every step would be expensive. But matrix multiplication is associative: the key up-projection can be **absorbed into the query projection**, and the value up-projection **into the output projection**. The model can compute attention directly against the cached latents without ever materialising full keys and values.

### The catch: RoPE, and the decoupled fix
[Rotary position embeddings](../../techniques/54-rope-rotary-position-embedding/summary.md) apply a position-dependent rotation between the query and key. That rotation sits in the middle of the product and **blocks the absorption trick**. MLA's solution is a **decoupled RoPE key**: a small extra key component (64 dimensions per token) that carries positional information separately and is cached alongside the latent.

```
MLA cache per token per layer = d_c + d_rope = 512 + 64 = 576 numbers
                              vs 32,768 for MHA with the same heads
```

The paper notes this is equivalent in size to GQA with only about **2.25 groups** - yet with quality at or above full multi-head attention. Queries are also compressed (to 1,536 dimensions) to save activation memory during training.

---

## Key Results

### Attention variants compared (paper appendix)
On 7B dense models trained on 1.33T tokens, standard attention beat its cheaper variants:

| Attention | MMLU |
|---|---|
| MHA (full multi-head) | 45.2 |
| GQA (8 groups) | 41.2 |
| MQA (1 shared KV head) | 37.9 |

On MoE models, **MLA beat MHA** while caching a fraction of the data:

| Model scale | MHA cache per token | MLA cache per token |
|---|---|---|
| Small MoE | 110.6K elements | 15.6K elements |
| Large MoE | 860.2K elements | 34.6K elements |

### DeepSeek-V2 overall
At release it was among the strongest open models on standard benchmarks while activating only 21B parameters per token - and the efficiency gains let DeepSeek price its API far below competitors, which contributed to an industry-wide price cut in China in mid-2024.

---

## Why This Was Revolutionary

- **It broke the memory-quality trade-off.** MQA and GQA bought memory with quality; MLA delivered both in the paper's tests.
- **It made long context cheaper to serve**, since the cache is what grows with context length.
- **It showed architecture still matters.** At a time when attention seemed settled, a substantial improvement came from rethinking what gets cached.

---

## Real-World Impact

- **[DeepSeek-V3](../../language-models/27-deepseek-v3/summary.md) and [DeepSeek-R1](../../language-models/26-deepseek-r1/summary.md)** kept MLA, which is part of why they were cheap to serve.
- **Kimi K2** (Moonshot AI, 2025) adopted MLA in its architecture.
- **FlashMLA** (DeepSeek, February 2025) released optimised GPU kernels for MLA decoding.
- **TransMLA** (2025) proposed converting existing GQA models to MLA after training.
- **Serving frameworks** such as vLLM and SGLang added dedicated MLA support.

---

## Key Takeaways for Practitioners

1. **KV cache size drives serving cost** for long contexts and large batches. Check an architecture's cache per token, not just its parameter count.
2. **MLA needs framework support.** Without absorption-aware kernels, you lose much of the benefit.
3. **Position encoding interacts with efficiency tricks** - the decoupled RoPE key is a reminder to check.
4. **Compare like with like.** GQA, MQA and MLA trade memory against quality differently; see the [KV cache explainer](../../../explainers/concepts/kv-cache.md).

---

## Limitations & Future Directions

- **More complex** to implement than GQA, with an extra position-encoding path.
- **Evidence mostly from DeepSeek's own models**; independent large-scale comparisons are fewer.
- **Compute is not reduced** - MLA saves memory and bandwidth, not attention FLOPs. Long-context cost still grows with sequence length.
- **Alternatives keep coming**: sliding-window hybrids ([Longformer](../131-longformer/summary.md)), sparse attention, and state-space models ([Mamba](../20-mamba/summary.md)).

---

## Further Reading

- **Original Paper:** [arxiv.org/abs/2405.04434](https://arxiv.org/abs/2405.04434)
- **FlashMLA kernels:** [github.com/deepseek-ai/FlashMLA](https://github.com/deepseek-ai/FlashMLA)
- **TransMLA:** [arxiv.org/abs/2502.07864](https://arxiv.org/abs/2502.07864)
- **In this collection:** [Grouped-Query Attention](../75-grouped-query-attention/summary.md), [DeepSeek-V3](../../language-models/27-deepseek-v3/summary.md), [Mixture of Experts](../37-mixture-of-experts/summary.md), [PagedAttention](../../techniques/52-pagedattention-vllm/summary.md), [RoPE](../../techniques/54-rope-rotary-position-embedding/summary.md)
- **Explainers:** [KV cache](../../../explainers/concepts/kv-cache.md), [DeepSeek model family](../../../explainers/model-families/deepseek.md)

## Citation

```bibtex
@article{deepseekai2024deepseekv2,
  title={DeepSeek-V2: A Strong, Economical, and Efficient Mixture-of-Experts Language Model},
  author={{DeepSeek-AI}},
  journal={arXiv preprint arXiv:2405.04434},
  year={2024}
}
```

<!-- related:start -->

---

## Related in This Collection

- [Mamba: Linear-Time Sequence Modeling with Selective State Spaces](../../architectures/20-mamba/summary.md)
- [DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning](../../language-models/26-deepseek-r1/summary.md)
- [DeepSeek-V3 Technical Report](../../language-models/27-deepseek-v3/summary.md)
- [Mixtral of Experts (and the Mixture-of-Experts Architecture)](../../architectures/37-mixture-of-experts/summary.md)
- [PagedAttention: Efficient LLM Serving with vLLM](../../techniques/52-pagedattention-vllm/summary.md)
- [RoFormer: Enhanced Transformer with Rotary Position Embedding (RoPE)](../../techniques/54-rope-rotary-position-embedding/summary.md)
- [GQA: Grouped-Query Attention (and Multi-Query Attention)](../../architectures/75-grouped-query-attention/summary.md)
- [Longformer: The Long-Document Transformer (Longformer)](../../architectures/131-longformer/summary.md)

<!-- related:end -->
