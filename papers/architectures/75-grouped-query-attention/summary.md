---
title: "GQA: Grouped-Query Attention (and Multi-Query Attention)"
slug: "75-grouped-query-attention"
number: 75
category: "architectures"
authors: "Joshua Ainslie, James Lee-Thorp, Michiel de Jong, Yury Zemlyanskiy, Federico Lebron, Sumit Sanghai (Google Research) - GQA; Noam Shazeer (Google) - Multi-Query Attention"
published: "May 2023 (GQA, EMNLP 2023); November 2019 (MQA)"
year: 2023
url: "https://arxiv.org/abs/2305.13245"
tags: ["attention", "architecture", "efficiency", "inference-optimization"]
---

# GQA: Grouped-Query Attention (and Multi-Query Attention)

**Authors:** Joshua Ainslie, James Lee-Thorp, Michiel de Jong, Yury Zemlyanskiy, Federico Lebron, Sumit Sanghai (Google Research) - GQA; Noam Shazeer (Google) - Multi-Query Attention
**Published:** May 2023 (GQA, EMNLP 2023); November 2019 (MQA)
**Papers:** [GQA arxiv.org/abs/2305.13245](https://arxiv.org/abs/2305.13245) | [MQA arxiv.org/abs/1911.02150](https://arxiv.org/abs/1911.02150)

---

## Why This Matters

Grouped-query attention is **the reason you can run a 70B model with a long context on a reasonable GPU**. It is a small change to the attention layer that cuts KV-cache memory by roughly 8x with almost no quality loss, and it is now in essentially every production LLM: Llama 2 70B and every Llama since, Mistral and Mixtral, Qwen, Gemma, and most open models released after mid-2023.

- **8x smaller KV cache** at the typical setting (8 key-value heads instead of 64).
- **Longer contexts and bigger batches** on the same hardware - KV cache, not weights, is what limits both.
- **Near-zero quality cost**, unlike multi-query attention which measurably degraded quality.
- **Uptraining, not retraining** - the paper showed you can convert an existing multi-head checkpoint using about 5 percent of original pretraining compute.

**The insight:** autoregressive generation is memory-bandwidth-bound, not compute-bound. Each new token requires reading the entire KV cache from GPU memory. Shrinking that cache by sharing key and value heads across query heads makes generation faster and longer contexts affordable, and the quality loss is small if you do not shrink it all the way to one.

---

## The Problem: The KV Cache Eats Everything

In a decoder-only transformer, generating token N requires attending to all previous tokens. Recomputing their keys and values every step would be quadratic, so they are cached. That cache is the problem.

```
KV cache size = 2 (K and V)
              x layers
              x heads
              x head_dim
              x sequence_length
              x batch_size
              x bytes_per_value

Llama 2 70B with standard multi-head attention:
  80 layers x 64 heads x 128 dim x 2 x 2 bytes = ~2.6 MB per token
  32,768 tokens of context                     = ~86 GB

That is more memory than the model weights, for ONE request.
```

Two consequences follow. First, memory: long contexts and large batches simply do not fit. [PagedAttention](../../techniques/52-pagedattention-vllm/summary.md) attacked the fragmentation side of this; GQA attacks the raw size. Second, and less obvious, **bandwidth**: every generated token requires streaming the whole cache through the memory bus. Modern GPUs have far more FLOPs than memory bandwidth, so decoding sits idle waiting on memory. Shrinking the cache directly speeds up generation.

---

## The Core Innovation

Standard multi-head attention gives every query head its own key and value heads. Multi-query attention (Shazeer, 2019) shares **one** key-value head across all query heads. GQA interpolates: split query heads into **G groups**, and give each group its own key-value head.

```
Multi-Head Attention (MHA)     Multi-Query Attention (MQA)     Grouped-Query (GQA)
    8 query heads                   8 query heads                8 query heads
    8 K/V heads                     1 K/V head                   2 K/V heads

  Q1 Q2 Q3 Q4 Q5 Q6 Q7 Q8       Q1 Q2 Q3 Q4 Q5 Q6 Q7 Q8      Q1 Q2 Q3 Q4 | Q5 Q6 Q7 Q8
  |  |  |  |  |  |  |  |         \  \  \  |  /  /  /  /       \  \  \  | | |  /  /  /
  K1 K2 K3 K4 K5 K6 K7 K8              KV1                        KV1   |    KV2

  cache: 8 units                 cache: 1 unit                 cache: 2 units
  quality: best                  quality: degraded             quality: ~MHA
  bandwidth: worst               bandwidth: best               bandwidth: near-best
```

MQA was the extreme version and had been known since 2019. It gave a huge speedup and a real quality drop, plus training instability, which is why it was not universally adopted. GQA's contribution is the observation that the curve is steep at the start and flat afterward: going from 64 KV heads to 8 costs almost nothing, and going from 8 to 1 costs a lot. Eight is the sweet spot, and it is what nearly everyone uses.

---

## Key Components Explained

### 1. Grouping
**What it does:** Trades KV cache size against quality on a smooth dial.
**How it works:** With H query heads and G groups, each group of H/G query heads shares one key head and one value head. G = H is standard multi-head attention; G = 1 is multi-query attention. The KV cache shrinks by a factor of H/G. Typical production setting: H = 64 query heads, G = 8 KV heads.

### 2. Uptraining from an Existing Checkpoint
**What it does:** Converts a trained MHA model into a GQA model cheaply.
**How it works:** **Mean-pool** the key and value projection matrices within each group to produce the shared head, then continue pretraining for a small number of steps. The paper found roughly 5 percent of original pretraining compute was enough to recover quality. Mean pooling beat both selecting a single head and random initialization. This mattered enormously in practice: labs did not have to retrain from scratch to adopt GQA.

### 3. Why the Quality Loss Is Small
**What it does:** Explains why this is nearly free.
**How it works:** Query heads specialize far more than key/value heads do. Keys and values encode "what information does this token carry," which is largely shared; queries encode "what am I looking for," which varies a lot per head. Sharing the shared thing costs little. Reducing to a single KV head goes too far because some genuine diversity in the key space is lost.

### 4. Interaction With Other Optimizations
**What it does:** Composes with the rest of the inference stack.
**How it works:** GQA multiplies with [FlashAttention](../../techniques/16-flash-attention/summary.md) (which reduces attention's memory traffic within a step), PagedAttention (which reduces cache fragmentation), and quantized KV caches (which reduce bytes per entry). Modern serving stacks use all four together. Note that GQA also helps **tensor parallelism**: with 8 KV heads and 8 GPUs, each GPU holds exactly one KV head with no replication.

---

## Key Results

- On summarization, translation, and question answering benchmarks, **GQA with 8 groups matched multi-head attention quality** while running at close to multi-query attention speed.
- **MQA showed a consistent quality gap** on the same tasks, quantifying what had been folklore.
- **Uptraining with about 5 percent of pretraining compute** recovered near-full quality from a converted checkpoint.
- Inference speedups of several times on long-sequence decoding, driven almost entirely by reduced memory traffic.

---

## Why This Was Revolutionary

- **A one-line architecture change with a system-level payoff.** Changing the shape of two projection matrices bought 8x on the dominant memory cost of serving.
- **Reframed attention design around memory bandwidth.** Before GQA, attention variants were evaluated on FLOPs and quality. After, the KV cache became a first-class design target, and the entire long-context race depends on managing it.
- **Made the uptraining pattern respectable.** "Change the architecture, then cheaply continue training" is now a standard move (it recurs in context extension, MoE upcycling, and quantization-aware fine-tuning).
- **Enabled the 100k+ context era in practice.** Long-context models are possible without GQA; they are not affordable without it or something like it.

---

## Real-World Impact

- **[Llama 2](../../language-models/17-llama2/summary.md) 70B, [Llama 3](../../language-models/33-llama3.3/summary.md) at all sizes, and [Llama 4](../../language-models/41-llama4/summary.md)** use GQA. Mistral 7B, [Mixtral](../37-mixture-of-experts/summary.md), Qwen, Gemma, Falcon, and most post-2023 open models do too.
- **Every serving framework** - vLLM, TensorRT-LLM, llama.cpp, SGLang - has GQA-aware attention kernels, because the grouping changes the memory access pattern.
- **Consumer local inference.** Running a 70B model on a workstation is feasible partly because its KV cache is 8x smaller than it would otherwise be.
- **Successor designs push further.** DeepSeek's **Multi-head Latent Attention (MLA)**, used in [DeepSeek-V2 and V3](../../language-models/27-deepseek-v3/summary.md), compresses keys and values into a low-rank latent instead of sharing heads, achieving even smaller caches than GQA. GQA established the target that MLA optimizes harder.

---

## Key Takeaways for Practitioners

1. **8 KV heads is the well-tested default.** If you are designing a model, start there rather than exploring the whole range.
2. **Set KV heads to a multiple of your tensor-parallel degree.** Otherwise KV heads get replicated across GPUs and you lose part of the benefit.
3. **KV cache size is your context-length budget.** Compute it before promising a context window: `2 * layers * kv_heads * head_dim * seq_len * batch * bytes`.
4. **You can convert an existing model.** Mean-pool the K/V projections per group and uptrain briefly - do not assume a from-scratch retrain is required.
5. **Stack it with quantized KV caches.** FP8 or INT8 KV cache on top of GQA is another 2x to 4x and is now common in production serving.

---

## Limitations & Future Directions

- **Still linear in sequence length.** GQA reduces the constant, not the growth rate. 1M-token contexts need something structurally different: sliding-window attention, sparse attention, latent compression, or state-space models like [Mamba](../20-mamba/summary.md).
- **Some quality is lost**, small but nonzero, and it may show up disproportionately on tasks requiring fine-grained retrieval from long contexts.
- **Group count is fixed at training time**, so it is a design commitment rather than an inference-time knob.
- **MLA and sparse-attention variants may supersede it.** The 2024-2025 trend is toward learned compression of the cache rather than head sharing.

---

## Further Reading

- **GQA paper:** [arxiv.org/abs/2305.13245](https://arxiv.org/abs/2305.13245)
- **Multi-Query Attention:** [arxiv.org/abs/1911.02150](https://arxiv.org/abs/1911.02150)
- **DeepSeek-V2 (Multi-head Latent Attention):** [arxiv.org/abs/2405.04434](https://arxiv.org/abs/2405.04434)
- **In this collection:** [FlashAttention](../../techniques/16-flash-attention/summary.md), [PagedAttention/vLLM](../../techniques/52-pagedattention-vllm/summary.md)

## Citation

```bibtex
@inproceedings{ainslie2023gqa,
  title={GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints},
  author={Ainslie, Joshua and Lee-Thorp, James and de Jong, Michiel and Zemlyanskiy, Yury and Lebr{\'o}n, Federico and Sanghai, Sumit},
  booktitle={Proceedings of EMNLP},
  year={2023}
}
```

<!-- related:start -->

---

## Related in This Collection

- [FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness](../../techniques/16-flash-attention/summary.md)
- [LLaMA 2: Open Foundation and Fine-Tuned Chat Models](../../language-models/17-llama2/summary.md)
- [Mamba: Linear-Time Sequence Modeling with Selective State Spaces](../../architectures/20-mamba/summary.md)
- [Qwen3: Technical Report](../../language-models/28-qwen3/summary.md)
- [LLaMA 3.3: Matching 405B Performance with 70B Parameters](../../language-models/33-llama3.3/summary.md)
- [Mixtral of Experts (and the Mixture-of-Experts Architecture)](../../architectures/37-mixture-of-experts/summary.md)
- [Llama 4: Natively Multimodal Open-Source AI](../../language-models/41-llama4/summary.md)
- [PagedAttention: Efficient LLM Serving with vLLM](../../techniques/52-pagedattention-vllm/summary.md)

<!-- related:end -->
