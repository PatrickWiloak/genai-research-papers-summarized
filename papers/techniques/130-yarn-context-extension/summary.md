---
title: "YaRN: Efficient Context Window Extension of Large Language Models (YaRN)"
slug: "130-yarn-context-extension"
number: 130
category: "techniques"
authors: "Bowen Peng, Jeffrey Quesnelle, Honglu Fan, Enrico Shippole (Nous Research, EleutherAI, University of Geneva)"
published: "August 2023 (ICLR 2024)"
year: 2023
url: "https://arxiv.org/abs/2309.00071"
tags: ["long-context", "position-encoding", "fine-tuning"]
---

# YaRN: Efficient Context Window Extension of Large Language Models (YaRN)

**Authors:** Bowen Peng, Jeffrey Quesnelle, Honglu Fan, Enrico Shippole (Nous Research, EleutherAI, University of Geneva)
**Published:** August 2023 (ICLR 2024)
**Paper:** [arxiv.org/abs/2309.00071](https://arxiv.org/abs/2309.00071)

---

## Why This Paper Matters

A language model trained on 4,096-token sequences does not gracefully handle 4,097. Push it past the length it saw in training and quality does not slowly degrade - it falls off a cliff, with perplexity exploding within a few hundred tokens. Pretraining from scratch at long lengths is expensive because attention cost grows with the square of sequence length. So in 2023 the practical question was: **can you take a model that already exists and teach it a longer context cheaply?**

YaRN is the answer most open models ended up using. It extends models that use [RoPE](../54-rope-rotary-position-embedding/summary.md) - which by then meant nearly every open LLM - by changing how positions are fed to the rotary embedding, plus one small adjustment to attention. The authors report reaching state-of-the-art context extension after fine-tuning on **less than about 0.1% of the original pretraining data**, with **10x fewer tokens and 2.5x fewer training steps** than earlier methods.

- **Llama 2 went from 4K to 64K and 128K context** with a few hundred fine-tuning steps.
- **It needs no architecture change.** The trick lives entirely in the precomputed rotary embeddings, so inference code and kernels do not change.
- **It became standard plumbing.** DeepSeek-V3 used YaRN to extend its context in two stages to 128K, and Qwen models ship YaRN as their `rope_scaling` option for long inputs.

**The insight:** RoPE encodes position as rotations at many different frequencies. Earlier methods stretched all of them the same way. YaRN observed that the high-frequency dimensions (which track *nearby* tokens) and the low-frequency dimensions (which track *far-apart* tokens) need different treatment - and that stretching positions also makes attention blurrier, which a single temperature constant can undo.

---

## The Problem: Models Break Past Their Training Length

RoPE rotates each query and key vector by an angle proportional to the token's position. It does this in pairs of dimensions, each pair rotating at its own frequency - fast for some pairs, slow for others:

```
dimension pair:   0      1      2    ...    d/2-1
frequency:      fast  ->  ->  ->  ->  ->   slow
wavelength:     ~6 tokens            ...  far longer than the context
```

When a model trained on 4K tokens sees position 6,000, the slow-rotating pairs reach angles they never saw during training. Attention scores computed from those unfamiliar angles are garbage, and the model degrades immediately. This is the **extrapolation** problem.

---

## Step One: Position Interpolation

The first clean fix came from Meta a few weeks earlier. **Position Interpolation** (Chen, Wong, Chen and Tian, June 2023, [arXiv 2306.15595](https://arxiv.org/abs/2306.15595)) said: do not extrapolate, *interpolate*. If you want 8x the context, divide every position index by 8 before computing the rotation, so position 32,000 is fed in as position 4,000.

```
Original training:   positions 0 ... 4096
Want 32K context:    positions 0 ... 32768
Position Interpolation: feed position p as p / 8  ->  0 ... 4096 again
```

Every angle is now one the model has seen. Position Interpolation extended LLaMA models from 7B to 65B up to 32,768 tokens with **fine-tuning within 1,000 steps**, and the paper showed the error bound for interpolation is far smaller (about 600x) than for extrapolation.

**What it cost:** squeezing positions by 8x also squeezes the fast-rotating dimensions. Two neighbouring tokens that used to differ by a clear angle now differ by an eighth of it. The model loses resolution on *local* order - exactly the information that matters for grammar and nearby reference.

---

## The Core Innovation: Treat Frequencies Differently, Then Cool the Attention

YaRN packages three ideas, two of which circulated as open-source experiments on Reddit and GitHub in mid-2023 before this paper wrote them up properly.

### 1. "NTK-aware" interpolation
Instead of dividing positions, change the RoPE **base** so that low frequencies are stretched a lot and high frequencies barely at all. This keeps local resolution. Code Llama shipped with this approach and Qwen 7B used a dynamic variant. Its weakness: the right base for a given extension factor is hard to pick, and some dimensions end up slightly extrapolated.

### 2. "NTK-by-parts" interpolation
Be explicit about it. Measure each dimension pair's wavelength against the original context length:

```
wavelength much shorter than context  ->  leave it alone (no interpolation)
wavelength longer than context        ->  interpolate fully (like PI)
in between                            ->  blend with a linear ramp
```

High-frequency pairs never needed help - they cycle many times within the training length, so the model has seen every angle. Low-frequency pairs are the ones that break, so only they get squeezed. For the Llama family the authors found good ramp boundaries at alpha = 1 and beta = 32 rotations.

### 3. Attention temperature
Stretching the context also flattens the attention distribution: with more positions competing, softmax spreads its probability more thinly and the model's attention gets blurrier. YaRN divides the attention logits by a temperature, fitted empirically on LLaMA models:

```
sqrt(1/t) = 0.1 * ln(s) + 1        where s is the extension factor
```

The neat engineering point is that this does not require touching the attention code. Scaling both queries and keys by a constant is equivalent, and that can be folded into the precomputed rotary embeddings - so it costs nothing at training or inference time.

**YaRN = NTK-by-parts interpolation + attention temperature scaling.**

### Dynamic scaling (no fine-tuning at all)
The paper also describes **Dynamic-YaRN**: set the scale factor at inference time from the current sequence length, so short inputs run unmodified and long inputs get just enough interpolation. Without any fine-tuning this gives more than a 2x context extension.

---

## Key Results

The authors fine-tuned Llama 2 7B and 13B using PG19 book data chunked into long segments.

- **s = 16 (to 64K):** 400 steps at global batch size 64.
- **s = 32 (to 128K):** started from the s = 16 checkpoint and trained only **200 more steps**, still on 64K-length data.
- **Extrapolation past the fine-tuning data:** the s = 32 model, trained only on 64K sequences, still achieved low perplexity at 128K. This matters because long training documents are scarce.
- **Beat the alternatives at equal budget:** with the same 400-step fine-tune, YaRN reached lower long-document perplexity (on Proof-pile) than Position Interpolation and NTK-aware scaling, and converged faster.
- **Short-context quality held:** on standard Hugging Face Open LLM Leaderboard benchmarks the extended models lost very little against the original Llama 2.
- **Passkey retrieval:** the extended models could find a hidden passkey across their full extended context.

---

## Why This Was Revolutionary

- **It made long context a fine-tuning problem, not a pretraining problem.** That put 64K-128K windows within reach of anyone with a modest GPU budget.
- **It explained why the folk methods worked.** NTK-aware and NTK-by-parts scaling were community discoveries; YaRN gave them a frequency-by-frequency account.
- **Zero inference cost.** Everything folds into the rotary embedding tables.
- **The two-stage recipe stuck.** Pretrain short, then extend in one or two cheap stages is now how most open models reach long context.

---

## Real-World Impact

- **DeepSeek-V3** ([summary](../../language-models/27-deepseek-v3/summary.md)) applied YaRN after pretraining, extending from 4K to 32K and then to 128K in two stages.
- **Qwen models** ([Qwen3](../../language-models/28-qwen3/summary.md)) document YaRN as the way to enable inputs beyond their native window.
- **Community long-context fine-tunes** of Llama and Mistral models throughout 2023 and 2024 were typically YaRN or one of its NTK predecessors.
- **Inference servers** such as vLLM ([PagedAttention](../52-pagedattention-vllm/summary.md)) and Hugging Face transformers accept a `rope_scaling: yarn` configuration.

---

## Key Takeaways for Practitioners

1. **A model's advertised context may be an extension.** Many "128K" models were pretrained at 4K-32K and extended with YaRN-style scaling. That is fine, but measure usable length with something like [RULER](../132-ruler/summary.md) rather than trusting the number.
2. **Enable scaling only when you need it.** Static YaRN scaling can slightly hurt short inputs; several model cards advise switching it on only for long workloads, or using the dynamic variant.
3. **Extension does not fix memory.** A longer window still means a bigger [KV cache](../../../explainers/concepts/kv-cache.md). YaRN solves the position problem, not the cost problem.
4. **Long data is the bottleneck.** YaRN's ability to extrapolate beyond its fine-tuning length is useful precisely because long, high-quality documents are rare.

---

## Limitations & Future Directions

- **RoPE only.** Models using ALiBi or learned absolute positions need other methods.
- **Perplexity is not comprehension.** Low perplexity at 128K shows the model is not broken there; it does not show the model can *use* information from 100K tokens back. Later evaluations such as [RULER](../132-ruler/summary.md) found that effective context is often much shorter than claimed context.
- **Hand-fitted constants.** The temperature formula and ramp boundaries were tuned on the Llama family; other models may need their own values.
- **Quadratic attention remains.** Extension makes long context *possible*, not cheap. Sparse and sliding-window attention ([Longformer](../../architectures/131-longformer/summary.md)), better KV caches ([MLA](../../architectures/141-multi-head-latent-attention/summary.md)) and alternatives like [Mamba](../../architectures/20-mamba/summary.md) attack the cost side.

---

## Further Reading

- **Original Paper:** [arxiv.org/abs/2309.00071](https://arxiv.org/abs/2309.00071)
- **Code:** [github.com/jquesnelle/yarn](https://github.com/jquesnelle/yarn)
- **Position Interpolation:** Chen et al. 2023, [arxiv.org/abs/2306.15595](https://arxiv.org/abs/2306.15595)
- **In this collection:** [RoPE](../54-rope-rotary-position-embedding/summary.md), [RULER](../132-ruler/summary.md), [Longformer](../../architectures/131-longformer/summary.md), [DeepSeek-V3](../../language-models/27-deepseek-v3/summary.md)
- **Explainer:** [Context windows](../../../explainers/concepts/context-windows.md)

## Citation

```bibtex
@inproceedings{peng2024yarn,
  title={YaRN: Efficient Context Window Extension of Large Language Models},
  author={Peng, Bowen and Quesnelle, Jeffrey and Fan, Honglu and Shippole, Enrico},
  booktitle={International Conference on Learning Representations},
  year={2024}
}
```

<!-- related:start -->

---

## Related in This Collection

- [LLaMA 2: Open Foundation and Fine-Tuned Chat Models](../../language-models/17-llama2/summary.md)
- [Mamba: Linear-Time Sequence Modeling with Selective State Spaces](../../architectures/20-mamba/summary.md)
- [DeepSeek-V3 Technical Report](../../language-models/27-deepseek-v3/summary.md)
- [Qwen3: Technical Report](../../language-models/28-qwen3/summary.md)
- [PagedAttention: Efficient LLM Serving with vLLM](../../techniques/52-pagedattention-vllm/summary.md)
- [RoFormer: Enhanced Transformer with Rotary Position Embedding (RoPE)](../../techniques/54-rope-rotary-position-embedding/summary.md)
- [GQA: Grouped-Query Attention (and Multi-Query Attention)](../../architectures/75-grouped-query-attention/summary.md)
- [Longformer: The Long-Document Transformer (Longformer)](../../architectures/131-longformer/summary.md)

<!-- related:end -->
