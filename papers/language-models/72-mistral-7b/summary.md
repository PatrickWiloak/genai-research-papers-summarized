# Mistral 7B

**Authors:** Albert Q. Jiang, Alexandre Sablayrolles, Arthur Mensch, et al. (Mistral AI)

**Published:** October 2023 (arXiv 2310.06825)

**Paper Link:** https://arxiv.org/abs/2310.06825

---

## Why This Paper Matters

Mistral 7B was the model that made "open-weight" mean something competitive. Released under Apache 2.0 by a then-three-month-old French startup, it beat Meta's LLaMA-2 13B on every benchmark tested while having half the parameters. It even outperformed LLaMA 1's 34B model on reasoning, math, and code. For the first time, a single open 7B model was simultaneously the best small model available and a genuinely useful general-purpose assistant.

The paper itself is short and modest — a six-page technical report. But its impact reshaped the open-source LLM landscape, established Mistral AI as a major lab, and made **Grouped-Query Attention** plus **Sliding Window Attention** standard parts of the modern LLM toolkit.

---

## The Problem Before Mistral 7B

By late 2023, open-weight LLMs had real momentum thanks to LLaMA and LLaMA 2. But the trade-offs were stark:

- **7B models** were small enough to run on a consumer GPU but noticeably weaker than 13B+
- **13B models** were the sweet spot in quality but harder to deploy
- **Inference costs** at the 13B+ scale were still painful for high-throughput applications
- **Long contexts** were impractical — full attention scales O(n^2) in memory

Mistral AI's bet: with careful architecture choices, a 7B model could match the quality of 13B models while keeping inference cheap, and could handle long contexts efficiently.

---

## The Core Innovation: Better Architecture, Not More Parameters

Mistral 7B didn't introduce one big idea — it combined several existing techniques into a tight, well-engineered package:

1. **Grouped-Query Attention (GQA)** for faster inference
2. **Sliding Window Attention (SWA)** for long-context efficiency
3. **Rolling buffer KV cache** for constant-memory inference at long contexts
4. **Pre-fill and chunking** for efficient prompt processing
5. A standard, well-tuned dense Transformer backbone with SwiGLU and RoPE

The combination made Mistral 7B punch dramatically above its parameter weight.

---

## How Mistral 7B Works

### Architecture Summary

| Parameter | Value |
|-----------|-------|
| Parameters | 7.3B |
| Layers | 32 |
| Hidden dim | 4096 |
| Heads (Q) | 32 |
| Heads (KV) | 8 (GQA) |
| Head dim | 128 |
| FFN dim | 14336 (SwiGLU) |
| Vocab | 32000 (SentencePiece BPE) |
| Context | 8192 (sliding window 4096) |

### Grouped-Query Attention (GQA)

Standard multi-head attention has separate Q, K, V projections per head. **Multi-Query Attention (MQA)**, from PaLM, shares a single K and V across all heads. GQA, from a Google paper, sits in between: groups of query heads share a K/V pair.

```
Multi-head:  32 Q heads, 32 K heads, 32 V heads  (heavy KV cache)
Multi-query: 32 Q heads,  1 K head,  1 V head    (lossy quality)
Grouped (8): 32 Q heads,  8 K heads,  8 V heads  (Mistral's choice)
```

The KV cache — which dominates memory during autoregressive generation — shrinks 4x compared to full multi-head, with negligible quality loss. This is why Mistral 7B is so fast to run.

### Sliding Window Attention (SWA)

Standard attention lets every token attend to every previous token: O(n^2) memory and compute. SWA restricts each token to attend only to the previous W tokens (W = 4096 in Mistral). This caps memory at O(n * W).

But information can still flow farther than W tokens because each layer's attention output feeds the next layer:

```
Layer 1: token at position 8192 sees positions 4096-8191
Layer 2: those positions saw 0-4095 in layer 1
         -> effective receptive field at layer L is L * W tokens
```

With 32 layers and W = 4096, the theoretical receptive field is 131,072 tokens — though information mixing dilutes with depth.

### Rolling Buffer KV Cache

Naively, a KV cache grows linearly with sequence length. With SWA, you only need the last W tokens of K/V, so Mistral uses a **rolling buffer** that overwrites old entries:

```python
# Pseudocode for rolling buffer cache
cache_pos = i % W
K_cache[cache_pos] = K_new
V_cache[cache_pos] = V_new
```

This gives constant memory usage regardless of sequence length — critical for long-context inference.

### Pre-fill and Chunking

For long prompts, the model processes the prompt in chunks of W tokens, building up the KV cache incrementally. This avoids materializing a huge attention matrix at any point.

---

## Key Results

### Standard Benchmarks (vs LLaMA-2)

| Benchmark | Mistral 7B | LLaMA-2 7B | LLaMA-2 13B | LLaMA-1 34B |
|-----------|------------|------------|-------------|-------------|
| MMLU | 60.1 | 44.4 | 55.6 | 62.6 |
| HellaSwag | 81.3 | 77.1 | 80.7 | 83.4 |
| WinoGrande | 75.3 | 69.5 | 72.9 | 76.9 |
| PIQA | 83.0 | 78.1 | 80.5 | 82.2 |
| Arc-C | 55.5 | 45.9 | 49.4 | 54.5 |
| NaturalQuestions | 28.8 | 22.6 | 26.7 | 29.4 |
| TriviaQA | 69.9 | 64.0 | 67.6 | 73.7 |
| HumanEval | 30.5 | 12.2 | 18.3 | 22.6 |
| MBPP | 47.5 | 20.8 | 30.6 | 33.6 |
| GSM8K | 52.1 | 14.6 | 28.7 | 35.7 |
| Math | 13.1 | 2.5 | 3.9 | 6.2 |

**The headlines:**
- Beats LLaMA-2 13B on every benchmark
- Beats LLaMA-1 34B on reasoning, math, and code
- Roughly matches CodeLlama-7B on code despite not being code-specialized

### Mistral 7B Instruct

A simple supervised fine-tuning pass on publicly available instruction datasets (no RLHF) produced **Mistral 7B Instruct**, which outperformed all 7B and 13B open chat models on MT-Bench. The fact that lightweight SFT on the base model produced such a strong chat model became a standard recipe.

### Inference Speed

GQA + SWA make Mistral 7B notably faster per token than comparably-sized LLaMA models, with smaller memory footprints. In practice, it became the default choice for many deployment scenarios.

---

## Why This Mattered

### A new ceiling for "small" models

Before Mistral 7B, the assumption was that a 7B parameter model would always trail a well-trained 13B by a meaningful margin. Mistral broke that assumption with three forces:

1. **Better architecture** (GQA, SWA) reduces overhead per parameter
2. **More training data** (exact figures undisclosed, but clearly more than LLaMA-2)
3. **Better data quality and curation**

The implication for the field: parameter count is a weak proxy for capability. A small, well-built, well-trained model can dominate larger, less polished ones.

### The European open-AI moment

Mistral AI was founded in May 2023 by former Meta and DeepMind researchers. Releasing Mistral 7B under Apache 2.0 (a far more permissive license than LLaMA-2's custom license) in October 2023 — five months after founding — instantly made the company a major player. It also gave Europe its first credible homegrown frontier LLM lab.

### Architectural choices that stuck

Many subsequent models — including Mistral's own Mixtral and Mistral Large, plus many fine-tunes — adopted the GQA + SWA + SwiGLU + RoPE template. It became the de facto modern small-LLM architecture.

---

## Impact and Legacy

### Direct successors from Mistral
- **Mixtral 8x7B (December 2023):** Sparse MoE built on the Mistral 7B block
- **Mistral 7B v0.2 and v0.3:** Refinements
- **Mistral Large, Mistral Medium, Mistral Small:** Closed/API frontier models
- **Codestral:** Code-specialized variant
- **Mathstral, Mistral NeMo, Mistral 3:** Domain or scale variants

### Ecosystem impact
- Became the default base model for thousands of fine-tunes on Hugging Face
- Powered countless production deployments where LLaMA-2's license was a blocker
- Standardized GQA as a baseline expectation for new LLMs
- Inspired similar small-model efforts (Phi, Gemma, Qwen) to push harder on quality-per-parameter

### Influence on bigger models
- LLaMA 3 (2024) adopted GQA
- Most modern frontier models use some form of GQA or MQA
- Sliding window or local attention variants appear in many long-context designs

---

## Limitations

- **No safety alignment in base model:** Released without RLHF; downstream users must add their own safety
- **English-dominant:** Strong on English, weaker on other languages
- **Effective context limit:** The 131K theoretical receptive field of SWA is much smaller in practice — long-range recall degrades faster than full-attention models
- **Closed training data:** Mistral never disclosed the training mix
- **Closed training code:** Only the weights and inference code are open

---

## Connections to Other Papers

- **Attention Is All You Need (#1):** Mistral 7B is a tightly engineered instance of this architecture
- **LLaMA (#15) and LLaMA 2 (#17):** Direct comparison points. Mistral 7B's architecture is essentially LLaMA with GQA and SWA added; the results showed how much those changes plus better data could matter
- **PaLM (#71):** Pioneered Multi-Query Attention, which Mistral generalized to GQA
- **FlashAttention (#16):** Critical for actually realizing the speed benefits of GQA and SWA in practice
- **RoPE (#54):** Mistral inherits PaLM/LLaMA's choice of rotary positional embeddings
- **Mixture of Experts (#37) and Mixtral (#73):** Mistral's follow-up applied MoE to the same backbone, scaling capability without proportionally scaling inference cost
- **Chinchilla (#18):** Mistral 7B is in the spirit of Chinchilla-optimal training — small model, lots of data
- **GPT-3 (#4):** Mistral 7B is not too far from GPT-3 175B on many benchmarks at 25x fewer parameters

---

## Key Takeaways

1. **A well-engineered 7B model can beat a generic 13B model** — parameter count is not the best proxy for quality
2. **Grouped-Query Attention** trades a sliver of quality for dramatically cheaper inference and has become standard
3. **Sliding Window Attention plus rolling KV cache** enables long-context inference at constant memory
4. **Apache 2.0 licensing** of a frontier-quality open model dramatically expanded what builders could ship without legal friction
5. **Architecture and data quality compound:** small improvements in each stage add up to a model that outperforms much larger predecessors
