# Mixtral of Experts

**Authors:** Albert Q. Jiang, Alexandre Sablayrolles, Antoine Roux, et al. (Mistral AI)

**Published:** January 2024 (arXiv 2401.04088)

**Paper Link:** https://arxiv.org/abs/2401.04088

---

## Why This Paper Matters

Mixtral 8x7B was the moment **Mixture of Experts (MoE)** moved from being a Google-only research curiosity to a mainstream, open, deployable architecture. With 47 billion total parameters but only ~13 billion active per token, Mixtral matched or beat LLaMA-2 70B and GPT-3.5 on most benchmarks while costing roughly the same to run as a 13B dense model. It was the first widely-available open-weight MoE that anyone could download, fine-tune, and serve.

The release also confirmed something the field had suspected since rumors of GPT-4's architecture: the path to better-than-dense quality at deployable cost was sparse mixture of experts. Mixtral made that path open-source.

---

## The Problem Before Mixtral

By late 2023, the field faced a tension:

- **Dense models** (LLaMA 2, Mistral 7B) were clean, easy to serve, and well-studied — but scaling them to GPT-4-level quality requires 100B+ parameters and proportional inference cost
- **MoE models** could in theory get more capability per FLOP — but Google's Switch Transformer, GShard, and GLaM were closed, hard to train, and notoriously difficult to fine-tune
- **GPT-4** was rumored to be a large MoE (~1.8T total parameters, ~280B active), but nobody outside OpenAI could verify or replicate this

The open question: could a well-engineered MoE be open-sourced and made practical for the broader community? Could MoE deliver on its theoretical efficiency promise outside Google?

---

## The Core Innovation: Sparse MoE on a Mistral Backbone

Mixtral 8x7B's contribution isn't a new idea — MoE was introduced for LLMs by Shazeer et al. years earlier. The contribution is making it work well, in the open, on a battle-tested architecture:

- Take the Mistral 7B Transformer block
- Replace each feed-forward network with **8 expert FFNs and a router**
- For each token, route to the **top-2 experts** and combine their outputs
- Train end-to-end on a large multilingual corpus

The result: 47B total parameters but only ~13B active per token, with quality that matches LLaMA-2 70B (a model with ~5x more active parameters per token).

---

## How Mixtral Works

### The Mixture-of-Experts Layer

In a standard Transformer, each block has Attention followed by a single FFN. In Mixtral, each block has Attention followed by an **MoE layer** with N experts (N=8) and a router.

```python
class MoELayer:
    def __init__(self, num_experts=8, top_k=2):
        self.experts = [FFN() for _ in range(num_experts)]
        self.router = Linear(hidden_dim, num_experts)
        self.top_k = top_k

    def forward(self, x):  # x shape: (tokens, hidden_dim)
        # 1. Compute routing logits for each token
        logits = self.router(x)               # (tokens, num_experts)

        # 2. Pick top-k experts per token
        weights, indices = topk(softmax(logits), k=self.top_k)

        # 3. Route each token to its chosen experts
        out = zeros_like(x)
        for i, expert in enumerate(self.experts):
            mask = (indices == i)            # tokens assigned to this expert
            out[mask] += weights[mask] * expert(x[mask])

        return out
```

Each token uses 2 of 8 experts, so only ~25% of the FFN parameters are active per token. Since FFN dominates parameter count in Transformers, total active parameters per token are much less than the model's total parameters.

### Architecture Summary

| Parameter | Value |
|-----------|-------|
| Total parameters | 46.7B |
| Active parameters per token | ~12.9B |
| Layers | 32 |
| Hidden dim | 4096 |
| FFN dim (per expert) | 14336 |
| Experts per layer | 8 |
| Active experts per token | 2 |
| Attention | GQA (32 Q heads, 8 KV heads) |
| Context | 32,768 tokens |
| Vocab | 32,000 |

The attention layers are dense and shared across tokens. Only the FFN is sparse. This means the KV cache is the same size as a 13B dense model — important for memory at long contexts.

### The Router

The router is a simple linear layer that outputs N logits per token. The top-2 experts are selected and softmax-normalized to give the combination weights. There's no auxiliary load-balancing loss in the released model; the authors found that the model learns reasonable routing on its own when trained at scale.

### Why "8x7B" Is Misleading

"8x7B" suggests 8 separate 7B models. In reality:
- The 7B parts are just the FFN expert parameters
- Attention layers, embeddings, and norms are **shared** across all experts
- Total parameters are 46.7B, not 56B
- Active per token is ~13B, not 7B (because 2 experts are active plus all dense components)

A more accurate naming would be "47B-MoE with 13B active." The community kept the catchier "8x7B."

---

## Key Results

### Benchmarks vs Dense Models

| Benchmark | Mixtral 8x7B | LLaMA-2 70B | GPT-3.5 |
|-----------|--------------|-------------|---------|
| MMLU | 70.6 | 69.9 | 70.0 |
| HellaSwag | 84.4 | 85.4 | 85.5 |
| WinoGrande | 77.2 | 80.5 | 81.6 |
| PIQA | 83.6 | 82.8 | - |
| ARC-c | 59.7 | 57.4 | 56.5 |
| TriviaQA | 77.6 | 79.6 | - |
| NaturalQuestions | 30.6 | 29.3 | - |
| GSM8K (maj@8) | 74.4 | 63.2 | 57.1 |
| MATH (maj@4) | 28.4 | 12.0 | 34.1 |
| HumanEval | 40.2 | 30.5 | 48.1 |
| MBPP | 60.7 | 49.8 | - |

Mixtral matched or beat LLaMA-2 70B on 9 of 11 benchmarks while using ~5x fewer active parameters per token. It roughly matched or beat GPT-3.5 on most non-coding benchmarks.

### Multilingual

The model was trained on substantial multilingual data and performed strongly on French, German, Italian, Spanish, and other European languages — much better than English-dominant LLaMA-2.

### Inference Cost

Active-parameter count drives FLOPs per token. So Mixtral's inference compute is closer to a 13B dense model than a 70B one, despite the larger parameter footprint. Memory is harder — you still need to load all 47B parameters, which means ~94GB at fp16 or ~24GB at 4-bit quantization.

In practice, Mixtral runs comfortably on a single A100 80GB or two consumer GPUs with quantization, with throughput much higher than LLaMA-2 70B.

### Mixtral 8x7B Instruct

The SFT + DPO instruct variant was state-of-the-art among open chat models at release, scoring 8.30 on MT-Bench (vs LLaMA-2-Chat 70B's 6.86).

---

## Expert Specialization Analysis

The paper investigated whether experts specialize by topic (math expert, code expert, etc.) — a popular hypothesis about MoE behavior. The finding was nuanced:

- **Experts do not clearly specialize by topic** (math vs code vs literature)
- **Experts do show some specialization by syntactic role** — certain experts fire more for whitespace, punctuation, or specific token types
- Routing is mostly determined by surface-level token features rather than high-level domains

This is somewhat surprising but important: the "experts as topic specialists" intuition is wrong. MoE works by allowing the model to allocate parameters non-uniformly across tokens, not by routing semantically-related content to a dedicated expert.

---

## Why This Mattered

### MoE went mainstream

Before Mixtral, MoE was associated with Google papers and the hard-to-replicate world of TPU pods. After Mixtral, every serious open model lab considered MoE for their next release. DeepSeek, Qwen, DBRX, Grok, and others followed with their own MoE designs within a year.

### The "active parameter" framing

Mixtral popularized the idea of reporting both **total** and **active** parameters. This became the standard way to think about MoE compute cost and to compare models fairly across dense and sparse architectures.

### A practical recipe

The fact that Mixtral was built by adding an MoE layer to a known-good dense backbone (Mistral 7B) showed that you don't need exotic infrastructure to train good MoE models. This recipe — take a dense model you trust, add expert FFNs — became widely adopted.

### Open MoE as a baseline

Mixtral established a high baseline for open MoE quality. Subsequent open MoE releases (DBRX 132B, Qwen2-MoE, DeepSeek-MoE) were measured against it.

---

## Limitations

- **Memory-heavy:** All 47B parameters must be loaded even though only 13B are active per token
- **Routing instability:** Training MoE is harder than training dense; small routing imbalances can hurt quality
- **Fine-tuning friction:** Full fine-tuning of MoE is more delicate than dense; many community fine-tunes had to rediscover good practices
- **No clear expert interpretability:** Specialization is not topic-based, making it hard to reason about or debug routing
- **Limited expert count:** 8 experts with top-2 is at the small end; later models (Mixtral 8x22B, DeepSeek-MoE) explored finer-grained sparsity
- **Closed training data:** Mistral AI did not disclose the corpus

---

## Connections to Other Papers

- **Mixture of Experts (#37):** Mixtral is the direct, practical instantiation of this architecture for modern LLMs
- **Mistral 7B (#72):** Mixtral's per-expert block is the Mistral 7B FFN; the attention layers are also inherited
- **Attention Is All You Need (#1):** Mixtral remains a Transformer; only the FFN sublayer is replaced
- **PaLM (#71):** A pure dense scaling effort — Mixtral represents the opposite design philosophy that began to dominate after PaLM
- **LLaMA 2 (#17):** Direct competitor; Mixtral matched its 70B variant at much lower inference cost
- **GPT-3 (#4) and GPT-4 (#36):** GPT-4 was widely rumored to be a large MoE; Mixtral made the MoE design space public
- **FlashAttention (#16):** Mixtral uses Flash Attention to keep the dense attention layers efficient
- **DeepSeek-V3 (#27):** Subsequent open MoE that pushed the design further (256 experts, finer sparsity) — built on the trail Mixtral blazed

---

## Key Takeaways

1. **MoE can match dense quality at a fraction of the active compute** — Mixtral 8x7B beats LLaMA-2 70B on most benchmarks with ~5x fewer active parameters per token
2. **Open-source MoE is practical** — Mixtral proved you can train, release, and deploy a state-of-the-art sparse model outside the closed Google ecosystem
3. **Reporting "active parameters" matters** — it's the right way to think about MoE compute and to compare models across architecture families
4. **Experts do not specialize by topic** — the intuitive picture is wrong; specialization is mostly syntactic
5. **Sparse architectures are likely the path forward** — Mixtral's success accelerated the field's shift away from pure dense scaling toward MoE and other sparsity techniques
