---
title: "Longformer: The Long-Document Transformer (Longformer)"
slug: "131-longformer"
number: 131
category: "architectures"
authors: "Iz Beltagy, Matthew E. Peters, Arman Cohan (Allen Institute for Artificial Intelligence)"
published: "April 2020 (arXiv; v2 December 2020 added the encoder-decoder LED)"
year: 2020
url: "https://arxiv.org/abs/2004.05150"
tags: ["attention", "long-context", "efficiency"]
---

# Longformer: The Long-Document Transformer (Longformer)

**Authors:** Iz Beltagy, Matthew E. Peters, Arman Cohan (Allen Institute for Artificial Intelligence)
**Published:** April 2020 (arXiv; v2 December 2020 added the encoder-decoder LED)
**Paper:** [arxiv.org/abs/2004.05150](https://arxiv.org/abs/2004.05150)

---

## Why This Paper Matters

Full self-attention compares every token with every other token. Double the input and the work quadruples. In 2020 that meant BERT-style models stopped at 512 tokens, and anything longer - a scientific paper, a legal contract, a Wikipedia article - had to be chopped into pieces that could not see each other.

Longformer showed a simple way out that has lasted: **most tokens only need to look at their neighbours, and a few special tokens need to see everything.** Replace the full attention matrix with a sliding window plus a handful of global positions and the cost grows linearly with length instead of quadratically.

- **Linear-cost attention as a drop-in replacement.** No new architecture to learn; the Transformer stays a Transformer.
- **It reused existing models.** Longformer started from RoBERTa's weights and continued pretraining, rather than training from scratch.
- **State of the art on long-document tasks** of its day, including WikiHop and TriviaQA.
- **The sliding window survives in modern LLMs.** [Mistral 7B](../../language-models/95-mistral-7b/summary.md) and later hybrid designs interleave windowed and global attention layers - the same bet in a decoder-only setting.

**The insight:** in text, relevance is mostly local. A window of a few hundred tokens covers the dependencies inside a paragraph, and stacking layers widens what each token can indirectly see. The rare long-range links - the question in a QA task, the classification token - can be handled by giving just those positions full attention.

---

## The Problem: Quadratic Attention

```
tokens (n)     attention scores per layer (n x n)
   512                    262,144
 4,096                 16,777,216     (64x more for 8x the length)
16,384                268,435,456
```

Memory, not just time, was the binding limit. At a few thousand tokens the attention matrices alone would not fit on a 2020 GPU. Workarounds existed - split the document, process chunks, combine - but information could not flow between chunks, and every task needed its own stitching logic.

---

## The Core Innovation: Three Attention Patterns

### 1. Sliding window attention
Each token attends to a fixed window of `w` tokens around it (w/2 on each side). Cost is `O(n x w)` - linear in length for a fixed window.

```
token:    1 2 3 4 5 6 7 8 9 ...
token 5 sees:   [3 4 5 6 7]       (w = 4 plus itself)
```

Stacking layers grows the receptive field like a convolutional network: with `l` layers, the top layer can draw on roughly `l x w` tokens of context. Longformer used a window of **512**, chosen so compute per token matched RoBERTa's.

### 2. Dilated sliding window
For language modelling, some heads skip positions within their window (look at every 2nd or 4th token), reaching further for the same cost. Lower layers use small windows for local detail; higher layers use larger ones.

### 3. Global attention
A small set of task-chosen positions attend to **every** token, and every token attends to them. For classification that is the `[CLS]` token; for question answering it is all the question tokens. These positions get their own separate query, key and value projections.

```
            window-only tokens:  see neighbours
                     ^   ^
     [CLS] <-----> every token <-----> question tokens
            global tokens: see everything, seen by everything
```

Global tokens are few, so the added cost is still linear in document length.

### Building on RoBERTa
Rather than pretraining from scratch, Longformer took RoBERTa and continued masked-language-model pretraining for **65K gradient updates at sequence length 4,096**. RoBERTa only had 512 learned position embeddings, so Longformer initialised the extra positions by **copying RoBERTa's 512 embeddings repeatedly** - a trick that let the model start from sensible positional knowledge.

### LED: the encoder-decoder version
The December 2020 revision added **Longformer-Encoder-Decoder (LED)**, built from BART, for long-input generation tasks like summarising scientific papers.

---

## Key Results

- **Character-level language modelling:** state of the art at the time on text8 (**1.10** bits per character) and enwik8 (**1.00**).
- **Base model against RoBERTa-base** on long-document tasks: WikiHop improved from 72.4 to **75.0**; Hyperpartisan news classification from 87.4 to **94.8**.
- **Longformer-large** set new best results on **WikiHop (81.9)** and **TriviaQA (77.3)**.
- **LED on arXiv summarisation** with 16K-token inputs: ROUGE-1/2/L of **46.63 / 19.62 / 41.83**, state of the art at the time.

The gains were largest on tasks whose evidence was spread across long documents - exactly where chunking hurt.

---

## Why This Was Revolutionary

- **It turned "long context" from a research problem into a configuration choice** for encoder models.
- **It proved continued pretraining works for length.** Copying position embeddings and training on longer sequences became a standard recipe, echoed later by [Position Interpolation and YaRN](../../techniques/130-yarn-context-extension/summary.md) for RoPE models.
- **Local plus global** became the design vocabulary for efficient attention.

---

## Real-World Impact

- **BigBird** (Google, NeurIPS 2020) arrived at the same time with window plus global plus *random* attention, and a proof that this sparse pattern keeps the full Transformer's expressive power.
- **Mistral 7B** ([summary](../../language-models/95-mistral-7b/summary.md)) used sliding window attention with a 4,096-token window in a decoder-only LLM.
- **Hybrid local/global layer stacks** - where most layers use a short window and some use full attention - appear in several open model families; Gemma 3, for example, raised its proportion of local layers to cut KV cache memory.
- **StreamingLLM** (ICLR 2024) found that decoder models dump attention on the first few tokens ("attention sinks"); keeping those plus a sliding window lets a model stream millions of tokens without its quality collapsing, though it still only *remembers* the window.

---

## Key Takeaways for Practitioners

1. **Windowed attention trades recall for cost.** A token can only reach distant information indirectly, through stacked layers or global tokens. Tasks that need exact lookup far back suffer.
2. **Choose global tokens deliberately.** In encoder use, what gets global attention is a task decision - the question, the label token, section headers.
3. **Most "long context" LLMs today use full attention plus better position handling, not Longformer.** FlashAttention made exact attention cheap enough at tens of thousands of tokens that sparse patterns became a memory optimisation rather than a necessity.
4. **Measure usable length.** Whatever the mechanism, test with a benchmark like [RULER](../../techniques/132-ruler/summary.md).

---

## Limitations & Future Directions

- **Custom kernels.** Efficient banded attention needed a hand-written CUDA kernel; naive implementations lose the speed advantage.
- **Encoder-first design.** The original work targeted BERT-style tasks; decoder LLMs took the sliding window but not the task-specific global tokens.
- **Superseded for exact attention.** [FlashAttention](../../techniques/16-flash-attention/summary.md) made full attention far cheaper in practice, and state-space models like [Mamba](../20-mamba/summary.md) offer linear cost with a different trade-off.

---

## Further Reading

- **Original Paper:** [arxiv.org/abs/2004.05150](https://arxiv.org/abs/2004.05150)
- **BigBird:** Zaheer et al. 2020, [arxiv.org/abs/2007.14062](https://arxiv.org/abs/2007.14062)
- **StreamingLLM:** Xiao et al. 2023, [arxiv.org/abs/2309.17453](https://arxiv.org/abs/2309.17453)
- **In this collection:** [Attention Is All You Need](../01-attention-is-all-you-need/summary.md), [BERT](../../language-models/03-bert/summary.md), [Mistral 7B](../../language-models/95-mistral-7b/summary.md), [YaRN](../../techniques/130-yarn-context-extension/summary.md)
- **Explainer:** [Context windows](../../../explainers/concepts/context-windows.md)

## Citation

```bibtex
@article{beltagy2020longformer,
  title={Longformer: The Long-Document Transformer},
  author={Beltagy, Iz and Peters, Matthew E. and Cohan, Arman},
  journal={arXiv preprint arXiv:2004.05150},
  year={2020}
}
```

<!-- related:start -->

---

## Related in This Collection

- [Attention Is All You Need](../../architectures/01-attention-is-all-you-need/summary.md)
- [BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](../../language-models/03-bert/summary.md)
- [FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness](../../techniques/16-flash-attention/summary.md)
- [Mamba: Linear-Time Sequence Modeling with Selective State Spaces](../../architectures/20-mamba/summary.md)
- [RoFormer: Enhanced Transformer with Rotary Position Embedding (RoPE)](../../techniques/54-rope-rotary-position-embedding/summary.md)
- [GQA: Grouped-Query Attention (and Multi-Query Attention)](../../architectures/75-grouped-query-attention/summary.md)
- [Mistral 7B](../../language-models/95-mistral-7b/summary.md)
- [YaRN: Efficient Context Window Extension of Large Language Models (YaRN)](../../techniques/130-yarn-context-extension/summary.md)

<!-- related:end -->
