# Context Windows

**In one line:** A context window is how many tokens a model can take in at once, and the number on the spec sheet is an upper bound on what fits, not a promise of how well the model uses it, so treat long context as a tool to test rather than a replacement for choosing what goes in.
**Last reviewed:** 2026-09-30

---

## The short version

- The **context window** is the maximum number of [tokens](tokenization.md) a model can attend to in one request: system prompt, conversation history, documents, tool results, the model's own hidden reasoning and its answer all share it.
- Advertised windows grew from a few thousand tokens in 2020-2022 to **about 1 million tokens** for the flagship APIs of Anthropic, OpenAI and Google as of September 2026, and Meta advertised 10 million for Llama 4 Scout in April 2025.
- **Advertised is not usable.** Benchmarks such as RULER and NoLiMa find that many models degrade well before their limit, often by 32K tokens, once the task is harder than finding an exact phrase.
- Models are best at using information at the **start and end** of the context and worst in the **middle** ("lost in the middle"), and performance drifts down as irrelevant material piles up ("context rot").
- Long context and **retrieval-augmented generation (RAG)** are complements, not rivals: long context is simpler and sees everything; RAG is cheaper, fresher and scales past any window. Most real systems use both.

## What the window actually contains

A model has no memory between requests. Everything it "knows" about your conversation is re-sent every turn and must fit inside the window.

```
+------------------------------------------------------------+
| system prompt | tools | history | documents | [thinking] | answer |
+------------------------------------------------------------+
<----------------------- context window ----------------------->
                                             ^ output tokens count too
```

Two practical consequences:

- **Output and reasoning share the budget.** Providers set a separate maximum output length, but output still sits in the same window. Reasoning models make this matter: OpenAI's reasoning guide, as of September 2026, suggests reserving at least 25,000 tokens for reasoning and output when you start experimenting (see [reasoning models](reasoning-models.md)).
- **Tokens are not words.** The same "1M tokens" holds different amounts of text on different tokenizers. Anthropic's documentation says 1M tokens is roughly 555,000 English words on its current tokenizer, versus about 750,000 on earlier Claude models.

## Why the window is limited at all

Three separate constraints, each with its own fixes.

| Constraint | Why it bites | Main fixes |
|---|---|---|
| **Position handling** | A model trained on sequences of length N has never seen position N+1 and often breaks there | Better position encodings and context extension (RoPE, Position Interpolation, YaRN) |
| **Attention compute** | Standard attention compares every token with every other, so cost grows with the square of length | Efficient kernels (FlashAttention), sparse or sliding-window attention (Longformer), hybrid architectures |
| **Memory** | The [KV cache](kv-cache.md) grows linearly with length and must sit in accelerator memory | GQA, MLA, cache quantization, paging |

**Position.** Most modern models encode position with rotary embeddings ([RoPE](../../papers/techniques/54-rope-rotary-position-embedding/summary.md)), which rotate query and key vectors by an angle that depends on position. Context extension methods such as Position Interpolation and **YaRN** rescale those rotations so that a model pre-trained at a short length can be fine-tuned briefly to work at a much longer one (see the [YaRN summary](../../papers/techniques/130-yarn-context-extension/summary.md)). The published configurations show this at work: Llama 3.3 70B was pre-trained with 8,192-token positions and extended by a factor of 8 to 131,072; DeepSeek-V3 applies YaRN with a factor of 40 on top of a 4,096-token base to reach 163,840.

**Compute.** Doubling the input roughly quadruples the attention work. [Longformer](../../papers/architectures/131-longformer/summary.md) showed one escape: let most tokens attend only to a local sliding window, with a few global tokens that see everything. Variants of windowed and sparse attention now appear in many long-context models, often mixed with some full-attention layers.

**Memory.** Even when the model can handle a million tokens, serving that to many users at once is a memory bill; see the worked example on the [KV cache](kv-cache.md) page.

## Advertised versus usable length

The classic test of a long context is **needle in a haystack**: hide one sentence in a long document and ask for it. Most modern models score near-perfectly, which is why it stopped being informative. Harder tests tell a different story.

- **RULER** (NVIDIA, 2024) added multi-needle retrieval, multi-hop tracing, aggregation and question answering. Of 17 long-context models tested, all claiming 32K tokens or more, only about half kept satisfactory performance at 32K. See the [RULER summary](../../papers/techniques/132-ruler/summary.md).
- **NoLiMa** (ICML 2025) removed the literal word overlap between question and answer, so the model has to make a semantic link rather than pattern-match. Of 13 models claiming at least 128K, 11 fell below half of their short-context score at 32K; GPT-4o dropped from 99.3% to 69.7%.
- **Context Rot** (Chroma, July 2025) tested 18 models including GPT-4.1, Claude 4, Gemini 2.5 and Qwen3 and found performance declined as input grew even on simple tasks such as repeating text, with larger drops when distractors were present or when the question and answer shared little wording.

```
accuracy
 100 |****
     |    ****                     "literal match" (needle in a haystack)
     |        ****************************
     |   ...
     |      .....
     |           ......            "needs understanding" (RULER, NoLiMa)
     |                 ........
     +-------------------------------------> input length
      4K   16K   32K   64K   128K   1M          (illustrative shape, not data)
```

Models have improved steadily since these studies, and results vary a lot by model and task. The durable lesson is the method: **test at the lengths and on the kinds of tasks you actually need**, not on the headline number.

## Lost in the middle

Liu et al. (2023, published in TACL) gave models a set of documents with the answer placed at different positions. Accuracy was highest when the relevant document was at the **beginning or end** and dropped markedly when it was in the **middle**, a U-shaped curve, even for models built for long inputs.

```
accuracy by position of the relevant document
 high |*                                *
      | *                             *
      |   *                        *
  low |      *  *  *  *  *  *  *
      +----------------------------------
       start          middle          end
```

Newer models show a weaker effect on simple retrieval, but the practical advice has held up: **put instructions and the question where the model will weight them** (often at the end, after long documents), keep the most important material near the edges, and do not pad the context with things that are merely possibly relevant.

## Long context versus RAG

**Retrieval-augmented generation** fetches the handful of passages most relevant to the question from an index and puts only those in the context (see the [RAG summary](../../papers/techniques/13-rag/summary.md) and the hands-on [RAG explained](https://github.com/PatrickWiloak/cloud-data-ai-security-zero-to-hero/blob/main/learn/concepts/rag-explained.md)). Long context instead puts the whole corpus in.

| | Long context ("just paste it all") | RAG |
|---|---|---|
| Setup | None: no chunking, no index | Chunking, embeddings, vector search, tuning |
| Sees everything | Yes, so whole-document questions ("what changed across these contracts?") work | Only what retrieval found; a miss is invisible |
| Cost per question | Pay for every token every time (reduced by prompt caching) | Pay for a few thousand tokens |
| Latency | Grows with input length | Mostly constant |
| Scale | Capped by the window | Millions of documents |
| Freshness and access control | Rebuild the prompt | Update the index; filter by user permissions |
| Failure mode | Dilution, lost in the middle, context rot | Retrieval errors, bad chunk boundaries |

The practical pattern is to **retrieve generously, then let a long-context model read**: use search to narrow millions of documents to a few hundred thousand tokens, then rely on the model to reason across them. [Prompt caching](https://github.com/PatrickWiloak/cloud-data-ai-security-zero-to-hero/blob/main/learn/concepts/prompt-caching.md) makes the long-context half cheaper when the same material is reused. The sibling repo's [fine-tuning vs RAG](https://github.com/PatrickWiloak/cloud-data-ai-security-zero-to-hero/blob/main/learn/concepts/fine-tuning-vs-rag.md) page covers the third option, baking knowledge into weights.

For agents that run for hours, the question becomes **context management**: summarising or dropping old tool results, keeping notes outside the window and re-reading them. This has become a design discipline of its own, often called context engineering.

## Where the numbers stand (as of September 2026)

| Model family | Advertised context | Source |
|---|---|---|
| Claude (Opus 5.5, Sonnet 5.5, Fable 5.1) | 1M tokens | Anthropic models overview |
| Claude Haiku 4.5 | 200K tokens | Anthropic models overview |
| OpenAI GPT-6 Astra, GPT-6.1 Sol, GPT-6 Luna | 1.05M tokens | OpenAI models page |
| Gemini 3.8 Flash | 1,048,576 input tokens | Google Gemini API docs |
| Llama 4 Scout (open weights, April 2025) | 10M tokens | Meta announcement |

These are maximums. What a given deployment allows, and what it charges above certain lengths, can differ; check the provider's pricing page.

## What to watch

- **Effective-length benchmarks catching up with advertised length.** Watch RULER-style and NoLiMa-style results for the newest 1M-token models, and independent long-context evaluations, rather than vendor needle tests.
- **Architectures that change the cost curve.** Hybrid attention and state-space designs could make multi-million-token contexts affordable to serve, which is a different question from whether they are used well.
- **Context management in agents.** Automatic compaction and memory tools may matter more than raw window size for long-running work.

## Read next

- [YaRN and Position Interpolation](../../papers/techniques/130-yarn-context-extension/summary.md)
- [RoPE](../../papers/techniques/54-rope-rotary-position-embedding/summary.md)
- [Longformer](../../papers/architectures/131-longformer/summary.md)
- [RULER](../../papers/techniques/132-ruler/summary.md)
- [RAG](../../papers/techniques/13-rag/summary.md), [Dense retrieval](../../papers/techniques/87-dense-retrieval/summary.md), [GraphRAG](../../papers/techniques/60-graph-rag/summary.md)
- [FlashAttention](../../papers/techniques/16-flash-attention/summary.md)
- [KV cache](kv-cache.md), [Tokenization](tokenization.md), [Reasoning models](reasoning-models.md)
- Hands-on: [RAG explained](https://github.com/PatrickWiloak/cloud-data-ai-security-zero-to-hero/blob/main/learn/concepts/rag-explained.md), [Embeddings and vector search](https://github.com/PatrickWiloak/cloud-data-ai-security-zero-to-hero/blob/main/learn/concepts/embeddings-and-vector-search.md)

## Sources

- Liu et al., "Lost in the Middle: How Language Models Use Long Contexts", TACL 2024. https://arxiv.org/abs/2307.03172
- Hsieh et al., "RULER: What's the Real Context Size of Your Long-Context Language Models?", 2024. https://arxiv.org/abs/2404.06654
- Modarressi et al., "NoLiMa: Long-Context Evaluation Beyond Literal Matching", ICML 2025. https://arxiv.org/abs/2502.05167
- Hong, Troynikov, Huber, "Context Rot: How Increasing Input Tokens Impacts LLM Performance", Chroma, July 2025. https://www.trychroma.com/research/context-rot
- Anthropic, Models overview (context windows, tokens-to-words), accessed 2026-09-30. https://platform.claude.com/docs/en/about-claude/models/overview
- OpenAI, Models page, accessed 2026-09-30. https://developers.openai.com/api/docs/models
- OpenAI, Reasoning models guide (reserve at least 25,000 tokens), accessed 2026-09-30. https://developers.openai.com/api/docs/guides/reasoning
- Google, Gemini 3.8 Flash model page, accessed 2026-09-30. https://ai.google.dev/gemini-api/docs/models/gemini-3.8-flash
- Meta, "The Llama 4 herd", April 2025. https://ai.meta.com/blog/llama-4-multimodal-intelligence/
- Llama 3.3 70B configuration (`original_max_position_embeddings` 8192, factor 8, 131,072). https://huggingface.co/meta-llama/Llama-3.3-70B-Instruct
- DeepSeek-V3 configuration (YaRN factor 40 from 4,096 to 163,840). https://huggingface.co/deepseek-ai/DeepSeek-V3/raw/main/config.json
