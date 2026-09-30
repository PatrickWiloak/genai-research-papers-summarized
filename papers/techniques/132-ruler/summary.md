---
title: "RULER: What's the Real Context Size of Your Long-Context Language Models? (RULER)"
slug: "132-ruler"
number: 132
category: "techniques"
authors: "Cheng-Ping Hsieh, Simeng Sun, Samuel Kriman, Shantanu Acharya, Dima Rekesh, Fei Jia, Yang Zhang, Boris Ginsburg (NVIDIA)"
published: "April 2024 (COLM 2024)"
year: 2024
url: "https://arxiv.org/abs/2404.06654"
tags: ["long-context", "evaluation", "benchmarks"]
---

# RULER: What's the Real Context Size of Your Long-Context Language Models? (RULER)

**Authors:** Cheng-Ping Hsieh, Simeng Sun, Samuel Kriman, Shantanu Acharya, Dima Rekesh, Fei Jia, Yang Zhang, Boris Ginsburg (NVIDIA)
**Published:** April 2024 (COLM 2024)
**Paper:** [arxiv.org/abs/2404.06654](https://arxiv.org/abs/2404.06654)

---

## Why This Paper Matters

By early 2024 model launches advertised context windows of 128K, 200K, even a million tokens. The evidence usually offered was a **needle-in-a-haystack** chart: hide one fact in a long document, ask for it back, and show a wall of green. Nearly every model passed.

RULER asked whether passing that test meant the model could actually *use* its context. The answer was mostly no. Across 17 long-context models, performance fell sharply as inputs grew, and **only about half of the models that claimed 32K or more still met a reasonable quality bar at 32K**.

- **It separated "claimed" from "effective" context length** and gave the field a number for the gap.
- **It went beyond retrieval.** Multi-hop tracing and aggregation tasks test whether a model can follow and combine information, not just find a string.
- **It is synthetic and configurable**, so it can be run at any length and cannot be memorised from the web in the usual way.
- **It became a standard long-context report card.** Model technical reports, such as Qwen2.5-1M's, publish RULER scores.

**The insight:** a single needle is too easy. Real long-context work involves many similar-looking distractors, several pieces of evidence, chains of reference and summaries over the whole input. Make the test look like that, and the advertised numbers shrink.

---

## The Problem: Needle-in-a-Haystack Is a Superficial Test

The popular needle test (Kamradt, 2023) and the older passkey task (Mohtashami and Jaggi, 2023) hide one distinctive fact in filler text. They measure one skill - **exact retrieval of a salient string** - and the filler is often unrelated essays, so the needle stands out. A model can score perfectly while being unable to do anything more demanding with its context.

---

## The Core Innovation: 13 Tasks in 4 Categories

RULER keeps the needle idea and makes it harder along several axes.

### 1. Retrieval (extended needle-in-a-haystack)
- **Single needle**, with different kinds of keys and values (words, numbers, UUIDs) and different haystacks.
- **Multi-key:** many needles present, only one is asked for - the rest are distractors.
- **Multi-value:** one key has several values; the model must return all of them.
- **Multi-query:** several needles must be retrieved in one answer.

### 2. Multi-hop tracing (variable tracking)
Chains of assignments are scattered through the text:

```
... X1 = 12345 ... X2 = X1 ... X3 = X2 ... X4 = X3 ...
Question: which variables hold the value 12345?
```

This is a proxy for coreference - following "it", "the company", "the defendant" back to the original entity across a long document.

### 3. Aggregation
- **Common words extraction:** find the most frequent words in a long list.
- **Frequent words extraction:** the same with a skewed distribution.

These test whether the model can summarise over the *whole* context rather than locate one part of it.

### 4. Question answering
Existing short-context QA datasets with the gold paragraph buried among many distractor paragraphs.

### The pass bar
To turn scores into an "effective length", the authors used **Llama 2 7B's performance at 4K tokens (85.6%)** as the threshold. A model's effective context is the longest length at which it still beats that bar.

---

## Key Results

The paper evaluated 17 models at lengths from 4K to 128K.

- **Almost all models scored near-perfectly on the vanilla needle test**, then dropped substantially on the full suite as length increased.
- **Claimed vs effective context** for some well-known models at the time:

| Model | Claimed | Effective (RULER) |
|---|---|---|
| GPT-4 | 128K | 64K |
| Yi-34B | 200K | 32K |
| LWM (Large World Model) | 1M | under 4K |
| Gemini 1.5 Pro | 1M | over 128K (the longest tested) |

- **Typical failure modes** as context grew: returning a distractor needle instead of the asked one, copying chunks of context verbatim, and falling back on parametric knowledge instead of reading the input.
- **Bigger models degraded less.** Model size helped more than training on longer sequences alone.
- **Non-Transformer architectures** tested (RWKV and Mamba variants) fell well behind Transformer baselines on these tasks.

---

## Why This Was Revolutionary

- **It moved the conversation from "how long" to "how well".** "Effective context" is now a standard way to talk about long-context models.
- **It made long-context evaluation cheap and repeatable.** Synthetic generation at any length, with controllable difficulty.
- **It set expectations for builders.** If a model's effective length is a fraction of its advertised window, retrieval ([RAG](../13-rag/summary.md)) and careful context design still matter.

---

## Real-World Impact

- **Model reports adopted it.** Long-context releases routinely publish RULER tables alongside needle charts.
- **Harder successors followed.** NoLiMa (ICML 2025) removed literal word overlap between question and needle, forcing the model to make a semantic link, and found steeper drops still.
- **It shaped context-extension work.** Methods like [YaRN](../130-yarn-context-extension/summary.md) are now judged on whether they raise effective length, not just keep perplexity low.

---

## Key Takeaways for Practitioners

1. **Treat advertised context as an upper bound.** Test at the lengths you plan to use.
2. **Distractors are the real enemy.** Documents full of similar entries (logs, contracts, tables) are harder than a needle in unrelated text.
3. **Put critical information where the model can use it.** Shorter, better-organised contexts often beat stuffing the whole window.
4. **Aggregation is weak.** "Count" or "summarise everything" over very long inputs is less reliable than "find X".

---

## Limitations & Future Directions

- **Synthetic tasks are proxies.** Variable tracking approximates coreference but is not natural language understanding.
- **Some tasks can be solved by shortcuts**, such as string matching, which newer benchmarks like NoLiMa deliberately remove.
- **The pass bar is arbitrary.** Llama 2 7B at 4K is a reasonable anchor, but "effective length" shifts if you change it.
- **Scores age.** The model table above is a 2024 snapshot; the method, not the numbers, is what lasts.

---

## Further Reading

- **Original Paper:** [arxiv.org/abs/2404.06654](https://arxiv.org/abs/2404.06654)
- **Code:** [github.com/NVIDIA/RULER](https://github.com/NVIDIA/RULER)
- **In this collection:** [YaRN](../130-yarn-context-extension/summary.md), [Longformer](../../architectures/131-longformer/summary.md), [RAG](../13-rag/summary.md), [Gemini 2.5](../../multimodal/29-gemini-2.5/summary.md)
- **Explainers:** [Context windows](../../../explainers/concepts/context-windows.md), [Contamination and saturation](../../../explainers/benchmarks/contamination-and-saturation.md)

## Citation

```bibtex
@inproceedings{hsieh2024ruler,
  title={RULER: What's the Real Context Size of Your Long-Context Language Models?},
  author={Hsieh, Cheng-Ping and Sun, Simeng and Kriman, Samuel and Acharya, Shantanu and Rekesh, Dima and Ginsburg, Boris and others},
  booktitle={Conference on Language Modeling (COLM)},
  year={2024}
}
```

<!-- related:start -->

---

## Related in This Collection

- [Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks (RAG)](../../techniques/13-rag/summary.md)
- [LLaMA 2: Open Foundation and Fine-Tuned Chat Models](../../language-models/17-llama2/summary.md)
- [Mamba: Linear-Time Sequence Modeling with Selective State Spaces](../../architectures/20-mamba/summary.md)
- [Qwen3: Technical Report](../../language-models/28-qwen3/summary.md)
- [Gemini 2.5: Pushing the Frontier with Advanced Reasoning, Multimodality, Long Context, and Next Generation Agentic Capabilities](../../multimodal/29-gemini-2.5/summary.md)
- [GPT-4 Technical Report](../../language-models/36-gpt4/summary.md)
- [Mastering Diverse Domains through World Models (DreamerV3)](../../techniques/105-dreamerv3/summary.md)
- [YaRN: Efficient Context Window Extension of Large Language Models (YaRN)](../../techniques/130-yarn-context-extension/summary.md)

<!-- related:end -->
