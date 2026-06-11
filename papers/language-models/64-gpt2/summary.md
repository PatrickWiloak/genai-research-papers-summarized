---
title: "Language Models are Unsupervised Multitask Learners (GPT-2)"
slug: "64-gpt2"
number: 64
category: "language-models"
authors: "Alec Radford, Jeffrey Wu, Rewon Child, David Luan, Dario Amodei, Ilya Sutskever (OpenAI)"
published: "February 2019"
year: 2019
url: "https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf"
tags: ["language-model", "scaling", "pretraining"]
---

# Language Models are Unsupervised Multitask Learners (GPT-2)

**Authors:** Alec Radford, Jeffrey Wu, Rewon Child, David Luan, Dario Amodei, Ilya Sutskever (OpenAI)

**Published:** February 2019

**Paper Link:** https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf

---

## Why This Paper Matters

GPT-2 is the bridge between GPT-1 and GPT-3. It took the decoder-only Transformer from GPT-1, scaled it dramatically (up to 1.5 billion parameters), trained it on a far larger and cleaner dataset, and made a bold claim: **a language model trained well enough will perform many NLP tasks without ever seeing task-specific training data**. The model just needs a prompt.

This established the zero-shot transfer principle that GPT-3 would later prove at scale, and it launched the "pre-train on everything, prompt to steer" recipe that now underlies virtually all large language model products.

---

## The Core Innovation: Zero-Shot Task Transfer

### The Previous Paradigm

Before GPT-2, the standard workflow was:

```
Pre-train on large text corpus
    down-arrow
Fine-tune on labeled task data (sentiment, translation, QA, ...)
    down-arrow
Deploy for that specific task
```

Each task required its own fine-tuned model. Collecting labeled data was expensive, and you could not reuse a fine-tuned model for a different job.

### The GPT-2 Claim

The paper argues that a sufficiently capable language model learns to perform tasks **implicitly** because those tasks appear naturally in text. A web page that reads "Translate to French: ... " followed by a French sentence is, in effect, a translation training example. A paragraph that poses a question and then answers it is a QA example. A model that has read billions of such pages has absorbed all these tasks.

At inference time you do not fine-tune at all - you just frame the task in the prompt:

```
Zero-shot translation:
"Translate the following sentence to French: 'The cat sat on the mat.'"

Zero-shot summarization:
"<long article> TL;DR:"

Zero-shot QA:
"Q: What is the capital of France?  A:"
```

No gradient updates. No task labels. Just text.

---

## Key Components Explained

### 1. Decoder-Only Transformer (Scaling GPT-1)

GPT-2 keeps the same architecture as GPT-1 - a stack of Transformer decoder blocks, each with:
- **Masked self-attention** (each token can only attend to tokens that came before it)
- **Feed-forward layers**
- **Layer normalization** (moved to the input of each sub-layer, a small but important tweak from the original Transformer)

The key change is size. GPT-1 had 117M parameters. GPT-2 was released in four sizes:

| Model  | Layers | Attention heads | Parameters |
|--------|--------|-----------------|------------|
| Small  | 12     | 12              | 117M       |
| Medium | 24     | 16              | 345M       |
| Large  | 36     | 20              | 762M       |
| XL     | 48     | 25              | 1.5B       |

Bigger context window too: 1,024 tokens vs. 512 in GPT-1.

### 2. WebText - The Training Dataset

Previous language models trained on Wikipedia or BookCorpus, which are large but curated toward formal writing. GPT-2 trained on **WebText**: 40 GB of web text scraped from outbound links on Reddit posts with at least 3 karma upvotes.

Why that filtering heuristic? Reddit upvotes act as a rough quality signal - humans found the linked pages interesting or useful. This gave the model exposure to an enormous variety of domains: news, fiction, code, Q&A, manuals, recipes, arguments, instructions.

The dataset was never publicly released, but its construction logic was documented and later replicated as "OpenWebText."

### 3. Byte-Level BPE Tokenization

GPT-1 used character-level BPE after some preprocessing. GPT-2 switched to **byte-level BPE**, which operates directly on raw UTF-8 bytes rather than Unicode characters.

Why does this matter? It means:
- The vocabulary can represent **any string** without an unknown-token fallback
- Rare words, URLs, code snippets, and non-English text all tokenize cleanly
- No text preprocessing is needed - the model handles punctuation, emoji, and foreign scripts natively

Vocabulary size: 50,257 tokens.

**Analogy:** Imagine you're building a set of Lego pieces to represent every English word. Character-level BPE starts with 26 letters. Standard BPE starts with common words. Byte-level BPE starts with 256 raw byte values - it can represent literally any document ever written in any language.

### 4. Language Modeling as the Universal Task

GPT-2's training objective is the same as GPT-1 and every other autoregressive LM: predict the next token. No special classification heads, no task labels.

The paper formalizes the intuition that any supervised NLP task can be expressed as:

```
p(output | input, task)
```

And "task" is just more text. A question-answering system is nothing more than a model that, given context + question as text, assigns high probability to the correct answer tokens.

---

## Key Results

GPT-2 XL (1.5B) was evaluated zero-shot - no fine-tuning - on a variety of benchmarks:

**Language Modeling:**
- Achieved state-of-the-art perplexity on 7 out of 8 tested language modeling datasets, despite never being trained on those datasets specifically.

**Reading Comprehension (CoQA):**
- 55 F1 zero-shot vs. 89 F1 for fine-tuned models. Impressive for zero-shot; still a meaningful gap.

**Summarization (CNN/DailyMail):**
- Reasonable summaries when prompted with "TL;DR:" - not state-of-the-art, but coherent without any fine-tuning.

**Translation (English to French):**
- 11.5 BLEU zero-shot. Far below supervised models, but non-trivial for a model that received no explicit translation training.

**Children's Book Test (Common Noun Task):**
- 93.3% - competitive with task-specific fine-tuned models.

The headline result was not "GPT-2 beats fine-tuned models" - it mostly did not. The headline was that a single model trained on one objective showed **non-trivial performance across all these tasks with no task-specific supervision at all**.

---

## Why This Was Revolutionary

### 1. Zero-Shot Transfer Proved Possible

Prior to GPT-2, zero-shot generalization across diverse NLP tasks was considered impractical. GPT-2 showed it could happen - and that the capability scaled with model size. The 1.5B model was consistently better than the 117M model in zero-shot settings.

### 2. Prompting as a Programming Interface

The paper implicitly introduced the idea of **prompt engineering** - framing your input carefully so the model interprets it as the desired task. This became central to how practitioners use LLMs today.

### 3. Data Quality Over Data Size

WebText (40 GB) was smaller than some competing corpora but better filtered. The paper helped establish that dataset quality and diversity matter at least as much as raw size.

### 4. The Staged Rollout Controversy

OpenAI made an unusual decision: they initially released only the smallest GPT-2 models (117M and 345M), citing concerns that the 1.5B model was "too dangerous to release" because of its ability to generate convincing fake text at scale.

This sparked significant debate in the AI research community:
- Critics argued it was security theater - the capability was not uniquely dangerous and withholding research hindered the field.
- Supporters argued it was responsible disclosure - demonstrating a new norm of evaluating potential misuse before releasing powerful models.
- All models were eventually released in stages over 9 months; no major misuse incidents occurred, but the debate about responsible release practices was lasting.

This was among the first times an AI lab publicly wrestled with dual-use concerns, foreshadowing the much larger debates that would follow GPT-3, GPT-4, and beyond.

---

## Real-World Impact and Descendants

### Direct Successors

- **GPT-3** ([see summary](../04-gpt3-few-shot-learners/summary.md), OpenAI, 2020): 175B parameters, proved the zero-shot/few-shot thesis at scale, launched the prompt-engineering era.
- **GPT-4** ([see summary](../36-gpt4/summary.md), OpenAI, 2023): Multimodal, substantially more capable, closed-source.
- **GPT-4o** ([see summary](../40-gpt4o/summary.md), OpenAI, 2024): Unified omni-modal model.

### Broader Influence

- Every decoder-only LLM today (LLaMA, Mistral, Claude, Gemini, Grok) traces its architecture lineage through GPT-2's design choices.
- The byte-level BPE tokenizer became standard practice; GPT-3, GPT-4, and others use variants of the same approach.
- WebText's construction methodology inspired OpenWebText, The Pile, RedPajama, and other open training datasets.
- The staged rollout debate directly shaped later norms around model cards, system cards, and responsible disclosure frameworks.

---

## Key Takeaways

1. **Zero-shot transfer is real** - a model trained only to predict text can perform many tasks without fine-tuning, if it is large and well-trained enough.
2. **Scale matters** - the 1.5B model showed substantially stronger zero-shot results than the 117M model; the scaling story continued straight into GPT-3.
3. **Data quality is a lever** - WebText's human-filtered quality signal produced a better model than raw crawl data of the same size would have.
4. **Byte-level BPE eliminates unknowns** - the tokenizer choice has lasting practical impact on robustness with rare text.
5. **Prompting is an interface** - how you frame a request changes what a language model does; this was demonstrated systematically here for the first time.
6. **Responsible release is a real tension** - GPT-2's staged rollout introduced questions the field is still working through.

---

## Limitations and Future Directions

### Limitations

- **Still behind fine-tuned models** on most tasks - zero-shot was impressive but not yet practical for production NLP in 2019.
- **No instruction following** - GPT-2 continues text; it does not reliably follow instructions unless they happen to match patterns in WebText. That required InstructGPT/RLHF ([see summary](../05-instructgpt-rlhf/summary.md)).
- **Repetition and incoherence over long outputs** - generation quality degrades over long passages; the model loses track of earlier context.
- **No grounding** - the model hallucinates facts confidently; it has no mechanism to verify claims against an external source.
- **Context window of 1,024 tokens** - adequate for 2019 but small by later standards (GPT-3: 2,048; GPT-4: 128,000+).

### What Came Next

- **GPT-3** (2020) validated the scaling hypothesis with 100x more parameters.
- **InstructGPT / RLHF** (2022) solved the instruction-following gap by fine-tuning with human feedback.
- **Retrieval-Augmented Generation (RAG)** addressed the grounding problem.
- **Flash Attention and longer context windows** addressed the 1,024-token ceiling.

---

## Further Reading

- **Original paper:** https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf
- **OpenAI blog post (Feb 2019):** https://openai.com/research/language-unsupervised
- **The Illustrated GPT-2 (Jay Alammar):** https://jalammar.github.io/illustrated-gpt2/
- **OpenWebText replication of WebText:** https://skylion007.github.io/OpenWebTextCorpus/
- **GPT-3 paper (direct successor):** https://arxiv.org/abs/2005.14165
- **InstructGPT paper (instruction tuning):** https://arxiv.org/abs/2203.02155

---

## Citation

```bibtex
@article{radford2019language,
  title={Language models are unsupervised multitask learners},
  author={Radford, Alec and Wu, Jeffrey and Child, Rewon and Luan, David and Amodei, Dario and Sutskever, Ilya},
  journal={OpenAI blog},
  volume={1},
  number={8},
  pages={9},
  year={2019}
}
```

<!-- related:start -->

---

## Related in This Collection

- [Language Models are Few-Shot Learners (GPT-3)](../../language-models/04-gpt3-few-shot-learners/summary.md)
- [Training Language Models to Follow Instructions with Human Feedback (InstructGPT)](../../language-models/05-instructgpt-rlhf/summary.md)
- [Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks (RAG)](../../techniques/13-rag/summary.md)
- [FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness](../../techniques/16-flash-attention/summary.md)
- [GPT-4 Technical Report](../../language-models/36-gpt4/summary.md)
- [GPT-4o: The First Omni Model](../../language-models/40-gpt4o/summary.md)

<!-- related:end -->
