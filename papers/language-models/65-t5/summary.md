---
title: "Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer (T5)"
slug: "65-t5"
number: 65
category: "language-models"
authors: "Colin Raffel, Noam Shazeer, Adam Roberts, Katherine Lee, Sharan Narang, Michael Matena, Yanqi Zhou, Wei Li, Peter J. Liu (Google)"
published: "October 2019 (JMLR 2020)"
year: 2019
url: "https://arxiv.org/abs/1910.10683"
tags: ["language-model", "architecture", "pretraining"]
---

# Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer (T5)

**Authors:** Colin Raffel, Noam Shazeer, Adam Roberts, Katherine Lee, Sharan Narang, Michael Matena, Yanqi Zhou, Wei Li, Peter J. Liu (Google)

**Published:** October 2019 (JMLR 2020)

**Paper Link:** https://arxiv.org/abs/1910.10683

---

## Why This Paper Matters

T5 standardized the way we think about language models by casting **every NLP task as a text-in, text-out problem**. Translation, summarization, classification, question answering, and even regression all feed the same encoder-decoder model a text prompt and read off a text answer. This "one interface to rule them all" made transfer learning dramatically simpler and laid the conceptual groundwork for modern instruction-tuned models. The paper is also a landmark **systematic study**: the authors ran hundreds of controlled experiments to identify which architectural choices, pretraining objectives, datasets, and scaling decisions actually matter - giving the field a rigorous empirical foundation rather than received wisdom.

---

## The Core Innovation: Unified Text-to-Text

### The Problem with Task-Specific Heads

Before T5, each NLP task had its own output format:
- Sentiment analysis returned a class label
- Translation returned a token sequence from a separate decoder
- Regression returned a floating-point number
- Question answering returned a span index into the input

Every task needed a custom output layer, and comparing techniques across tasks was awkward because the interfaces were incompatible.

### T5's Solution: Everything is Text

T5 frames every task identically:

```
Input:  "<task prefix> <input text>"
Output: "<answer as a string>"
```

Concrete examples:

```
Translation:
  Input:  "translate English to German: The house is wonderful."
  Output: "Das Haus ist wunderbar."

Sentiment Classification:
  Input:  "sst2 sentence: The movie was fantastic."
  Output: "positive"

Summarization:
  Input:  "summarize: Scientists discover that..."
  Output: "Researchers found..."

Question Answering:
  Input:  "question: Who wrote Hamlet? context: William Shakespeare..."
  Output: "William Shakespeare"

Regression (STS-B similarity, scaled 1-5):
  Input:  "stsb sentence1: The cat sat. sentence2: A cat is sitting."
  Output: "3.8"
```

The model always does the same thing: generate tokens. No task-specific heads, no conditional output formats. A single model, fine-tuned once per task, with task identity encoded in the input prefix.

---

## Key Components Explained

### 1. Encoder-Decoder Transformer Architecture

T5 uses the original **encoder-decoder Transformer** from "Attention is All You Need" with minor modifications (relative position biases instead of sinusoidal encodings, pre-norm layer normalization).

```
Input Text + Task Prefix
        |
+-------------------------+
|  ENCODER                |
|  - Reads full input     |
|  - Bidirectional        |
|  - Builds rich context  |
+-------------------------+
        |
  Encoded Representation
        |
+-------------------------+
|  DECODER                |
|  - Generates output     |
|  - Left-to-right        |
|  - Cross-attends to     |
|    encoder output       |
+-------------------------+
        |
   Output Text (token by token)
```

Why encoder-decoder over encoder-only (BERT) or decoder-only (GPT)?
- Encoder-only models cannot generate text naturally
- Decoder-only models see the input autoregressively, not with full bidirectional context
- Encoder-decoder gets both: full bidirectional understanding of the input, autoregressive generation of the output

This choice proved well-suited to the text-to-text framing.

### 2. C4 - Colossal Clean Crawled Corpus

Previous pretraining datasets were relatively small or not publicly documented. T5 introduced **C4**, a massive, carefully cleaned web corpus:

- Sourced from **Common Crawl** - petabytes of raw web text scraped monthly
- Aggressive cleaning pipeline:
  - Kept only lines ending in terminal punctuation
  - Removed pages with fewer than 5 sentences
  - Deduplicated at the 3-sentence-span level
  - Filtered pages containing offensive content (word blocklist)
  - Removed non-English pages (using langdetect)
  - Discarded pages from known code/boilerplate sources (Lorem ipsum, etc.)
- Result: ~750 GB of clean English text (about 156 billion tokens)

The cleaning choices matter enormously. The paper's ablations showed that noisier pretraining data consistently hurt downstream performance, even when the noisy corpus was larger.

```
Raw Common Crawl
      |
  ~Petabytes of web text
      | (cleaning pipeline)
  C4: ~750 GB clean English
      |
  Pretrain T5 here
```

### 3. Span Corruption Pretraining Objective

T5 tested many pretraining objectives in its systematic study. The winner was **span corruption** (a variant of masked language modeling):

**Standard masked LM (BERT-style):**
```
Original: "The quick brown fox jumps over the lazy dog"
Masked:   "The [MASK] brown [MASK] jumps over the [MASK] dog"
Target:   predict each [MASK] independently
```

**Span corruption (T5):**
```
Original: "The quick brown fox jumps over the lazy dog"
Corrupt:  Replace *spans* of consecutive tokens with a single sentinel:
Input:    "The <X> fox <Y> the lazy dog"
Target:   "<X> quick brown <Y> jumps over <Z>"
```

Key differences from BERT's MLM:
- Corrupts **spans** (consecutive tokens), not individual tokens - forces the model to recover multi-word phrases
- Uses **unique sentinel tokens** (`<X>`, `<Y>`, ...) rather than `[MASK]`, so the target is a single concatenated output sequence
- The **decoder** generates the corrupted spans - making pretraining format-consistent with fine-tuning (text generation)
- Default corruption rate: 15%; default mean span length: 3 tokens

This is more efficient than predicting every token (only corrupted spans are generated) and better aligned with downstream text generation.

```
Pretraining (span corruption):
  Input  -> "The <X> fox <Y> the lazy dog"
  Target -> "<X> quick brown <Y> jumps over <Z>"

Fine-tuning (translation):
  Input  -> "translate English to German: The quick brown fox"
  Target -> "Der schnelle braune Fuchs"

Same interface. Same model. Same training loop.
```

### 4. The Paper as a Systematic Study

A large fraction of the paper's value is not the final model but the **controlled experiments** comparing:

| Dimension | What was varied |
|-----------|-----------------|
| Architecture | Encoder-decoder vs. decoder-only vs. encoder-only |
| Pretraining objective | MLM, span corruption, deshuffling, GPT-style LM, many variants |
| Dataset | C4 vs. Wikipedia+Books vs. unfiltered Common Crawl vs. domain-specific |
| Training duration | Number of pretraining steps |
| Fine-tuning approach | Adapter layers vs. full fine-tuning vs. gradual unfreezing |
| Multi-task learning | Mixing tasks during pretraining vs. task-specific pretraining |
| Scale | Model size from ~60M to 11B parameters |

This breadth of ablations made T5 a reference paper researchers could cite to justify design decisions without rerunning the experiments themselves.

---

## T5 Model Sizes

| Variant | Parameters | Encoder Layers | Decoder Layers | d_model |
|---------|-----------|----------------|----------------|---------|
| T5-Small | 60M | 6 | 6 | 512 |
| T5-Base | 220M | 12 | 12 | 768 |
| T5-Large | 770M | 24 | 24 | 1024 |
| T5-3B | 3B | 24 | 24 | 1024 |
| T5-11B | 11B | 24 | 24 | 1024 |

T5-11B was the largest model in the paper and achieved the strongest results.

---

## Key Results

T5 set new state-of-the-art results across a wide range of benchmarks at publication:

**GLUE / SuperGLUE (language understanding):**
- SuperGLUE score of **88.9** with T5-11B (human baseline: 89.8)
- Near-human performance on a diverse multi-task benchmark

**SQuAD (question answering):**
- Exact match of **90.1** on SQuAD v1.1

**CNN/Daily Mail (summarization):**
- ROUGE-2 of **21.55** (strong improvement over baselines)

**WMT Translation (English-German, English-French, English-Romanian):**
- Competitive with dedicated translation models despite being a general-purpose model

**Key empirical findings from ablations:**
- Span corruption consistently outperformed other pretraining objectives
- Encoder-decoder beat decoder-only for most tasks tested
- Cleaner pretraining data beat larger but noisier data
- Scale (more parameters, more data, more compute) reliably improved results across all axes

---

## Why This Was Revolutionary

### 1. Single Unified Interface

Before T5, building an NLP system meant choosing task-specific architectures, loss functions, and output heads. After T5, the question became simply: "What text should my model output?" This conceptual simplification made systems far easier to build, compare, and extend.

### 2. Validated the Encoder-Decoder for Transfer Learning

BERT had popularized encoder-only pretraining. GPT had shown decoder-only scaling. T5 showed that the original encoder-decoder architecture from "Attention is All You Need" was highly competitive - and in many cases superior - for tasks requiring both understanding and generation.

### 3. C4 as a Public Benchmark Dataset

By releasing C4 and documenting the cleaning pipeline in detail, T5 gave the research community a reproducible large-scale pretraining corpus. Many subsequent models used C4 or variants of it.

### 4. Rigorous Empirical Grounding

The paper's systematic ablations answered questions that had been debated informally (Should I use masked LM or a language model objective? How much does data quality matter?) with controlled evidence. This changed how practitioners justified architectural choices.

### 5. The Instruction Framing Seed

The task-prefix format ("translate English to German: ...") is a direct ancestor of instruction tuning. When later work (InstructGPT, FLAN, instruction-tuned LLaMA) trained models to follow natural-language instructions, they were extending the intuition T5 systematized: the task identity belongs in the input text.

---

## Impact and Descendants

T5's influence runs through nearly every major model family that followed:

- **Transformer (2017)** - the encoder-decoder backbone T5 built on
  - See: [Attention Is All You Need](../../architectures/01-attention-is-all-you-need/summary.md)

- **BERT (2018)** - established pretraining + fine-tuning; T5 extended this to generative tasks
  - See: [BERT](../../language-models/03-bert/summary.md)

- **FLAN (2021, Google)** - fine-tuned T5 on 60+ tasks with natural-language instructions; direct bridge to instruction tuning

- **mT5 (2021, Google)** - multilingual T5 covering 101 languages

- **T5v1.1 / ExT5 / UL2** - improved pretraining objectives and efficiency on the T5 base

- **Flan-T5 (2022, Google)** - T5 with instruction fine-tuning, widely used open model

- **InstructGPT / ChatGPT** - adopted the "instruction as input text" framing at massive scale
  - See: [InstructGPT](../../language-models/05-instructgpt-rlhf/summary.md)

- **LLaMA family** - decoder-only successors that absorbed many of T5's transfer-learning lessons
  - See: [LLaMA](../../language-models/15-llama/summary.md)

---

## Key Takeaways

1. **Text-to-text is a powerful unifying abstraction** - every NLP task can be expressed as a text generation problem with minimal engineering overhead
2. **Data quality beats data quantity** - a cleaner 750 GB corpus outperformed noisier multi-terabyte alternatives
3. **Span corruption is a strong pretraining objective** - better than token-level masking for most downstream tasks
4. **Encoder-decoder scales well** - competitive with or better than encoder-only and decoder-only alternatives across understanding and generation tasks
5. **Scale matters consistently** - larger models, more data, and more compute all improved results reliably
6. **Empirical rigor is a contribution in itself** - systematic ablations give the community shared evidence to build on

---

## Limitations and Future Directions

### Limitations

**Computational cost of pretraining:**
- T5-11B required substantial TPU resources to pretrain from scratch
- Not reproducible for most academic labs without a precomputed checkpoint

**Fixed input/output length:**
- Like other Transformers of its era, T5 had a practical limit on sequence length (512 tokens for most experiments)
- Long documents required truncation

**Text-only:**
- T5 covers text tasks only; extending the text-to-text framing to images, audio, or structured data required follow-on work

**English-centric pretraining:**
- C4 is predominantly English; multilingual coverage required a separate model (mT5)

**Autoregressive decoding is slow:**
- At inference time, generating one token at a time is expensive for long outputs; this remained a challenge until speculative decoding and other acceleration techniques

### Directions That Followed

- **Instruction tuning** (FLAN, InstructGPT): replacing task-prefix strings with natural-language instructions and scaling to many more tasks
- **Efficient attention** (FlashAttention, Longformer): addressing the sequence-length limit
- **Mixture of Experts** (Switch Transformer, 2021): T5-style architecture with sparse routing to increase model capacity without proportional compute cost
- **Multi-modal text-to-text** (Flamingo, Gemini): extending the text-to-text interface to image and audio inputs

---

## Further Reading

- **Original Paper:** https://arxiv.org/abs/1910.10683
- **T5 GitHub (original):** https://github.com/google-research/text-to-text-transfer-transformer
- **Flan-T5 on HuggingFace:** https://huggingface.co/docs/transformers/model_doc/flan-t5
- **The Illustrated T5:** https://jalammar.github.io/illustrated-t5/
- **C4 dataset on HuggingFace:** https://huggingface.co/datasets/allenai/c4
- **FLAN paper (instruction-tuned T5):** https://arxiv.org/abs/2109.01652

---

## Citation

```bibtex
@article{raffel2020exploring,
  title={Exploring the limits of transfer learning with a unified text-to-text transformer},
  author={Raffel, Colin and Shazeer, Noam and Roberts, Adam and Lee, Katherine and Narang, Sharan and Matena, Michael and Zhou, Yanqi and Li, Wei and Liu, Peter J},
  journal={Journal of Machine Learning Research},
  volume={21},
  number={140},
  pages={1--67},
  year={2020}
}
```

<!-- related:start -->

---

## Related in This Collection

- [Attention Is All You Need](../../architectures/01-attention-is-all-you-need/summary.md)
- [BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](../../language-models/03-bert/summary.md)
- [Training Language Models to Follow Instructions with Human Feedback (InstructGPT)](../../language-models/05-instructgpt-rlhf/summary.md)
- [FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness](../../techniques/16-flash-attention/summary.md)
- [Mixtral of Experts (and the Mixture-of-Experts Architecture)](../../architectures/37-mixture-of-experts/summary.md)
- [Speculative Decoding: Fast Inference from Transformers](../../techniques/45-speculative-decoding/summary.md)
- [Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity](../../architectures/67-switch-transformer/summary.md)

<!-- related:end -->
