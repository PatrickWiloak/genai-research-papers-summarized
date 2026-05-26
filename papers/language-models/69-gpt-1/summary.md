# Improving Language Understanding by Generative Pre-Training (GPT-1)

**Authors:** Alec Radford, Karthik Narasimhan, Tim Salimans, Ilya Sutskever (OpenAI)

**Published:** June 2018 (OpenAI Technical Report)

**Paper Link:** https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf

---

## Why This Paper Matters

This is the paper that started the **GPT** lineage. GPT-1 introduced the now-standard recipe for modern language models: **unsupervised pretraining of a Transformer decoder on a large text corpus**, followed by **supervised fine-tuning on individual downstream tasks**. Every model in the GPT family — GPT-2, GPT-3 (#4), GPT-4 (#36), GPT-4o (#40), GPT-5 (#42), o1 (#31), Codex (#56) — descends from the architectural and methodological choices made here. So do LLaMA (#15, #17, #33, #41), Claude (#30, #43), and essentially every decoder-only LLM in production today.

Released in June 2018, GPT-1 was a 117-million-parameter model. By 2025 standards it's tiny. But the recipe it pioneered — pretrain a causal-attention Transformer on raw text, fine-tune it on whatever you need — turned out to be the most consequential idea in NLP of the last decade.

---

## The Problem in 2018

### Supervised NLP Was Data-Hungry
By 2018, deep learning had taken over NLP, but each task still required its own carefully labeled dataset:
- Sentiment classification: ~10k labeled reviews per dataset
- Natural language inference: ~500k annotated sentence pairs (SNLI)
- Question answering: ~100k labeled (question, answer) pairs (SQuAD)

Annotating data is slow and expensive. Meanwhile, the internet had **billions of words of unlabeled text** sitting unused.

### Previous Transfer Learning Was Limited
Word embeddings like **Word2Vec** (#63) and **GloVe** provided pretrained representations of individual words, but only at the input layer. The rest of each model still had to be trained from scratch per task.

**ELMo** (Peters et al., 2018) had recently shown that pretraining a bidirectional LSTM language model and using its contextual representations as features improved many tasks. But ELMo's representations were still fed into custom task-specific models — there was no unified architecture.

### The Open Question
Could a single large neural network, pretrained on unlabeled text, transfer to many downstream tasks with **minimal task-specific architecture changes**?

---

## The Core Innovation: Generative Pretraining + Discriminative Fine-Tuning

GPT-1's two-stage recipe:

### Stage 1: Unsupervised Pretraining

Train a **Transformer decoder** as a causal language model on a large corpus. Given a sequence of tokens, predict each next token from the ones that came before:

```
Loss = - Σ_i log P(token_i | token_1, ..., token_{i-1})
```

This is the same objective Shannon and others studied half a century earlier — predict the next word — applied to a deep neural network and a large dataset.

**Why this works:** To predict the next word well, the model has to learn syntax, semantics, world knowledge, reasoning patterns, and styles of writing. All of this is "for free" — no labels needed.

**Dataset:** BookCorpus, a collection of about **7,000 unpublished books** (~800 million words). The books contained long stretches of coherent prose, encouraging the model to learn long-range structure.

### Stage 2: Supervised Fine-Tuning

After pretraining, take the same model and fine-tune it on a labeled task. The original architecture is kept; only minimal changes are made to the input format and a small linear output head is added.

**The trick: task-specific input transformations.** Different tasks have different shapes:

```
Classification:    [START] document [EXTRACT] → predict class
Entailment:        [START] premise [DELIM] hypothesis [EXTRACT] → predict entail/contradict
Similarity:        [START] text1 [DELIM] text2 [EXTRACT] → predict score
Multiple choice:   [START] context [DELIM] answer_i [EXTRACT] → score each answer i
```

Each task is restructured into a sequence the same Transformer can process. The output of the final transformer block at the special `[EXTRACT]` position is fed through a small linear layer to produce the prediction. The entire model — pretrained weights plus new head — is fine-tuned jointly.

**Auxiliary objective:** During fine-tuning, the language modeling objective is also retained as a secondary loss. This helped generalization and accelerated convergence.

---

## The Architecture: Decoder-Only Transformer

GPT-1 takes the **decoder** half of the original Transformer (#1) and uses it standalone. Key properties:

- **Causal (masked) self-attention:** Each position can only attend to itself and previous positions. This makes the model autoregressive — well-suited to next-token prediction.
- **No encoder, no cross-attention:** Unlike the original Transformer and T5 (#68), GPT-1 is decoder-only. There is no separate input encoder.
- **12 Transformer blocks**
- **12 attention heads per block**
- **768-dimensional embeddings**
- **3072-dimensional feedforward layers**
- **Total: ~117 million parameters**

This decoder-only choice — and the staggering success of scaling it up — is one of the most consequential architectural decisions in modern AI. Years later, T5 (#68) would explicitly study encoder-decoder vs decoder-only and find encoder-decoder slightly better at moderate scales. But at very large scale (GPT-3, GPT-4, LLaMA), decoder-only won on simplicity and flexibility, and the GPT lineage starting here is why.

### Why Decoder-Only Works

- **Generative pretraining is autoregressive by nature** — the decoder shape fits perfectly.
- **Simpler architecture** — only one stack of layers, only one attention pattern.
- **Naturally handles open-ended generation** — chat, completion, code generation.
- **In-context learning emerges with scale** — feeding examples as part of the input prompt works because the same model handles both context and continuation.

---

## How Fine-Tuning Worked: One Architecture, Many Tasks

The clever part of GPT-1 is how it handled the diversity of NLP tasks **without changing the architecture**. The paper introduces input format transformations for four task types:

### 1. Classification
```
[START] text [EXTRACT]
```
Feed the document, take the final hidden state at [EXTRACT], project to class logits.

### 2. Entailment (does sentence A imply sentence B?)
```
[START] premise [DELIM] hypothesis [EXTRACT]
```
Concatenate with a delimiter, classify as entailment/contradiction/neutral.

### 3. Sentence Similarity
```
[START] sentence_A [DELIM] sentence_B [EXTRACT]   → h_AB
[START] sentence_B [DELIM] sentence_A [EXTRACT]   → h_BA
```
Process both orderings (since similarity is symmetric), sum the representations, project to similarity score.

### 4. Multiple Choice / QA
```
[START] context [DELIM] answer_1 [EXTRACT]   → score_1
[START] context [DELIM] answer_2 [EXTRACT]   → score_2
...
```
Score each candidate answer independently, softmax over scores to pick.

In each case, the **transformer body is unchanged** — only the input formatting and a tiny output head differ. This was a striking contrast to previous work where each task had its own bespoke architecture.

---

## Key Results

GPT-1 evaluated on 12 NLP benchmarks across 4 task categories. It set new state-of-the-art on **9 of 12** tasks:

| Task | Previous SOTA | GPT-1 |
|------|---------------|-------|
| Natural Language Inference (MultiNLI) | 80.6 | **82.1** |
| Question Answering (RACE) | 53.3 | **59.0** |
| Semantic Similarity (STS-B) | 81.0 | **82.0** |
| Story Cloze | 77.6 | **86.5** |
| Linguistic acceptability (CoLA) | 35.0 | **45.4** |

The improvements were significant: 5-9 point absolute gains on some tasks.

### Ablations
The paper showed that:
- **Pretraining was crucial:** Without it, the model performed dramatically worse.
- **All 12 layers helped:** Deeper transferred features helped most tasks.
- **The auxiliary LM objective helped on large datasets**, less so on small ones.

---

## Why This Was Revolutionary

### 1. Unified Architecture
A single Transformer body handles classification, entailment, similarity, multiple choice — all by reformatting the input. This was a huge simplification compared to bespoke architectures per task.

### 2. Unsupervised Pretraining at Scale Worked
GPT-1 demonstrated convincingly that pretraining on raw text gave a useful initialization for downstream tasks. This was the empirical foundation on which BERT (#3), GPT-2, GPT-3 (#4), and every modern LLM would build.

### 3. Decoder-Only Choice
By choosing the decoder-only path (vs BERT's encoder-only or T5's encoder-decoder), GPT-1 staked out the lineage that would eventually dominate generative AI. The decision looks obvious in hindsight; in 2018 it was a bet.

### 4. The Recipe Generalized
The same recipe — pretrain on unlabeled text + fine-tune on labels — would soon be applied at every scale, in every domain, with consistent results. GPT-1 was the proof of concept.

---

## Limitations

### 1. Small by Modern Standards
117M parameters is tiny. GPT-2 (1.5B), GPT-3 (175B), and GPT-4 (rumored to be much larger) showed that scaling this same architecture by orders of magnitude produced qualitatively new capabilities (#12: scaling laws).

### 2. BookCorpus Was Limited
Only 7,000 books — narrow style, limited domains. GPT-2 expanded to WebText (Reddit-linked pages), and GPT-3 used Common Crawl + books + Wikipedia, dramatically broadening the knowledge base.

### 3. Required Fine-Tuning
GPT-1 still needed labeled task data to perform well. The dream of "zero-shot" task performance — just describe the task in words and have the model do it — wouldn't arrive until GPT-2 / GPT-3 scaled the recipe far enough that **in-context learning** emerged.

### 4. Beaten Briefly by BERT
A few months after GPT-1, BERT (#3) came out with an **encoder-only** architecture and a **bidirectional masked language modeling** objective. BERT outperformed GPT-1 on most discriminative tasks (NLI, QA, classification), and BERT-style models dominated NLP benchmarks from 2018 through ~2020. GPT-1's revenge would come from sticking with decoder-only and scaling up.

---

## Impact and Legacy

### Direct Descendants
- **GPT-2** (Feb 2019, 1.5B parameters): Same architecture, much larger, trained on WebText. Showed surprisingly good zero-shot capabilities.
- **GPT-3** (#4, May 2020, 175B parameters): The breakthrough. Demonstrated that scaling enables **few-shot in-context learning** — solving tasks just from prompts, no fine-tuning required.
- **Codex / GitHub Copilot** (#56): GPT-3 fine-tuned on code.
- **InstructGPT and ChatGPT** (#5): GPT-3 + RLHF — turned the raw generator into a helpful assistant.
- **GPT-4** (#36, March 2023): Multimodal, much more capable, exact architecture details proprietary.
- **GPT-4o** (#40), **GPT-5** (#42), **o1** (#31): Successive generations, all built on the GPT-1 recipe.

### Inspired the Whole Decoder-Only LLM Family
- **LLaMA** (#15) and its successors (#17, #33, #41): Meta's open-weight decoder-only LLMs.
- **Claude** (#30, #43): Anthropic's decoder-only Transformer family.
- **Mistral (#72), Mixtral (#73), DeepSeek (#26, #27), Qwen (#28), Gemini**: All decoder-only descendants of the GPT-1 paradigm.

### Established the Pretrain-then-Adapt Paradigm
The "pretrain a big model, adapt to many tasks" pattern is now the default in NLP, computer vision (Vision Transformer #11 used the same idea for images), audio (Whisper), code, biology, and beyond.

### Made Generative Modeling Central
Before GPT-1, NLP was dominated by discriminative tasks. GPT-1 placed **generation** at the center of pretraining, which proved to be the right bet — generative objectives turned out to teach models more about language than discriminative ones.

---

## Connections to Other Papers

- **Attention Is All You Need (#1):** GPT-1 uses the decoder half of the original Transformer, with causal masking. Without the Transformer, no GPT.
- **BERT (#3):** The contemporary alternative — encoder-only, bidirectional masked LM. BERT initially outperformed GPT-1 on most benchmarks; the GPT lineage's revenge was at scale.
- **GPT-3 (#4):** The direct scaled-up successor that demonstrated few-shot in-context learning and made the GPT recipe famous.
- **InstructGPT / RLHF (#5):** Adds human-preference fine-tuning on top of GPT-style pretraining — the final ingredient that turned raw language models into useful assistants.
- **T5 (#68):** The encoder-decoder alternative that explicitly compared architectures and found encoder-decoder slightly better — but the decoder-only path GPT-1 chose ultimately won at extreme scale.
- **LLaMA (#15):** Open-weight decoder-only Transformer family, directly descended from the GPT-1 architectural lineage.
- **Word2Vec (#63):** The earlier transfer learning paradigm that GPT-1 superseded — instead of pretrained word vectors as the input layer only, GPT-1 pretrains the whole model.
- **Sequence to Sequence Learning (#64):** Provides the autoregressive generation pattern that GPT-1's decoder inherits.
- **ResNet (#66):** Provides the residual connections that make GPT-1's (and all later Transformers') deep stacks trainable.
- **Scaling Laws (#12):** Came after GPT-1 but formalized what the GPT-1 → GPT-2 → GPT-3 progression demonstrated empirically: bigger Transformers, more data, more compute = predictable capability gains.

---

## Key Takeaways

1. **The two-stage recipe became the standard:** Pretrain a Transformer on unlabeled text, fine-tune on labeled tasks. Every modern LLM workflow descends from this.
2. **Decoder-only Transformers can be general-purpose:** With clever input formatting, one autoregressive model handles classification, entailment, similarity, multiple choice — and eventually chat, code, and reasoning.
3. **Generative pretraining teaches deep knowledge:** Predicting the next token forces the model to learn syntax, semantics, facts, and reasoning — all without labels.
4. **A modest model in 2018 launched an entire industry:** 117M parameters trained on 7k books was the seed of GPT-3, ChatGPT, GPT-4, Claude, LLaMA, and the modern LLM era.
5. **The bet on scale would pay off enormously:** GPT-1 was the first step in the GPT-2 → GPT-3 → GPT-4 progression that defined the AI explosion of the 2020s.
