# Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer (T5)

**Authors:** Colin Raffel, Noam Shazeer, Adam Roberts, Katherine Lee, Sharan Narang, Michael Matena, Yanqi Zhou, Wei Li, Peter J. Liu (Google Research)

**Published:** October 2019 (JMLR 2020)

**Paper Link:** https://arxiv.org/abs/1910.10683

---

## Why This Paper Matters

T5 ("Text-to-Text Transfer Transformer") is one of the most thorough and influential papers in NLP. It did two things that mattered enormously: (1) reframed **every** NLP task — classification, translation, summarization, question answering, regression, you name it — as a single unified problem of mapping input text to output text; and (2) systematically explored the design space of pretraining objectives, model architectures, datasets, and scaling, producing a 50+ page paper of careful ablations.

The result was both a state-of-the-art model (at the time) and a kind of empirical map of transfer learning for NLP. T5's text-to-text framing influenced **instruction tuning**, **prompting**, and the design philosophy of subsequent models from InstructGPT to FLAN to GPT-4. The **C4 dataset** introduced by T5 became a standard pretraining corpus. And T5 itself, in its various sizes (Small, Base, Large, 3B, 11B), remains a widely-used encoder-decoder model for tasks where conditional generation matters.

---

## The Core Idea: Text-to-Text Everything

In 2019, transfer learning for NLP looked messy. BERT (#3) had shown that pretraining a Transformer encoder, then fine-tuning task-specific heads (a classification head for sentiment, a span head for QA, a sequence-tagging head for NER), was the right pattern. But each task required its own head, its own loss function, its own evaluation pipeline.

T5's proposal: **make every task look like a text-in, text-out problem.** Then a single model architecture and single training procedure handles everything.

### Examples

```
Input:  "translate English to German: That is good."
Output: "Das ist gut."

Input:  "summarize: studies have shown that owning a dog is good for you..."
Output: "owning a dog is good for you."

Input:  "cola sentence: The course is jumping well."
Output: "not acceptable"

Input:  "stsb sentence1: A plane is taking off. sentence2: An air plane is taking off."
Output: "5.0"     ← regression as a string!

Input:  "question: What is the capital of France? context: France's capital..."
Output: "Paris"
```

Every task — translation, sentiment, similarity (a regression problem!), question answering, summarization — uses the **same encoder-decoder model**, the **same cross-entropy loss**, and only the **input prefix** changes to tell the model what task to perform. This is a clear ancestor of modern **prompting** and **instruction tuning**.

---

## The Architecture: A Standard Encoder-Decoder Transformer

T5 keeps things deliberately close to the original Transformer (#1) — encoder + decoder, multi-head attention, feedforward layers — to make the experiments interpretable.

### Why Encoder-Decoder?

This is one of T5's important findings. The paper systematically compared three Transformer architectures:

1. **Encoder-decoder** (original Transformer): separate encoder reads the input, decoder generates output, with cross-attention between them. T5 uses this.
2. **Decoder-only** (GPT-style): one stack, autoregressive, input and output concatenated. Used by GPT-1 (#69), GPT-3 (#4), LLaMA (#15).
3. **Prefix-LM**: hybrid — bidirectional attention on the input prefix, causal attention on the output.

**T5's finding:** Encoder-decoder consistently performed best for transfer learning, when controlled for parameter count. (Note: the decoder-only paradigm later won out in practice for very large generative models, but for T5's size range and task mix, encoder-decoder was superior.)

### Architecture Details

- **Relative position embeddings** instead of absolute (or sinusoidal) — a small but important tweak for handling variable-length inputs.
- **Layer normalization** before each sub-layer (pre-norm), without a bias term.
- **No bias** in many layers — small simplification.
- Standard ReLU activations.
- Model sizes: Small (60M), Base (220M), Large (770M), 3B, 11B.

---

## The Pretraining Objective: Span Corruption

T5 also systematically explored pretraining objectives, eventually settling on **span corruption** (a variant of BERT's masked language modeling adapted for encoder-decoder):

### How It Works

1. Take a span of text from the corpus.
2. Randomly mask **contiguous spans** of tokens (not just single tokens), replacing each masked span with a unique sentinel token.
3. The model is trained to output the **masked spans**, separated by the sentinel tokens.

```
Original:    "Thank you for inviting me to your party last week."
Corrupted input: "Thank you <X> me to your party <Y> week."
Target output:   "<X> for inviting <Y> last <Z>"
```

This is a denoising autoencoder objective in text-to-text form. It works well because:
- It's a generative objective (suits encoder-decoder)
- Span masking is harder and more realistic than single-token masking
- It matches the input-output shape of downstream tasks

### Other Objectives Compared

The paper tested:
- BERT-style masked language modeling
- Standard causal LM (GPT-style)
- Prefix LM
- Deshuffling

Span corruption with average span length 3 won — modestly but consistently.

---

## The Dataset: Colossal Clean Crawled Corpus (C4)

To pretrain at scale, T5 built **C4**, derived from Common Crawl. Raw Common Crawl is enormous (petabytes) but also full of garbage. The C4 cleaning pipeline:

- Keep only English text (langdetect)
- Drop pages with offensive words from a blocklist
- Drop pages with "lorem ipsum" placeholder text
- Drop pages with curly braces (indicates code/HTML)
- Drop very short or repetitive text
- Deduplicate at line and paragraph level

The result: **~750 GB of clean English text** (about 156 billion tokens). C4 was released publicly and became a standard corpus, used in many subsequent models (including the open replication community's training of T5-derived models).

---

## The Systematic Study

T5's main contribution is its **rigor** — the paper is essentially a giant ablation study. The authors fixed a baseline (Base-sized model, span corruption, C4) and then varied one factor at a time:

### What They Tested
- **Model architecture:** encoder-decoder vs decoder-only vs prefix-LM
- **Pretraining objective:** span corruption vs MLM vs causal LM vs deshuffling
- **Pretraining dataset:** C4, Wikipedia, web text variants
- **Dataset size:** how much pretraining data is enough?
- **Pretraining length:** more steps vs more data
- **Multi-task vs pretrain-then-finetune**
- **Model size:** Small, Base, Large, 3B, 11B
- **Fine-tuning method:** full fine-tuning vs adapter modules vs gradual unfreezing

### What They Found
- **Encoder-decoder + span corruption + C4 + scale** is the winning combination.
- **Bigger is consistently better** (foreshadowing scaling laws (#12)).
- **More data is better than more pretraining steps** on the same data.
- **Multi-task pretraining hurt slightly** vs pretraining then fine-tuning — but the gap was small and disappeared at larger scales.
- **Cleaning the data really matters:** training on dirty Common Crawl was much worse than training on C4.

---

## Scaling to 11B Parameters

T5's flagship model, **T5-11B**, had 11 billion parameters — at the time, one of the largest publicly trained NLP models. It achieved **state-of-the-art on many GLUE and SuperGLUE tasks**, matched or beat humans on SuperGLUE, and produced high-quality summaries and translations.

The 11B model required 4 days of training on 1024 TPU v3 chips — an extraordinary compute budget for 2019.

### Key Benchmarks
- **SuperGLUE:** 88.9 average score (close to human 89.8)
- **CNN/DailyMail summarization:** state-of-the-art ROUGE
- **WMT translation:** competitive with specialized translation systems
- **SQuAD QA:** strong span extraction via text generation

---

## Why Encoder-Decoder vs Decoder-Only?

This is one of the most debated design questions in NLP. T5 chose encoder-decoder; GPT-3 and the entire LLM mainstream chose decoder-only. Why?

**Encoder-decoder strengths:**
- Bidirectional attention on the input → better understanding of context
- Cleaner separation of "what's given" from "what's generated"
- Empirically better for tasks with a clear input/output distinction (translation, summarization, QA)

**Decoder-only strengths:**
- Simpler architecture, easier to scale
- Naturally suited to open-ended generation (chat, completion)
- Easy in-context learning with few-shot examples
- More parameter-efficient at extreme scale (no separate encoder)

T5 found encoder-decoder won at its scale, but GPT-3 (#4) and the subsequent generation of LLMs showed that decoder-only models, scaled up enough, can match or beat encoder-decoder approaches while being more flexible for generation. Today, both paradigms coexist: decoder-only LLMs dominate chat (GPT-4, Claude, LLaMA), encoder-decoder models dominate dedicated translation/summarization/code-translation systems (T5, BART, Whisper).

---

## Impact and Legacy

### The Text-to-Text Paradigm Won
Even though most modern LLMs are decoder-only, the **text-to-text framing** of every task is now universal. When you prompt ChatGPT to "Translate this to French" or "Summarize this paragraph," you're using T5's framing — task instruction + input → output text. **Instruction tuning** (InstructGPT, FLAN, Alpaca) is essentially text-to-text taken to its logical conclusion.

### C4 as a Foundational Dataset
C4 has been used to pretrain countless subsequent models. It pioneered the careful cleaning and deduplication practices now considered standard for LLM pretraining corpora.

### Inspired Follow-Up Work
- **mT5**: multilingual T5, trained on 101 languages.
- **ByT5**: byte-level T5 (no tokenizer).
- **FLAN-T5**: T5 with instruction tuning, a major step toward useful instruction-following models.
- **UL2**: improved pretraining objective combining T5-style and GPT-style.
- **PaLM, GPT-4, Claude**: all benefit from the methodology of careful ablations T5 normalized.

### Empirical Rigor as a Norm
T5 set a new bar for thoroughness in machine learning papers. Its sprawling, careful experimental matrix influenced how subsequent foundation model papers (PaLM, LLaMA, Chinchilla scaling laws) approached their empirical claims.

---

## Limitations

### 1. Encoder-Decoder is Awkward for Chat
Conversational AI is fundamentally autoregressive — each turn builds on the last. Decoder-only models like GPT fit this shape more naturally. T5 works for chat but with extra plumbing.

### 2. 11B Parameters Was Big in 2019, Small in 2024
T5-11B is dwarfed by GPT-4, Claude, and LLaMA-405B. Its capability ceiling is much lower than modern frontier models.

### 3. Span Corruption Objective Isn't Universal
While span corruption is great for fine-tuning on structured tasks, it doesn't produce the same kind of fluid generative capability as causal language modeling at scale. This is one reason later models leaned decoder-only.

### 4. Fine-Tuning Required
T5 was designed around the pretrain-then-finetune paradigm. GPT-3 showed that with enough scale, you could skip fine-tuning entirely and use **few-shot prompting** — a workflow T5 doesn't really enable in the same way.

---

## Connections to Other Papers

- **Attention Is All You Need (#1):** T5 is a careful re-engineering of the original Transformer encoder-decoder, with modern tweaks (relative position embeddings, pre-norm, sentinel tokens).
- **BERT (#3):** T5 builds on BERT's masked-language-modeling insight, but generalizes it to span corruption and to a generative encoder-decoder setup, making fine-tuning more uniform across tasks.
- **GPT-3 (#4):** Represents the alternative path — decoder-only, in-context learning, no fine-tuning. T5 and GPT-3 together define the two main poles of modern LLM design.
- **GPT-1 (#69):** The decoder-only ancestor whose paradigm GPT-3 scaled up. T5 chose the other branch.
- **Sequence to Sequence Learning (#64):** T5 is the modern-Transformer-era successor to LSTM-based encoder-decoders, taking the same "text-in, text-out" philosophy to its logical extreme.
- **Bahdanau Attention (#65):** T5's decoder uses cross-attention to the encoder — a direct descendant of Bahdanau's attention mechanism.
- **Scaling Laws (#12) and Chinchilla (#18):** T5's systematic scaling experiments (Small to 11B) presaged the more rigorous scaling laws work that would come from OpenAI and DeepMind shortly after.
- **Imagen (#78):** Uses a frozen T5-XXL as its text encoder — direct downstream usage of T5 in modern image generation.

---

## Key Takeaways

1. **Text-to-text unifies NLP:** Reframing every task as input-text → output-text simplifies training, evaluation, and multi-task learning, and laid the groundwork for instruction tuning and prompting.
2. **Span corruption is a strong pretraining objective** for encoder-decoder models — better than BERT's single-token masking when generating text.
3. **C4 set the standard for cleaned, large-scale pretraining corpora** and is still in wide use.
4. **Systematic ablations matter:** T5's rigorous empirical study (architecture × objective × data × scale) clarified which design choices actually matter.
5. **Encoder-decoder is alive and well for translation, summarization, and structured generation**, even as decoder-only models dominate open-ended chat. Both paradigms trace back to the original Transformer.
