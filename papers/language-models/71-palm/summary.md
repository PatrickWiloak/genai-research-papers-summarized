# PaLM: Scaling Language Modeling with Pathways

**Authors:** Aakanksha Chowdhery, Sharan Narang, Jacob Devlin, et al. (Google Research) — 67 authors

**Published:** April 2022 (arXiv 2204.02311)

**Paper Link:** https://arxiv.org/abs/2204.02311

---

## Why This Paper Matters

PaLM (Pathways Language Model) was Google's answer to GPT-3, and at 540 billion parameters it became the largest dense Transformer ever trained at the time. But the parameter count wasn't the main story — PaLM's striking result was that scaling brought **qualitatively new reasoning ability**. On the BIG-bench evaluation suite (200+ tasks designed to probe hard, novel capabilities), PaLM-540B not only crushed prior LLMs but matched the average human rater on many tasks. It could explain jokes, perform multi-step logical inference, and solve grade-school math problems via chain-of-thought prompting.

PaLM also marked the debut of **Pathways**, Google's new ML training infrastructure, which made it possible to train a single dense model efficiently across two entire TPU v4 Pods (6,144 chips). The lessons learned here fed directly into Gemini and the modern Google DeepMind training stack.

---

## The Problem Before PaLM

By early 2022, large language models had clear strengths and frustrating weaknesses:

- **GPT-3 (175B):** Strong on language tasks, but reasoning was unreliable. Multi-step arithmetic, logic puzzles, and commonsense inference often failed
- **Gopher (280B), MT-NLG (530B):** Bigger, but the qualitative leap people hoped for hadn't appeared
- **Chinchilla (70B):** Showed many large models were undertrained, but didn't address the reasoning ceiling

The open questions:
1. Were the reasoning failures a fundamental limitation, or just a scale issue?
2. Could a single model handle hundreds of disparate hard tasks?
3. Could you actually train a half-trillion-parameter dense model efficiently across multiple data centers?

PaLM was Google's attempt to answer all three with a giant, well-engineered push.

---

## The Core Innovation: Dense Scaling Plus Pathways

PaLM's contribution had three pieces:

1. **A single 540B dense Transformer** — no mixture-of-experts shortcuts, no model sparsity
2. **The Pathways training system** — efficient training across two TPU v4 Pods (separate hardware islands)
3. **Demonstration that scale unlocks reasoning** — especially when combined with chain-of-thought prompting

The dense choice was deliberate. Sparse models (MoE) can have more parameters per dollar of compute, but PaLM's authors wanted a clean test of what pure scale buys in a standard architecture.

---

## How PaLM Works

### Architecture

PaLM is a decoder-only Transformer, but with several careful design choices:

- **SwiGLU activations** in the FFN (better than ReLU/GELU at scale)
- **Parallel layers:** attention and FFN computed in parallel rather than sequentially, reducing latency
- **Multi-Query Attention (MQA):** all heads share K and V projections, dramatically speeding up autoregressive inference
- **RoPE positional embeddings:** rotary embeddings instead of learned absolute positions
- **No biases** in dense and layer-norm layers (more stable training)
- **Shared input-output embeddings**
- **256K SentencePiece vocabulary** with byte-fallback for any Unicode

```python
# Parallel block (PaLM-style) vs sequential block (standard)

# Standard:
x = x + Attention(LayerNorm(x))
x = x + FFN(LayerNorm(x))

# PaLM parallel:
norm_x = LayerNorm(x)
x = x + Attention(norm_x) + FFN(norm_x)
```

This parallel formulation saved roughly 15% of training time at the 540B scale.

### Model Sizes

| Model | Params | Layers | Hidden | Heads | Heads x Dim |
|-------|--------|--------|--------|-------|-------------|
| PaLM 8B | 8.6B | 32 | 4096 | 16 | 16 x 256 |
| PaLM 62B | 62B | 64 | 8192 | 32 | 32 x 256 |
| **PaLM 540B** | **540B** | **118** | **18432** | **48** | **48 x 384** |

### Training Data

780 billion tokens from a mixture spanning:
- Multilingual web pages (filtered Common Crawl)
- English books
- GitHub code (24 languages)
- Multilingual Wikipedia
- News
- Social media conversations

Roughly 78% English, 22% multilingual + code. This data composition made PaLM unusually strong on translation and code despite not being specialized.

### Pathways Infrastructure

The training run is itself a contribution. Pathways enabled:
- **6,144 TPU v4 chips** organized as two pods (3072 each) connected over data center networks
- **Two-way data parallelism** between pods, model parallelism within
- **Hardware FLOPs utilization of 46.2%** — extraordinary for a model this size
- A single Python program orchestrating computation across two separate hardware islands

This was the first publicly described training run at the multi-pod scale.

---

## Key Results

### BIG-bench: The Headline Demonstration

BIG-bench is a 200+ task benchmark designed to probe hard, novel capabilities (logic puzzles, mathematical induction, code generation, joke understanding, semantic parsing, etc.).

- PaLM 540B substantially outperformed prior models on BIG-bench
- On a "hard subset" of 58 tasks where Gopher and Chinchilla underperformed humans, PaLM matched or beat average human raters on many
- Discontinuous jumps appeared between 62B and 540B — capabilities that simply didn't exist at smaller scales

This was strong evidence for **emergent capabilities**: abilities absent at small scale that appear suddenly when models cross some threshold.

### Reasoning + Chain-of-Thought

PaLM popularized the combination of large dense models with chain-of-thought (CoT) prompting:

```
Question: Roger has 5 tennis balls. He buys 2 more cans of tennis
balls. Each can has 3 tennis balls. How many tennis balls does he
have now?

Answer (CoT): Roger started with 5 balls. 2 cans of 3 balls each is
6 balls. 5 + 6 = 11. The answer is 11.
```

Results on math/reasoning benchmarks:
- **GSM8K (grade-school math):** 58% with 8-shot CoT (prior SOTA: 55%, using a fine-tuned model plus external calculator)
- **MATH:** Strong gains over prior LLMs
- **MMLU:** Competitive with the best fine-tuned systems
- Solved 65% of a held-out math word problem set zero-shot

### Code Understanding

Despite code being only ~5% of training data:
- Strong performance on HumanEval and MBPP (Python)
- Could explain, debug, and translate between programming languages
- Fine-tuning gave PaLM-Coder, which became a strong baseline for subsequent code models

### Multilingual Translation

Without any task-specific fine-tuning, PaLM matched or exceeded specialized translation systems for high-resource languages and beat prior LLMs on low-resource pairs.

### Explaining Jokes

PaLM produced explanations of original jokes — a task requiring genuine semantic understanding rather than memorization. Example:

> **Joke:** "I was supposed to start writing the paper at 5:00 PM. But then I started playing with this cool new language model for 10 minutes. 10 minutes later, it's suddenly 9:30 PM!"
>
> **PaLM explanation:** "The joke is that the speaker intended to start writing at 5 PM, but instead lost track of time playing with a language model, and when they checked the clock again it was 9:30 PM — meaning they wasted four and a half hours instead of just ten minutes."

This kind of inferential humor explanation was a qualitatively new capability.

---

## Discontinuous Improvements at Scale

Across many BIG-bench tasks, PaLM showed a striking pattern: performance was flat or near-random at 8B and 62B, then jumped dramatically at 540B. The paper documents this for tasks like:

- Logical deduction
- Modified arithmetic
- Code line description
- Conceptual combinations

These step-function jumps were among the strongest pieces of evidence for **emergent abilities** in LLMs — a concept that would be formalized in subsequent papers and later contested by reanalyses suggesting some "emergence" is an artifact of discontinuous metrics.

---

## Impact and Legacy

### Direct descendants
- **PaLM 2 (2023):** Smaller, better-trained, multilingual-focused; powered Bard
- **Med-PaLM, Med-PaLM 2:** Medical question-answering systems
- **PaLM-E:** Embodied PaLM for robotics (vision + language + action)
- **Minerva:** PaLM fine-tuned on scientific/mathematical text
- **Gemini (2023-2024):** Inherits the lessons of PaLM but moves to multimodal-first design and uses MoE

### Conceptual contributions
- Demonstrated that **dense scaling still pays dividends** even at hundreds of billions of parameters
- Provided the strongest evidence to date for **emergent capabilities**
- Made **chain-of-thought** a standard tool for unlocking reasoning in large LLMs
- Showed that **Multi-Query Attention** is practical at scale (later refined to Grouped-Query Attention in LLaMA 2 and Mistral)
- Validated **parallel block design** for efficiency

### Infrastructure contributions
- Pathways became Google's training backbone
- Public demonstration that multi-pod training works
- Influenced training systems at every major lab

PaLM was the high-water mark for pure dense scaling. After PaLM, the field largely shifted attention to better-trained smaller models (Chinchilla, LLaMA), mixture-of-experts (Mixtral, GPT-4 rumored), and multimodality (GPT-4V, Gemini). But the questions PaLM answered — does dense scaling keep working, do reasoning capabilities emerge, can you train across data centers — defined the trajectory.

---

## Limitations

- **Severely undertrained by Chinchilla standards:** 780B tokens for a 540B model is roughly 1.4 tokens per parameter, vs Chinchilla's recommended ~20. PaLM would have been even better with more training data
- **Astronomical inference cost:** 540B dense is impractical to serve; Multi-Query Attention helped but not enough
- **Single-modality:** Pure text, no vision
- **English-dominated:** Despite multilingual data, English still dominated
- **Closed weights:** Never released publicly

---

## Connections to Other Papers

- **Attention Is All You Need (#1):** PaLM is a 118-layer instance of this architecture with carefully tuned modifications
- **GPT-3 (#4):** Direct competitor and inspiration; PaLM showed that even larger dense models keep improving and unlock new capabilities
- **Scaling Laws (#12):** PaLM was sized using these laws (though pre-Chinchilla)
- **Chinchilla (#18):** Published the same month; showed PaLM was undertrained, sparking the shift in training recipes
- **LLaMA (#15) and LLaMA 2 (#17):** Adopted PaLM's SwiGLU, RoPE, and architectural cleanups; LLaMA 2 adopted Grouped-Query Attention, a refinement of PaLM's MQA
- **Mixture of Experts (#37) and Mixtral (#73):** The sparse alternative — gets PaLM-like quality at GPT-3.5-like compute
- **GPT-4 (#36) and Gemini:** Both build on lessons from PaLM about reasoning, scale, and infrastructure

---

## Key Takeaways

1. **Dense scaling still worked at 540B** — capabilities continued to improve, and some appeared discontinuously
2. **Reasoning is unlocked by scale plus prompting strategy** (chain-of-thought), not by scale alone
3. **Architectural details matter at scale:** SwiGLU, parallel blocks, MQA, RoPE, and no biases compounded into substantial efficiency gains
4. **Multi-pod training is feasible** — Pathways proved you can train across separate hardware islands at high utilization
5. **Per-token efficiency was suboptimal** — PaLM's later reassessment in the light of Chinchilla helped pivot the entire field toward training smaller models on more data
