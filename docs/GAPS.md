# Coverage & Gaps

What this collection covers, where it is thin, and what is queued next - papers and explainers.

This page exists so the collection's boundaries are explicit. A curated list is only
trustworthy if it says what it left out. Entries here are candidates, not promises -
see [CONTRIBUTING.md](../CONTRIBUTING.md) if you want to write one.

**Last reviewed:** 2026-09-30 · **Papers at review time:** 144

On 2026-09-30 the collection broadened from research papers to "everything AI": landmark
essays joined as a paper category, robotics got its own category, and unnumbered
[explainers](../EXPLAINERS.md) now cover what no single paper does - model families,
benchmarks, compute and economics, policy, the ecosystem and the open questions.

---

## Coverage map

| Area | Papers | State |
|---|---|---|
| Transformer architecture & attention variants | 01, 11, 16, 54, 66, 73, 75, 141 | **Strong** |
| Sequence-model alternatives (SSM, MoE, sparse) | 20, 37, 55, 67, 131 | **Good** |
| Language model lineage (GPT, LLaMA, Claude, Gemini, DeepSeek, Qwen, Mistral, Phi) | 03, 04, 15, 17, 26-28, 30-31, 33, 36, 40-43, 47, 64-65, 93-95, 135 | **Strong** - plus a lineage explainer per family |
| Alignment (RLHF, CAI, DPO, KTO, GRPO, RLVR) | 05, 14, 19, 38, 39, 63, 103 | **Strong** |
| Instruction tuning & synthetic data | 79, 80, 140 | **Good** |
| Reasoning (CoT, ToT, PRM, STaR, test-time compute) | 09, 25, 34, 35, 50, 51, 77, 97-99 | **Strong** |
| Agents & tool use | 21, 24, 58, 59, 78, 100, 115 | **Strong** |
| Diffusion & image generation | 02, 06, 07, 44, 48, 57, 69-72, 74, 89-92, 119, 120 | **Strong** |
| 3D generation | 117, 118 | **Good** |
| Self-supervised vision pretraining | 11, 88 | **Good** |
| Multimodal (vision-language, speech, video) | 08, 23, 29, 32, 40, 46, 47, 49 | **Good** |
| Audio generation | 121, 122 | **Good** |
| Robotics & embodied AI | 123-125 | **Good** |
| Retrieval | 13, 60, 87 | **Good** |
| Long context | 130, 132 | **Good** |
| Inference & serving efficiency | 16, 45, 52, 86 | **Strong** |
| Training systems, numerics & optimizers | 76, 94, 142-144 | **Good** |
| Data curation & tokenization | 133, 136 | **Good** |
| Efficiency of fine-tuning & distillation | 10, 22, 134 | **Good** |
| Scaling behaviour | 12, 18, 81 | **Good** |
| Interpretability | 82, 126 | **Adequate** |
| Safety & adversarial robustness | 83, 96, 127-129 | **Good** |
| Evaluation & benchmarks | 84, 85, 137-139 | **Good** - plus five benchmark explainers |
| Code generation & software engineering | 56 | **Thin** |
| Reinforcement learning & world models | 102, 104, 105 | **Adequate for scope** |
| Science applications (biology, mathematics, algorithms) | 61, 62, 68, 101, 106, 107 | **Good** |
| Deep learning prerequisites (pre-2015) | 53, 55, 57, 66, 73, 74 | **Deliberately partial** |
| Essays & landmark posts | 108-116 | **Good** |

Explainers are not numbered and so are not in this map; [EXPLAINERS.md](../EXPLAINERS.md)
lists them all with their review dates.

---

## Queued: high priority

Papers whose absence is most likely to leave a reader with a hole in their mental model.

### Code and software engineering
- **SWE-agent** (Yang et al., 2024) - the agent-computer interface paper, the other half of
  [SWE-bench](../papers/techniques/84-swe-bench/summary.md).
- **AlphaCode** (Li et al., 2022) - competition-level code generation by sampling and filtering at scale.

### Interpretability & safety
- **Toy Models of Superposition** (Elhage et al., 2022) - why features share neurons; the setup
  that [Sparse Autoencoders](../papers/techniques/82-sparse-autoencoders/summary.md) answer.
- **Alignment Faking in Large Language Models** (Greenblatt et al., 2024) - pairs with
  [Sleeper Agents](../papers/techniques/83-sleeper-agents/summary.md).
- **Constitutional Classifiers** (Anthropic, 2025) - the production defence against universal jailbreaks.

### Long context & efficiency
- **Ring Attention / context parallelism** - the systems answer to million-token contexts.
- **Medusa / EAGLE** - the successors to
  [Speculative Decoding](../papers/techniques/45-speculative-decoding/summary.md).

### Evaluation
- **HELM** (Liang et al., 2022) and **BIG-Bench** (Srivastava et al., 2022) - holistic evaluation
  and the crowd-sourced benchmark suite.
- **GPQA** and **Humanity's Last Exam** - covered in the
  [knowledge benchmarks explainer](../explainers/benchmarks/knowledge-and-reasoning.md), not yet
  summarised on their own.

---

## Queued: medium priority

### Generative modelling beyond images
- **MusicLM / Stable Audio** - music generation, next to [AudioLM](../papers/multimodal/121-audiolm/summary.md).
- **Veo, Movie Gen and open video models** - video generation after
  [Sora](../papers/image-generation/44-sora-dit/summary.md).

### Techniques
- **Self-RAG, HyDE, and query rewriting** - the retrieval techniques practitioners reach for after
  [Dense Retrieval](../papers/techniques/87-dense-retrieval/summary.md).
- **Mixture-of-Depths and early-exit** - conditional compute in the depth dimension.
- **DPO variants** (IPO, SimPO, ORPO) - the preference-optimisation family after
  [DPO](../papers/language-models/19-dpo/summary.md).

### Robotics
- **Diffusion Policy** (Chi et al., 2023) and **Gemini Robotics** (2025) - the other major lines
  beside [RT-2](../papers/robotics/123-rt2/summary.md) and [pi0](../papers/robotics/124-pi0/summary.md).

### Deep learning prerequisites
The collection covers roots selectively (Word2Vec, Seq2Seq, VAE, PPO, ResNet, U-Net, Adam). Candidates
for completing that layer: **AlexNet** (2012), **Batch/Layer Normalization**, **Dropout**,
**LSTM** (1997), and **DQN / AlphaGo** - the run-up to
[AlphaZero](../papers/techniques/102-alphazero/summary.md). Each is foundational; each is one step
further from generative AI, so they stay optional rather than assumed.

---

## Queued: explainers

- **Model families:** xAI Grok, Microsoft Phi (beyond [phi-1](../papers/language-models/135-phi-1-textbooks/summary.md)),
  Moonshot Kimi.
- **Concepts:** mixture-of-experts routing in practice, multimodal tokenization (how images and
  audio become tokens), embeddings beyond retrieval.
- **Applications:** AI for code (the coding-assistant landscape), AI in science, AI search.
- **Safety:** jailbreaks and prompt injection as a threat class (linking the sibling repo's
  [AI threat modelling](https://github.com/PatrickWiloak/cloud-data-ai-security-zero-to-hero/blob/main/learn/concepts/ai-threat-modeling.md)).
- **Society:** AI and jobs, copyright and training data litigation, energy use.

---

## Deliberately out of scope

- **Model cards and system cards** that contain no methodological contribution - their facts go
  into the [model-family explainers](../EXPLAINERS.md) instead.
- **Incremental version bumps** where the previous entry already covers the technique.
- **Papers with no public write-up** - if there is nothing citable to link, there is nothing to summarize.
- **Hands-on build guides and certification prep** - that is the sibling
  [Zero to Hero](https://github.com/PatrickWiloak/cloud-data-ai-security-zero-to-hero) repo.
- **Anything the summary would have to invent details about.** Every entry here is written from the
  published record; where a number is uncertain the summary says so rather than guessing.

---

## How to close a gap

1. Pick an entry above (or propose one).
2. For a paper, follow "Adding a new paper summary" in [CONTRIBUTING.md](../CONTRIBUTING.md); for an
   explainer, follow "Adding an explainer" there.
3. Update the coverage table (papers only) and remove the entry from the queue above.
