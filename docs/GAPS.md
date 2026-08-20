# Coverage & Gaps

What this collection covers, where it is thin, and which papers are queued next.

This page exists so the collection's boundaries are explicit. A curated list is only
trustworthy if it says what it left out. Entries here are candidates, not promises -
see [CONTRIBUTING.md](../CONTRIBUTING.md) if you want to write one.

**Last reviewed:** 2026-08-20 · **Papers at review time:** 107

---

## Coverage map

| Area | Papers | State |
|---|---|---|
| Transformer architecture & attention variants | 01, 11, 16, 54, 66, 73, 75 | **Strong** |
| Sequence-model alternatives (SSM, MoE, sparse) | 20, 37, 55, 67 | **Good** |
| Language model lineage (GPT, LLaMA, Claude, Gemini, DeepSeek, Qwen, Mistral) | 03, 04, 15, 17, 26-28, 30-31, 33, 36, 40-43, 47, 64-65, 93-95 | **Strong** |
| Alignment (RLHF, CAI, DPO, KTO, GRPO, RLVR) | 05, 14, 19, 38, 39, 63, 103 | **Strong** |
| Instruction tuning & synthetic data | 79, 80 | **Good** |
| Reasoning (CoT, ToT, PRM, STaR, test-time compute) | 09, 25, 34, 35, 50, 51, 77, 97-99 | **Strong** |
| Agents & tool use | 21, 24, 58, 59, 78, 100 | **Strong** |
| Diffusion & image generation | 02, 06, 07, 44, 48, 57, 69-72, 74, 89-92 | **Strong** |
| Self-supervised vision pretraining | 11, 88 | **Good** |
| Multimodal (vision-language, audio, video) | 08, 23, 29, 32, 40, 46, 47, 49 | **Good** |
| Retrieval | 13, 60, 87 | **Good** |
| Inference & serving efficiency | 16, 45, 52, 75, 86 | **Strong** |
| Training systems & parallelism | 76, 94 | **Thin** |
| Efficiency of fine-tuning | 10, 22 | **Good** |
| Scaling behaviour | 12, 18, 81 | **Good** |
| Interpretability | 82 | **Thin** |
| Safety beyond alignment training | 14, 83, 96 | **Thin** |
| Evaluation & benchmarks | 81, 84, 85 | **Thin** |
| Code generation & software engineering | 56, 84 | **Thin** |
| Reinforcement learning & world models | 63, 102, 104, 105 | **Adequate for scope** |
| Science applications (biology, mathematics, algorithms) | 61, 62, 68, 101, 106, 107 | **Good** |
| Deep learning prerequisites (pre-2015) | 53, 55, 57, 66, 73, 74 | **Deliberately partial** |

---

## Queued: high priority

Papers whose absence is most likely to leave a reader with a hole in their mental model.

### Interpretability & safety
- **In-context Learning and Induction Heads** (Olsson et al., 2022) - the mechanism behind in-context
  learning, and the clearest example of a capability traced to a specific circuit. Pairs directly with
  [Emergent Abilities](../papers/techniques/81-emergent-abilities/summary.md).
- **Universal and Transferable Adversarial Attacks on Aligned Language Models** (Zou et al., 2023) -
  the GCG jailbreak paper. The collection currently has no adversarial-robustness entry at all.
- **Weak-to-Strong Generalization** (Burns et al., 2023) - can a weaker supervisor align a stronger
  model? The core scalable-oversight question.
- **Red Teaming Language Models** (Perez et al., 2022) and **Constitutional Classifiers** - the
  practical defence side, to balance [Sleeper Agents](../papers/techniques/83-sleeper-agents/summary.md).

### Long context
- **Sliding-window and sparse attention** (Longformer, BigBird, StreamingLLM) - how attention was
  first made sub-quadratic.
- **Position interpolation and YaRN** - how a 4K model becomes a 128K model after training. Directly
  extends [RoPE](../papers/techniques/54-rope-rotary-position-embedding/summary.md).
- **Ring Attention / context parallelism** - the systems answer to million-token contexts.
- **Needle-in-a-haystack and RULER** - long-context evaluation, since advertised context length and
  usable context length differ substantially.

### Training systems
- **Mixed-precision and FP8 training** - the numerics that make large runs affordable.
- **Muon and modern optimizers** - the first serious challenge to Adam's dominance in LLM pretraining.
- **Data curation at scale** (The Pile, RefinedWeb, FineWeb, DCLM) - data quality is now a bigger
  lever than architecture, and the collection has nothing on it.

### Evaluation
- **MMLU, HELM, and BIG-Bench** - the benchmarks everyone quotes, and their known flaws.
- **Contamination studies** - why benchmark scores drift upward faster than capability.
- **ARC-AGI** - the benchmark frontier reasoning models are still measured against.

---

## Queued: medium priority

### Generative modelling beyond images
- **Consistency Models** (Song et al., 2023) - one-to-four step generation.
- **AudioLM / MusicLM / VALL-E** - the collection has [Whisper](../papers/multimodal/49-whisper/summary.md)
  for speech *recognition* and nothing for audio *generation*.
- **NeRF and 3D Gaussian Splatting** - 3D generative representation.
- **DALL-E 1 and DALL-E 2 (unCLIP)** - the missing middle of the text-to-image lineage; the collection
  jumps from [CLIP](../papers/multimodal/08-clip/summary.md) to
  [Imagen](../papers/image-generation/91-imagen/summary.md) and
  [DALL-E 3](../papers/image-generation/48-dalle3/summary.md).

### Techniques
- **Knowledge Distillation** (Hinton et al., 2015) - referenced repeatedly across these summaries,
  never explained on its own page.
- **Self-RAG, HyDE, and query rewriting** - the retrieval techniques practitioners reach for after
  [Dense Retrieval](../papers/techniques/87-dense-retrieval/summary.md).
- **Medusa / EAGLE** - the successors to
  [Speculative Decoding](../papers/techniques/45-speculative-decoding/summary.md).
- **Multi-head Latent Attention (DeepSeek-V2)** - the successor to
  [GQA](../papers/architectures/75-grouped-query-attention/summary.md).
- **Mixture-of-Depths and early-exit** - conditional compute in the depth dimension.

### Deep learning prerequisites
The collection covers roots selectively (Word2Vec, Seq2Seq, VAE, PPO, ResNet, U-Net). Candidates for
completing that layer: **AlexNet** (2012), **Adam** (2014), **Batch/Layer Normalization**,
**LSTM** (1997), and **DQN / AlphaGo** - the run-up to
[AlphaZero](../papers/techniques/102-alphazero/summary.md), which the collection now covers. Each is
foundational; each is one step further from generative AI, so they stay optional rather than assumed.

---

## Deliberately out of scope

- **Model cards and system cards** that contain no methodological contribution.
- **Incremental version bumps** where the previous entry already covers the technique.
- **Papers with no public write-up** - if there is nothing citable to link, there is nothing to summarize.
- **Application-domain surveys** (AI in medicine, law, finance) - a different collection.
- **Anything the summary would have to invent details about.** Every entry here is written from the
  published record; where a number is uncertain the summary says so rather than guessing.

---

## How to close a gap

1. Pick an entry above (or propose one).
2. Copy [`papers/_TEMPLATE.md`](../papers/_TEMPLATE.md) into `papers/<category>/<NN-slug>/summary.md`
   with the next free number.
3. Add aliases to `ALIASES` in `scripts/add_cross_links.py` and topics to `TOPICS` in
   `scripts/build_manifest.py`.
4. Run the regeneration pipeline documented in [CONTRIBUTING.md](../CONTRIBUTING.md) and commit the
   generated output.
5. Update the coverage table and remove the entry from the queue above.
