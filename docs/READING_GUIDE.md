# Reading Guide - Historical Significance vs Current Relevance

Not every important paper is a paper you still need to read. This guide separates the two: how
much a paper *mattered* when it landed, and how much it *matters to your work today*. Where those
two diverge, it says which way to lean and what to read instead.

It is a prioritisation guide, not a catalogue. Every one of the 107 papers carries a relevance
badge in [BROWSE.md](../BROWSE.md), a one-line contribution in
[QUICK_REFERENCE.md](./QUICK_REFERENCE.md), and a full entry in [INDEX.md](../INDEX.md). This page
covers the essentials plus the cases where the two axes pull apart.

---

## 📊 Priority Matrix

**Historical** = how much it changed the field at the time. **Current** = how much you lose today
by not having read it.

### Foundations you cannot skip

| Paper | Year | Historical | Current | Priority | Read For |
|-------|------|-----------|---------|----------|----------|
| [Transformer](../papers/architectures/01-attention-is-all-you-need/summary.md) | 2017 | 🔥🔥🔥🔥🔥 | 🔥🔥🔥🔥🔥 | **CRITICAL** | The architecture under everything |
| [Scaling Laws](../papers/techniques/12-scaling-laws/summary.md) | 2020 | 🔥🔥🔥🔥🔥 | 🔥🔥🔥🔥🔥 | **CRITICAL** | Why compute budgets look like they do |
| [Chinchilla](../papers/techniques/18-chinchilla/summary.md) | 2022 | 🔥🔥🔥🔥 | 🔥🔥🔥🔥🔥 | **CRITICAL** | The data/parameter ratio everyone now uses |
| [InstructGPT](../papers/language-models/05-instructgpt-rlhf/summary.md) | 2022 | 🔥🔥🔥🔥🔥 | 🔥🔥🔥🔥🔥 | **CRITICAL** | How a base model becomes an assistant |
| [Chain-of-Thought](../papers/techniques/09-chain-of-thought/summary.md) | 2022 | 🔥🔥🔥🔥 | 🔥🔥🔥🔥🔥 | **CRITICAL** | The root of every reasoning method |
| [RAG](../papers/techniques/13-rag/summary.md) | 2020 | 🔥🔥🔥🔥 | 🔥🔥🔥🔥🔥 | **CRITICAL** | The most-deployed LLM pattern there is |
| [LoRA](../papers/techniques/10-lora/summary.md) | 2021 | 🔥🔥🔥🔥 | 🔥🔥🔥🔥🔥 | **CRITICAL** | How fine-tuning is actually done |
| [GPT-3](../papers/language-models/04-gpt3-few-shot-learners/summary.md) | 2020 | 🔥🔥🔥🔥🔥 | 🔥🔥🔥🔥 | **HIGH** | Where prompting replaced fine-tuning |

### The modern layer (2023-2026)

These did not exist when most "essential papers" lists were written, and they are where current
practice actually lives.

| Paper | Year | Historical | Current | Priority | Read For |
|-------|------|-----------|---------|----------|----------|
| [DeepSeek-R1](../papers/language-models/26-deepseek-r1/summary.md) | 2025 | 🔥🔥🔥🔥 | 🔥🔥🔥🔥🔥 | **CRITICAL** | Reasoning training, fully disclosed |
| [RLVR](../papers/techniques/39-rlvr/summary.md) | 2024-25 | 🔥🔥🔥🔥 | 🔥🔥🔥🔥🔥 | **CRITICAL** | The signal reasoning models train on |
| [GRPO](../papers/techniques/38-grpo/summary.md) | 2024 | 🔥🔥🔥 | 🔥🔥🔥🔥🔥 | **CRITICAL** | RLHF without the critic model |
| [Test-Time Compute](../papers/techniques/50-test-time-compute/summary.md) | 2024 | 🔥🔥🔥🔥 | 🔥🔥🔥🔥🔥 | **CRITICAL** | Why "think longer" beats "train bigger" |
| [Mixtral / MoE](../papers/architectures/37-mixture-of-experts/summary.md) | 2024 | 🔥🔥🔥🔥 | 🔥🔥🔥🔥🔥 | **CRITICAL** | The frontier architecture default |
| [DPO](../papers/language-models/19-dpo/summary.md) | 2023 | 🔥🔥🔥🔥 | 🔥🔥🔥🔥🔥 | **CRITICAL** | Alignment without an RL loop |
| [RoPE](../papers/techniques/54-rope-rotary-position-embedding/summary.md) | 2021 | 🔥🔥🔥 | 🔥🔥🔥🔥🔥 | **CRITICAL** | Positions, in nearly every LLM you use |
| [FlashAttention](../papers/techniques/16-flash-attention/summary.md) | 2022 | 🔥🔥🔥🔥 | 🔥🔥🔥🔥🔥 | **CRITICAL** | Why long context is affordable |
| [PagedAttention / vLLM](../papers/techniques/52-pagedattention-vllm/summary.md) | 2023 | 🔥🔥🔥 | 🔥🔥🔥🔥🔥 | **HIGH** | How models are actually served |
| [MCP](../papers/techniques/59-model-context-protocol/summary.md) | 2024 | 🔥🔥🔥 | 🔥🔥🔥🔥🔥 | **HIGH** | How tools get wired to models now |
| [ReAct](../papers/techniques/21-react/summary.md) | 2023 | 🔥🔥🔥🔥 | 🔥🔥🔥🔥 | **HIGH** | The agent loop, in its original form |
| [GQA](../papers/architectures/75-grouped-query-attention/summary.md) | 2023 | 🔥🔥🔥 | 🔥🔥🔥🔥 | **HIGH** | KV-cache cost in production |
| [GPTQ & AWQ](../papers/techniques/86-gptq-awq-quantization/summary.md) | 2022-23 | 🔥🔥🔥 | 🔥🔥🔥🔥 | **HIGH** | Why local inference is possible |
| [Dense Retrieval](../papers/techniques/87-dense-retrieval/summary.md) | 2019-20 | 🔥🔥🔥 | 🔥🔥🔥🔥 | **HIGH** | The layer under every RAG system |
| [SWE-bench](../papers/techniques/84-swe-bench/summary.md) | 2023 | 🔥🔥🔥 | 🔥🔥🔥🔥 | **HIGH** | What coding-agent claims mean |
| [Sparse Autoencoders](../papers/techniques/82-sparse-autoencoders/summary.md) | 2022-24 | 🔥🔥🔥🔥 | 🔥🔥🔥🔥 | **HIGH** | The state of interpretability |

### Where the two axes diverge

Read these for understanding, not for practice. The "instead" column is what to spend the time on.

| Paper | Year | Historical | Current | Priority | Instead |
|-------|------|-----------|---------|----------|---------|
| [BERT](../papers/language-models/03-bert/summary.md) | 2018 | 🔥🔥🔥🔥🔥 | 🔥🔥 | MEDIUM | Decoder-only models; [Dense Retrieval](../papers/techniques/87-dense-retrieval/summary.md) for embeddings |
| [GANs](../papers/image-generation/02-generative-adversarial-networks/summary.md) | 2014 | 🔥🔥🔥🔥🔥 | 🔥🔥 | MEDIUM | [Stable Diffusion](../papers/image-generation/07-stable-diffusion/summary.md), [Flow Matching](../papers/image-generation/72-flow-matching-sd3/summary.md) |
| [Word2Vec](../papers/techniques/53-word2vec/summary.md) | 2013 | 🔥🔥🔥🔥🔥 | 🔥🔥 | MEDIUM | [Dense Retrieval](../papers/techniques/87-dense-retrieval/summary.md) |
| [Seq2Seq](../papers/architectures/55-seq2seq/summary.md) | 2014 | 🔥🔥🔥🔥 | 🔥🔥 | MEDIUM | [Transformer](../papers/architectures/01-attention-is-all-you-need/summary.md) |
| [GPT-1](../papers/language-models/93-gpt1/summary.md) | 2018 | 🔥🔥🔥🔥 | 🔥🔥 | MEDIUM | [GPT-3](../papers/language-models/04-gpt3-few-shot-learners/summary.md) |
| [GPT-2](../papers/language-models/64-gpt2/summary.md) | 2019 | 🔥🔥🔥🔥 | 🔥🔥 | MEDIUM | [GPT-3](../papers/language-models/04-gpt3-few-shot-learners/summary.md) |
| [DDPM](../papers/image-generation/06-diffusion-models/summary.md) | 2020 | 🔥🔥🔥🔥🔥 | 🔥🔥🔥 | MEDIUM | Read if you need the maths, not to use diffusion |
| [PPO](../papers/techniques/63-ppo/summary.md) | 2017 | 🔥🔥🔥🔥🔥 | 🔥🔥🔥 | MEDIUM | [DPO](../papers/language-models/19-dpo/summary.md), [GRPO](../papers/techniques/38-grpo/summary.md) unless you implement RLHF |
| [PaLM](../papers/language-models/94-palm/summary.md) | 2022 | 🔥🔥🔥🔥 | 🔥🔥 | MEDIUM | MoE models; dense scaling stopped here |

---

## 🎯 What Changed, and Why

The three cases readers ask about most.

### BERT-style encoders

- **2018-2020:** encoders dominated NLP; fine-tune BERT for every task.
- **Today:** decoder-only models do understanding *and* generation, so the split disappeared.
- **Still used for:** embeddings and reranking - Sentence-BERT and ColBERT, covered in
  [Dense Retrieval](../papers/techniques/87-dense-retrieval/summary.md).
- **Read BERT if:** you work on retrieval, or maintain a pre-2021 NLP system.

### GANs

- **2014-2020:** the default for image synthesis.
- **Today:** diffusion and flow matching win on stability, controllability and quality.
- **Still used for:** real-time generation and super-resolution, where a single forward pass
  matters. The adversarial loss also survives inside
  [VQ-GAN](../papers/image-generation/90-vq-gan/summary.md), which is very much still in use.
- **Read GANs if:** you need the adversarial idea itself; otherwise read the diffusion line.

### Dense scaling

- **2020-2022:** make the dense model bigger - GPT-3 at 175B, PaLM at 540B.
- **Today:** sparse MoE gives more capacity per unit of inference cost, and
  [Chinchilla](../papers/techniques/18-chinchilla/summary.md) showed the big dense models were
  under-trained anyway. Frontier models activate a fraction of their parameters.
- **Read PaLM if:** you want the high-water mark of the dense era; otherwise read
  [Mixtral](../papers/architectures/37-mixture-of-experts/summary.md) and
  [DeepSeek-V3](../papers/language-models/27-deepseek-v3/summary.md).

---

## 📈 What's Increasingly Important

**Reasoning as a training target, not a prompt.** Chain-of-thought was a prompting trick in 2022.
By 2025 it is what the model is trained to do:
[STaR](../papers/techniques/97-star/summary.md) →
[Process Reward Models](../papers/techniques/51-process-reward-models/summary.md) →
[Test-Time Compute](../papers/techniques/50-test-time-compute/summary.md) →
[o1](../papers/language-models/31-openai-o1/summary.md) →
[RLVR](../papers/techniques/39-rlvr/summary.md) /
[DeepSeek-R1](../papers/language-models/26-deepseek-r1/summary.md).

**Alignment without the RL machinery.** [DPO](../papers/language-models/19-dpo/summary.md) removed
the reward model, [KTO](../papers/techniques/103-kto/summary.md) removed the need for paired data,
and [GRPO](../papers/techniques/38-grpo/summary.md) removed the critic. Each step made alignment
cheaper to run.

**Inference economics.** Serving cost now drives architecture:
[GQA](../papers/architectures/75-grouped-query-attention/summary.md),
[PagedAttention](../papers/techniques/52-pagedattention-vllm/summary.md),
[Speculative Decoding](../papers/techniques/45-speculative-decoding/summary.md),
[quantisation](../papers/techniques/86-gptq-awq-quantization/summary.md), and MoE all exist to make
the forward pass cheaper.

**Agents and tool use.** [ReAct](../papers/techniques/21-react/summary.md) →
[Reflexion](../papers/techniques/78-reflexion/summary.md) →
[Generative Agents](../papers/techniques/58-generative-agents/summary.md) →
[Voyager](../papers/techniques/100-voyager/summary.md), with
[MCP](../papers/techniques/59-model-context-protocol/summary.md) as the integration layer and
[SWE-bench](../papers/techniques/84-swe-bench/summary.md) as the scoreboard.

**Interpretability and safety with teeth.**
[Sparse Autoencoders](../papers/techniques/82-sparse-autoencoders/summary.md) made features
readable and steerable; [Sleeper Agents](../papers/techniques/83-sleeper-agents/summary.md) showed
safety fine-tuning can leave a backdoor intact.

**Everything outside language.** The same toolkit now does biology
([AlphaFold 2](../papers/techniques/68-alphafold/summary.md),
[AlphaFold 3](../papers/techniques/101-alphafold3/summary.md),
[ESM-2](../papers/techniques/106-esm/summary.md)), mathematics
([AlphaGeometry](../papers/techniques/61-alphageometry/summary.md)), algorithm design
([AlphaEvolve](../papers/techniques/62-alphaevolve/summary.md)) and world modelling
([Genie](../papers/techniques/104-genie/summary.md),
[DreamerV3](../papers/techniques/105-dreamerv3/summary.md)).

---

## 📉 What's Becoming Less Relevant

- **Encoder-only architectures** - outside embeddings and reranking.
- **GAN training tricks** - the failure modes they solved mostly don't arise in diffusion.
- **Separate vision and language stacks** - [ViT](../papers/architectures/11-vision-transformer/summary.md)
  and [CLIP](../papers/multimodal/08-clip/summary.md) collapsed them into one.
- **Prompt engineering as a discipline** - [DALL-E 3](../papers/image-generation/48-dalle3/summary.md)
  and instruction-tuned LLMs absorbed most of it.
- **Absolute positional embeddings** - [RoPE](../papers/techniques/54-rope-rotary-position-embedding/summary.md) won.
- **Dense-only scaling** - see above.

---

## 🎯 Minimum Viable Reading

**3 hours - you need to hold a conversation about modern AI**

1. [Transformer](../papers/architectures/01-attention-is-all-you-need/summary.md) - the architecture
2. [GPT-3](../papers/language-models/04-gpt3-few-shot-learners/summary.md) - in-context learning
3. [InstructGPT](../papers/language-models/05-instructgpt-rlhf/summary.md) - how it became an assistant
4. [Chain-of-Thought](../papers/techniques/09-chain-of-thought/summary.md) - how it reasons

**+3 hours - you are building on top of models**

5. [RAG](../papers/techniques/13-rag/summary.md) and [Dense Retrieval](../papers/techniques/87-dense-retrieval/summary.md) - grounding
6. [LoRA](../papers/techniques/10-lora/summary.md) - adaptation
7. [ReAct](../papers/techniques/21-react/summary.md) and [MCP](../papers/techniques/59-model-context-protocol/summary.md) - tools and agents

**+3 hours - you need to reason about cost and capability**

8. [Scaling Laws](../papers/techniques/12-scaling-laws/summary.md) and [Chinchilla](../papers/techniques/18-chinchilla/summary.md)
9. [Mixtral / MoE](../papers/architectures/37-mixture-of-experts/summary.md)
10. [Test-Time Compute](../papers/techniques/50-test-time-compute/summary.md) and [DeepSeek-R1](../papers/language-models/26-deepseek-r1/summary.md)

---

## 📚 Suggested Reading Orders

### Relevance-first (recommended)

Transformer → GPT-3 → InstructGPT → Chain-of-Thought → RAG → LoRA → Chinchilla → MoE →
DPO → Test-Time Compute → DeepSeek-R1 → RoPE → FlashAttention → vLLM → ReAct → MCP

Most useful material first; you can stop at any point and still have a coherent picture.

### Chronological (how the field actually moved)

Word2Vec → Seq2Seq → Bahdanau Attention → ResNet → U-Net → Transformer → GPT-1 → BERT →
GPT-2 → Scaling Laws → GPT-3 → ViT → DDPM → CLIP → RoPE → LoRA → Chinchilla → InstructGPT →
Chain-of-Thought → Stable Diffusion → LLaMA → DPO → Mixtral → o1 → DeepSeek-R1 → GPT-5

Slower start, but the causal chain is much clearer.

### Architecture-first

Transformer → RoPE → GQA → FlashAttention → MoE / Switch → Mamba → ViT → MAE → ResNet →
U-Net → VQ-VAE → VQ-GAN → DiT

### Track: image and video generation

VAE → GANs → DDPM → DDIM → Classifier-Free Guidance → Stable Diffusion → Imagen → DALL-E 3 →
ControlNet → DreamBooth → Flow Matching / SD3 → Sora / DiT → Genie

### Track: reasoning models

Chain-of-Thought → Self-Consistency → Tree of Thoughts → STaR → Process Reward Models →
Quiet-STaR → Test-Time Compute → o1 → GRPO → RLVR → DeepSeek-R1 → rStar-Math → Meta-CoT

### Track: production systems

RAG → Dense Retrieval → GraphRAG → LoRA → QLoRA → GPTQ & AWQ → GQA → FlashAttention →
PagedAttention → Speculative Decoding → ReAct → MCP → SWE-bench → LLM-as-a-Judge → Llama Guard

---

## 💡 Pro Tips

**Skimming a summary:** read "Why This Matters", then the results table, then "Key Takeaways".
Skip the derivations on a first pass.

**Going deep:** read the summary, then the original paper, then implement the core idea on a toy
problem. The gap between "I followed that" and "I can build it" is where the learning is.

**Building something:** go straight to the practical-applications and limitations sections. The
limitations are the more valuable half - they tell you what will break in production.

---

## ✅ Progress Tracker

**Essentials (15)**

- [ ] Transformer · [ ] Scaling Laws · [ ] Chinchilla · [ ] GPT-3 · [ ] InstructGPT
- [ ] Chain-of-Thought · [ ] RAG · [ ] LoRA · [ ] DPO · [ ] Mixtral / MoE
- [ ] RoPE · [ ] FlashAttention · [ ] Test-Time Compute · [ ] GRPO · [ ] DeepSeek-R1

**Production layer (10)**

- [ ] Dense Retrieval · [ ] GraphRAG · [ ] QLoRA · [ ] GPTQ & AWQ · [ ] GQA
- [ ] PagedAttention · [ ] Speculative Decoding · [ ] ReAct · [ ] MCP · [ ] SWE-bench

**Historical context (9)**

- [ ] Word2Vec · [ ] Seq2Seq · [ ] Bahdanau Attention · [ ] GPT-1 · [ ] GPT-2
- [ ] BERT · [ ] GANs · [ ] PPO · [ ] PaLM

Then pick a track above, or work through [INDEX.md](../INDEX.md) for all 107.

---

**Last updated:** 2026-08-20 · **Collection:** 107 papers · **See also:**
[BROWSE.md](../BROWSE.md) · [QUICK_REFERENCE.md](./QUICK_REFERENCE.md) · [ROADMAP.md](./ROADMAP.md)
