# Paper Comparisons and Analysis

Detailed side-by-side comparisons of related papers to understand trade-offs and evolution.

---

## Table of Contents
1. [Architecture Comparisons](#architecture-comparisons)
2. [Training Approaches](#training-approaches)
3. [Alignment Methods](#alignment-methods)
4. [Reasoning Methods](#reasoning-methods)
5. [Image Generation](#image-generation)
6. [Efficiency Techniques](#efficiency-techniques)
7. [Inference and Serving](#inference-and-serving)
8. [Retrieval and Knowledge](#retrieval-and-knowledge)
9. [Agents and Tool Use](#agents-and-tool-use)
10. [Evaluation](#evaluation)
11. [Beyond Language](#beyond-language)
12. [Evolution Over Time](#evolution-over-time)
13. [Technique Combinations](#technique-combinations)
14. [When to Use Which Paper's Techniques](#when-to-use-which-papers-techniques)

---

## Architecture Comparisons

### Transformer Variants: Encoder vs Decoder vs Encoder-Decoder

| Aspect | BERT (Encoder) | GPT-3 (Decoder) | T5 (Enc-Dec) |
|--------|----------------|-----------------|--------------|
| **Architecture** | Bidirectional encoder | Unidirectional decoder | Full encoder-decoder |
| **Attention** | Bidirectional | Causal (left-to-right) | Encoder: bi, Decoder: causal |
| **Training** | Masked language modeling | Next token prediction | Span corruption |
| **Best For** | Understanding, classification | Generation, completion | Translation, summarization |
| **Context** | Can see full input | Only sees left context | Best of both |
| **Parameters** | 110M-340M | 175B (GPT-3) | 11B (T5-XXL) |
| **Use Cases** | Search, NER, QA | Chatbots, code, creative | Translation, QA with generation |
| **Strength** | Deep understanding | Fluent generation | Flexible seq-to-seq |
| **Weakness** | Can't generate long text | No bidirectional context | More complex |

**When to use:**
- **BERT**: Classification, entity recognition, semantic search
- **GPT**: Text generation, chatbots, creative writing
- **T5**: Translation, summarization, structured generation

---

### Seq2Seq Progression: RNN to Attention to Transformer

The encoder-decoder idea predates the Transformer by several years. Understanding the lineage clarifies why each component was invented.

| Aspect | Seq2Seq / LSTM (2014) | + Bahdanau Attention (2014) | Transformer (2017) |
|--------|-----------------------|-----------------------------|--------------------|
| **Paper** | [Sutskever et al.](../papers/architectures/55-seq2seq/summary.md) | [Bahdanau et al.](../papers/architectures/66-bahdanau-attention/summary.md) | Vaswani et al. |
| **Encoder** | LSTM (sequential) | LSTM (sequential) | Self-attention (parallel) |
| **Decoder** | LSTM + fixed context vector | LSTM + dynamic context | Self-attention + cross-attention |
| **Bottleneck** | Single fixed vector | None - all encoder states used | None |
| **Long-range deps** | Poor (vanishing gradients) | Better | Excellent (O(1) path length) |
| **Parallelism** | Sequential only | Sequential only | Fully parallel |
| **Alignment** | Implicit / none | Explicit soft alignment | Multi-head attention |
| **Translation quality** | Baseline | +3-4 BLEU | State-of-the-art |

**The key insight chain:**
1. Seq2Seq: "Compress input to a vector, then decode" - works but bottleneck hurts long sentences
2. Bahdanau attention: "Let the decoder look back at every encoder state, weighted by relevance" - solved the bottleneck
3. Transformer: "What if attention is the whole model?" - removed RNNs entirely, enabling parallelism

---

### Vision: CNN vs Transformer

| Aspect | ResNet (CNN) | Vision Transformer (ViT) |
|--------|--------------|--------------------------|
| **Architecture** | Convolutional layers | Pure Transformer (self-attention) |
| **Inductive Bias** | Strong (locality, translation) | Minimal |
| **Receptive Field** | Grows with depth | Global from layer 1 |
| **Data Requirement** | Lower (works on ImageNet-1k) | Higher (needs ImageNet-21k+) |
| **Compute (Training)** | Lower | Higher |
| **Compute (Inference)** | Lower for small images | Quadratic with image size |
| **Scalability** | Plateaus with more data | Improves with more data |
| **Multimodal** | Hard to combine with text | Natural integration |
| **Performance (small data)** | Better | Worse |
| **Performance (large data)** | Good | Better |
| **Transfer Learning** | Good | Excellent |

**Key Insight:** ViT needs more data but scales better. For production with limited data, CNNs still competitive. For multimodal or large-scale, ViT wins.

---

## Training Approaches

### Pre-training Paradigms

| Approach | Papers | Method | Pros | Cons |
|----------|--------|--------|------|------|
| **Masked Modeling** | BERT | Mask tokens, predict them | Bidirectional context | Can't generate naturally |
| **Autoregressive** | GPT-2, GPT-3, LLaMA | Predict next token | Natural generation | Only sees left context |
| **Contrastive** | CLIP | Match positive pairs | Learn alignments | Needs paired data |
| **Denoising** | Diffusion, DDPM | Remove noise iteratively | High quality | Slow generation |
| **Adversarial** | GANs | Generator vs discriminator | Fast generation | Training instability |
| **Text-to-Text** | [T5](../papers/language-models/65-t5/summary.md) | Unify all tasks as seq-to-seq | One model for all tasks | More complex fine-tuning |

---

### GPT Scaling Progression

| Aspect | GPT-2 (2019) | GPT-3 (2020) | GPT-3.5 (2022) | GPT-4 (2023) |
|--------|--------------|--------------|-----------------|--------------|
| **Paper** | [Radford et al.](../papers/language-models/64-gpt2/summary.md) | Brown et al. | - | - |
| **Params** | 1.5B | 175B | ~175B (est.) | Unknown |
| **Training Tokens** | ~40B | ~300B | ~300B + RLHF | Unknown |
| **Key Innovation** | Zero-shot tasks | Few-shot learning | RLHF alignment | Multimodal + reasoning |
| **Release** | Staged (safety concerns) | API only | ChatGPT base | API + ChatGPT |
| **MMLU** | - | 43.9% | ~70% | 86.4% |
| **GSM8k** | - | 17% | 57% | 92% |

**The GPT-2 moment:** OpenAI staged the release because they feared misuse - the first time a language model was considered too capable to release freely. GPT-2 was the proof that scale alone enables emergent zero-shot behavior.

---

### Scaling: GPT-3 vs LLaMA

| Aspect | GPT-3 (2020) | LLaMA (2023) |
|--------|--------------|--------------|
| **Largest Model** | 175B params | 65B params |
| **Training Tokens** | ~300B | 1.4T (4.7× more) |
| **Training Approach** | Scale parameters | Scale tokens (Chinchilla-optimal) |
| **Data** | Proprietary + public | Public only |
| **Accessibility** | API only | Open weights |
| **Compute Efficiency** | Lower | Higher (4× better) |
| **Performance** | Strong | LLaMA-65B > GPT-3 |
| **Fine-tuning** | Not available | Fully supported |
| **Inference Cost** | High (175B) | Lower (13B-65B) |
| **Key Innovation** | Few-shot learning | Compute-optimal training |

**What Changed:** Scaling laws (paper #12) showed GPT-3 was undertrained. LLaMA applied this learning to train smaller models longer, proving 13B with proper training matches 175B.

**Practical Impact:**
- **GPT-3 era**: "Bigger is better"
- **LLaMA era**: "Better trained is better"
- Democratized LLMs for researchers and startups

---

### Compute Efficiency: Original vs Chinchilla-Optimal

| Training Budget | Original Approach | Chinchilla-Optimal | Improvement |
|----------------|-------------------|--------------------| ------------|
| **1× compute** | 1B params, 20B tokens | 400M params, 8B tokens | More data > more params |
| **10× compute** | 10B params, 200B tokens | 2B params, 200B tokens | Same tokens, smaller model |
| **100× compute** | 100B params, 300B tokens | 10B params, 2T tokens | Way more tokens |

**Models following Chinchilla:**
- ✅ LLaMA - 1.4T tokens for 65B params
- ✅ Chinchilla - 1.4T tokens for 70B params
- ❌ GPT-3 - 300B tokens for 175B params (undertrained)
- ❌ PaLM - 780B tokens for 540B params (undertrained)

---

## Alignment Methods

### RLHF vs Constitutional AI vs DPO vs KTO vs GRPO vs RLVR

| Aspect | RLHF / InstructGPT | Constitutional AI | [DPO](../papers/language-models/19-dpo/summary.md) | [KTO](../papers/techniques/103-kto/summary.md) | [GRPO](../papers/techniques/38-grpo/summary.md) | [RLVR](../papers/techniques/39-rlvr/summary.md) |
|--------|--------------------|-------------------|-----|-----|------|------|
| **RL optimizer** | [PPO](../papers/techniques/63-ppo/summary.md) | PPO | None (direct) | None (direct) | Group-relative policy opt. | Any (usually GRPO) |
| **Critic / value model** | Yes | Yes | No | No | No | No |
| **Reward model** | Separate RM trained | AI-based RM | Implicit (closed-form) | Implicit | Relative group scores | None - a verifier |
| **Data needed** | 10,000+ human comparisons | ~100 written principles | Preference *pairs* | Unpaired 👍/👎 | Sampled groups + scores | Problems with checkable answers |
| **Signal** | Human preference | AI self-critique | Human/AI preference | Binary desirability | Outcome correctness | Verified correctness |
| **Transparency** | Opaque | Transparent (principles) | Moderate | Moderate | High | Highest - the rule is the reward |
| **Relative cost** | Highest | Low | Low | Lowest labelling cost | Low | Low, where a verifier exists |
| **Stability** | Moderate (PPO-sensitive) | Moderate | High | High | High | High |
| **Best for** | General helpfulness | Harmlessness | Efficient preference learning | When pairs are impractical | Maths and reasoning | Anything auto-checkable |
| **Limitation** | Expensive, reward hacking | Principles must be written | Needs pairs | Coarser signal | Needs sampled groups | No verifier, no method |
| **Used by** | ChatGPT, InstructGPT | Claude | Many open models | Open fine-tunes | DeepSeek-R1 | o1, R1, reasoning models |

**PPO's role:** PPO ([Schulman et al. 2017](../papers/techniques/63-ppo/summary.md)) is the RL backbone of classic RLHF. It clips policy updates to prevent catastrophic reward hacking - the "proximal" constraint keeps the fine-tuned model close to the base model. DPO and GRPO emerged partly to sidestep PPO's complexity and hyperparameter sensitivity.

**Stage-by-Stage (RLHF vs Constitutional AI):**

| Stage | RLHF | Constitutional AI |
|-------|------|-------------------|
| **1. Initial Data** | Humans write demonstrations | AI generates + self-critiques |
| **2. Preference Data** | Humans compare outputs | AI compares via principles |
| **3. Reward Model** | Train on human preferences | Train on AI preferences |
| **4. RL** | PPO with human reward model | PPO with AI reward model |

**Hybrid Approach (Best Practice):**
- Use RLHF for helpfulness (harder to specify as principles)
- Use Constitutional AI for harmlessness (easier to write rules)
- Combine both signals in final model

**Real-World:**
- **ChatGPT**: Primarily RLHF
- **Claude**: Constitutional AI + some RLHF
- **Trend**: Moving toward more Constitutional AI for scalability

---

## Reasoning Methods

### Prompt-time vs Train-time Reasoning

The field moved from *asking* a model to reason to *training* it to. These are the stops along the way.

| Method | Year | Where the work happens | Extra cost | Needs labels? | Key idea |
|--------|------|------------------------|-----------|---------------|----------|
| [Chain-of-Thought](../papers/techniques/09-chain-of-thought/summary.md) | 2022 | Prompt | ~1 longer generation | No | Ask for intermediate steps |
| [Self-Consistency](../papers/techniques/77-self-consistency/summary.md) | 2022 | Sampling | N generations | No | Majority vote over N chains |
| [Tree of Thoughts](../papers/techniques/25-tree-of-thoughts/summary.md) | 2023 | Search | Many generations + search | No | Branch, evaluate, backtrack |
| [Self-Refine](../papers/techniques/99-self-refine/summary.md) | 2023 | Iteration | 2-3x generations | No | Model critiques and rewrites itself |
| [STaR](../papers/techniques/97-star/summary.md) | 2022 | Training | Fine-tuning rounds | Answers only | Train on rationales that got it right |
| [Process Reward Models](../papers/techniques/51-process-reward-models/summary.md) | 2023 | Training + search | PRM training + scoring | Step-level labels | Score each step, not the answer |
| [Quiet-STaR](../papers/techniques/98-quiet-star/summary.md) | 2024 | Training | Expensive pretraining | No | Latent rationale before every token |
| [Test-Time Compute](../papers/techniques/50-test-time-compute/summary.md) | 2024 | Inference budget | Tunable | No | Spend inference compute optimally |
| [RLVR](../papers/techniques/39-rlvr/summary.md) / [GRPO](../papers/techniques/38-grpo/summary.md) | 2024-25 | Training | RL run | Verifiable answers | Reward correctness, let reasoning emerge |

**How to choose**

- **You have an API and a problem today** → Chain-of-Thought, then Self-Consistency if accuracy
  matters more than cost. This is the entire budget-conscious answer.
- **Answers are checkable and you can train** → RLVR with GRPO. This is what produced
  [DeepSeek-R1](../papers/language-models/26-deepseek-r1/summary.md).
- **Answers are checkable but you cannot afford RL** → STaR: sample, filter by correctness,
  fine-tune, repeat.
- **The failure mode is a wrong intermediate step** → Process Reward Models. Outcome supervision
  rewards lucky guesses; process supervision does not.
- **The problem needs exploration, not a single chain** → Tree of Thoughts, or
  [rStar-Math](../papers/techniques/35-rstar-math/summary.md) for the MCTS version.

**The cost trap:** Self-Consistency at N=40 costs 40x a single answer. A reasoning model trained
with RLVR gets similar or better accuracy in one (longer) generation. Above a certain query volume,
training is cheaper than sampling.

### Outcome vs Process Supervision

| | Outcome supervision | Process supervision |
|---|---|---|
| **Labels** | Final answer only | Every reasoning step |
| **Cost** | Cheap, often automatic | Expensive - [PRM800K](../papers/techniques/51-process-reward-models/summary.md) took 800k human labels |
| **Failure mode** | Rewards right answers reached by bad reasoning | Needs a labelled domain |
| **Best for** | Any verifiable domain, at scale | High-stakes reasoning where the chain must be sound |
| **Used by** | [RLVR](../papers/techniques/39-rlvr/summary.md), [GRPO](../papers/techniques/38-grpo/summary.md) | [o1](../papers/language-models/31-openai-o1/summary.md)-style verification |

---

## Image Generation

### VAE vs GAN vs Diffusion vs Latent Diffusion

| Aspect | [VAE](../papers/image-generation/57-vae/summary.md) (2013) | GANs (2014) | Diffusion / DDPM (2020) | Stable Diffusion (2022) |
|--------|------------------------------------------------------------|-------------|--------------------------|-------------------------|
| **Training** | Stable (ELBO) | Adversarial (unstable) | Stable denoising | Stable (in latent space) |
| **Generation Speed** | Fast (1 pass) | Fast (1 pass) | Slow (50-1000 steps) | Medium (50 steps, smaller) |
| **Quality** | Moderate (blurry) | Good (sharp) | Excellent | Excellent |
| **Diversity** | High | Lower (mode collapse) | Higher | Higher |
| **Latent Space** | Structured, interpolatable | Unstructured | None (pixel-space noise) | Structured (VAE encoder) |
| **Control** | Smooth interpolation | Harder | Moderate | Easy (text conditioning) |
| **Memory** | Low | Moderate | High (pixel space) | Lower (latent space) |
| **Compute (Inference)** | Low | Low | Very high | Medium |
| **Text-to-Image** | Hard | Hard | Moderate | Native support |
| **Key Weakness** | Blurry outputs | Training instability | Very slow | Still needs many steps |

**The generative model lineage:**
1. **VAE (2013)**: First principled probabilistic generative model - structured latent space but blurry outputs due to pixel-wise reconstruction loss
2. **GANs (2014)**: Sharp images via adversarial training, but mode collapse and instability
3. **DDPM (2020)**: Beat both on quality and diversity, sacrificed speed
4. **Stable Diffusion (2022)**: Runs diffusion in VAE's latent space - borrows VAE's compression to make diffusion tractable

**VAE's lasting contribution:** Even though VAEs lost the image quality race to GANs and diffusion, the VAE encoder/decoder is the latent space backbone of Stable Diffusion.

**Use Cases:**
- **VAE**: Structured generation, interpolation, anomaly detection, latent representations
- **GANs**: Real-time applications, style transfer (when speed matters)
- **DDPM**: Research, highest quality needs
- **Stable Diffusion**: Production text-to-image, balance of quality and speed

---

### Image Generation: Computational Cost Comparison

| Method | Training Cost | Inference Time (1 image) | Memory (Inference) |
|--------|---------------|--------------------------|-------------------|
| **StyleGAN** | ~1 week (8× V100) | ~0.1s | ~4GB |
| **DDPM** | ~2 weeks (8× V100) | ~30s (1000 steps) | ~16GB |
| **Stable Diffusion** | ~1 week (256× A100) | ~3s (50 steps) | ~8GB |
| **DALL-E 2** | Unknown (massive) | ~10s | Unknown (API only) |

**Speedup techniques:**
- DDIM sampling: 50× faster than DDPM
- Latent space: 8-10× compression
- Distillation: Train student to match in fewer steps

---

### Diffusion Samplers: DDPM vs DDIM vs Flow Matching

The generator quality is set by training; the *speed* is set by how you sample from it.

| | [DDPM](../papers/image-generation/06-diffusion-models/summary.md) | [DDIM](../papers/image-generation/70-ddim/summary.md) | [Flow Matching / SD3](../papers/image-generation/72-flow-matching-sd3/summary.md) |
|---|---|---|---|
| **Year** | 2020 | 2020 | 2022-2024 |
| **Typical steps** | 1,000 | 20-50 | 20-30, and straighter |
| **Deterministic?** | No (stochastic) | Yes | Yes |
| **Retraining needed** | - | None - resamples a trained DDPM | Yes, different objective |
| **Same seed → same image** | No | Yes | Yes |
| **Enables** | The theory | Image editing, inversion, interpolation | SD3, Flux |
| **Read it for** | Understanding the maths | Understanding why sampling got fast | Understanding current models |

**Practical read:** if you use diffusion, you are almost certainly using a DDIM-style sampler on a
latent model. DDPM is the theory; DDIM is what made it usable; flow matching is what current
frontier image models train with.

### Conditioning and Control

Four different problems that get confused with one another.

| Technique | Controls | Needs training? | Data needed | Use when |
|-----------|----------|-----------------|-------------|----------|
| [Classifier-Free Guidance](../papers/image-generation/69-classifier-free-guidance/summary.md) | How hard the model follows the prompt | No (a sampling knob) | None | Always - this is the "CFG scale" slider |
| [ControlNet](../papers/image-generation/71-controlnet/summary.md) | Spatial layout: pose, depth, edges | Yes, an encoder copy | Paired condition images | Composition must match a reference |
| [DreamBooth](../papers/image-generation/92-dreambooth/summary.md) | *Which* subject appears | Yes, fine-tunes the model | 3-5 images of the subject | A specific person, pet or product |
| [LoRA](../papers/techniques/10-lora/summary.md) | Style or subject, cheaply | Yes, low-rank adapters | Small set | You want a shareable ~10-100MB file |

**These compose.** A typical production stack runs a latent model with CFG, a subject LoRA, and a
ControlNet for pose - all at once. They are answering different questions.

### Image Tokenizers: VAE vs VQ-VAE vs VQ-GAN

| | [VAE](../papers/image-generation/57-vae/summary.md) | [VQ-VAE](../papers/image-generation/89-vq-vae/summary.md) | [VQ-GAN](../papers/image-generation/90-vq-gan/summary.md) |
|---|---|---|---|
| **Latent** | Continuous | Discrete codebook | Discrete codebook |
| **Extra losses** | KL to a prior | Codebook + commitment | + adversarial + perceptual |
| **Reconstruction** | Blurry | Sharper | Sharp at high resolution |
| **Feeds** | [Stable Diffusion](../papers/image-generation/07-stable-diffusion/summary.md)'s latent space | Autoregressive priors | Transformers over image tokens |
| **Why it matters** | Made latent diffusion possible | Made images tokenisable | Made token-based high-res synthesis work |

**The through-line:** compress first, model second. Both branches - continuous latents for
diffusion, discrete tokens for autoregressive models - come from the same insight, and both trace
back to the VAE.

---

## Efficiency Techniques

### Mixture of Experts: Dense vs Sparse Routing

| Aspect | Dense Transformer | [Switch Transformer](../papers/architectures/67-switch-transformer/summary.md) (top-1) | Mixtral (top-2) |
|--------|-------------------|----------------------------------------------------------------------------------------|-----------------|
| **Routing** | All params active | Top-1 expert per token | Top-2 experts per token |
| **Active Params** | 100% | ~1/N (N = num experts) | ~2/N |
| **Total Params** | Baseline | 4-8× more | 4-8× more |
| **Compute/Token** | Baseline | Same as smaller dense model | Slightly more than Switch |
| **Training Stability** | High | Lower (load balancing needed) | Higher than Switch |
| **Expert Utilization** | N/A | Uneven without aux loss | More balanced |
| **Quality vs Compute** | Good | 7× more compute-efficient (Switch paper) | Better quality than top-1 |
| **Communication Cost** | Low | High (all-to-all expert routing) | High |
| **Examples** | GPT-3, LLaMA | Switch-Base/Large | Mixtral 8×7B, 8×22B |

**The MoE trade-off:** Sparse MoE gives near-dense-model quality at a fraction of the FLOPs per token, but at the cost of much larger total parameter counts, complex routing, and communication overhead across devices.

**Top-1 vs top-2 routing:**
- **Switch (top-1)**: Simpler, lower compute, but each token sees only one expert - higher variance
- **Mixtral (top-2)**: Each token mixes two experts - more stable, better quality, slightly more compute

---

### Fine-Tuning Methods

| Method | Trainable Params | Memory | Speed | Quality | Use Case |
|--------|-----------------|--------|-------|---------|----------|
| **Full Fine-tuning** | 100% | Very high | Slow | Best | Unlimited resources |
| **LoRA** | 0.01-1% | Low | Fast | Near-full | Most practical cases |
| **Prefix Tuning** | 0.1-0.5% | Very low | Very fast | Good | Quick adaptation |
| **Adapter Layers** | 1-5% | Low | Fast | Very good | Multiple tasks |
| **Prompt Tuning** | 0.001-0.01% | Minimal | Fastest | Moderate | Simple tasks |

**LoRA in Detail (7B model example):**
```
Full fine-tuning: 7B trainable params, ~28GB memory
LoRA (r=8):       ~4M trainable params, ~8GB memory
Reduction:        1,750× fewer params, 3.5× less memory
```

**When to use LoRA:**
- ✅ Limited GPU memory
- ✅ Need to fine-tune multiple times
- ✅ Want to deploy multiple adaptations
- ❌ Unlimited resources and want absolute best
- ❌ Catastrophic domain shift (full fine-tuning better)

---

### Knowledge Enhancement: RAG vs Fine-tuning vs Prompting

| Aspect | RAG | Fine-tuning | In-Context (Prompting) |
|--------|-----|-------------|------------------------|
| **Knowledge Update** | Instant (update DB) | Slow (retrain) | Instant (change prompt) |
| **Accuracy** | High (grounded) | High (internalized) | Moderate (limited context) |
| **Cost (Setup)** | Medium (build index) | High (training) | Low (just prompt) |
| **Cost (Inference)** | High (retrieval + gen) | Low (just gen) | Low (just gen) |
| **Latency** | Higher (2-stage) | Lower (1-stage) | Lowest |
| **Citations** | Native support | Not possible | Not possible |
| **Hallucination** | Lower | Moderate | Higher |
| **Context Limit** | Bypassed (retrieval) | Model context limit | Model context limit |
| **Domain Adaptation** | Good | Excellent | Poor |

**Decision Matrix:**

| Scenario | Best Approach | Why |
|----------|---------------|-----|
| Customer support with docs | **RAG** | Need citations, docs update frequently |
| Domain-specific language | **Fine-tuning** | Need internalized knowledge |
| Quick experiments | **Prompting** | Fast, no infrastructure |
| Factual Q&A | **RAG** | Reduces hallucination |
| Style/tone adaptation | **Fine-tuning** | Deep behavioral change |
| Multi-task with shared knowledge | **RAG** | One knowledge base, many tasks |

**Combination (Best):**
```
Base Model
    ↓
Fine-tune (domain language/style)
    ↓
RAG (dynamic facts)
    ↓
Prompting (task-specific instructions)
```

---

## Inference and Serving

### Where the Money Goes

Four techniques, four different bottlenecks. They stack.

| Technique | Attacks | Typical gain | Quality cost | Needs retraining? |
|-----------|---------|--------------|--------------|-------------------|
| [FlashAttention](../papers/techniques/16-flash-attention/summary.md) | Attention memory traffic | Longer context, faster steps | None - exact | No |
| [GQA](../papers/architectures/75-grouped-query-attention/summary.md) | KV-cache size | Several-fold smaller cache | Slight | Yes (architecture choice) |
| [PagedAttention](../papers/techniques/52-pagedattention-vllm/summary.md) | KV-cache *waste* | Up to 24x throughput | None | No |
| [Speculative Decoding](../papers/techniques/45-speculative-decoding/summary.md) | Sequential decode latency | 2-3x | None - provably identical | No (needs a draft model) |
| [GPTQ / AWQ](../papers/techniques/86-gptq-awq-quantization/summary.md) | Weight memory | ~4x smaller at 4-bit | Small but real | No (post-training) |
| [MoE](../papers/architectures/37-mixture-of-experts/summary.md) | Compute per token | Large capacity, small active cost | None | Yes (architecture choice) |

**Order to reach for them:** if you serve someone else's model, start with vLLM (PagedAttention)
and quantisation - no retraining, largest wins. Speculative decoding next if latency is the
complaint. GQA and MoE are decisions made before training, not fixes afterwards.

### Attention Variants

| | MHA | [MQA](../papers/architectures/75-grouped-query-attention/summary.md) | [GQA](../papers/architectures/75-grouped-query-attention/summary.md) |
|---|---|---|---|
| **KV heads** | One per query head | One, shared by all | One per group |
| **KV cache** | Largest | Smallest | Tunable middle |
| **Quality** | Best | Noticeable degradation | Near-MHA |
| **Used by** | Original Transformer | PaLM | LLaMA 2/3, [Mistral](../papers/language-models/95-mistral-7b/summary.md), most modern LLMs |

### Positional Encoding

| | Sinusoidal / learned absolute | [RoPE](../papers/techniques/54-rope-rotary-position-embedding/summary.md) |
|---|---|---|
| **Encodes** | Absolute position | Relative position, via absolute rotation |
| **Parameters** | Learned variant adds some | None |
| **Extrapolates beyond training length** | Poorly | Better, and extendable (NTK-aware, YaRN) |
| **Used by** | Original Transformer, BERT, GPT-2 | Nearly every LLM since 2022 |

---

## Retrieval and Knowledge

### Sparse vs Dense vs Late-Interaction vs Graph

| | BM25 (sparse) | [DPR / Sentence-BERT](../papers/techniques/87-dense-retrieval/summary.md) | [ColBERT](../papers/techniques/87-dense-retrieval/summary.md) | [GraphRAG](../papers/techniques/60-graph-rag/summary.md) |
|---|---|---|---|---|
| **Matches on** | Exact terms | Meaning | Meaning, per token | Entities and relationships |
| **Handles synonyms** | No | Yes | Yes | Yes |
| **Handles rare terms / IDs** | Excellent | Poorly | Well | Depends on extraction |
| **Index cost** | Low | Moderate | High (per-token vectors) | Highest - LLM extraction pass |
| **Query cost** | Lowest | Low | Moderate | Moderate, pre-computed summaries |
| **Answers "what are the themes?"** | No | No | No | **Yes** |
| **Best for** | Keyword and code search | General semantic search | Precision-critical retrieval | Corpus-level questions |

**The practical answer is hybrid.** Dense retrieval alone fails on product codes, error strings and
proper nouns; BM25 alone fails on paraphrase. Most production systems run both and fuse the
rankings. Reach for GraphRAG only when the questions are genuinely global - "what themes run
through these 10,000 documents?" - because the indexing pass costs real money.

### Grounding Strategies

| | Prompting | [RAG](../papers/techniques/13-rag/summary.md) | Fine-tuning | Long context |
|---|---|---|---|---|
| **Knowledge updates** | Instantly | Instantly (re-index) | Retraining | Instantly |
| **Cost per query** | Lowest | Low + retrieval | Lowest after training | High - you pay for the tokens |
| **Cites sources** | No | Yes | No | Sometimes |
| **Teaches new *behaviour*** | Weakly | No | **Yes** | No |
| **Teaches new *facts*** | Small amounts | **Yes** | Unreliably | Yes, within the window |

**The rule that saves the most money:** RAG for facts, fine-tuning for behaviour. Fine-tuning to
inject knowledge is the single most common expensive mistake - it is unreliable, and the facts go
stale the moment the model is trained.

---

## Agents and Tool Use

| | [ReAct](../papers/techniques/21-react/summary.md) | [Reflexion](../papers/techniques/78-reflexion/summary.md) | [Generative Agents](../papers/techniques/58-generative-agents/summary.md) | [Voyager](../papers/techniques/100-voyager/summary.md) |
|---|---|---|---|---|
| **Year** | 2023 | 2023 | 2023 | 2023 |
| **Core loop** | Think → act → observe | + verbal self-critique | + memory, reflection, planning | + skill library as code |
| **Learns across episodes** | No | Yes, in episodic memory | Yes, via reflection | Yes, as reusable code |
| **Weight updates** | None | None | None | None |
| **Horizon** | One task | One task, retried | Days of simulated life | Open-ended |
| **Best for** | Any tool-using agent | Tasks with a failure signal | Simulation, multi-agent social behaviour | Open-ended skill acquisition |

**[Toolformer](../papers/techniques/24-toolformer/summary.md)** is the odd one out: it *trains* the
model to call APIs rather than prompting it to. **[MCP](../papers/techniques/59-model-context-protocol/summary.md)**
is orthogonal to all of them - it standardises how the tools are exposed, not how the agent thinks.

---

## Evaluation

| | Static benchmarks (MMLU, HumanEval) | [LLM-as-a-Judge](../papers/techniques/85-llm-as-judge/summary.md) | Human arena (Elo) | [SWE-bench](../papers/techniques/84-swe-bench/summary.md) |
|---|---|---|---|---|
| **Cost** | Lowest | Low | High | Moderate (sandboxed runs) |
| **Reproducible** | Yes | Mostly | No | Yes |
| **Contamination risk** | **High** | Moderate | Low | Lower - real repos, held-out issues |
| **Measures** | Knowledge, narrow skills | Preference on open tasks | Real user preference | End-to-end task completion |
| **Known biases** | Saturation, leakage | Position, verbosity, self-preference | Popularity, presentation | Repo and language skew |

**Read [Emergent Abilities](../papers/techniques/81-emergent-abilities/summary.md) alongside these.**
Its pairing with the "Mirage" rebuttal is the clearest lesson in the collection that a metric
choice - exact-match versus partial credit - can manufacture a discontinuity that isn't there.

---

## Beyond Language

### Protein Structure

| | [AlphaFold 2](../papers/techniques/68-alphafold/summary.md) | [ESM-2 / ESMFold](../papers/techniques/106-esm/summary.md) | [AlphaFold 3](../papers/techniques/101-alphafold3/summary.md) |
|---|---|---|---|
| **Year** | 2021 | 2023 | 2024 |
| **Input** | Sequence + MSA | Single sequence | Sequences + ligands, DNA, RNA, ions |
| **Needs MSA search** | Yes (slow) | **No** | Yes |
| **Speed** | Baseline | Up to 60x faster | Slower, far broader |
| **Predicts** | Single protein structure | Single protein structure | Biomolecular complexes |
| **Trade-off** | Most accurate for single proteins | Speed and metagenomic coverage | Interactions, which is what drugs are |

### Self-Play and World Models

| | [AlphaZero](../papers/techniques/102-alphazero/summary.md) | [DreamerV3](../papers/techniques/105-dreamerv3/summary.md) | [Genie](../papers/techniques/104-genie/summary.md) |
|---|---|---|---|
| **Learns from** | Self-play, rules known | Interaction, model learned | Internet video, no actions labelled |
| **Plans in** | Real game tree (MCTS) | Imagined latent rollouts | - (generates the world) |
| **Output** | Superhuman play | General control across 150+ tasks | A playable environment |
| **Relevance to LLMs** | The ancestor of self-improvement loops | Model-based planning | Foundation world models |

---

## Evolution Over Time

### Word Embeddings: Static vs Contextual

| Aspect | [Word2Vec](../papers/techniques/53-word2vec/summary.md) (2013) | GloVe (2014) | ELMo (2018) | BERT (2018) |
|--------|----------------------------------------------------------------|--------------|-------------|-------------|
| **Embedding type** | Static (1 vector per word) | Static (1 vector per word) | Contextual (BiLSTM) | Contextual (Transformer) |
| **"Bank" the word** | Same vector always | Same vector always | Different by sentence | Different by sentence |
| **Training** | Skip-gram / CBOW | Co-occurrence matrix | Language model (BiLSTM) | Masked language modeling |
| **Params** | Small (vocab × dim) | Small (vocab × dim) | Medium (LSTM layers) | Large (110M+) |
| **Inference Speed** | Lookup (instant) | Lookup (instant) | Forward pass (moderate) | Forward pass (slower) |
| **Polysemy handling** | None | None | Partial | Full |
| **Sentence context** | No | No | Full sentence (BiLSTM) | Full sentence (Transformer) |
| **Downstream tasks** | Feature input to model | Feature input to model | Feature input or fine-tune | Fine-tune end-to-end |
| **Still used?** | Yes (fast, no GPU needed) | Yes (NLP basics) | Largely replaced | Yes (or its successors) |

**The key shift:** Word2Vec proved that dense vector representations capture semantic relationships (king - man + woman = queen). BERT proved those representations should be contextual - the same word needs a different embedding depending on its sentence. This shift from static to contextual embeddings is the foundation of modern NLP.

---

### Language Model Performance (on Common Benchmarks)

| Model (Year) | Params | MMLU | HellaSwag | GSM8k | HumanEval |
|--------------|--------|------|-----------|-------|-----------|
| BERT (2018) | 340M | - | 78% | - | - |
| [GPT-2](../papers/language-models/64-gpt2/summary.md) (2019) | 1.5B | - | ~70% | - | - |
| GPT-3 (2020) | 175B | 43.9% | 78.9% | 17% | - |
| GPT-3.5 (2022) | ? | ~70% | ~95% | 57% | 48% |
| LLaMA-65B (2023) | 65B | 63.4% | 84.2% | 50.9% | 23% |
| GPT-4 (2023) | ? | 86.4% | ~95% | 92% | 67% |
| Claude 3 (2024) | ? | 86.8% | - | - | - |

**Trends:**
- 2018-2020: Scale up parameters
- 2020-2022: Alignment via RLHF
- 2022-2023: Compute-optimal training
- 2023+: Multimodal, reasoning, efficiency

---

### Parameter Efficiency Over Time

| Year | Model | Params | Tokens | Performance | Efficiency Gain |
|------|-------|--------|--------|-------------|----------------|
| 2020 | GPT-3 | 175B | 300B | Baseline | 1× |
| 2022 | Chinchilla | 70B | 1.4T | Same | 2.5× fewer params |
| 2023 | LLaMA-13B | 13B | 1T | Same | 13.5× fewer params |
| 2023 | LLaMA-65B | 65B | 1.4T | Better | 2.7× fewer params |

**What this means:**
- 2020: "Need 175B params for GPT-3 performance"
- 2023: "Need only 13B params with better training"
- **13× parameter reduction** in 3 years through better training

---

### Image Generation Quality Over Time

| Year | Model | Method | Resolution | Speed | Quality (FID) |
|------|-------|--------|------------|-------|---------------|
| 2014 | Original GAN | Adversarial | 64×64 | Fast | Poor (~50) |
| 2018 | StyleGAN | GAN | 1024×1024 | Fast | Good (~4) |
| 2020 | DDPM | Diffusion | 256×256 | Very slow | Excellent (~3) |
| 2022 | Stable Diffusion | Latent Diffusion | 512×512+ | Medium | Excellent (~10) |
| 2022 | DALL-E 2 | Diffusion + CLIP | 1024×1024 | Medium | Excellent |

**FID Score:** Lower is better (measures distribution similarity)

---

### Transformers Beyond Language

The Transformer architecture generalized far beyond NLP. These applications use the same attention mechanism but on fundamentally different sequence types.

| Domain | Model (Year) | Sequence Type | Key Adaptation | Impact |
|--------|--------------|---------------|----------------|--------|
| NLP | BERT / GPT (2018-20) | Tokens | None - native domain | State-of-the-art on all NLP tasks |
| Vision | ViT (2020) | Image patches | Flatten patches as tokens | Matches CNN at scale |
| Images (gen.) | Stable Diffusion (2022) | Latent patches | Cross-attention for text conditioning | Best text-to-image |
| Protein structure | [AlphaFold 2](../papers/techniques/68-alphafold/summary.md) (2021) | Amino acid residues | Evoformer + structure module (triangle attention) | Solved 50-year protein folding problem |
| Code | Codex / GPT-4 (2021+) | Code tokens | Fine-tuned on code corpora | Near-human code generation |
| Audio | Whisper (2022) | Spectrogram patches | Conv frontend + Transformer | Robust multilingual ASR |

**AlphaFold 2's significance in this context:** It demonstrated that the core insight of attention - letting every element attend to every other element - applies to non-linguistic structure prediction. Amino acid residues attending to each other to infer 3D spatial relationships is conceptually identical to tokens attending to each other to infer semantic relationships. AlphaFold 2 effectively closed the protein structure prediction problem (GDT > 90 on CASP14), a benchmark that had resisted 50 years of computational biology.

---

## Technique Combinations

### What Works Well Together

| Combination | Use Case | Example |
|-------------|----------|---------|
| **LoRA + RAG** | Efficient domain chatbot | Domain-tuned LLaMA + company docs |
| **CLIP + Stable Diffusion** | Text-to-image | How SD does text conditioning |
| **RLHF + Constitutional AI** | Aligned assistant | Helpful via RLHF, safe via CAI |
| **ViT + Diffusion** | High-quality generation | Modern text-to-image models |
| **RAG + Chain-of-Thought** | Grounded reasoning | Retrieve facts, reason step-by-step |
| **LLaMA + LoRA** | Accessible fine-tuning | Most popular open-source combo |
| **VAE + Diffusion** | Efficient image generation | Stable Diffusion's latent space backbone |
| **Seq2Seq + Attention** | Translation / summarization | Pre-Transformer NMT (still used in constrained settings) |
| **PPO + Reward Model** | RLHF fine-tuning | InstructGPT, ChatGPT alignment |
| **Switch MoE + Decoder** | Scalable generation | Mixtral - dense quality at sparse compute |

---

### Production Stack Comparison

**Scenario: Enterprise Chatbot**

| Stack | Description | Pros | Cons |
|-------|-------------|------|------|
| **GPT-4 API** | Direct API calls | Highest quality, no infra | Expensive, no customization |
| **LLaMA + LoRA + RAG** | Self-hosted optimized | Full control, lower cost | Setup complexity |
| **Claude API** | Constitutional AI aligned | Good safety, citations | API dependency |
| **Fine-tuned BERT + Rules** | Traditional NLP | Fast, cheap, reliable | Limited generalization |

**Cost Comparison (1M tokens):**
- GPT-4 API: $60 (generation)
- Claude API: $24 (generation)
- Self-hosted LLaMA-13B: ~$2 (compute only)
- BERT: <$1 (compute only)

---

## Research Impact Comparison

### Influence by Descendants

Citation counts go stale the month you write them down, so this ranks by something more durable:
how much of the current stack descends from the paper.

| Paper | What descends from it |
|-------|-----------------------|
| [Transformer](../papers/architectures/01-attention-is-all-you-need/summary.md) | Every model in this collection except the pre-2017 roots |
| [ResNet](../papers/architectures/73-resnet/summary.md) | Residual connections in every Transformer block; the whole deep-vision era |
| [Bahdanau Attention](../papers/architectures/66-bahdanau-attention/summary.md) | Attention itself, and therefore the Transformer |
| [VAE](../papers/image-generation/57-vae/summary.md) | Latent diffusion, VQ-VAE, VQ-GAN - the entire compress-then-model line |
| [U-Net](../papers/architectures/74-unet/summary.md) | The denoiser in every diffusion model until DiT |
| [PPO](../papers/techniques/63-ppo/summary.md) | RLHF, and by reaction DPO, KTO and GRPO |
| [LLaMA](../papers/language-models/15-llama/summary.md) | The open-weight ecosystem: Alpaca, Vicuna, Mistral, and thousands of fine-tunes |
| [Chain-of-Thought](../papers/techniques/09-chain-of-thought/summary.md) | Self-consistency, ToT, STaR, PRMs, and every reasoning model |
| [LoRA](../papers/techniques/10-lora/summary.md) | QLoRA, and the adapter-sharing ecosystem in both text and image |
| [CLIP](../papers/multimodal/08-clip/summary.md) | Text conditioning in image generators; the vision tower in VLMs |

**Sleeper hits** - unglamorous papers doing enormous work in production:
[RoPE](../papers/techniques/54-rope-rotary-position-embedding/summary.md),
[GQA](../papers/architectures/75-grouped-query-attention/summary.md),
[FlashAttention](../papers/techniques/16-flash-attention/summary.md) and
[PagedAttention](../papers/techniques/52-pagedattention-vllm/summary.md). None changed what models
can do; all four changed what they cost.

---

## When to Use Which Paper's Techniques

### Quick Decision Tree

**Need to generate text?**
- Long-form, creative → GPT-3 / LLaMA
- Factual, grounded → RAG
- With reasoning → Chain-of-Thought
- Specific style → Fine-tuning + LoRA
- Translation or structured output → T5 (encoder-decoder)

**Need to understand text?**
- Classification → BERT
- Semantic search → BERT / CLIP (for images)
- Q&A → BERT + RAG
- Simple/fast embeddings (no GPU) → Word2Vec / GloVe

**Need to generate images?**
- Artistic, text-to-image → Stable Diffusion, or SD3 / Flux (flow matching)
- Faster sampling from a trained model → DDIM
- Prompt adherence → Classifier-Free Guidance (the CFG slider)
- Control the composition → ControlNet
- A specific subject → DreamBooth, or a LoRA
- Video → Sora / DiT
- Fast, real-time single pass → GANs
- Structured latent space / interpolation → VAE

**Need better reasoning?**
- Cheapest improvement → Chain-of-Thought
- More accuracy, more budget → Self-Consistency
- Needs exploration and backtracking → Tree of Thoughts
- You can verify answers and train → RLVR + GRPO
- Bad intermediate steps → Process Reward Models

**Need to align a model?**
- General helpfulness → RLHF / PPO (InstructGPT)
- Safety focus → Constitutional AI, Llama Guard for moderation
- Efficient preference learning → DPO
- Only thumbs-up/down data → KTO
- Maths / verifiable reasoning → GRPO + RLVR

**Need to adapt a model?**
- Full resources → Fine-tuning
- Limited resources → LoRA, or QLoRA on one GPU
- New facts, not new behaviour → RAG (do not fine-tune for this)
- No training at all → Prompting

**Need it cheaper or faster in production?**
- Throughput → vLLM / PagedAttention
- Latency → Speculative Decoding
- Won't fit on the GPU → GPTQ / AWQ 4-bit
- KV cache too large → GQA
- Capacity without inference cost → Mixture of Experts

**Need to retrieve over your own data?**
- General semantic search → Dense Retrieval, hybrid with BM25
- Precision-critical → ColBERT late interaction
- Corpus-level "what are the themes" questions → GraphRAG

**Building an agent?**
- Core loop → ReAct
- Learn from failures → Reflexion
- Long-lived, accumulating skills → Voyager
- Wiring tools to models → MCP
- Measuring it → SWE-bench

**Planning a project?**
- Estimate resources → Scaling Laws
- Choose model size and token budget → Chinchilla
- Decide train vs. think-longer → Test-Time Compute

**Applying this outside NLP?**
- Protein structure → AlphaFold 2, ESMFold if speed matters
- Molecular complexes and drug binding → AlphaFold 3
- Images → ViT, MAE for self-supervised pretraining
- Code → Codex, and SWE-bench to evaluate
- Games and control → AlphaZero, DreamerV3
- Interactive worlds → Genie

---

## Key Insights from Comparisons

1. **Scaling isn't everything** - LLaMA proved training matters more than size
2. **Hybrid is best** - Combine RLHF + Constitutional AI, RAG + fine-tuning
3. **Efficiency advances** - LoRA makes fine-tuning accessible, Stable Diffusion makes diffusion practical, MoE makes scale affordable
4. **Open vs closed** - Open models (LLaMA) spawned more innovation than closed (GPT-3)
5. **Architecture consolidation** - Transformers won for text, vision, protein structure, and more
6. **Alignment evolution** - From RLHF/PPO to DPO to GRPO (each iteration simpler or more targeted)
7. **Knowledge grounding** - RAG reduces hallucination better than any architecture change
8. **Embeddings matured** - Static (Word2Vec) to contextual (ELMo) to Transformer-based (BERT) over ~5 years
9. **Seq2Seq lineage** - Every encoder-decoder model (T5, Stable Diffusion decoder, etc.) inherits from Sutskever 2014 via Bahdanau attention
10. **Compute moved from training to inference** - Test-Time Compute and o1 showed a small model that thinks longer can beat a much larger one that answers immediately
11. **Alignment kept shedding machinery** - PPO needed a reward model and a critic; DPO dropped the reward model, GRPO dropped the critic, KTO dropped paired data, RLVR dropped learned rewards entirely
12. **Sparsity beat density** - the dense scaling line topped out at PaLM's 540B; every frontier model since activates a fraction of its parameters
13. **Compress first, then model** - the VAE insight recurs everywhere: latent diffusion, VQ-VAE tokens, VQ-GAN, and video as spacetime patches
14. **The cheap wins are in serving** - RoPE, GQA, FlashAttention and PagedAttention changed no capability and changed every budget
15. **Evaluation is the weak link** - static benchmarks leak, LLM judges are biased, and metric choice alone can invent an "emergent" jump

---

**Last updated:** 2026-08-20 · Covers all 107 papers in the collection.
