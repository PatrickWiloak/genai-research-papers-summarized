# Foundational Generative AI Research Papers - Summarized

A curated collection of the **94 most influential papers** that shaped modern generative AI, with comprehensive summaries designed to make cutting-edge research accessible to everyone.

[![Papers](https://img.shields.io/badge/Papers-94-blue.svg)](./papers/)
[![Guides](https://img.shields.io/badge/Guides-6-green.svg)](./docs/)
[![License](https://img.shields.io/badge/License-Educational-orange.svg)](./LICENSE)
[![Updated](https://img.shields.io/badge/Updated-May_2026-green.svg)](./README.md)

---

## 🚀 Quick Start

**New to AI?** Start with the [Learning Roadmap](./docs/ROADMAP.md)
**Want to browse visually?** See [BROWSE.md](./BROWSE.md) for table/grid view
**Not sure what to read?** Check the [Reading Guide](./docs/READING_GUIDE.md) - Historical vs Modern relevance
**Need quick lookup?** See the [Quick Reference](./docs/QUICK_REFERENCE.md)
**Building something?** Read [Comparisons](./docs/COMPARISONS.md) for decision guides
**Confused by terms?** Browse the [Glossary](./docs/GLOSSARY.md)

---

## 📚 Documentation

### Core Guides
- 🗺️ **[Learning Roadmap](./docs/ROADMAP.md)** - Structured paths from beginner to expert (4 tracks)
- ⭐ **[Reading Guide](./docs/READING_GUIDE.md)** - What's still relevant vs historical context
- 📖 **[Quick Reference](./docs/QUICK_REFERENCE.md)** - One-page overview of all 94 papers
- 🔍 **[Comparisons](./docs/COMPARISONS.md)** - Side-by-side analysis and trade-offs
- 📚 **[Glossary](./docs/GLOSSARY.md)** - 150+ key terms explained

---

## 📁 Repository Structure

```
genai-research-papers-summarized/
├── README.md                          # You are here
├── docs/                              # All guides and documentation
│   ├── ROADMAP.md                     # Learning paths
│   ├── QUICK_REFERENCE.md             # Quick lookup
│   ├── COMPARISONS.md                 # Decision guides
│   └── GLOSSARY.md                    # Term definitions
├── papers/                            # All paper summaries organized by category
│   ├── architectures/                 # Foundational architectures
│   │   ├── 01-attention-is-all-you-need/
│   │   ├── 11-vision-transformer/
│   │   ├── 20-mamba/
│   │   ├── 37-mixture-of-experts/
│   │   ├── 63-word2vec/
│   │   ├── 64-seq2seq/
│   │   ├── 65-bahdanau-attention/
│   │   ├── 66-resnet/
│   │   └── 74-mae-masked-autoencoders/
│   ├── language-models/               # LLM papers
│   │   ├── 03-bert/  04-gpt3-few-shot-learners/  05-instructgpt-rlhf/
│   │   ├── 14-constitutional-ai/  15-llama/  17-llama2/  19-dpo/
│   │   ├── 26-deepseek-r1/  27-deepseek-v3/  28-qwen3/  30-claude-3.5-sonnet/
│   │   ├── 31-openai-o1/  33-llama3.3/  36-gpt4/  40-gpt4o/  41-llama4/
│   │   ├── 42-gpt5/  43-claude4/  56-codex/  68-t5/  69-gpt-1/  70-gpt-2/
│   │   └── 71-palm/  72-mistral-7b/  73-mixtral/  92-llama-guard/
│   ├── image-generation/              # Image generation papers
│   │   ├── 02-generative-adversarial-networks/  06-diffusion-models/
│   │   ├── 07-stable-diffusion/  44-sora-dit/  48-dalle3/  67-vae/
│   │   ├── 75-ddpm/  76-vq-vae/  77-vq-gan/  78-imagen/
│   │   └── 79-controlnet/  80-dreambooth/
│   ├── multimodal/                    # Cross-modal papers
│   │   ├── 08-clip/  23-gpt4v/  29-gemini-2.5/  32-sam2/
│   │   └── 46-llava/  47-gemini3/  49-whisper/
│   └── techniques/                    # Methods and techniques
│       ├── 09-chain-of-thought/  10-lora/  12-scaling-laws/  13-rag/
│       ├── 16-flash-attention/  18-chinchilla/  21-react/  22-qlora/
│       ├── 24-toolformer/  25-tree-of-thoughts/  34-meta-cot/  35-rstar-math/
│       ├── 38-grpo/  39-rlvr/  45-speculative-decoding/
│       ├── 50-test-time-compute/  51-process-reward-models/  52-pagedattention-vllm/
│       ├── 54-rope-rotary-position-embedding/  58-generative-agents/
│       ├── 59-model-context-protocol/  60-graph-rag/  61-alphageometry/  62-alphaevolve/
│       ├── 81-star-self-taught-reasoner/  82-quiet-star/  83-reflexion/
│       ├── 84-self-refine/  85-self-consistency/  86-voyager/
│       ├── 87-alphafold2/  88-alphafold3/  89-alphazero/
│       ├── 90-sparse-autoencoders/  91-swe-bench/  93-kto/
│       └── 94-genie/  95-dreamerv3/  96-esm/  97-cicero/
└── resources/                         # Additional resources
    ├── images/                        # Diagrams and visualizations
    └── notebooks/                     # Jupyter notebooks
```

---

## 📄 Papers by Category

### 🏗️ Foundational Architectures
**Recommended Reading Order:** 63 → 64 → 65 → 1 → 11 → 74 → 20 → 37

#### **Pre-Transformer Foundations**

**1.** [Word2Vec](./papers/architectures/63-word2vec/) (2013)
- 📚 **HISTORICAL** - First widely-used dense word embeddings
- Vector arithmetic for semantics (king - man + woman ≈ queen)
- [Paper](https://arxiv.org/abs/1301.3781)

**2.** [Seq2Seq](./papers/architectures/64-seq2seq/) (2014)
- 📚 **HISTORICAL** - Invented encoder-decoder paradigm
- LSTM encoder-decoder for translation, beat phrase-based SMT
- [Paper](https://arxiv.org/abs/1409.3215)

**3.** [Bahdanau Attention](./papers/architectures/65-bahdanau-attention/) (2014)
- 🔥 **CRITICAL** - Invented attention; direct ancestor of the Transformer
- Solved the seq2seq bottleneck with soft alignment
- [Paper](https://arxiv.org/abs/1409.0473)

**4.** [ResNet](./papers/architectures/66-resnet/) (2015)
- 🔥 **CRITICAL** - Residual connections; used in every Transformer block
- Made 100+ layer networks trainable for the first time
- [Paper](https://arxiv.org/abs/1512.03385)

#### **The Transformer Era**

**5. Start Here:** [Attention Is All You Need](./papers/architectures/01-attention-is-all-you-need/) (2017)
- 🔥 **CRITICAL** - Foundation of everything
- Transformer architecture, self-attention mechanism
- [Paper](https://arxiv.org/abs/1706.03762)

**6.** [Vision Transformer (ViT)](./papers/architectures/11-vision-transformer/) (2020)
- ⭐ **HIGH** - Transformers for computer vision
- Images as patch sequences, enables multimodal models
- [Paper](https://arxiv.org/abs/2010.11929)

**7.** [MAE - Masked Autoencoders](./papers/architectures/74-mae-masked-autoencoders/) (2021)
- ⭐ **HIGH** - BERT-style self-supervised pretraining for vision
- Mask 75% of patches, reconstruct with asymmetric encoder-decoder
- [Paper](https://arxiv.org/abs/2111.06377)

#### **Beyond Transformers**

**8.** [Mamba](./papers/architectures/20-mamba/) (2023)
- 🔥 **CRITICAL** - First viable Transformer alternative
- Linear-time sequence modeling (O(n) vs O(n²)), selective state spaces
- [Paper](https://arxiv.org/abs/2312.00752)

**9.** [Mixture-of-Experts (Mixtral)](./papers/architectures/37-mixture-of-experts/) (2024)
- 🔥 **CRITICAL** - Architecture behind every frontier model
- Sparse routing, expert specialization, used by DeepSeek-V3, Llama 4, Qwen3
- [Paper](https://arxiv.org/abs/2401.04088)

### 🤖 Language Models
**Recommended Reading Order:** Evolution → Frontier → Alignment → Open Source → Reasoning → Unified

#### **Early Evolution (Historical Context)**

**1.** [GPT-1](./papers/language-models/69-gpt-1/) (2018)
- 📚 **HISTORICAL** - First of the GPT lineage
- Decoder-only Transformer, unsupervised pretrain + supervised fine-tune
- [Paper](https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf)

**2.** [BERT](./papers/language-models/03-bert/) (2018)
- 📚 **HISTORICAL** - Pre-training revolution
- Bidirectional pre-training, masked language modeling
- [Paper](https://arxiv.org/abs/1810.04805)

**3.** [GPT-2](./papers/language-models/70-gpt-2/) (2019)
- 📚 **HISTORICAL** - 1.5B params, zero-shot task transfer
- Staged release, foreshadowed in-context learning
- [Paper](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)

**4.** [T5](./papers/language-models/68-t5/) (2019)
- ⭐ **HIGH** - Text-to-text framing of every NLP task
- C4 dataset, 11B params, encoder-decoder design study
- [Paper](https://arxiv.org/abs/1910.10683)

**5.** [GPT-3](./papers/language-models/04-gpt3-few-shot-learners/) (2020)
- ⭐ **HIGH** - Few-shot learning paradigm
- 175B parameters, foundation for ChatGPT
- [Paper](https://arxiv.org/abs/2005.14165)

**6.** [PaLM](./papers/language-models/71-palm/) (2022)
- ⭐ **HIGH** - 540B dense, Pathways system
- Breakthrough reasoning on BIG-bench; precursor to Gemini
- [Paper](https://arxiv.org/abs/2204.02311)

#### **The Frontier Era**

**3.** [GPT-4](./papers/language-models/36-gpt4/) (2023)
- 🔥 **CRITICAL** - Defined the frontier model era
- Passed bar exam (90th percentile), MMLU 86.4%
- First multimodal GPT, predictable scaling
- [Paper](https://arxiv.org/abs/2303.08774)

**4.** [GPT-4o](./papers/language-models/40-gpt4o/) (2024)
- 🔥 **CRITICAL** - First native omni-model
- Text, audio, image in one model, 232ms voice latency
- 2x faster, 50% cheaper than GPT-4 Turbo
- [System Card](https://cdn.openai.com/gpt-4o-system-card.pdf)

#### **Alignment Methods (How to Make Them Helpful)**

**3.** [InstructGPT (RLHF)](./papers/language-models/05-instructgpt-rlhf/) (2022)
- 🔥 **CRITICAL** - Human preference learning
- Enabled ChatGPT
- [Paper](https://arxiv.org/abs/2203.02155)

**4.** [Constitutional AI](./papers/language-models/14-constitutional-ai/) (2022)
- ⭐ **HIGH** - Alternative to RLHF
- AI self-critique via principles, powers Claude
- [Paper](https://arxiv.org/abs/2212.08073)

**5.** [DPO](./papers/language-models/19-dpo/) (2023)
- 🔥 **CRITICAL** - Simpler than RLHF
- No reward model needed
- [Paper](https://arxiv.org/abs/2305.18290)

#### **Open-Source Revolution (2023)**

**6.** [LLaMA](./papers/language-models/15-llama/) (2023)
- 🔥 **CRITICAL** - Compute-optimal training
- 13B matches GPT-3 175B
- [Paper](https://arxiv.org/abs/2302.13971)

**7.** [LLaMA 2](./papers/language-models/17-llama2/) (2023)
- 🔥 **CRITICAL** - Production-ready open model
- Commercial license, RLHF alignment
- [Paper](https://arxiv.org/abs/2307.09288)

**8.** [Mistral 7B](./papers/language-models/72-mistral-7b/) (2023)
- 🔥 **CRITICAL** - European open-weight breakthrough
- GQA + sliding-window attention, beat LLaMA 2 13B at 7B
- [Paper](https://arxiv.org/abs/2310.06825)

**9.** [Mixtral 8x7B](./papers/language-models/73-mixtral/) (2024)
- 🔥 **CRITICAL** - First high-quality open-weight MoE
- 47B total / 13B active, matched GPT-3.5
- [Paper](https://arxiv.org/abs/2401.04088)

**10.** [LLaMA 3.3](./papers/language-models/33-llama3.3/) (2024)
- 🔥 **HIGH** - Distillation breakthrough
- 70B matches 405B performance
- [Paper](https://www.meta.ai/blog/meta-llama-3-3/)

**11.** [Llama 4](./papers/language-models/41-llama4/) (2025)
- 🔥 **HIGH** - First open-source multimodal MoE
- Scout: 10M token context, Maverick: beats GPT-4o
- [Blog](https://ai.meta.com/blog/llama-4-multimodal-intelligence/)

#### **Safety & Moderation**

**12.** [Llama Guard](./papers/language-models/92-llama-guard/) (2023)
- ⭐ **HIGH** - Open-weight safety classifier
- LLM-based input/output moderation, customizable taxonomy
- [Paper](https://arxiv.org/abs/2312.06674)

#### **Efficiency Breakthroughs (2024)**

**9.** [DeepSeek-V3](./papers/language-models/27-deepseek-v3/) (2024)
- 🔥 **CRITICAL** - $5.76M training cost
- 671B MoE, matches GPT-4
- [Paper](https://arxiv.org/abs/2412.19437)

#### **Reasoning Era (2024-2025)**

**11.** [OpenAI o1](./papers/language-models/31-openai-o1/) (2024)
- 🔥 **CRITICAL** - Started reasoning model era
- PhD-level performance, RL for reasoning
- [Announcement](https://openai.com/index/learning-to-reason-with-llms/)

**11.** [DeepSeek-R1](./papers/language-models/26-deepseek-r1/) (2025)
- 🔥 **CRITICAL** - Pure RL reasoning
- Matches o1, fully open source
- [Paper](https://arxiv.org/abs/2501.12948)

**12.** [Qwen3](./papers/language-models/28-qwen3/) (2025)
- 🔥 **CRITICAL** - Unified thinking/non-thinking
- Adaptive reasoning, beats competitors
- [Paper](https://arxiv.org/abs/2505.09388)

**15.** [Claude 3.5 Sonnet](./papers/language-models/30-claude-3.5-sonnet/) (2024)
- 🔥 **CRITICAL** - Computer use capability
- Best coding model (49% SWE-Bench)
- [Announcement](https://www.anthropic.com/news/3-5-models-and-computer-use)

#### **Unified Intelligence (2025-2026)**

**16.** [GPT-5](./papers/language-models/42-gpt5/) (2025)
- 🔥 **CRITICAL** - Unified fast + reasoning model
- 94.6% AIME, 74.9% SWE-bench, 80% fewer hallucinations
- Adaptive routing between fast and deep thinking
- [System Card](https://cdn.openai.com/gpt-5-system-card.pdf)

**17.** [Claude 4 Family](./papers/language-models/43-claude4/) (2025-2026)
- 🔥 **CRITICAL** - Agentic AI leader
- 80.9% SWE-bench (Opus 4.5), best coding model
- Multi-agent orchestration, extended thinking
- [Announcement](https://www.anthropic.com/news/claude-4)

### 🎨 Image Generation
**Recommended Reading Order:** VAE → GAN → DDPM → VQ-VAE → VQ-GAN → Stable Diffusion → Imagen → DALL-E 3 → Sora → ControlNet/DreamBooth

#### **Foundations**

**1.** [VAE](./papers/image-generation/67-vae/) (2013)
- 📚 **HISTORICAL** - Variational Autoencoders + reparameterization trick
- Foundation for latent-space generative models
- [Paper](https://arxiv.org/abs/1312.6114)

**2.** [GANs](./papers/image-generation/02-generative-adversarial-networks/) (2014)
- 📚 **HISTORICAL** - Generative modeling origins
- Adversarial training: generator vs discriminator
- [Paper](https://arxiv.org/abs/1406.2661)

**3.** [VQ-VAE](./papers/image-generation/76-vq-vae/) (2017)
- ⭐ **HIGH** - Discrete-token image representations
- Codebook learning, foundation for DALL-E and modern tokenizers
- [Paper](https://arxiv.org/abs/1711.00937)

#### **The Diffusion Era**

**4.** [DDPM](./papers/image-generation/75-ddpm/) (2020)
- 🔥 **CRITICAL** - The seminal diffusion paper
- Forward/reverse processes, simplified loss, beat GANs on image quality
- [Paper](https://arxiv.org/abs/2006.11239)

**5.** [Diffusion Models (overview)](./papers/image-generation/06-diffusion-models/) (2020)
- 📖 **THEORY** - General diffusion theory and intuition
- Iterative denoising, score matching
- [Paper](https://arxiv.org/abs/2006.11239)

**6.** [VQ-GAN](./papers/image-generation/77-vq-gan/) (2021)
- ⭐ **HIGH** - VQ-VAE + adversarial + perceptual losses
- Direct ancestor of Stable Diffusion's latent space
- [Paper](https://arxiv.org/abs/2012.09841)

**7.** [Stable Diffusion](./papers/image-generation/07-stable-diffusion/) (2022)
- 🔥 **CRITICAL** - Latent diffusion, democratized AI art
- 10-100× faster than pixel-space diffusion
- [Paper](https://arxiv.org/abs/2112.10752)

**8.** [Imagen](./papers/image-generation/78-imagen/) (2022)
- ⭐ **HIGH** - Google's text-to-image
- Frozen T5-XXL text encoder beats CLIP for prompts
- [Paper](https://arxiv.org/abs/2205.11487)

#### **Control & Personalization**

**9.** [DreamBooth](./papers/image-generation/80-dreambooth/) (2022)
- ⭐ **HIGH** - Personalize diffusion from 3-5 images
- Subject-specific token + prior preservation loss
- [Paper](https://arxiv.org/abs/2208.12242)

**10.** [ControlNet](./papers/image-generation/79-controlnet/) (2023)
- 🔥 **CRITICAL** - Controllable diffusion generation
- Edges, poses, depth, sketches as conditioning
- [Paper](https://arxiv.org/abs/2302.05543)

**11.** [DALL-E 3](./papers/image-generation/48-dalle3/) (2023)
- ⭐ **HIGH** - Solved prompt adherence
- Better captions = better images, first readable text in images
- [Paper](https://cdn.openai.com/papers/dall-e-3.pdf)

**12.** [Sora / DiT](./papers/image-generation/44-sora-dit/) (2024)
- 🔥 **CRITICAL** - Video generation + Diffusion Transformers
- Transformers replaced U-Net in diffusion models
- [DiT Paper](https://arxiv.org/abs/2212.09748) | [Sora Report](https://openai.com/index/video-generation-models-as-world-simulators/)

### 🔗 Multimodal
**Recommended Reading Order:** Vision-language bridge → Practical multimodal → Next-gen unified AI

**1.** [CLIP](./papers/multimodal/08-clip/) (2021)
- ⭐ **HIGH** - Vision-language bridge
- Vision-language contrastive learning
- Zero-shot image classification, powers text-to-image models
- [Paper](https://arxiv.org/abs/2103.00020)

**2.** [Whisper](./papers/multimodal/49-whisper/) (2022)
- 🔥 **CRITICAL** - Foundation model for speech
- 680K hours of training data, 99 languages, zero-shot robustness
- 50% fewer errors than specialized models
- [Paper](https://arxiv.org/abs/2212.04356)

**3.** [LLaVA](./papers/multimodal/46-llava/) (2023)
- 🔥 **HIGH** - Blueprint for open-source multimodal
- Vision encoder + projection + LLM = visual assistant
- 85% of GPT-4V quality, spawned dozens of derivatives
- [Paper](https://arxiv.org/abs/2304.08485)

**4.** [GPT-4V(ision)](./papers/multimodal/23-gpt4v/) (2023)
- 🔥 **CRITICAL** - Multimodal frontier model
- GPT-4 with vision capabilities
- State-of-the-art VQA and OCR, real-world applications
- [Paper](https://cdn.openai.com/papers/GPTV_System_Card.pdf)

**5.** [SAM 2](./papers/multimodal/32-sam2/) (2024)
- 🔥 **HIGH** - Universal video segmentation
- 44 FPS real-time performance
- Zero-shot generalization across domains
- [Paper](https://arxiv.org/abs/2408.00714)

**6.** [Gemini 2.5](./papers/multimodal/29-gemini-2.5/) (2025)
- 🔥 **CRITICAL** - Advanced multimodal AI
- Native multimodal (text, image, audio, video)
- 1M context, 3-hour video understanding, integrated thinking mode
- [Paper](https://arxiv.org/abs/2507.06261)

**7.** [Gemini 3](./papers/multimodal/47-gemini3/) (2025)
- 🔥 **CRITICAL** - First model to cross 1500 LMArena ELO
- 91.8% MMLU, 95% AIME, best video understanding
- Deep Think mode: 45.1% ARC-AGI-2
- [Announcement](https://blog.google/products-and-platforms/products/gemini/gemini-3/)

### ⚡ Techniques & Methods
**Recommended Reading Order:** Scaling foundations → Efficiency → Reasoning → Agents

#### **Scaling Foundations (Start Here)**

**1.** [Scaling Laws](./papers/techniques/12-scaling-laws/) (2020)
- 🔥 **CRITICAL** - Predictive theory
- Predictable power laws, guides compute allocation
- [Paper](https://arxiv.org/abs/2001.08361)

**2.** [Chinchilla](./papers/techniques/18-chinchilla/) (2022)
- 🔥 **CRITICAL** - Rewrote scaling laws
- Equal scaling of params and tokens, proved GPT-3 was undertrained 4×
- [Paper](https://arxiv.org/abs/2203.15556)

#### **Efficiency Techniques**

**3.** [FlashAttention](./papers/techniques/16-flash-attention/) (2022)
- 🔥 **CRITICAL** - IO-aware attention
- 10-20× faster, enables 64k+ context lengths
- [Paper](https://arxiv.org/abs/2205.14135)

**4.** [LoRA](./papers/techniques/10-lora/) (2021)
- 🔥 **CRITICAL** - Efficient fine-tuning
- Low-rank adaptation, 10,000× fewer trainable parameters
- [Paper](https://arxiv.org/abs/2106.09685)

**5.** [QLoRA](./papers/techniques/22-qlora/) (2023)
- 🔥 **CRITICAL** - Efficient fine-tuning at scale
- 4-bit quantization + LoRA, 16× memory reduction
- [Paper](https://arxiv.org/abs/2305.14314)

#### **Inference Optimization**

**6.** [Speculative Decoding](./papers/techniques/45-speculative-decoding/) (2023)
- 🔥 **CRITICAL** - 2-3x faster inference, identical output
- Draft model guesses, target model verifies in parallel
- Used by every major LLM provider
- [Paper](https://arxiv.org/abs/2211.17192)

**7.** [PagedAttention / vLLM](./papers/techniques/52-pagedattention-vllm/) (2023)
- 🔥 **CRITICAL** - Made LLM serving practical
- Virtual memory for GPU KV-cache, 24x throughput improvement
- Near-zero memory waste, powers most production LLM deployments
- [Paper](https://arxiv.org/abs/2309.06180)

#### **Production Techniques**

**7.** [RAG](./papers/techniques/13-rag/) (2020)
- 🔥 **CRITICAL** - Production standard
- Retrieval-augmented generation, reduces hallucinations
- [Paper](https://arxiv.org/abs/2005.11401)

#### **Reasoning Methods**

**7.** [Chain-of-Thought](./papers/techniques/09-chain-of-thought/) (2022)
- 🔥 **CRITICAL** - Reasoning breakthrough
- Step-by-step reasoning prompts, "Let's think step by step"
- [Paper](https://arxiv.org/abs/2201.11903)

**8.** [Tree of Thoughts](./papers/techniques/25-tree-of-thoughts/) (2023)
- ⭐ **HIGH** - Advanced reasoning
- Tree search over reasoning paths, 18× better than CoT
- [Paper](https://arxiv.org/abs/2305.10601)

**9.** [Meta-CoT](./papers/techniques/34-meta-cot/) (2025)
- 🔥 **HIGH** - System 2 reasoning
- Metacognitive strategies, deliberate problem-solving
- [Paper](https://arxiv.org/abs/2501.xxxxx)

**10.** [rStar-Math](./papers/techniques/35-rstar-math/) (2025)
- 🔥 **HIGH** - Small models rival large ones
- MCTS for math, 7B model beats 70B+ competitors
- [Paper](https://arxiv.org/abs/2501.04519)

**11.** [Test-Time Compute Scaling](./papers/techniques/50-test-time-compute/) (2024)
- 🔥 **CRITICAL** - Theoretical foundation for reasoning models
- Think harder, not bigger - small model + more compute matches 14x larger model
- Compute-optimal strategies for easy vs. hard problems
- [Paper](https://arxiv.org/abs/2408.03314)

**12.** [Process Reward Models (Let's Verify Step by Step)](./papers/techniques/51-process-reward-models/) (2023)
- 🔥 **CRITICAL** - Step-by-step verification for reasoning
- Process supervision beats outcome supervision (78.2% vs 72.4% on MATH)
- PRM800K dataset, foundation for o1/R1 verification
- [Paper](https://arxiv.org/abs/2305.20050)

#### **RL Training Methods**

**11.** [GRPO](./papers/techniques/38-grpo/) (2024)
- 🔥 **CRITICAL** - The algorithm behind reasoning models
- No critic model needed, 50% less memory than PPO
- Powers DeepSeek-R1, industry standard for reasoning training
- [Paper](https://arxiv.org/abs/2402.03300)

**12.** [RLVR](./papers/techniques/39-rlvr/) (2024-2025)
- 🔥 **CRITICAL** - New training paradigm
- Verifiable rewards replace human preferences for reasoning
- Emergent reasoning from correctness signal alone
- [Key Paper](https://arxiv.org/abs/2501.12948)

#### **Agentic Capabilities**

**11.** [ReAct](./papers/techniques/21-react/) (2023)
- 🔥 **CRITICAL** - AI agents foundation
- Synergizing reasoning and acting, powers ChatGPT plugins
- [Paper](https://arxiv.org/abs/2210.03629)

**12.** [Toolformer](./papers/techniques/24-toolformer/) (2023)
- ⭐ **HIGH** - Self-taught tool use
- LLMs learn to use tools automatically, inspired ChatGPT function calling
- [Paper](https://arxiv.org/abs/2302.04761)

**13.** [Reflexion](./papers/techniques/83-reflexion/) (2023)
- ⭐ **HIGH** - Verbal RL for agent loops
- LLM agents reflect on failures, store reflections, retry
- [Paper](https://arxiv.org/abs/2303.11366)

**14.** [Self-Refine](./papers/techniques/84-self-refine/) (2023)
- ⭐ **HIGH** - LLM as its own critic
- Generate → critique → refine loop, no extra training
- [Paper](https://arxiv.org/abs/2303.17651)

**15.** [Voyager](./papers/techniques/86-voyager/) (2023)
- ⭐ **HIGH** - Landmark LLM Minecraft agent
- Self-expanding skill library + automatic curriculum
- [Paper](https://arxiv.org/abs/2305.16291)

**16.** [SWE-bench](./papers/techniques/91-swe-bench/) (2023)
- 🔥 **CRITICAL** - The coding-agent benchmark
- Real GitHub issues; SWE-bench Verified is the standard
- [Paper](https://arxiv.org/abs/2310.06770)

#### **Self-Taught Reasoning**

**17.** [Self-Consistency](./papers/techniques/85-self-consistency/) (2022)
- 🔥 **CRITICAL** - Majority vote over CoT samples
- Massive accuracy gains; foundation of test-time-compute strategies
- [Paper](https://arxiv.org/abs/2203.11171)

**18.** [STaR](./papers/techniques/81-star-self-taught-reasoner/) (2022)
- ⭐ **HIGH** - Bootstrap reasoning from correct answers
- Conceptual precursor to o1/R1-style RL on reasoning
- [Paper](https://arxiv.org/abs/2203.14465)

**19.** [Quiet-STaR](./papers/techniques/82-quiet-star/) (2024)
- ⭐ **HIGH** - Internal thoughts at every token
- Direct precursor to o1-style test-time thinking
- [Paper](https://arxiv.org/abs/2403.09629)

#### **Preference Optimization**

**20.** [KTO](./papers/techniques/93-kto/) (2024)
- ⭐ **HIGH** - Alignment from thumbs-up/down (not pairs)
- Based on Kahneman-Tversky prospect theory; simpler data than DPO
- [Paper](https://arxiv.org/abs/2402.01306)

#### **Interpretability**

**21.** [Sparse Autoencoders / Scaling Monosemanticity](./papers/techniques/90-sparse-autoencoders/) (2024)
- 🔥 **CRITICAL** - Millions of monosemantic features in Claude 3 Sonnet
- Landmark interpretability for production-scale models
- [Anthropic Report](https://transformer-circuits.pub/2024/scaling-monosemanticity/)

### 🔬 Scientific & World-Model AI

**1.** [AlphaZero](./papers/techniques/89-alphazero/) (2017)
- 🔥 **CRITICAL** - Self-play RL + MCTS, no human data
- Mastered Go/Chess/Shogi; conceptual ancestor of RLVR
- [Paper](https://arxiv.org/abs/1712.01815)

**2.** [AlphaFold 2](./papers/techniques/87-alphafold2/) (2021)
- 🔥 **CRITICAL** - Solved 50-year protein folding challenge
- ~atomic accuracy on CASP14; Nobel Prize 2024
- [Nature](https://www.nature.com/articles/s41586-021-03819-2)

**3.** [CICERO](./papers/techniques/97-cicero/) (2022)
- ⭐ **HIGH** - Human-level Diplomacy via LLM + planning
- Landmark for negotiation/multi-agent LLM systems
- [Science](https://www.science.org/doi/10.1126/science.ade9097)

**4.** [DreamerV3](./papers/techniques/95-dreamerv3/) (2023)
- ⭐ **HIGH** - First RL algorithm to mine diamonds in Minecraft
- Model-based RL with learned world model
- [Paper](https://arxiv.org/abs/2301.04104)

**5.** [ESM-2 / ESMFold](./papers/techniques/96-esm/) (2023)
- ⭐ **HIGH** - 15B-param protein language model
- Structure prediction without MSAs; scaling laws for biology
- [Science](https://www.science.org/doi/10.1126/science.ade2574)

**6.** [Genie](./papers/techniques/94-genie/) (2024)
- ⭐ **HIGH** - Foundation world model from videos
- Playable interactive 3D worlds from a single image prompt
- [Paper](https://arxiv.org/abs/2402.15391)

**7.** [AlphaFold 3](./papers/techniques/88-alphafold3/) (2024)
- 🔥 **CRITICAL** - Diffusion-based biomolecular interactions
- Proteins + ligands + DNA/RNA + ions; drug discovery
- [Nature](https://www.nature.com/articles/s41586-024-07487-w)

---

## 🎯 Learning Paths

### For Beginners
**Goal:** Understand what modern AI is and how it works
**Time:** 20-30 hours
**Path:** [Beginner Track](./docs/ROADMAP.md#path-1-complete-beginner)

1. Transformers → GPT-3 → Scaling Laws → LLaMA
2. GANs → Diffusion → Stable Diffusion → CLIP
3. InstructGPT → Chain-of-Thought → RAG → LoRA

### For Engineers
**Goal:** Build AI applications
**Time:** 15-20 hours
**Path:** [Engineer Track](./docs/ROADMAP.md#path-2-software-engineer)

1. **Sprint 1:** Transformers, ViT, Scaling Laws
2. **Sprint 2:** RAG, LoRA, Chain-of-Thought (focus here!)
3. **Sprint 3:** LLaMA, Alignment methods

### For Researchers
**Goal:** Deep technical understanding
**Time:** 30-40 hours
**Path:** [Researcher Track](./docs/ROADMAP.md#path-3-ml-studentresearcher)

1. **Phase 1:** Transformers, Scaling Laws, ViT (foundations)
2. **Phase 2:** BERT, GPT-3, LLaMA (training methods)
3. **Phase 3:** GANs, DDPM, Stable Diffusion, CLIP (generative)
4. **Phase 4:** InstructGPT, Constitutional AI, LoRA, RAG (alignment & efficiency)

### For Product Managers
**Goal:** Understand capabilities and trade-offs
**Time:** 10-15 hours
**Path:** [PM Track](./docs/ROADMAP.md#path-4-ai-product-manager)

Focus on "Why This Matters" sections + [Comparisons Guide](./docs/COMPARISONS.md)

---

## 📊 Quick Stats

| Category | Count | Total Reading Time |
|----------|-------|-------------------|
| **Papers** | 94 | 80-100 hours |
| **Words** | 600,000+ | - |
| **Guides** | 6 | 3-4 hours |
| **Terms Explained** | 250+ | - |

### By Year
- **2013:** 2 papers (Word2Vec, VAE)
- **2014:** 3 papers (GANs, Seq2Seq, Bahdanau Attention)
- **2015:** 1 paper (ResNet)
- **2017:** 3 papers (Transformers, VQ-VAE, AlphaZero)
- **2018:** 2 papers (BERT, GPT-1)
- **2019:** 2 papers (GPT-2, T5)
- **2020:** 5 papers (GPT-3, Scaling Laws, ViT, DDPM, RAG)
- **2021:** 4 papers (CLIP, LoRA, MAE, VQ-GAN, AlphaFold 2)
- **2022:** 13 papers (InstructGPT, Whisper, CoT, Stable Diffusion, Constitutional AI, FlashAttention, Chinchilla, DiT, Speculative Decoding, STaR, Self-Consistency, PaLM, Imagen, DreamBooth, CICERO)
- **2023:** 22 papers (GPT-4, LLaVA, DALL-E 3, PRMs, vLLM, LLaMA, LLaMA 2, Mamba, DPO, GPT-4V, ReAct, QLoRA, Toolformer, ToT, Mistral 7B, ControlNet, Self-Refine, Reflexion, Voyager, SWE-bench, Llama Guard, DreamerV3, ESM-2)
- **2024:** 17 papers (Mixtral/MoE, GPT-4o, Sora, GRPO, Test-Time Compute, DeepSeek-V3, o1, Claude 3.5 Sonnet, SAM 2, LLaMA 3.3, Quiet-STaR, KTO, AlphaFold 3, Sparse Autoencoders, Genie)
- **2025-2026:** 14 papers (DeepSeek-R1, RLVR, Qwen3, Gemini 2.5, Gemini 3, Llama 4, GPT-5, Claude 4, Meta-CoT, rStar-Math, AlphaGeometry, AlphaEvolve)

---

## 🌟 Key Concepts

### Self-Attention (Transformers)
Process all positions in parallel, enabling better context understanding.

### Scaling Laws
Predictable power-law relationships between model performance and compute/data/parameters.

### RLHF (InstructGPT)
Align models using human preferences as reward signal.

### Constitutional AI
Self-supervised alignment using explicit written principles.

### RAG (Retrieval-Augmented Generation)
Combine retrieval with generation to ground responses in facts.

### LoRA (Low-Rank Adaptation)
Efficient fine-tuning using small adapter matrices.

[See full glossary →](./docs/GLOSSARY.md)

---

## 🎓 What You'll Learn

After working through this repository, you'll be able to:

✅ Explain the key innovation of each foundational AI paper
✅ Choose the right technique for a given problem
✅ Understand trade-offs between different approaches
✅ Read new AI papers and understand them
✅ Build or deploy AI applications
✅ Critically evaluate AI products and claims

---

## 📖 Additional Resources

### Interactive Learning
- [The Illustrated Transformer](http://jalammar.github.io/illustrated-transformer/)
- [CLIP Playground](https://replicate.com/openai/clip)
- [Stable Diffusion Demo](https://huggingface.co/spaces/stabilityai/stable-diffusion)

### Code Implementations
- [Hugging Face Transformers](https://github.com/huggingface/transformers)
- [LLaMA](https://github.com/facebookresearch/llama)
- [Stable Diffusion](https://github.com/Stability-AI/stablediffusion)
- [LangChain (RAG)](https://github.com/langchain-ai/langchain)
- [PEFT (LoRA)](https://github.com/huggingface/peft)

### Communities
- [r/MachineLearning](https://reddit.com/r/MachineLearning)
- [Hugging Face Forums](https://discuss.huggingface.co/)
- [Papers with Code](https://paperswithcode.com/)

---

## 📜 Citation

If you use these summaries in your work:

```bibtex
@misc{genai-papers-summarized-2025,
  title={Foundational Generative AI Research Papers - Summarized},
  author={[Your Name]},
  year={2025},
  url={https://github.com/[your-repo]/genai-research-papers-summarized}
}
```

---

## ⭐ Star History

If you find this repository helpful, please consider giving it a star! It helps others discover these resources.

---

## 📄 License

This work is provided for educational purposes. Original papers retain their respective copyrights and licenses. Summaries are original interpretations created for accessibility and learning.

---

## 🙏 Acknowledgments

Gratitude to the researchers who created these foundational works:
- Google Research (Transformers, ViT)
- OpenAI (GPT-3, CLIP, Scaling Laws, InstructGPT)
- Meta AI (LLaMA)
- Anthropic (Constitutional AI)
- Stability AI (Stable Diffusion)
- And many more brilliant researchers

**Special thanks to the open-source AI community for making research accessible.**

---

**Last Updated:** 2026-05-26
**Papers:** 94 foundational works (2013-2026)
**Total Content:** 600,000+ words
**Includes:** Foundations (Word2Vec, VAE, Seq2Seq, ResNet, GPT-1/2, T5, PaLM), latest frontier models (GPT-5, Claude 4, Llama 4, Gemini 3), reasoning era (o1, R1, STaR, Quiet-STaR), scientific AI (AlphaFold 2/3, AlphaZero, ESM, CICERO, Genie), and more.
**Repository:** [github.com/PatrickWiloak/genai-research-papers-summarized](https://github.com/PatrickWiloak/genai-research-papers-summarized)
