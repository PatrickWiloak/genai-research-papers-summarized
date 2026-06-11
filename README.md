# Foundational Generative AI Research Papers - Summarized

A curated collection of the **68 most influential papers** that shaped modern generative AI, with comprehensive summaries designed to make cutting-edge research accessible to everyone.

[![Papers](https://img.shields.io/badge/Papers-68-blue.svg)](./INDEX.md)
[![Guides](https://img.shields.io/badge/Guides-6-green.svg)](./docs/ROADMAP.md)
[![License](https://img.shields.io/badge/License-CC_BY_4.0-orange.svg)](./LICENSE)
[![Updated](https://img.shields.io/badge/Updated-June_2026-green.svg)](./README.md)

---

## 🚀 Quick Start

**New to AI?** Start with the [Learning Roadmap](./docs/ROADMAP.md)
**Want the full list?** See the [Paper Index](./INDEX.md) - all 68 papers grouped by category
**Exploring a topic?** Browse [by topic tag](./TAGS.md) (reasoning, RLHF, efficiency, agents, multimodal, ...)
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
- 📖 **[Quick Reference](./docs/QUICK_REFERENCE.md)** - One-page overview of the papers
- 🔍 **[Comparisons](./docs/COMPARISONS.md)** - Side-by-side analysis and trade-offs
- 📚 **[Glossary](./docs/GLOSSARY.md)** - 150+ key terms explained
- 🗂️ **[Paper Index](./INDEX.md)** - Complete category-grouped list (generated)
- 🏷️ **[Browse by Topic](./TAGS.md)** - Tag-filtered index across 37 topics (generated)

### Data & Tooling
- 🧾 **[papers.json](./papers.json)** / **[papers.csv](./papers.csv)** - machine-readable manifest of every paper
- 🛠️ **[scripts/build_manifest.py](./scripts/build_manifest.py)** - regenerates frontmatter, the manifest, `INDEX.md`, and the site nav
- 🌐 **Docs site** - run `pip install -r requirements.txt && mkdocs serve` for a searchable browsable version (auto-deploys to GitHub Pages on push)
- 🤝 **[CONTRIBUTING.md](./CONTRIBUTING.md)** - how to add a paper, house style, and the build workflow

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
│   │   └── 37-mixture-of-experts/
│   ├── language-models/               # LLM papers
│   │   ├── 03-bert/
│   │   ├── 04-gpt3-few-shot-learners/
│   │   ├── 05-instructgpt-rlhf/
│   │   ├── 14-constitutional-ai/
│   │   ├── 15-llama/
│   │   ├── 17-llama2/
│   │   ├── 19-dpo/
│   │   ├── 36-gpt4/
│   │   ├── 40-gpt4o/
│   │   ├── 41-llama4/
│   │   ├── 42-gpt5/
│   │   └── 43-claude4/
│   ├── image-generation/              # Image generation papers
│   │   ├── 02-generative-adversarial-networks/
│   │   ├── 06-diffusion-models/
│   │   ├── 07-stable-diffusion/
│   │   ├── 44-sora-dit/
│   │   └── 48-dalle3/
│   ├── multimodal/                    # Cross-modal papers
│   │   ├── 08-clip/
│   │   ├── 23-gpt4v/
│   │   ├── 46-llava/
│   │   ├── 47-gemini3/
│   │   └── 49-whisper/
│   └── techniques/                    # Methods and techniques
│       ├── 09-chain-of-thought/
│       ├── 10-lora/
│       ├── 12-scaling-laws/
│       ├── 13-rag/
│       ├── 16-flash-attention/
│       ├── 18-chinchilla/
│       ├── 21-react/
│       ├── 22-qlora/
│       ├── 24-toolformer/
│       ├── 25-tree-of-thoughts/
│       ├── 38-grpo/
│       ├── 39-rlvr/
│       ├── 45-speculative-decoding/
│       ├── 50-test-time-compute/
│       ├── 51-process-reward-models/
│       └── 52-pagedattention-vllm/
└── resources/                         # Additional resources
    ├── images/                        # Diagrams and visualizations
    └── notebooks/                     # Jupyter notebooks
```

---

## 📄 Papers by Category

### 🏗️ Foundational Architectures
**Recommended Reading Order:** 1 → 2 → 3

**1. Start Here:** [Attention Is All You Need](./papers/architectures/01-attention-is-all-you-need/summary.md) (2017)
- 🔥 **CRITICAL** - Foundation of everything
- Introduced Transformer architecture
- Self-attention mechanism
- **Read this first** - Everything else builds on this
- [Paper](https://arxiv.org/abs/1706.03762)

**2. Then:** [Vision Transformer (ViT)](./papers/architectures/11-vision-transformer/summary.md) (2020)
- ⭐ **HIGH** - Transformers for computer vision
- Images as patch sequences
- Enables multimodal models
- [Paper](https://arxiv.org/abs/2010.11929)

**3. Alternative Architecture:** [Mamba](./papers/architectures/20-mamba/summary.md) (2023)
- 🔥 **CRITICAL** - First viable Transformer alternative
- Linear-time sequence modeling (O(n) vs O(n²))
- Selective state spaces
- [Paper](https://arxiv.org/abs/2312.00752)

**4. Sparse Architecture:** [Mixture-of-Experts (Mixtral)](./papers/architectures/37-mixture-of-experts/summary.md) (2024)
- 🔥 **CRITICAL** - Architecture behind every frontier model
- 47B params, 13B active - matches LLaMA 2 70B
- Sparse routing, expert specialization
- Now used by DeepSeek-V3, Llama 4, Qwen3
- [Paper](https://arxiv.org/abs/2401.04088)

### 🤖 Language Models
**Recommended Reading Order:** Evolution → Frontier → Alignment → Open Source → Reasoning → Unified

#### **Early Evolution (Historical Context)**

**1.** [BERT](./papers/language-models/03-bert/summary.md) (2018)
- 📚 **HISTORICAL** - Pre-training revolution
- Bidirectional pre-training, masked language modeling
- [Paper](https://arxiv.org/abs/1810.04805)

**2.** [GPT-3](./papers/language-models/04-gpt3-few-shot-learners/summary.md) (2020)
- ⭐ **HIGH** - Few-shot learning paradigm
- 175B parameters, foundation for ChatGPT
- [Paper](https://arxiv.org/abs/2005.14165)

#### **The Frontier Era**

**3.** [GPT-4](./papers/language-models/36-gpt4/summary.md) (2023)
- 🔥 **CRITICAL** - Defined the frontier model era
- Passed bar exam (90th percentile), MMLU 86.4%
- First multimodal GPT, predictable scaling
- [Paper](https://arxiv.org/abs/2303.08774)

**4.** [GPT-4o](./papers/language-models/40-gpt4o/summary.md) (2024)
- 🔥 **CRITICAL** - First native omni-model
- Text, audio, image in one model, 232ms voice latency
- 2x faster, 50% cheaper than GPT-4 Turbo
- [System Card](https://cdn.openai.com/gpt-4o-system-card.pdf)

#### **Alignment Methods (How to Make Them Helpful)**

**3.** [InstructGPT (RLHF)](./papers/language-models/05-instructgpt-rlhf/summary.md) (2022)
- 🔥 **CRITICAL** - Human preference learning
- Enabled ChatGPT
- [Paper](https://arxiv.org/abs/2203.02155)

**4.** [Constitutional AI](./papers/language-models/14-constitutional-ai/summary.md) (2022)
- ⭐ **HIGH** - Alternative to RLHF
- AI self-critique via principles, powers Claude
- [Paper](https://arxiv.org/abs/2212.08073)

**5.** [DPO](./papers/language-models/19-dpo/summary.md) (2023)
- 🔥 **CRITICAL** - Simpler than RLHF
- No reward model needed
- [Paper](https://arxiv.org/abs/2305.18290)

#### **Open-Source Revolution (2023)**

**6.** [LLaMA](./papers/language-models/15-llama/summary.md) (2023)
- 🔥 **CRITICAL** - Compute-optimal training
- 13B matches GPT-3 175B
- [Paper](https://arxiv.org/abs/2302.13971)

**7.** [LLaMA 2](./papers/language-models/17-llama2/summary.md) (2023)
- 🔥 **CRITICAL** - Production-ready open model
- Commercial license, RLHF alignment
- [Paper](https://arxiv.org/abs/2307.09288)

**8.** [LLaMA 3.3](./papers/language-models/33-llama3.3/summary.md) (2024)
- 🔥 **HIGH** - Distillation breakthrough
- 70B matches 405B performance
- [Paper](https://www.meta.ai/blog/meta-llama-3-3/)

**9.** [Llama 4](./papers/language-models/41-llama4/summary.md) (2025)
- 🔥 **HIGH** - First open-source multimodal MoE
- Scout: 10M token context, Maverick: beats GPT-4o
- 17B active params, natively multimodal
- [Blog](https://ai.meta.com/blog/llama-4-multimodal-intelligence/)

#### **Efficiency Breakthroughs (2024)**

**9.** [DeepSeek-V3](./papers/language-models/27-deepseek-v3/summary.md) (2024)
- 🔥 **CRITICAL** - $5.76M training cost
- 671B MoE, matches GPT-4
- [Paper](https://arxiv.org/abs/2412.19437)

#### **Reasoning Era (2024-2025)**

**11.** [OpenAI o1](./papers/language-models/31-openai-o1/summary.md) (2024)
- 🔥 **CRITICAL** - Started reasoning model era
- PhD-level performance, RL for reasoning
- [Announcement](https://openai.com/index/learning-to-reason-with-llms/)

**11.** [DeepSeek-R1](./papers/language-models/26-deepseek-r1/summary.md) (2025)
- 🔥 **CRITICAL** - Pure RL reasoning
- Matches o1, fully open source
- [Paper](https://arxiv.org/abs/2501.12948)

**12.** [Qwen3](./papers/language-models/28-qwen3/summary.md) (2025)
- 🔥 **CRITICAL** - Unified thinking/non-thinking
- Adaptive reasoning, beats competitors
- [Paper](https://arxiv.org/abs/2505.09388)

**15.** [Claude 3.5 Sonnet](./papers/language-models/30-claude-3.5-sonnet/summary.md) (2024)
- 🔥 **CRITICAL** - Computer use capability
- Best coding model (49% SWE-Bench)
- [Announcement](https://www.anthropic.com/news/3-5-models-and-computer-use)

#### **Unified Intelligence (2025-2026)**

**16.** [GPT-5](./papers/language-models/42-gpt5/summary.md) (2025)
- 🔥 **CRITICAL** - Unified fast + reasoning model
- 94.6% AIME, 74.9% SWE-bench, 80% fewer hallucinations
- Adaptive routing between fast and deep thinking
- [System Card](https://cdn.openai.com/gpt-5-system-card.pdf)

**17.** [Claude 4 Family](./papers/language-models/43-claude4/summary.md) (2025-2026)
- 🔥 **CRITICAL** - Agentic AI leader
- 80.9% SWE-bench (Opus 4.5), best coding model
- Multi-agent orchestration, extended thinking
- [Announcement](https://www.anthropic.com/news/claude-4)

### 🎨 Image Generation
**Recommended Reading Order:** GANs (historical) → Diffusion theory → Practical implementation

**1.** [GANs](./papers/image-generation/02-generative-adversarial-networks/summary.md) (2014)
- 📚 **HISTORICAL** - Generative modeling origins
- Adversarial training: generator vs discriminator
- [Paper](https://arxiv.org/abs/1406.2661)

**2.** [Diffusion Models (DDPM)](./papers/image-generation/06-diffusion-models/summary.md) (2020)
- 📖 **THEORY** - Diffusion foundations
- Iterative denoising, better than GANs
- [Paper](https://arxiv.org/abs/2006.11239)

**3.** [Stable Diffusion](./papers/image-generation/07-stable-diffusion/summary.md) (2022)
- ⭐ **HIGH** - Practical implementation
- Latent space diffusion (10-100× faster)
- Open-source, democratized AI art
- [Paper](https://arxiv.org/abs/2112.10752)

**4.** [DALL-E 3](./papers/image-generation/48-dalle3/summary.md) (2023)
- ⭐ **HIGH** - Solved prompt adherence
- Better captions = better images, first readable text in images
- ChatGPT integration eliminated prompt engineering
- [Paper](https://cdn.openai.com/papers/dall-e-3.pdf)

**5.** [Sora / DiT](./papers/image-generation/44-sora-dit/summary.md) (2024)
- 🔥 **CRITICAL** - Video generation + Diffusion Transformers
- Transformers replaced U-Net in diffusion models
- Spacetime patches enable flexible video generation
- [DiT Paper](https://arxiv.org/abs/2212.09748) | [Sora Report](https://openai.com/index/video-generation-models-as-world-simulators/)

### 🔗 Multimodal
**Recommended Reading Order:** Vision-language bridge → Practical multimodal → Next-gen unified AI

**1.** [CLIP](./papers/multimodal/08-clip/summary.md) (2021)
- ⭐ **HIGH** - Vision-language bridge
- Vision-language contrastive learning
- Zero-shot image classification, powers text-to-image models
- [Paper](https://arxiv.org/abs/2103.00020)

**2.** [Whisper](./papers/multimodal/49-whisper/summary.md) (2022)
- 🔥 **CRITICAL** - Foundation model for speech
- 680K hours of training data, 99 languages, zero-shot robustness
- 50% fewer errors than specialized models
- [Paper](https://arxiv.org/abs/2212.04356)

**3.** [LLaVA](./papers/multimodal/46-llava/summary.md) (2023)
- 🔥 **HIGH** - Blueprint for open-source multimodal
- Vision encoder + projection + LLM = visual assistant
- 85% of GPT-4V quality, spawned dozens of derivatives
- [Paper](https://arxiv.org/abs/2304.08485)

**4.** [GPT-4V(ision)](./papers/multimodal/23-gpt4v/summary.md) (2023)
- 🔥 **CRITICAL** - Multimodal frontier model
- GPT-4 with vision capabilities
- State-of-the-art VQA and OCR, real-world applications
- [Paper](https://cdn.openai.com/papers/GPTV_System_Card.pdf)

**5.** [SAM 2](./papers/multimodal/32-sam2/summary.md) (2024)
- 🔥 **HIGH** - Universal video segmentation
- 44 FPS real-time performance
- Zero-shot generalization across domains
- [Paper](https://arxiv.org/abs/2408.00714)

**6.** [Gemini 2.5](./papers/multimodal/29-gemini-2.5/summary.md) (2025)
- 🔥 **CRITICAL** - Advanced multimodal AI
- Native multimodal (text, image, audio, video)
- 1M context, 3-hour video understanding, integrated thinking mode
- [Paper](https://arxiv.org/abs/2507.06261)

**7.** [Gemini 3](./papers/multimodal/47-gemini3/summary.md) (2025)
- 🔥 **CRITICAL** - First model to cross 1500 LMArena ELO
- 91.8% MMLU, 95% AIME, best video understanding
- Deep Think mode: 45.1% ARC-AGI-2
- [Announcement](https://blog.google/products-and-platforms/products/gemini/gemini-3/)

### ⚡ Techniques & Methods
**Recommended Reading Order:** Scaling foundations → Efficiency → Reasoning → Agents

#### **Scaling Foundations (Start Here)**

**1.** [Scaling Laws](./papers/techniques/12-scaling-laws/summary.md) (2020)
- 🔥 **CRITICAL** - Predictive theory
- Predictable power laws, guides compute allocation
- [Paper](https://arxiv.org/abs/2001.08361)

**2.** [Chinchilla](./papers/techniques/18-chinchilla/summary.md) (2022)
- 🔥 **CRITICAL** - Rewrote scaling laws
- Equal scaling of params and tokens, proved GPT-3 was undertrained 4×
- [Paper](https://arxiv.org/abs/2203.15556)

#### **Efficiency Techniques**

**3.** [FlashAttention](./papers/techniques/16-flash-attention/summary.md) (2022)
- 🔥 **CRITICAL** - IO-aware attention
- 10-20× faster, enables 64k+ context lengths
- [Paper](https://arxiv.org/abs/2205.14135)

**4.** [LoRA](./papers/techniques/10-lora/summary.md) (2021)
- 🔥 **CRITICAL** - Efficient fine-tuning
- Low-rank adaptation, 10,000× fewer trainable parameters
- [Paper](https://arxiv.org/abs/2106.09685)

**5.** [QLoRA](./papers/techniques/22-qlora/summary.md) (2023)
- 🔥 **CRITICAL** - Efficient fine-tuning at scale
- 4-bit quantization + LoRA, 16× memory reduction
- [Paper](https://arxiv.org/abs/2305.14314)

#### **Inference Optimization**

**6.** [Speculative Decoding](./papers/techniques/45-speculative-decoding/summary.md) (2023)
- 🔥 **CRITICAL** - 2-3x faster inference, identical output
- Draft model guesses, target model verifies in parallel
- Used by every major LLM provider
- [Paper](https://arxiv.org/abs/2211.17192)

**7.** [PagedAttention / vLLM](./papers/techniques/52-pagedattention-vllm/summary.md) (2023)
- 🔥 **CRITICAL** - Made LLM serving practical
- Virtual memory for GPU KV-cache, 24x throughput improvement
- Near-zero memory waste, powers most production LLM deployments
- [Paper](https://arxiv.org/abs/2309.06180)

#### **Production Techniques**

**7.** [RAG](./papers/techniques/13-rag/summary.md) (2020)
- 🔥 **CRITICAL** - Production standard
- Retrieval-augmented generation, reduces hallucinations
- [Paper](https://arxiv.org/abs/2005.11401)

#### **Reasoning Methods**

**7.** [Chain-of-Thought](./papers/techniques/09-chain-of-thought/summary.md) (2022)
- 🔥 **CRITICAL** - Reasoning breakthrough
- Step-by-step reasoning prompts, "Let's think step by step"
- [Paper](https://arxiv.org/abs/2201.11903)

**8.** [Tree of Thoughts](./papers/techniques/25-tree-of-thoughts/summary.md) (2023)
- ⭐ **HIGH** - Advanced reasoning
- Tree search over reasoning paths, 18× better than CoT
- [Paper](https://arxiv.org/abs/2305.10601)

**9.** [Meta-CoT](./papers/techniques/34-meta-cot/summary.md) (2025)
- 🔥 **HIGH** - System 2 reasoning
- Metacognitive strategies, deliberate problem-solving
- [Paper](https://arxiv.org/abs/2501.xxxxx)

**10.** [rStar-Math](./papers/techniques/35-rstar-math/summary.md) (2025)
- 🔥 **HIGH** - Small models rival large ones
- MCTS for math, 7B model beats 70B+ competitors
- [Paper](https://arxiv.org/abs/2501.04519)

**11.** [Test-Time Compute Scaling](./papers/techniques/50-test-time-compute/summary.md) (2024)
- 🔥 **CRITICAL** - Theoretical foundation for reasoning models
- Think harder, not bigger - small model + more compute matches 14x larger model
- Compute-optimal strategies for easy vs. hard problems
- [Paper](https://arxiv.org/abs/2408.03314)

**12.** [Process Reward Models (Let's Verify Step by Step)](./papers/techniques/51-process-reward-models/summary.md) (2023)
- 🔥 **CRITICAL** - Step-by-step verification for reasoning
- Process supervision beats outcome supervision (78.2% vs 72.4% on MATH)
- PRM800K dataset, foundation for o1/R1 verification
- [Paper](https://arxiv.org/abs/2305.20050)

#### **RL Training Methods**

**11.** [GRPO](./papers/techniques/38-grpo/summary.md) (2024)
- 🔥 **CRITICAL** - The algorithm behind reasoning models
- No critic model needed, 50% less memory than PPO
- Powers DeepSeek-R1, industry standard for reasoning training
- [Paper](https://arxiv.org/abs/2402.03300)

**12.** [RLVR](./papers/techniques/39-rlvr/summary.md) (2024-2025)
- 🔥 **CRITICAL** - New training paradigm
- Verifiable rewards replace human preferences for reasoning
- Emergent reasoning from correctness signal alone
- [Key Paper](https://arxiv.org/abs/2501.12948)

#### **Agentic Capabilities**

**11.** [ReAct](./papers/techniques/21-react/summary.md) (2023)
- 🔥 **CRITICAL** - AI agents foundation
- Synergizing reasoning and acting, powers ChatGPT plugins
- [Paper](https://arxiv.org/abs/2210.03629)

**12.** [Toolformer](./papers/techniques/24-toolformer/summary.md) (2023)
- ⭐ **HIGH** - Self-taught tool use
- LLMs learn to use tools automatically, inspired ChatGPT function calling
- [Paper](https://arxiv.org/abs/2302.04761)

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
| **Papers** | 68 | 58-70 hours |
| **Words** | 390,000+ | - |
| **Guides** | 6 | 3-4 hours |
| **Terms Explained** | 250+ | - |

### By Year
_This distribution is generated from [`papers.json`](./papers.json) - see [INDEX.md](./INDEX.md) for the full clickable list._

- 2013: 2 papers (Word2Vec, VAE)
- 2014: 3 papers (GANs, Seq2Seq, Bahdanau Attention)
- 2017: 2 papers (Transformers, PPO)
- 2018: 1 paper (BERT)
- 2019: 2 papers (GPT-2, T5)
- 2020: 5 papers (GPT-3, DDPM, ViT, Scaling Laws, RAG)
- 2021: 7 papers (Stable Diffusion, CLIP, LoRA, RoPE, Codex, Switch Transformer, AlphaFold)
- 2022: 9 papers (InstructGPT, Chain-of-Thought, Constitutional AI, FlashAttention, Chinchilla, ReAct, DiT/Sora, Speculative Decoding, Whisper)
- 2023: 14 papers (LLaMA, LLaMA 2, DPO, Mamba, QLoRA, GPT-4V, Toolformer, Tree of Thoughts, GPT-4, LLaVA, DALL-E 3, Process Reward Models, PagedAttention/vLLM, Generative Agents)
- 2024: 13 papers (DeepSeek-V3, Claude 3.5 Sonnet, o1, SAM 2, LLaMA 3.3, Mixtral/MoE, GRPO, RLVR, GPT-4o, Test-Time Compute, MCP, GraphRAG, AlphaGeometry)
- 2025: 10 papers (DeepSeek-R1, Qwen3, Gemini 2.5, Meta-CoT, rStar-Math, Llama 4, GPT-5, Claude 4, Gemini 3, AlphaEvolve)

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

**Last Updated:** 2026-06-10
**Papers:** 68 foundational works (2013-2026)
**Total Content:** 390,000+ words
**Includes:** Roots (Word2Vec, Seq2Seq, VAE, PPO) through the latest breakthroughs of early 2026 (GPT-5, Claude 4, Llama 4, GRPO, RLVR, and more)
**Repository:** [github.com/PatrickWiloak/genai-research-papers-summarized](https://github.com/PatrickWiloak/genai-research-papers-summarized)
