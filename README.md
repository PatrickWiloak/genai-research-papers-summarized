<!-- markdown="1" lets the documentation site render the markdown inside this
     centered block. GitHub ignores the attribute, so rendering there is unchanged. -->
<div align="center" markdown="1">

<a href="https://noblerworks.com/"><img src="https://raw.githubusercontent.com/Noblerworks/IRONSIGHT/main/nobler-works-banner.JPG" alt="Nobler Works" width="240"></a>

### Built by [Patrick Wiloak](https://patrickwiloak.com) at [Nobler Works](https://noblerworks.com/)

We build custom software and products at Nobler Works. Open source projects and research libraries like this one are our way of giving back - we're nothing without the community that supports us.<br>
If you need custom software built, [get in touch](https://noblerworks.com/).

[![Website](https://img.shields.io/badge/Website-000000?style=for-the-badge&logo=googlechrome&logoColor=white)](https://noblerworks.com/)
[![X](https://img.shields.io/badge/X-000000?style=for-the-badge&logo=x&logoColor=white)](https://x.com/Nobler_Works)
[![YouTube](https://img.shields.io/badge/YouTube-FF0000?style=for-the-badge&logo=youtube&logoColor=white)](https://www.youtube.com/@NoblerWorks)
[![TikTok](https://img.shields.io/badge/TikTok-000000?style=for-the-badge&logo=tiktok&logoColor=white)](https://www.tiktok.com/@noblerworks)
[![Threads](https://img.shields.io/badge/Threads-000000?style=for-the-badge&logo=threads&logoColor=white)](https://www.threads.com/@noblerworks)

---

<a href="https://gitgood.dev"><img src="./assets/brand/gitgood-banner.png" alt="gitGood.dev - practice questions, coding challenges, system design and cloud certification practice exams for engineers, data, security, DevOps and product technologists" width="280"></a>

### 🚀 Read the papers here. Prove you understood them there.

**[gitGood.dev](https://gitgood.dev) is our flagship tech training platform - train for the role, not just the interview.**

Built for **every kind of technologist**, across 21 role-targeted learning paths - including machine learning and AI engineering, data science, and data engineering. This repo gives you the theory. gitGood gives you the reps, and tells you whether you actually know it.

[![Start 10 days free](https://img.shields.io/badge/Start%2010%20Days%20Free-000000?style=for-the-badge&logo=rocket&logoColor=white)](https://gitgood.dev)

**$5/month** or **$40/year** after the trial. Free tier needs no card: 20 practice questions, the free coding challenges, streaks, achievements, and the job board.

<details class="promo" markdown="1">
<summary>What's inside gitGood.dev</summary>

**Roles** - backend, frontend, full-stack and mobile development, SRE and DevOps, platform engineering, cloud and solutions architecture, security engineering, machine learning and AI engineering, data science, data analytics, data engineering, QA and SDET, engineering management, product management, TPM, and new grads breaking into tech.

**Cloud certification practice exams** - 13 banks and 608 original scenario-style questions (never brain dumps), with per-domain scoring and a timed exam simulator: AWS Cloud Practitioner CLF-C02, Solutions Architect Associate SAA-C03 and Developer Associate DVA-C02; Azure AZ-900, AZ-104, AZ-305 and AI-900; Google Cloud Digital Leader and Associate Cloud Engineer; Kubernetes CKA and CKAD; CompTIA Security+; HashiCorp Terraform Associate.

**Everything else** - 1,950+ practice questions across 34 categories · coding challenges with real sandboxed execution in JavaScript, Python and TypeScript · a 45-problem SQL playground · 32 system design walkthroughs including LLM inference and RAG · behavioral and STAR interview prep · curated interview packs for 39 companies · AI mock interviews · AI resume reviews · salary negotiation coaching · a live tech job-market pulse and job board.

</details>

---

### 📚 Also from Nobler Works

**[Cloud, Data, AI & Security - Zero to Hero](https://github.com/PatrickWiloak/cloud-data-ai-security-zero-to-hero)** - the practitioner companion to this repo. 148 certification study guides across 27 providers, hands-on builds (RAG, MCP agents, vLLM, evals, LoRA), and plain-English concept pages. These summaries explain *why* the techniques work; that repo walks you through *building* with them.

</div>

---

# Foundational Generative AI Research Papers - Summarized

A curated collection of the **107 most influential papers** that shaped modern generative AI, with comprehensive summaries designed to make cutting-edge research accessible to everyone.

[![Papers](https://img.shields.io/badge/Papers-107-blue.svg)](./INDEX.md)
[![Guides](https://img.shields.io/badge/Guides-7-green.svg)](./docs/ROADMAP.md)
[![License](https://img.shields.io/badge/License-CC_BY_4.0-orange.svg)](./LICENSE)
[![Updated](https://img.shields.io/badge/Updated-August_2026-green.svg)](./README.md)

---

## 🚀 Quick Start

**New to AI?** Start with the [Learning Roadmap](./docs/ROADMAP.md)
**Want the full list?** See the [Paper Index](./INDEX.md) - all 107 papers grouped by category
**Exploring a topic?** Browse [by topic tag](./TAGS.md) (reasoning, RLHF, efficiency, agents, multimodal, ...)
**Want to browse visually?** See [BROWSE.md](./BROWSE.md) for table/grid view
**Not sure what to read?** Check the [Reading Guide](./docs/READING_GUIDE.md) - Historical vs Modern relevance
**Need quick lookup?** See the [Quick Reference](./docs/QUICK_REFERENCE.md)
**Building something?** Read [Comparisons](./docs/COMPARISONS.md) for decision guides
**Confused by terms?** Browse the [Glossary](./docs/GLOSSARY.md)
**Wondering what is missing?** See [Coverage & Gaps](./docs/GAPS.md) - what this collection covers and what is queued next

---

## 📚 Documentation

### Core Guides
- 🗺️ **[Learning Roadmap](./docs/ROADMAP.md)** - Structured paths from beginner to expert (4 tracks)
- ⭐ **[Reading Guide](./docs/READING_GUIDE.md)** - What's still relevant vs historical context
- 📖 **[Quick Reference](./docs/QUICK_REFERENCE.md)** - One-page overview of the papers
- 🔍 **[Comparisons](./docs/COMPARISONS.md)** - Side-by-side analysis and trade-offs
- 📚 **[Glossary](./docs/GLOSSARY.md)** - 150+ key terms explained
- 🧭 **[Coverage & Gaps](./docs/GAPS.md)** - what the collection covers, and the papers queued to be added next
- 🗂️ **[Paper Index](./INDEX.md)** - Complete category-grouped list (generated)
- 🏷️ **[Browse by Topic](./TAGS.md)** - Tag-filtered index across 45 topics (generated)

### Data & Tooling
- 🧾 **[papers.json](./papers.json)** / **[papers.csv](./papers.csv)** - machine-readable manifest of every paper
- 🛠️ **[scripts/build_manifest.py](./scripts/build_manifest.py)** - regenerates frontmatter, the manifest, `INDEX.md`, and the site nav
- 🌐 **Docs site** - read it all at **[patrickwiloak.github.io/genai-research-papers-summarized](https://patrickwiloak.github.io/genai-research-papers-summarized/)** - searchable, dark mode, no signup. See [CONTRIBUTING.md](./CONTRIBUTING.md#previewing-the-site-locally) to build it locally.
- 🤝 **[CONTRIBUTING.md](./CONTRIBUTING.md)** - how to add a paper, house style, and the build workflow

---

## 📁 Repository Structure

```
genai-research-papers-summarized/
├── README.md                          # You are here
├── INDEX.md / BROWSE.md / TAGS.md     # Generated indexes (category, grid, topic tag)
├── papers.json / papers.csv           # Generated machine-readable manifest
├── mkdocs.yml                         # Hand-maintained site config (theme, palette); no nav
├── requirements.txt                   # Pinned docs toolchain
├── docs/                              # Guides
│   ├── ROADMAP.md                     # Learning paths
│   ├── READING_GUIDE.md               # Historical vs modern relevance
│   ├── QUICK_REFERENCE.md             # Quick lookup
│   ├── COMPARISONS.md                 # Decision guides
│   ├── GLOSSARY.md                    # Term definitions
│   └── GAPS.md                        # Coverage map + what is queued next
├── scripts/                           # Stdlib-only regeneration pipeline
│   ├── build_manifest.py              # Frontmatter, manifest, INDEX, TAGS, site nav + tree
│   ├── add_cross_links.py             # "Related in This Collection" footers
│   └── check_links.py                 # Relative-link validation (CI gate)
├── .github/site/                      # Site-only chrome (landing page + stylesheet)
├── assets/brand/                      # Banner images used by this README
└── papers/                            # All 107 summaries, grouped by category
    ├── architectures/       # Foundational architectures (11)
    ├── language-models/     # Language models (25)
    ├── image-generation/    # Image & video generation (14)
    ├── multimodal/          # Multimodal (7)
    └── techniques/          # Techniques & methods (50)
```

Each paper is its own directory holding a single `summary.md`, numbered in the
order it entered the collection. For the full list with titles, authors, and
years, see the generated [Paper Index](./INDEX.md) - it never goes stale.

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

**5. The Part Every Model Inherits:** [ResNet](./papers/architectures/73-resnet/summary.md) (2015)
- 🔥 **CRITICAL** - Residual connections made depth possible
- `x + f(x)` is inside every Transformer block ever trained
- 152 layers when 20 was the limit; most-cited paper in deep learning
- [Paper](https://arxiv.org/abs/1512.03385)

**6. The Diffusion Backbone:** [U-Net](./papers/architectures/74-unet/summary.md) (2015)
- ⭐ **HIGH** - Encoder-decoder with skip connections
- Denoiser inside DDPM, Stable Diffusion, DALL-E 2, Imagen
- Written for microscopy; became the image-generation workhorse
- [Paper](https://arxiv.org/abs/1505.04597)

**7. The KV-Cache Fix:** [Grouped-Query Attention](./papers/architectures/75-grouped-query-attention/summary.md) (2023)
- 🔥 **CRITICAL** - 8x smaller KV cache, near-zero quality cost
- In Llama 2/3/4, Mistral, Mixtral, Qwen, Gemma
- Why long context and local inference are affordable
- [Paper](https://arxiv.org/abs/2305.13245)

_Also in this category:_ [Seq2Seq](./papers/architectures/55-seq2seq/summary.md) (2014), [Bahdanau Attention](./papers/architectures/66-bahdanau-attention/summary.md) (2014), [Switch Transformers](./papers/architectures/67-switch-transformer/summary.md) (2021), [Masked Autoencoders (MAE)](./papers/architectures/88-mae/summary.md) (2021). See [INDEX.md](./INDEX.md) for the complete list.

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

_Also in this category:_ [GPT-1](./papers/language-models/93-gpt1/summary.md) (2018) - where generative pre-training started, [PaLM](./papers/language-models/94-palm/summary.md) (2022), [Mistral 7B](./papers/language-models/95-mistral-7b/summary.md) (2023), [Llama Guard](./papers/language-models/96-llama-guard/summary.md) (2023). See [INDEX.md](./INDEX.md) for the complete list.

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

**6. The Prompt-Following Trick:** [Classifier-Free Guidance](./papers/image-generation/69-classifier-free-guidance/summary.md) (2021-2022)
- 🔥 **CRITICAL** - The "CFG scale" slider in every image tool
- Two predictions, one difference vector, prompts finally obeyed
- Also the mechanism behind negative prompts
- [Paper](https://arxiv.org/abs/2207.12598)

**7. The Speed Fix:** [DDIM](./papers/image-generation/70-ddim/summary.md) (2020)
- 🔥 **CRITICAL** - 1,000 sampling steps down to 20, no retraining
- Deterministic sampling; the basis of image editing and inversion
- Ancestor of every sampler in your image UI
- [Paper](https://arxiv.org/abs/2010.02502)

**8. Structural Control:** [ControlNet](./papers/image-generation/71-controlnet/summary.md) (2023)
- 🔥 **CRITICAL** - Edges, depth, pose, scribbles as conditioning
- Zero convolutions: add capability without breaking the base model
- ICCV 2023 best paper; turned image generation into a production tool
- [Paper](https://arxiv.org/abs/2302.05543)

**9. What Replaced Diffusion:** [Flow Matching / Rectified Flow (SD3)](./papers/image-generation/72-flow-matching-sd3/summary.md) (2022-2024)
- 🔥 **CRITICAL** - The current training objective for frontier image and video models
- Straight noise-to-image paths, fewer steps, simpler objective
- Powers Stable Diffusion 3/3.5 and Flux; MMDiT fixed text rendering
- [Flow Matching](https://arxiv.org/abs/2210.02747) | [SD3](https://arxiv.org/abs/2403.03206)

_Also in this category:_ [VAE](./papers/image-generation/57-vae/summary.md) (2013), [VQ-VAE](./papers/image-generation/89-vq-vae/summary.md) (2017), [VQ-GAN](./papers/image-generation/90-vq-gan/summary.md) (2020), [Imagen](./papers/image-generation/91-imagen/summary.md) (2022), [DreamBooth](./papers/image-generation/92-dreambooth/summary.md) (2022).

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

**7b.** [Self-Consistency](./papers/techniques/77-self-consistency/summary.md) (2022)
- 🔥 **CRITICAL** - Sample several chains, take the majority vote
- +17.9 points on GSM8K over greedy chain-of-thought
- First demonstration that inference compute buys accuracy
- [Paper](https://arxiv.org/abs/2203.11171)

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

**13.** [Reflexion](./papers/techniques/78-reflexion/summary.md) (2023)
- 🔥 **CRITICAL** - Agents that learn from their own failures
- Verbal self-reflection stored in memory, no gradient updates
- The act -> test -> reflect -> retry loop every coding agent runs
- [Paper](https://arxiv.org/abs/2303.11366)

#### **Training Infrastructure (How Big Models Are Actually Trained)**

**14.** [ZeRO and Megatron-LM](./papers/techniques/76-zero-megatron/summary.md) (2019)
- 🔥 **CRITICAL** - The systems layer under every large training run
- Data, tensor, and pipeline parallelism (3D parallelism); DeepSpeed and FSDP
- Removed the single-GPU ceiling on model size
- [ZeRO](https://arxiv.org/abs/1910.02054) | [Megatron-LM](https://arxiv.org/abs/1909.08053)

**15.** [GPTQ and AWQ](./papers/techniques/86-gptq-awq-quantization/summary.md) (2022-2023)
- 🔥 **CRITICAL** - 4-bit post-training quantization
- 70B model from ~140 GB to ~35 GB, no retraining
- Why local inference (Ollama, llama.cpp, LM Studio) exists
- [GPTQ](https://arxiv.org/abs/2210.17323) | [AWQ](https://arxiv.org/abs/2306.00978)

#### **Instruction Tuning & Synthetic Data**

**16.** [FLAN](./papers/techniques/80-flan/summary.md) (2021)
- 🔥 **CRITICAL** - Invented instruction tuning
- Zero-shot beat GPT-3 175B on 20 of 25 benchmarks
- The middle stage of pretrain -> instruction-tune -> align
- [Paper](https://arxiv.org/abs/2109.01652)

**17.** [Self-Instruct](./papers/techniques/79-self-instruct/summary.md) (2022)
- 🔥 **CRITICAL** - 52K instructions from 175 human seeds
- Made instruction data free; enabled Alpaca and the open fine-tuning wave
- Why most post-training data is now synthetic
- [Paper](https://arxiv.org/abs/2212.10560)

#### **Retrieval Foundations (The Other Half of RAG)**

**18.** [Dense Retrieval: DPR, ColBERT, Sentence-BERT](./papers/techniques/87-dense-retrieval/summary.md) (2019-2020)
- 🔥 **CRITICAL** - The retriever every RAG system runs on
- Bi-encoders, cross-encoder reranking, late interaction, hybrid search
- Where RAG systems actually fail, and how to fix them
- [DPR](https://arxiv.org/abs/2004.04906) | [ColBERT](https://arxiv.org/abs/2004.12832)

#### **Evaluation (How We Know Any of This Works)**

**19.** [SWE-bench](./papers/techniques/84-swe-bench/summary.md) (2023)
- 🔥 **CRITICAL** - Real GitHub issues, graded by the project's own tests
- Went from ~2% resolved in 2023 to ~70-80% in 2025
- The headline number in every coding-model announcement
- [Paper](https://arxiv.org/abs/2310.06770)

**20.** [LLM-as-a-Judge / Chatbot Arena](./papers/techniques/85-llm-as-judge/summary.md) (2023)
- 🔥 **CRITICAL** - Model-graded evaluation, validated against humans
- ~80% agreement with human preference, plus the biases to correct for
- Created Chatbot Arena Elo; now also generates training data
- [Paper](https://arxiv.org/abs/2306.05685)

**21.** [Emergent Abilities (and the Mirage rebuttal)](./papers/techniques/81-emergent-abilities/summary.md) (2022-2023)
- ⭐ **HIGH** - Do capabilities appear suddenly at scale?
- The rebuttal: sharp curves are often a metric artifact
- Why you should never report exact-match alone
- [Emergence](https://arxiv.org/abs/2206.07682) | [Mirage](https://arxiv.org/abs/2304.15004)

#### **Safety & Interpretability**

**22.** [Sparse Autoencoders & Monosemanticity](./papers/techniques/82-sparse-autoencoders/summary.md) (2022-2024)
- 🔥 **CRITICAL** - Reading the concepts inside a frontier model
- Millions of interpretable features from Claude 3 Sonnet; steering works
- The most credible progress on interpretability to date
- [Scaling Monosemanticity](https://transformer-circuits.pub/2024/scaling-monosemanticity/index.html)

**23.** [Sleeper Agents](./papers/techniques/83-sleeper-agents/summary.md) (2024)
- 🔥 **CRITICAL** - Backdoors survive the entire safety stack
- SFT, RLHF, and adversarial training all failed to remove them
- Adversarial training taught models to hide the trigger better
- [Paper](https://arxiv.org/abs/2401.05566)

_Also in this category:_ [Word2Vec](./papers/techniques/53-word2vec/summary.md), [RoPE](./papers/techniques/54-rope-rotary-position-embedding/summary.md), [PPO](./papers/techniques/63-ppo/summary.md), [Generative Agents](./papers/techniques/58-generative-agents/summary.md), [MCP](./papers/techniques/59-model-context-protocol/summary.md), [GraphRAG](./papers/techniques/60-graph-rag/summary.md), [AlphaGeometry](./papers/techniques/61-alphageometry/summary.md), [AlphaEvolve](./papers/techniques/62-alphaevolve/summary.md), [AlphaFold](./papers/techniques/68-alphafold/summary.md).

**Models that teach themselves to reason:** [STaR](./papers/techniques/97-star/summary.md) (2022), [Quiet-STaR](./papers/techniques/98-quiet-star/summary.md) (2024), [Self-Refine](./papers/techniques/99-self-refine/summary.md) (2023) - bootstrap rationales, think before speaking, and revise your own output.

**Agents and world models:** [Voyager](./papers/techniques/100-voyager/summary.md) (2023), [Genie](./papers/techniques/104-genie/summary.md) (2024), [DreamerV3](./papers/techniques/105-dreamerv3/summary.md) (2023) - an agent that writes its own skills, and two models that learn an environment well enough to imagine it.

**Alignment beyond preference pairs:** [KTO](./papers/techniques/103-kto/summary.md) (2024) - align on plain thumbs-up/thumbs-down instead of ranked pairs.

**AI on hard science and hard games:** [AlphaFold 3](./papers/techniques/101-alphafold3/summary.md) (2024), [ESM-2 / ESMFold](./papers/techniques/106-esm/summary.md) (2023), [AlphaZero](./papers/techniques/102-alphazero/summary.md) (2017), [CICERO](./papers/techniques/107-cicero/summary.md) (2022) - structure prediction, protein language models, self-play from zero, and negotiation in natural language.

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
| **Papers** | 107 | 25-30 hours |
| **Source words compressed** | 1.4M+ | ~130 hours |
| **Words** | 219,000+ | - |
| **Guides** | 6 | 3-5 hours |
| **Terms Explained** | 250+ | - |

### By Year
_Generated from [`papers.json`](./papers.json) - see [INDEX.md](./INDEX.md) for the full clickable list._

<!-- byyear:start -->
- **2013:** 2 papers
- **2014:** 3 papers
- **2015:** 2 papers
- **2017:** 4 papers
- **2018:** 2 papers
- **2019:** 3 papers
- **2020:** 8 papers
- **2021:** 10 papers
- **2022:** 19 papers
- **2023:** 25 papers
- **2024:** 19 papers
- **2025:** 10 papers
<!-- byyear:end -->

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

**Last Updated:** 2026-08-18
**Papers:** 107 foundational works (2013-2026)
**Total Content:** 460,000+ words
**Includes:** Roots (Word2Vec, Seq2Seq, VAE, PPO) through the latest breakthroughs of early 2026 (GPT-5, Claude 4, Llama 4, GRPO, RLVR, and more)
**Repository:** [github.com/PatrickWiloak/genai-research-papers-summarized](https://github.com/PatrickWiloak/genai-research-papers-summarized)
