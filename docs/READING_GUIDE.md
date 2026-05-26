# Reading Guide - Historical Significance vs Current Relevance

A practical guide to understanding which of the 94 papers are essential for modern AI work vs historical context.

> Symbols: 🔥 = significance/relevance score (1-5). **CRITICAL** = read this. **HIGH** = read this if relevant to your work. **MEDIUM** = historical/specialized context.

---

## 📊 Quick Reference Matrix

### Original 15 (the core)

| Paper | Year | Historical | Current | Priority | Read For |
|-------|------|------------|---------|----------|----------|
| **Transformers** | 2017 | 🔥🔥🔥🔥🔥 | 🔥🔥🔥🔥🔥 | **CRITICAL** | Architecture foundation |
| **Scaling Laws** | 2020 | 🔥🔥🔥🔥🔥 | 🔥🔥🔥🔥🔥 | **CRITICAL** | Theory & planning |
| **LLaMA** | 2023 | 🔥🔥🔥 | 🔥🔥🔥🔥🔥 | **CRITICAL** | Modern training |
| **RAG** | 2020 | 🔥🔥🔥🔥 | 🔥🔥🔥🔥🔥 | **CRITICAL** | Production apps |
| **LoRA** | 2021 | 🔥🔥🔥🔥 | 🔥🔥🔥🔥🔥 | **CRITICAL** | Fine-tuning |
| **InstructGPT** | 2022 | 🔥🔥🔥🔥🔥 | 🔥🔥🔥🔥🔥 | **CRITICAL** | Alignment |
| **Chain-of-Thought** | 2022 | 🔥🔥🔥🔥 | 🔥🔥🔥🔥🔥 | **CRITICAL** | Reasoning |
| **GPT-3** | 2020 | 🔥🔥🔥🔥🔥 | 🔥🔥🔥🔥 | **HIGH** | Few-shot learning |
| **Constitutional AI** | 2022 | 🔥🔥🔥 | 🔥🔥🔥🔥 | **HIGH** | Alternative alignment |
| **Vision Transformer** | 2020 | 🔥🔥🔥🔥 | 🔥🔥🔥🔥 | **HIGH** | Multimodal foundation |
| **CLIP** | 2021 | 🔥🔥🔥🔥 | 🔥🔥🔥🔥 | **HIGH** | Vision-language |
| **Stable Diffusion** | 2022 | 🔥🔥🔥🔥 | 🔥🔥🔥🔥 | **HIGH** | Image generation |
| **BERT** | 2018 | 🔥🔥🔥🔥🔥 | 🔥🔥 | **MEDIUM** | Historical context |
| **GANs** | 2014 | 🔥🔥🔥🔥🔥 | 🔥🔥 | **MEDIUM** | Historical context |
| **DDPM** | 2020 | 🔥🔥🔥🔥 | 🔥🔥🔥🔥 | **HIGH** | Diffusion theory |

### Foundations Below the Transformer

| Paper | Year | Historical | Current | Priority | Read For |
|-------|------|------------|---------|----------|----------|
| **Word2Vec** | 2013 | 🔥🔥🔥🔥🔥 | 🔥🔥 | **MEDIUM** | Embedding origins |
| **VAE** | 2013 | 🔥🔥🔥🔥 | 🔥🔥🔥 | **MEDIUM** | Latent spaces, reparameterization trick |
| **Seq2Seq** | 2014 | 🔥🔥🔥🔥🔥 | 🔥🔥 | **MEDIUM** | Encoder-decoder origin |
| **Bahdanau Attention** | 2014 | 🔥🔥🔥🔥🔥 | 🔥🔥🔥 | **HIGH** | Attention invented |
| **ResNet** | 2015 | 🔥🔥🔥🔥🔥 | 🔥🔥🔥🔥 | **HIGH** | Residual connections, in every Transformer |
| **GPT-1** | 2018 | 🔥🔥🔥🔥🔥 | 🔥🔥 | **MEDIUM** | Started GPT lineage |
| **GPT-2** | 2019 | 🔥🔥🔥🔥 | 🔥🔥 | **MEDIUM** | Zero-shot transfer |
| **T5** | 2019 | 🔥🔥🔥🔥 | 🔥🔥🔥🔥 | **HIGH** | Text-to-text framing |
| **PaLM** | 2022 | 🔥🔥🔥🔥 | 🔥🔥🔥 | **MEDIUM** | Pre-Gemini Google flagship |

### Open-Source Frontier

| Paper | Year | Priority | Read For |
|-------|------|----------|----------|
| **Mistral 7B** | 2023 | **HIGH** | GQA + SWA, beat LLaMA-2 13B |
| **Mixtral 8x7B** | 2024 | **CRITICAL** | First high-quality open MoE |
| **DeepSeek-V3** | 2024 | **CRITICAL** | $5.76M frontier-quality training |
| **DeepSeek-R1** | 2025 | **CRITICAL** | Open reasoning model |
| **Qwen3** | 2025 | **CRITICAL** | Unified thinking/non-thinking |

### Reasoning & Test-Time Compute

| Paper | Year | Priority | Read For |
|-------|------|----------|----------|
| **Self-Consistency** | 2022 | **CRITICAL** | Foundation of test-time-compute |
| **STaR** | 2022 | **HIGH** | Self-taught reasoning origin |
| **Quiet-STaR** | 2024 | **HIGH** | Direct o1 precursor |
| **Process Reward Models** | 2023 | **CRITICAL** | Step-level verification for reasoning |
| **Test-Time Compute** | 2024 | **CRITICAL** | Compute-optimal reasoning scaling |
| **o1** | 2024 | **CRITICAL** | Started reasoning era |
| **DeepSeek-R1** | 2025 | **CRITICAL** | Open RL-trained reasoning |

### Agents & Tools

| Paper | Year | Priority | Read For |
|-------|------|----------|----------|
| **ReAct** | 2023 | **CRITICAL** | Reasoning + acting foundation |
| **Reflexion** | 2023 | **HIGH** | Verbal RL for agents |
| **Self-Refine** | 2023 | **HIGH** | LLM as critic |
| **Voyager** | 2023 | **HIGH** | LLM Minecraft agent |
| **SWE-bench** | 2023 | **CRITICAL** | The coding-agent benchmark |
| **MCP** | 2024 | **HIGH** | Open standard for tool integration |

### Image / Video Generation

| Paper | Year | Priority | Read For |
|-------|------|----------|----------|
| **VQ-VAE** | 2017 | **HIGH** | Discrete image tokens |
| **VQ-GAN** | 2021 | **HIGH** | SD latent space ancestor |
| **Imagen** | 2022 | **MEDIUM** | T5 text encoder finding |
| **DreamBooth** | 2022 | **HIGH** | Personalization |
| **ControlNet** | 2023 | **CRITICAL** | Conditional control |
| **Sora / DiT** | 2024 | **CRITICAL** | Diffusion Transformers; video |

### Alignment & Safety

| Paper | Year | Priority | Read For |
|-------|------|----------|----------|
| **DPO** | 2023 | **CRITICAL** | Simpler than RLHF |
| **KTO** | 2024 | **HIGH** | Thumbs-up/down alignment |
| **Llama Guard** | 2023 | **HIGH** | Production safety classifier |
| **Sparse Autoencoders** | 2024 | **CRITICAL** | Production-scale interpretability |

### Scientific & World-Model AI

| Paper | Year | Priority | Read For |
|-------|------|----------|----------|
| **AlphaZero** | 2017 | **HIGH** | Self-play RL ancestor of RLVR |
| **AlphaFold 2** | 2021 | **CRITICAL** | Solved protein folding |
| **CICERO** | 2022 | **HIGH** | Multi-agent LLM + planning |
| **DreamerV3** | 2023 | **HIGH** | Model-based RL |
| **ESM-2** | 2023 | **HIGH** | Protein language model |
| **AlphaFold 3** | 2024 | **CRITICAL** | Biomolecular complexes; drug discovery |
| **Genie** | 2024 | **HIGH** | Foundation world model |
| **AlphaGeometry / AlphaEvolve** | 2024-25 | **HIGH** | Discovery-engine AI |

---

## 🎯 Priority Tiers Explained

### Tier 1: CRITICAL - Must Read (7 papers)
**These define modern AI development in 2024/2025**

#### 1. Transformers (2017)
**Historical:** Founded the entire modern AI era
**Current:** Every major model uses this architecture
**Why Read:** Cannot understand anything else without this
**Skip If:** You literally never work with AI
**Time Investment:** 1 hour
**Key Takeaway:** Self-attention is the foundation

#### 2. Scaling Laws (2020)
**Historical:** Proved AI progress is predictable
**Current:** Guides every major AI investment and model design
**Why Read:** Understand why companies spend billions on compute
**Skip If:** You only use APIs and don't care about economics
**Time Investment:** 45 minutes
**Key Takeaway:** Loss scales predictably with compute/data/params

#### 3. LLaMA (2023)
**Historical:** Recent but already transformative
**Current:** Most active open-source ecosystem (Alpaca, Vicuna, etc.)
**Why Read:** Modern best practices for training
**Skip If:** You only use closed models
**Time Investment:** 1 hour
**Key Takeaway:** Training > size, open > closed

#### 4. RAG (2020)
**Historical:** Solved the hallucination problem
**Current:** ~80% of production LLM apps use this
**Why Read:** Essential for building real applications
**Skip If:** You only do research, not deployment
**Time Investment:** 1 hour
**Key Takeaway:** Retrieval grounds generation in facts

#### 5. LoRA (2021)
**Historical:** Made fine-tuning accessible
**Current:** Standard for customizing models
**Why Read:** If you ever fine-tune, you'll use this
**Skip If:** You never customize models
**Time Investment:** 45 minutes
**Key Takeaway:** 10,000× fewer params, same quality

#### 6. InstructGPT / RLHF (2022)
**Historical:** Enabled ChatGPT
**Current:** Powers all major AI assistants
**Why Read:** Understand how models become helpful
**Skip If:** You don't care about alignment
**Time Investment:** 1 hour
**Key Takeaway:** Human feedback shapes behavior

#### 7. Chain-of-Thought (2022)
**Historical:** Unlocked reasoning capabilities
**Current:** Active area, getting better (Tree of Thoughts, etc.)
**Why Read:** Dramatically improves prompt effectiveness
**Skip If:** You only do simple tasks
**Time Investment:** 30 minutes
**Key Takeaway:** "Let's think step by step" works

---

### Tier 2: HIGH - Strongly Recommended (5 papers)
**Important for comprehensive understanding**

#### 8. GPT-3 (2020)
**Historical:** 🔥🔥🔥🔥🔥 Proved scaling works, introduced few-shot
**Current:** 🔥🔥🔥🔥 Still relevant but surpassed by GPT-4, LLaMA 2
**Why Read:** Understand the paradigm shift to prompting
**Skip If:** You've used ChatGPT and get the concept
**Time Investment:** 1 hour
**Modern Alternative:** Just read LLaMA (covers similar ground)

#### 9. Constitutional AI (2022)
**Historical:** 🔥🔥🔥 Alternative alignment method
**Current:** 🔥🔥🔥🔥 Powers Claude, growing adoption
**Why Read:** Understand alignment alternatives to RLHF
**Skip If:** You only use RLHF models
**Time Investment:** 1 hour
**Modern Status:** Increasingly important as RLHF limitations appear

#### 10. Vision Transformer (2020)
**Historical:** 🔥🔥🔥🔥 Unified vision and language architectures
**Current:** 🔥🔥🔥🔥 Foundation for GPT-4V, DALL-E, Gemini Vision
**Why Read:** Essential for multimodal understanding
**Skip If:** You only work with text
**Time Investment:** 1 hour
**Modern Status:** More relevant than ever (multimodal boom)

#### 11. CLIP (2021)
**Historical:** 🔥🔥🔥🔥 Connected vision and language
**Current:** 🔥🔥🔥🔥 Powers text-to-image, visual search
**Why Read:** Understand vision-language alignment
**Skip If:** You don't touch images
**Time Investment:** 45 minutes
**Modern Status:** Core component of modern image AI

#### 12. Stable Diffusion (2022)
**Historical:** 🔥🔥🔥🔥 Democratized AI art
**Current:** 🔥🔥🔥🔥 Active ecosystem, constantly evolving
**Why Read:** Understand practical image generation
**Skip If:** You don't care about image generation
**Time Investment:** 1 hour
**Modern Status:** Still dominant, but competition from DALL-E 3, Midjourney

---

### Tier 3: MEDIUM - Historical Context (3 papers)
**Important for understanding AI history, less critical for modern work**

#### 13. BERT (2018)
**Historical:** 🔥🔥🔥🔥🔥 Revolutionized NLP, pre-train + fine-tune
**Current:** 🔥🔥 Mostly replaced by decoder-only models
**Why Read:** Understand the pre-training revolution
**Skip If:** You focus on modern LLMs (all decoder-only now)
**Time Investment:** 45 minutes
**Modern Alternative:** Focus on GPT-3/LLaMA instead

**What Changed:**
- 2018-2020: BERT-style encoders dominated
- 2020+: GPT-style decoders took over (better at generation)
- Today: Encoder-only models are niche (embedding models mainly)

**Still Used For:**
- Sentence embeddings (Sentence-BERT)
- Some classification tasks
- Legacy systems

**Modern Descendants:**
- RoBERTa, DeBERTa (improved BERT)
- Sentence-BERT (embeddings)

#### 14. GANs (2014)
**Historical:** 🔥🔥🔥🔥🔥 Launched generative modeling field
**Current:** 🔥🔥 Mostly replaced by diffusion models
**Why Read:** Understand adversarial training concept
**Skip If:** You only care about modern image generation
**Time Investment:** 45 minutes
**Modern Alternative:** Read Stable Diffusion instead

**What Changed:**
- 2014-2020: GANs dominated image generation
- 2020+: Diffusion models proved more stable, higher quality
- Today: GANs used for niche applications (real-time, style transfer)

**Still Used For:**
- Real-time video generation (fast inference)
- Style transfer
- Some specific domains

**Modern Descendants:**
- StyleGAN series (still used for faces)
- BigGAN (large-scale)

#### 15. Diffusion Models / DDPM (2020)
**Historical:** 🔥🔥🔥🔥 Introduced diffusion for images
**Current:** 🔥🔥🔥 Theory covered in Stable Diffusion
**Why Read:** Deep understanding of diffusion mathematics
**Skip If:** You just want to use diffusion models
**Time Investment:** 1 hour
**Modern Alternative:** Stable Diffusion summary covers key concepts

**What Changed:**
- 2020: DDPM introduced theory
- 2022: Stable Diffusion made it practical (latent space)
- Today: Latent diffusion is standard

**Still Relevant For:**
- Research into diffusion theory
- Understanding mathematical foundations
- Optimizing diffusion models

---

## 🎓 Reading Strategies by Goal

### Goal: Build Modern LLM Applications
**Time: 5-7 hours total**

**Must Read (4 papers, 4 hours):**
1. Transformers - 1h (understand architecture)
2. RAG - 1h (most important for apps!)
3. LoRA - 45min (for customization)
4. Chain-of-Thought - 30min (for better prompts)

**Strongly Recommended (2 papers, 2 hours):**
5. LLaMA - 1h (modern training best practices)
6. InstructGPT - 1h (understand alignment)

**Optional:**
7. Scaling Laws - 45min (for planning)
8. Constitutional AI - 1h (alternative alignment)

**Skip:**
- BERT (old paradigm)
- GANs, DDPM (not relevant for LLMs)
- ViT, CLIP (unless doing multimodal)

---

### Goal: Understand Modern AI Landscape
**Time: 8-10 hours total**

**Critical Foundation (5 papers, 5 hours):**
1. Transformers - 1h
2. Scaling Laws - 45min
3. GPT-3 - 1h
4. InstructGPT - 1h
5. LLaMA - 1h

**Choose Your Focus (3-4 papers, 3-4 hours):**

*If interested in applications:*
6. RAG - 1h
7. LoRA - 45min
8. Chain-of-Thought - 30min

*If interested in image generation:*
6. Vision Transformer - 1h
7. CLIP - 45min
8. Stable Diffusion - 1h

**Historical Context (optional, 2 hours):**
9. BERT - 45min (pre-training revolution)
10. GANs - 45min (generative modeling origins)

---

### Goal: Deep Research Understanding
**Time: 15-20 hours total**

**Read all 15 papers chronologically to understand evolution:**

**Phase 1: Foundations (2014-2018)**
1. GANs (2014) - Generative modeling begins
2. Transformers (2017) - Architecture revolution
3. BERT (2018) - Pre-training paradigm

**Phase 2: Scaling Era (2020)**
4. Scaling Laws (2020) - Predictive theory
5. GPT-3 (2020) - Scaling in practice
6. Vision Transformer (2020) - Transformers to vision
7. DDPM (2020) - Diffusion introduced
8. RAG (2020) - Grounded generation

**Phase 3: Efficient & Aligned (2021-2022)**
9. CLIP (2021) - Multimodal alignment
10. LoRA (2021) - Efficient adaptation
11. InstructGPT (2022) - RLHF alignment
12. Chain-of-Thought (2022) - Reasoning
13. Stable Diffusion (2022) - Practical diffusion
14. Constitutional AI (2022) - Alternative alignment

**Phase 4: Modern Era (2023+)**
15. LLaMA (2023) - Compute-optimal training

---

## 📉 What's Becoming Less Relevant

### BERT-Style Encoders
**Then (2018-2020):** Dominant for NLP tasks
**Now (2024+):** Mostly replaced by decoder-only models
**Why:** Decoders can do both understanding AND generation
**Exception:** Still used for embeddings (Sentence-BERT)

### GANs
**Then (2014-2020):** State-of-art for image generation
**Now (2024+):** Diffusion models are standard
**Why:** More stable training, better quality, easier to control
**Exception:** Real-time applications where speed matters

### Separate Vision & Language Models
**Then (pre-2020):** CNNs for vision, Transformers for language
**Now (2024+):** Unified Transformer architecture for everything
**Why:** ViT + multimodal models like GPT-4V
**Exception:** Some specialized vision tasks still use CNNs

---

## 📈 What's Increasingly Important

### Alignment Methods
**Papers:** InstructGPT, Constitutional AI
**Why:** AI safety and helpfulness are critical
**Trend:** More sophisticated alignment techniques emerging

### Efficient Adaptation
**Papers:** LoRA, RAG
**Why:** Can't always retrain from scratch
**Trend:** More efficient methods (QLoRA, PEFT variants)

### Multimodal
**Papers:** ViT, CLIP
**Why:** GPT-4V, Gemini Vision, etc. are the future
**Trend:** Everything becoming multimodal

### Reasoning
**Papers:** Chain-of-Thought
**Why:** LLMs need to handle complex tasks
**Trend:** Tree of Thoughts, graph reasoning, etc.

---

## 🎯 Minimum Viable Reading

**If you only have 3 hours, read these 4 papers:**

1. **Transformers** (1h) - Foundation
2. **Scaling Laws** (45min) - Theory
3. **RAG** (45min) - Most practical
4. **Chain-of-Thought** (30min) - Immediate application

This gives you:
- ✅ Architecture foundation
- ✅ Understanding of why AI works
- ✅ Most important production technique
- ✅ Simple prompt improvement method

---

## 📚 Suggested Reading Orders

### Chronological (Historical Understanding)
GANs → Transformers → BERT → Scaling Laws → GPT-3 → ViT → DDPM → RAG → CLIP → LoRA → InstructGPT → Chain-of-Thought → Stable Diffusion → Constitutional AI → LLaMA

**Pros:** Understand evolution and context
**Cons:** Start with less relevant material
**Time:** 15+ hours

### Relevance-First (Modern Focus)
Transformers → Scaling Laws → LLaMA → RAG → LoRA → InstructGPT → Chain-of-Thought → GPT-3 → Constitutional AI → ViT → CLIP → Stable Diffusion → BERT → GANs → DDPM

**Pros:** Most useful information first
**Cons:** Miss historical context
**Time:** 12-15 hours (can stop earlier)

### Architecture-First (Technical Deep Dive)
Transformers → ViT → BERT → GPT-3 → LLaMA → Scaling Laws → InstructGPT → Constitutional AI → RAG → LoRA → Chain-of-Thought → GANs → DDPM → Stable Diffusion → CLIP

**Pros:** Deep architecture understanding first
**Cons:** Less practical focus
**Time:** 15+ hours

---

## 💡 Pro Tips

### For Skimming
1. Read "Why This Matters" section only
2. Check the results tables
3. Read "Key Takeaways"
4. Skip mathematical derivations first time

### For Deep Learning
1. Read summary fully
2. Read original paper
3. Implement core concepts
4. Compare with modern alternatives

### For Practical Application
1. Focus on "Practical Applications" sections
2. Try code examples
3. Skip heavy theory
4. Focus on limitations (what NOT to do)

---

## 🔄 When to Re-read

**Re-read BERT if:**
- Working with legacy NLP systems
- Doing sentence embedding work
- Need bidirectional understanding specifically

**Re-read GANs if:**
- Need real-time generation
- Doing style transfer work
- Researching adversarial methods

**Re-read DDPM if:**
- Optimizing diffusion models
- Research into diffusion theory
- Building custom diffusion variants

---

## ✅ Reading Progress Tracker

Mark off as you complete:

**CRITICAL (7 papers):**
- [ ] Transformers
- [ ] Scaling Laws
- [ ] LLaMA
- [ ] RAG
- [ ] LoRA
- [ ] InstructGPT
- [ ] Chain-of-Thought

**HIGH (5 papers):**
- [ ] GPT-3
- [ ] Constitutional AI
- [ ] Vision Transformer
- [ ] CLIP
- [ ] Stable Diffusion

**MEDIUM (3 papers):**
- [ ] BERT
- [ ] GANs
- [ ] DDPM

**Total Progress:** ___ / 15 papers (core); see Quick Reference for full 94

---

**Last Updated:** 2026-05-26
**Total Papers in Collection:** 94
**Next Review:** This guide will be updated as the field evolves
