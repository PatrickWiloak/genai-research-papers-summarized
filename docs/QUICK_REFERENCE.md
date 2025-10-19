# Quick Reference Guide - All Papers at a Glance

A one-page overview of all 15 foundational GenAI papers for quick lookup.

---

## Architecture Papers

| # | Paper | Year | Key Contribution | Impact |
|---|-------|------|------------------|--------|
| 1 | **Attention Is All You Need** | 2017 | Self-attention mechanism, Transformer architecture | Foundation for all modern LLMs (GPT, BERT, etc.) |
| 2 | **GANs** | 2014 | Generator-discriminator adversarial training | Launched generative modeling, deepfakes, image synthesis |
| 6 | **Diffusion Models (DDPM)** | 2020 | Iterative denoising process | Alternative to GANs, foundation for DALL-E 2, Midjourney |
| 11 | **Vision Transformer (ViT)** | 2020 | Images as patch sequences for Transformers | Unified vision-language architectures, multimodal models |

---

## Language Model Papers

| # | Paper | Year | Key Contribution | Impact |
|---|-------|------|------------------|--------|
| 3 | **BERT** | 2018 | Bidirectional pre-training, masked language modeling | Google Search, NLP tasks, encoder-based models |
| 4 | **GPT-3** | 2020 | 175B params, few-shot in-context learning | ChatGPT foundation, prompt engineering era |
| 5 | **InstructGPT (RLHF)** | 2022 | Reinforcement learning from human feedback | ChatGPT, Claude, all modern AI assistants |
| 14 | **Constitutional AI** | 2022 | AI self-critique guided by principles | Powers Claude, alternative to RLHF |
| 15 | **LLaMA** | 2023 | Compute-optimal training, 13B = 175B performance | Open-source LLM revolution, Alpaca, Vicuna |

---

## Multimodal & Vision Papers

| # | Paper | Year | Key Contribution | Impact |
|---|-------|------|------------------|--------|
| 7 | **Stable Diffusion** | 2022 | Latent diffusion, 10-100× speedup | Democratized AI art, open-source text-to-image |
| 8 | **CLIP** | 2021 | Vision-language contrastive learning | Zero-shot classification, text-to-image guidance |
| 11 | **Vision Transformer** | 2020 | Transformers for image classification | Enabled GPT-4 Vision, DALL-E, multimodal AI |

---

## Technique & Method Papers

| # | Paper | Year | Key Contribution | Impact |
|---|-------|------|------------------|--------|
| 9 | **Chain-of-Thought** | 2022 | Step-by-step reasoning prompts | Improved LLM reasoning, math, logic tasks |
| 10 | **LoRA** | 2021 | Low-rank adaptation, 10,000× parameter reduction | Accessible fine-tuning, Stable Diffusion LoRAs |
| 12 | **Scaling Laws** | 2020 | Predictable power laws for loss vs compute/data | Justified GPT-3/4, guided all major AI development |
| 13 | **RAG** | 2020 | Retrieval + generation with external knowledge | Production LLM standard, reduced hallucinations |

---

## By Research Area

### Natural Language Processing
- **Transformers** (1) - Architecture foundation
- **BERT** (3) - Bidirectional understanding
- **GPT-3** (4) - Few-shot learning
- **Scaling Laws** (12) - Performance prediction
- **LLaMA** (15) - Efficient open models

### Computer Vision
- **GANs** (2) - Adversarial generation
- **Diffusion Models** (6) - Denoising generation
- **Vision Transformer** (11) - Transformer for vision
- **Stable Diffusion** (7) - Latent diffusion

### Multimodal AI
- **CLIP** (8) - Vision-language bridge
- **Stable Diffusion** (7) - Text-to-image
- **Vision Transformer** (11) - Unified architecture

### AI Alignment & Safety
- **InstructGPT** (5) - RLHF alignment
- **Constitutional AI** (14) - Principle-based alignment
- **Chain-of-Thought** (9) - Interpretable reasoning

### Efficient Training & Deployment
- **LoRA** (10) - Efficient fine-tuning
- **Scaling Laws** (12) - Compute optimization
- **LLaMA** (15) - Training efficiency
- **RAG** (13) - Knowledge efficiency

---

## By Year

### 2014
- **GANs** - Adversarial training framework

### 2017
- **Transformers** - Self-attention architecture

### 2018
- **BERT** - Bidirectional pre-training

### 2020 (Breakthrough Year)
- **GPT-3** - 175B parameter scaling
- **Scaling Laws** - Mathematical AI foundations
- **Vision Transformer** - Transformers conquer vision
- **Diffusion Models** - Denoising generation
- **RAG** - Retrieval-augmented generation

### 2021
- **CLIP** - Vision-language alignment
- **LoRA** - Efficient fine-tuning

### 2022 (Alignment & Practical Year)
- **InstructGPT** - RLHF for alignment
- **Chain-of-Thought** - Reasoning prompts
- **Stable Diffusion** - Open text-to-image
- **Constitutional AI** - AI feedback alignment

### 2023
- **LLaMA** - Open, efficient models

---

## Quick Stats

### Model Sizes
| Paper | Largest Model | Key Finding |
|-------|--------------|-------------|
| GPT-3 | 175B params | Few-shot learning emerges |
| BERT | 340M params | Pre-training + fine-tuning works |
| LLaMA | 65B params | Matches 175B with better training |
| InstructGPT | 175B params | 1.3B + RLHF > 175B base |
| ViT | 632M params | Transformers work for vision |

### Training Data
| Paper | Dataset Size | Source |
|-------|-------------|--------|
| GPT-3 | 300B tokens | Web crawl, books |
| LLaMA | 1.4T tokens | Public datasets only |
| BERT | 3.3B words | Wikipedia, books |
| CLIP | 400M pairs | Image-text from web |
| Stable Diffusion | 5B images | LAION dataset |

### Citations (Approximate)
| Paper | Citations | Year | Impact |
|-------|-----------|------|--------|
| Transformers | 100k+ | 2017 | Most cited |
| GANs | 50k+ | 2014 | Foundational |
| BERT | 80k+ | 2018 | Huge NLP impact |
| GPT-3 | 15k+ | 2020 | Recent landmark |
| Diffusion | 10k+ | 2020 | Rapidly growing |

---

## Problem → Solution Mapping

### Problem: Can't process long sequences in parallel
**Solution:** Transformers (1) - Self-attention for parallel processing

### Problem: Need realistic image generation
**Solutions:**
- GANs (2) - Adversarial training
- Diffusion (6) - Iterative denoising
- Stable Diffusion (7) - Efficient latent space

### Problem: Models don't understand context bidirectionally
**Solution:** BERT (3) - Masked language modeling

### Problem: Need task-specific training for every application
**Solution:** GPT-3 (4) - Few-shot in-context learning

### Problem: Models are helpful but harmful
**Solutions:**
- InstructGPT (5) - RLHF alignment
- Constitutional AI (14) - Principle-based self-improvement

### Problem: Models hallucinate facts
**Solution:** RAG (13) - Retrieve then generate from evidence

### Problem: Transformers only work for text
**Solution:** Vision Transformer (11) - Treat images as patch sequences

### Problem: Can't connect vision and language
**Solution:** CLIP (8) - Contrastive learning on image-text pairs

### Problem: Models too expensive to fine-tune
**Solution:** LoRA (10) - Low-rank adaptation matrices

### Problem: Models can't reason through complex problems
**Solution:** Chain-of-Thought (9) - Step-by-step reasoning prompts

### Problem: Don't know how much compute/data needed
**Solution:** Scaling Laws (12) - Predictable power laws

### Problem: Only big companies can build good LLMs
**Solution:** LLaMA (15) - Smaller models + more training

---

## Key Innovations Summary

### Self-Attention (Transformers)
```
Q(query) · K(key) / √d → attention weights → weighted V(values)
Enables parallel processing, long-range dependencies
```

### Adversarial Training (GANs)
```
Generator creates fake data
Discriminator tries to detect fakes
Both improve through competition
```

### Masked Language Modeling (BERT)
```
The cat [MASK] on the mat → predict "sat"
Learn bidirectional context
```

### In-Context Learning (GPT-3)
```
Provide examples in prompt
Model learns pattern without training
Few-shot → strong performance
```

### RLHF (InstructGPT)
```
1. Supervised fine-tuning
2. Reward model from preferences
3. RL optimization (PPO)
```

### Diffusion Process (DDPM)
```
Forward: Add noise gradually
Reverse: Learn to denoise
Generate by denoising random noise
```

### Latent Diffusion (Stable Diffusion)
```
Encode image to latent space (8× compression)
Run diffusion in latent space
Decode to pixel space
10-100× faster than pixel diffusion
```

### Contrastive Learning (CLIP)
```
Match: image of cat ↔ "a cat"
Don't match: image of cat ↔ "a dog"
Learn alignment through contrast
```

### Low-Rank Adaptation (LoRA)
```
W + ΔW = W + BA
where B, A are low-rank matrices
10,000× fewer trainable parameters
```

### Retrieval-Augmented (RAG)
```
Query → Retrieve docs → Generate with context
Knowledge in database, not just parameters
```

### Constitutional AI
```
1. AI critiques own output vs principle
2. AI revises to align with principle
3. Train on self-improvements
```

### Scaling Laws
```
L(N) ∝ N^(-α)  [loss vs params]
L(D) ∝ D^(-α)  [loss vs data]
L(C) ∝ C^(-α)  [loss vs compute]
Predictable across 7+ orders of magnitude
```

---

## Reading Time Estimates

| Paper | Summary Length | Reading Time |
|-------|---------------|--------------|
| Transformers | ~8000 words | 45-60 min |
| GANs | ~6000 words | 30-45 min |
| BERT | ~7000 words | 40-50 min |
| GPT-3 | ~8000 words | 45-60 min |
| InstructGPT | ~7500 words | 40-55 min |
| Diffusion | ~7000 words | 40-50 min |
| Stable Diffusion | ~7500 words | 40-55 min |
| CLIP | ~7000 words | 40-50 min |
| Chain-of-Thought | ~6000 words | 30-45 min |
| LoRA | ~6500 words | 35-45 min |
| Vision Transformer | ~8500 words | 50-65 min |
| Scaling Laws | ~8000 words | 45-60 min |
| RAG | ~8500 words | 50-65 min |
| Constitutional AI | ~9000 words | 55-70 min |
| LLaMA | ~8500 words | 50-65 min |
| **TOTAL** | **~110k words** | **12-15 hours** |

---

## Paper Dependencies

**Read these first (foundations):**
1. Transformers - Everything builds on this
2. Scaling Laws - Theoretical foundation

**Then language models:**
3. BERT or GPT-3 - Encoder vs decoder approaches
4. LLaMA - Modern efficient training

**Then vision:**
5. GANs or Diffusion - Generative approaches
6. Vision Transformer - Transformers for images
7. CLIP - Bridging vision and language

**Then practical techniques:**
8. LoRA - Efficient adaptation
9. RAG - Knowledge grounding
10. Chain-of-Thought - Better reasoning

**Finally alignment:**
11. InstructGPT - RLHF approach
12. Constitutional AI - Alternative approach

**Applied vision:**
13. Stable Diffusion - Practical text-to-image

---

## When to Reference Each Paper

### Building a chatbot
→ GPT-3, InstructGPT, Constitutional AI, RAG

### Fine-tuning models
→ BERT, LoRA, LLaMA

### Image generation
→ GANs, Diffusion, Stable Diffusion

### Multimodal applications
→ CLIP, Vision Transformer, Stable Diffusion

### Improving reasoning
→ Chain-of-Thought, InstructGPT

### Planning compute budget
→ Scaling Laws, LLaMA

### Understanding modern AI architectures
→ Transformers, Vision Transformer

### Reducing hallucinations
→ RAG, Constitutional AI

### Zero-shot classification
→ CLIP, GPT-3

### Efficient deployment
→ LoRA, LLaMA, RAG

---

**Updated:** 2025-10-19
**Total Papers:** 15
**Total Reading Time:** 12-15 hours (summaries only)
