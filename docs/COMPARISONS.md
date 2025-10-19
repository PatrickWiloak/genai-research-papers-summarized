# Paper Comparisons and Analysis

Detailed side-by-side comparisons of related papers to understand trade-offs and evolution.

---

## Table of Contents
1. [Architecture Comparisons](#architecture-comparisons)
2. [Training Approaches](#training-approaches)
3. [Alignment Methods](#alignment-methods)
4. [Image Generation](#image-generation)
5. [Efficiency Techniques](#efficiency-techniques)
6. [Evolution Over Time](#evolution-over-time)

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
| **Autoregressive** | GPT-3, LLaMA | Predict next token | Natural generation | Only sees left context |
| **Contrastive** | CLIP | Match positive pairs | Learn alignments | Needs paired data |
| **Denoising** | Diffusion, DDPM | Remove noise iteratively | High quality | Slow generation |
| **Adversarial** | GANs | Generator vs discriminator | Fast generation | Training instability |

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

### RLHF vs Constitutional AI

| Aspect | InstructGPT (RLHF) | Constitutional AI |
|--------|-------------------|-------------------|
| **Human Labels** | 10,000+ comparisons | ~100 (for helpfulness only) |
| **Data Source** | Human preferences | AI self-critique |
| **Transparency** | Opaque (implicit values) | Transparent (written principles) |
| **Cost** | $50k-$100k | $1k-$5k |
| **Iteration Speed** | Weeks (recollect data) | Days (update principles) |
| **Consistency** | Variable (human annotators differ) | High (AI consistent) |
| **Best For** | Helpfulness | Harmlessness |
| **Scalability** | Limited by human bandwidth | High (AI scales) |
| **Bias** | Reflects annotator biases | Reflects principle writers' biases |
| **Training Stages** | SFT → Reward Model → PPO | SL-CAI → RL-CAI |

**Stage-by-Stage:**

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

## Image Generation

### GANs vs Diffusion vs Latent Diffusion

| Aspect | GANs | Diffusion (DDPM) | Stable Diffusion |
|--------|------|------------------|------------------|
| **Training** | Adversarial (unstable) | Stable denoising | Stable (in latent space) |
| **Generation Speed** | Fast (1 pass) | Slow (50-1000 steps) | Medium (50 steps, but smaller) |
| **Quality** | Good | Excellent | Excellent |
| **Diversity** | Lower (mode collapse) | Higher | Higher |
| **Control** | Harder | Moderate | Easy (text conditioning) |
| **Memory** | Moderate | High (pixel space) | Lower (latent space) |
| **Training Data** | Can work with less | Needs more | Needs more |
| **Compute (Train)** | Moderate | High | Moderate |
| **Compute (Inference)** | Low | Very high | Medium |
| **Text-to-Image** | Harder to integrate | Moderate | Native support |

**Evolution:**
1. **GANs (2014)**: First high-quality generation, but unstable training
2. **DDPM (2020)**: Better quality and diversity, but very slow
3. **Stable Diffusion (2022)**: Best of both - quality + speed via latent space

**Use Cases:**
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

## Efficiency Techniques

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

## Evolution Over Time

### Language Model Performance (on Common Benchmarks)

| Model (Year) | Params | MMLU | HellaSwag | GSM8k | HumanEval |
|--------------|--------|------|-----------|-------|-----------|
| BERT (2018) | 340M | - | 78% | - | - |
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

### Citations and Influence (Approximate)

| Paper | Citations | Years Since | Cites/Year | Derivatives |
|-------|-----------|-------------|------------|-------------|
| Transformers | 100,000+ | 7 | 14,000+ | Hundreds |
| GANs | 50,000+ | 10 | 5,000+ | 100+ |
| BERT | 80,000+ | 6 | 13,000+ | 50+ |
| GPT-3 | 15,000+ | 4 | 3,750+ | 20+ |
| ResNet | 120,000+ | 9 | 13,000+ | 100+ |
| Diffusion (DDPM) | 10,000+ | 4 | 2,500+ | 30+ |
| CLIP | 8,000+ | 4 | 2,000+ | 40+ |
| LLaMA | 3,000+ | 1 | 3,000+ | 500+ (forks) |

**Most Influential by Citations:** ResNet, Transformers, BERT
**Most Influential by Derivatives:** LLaMA (spawned entire ecosystem in 1 year)
**Fastest Growing:** Scaling Laws, Constitutional AI (recent but accelerating)

---

## When to Use Which Paper's Techniques

### Quick Decision Tree

**Need to generate text?**
- Long-form, creative → GPT-3 / LLaMA
- Factual, grounded → RAG
- With reasoning → Chain-of-Thought
- Specific style → Fine-tuning + LoRA

**Need to understand text?**
- Classification → BERT
- Semantic search → BERT / CLIP (for images)
- Q&A → BERT + RAG

**Need to generate images?**
- Artistic, text-to-image → Stable Diffusion
- Fast, real-time → GANs
- Highest quality → DDPM
- With vision-language → CLIP + Diffusion

**Need to align model?**
- General alignment → RLHF (InstructGPT)
- Safety focus → Constitutional AI
- Both → Hybrid approach

**Need to adapt model?**
- Full resources → Fine-tuning
- Limited resources → LoRA
- No training → RAG or Prompting

**Planning a project?**
- Estimate resources → Scaling Laws
- Choose model size → Scaling Laws + LLaMA lessons

---

## Key Insights from Comparisons

1. **Scaling isn't everything** - LLaMA proved training matters more than size
2. **Hybrid is best** - Combine RLHF + Constitutional AI, RAG + fine-tuning
3. **Efficiency advances** - LoRA makes fine-tuning accessible, Stable Diffusion makes diffusion practical
4. **Open vs closed** - Open models (LLaMA) spawned more innovation than closed (GPT-3)
5. **Architecture consolidation** - Transformers won for both text and vision
6. **Alignment evolution** - From RLHF to Constitutional AI (more scalable)
7. **Knowledge grounding** - RAG reduces hallucination better than any architecture change

---

**Last Updated:** 2025-10-19
