# Quick Reference Guide - All 94 Papers at a Glance

One-page lookup for every paper in the collection. Sorted by category and rough chronology within category. Use Ctrl-F to find a paper by name.

---

## 🏗️ Architectures (9 papers)

| # | Paper | Year | Key Contribution |
|---|-------|------|------------------|
| 63 | **[Word2Vec](../papers/architectures/63-word2vec/)** | 2013 | Dense word embeddings; king − man + woman ≈ queen |
| 64 | **[Seq2Seq](../papers/architectures/64-seq2seq/)** | 2014 | LSTM encoder-decoder for translation |
| 65 | **[Bahdanau Attention](../papers/architectures/65-bahdanau-attention/)** | 2014 | Invented attention; ancestor of the Transformer |
| 66 | **[ResNet](../papers/architectures/66-resnet/)** | 2015 | Residual connections; in every Transformer block |
| 1 | **[Attention Is All You Need](../papers/architectures/01-attention-is-all-you-need/)** | 2017 | The Transformer; self-attention replaces RNNs |
| 11 | **[Vision Transformer (ViT)](../papers/architectures/11-vision-transformer/)** | 2020 | Treats images as patch sequences |
| 74 | **[MAE](../papers/architectures/74-mae-masked-autoencoders/)** | 2021 | BERT-style self-supervised pretraining for vision |
| 20 | **[Mamba](../papers/architectures/20-mamba/)** | 2023 | Selective state spaces, O(n) sequence modeling |
| 37 | **[Mixture-of-Experts](../papers/architectures/37-mixture-of-experts/)** | 2024 | Sparse routing; behind GPT-4, DeepSeek, Llama 4 |

---

## 🤖 Language Models (26 papers)

### Foundations & Open Source
| # | Paper | Year | Key Contribution |
|---|-------|------|------------------|
| 69 | **[GPT-1](../papers/language-models/69-gpt-1/)** | 2018 | Decoder-only Transformer + pretrain/fine-tune recipe |
| 3 | **[BERT](../papers/language-models/03-bert/)** | 2018 | Bidirectional masked language modeling |
| 70 | **[GPT-2](../papers/language-models/70-gpt-2/)** | 2019 | 1.5B params, zero-shot task transfer |
| 68 | **[T5](../papers/language-models/68-t5/)** | 2019 | Text-to-text framing of every NLP task |
| 4 | **[GPT-3](../papers/language-models/04-gpt3-few-shot-learners/)** | 2020 | 175B params, few-shot in-context learning |
| 71 | **[PaLM](../papers/language-models/71-palm/)** | 2022 | 540B dense, Pathways system, BIG-bench |
| 15 | **[LLaMA](../papers/language-models/15-llama/)** | 2023 | 13B matches GPT-3; open-source revolution |
| 17 | **[LLaMA 2](../papers/language-models/17-llama2/)** | 2023 | Commercial license, RLHF alignment |
| 72 | **[Mistral 7B](../papers/language-models/72-mistral-7b/)** | 2023 | GQA + SWA; beat LLaMA-2 13B at 7B |
| 73 | **[Mixtral 8x7B](../papers/language-models/73-mixtral/)** | 2024 | Sparse MoE open weights, 47B/13B active |
| 33 | **[LLaMA 3.3](../papers/language-models/33-llama3.3/)** | 2024 | 70B matches 405B via distillation |
| 27 | **[DeepSeek-V3](../papers/language-models/27-deepseek-v3/)** | 2024 | $5.76M training cost; 671B MoE |
| 41 | **[Llama 4](../papers/language-models/41-llama4/)** | 2025 | Open multimodal MoE; 10M token context |
| 28 | **[Qwen3](../papers/language-models/28-qwen3/)** | 2025 | Unified thinking / non-thinking modes |

### Alignment
| # | Paper | Year | Key Contribution |
|---|-------|------|------------------|
| 5 | **[InstructGPT (RLHF)](../papers/language-models/05-instructgpt-rlhf/)** | 2022 | Human preference fine-tuning enabled ChatGPT |
| 14 | **[Constitutional AI](../papers/language-models/14-constitutional-ai/)** | 2022 | AI self-critique with written principles |
| 19 | **[DPO](../papers/language-models/19-dpo/)** | 2023 | Preference optimization without a reward model |
| 92 | **[Llama Guard](../papers/language-models/92-llama-guard/)** | 2023 | LLM-based safety classifier |

### Frontier & Reasoning Models
| # | Paper | Year | Key Contribution |
|---|-------|------|------------------|
| 36 | **[GPT-4](../papers/language-models/36-gpt4/)** | 2023 | Defined the frontier model era |
| 30 | **[Claude 3.5 Sonnet](../papers/language-models/30-claude-3.5-sonnet/)** | 2024 | Computer use; best coding model of its era |
| 40 | **[GPT-4o](../papers/language-models/40-gpt4o/)** | 2024 | Native omni-model: text/audio/image, 232ms voice |
| 31 | **[OpenAI o1](../papers/language-models/31-openai-o1/)** | 2024 | Started reasoning model era; PhD-level performance |
| 26 | **[DeepSeek-R1](../papers/language-models/26-deepseek-r1/)** | 2025 | Pure-RL reasoning, fully open source |
| 42 | **[GPT-5](../papers/language-models/42-gpt5/)** | 2025 | Unified fast + reasoning model |
| 43 | **[Claude 4 Family](../papers/language-models/43-claude4/)** | 2025-26 | Agentic AI leader; 80.9% SWE-bench |
| 56 | **[Codex](../papers/language-models/56-codex/)** | 2021 | GPT-3 fine-tuned on code; Copilot foundation |

---

## 🎨 Image Generation (12 papers)

| # | Paper | Year | Key Contribution |
|---|-------|------|------------------|
| 67 | **[VAE](../papers/image-generation/67-vae/)** | 2013 | Variational Autoencoders, reparameterization trick |
| 2 | **[GANs](../papers/image-generation/02-generative-adversarial-networks/)** | 2014 | Adversarial generator vs discriminator |
| 76 | **[VQ-VAE](../papers/image-generation/76-vq-vae/)** | 2017 | Discrete-token image representations |
| 75 | **[DDPM](../papers/image-generation/75-ddpm/)** | 2020 | The seminal diffusion paper; beat GANs |
| 6 | **[Diffusion (overview)](../papers/image-generation/06-diffusion-models/)** | 2020 | General diffusion theory |
| 77 | **[VQ-GAN](../papers/image-generation/77-vq-gan/)** | 2021 | VQ-VAE + GAN losses; SD's latent space ancestor |
| 7 | **[Stable Diffusion](../papers/image-generation/07-stable-diffusion/)** | 2022 | Latent diffusion; democratized AI art |
| 78 | **[Imagen](../papers/image-generation/78-imagen/)** | 2022 | Frozen T5-XXL text encoder beats CLIP for prompts |
| 80 | **[DreamBooth](../papers/image-generation/80-dreambooth/)** | 2022 | Personalize diffusion from 3-5 images |
| 79 | **[ControlNet](../papers/image-generation/79-controlnet/)** | 2023 | Conditional control: edges, poses, depth, sketches |
| 48 | **[DALL-E 3](../papers/image-generation/48-dalle3/)** | 2023 | Solved prompt adherence; readable text in images |
| 44 | **[Sora / DiT](../papers/image-generation/44-sora-dit/)** | 2024 | Diffusion Transformers; spacetime patches |

---

## 🔗 Multimodal (7 papers)

| # | Paper | Year | Key Contribution |
|---|-------|------|------------------|
| 8 | **[CLIP](../papers/multimodal/08-clip/)** | 2021 | Vision-language contrastive pretraining |
| 49 | **[Whisper](../papers/multimodal/49-whisper/)** | 2022 | 680K hours speech; foundation model for ASR |
| 46 | **[LLaVA](../papers/multimodal/46-llava/)** | 2023 | Open-source blueprint: vision encoder + LLM |
| 23 | **[GPT-4V(ision)](../papers/multimodal/23-gpt4v/)** | 2023 | GPT-4 with vision |
| 32 | **[SAM 2](../papers/multimodal/32-sam2/)** | 2024 | Universal video segmentation; real-time |
| 29 | **[Gemini 2.5](../papers/multimodal/29-gemini-2.5/)** | 2025 | 1M context; 3-hour video understanding |
| 47 | **[Gemini 3](../papers/multimodal/47-gemini3/)** | 2025 | First 1500+ LMArena ELO; Deep Think mode |

---

## ⚡ Techniques & Methods (40 papers)

### Scaling & Efficient Training
| # | Paper | Year | Key Contribution |
|---|-------|------|------------------|
| 12 | **[Scaling Laws (Kaplan)](../papers/techniques/12-scaling-laws/)** | 2020 | Predictable power laws |
| 18 | **[Chinchilla](../papers/techniques/18-chinchilla/)** | 2022 | Equal scaling of params and tokens |
| 10 | **[LoRA](../papers/techniques/10-lora/)** | 2021 | Low-rank adaptation; 10,000× param reduction |
| 22 | **[QLoRA](../papers/techniques/22-qlora/)** | 2023 | 4-bit quantization + LoRA |

### Inference Efficiency
| # | Paper | Year | Key Contribution |
|---|-------|------|------------------|
| 16 | **[FlashAttention](../papers/techniques/16-flash-attention/)** | 2022 | IO-aware attention, 10-20× faster |
| 45 | **[Speculative Decoding](../papers/techniques/45-speculative-decoding/)** | 2023 | Draft + verify; 2-3× faster inference |
| 52 | **[PagedAttention / vLLM](../papers/techniques/52-pagedattention-vllm/)** | 2023 | Virtual memory for KV cache; 24× throughput |
| 54 | **[RoPE](../papers/techniques/54-rope-rotary-position-embedding/)** | 2021 | Rotary position embeddings, standard in modern LLMs |

### Retrieval & Knowledge
| # | Paper | Year | Key Contribution |
|---|-------|------|------------------|
| 13 | **[RAG](../papers/techniques/13-rag/)** | 2020 | Retrieval-augmented generation |
| 60 | **[Graph RAG](../papers/techniques/60-graph-rag/)** | 2024 | Knowledge-graph augmented retrieval |

### Reasoning & Test-Time Compute
| # | Paper | Year | Key Contribution |
|---|-------|------|------------------|
| 9 | **[Chain-of-Thought](../papers/techniques/09-chain-of-thought/)** | 2022 | "Let's think step by step" |
| 85 | **[Self-Consistency](../papers/techniques/85-self-consistency/)** | 2022 | Majority vote over CoT samples |
| 81 | **[STaR](../papers/techniques/81-star-self-taught-reasoner/)** | 2022 | Bootstrap reasoning from correct answers |
| 25 | **[Tree of Thoughts](../papers/techniques/25-tree-of-thoughts/)** | 2023 | Tree search over reasoning paths |
| 84 | **[Self-Refine](../papers/techniques/84-self-refine/)** | 2023 | LLM as its own critic |
| 51 | **[Process Reward Models](../papers/techniques/51-process-reward-models/)** | 2023 | Step-by-step verification |
| 82 | **[Quiet-STaR](../papers/techniques/82-quiet-star/)** | 2024 | Internal thoughts at every token |
| 50 | **[Test-Time Compute](../papers/techniques/50-test-time-compute/)** | 2024 | Think harder, not bigger |
| 34 | **[Meta-CoT](../papers/techniques/34-meta-cot/)** | 2025 | System 2 metacognitive reasoning |
| 35 | **[rStar-Math](../papers/techniques/35-rstar-math/)** | 2025 | MCTS for math; 7B beats 70B+ |

### RL Training & Preferences
| # | Paper | Year | Key Contribution |
|---|-------|------|------------------|
| 38 | **[GRPO](../papers/techniques/38-grpo/)** | 2024 | RL training without a critic; powers R1 |
| 39 | **[RLVR](../papers/techniques/39-rlvr/)** | 2024-25 | Verifiable rewards replace human prefs |
| 93 | **[KTO](../papers/techniques/93-kto/)** | 2024 | Thumbs-up/down alignment (prospect theory) |

### Agents
| # | Paper | Year | Key Contribution |
|---|-------|------|------------------|
| 21 | **[ReAct](../papers/techniques/21-react/)** | 2023 | Synergize reasoning + acting |
| 24 | **[Toolformer](../papers/techniques/24-toolformer/)** | 2023 | Self-taught tool use |
| 83 | **[Reflexion](../papers/techniques/83-reflexion/)** | 2023 | Verbal RL for agent loops |
| 86 | **[Voyager](../papers/techniques/86-voyager/)** | 2023 | LLM Minecraft agent w/ skill library |
| 58 | **[Generative Agents](../papers/techniques/58-generative-agents/)** | 2023 | Believable simulated humans |
| 91 | **[SWE-bench](../papers/techniques/91-swe-bench/)** | 2023 | The coding-agent benchmark |
| 59 | **[Model Context Protocol](../papers/techniques/59-model-context-protocol/)** | 2024 | Open standard for tool/data integration |

### Interpretability
| # | Paper | Year | Key Contribution |
|---|-------|------|------------------|
| 90 | **[Sparse Autoencoders](../papers/techniques/90-sparse-autoencoders/)** | 2024 | Monosemantic features in Claude 3 Sonnet |

### Scientific & World-Model AI
| # | Paper | Year | Key Contribution |
|---|-------|------|------------------|
| 89 | **[AlphaZero](../papers/techniques/89-alphazero/)** | 2017 | Self-play RL + MCTS; no human data |
| 87 | **[AlphaFold 2](../papers/techniques/87-alphafold2/)** | 2021 | Solved protein folding (Nobel 2024) |
| 97 | **[CICERO](../papers/techniques/97-cicero/)** | 2022 | Human-level Diplomacy AI |
| 95 | **[DreamerV3](../papers/techniques/95-dreamerv3/)** | 2023 | Model-based RL mined diamonds in Minecraft |
| 96 | **[ESM-2 / ESMFold](../papers/techniques/96-esm/)** | 2023 | Protein structure from a language model |
| 94 | **[Genie](../papers/techniques/94-genie/)** | 2024 | Foundation world model from videos |
| 88 | **[AlphaFold 3](../papers/techniques/88-alphafold3/)** | 2024 | Diffusion-based biomolecular interactions |
| 61 | **[AlphaGeometry](../papers/techniques/61-alphageometry/)** | 2024 | IMO-level geometry from LM + symbolic |
| 62 | **[AlphaEvolve](../papers/techniques/62-alphaevolve/)** | 2025 | LLM-driven algorithm discovery |

---

## 📅 By Year

- **2013:** Word2Vec, VAE
- **2014:** GANs, Seq2Seq, Bahdanau Attention
- **2015:** ResNet
- **2017:** Transformer, VQ-VAE, AlphaZero
- **2018:** BERT, GPT-1
- **2019:** GPT-2, T5
- **2020:** GPT-3, ViT, DDPM, Diffusion overview, Scaling Laws, RAG
- **2021:** CLIP, LoRA, MAE, VQ-GAN, AlphaFold 2, RoPE, Codex
- **2022:** Transformer-era cascade — InstructGPT, Whisper, CoT, Stable Diffusion, Constitutional AI, FlashAttention, Chinchilla, Self-Consistency, STaR, PaLM, Imagen, DreamBooth, CICERO
- **2023:** GPT-4, LLaVA, DALL-E 3, PRMs, vLLM, LLaMA 1+2, Mamba, DPO, GPT-4V, ReAct, QLoRA, Toolformer, ToT, Mistral 7B, ControlNet, Self-Refine, Reflexion, Voyager, SWE-bench, Llama Guard, DreamerV3, ESM-2, Generative Agents
- **2024:** Mixtral, GPT-4o, Sora/DiT, GRPO, Test-Time Compute, DeepSeek-V3, o1, Claude 3.5, SAM 2, LLaMA 3.3, Quiet-STaR, KTO, AlphaFold 3, Sparse Autoencoders, Genie, MCP, Graph RAG, AlphaGeometry
- **2025-26:** DeepSeek-R1, RLVR, Qwen3, Gemini 2.5/3, Llama 4, GPT-5, Claude 4, Meta-CoT, rStar-Math, AlphaEvolve

---

## 🎯 Problem → Solution Mapping

**Can't process long sequences in parallel** → Transformer (1)
**Need long-range dependencies before attention** → Bahdanau Attention (65), Seq2Seq (64)
**Need long-range dependencies cheaply** → Mamba (20), RoPE (54), FlashAttention (16)
**Realistic image generation** → GANs (2), DDPM (75), Stable Diffusion (7), Sora (44)
**Personalize a diffusion model** → DreamBooth (80), LoRA (10), ControlNet (79)
**Bidirectional understanding** → BERT (3), MAE (74)
**Few-shot task transfer** → GPT-3 (4), GPT-2 (70), in-context learning
**Help → harmless** → InstructGPT (5), Constitutional AI (14), DPO (19), KTO (93), Llama Guard (92)
**Hallucinations** → RAG (13), Graph RAG (60), PRMs (51)
**Transformers for images** → ViT (11), MAE (74)
**Connect vision and language** → CLIP (8), LLaVA (46), GPT-4V (23)
**Fine-tune cheaply** → LoRA (10), QLoRA (22)
**Better reasoning** → CoT (9), Self-Consistency (85), STaR (81), Quiet-STaR (82), Test-Time Compute (50), o1 (31), R1 (26)
**Open-source frontier model** → LLaMA family (15/17/33/41), Mistral/Mixtral (72/73), DeepSeek (26/27), Qwen3 (28)
**Coding agent** → Codex (56), SWE-bench (91), Claude 3.5 (30), Claude 4 (43)
**Embodied agent** → Voyager (86), DreamerV3 (95), Genie (94), Generative Agents (58)
**Discover new algorithms / science** → AlphaZero (89), AlphaFold 2/3 (87/88), AlphaGeometry (61), AlphaEvolve (62), ESM (96), CICERO (97)
**Understand what a model is "thinking"** → Sparse Autoencoders (90)
**Plan compute budget** → Scaling Laws (12), Chinchilla (18)
**Serve LLMs efficiently** → vLLM (52), Speculative Decoding (45), FlashAttention (16)

---

## 📐 Key Innovations (formulas & diagrams)

```
Self-Attention:     softmax(QKᵀ / √d) · V
Residual block:     y = F(x) + x
VAE ELBO:           E[log p(x|z)] − KL(q(z|x) ‖ p(z))
Diffusion (DDPM):   x_{t-1} = denoise(x_t, t)
LoRA:               W ← W + B·A   with rank(B·A) ≪ rank(W)
RLHF (PPO):         max E[r(x,y)] − β·KL(π ‖ π_ref)
DPO:                max log σ(β·log[π(y_w)/π_ref(y_w)] − β·log[π(y_l)/π_ref(y_l)])
GRPO:               group-relative advantage, no critic
Scaling Laws:       L(N) ∝ N^(−α), L(D) ∝ D^(−α), L(C) ∝ C^(−α)
Chinchilla:         N_opt ∝ √C, D_opt ∝ √C (equal scaling)
RoPE:               rotate Q,K by frequency-dependent angle per position
```

---

**Updated:** 2026-05-26
**Total Papers:** 94
**Estimated Reading Time (all summaries):** 80-100 hours
