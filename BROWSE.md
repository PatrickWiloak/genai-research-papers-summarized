# Browse All Papers

A visual grid of **every one of the 107 summaries**, grouped by category and ordered by paper number, with an editorial relevance badge and a three-line pitch each. Use it to scan the whole collection quickly; for the generated, always-current listings see the [Paper Index](./INDEX.md) or browse [by topic tag](./TAGS.md). **Looking for a learning path?** See [docs/ROADMAP.md](./docs/ROADMAP.md) or the [README](./README.md) for numbered reading orders. **Wondering what is not here yet?** See [Coverage & Gaps](./docs/GAPS.md).

The badges and pitches on this page are editorial judgements, not claims from the papers.

---

## 🏗️ Foundational Architectures

The backbone papers - what the models are actually built out of, from the 2014 attention mechanism through the Transformer, ViT, sparse MoE and the state-space challengers. **11 papers.**

<table>
<tr>
<td width="33%">

**[Attention Is All You Need](./papers/architectures/01-attention-is-all-you-need/summary.md)** (2017)
- 🔥 **CRITICAL** - Foundation of everything
- Introduced Transformer architecture
- Self-attention mechanism
- [Paper](https://arxiv.org/abs/1706.03762)

</td>
<td width="33%">

**[Vision Transformer (ViT)](./papers/architectures/11-vision-transformer/summary.md)** (2020)
- ⭐ **HIGH** - Transformers for computer vision
- Images as patch sequences
- Enables multimodal models
- [Paper](https://arxiv.org/abs/2010.11929)

</td>
<td width="33%">

**[Mamba](./papers/architectures/20-mamba/summary.md)** (2023)
- 🔥 **CRITICAL** - First viable Transformer alternative
- Linear-time sequence modeling (O(n) vs O(n²))
- Selective state spaces
- [Paper](https://arxiv.org/abs/2312.00752)

</td>
</tr>
<tr>
<td width="33%">

**[Mixtral & Mixture-of-Experts](./papers/architectures/37-mixture-of-experts/summary.md)** (2024)
- 🔥 **CRITICAL** - 47B parameters, 13B active
- Matches LLaMA 2 70B at 6x faster inference
- The architecture DeepSeek-V3, Qwen3 and Llama 4 adopted
- [Paper](https://arxiv.org/abs/2401.04088)

</td>
<td width="33%">

**[Seq2Seq](./papers/architectures/55-seq2seq/summary.md)** (2014)
- 📚 **HISTORICAL** - Variable-length in, variable-length out
- Stacked LSTM encoder-decoder for translation
- Its fixed-size bottleneck is what attention fixed
- [Paper](https://arxiv.org/abs/1409.3215)

</td>
<td width="33%">

**[Bahdanau Attention](./papers/architectures/66-bahdanau-attention/summary.md)** (2014)
- 🔥 **CRITICAL** - Where attention was invented
- Soft alignment over every encoder state
- The conceptual seed of self-attention
- [Paper](https://arxiv.org/abs/1409.0473)

</td>
</tr>
<tr>
<td width="33%">

**[Switch Transformer](./papers/architectures/67-switch-transformer/summary.md)** (2021)
- ⭐ **HIGH** - Trillion parameters at near-constant compute
- Top-1 routing: one expert per token
- The sparsity recipe Mixtral productionized
- [Paper](https://arxiv.org/abs/2101.03961)

</td>
<td width="33%">

**[ResNet](./papers/architectures/73-resnet/summary.md)** (2015)
- 🔥 **CRITICAL** - Residual connections
- `x + f(x)` is in every Transformer
- Most-cited deep learning paper
- [Paper](https://arxiv.org/abs/1512.03385)

</td>
<td width="33%">

**[U-Net](./papers/architectures/74-unet/summary.md)** (2015)
- ⭐ **HIGH** - Encoder-decoder + skips
- The denoiser inside diffusion models
- Trained on 30 images, won its challenge
- [Paper](https://arxiv.org/abs/1505.04597)

</td>
</tr>
<tr>
<td width="33%">

**[Grouped-Query Attention](./papers/architectures/75-grouped-query-attention/summary.md)** (2023)
- 🔥 **CRITICAL** - 8x smaller KV cache
- In Llama 2/3/4, Mistral, Qwen, Gemma
- Why long context is affordable
- [Paper](https://arxiv.org/abs/2305.13245)

</td>
<td width="33%">

**[Masked Autoencoders (MAE)](./papers/architectures/88-mae/summary.md)** (2021)
- ⭐ **HIGH** - BERT-style pretraining for vision
- Hide 75% of patches, reconstruct the rest
- Asymmetric encoder-decoder, cheap to scale
- [Paper](https://arxiv.org/abs/2111.06377)

</td>
<td width="33%"></td>
</tr>
</table>

---

## 🤖 Language Models

The model releases themselves, in lineage order: GPT-1 through GPT-5, the BERT and T5 branch, the Llama and Mistral open-weight line, and the frontier reasoning models. **25 papers.**

<table>
<tr>
<td width="33%">

**[BERT](./papers/language-models/03-bert/summary.md)** (2018)
- 📚 **HISTORICAL** - Pre-training revolution
- Bidirectional pre-training
- Masked language modeling
- [Paper](https://arxiv.org/abs/1810.04805)

</td>
<td width="33%">

**[GPT-3](./papers/language-models/04-gpt3-few-shot-learners/summary.md)** (2020)
- ⭐ **HIGH** - Few-shot learning paradigm
- 175B parameters
- Foundation for ChatGPT
- [Paper](https://arxiv.org/abs/2005.14165)

</td>
<td width="33%">

**[InstructGPT (RLHF)](./papers/language-models/05-instructgpt-rlhf/summary.md)** (2022)
- 🔥 **CRITICAL** - Human preference learning
- Enabled ChatGPT
- RLHF methodology
- [Paper](https://arxiv.org/abs/2203.02155)

</td>
</tr>
<tr>
<td width="33%">

**[Constitutional AI](./papers/language-models/14-constitutional-ai/summary.md)** (2022)
- ⭐ **HIGH** - Alternative to RLHF
- AI self-critique via principles
- Powers Claude
- [Paper](https://arxiv.org/abs/2212.08073)

</td>
<td width="33%">

**[LLaMA](./papers/language-models/15-llama/summary.md)** (2023)
- 🔥 **CRITICAL** - Compute-optimal training
- 13B matches GPT-3 175B
- Open weights
- [Paper](https://arxiv.org/abs/2302.13971)

</td>
<td width="33%">

**[LLaMA 2](./papers/language-models/17-llama2/summary.md)** (2023)
- 🔥 **CRITICAL** - Production-ready open model
- Commercial license
- RLHF alignment
- [Paper](https://arxiv.org/abs/2307.09288)

</td>
</tr>
<tr>
<td width="33%">

**[DPO](./papers/language-models/19-dpo/summary.md)** (2023)
- 🔥 **CRITICAL** - Simpler than RLHF
- Direct preference optimization
- No reward model needed
- [Paper](https://arxiv.org/abs/2305.18290)

</td>
<td width="33%">

**[DeepSeek-R1](./papers/language-models/26-deepseek-r1/summary.md)** (2025)
- 🔥 **CRITICAL** - Pure RL reasoning
- Matches OpenAI o1
- Fully open source
- [Paper](https://arxiv.org/abs/2501.12948)

</td>
<td width="33%">

**[DeepSeek-V3](./papers/language-models/27-deepseek-v3/summary.md)** (2024)
- 🔥 **CRITICAL** - $5.76M training cost
- 671B MoE architecture
- Matches GPT-4 efficiency
- [Paper](https://arxiv.org/abs/2412.19437)

</td>
</tr>
<tr>
<td width="33%">

**[Qwen3](./papers/language-models/28-qwen3/summary.md)** (2025)
- 🔥 **CRITICAL** - Unified thinking/non-thinking
- Adaptive reasoning modes
- Best of both worlds
- [Paper](https://arxiv.org/abs/2505.09388)

</td>
<td width="33%">

**[Claude 3.5 Sonnet](./papers/language-models/30-claude-3.5-sonnet/summary.md)** (2024)
- 🔥 **CRITICAL** - Computer use capability
- Best coding model (49% SWE-Bench)
- AI controls computers
- [Announcement](https://www.anthropic.com/news/3-5-models-and-computer-use)
- [Paper](https://www.anthropic.com/news/3-5-models-and-computer-use)

</td>
<td width="33%">

**[OpenAI o1](./papers/language-models/31-openai-o1/summary.md)** (2024)
- 🔥 **CRITICAL** - Started reasoning model era
- PhD-level performance
- RL for reasoning
- [Announcement](https://openai.com/index/learning-to-reason-with-llms/)
- [Paper](https://openai.com/index/learning-to-reason-with-llms/)

</td>
</tr>
<tr>
<td width="33%">

**[LLaMA 3.3](./papers/language-models/33-llama3.3/summary.md)** (2024)
- ⭐ **HIGH** - Distillation breakthrough
- 70B matches 405B performance
- Knowledge transfer success
- [Paper](https://www.meta.ai/blog/meta-llama-3-3/)

</td>
<td width="33%">

**[GPT-4](./papers/language-models/36-gpt4/summary.md)** (2023)
- 🔥 **CRITICAL** - The capability jump over GPT-3.5
- Bar exam top 10%; first multimodal GPT
- Loss predicted in advance from smaller runs
- [Paper](https://arxiv.org/abs/2303.08774)

</td>
<td width="33%">

**[GPT-4o](./papers/language-models/40-gpt4o/summary.md)** (2024)
- ⭐ **HIGH** - Text, audio and vision in one model
- 232ms average voice response
- Half the price and 2x the speed of GPT-4 Turbo
- [Paper](https://cdn.openai.com/gpt-4o-system-card.pdf)

</td>
</tr>
<tr>
<td width="33%">

**[Llama 4](./papers/language-models/41-llama4/summary.md)** (2025)
- ⭐ **HIGH** - Meta's first MoE models
- 10M-token context window on Scout
- Natively multimodal, open weights
- [Paper](https://ai.meta.com/blog/llama-4-multimodal-intelligence/)

</td>
<td width="33%">

**[GPT-5](./papers/language-models/42-gpt5/summary.md)** (2025)
- 🔥 **CRITICAL** - One model routing fast vs deep reasoning
- 94.6% AIME 2025, 74.9% SWE-bench Verified
- Ended the GPT-4 / o1 / o3 product split
- [Paper](https://cdn.openai.com/gpt-5-system-card.pdf)

</td>
<td width="33%">

**[Claude 4 Family](./papers/language-models/43-claude4/summary.md)** (2025)
- 🔥 **CRITICAL** - State of the art at agentic work
- 80.9% SWE-bench Verified (Opus 4.5)
- Extended thinking with controllable depth
- [Paper](https://www.anthropic.com/news/claude-4)

</td>
</tr>
<tr>
<td width="33%">

**[Codex](./papers/language-models/56-codex/summary.md)** (2021)
- ⭐ **HIGH** - The model behind GitHub Copilot
- Introduced HumanEval and the pass@k metric
- Ancestor of every code model since
- [Paper](https://arxiv.org/abs/2107.03374)

</td>
<td width="33%">

**[GPT-2](./papers/language-models/64-gpt2/summary.md)** (2019)
- 📚 **HISTORICAL** - GPT-1 scaled to 1.5B parameters
- Zero-shot transfer with no fine-tuning
- The staged-release safety debate started here
- [Paper](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)

</td>
<td width="33%">

**[T5](./papers/language-models/65-t5/summary.md)** (2019)
- ⭐ **HIGH** - Every NLP task as text-in, text-out
- Systematic ablation of the whole pretraining recipe
- Gave the field the C4 corpus
- [Paper](https://arxiv.org/abs/1910.10683)

</td>
</tr>
<tr>
<td width="33%">

**[GPT-1](./papers/language-models/93-gpt1/summary.md)** (2018)
- 📚 **HISTORICAL** - Where the GPT recipe begins
- Generative pretraining, then task fine-tuning
- 117M parameters, decoder-only Transformer
- [Paper](https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf)

</td>
<td width="33%">

**[PaLM](./papers/language-models/94-palm/summary.md)** (2022)
- ⭐ **HIGH** - 540B, largest dense Transformer of its day
- Trained across two TPU pods via Pathways
- Where chain-of-thought results broke out
- [Paper](https://arxiv.org/abs/2204.02311)

</td>
<td width="33%">

**[Mistral 7B](./papers/language-models/95-mistral-7b/summary.md)** (2023)
- 🔥 **CRITICAL** - Made open weights genuinely competitive
- Beat LLaMA 2 13B on every benchmark tested
- Apache 2.0, with GQA and sliding-window attention
- [Paper](https://arxiv.org/abs/2310.06825)

</td>
</tr>
<tr>
<td width="33%">

**[Llama Guard](./papers/language-models/96-llama-guard/summary.md)** (2023)
- ⭐ **HIGH** - Safety classification as an LLM task
- Moderates both the prompt and the response
- The open-source production reference
- [Paper](https://arxiv.org/abs/2312.06674)

</td>
<td width="33%"></td>
<td width="33%"></td>
</tr>
</table>

---

## 🎨 Image & Video Generation

From VAEs and GANs through the diffusion pipeline that replaced them, the tokenizer line that made images modellable, and the video models built on top. **14 papers.**

<table>
<tr>
<td width="33%">

**[GANs](./papers/image-generation/02-generative-adversarial-networks/summary.md)** (2014)
- 📚 **HISTORICAL** - Generative modeling origins
- Adversarial training
- Generator vs discriminator
- [Paper](https://arxiv.org/abs/1406.2661)

</td>
<td width="33%">

**[Diffusion Models (DDPM)](./papers/image-generation/06-diffusion-models/summary.md)** (2020)
- 📖 **THEORY** - Diffusion foundations
- Iterative denoising
- Better than GANs
- [Paper](https://arxiv.org/abs/2006.11239)

</td>
<td width="33%">

**[Stable Diffusion](./papers/image-generation/07-stable-diffusion/summary.md)** (2022)
- ⭐ **HIGH** - Practical implementation
- Latent space diffusion (10-100× faster)
- Open-source, democratized AI art
- [Paper](https://arxiv.org/abs/2112.10752)

</td>
</tr>
<tr>
<td width="33%">

**[Sora & Diffusion Transformers](./papers/image-generation/44-sora-dit/summary.md)** (2022-2024)
- 🔥 **CRITICAL** - Diffusion on a Transformer backbone
- Video as spacetime patches, any resolution
- Launched the AI video generation industry
- [Paper](https://arxiv.org/abs/2212.09748)

</td>
<td width="33%">

**[DALL-E 3](./papers/image-generation/48-dalle3/summary.md)** (2023)
- ⭐ **HIGH** - Follows the prompt you actually wrote
- Recaptioned training data was the whole trick
- First model to render readable text
- [Paper](https://cdn.openai.com/papers/dall-e-3.pdf)

</td>
<td width="33%">

**[Variational Autoencoder (VAE)](./papers/image-generation/57-vae/summary.md)** (2013)
- 🔥 **CRITICAL** - The reparameterization trick
- A smooth, interpolatable latent space
- The latent half of latent diffusion
- [Paper](https://arxiv.org/abs/1312.6114)

</td>
</tr>
<tr>
<td width="33%">

**[Classifier-Free Guidance](./papers/image-generation/69-classifier-free-guidance/summary.md)** (2021)
- 🔥 **CRITICAL** - The "CFG scale" slider
- Two predictions, one difference vector
- Also how negative prompts work
- [Paper](https://arxiv.org/abs/2207.12598)

</td>
<td width="33%">

**[DDIM](./papers/image-generation/70-ddim/summary.md)** (2020)
- 🔥 **CRITICAL** - 1,000 steps down to 20
- Deterministic sampling, no retraining
- Basis of diffusion image editing
- [Paper](https://arxiv.org/abs/2010.02502)

</td>
<td width="33%">

**[ControlNet](./papers/image-generation/71-controlnet/summary.md)** (2023)
- 🔥 **CRITICAL** - Edges, depth, pose control
- Zero convolutions, base model frozen
- ICCV 2023 best paper
- [Paper](https://arxiv.org/abs/2302.05543)

</td>
</tr>
<tr>
<td width="33%">

**[Flow Matching / SD3](./papers/image-generation/72-flow-matching-sd3/summary.md)** (2022-2024)
- 🔥 **CRITICAL** - What replaced diffusion
- Straight noise-to-image paths
- Powers SD3, SD3.5 and Flux
- [Paper](https://arxiv.org/abs/2403.03206)

</td>
<td width="33%">

**[VQ-VAE](./papers/image-generation/89-vq-vae/summary.md)** (2017)
- ⭐ **HIGH** - Discrete codebook latents
- Straight-through estimator for the argmin
- Made images tokenizable like text
- [Paper](https://arxiv.org/abs/1711.00937)

</td>
<td width="33%">

**[VQ-GAN](./papers/image-generation/90-vq-gan/summary.md)** (2020)
- ⭐ **HIGH** - VQ-VAE latents plus an adversarial loss
- Transformers autoregress over image tokens
- The compress-then-model recipe
- [Paper](https://arxiv.org/abs/2012.09841)

</td>
</tr>
<tr>
<td width="33%">

**[Imagen](./papers/image-generation/91-imagen/summary.md)** (2022)
- ⭐ **HIGH** - A frozen T5 encoder beat bigger image models
- Text understanding was the real bottleneck
- Cascaded super-resolution diffusion
- [Paper](https://arxiv.org/abs/2205.11487)

</td>
<td width="33%">

**[DreamBooth](./papers/image-generation/92-dreambooth/summary.md)** (2022)
- 🔥 **CRITICAL** - Your subject, any scene, from 3-5 images
- Rare-token binding plus prior-preservation loss
- The blueprint every personalization method copies
- [Paper](https://arxiv.org/abs/2208.12242)

</td>
<td width="33%"></td>
</tr>
</table>

---

## 🔗 Multimodal

Models that cross modalities - vision-language, speech, video, and the omni-models that fold them into a single network. **7 papers.**

<table>
<tr>
<td width="33%">

**[CLIP](./papers/multimodal/08-clip/summary.md)** (2021)
- ⭐ **HIGH** - Vision-language bridge
- Vision-language contrastive learning
- Zero-shot image classification
- Powers text-to-image models
- [Paper](https://arxiv.org/abs/2103.00020)

</td>
<td width="33%">

**[GPT-4V(ision)](./papers/multimodal/23-gpt4v/summary.md)** (2023)
- 🔥 **CRITICAL** - Multimodal frontier model
- GPT-4 with vision capabilities
- State-of-the-art VQA and OCR
- Real-world applications
- [Paper](https://cdn.openai.com/papers/GPTV_System_Card.pdf)

</td>
<td width="33%">

**[Gemini 2.5](./papers/multimodal/29-gemini-2.5/summary.md)** (2025)
- 🔥 **CRITICAL** - Most advanced multimodal AI
- Native multimodal (text, image, audio, video)
- 1M context, 3-hour video understanding
- Integrated thinking mode
- [Paper](https://arxiv.org/abs/2507.06261)

</td>
</tr>
<tr>
<td width="33%">

**[SAM 2](./papers/multimodal/32-sam2/summary.md)** (2024)
- ⭐ **HIGH** - Universal video segmentation
- 44 FPS real-time performance
- Zero-shot generalization
- Segment anything in video
- [Paper](https://arxiv.org/abs/2408.00714)

</td>
<td width="33%">

**[LLaVA](./papers/multimodal/46-llava/summary.md)** (2023)
- 🔥 **CRITICAL** - Vision encoder + projection + LLM
- Instruction data generated by text-only GPT-4
- Democratized open vision-language models
- [Paper](https://arxiv.org/abs/2304.08485)

</td>
<td width="33%">

**[Gemini 3](./papers/multimodal/47-gemini3/summary.md)** (2025)
- ⭐ **HIGH** - First model past 1500 LMArena Elo
- 95% AIME 2025, 91.8% MMLU
- Deep Think mode for the hardest reasoning
- [Paper](https://blog.google/products-and-platforms/products/gemini/gemini-3/)

</td>
</tr>
<tr>
<td width="33%">

**[Whisper](./papers/multimodal/49-whisper/summary.md)** (2022)
- 🔥 **CRITICAL** - 680,000 hours of weakly supervised audio
- 99 languages, zero-shot, no fine-tuning
- The default open speech recognition model
- [Paper](https://arxiv.org/abs/2212.04356)

</td>
<td width="33%"></td>
<td width="33%"></td>
</tr>
</table>

---

## ⚡ Techniques & Methods

The methods layer: how models are trained, aligned, made to reason, served fast, evaluated, interpreted, and pointed at problems outside language. **50 papers.**

<table>
<tr>
<td width="33%">

**[Chain-of-Thought](./papers/techniques/09-chain-of-thought/summary.md)** (2022)
- 🔥 **CRITICAL** - Reasoning breakthrough
- Step-by-step reasoning prompts
- "Let's think step by step"
- Improves complex problem-solving
- [Paper](https://arxiv.org/abs/2201.11903)

</td>
<td width="33%">

**[LoRA](./papers/techniques/10-lora/summary.md)** (2021)
- 🔥 **CRITICAL** - Efficient fine-tuning
- Low-rank adaptation
- 10,000× fewer trainable parameters
- Enables custom models
- [Paper](https://arxiv.org/abs/2106.09685)

</td>
<td width="33%">

**[Scaling Laws](./papers/techniques/12-scaling-laws/summary.md)** (2020)
- 🔥 **CRITICAL** - Predictive theory
- Predictable power laws
- Guides compute allocation
- Justified massive investments
- [Paper](https://arxiv.org/abs/2001.08361)

</td>
</tr>
<tr>
<td width="33%">

**[RAG](./papers/techniques/13-rag/summary.md)** (2020)
- 🔥 **CRITICAL** - Production standard
- Retrieval-augmented generation
- Reduces hallucinations
- Production LLM standard
- [Paper](https://arxiv.org/abs/2005.11401)

</td>
<td width="33%">

**[FlashAttention](./papers/techniques/16-flash-attention/summary.md)** (2022)
- 🔥 **CRITICAL** - IO-aware attention
- 10-20× faster than standard attention
- Enables 64k+ context lengths
- Powers all modern long-context LLMs
- [Paper](https://arxiv.org/abs/2205.14135)

</td>
<td width="33%">

**[Chinchilla](./papers/techniques/18-chinchilla/summary.md)** (2022)
- 🔥 **CRITICAL** - Rewrote scaling laws
- Equal scaling of params and tokens
- Proved GPT-3 was undertrained 4×
- Validated by LLaMA
- [Paper](https://arxiv.org/abs/2203.15556)

</td>
</tr>
<tr>
<td width="33%">

**[ReAct](./papers/techniques/21-react/summary.md)** (2023)
- 🔥 **CRITICAL** - AI agents foundation
- Synergizing reasoning and acting
- Interleaves thought and action
- Powers ChatGPT plugins, LangChain
- [Paper](https://arxiv.org/abs/2210.03629)

</td>
<td width="33%">

**[QLoRA](./papers/techniques/22-qlora/summary.md)** (2023)
- 🔥 **CRITICAL** - Efficient fine-tuning at scale
- 4-bit quantization + LoRA
- Fine-tune 65B on consumer GPU
- 16× memory reduction
- [Paper](https://arxiv.org/abs/2305.14314)

</td>
<td width="33%">

**[Toolformer](./papers/techniques/24-toolformer/summary.md)** (2023)
- ⭐ **HIGH** - Self-taught tool use
- LLMs learn to use tools automatically
- No manual annotations needed
- Inspired ChatGPT function calling
- [Paper](https://arxiv.org/abs/2302.04761)

</td>
</tr>
<tr>
<td width="33%">

**[Tree of Thoughts](./papers/techniques/25-tree-of-thoughts/summary.md)** (2023)
- ⭐ **HIGH** - Advanced reasoning
- Tree search over reasoning paths
- Deliberate problem solving
- 18× better than CoT on hard problems
- [Paper](https://arxiv.org/abs/2305.10601)

</td>
<td width="33%">

**[Meta-CoT](./papers/techniques/34-meta-cot/summary.md)** (2025)
- ⭐ **HIGH** - System 2 reasoning
- Metacognitive strategies
- Deliberate problem-solving
- Next-gen reasoning approach
- [Paper](https://arxiv.org/abs/2501.xxxxx)

</td>
<td width="33%">

**[rStar-Math](./papers/techniques/35-rstar-math/summary.md)** (2025)
- ⭐ **HIGH** - Small models rival large ones
- MCTS for math reasoning
- 7B model beats 70B+ competitors
- Efficient reasoning breakthrough
- [Paper](https://arxiv.org/abs/2501.04519)

</td>
</tr>
<tr>
<td width="33%">

**[GRPO](./papers/techniques/38-grpo/summary.md)** (2024)
- 🔥 **CRITICAL** - The algorithm behind DeepSeek-R1
- Drops PPO's critic model entirely
- Group-relative advantage, ~50% less memory
- [Paper](https://arxiv.org/abs/2402.03300)

</td>
<td width="33%">

**[RLVR](./papers/techniques/39-rlvr/summary.md)** (2024-2025)
- 🔥 **CRITICAL** - Reward correctness, not human preference
- No reward model to train or to game
- Where reasoning models get their training signal
- [Paper](https://arxiv.org/abs/2501.12948)

</td>
<td width="33%">

**[Speculative Decoding](./papers/techniques/45-speculative-decoding/summary.md)** (2022)
- 🔥 **CRITICAL** - 2-3x faster, outputs provably unchanged
- Small draft model proposes, big model verifies
- Deployed in every serious serving stack
- [Paper](https://arxiv.org/abs/2211.17192)

</td>
</tr>
<tr>
<td width="33%">

**[Scaling Test-Time Compute](./papers/techniques/50-test-time-compute/summary.md)** (2024)
- 🔥 **CRITICAL** - Think longer instead of training bigger
- A small model that searches can match a 14x larger one
- The theory underneath o1 and R1
- [Paper](https://arxiv.org/abs/2408.03314)

</td>
<td width="33%">

**[Process Reward Models](./papers/techniques/51-process-reward-models/summary.md)** (2023)
- 🔥 **CRITICAL** - Grade every step, not just the answer
- PRM800K: 800,000 step-level human labels
- How reasoning models verify themselves
- [Paper](https://arxiv.org/abs/2305.20050)

</td>
<td width="33%">

**[PagedAttention & vLLM](./papers/techniques/52-pagedattention-vllm/summary.md)** (2023)
- 🔥 **CRITICAL** - OS paging applied to the KV cache
- KV memory waste from 60-80% down to under 4%
- 24x throughput; vLLM became the standard
- [Paper](https://arxiv.org/abs/2309.06180)

</td>
</tr>
<tr>
<td width="33%">

**[Word2Vec](./papers/techniques/53-word2vec/summary.md)** (2013)
- 📚 **HISTORICAL** - Dense embeddings replace one-hot vectors
- king - man + woman ≈ queen
- Started representation learning for text
- [Paper](https://arxiv.org/abs/1301.3781)

</td>
<td width="33%">

**[RoPE](./papers/techniques/54-rope-rotary-position-embedding/summary.md)** (2021)
- 🔥 **CRITICAL** - Relative position by rotating the query and key
- Zero learned parameters, a few dozen lines
- In LLaMA, PaLM, Gemini, Mistral, DeepSeek, Qwen
- [Paper](https://arxiv.org/abs/2104.09864)

</td>
<td width="33%">

**[Generative Agents](./papers/techniques/58-generative-agents/summary.md)** (2023)
- ⭐ **HIGH** - 25 agents living in a sandbox town
- The memory -> reflection -> planning loop
- Unscripted emergent social behavior
- [Paper](https://arxiv.org/abs/2304.03442)

</td>
</tr>
<tr>
<td width="33%">

**[Model Context Protocol (MCP)](./papers/techniques/59-model-context-protocol/summary.md)** (2024)
- 🔥 **CRITICAL** - One protocol for model-to-tool wiring
- Turns M x N bespoke integrations into M + N
- Industry-wide adoption within months
- [Paper](https://www.anthropic.com/news/model-context-protocol)

</td>
<td width="33%">

**[GraphRAG](./papers/techniques/60-graph-rag/summary.md)** (2024)
- ⭐ **HIGH** - Answers global "what are the themes?" queries
- LLM-extracted knowledge graph over the corpus
- Hierarchical community summaries, map-reduce answering
- [Paper](https://arxiv.org/abs/2404.16130)

</td>
<td width="33%">

**[AlphaGeometry](./papers/techniques/61-alphageometry/summary.md)** (2024)
- ⭐ **HIGH** - 25/30 IMO geometry problems solved
- Neuro-symbolic: LM proposes, solver deduces
- Trained on 100M synthetic theorems, zero human proofs
- [Paper](https://www.nature.com/articles/s41586-023-06747-5)

</td>
</tr>
<tr>
<td width="33%">

**[AlphaEvolve](./papers/techniques/62-alphaevolve/summary.md)** (2025)
- ⭐ **HIGH** - Broke Strassen's 1969 matrix-multiply record
- Gemini proposes, evolutionary search selects
- Already running in Google production
- [Paper](https://deepmind.google/discover/blog/alphaevolve-a-gemini-powered-coding-agent-for-designing-advanced-algorithms/)

</td>
<td width="33%">

**[PPO](./papers/techniques/63-ppo/summary.md)** (2017)
- 🔥 **CRITICAL** - The RL algorithm underneath RLHF
- Clipped objective: TRPO's stability without the pain
- What InstructGPT and ChatGPT were trained with
- [Paper](https://arxiv.org/abs/1707.06347)

</td>
<td width="33%">

**[AlphaFold 2](./papers/techniques/68-alphafold/summary.md)** (2021)
- 🔥 **CRITICAL** - Solved 50-year protein folding problem
- Attention generalizing far beyond language
- 200M+ predicted structures released free
- [Paper](https://www.nature.com/articles/s41586-021-03819-2)

</td>
</tr>
<tr>
<td width="33%">

**[ZeRO & Megatron-LM](./papers/techniques/76-zero-megatron/summary.md)** (2019)
- 🔥 **CRITICAL** - 3D parallelism
- The systems layer under every big run
- DeepSpeed, FSDP, tensor parallelism
- [Paper](https://arxiv.org/abs/1910.02054)

</td>
<td width="33%">

**[Self-Consistency](./papers/techniques/77-self-consistency/summary.md)** (2022)
- 🔥 **CRITICAL** - Sample N, majority vote
- +17.9 points on GSM8K over greedy CoT
- Inference compute buys accuracy
- [Paper](https://arxiv.org/abs/2203.11171)

</td>
<td width="33%">

**[Reflexion](./papers/techniques/78-reflexion/summary.md)** (2023)
- 🔥 **CRITICAL** - Agents learn from failure
- Verbal reflection, no weight updates
- The retry loop every coding agent runs
- [Paper](https://arxiv.org/abs/2303.11366)

</td>
</tr>
<tr>
<td width="33%">

**[Self-Instruct](./papers/techniques/79-self-instruct/summary.md)** (2022)
- 🔥 **CRITICAL** - 52K instructions from 175
- Made instruction data free; led to Alpaca
- Why post-training data is synthetic now
- [Paper](https://arxiv.org/abs/2212.10560)

</td>
<td width="33%">

**[FLAN](./papers/techniques/80-flan/summary.md)** (2021)
- 🔥 **CRITICAL** - Invented instruction tuning
- Beat GPT-3 175B on 20 of 25 benchmarks
- The middle stage of the modern pipeline
- [Paper](https://arxiv.org/abs/2109.01652)

</td>
<td width="33%">

**[Emergent Abilities](./papers/techniques/81-emergent-abilities/summary.md)** (2022-2023)
- ⭐ **HIGH** - Do capabilities jump at scale?
- Paired with the "Mirage" rebuttal
- Why exact-match alone misleads
- [Paper](https://arxiv.org/abs/2206.07682)

</td>
</tr>
<tr>
<td width="33%">

**[Sparse Autoencoders](./papers/techniques/82-sparse-autoencoders/summary.md)** (2022-2024)
- 🔥 **CRITICAL** - Reading a model's concepts
- Millions of features from Claude 3 Sonnet
- Steering works: "Golden Gate Claude"
- [Paper](https://transformer-circuits.pub/2024/scaling-monosemanticity/index.html)

</td>
<td width="33%">

**[Sleeper Agents](./papers/techniques/83-sleeper-agents/summary.md)** (2024)
- 🔥 **CRITICAL** - Backdoors survive safety training
- SFT, RLHF and adversarial training all failed
- Adversarial training hid the trigger better
- [Paper](https://arxiv.org/abs/2401.05566)

</td>
<td width="33%">

**[SWE-bench](./papers/techniques/84-swe-bench/summary.md)** (2023)
- 🔥 **CRITICAL** - Real GitHub issues, real tests
- ~2% resolved in 2023, ~70-80% in 2025
- The coding-agent headline metric
- [Paper](https://arxiv.org/abs/2310.06770)

</td>
</tr>
<tr>
<td width="33%">

**[LLM-as-a-Judge](./papers/techniques/85-llm-as-judge/summary.md)** (2023)
- 🔥 **CRITICAL** - Model-graded evaluation
- ~80% agreement with humans, plus the biases
- Created Chatbot Arena Elo
- [Paper](https://arxiv.org/abs/2306.05685)

</td>
<td width="33%">

**[GPTQ & AWQ](./papers/techniques/86-gptq-awq-quantization/summary.md)** (2022-2023)
- 🔥 **CRITICAL** - 4-bit post-training quantization
- 70B from ~140 GB to ~35 GB
- Why local inference exists
- [Paper](https://arxiv.org/abs/2306.00978)

</td>
<td width="33%">

**[Dense Retrieval](./papers/techniques/87-dense-retrieval/summary.md)** (2019-2020)
- 🔥 **CRITICAL** - DPR, ColBERT, Sentence-BERT
- The retriever every RAG system runs on
- Where RAG actually fails, and the fixes
- [Paper](https://arxiv.org/abs/2004.04906)

</td>
</tr>
<tr>
<td width="33%">

**[STaR](./papers/techniques/97-star/summary.md)** (2022)
- ⭐ **HIGH** - Generate rationales, keep the ones that work
- Rationalize backwards from known answers
- The quiet ancestor of o1 and R1
- [Paper](https://arxiv.org/abs/2203.14465)

</td>
<td width="33%">

**[Quiet-STaR](./papers/techniques/98-quiet-star/summary.md)** (2024)
- ⭐ **HIGH** - Learn to think before every token
- Generalizes STaR past math to arbitrary text
- Predates o1's hidden chain-of-thought
- [Paper](https://arxiv.org/abs/2403.09629)

</td>
<td width="33%">

**[Self-Refine](./papers/techniques/99-self-refine/summary.md)** (2023)
- ⭐ **HIGH** - One model critiques and rewrites itself
- No extra training, no second model
- Codified the LLM-as-critic pattern
- [Paper](https://arxiv.org/abs/2303.17651)

</td>
</tr>
<tr>
<td width="33%">

**[Voyager](./papers/techniques/100-voyager/summary.md)** (2023)
- ⭐ **HIGH** - Lifelong open-ended agent in Minecraft
- Grows a reusable skill library written as code
- GPT-4 planning over multi-hour horizons
- [Paper](https://arxiv.org/abs/2305.16291)

</td>
<td width="33%">

**[AlphaFold 3](./papers/techniques/101-alphafold3/summary.md)** (2024)
- ⭐ **HIGH** - Proteins plus DNA, RNA, ligands and ions
- A diffusion module replaces the structure head
- Predicts interactions, not just single shapes
- [Paper](https://www.nature.com/articles/s41586-024-07487-w)

</td>
<td width="33%">

**[AlphaZero](./papers/techniques/102-alphazero/summary.md)** (2017)
- 🔥 **CRITICAL** - Superhuman from self-play and the rules alone
- MCTS guided by a single policy-value network
- Ancestor of every self-improvement loop in AI
- [Paper](https://arxiv.org/abs/1712.01815)

</td>
</tr>
<tr>
<td width="33%">

**[KTO](./papers/techniques/103-kto/summary.md)** (2024)
- ⭐ **HIGH** - Alignment without paired preference data
- Prospect theory turned into a loss function
- Thumbs-up / thumbs-down signals are enough
- [Paper](https://arxiv.org/abs/2402.01306)

</td>
<td width="33%">

**[Genie](./papers/techniques/104-genie/summary.md)** (2024)
- ⭐ **HIGH** - A playable world from a single image
- Latent actions learned unsupervised from video
- The first foundation world model
- [Paper](https://arxiv.org/abs/2402.15391)

</td>
<td width="33%">

**[DreamerV3](./papers/techniques/105-dreamerv3/summary.md)** (2023)
- ⭐ **HIGH** - Minecraft diamonds from scratch, no demos
- One hyperparameter set across 150+ tasks
- Learn a world model, then plan in imagination
- [Paper](https://arxiv.org/abs/2301.04104)

</td>
</tr>
<tr>
<td width="33%">

**[ESM-2 / ESMFold](./papers/techniques/106-esm/summary.md)** (2023)
- ⭐ **HIGH** - A 15B language model trained on protein sequences
- Structure with no MSA search, up to 60x faster
- 617M predicted metagenomic structures released
- [Paper](https://doi.org/10.1126/science.ade2574)

</td>
<td width="33%">

**[CICERO](./papers/techniques/107-cicero/summary.md)** (2022)
- ⭐ **HIGH** - Human-level Diplomacy, negotiation included
- Language generation fused with strategic planning
- Top 10% across 40 games against humans
- [Paper](https://doi.org/10.1126/science.ade9097)

</td>
<td width="33%"></td>
</tr>
</table>

---

## 📊 Quick Stats

Every paper in the collection has a card on this page.

| Category | Papers |
|----------|--------|
| **Foundational Architectures** | 11 |
| **Language Models** | 25 |
| **Image & Video Generation** | 14 |
| **Multimodal** | 7 |
| **Techniques & Methods** | 50 |
| **Total** | **107** |

The authoritative counts live in [`papers.json`](./papers.json) and are regenerated by
`scripts/build_manifest.py`; `scripts/check_counts.py` fails CI if the table above drifts from it.

---

## 🔍 Filter by Badge

Relevance ratings are editorial, and describe how much the paper matters to a reader today:

- 🔥 **CRITICAL** (60 papers) - Essential - read these to understand the field
- ⭐ **HIGH** (40 papers) - Important, with significant downstream impact
- 📚 **HISTORICAL** (6 papers) - Formative context; superseded in practice
- 📖 **THEORY** (1 paper) - Theoretical foundations

---

**Want a structured learning path?** → [docs/ROADMAP.md](./docs/ROADMAP.md) · [README.md](./README.md)

**Want the generated listings?** → [INDEX.md](./INDEX.md) · [TAGS.md](./TAGS.md)

**Wondering what is missing?** → [docs/GAPS.md](./docs/GAPS.md)

