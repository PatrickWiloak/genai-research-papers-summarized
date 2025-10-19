# Browse All Papers - Quick Reference

A visual grid view of all 35 papers for quick browsing. **Looking for a learning path?** See the [README](./README.md) for numbered reading orders.

---

## 🏗️ Foundational Architectures

<table>
<tr>
<td width="33%">

**[Attention Is All You Need](./papers/architectures/01-attention-is-all-you-need/)** (2017)
- 🔥 **CRITICAL** - Foundation of everything
- Introduced Transformer architecture
- Self-attention mechanism
- [Paper](https://arxiv.org/abs/1706.03762)

</td>
<td width="33%">

**[Vision Transformer (ViT)](./papers/architectures/11-vision-transformer/)** (2020)
- ⭐ **HIGH** - Transformers for computer vision
- Images as patch sequences
- Enables multimodal models
- [Paper](https://arxiv.org/abs/2010.11929)

</td>
<td width="33%">

**[Mamba](./papers/architectures/20-mamba/)** (2023)
- 🔥 **CRITICAL** - First viable Transformer alternative
- Linear-time sequence modeling (O(n) vs O(n²))
- Selective state spaces
- [Paper](https://arxiv.org/abs/2312.00752)

</td>
</tr>
</table>

---

## 🤖 Language Models

<table>
<tr>
<td width="33%">

**[BERT](./papers/language-models/03-bert/)** (2018)
- 📚 **HISTORICAL** - Pre-training revolution
- Bidirectional pre-training
- Masked language modeling
- [Paper](https://arxiv.org/abs/1810.04805)

</td>
<td width="33%">

**[GPT-3](./papers/language-models/04-gpt3-few-shot-learners/)** (2020)
- ⭐ **HIGH** - Few-shot learning paradigm
- 175B parameters
- Foundation for ChatGPT
- [Paper](https://arxiv.org/abs/2005.14165)

</td>
<td width="33%">

**[InstructGPT (RLHF)](./papers/language-models/05-instructgpt-rlhf/)** (2022)
- 🔥 **CRITICAL** - Human preference learning
- Enabled ChatGPT
- RLHF methodology
- [Paper](https://arxiv.org/abs/2203.02155)

</td>
</tr>
<tr>
<td width="33%">

**[Constitutional AI](./papers/language-models/14-constitutional-ai/)** (2022)
- ⭐ **HIGH** - Alternative to RLHF
- AI self-critique via principles
- Powers Claude
- [Paper](https://arxiv.org/abs/2212.08073)

</td>
<td width="33%">

**[LLaMA](./papers/language-models/15-llama/)** (2023)
- 🔥 **CRITICAL** - Compute-optimal training
- 13B matches GPT-3 175B
- Open weights
- [Paper](https://arxiv.org/abs/2302.13971)

</td>
<td width="33%">

**[LLaMA 2](./papers/language-models/17-llama2/)** (2023)
- 🔥 **CRITICAL** - Production-ready open model
- Commercial license
- RLHF alignment
- [Paper](https://arxiv.org/abs/2307.09288)

</td>
</tr>
<tr>
<td width="33%">

**[DPO](./papers/language-models/19-dpo/)** (2023)
- 🔥 **CRITICAL** - Simpler than RLHF
- Direct preference optimization
- No reward model needed
- [Paper](https://arxiv.org/abs/2305.18290)

</td>
<td width="33%">

**[DeepSeek-R1](./papers/language-models/26-deepseek-r1/)** (2025)
- 🔥 **CRITICAL** - Pure RL reasoning
- Matches OpenAI o1
- Fully open source
- [Paper](https://arxiv.org/abs/2501.12948)

</td>
<td width="33%">

**[DeepSeek-V3](./papers/language-models/27-deepseek-v3/)** (2024)
- 🔥 **CRITICAL** - $5.76M training cost
- 671B MoE architecture
- Matches GPT-4 efficiency
- [Paper](https://arxiv.org/abs/2412.19437)

</td>
</tr>
<tr>
<td width="33%">

**[Qwen3](./papers/language-models/28-qwen3/)** (2025)
- 🔥 **CRITICAL** - Unified thinking/non-thinking
- Adaptive reasoning modes
- Best of both worlds
- [Paper](https://arxiv.org/abs/2505.09388)

</td>
<td width="33%">

**[Claude 3.5 Sonnet](./papers/language-models/30-claude-3.5-sonnet/)** (2024)
- 🔥 **CRITICAL** - Computer use capability
- Best coding model (49% SWE-Bench)
- AI controls computers
- [Announcement](https://www.anthropic.com/news/3-5-models-and-computer-use)

</td>
<td width="33%">

**[OpenAI o1](./papers/language-models/31-openai-o1/)** (2024)
- 🔥 **CRITICAL** - Started reasoning model era
- PhD-level performance
- RL for reasoning
- [Announcement](https://openai.com/index/learning-to-reason-with-llms/)

</td>
</tr>
<tr>
<td width="33%">

**[LLaMA 3.3](./papers/language-models/33-llama3.3/)** (2024)
- 🔥 **HIGH** - Distillation breakthrough
- 70B matches 405B performance
- Knowledge transfer success
- [Paper](https://www.meta.ai/blog/meta-llama-3-3/)

</td>
</tr>
</table>

---

## 🎨 Image Generation

<table>
<tr>
<td width="33%">

**[GANs](./papers/image-generation/02-generative-adversarial-networks/)** (2014)
- 📚 **HISTORICAL** - Generative modeling origins
- Adversarial training
- Generator vs discriminator
- [Paper](https://arxiv.org/abs/1406.2661)

</td>
<td width="33%">

**[Diffusion Models (DDPM)](./papers/image-generation/06-diffusion-models/)** (2020)
- 📖 **THEORY** - Diffusion foundations
- Iterative denoising
- Better than GANs
- [Paper](https://arxiv.org/abs/2006.11239)

</td>
<td width="33%">

**[Stable Diffusion](./papers/image-generation/07-stable-diffusion/)** (2022)
- ⭐ **HIGH** - Practical implementation
- Latent space diffusion (10-100× faster)
- Open-source, democratized AI art
- [Paper](https://arxiv.org/abs/2112.10752)

</td>
</tr>
</table>

---

## 🔗 Multimodal

<table>
<tr>
<td width="50%">

**[CLIP](./papers/multimodal/08-clip/)** (2021)
- ⭐ **HIGH** - Vision-language bridge
- Vision-language contrastive learning
- Zero-shot image classification
- Powers text-to-image models
- [Paper](https://arxiv.org/abs/2103.00020)

</td>
<td width="50%">

**[GPT-4V(ision)](./papers/multimodal/23-gpt4v/)** (2023)
- 🔥 **CRITICAL** - Multimodal frontier model
- GPT-4 with vision capabilities
- State-of-the-art VQA and OCR
- Real-world applications
- [Paper](https://cdn.openai.com/papers/GPTV_System_Card.pdf)

</td>
</tr>
<tr>
<td width="50%">

**[Gemini 2.5](./papers/multimodal/29-gemini-2.5/)** (2025)
- 🔥 **CRITICAL** - Most advanced multimodal AI
- Native multimodal (text, image, audio, video)
- 1M context, 3-hour video understanding
- Integrated thinking mode
- [Paper](https://arxiv.org/abs/2507.06261)

</td>
<td width="50%">

**[SAM 2](./papers/multimodal/32-sam2/)** (2024)
- 🔥 **HIGH** - Universal video segmentation
- 44 FPS real-time performance
- Zero-shot generalization
- Segment anything in video
- [Paper](https://arxiv.org/abs/2408.00714)

</td>
</tr>
</table>

---

## ⚡ Techniques & Methods

<table>
<tr>
<td width="33%">

**[Chain-of-Thought](./papers/techniques/09-chain-of-thought/)** (2022)
- 🔥 **CRITICAL** - Reasoning breakthrough
- Step-by-step reasoning prompts
- "Let's think step by step"
- Improves complex problem-solving
- [Paper](https://arxiv.org/abs/2201.11903)

</td>
<td width="33%">

**[LoRA](./papers/techniques/10-lora/)** (2021)
- 🔥 **CRITICAL** - Efficient fine-tuning
- Low-rank adaptation
- 10,000× fewer trainable parameters
- Enables custom models
- [Paper](https://arxiv.org/abs/2106.09685)

</td>
<td width="33%">

**[Scaling Laws](./papers/techniques/12-scaling-laws/)** (2020)
- 🔥 **CRITICAL** - Predictive theory
- Predictable power laws
- Guides compute allocation
- Justified massive investments
- [Paper](https://arxiv.org/abs/2001.08361)

</td>
</tr>
<tr>
<td width="33%">

**[RAG](./papers/techniques/13-rag/)** (2020)
- 🔥 **CRITICAL** - Production standard
- Retrieval-augmented generation
- Reduces hallucinations
- Production LLM standard
- [Paper](https://arxiv.org/abs/2005.11401)

</td>
<td width="33%">

**[FlashAttention](./papers/techniques/16-flash-attention/)** (2022)
- 🔥 **CRITICAL** - IO-aware attention
- 10-20× faster than standard attention
- Enables 64k+ context lengths
- Powers all modern long-context LLMs
- [Paper](https://arxiv.org/abs/2205.14135)

</td>
<td width="33%">

**[Chinchilla](./papers/techniques/18-chinchilla/)** (2022)
- 🔥 **CRITICAL** - Rewrote scaling laws
- Equal scaling of params and tokens
- Proved GPT-3 was undertrained 4×
- Validated by LLaMA
- [Paper](https://arxiv.org/abs/2203.15556)

</td>
</tr>
<tr>
<td width="33%">

**[ReAct](./papers/techniques/21-react/)** (2023)
- 🔥 **CRITICAL** - AI agents foundation
- Synergizing reasoning and acting
- Interleaves thought and action
- Powers ChatGPT plugins, LangChain
- [Paper](https://arxiv.org/abs/2210.03629)

</td>
<td width="33%">

**[QLoRA](./papers/techniques/22-qlora/)** (2023)
- 🔥 **CRITICAL** - Efficient fine-tuning at scale
- 4-bit quantization + LoRA
- Fine-tune 65B on consumer GPU
- 16× memory reduction
- [Paper](https://arxiv.org/abs/2305.14314)

</td>
<td width="33%">

**[Toolformer](./papers/techniques/24-toolformer/)** (2023)
- ⭐ **HIGH** - Self-taught tool use
- LLMs learn to use tools automatically
- No manual annotations needed
- Inspired ChatGPT function calling
- [Paper](https://arxiv.org/abs/2302.04761)

</td>
</tr>
<tr>
<td width="33%">

**[Tree of Thoughts](./papers/techniques/25-tree-of-thoughts/)** (2023)
- ⭐ **HIGH** - Advanced reasoning
- Tree search over reasoning paths
- Deliberate problem solving
- 18× better than CoT on hard problems
- [Paper](https://arxiv.org/abs/2305.10601)

</td>
<td width="33%">

**[Meta-CoT](./papers/techniques/34-meta-cot/)** (2025)
- 🔥 **HIGH** - System 2 reasoning
- Metacognitive strategies
- Deliberate problem-solving
- Next-gen reasoning approach
- [Paper](https://arxiv.org/abs/2501.xxxxx)

</td>
<td width="33%">

**[rStar-Math](./papers/techniques/35-rstar-math/)** (2025)
- 🔥 **HIGH** - Small models rival large ones
- MCTS for math reasoning
- 7B model beats 70B+ competitors
- Efficient reasoning breakthrough
- [Paper](https://arxiv.org/abs/2501.04519)

</td>
</tr>
</table>

---

## 📊 Quick Stats

| Category | Count |
|----------|-------|
| **Total Papers** | 35 |
| **Foundational Architectures** | 3 |
| **Language Models** | 13 |
| **Image Generation** | 3 |
| **Multimodal** | 4 |
| **Techniques & Methods** | 12 |

---

## 🔍 Filter by Badge

- 🔥 **CRITICAL** (27 papers) - Essential foundational papers
- ⭐ **HIGH** (6 papers) - Important papers with significant impact
- 📚 **HISTORICAL** (2 papers) - Important historical context
- 📖 **THEORY** (1 paper) - Theoretical foundations

---

**Want a structured learning path?** → [README.md](./README.md)

**Need detailed summaries?** → Browse the `/papers` directory

**Total Reading Time:** 30-38 hours | **Words:** 220,000+
