# The Gemini Family (Google), with PaLM, Bard and Gemma

**In one line:** Gemini is Google's closed, natively multimodal model line that replaced PaLM and absorbed the Bard brand, sold as Pro / Flash / Flash-Lite tiers with a frequent-update Flash line, while Gemma is its smaller open-weight sibling - read a name as "generation, then tier".
**Last reviewed:** 2026-09-30

---

## The short version

- **Two eras at Google.** First came separate research models (LaMDA for dialogue, PaLM for scale). In 2023 Google merged Google Brain and DeepMind into Google DeepMind, and Gemini (December 2023) became the single model family behind Google's products.
- **Multimodal and long-context from the start.** Gemini was trained on text, images, audio and video together, not as a text model with add-ons. Gemini 1.5 (February 2024) made million-token context a headline feature.
- **Mixture-of-experts at the core.** Gemini 1.5 introduced a mixture-of-experts (MoE) design, and Google's Gemini 3 Pro model card describes it as a sparse MoE transformer. MoE means only part of the network runs for each token, so total size and per-token cost are decoupled.
- **Flash became the fast-moving line.** Since Gemini 3.5 (May 2026) Google has shipped a new Flash roughly monthly (3.5, 3.6, 3.7, 3.8), while the Pro tier has stayed at 3.1 Pro since February 2026. Gemini 4 is in post-training as of September 2026.
- **Gemini is closed; Gemma is open.** Gemma models (from February 2024) are released as downloadable weights. Gemma 4 (March 31, 2026) moved to the Apache 2.0 licence.

## Release timeline

"Summary" links go to paper summaries in this repo where one exists. Image, speech and robotics offshoots are left out except where they mark a turn in the family.

### Before Gemini

| Model | Date | What changed | Summary |
|---|---|---|---|
| T5 | October 2019 | "Every task is text-to-text"; Google's encoder-decoder baseline. | [65](../../papers/language-models/65-t5/summary.md) |
| FLAN | 2021 | Instruction tuning across many tasks improves zero-shot performance. | [80](../../papers/techniques/80-flan/summary.md) |
| PaLM | April 2022 | 540B-parameter dense model trained with the Pathways system; strong reasoning with chain-of-thought prompting. | [94](../../papers/language-models/94-palm/summary.md) |
| Bard | March 21, 2023 | Google's chat product, first on LaMDA, later on PaLM 2. | - |
| PaLM 2 | May 10, 2023 | Smaller, more multilingual successor to PaLM; powered Bard. | - |

### Gemini

| Model | Date | What changed | Summary |
|---|---|---|---|
| Gemini 1.0 (Ultra, Pro, Nano) | December 6, 2023 | First Gemini; natively multimodal; three sizes from data centre to phone. | - |
| Bard renamed Gemini | February 8, 2024 | Bard becomes the Gemini app; Gemini Ultra 1.0 in a paid plan. | - |
| Gemini 1.5 Pro | February 15, 2024 | Mixture-of-experts; 1M-token context window. | - |
| Gemini 1.5 Flash | May 14, 2024 | First "Flash": a smaller, faster, cheaper tier. | - |
| Gemini 2.0 Flash | December 11, 2024 (experimental); general availability early 2025 | Built for agents: native tool use and image and audio output. | - |
| Gemini 2.5 Pro | March 25, 2025 | "Thinking" model: reasons before answering. | [29](../../papers/multimodal/29-gemini-2.5/summary.md) |
| Gemini 2.5 Flash / Flash-Lite | April 17, 2025 / June 17, 2025 | Thinking brought to the cheaper tiers, with a thinking budget. | [29](../../papers/multimodal/29-gemini-2.5/summary.md) |
| Gemini 3 Pro | November 18, 2025 | Sparse MoE; 1M input tokens, 64K output; text, image, audio and video input. | [47](../../papers/multimodal/47-gemini3/summary.md) |
| Gemini 3 Deep Think | December 2025 (app, Ultra subscribers) | Extended-reasoning mode for hard maths, science and logic. | [47](../../papers/multimodal/47-gemini3/summary.md) |
| Gemini 3 Flash | December 17, 2025 | Frontier-class performance at Flash cost. | - |
| Gemini 3.1 Pro | February 19, 2026 | Stronger core reasoning; still the latest Pro as of September 2026. | - |
| Gemini 3.1 Flash-Lite | March 3, 2026 (preview), May 7, 2026 (GA) | First Flash-Lite of the Gemini 3 generation. | - |
| Gemini 3.5 Flash | May 19, 2026 (Google I/O) | Generally available; becomes the `gemini-flash-latest` alias. | - |
| Gemini 3.6 Flash, 3.5 Flash-Lite, 3.5 Flash Cyber | July 21, 2026 | 3.6 Flash uses 17% fewer output tokens than 3.5 Flash (Artificial Analysis Index); a security-specialised model for government and trusted partners. | - |
| Gemini 3.7 Flash | August 13, 2026 | Software engineering, web development and agent gains. | - |
| Gemini 3.8 Flash, 3.8 Flash Cyber | September 2, 2026 | "Works harder": more reasoning steps and repeated tool calls on complex tasks; 1M input, 64K output. | - |

### Gemma (open weights)

| Model | Date | What changed |
|---|---|---|
| Gemma | February 21, 2024 | 2B and 7B open-weight models built from Gemini research. |
| Gemma 2 | June 27, 2024 (2B on July 31) | 9B and 27B, then 2B. |
| Gemma 3 | March 10, 2025 | 1B, 4B, 12B, 27B; image input on the larger sizes. |
| Gemma 3n | June 26, 2025 | "Effective" E2B and E4B sizes designed for phones and laptops. |
| Gemma 4 | March 31, 2026 | E2B, E4B, 31B dense and 26B MoE (about 4B active); up to 256K context; Apache 2.0 licence. |
| Gemma 4 12B | June 3, 2026 | A unified multimodal model added to the Gemma 4 line. |

Around the core line Google has also released specialised Gemma variants, for example CodeGemma, PaliGemma (vision), ShieldGemma (safety classification), MedGemma (medical), EmbeddingGemma and VaultGemma (trained with differential privacy).

## The through-line

### 1. Scale, then consolidate

Google's research labs produced many of the ideas every model family now uses: the [transformer](../../papers/architectures/01-attention-is-all-you-need/summary.md) itself, [BERT](../../papers/language-models/03-bert/summary.md), [T5](../../papers/language-models/65-t5/summary.md), [mixture-of-experts at scale](../../papers/architectures/67-switch-transformer/summary.md), [chain-of-thought prompting](../../papers/techniques/09-chain-of-thought/summary.md) and [Chinchilla's compute-optimal scaling](../../papers/techniques/18-chinchilla/summary.md). But until 2023 those ideas were spread across separate model lines. [PaLM](../../papers/language-models/94-palm/summary.md) showed Google could train at the largest scale; ChatGPT showed that product was what mattered. Gemini is the consolidation: one family, one lab (Google DeepMind), deployed across Search, Workspace, Android and Cloud.

### 2. Native multimodality

Where other families added vision to a text model, Gemini was trained from the start on interleaved text, images, audio and video. That choice is why Gemini has tended to lead on video and long-document understanding, and why Google ships voice ("Live"), image ("Nano Banana") and robotics variants from the same base. See [CLIP](../../papers/multimodal/08-clip/summary.md) and [LLaVA](../../papers/multimodal/46-llava/summary.md) for the bolt-on approach it is contrasted with.

### 3. Long context as a feature

Gemini 1.5 Pro's 1M-token context (February 2024) was the first time a major model could take in hours of video or a whole codebase in one prompt. The 1M-input, 64K-output shape has held from Gemini 3 Pro through Gemini 3.8 Flash. A big context window is not the same as using it well; see [context windows](../concepts/context-windows.md) and the [RULER benchmark](../../papers/techniques/132-ruler/summary.md) for how effective context is measured.

### 4. Sparse compute: MoE plus distillation

```
            Pro (largest, slowest to update)
             |   knowledge passed down by distillation
             v
            Flash (workhorse, updated roughly monthly in 2026)
             |
             v
            Flash-Lite (cheapest, highest throughput)

   each: sparse mixture-of-experts -> only some "experts" run per token
```

Sparse MoE lets Google grow total capacity without growing per-token cost in proportion. The tier structure leans on [knowledge distillation](../../papers/techniques/134-knowledge-distillation/summary.md), where a smaller model is trained to imitate a larger one. See [mixture of experts](../../papers/architectures/37-mixture-of-experts/summary.md) for the architecture, and the sibling repo's [quantization and distillation](https://github.com/PatrickWiloak/cloud-data-ai-security-zero-to-hero/blob/main/learn/concepts/quantization-and-distillation.md) page.

### 5. Thinking and agents

Gemini 2.5 (March 2025) made "thinking" the default across the family, with a thinking budget developers can set. Deep Think adds a heavier reasoning mode for the hardest problems. By 2026 launch posts lead with agentic coding and tool use: Gemini 3.8 Flash's own announcement says its gains come from executing extra reasoning steps and calling tools iteratively. Background: [reasoning models](../concepts/reasoning-models.md) and [test-time compute](../../papers/techniques/50-test-time-compute/summary.md).

### 6. Security-specialised variants

In 2026 Google began shipping "Cyber" variants of Flash (3.5 Flash Cyber in July, 3.8 Flash Cyber in September) aimed at vulnerability discovery, with the July release limited to government and trusted partners. The general models ship with safeguards against chemical, biological, radiological and nuclear (CBRN) misuse and cyber offence. This mirrors the gated-access pattern in the [Claude](claude.md) and [GPT](gpt.md) families; see [frontier safety frameworks](../policy/frontier-safety-frameworks.md).

## Open vs closed

| Line | Weights | How you get it |
|---|---|---|
| Gemini (all tiers) | Closed | Gemini app, Gemini API (Google AI Studio), Vertex AI, Gemini Enterprise |
| Gemma 1 to 3n | Open, under Google's own Gemma licence and use policy | Hugging Face, Kaggle, local runtimes |
| Gemma 4 | Open, Apache 2.0 | Hugging Face, Kaggle, local runtimes |

Google runs a two-track strategy: its best models stay closed, while Gemma gives developers something to download, fine-tune and run on their own hardware. The switch to Apache 2.0 for Gemma 4 removed custom licence terms, putting Gemma on the same footing as other permissively licensed open models such as [Qwen](qwen.md). See [open vs closed weights](../concepts/open-vs-closed-weights.md) and the [open-source stack](../ecosystem/open-source-stack.md).

## How to read the naming

```
Gemini  3.8   Flash   (-Lite / Cyber / Live / Image ...)
          |     |              |
          |     |              +-- specialised variant or modality
          |     +-- tier: Pro > Flash > Flash-Lite   (Ultra and Nano in 1.0 only)
          +-- generation.point release
```

- **Tiers.** Gemini 1.0 had Ultra, Pro and Nano. From 1.5 the main tiers became **Pro** and **Flash**, with **Flash-Lite** added in 2.0. "Ultra" now mostly names Google's top consumer subscription (Google AI Ultra), not a model.
- **Point numbers move per tier.** As of September 30, 2026 the latest Pro is 3.1, the latest Flash is 3.8, and the latest Flash-Lite is 3.5. A higher number in one tier does not imply a matching release in another. Google's July 2026 post said Gemini 3.5 Pro was "testing with partners".
- **Deep Think** is a reasoning mode on top of a model, not a separate generation.
- **Nano Banana** is the public nickname for Gemini's image generation and editing models (for example Gemini 2.5 Flash Image), not a language model tier.
- **Preview, then stable.** API model IDs often start with a `-preview` suffix (for example `gemini-3-pro-preview`) before a stable ID; aliases like `gemini-flash-latest` move to the newest stable model.
- **Gemma sizes.** "E4B" means "effective 4 billion" parameters in memory at run time. "26B A4B" means 26 billion total, about 4 billion active per token (a mixture-of-experts model).

## What to watch

- **Gemini 4.** Google said in July 2026 it had started "our most ambitious pre-training run yet, for Gemini 4". In late September 2026 Google DeepMind's Koray Kavukcuoglu said the model was in post-training and that Google hoped to release it "much earlier" than the end of 2026. No date has been announced as of this review.
- **The Pro gap.** Gemini 3.1 Pro (February 2026) is now seven months old while Flash has had four updates. Whether 3.5 Pro ships or Google goes straight to Gemini 4 will show whether the frequent-Flash, slower-Pro cadence is deliberate.
- **Retirements.** Gemini 2.0 Flash and Flash-Lite were shut down on June 1, 2026, and from September 18, 2026 access to the 2.5 models is limited to projects with prior usage. Expect the same cycle for early Gemini 3 models.
- **Gemma after Apache 2.0.** Gemma 4's licence change makes it a stronger base for fine-tuning. Watch whether Gemma keeps pace with the closed line and with open models from [Qwen](qwen.md), [DeepSeek](deepseek.md) and [Mistral](mistral.md).

## Read next

- Paper summaries: [PaLM](../../papers/language-models/94-palm/summary.md), [Gemini 2.5](../../papers/multimodal/29-gemini-2.5/summary.md), [Gemini 3](../../papers/multimodal/47-gemini3/summary.md), [T5](../../papers/language-models/65-t5/summary.md), [FLAN](../../papers/techniques/80-flan/summary.md), [Chinchilla](../../papers/techniques/18-chinchilla/summary.md), [Switch Transformer](../../papers/architectures/67-switch-transformer/summary.md), [Imagen](../../papers/image-generation/91-imagen/summary.md), [RT-2](../../papers/robotics/123-rt2/summary.md), [AlphaFold 3](../../papers/techniques/101-alphafold3/summary.md)
- Other families: [GPT and o-series](gpt.md), [Claude](claude.md), [Llama](llama.md)
- Explainers: [Context windows](../concepts/context-windows.md), [Open vs closed weights](../concepts/open-vs-closed-weights.md), [AI hardware landscape](../compute/ai-hardware-landscape.md) (Google trains Gemini on its own TPUs), [Labs landscape](../ecosystem/labs-landscape.md)
- Hands-on: [Transformer architecture](https://github.com/PatrickWiloak/cloud-data-ai-security-zero-to-hero/blob/main/learn/concepts/transformer-architecture.md), [GPUs for AI](https://github.com/PatrickWiloak/cloud-data-ai-security-zero-to-hero/blob/main/learn/concepts/gpus-for-ai.md)

## Sources

- Google AI for Developers, Gemini API changelog (model launch and retirement dates, November 2025 to September 2026): https://ai.google.dev/gemini-api/docs/changelog
- Google AI for Developers, Gemma releases (all Gemma dates and sizes): https://ai.google.dev/gemma/docs/releases
- Google, "Gemma 4: Byte for byte, the most capable open models" (Apache 2.0, sizes, 256K context): https://blog.google/innovation-and-ai/technology/developers-tools/gemma-4/
- Google, "Introducing Gemini 3.8 Flash and 3.8 Flash Cyber": https://blog.google/innovation-and-ai/models-and-research/gemini-models/3-8-flash-and-3-8-flash-cyber/
- Google DeepMind, Gemini 3.8 Flash model card (1M input, 64K output): https://deepmind.google/models/model-cards/gemini-3-8-flash/
- Google, "Introducing Gemini 3.6 Flash, 3.5 Flash-Lite, and 3.5 Flash Cyber" (July 21, 2026; 3.5 Pro testing; Gemini 4 pre-training): https://blog.google/innovation-and-ai/models-and-research/gemini-models/gemini-3-6-flash-3-5-flash-lite-3-5-flash-cyber/
- Google, "Gemini 3.1 Pro: A smarter model for your most complex tasks" (February 19, 2026): https://blog.google/innovation-and-ai/models-and-research/gemini-models/gemini-3-1-pro/
- Google DeepMind, Gemini 3 Pro model card (sparse MoE, 1M input, 64K output): https://storage.googleapis.com/deepmind-media/Model-Cards/Gemini-3-Pro-Model-Card.pdf
- Google, "Gemini 3 Deep Think is now available in the Gemini app": https://blog.google/products/gemini/gemini-3-deep-think/
- Google, "Introducing Gemini 3 Flash": https://blog.google/products-and-platforms/products/gemini/gemini-3-flash/
- Google, "Bard is now Gemini" (February 8, 2024): https://blog.google/products-and-platforms/products/gemini/bard-gemini-advanced-app/
- Google, "Introducing Gemini: our largest and most capable AI model" (December 6, 2023): https://blog.google/innovation-and-ai/technology/ai/google-gemini-ai/
- Gemini Team, "Gemini 1.5: Unlocking multimodal understanding across millions of tokens of context" (MoE, 1M context): https://arxiv.org/abs/2403.05530
- Gemini Team, "Gemini 2.5" technical report: https://arxiv.org/abs/2507.06261
- 9to5Google, "Google says Gemini 4 release is coming 'as soon as possible'" (September 24, 2026, Kavukcuoglu remarks): https://9to5google.com/2026/09/24/google-says-gemini-4-release-is-coming-as-soon-as-possible/
- Google, "Introducing PaLM 2" (May 10, 2023): https://blog.google/technology/ai/google-palm-2-ai-large-language-model/
- Wikipedia, "Gemini (language model)" (cross-check of release dates): https://en.wikipedia.org/wiki/Gemini_(language_model)
