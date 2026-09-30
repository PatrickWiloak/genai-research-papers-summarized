# The AI Labs Landscape

**In one line:** As of September 2026 the frontier is set by a handful of US labs that keep their best weights closed (OpenAI, Anthropic, Google DeepMind, xAI), while the strongest downloadable models come mostly from Chinese labs (DeepSeek, Alibaba Qwen, Moonshot, Z.ai) and Europe's Mistral - so when you pick a model, the lab's weights policy and licence matter as much as its benchmark scores.
**Last reviewed:** 2026-09-30

---

## The short version

- **Three kinds of organisation build large models.** Independent frontier labs (OpenAI, Anthropic, xAI, Mistral, DeepSeek, Moonshot, Z.ai), labs inside big tech companies (Google DeepMind, Meta Superintelligence Labs, Microsoft AI, Amazon, Alibaba's Qwen team, Apple), and a long tail of smaller or specialist groups. The first two groups account for nearly every model in this repo.
- **Closed at the top, open underneath.** The US frontier labs keep their flagship weights private and sell access through APIs and apps. Several of them also publish smaller open-weight models (OpenAI's gpt-oss, Google's Gemma, Meta's Muse Glimmer).
- **Open weights are led from China and France.** DeepSeek, Qwen, Moonshot (Kimi), Z.ai (GLM) and MiniMax release large models with downloadable weights, some of them over 2 trillion parameters. Mistral is the main European open-weight lab.
- **"Open" licences diverged in 2026.** DeepSeek stays on plain MIT; Qwen, Kimi, GLM and MiniMax added clauses aimed at companies that would resell the model as a hosted service; Mistral's large models cap free use by company revenue. Read the licence, not the label.
- **Partnerships blur the lines.** Microsoft owns about 27% of OpenAI; Amazon and Google are investors in Anthropic; Apple builds its next foundation models on Google's Gemini; Alibaba holds a large stake in Moonshot. The "who competes with whom" picture is a web, not a list.

## How to read the landscape

Three questions sort most labs quickly.

```
1. Do they release weights?     closed  <-------- mixed -------->  open
                                Anthropic    OpenAI, Google,       DeepSeek, Qwen,
                                             Meta, xAI, Mistral    Moonshot, Z.ai

2. How do they make money?      API + apps | cloud platform | ads / devices | none yet (research-led)

3. Where are they, and whose    US | China | Europe
   chips can they buy?          (export controls shape Chinese labs' designs)
```

A lab's answers explain a lot of its technical choices. Chinese labs working with export-restricted chips publish heavily on efficiency (see the [DeepSeek](../model-families/deepseek.md) and [Qwen](../model-families/qwen.md) pages). Labs that sell API access guard weights because the weights are the product. Big tech labs with other revenue (ads, cloud, devices) can afford to give some models away to shape the ecosystem. The [open vs closed weights explainer](../concepts/open-vs-closed-weights.md) covers the arguments on both sides.

## At a glance

"Flagship" is the newest top model we could confirm from a primary source as of September 30, 2026. Weights stance describes the lab's language models.

| Lab | Based in | Known for | Flagship (as of Sept 2026) | Weights stance |
|---|---|---|---|---|
| OpenAI | US | ChatGPT; GPT and o-series; first mass-market chatbot | GPT-6 Astra (September 3, 2026) | Closed flagships; open gpt-oss (Apache 2.0, August 2025) |
| Anthropic | US | Claude; safety research (Constitutional AI, interpretability); agentic coding | Claude Fable 5.1 / Mythos 5.1 (September 1, 2026); Opus 5.5 (September 22); Sonnet 5.5 (September 28) | Closed |
| Google DeepMind | UK / US (Google) | Gemini; AlphaGo, AlphaFold; the Transformer came from Google | Gemini 3.8 Flash (September 2026); Gemini 4 in post-training | Closed Gemini; open Gemma (Gemma 4 under Apache 2.0) |
| Meta (Superintelligence Labs) | US | Llama, which started the open-weight wave in 2023 | Muse Spark (served in Meta AI) | Mixed: Llama 4 under Meta's licence; Muse Glimmer 30B under Apache 2.0 (August 2026) |
| xAI (now part of SpaceX) | US | Grok; very large training clusters (Colossus) | Grok series | Mostly closed; older Grok-1 (Apache 2.0) and Grok-2 (xAI community licence) weights released |
| Microsoft AI | US | Largest OpenAI investor; own MAI models since 2025 | MAI model family (voice, image, code, reasoning) | Closed MAI models; open small research models (for example Phi) |
| Amazon | US | AWS Bedrock marketplace; Nova models; major Anthropic investor | Nova 2 generation | Closed |
| Apple | US | On-device Apple Intelligence; privacy-focused server models | Apple Foundation Models, next generation built on Gemini | Closed flagship models; publishes research models |
| DeepSeek | China | Efficiency research (MLA, sparse attention); R1 open reasoning model | DeepSeek-V4.1-Flash (September 10, 2026) | Open, MIT |
| Alibaba (Qwen) | China | Widest open catalogue; most-used fine-tuning base | Qwen3.8 (August 2026); Qwen4 in training | Open for most sizes (Apache 2.0); largest models with service carve-outs; some Max tiers hosted only |
| Moonshot AI | China | Kimi; very large open MoE models | Kimi K3 (July 2026, about 2.8T parameters) | Open, custom licence with service carve-out |
| Z.ai (Zhipu) | China | GLM models; Tsinghua spin-out | GLM-5.3 (August 2026) | Open, custom licence |
| MiniMax | China | Long-context MoE models; audio and music models | MiniMax-M3 (mid-2026) | Open weights, commercial use needs notice or authorisation |
| Mistral AI | France | Efficient open models; "sovereign" European AI | Mistral Medium 3.5 (April 2026); Large 3 is its largest open model | Mixed: Apache 2.0 and revenue-capped modified MIT |

## US frontier labs

**OpenAI.** Founded in December 2015 as a non-profit research lab. In October 2025 it completed a recapitalisation: the non-profit, now the OpenAI Foundation, controls a for-profit public benefit corporation, OpenAI Group PBC. Microsoft's announcement of the new agreement puts its stake at about 27% on an as-converted diluted basis, and says the deal lets OpenAI release open-weight models and use other cloud providers. OpenAI's line runs GPT-1 to GPT-6; see the [GPT family explainer](../model-families/gpt.md).

**Anthropic.** Founded in 2021 by former OpenAI staff including Dario and Daniela Amodei. It is a public benefit corporation whose board is elected partly by a Long-Term Benefit Trust, a body of trustees meant to weigh long-term public benefit. It does not release model weights for its Claude models. Amazon and Google are both major investors. Its model line and tiers (Haiku, Sonnet, Opus, and since 2026 a tier above Opus) are covered in the [Claude family explainer](../model-families/claude.md). Its published safety framework is covered in [frontier safety frameworks](../policy/frontier-safety-frameworks.md).

**Google DeepMind.** Formed on April 20, 2023 by merging Google Brain (from Google Research) with DeepMind, led by Demis Hassabis. It builds Gemini, and continues the scientific line of AlphaGo and AlphaFold (see the [AlphaFold summary](../../papers/techniques/68-alphafold/summary.md)). Google keeps Gemini weights closed but publishes the smaller Gemma models; Gemma 4 (2026) moved to the Apache 2.0 licence from the custom Gemma licence used for Gemma 3. In late September 2026 DeepMind's Koray Kavukcuoglu said Gemini 4 was in post-training. See the [Gemini family explainer](../model-families/gemini.md).

**xAI.** Founded by Elon Musk in 2023. It merged with X Corp in 2025 and was acquired by SpaceX in February 2026. It builds the Grok models and trains on its Colossus clusters in Memphis. It released the weights of Grok-1 (Apache 2.0, March 2024) and later Grok-2 (under an xAI community licence), but not its current models. We could not confirm the latest Grok version and its date from a primary source, so it is not listed here.

## US big tech

**Meta.** Meta's Llama models (2023-2025) made capable open weights normal; see the [Llama family explainer](../model-families/llama.md). In June 2025 it formed Meta Superintelligence Labs. In 2026 that group released Muse Spark, which powers Meta AI, and in August 2026 the Muse Glimmer 30B model, distilled from Muse Spark and released under Apache 2.0. As of September 30, 2026 there are no official Muse Spark weights on Hugging Face.

**Microsoft.** Microsoft is OpenAI's largest outside shareholder and its main cloud partner, and it resells OpenAI, Anthropic and other models through Azure. Its Microsoft AI group also builds its own models under the MAI name (voice, image, transcription, code and reasoning models in 2025-2026), and Microsoft Research publishes small open models such as Phi.

**Amazon.** Amazon's main role is as a platform: AWS Bedrock hosts models from many labs, and Amazon is a large Anthropic investor. It also builds its own closed Nova models; the current generation includes Nova 2 Lite and Nova 2 Sonic.

**Apple.** Apple runs a roughly 3B-parameter model on devices and a larger mixture-of-experts model in its Private Cloud Compute, and lets app developers call the on-device model through its Foundation Models framework. Weights are not released. On January 12, 2026, Apple and Google announced that Apple's next-generation foundation models will be built on Gemini technology, while still running on Apple devices and Private Cloud Compute.

## Chinese labs

The Chinese labs that matter most for this repo share three traits: they release weights, they publish detailed technical reports, and their designs focus on cost per token.

**DeepSeek.** A Hangzhou lab spun out of the High-Flyer hedge fund, which remains its main owner. Known for Multi-head Latent Attention, cheap FP8 training and the R1 reasoning model, all under MIT. See the [DeepSeek explainer](../model-families/deepseek.md) and the [DeepSeek-V2 / MLA summary](../../papers/architectures/141-multi-head-latent-attention/summary.md).

**Alibaba (Qwen).** Qwen is built inside Alibaba Cloud and ships the broadest range of open sizes and modalities of any lab. Its biggest models moved to licences with service carve-outs in 2026. Alibaba is also a leading investor in Moonshot. See the [Qwen explainer](../model-families/qwen.md).

**Moonshot AI.** A Beijing lab founded in 2023, maker of the Kimi assistant and models. Kimi K2 (July 2025) was a 1-trillion-parameter open MoE model under a modified MIT licence. Kimi K3 (July 2026) is about 2.8 trillion total parameters with 104B active, 1M-token context and native image and video input. Its licence requires a separate agreement for "model as a service" businesses above US$20M revenue over 12 months.

**Z.ai (Zhipu AI).** A 2019 Tsinghua University spin-out that rebranded internationally as Z.ai in 2025. The US Commerce Department added it to the Entity List in January 2025. It listed on the Hong Kong Stock Exchange on January 8, 2026. Its GLM-5 line runs through GLM-5.3 (August 2026), released with open weights under a custom licence that asks only very large service providers (over US$10B revenue) to pass a security review.

**MiniMax.** A Chinese lab publishing long-context MoE language models (MiniMax-M series) plus speech and music models. The MiniMax-M3 weights (2026) are free for non-commercial use; commercial users must credit the model and notify MiniMax, and those above US$20M yearly revenue need written authorisation.

## Europe

**Mistral AI.** Founded in Paris in 2023 by alumni of Google DeepMind and Meta. It is the largest European model developer and pitches "sovereign" AI that governments and companies can run under their own control. Its September 2026 Series D raised EUR 3B, led by Samsung, at a post-money valuation above EUR 21B; ASML led its 2025 round. See the [Mistral explainer](../model-families/mistral.md).

## Patterns across labs

- **Mixture of experts everywhere.** Almost every large 2025-2026 model listed here, open or closed where disclosed, is a mixture-of-experts model. See the [MoE summary](../../papers/architectures/37-mixture-of-experts/summary.md).
- **Reasoning is a mode, not a separate model.** In 2024 labs shipped separate reasoning models (o1, R1, QwQ, Magistral). By 2026 most flagships are one model with a thinking switch or effort setting. See [reasoning models](../concepts/reasoning-models.md).
- **Long context as a default.** 1M-token context is now standard at DeepSeek, OpenAI's GPT-6 Astra, Kimi K3 and the hosted Qwen3.8-Max. See [context windows](../concepts/context-windows.md).
- **Licences as competitive tools.** The "model as a service" clauses in Qwen, Kimi and GLM licences leave individuals and most companies free to use the weights while stopping large cloud providers from reselling them without a deal.
- **Compute sets the ceiling.** Training frontier models needs very large GPU or accelerator clusters, which is why most labs are tied to a cloud or hardware partner. See [AI hardware landscape](../compute/ai-hardware-landscape.md) and [cost of training](../compute/cost-of-training.md).

## What to watch

- **Next-generation releases already announced as in progress:** Gemini 4 (in post-training as of late September 2026), Qwen4 (in training as of Apsara, September 2026) and DeepSeek-V4.1-Pro (announced, no date).
- **Meta's open-weight direction.** Muse Glimmer is open under Apache 2.0; whether the larger Muse Spark follows will show whether Meta returns to releasing its best models.
- **Licence drift among open labs.** If DeepSeek, the last big lab on plain MIT, adds service carve-outs, "open weights" will mean something narrower than it did in 2025.
- **Consolidation and capital.** 2025-2026 saw a recapitalisation (OpenAI), a merger into a rocket company (xAI into SpaceX), an IPO (Z.ai) and record European fundraising (Mistral). Expect more structural changes; treat any valuation as a snapshot.
- **Policy.** Export controls, US state and federal rules and the EU AI Act shape which labs can sell where. See [US AI policy](../policy/us-ai-policy.md) and the [EU AI Act](../policy/eu-ai-act.md).

## Read next

- Model families: [GPT](../model-families/gpt.md), [Claude](../model-families/claude.md), [Gemini](../model-families/gemini.md), [Llama](../model-families/llama.md), [DeepSeek](../model-families/deepseek.md), [Qwen](../model-families/qwen.md), [Mistral](../model-families/mistral.md)
- Ecosystem: [open-source stack](open-source-stack.md), [agent protocols](agent-protocols.md)
- History: [timeline of AI](../history/timeline-of-ai.md)
- Essays that shaped how labs think: [The Bitter Lesson](../../papers/essays/111-bitter-lesson/summary.md), [The Scaling Hypothesis](../../papers/essays/112-scaling-hypothesis/summary.md), [Machines of Loving Grace](../../papers/essays/114-machines-of-loving-grace/summary.md), [Situational Awareness](../../papers/essays/113-situational-awareness/summary.md)
- Hands-on: [LLM basics](https://github.com/PatrickWiloak/cloud-data-ai-security-zero-to-hero/blob/main/learn/concepts/llm-basics.md)

## Sources

- OpenAI GPT-6 Astra announcement (September 3, 2026): https://openai.com/index/gpt-6-astra/ and https://community.openai.com/t/introducing-gpt-6-astra-the-most-intelligent-and-aligned-model-in-the-world/1394703
- Microsoft, "The next chapter of the Microsoft-OpenAI partnership" (October 28, 2025; ~27% stake, open-weight and cloud terms): https://blogs.microsoft.com/blog/2025/10/28/the-next-chapter-of-the-microsoft-openai-partnership/
- OpenAI structure and founding (OpenAI Foundation, OpenAI Group PBC): https://en.wikipedia.org/wiki/OpenAI
- gpt-oss model card (Apache 2.0, August 2025): https://huggingface.co/openai/gpt-oss-120b
- Anthropic newsroom (Fable 5.1 and Mythos 5.1 September 1, Opus 5.5 September 22, Sonnet 5.5 September 28, 2026): https://www.anthropic.com/news and https://www.anthropic.com/news/claude-opus-5-5
- Anthropic company page (PBC, Long-Term Benefit Trust): https://www.anthropic.com/company
- Anthropic founding and investors: https://en.wikipedia.org/wiki/Anthropic
- Google DeepMind formation (April 20, 2023): https://blog.google/technology/ai/april-ai-update/
- Gemini API release notes (Gemini 3.8 Flash, September 2026): https://ai.google.dev/gemini-api/docs/changelog
- Gemini 4 in post-training (September 24, 2026): https://9to5google.com/2026/09/24/google-says-gemini-4-release-is-coming-as-soon-as-possible/
- Gemma 4 model card (Apache 2.0): https://huggingface.co/google/gemma-4-31B-it
- Llama 4 announcement (April 5, 2025): https://ai.meta.com/blog/llama-4-multimodal-intelligence/
- Muse Glimmer 30B model card (Meta Superintelligence Labs, Apache 2.0, distilled from Muse Spark): https://huggingface.co/meta-models/Muse-Glimmer-30B
- Meta Superintelligence Labs formation and Muse Spark: https://en.wikipedia.org/wiki/Meta_Superintelligence_Labs
- xAI history (X merger, SpaceX acquisition, Colossus): https://en.wikipedia.org/wiki/XAI_(company)
- Grok-1 and Grok-2 weights and licences: https://huggingface.co/xai-org/grok-1 and https://huggingface.co/xai-org/grok-2
- Microsoft AI news (MAI models): https://microsoft.ai/news/
- Amazon Nova: https://aws.amazon.com/nova/
- Apple Foundation Models 2025 update: https://machinelearning.apple.com/research/apple-foundation-models-2025-updates
- Apple and Google joint statement (January 12, 2026): https://blog.google/company-news/inside-google/company-announcements/joint-statement-google-apple/
- DeepSeek API changelog: https://api-docs.deepseek.com/updates/
- Qwen organisation and licences on Hugging Face: https://huggingface.co/Qwen
- Qwen4 status at Apsara 2026: https://pandaily.com/alibaba-qwen4-training-roadmap-5-10t-apsara-2026
- Kimi K3 model card and LICENSE: https://huggingface.co/moonshotai/Kimi-K3
- Kimi K2 LICENSE (modified MIT): https://huggingface.co/moonshotai/Kimi-K2-Instruct
- Moonshot AI background (founding, Alibaba stake): https://en.wikipedia.org/wiki/Moonshot_AI
- GLM-5.3 model card and LICENSE: https://huggingface.co/zai-org/GLM-5.3
- Z.ai background (Tsinghua origin, rebrand, Entity List, Hong Kong listing): https://en.wikipedia.org/wiki/Zhipu_AI
- MiniMax-M3 LICENSE: https://huggingface.co/MiniMaxAI/MiniMax-M3
- Mistral Series D (September 8, 2026): https://mistral.ai/news/mistral-makes-sovereign-open-weight-ai-to-frontier/
- Mistral Series C (September 9, 2025): https://mistral.ai/news/mistral-ai-raises-1-7-b-to-accelerate-technological-progress-with-ai
