# The Qwen Family (Alibaba)

**In one line:** Qwen is Alibaba's model family and the widest open-weight catalogue in the field - every size from under 1B to trillions of parameters, text, vision, audio, code and images - with the top model usually kept hosted-only and, since mid-2026, the biggest open releases carrying commercial carve-outs, so always check which Qwen and which licence you are looking at.
**Last reviewed:** 2026-09-30

---

## The short version

- **Breadth is the strategy.** Where most labs release one or two open models per generation, Qwen releases a full ladder of sizes plus specialist lines (Coder, VL for vision, Omni for speech, Image, ASR). That makes Qwen the default base model for a large share of community fine-tunes.
- **Two tiers: open weights and hosted "Max/Plus".** Since 2025 the pattern has been open-weight models for most sizes, and a proprietary flagship (Qwen2.5-Max, Qwen3-Max, Qwen3.6-Plus, Qwen3.8-Max) served only through Alibaba Cloud. In August 2026 Alibaba broke that pattern by releasing the Qwen3.8 flagship's weights (2.4T parameters).
- **Licences are per model, not per family.** Most small and mid-size Qwen models are Apache 2.0. The largest 2026 releases use Qwen-specific licences that require a separate agreement for large "model as a service" providers.
- **The architecture has moved away from plain transformers.** Qwen3.5 (February 2026) onward mixes a linear-attention layer type (Gated DeltaNet) with ordinary attention, three to one, for cheaper long context.
- **As of September 2026** the newest models are the Qwen3.8 generation (August 2026) and Qwen3.8-Flash-Next, which Alibaba calls a preview of the Qwen4 architecture. Alibaba said at its Apsara conference (September 2026) that Qwen4 is in training, with no release date.

## Who makes Qwen

Qwen (from Tongyi Qianwen, the Chinese product name) is built by Alibaba's Qwen team inside Alibaba Cloud. The first public beta was in April 2023, and the first technical report on arXiv is dated September 28, 2023. Alibaba is also a major investor in other Chinese labs, notably Moonshot AI (see the [labs landscape](../ecosystem/labs-landscape.md)).

## Release timeline

Dates are first public release. Parameter notation: "235B-A22B" means 235 billion total parameters of which 22 billion are active per token (a mixture-of-experts model). "Summary" links go to the paper summary in this repo where one exists.

| Model | Date | What changed | Licence of open weights | Summary |
|---|---|---|---|---|
| Qwen (1st gen) | August - December 2023 (report September 28, 2023) | 1.8B, 7B, 14B, 72B base and chat models. | Custom Tongyi Qianwen licence | - |
| Qwen2 | June 2024 | 0.5B to 72B, including a 57B-A14B mixture-of-experts. | Apache 2.0 for most sizes; 72B under the Qwen licence | - |
| Qwen2.5 | September 2024 | 0.5B to 72B, plus Coder and Math lines. Became a common fine-tuning base. | Apache 2.0 for most; 3B research-only; 72B Qwen licence | - |
| QwQ-32B-Preview | November 2024 | Qwen's first open reasoning ("thinking") model. | Apache 2.0 | - |
| Qwen2.5-VL / Qwen2.5-Max | January 2025 | VL: vision-language family. Max: large MoE flagship, hosted only. | VL open; Max closed | - |
| Qwen3 | April 2025 | Dense 0.6B to 32B plus 30B-A3B and 235B-A22B MoE. One model switches between thinking and non-thinking modes; 36T training tokens; 119 languages. | Apache 2.0 | [28](../../papers/language-models/28-qwen3/summary.md) |
| Qwen3-Coder | July 2025 | Agentic coding specialist line. | Apache 2.0 | - |
| Qwen3-Max | September 2025 (Apsara, September 24) | Over 1 trillion parameters; Alibaba's largest model at the time. | Hosted only | - |
| Qwen3-Next, Qwen3-Omni, Qwen3-VL | September 2025 | Next: 80B-A3B testbed for hybrid linear attention. Omni: speech in and out. | Next: Apache 2.0; Omni: custom | - |
| Qwen3-Coder-Next | February 2026 | Coding line on the Next architecture. | Apache 2.0 | - |
| Qwen3.5 | February 2026 | Flagship 397B-A17B plus a ladder down to 0.8B. Natively multimodal (images trained in from the start), 262K context extendable to about 1M, 201 languages. Hybrid Gated DeltaNet + attention. | Apache 2.0 | - |
| Qwen3.6 | April 2026 | Open 35B-A3B and 27B; a larger Qwen3.6-Plus served via API. | Apache 2.0 (open sizes) | - |
| Qwen3.8 (Max / 2.4T-A95B / 27B) | August 2026 | Qwen3.8-Max on Alibaba Cloud first (early August), then the flagship's weights as Qwen3.8-2.4T-A95B (August 12) and a dense 27B (August 14). The open 2.4T model is text-only and thinking-only; the hosted Max adds vision and non-thinking mode. | 2.4T: "Qwen3.8-Max License" (see below); 27B: Apache 2.0 | - |
| Qwen3.8-Flash-Next | Late August 2026 | 125B total / 6B active; image, video and text input; new attention and residual designs. Described as the architecture that will underpin Qwen4. | Qwen Community License 1.0 | - |
| Qwen-Image-2.1 | September 2026 | 7B image generation and editing model. | Custom, non-Apache | - |

## The through-line: a ladder, then a cheaper rung

Two ideas explain most Qwen decisions.

**1. Ship the whole ladder.** Every generation is released at many sizes so that the same family runs on a phone, a laptop, a single GPU and a cluster. Small models are often distilled from large ones (see [knowledge distillation](../../papers/techniques/134-knowledge-distillation/summary.md)). For a builder this means you can prototype on a small Qwen and scale up without changing tokenizer, chat format or behaviour much.

```
           hosted only          open weights
          +-----------+   +---------------------------------------------+
Qwen3.8   |  3.8-Max  |   | 2.4T-A95B | ... | 27B dense |  Flash-Next   |
          +-----------+   +---------------------------------------------+
Qwen3.5/6 | 3.6-Plus  |   | 397B-A17B | 122B-A10B | 35B-A3B | 27B | 9B | 4B | 2B | 0.8B
          +-----------+   +---------------------------------------------+
Qwen3     |   3-Max   |   | 235B-A22B | 32B | 30B-A3B | 14B | 8B | 4B | 1.7B | 0.6B
          +-----------+   +---------------------------------------------+
```

**2. Make each rung cheaper to run.** The architecture has moved step by step toward lower cost per token at long context:

- **Mixture of experts** from Qwen2 onward: most parameters sit idle for any given token. See the [MoE summary](../../papers/architectures/37-mixture-of-experts/summary.md).
- **Hybrid thinking** (Qwen3): one model can answer quickly or reason at length, with a budget the caller controls. The [Qwen3 summary](../../papers/language-models/28-qwen3/summary.md) covers the thinking budget; the [reasoning models explainer](../concepts/reasoning-models.md) gives the wider picture.
- **Hybrid linear attention** (Qwen3-Next, then Qwen3.5 and Qwen3.8): three out of every four layers use Gated DeltaNet, a linear-attention layer whose memory does not grow with context length the way a KV cache does; every fourth layer uses ordinary (gated) attention to keep precise recall. The Qwen3.5 flagship's layout is 15 repeats of that 3-plus-1 block. Why this matters is covered in the [KV cache](../concepts/kv-cache.md) and [context windows](../concepts/context-windows.md) explainers.
- **Qwen4 preview** (Qwen3.8-Flash-Next): block-level sparse attention, a gated residual path, a large "n-gram embedding" table, and training with the Muon optimizer for some weights (see the [Muon summary](../../papers/techniques/144-muon/summary.md)).

## Licensing

This is where Qwen needs the most care, because terms vary within a single generation.

| Licence | Used for | Plain meaning |
|---|---|---|
| Apache 2.0 | Most Qwen2 to Qwen3.6 sizes, Qwen3.8-27B | Use, modify and sell freely; keep notices. Includes a patent grant. |
| Qwen licence / Qwen research licence | Some 2024 models (for example Qwen2.5-72B, Qwen2.5-3B) | Custom terms; the research licence bars commercial use. |
| Qwen3.8-Max License | Qwen3.8-2.4T-A95B (August 2026) | Permissive, but: products above 100M monthly users or US$20M monthly revenue must display the model name, and a company running a "Model as a Service" or AI coding/office assistant business with over US$50M revenue in 12 months needs a separate licence from Qwen. Internal use is exempt. |
| Qwen Community License 1.0 | Qwen3.8-Flash-Next | Same structure, but any "Model as a Service" or AI work-assistant business needs a separate licence for commercial use, with no revenue threshold. |

In short: the small and medium Qwen models are about as open as models get, while the largest and newest ones are free for most users but not for companies that would resell them as a hosted API or a competing coding assistant. Moonshot AI (Kimi K3) and Z.ai (GLM-5.3) adopted similar "model as a service" clauses in 2026. See [open vs closed weights](../concepts/open-vs-closed-weights.md) for why labs write licences this way.

## Reading the names

- **Qwen + number** is the generation. Half-steps (2.5, 3.5) and odd point releases (3.6, 3.7, 3.8) are frequent; a point release can change architecture, not just training.
- **Size suffix:** "32B" is a dense model; "235B-A22B" is MoE with total and active parameters. "2.4T" means 2.4 trillion.
- **Max / Plus / Flash** are service tiers on Alibaba Cloud. Max is the flagship; Plus and Flash are cheaper. A "Max" model is usually hosted only, except for the Qwen3.8 flagship weights noted above.
- **Next** marks an architecture preview (Qwen3-Next, Qwen3.8-Flash-Next) that tests ideas before the next generation.
- **Specialist suffixes:** Coder (programming), VL (vision-language), Omni (speech and vision), Math, ASR (speech recognition), Image (image generation), QwQ (early reasoning models).
- **Instruct / Thinking / Base:** Base is the raw pretrained model; Instruct is tuned to follow instructions; Thinking variants reason before answering.

## What to watch

- **Qwen4.** Announced as in training at Apsara 2026 (September), with no date, size or licence published. Qwen3.8-Flash-Next is the best guide to its architecture. Attendee reports mention Max, Plus, Flash and 27B tiers, but Alibaba's own materials did not list them; treat tier names as unconfirmed until release.
- **Licence direction.** The 2026 flagship releases added commercial carve-outs. Whether Qwen4's open weights keep Apache 2.0 for small sizes and carve-outs for large ones is the key question for anyone building on Qwen.
- **Open-weight scale race.** Qwen3.8-2.4T-A95B and Moonshot's Kimi K3 (2.8T) pushed open weights into the multi-trillion range in mid-2026. Few organisations can run models that size; the practical impact comes through distilled smaller models.
- **Independent evaluation.** Qwen's reported benchmark numbers are self-reported; check community leaderboards and the [contamination and saturation explainer](../benchmarks/contamination-and-saturation.md) before relying on them.

## Read next

- [Qwen3 Technical Report](../../papers/language-models/28-qwen3/summary.md)
- [Mixture of Experts](../../papers/architectures/37-mixture-of-experts/summary.md), [Grouped-Query Attention](../../papers/architectures/75-grouped-query-attention/summary.md), [YaRN context extension](../../papers/techniques/130-yarn-context-extension/summary.md)
- [DeepSeek-R1](../../papers/language-models/26-deepseek-r1/summary.md) (its distilled models were partly built on Qwen bases)
- Explainers: [DeepSeek](deepseek.md), [Mistral](mistral.md), [Llama](llama.md), [labs landscape](../ecosystem/labs-landscape.md), [open-source stack](../ecosystem/open-source-stack.md)
- Hands-on: [Fine-tuning vs RAG](https://github.com/PatrickWiloak/cloud-data-ai-security-zero-to-hero/blob/main/learn/concepts/fine-tuning-vs-rag.md), [Quantization and distillation](https://github.com/PatrickWiloak/cloud-data-ai-security-zero-to-hero/blob/main/learn/concepts/quantization-and-distillation.md)

## Sources

- Qwen Technical Report, arXiv 2309.16609 (September 28, 2023): https://arxiv.org/abs/2309.16609
- Qwen3 Technical Report, arXiv 2505.09388: https://arxiv.org/abs/2505.09388
- Qwen organisation on Hugging Face (model list, creation dates, licence tags): https://huggingface.co/Qwen
- Qwen3.5-397B-A17B model card (architecture, 201 languages, context, Apache 2.0): https://huggingface.co/Qwen/Qwen3.5-397B-A17B
- Qwen3.6-35B-A3B and Qwen3.6-27B model cards (Apache 2.0): https://huggingface.co/Qwen/Qwen3.6-35B-A3B
- Qwen3.8-2.4T-A95B model card and LICENSE (Qwen3.8-Max License): https://huggingface.co/Qwen/Qwen3.8-2.4T-A95B
- Qwen3.8-27B model card (Apache 2.0): https://huggingface.co/Qwen/Qwen3.8-27B
- Qwen3.8-Flash-Next model card and LICENSE (Qwen Community License 1.0, Qwen4 architecture statement): https://huggingface.co/Qwen/Qwen3.8-Flash-Next
- Qwen-Image-2.1 model card: https://huggingface.co/Qwen/Qwen-Image-2.1
- Qwen2.5-72B-Instruct and Qwen2.5-3B-Instruct licence files: https://huggingface.co/Qwen/Qwen2.5-72B-Instruct
- Qwen3-Max launch at Apsara 2025 (closed weights, over 1T parameters): https://cybernews.com/ai-news/alibaba-released-trillion-parameter-model-qwen3-max/
- Qwen4 in training, Apsara 2026: https://pandaily.com/alibaba-qwen4-training-roadmap-5-10t-apsara-2026 and https://cellcog.ai/blog/qwen-4-release-date/
- Release history cross-check (Tongyi Qianwen beta, Qwen2/2.5 licences, Qwen3.6-Plus hosted-only, Qwen3.8 dates): https://en.wikipedia.org/wiki/Qwen
