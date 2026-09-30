# The Llama Family (Meta), and What Came After

**In one line:** Llama made "download a strong model and run it yourself" normal between 2023 and 2025, but its last release was Llama 4 (April 2025); in 2026 Meta moved its frontier work to a closed Muse line and returned to open weights only with the smaller Muse Glimmer - so "Meta = open" is no longer a safe assumption.
**Last reviewed:** 2026-09-30

---

## The short version

- **Llama was the open-weight default.** From LLaMA (February 2023) through Llama 3.3 (December 2024), each release gave developers weights they could download, fine-tune and run on their own hardware, and most of the open-model ecosystem was built on them.
- **"Open weights", not "open source".** The weights are downloadable, but the licence restricts use: companies above 700 million monthly active users need Meta's permission, derived models must carry the "Llama" name, and the multimodal Llama models exclude EU-based companies and individuals from the licence grant. The training data was never released.
- **The design was deliberately conservative until Llama 4.** Llama 1 to 3 were dense decoder-only transformers that won by training small models on far more data than compute-optimal rules suggested. Llama 4 (April 2025) switched to mixture-of-experts and native multimodality.
- **Llama 4 was the last Llama.** Its largest model, Behemoth, was previewed but never released. Meta formed Meta Superintelligence Labs in 2025, and in April 2026 launched **Muse Spark**, a closed model, as the new engine of Meta AI.
- **Open weights returned in a different form.** In August 2026 Meta released **Muse Glimmer**, a 30B model under Apache 2.0, and said an open-weight version of Muse Spark 1.2 was coming. As of September 30, 2026 that Spark release has not been published on Meta's channels.

## Release timeline

"Summary" links go to paper summaries in this repo where one exists.

### Llama

| Model | Date | What changed | Summary |
|---|---|---|---|
| LLaMA | February 24, 2023 | 7B to 65B, trained only on public data; 13B beat GPT-3 on most benchmarks. Research-only licence; weights leaked publicly within about two weeks. | [15](../../papers/language-models/15-llama/summary.md) |
| Llama 2 (and Llama 2-Chat) | July 18, 2023 | 7B, 13B, 70B; commercial use allowed; RLHF-tuned chat models; 4K context; grouped-query attention on 70B. | [17](../../papers/language-models/17-llama2/summary.md) |
| Code Llama | August 24, 2023 | Llama 2 further trained on code. | - |
| Llama Guard | December 2023 | A Llama model fine-tuned as a safety classifier for prompts and responses. | [96](../../papers/language-models/96-llama-guard/summary.md) |
| Llama 3 | April 18, 2024 | 8B and 70B; about 15T training tokens; larger tokenizer; 8K context. | - |
| Llama 3.1 | July 23, 2024 | 405B, the first open-weight model in the frontier class; 128K context. | - |
| Llama 3.2 | September 25, 2024 | 1B and 3B for devices; 11B and 90B with image input (the first multimodal Llamas). | - |
| Llama 3.3 | December 2024 | 70B that approaches 3.1 405B quality, via distillation and better post-training. | [33](../../papers/language-models/33-llama3.3/summary.md) |
| Llama 4 Scout / Maverick | April 5, 2025 | Mixture-of-experts; natively multimodal (early fusion); Scout 17B active of 109B total with a 10M-token context claim; Maverick 17B active of 400B total. | [41](../../papers/language-models/41-llama4/summary.md) |
| Llama 4 Behemoth | Previewed April 5, 2025 | About 288B active and roughly 2T total parameters, used as a teacher for Scout and Maverick. Never released. | [41](../../papers/language-models/41-llama4/summary.md) |

### After Llama: the Muse line

| Model | Date | What changed | Weights |
|---|---|---|---|
| Muse Spark | April 8, 2026 | First Meta Superintelligence Labs model; multimodal reasoning with tool use; Meta says it reaches Llama 4 Maverick's capabilities with over an order of magnitude less compute. Powers Meta AI. | Closed |
| Muse Spark 1.1 | July 9, 2026 | Agentic tasks, computer use, coding; 1M-token context; Meta Model API in public preview. | Closed |
| Muse Glimmer | August 10, 2026 | 30B dense multimodal model (2B vision encoder plus 28B decoder), 32K context, built for local agents on a single 24 to 32 GB GPU. | Open, Apache 2.0 |
| Muse Spark 1.2 (open version) | Announced August 10, 2026 | Alexandr Wang, Meta's Chief AI Officer, said open weights for "a version of muse spark 1.2" were "coming soon". | Pending |

## The through-line

### 1. Small models, lots of data

The founding idea of LLaMA was to spend compute on data, not parameters. [Chinchilla](../../papers/techniques/18-chinchilla/summary.md) had worked out the compute-optimal ratio of model size to training tokens for a fixed training budget. LLaMA deliberately trained past that point, because a model that is used a lot is cheaper to serve when it is small:

```
Chinchilla question:  best model for a fixed TRAINING budget?
LLaMA question:       best model for a fixed INFERENCE budget?
                      -> smaller model, many more tokens
LLaMA 1:  up to 1.4T tokens    Llama 2:  2T    Llama 3:  ~15T    Llama 4:  30T+
```

This is why a 7B or 8B Llama could run on a laptop and still be useful, and why "Llama-sized" became shorthand for what open models could do.

### 2. Standard parts, carefully assembled

Llama 1 to 3 introduced little new architecture. They combined proven parts: [RoPE](../../papers/techniques/54-rope-rotary-position-embedding/summary.md) position embeddings, RMSNorm, SwiGLU activations, and from Llama 2 70B onward [grouped-query attention](../../papers/architectures/75-grouped-query-attention/summary.md) to shrink the KV cache (see the [KV cache explainer](../concepts/kv-cache.md)). Post-training moved from RLHF in Llama 2 toward [DPO](../../papers/language-models/19-dpo/summary.md)-style preference optimisation and teacher-generated data in Llama 3.x. Llama 3.3 is the clearest case: a 70B model trained with help from the 405B model as a teacher, a form of [knowledge distillation](../../papers/techniques/134-knowledge-distillation/summary.md).

### 3. Llama 4: a change of direction

Llama 4 broke with the conservative recipe in three ways: [mixture-of-experts](../../papers/architectures/37-mixture-of-experts/summary.md) (only a fraction of parameters runs per token), early-fusion multimodality (image and text tokens in one model from the start of training), and a very long context claim for Scout (10M tokens, from 256K in pre- and post-training). The launch was also controversial: LMArena said the Maverick version Meta submitted to its leaderboard was an experimental chat-tuned variant that differed from the public release. See [human preference arenas](../benchmarks/human-preference-arenas.md) and [contamination and saturation](../benchmarks/contamination-and-saturation.md) for why that matters.

### 4. Safety as a toolkit, not only a model

Because anyone could fine-tune the weights, Meta could not rely on the model alone to behave. It released separate safety components: [Llama Guard](../../papers/language-models/96-llama-guard/summary.md) (an input and output classifier), and an acceptable use policy attached to the licence. This "ship the guardrails alongside the model" approach is now common in open-weight deployments; see the sibling repo's [guardrails and safety](https://github.com/PatrickWiloak/cloud-data-ai-security-zero-to-hero/blob/main/learn/concepts/guardrails-and-safety.md) page.

### 5. The 2026 pivot

With Muse Spark, Meta put its frontier model behind its own apps and an API, like OpenAI, Anthropic and Google. Meta said at the time that its current Llama models would remain available. Four months later it returned to open weights with Muse Glimmer, a model sized for a single consumer GPU under a standard Apache 2.0 licence rather than a custom Llama licence. The resulting shape resembles Google's: a closed frontier line plus a smaller open line (compare [Gemini and Gemma](gemini.md)).

## Open vs closed

| Release | Weights | Licence and key conditions |
|---|---|---|
| LLaMA (1) | Released to approved researchers; leaked | Non-commercial research licence |
| Llama 2, 3, 3.1, 3.2, 3.3 | Open weights | Llama Community Licence: permission needed above 700M monthly active users; "Built with Llama" attribution; acceptable use policy. Llama 3.2 vision models carry the same EU restriction as the Llama 4 multimodal models (next row). |
| Llama 4 Scout, Maverick | Open weights | Llama 4 Community Licence: same 700M MAU clause; derived models must start their name with "Llama"; the acceptable use policy withholds the licence grant for the multimodal models from individuals and companies based in the EU (end users of products built on them are not affected). |
| Muse Spark, Spark 1.1 | Closed | Meta AI app and website; Meta Model API |
| Muse Glimmer | Open weights | Apache 2.0 |

The Open Source Initiative does not consider the Llama licences open source, because of the use restrictions and because training data is not released. The more precise term is "open weights". This distinction is the subject of [open vs closed weights](../concepts/open-vs-closed-weights.md).

## How to read the naming

- **Llama generations.** "LLaMA" (with that capitalisation) was the 2023 original. From Llama 2 it is written "Llama". Point releases (3.1, 3.2, 3.3) added sizes or modalities to the same generation rather than retraining everything.
- **Sizes in the name.** Up to Llama 3.3, the suffix is the parameter count (Llama-3.1-8B, -70B, -405B). "Instruct" or "Chat" marks the instruction-tuned version; without it you have the base model.
- **Llama 4 animal names.** Scout, Maverick and Behemoth are sizes. The official model IDs spell out active parameters and expert count, for example `Llama-4-Maverick-17B-128E-Instruct` (17B active, 128 experts).
- **Muse.** Muse is the new family. "Spark" is the closed frontier line with point versions (1.1, 1.2); "Glimmer" is the small open model; "Muse Image" is image generation; "Muse" is also the name of Meta's consumer AI agent. Expect this scheme to settle as more models ship.

## Why Llama mattered even if it stops

- It showed that a strong model could be released openly without the predicted immediate harms, which shifted the policy debate (see [US AI policy](../policy/us-ai-policy.md)).
- It created the tooling ecosystem: llama.cpp, Ollama, vLLM, LoRA fine-tunes and quantised builds grew up around Llama weights. See [LoRA](../../papers/techniques/10-lora/summary.md), [QLoRA](../../papers/techniques/22-qlora/summary.md), [GPTQ and AWQ quantization](../../papers/techniques/86-gptq-awq-quantization/summary.md), [vLLM](../../papers/techniques/52-pagedattention-vllm/summary.md), and the [open-source stack explainer](../ecosystem/open-source-stack.md).
- Its gap is now being filled largely by Chinese labs; see [DeepSeek](deepseek.md) and [Qwen](qwen.md), and by [Mistral](mistral.md) and Google's Gemma.

## What to watch

- **Whether the open Muse Spark 1.2 weights ship.** This is the single clearest test of Meta's renewed open-weights commitment. As of September 30, 2026 they have been announced but not released.
- **Licence choice.** Muse Glimmer uses Apache 2.0, with none of the Llama licence's MAU cap or EU carve-out. Whether larger open Muse models keep that licence or return to Llama-style terms will decide how widely they are adopted.
- **The fate of existing Llama models.** Meta says they remain available. Watch for deprecations on cloud platforms and whether Meta keeps publishing Llama safety tools.
- **Meta's compute efficiency claim.** Meta says Muse Spark matches Llama 4 Maverick's capabilities with over an order of magnitude less compute. Independent replication or a technical report would make this checkable; see [cost of training](../compute/cost-of-training.md).

## Read next

- Paper summaries: [LLaMA](../../papers/language-models/15-llama/summary.md), [Llama 2](../../papers/language-models/17-llama2/summary.md), [Llama 3.3](../../papers/language-models/33-llama3.3/summary.md), [Llama 4](../../papers/language-models/41-llama4/summary.md), [Llama Guard](../../papers/language-models/96-llama-guard/summary.md), [Chinchilla](../../papers/techniques/18-chinchilla/summary.md), [Grouped-query attention](../../papers/architectures/75-grouped-query-attention/summary.md), [RoPE](../../papers/techniques/54-rope-rotary-position-embedding/summary.md), [YaRN](../../papers/techniques/130-yarn-context-extension/summary.md)
- Other families: [GPT and o-series](gpt.md), [Claude](claude.md), [Gemini](gemini.md), [DeepSeek](deepseek.md), [Qwen](qwen.md), [Mistral](mistral.md)
- Explainers: [Open vs closed weights](../concepts/open-vs-closed-weights.md), [Open-source stack](../ecosystem/open-source-stack.md), [Labs landscape](../ecosystem/labs-landscape.md)
- Hands-on: [Fine-tuning vs RAG](https://github.com/PatrickWiloak/cloud-data-ai-security-zero-to-hero/blob/main/learn/concepts/fine-tuning-vs-rag.md), [Quantization and distillation](https://github.com/PatrickWiloak/cloud-data-ai-security-zero-to-hero/blob/main/learn/concepts/quantization-and-distillation.md), [Inference servers](https://github.com/PatrickWiloak/cloud-data-ai-security-zero-to-hero/blob/main/learn/concepts/inference-servers.md)

## Sources

- Meta, "The Llama 4 herd" (April 5, 2025; parameters, experts, context, 30T+ tokens, Behemoth still training): https://ai.meta.com/blog/llama-4-multimodal-intelligence/
- Meta, Llama 4 Community License Agreement (700M MAU clause, "Built with Llama", "Llama" name prefix): https://dev.meta.ai/llama/llama4/license/
- Meta, Llama 4 Acceptable Use Policy (EU restriction for multimodal models): https://dev.meta.ai/llama/llama4/use-policy/
- Meta, "Introducing Llama 3.1": https://ai.meta.com/blog/meta-llama-3-1/
- Meta, "Llama 3.2: Revolutionizing edge AI and vision with open, customizable models": https://ai.meta.com/blog/llama-3-2-connect-2024-vision-edge-mobile-devices/
- Meta, "Introducing Meta Llama 3": https://ai.meta.com/blog/meta-llama-3/
- Meta, "Introducing Code Llama": https://ai.meta.com/blog/code-llama-large-language-model-coding/
- Touvron et al., "LLaMA: Open and Efficient Foundation Language Models": https://arxiv.org/abs/2302.13971
- Touvron et al., "Llama 2: Open Foundation and Fine-Tuned Chat Models": https://arxiv.org/abs/2307.09288
- Meta, "Introducing Muse Spark: Scaling Towards Personal Superintelligence" (April 8, 2026): https://ai.meta.com/blog/introducing-muse-spark-msl/ and https://about.fb.com/news/2026/04/introducing-muse-spark-meta-superintelligence-labs/
- Meta, "Introducing Muse Spark 1.1" (July 9, 2026; 1M context, Meta Model API public preview): https://ai.meta.com/blog/introducing-muse-spark-meta-model-api/
- Hugging Face, "Meta is back with Muse Glimmer" (August 10, 2026; 30B dense, Apache 2.0, 32K context): https://huggingface.co/blog/muse-glimmer and model card https://huggingface.co/meta-models/Muse-Glimmer-30B
- Alexandr Wang on X (August 10, 2026), open weights for Muse Glimmer and "a version of muse spark 1.2 coming soon": https://x.com/alexandr_wang/status/2086755368596902004
- VentureBeat, "Goodbye, Llama? Meta launches new proprietary AI model Muse Spark" (Meta spokesperson: current Llama models stay available): https://venturebeat.com/technology/goodbye-llama-meta-launches-new-proprietary-ai-model-muse-spark-first-since
- Wikipedia, "Llama (language model)" (cross-check of release dates, OSI position, LMArena controversy): https://en.wikipedia.org/wiki/Llama_(language_model)
