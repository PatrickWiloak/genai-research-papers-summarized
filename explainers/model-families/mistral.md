# The Mistral Family (Mistral AI)

**In one line:** Mistral is the Paris lab that made small, efficient open-weight models its identity, wandered toward closed and restricted licences in 2024, came back to Apache 2.0 in 2025, and in 2026 sells "sovereign" European AI built on a mix of fully open and revenue-capped models - so check the licence of the exact model before you build on it.
**Last reviewed:** 2026-09-30

---

## The short version

- **Known for efficiency at small sizes.** Mistral 7B (September 2023) beat larger Llama 2 models using grouped-query and sliding-window attention; Mixtral 8x7B (December 2023) brought mixture-of-experts to open weights.
- **The licence story has three phases.** Apache 2.0 (2023), then a mix of API-only and research-only licences (2024), then a public return to Apache 2.0 for general-purpose models (January 2025). In 2025-2026 some large models use a "modified MIT" licence that is free only for companies under US$20M monthly revenue.
- **2026 is about merging lines.** Mistral used to ship separate models for chat, reasoning (Magistral), code (Devstral) and vision (Pixtral). Mistral Small 4 (March 2026) and Mistral Medium 3.5 (April 2026) fold those into single models with a reasoning switch.
- **As of September 2026** the largest open model is Mistral Large 3 (675B total / 41B active, Apache 2.0, December 2025), and the newest general flagship is Mistral Medium 3.5 (128B dense, April 2026).
- **Positioning: European and "sovereign".** Mistral's pitch is models and infrastructure that European governments and companies can run under their own control. Its September 2026 funding round (EUR 3B, led by Samsung) was framed around that mission.

## Who Mistral is

Mistral AI was founded in Paris in April 2023 by Arthur Mensch (previously at Google DeepMind), Guillaume Lample and Timothée Lacroix (both previously at Meta, where Lample worked on the original [Llama](../../papers/language-models/15-llama/summary.md)). It sells API access, the Le Chat assistant, the Vibe coding agent, and on-premises deployments, alongside its open-weight releases.

Funding facts from Mistral's own announcements:

| Date | Round | Detail |
|---|---|---|
| September 9, 2025 | Series C | EUR 1.7B, led by ASML, at EUR 11.7B post-money. |
| September 8, 2026 | Series D | EUR 3B, led by Samsung Electronics with co-leads Scaleup Europe Fund (EQT) and PSG Equity, at over EUR 21B post-money. Mistral calls it the largest equity round by a European tech company. |

## Release timeline

Dates are first public release, from Mistral's news posts, changelog and model cards. Only the main language models are listed; Mistral also ships OCR, speech (Voxtral), moderation and specialist models. "Summary" links go to the paper summary in this repo where one exists.

| Model | Date | What changed | Weights and licence | Summary |
|---|---|---|---|---|
| Mistral 7B | September 27, 2023 | 7.3B parameters; outperformed Llama 2 13B. Grouped-query attention plus sliding-window attention. | Apache 2.0 | [95](../../papers/language-models/95-mistral-7b/summary.md) |
| Mixtral 8x7B | December 11, 2023 | Sparse mixture of experts: 46.7B total, 12.9B active per token; router picks 2 of 8 experts. | Apache 2.0 | [37](../../papers/architectures/37-mixture-of-experts/summary.md) |
| Mistral Large | February 26, 2024 | First flagship; launched with Microsoft Azure as distribution partner. | API only | - |
| Codestral | May 29, 2024 | 22B code model, 80+ languages, fill-in-the-middle. | Mistral Non-Production License (no commercial use without a separate licence) | - |
| Mistral Large 2 | July 24, 2024 | 123B dense, 128K context, designed for single-node inference. | Mistral Research License (commercial self-hosting needs a paid licence) | - |
| Mistral Small 3 | January 30, 2025 | 24B; announcement commits to Apache 2.0 for general-purpose models, "progressively moving away" from research-licensed ones. | Apache 2.0 | - |
| Mistral Medium 3 | May 7, 2025 | Enterprise model; API and self-hosted deployments on four or more GPUs. | Not released as open weights | - |
| Magistral Small / Medium | June 10, 2025 | First Mistral reasoning models. | Small (24B): Apache 2.0; Medium: API only | - |
| Magistral 1.2 | September 17, 2025 | Reasoning update. | Small: Apache 2.0 | - |
| Mistral 3 (Large 3, Ministral 3) | December 2, 2025 | Large 3: 675B total / 41B active MoE with vision, 256K context, trained on 3,000 NVIDIA H200s. Ministral 3: 3B, 8B, 14B in base, instruct and reasoning versions. | Apache 2.0 | - |
| Devstral 2 / Devstral Small 2 | December 9, 2025 | Coding models with the Vibe CLI agent. | Devstral 2 (123B): modified MIT; Small 2 (24B): Apache 2.0 | - |
| Mistral Small 4 | March 16, 2026 | 119B total / 6.5B active MoE (128 experts, 4 active); images in; 256K context; one model covering instruct, reasoning (ex-Magistral) and coding (ex-Devstral), with a `reasoning_effort` switch. | Apache 2.0 | - |
| Mistral Medium 3.5 | April 28, 2026 | 128B dense, multimodal, 256K context; Mistral's "first flagship merged model", replacing Medium 3.1 and Devstral 2 and powering Le Chat and Vibe. | Modified MIT (see below) | - |
| Leanstral 1.5 | June 30 / July 2, 2026 | 119B-A6B model for Lean 4 formal proofs. | Apache 2.0 | - |
| Shieldstral 1.0 | August 4, 2026 | 3B safety and moderation model. | Apache 2.0 | - |

## The through-line: do more per active parameter

Mistral's technical identity is **efficiency per active parameter**: how much capability you get for the compute each token actually uses.

```
2023  Mistral 7B     small dense model, cheap attention tricks
        |            (grouped-query attention, sliding-window attention)
        v
2023  Mixtral 8x7B   mixture of experts: big total, small active
        |            46.7B stored, 12.9B used per token
        v
2025  Large 3        same idea at frontier scale
        |            675B stored, 41B used per token
        v
2026  Small 4        one MoE model for chat + reasoning + code
                     119B stored, 6.5B used per token
```

- **Grouped-query attention (GQA)** lets several attention heads share one set of keys and values, shrinking the KV cache. See the [GQA summary](../../papers/architectures/75-grouped-query-attention/summary.md) and the [KV cache explainer](../concepts/kv-cache.md).
- **Sliding-window attention** limits each token to looking back a fixed distance, so memory stays bounded at long lengths; information still travels further through stacked layers. See the [Longformer summary](../../papers/architectures/131-longformer/summary.md) for the general idea.
- **Mixture of experts** stores many expert sub-networks and routes each token to a few. The [Mixtral summary](../../papers/architectures/37-mixture-of-experts/summary.md) explains routing and why the active count, not the total, drives speed.

The 2026 counter-example is Medium 3.5, a 128B **dense** model: every parameter is used for every token. Mistral chose a dense design for the model it self-hosts for coding agents, which is a reminder that MoE is a trade-off (more memory, less compute per token), not a free win.

## Licensing

Mistral's licences have changed more than most labs', so it helps to group them.

| Licence | Examples | Plain meaning |
|---|---|---|
| Apache 2.0 | Mistral 7B, Mixtral, Small 3, Magistral Small, Large 3, Ministral 3, Small 4, Leanstral, Shieldstral | Use, modify and sell freely; keep notices; includes a patent grant. |
| Mistral Research License (2024) | Mistral Large 2 | Research and non-commercial use; commercial self-hosting needs a paid licence. Mistral said in January 2025 it was moving away from this for general-purpose models. |
| Mistral Non-Production License (2024) | Codestral (first version) | Testing and research only. |
| Modified MIT (2025-2026) | Devstral 2, Medium 3.5 | MIT terms, except that you may not use the model at all if your company's (or your employer's) global monthly revenue exceeded US$20M in the previous month. Larger companies must buy a commercial licence or use Mistral's hosted service. |
| Closed | Mistral Large (2024), Medium 3, Magistral Medium | API or managed deployment only. |

The modified MIT licence is worth reading carefully: it is permissive for individuals, startups and researchers, but it is not "open source" for large enterprises, which is exactly the customer group Mistral sells to. For the wider debate on what "open" should mean, see [open vs closed weights](../concepts/open-vs-closed-weights.md).

## Reading the names

- **Size words, not numbers:** Ministral (smallest, edge devices), Small, Medium, Large. "Small" is relative: Small 4 has 119B total parameters.
- **Version numbers apply per size line.** Small 4 and Medium 3.5 are not the same generation; each line is numbered separately.
- **Four-digit suffixes are dates, year then month:** `2512` is December 2025, `2603` is March 2026. `mistral-large-2512` is Large 3.
- **"-stral" names are specialists:** Codestral and Devstral (code), Magistral (reasoning), Pixtral (vision), Voxtral (speech), Leanstral (formal proofs), Shieldstral (safety). Several of these have now been folded back into the general models.
- **"Mixtral"** was the MoE line; since Large 3 and Small 4 are MoE themselves, the separate name has fallen out of use.

## What to watch

- **The next open frontier model.** In July 2026 press coverage reported an open-weight model in early access with partners and a broader release expected later in the year. As of September 30, 2026 there is no official release on Mistral's news page or Hugging Face; treat it as unconfirmed.
- **Which licence the next large model uses.** Large 3 (Apache 2.0) and Medium 3.5 (revenue-capped) point in different directions. The Series D announcement restates an open-weight mission; the next flagship's licence will show what that means in practice.
- **Compute.** Mistral has been building its own European data centre capacity, and its funding announcements are tied to that. Whether it can train at the scale of US and Chinese labs is the open question.
- **Merged models.** If Small 4 and Medium 3.5 hold up, expect fewer specialist "-stral" releases and more single models with a reasoning setting.

## Read next

- [Mistral 7B](../../papers/language-models/95-mistral-7b/summary.md)
- [Mixtral of Experts](../../papers/architectures/37-mixture-of-experts/summary.md)
- [Grouped-Query Attention](../../papers/architectures/75-grouped-query-attention/summary.md), [Longformer / sliding-window attention](../../papers/architectures/131-longformer/summary.md)
- Explainers: [Llama](llama.md), [DeepSeek](deepseek.md), [Qwen](qwen.md), [labs landscape](../ecosystem/labs-landscape.md), [EU AI Act](../policy/eu-ai-act.md)
- Hands-on: [Inference servers](https://github.com/PatrickWiloak/cloud-data-ai-security-zero-to-hero/blob/main/learn/concepts/inference-servers.md), [GPUs for AI](https://github.com/PatrickWiloak/cloud-data-ai-security-zero-to-hero/blob/main/learn/concepts/gpus-for-ai.md)

## Sources

- Mistral 7B announcement (September 27, 2023): https://mistral.ai/news/announcing-mistral-7b
- Mixtral of Experts announcement (December 11, 2023): https://mistral.ai/news/mixtral-of-experts
- Mistral Large announcement (February 26, 2024): https://mistral.ai/news/mistral-large
- Codestral announcement (May 29, 2024): https://mistral.ai/news/codestral
- Mistral Large 2 announcement (July 24, 2024): https://mistral.ai/news/mistral-large-2407
- Mistral Small 3 announcement (January 30, 2025): https://mistral.ai/news/mistral-small-3
- Mistral Medium 3 announcement (May 7, 2025): https://mistral.ai/news/mistral-medium-3
- Magistral announcement (June 10, 2025): https://mistral.ai/news/magistral
- Mistral 3 announcement (December 2, 2025): https://mistral.ai/news/mistral-3
- Devstral 2 and Vibe CLI (December 9, 2025): https://mistral.ai/news/devstral-2-vibe-cli
- Mistral docs changelog: https://docs.mistral.ai/resources/changelogs
- Mistral news index (2026 posts: Small 4, Leanstral 1.5, Shieldstral, Series D): https://mistral.ai/news/
- Mistral Small 4 model card: https://huggingface.co/mistralai/Mistral-Small-4-119B-2603
- Mistral Medium 3.5 model card and LICENSE: https://huggingface.co/mistralai/Mistral-Medium-3.5-128B
- Mistral Medium 3.5 docs page (April 28, 2026, modified MIT): https://docs.mistral.ai/models/model-cards/mistral-medium-3-5-26-04
- Mistral Large 3 model card: https://huggingface.co/mistralai/Mistral-Large-3-675B-Instruct-2512
- Devstral 2 LICENSE: https://huggingface.co/mistralai/Devstral-2-123B-Instruct-2512
- Series C (September 9, 2025): https://mistral.ai/news/mistral-ai-raises-1-7-b-to-accelerate-technological-progress-with-ai
- Series D (September 8, 2026): https://mistral.ai/news/mistral-makes-sovereign-open-weight-ai-to-frontier/ and https://techcrunch.com/2026/09/08/mistral-raises-e3b-as-sovereign-ai-becomes-big-business/
- July 2026 early-access report: https://www.techtimes.com/articles/319798/20260706/mistral-ai-targets-frontier-gap-open-weight-model-entering-july-early-access.htm
- Founding and founders: https://en.wikipedia.org/wiki/Mistral_AI
- Mistral 7B paper, arXiv 2310.06825: https://arxiv.org/abs/2310.06825
