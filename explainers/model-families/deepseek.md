# The DeepSeek Family

**In one line:** DeepSeek is a Hangzhou lab whose models are defined by one habit - make the same capability cheaper to train and to serve through architecture changes (sparse experts, compressed attention, low precision), then publish the weights under MIT - so read each release as "which cost did they cut this time".
**Last reviewed:** 2026-09-30

---

## The short version

- **Efficiency is the product.** Almost every DeepSeek generation is remembered for a way of spending less compute or memory for the same quality: Multi-head Latent Attention (V2), cheap FP8 training of a 671B model (V3), sparse attention (V3.2), 1M-token context as the default (V4), and an asymmetric encoder-decoder design (V4.1).
- **Open weights, permissive licence.** Since R1 (January 2025) DeepSeek's models ship under the MIT licence, which lets anyone use, modify and sell them. That has stayed true into V4.1-Flash (September 2026), while several other Chinese labs moved to licences with commercial carve-outs in 2026.
- **R1 made "reasoning via reinforcement learning" public.** DeepSeek-R1 (January 20, 2025) was the first openly documented recipe for an o1-style reasoning model, and it was released with weights. Its reasoning line was later folded into the main V line.
- **As of September 2026** the current generation is V4 (preview April 24, 2026) and its successor V4.1-Flash (September 10, 2026). V4.1-Pro has been announced as coming but has no date.
- **It is a research-led company with an unusual owner.** DeepSeek was spun out of the Chinese quantitative hedge fund High-Flyer, which remains its principal owner.

## Who DeepSeek is

DeepSeek was founded in July 2023 by Liang Wenfeng, co-founder of High-Flyer, a quantitative trading fund that had set up an AI research lab and then spun it off as a separate company. It is based in Hangzhou, China. Its public output is unusually paper-heavy: nearly every major model ships with a technical report on arXiv or Hugging Face, and several of those reports became reference designs for other labs.

Because it trains on export-restricted hardware (the V3 report describes a cluster of NVIDIA H800s, the China-market version of the H100), the lab's constraints are visible in its designs. Many of its innovations are ways to do more with less memory bandwidth or fewer GPU hours.

## Release timeline

Dates are the first public release or API availability, taken from DeepSeek's own API changelog and model cards. "Summary" links go to the paper summary in this repo where one exists.

| Model | Date | What changed | Summary |
|---|---|---|---|
| DeepSeek LLM 7B / 67B | Late 2023 (report January 5, 2024) | First general models; 2 trillion training tokens; the 67B chat model compared against GPT-3.5 and Llama 2 70B. | - |
| DeepSeekMath | February 2024 | Introduced GRPO, a cheaper reinforcement learning algorithm that later trained R1. | [38](../../papers/techniques/38-grpo/summary.md) |
| DeepSeek-V2 | May 2024 | Mixture-of-experts, 236B total / 21B active parameters, 128K context. Introduced Multi-head Latent Attention (MLA): KV cache cut 93.3% and training cost 42.5% versus the 67B model. | [141](../../papers/architectures/141-multi-head-latent-attention/summary.md) |
| DeepSeek-Coder-V2 | June 2024 | Code-specialised V2. | - |
| DeepSeek-V2.5 | September 5, 2024 | Chat and coder models merged into one. | - |
| DeepSeek-V3 | December 26, 2024 | 671B total / 37B active; 14.8T training tokens; full training in 2.788M H800 GPU hours using FP8. | [27](../../papers/language-models/27-deepseek-v3/summary.md) |
| DeepSeek-R1 | January 20, 2025 | Reasoning model trained largely with RL on verifiable answers; MIT licence; six smaller "distilled" models released alongside. | [26](../../papers/language-models/26-deepseek-r1/summary.md) |
| DeepSeek-V3-0324 | March 24, 2025 | V3 refresh; same architecture; weights now under MIT. | - |
| DeepSeek-R1-0528 | May 28, 2025 | Stronger reasoning update to R1. | - |
| DeepSeek-V3.1 | August 21, 2025 | "Hybrid" model: one set of weights with a thinking mode and a non-thinking mode. The separate R line ends here. | - |
| DeepSeek-V3.1-Terminus | September 22, 2025 | Fixes for language mixing and agent behaviour. | - |
| DeepSeek-V3.2-Exp | September 29, 2025 | Experimental release introducing DeepSeek Sparse Attention (DSA). | - |
| DeepSeek-V3.2 / V3.2-Speciale | December 1, 2025 | V3.2 is the first DeepSeek model to use tools inside its thinking. Speciale is a reasoning-maximised variant that DeepSeek reports at gold-medal level on IMO, CMO, ICPC World Finals and IOI 2025. | - |
| DeepSeek-V4 Preview (V4-Pro, V4-Flash) | April 24, 2026 | Two tiers: Pro is 1.6T total / 49B active, Flash is 284B / 13B. 1M-token context becomes the default. Hybrid compressed sparse attention, Muon optimizer, FP4 expert weights. MIT. | - |
| DeepSeek-V4-Flash (official) | July 31, 2026 | Preview replaced; better agent performance. | - |
| DeepSeek-V4-Pro (GA, "0813") | August 13, 2026 | Production release focused on agents; low / high / max reasoning effort; peak and off-peak API pricing. | - |
| DeepSeek-V4-Flash-Vision-Exp | August 21, 2026 | Experimental image understanding. | - |
| DeepSeek-V4.1-Flash | September 10, 2026 | New "causal encoder-decoder" MoE: 552B parameters, 8B active when reading input, 16B when generating. Native image input. V4-Flash retired; from September 14 V4-Pro API traffic is routed to V4.1-Flash until V4.1-Pro ships. | - |

## The through-line: attack the expensive part

A useful mental model is that a large language model has three big bills: the **training bill** (GPU hours), the **memory bill at inference** (mostly the KV cache, the stored keys and values for every past token), and the **compute bill at inference** (how many parameters each token touches). DeepSeek's releases read like a campaign against each bill in turn.

```
Bill                      DeepSeek's answer                      Release
------------------------  -------------------------------------  -----------
Compute per token         Fine-grained mixture of experts:       V2, V3
                          many small experts, few active
Memory (KV cache)         Multi-head Latent Attention:           V2
                          store a small compressed "latent"
                          instead of full keys and values
Training cost             FP8 mixed precision, load balancing    V3
                          without an auxiliary loss
Reasoning cost            GRPO: RL without a separate value      DeepSeekMath,
                          model; rewards from checkable answers  R1
Attention at long length  DeepSeek Sparse Attention: each        V3.2
                          token attends to a selected subset
Long context as default   Compressed Sparse + Heavily            V4
                          Compressed Attention; FP4 experts
Prefill vs decode split   Causal encoder-decoder: fewer          V4.1
                          active params to read, more to write
```

**Mixture of experts (MoE)** means the model contains many sub-networks ("experts") and a router picks a few for each token. Total parameters can be enormous while the compute per token stays small. DeepSeek's version uses many small experts plus some always-on "shared" experts. See the [Mixtral / MoE summary](../../papers/architectures/37-mixture-of-experts/summary.md) for the general idea.

**MLA** is the idea most worth understanding. Standard attention caches a key and a value vector for every head, for every past token. MLA caches one compressed vector per token and reconstructs keys and values from it on the fly. That shrinks the memory needed per user, which is what limits how many users one GPU can serve. The [MLA summary](../../papers/architectures/141-multi-head-latent-attention/summary.md) covers it in detail, and the [KV cache explainer](../concepts/kv-cache.md) explains why the cache matters.

**Sparse attention** (V3.2 onward) goes further: rather than every new token looking at every past token, a cheap scoring step picks which past tokens matter. By V4 DeepSeek describes a hybrid of "Compressed Sparse Attention" and "Heavily Compressed Attention", and says 1M-token context is now the default across its services. The [context windows explainer](../concepts/context-windows.md) covers why long context is costly.

**Reasoning via RL.** R1 showed that if you reward a model only for reaching checkable correct answers (maths, code that passes tests), it learns to write long chains of thought on its own. The [R1 summary](../../papers/language-models/26-deepseek-r1/summary.md), [GRPO summary](../../papers/techniques/38-grpo/summary.md) and [RLVR summary](../../papers/techniques/39-rlvr/summary.md) cover the recipe; the [reasoning models explainer](../concepts/reasoning-models.md) puts it in context.

The V4 model card also lists two training changes other labs are watching: the **Muon optimizer** (see the [Muon summary](../../papers/techniques/144-muon/summary.md)) and "Manifold-Constrained Hyper-Connections", a change to how residual connections are wired.

## Licensing

| Period | Weights licence | What it means in practice |
|---|---|---|
| V2, original V3 | Code under MIT; weights under a separate DeepSeek Model Agreement that permits commercial use | Usable commercially, but with use-based restrictions in a custom document. |
| R1 onward (January 2025), V3-0324 onward | MIT for code and weights | Among the most permissive terms of any frontier-scale model: use, modify, distil and sell freely, keep the notice. |
| V4, V4.1-Flash (2026) | MIT | Unchanged. |

The R1 announcement explicitly invited people to "distill and commercialize freely", and the six distilled R1 models were built on other labs' open models (Qwen and Llama bases). That is one reason DeepSeek's influence runs wider than its own downloads. For how licence terms differ across the open ecosystem, see the [open vs closed weights explainer](../concepts/open-vs-closed-weights.md).

Open weights do not mean open everything. DeepSeek publishes weights, inference code and reports, but not its training data.

## Reading the names

- **V + number** is the base generation (V2, V3, V4). A **point release** (V3.1, V3.2, V4.1) usually keeps the parameter budget but changes architecture or training.
- **R** was the reasoning line (R1, R1-0528). Since V3.1 (August 2025) reasoning is a mode of the main model, not a separate model.
- **A four-digit suffix is a date**, month then day: V3-0324 is March 24, V4-Pro-0813 is August 13.
- **-Exp** marks an experimental architecture test, **-Terminus** marked the last V3.1 update, and **-Speciale** marked a reasoning-maximised variant.
- **Pro / Flash** (from V4) are size tiers: Pro is the large flagship, Flash the smaller, cheaper model.
- **API aliases are not model names.** For most of 2025 `deepseek-chat` and `deepseek-reasoner` pointed at whatever was current; DeepSeek retired both on July 24, 2026, and in September 2026 told developers to use `deepseek-flash`.

## Why it matters beyond DeepSeek

- **It moved the open-weight frontier.** V3 and R1 were, at release, the strongest openly downloadable models in their categories, and their reports were detailed enough to reproduce ideas from.
- **It made cost a headline.** The V3 report's figure of 2.788M H800 GPU hours for the final training run started a wide debate about what frontier training costs. Note that this figure covers the final run only, not research, failed runs, data or staff. The [cost of training explainer](../compute/cost-of-training.md) covers how to read such numbers.
- **It is part of a policy story.** DeepSeek models are restricted on some government systems in several countries; the [US AI policy explainer](../policy/us-ai-policy.md) covers the wider export-control context.

## What to watch

- **V4.1-Pro.** DeepSeek's September 10, 2026 notice says V4.1-Pro follows the V4-Pro transition that began September 14, with no date given. Until then, V4-Pro API calls are served by V4.1-Flash.
- **Whether the encoder-decoder turn spreads.** V4.1-Flash departs from the decoder-only design almost every large model uses. If V4.1-Pro keeps it and holds up on independent benchmarks, expect other labs to test it.
- **Licence stability.** DeepSeek is now one of the few large Chinese labs still on plain MIT. A change there would be a signal for the whole open-weight ecosystem.
- **Multimodality.** Image input arrived only in August and September 2026; check whether V4.1-Pro is natively multimodal.
- **Independent verification.** Benchmark figures on DeepSeek's pages are self-reported. Look for third-party results (see the [contamination and saturation explainer](../benchmarks/contamination-and-saturation.md)).

## Read next

- [DeepSeek-V2 / Multi-head Latent Attention](../../papers/architectures/141-multi-head-latent-attention/summary.md)
- [DeepSeek-V3 Technical Report](../../papers/language-models/27-deepseek-v3/summary.md)
- [DeepSeek-R1](../../papers/language-models/26-deepseek-r1/summary.md)
- [GRPO](../../papers/techniques/38-grpo/summary.md) and [Muon](../../papers/techniques/144-muon/summary.md)
- [Mixture of Experts](../../papers/architectures/37-mixture-of-experts/summary.md) and [Knowledge Distillation](../../papers/techniques/134-knowledge-distillation/summary.md)
- Explainers: [Qwen](qwen.md), [Mistral](mistral.md), [Llama](llama.md), [labs landscape](../ecosystem/labs-landscape.md), [KV cache](../concepts/kv-cache.md), [reasoning models](../concepts/reasoning-models.md)
- Hands-on: [Quantization and distillation](https://github.com/PatrickWiloak/cloud-data-ai-security-zero-to-hero/blob/main/learn/concepts/quantization-and-distillation.md), [Inference servers](https://github.com/PatrickWiloak/cloud-data-ai-security-zero-to-hero/blob/main/learn/concepts/inference-servers.md)

## Sources

- DeepSeek API change log (all dated releases 2024-2026): https://api-docs.deepseek.com/updates/
- DeepSeek V4 Preview release, April 24, 2026: https://api-docs.deepseek.com/news/news260424/
- DeepSeek-V4-Pro GA release, August 13, 2026: https://api-docs.deepseek.com/news/news260813/
- DeepSeek-V4.1-Flash release, September 10, 2026: https://api-docs.deepseek.com/news/news260910/
- DeepSeek-V4-Pro model card (MIT, 1.6T / 49B, CSA + HCA, mHC, Muon, FP4 + FP8): https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro
- DeepSeek-V4.1-Flash model card (MIT, 552B, causal encoder-decoder, 1M context): https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash
- DeepSeek-V3.2 release, December 1, 2025: https://api-docs.deepseek.com/news/news251201/
- DeepSeek-V3.1 release, August 21, 2025: https://api-docs.deepseek.com/news/news250821/
- DeepSeek-R1 release, January 20, 2025 (MIT, distilled models): https://api-docs.deepseek.com/news/news250120/
- DeepSeek-V3 model card (671B / 37B, 2.788M H800 GPU hours, licence split): https://huggingface.co/deepseek-ai/DeepSeek-V3
- DeepSeek-V3-0324 model card (MIT): https://huggingface.co/deepseek-ai/DeepSeek-V3-0324
- DeepSeek-V2 paper, arXiv 2405.04434: https://arxiv.org/abs/2405.04434
- DeepSeek LLM paper, arXiv 2401.02954: https://arxiv.org/abs/2401.02954
- DeepSeekMath / GRPO, arXiv 2402.03300: https://arxiv.org/abs/2402.03300
- Company background (founding, High-Flyer, Hangzhou): https://en.wikipedia.org/wiki/DeepSeek
