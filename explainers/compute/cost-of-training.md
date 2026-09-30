# The Cost of Training

**In one line:** Training compute can be estimated with one formula (about 6 x parameters x tokens), and by that measure the cost of a frontier training run has grown from about $2 million (GPT-3, 2020) to roughly $500 million (Grok 4, 2025), so treat any single headline cost, including DeepSeek's "$5.6 million", as one slice of a much larger bill.
**Last reviewed:** 2026-09-30

---

## The short version

- **Training compute is predictable.** For a standard transformer, total training FLOPs (floating-point operations) are roughly `6 x N x D`, where N is parameter count and D is training tokens.
- **Compute turns into dollars through GPU-hours.** Divide FLOPs by what the hardware actually achieves per second, multiply by the price of an hour of that hardware.
- **The trend is steep.** Epoch AI estimates frontier training compute grows about 5x per year and training cost about 3.5x per year (as of February 2026). Its cost estimates run from about $2M for GPT-3 to about $500M for Grok 4.
- **The final run is not the whole bill.** Epoch's study of GPT-4 and Gemini found staff costs almost as large as hardware, and experiments, failed runs and research come on top.
- **DeepSeek-V3's $5.576M is real but narrow.** It is 2.788M H800 GPU-hours at an assumed $2 per hour, for the final run only. DeepSeek's paper says so explicitly.
- **Energy is now the binding constraint.** Frontier runs draw tens to hundreds of megawatts; data centre projects are now announced in gigawatts.

## The mental model: FLOPs are the currency

Think of training as a very large, very regular calculation. Every token of training data passes forward through the network and then backward to compute gradients. For each parameter, that is about:

```
forward pass:   ~2 FLOPs per parameter per token   (one multiply, one add)
backward pass:  ~4 FLOPs per parameter per token   (roughly twice the forward)
--------------------------------------------------
total:          ~6 FLOPs per parameter per token

Training FLOPs  C  ~=  6 x N x D
```

This is the approximation used in the [scaling laws](../../papers/techniques/12-scaling-laws/summary.md) paper and nearly all later compute estimates. It ignores attention's extra cost at long context, which is small for most pretraining setups.

For a [mixture-of-experts](../../papers/architectures/37-mixture-of-experts/summary.md) model, N is the number of **active** parameters per token, not the total. That distinction is why sparse models are cheap to train for their size.

### A worked check: GPT-3

GPT-3 has 175 billion parameters and was trained on about 300 billion tokens:

```
6 x 175e9 x 300e9  =  3.15e23 FLOPs
```

The GPT-3 paper reports 3.14e23, so the rule of thumb lands within a rounding error.

### From FLOPs to dollars

```
GPU-hours  =  C  /  (peak FLOPS per GPU  x  utilisation  x  3600)
Cost       =  GPU-hours  x  price per GPU-hour
```

**Utilisation** (often called MFU, model FLOPs utilisation) is the fraction of peak the run actually achieves. Communication between chips, memory stalls, restarts after hardware failures and pipeline "bubbles" all eat into it, so real runs achieve well below the vendor's peak figure. This is where engineering skill shows up as money: the same model costs very different amounts to train at different utilisation. Techniques such as [mixed precision](../../papers/techniques/143-mixed-precision-training/summary.md) (doing most arithmetic in 16-bit or 8-bit formats) and [ZeRO and Megatron-style parallelism](../../papers/techniques/76-zero-megatron/summary.md) exist largely to push utilisation and per-chip throughput up.

**Price per GPU-hour** depends on whether you rent (cloud rate) or own (hardware amortised over its useful life plus power). Epoch AI's cost studies use both methods and report them separately.

### How big should N and D be?

For a fixed compute budget, you can choose a bigger model trained on fewer tokens or a smaller model trained on more. [Chinchilla](../../papers/techniques/18-chinchilla/summary.md) found the compute-optimal balance is roughly 20 training tokens per parameter. In practice labs now train smaller models on far more tokens than that, because a smaller model is cheaper to serve for its whole life. Training cost and [inference cost](inference-economics.md) are one budget, not two.

## The trend: 2020 to 2026

Epoch AI maintains the most widely cited public dataset of training runs. Its estimates are reconstructions from public information, often with wide uncertainty, but the direction is not in doubt.

| Year | Model | Epoch AI estimated training cost | What changed |
|---|---|---|---|
| 2020 | GPT-3 175B | about $2 million | The first widely used large language model |
| 2023 | GPT-4 | about $40 million | Frontier moves to tens of millions |
| 2024 | Llama 3.1 405B | about $50 million | Largest openly released dense model of its time |
| 2025 | Grok 4 | about $490-500 million | Largest run in Epoch's dataset as of its September 2025 analysis |

Headline rates from Epoch AI (as of its trends page, updated February 2026):

- **Training compute:** about 5x per year for frontier language models, roughly 10,000x since 2020.
- **Training cost:** about 3.5x per year (doubling every 7 months). An earlier Epoch paper (2024) estimated 2.4x per year for amortised hardware and energy since 2016; the rates differ because they cover different periods and methods.
- **Hardware price-performance:** AI chip performance per dollar up about 49% per year since 2023.
- **Algorithmic efficiency:** the compute needed to reach a given pretraining result falls about 3x per year.

The last two numbers are why cost grows slower than compute: each year's dollar buys more FLOPs, and each year's FLOP is used more cleverly.

The Grok 4 estimate illustrates the method. Epoch inferred about 246 million H100-hours from xAI's public statements, priced them two ways (rental rates, and depreciation plus power) and arrived at a median of $490M and about 310 GWh of electricity. Epoch itself warns that xAI's statements are vague, so the uncertainty is large.

### What a frontier run leaves out

Epoch's 2024 paper broke down the full development cost of GPT-4 and Gemini Ultra, not just the final run:

| Component | Share of development cost |
|---|---|
| AI accelerator chips (amortised) | the largest single item |
| R&D staff | 29-49% |
| Other server components | 15-22% |
| Cluster interconnect | 9-13% |
| Energy | 2-6% |

The paper projected that if trends held, the largest training runs would cost more than $1 billion by 2027. Staff and experimentation are real costs that a "cost to train" headline usually omits.

## The DeepSeek-V3 "$5.6 million"

In December 2024 DeepSeek published the [DeepSeek-V3](../../papers/language-models/27-deepseek-v3/summary.md) technical report, and a single number from it moved markets in January 2025. Here is what the report actually says:

```
Pretraining (14.8T tokens)      2,664K H800 GPU-hours   (under 2 months on 2,048 H800s)
Context extension                 119K GPU-hours
Post-training                       5K GPU-hours
-------------------------------------------------
Total                           2,788K GPU-hours
x assumed rental price $2/GPU-hour  =  $5.576M
```

And the sentence that followed: the figure "include[s] only the official training of DeepSeek-V3, excluding the costs associated with prior research and ablation experiments on architectures, algorithms, or data."

**What it did include:** the GPU time of the final pretraining run, long-context extension and post-training, priced at a rental rate.

**What it did not include:**

- The research and ablation runs that produced the design (multi-head latent attention, auxiliary-loss-free load balancing, multi-token prediction, FP8 training).
- Earlier models it built on, such as [DeepSeek-V2](../../papers/architectures/141-multi-head-latent-attention/summary.md).
- Buying and running the cluster, staff, data acquisition, and the reinforcement learning work behind [DeepSeek-R1](../../papers/language-models/26-deepseek-r1/summary.md), which was released separately.

**Is the number plausible?** Check it with 6ND. DeepSeek-V3 activates 37B of its 671B parameters per token:

```
6 x 37e9 x 14.8e12  =  3.3e24 FLOPs
```

By the same formula, Llama 3.1 405B (a dense model, trained on over 15 trillion tokens) comes to about 6 x 405e9 x 15e12 = 3.6e25 FLOPs, roughly ten times more, for a model DeepSeek-V3 matched or beat on many benchmarks. The claim is consistent: DeepSeek's efficiency came from the mixture-of-experts design (few active parameters) and aggressive engineering around H800 interconnect limits, not from an accounting trick.

**The strongest critique:** SemiAnalysis estimated in early 2025 that DeepSeek had access to about 50,000 Hopper-generation GPUs and had spent about $1.6 billion on servers overall. That figure is an outside estimate, not a DeepSeek disclosure, and it describes total capital across all of the company's work, not one model. Both numbers can be true at once: a cheap final run sitting on top of an expensive research operation.

**The lesson for reading any cost claim:** ask which of these it covers - final run, all experiments, hardware purchase, staff - and whether the GPU-hour price is a rental rate or an owner's cost.

## Energy and the data centre build-out

Energy is a small share of a training run's cost (2-6% in Epoch's breakdown), but power availability has become the constraint on building bigger clusters.

- **Frontier runs:** Epoch estimates power for frontier training roughly doubles each year, with current runs drawing tens to hundreds of megawatts. Its 2024 paper estimated Gemini Ultra needed about 35 MW.
- **Data centres overall:** the International Energy Agency estimated data centres used about 415 TWh in 2024 (about 1.5% of global electricity) and projects about 945 TWh by 2030 in its base case, with AI-accelerated servers growing about 30% per year.
- **Gigawatt campuses:** OpenAI's Stargate programme with Oracle and SoftBank targets about 10 GW of US capacity. Epoch AI's April 2026 review found about 0.3 GW operating at Abilene, Texas, six more US sites under construction, and total capacity projected to exceed 9 GW by 2029, while noting plans had already changed (OpenAI dropped a planned Abilene expansion) and that local opposition and financing are real risks.
- **Other labs:** Anthropic's October 2025 Google TPU agreement brings "well over a gigawatt" online in 2026. xAI trained Grok 4 on its Colossus cluster in Memphis.

For the hardware inside these buildings, see [The AI Hardware Landscape](ai-hardware-landscape.md).

## Two honest caveats

1. **Most frontier labs do not publish costs.** Every number above for a closed model is an outside estimate. Treat single figures as order-of-magnitude, and prefer ranges.
2. **Pretraining is no longer the whole story.** Since 2024, reinforcement learning on reasoning tasks (see [Reasoning Models](../concepts/reasoning-models.md) and [GRPO](../../papers/techniques/38-grpo/summary.md)) has become a growing share of compute, and it does not fit the simple 6ND formula, because much of its cost is generating samples (inference) rather than gradient steps.

## What to watch

- **Whether a disclosed run crosses $1 billion.** Epoch's 2024 paper projected this by 2027.
- **Rubin-generation clusters (from H2 2026).** Vendor claims of 4x fewer GPUs for mixture-of-experts training would, if borne out, slow cost growth.
- **RL compute share.** If post-training grows to rival pretraining, cost estimates built on 6ND will understate true cost.
- **Power and permitting.** Grid connections, gas turbines and local opposition now set the pace of new capacity more than chip supply.
- **Stargate and peer projects.** Epoch's site tracker is the best public check on announced versus operating gigawatts.

## Read next

- [Scaling Laws for Neural Language Models](../../papers/techniques/12-scaling-laws/summary.md) - where compute as the key variable comes from
- [Chinchilla](../../papers/techniques/18-chinchilla/summary.md) - how to split a compute budget between size and data
- [DeepSeek-V3](../../papers/language-models/27-deepseek-v3/summary.md) - the model behind the $5.6M figure
- [ZeRO and Megatron-LM](../../papers/techniques/76-zero-megatron/summary.md) - how training is spread over thousands of chips
- [Mixed Precision Training](../../papers/techniques/143-mixed-precision-training/summary.md) - the lower-precision arithmetic that makes runs cheaper
- [Inference Economics](inference-economics.md) - the other half of the budget
- [The AI Hardware Landscape](ai-hardware-landscape.md)
- [Scaling Limits](../open-questions/scaling-limits.md) - whether this curve can continue
- [DeepSeek model family](../model-families/deepseek.md)

## Sources

- Kaplan et al., "Scaling Laws for Neural Language Models" (2020), 6ND approximation: https://arxiv.org/abs/2001.08361
- Brown et al., "Language Models are Few-Shot Learners" (GPT-3, 2020), 3.14e23 FLOPs: https://arxiv.org/abs/2005.14165
- Hoffmann et al., "Training Compute-Optimal Large Language Models" (Chinchilla, 2022): https://arxiv.org/abs/2203.15556
- DeepSeek-AI, "DeepSeek-V3 Technical Report" (December 2024), GPU-hours, $2/hour assumption and exclusions, section 1: https://arxiv.org/abs/2412.19437
- Epoch AI, Trends in AI (updated February 5, 2026): https://epoch.ai/trends
- Cottier et al. (Epoch AI), "The rising costs of training frontier AI models" (2024): https://arxiv.org/abs/2405.21015
- Epoch AI, "How much does it cost to train frontier AI models?" (June 3, 2024): https://epoch.ai/blog/how-much-does-it-cost-to-train-frontier-ai-models
- Epoch AI, "Training compute costs are doubling every eight months for the largest AI models" (per-model cost table, GPT-3 to Grok 4): https://epoch.ai/data-insights/cost-trend-large-scale
- Epoch AI, "What did it take to train Grok 4?" (September 12, 2025): https://epoch.ai/data-insights/grok-4-training-resources
- Epoch AI, "OpenAI Stargate: where the US sites stand" (April 17, 2026): https://epoch.ai/publications/openai-stargate-where-the-us-sites-stand
- SemiAnalysis, "DeepSeek Debates" (January 2025): https://newsletter.semianalysis.com/p/deepseek-debates
- Tom's Hardware, summary of the SemiAnalysis estimate: https://www.tomshardware.com/tech-industry/artificial-intelligence/deepseek-might-not-be-as-disruptive-as-claimed-firm-reportedly-has-50-000-nvidia-gpus-and-spent-usd1-6-billion-on-buildouts
- Llama Team, Meta, "The Llama 3 Herd of Models" (2024), 405B parameters and training tokens: https://arxiv.org/abs/2407.21783
- IEA, "Energy and AI" (2025), energy demand from AI: https://www.iea.org/reports/energy-and-ai/energy-demand-from-ai
- CNBC, "OpenAI's first data center in $500 billion Stargate project is open in Texas" (September 23, 2025): https://www.cnbc.com/2025/09/23/openai-first-data-center-in-500-billion-stargate-project-up-in-texas.html
- Google Cloud Press Corner, "Anthropic to Expand Use of Google Cloud TPUs and Services" (October 23, 2025): https://www.googlecloudpresscorner.com/2025-10-23-Anthropic-to-Expand-Use-of-Google-Cloud-TPUs-and-Services
- NVIDIA Newsroom, Rubin platform (January 5, 2026): https://nvidianews.nvidia.com/news/rubin-platform-ai-supercomputer
