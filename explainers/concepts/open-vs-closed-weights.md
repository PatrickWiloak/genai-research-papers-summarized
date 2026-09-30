# Open vs Closed Weights

**In one line:** "Open" in AI is a spectrum, from API-only access, through downloadable weights under restrictive or permissive licences, to fully open source with code and data, so ask exactly what is released and under which licence before deciding what you can build, and read the safety debate as a disagreement about marginal risk rather than about openness in general.
**Last reviewed:** 2026-09-30

---

## The short version

- A trained model is mostly its **weights**: billions of numbers learned in training. Whoever holds the weights can run, inspect, fine-tune and copy the model. Whether the public gets the weights is the central question in "open vs closed".
- **Closed (API-only)** models, such as the flagship GPT, Claude and Gemini models, are used over the internet on the provider's terms. You never see the weights.
- **Open-weight** models, such as Llama, DeepSeek, Qwen, Mistral's open releases, Gemma and OpenAI's gpt-oss, can be downloaded and run yourself. Their **licences differ a lot**: some are standard permissive licences (Apache 2.0, MIT), others are custom licences with usage restrictions.
- **Open source AI**, as defined by the Open Source Initiative in October 2024, asks for more than weights: the training and inference code and detailed information about the training data, all under terms that allow use, study, modification and sharing. Few prominent models meet it; Ai2's Olmo family is a well-known one built to.
- The **trade-offs** are control, privacy, cost and customisation (favouring open weights) against peak capability, managed safety and zero operations (favouring closed APIs). Many teams use both.
- The **policy debate** is about whether releasing the weights of the most capable models adds meaningful risk (misuse, removable safeguards) compared with what is already available. As of September 2026, US federal policy encourages open weights and the EU AI Act gives open-source models lighter duties, except for models deemed to carry systemic risk.

## The spectrum

Irene Solaiman's "The Gradient of Generative AI Release" (2023) described release as a gradient rather than a switch, running from fully closed through staged, hosted and API access to downloadable and fully open. A simplified version:

```
CLOSED <------------------------------------------------------------> OPEN

internal   API / chat app    open weights,      open weights,     open source
only       (weights stay     custom licence     permissive        (weights + code
           with provider)    (use limits,       licence           + data info,
                             attribution)       (Apache, MIT)     OSI definition)

           GPT, Claude,      Llama 4,           DeepSeek-R1,      Olmo
           Gemini flagships  Gemma 1-3          Qwen3, gpt-oss,
                                                Gemma 4
```

What is released at each point:

| Level | Weights | Training code | Training data | Licence freedom |
|---|---|---|---|---|
| API-only | No | No | No | Terms of service only |
| Open weights, custom licence | Yes | Usually not | Rarely; often a summary at most | Restricted (use policies, size thresholds, naming rules) |
| Open weights, permissive licence | Yes | Sometimes inference code only | Rarely | Broad: use, modify, redistribute, commercial |
| Open source (OSI definition) | Yes | Yes, training and inference | Detailed data information; datasets where they can be shared | Terms that grant all four freedoms |

## The Open Source AI Definition

The **Open Source Initiative (OSI)**, which maintains the long-standing definition of open source software, released the **Open Source AI Definition 1.0** at the All Things Open conference in October 2024. It carries the four software freedoms over to AI systems: freedom to **use** the system for any purpose, **study** how it works, **modify** it, and **share** it. To make those freedoms practical, it requires the "preferred form for making modifications", which it spells out as:

- **Data information**: sufficiently detailed information about the training data (sources, selection, labelling, processing, how to obtain public datasets) that a skilled person could build a substantially equivalent system.
- **Code**: the complete source code used to train and run the system.
- **Parameters**: the model weights and relevant configuration, including checkpoints.

The definition does **not** require releasing the training data itself, only detailed information about it. That compromise was contested: some free-software advocates argue that without the data you cannot truly study or rebuild a model, while developers point to copyright, privacy and licensing limits on redistributing web-scale data. In OSI's own assessment of models against the definition, Meta's Llama 2 fell short.

Why this matters in practice: calling a model "open source" is partly marketing. Many "open" models are open weights under a custom licence, which is useful but is not open source in the OSI sense. For how open models plug into the wider tooling stack, see the [open-source stack](../ecosystem/open-source-stack.md) explainer.

## Licences: what the fine print says

Licences are where "open" gets specific. Some representative examples, as of September 2026:

| Model family | Licence | Notable terms |
|---|---|---|
| **Llama 4** (Meta, April 2025) | Llama 4 Community License | Companies whose products exceeded 700 million monthly active users must request a separate licence; must display "Built with Llama"; fine-tuned derivatives must start their name with "Llama"; bound by an Acceptable Use Policy |
| **Gemma 1-3** (Google) | Gemma Terms of Use | Custom terms with a Prohibited Use Policy |
| **Gemma 4** (Google, March 2026) | Apache 2.0 | Standard permissive licence |
| **DeepSeek-R1** (January 2025) | MIT | Explicitly allows commercial use and distillation into other models |
| **Qwen3** (Alibaba) | Apache 2.0 | Standard permissive licence |
| **Mistral 7B** (2023) | Apache 2.0 | Not every later Mistral model uses the same terms |
| **gpt-oss-120b / 20b** (OpenAI, August 2025) | Apache 2.0 | OpenAI's first open-weight language models since GPT-2 |
| **Muse Glimmer** (Meta, August 2026) | Apache 2.0 | A 30B dense multimodal model, released after Meta's flagship Muse Spark launched as closed |
| **Olmo 3** (Ai2, 2025) | Open weights plus training data, code and intermediate checkpoints | Built to meet the fully open end of the spectrum |

A clear 2025-2026 trend is **towards standard permissive licences** for open-weight releases: OpenAI (gpt-oss), Google (from Gemma 4) and Meta (Muse Glimmer) all used Apache 2.0, joining DeepSeek, Qwen and Mistral's open models. The opposite move is visible at the very top: Meta launched **Muse Spark**, the first model from Meta Superintelligence Labs, in April 2026 as a closed model offered by API to select partners, saying it hopes to open-source future versions. Model-by-model history is in the [Llama](../model-families/llama.md), [DeepSeek](../model-families/deepseek.md), [Qwen](../model-families/qwen.md) and [Mistral](../model-families/mistral.md) explainers.

Licence reading tips:

- **"Free for commercial use" may have a ceiling** (the 700M MAU clause) or conditions (attribution, naming).
- **Acceptable use policies are licence terms**, not just guidelines, and some must be passed on to your own users.
- **Derivatives inherit.** A model fine-tuned from Llama carries Llama's terms; DeepSeek's own model card notes that its distilled variants built on Qwen and Llama keep those base models' licences.

## Trade-offs for builders

| | Closed API | Open weights (self-hosted or third-party hosted) |
|---|---|---|
| **Peak capability** | Usually the frontier | Typically close behind; the gap has varied over time (see [labs landscape](../ecosystem/labs-landscape.md)) |
| **Data control and privacy** | Data goes to the provider under their terms | Can run entirely on your own hardware or network |
| **Customisation** | Prompting, plus provider fine-tuning where offered | Full fine-tuning, [LoRA](../../papers/techniques/10-lora/summary.md), quantization, architecture changes |
| **Cost shape** | Pay per token; no fixed cost | Pay for hardware or hosting; cheaper at high steady volume, more expensive when idle |
| **Operations** | None | You own serving, scaling, updates and security (see [inference servers](https://github.com/PatrickWiloak/cloud-data-ai-security-zero-to-hero/blob/main/learn/concepts/inference-servers.md)) |
| **Stability** | Models retired on the provider's schedule | A version you downloaded never changes or disappears |
| **Safety layers** | Provider runs filters and monitoring | Your responsibility (for example [Llama Guard](../../papers/language-models/96-llama-guard/summary.md)); see [guardrails](https://github.com/PatrickWiloak/cloud-data-ai-security-zero-to-hero/blob/main/learn/concepts/guardrails-and-safety.md) |
| **Research access** | Behaviour only | Weights, activations, internals (for example [sparse autoencoders](../../papers/techniques/82-sparse-autoencoders/summary.md)) |

The sibling repo's [fine-tuning vs RAG](https://github.com/PatrickWiloak/cloud-data-ai-security-zero-to-hero/blob/main/learn/concepts/fine-tuning-vs-rag.md), [quantization and distillation](https://github.com/PatrickWiloak/cloud-data-ai-security-zero-to-hero/blob/main/learn/concepts/quantization-and-distillation.md) and [GPUs for AI](https://github.com/PatrickWiloak/cloud-data-ai-security-zero-to-hero/blob/main/learn/concepts/gpus-for-ai.md) pages cover the hands-on side of running open models, and [AI threat modeling](https://github.com/PatrickWiloak/cloud-data-ai-security-zero-to-hero/blob/main/learn/concepts/ai-threat-modeling.md) covers securing them. Cost comparisons are in [inference economics](../compute/inference-economics.md).

## The policy debate

The argument is not about whether openness is good in general. It is about **whether releasing the weights of the most capable models adds risk that closed release would avoid**, and whether that outweighs the benefits.

**The case for open weights, at its strongest:**

- **Research and transparency.** Independent scientists can study, audit and red-team a model's internals only if they have the weights.
- **Competition and power.** Open weights stop a few companies from controlling access to a general-purpose technology, and let smaller firms, universities and other countries build on it.
- **Privacy and sovereignty.** Sensitive data can stay on-premises, and organisations are not dependent on one provider's pricing, policies or continued existence.
- **Marginal risk.** Kapoor, Bommasani, Klyman and colleagues (2024) argued that risks should be judged by what open release adds beyond existing tools such as search engines, closed models and published science, and found current evidence insufficient to characterise that marginal risk.

**The case for caution, at its strongest:**

- **Irreversibility.** Once weights are published they cannot be recalled or patched. A dangerous capability discovered later stays in circulation.
- **Safeguards come off.** Safety training can be undone by fine-tuning. Qi et al. (2023) removed much of GPT-3.5 Turbo's safety behaviour by fine-tuning on just 10 harmful examples, for under $0.20, through OpenAI's own API; with open weights there is no API to monitor or block that.
- **Capability thresholds.** Several frontier labs commit to restricting models that cross defined danger thresholds, for example in biology or cybersecurity (see [frontier safety frameworks](../policy/frontier-safety-frameworks.md)). Those commitments are hard to apply once weights are public.
- **Geopolitics.** Weights released by any lab are available to everyone, including rival states and competitors.

**Where governments stand (as of September 2026):**

- **United States.** NTIA's July 30, 2024 report concluded the government should **not restrict** widely available model weights at that time, and should instead build the capacity to monitor risks and act if evidence warranted it. **America's AI Action Plan** (July 23, 2025) went further, with a section on encouraging open-source and open-weight AI. See [US AI policy](../policy/us-ai-policy.md).
- **European Union.** The AI Act's Article 53(2) exempts general-purpose models released under a free and open-source licence, with public weights and architecture information, from two documentation duties. They must still have a copyright policy and publish a summary of training content, and **the exemption does not apply to models with systemic risk**. See the [EU AI Act](../policy/eu-ai-act.md).

## What to watch

- **Where the frontier labs land.** OpenAI and Google now release permissively licensed open models below their flagships, while Meta's newest flagship is closed. Which labs ship the most capable open-weight models, and how far behind the closed frontier they sit, shifts from quarter to quarter.
- **Systemic-risk designations.** How the EU applies the systemic-risk carve-out to open models will decide whether frontier-scale open releases carry the full compliance load.
- **Fully open models.** Whether releases with data and training code (Olmo-style) close the gap with open-weight-only models.
- **Tamper-resistant safeguards.** Safety training that survived fine-tuning would change the risk calculation substantially, if it could be made to work.

## Read next

- Open-weight releases in this repo: [LLaMA](../../papers/language-models/15-llama/summary.md), [Llama 2](../../papers/language-models/17-llama2/summary.md), [Llama 4](../../papers/language-models/41-llama4/summary.md), [Mistral 7B](../../papers/language-models/95-mistral-7b/summary.md), [DeepSeek-V3](../../papers/language-models/27-deepseek-v3/summary.md), [DeepSeek-R1](../../papers/language-models/26-deepseek-r1/summary.md), [Qwen3](../../papers/language-models/28-qwen3/summary.md)
- Closed releases: [GPT-4](../../papers/language-models/36-gpt4/summary.md), [GPT-5](../../papers/language-models/42-gpt5/summary.md), [Claude 4](../../papers/language-models/43-claude4/summary.md), [Gemini 3](../../papers/multimodal/47-gemini3/summary.md)
- Safety context: [Red teaming LMs with LMs](../../papers/techniques/129-red-teaming-lms/summary.md), [GCG adversarial attacks](../../papers/techniques/127-gcg-adversarial-attacks/summary.md), [Llama Guard](../../papers/language-models/96-llama-guard/summary.md)
- Explainers: [Labs landscape](../ecosystem/labs-landscape.md), [Open-source stack](../ecosystem/open-source-stack.md), [EU AI Act](../policy/eu-ai-act.md), [US AI policy](../policy/us-ai-policy.md), [Frontier safety frameworks](../policy/frontier-safety-frameworks.md)

## Sources

- Solaiman, "The Gradient of Generative AI Release: Methods and Considerations", February 2023. https://arxiv.org/abs/2302.04844
- Open Source Initiative, "The Open Source AI Definition 1.0". https://opensource.org/ai/open-source-ai-definition
- Open Source Initiative, "2024 end-of-year review: Open Source AI Definition v1.0" (released at All Things Open, October 2024; Llama 2 assessed as falling short). https://opensource.org/blog/2024-end-of-year-review-open-source-ai-definition-v1-0
- Meta, Llama 4 Community License Agreement. https://github.com/meta-llama/llama-models/blob/main/models/llama4/LICENSE
- Meta, "The Llama 4 herd", April 5, 2025. https://ai.meta.com/blog/llama-4-multimodal-intelligence/
- Meta, "Introducing Muse Spark", April 8, 2026. https://about.fb.com/news/2026/04/introducing-muse-spark-meta-superintelligence-labs/
- Hugging Face, "Meta is back with Muse Glimmer", August 10, 2026. https://huggingface.co/blog/muse-glimmer
- Google, Gemma releases (Gemma 4, March 31, 2026, Apache 2.0) and Gemma Terms of Use. https://ai.google.dev/gemma/docs/releases and https://ai.google.dev/gemma/terms
- DeepSeek-R1 model card (MIT licence; distillation allowed; distilled variants keep base licences). https://huggingface.co/deepseek-ai/DeepSeek-R1
- OpenAI, "Introducing gpt-oss", August 5, 2025, and the gpt-oss model card. https://openai.com/index/introducing-gpt-oss/ and https://arxiv.org/abs/2508.10925
- Ai2, Olmo overview (weights, data, code, checkpoints). https://allenai.org/olmo
- Kapoor, Bommasani, Klyman et al., "On the Societal Impact of Open Foundation Models", February 2024. https://arxiv.org/abs/2403.07918
- Qi et al., "Fine-tuning Aligned Language Models Compromises Safety, Even When Users Do Not Intend To!", October 2023. https://arxiv.org/abs/2310.03693
- NTIA, "Dual-Use Foundation Models with Widely Available Model Weights", July 30, 2024. https://www.ntia.gov/programs-and-initiatives/artificial-intelligence/open-model-weights-report
- The White House, "Winning the Race: America's AI Action Plan", July 2025. https://www.whitehouse.gov/wp-content/uploads/2025/07/Americas-AI-Action-Plan.pdf
- Open Source Initiative, "White House releases AI Action Plan, includes Open Source" (plan released July 23, 2025). https://opensource.org/blog/white-house-releases-ai-action-plan-includes-open-source
- EU AI Act, Article 53. https://artificialintelligenceact.eu/article/53/
