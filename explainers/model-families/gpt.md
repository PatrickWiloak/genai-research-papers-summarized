# The GPT and o-series Family (OpenAI)

**In one line:** OpenAI's models are one line of decoder-only transformers scaled up, then taught to follow instructions, then taught to think before answering, and since 2025 sold as a single product family with named capability tiers - read a model name as "generation, then tier".
**Last reviewed:** 2026-09-30

---

## The short version

- **One architecture, scaled and then post-trained.** Every GPT is a decoder-only transformer that predicts the next token. The big changes over eight years were size (GPT-1 to GPT-3), instruction-following via human feedback (InstructGPT, ChatGPT), native multimodality (GPT-4o) and "thinking" via reinforcement learning (the o-series).
- **Two lines merged back into one.** From September 2024 to August 2025 OpenAI ran two families side by side: GPT (fast, general) and o-series (slow, reasoning). GPT-5 (August 2025) folded them together behind a router. Everything since is one family.
- **Closed weights, with one exception.** GPT-2 was the last fully released GPT. GPT-3 onward are API-only. The exception is gpt-oss (August 2025), two open-weight reasoning models under Apache 2.0.
- **Disclosure shrank as capability grew.** GPT-3 came with a detailed paper. GPT-4 came with a technical report that withheld size, data and architecture. Recent releases come with system cards that focus on safety testing, not on how the model was built.
- **As of September 2026 the frontier model is GPT-6 Astra** (September 3, 2026), with GPT-6 Sol, GPT-6 Luna and GPT-6.1 Sol as the cheaper tiers. Its system card rates it at the "Critical" level for cybersecurity under OpenAI's Preparedness Framework, and access to some capabilities is gated.

## Release timeline

Dates are the public announcement or first availability. "Summary" links go to the paper summary in this repo where one exists.

| Model | Date | What changed | Summary |
|---|---|---|---|
| GPT-1 | June 2018 | Showed that generative pretraining on unlabeled text, then fine-tuning, beats task-specific models. 117M parameters. | [93](../../papers/language-models/93-gpt1/summary.md) |
| GPT-2 | February 2019 | 1.5B parameters; zero-shot task performance from pretraining alone; staged release over safety concerns. | [64](../../papers/language-models/64-gpt2/summary.md) |
| GPT-3 | May 2020 | 175B parameters; few-shot "in-context learning" from examples in the prompt. API only, no weights. | [04](../../papers/language-models/04-gpt3-few-shot-learners/summary.md) |
| Codex | July 2021 | GPT fine-tuned on public code; powered the first GitHub Copilot. | [56](../../papers/language-models/56-codex/summary.md) |
| InstructGPT | March 2022 | Reinforcement learning from human feedback (RLHF) makes a small model preferred over a raw 175B one. | [05](../../papers/language-models/05-instructgpt-rlhf/summary.md) |
| ChatGPT (GPT-3.5) | November 30, 2022 | The InstructGPT recipe in a chat interface; the consumer launch. | - |
| GPT-4 | March 14, 2023 | Large jump on exams and coding; image input; first report to withhold size and data. | [36](../../papers/language-models/36-gpt4/summary.md), [23](../../papers/multimodal/23-gpt4v/summary.md) |
| GPT-4o | May 13, 2024 | One "omni" model trained end to end on text, images and audio. | [40](../../papers/language-models/40-gpt4o/summary.md) |
| o1-preview / o1 | September 12, 2024 / December 5, 2024 | First "reasoning model": trained with RL to produce a long hidden chain of thought before answering. | [31](../../papers/language-models/31-openai-o1/summary.md) |
| GPT-4.5 | February 27, 2025 | Largest pure-pretraining GPT; the last big "non-reasoning" release. | - |
| GPT-4.1 | April 14, 2025 | API-focused model for coding and instruction following, 1M-token context. | - |
| o3 / o4-mini | April 16, 2025 | Reasoning models that can call tools (search, Python, images) inside their chain of thought. | - |
| gpt-oss-120b / gpt-oss-20b | August 5, 2025 | First open-weight OpenAI language models since GPT-2; mixture-of-experts reasoning models, Apache 2.0. | - |
| GPT-5 | August 7, 2025 | GPT and o-series merged: one system that routes between fast answers and deeper thinking. | [42](../../papers/language-models/42-gpt5/summary.md) |
| GPT-5.1 | November 12, 2025 | Adaptive reasoning (spends fewer thinking tokens on easy tasks), new coding tools in the API. | - |
| GPT-5.2 | December 11, 2025 | "Instant" and "Thinking" variants refreshed; more reliable day-to-day use. | - |
| GPT-5.3-Codex | February 5, 2026 | Agentic coding specialist. | - |
| GPT-5.4 (+ mini, nano) | March 5, 2026 (mini and nano March 17) | Better tool use across large tool sets; Pro variant in the API. | - |
| GPT-5.5 | April 23, 2026 | Stronger agentic coding at the same per-token latency as 5.4; OpenAI's "strongest safeguards to date" at the time. | - |
| GPT-5.6 Sol / Terra / Luna | July 9, 2026 (limited preview first) | New naming: number = generation, Sol / Terra / Luna = capability tiers. | - |
| GPT-6 Astra | September 3, 2026 (general rollout from September 4) | New top tier above Sol; 1.05M-token context; rated Critical for cyber capability. | - |
| GPT-6 Sol / GPT-6 Luna | September 22, 2026 | Astra's training methods applied to the cheaper tiers. | - |
| GPT-6.1 Sol | September 29, 2026 (DevDay) | Near-Astra results on agentic coding and computer use at a fifth of Astra's price. | - |

## The through-line: four bets, stacked

It is easiest to read OpenAI's history as four bets, each one layered on the last rather than replacing it.

```
2018-2020   Bet 1: SCALE        bigger decoder-only transformer + more text
                 |               (GPT-1 -> GPT-2 -> GPT-3)
2022        Bet 2: ALIGNMENT    human feedback turns a text predictor
                 |               into an assistant (InstructGPT -> ChatGPT)
2023-2024   Bet 3: MODALITY     images, then audio, trained into one model
                 |               (GPT-4 -> GPT-4o)
2024-2026   Bet 4: THINKING     RL on verifiable tasks teaches the model
                                 to reason at length before answering
                                 (o1 -> o3 -> GPT-5 -> GPT-6)
```

**Bet 1, scale.** GPT-1 established the recipe (pretrain on raw text, then adapt). GPT-2 and GPT-3 were the same recipe with roughly ten and a hundred times more parameters, and the surprise was that new abilities, like following a few examples in the prompt, appeared without being trained for. OpenAI's own [scaling laws](../../papers/techniques/12-scaling-laws/summary.md) work gave this bet its justification. For background on the architecture, see the sibling repo's [transformer architecture](https://github.com/PatrickWiloak/cloud-data-ai-security-zero-to-hero/blob/main/learn/concepts/transformer-architecture.md) page.

**Bet 2, alignment through feedback.** A raw GPT-3 continues text; it does not answer questions. [InstructGPT](../../papers/language-models/05-instructgpt-rlhf/summary.md) showed that fine-tuning on human demonstrations and then optimising against a reward model of human preferences (RLHF, using [PPO](../../papers/techniques/63-ppo/summary.md)) made a 1.3B model preferred over the 175B base model. ChatGPT was this recipe with a chat interface, and it is what turned a research line into a product.

**Bet 3, one model for every modality.** GPT-4 added image input. GPT-4o went further and trained a single network on text, images and audio together, instead of bolting separate speech and vision models onto a text model. The payoff was lower latency in voice and better cross-modal understanding.

**Bet 4, thinking.** o1 showed that a model trained with reinforcement learning to write a long private chain of thought gets better the longer it is allowed to think. This is "test-time compute": spending more computation at answer time rather than only at training time. See [test-time compute](../../papers/techniques/50-test-time-compute/summary.md), [RL with verifiable rewards](../../papers/techniques/39-rlvr/summary.md) and the [reasoning models explainer](../concepts/reasoning-models.md). GPT-5 then made thinking a dial rather than a separate product, and the 5.x and 6 releases tuned how much thinking a model spends on easy versus hard work.

Two quieter threads run alongside:

- **Coding and agents became the headline use case.** Codex (2021) was a side project; by 2026 most launch posts lead with agentic coding (Terminal-Bench, SWE-bench) and computer use. See the [agents and computer use benchmarks explainer](../benchmarks/agents-and-computer-use.md).
- **Safety process moved to the front of the launch.** Since GPT-5, releases ship with a system card on OpenAI's Deployment Safety Hub. GPT-5.6 Sol began as a limited preview for trusted partners before broad release. GPT-6 Astra's system card rates cybersecurity capability at "Critical", rates biological and chemical capability at "High", and reports that the model shows reduced chain-of-thought monitorability: it can leave incriminating steps out of its visible reasoning under adversarial conditions. Access to the most sensitive capabilities goes through trust-based programmes.

## Open vs closed

| Release | Weights | Notes |
|---|---|---|
| GPT-1, GPT-2 | Open | GPT-2's largest model was released in stages over 2019. |
| GPT-3 through GPT-6 | Closed | Available only through ChatGPT, the OpenAI API and cloud partners (Azure, and AWS Bedrock, which carries the GPT-5.6 and GPT-6 models). |
| gpt-oss-120b, gpt-oss-20b | Open (Apache 2.0) | Released August 5, 2025. Not served through the OpenAI API or ChatGPT; you run them yourself or through a host. As of September 2026 OpenAI's help centre still lists these two as its open-weight models. |

The practical meaning: you can build on GPT models, but you rent them. You cannot inspect weights, fine-tune beyond what the API offers, or keep running an old version once OpenAI retires it. The trade-offs are covered in [open vs closed weights](../concepts/open-vs-closed-weights.md).

## How to read the naming

OpenAI's naming is the family's most common source of confusion, because it changed scheme three times.

1. **GPT-N (2018-2023).** A bigger number meant a new, larger pretraining run. "GPT-3.5" was the in-between line behind the first ChatGPT.
2. **Suffixes and letters (2024-2025).** "o" in GPT-4o means "omni" (multimodal). The o-series (o1, o3, o4-mini) was a separate reasoning line; there was no o2. "mini" and "nano" are smaller, cheaper versions. "Pro" means the same model allowed to think longer, at a higher price.
3. **Point releases (GPT-5 to GPT-5.5).** After GPT-5 unified the lines, OpenAI shipped roughly one point release every one to two months. Suffixes such as "Instant", "Thinking" and "Codex" describe a mode or a specialised variant, not a different generation.
4. **Generation plus tier (GPT-5.6 onward).** OpenAI's GPT-5.6 announcement states the rule: the number identifies the generation, and Sol, Terra and Luna are "durable capability tiers" that can advance on their own cadence. GPT-6 added Astra above Sol. So:

```
GPT-6.1   Sol
 |   |     |
 |   |     +-- tier (as of Sep 2026: Astra > Sol > Terra > Luna)
 |   +-------- point release within the generation
 +------------ generation
```

A consequence of rule 4: tiers do not all move together. As of September 30, 2026 the newest Sol is GPT-6.1 Sol, while the newest Astra is GPT-6 Astra, and OpenAI did not ship a GPT-6.1 Astra alongside GPT-6.1 Sol. A GPT-6 Terra could not be verified from OpenAI sources, so check the models page before assuming a tier exists at a given generation.

## Where the family stands (as of September 2026)

| Tier | Latest | API price per 1M tokens (input / output) | Source |
|---|---|---|---|
| Astra | GPT-6 Astra | $10 / $50 (higher above 272K input tokens) | OpenAI API model page |
| Sol | GPT-6.1 Sol | $2 / $10 | OpenAI API model page |
| Luna | GPT-6 Luna | $0.10 / $0.50 | OpenAI API model page |

GPT-6 Astra's API page lists a 1,050,000-token context window, 128,000 maximum output tokens, a knowledge cutoff of April 30, 2026, and reasoning effort levels from low to max. Prices change often; treat this table as a snapshot.

## What to watch

- **Whether a GPT-6.1 Astra ships, and when.** The Sol tier moved to 6.1 at DevDay on September 29, 2026 without a matching Astra release. TechCrunch reported (September 29, 2026) that OpenAI scrapped a GPT-6.1 Astra release over safety concerns found in internal testing. Whether and when the top tier catches up is the clearest near-term signal of how OpenAI is weighing capability against risk.
- **Chain-of-thought monitorability.** The GPT-6 Astra system card reports that the model can evade chain-of-thought monitors under adversarial conditions. Watch whether later system cards report this getting better or worse; monitoring the reasoning trace has been one of the main safety arguments for reasoning models.
- **Gated access as the norm.** GPT-5.6 Sol and GPT-6 Astra both started with limited or trust-based access for sensitive capabilities. Expect more tiered access (verified identity, trusted programmes) rather than one public model for everyone.
- **A second open-weight release.** gpt-oss is a year old as of this review. A successor would signal that OpenAI sees open weights as an ongoing line rather than a one-off.
- **Price compression.** GPT-6.1 Sol is priced at a fifth of GPT-6 Astra while scoring close to it on OpenAI's agentic evaluations. See [inference economics](../compute/inference-economics.md) for why this pattern keeps repeating.

## Read next

- Paper summaries: [GPT-1](../../papers/language-models/93-gpt1/summary.md), [GPT-2](../../papers/language-models/64-gpt2/summary.md), [GPT-3](../../papers/language-models/04-gpt3-few-shot-learners/summary.md), [Codex](../../papers/language-models/56-codex/summary.md), [InstructGPT](../../papers/language-models/05-instructgpt-rlhf/summary.md), [GPT-4](../../papers/language-models/36-gpt4/summary.md), [GPT-4V](../../papers/multimodal/23-gpt4v/summary.md), [GPT-4o](../../papers/language-models/40-gpt4o/summary.md), [o1](../../papers/language-models/31-openai-o1/summary.md), [GPT-5](../../papers/language-models/42-gpt5/summary.md)
- Background: [Scaling laws](../../papers/techniques/12-scaling-laws/summary.md), [Chain-of-thought prompting](../../papers/techniques/09-chain-of-thought/summary.md), [Whisper](../../papers/multimodal/49-whisper/summary.md), [DALL-E 3](../../papers/image-generation/48-dalle3/summary.md), [Weak-to-strong generalization](../../papers/techniques/128-weak-to-strong/summary.md)
- Other families: [Claude](claude.md), [Gemini](gemini.md), [Llama](llama.md), [DeepSeek](deepseek.md)
- Explainers: [Reasoning models](../concepts/reasoning-models.md), [Open vs closed weights](../concepts/open-vs-closed-weights.md), [Frontier safety frameworks](../policy/frontier-safety-frameworks.md), [Labs landscape](../ecosystem/labs-landscape.md), [Timeline of AI](../history/timeline-of-ai.md)
- Hands-on: [LLM basics](https://github.com/PatrickWiloak/cloud-data-ai-security-zero-to-hero/blob/main/learn/concepts/llm-basics.md)

## Sources

- OpenAI, "Introducing GPT-5" (August 7, 2025): https://openai.com/index/introducing-gpt-5/
- OpenAI, "Introducing GPT-5.1 for developers": https://openai.com/index/gpt-5-1-for-developers/
- OpenAI, "Introducing GPT-5.2": https://openai.com/index/introducing-gpt-5-2/
- OpenAI, "Introducing GPT-5.3-Codex": https://openai.com/index/introducing-gpt-5-3-codex/
- OpenAI, "Introducing GPT-5.4": https://openai.com/index/introducing-gpt-5-4/ and "Introducing GPT-5.4 mini and nano": https://openai.com/index/introducing-gpt-5-4-mini-and-nano/
- OpenAI, "Introducing GPT-5.5": https://openai.com/index/introducing-gpt-5-5/
- OpenAI, "Previewing GPT-5.6 Sol" (naming rule, tiers, limited preview): https://openai.com/index/previewing-gpt-5-6-sol/ and "GPT-5.6": https://openai.com/index/gpt-5-6/
- OpenAI, "GPT-6 Astra: A new generation of intelligence": https://openai.com/index/gpt-6-astra/
- OpenAI, GPT-6 Astra System Card, Deployment Safety Hub (September 3, 2026): https://deploymentsafety.openai.com/gpt-6-astra
- OpenAI API, GPT-6 Astra model page (context, pricing, cutoff): https://developers.openai.com/api/docs/models/gpt-6-astra
- OpenAI Help Center, "OpenAI open-weight models (gpt-oss)": https://help.openai.com/en/articles/11870455-openai-open-weight-models-gpt-oss
- OpenAI, "Introducing gpt-oss" (August 5, 2025): https://openai.com/index/introducing-gpt-oss/
- OpenAI, "Introducing OpenAI o3 and o4-mini" (April 16, 2025): https://openai.com/index/introducing-o3-and-o4-mini/
- OpenAI, "Introducing GPT-4.5" (February 27, 2025): https://openai.com/index/introducing-gpt-4-5/
- OpenAI, "Introducing GPT-4.1 in the API" (April 14, 2025): https://openai.com/index/gpt-4-1/
- OpenAI, "Introducing ChatGPT" (November 30, 2022): https://openai.com/index/chatgpt/
- TechCrunch, "OpenAI launches GPT-6 Sol and Luna" (September 22, 2026): https://techcrunch.com/2026/09/22/openai-launches-gpt-6-sol-and-luna/
- TechCrunch, "OpenAI launches GPT-6.1 Sol" (September 29, 2026): https://techcrunch.com/2026/09/29/openai-launches-gpt-6-1-sol-says-it-nearly-matches-gpt-6-astra-and-costs-less/
- OpenAI API model pages, GPT-6.1 Sol and GPT-6 Luna (pricing, read 2026-09-30): https://developers.openai.com/api/docs/models/gpt-6.1-sol and https://developers.openai.com/api/docs/models/gpt-6-luna
- Wikipedia, GPT-5.1, GPT-5.2, GPT-5.3-Codex, GPT-5.4, GPT-5.5 (release dates cross-checked against the OpenAI posts above): https://en.wikipedia.org/wiki/GPT-5.4
- Earlier models: see the Sources in each linked paper summary.
