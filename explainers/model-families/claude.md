# The Claude Family (Anthropic)

**In one line:** Claude is a closed-weight family sold in size tiers (Haiku, Sonnet, Opus, and since 2026 Fable and Mythos above them), shaped by a written constitution and a scaling policy that gates each release on safety testing - read a name as "tier, then generation".
**Last reviewed:** 2026-09-30

---

## The short version

- **Tiers, not one model.** Since Claude 3 (March 2024) every generation comes in sizes: Haiku (small, fast), Sonnet (balanced), Opus (largest of the general tiers). In 2026 Anthropic added a "Mythos-class" tier above Opus, sold as **Fable** (public, with safeguards) and **Mythos** (restricted access, fewer safeguards).
- **Alignment is written down.** Claude is trained against an explicit set of principles, first described in the [Constitutional AI](../../papers/language-models/14-constitutional-ai/summary.md) paper (December 2022), rather than only against human ratings.
- **Releases are gated by a scaling policy.** Anthropic's Responsible Scaling Policy ties each capability level to required safeguards. Claude Opus 4 (May 2025) was the first model deployed under the stricter ASL-3 protections, and the 2026 Mythos-class models ship behind classifiers and trusted-access programmes.
- **Agents and code became the centre of gravity.** Computer use (October 2024), extended thinking (February 2025), Claude Code, and the Model Context Protocol pushed the family toward long-running agentic work. Most 2026 launch posts lead with coding and agent benchmarks.
- **No open weights.** No Claude model has had its weights released, as of September 2026.

## Release timeline

"Summary" links go to paper summaries in this repo where one exists. Minor snapshots and regional launches are left out.

| Model | Date | What changed | Summary |
|---|---|---|---|
| Constitutional AI (paper) | December 2022 | Harmlessness trained from AI feedback against a written list of principles. | [14](../../papers/language-models/14-constitutional-ai/summary.md) |
| Claude | March 14, 2023 | First release, limited access; plus a faster Claude Instant. | - |
| Claude 2 | July 11, 2023 | First broadly available Claude; 100K-token context. | - |
| Claude 2.1 | November 21, 2023 | 200K-token context; tool use in beta; fewer hallucinations. | - |
| Claude 3 (Haiku, Sonnet, Opus) | March 4, 2024 (Haiku March 13) | The three-tier structure; image input across the family. | - |
| Claude 3.5 Sonnet | June 20, 2024 | Mid tier beats the previous top tier; Artifacts in the app. | [30](../../papers/language-models/30-claude-3.5-sonnet/summary.md) |
| Claude 3.5 Sonnet (new), 3.5 Haiku | October 22, 2024 | Computer use in beta: the model operates a desktop through screenshots and mouse and keyboard actions. | [30](../../papers/language-models/30-claude-3.5-sonnet/summary.md) |
| Claude 3.7 Sonnet | February 24, 2025 | First "hybrid" reasoning model: one model with an optional extended-thinking mode; Claude Code preview. | - |
| Claude Opus 4, Sonnet 4 | May 22, 2025 | Naming flips to tier-first; Opus 4 deployed under ASL-3 safeguards. | [43](../../papers/language-models/43-claude4/summary.md) |
| Claude Opus 4.1 | August 5, 2025 | Incremental coding and agent gains. | [43](../../papers/language-models/43-claude4/summary.md) |
| Claude Sonnet 4.5 | September 29, 2025 | Coding and long-running agent focus. | [43](../../papers/language-models/43-claude4/summary.md) |
| Claude Haiku 4.5 | October 15, 2025 | Small tier at roughly Sonnet 4 coding level, a third of the cost. Still the current Haiku as of September 2026. | - |
| Claude Opus 4.5 | November 24, 2025 | Opus price cut to $5 / $25 per million tokens. | [43](../../papers/language-models/43-claude4/summary.md) |
| Claude Opus 4.6 | February 5, 2026 | Adaptive thinking (model decides how much to think), effort controls, context compaction, agent teams in Claude Code. | [43](../../papers/language-models/43-claude4/summary.md) |
| Claude Sonnet 4.6 | February 17, 2026 | Sonnet approaching Opus-level work at Sonnet prices. | - |
| Claude Mythos Preview | April 7, 2026 | Unreleased frontier model offered only to Project Glasswing partners for defensive security work. | - |
| Claude Opus 4.7 | April 16, 2026 | Harder software tasks; higher-resolution vision; new tokenizer; `xhigh` effort level; automated cyber-misuse safeguards. | - |
| Claude Opus 4.8 | May 28, 2026 | Effort control in the app; large parallel-subagent workflows in Claude Code; cheaper fast mode. | - |
| Claude Fable 5 / Mythos 5 | June 9, 2026 | Same underlying Mythos-class model sold two ways (see below). Access suspended June 12 to July 1 under US export controls. | - |
| Claude Sonnet 5 | June 30, 2026 | Close to Opus 4.8 performance at Sonnet pricing ($2 / $10). | - |
| Claude Opus 5 | July 24, 2026 | Long-running agentic coding; Anthropic's "most aligned model to date" on its automated behavioural audit. | - |
| Claude Fable 5.1 / Mythos 5.1 | September 1, 2026 | Roughly 25% cheaper on typical workloads (up to 45% on agentic tasks). | - |
| Claude Opus 5.5 | September 22, 2026 | About Fable 5.1 level on most work, at about 40% lower running cost than Opus 5 on typical workloads (20% lower per-token price); thinking always on. | - |
| Claude Sonnet 5.5 | September 28, 2026 | 30%+ faster output than Sonnet 5 at the same token price. | - |

## The through-line

### 1. Principles first, then preferences

Most labs aligned their chat models with RLHF: people rate answers and the model is tuned toward the preferred ones (see [InstructGPT](../../papers/language-models/05-instructgpt-rlhf/summary.md)). Anthropic's distinctive move was [Constitutional AI](../../papers/language-models/14-constitutional-ai/summary.md): write the principles down, have the model critique and revise its own answers against them, and train on AI-generated preference labels. This makes the target behaviour inspectable (you can read the constitution) and cheaper to scale than human labelling alone. Anthropic's safety research, such as [sleeper agents](../../papers/techniques/83-sleeper-agents/summary.md) and [sparse autoencoders](../../papers/techniques/82-sparse-autoencoders/summary.md) for interpretability, feeds into the system cards that accompany each release.

### 2. Capability levels decide how a model ships

The Responsible Scaling Policy (RSP) defines AI Safety Levels (ASLs), modelled loosely on biosafety levels: the more dangerous a model's capabilities, the stronger the security and deployment safeguards required before release. Opus 4 in May 2025 was the first model Anthropic deployed with ASL-3 protections. The 2026 releases show how this has turned into product structure:

```
                  same underlying Mythos-class model
                    /                             \
        Claude Fable 5 / 5.1                Claude Mythos 5 / 5.1
        generally available                 trusted-access programmes only
        classifiers watch for cyber,        (defensive security, life sciences)
        bio/chem and distillation requests  fewer safeguards in those areas
        -> flagged requests fall back to
           an Opus model instead of refusing
```

Anthropic's Fable 5 announcement says the classifiers trigger in under 5% of sessions on average. Mythos Preview, the April 2026 predecessor, was never generally available; it went to Project Glasswing, a group of launch partners including AWS, Apple, Google, Microsoft, NVIDIA and the Linux Foundation, to find and fix vulnerabilities in critical software.

Anthropic's CEO argued in a September 12, 2026 essay, "We Must Pace the Frontier", for embedded third-party evaluators, targeted US regulation and coordination between frontier labs in democracies to slow the riskiest capability gains. Anthropic describes Opus 5.5 as its first release after that call. See [frontier safety frameworks](../policy/frontier-safety-frameworks.md) for how this compares with OpenAI's and Google's frameworks.

### 3. From chat to agents

The capability story since late 2024 is about doing, not answering:

- **Computer use** (October 2024): the model sees screenshots and issues mouse and keyboard actions. See [OSWorld](../../papers/techniques/139-osworld/summary.md) for how this is measured.
- **Extended thinking** (February 2025), replaced by **adaptive thinking** (from Opus 4.6, February 2026), where the model picks how much to reason, steered by an `effort` setting. On the current Opus and Fable models thinking is always on. Background: [reasoning models](../concepts/reasoning-models.md).
- **Tools and protocols**: Anthropic introduced the [Model Context Protocol](../../papers/techniques/59-model-context-protocol/summary.md) in November 2024 as an open standard for connecting models to tools and data. See the [agent protocols explainer](../ecosystem/agent-protocols.md) and the hands-on [MCP page](https://github.com/PatrickWiloak/cloud-data-ai-security-zero-to-hero/blob/main/learn/concepts/mcp-explained.md).
- **Long context and compaction**: context grew from 100K (Claude 2) to 200K (Claude 2.1) to 1M tokens on the current Fable, Opus and Sonnet models, with compaction letting a model summarise its own history to keep working. See [context windows](../concepts/context-windows.md).

Anthropic's essay [Building Effective Agents](../../papers/essays/115-building-effective-agents/summary.md) is the plain-language statement of the design philosophy behind this shift.

### 4. The mid tier keeps catching the top tier

A repeated pattern: a new Sonnet matches or beats the previous Opus at a fraction of the price (3.5 Sonnet over 3 Opus in 2024; Sonnet 4.6 preferred by developers to Opus 4.5; Sonnet 5 close to Opus 4.8). Opus 5.5 continues the pattern one level up, performing at about Fable 5.1 level for less. Prices for the top of the general line have fallen too: Opus went from $15 / $75 per million tokens to $5 / $25 with Opus 4.5, and to $4 / $20 with Opus 5.5.

## Open vs closed

Claude is closed-weight. You reach it through claude.ai, the Claude API, Amazon Bedrock, Google Cloud Vertex AI and Microsoft Foundry. Anthropic publishes system cards, a transparency hub with training-data cutoffs, and interpretability research, but not weights, parameter counts or training-data details. What the lab does publish, and what it does not, is compared across labs in [open vs closed weights](../concepts/open-vs-closed-weights.md).

## How to read the naming

```
Claude  Opus  5.5          API id: claude-opus-5-5
         |     |
         |     +-- generation (point releases move one tier at a time)
         +-------- tier: Haiku < Sonnet < Opus < Fable/Mythos
```

- **Order flipped in 2025.** Up to 3.7 the number came first ("Claude 3.5 Sonnet"). From Claude 4 the tier comes first ("Claude Sonnet 4"). Both orders refer to the same scheme.
- **Tiers move independently.** As of September 30, 2026 the current lineup is Fable 5.1, Opus 5.5, Sonnet 5.5 and Haiku 4.5. There is no Haiku 5 yet, and no Opus or Sonnet "5.1".
- **Fable and Mythos are one model, two access levels.** They share a version number and a price ($10 / $50 per million tokens for 5.1), and differ in who can use them and which safeguards apply.
- **Model IDs are pinned snapshots.** Before the 4.6 generation, IDs carried a date (for example `claude-haiku-4-5-20251001`) and a dateless alias pointed at it. From 4.6 onward the dateless ID is itself the pinned snapshot.

## Where the family stands (as of September 2026)

| Model | Price per 1M tokens (input / output) | Context | Max output | Default effort |
|---|---|---|---|---|
| Claude Fable 5.1 | $10 / $50 | 1M | 128K | high |
| Claude Opus 5.5 | $4 / $20 | 1M | 128K | medium |
| Claude Sonnet 5.5 | $2 / $10 | 1M | 128K | high |
| Claude Haiku 4.5 | $1 / $5 | 200K | 64K | not supported |

Source: Anthropic's models overview page. Anthropic's own guidance at this date is to start with Opus 5.5 and move to Fable 5.1 for the hardest reasoning and long-horizon agent work. The three 5.x models have a reliable knowledge cutoff of June 2026.

## What to watch

- **What "pacing the frontier" means in practice.** Anthropic has publicly argued for slowing the riskiest capability gains. Watch whether the gap between Fable/Mythos releases lengthens, whether embedded third-party evaluators appear in system cards, and whether other labs follow.
- **Government access controls.** Fable 5 and Mythos 5 were suspended for all users from June 12 to July 1, 2026 after the US government applied export controls. Future Mythos-class releases may face similar controls, and nationality or identity verification could become part of access.
- **A Haiku 5.** Haiku 4.5's retirement commitment runs only to "not sooner than October 15, 2026", earlier than the other current models. A new small model is the obvious gap.
- **The fallback pattern.** Routing flagged requests to a less capable model instead of refusing is a new safeguard design. Watch whether it holds up against bypass attempts; the June 2026 suspension followed a reported bypass technique.
- **Whether Opus keeps absorbing the top tier.** If Opus 5.5 at $4 / $20 matches Fable 5.1 for most work, the case for a separate public Fable tier rests on the hardest tasks only.

## Read next

- Paper summaries: [Constitutional AI](../../papers/language-models/14-constitutional-ai/summary.md), [Claude 3.5 Sonnet and computer use](../../papers/language-models/30-claude-3.5-sonnet/summary.md), [Claude 4 family](../../papers/language-models/43-claude4/summary.md), [Model Context Protocol](../../papers/techniques/59-model-context-protocol/summary.md), [Sleeper agents](../../papers/techniques/83-sleeper-agents/summary.md), [Sparse autoencoders](../../papers/techniques/82-sparse-autoencoders/summary.md), [Scaling laws](../../papers/techniques/12-scaling-laws/summary.md)
- Essays: [Machines of Loving Grace](../../papers/essays/114-machines-of-loving-grace/summary.md), [Building Effective Agents](../../papers/essays/115-building-effective-agents/summary.md)
- Other families: [GPT and o-series](gpt.md), [Gemini](gemini.md), [Llama](llama.md)
- Explainers: [Frontier safety frameworks](../policy/frontier-safety-frameworks.md), [Agents and computer use benchmarks](../benchmarks/agents-and-computer-use.md), [Reasoning models](../concepts/reasoning-models.md), [Labs landscape](../ecosystem/labs-landscape.md)
- Hands-on: [Prompt caching](https://github.com/PatrickWiloak/cloud-data-ai-security-zero-to-hero/blob/main/learn/concepts/prompt-caching.md), [Tool use and function calling](https://github.com/PatrickWiloak/cloud-data-ai-security-zero-to-hero/blob/main/learn/concepts/tool-use-and-function-calling.md), [Guardrails and safety](https://github.com/PatrickWiloak/cloud-data-ai-security-zero-to-hero/blob/main/learn/concepts/guardrails-and-safety.md)

## Sources

- Anthropic, Models overview (current lineup, prices, context, IDs, cutoffs, retirement dates), retrieved 2026-09-30: https://platform.claude.com/docs/en/about-claude/models/overview
- Anthropic, "Introducing Claude Sonnet 5.5" (September 28, 2026): https://www.anthropic.com/claude-sonnet-5-5
- Anthropic, "Introducing Claude Opus 5.5" (September 22, 2026): https://www.anthropic.com/claude-opus-5-5
- Anthropic, "Introducing Claude Fable 5.1 and Claude Mythos 5.1" (September 1, 2026): https://www.anthropic.com/claude-fable-and-mythos-5-1
- Anthropic, "Introducing Claude Opus 5" (July 24, 2026): https://www.anthropic.com/news/claude-opus-5
- Anthropic, "Introducing Claude Sonnet 5" (June 30, 2026): https://www.anthropic.com/news/claude-sonnet-5
- Anthropic, "Claude Fable 5 and Claude Mythos 5" (June 9, 2026): https://www.anthropic.com/news/claude-fable-5-mythos-5
- Anthropic, "Redeploying Claude Fable 5" (export-control suspension and restoration): https://www.anthropic.com/news/redeploying-fable-5
- Anthropic, "Introducing Claude Opus 4.8" (May 28, 2026): https://www.anthropic.com/news/claude-opus-4-8
- Anthropic, "Introducing Claude Opus 4.7" (April 16, 2026): https://www.anthropic.com/news/claude-opus-4-7
- Anthropic, "Project Glasswing" (April 7, 2026): https://www.anthropic.com/glasswing
- Anthropic, "Introducing Sonnet 4.6": https://www.anthropic.com/news/claude-sonnet-4-6 and Claude Sonnet 4.6 System Card (February 17, 2026)
- Anthropic, "Claude Opus 4.6": https://www.anthropic.com/news/claude-opus-4-6
- Anthropic, "Introducing Claude Opus 4.5": https://www.anthropic.com/news/claude-opus-4-5
- Anthropic, "Introducing Claude Haiku 4.5": https://www.anthropic.com/news/claude-haiku-4-5
- Anthropic, "Introducing Claude 4": https://www.anthropic.com/news/claude-4 and "Activating AI Safety Level 3 protections": https://www.anthropic.com/news/activating-asl3-protections
- Anthropic, "Claude 3.7 Sonnet and Claude Code": https://www.anthropic.com/news/claude-3-7-sonnet
- Anthropic, "Introducing computer use, a new Claude 3.5 Sonnet, and Claude 3.5 Haiku": https://www.anthropic.com/news/3-5-models-and-computer-use
- Anthropic, "Introducing the next generation of Claude" (Claude 3): https://www.anthropic.com/news/claude-3-family
- Anthropic, "Introducing Claude 2.1": https://www.anthropic.com/news/claude-2-1 ; "Claude 2": https://www.anthropic.com/news/claude-2 ; "Introducing Claude": https://www.anthropic.com/news/introducing-claude
- Anthropic, Responsible Scaling Policy: https://www.anthropic.com/responsible-scaling-policy
- Dario Amodei, "We Must Pace the Frontier" (September 12, 2026): https://darioamodei.com/post/we-must-pace-the-frontier
