# Frontier Safety Frameworks

**In one line:** The big labs publish voluntary "if-then" policies - if a model reaches a dangerous capability level, then specific safeguards must be in place before it is deployed - and the live debates are whether those policies bind anyone, and whether they are being weakened as competition heats up.
**Last reviewed:** 2026-09-30

---

## The short version

- **The core idea is an if-then commitment.** Define capability thresholds in advance (for example, "meaningfully helps a novice make a biological weapon"), test each new model against them, and require stronger safeguards once a threshold is reached.
- **Three flagship frameworks:** Anthropic's Responsible Scaling Policy (RSP, with "AI Safety Levels"), OpenAI's Preparedness Framework, and Google DeepMind's Frontier Safety Framework. Most other frontier developers published something similar after the May 2024 Seoul summit.
- **They have been triggered.** Anthropic activated its ASL-3 protections for Claude Opus 4 in May 2025. In September 2026 OpenAI said a new model was the first it had classified at its "Critical" level for cybersecurity.
- **They are voluntary and self-assessed**, though California's SB 53 (2026) and New York's RAISE Act (2027) now require large developers to publish a framework and follow it, and the EU's Code of Practice builds on the same structure.
- **The frameworks keep changing.** Anthropic's RSP v3.0 (February 2026) dropped its earlier commitment to pause if safeguards were not ready; OpenAI's framework lets it adjust safeguards if a competitor ships without them. Supporters call this honesty about what one company can do alone; critics call it erosion.

## The mental model: tripwires tied to safeguards

```
  capability evals on each new model
               |
               v
   +-----------------------+      below threshold
   | crosses a threshold?  |------------------------->  deploy with baseline safeguards
   +-----------------------+
               | yes
               v
   required safeguards in place?
     - deployment safeguards (misuse filters, access controls, monitoring)
     - security safeguards (protect the weights from theft)
               |
       yes     |      no
     deploy <--+--> (older policies: pause / do not deploy)
                    (newer policies: mitigate, disclose, justify publicly)
```

Two families of risk dominate every framework:

- **Misuse:** a person uses the model to cause mass harm - chemical and biological weapons, large-scale cyberattacks.
- **Loss of control / autonomy:** the model itself acts against its developers' intent, or AI accelerates AI research so fast that oversight cannot keep up.

Frameworks differ mainly in which risks they track, where they set thresholds, what safeguards each level requires, and how binding the response is.

## The Seoul commitments (May 2024)

At the AI Seoul Summit on 21 May 2024, the UK and South Korean governments announced the **Frontier AI Safety Commitments**, signed by 16 companies including Amazon, Anthropic, Google, Meta, Microsoft, Mistral AI, OpenAI, xAI and Zhipu.ai. Four more (including NVIDIA and MiniMax) joined later. Signatories agreed to:

- assess risks across the model lifecycle, including before deployment and with external evaluators where appropriate;
- set thresholds beyond which risks would be "intolerable" unless mitigated;
- commit **not to develop or deploy a model at all "if mitigations cannot be applied to keep risks below the thresholds"**;
- publish a safety framework before the next summit (France, February 2025).

The commitments are voluntary and have no enforcement mechanism. They matter because they turned a practice pioneered by a few labs into an industry expectation, and because later laws (SB 53, RAISE, the EU Code of Practice) borrowed the "publish a framework and follow it" structure.

## Anthropic: Responsible Scaling Policy

**Structure.** First published in September 2023. It borrows from biosafety levels: **AI Safety Levels (ASL)** pair a level of capability with required safeguards. ASL-2 is the baseline for current models; ASL-3 adds stronger protections against misuse for chemical and biological weapons and stronger security against weight theft by non-state attackers; higher levels were sketched for more capable systems.

**First activation.** On 22 May 2025, Anthropic deployed Claude Opus 4 under the ASL-3 Deployment and Security Standards. It said it had not determined that the model definitely required ASL-3, but could not clearly rule it out, and chose to activate the protections as a precaution. See the [Claude 4 summary](../../papers/language-models/43-claude4/summary.md).

**RSP v3.0 (effective 24 February 2026).** A major rewrite. The main changes, per Anthropic and an analysis by the Centre for the Governance of AI (GovAI):

- **Split between what Anthropic will do alone and what it thinks the industry should do.** Some previous commitments became "industry-wide recommendations" rather than unilateral pledges. Anthropic says it will keep ASL-3 protections for chemical and biological risks regardless of what others do.
- **The pause language was removed.** Earlier versions committed not to train or deploy models capable of catastrophic harm unless safeguards kept risk acceptable. Anthropic said pre-set capability levels had proved "far more ambiguous than we anticipated".
- **A Frontier Safety Roadmap** of public, "nonbinding" goals across security, alignment, safeguards and policy.
- **Risk Reports** on deployed and internal models every few months, with **external third-party review** for highly capable models.
- Radiological, nuclear and cyber operations categories were removed from the threshold list; the automated AI R&D threshold was redefined.

Further point releases followed; as of September 2026 the current version is 3.4, effective 8 July 2026.

## OpenAI: Preparedness Framework

**Structure.** First released as a beta in December 2023; Version 2 was published on 15 April 2025 and is the current version as of September 2026.

- **Tracked Categories** (measured on every covered model): Biological and Chemical, Cybersecurity, and AI Self-improvement.
- **Research Categories** (not yet tracked, being studied): including Long-range Autonomy, Sandbagging (deliberately underperforming on tests), Autonomous Replication and Adaptation, Undermining Safeguards, and Nuclear and Radiological.
- **Two thresholds.** At **High**, a model is not deployed until its risks are "sufficiently minimized". At **Critical**, safeguards are also required *during development*, whatever the deployment plans.
- **Governance.** An internal Safety Advisory Group reviews evidence and recommends; OpenAI leadership decides; the board's Safety and Security Committee oversees.
- **Persuasion was dropped** as a tracked category in v2. OpenAI said persuasion risks do not fit its definition of severe harm and are better addressed through usage policies, content provenance work and society-level measures.
- **The marginal-risk clause.** If another developer releases a High or Critical system without comparable safeguards, OpenAI "could adjust accordingly the level of safeguards that we require", but only if doing so does not meaningfully increase overall risk, it says so publicly, and it keeps its safeguards more protective than the other developer's.

**First Critical classification.** In early September 2026 OpenAI said its upcoming model Astra was the first it had classified at the Critical cybersecurity level: able to find and exploit previously unknown vulnerabilities in hardened systems with limited human involvement. OpenAI said it would not make the model's full cyber capabilities widely available at launch - advanced cyber capabilities go only to members of its Daybreak coalition - and that its safeguards "sufficiently minimize the risk of severe harm for release". CNBC reported that OpenAI had delayed parts of Astra's development after an August security incident at Hugging Face that did not involve Astra.

## Google DeepMind: Frontier Safety Framework

**Structure.** Version 1.0 in May 2024, 2.0 in February 2025, 3.0 on 22 September 2025, and 3.1 on 17 April 2026.

- **Critical Capability Levels (CCLs)** across misuse (CBRN, cyber, and since v3, **harmful manipulation** - models that could systematically and substantially change beliefs and behaviours in high-stakes contexts), machine-learning R&D, and **misalignment** (including models that might resist operators' attempts to modify or shut them down).
- **Safety case reviews** before external launches and, since v3, before large-scale internal deployments of models with advanced ML research capabilities.
- **Tracked Capability Levels (TCLs)**, new in v3.1, flag rising risk earlier, before a CCL is reached.
- **Security levels** for protecting weights, which v3.1 (April 2026) revised alongside its new tracked capability levels; check the framework text for the current security-level definitions.

## How they compare

| | Anthropic RSP | OpenAI Preparedness | Google DeepMind FSF |
|---|---|---|---|
| Current version (Sep 2026) | 3.4 (Jul 2026) | 2 (Apr 2025) | 3.1 (Apr 2026) |
| Threshold vocabulary | ASL levels; capability thresholds | High / Critical | CCLs, plus TCLs |
| Misuse areas | Chem/bio | Bio/chem, cyber | CBRN, cyber, harmful manipulation |
| Autonomy / AI R&D | Automated AI R&D threshold | AI self-improvement | ML R&D, misalignment / shutdown resistance |
| Explicit pause language | Removed in v3.0 | Critical requires safeguards in development | Safety case review before deployment |
| Competitor clause | Commits to match competitors' more effective mitigations; industry recommendations separated from unilateral commitments | Marginal-risk adjustment, with conditions | - |
| External review | Third-party review of Risk Reports for highly capable models | Internal Safety Advisory Group | Internal safety case review |

## The criticisms, and the replies

**"They are marketing, not constraints."** Critics note that each lab writes its own thresholds, runs its own tests, judges the results, and can rewrite the policy. A 2025 study scoring 12 companies' frameworks against 65 risk-management criteria drawn from aviation and nuclear safety found scores from 8% to 34% (median 18%) and called the commitments vague as accountability tools. An analysis of OpenAI's framework argued its wording "does not guarantee any AI risk mitigation practices".
*Reply:* transparency is the point. Published thresholds and model cards let outsiders check claims, and laws like SB 53 now make following one's own framework legally enforceable in California.

**"The commitments weaken exactly when they would bite."** Anthropic's removal of pause language and OpenAI's competitor clause and removal of persuasion are cited as evidence that commercial pressure wins, and that a pledge which can be rewritten is not a pledge.
*Reply:* Anthropic and GovAI argue that keeping commitments a company expects to break creates worse incentives, such as reluctance to admit a threshold has been crossed; honest, transparent, conditional policies may do more good than rigid ones nobody keeps. GovAI wrote that its "initial reaction to the update was rather negative" but that after closer engagement its "overall view became more positive" about RSP v3.0, conditional on faithful implementation.

**"Evals can't reliably detect what matters."** Capability evaluations can underestimate a model (poor elicitation, sandbagging), and thresholds like "significant uplift to a novice" are hard to operationalise. Work such as [Sleeper Agents](../../papers/techniques/83-sleeper-agents/summary.md) shows deceptive behaviour can survive standard safety training.
*Reply:* labs have moved toward conservative, precautionary activation (Anthropic's ASL-3 and OpenAI's Critical classification were both made under uncertainty), automated [red teaming](../../papers/techniques/129-red-teaming-lms/summary.md), and "early warning" levels like DeepMind's TCLs.

**"They cover the wrong risks."** Some argue the frameworks focus on catastrophic, speculative scenarios while ignoring present harms (bias, misinformation, labour effects). Others argue the opposite: that alignment of systems smarter than their overseers, the subject of [Weak-to-Strong Generalization](../../papers/techniques/128-weak-to-strong/summary.md), is barely addressed.
*Reply:* the frameworks are explicitly scoped to severe, large-scale harms; other policies (usage policies, model specs such as [Constitutional AI](../../papers/language-models/14-constitutional-ai/summary.md), classifiers like [Llama Guard](../../papers/language-models/96-llama-guard/summary.md)) cover everyday harms.

## What to watch

- **Risk Reports and external reviews** under Anthropic's RSP v3, and whether reviewers publicly disagree with Anthropic's conclusions.
- **Astra's release** and what OpenAI publishes about the safeguards that justified deploying a Critical-level cyber model.
- **Whether OpenAI publishes a Version 3** of its framework now that a model has reached the Critical cyber level.
- **SB 53 enforcement** and the first RAISE Act filings after 1 January 2027: the first time a regulator can act on a lab not following its own framework.
- **EU Code of Practice alignment:** the Safety and Security chapter asks systemic-risk model providers for a similar framework; enforcement powers began 2 August 2026.

## Read next

- [US AI policy](./us-ai-policy.md) - SB 53 and the RAISE Act, which make these frameworks legally relevant
- [The EU AI Act](./eu-ai-act.md) - the Code of Practice's Safety and Security chapter
- [Sleeper Agents](../../papers/techniques/83-sleeper-agents/summary.md), [Red Teaming Language Models with Language Models](../../papers/techniques/129-red-teaming-lms/summary.md), [Weak-to-Strong Generalization](../../papers/techniques/128-weak-to-strong/summary.md)
- [Constitutional AI](../../papers/language-models/14-constitutional-ai/summary.md), [Llama Guard](../../papers/language-models/96-llama-guard/summary.md)
- [Evals for LLMs](https://github.com/PatrickWiloak/cloud-data-ai-security-zero-to-hero/blob/main/learn/concepts/evals-for-llms.md) and [AI threat modeling](https://github.com/PatrickWiloak/cloud-data-ai-security-zero-to-hero/blob/main/learn/concepts/ai-threat-modeling.md) (sibling repo, hands-on)

## Sources

- UK Government, "Frontier AI Safety Commitments, AI Seoul Summit 2024": https://www.gov.uk/government/publications/frontier-ai-safety-commitments-ai-seoul-summit-2024/frontier-ai-safety-commitments-ai-seoul-summit-2024
- UK Government press release, 21 May 2024 (16 companies): https://www.gov.uk/government/news/historic-first-as-companies-spanning-north-america-asia-europe-and-middle-east-agree-safety-commitments-on-development-of-ai
- Anthropic, Responsible Scaling Policy (current and version history): https://www.anthropic.com/responsible-scaling-policy ; updates log: https://www.anthropic.com/rsp-updates
- Anthropic, "Responsible Scaling Policy v3" announcement (24 Feb 2026): https://anthropic.com/news/responsible-scaling-policy-v3
- Anthropic, RSP Version 3.4, effective 8 July 2026: https://www-cdn.anthropic.com/files/4zrzovbb/website/0bacdc8440ea96e62a8766d99ebe1d4eea6d5f3a.pdf
- Anthropic, "Activating AI Safety Level 3 Protections" (May 2025): https://anthropic.com/news/activating-asl3-protections
- GovAI, "Anthropic's RSP v3.0: How it Works, What's Changed, and Some Reflections": https://www.governance.ai/analysis/anthropics-rsp-v3-0-how-it-works-whats-changed-and-some-reflections
- OpenAI, Preparedness Framework Version 2 (15 Apr 2025), including Section 4.3 "Marginal risk": https://cdn.openai.com/pdf/18a02b5d-6b67-4cec-ab64-68cdfbddebcd/preparedness-framework-v2.pdf
- OpenAI, "Responding to the next frontier of critical cyber capabilities" (Sep 2026): https://openai.com/index/responding-next-frontier-critical-cyber-capabilities/ ; SecurityWeek, 2 Sep 2026: https://www.securityweek.com/openais-astra-becomes-first-model-to-cross-critical-cybersecurity-threshold/ ; CNBC, 1 Sep 2026: https://www.cnbc.com/2026/09/01/open-ai-astra-cyber-model.html
- Google DeepMind, "Strengthening our Frontier Safety Framework" (v3.0 22 Sep 2025; v3.1 17 Apr 2026): https://deepmind.google/blog/strengthening-our-frontier-safety-framework/ ; original FSF (May 2024): https://deepmind.google/blog/introducing-the-frontier-safety-framework/
- Stelling et al., "Evaluating AI Providers' Frontier Safety Frameworks" (arXiv 2512.01166, Dec 2025, rev. Apr 2026): https://arxiv.org/abs/2512.01166
- Coggins et al., "The 2025 OpenAI Preparedness Framework does not guarantee any AI risk mitigation practices" (arXiv 2509.24394): https://arxiv.org/abs/2509.24394
