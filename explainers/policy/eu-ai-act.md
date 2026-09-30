# The EU AI Act

**In one line:** The EU AI Act sorts AI by risk, bans a short list of practices outright, puts heavy duties on "high-risk" uses and on the biggest general-purpose models, and phases in over 2025-2028 - with the high-risk deadlines pushed back by the 2026 "digital omnibus".
**Last reviewed:** 2026-09-30

---

## The short version

- **It regulates uses, not technology in general.** The Act (Regulation (EU) 2024/1689) sorts AI systems into four risk tiers. Most AI (spam filters, game AI, recommendation widgets) falls in the bottom tier and has no new obligations.
- **A short list of practices is banned** outright, such as social scoring and untargeted scraping of faces to build recognition databases. These bans have applied since 2 February 2025.
- **"High-risk" systems** (AI used in hiring, credit, education, critical infrastructure, border control and similar) must meet requirements on risk management, data quality, logging, documentation and human oversight.
- **General-purpose AI models** (the foundation models behind chatbots) have their own track. Duties started on 2 August 2025; the biggest models, presumed to carry "systemic risk", have extra safety duties. A voluntary Code of Practice, published 10 July 2025, is the main way providers show compliance.
- **The timeline moved.** The AI Omnibus, proposed by the Commission on 19 November 2025 and in force since 27 July 2026, delays the high-risk rules to 2 December 2027 (stand-alone uses) and 2 August 2028 (AI inside regulated products). The bans and the general-purpose model duties were not delayed.

## The mental model: a risk pyramid

The Act is built like product-safety law. Instead of asking "is this AI?", it asks "what is this AI being used for, and how badly could that go wrong?"

```
                 /\
                /  \      UNACCEPTABLE RISK - banned (Article 5)
               /----\
              /      \    HIGH RISK - allowed, with strict duties
             /--------\
            /          \  TRANSPARENCY RISK - must disclose (Article 50)
           /------------\
          /              \ MINIMAL RISK - no new obligations
         /________________\
```

Running alongside the pyramid is a separate track for **general-purpose AI (GPAI) models** - models like GPT, Claude, Gemini or Llama that can be built into many different products. A GPAI model is not itself "high-risk"; what someone builds with it might be. So the Act regulates the model maker (the "provider") for what only they can control - documentation, copyright policy, and for the largest models, safety testing - and regulates the product builder for how the model is used.

### Who is on the hook

The Act assigns duties by role:

| Role | Plain meaning | Example |
|---|---|---|
| Provider | Develops the AI system or model and places it on the EU market | A lab releasing a model; a vendor selling a CV-screening tool |
| Deployer | Uses an AI system in a professional capacity | A bank using that tool to screen job applicants |
| Importer / distributor | Brings a non-EU system into the EU market or resells it | A reseller of a US product |

It applies to non-EU companies whose systems are used in the EU, much as GDPR does.

## Tier 1: prohibited practices

Article 5 lists practices considered incompatible with EU fundamental rights. As of September 2026 the list covers:

- Subliminal, manipulative or deceptive techniques that distort behaviour and cause significant harm.
- Exploiting vulnerabilities due to age, disability or social or economic situation.
- Social scoring that leads to unjustified or disproportionate detrimental treatment.
- Predicting that a person will commit a crime based solely on profiling or personality traits.
- Building facial-recognition databases by untargeted scraping of images from the internet or CCTV.
- Inferring emotions in workplaces and schools (except for medical or safety reasons).
- Biometric categorisation to infer race, political opinions, religion, sex life or sexual orientation.
- Real-time remote biometric identification in public spaces for law enforcement, except in narrow listed cases with safeguards.
- **New from the omnibus:** generating non-consensual intimate imagery and child sexual abuse material, including so-called "nudifier" apps. This addition has a transition period ending 2 December 2026.

The maximum fine for a prohibited practice is EUR 35 million or 7% of worldwide annual turnover, whichever is higher (Article 99). For small and medium enterprises the cap is whichever is lower.

## Tier 2: high-risk systems

A system is high-risk in two ways:

1. **Annex III uses** - stand-alone systems used in listed sensitive areas: biometrics, critical infrastructure, education and vocational training, employment and worker management, access to essential private and public services (such as credit scoring), law enforcement, migration and border control, and the administration of justice and democratic processes.
2. **Annex I products** - AI that is a safety component of a product already covered by EU product-safety law (machinery, medical devices, toys, lifts and so on).

Providers of high-risk systems must run a risk-management system, use training data that meets quality standards, keep technical documentation and automatic logs, give deployers clear instructions, design for human oversight, meet accuracy and cybersecurity requirements, pass a conformity assessment, and register in an EU database. Deployers have lighter but real duties, such as using the system as instructed and assigning trained people to oversee it.

Breaching most of these obligations can cost up to EUR 15 million or 3% of turnover.

## Tier 3: transparency

Article 50 covers systems whose main risk is that people are misled:

- Chatbots must tell people they are talking to an AI, unless it is obvious.
- Providers of generative systems must mark synthetic audio, image, video and text in a machine-readable way (watermarking or similar).
- Deployers of deepfakes must disclose that the content is artificially generated.

These duties apply from 2 August 2026. The omnibus gave systems already on the market before that date a grace period until 2 December 2026 for the machine-readable marking duty in Article 50(2). The Commission published transparency guidelines on 20 July 2026.

## The general-purpose AI track

### Two layers of duties

| Layer | Who | Main duties |
|---|---|---|
| All GPAI providers | Anyone placing a general-purpose model on the EU market | Technical documentation for the AI Office and for downstream builders; a policy to comply with EU copyright law (including respecting text-and-data-mining opt-outs); a public summary of the content used for training |
| GPAI with systemic risk | Models presumed to have "high-impact capabilities" - by default, trained with more than 10^25 floating-point operations (Article 51), or designated by the Commission | All of the above, plus model evaluations including adversarial testing, assessing and mitigating systemic risks, reporting serious incidents to the AI Office, and adequate cybersecurity |

Open-source models released under a free licence get some exemptions from the documentation duties, but not if they carry systemic risk.

### Dates specific to general-purpose models

- **2 August 2025:** obligations apply to new models.
- **2 August 2026:** the Commission's enforcement powers (including fines) begin.
- **2 August 2027:** deadline for models already on the market before 2 August 2025.

Fines for GPAI providers can reach EUR 15 million or 3% of worldwide turnover (Article 101). The omnibus also gave the Commission's AI Office exclusive competence over general-purpose AI systems, with new powers to investigate, inspect on site, accept binding commitments and impose fines.

### The Code of Practice

The General-Purpose AI Code of Practice was published on 10 July 2025. It was drafted by independent experts through a multi-stakeholder process and has three chapters: **Transparency** (including a model documentation form), **Copyright**, and **Safety and Security** (only relevant to systemic-risk models). It is voluntary. Signing it is a way to demonstrate compliance; a provider that does not sign has to show compliance by other means.

As of September 2026 the Commission lists signatories including Amazon, Anthropic, Google, IBM, Microsoft, Mistral AI and OpenAI. xAI signed only the Safety and Security chapter. Meta publicly declined to sign in July 2025, with its global affairs chief saying the Code introduced legal uncertainties and went beyond the scope of the Act.

For how the labs' own voluntary safety policies compare with the Code's safety chapter, see [Frontier safety frameworks](./frontier-safety-frameworks.md).

## The timeline

The Act entered into force on 1 August 2024 and applies in stages. The table shows the timeline as amended by the AI Omnibus.

| Date | What applies |
|---|---|
| 2 Feb 2025 | Prohibited practices; AI literacy duty (Article 4) |
| 2 Aug 2025 | Governance bodies; general-purpose AI model obligations; penalties framework |
| 2 Aug 2026 | Most remaining provisions, including Article 50 transparency; Commission enforcement powers over GPAI models |
| 2 Dec 2026 | End of grace period for Article 50(2) marking (systems already on market); end of transition for the new NCII / CSAM ban |
| 2 Aug 2027 | Legacy GPAI models (on market before Aug 2025) must comply; deadline for national regulatory sandboxes (moved back a year) |
| 2 Dec 2027 | High-risk obligations for Annex III stand-alone systems (originally 2 Aug 2026) |
| 2 Aug 2028 | High-risk obligations for Annex I AI embedded in regulated products (originally 2 Aug 2027) |

## The digital omnibus: what changed in 2026

In November 2025 the Commission proposed a "digital omnibus" package to simplify several digital laws. The AI part moved fast:

| Step | Date |
|---|---|
| Commission proposal | 19 November 2025 |
| Provisional political agreement between Parliament and Council | 7 May 2026 |
| Entry into force | 27 July 2026 |

The main changes, as reported by the Commission and law firms tracking the text:

- **High-risk deadlines postponed** to 2 December 2027 (Annex III) and 2 August 2028 (Annex I). The new dates are fixed, not conditional on harmonised technical standards being ready.
- **AI literacy softened.** Article 4 originally required providers and deployers to ensure a sufficient level of AI literacy among staff. It now requires them to take measures to support it, without guaranteeing any particular level.
- **New prohibition** on AI systems that generate non-consensual intimate imagery or child sexual abuse material.
- **SME relief extended** to "small mid-caps", including simplified technical documentation and proportionate penalties, and fewer data points required for registration.
- **Bias detection.** Processing special-category personal data to detect and correct bias is permitted for all AI systems and GPAI models under strict necessity conditions, not only for high-risk providers.
- **Stronger central enforcement** through the AI Office, as described above.

### The debate around the delay

Supporters of the delay argued that the harmonised technical standards companies need to demonstrate compliance were not ready, that national enforcement authorities were not all in place, and that Europe risked falling behind on AI adoption. Fixing firm dates, they said, is more honest than enforcing rules nobody can yet show they meet.

Critics argued that the delay postpones protections for people affected by hiring, credit and public-services algorithms that are already in use, that "simplification" risks becoming deregulation under competitive pressure, and that repeated changes undermine legal certainty for companies that had already invested in compliance.

## What to watch

- **2 August 2026 onward:** the first enforcement actions by the AI Office against GPAI providers, now that fines are available.
- **Harmonised standards** from the European standardisation bodies (CEN-CENELEC) for high-risk systems, needed well before December 2027.
- **2 August 2027:** legacy GPAI models must comply; watch whether any widely used older model is withdrawn from the EU rather than documented.
- **Further revisions.** The Act contains review clauses, and the omnibus showed the Commission is willing to reopen it. Whether the 10^25 FLOP threshold is updated as training compute grows is an open question.
- **Signatory list** for the Code of Practice, and whether non-signatories face more scrutiny.

## Read next

- [US AI policy](./us-ai-policy.md) - the contrasting, largely state-led US approach
- [Frontier safety frameworks](./frontier-safety-frameworks.md) - the voluntary lab policies that the Code of Practice's safety chapter builds on
- [Open vs closed weights](../concepts/open-vs-closed-weights.md) - relevant to the Act's open-source exemptions
- [Cost of training](../compute/cost-of-training.md) - context for what a 10^25 FLOP run means
- [Llama Guard](../../papers/language-models/96-llama-guard/summary.md) - an example of the kind of safeguard tooling providers deploy
- [Guardrails and safety](https://github.com/PatrickWiloak/cloud-data-ai-security-zero-to-hero/blob/main/learn/concepts/guardrails-and-safety.md) (sibling repo, hands-on)

## Sources

- European Commission, "AI Act" regulatory framework page (risk tiers, timeline, AI Omnibus dates): https://digital-strategy.ec.europa.eu/en/policies/regulatory-framework-ai
- European Commission, "The General-Purpose AI Code of Practice" (publication date, chapters, signatories, xAI): https://digital-strategy.ec.europa.eu/en/policies/contents-code-gpai
- European Commission, "Guidelines for providers of general-purpose AI models" (2 Aug 2025 / 2 Aug 2026 / 2 Aug 2027 dates): https://digital-strategy.ec.europa.eu/en/policies/guidelines-gpai-providers
- Regulation (EU) 2024/1689 (AI Act), EUR-Lex: https://eur-lex.europa.eu/eli/reg/2024/1689/oj - consolidated article text consulted via https://artificialintelligenceact.eu/article/5/, /article/51/, /article/99/, /article/101/
- Council of the EU press release, 7 May 2026, "Artificial intelligence: Council and Parliament agree to simplify and streamline rules": https://www.consilium.europa.eu/en/press/press-releases/2026/05/07/artificial-intelligence-council-and-parliament-agree-to-simplify-and-streamline-rules/
- Gibson Dunn, "EU AI Act Omnibus Agreement - Postponed High-Risk Deadlines and Other Key Changes" (fixed dates, AI Office powers, sandboxes, bias data): https://www.gibsondunn.com/eu-ai-act-omnibus-agreement-postponed-high-risk-deadlines-and-other-key-changes/
- Sidley Austin Data Matters, 22 June 2026, "EU Lawmakers Reach Provisional Agreement to Delay Key EU AI Act Obligations": https://datamatters.sidley.com/2026/06/22/eu-lawmakers-reach-provisional-agreement-to-delay-key-eu-ai-act-obligations/
- CNBC, 18 July 2025, "Meta says it won't sign Europe AI agreement": https://www.cnbc.com/2025/07/18/meta-europe-ai-code.html
