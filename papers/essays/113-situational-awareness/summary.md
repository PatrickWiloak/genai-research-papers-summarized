---
title: "Situational Awareness: The Decade Ahead (Situational Awareness)"
slug: "113-situational-awareness"
number: 113
category: "essays"
authors: "Leopold Aschenbrenner (independent; formerly OpenAI Superalignment team)"
published: "June 2024 (self-published essay series)"
year: 2024
url: "https://situational-awareness.ai/"
tags: ["essay", "scaling", "policy"]
---

# Situational Awareness: The Decade Ahead (Situational Awareness)

**Authors:** Leopold Aschenbrenner (independent; formerly OpenAI Superalignment team)
**Published:** June 2024 (self-published essay series)
**Essay:** [situational-awareness.ai](https://situational-awareness.ai/)

---

## Why This Matters

"Situational Awareness" is the best-known attempt to turn scaling-law trend lines into a concrete, dated forecast of artificial general intelligence (AGI) and of what governments will do about it. Its author is a former member of OpenAI's Superalignment team, and it is dedicated to Ilya Sutskever. It runs to five long chapters plus an introduction and a conclusion. It appeared in June 2024, just as Washington was starting to treat frontier AI as a national-security issue.

- **It made a falsifiable bet.** "AGI by 2027 is strikingly plausible" is a sentence with a date on it, which is rare in this genre.
- **It gave people a counting method.** The "count the OOMs" framework (orders of magnitude of effective compute) is now common vocabulary in forecasting debates.
- **It reframed AI as geopolitics.** Much of the essay is not about models at all. It is about power plants, chip fabs, espionage and a US-China race.
- **It is contested.** Admirers treat it as prescient. Critics treat it as a hawkish, self-fulfilling pitch for an arms race. Both camps read it, which is why it belongs in a reading list.

This summary presents the argument on its own terms, then gives the main criticisms and checks the dated predictions that can be checked as of September 2026. It does not take a side on whether the central forecast will come true.

---

## The Problem It Addresses

By mid-2024 the public conversation about AI had split in two. Some people saw chatbots as a fad. Others, inside frontier labs, believed they were watching a smooth, predictable exponential. Aschenbrenner's opening claim is that only "perhaps a few hundred people" had what he calls *situational awareness*: an understanding of how fast things were moving and what would follow. The essay tries to hand that picture of the world to everyone else, above all to policymakers.

The intellectual background is the scaling tradition: [Scaling Laws](../../techniques/12-scaling-laws/summary.md), [Chinchilla](../../techniques/18-chinchilla/summary.md), Gwern's [Scaling Hypothesis](../112-scaling-hypothesis/summary.md) and Sutton's [Bitter Lesson](../111-bitter-lesson/summary.md). Aschenbrenner takes their premise, that more compute reliably buys more capability, and pushes it forward year by year.

---

## The Argument

The essay has a simple spine, and each chapter hangs off one link in the chain.

```
Chapter I     trend lines in compute + algorithms + "unhobbling"
                  |
                  v
              AGI (a "drop-in remote worker") around 2027
                  |
Chapter II        v
              millions of AGI copies automate AI research
                  |
                  v
              intelligence explosion -> superintelligence by ~2030
                  |
Chapter III       v
   a) trillion-dollar clusters, power, chips  (can we build it?)
   b) lab security                            (can we keep it?)
   c) superalignment                          (can we control it?)
   d) the free world must prevail             (who gets it first?)
                  |
Chapter IV        v
              "The Project": a US government AGI effort by 2027/28
```

### I. From GPT-4 to AGI: Counting the OOMs

This chapter makes the core quantitative move. An OOM is one order of magnitude, a factor of 10. Aschenbrenner argues that capability tracks *effective compute*, which grows from three sources:

```
effective compute growth per year (his estimates)

  physical compute         ~0.5 OOM/yr   (bigger clusters, more spending)
+ algorithmic efficiency   ~0.5 OOM/yr   (same result with less compute)
+ "unhobbling"             step changes  (RLHF, chain of thought, tools,
                                          agent scaffolding, long context)
```

GPT-2 to GPT-4, he writes, took us "from ~preschooler to ~smart high-schooler abilities in 4 years". He projects about another 100,000x in effective compute over the next four years. He also argues that unhobbling will turn chatbots into agents that can do weeks of work on their own, a "drop-in remote worker". Hence "AGI by 2027 is strikingly plausible."

He names the main obstacle himself: the **data wall**. Frontier models were already training on much of the useful internet (he cites Llama 3's 15 trillion tokens). He expects synthetic data, self-play and reinforcement learning to get around it, while admitting this is uncertain.

### II. From AGI to Superintelligence: the Intelligence Explosion

Once an AI can do an AI researcher's job, you can run enormous numbers of copies. He estimates that inference fleets will be large enough for the equivalent of "100 million human-equivalents" working on AI research, soon at many times human speed. On that basis he argues they could compress "a decade of algorithmic progress (5+ OOMs) into ≤1 year". The main bottleneck he accepts is compute for experiments, which he argues automated researchers would use far more efficiently. The result is superintelligence by about the end of the decade.

### IIIa. Racing to the Trillion-Dollar Cluster

This is the industrial chapter. His table of the largest single training cluster by year:

| Year | vs GPT-4 | H100-equivalents | Cost | Power |
|---|---|---|---|---|
| 2022 | GPT-4 | ~10k | ~$500M | ~10 MW |
| 2024 | +1 OOM | ~100k | billions | ~100 MW |
| 2026 | +2 OOM | ~1M | tens of billions | ~1 GW |
| 2028 | +3 OOM | ~10M | hundreds of billions | ~10 GW |
| 2030 | +4 OOM | ~100M | $1T+ | ~100 GW (over 20% of US electricity) |

He argues that electricity, not chips, will be the binding constraint, and that US natural gas could supply it. He puts total AI investment at about $1 trillion a year by 2027. He also proposes a revenue milestone: a big tech company reaching a $100B annual run rate from AI, which a naive extrapolation puts at mid-2026.

### IIIb. Lock Down the Labs

Model weights and, even more, *algorithmic secrets* are the crown jewels. He argues that frontier labs protect them at roughly "random startup" level against state actors, above all China. He calls for government-grade security: air-gapped datacenters, vetted staff, and work done in secure facilities.

### IIIc. Superalignment

Techniques like [RLHF](../../language-models/05-instructgpt-rlhf/summary.md) depend on humans judging outputs, and that fails once models are smarter than the judges. He sketches a "default plan": scalable oversight, studying how behaviour generalises, interpretability, and using early AGIs to automate alignment research. He is openly worried that an intelligence explosion leaves little time to iterate: "We're counting way too much on luck here." Aschenbrenner co-authored OpenAI's [Weak-to-Strong Generalization](../../techniques/128-weak-to-strong/summary.md) paper, one strand of that plan.

### IIId. The Free World Must Prevail

Superintelligence, he argues, will give a decisive military and economic advantage, so it matters enormously whether democracies or authoritarian states get there first. He wants the US and its allies to hold a "healthy lead", partly so there is room to slow down for safety.

### IV. The Project

This is the most provocative prediction: "By 27/28 we'll get some form of government AGI project." He compares it to the Manhattan Project and argues it is inevitable once officials see the evidence: "I find it an insane proposition that the US government will let a random SF startup develop superintelligence." He sketches an allied coalition, plus an "Atoms for Peace"-style offer of benefits to other countries.

---

## Key Claims

1. **Trend lines are the best forecast.** Effective compute has grown about 1 OOM per year, and there is no strong reason to expect that to stop before 2027.
2. **AGI around 2027 is plausible, not certain.** He calls it "strikingly plausible", not guaranteed.
3. **AGI leads quickly to superintelligence** through automated AI research.
4. **Power is the binding constraint** on the buildout, and the US has the natural gas to meet it.
5. **Security is the most neglected problem.** Leaking algorithmic secrets to China is the likeliest way to lose a lead.
6. **Aligning superhuman systems is unsolved**, and the timeline may not leave room for trial and error.
7. **The US government will take over** frontier development in some form by 2027-2028.

---

## How It Has Aged (as of September 2026)

Only items with a public source are listed. The central forecast, AGI around 2027, cannot be scored yet.

**Broadly on track or confirmed**

- **The compute and investment trend.** A June 2025 one-year audit on LessWrong (Nathan Delisle) found the roughly half-OOM-per-year compute trend "roughly supported". Capital spending, accelerator shipments and committed power met or beat his curves. Revenue was the weakest metric.
- **The gigawatt cluster in 2026.** His table put a ~1 GW cluster in 2026. xAI said its Colossus 2 training cluster was running at about 1 GW in January 2026, and other gigawatt-scale sites, including OpenAI's Stargate site in Abilene, were scheduled for 2026.
- **Private megaprojects.** The Stargate joint venture, announced in January 2025 with up to $500 billion of planned investment over four years, is the kind of spending he described. It is private, though, not governmental.
- **Unhobbling via reasoning.** Test-time reasoning, which he listed as an unhobbling gain, became the main driver of capability after [OpenAI o1](../../language-models/31-openai-o1/summary.md) (September 2024) and [DeepSeek-R1](../../language-models/26-deepseek-r1/summary.md) (January 2025). See also [test-time compute](../../techniques/50-test-time-compute/summary.md).
- **Power as a constraint.** Access to the grid and to gas turbines became a widely reported bottleneck for datacenter builds through 2025 and 2026.

**Not happened (yet), or went the other way**

- **"The Project".** No nationalised or government-run AGI effort exists as of September 2026. The nearest thing is the Department of Energy's "Genesis Mission" (executive order of November 24, 2025). It invokes the Manhattan Project, but it is a programme to speed up science using the national labs and partners, not a takeover of frontier labs.
- **Open weights did not fade.** He expected the frontier to become closed and secret. Open-weight models from DeepSeek, Qwen and others stayed close to the frontier through 2025 and 2026.
- **Export controls loosened rather than tightened** in several respects. The US rescinded the "AI Diffusion Rule" in May 2025, and large chip sales to Gulf states were approved in 2025.
- **Safety institutions.** In June 2025 the US AI Safety Institute was renamed the Center for AI Standards and Innovation (CAISI) and given a changed remit. It was not folded into the security apparatus he called for.
- **The $100B AI revenue run rate by mid-2026.** Scorecards published in 2026 did not find any big tech company disclosing a $100B AI-specific run rate by mid-2026. Companies do not report AI revenue consistently, so this one is hard to score cleanly.

**Still open:** AGI by 2027, the intelligence explosion, a 10 GW cluster by 2028, and a government project by 2028.

---

## Criticisms

- **Straight lines on log plots are not laws.** The whole forecast rests on extrapolating effective compute and on assuming that more OOMs means more general capability. Critics point out that the "preschooler to high schooler" mapping is a loose analogy, not a measurement, and that benchmark scores can rise faster than real-world usefulness.
- **"Unhobbling" does a lot of unmeasured work.** It is the least quantified term in the equation, yet it carries the jump from chatbot to autonomous worker.
- **Bottlenecks outside compute.** Economists and forecasters argue that automated AI research is limited by experiment compute, data and serial time, and that the rest of the economy is limited by regulation, physical infrastructure and adoption. The essay acknowledges these but argues they are small, and many readers are not convinced.
- **Race framing can be self-fulfilling.** The strongest objection from the AI-safety community is that describing AI as a winner-take-all US-China race encourages exactly the rushed, secretive development that makes alignment harder. It also makes international coordination less likely.
- **Hawkishness and China.** Critics argue the essay takes conflict with China as a given and gives little weight to diplomacy, arms control, or the chance that China is not racing in the same way.
- **Conflict of interest.** After publishing, Aschenbrenner started an AI-focused investment fund named Situational Awareness. Critics note that a forecast of huge AI capital spending matches the fund's thesis. Defenders note that he says so openly.
- **The "few hundred people" framing.** Treating disagreement as a lack of awareness makes the argument hard to engage with on its merits.

---

## Real-World Impact

- **Policy debate.** It became required reading in parts of Washington and in AI-policy circles in 2024, and its vocabulary is widely reused: OOMs, "the Project", "lock down the labs", the trillion-dollar cluster.
- **Security.** Frontier-lab security and the theft of weights or secrets moved up lab and government agendas after 2024. Many people were pushing on this at the same time, so how much is due to the essay is uncertain.
- **Forecasting culture.** It spawned a small industry of scorecards and retrospectives. That is useful in itself: it is one of the few essays about the AI future specific enough to check.

---

## Key Takeaways for Practitioners

1. **Learn the OOM habit, but check its inputs.** Splitting progress into compute, algorithmic efficiency and qualitative "unhobbling" is a useful way to think, even if you doubt the conclusion.
2. **Infrastructure is strategy.** Power, chips and datacenters became headline issues, as he predicted. Plans that assume unlimited cheap compute may be wrong in either direction.
3. **Treat weights and training recipes as sensitive assets.** Whatever you think of the geopolitics, basic security for models and data is cheap insurance.
4. **Read forecasts for their mechanisms, not their dates.** The dates are where essays like this are most likely to be wrong. The causal chain is where they are most useful.

---

## Limitations & Future Directions

The essay is one author's argument, not a study. Its numbers are estimates drawn from public sources and insider impressions, and it does little to model uncertainty or alternative scenarios. Its most important claims (AGI by 2027, an intelligence explosion, a government project) resolve between 2027 and 2030. For a different view of the same period, compare Dario Amodei's [Machines of Loving Grace](../114-machines-of-loving-grace/summary.md), which shares the fast-timelines premise but focuses on benefits, and Silver and Sutton's [Era of Experience](../116-era-of-experience/summary.md), which argues that progress will come from reinforcement learning rather than more human data.

---

## Further Reading

- **Original essay:** [situational-awareness.ai](https://situational-awareness.ai/)
- **One-year retrospective (Delisle, June 2025):** [lesswrong.com/posts/EGGruXRxGQx6RQt8x](https://www.lesswrong.com/posts/EGGruXRxGQx6RQt8x/situational-awareness-a-one-year-retrospective)
- **Genesis Mission executive order (November 2025):** [whitehouse.gov](https://www.whitehouse.gov/presidential-actions/2025/11/launching-the-genesis-mission/)
- **In this collection:** [Scaling Laws](../../techniques/12-scaling-laws/summary.md), [Chinchilla](../../techniques/18-chinchilla/summary.md), [The Scaling Hypothesis](../112-scaling-hypothesis/summary.md), [The Bitter Lesson](../111-bitter-lesson/summary.md), [Weak-to-Strong Generalization](../../techniques/128-weak-to-strong/summary.md)

## Citation

```bibtex
@misc{aschenbrenner2024situational,
  title={Situational Awareness: The Decade Ahead},
  author={Aschenbrenner, Leopold},
  year={2024},
  month={June},
  howpublished={\url{https://situational-awareness.ai/}}
}
```

<!-- related:start -->

---

## Related in This Collection

- [Training Language Models to Follow Instructions with Human Feedback (InstructGPT)](../../language-models/05-instructgpt-rlhf/summary.md)
- [Training Compute-Optimal Large Language Models (Chinchilla)](../../techniques/18-chinchilla/summary.md)
- [DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning](../../language-models/26-deepseek-r1/summary.md)
- [Qwen3: Technical Report](../../language-models/28-qwen3/summary.md)
- [OpenAI o1: Learning to Reason with Reinforcement Learning](../../language-models/31-openai-o1/summary.md)
- [LLaMA 3.3: Matching 405B Performance with 70B Parameters](../../language-models/33-llama3.3/summary.md)
- [GPT-4 Technical Report](../../language-models/36-gpt4/summary.md)
- [Scaling LLM Test-Time Compute: The Theoretical Foundation for Reasoning Models](../../techniques/50-test-time-compute/summary.md)

<!-- related:end -->
