# Is Scaling Hitting a Wall?

**In one line:** "Scaling" used to mean one thing - bigger models on more text - and that recipe is running into data and cost limits; the open question is whether newer ways of spending compute (reinforcement learning and thinking longer at inference) keep progress going, and for how long.
**Last reviewed:** 2026-09-30

---

## The short version

- **Scaling laws are empirical, not physical laws.** From 2020 to 2022, researchers found that language-model loss falls smoothly and predictably as you add parameters, data and compute. That predictability drove the industry's investment.
- **The pretraining recipe has a data problem.** Epoch AI estimates the stock of high-quality public human text at about 300 trillion tokens and projects that frontier training will use it up sometime between 2026 and 2032.
- **Late 2024 brought the "wall" narrative.** Ilya Sutskever said "pre-training as we know it will end" because "we have but one internet", and OpenAI's very large GPT-4.5 (February 2025) was seen by many as a modest step for its cost.
- **The field shifted what it scales.** Reasoning models trained with reinforcement learning (RL) on checkable problems, and allowed to think longer at inference time, produced large gains in math, code and agentic tasks through 2025 and 2026.
- **Both sides have strong evidence.** Measures of real-world agent capability kept improving on a steady exponential into 2026, but analysts argue RL is far less efficient than pretraining and can only grow faster than total compute for a short time. There is no consensus.

## The mental model: three compute knobs

```
   KNOB 1: PRETRAINING          KNOB 2: POST-TRAINING RL       KNOB 3: TEST-TIME COMPUTE
   predict next token on        reward the model for           let the model "think"
   trillions of tokens          correct, checkable answers      longer on each question
   ---------------------        ------------------------        -----------------------
   limited by: data, money,     limited by: supply of           limited by: cost per query,
   power, chips                 verifiable tasks, efficiency    latency, diminishing returns
   2018-2024 main driver        2024- main driver              2024- main driver
```

"Is scaling hitting a wall?" is really three questions, one per knob. A plateau in knob 1 does not imply one in knobs 2 and 3, and vice versa.

## Background: why anyone believed in scaling

- **Kaplan et al. (2020)** found smooth power-law relationships between loss and model size, data and compute across many orders of magnitude. See [Scaling Laws for Neural Language Models](../../papers/techniques/12-scaling-laws/summary.md).
- **Chinchilla (2022)** corrected the recipe: for a fixed compute budget, models had been too big and trained on too little data. Compute-optimal training roughly scales parameters and tokens together. See [Chinchilla](../../papers/techniques/18-chinchilla/summary.md). This made data, not parameters, the scarcer input.
- **The philosophical case** came earlier. Rich Sutton's [The Bitter Lesson](../../papers/essays/111-bitter-lesson/summary.md) (2019) argued that general methods that exploit more computation beat hand-built cleverness in the long run. Gwern's [The Scaling Hypothesis](../../papers/essays/112-scaling-hypothesis/summary.md) (2020) argued that GPT-3 showed intelligence-like abilities emerge from scale itself.

An important caveat: a scaling law predicts *loss* (how well the model predicts text), not *usefulness*. Whether lower loss turns into abilities people care about is a separate, noisier relationship. See [Emergent Abilities](../../papers/techniques/81-emergent-abilities/summary.md) for the debate over whether abilities appear smoothly or suddenly.

## The case that pretraining is hitting limits

**1. Data exhaustion.** Epoch AI's June 2024 analysis estimated the effective stock of quality-adjusted public human text at roughly 300 trillion tokens (range: 100 to 1,000 trillion), and projected full use sometime between 2026 and 2032, depending on how aggressively models are "overtrained" (trained on more tokens than Chinchilla-optimal to make them cheaper to run). Their earlier 2022 estimate had said 2024; better filtering and training for multiple passes over the same data pushed it back.

**2. The people building it said so.** At NeurIPS in December 2024, Sutskever described data as "the fossil fuel of AI" and said "we've achieved peak data and there'll be no more".

**3. Expensive, modest-looking gains.** OpenAI released GPT-4.5 on 27 February 2025, describing it as its largest and most compute-intensive model to date. It was priced at $75 per million input tokens and $150 per million output tokens, and OpenAI announced in April 2025 that it would be removed from the API on 14 July 2025 in favour of the cheaper GPT-4.1. Critics read this as evidence that a big pretraining step no longer bought a proportionate jump.

**4. Physical and financial constraints.** Epoch's August 2024 analysis of whether scaling can continue to 2030 found runs of about 2x10^29 FLOP likely feasible, but identified power as the most binding constraint, followed by chip manufacturing (advanced packaging and high-bandwidth memory). Each 10x step requires new data centers measured in gigawatts. See [Cost of training](../compute/cost-of-training.md) and [AI hardware landscape](../compute/ai-hardware-landscape.md).

## The case that scaling continues (in new forms)

**1. RL and reasoning opened a new axis.** OpenAI's [o1](../../papers/language-models/31-openai-o1/summary.md) (September 2024) showed performance improving with both more RL training compute and more thinking time at inference. [DeepSeek-R1](../../papers/language-models/26-deepseek-r1/summary.md) (January 2025) showed that pure RL with verifiable rewards could produce long reasoning chains in an open model. See [RLVR](../../papers/techniques/39-rlvr/summary.md) and [Reasoning models](../concepts/reasoning-models.md).

**2. Test-time compute is its own scaling law.** Work such as [Scaling LLM Test-Time Compute](../../papers/techniques/50-test-time-compute/summary.md) (2024) showed that spending compute at inference, by sampling, searching or verifying, can substitute for a much larger model on some problems.

**3. Headline capabilities kept rising.** In July 2025 an advanced Gemini Deep Think model reached gold-medal standard at the International Mathematical Olympiad (35 of 42 points), working end to end in natural language within the time limit. METR's measurement of the length of tasks AI agents can complete with 50% reliability found it had been doubling roughly every 7 months for six years, with 2024-2025 data suggesting a faster pace; an updated methodology (Time Horizon 1.1) was released in January 2026.

**4. Data is less fixed than it looks.** Epoch's own work notes that multimodal data (images, video, audio) and synthetic data could extend supply by orders of magnitude, and that multi-epoch training on filtered data works better than once assumed. RL generates its own training signal from problems with checkable answers. See [Synthetic data and model collapse](./synthetic-data-and-model-collapse.md) and [FineWeb](../../papers/techniques/133-fineweb/summary.md).

**5. Physical headroom remains.** The same Epoch analysis that names power as the bottleneck still concludes a GPT-2-to-GPT-4-sized jump in compute is feasible by 2030.

## The strongest skeptical case against RL scaling

The reasoning boom has its own critics, and their argument is more specific than "a wall":

- **RL is inefficient.** Toby Ord argues that RL on long reasoning chains gives the model roughly one bit of feedback (right or wrong) per episode of thousands of tokens, reducing the information learned per unit of compute by a factor of 1,000 to 1,000,000 compared with pretraining. It may also generalise less, making each unit of general capability more expensive.
- **RL cannot outgrow the total compute budget for long.** Epoch's Josh You (May 2025) noted that reasoning training was then a small fraction of pretraining compute and growing about 10x every few months (o1 to o3). At that rate it would reach the frontier of total training compute "perhaps within a year", after which it could only grow as fast as compute overall (around 4x per year).
- **Much of the recent gain is paid at inference.** If progress comes from thinking longer on every query, each capability improvement raises running costs, which shifts the economics. See [Inference economics](../compute/inference-economics.md).
- **Verifiable domains are narrow.** RL works best where answers can be checked automatically (math, code, some games). Whether it transfers to open-ended tasks like research judgment or writing is disputed.

## The strongest optimistic reply

- Each time a single recipe slowed (bigger RNNs, then bigger pretraining), researchers found a new axis to scale. The Bitter Lesson predicts this pattern: the winners are general methods that absorb more compute, not any particular method.
- Agentic benchmarks that measure real work, not just test questions, have continued to improve on a steady trend.
- Efficiency gains (better data, architectures, optimisers such as [Muon](../../papers/techniques/144-muon/summary.md)) mean each FLOP buys more over time, independent of raw compute growth.
- Silver and Sutton's [Era of Experience](../../papers/essays/116-era-of-experience/summary.md) (2025) argues the next source of data is agents learning from their own interaction with the world, which is not bounded by the stock of human text.

## Where that leaves the question

| Claim | Status as of September 2026 |
|---|---|
| Public human text is finite and nearly used | Widely accepted; timing debated (Epoch: 2026-2032) |
| Pure pretraining scale-up gives smaller visible gains per dollar than before | Widely believed; hard to measure because labs rarely publish controlled comparisons |
| RL and test-time compute produced large gains in 2024-2026 | Well supported on math, code and agent benchmarks |
| RL scaling can continue at its 2024-2025 pace | Disputed; strong arguments it must slow to the pace of overall compute |
| Overall AI capability progress is slowing | Disputed; task-horizon measures had not shown a slowdown as of early 2026 |

## What to watch

- **Task-horizon trends** from METR and similar groups: a bend in the doubling curve would be the clearest sign of slowdown.
- **Very large training clusters** coming online, and whether the models trained on them show gains proportionate to their cost.
- **RL outside verifiable domains:** evidence that it improves open-ended skills would weaken the "narrow" objection.
- **Data sources beyond text:** video, interaction logs and agent experience at scale.
- **Inference cost curves:** if capability keeps rising only by thinking longer, prices per task will tell.

## Read next

- [Scaling Laws](../../papers/techniques/12-scaling-laws/summary.md), [Chinchilla](../../papers/techniques/18-chinchilla/summary.md), [Test-Time Compute](../../papers/techniques/50-test-time-compute/summary.md)
- [The Bitter Lesson](../../papers/essays/111-bitter-lesson/summary.md), [The Scaling Hypothesis](../../papers/essays/112-scaling-hypothesis/summary.md)
- [OpenAI o1](../../papers/language-models/31-openai-o1/summary.md), [DeepSeek-R1](../../papers/language-models/26-deepseek-r1/summary.md), [RLVR](../../papers/techniques/39-rlvr/summary.md)
- [Reasoning models](../concepts/reasoning-models.md), [Cost of training](../compute/cost-of-training.md), [Inference economics](../compute/inference-economics.md)
- [Do LLMs reason?](./do-llms-reason.md), [Synthetic data and model collapse](./synthetic-data-and-model-collapse.md)
- [LLM basics](https://github.com/PatrickWiloak/cloud-data-ai-security-zero-to-hero/blob/main/learn/concepts/llm-basics.md) (sibling repo)

## Sources

- Villalobos et al. / Epoch AI, "Will we run out of data? Limits of LLM scaling based on human-generated data" (June 2024): https://epoch.ai/blog/will-we-run-out-of-data-limits-of-llm-scaling-based-on-human-generated-data
- Epoch AI, "Can AI Scaling Continue Through 2030?" (20 Aug 2024): https://epoch.ai/blog/can-ai-scaling-continue-through-2030
- Josh You / Epoch AI, "How far can reasoning models scale?" (9 May 2025): https://epoch.ai/gradient-updates/how-far-can-reasoning-models-scale
- Toby Ord, "The Extreme Inefficiency of RL for Frontier Models": https://www.tobyord.com/writing/inefficiency-of-reinforcement-learning
- The Verge via Techmeme, Sutskever at NeurIPS, 13 Dec 2024: https://www.techmeme.com/241213/p33
- OpenAI, "Introducing GPT-4.5" (27 Feb 2025): https://openai.com/index/introducing-gpt-4-5/ ; MIT Technology Review, 27 Feb 2025: https://www.technologyreview.com/2025/02/27/1112619/openai-just-released-gpt-4-5-and-says-it-is-its-biggest-and-best-chat-model-yet/ ; API deprecation date: https://help.openai.com/en/articles/9624314-model-release-notes
- METR, "Measuring AI Ability to Complete Long Tasks" (19 Mar 2025, with Time Horizon 1.1 update, Jan 2026): https://metr.org/blog/2025-03-19-measuring-ai-ability-to-complete-long-tasks/
- Google DeepMind, "Advanced version of Gemini with Deep Think officially achieves gold-medal standard at the IMO" (July 2025): https://deepmind.google/discover/blog/advanced-version-of-gemini-with-deep-think-officially-achieves-gold-medal-standard-at-the-international-mathematical-olympiad/
- Kaplan et al. (2020) and Hoffmann et al. (2022): see the linked summaries in this repo.
