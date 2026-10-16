---
title: "Measuring AI Ability to Complete Long Tasks (METR Time Horizons)"
slug: "145-metr-time-horizons"
number: 145
category: "techniques"
authors: "Thomas Kwa, Ben West, Joel Becker, Amy Deng, Katharyn Garcia, Max Hasin, Sami Jawhar, Megan Kinniment, Nate Rush, Sydney Von Arx, Ryan Bloom, Thomas Broadley, Haoxing Du, Brian Goodrich, Nikola Jurkovic, Luke Harold Miles, Seraphina Nix, Tao Lin, Chris Painter, Neev Parikh, David Rein, Lucas Jun Koba Sato, Hjalmar Wijk, Daniel M. Ziegler, Elizabeth Barnes, Lawrence Chan (METR)"
published: "March 2025 (arXiv v1, 18 March 2025; later revised - the current arXiv title is \"Measuring AI Ability to Complete Long Software Tasks\")"
year: 2025
url: "https://arxiv.org/abs/2503.14499"
tags: ["evaluation", "agents", "scaling"]
---

# Measuring AI Ability to Complete Long Tasks (METR Time Horizons)

**Authors:** Thomas Kwa, Ben West, Joel Becker, Amy Deng, Katharyn Garcia, Max Hasin, Sami Jawhar, Megan Kinniment, Nate Rush, Sydney Von Arx, Ryan Bloom, Thomas Broadley, Haoxing Du, Brian Goodrich, Nikola Jurkovic, Luke Harold Miles, Seraphina Nix, Tao Lin, Chris Painter, Neev Parikh, David Rein, Lucas Jun Koba Sato, Hjalmar Wijk, Daniel M. Ziegler, Elizabeth Barnes, Lawrence Chan (METR)
**Published:** March 2025 (arXiv v1, 18 March 2025; later revised - the current arXiv title is "Measuring AI Ability to Complete Long Software Tasks")
**Paper Link:** [arxiv.org/abs/2503.14499](https://arxiv.org/abs/2503.14499)

---

## Why This Paper Matters

Ask "how good is this model?" and the usual answer is a benchmark score: 72% on this test, 91% on that one. The trouble is that benchmark scores saturate. Once models hit 90% the number stops telling you anything, and a score on one benchmark cannot be compared with a score on another. Worse, a percentage says nothing about *what kind of work* a model can actually take off your plate.

METR (Model Evaluation and Threat Research, a nonprofit that evaluates frontier models) proposed a different yardstick: **how long is the task, measured in human working time, that an AI can complete reliably?** A task a skilled person finishes in 2 minutes and a task that takes them 8 hours are both "tasks", but they are very different jobs. If you plot the length of task that models can handle over time, you get a trend line you can extrapolate, and it never saturates in the same way a percentage does.

The headline finding became one of the most-cited charts in AI discussion: the task length that frontier models complete with 50% success has been **doubling roughly every seven months** since 2019. It gave forecasters, policymakers and engineers a single concrete number to argue about, instead of vibes.

The analogy: instead of asking a new hire "what did you score on the exam?", you ask "what is the longest project you can finish on your own without me checking in?" That second question is the one a manager cares about.

---

## The Core Innovation

The paper defines the **50% time horizon** of a model:

```
50% time horizon = the length of task (in human time) that the model
                   completes successfully about half the time
```

Here "human time" means how long a skilled human professional takes to do the task, not how long the AI takes. The method:

```
1. Build a suite of tasks spanning seconds to many hours
2. Time skilled humans doing each task (human baselines)
3. Run each AI model on every task, several attempts each
4. Fit a curve: success probability vs. log(human time)
5. Read off the task length where the curve crosses 50%
```

Because success falls as tasks get longer, the curve slopes downward. The point where it crosses 50% is the horizon. Do this for models released from 2019 to 2025, plot horizon against release date on a log scale, and a straight line appears.

---

## Key Components Explained

### 1. The task suite
**What it does:** Covers a wide range of task lengths so the curve can be fit.
**How it works:** Three sources, per the paper's own counts (the text gives 170 tasks in one place and 169 in another, so treat the total as approximate):
- **HCAST:** 97 tasks from 46 families, from about 1 minute to 30 hours of human time (software engineering, ML, cybersecurity and general reasoning).
- **RE-Bench:** 7 ML research-engineering tasks, about 8 hours each.
- **SWAA (Software Atomic Actions):** 66 new single-step tasks of roughly 1 to 30 seconds, added so that even GPT-2-era models have something to score on.

### 2. Human baselines
**What it does:** Gives each task a "how long does it take a person" label.
**How it works:** Over 800 baselines totalling 2,529 hours were collected from skilled professionals in software engineering, ML and cybersecurity (about 5 years of relevant experience on average). Task length is the geometric mean of the *successful* human times. Not every task has a real baseline: the paper reports human baselines for 148 tasks and uses researcher estimates for 21 HCAST tasks.

### 3. The logistic fit
**What it does:** Turns pass/fail results into a single number per model.
**How it works:** A logistic regression of success against log task length gives a smooth decline from "always succeeds" to "never succeeds". The 50% crossing is the horizon. Continuous scores are binarized with a task-specific threshold.

```
P(success)
  1.0 |*****
      |     ****
  0.5 |---------*****------   <- 50% horizon is here
      |              ****
  0.0 |                  ******
      +------------------------> log(human time for the task)
        seconds   minutes   hours
```

### 4. The trend fit
**What it does:** Measures how fast horizons grow.
**How it works:** Regress log(horizon) on release date. The slope gives a doubling time.

---

## Key Results

- **Doubling time of about 207 days** (95% bootstrap confidence interval 166 to 240 days), i.e. roughly seven months, over 2019 to 2025.
- **Faster recently:** the 2023 to 2025 trend is about 20% faster than the 2019 to 2025 fit, and the 2024-only estimate on the paper's suite is about 143 days. The paper itself flags low confidence here, since only seven frontier models fall in the 2024 to 2025 window.
- **Scale:** GPT-2 sits at about 2 seconds. Claude 3.7 Sonnet, the frontier model at the time of the paper, was around 50 minutes (abstract). o3 was about 110 minutes in the later analysis, above the long-run trend line (p = 0.006).
- **The 80% horizon is much shorter:** about 4 to 6 times shorter than the 50% horizon (roughly 5x), but it doubles at a similar rate (about 204 days).
- **Why the gains happen:** the authors attribute them mainly to greater reliability, better recovery from mistakes, stronger logical reasoning and better tool use, rather than raw one-shot ability.
- **Extrapolation:** if the trend continued, the paper gives mid-2028 to roughly 2030 as the window in which models could reach a one-month (about 167 working hours) horizon, with a central estimate near 2029. The text states this range slightly differently in different sections, so treat the exact dates as soft. The abstract's version: if results generalize to real software tasks, AI could automate many tasks that take humans a month within five years.

---

## Why This Was Revolutionary

- **A unit people can feel.** "Can do 1-hour tasks" is more meaningful than "scores 63%".
- **No saturation ceiling.** Add longer tasks and the metric keeps going.
- **A trend you can forecast.** An exponential on a log chart is easy to extend, argue with and update.
- **Common scale across eras.** One axis covers GPT-2 and 2025 reasoning models.
- **Safety relevance.** Long autonomous tasks are exactly where dangerous capabilities (cyber operations, AI R&D) become possible, so the metric ties capability measurement to risk thresholds.

---

## Real-World Impact

- The time-horizon chart became a standard reference in discussions of agent progress, capability forecasting and AI policy, and METR has continued to publish horizons for new frontier models. (Later per-model figures are not summarised here because they change with every release; check METR's site.)
- It reframed benchmark design: newer agent benchmarks report task length or "hours of work" alongside accuracy. See the software-engineering benchmark [SWE-bench](../84-swe-bench/summary.md), which the paper uses as a cross-check.
- It connects to the reasoning-model wave: longer horizons track the rise of RL-trained reasoning, see [GRPO](../38-grpo/summary.md) and [RLVR](../39-rlvr/summary.md).

---

## Key Takeaways for Practitioners

1. **Read a horizon as "at 50% reliability".** A 2-hour 50% horizon does not mean the model reliably does 2-hour jobs; at 80% reliability the task length is roughly 5x shorter.
2. **Horizon is relative to a task distribution and a human baseline.** Different task types, or more skilled baseliners, change the number.
3. **Messy real work is harder.** The paper scored tasks on 16 "messiness" factors; controlling for length, models do worse on messier tasks.
4. **Plan around the slope, not the point.** The useful signal is how fast horizons grow, with wide error bars.
5. **Verification still costs human time.** A model that attempts a long task may need review that eats the savings.

---

## Limitations & Future Directions

- **External validity.** The tasks are largely automatically scored, self-contained and well specified. Real jobs have unclear goals, shifting requirements and no test harness. The authors say they cannot rule out a slower trend on the task distribution needed to automate real software jobs.
- **Messiness.** Lower performance on messy tasks was found, though no plateau in the trend showed up in the messier subset.
- **Whose time?** In a side study on internal pull requests, contractors took 5 to 18 times longer than repo maintainers, so "human time" depends heavily on who is the human. AI performance matched contractor times better than maintainer times.
- **Different doubling time elsewhere.** On SWE-bench Verified the trend is also exponential but doubles faster (about 70 days), partly because annotator time estimates likely understate contractor time on easy tasks.
- **Measurement is hard at the edges.** Very long horizons and success rates near 100% are costly to measure, and the fit is sensitive to a handful of models.
- **Extrapolation assumes the trend holds.** Compute, data, and algorithmic limits could bend the curve either way.

I could not verify updates after the paper's own v4 revision beyond the figures above, so later model-by-model horizons and any revised doubling times are omitted on purpose.

---

## Further Reading

- **Original Paper:** https://arxiv.org/abs/2503.14499
- **HTML version:** https://arxiv.org/html/2503.14499
- **METR (publisher, with updated horizon charts):** https://metr.org
- **Related in this repo:** [SWE-bench](../84-swe-bench/summary.md), [GRPO](../38-grpo/summary.md), [RLVR](../39-rlvr/summary.md)

## Citation

```bibtex
@article{kwa2025measuring,
  title={Measuring AI Ability to Complete Long Tasks},
  author={Kwa, Thomas and West, Ben and Becker, Joel and Deng, Amy and Garcia, Katharyn and Hasin, Max and Jawhar, Sami and Kinniment, Megan and Rush, Nate and Von Arx, Sydney and others},
  journal={arXiv preprint arXiv:2503.14499},
  year={2025}
}
```

<!-- related:start -->

---

## Related in This Collection

- [GRPO: Group Relative Policy Optimization](../../techniques/38-grpo/summary.md)
- [RLVR: Reinforcement Learning from Verifiable Rewards](../../techniques/39-rlvr/summary.md)
- [Language Models are Unsupervised Multitask Learners (GPT-2)](../../language-models/64-gpt2/summary.md)
- [SWE-bench: Can Language Models Resolve Real-World GitHub Issues?](../../techniques/84-swe-bench/summary.md)

<!-- related:end -->
