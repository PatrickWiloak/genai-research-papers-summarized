---
title: "Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena"
slug: "85-llm-as-judge"
number: 85
category: "techniques"
authors: "Lianmin Zheng, Wei-Lin Chiang, Ying Sheng, Siyuan Zhuang, Zhanghao Wu, Yonghao Zhuang, Zi Lin, Zhuohan Li, Dacheng Li, Eric P. Xing, Hao Zhang, Joseph E. Gonzalez, Ion Stoica (UC Berkeley, UCSD, CMU, Stanford, MBZUAI)"
published: "June 2023 (NeurIPS 2023 Datasets and Benchmarks Track)"
year: 2023
url: "https://arxiv.org/abs/2306.05685"
tags: ["evaluation", "benchmarks", "alignment"]
---

# Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena

**Authors:** Lianmin Zheng, Wei-Lin Chiang, Ying Sheng, Siyuan Zhuang, Zhanghao Wu, Yonghao Zhuang, Zi Lin, Zhuohan Li, Dacheng Li, Eric P. Xing, Hao Zhang, Joseph E. Gonzalez, Ion Stoica (UC Berkeley, UCSD, CMU, Stanford, MBZUAI)
**Published:** June 2023 (NeurIPS 2023 Datasets and Benchmarks Track)
**Paper:** [arxiv.org/abs/2306.05685](https://arxiv.org/abs/2306.05685)

---

## Why This Matters

This paper legitimized **using a strong language model to grade other language models**, and it created **Chatbot Arena**, which became the most-watched leaderboard in AI. Together they solved the evaluation crisis that hit when models started producing open-ended text that no automatic metric could score.

- **GPT-4 as a judge agreed with human preferences at roughly 80 percent** - about the same rate two humans agree with each other.
- **Made open-ended evaluation cheap.** Human evaluation costs dollars and days per comparison; a judge model costs cents and seconds.
- **Chatbot Arena's Elo ratings** became the de facto public ranking of chat models, based on millions of blind pairwise votes from real users.
- **It also documented the biases**, which is what makes it a serious paper rather than a convenient shortcut: position bias, verbosity bias, self-enhancement bias, and weak grading of math and reasoning.

**The insight:** if evaluating a response is easier than producing it - and it usually is - then a model capable of producing good responses is capable of ranking them. Validate that claim against human preferences, quantify where it fails, and you have a usable measurement instrument.

---

## The Problem: Nothing Could Measure a Chatbot

By mid-2023, dozens of instruction-tuned models were being released weekly, and there was no credible way to rank them.

- **Traditional benchmarks measure the wrong thing.** MMLU and HellaSwag score knowledge and multiple-choice reasoning. A model can be excellent at those and an unpleasant, unhelpful assistant.
- **Automatic text metrics do not work.** BLEU and ROUGE compare against a reference; for "explain recursion to a child" there is no reference and a thousand good answers.
- **Human evaluation does not scale.** Expensive, slow, and hard to reproduce across studies.
- **Benchmarks saturate and leak.** Reported scores drifted upward faster than models actually improved, and contamination was rampant.

Meanwhile the actual question users had - "which of these is the better assistant" - was going unanswered by any published number.

---

## The Core Innovation

Two complementary instruments, plus a study of whether the automatic one can be trusted.

### MT-Bench: a curated multi-turn benchmark

80 carefully written multi-turn questions across 8 categories (writing, roleplay, extraction, reasoning, math, coding, STEM knowledge, humanities). Each has a follow-up turn, which is the point: single-turn benchmarks miss whether a model can maintain context and take correction. A judge model scores the responses.

### Chatbot Arena: crowdsourced blind pairwise comparison

```
  A real user types their own question.
  It goes to TWO anonymous models, side by side.
  The user reads both answers and votes: A better / B better / tie.
  Model identities are revealed only after the vote.
  Elo ratings (as in chess) are computed from millions of votes.
```

The design choices matter: **real user questions** (not a fixed set, so it cannot be gamed or contaminated), **blind** (no brand halo), **pairwise** (humans are far better at comparing than at scoring absolutely), and **continuously running** (new models slot into an existing rating system).

### The validation study

Compare GPT-4's judgments against human votes on the same pairs. Agreement was roughly 80 percent-plus, comparable to inter-human agreement - the headline result that made LLM judging respectable.

---

## Key Components Explained

### 1. Judging Modes
**What it does:** Different questions need different judge protocols.
**How it works:**
- **Pairwise comparison** - "which is better, A or B?" Most reliable, and matches how Arena collects human data.
- **Single-answer grading** - "score this 1 to 10." Scalable and cheaper, but scores drift over time and across batches.
- **Reference-guided grading** - give the judge a correct reference answer first. Essential for math and reasoning, where an unaided judge is unreliable.

### 2. Position Bias
**What it does:** The most important practical failure mode.
**How it works:** Judges systematically prefer whichever response is presented first (in some settings the effect is large). **The fix is mandatory: run every comparison twice with the order swapped, and count it as a tie if the verdict flips.** Any LLM-judge pipeline that does not do this is producing partly meaningless numbers, and this is the single most common implementation error.

### 3. Verbosity Bias
**What it does:** Judges prefer longer answers.
**How it works:** Longer responses win more often even when they add nothing, which creates a direct optimization hazard: models trained against judge feedback learn to pad. This is a documented cause of the verbose house style of many RLHF-trained assistants. Length-controlled scoring (used in later versions of AlpacaEval) is the standard correction.

### 4. Self-Enhancement Bias
**What it does:** Judges favor their own outputs and their own style.
**How it works:** GPT-4 rates GPT-4-family responses more favorably than humans do. This makes single-judge evaluation of a model family by a sibling model suspect, and it is why serious evaluations use a judge from a different family, a panel of judges, or human validation on a sample.

### 5. Limited Grading of Math and Reasoning
**What it does:** Bounds where judges are usable.
**How it works:** A judge cannot reliably verify a proof or a long calculation it could not perform itself, and it is easily persuaded by confident-sounding wrong reasoning. Reference-guided grading helps; for verifiable domains, **execute or check the answer instead of judging it** - which is the logic behind [RLVR](../39-rlvr/summary.md) and [process reward models](../51-process-reward-models/summary.md).

---

## Key Results

- **GPT-4 judge versus human agreement: roughly 80 percent-plus**, matching human-human agreement rates on the same comparisons.
- MT-Bench and Arena rankings correlated strongly with each other and with expert judgment, validating both instruments.
- Position, verbosity, and self-enhancement biases were measured and reported rather than glossed over.
- Chatbot Arena accumulated millions of human votes, becoming the largest public dataset of preference comparisons on real user traffic.

---

## Why This Was Revolutionary

- **Made open-ended evaluation tractable.** Cheap, fast, reproducible scoring of things no metric could score.
- **Established the methodology honestly.** Quantified biases and prescribed mitigations, which is why the technique survived scrutiny.
- **Chatbot Arena became the field's scoreboard.** For a period, Arena Elo was the number labs cared most about publicly, and it remains the least gameable public ranking because its questions come from users in real time.
- **Judges became training infrastructure, not just evaluation.** LLM judges now generate preference data for [DPO](../../language-models/19-dpo/summary.md) and RLHF, score candidates for rejection sampling, and gate synthetic data pipelines. Most models are now partly trained on judgments from other models.

---

## Real-World Impact

- **LLM-as-judge is standard practice.** Every production LLM application with quality monitoring uses some form of it; evaluation frameworks (LangSmith, Braintrust, OpenAI Evals, Ragas) ship judge templates by default.
- **AlpacaEval, Arena-Hard, WildBench** and similar automatic benchmarks are direct descendants, with length-control and bias corrections layered on.
- **RLAIF and [Constitutional AI](../../language-models/14-constitutional-ai/summary.md)** rely on model-generated feedback, which is judging by another name applied to training.
- **Reward models are judges too.** The scoring model in an RLHF loop is doing exactly this task, and shares the same biases - a link worth keeping in mind when reasoning about why aligned models are verbose.
- **The Arena's design influenced everything downstream:** blind, pairwise, real-traffic, continuously updated is now the template for credible leaderboards.

---

## Key Takeaways for Practitioners

1. **Always swap positions and average.** Two calls per comparison. Non-negotiable.
2. **Prefer pairwise to absolute scoring.** "Which is better" is far more stable than "rate this 1-10," whose scale drifts.
3. **Control for length.** Either instruct the judge to ignore length, or use length-controlled scoring. Otherwise you are measuring verbosity.
4. **Do not judge with a sibling model.** Use a different family, or a panel, and validate against human labels on a sample.
5. **Write a rubric.** Judges with explicit criteria (accuracy, completeness, tone, format compliance) are substantially more consistent than judges asked "which is better."
6. **Use execution, not judgment, wherever it exists.** Tests, schema validation, exact match on verifiable answers. Judges are for what cannot be checked.
7. **Sample-validate continuously.** Have a human review a slice of judge decisions periodically; judges drift as models are updated behind the API.

---

## Limitations & Future Directions

- **Judges cannot exceed their own capability.** They cannot verify what they could not produce, which caps them exactly where evaluation is hardest.
- **Optimizing against a judge is Goodhart's law in action.** Train hard against a judge and you get a model that games the judge - length, structure, confident tone, hedging patterns.
- **Arena has its own biases.** Voters are self-selected and skew technical; questions skew toward what people bring to a demo site; style can beat substance in a fast side-by-side read.
- **Vulnerable to style over substance.** Well-formatted, confident wrong answers beat correct terse ones with judges and humans alike.
- **Reproducibility.** Judge models change behind versioned API names; a score from a year ago may not be reproducible today. Pin versions and record them.

---

## Further Reading

- **Original Paper:** [arxiv.org/abs/2306.05685](https://arxiv.org/abs/2306.05685)
- **Chatbot Arena:** [lmarena.ai](https://lmarena.ai/)
- **Chatbot Arena methodology paper:** [arxiv.org/abs/2403.04132](https://arxiv.org/abs/2403.04132)
- **AlpacaEval (length-controlled):** [arxiv.org/abs/2404.04475](https://arxiv.org/abs/2404.04475)

## Citation

```bibtex
@inproceedings{zheng2023judging,
  title={Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena},
  author={Zheng, Lianmin and Chiang, Wei-Lin and Sheng, Ying and Zhuang, Siyuan and Wu, Zhanghao and Zhuang, Yonghao and Lin, Zi and Li, Zhuohan and Li, Dacheng and Xing, Eric P. and Zhang, Hao and Gonzalez, Joseph E. and Stoica, Ion},
  booktitle={Advances in Neural Information Processing Systems (NeurIPS) Datasets and Benchmarks Track},
  year={2023}
}
```

<!-- related:start -->

---

## Related in This Collection

- [Training Language Models to Follow Instructions with Human Feedback (InstructGPT)](../../language-models/05-instructgpt-rlhf/summary.md)
- [Constitutional AI: Harmlessness from AI Feedback](../../language-models/14-constitutional-ai/summary.md)
- [Direct Preference Optimization (DPO): Your Language Model is Secretly a Reward Model](../../language-models/19-dpo/summary.md)
- [GPT-4 Technical Report](../../language-models/36-gpt4/summary.md)
- [RLVR: Reinforcement Learning from Verifiable Rewards](../../techniques/39-rlvr/summary.md)
- [Self-Instruct: Aligning Language Models with Self-Generated Instructions](../../techniques/79-self-instruct/summary.md)

<!-- related:end -->
