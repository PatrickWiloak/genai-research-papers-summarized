---
title: "Self-Consistency Improves Chain of Thought Reasoning in Language Models"
slug: "77-self-consistency"
number: 77
category: "techniques"
authors: "Xuezhi Wang, Jason Wei, Dale Schuurmans, Quoc Le, Ed Chi, Sharan Narang, Aakanksha Chowdhery, Denny Zhou (Google Research, Brain Team)"
published: "March 2022 (ICLR 2023)"
year: 2022
url: "https://arxiv.org/abs/2203.11171"
tags: ["reasoning", "chain-of-thought", "test-time-compute", "prompting"]
---

# Self-Consistency Improves Chain of Thought Reasoning in Language Models

**Authors:** Xuezhi Wang, Jason Wei, Dale Schuurmans, Quoc Le, Ed Chi, Sharan Narang, Aakanksha Chowdhery, Denny Zhou (Google Research, Brain Team)
**Published:** March 2022 (ICLR 2023)
**Paper:** [arxiv.org/abs/2203.11171](https://arxiv.org/abs/2203.11171)

---

## Why This Matters

Self-consistency is **the simplest way to make a model reason better, and it works on every model**. Sample several [chain-of-thought](../09-chain-of-thought/summary.md) answers instead of one, then take the majority vote. That is the whole method, and it added double-digit accuracy on math benchmarks at a time when nobody had a better idea.

- **Prompting-only, no training.** Works with any model behind any API.
- **Large gains on reasoning benchmarks** - reported improvements of roughly +18 points on GSM8K, +11 on SVAMP, and +12 on AQuA with PaLM 540B.
- **The conceptual ancestor of the reasoning-model era.** "Spend more compute at inference to get better answers" is the thesis that [o1](../../language-models/31-openai-o1/summary.md), [test-time compute scaling](../50-test-time-compute/summary.md), and [DeepSeek-R1](../../language-models/26-deepseek-r1/summary.md) built on.
- **Still the default trick** for squeezing accuracy out of an existing model, under the name "majority voting" or "self-consistency decoding."

**The insight:** for a problem with one correct answer, there are many valid reasoning paths that reach it and many *different* wrong paths that reach different wrong answers. Correct reasoning converges; incorrect reasoning scatters. So sample many paths and let them vote - agreement is evidence of correctness.

---

## The Problem: Greedy Decoding Commits to One Path

Chain-of-thought prompting had just shown that asking a model to "think step by step" dramatically improved arithmetic and commonsense reasoning. But standard decoding is greedy: the model produces one chain, and if it makes an arithmetic slip in step three, everything after is wrong and the final answer is wrong. There is no recovery mechanism and no second opinion.

This is a brittle failure mode. A model that knows how to solve a problem still fails it a meaningful fraction of the time because of a single unlucky token. Worse, greedy decoding is not even guaranteed to find the highest-probability *reasoning path* - it is a local heuristic.

Prior attempts to fix this used trained verifiers or re-ranking models, which required extra training data and extra models.

---

## The Core Innovation

Replace greedy decoding with sample-and-vote:

```
Question: "Janet has 3 boxes of 12 pencils. She gives away 8.
           How many does she have left?"

--- Standard CoT (greedy, 1 sample) ---
"3 x 12 = 36, minus 8 is 28."          -> 28   (submitted)

--- Self-Consistency (sample 5 at temperature ~0.7) ---
Path 1: "3 boxes x 12 = 36. 36 - 8 = 28."                -> 28
Path 2: "Each box 12, three boxes 36 pencils total,
         after giving 8 away: 28."                       -> 28
Path 3: "12 + 12 + 12 = 36; 36 - 8 = 28."                -> 28
Path 4: "3 x 12 = 32. 32 - 8 = 24."                      -> 24  (arithmetic slip)
Path 5: "36 pencils, gave away 8 boxes... 36 - 8 = 28."   -> 28

Majority vote over ANSWERS (not reasoning): 28 appears 4 times
Final answer: 28
```

Two design choices make this work:

1. **Sample with temperature** (or top-k / nucleus sampling) rather than decode greedily, so the paths genuinely differ.
2. **Marginalize over the reasoning, vote on the answer.** The reasoning chains are treated as latent variables. Two chains that phrase things differently but reach 28 both count for 28. This is why the method needs a task with an extractable final answer.

---

## Key Components Explained

### 1. Marginalization Over Reasoning Paths
**What it does:** Turns a single sample into an estimate of the model's actual answer distribution.
**How it works:** The model implicitly defines a distribution over (reasoning, answer) pairs. Greedy decoding takes one high-probability point. Self-consistency samples from the distribution and marginalizes out the reasoning, approximating `argmax_answer P(answer | question)` rather than `argmax_path`. This is a better-posed objective, and it is why the method is not merely an ensemble heuristic.

### 2. Sample Count as a Compute Dial
**What it does:** Trades inference cost for accuracy, smoothly.
**How it works:** Accuracy rises steeply from 1 to about 10 samples and then flattens, with most benchmarks near-saturated by 40 samples. Practically: 5 samples for a cheap boost, 10 to 20 for serious use, 40 for benchmark maximization. This monotone-but-saturating curve is the empirical seed of the [test-time compute](../50-test-time-compute/summary.md) literature.

### 3. Answer Extraction and Equivalence
**What it does:** Determines what "the same answer" means.
**How it works:** Numeric answers vote directly. Multiple choice votes on the letter. Free-form text is the hard case - "28 pencils" and "twenty-eight" must be normalized to the same bucket. Poor normalization silently destroys the method's benefit, and this is the most common implementation bug.

### 4. Emergence With Scale
**What it does:** Explains why this helps big models more.
**How it works:** Self-consistency requires the model to be right more often than any single wrong answer. If the model's most likely answer is wrong, voting confidently amplifies the wrong answer. Small models on hard tasks can be made *worse*. The gains grew reliably with model scale in the paper's experiments.

---

## Key Results

With PaLM 540B and chain-of-thought prompting, self-consistency (40 samples) versus greedy CoT:

| Benchmark | CoT (greedy) | + Self-Consistency | Gain |
|---|---|---|---|
| GSM8K (grade-school math) | 56.5% | 74.4% | +17.9 |
| SVAMP (math word problems) | 79.0% | 86.6% | +7.6 |
| AQuA (algebraic QA) | 35.8% | 48.3% | +12.5 |
| StrategyQA (multi-hop) | 75.3% | 81.6% | +6.3 |
| ARC-challenge (science) | 85.2% | 88.7% | +3.5 |

- Gains held across model families (GPT-3, LaMDA, PaLM) and across sizes, growing with scale.
- Self-consistency outperformed sample-and-rank and beam search under matched sample budgets.
- Robust to sampling hyperparameters - the method is not fragile to temperature choice within a reasonable range.

---

## Why This Was Revolutionary

- **Established inference-time compute as a quality lever.** In 2022 the assumption was that model quality was fixed at training time. Self-consistency showed you could buy accuracy with sampling, no retraining involved.
- **Introduced a confidence signal for free.** Vote share (say, 18 of 20 agreeing) correlates with correctness and is usable for routing, abstention, or escalation - none of which the paper's title advertises.
- **Required nothing but an API.** No verifier, no fine-tuning, no labeled data. Adoption was immediate.
- **Set the frame for the reasoning era.** Best-of-N with a verifier, [process reward models](../51-process-reward-models/summary.md), [tree of thoughts](../25-tree-of-thoughts/summary.md), and o1-style long chains are all refinements of "generate multiple candidates, then select."

---

## Real-World Impact

- **Standard in evaluation harnesses.** Benchmark numbers reported as "maj@8" or "cons@64" are self-consistency; the distinction from pass@k matters and is often confused.
- **Production reliability.** Extraction pipelines, classification services, and agent decision points often run 3 to 5 samples and vote, because the cost is small relative to the error reduction.
- **RL training data.** Majority-vote answers are used as pseudo-labels to generate training data when ground truth is unavailable, a technique that shows up in self-improvement pipelines and in [RLVR](../39-rlvr/summary.md)-adjacent work.
- **Reasoning models internalized it.** Part of what long-chain reasoning models do during their thinking phase is generate and compare alternative approaches - the same logic moved inside the model.

---

## Key Takeaways for Practitioners

1. **Use 5 samples before you use a bigger model.** On reasoning-shaped tasks it is usually cheaper and often more effective.
2. **Temperature 0.6 to 0.8 is the working range.** Too low and the samples are identical (no benefit); too high and reasoning degrades.
3. **Vote share is a usable confidence score.** Route low-agreement queries to a stronger model or to a human; this pattern is more valuable than the accuracy gain in many products.
4. **Normalize answers carefully.** Most disappointing self-consistency implementations are failing at answer extraction, not at reasoning.
5. **Do not use it on open-ended generation.** There is no majority to take over essays or code with many valid forms - use a verifier or a judge instead.
6. **Check that the base accuracy is above chance-plus.** If the model is usually wrong, voting makes it confidently wrong.

---

## Limitations & Future Directions

- **N times the cost.** Forty samples is forty times the inference bill; this is the entire trade.
- **Only for extractable answers.** Math, multiple choice, and structured extraction, yes. Essays, code, and open-ended plans, no.
- **Amplifies systematic errors.** If a model consistently misreads a problem type, all samples share the misreading and voting reinforces it. Consensus is not correctness.
- **Superseded in part by verifiers.** Weighted voting using a [process reward model](../51-process-reward-models/summary.md) or a trained verifier beats plain majority vote, which is the main direction later work took.
- **Reasoning models blur the comparison.** With a model that already reasons at length internally, the marginal value of external voting is smaller, though still nonzero.

---

## Further Reading

- **Original Paper:** [arxiv.org/abs/2203.11171](https://arxiv.org/abs/2203.11171)
- **Universal Self-Consistency (for free-form answers):** [arxiv.org/abs/2311.17311](https://arxiv.org/abs/2311.17311)
- **In this collection:** [Chain-of-Thought](../09-chain-of-thought/summary.md), [Tree of Thoughts](../25-tree-of-thoughts/summary.md), [Process Reward Models](../51-process-reward-models/summary.md), [Test-Time Compute Scaling](../50-test-time-compute/summary.md)

## Citation

```bibtex
@inproceedings{wang2023selfconsistency,
  title={Self-Consistency Improves Chain of Thought Reasoning in Language Models},
  author={Wang, Xuezhi and Wei, Jason and Schuurmans, Dale and Le, Quoc and Chi, Ed and Narang, Sharan and Chowdhery, Aakanksha and Zhou, Denny},
  booktitle={International Conference on Learning Representations},
  year={2023}
}
```

<!-- related:start -->

---

## Related in This Collection

- [Language Models are Few-Shot Learners (GPT-3)](../../language-models/04-gpt3-few-shot-learners/summary.md)
- [Chain-of-Thought Prompting Elicits Reasoning in Large Language Models](../../techniques/09-chain-of-thought/summary.md)
- [Tree of Thoughts: Deliberate Problem Solving with Large Language Models](../../techniques/25-tree-of-thoughts/summary.md)
- [DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning](../../language-models/26-deepseek-r1/summary.md)
- [OpenAI o1: Learning to Reason with Reinforcement Learning](../../language-models/31-openai-o1/summary.md)
- [RLVR: Reinforcement Learning from Verifiable Rewards](../../techniques/39-rlvr/summary.md)
- [Scaling LLM Test-Time Compute: The Theoretical Foundation for Reasoning Models](../../techniques/50-test-time-compute/summary.md)
- [Let's Verify Step by Step: Process Reward Models](../../techniques/51-process-reward-models/summary.md)

<!-- related:end -->
