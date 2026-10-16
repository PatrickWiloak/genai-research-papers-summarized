---
title: "Why Language Models Hallucinate"
slug: "149-why-language-models-hallucinate"
number: 149
category: "techniques"
authors: "Adam Tauman Kalai, Ofir Nachum, Santosh S. Vempala, Edwin Zhang (OpenAI, Georgia Tech)"
published: "September 2025 (arXiv preprint, v1 submitted 4 September 2025)"
year: 2025
url: "https://arxiv.org/abs/2509.04664"
tags: ["safety", "evaluation"]
---

# Why Language Models Hallucinate

**Authors:** Adam Tauman Kalai, Ofir Nachum, Santosh S. Vempala, Edwin Zhang (OpenAI, Georgia Tech)
**Published:** September 2025 (arXiv preprint, v1 submitted 4 September 2025)
**Paper Link:** [arxiv.org/abs/2509.04664](https://arxiv.org/abs/2509.04664)

---

## Why This Paper Matters

Hallucination, a model stating something false with total confidence, is the best-known failure of language models. Before this paper it was usually described as a mysterious quirk, or as a bug that more data and bigger models would slowly wear away. Kalai and colleagues argue it is neither. Hallucination is what you should expect from a system trained and graded the way language models are, and the causes can be named and, in part, measured.

The paper makes two connected claims. First, hallucinations begin in pretraining, where a model that cannot perfectly separate valid statements from plausible-sounding invalid ones will sometimes generate the invalid ones. Second, they survive post-training because almost every popular benchmark grades answers as right or wrong, with no credit for saying "I don't know". Under that scoring, a model that guesses always does at least as well as one that abstains.

The practical upshot is unusual for a theory paper: the proposed fix is not a new training algorithm but a change to how existing leaderboards are scored.

**The insight:** a model that is rewarded for every lucky guess and never penalised for a wrong one will learn to bluff, exactly like a student on a multiple-choice exam with no negative marking.

---

## The Core Innovation

The authors reframe generation as classification. Imagine every possible output string sorted into "valid" and "error". Producing a valid answer is at least as hard as answering the yes/no question "is this output valid?" (the paper calls it the Is-It-Valid, or IIV, problem). Any language model can be used as an IIV classifier, so its generation errors can be bounded using its classification errors.

```
Generative error rate  >~  2 x (IIV misclassification rate)
```

(The paper states this as an approximate bound, with additional terms; read the proof for the exact form.)

This turns a vague complaint into a statistical question. If a model cannot reliably tell valid from invalid statements about some topic, it will generate invalid ones at a predictable rate, with no need for any flaw in the architecture.

---

## Key Components Explained

### 1. Errors that arise in pretraining
**What it does:** Explains why a model trained on clean data still hallucinates.
**How it works:** For facts with no learnable pattern, such as a person's birthday, there is nothing to generalise from. If the date is written in a fixed format, each person has 364 wrong claims for every correct one, so the model can only memorise what it saw. The paper ties the unavoidable error rate to the **singleton rate**: the fraction of facts that appear exactly once in the training data. This is a Good-Turing style idea (the number of things seen once estimates the probability mass of things not seen at all). The paper's own illustration: if 20% of birthday facts appear once, base models should be expected to hallucinate on at least about 20% of them.

### 2. Calibration makes bluffing the default
**What it does:** Explains why a base model guesses rather than abstains.
**How it works:** Standard cross-entropy training tends to produce well-calibrated base models, meaning their probabilities match how often they are right. A calibrated model that is 30% sure of an answer will, by design, produce it some of the time. The paper argues that this is the right behaviour for predicting text and, by the same token, the source of error when the output is read as a fact.

### 3. Other contributors
The authors also list limited model families (their example: a trigram model must have an error rate of at least 1/2 on a specific two-prompt case), computational hardness, distribution shift, and "garbage in, garbage out" from errors already in the training data.

### 4. Why hallucination survives post-training
**What it does:** Explains why RLHF-style polishing has not removed the problem.
**How it works:** The paper proves that, under binary 0-1 grading, abstaining is never optimal for a test-taker who believes the rubric is binary. An honest "I don't know" scores zero, a guess scores zero or one. The authors survey popular evaluations and report that most, including GPQA, MMLU-Pro, SWE-bench and Humanity's Last Exam, use binary grading with no credit for uncertainty. WildBench is the exception that gives partial credit for "I don't know", though the paper notes its rubric can still score such an answer below a fair answer containing hallucinations. The authors conclude that a handful of extra hallucination-specific benchmarks will not fix this, because the dominant leaderboards keep teaching models to bluff.

### 5. The proposed fix: explicit confidence targets
**What it does:** Changes the incentive without changing the model.
**How it works:** State a confidence threshold t in the prompt or rubric: answer only if you are more than t confident, since a wrong answer costs t/(1-t) points.

| Threshold t | Penalty per wrong answer |
|---|---|
| 0.5 | 1 |
| 0.75 | 2 |
| 0.9 | 9 |

A "behaviourally calibrated" model would then abstain whenever its chance of being right falls below the stated target. The authors want such targets written into mainstream evaluations rather than added as a separate test.

---

## Key Results

This is mainly a theory paper, so its "results" are theorems and a benchmark audit rather than a new score.

- **A bound linking generation to classification:** generative error is roughly at least twice the IIV misclassification rate.
- **A floor for arbitrary facts:** the hallucination rate is lower-bounded by roughly the singleton rate.
- **A proof that binary grading never rewards abstention.**
- **Illustrative failures:** DeepSeek-V3, asked for Adam Kalai's birthday, gave three different wrong dates across attempts. Models asked to count the Ds in "DEEPSEEK" gave wrong answers (2, 3 and sometimes higher), while reasoning models counted correctly. A table shows wrong dissertation titles and years from three popular models. These are anecdotes, not a measured benchmark.
- **A benchmark audit:** the evaluations above are almost all binary. The paper also notes that Humanity's Last Exam reports a high calibration error, but argues calibration error is not a proper hallucination metric, since a model that hallucinated every time could still score zero calibration error.

---

## Why This Was Revolutionary

- **It gave hallucination a statistical origin story,** linking it to learnability rather than to a defect to be patched.
- **It put the blame partly on evaluation,** a socio-technical cause the field controls directly.
- **It made a testable, cheap proposal:** change the scoring rule, not the architecture.
- **It separated kinds of error.** Arbitrary facts (birthdays) are unlearnable from a single mention; patterned errors (spelling, arithmetic) are failures of model capacity and shrink with better models. This explains why some hallucinations vanish with scale and others do not.

---

## Real-World Impact

The paper arrived from inside a major lab, and its message, that models should be rewarded for admitting uncertainty, aligns with a visible product direction: refusing or hedging rather than answering when unsure. It reframes the debate about leaderboards, adding a hallucination angle to the older worry that benchmarks reward the wrong behaviour. Related summaries in this collection cover the training stages it critiques: [InstructGPT / RLHF](../../language-models/05-instructgpt-rlhf/summary.md), [PPO](../63-ppo/summary.md) and [RLVR](../39-rlvr/summary.md), where binary verifiable rewards face the same incentive. [RAG](../13-rag/summary.md) attacks hallucination from the retrieval side.

---

## Key Takeaways for Practitioners

1. **Read your eval's scoring rule before reading its scores.** If "I don't know" earns zero, you are measuring and training bluffing.
2. **Tell models the stakes.** Prompts or graders that state a confidence threshold and a wrong-answer penalty give the model a reason to abstain.
3. **Expect irreducible error on rare facts.** Facts seen once in training cannot be reliably recalled; use retrieval or tools for them.
4. **Separate hallucination types** when debugging: unlearnable facts, model-capacity errors, and bad training data need different fixes.
5. **Calibration error is not a hallucination rate.** Track both.

---

## Limitations & Future Directions

The authors list their own caveats: the framework considers only plausible strings and treats errors as binary; open-ended generation is hard to score; retrieval and search reduce hallucination but do not remove the grading incentive; some errors depend on context outside the prompt and response; and uncertainty can be expressed in many ways beyond "I don't know", while the paper studies only the top-level decision of what to say.

Beyond those, note that the paper is an argument about incentives with limited direct experiments. Whether rewriting benchmark rubrics changes real model behaviour at scale, and whether a model trained with penalties stays useful rather than over-refusing, are open questions as of this summary. Treat the 20% singleton example as an illustration of the bound, not a measured rate.

---

## Further Reading

- **Original Paper:** https://arxiv.org/abs/2509.04664
- **Good-Turing estimation and missing mass:** background on why once-seen items estimate the unseen
- **Related in this collection:** [Chain-of-Thought](../09-chain-of-thought/summary.md), [Scaling Laws](../12-scaling-laws/summary.md)

---

## Citation

```bibtex
@article{kalai2025hallucinate,
  title={Why Language Models Hallucinate},
  author={Kalai, Adam Tauman and Nachum, Ofir and Vempala, Santosh S. and Zhang, Edwin},
  journal={arXiv preprint arXiv:2509.04664},
  year={2025}
}
```

<!-- related:start -->

---

## Related in This Collection

- [Training Language Models to Follow Instructions with Human Feedback (InstructGPT)](../../language-models/05-instructgpt-rlhf/summary.md)
- [Chain-of-Thought Prompting Elicits Reasoning in Large Language Models](../../techniques/09-chain-of-thought/summary.md)
- [Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks (RAG)](../../techniques/13-rag/summary.md)
- [DeepSeek-V3 Technical Report](../../language-models/27-deepseek-v3/summary.md)
- [RLVR: Reinforcement Learning from Verifiable Rewards](../../techniques/39-rlvr/summary.md)
- [Proximal Policy Optimization Algorithms (PPO)](../../techniques/63-ppo/summary.md)
- [SWE-bench: Can Language Models Resolve Real-World GitHub Issues?](../../techniques/84-swe-bench/summary.md)
- [Measuring Massive Multitask Language Understanding (MMLU)](../../techniques/137-mmlu/summary.md)

<!-- related:end -->
