---
title: "Measuring Massive Multitask Language Understanding (MMLU)"
slug: "137-mmlu"
number: 137
category: "techniques"
authors: "Dan Hendrycks, Collin Burns, Steven Basart, Andy Zou, Mantas Mazeika, Dawn Song, Jacob Steinhardt (UC Berkeley, Columbia University, University of Chicago, UIUC)"
published: "September 2020 (ICLR 2021)"
year: 2020
url: "https://arxiv.org/abs/2009.03300"
tags: ["evaluation", "benchmarks"]
---

# Measuring Massive Multitask Language Understanding (MMLU)

**Authors:** Dan Hendrycks, Collin Burns, Steven Basart, Andy Zou, Mantas Mazeika, Dawn Song, Jacob Steinhardt (UC Berkeley, Columbia University, University of Chicago, UIUC)
**Published:** September 2020 (ICLR 2021)
**Paper:** [arxiv.org/abs/2009.03300](https://arxiv.org/abs/2009.03300)

---

## Why This Matters

For about three years, MMLU was **the number**. From GPT-3 to GPT-4, Llama, Claude and Gemini, nearly every model launch led with an MMLU score, and "MMLU percent" became shorthand for how much a model knows. It was the first benchmark to test a language model across the breadth of a university education: 57 subjects from elementary maths to professional law, all in one multiple-choice format.

- **Broad by design.** 57 subjects across the humanities, social sciences, STEM and "other" (medicine, business, and so on), with 15,908 questions in total.
- **Hard at launch.** The largest GPT-3 scored 43.9 percent, where random guessing scores 25 percent; smaller GPT-3 models were at chance.
- **Then saturated.** GPT-4 reported 86.4 percent in 2023, and by early 2025 frontier models were above 90 percent, around the paper's own estimate of expert-level human performance.
- **Then audited.** A 2024 study found that about 6.5 percent of MMLU questions contain errors, which puts a ceiling on how meaningful the top scores are.

**The insight:** earlier benchmarks such as GLUE and SuperGLUE tested linguistic skill, and models passed human level within about a year. Models pretrained on the whole internet should be judged on what they **know and can apply** across many fields, the way we examine students: many subjects, many difficulty levels, evaluated zero-shot or few-shot without task-specific training.

---

## The Problem: Benchmarks Were Too Narrow and Too Easy

By 2020, [GPT-3](../../language-models/04-gpt3-few-shot-learners/summary.md) had shown that very large models could do tasks from a handful of examples in the prompt. The benchmarks of the day could not tell whether they had real knowledge:

- **GLUE and SuperGLUE** were saturated by fine-tuned models at or above human baselines.
- **Commonsense benchmarks** (HellaSwag, PIQA and others) tested everyday reasoning, not expert knowledge.
- **Single-domain question sets** measured one field at a time, so no single number captured breadth.

The authors wanted a test where the gap between models and human experts was large, broad and easy to measure.

---

## The Core Innovation

Collect real exam and practice questions from freely available sources across 57 subjects, all as four-option multiple choice, and evaluate models with **no fine-tuning on the test's subjects**, using zero or five examples in the prompt.

```
Format (5-shot)

  The following are multiple choice questions about high school biology.

  [5 solved example questions from the dev set]

  Question: <new question>
  (A) ...  (B) ...  (C) ...  (D) ...
  Answer:

  Score = fraction answered correctly, averaged over subjects.
  Random guessing = 25%.
```

| Split | Size | Use |
|---|---|---|
| Few-shot dev | 5 per subject | The examples placed in the prompt |
| Validation | 1,540 | Hyperparameter selection |
| Test | 14,079 | Reported scores (at least 100 per subject) |

Subjects cover levels from "Elementary" through "High School" and "College" to "Professional" (for example, Professional Medicine draws on United States Medical Licensing Examination style questions, and Professional Psychology on licensing exam practice questions).

---

## Key Components Explained

### 1. Human Baselines
**What it does:** Sets the scale for "good".
**How it works:** Amazon Mechanical Turk workers, not specialists, scored **34.5 percent**. For experts, the authors took the 95th-percentile human score on each underlying exam where available, guessed where it was not, and estimated expert-level accuracy at about **89.8 percent**. That estimate later became the informal finish line.

### 2. Per-Subject Breakdown
**What it does:** Shows where a model is weak, not just how good it is on average.
**How it works:** Scores are reported per subject and per category. GPT-3's accuracy ranged from 69 percent (US Foreign Policy) to 26 percent (College Chemistry). The authors call this "lopsided": broad but shallow knowledge, near-random on calculation-heavy subjects and on socially important ones such as law and morality.

### 3. Calibration
**What it does:** Checks whether a model knows when it is wrong.
**How it works:** Compare the model's confidence to its accuracy per subject. GPT-3's zero-shot confidence was "only weakly related to its actual accuracy", with gaps of up to 24 points on some subjects. Calibration is less often quoted than accuracy but was one of the paper's more prescient warnings.

---

## Key Results (from the paper)

| Model | Average accuracy |
|---|---|
| Random baseline | 25.0% |
| RoBERTa | 27.9% |
| GPT-2 | 32.4% |
| GPT-3 Small / Medium / Large (few-shot) | 25.9% / 24.9% / 26.0% |
| **GPT-3 X-Large, 175B (few-shot)** | **43.9%** |
| UnifiedQA (11B, T5-based, fine-tuned on other QA data) | 48.9% |

- **Scale appeared to "switch on" performance.** Smaller GPT-3 models were at chance and only the largest was well above it. This was later cited as an example of [emergent abilities](../81-emergent-abilities/summary.md), though that framing has since been challenged.
- **Fine-tuning on other QA datasets helped a lot.** An 11B UnifiedQA model beat 175B GPT-3, an early sign that [instruction tuning](../80-flan/summary.md) would matter.
- **No model reached expert level on any subject.**

---

## Saturation

MMLU went from hard to nearly solved in about four years:

```
  MMLU progress (as reported by each source)

  2020  GPT-3 175B (few-shot)        43.9%   this paper
  2023  GPT-4                        86.4%   GPT-4 Technical Report
  2025  frontier models            > 90%     "Humanity's Last Exam" paper, Jan 2025
        expert-level estimate       ~89.8%   this paper
```

The authors of **[Humanity's Last Exam](https://arxiv.org/abs/2501.14249)** (January 2025, a large collaboration coordinated by the Center for AI Safety and Scale AI, which includes Hendrycks) opened their paper with the observation that "LLMs now achieve over 90% accuracy on popular benchmarks like MMLU, limiting informed measurement of state-of-the-art LLM capabilities."

Once scores cluster at the top, three problems dominate:

- **Contamination.** MMLU questions are public and widely copied, so they enter pretraining data.
- **Prompt sensitivity.** Small changes to the prompt template or answer extraction method move scores by several points, as much as the gaps between models.
- **Label noise.** When models are near the ceiling, wrong answer keys decide the ranking.

---

## The Label-Error Findings: MMLU-Redux

"[Are We Done with MMLU?](https://arxiv.org/abs/2406.04127)" (Gema et al., University of Edinburgh and collaborators, NAACL 2025) manually re-annotated 100 random questions from each of the 57 subjects, 5,700 in all, released as **MMLU-Redux**. Their error taxonomy:

```
  Type 1: question problems             Type 2: ground-truth problems
    1a  question unclear                   2a  no correct option
    1b  options unclear                    2b  multiple correct options
                                           2c  wrong answer key
```

Findings:

- An estimated **6.49 percent** of all MMLU questions contain errors.
- **Virology is the worst**: 57 percent of the analysed questions had errors, many with the wrong answer key.
- Several other subjects, including Logical Fallacies, College Chemistry, Professional Law, Business Ethics and Formal Logic, had error rates above 10 percent.
- **Rankings change** when models are scored only on correct questions. In Virology, a model ranked 16th on all questions ranked first on the correct ones.

At frontier accuracy levels, a 6.5 percent error rate means the last several points of "progress" can reflect agreeing with wrong answer keys as much as knowing more.

---

## MMLU-Pro

**[MMLU-Pro](https://arxiv.org/abs/2406.01574)** (Wang et al., University of Waterloo and collaborators, NeurIPS 2024 Datasets and Benchmarks track) is the most widely adopted harder successor:

- **12,032 questions across 14 disciplines**, about 57 percent drawn from MMLU (after removing trivial and noisy items) and the rest from STEM problem sites, TheoremQA and SciBench.
- **Ten options instead of four**, with extra distractors generated by GPT-4-Turbo and reviewed by experts, so random guessing drops to 10 percent.
- **More reasoning-heavy.** Chain-of-thought prompting helps on MMLU-Pro, which it mostly did not on MMLU.
- **Accuracy drops of 16 to 33 points** relative to MMLU; GPT-4o, the top model in the paper, scored 72.6 percent.
- **Less prompt-sensitive**: score variation across 24 prompt styles fell from 4-5 percent to about 2 percent.

MMLU-Pro became a standard line in 2024-2025 model reports (for example [DeepSeek-R1](../../language-models/26-deepseek-r1/summary.md)).

---

## Why This Was Revolutionary

- **Defined "knowledge" as a measurable quantity** for pretrained models: one number, many subjects, no task-specific training.
- **Anticipated the few-shot era.** Its protocol (zero-shot and 5-shot prompts, no fine-tuning) became the default way to evaluate base models.
- **Tracked a whole generation of scaling.** From 44 to about 90 percent, MMLU is one of the clearest records of what [scaling](../12-scaling-laws/summary.md) bought between 2020 and 2024.
- **Highlighted calibration and lopsidedness**, issues that still matter for trustworthy deployment.

---

## Real-World Impact

- **The standard launch metric** for GPT-4, Llama 2 and 3, Claude 3, Gemini, Mistral and nearly every other model from 2021 to 2024.
- **A template for successors**: MMLU-Pro, MMLU-Redux, multilingual MMLU variants, GPQA and Humanity's Last Exam all follow the same "exam across many fields" pattern with harder or cleaner questions.
- **A cautionary tale** about benchmark lifecycles, together with [SWE-bench](../84-swe-bench/summary.md) and [ARC](../138-arc-agi/summary.md): benchmarks saturate, leak and carry label noise, so they need successors and audits.

---

## Key Takeaways for Practitioners

1. **Do not choose between frontier models on MMLU.** Above about 85 percent, differences are within prompt sensitivity and label noise.
2. **Check the evaluation setup.** 0-shot vs 5-shot, chain-of-thought or not, and the answer-extraction method can each move scores by several points; numbers from different harnesses are not comparable.
3. **Prefer MMLU-Pro or MMLU-Redux** when you need a knowledge benchmark in 2026, and a domain-specific test when you need a decision.
4. **Look at per-subject scores.** Averages hide the lopsidedness the paper warned about.
5. **Measure calibration** if your application depends on the model knowing when it does not know.

---

## Limitations & Future Directions

- **Multiple choice only.** It tests recognition, not generating an answer or explaining it.
- **Contaminated by now.** The questions are all over the web.
- **Label errors** (about 6.5 percent, and far higher in some subjects).
- **US-centric content** in law, history and some professional subjects.
- **Saturated.** It no longer separates frontier models; successors such as MMLU-Pro, GPQA and Humanity's Last Exam took over that role.

---

## Further Reading

- **Original Paper:** [arxiv.org/abs/2009.03300](https://arxiv.org/abs/2009.03300)
- **Dataset and code:** [github.com/hendrycks/test](https://github.com/hendrycks/test)
- **Are We Done with MMLU? (MMLU-Redux):** [arxiv.org/abs/2406.04127](https://arxiv.org/abs/2406.04127)
- **MMLU-Pro:** [arxiv.org/abs/2406.01574](https://arxiv.org/abs/2406.01574)
- **Humanity's Last Exam:** [arxiv.org/abs/2501.14249](https://arxiv.org/abs/2501.14249)
- **In this collection:** [GPT-3](../../language-models/04-gpt3-few-shot-learners/summary.md), [GPT-4](../../language-models/36-gpt4/summary.md), [Emergent Abilities](../81-emergent-abilities/summary.md), [SWE-bench](../84-swe-bench/summary.md), [LLM-as-a-Judge](../85-llm-as-judge/summary.md), [phi-1](../../language-models/135-phi-1-textbooks/summary.md)

## Citation

```bibtex
@inproceedings{hendrycks2021measuring,
  title={Measuring Massive Multitask Language Understanding},
  author={Hendrycks, Dan and Burns, Collin and Basart, Steven and Zou, Andy and Mazeika, Mantas and Song, Dawn and Steinhardt, Jacob},
  booktitle={International Conference on Learning Representations},
  year={2021}
}
```

<!-- related:start -->

---

## Related in This Collection

- [Language Models are Few-Shot Learners (GPT-3)](../../language-models/04-gpt3-few-shot-learners/summary.md)
- [LLaMA 2: Open Foundation and Fine-Tuned Chat Models](../../language-models/17-llama2/summary.md)
- [DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning](../../language-models/26-deepseek-r1/summary.md)
- [GPT-4 Technical Report](../../language-models/36-gpt4/summary.md)
- [GPT-4o: The First Omni Model](../../language-models/40-gpt4o/summary.md)
- [Language Models are Unsupervised Multitask Learners (GPT-2)](../../language-models/64-gpt2/summary.md)
- [Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer (T5)](../../language-models/65-t5/summary.md)
- [FLAN: Finetuned Language Models Are Zero-Shot Learners (Instruction Tuning)](../../techniques/80-flan/summary.md)

<!-- related:end -->
