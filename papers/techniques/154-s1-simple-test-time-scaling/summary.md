---
title: "s1: Simple Test-Time Scaling (s1)"
slug: "154-s1-simple-test-time-scaling"
number: 154
category: "techniques"
authors: "Niklas Muennighoff, Zitong Yang, Weijia Shi, Xiang Lisa Li, Li Fei-Fei, Hannaneh Hajishirzi, Luke Zettlemoyer, Percy Liang, Emmanuel Candes, Tatsunori Hashimoto (Stanford University, Allen Institute for AI, University of Washington, Contextual AI)"
published: "January 2025 (arXiv v1, 31 January 2025; latest version v3, 1 March 2025)"
year: 2025
url: "https://arxiv.org/abs/2501.19393"
tags: ["reasoning", "scaling", "test-time-compute"]
---

# s1: Simple Test-Time Scaling (s1)

**Authors:** Niklas Muennighoff, Zitong Yang, Weijia Shi, Xiang Lisa Li, Li Fei-Fei, Hannaneh Hajishirzi, Luke Zettlemoyer, Percy Liang, Emmanuel Candes, Tatsunori Hashimoto (Stanford University, Allen Institute for AI, University of Washington, Contextual AI)
**Published:** January 2025 (arXiv v1, 31 January 2025; latest version v3, 1 March 2025)
**Paper Link:** [arxiv.org/abs/2501.19393](https://arxiv.org/abs/2501.19393)

---

## Why This Paper Matters

In September 2024, OpenAI's o1 showed that a model can get much better at hard reasoning problems by "thinking" longer at inference time, and that more thinking tokens gives better answers. This is test-time scaling (see [Test-Time Compute](../50-test-time-compute/summary.md)). But o1's recipe was private. The open question was: what is the minimum you need to reproduce it?

DeepSeek-R1 ([DeepSeek-R1](../../language-models/26-deepseek-r1/summary.md)) answered with a large reinforcement learning pipeline over hundreds of thousands of examples. s1 answers from the other end. It asks how little you can do: no reinforcement learning, only 1,000 carefully chosen examples for supervised finetuning, plus a trivial trick at inference time. The result is a 32B model that beats o1-preview on competition math and gets better as you let it think longer.

For a motivated beginner the lesson is that reasoning ability is mostly already inside a strong pretrained model; a small, well-chosen set of examples can unlock it, and a one-word intervention can control it.

## The Core Innovation

Two ingredients.

**1. s1K, a tiny reasoning dataset.** 1,000 questions chosen from a pool of 59,029, with reasoning traces and answers.

**2. Budget forcing, a decoding-time control of thinking length.**

```
Thinking too long?   Force it to stop:
   append the end-of-thinking token (optionally "Final Answer:")
   -> the model must answer now

Thinking too short?  Force it to continue:
   suppress the end-of-thinking token and append "Wait"
   -> the model re-checks its work and often fixes mistakes
```

An analogy: a student taking an exam. Budget forcing is a proctor who either says "time is up, write your answer" or taps the desk and says "wait, are you sure?" The second move turns out to make the student find and fix errors.

## Key Components Explained

### 1. Building s1K
**What it does:** Picks the 1,000 most useful training examples.
**How it works:** The pool has 59,029 questions from 16 sources; the largest is NuminaMATH (30,660 questions). The authors added two new sets: s1-prob (182 probability questions from Stanford statistics qualifying exams) and s1-teasers (23 hard brain-teasers). Reasoning traces and solutions were generated with the Gemini Flash Thinking API. Filtering then went through three criteria:
- **Quality:** drop API errors (54,116 left) and badly formatted samples (51,581 left).
- **Difficulty:** drop questions that Qwen2.5-7B-Instruct or Qwen2.5-32B-Instruct could already solve (24,496 left). Easy questions teach little.
- **Diversity:** sample across subject domains, favoring longer reasoning traces, giving 1,000 samples across 50 domains.

Samples overlapping with the test sets (8-gram match with MATH500, GPQA Diamond or AIME24) were removed.

### 2. Supervised finetuning
**What it does:** Teaches the model to produce long reasoning in a thinking section before answering.
**How it works:** Starting from Qwen2.5-32B-Instruct, the authors train on s1K for 5 epochs (batch size 16, 315 gradient steps, learning rate 1e-5). It takes about 26 minutes on 16 H100 GPUs, roughly 7 H100 GPU hours. Training on all 59K samples cost 394 H100 GPU hours, which is over 50 times more.

### 3. Budget forcing
**What it does:** Sets how long the model thinks at test time.
**How it works:** To cap thinking, append the end-of-thinking delimiter. To extend, ignore the model's attempt to stop and append "Wait". It can be repeated. Of the strings tried, "Wait" gave the best AIME24 result in the paper's ablation.

### 4. Alternatives the authors compare against
**What it does:** Shows why budget forcing is preferred.
**How it works:** Three variants control length by telling the model a budget in the prompt: token-conditional, step-conditional and class-conditional control. Rejection sampling keeps only samples that happen to be short enough. The comparison metrics are control (how often the length target is met), scaling slope, and best performance.

## Key Results

Main table (accuracy, percent), from the paper:

| Model | Examples | AIME24 | MATH500 | GPQA Diamond |
|---|---|---|---|---|
| o1-preview | n/a | 44.6 | 85.5 | 73.3 |
| o1 | n/a | 74.4 | 94.8 | 77.3 |
| DeepSeek-R1 | well over 800K | 79.8 | 97.3 | 71.5 |
| QwQ-32B | n/a | 50.0 | 90.6 | 54.5 |
| Qwen2.5-32B-Instruct | n/a | 26.7 | 84.0 | 49.0 |
| s1 without budget forcing | 1K | 50.0 | 92.6 | 56.6 |
| **s1-32B** | **1K** | **56.7** | **93.0** | **59.6** |

- The abstract reports that s1-32B exceeds o1-preview on competition math (MATH and AIME24) by up to 27%, and that budget forcing extrapolates AIME24 from 50% to 57%.
- It does not catch o1 or R1, and it trails o1-preview on GPQA Diamond.
- **Data ablations (AIME24 / MATH500 / GPQA):** 1K random 36.7 / 90.6 / 52.0; 1K diverse 26.7 / 91.2 / 54.6; 1K longest 33.3 / 90.4 / 59.6; full 59K 53.3 / 92.8 / 58.1; s1K 50.0 / 93.0 / 57.6. Combining all three criteria matters, and 59K examples barely beat 1K.
- **Test-time control methods:** budget forcing reached 100% control with a positive scaling slope of 15. Token-, step- and class-conditional control got 40%, 60% and 50% control. Rejection sampling got 100% control but a negative slope (-35): picking short generations selects easy questions, which inverts scaling.
- **Sequential vs parallel:** majority voting over many samples did not match budget forcing's sequential scaling; combining with REBASE tree search scaled better still but needs a reward model.
- **s1.1 (appendix):** regenerating traces with DeepSeek-R1 gives s1K-1.1. Without budget forcing it scores 94.4 MATH500, 60.6 GPQA and 56.7 AIME24 (the AIME25 score is 50.0). R1 traces improve on the original Gemini traces.

## Why This Was Revolutionary

- It showed that a sample-efficient recipe (1,000 examples, under 30 minutes of 16-GPU training) can produce a competitive open reasoning model.
- Budget forcing is simple enough to implement in a few lines of decoding code.
- It put data curation, not data quantity, at the center of reasoning finetuning, in line with the "less is more" lines of work for alignment.
- It made test-time scaling reproducible for academic labs, with the model, data and code all released.

## Real-World Impact

s1 became a standard cheap baseline for reasoning research. "Wait"-style prompting and thinking-length control became common in later open reasoning work. The paper also sharpens the interpretation of [DeepSeek-R1](../../language-models/26-deepseek-r1/summary.md): much of what reinforcement learning does may be eliciting latent ability, and distillation from stronger reasoners (s1 itself distills Gemini traces, s1.1 distills R1 traces) is a cheap shortcut. For the broader framing of spending compute at inference, see [Test-Time Compute](../50-test-time-compute/summary.md).

## Key Takeaways for Practitioners

1. If you have a strong base model, try a small, hard, diverse set of reasoning traces before reaching for reinforcement learning.
2. Filter by difficulty. Questions the base model already solves add little.
3. Control thinking length at decode time. "Wait" is a cheap extension; end-of-thinking forcing is a cheap cap.
4. Measure scaling slope, not just accuracy at one length.
5. Expect diminishing returns; and set the context window high enough.
6. Better teachers give better students (s1.1 over s1).

## Limitations & Future Directions

- Scaling flattens out at about six times the default compute, and over-suppressing the delimiter can cause repetitive loops.
- The context window is a hard limit: with 512 steps, 12 of 30 AIME24 questions exceeded it.
- Evaluation scores vary across runs with batch size, continued generation and tensor parallelism, so small differences are noise.
- The student relies on traces from stronger proprietary or other open models, so it does not remove the need for a strong teacher.
- Open directions include reinforcement learning to push past the plateau, better ways to extend thinking than a fixed string, and combining sequential and parallel scaling.

## Further Reading

- **Original Paper:** [arxiv.org/abs/2501.19393](https://arxiv.org/abs/2501.19393)
- [Test-Time Compute](../50-test-time-compute/summary.md)
- [DeepSeek-R1](../../language-models/26-deepseek-r1/summary.md)

## Citation

```bibtex
@article{muennighoff2025s1,
  title={s1: Simple test-time scaling},
  author={Muennighoff, Niklas and Yang, Zitong and Shi, Weijia and Li, Xiang Lisa and Fei-Fei, Li and Hajishirzi, Hannaneh and Zettlemoyer, Luke and Liang, Percy and Cand{\`e}s, Emmanuel and Hashimoto, Tatsunori},
  journal={arXiv preprint arXiv:2501.19393},
  year={2025}
}
```

<!-- related:start -->

---

## Related in This Collection

- [DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning](../../language-models/26-deepseek-r1/summary.md)
- [Qwen3: Technical Report](../../language-models/28-qwen3/summary.md)
- [OpenAI o1: Learning to Reason with Reinforcement Learning](../../language-models/31-openai-o1/summary.md)
- [Scaling LLM Test-Time Compute: The Theoretical Foundation for Reasoning Models](../../techniques/50-test-time-compute/summary.md)
- [ZeRO and Megatron-LM: How Trillion-Parameter Models Are Actually Trained](../../techniques/76-zero-megatron/summary.md)
- [Self-Consistency Improves Chain of Thought Reasoning in Language Models](../../techniques/77-self-consistency/summary.md)

<!-- related:end -->
