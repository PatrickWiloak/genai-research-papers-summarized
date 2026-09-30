---
title: "Weak-to-Strong Generalization: Eliciting Strong Capabilities With Weak Supervision (Weak-to-Strong)"
slug: "128-weak-to-strong"
number: 128
category: "techniques"
authors: "Collin Burns, Pavel Izmailov, Jan Hendrik Kirchner, Bowen Baker, Leo Gao, Leopold Aschenbrenner, Yining Chen, Adrien Ecoffet, Manas Joglekar, Jan Leike, Ilya Sutskever, Jeff Wu (OpenAI)"
published: "December 2023 (ICML 2024)"
year: 2023
url: "https://arxiv.org/abs/2312.09390"
tags: ["alignment", "safety", "scaling"]
---

# Weak-to-Strong Generalization: Eliciting Strong Capabilities With Weak Supervision (Weak-to-Strong)

**Authors:** Collin Burns, Pavel Izmailov, Jan Hendrik Kirchner, Bowen Baker, Leo Gao, Leopold Aschenbrenner, Yining Chen, Adrien Ecoffet, Manas Joglekar, Jan Leike, Ilya Sutskever, Jeff Wu (OpenAI)
**Published:** December 2023 (ICML 2024)
**Paper:** [arxiv.org/abs/2312.09390](https://arxiv.org/abs/2312.09390)

---

## Why This Paper Matters

Today's alignment methods, above all [RLHF](../../language-models/05-instructgpt-rlhf/summary.md), rely on humans judging model outputs. That works while humans can tell good answers from bad ones. It stops working when models are better than their judges - when a model writes a million lines of code, or a proof, or a plan that no human can fully check. How do you supervise something smarter than you?

You cannot experiment on superhuman models yet. This paper's move was to build an **analogy you can run today**: use a *weak* model to supervise a *strong* one, and measure how much of the strong model's ability survives the weak supervision. If a GPT-2-level supervisor can elicit most of GPT-4's capability, that is encouraging; if the strong model just learns to imitate the weak one's mistakes, that is a warning.

- **A testable proxy for superalignment.** It turned a speculative question into an empirical research programme.
- **A clear metric:** "performance gap recovered" (PGR).
- **A mixed but informative answer.** Strong models generalise *beyond* their weak supervisors - often substantially - but nowhere near fully, and the result varies a lot by task.
- **It was the flagship paper of OpenAI's Superalignment team**, released alongside a $10 million grants programme for work on the problem.

**The insight:** a strong pretrained model may already *know* the right answer; supervision only needs to *elicit* it, not teach it. If so, even imperfect labels from a weaker supervisor could point the strong model at knowledge it already has.

---

## The Setup

```
 1. Train a WEAK model on ground truth              ->  weak performance
 2. Weak model labels a fresh dataset (with errors)
 3. Train a STRONG model on those weak labels       ->  weak-to-strong performance
 4. Train the STRONG model on ground truth          ->  strong ceiling

          weak-to-strong  -  weak
   PGR = ------------------------------
          strong ceiling  -  weak
```

- **PGR = 0%**: the strong model is no better than its supervisor - it just imitated the weak labels.
- **PGR = 100%**: weak supervision was as good as ground truth.

The weak and strong models came from the GPT-4 family's model series, spanning several orders of magnitude of compute.

### Three task settings
1. **NLP classification:** 22 standard benchmarks.
2. **Chess puzzles:** predict the best first move in lichess puzzles.
3. **ChatGPT reward modelling:** predict which of two responses human raters preferred, on OpenAI's own data.

---

## Key Results

### Naive fine-tuning already recovers something
- **NLP tasks:** PGR was nearly always positive. Even with the largest gaps between supervisor and student, more than 20% of the gap was recovered, often over 50% for the largest students.
- **Chess:** PGR was near zero with the weakest supervisors, but could exceed 40% when the gap was small. It *fell* as the student got larger.
- **Reward modelling:** the hardest case. PGR almost never exceeded 20%.

The strong student's main failure mode: it **learns to imitate the supervisor's errors**, overfitting to the weak labels.

### Simple methods help a lot in some settings
- **Auxiliary confidence loss** (NLP): encourage the strong model to make confident predictions even when they disagree with the weak labels. For the smallest supervisor and largest student, median PGR rose **from about 25% to nearly 80%**. The headline: a GPT-2-level supervisor could elicit performance **close to GPT-3.5 level** from a GPT-4-class model on these tasks.
- **Bootstrapping** (chess): go weak to slightly stronger to stronger in steps, rather than one big jump.
- **Unsupervised generative fine-tuning** (reward modelling): first fine-tune the strong model on relevant text without labels, making the target concept more salient, then apply weak supervision.

No single method worked across all three settings.

### Honest caveats from the authors
- **Early stopping on ground truth** added about 15 PGR points - but uses information a real superhuman setting would not have, so it is "cheating".
- **Disanalogies:** today's weak labels are errors a strong model can easily recognise as the weak model's typical mistakes ("imitation saliency"), whereas human errors on superhuman tasks may be harder to separate. And pretraining may already have exposed the strong model to human-level supervision on similar tasks ("pretraining leakage").

---

## Why This Was Revolutionary

- **It made scalable oversight empirical.** Before, most work was theoretical proposals.
- **It set the "elicitation" framing:** alignment as drawing out what a model already knows, rather than instilling new knowledge.
- **It showed the problem is real but not hopeless**, with concrete methods to improve on.

---

## Real-World Impact

- **OpenAI Superalignment Fast Grants** - a $10 million programme announced the same month, funding outside researchers on this and related problems.
- **Follow-up research** on eliciting latent knowledge without labels, including Anthropic-led work on Internal Coherence Maximization ("Unsupervised Elicitation of Language Models", 2025).
- **Safety frameworks** at frontier labs cite scalable oversight as an open problem; see [frontier safety frameworks](../../../explainers/policy/frontier-safety-frameworks.md).
- One author, Leopold Aschenbrenner, later wrote [Situational Awareness](../../essays/113-situational-awareness/summary.md), which lists this line of research as part of the "superalignment" plan.

---

## Key Takeaways for Practitioners

1. **Noisy labels can still teach a strong model**, if the model already represents the concept. Fine-tuning large models on imperfect labels often beats the labeller.
2. **Watch for imitation of labeller errors.** Regularise towards the model's own confident predictions where appropriate.
3. **Reward models are the weak link.** Of the three settings, preference modelling generalised worst - relevant to anyone training reward models.
4. **Step up gradually.** Intermediate models can bridge large capability gaps.

---

## Limitations & Future Directions

- **An analogy, not the real thing.** Weak models are not humans, and today's strong models are not superhuman.
- **Task-dependent results.** Methods that help on NLP classification did not transfer to reward modelling.
- **Classification-focused.** Real oversight involves open-ended generation and long-horizon behaviour, not just labels.
- **Complementary approaches** - debate, interpretability ([Sparse Autoencoders](../82-sparse-autoencoders/summary.md)), and red teaming ([Red Teaming LMs](../129-red-teaming-lms/summary.md)) - are needed alongside.

---

## Further Reading

- **Original Paper:** [arxiv.org/abs/2312.09390](https://arxiv.org/abs/2312.09390)
- **Unsupervised Elicitation of Language Models:** [arxiv.org/abs/2506.10139](https://arxiv.org/abs/2506.10139)
- **In this collection:** [InstructGPT / RLHF](../../language-models/05-instructgpt-rlhf/summary.md), [Constitutional AI](../../language-models/14-constitutional-ai/summary.md), [Sleeper Agents](../83-sleeper-agents/summary.md), [Situational Awareness](../../essays/113-situational-awareness/summary.md)

## Citation

```bibtex
@inproceedings{burns2024weak,
  title={Weak-to-Strong Generalization: Eliciting Strong Capabilities With Weak Supervision},
  author={Burns, Collin and Izmailov, Pavel and Kirchner, Jan Hendrik and Baker, Bowen and Gao, Leo and Aschenbrenner, Leopold and Chen, Yining and Ecoffet, Adrien and Joglekar, Manas and Leike, Jan and Sutskever, Ilya and Wu, Jeff},
  booktitle={International Conference on Machine Learning},
  pages={4971--5012},
  year={2024}
}
```

<!-- related:start -->

---

## Related in This Collection

- [Language Models are Few-Shot Learners (GPT-3)](../../language-models/04-gpt3-few-shot-learners/summary.md)
- [Training Language Models to Follow Instructions with Human Feedback (InstructGPT)](../../language-models/05-instructgpt-rlhf/summary.md)
- [Constitutional AI: Harmlessness from AI Feedback](../../language-models/14-constitutional-ai/summary.md)
- [GPT-4 Technical Report](../../language-models/36-gpt4/summary.md)
- [Language Models are Unsupervised Multitask Learners (GPT-2)](../../language-models/64-gpt2/summary.md)
- [Sparse Autoencoders and Monosemanticity: Reading the Features Inside a Model](../../techniques/82-sparse-autoencoders/summary.md)
- [Sleeper Agents: Training Deceptive LLMs that Persist Through Safety Training](../../techniques/83-sleeper-agents/summary.md)
- [Situational Awareness: The Decade Ahead (Situational Awareness)](../../essays/113-situational-awareness/summary.md)

<!-- related:end -->
