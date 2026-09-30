---
title: "Red Teaming Language Models with Language Models (LM Red Teaming)"
slug: "129-red-teaming-lms"
number: 129
category: "techniques"
authors: "Ethan Perez, Saffron Huang, Francis Song, Trevor Cai, Roman Ring, John Aslanides, Amelia Glaese, Nat McAleese, Geoffrey Irving (DeepMind; Perez also New York University)"
published: "February 2022 (EMNLP 2022)"
year: 2022
url: "https://arxiv.org/abs/2202.03286"
tags: ["safety", "evaluation", "alignment"]
---

# Red Teaming Language Models with Language Models (LM Red Teaming)

**Authors:** Ethan Perez, Saffron Huang, Francis Song, Trevor Cai, Roman Ring, John Aslanides, Amelia Glaese, Nat McAleese, Geoffrey Irving (DeepMind; Perez also New York University)
**Published:** February 2022 (EMNLP 2022)
**Paper:** [arxiv.org/abs/2202.03286](https://arxiv.org/abs/2202.03286)

---

## Why This Paper Matters

Before a model is released, someone has to find the ways it misbehaves. Traditionally that meant **human red teams**: people writing adversarial prompts by hand. Human red teaming is valuable but slow, expensive and limited by imagination - a team might write thousands of test cases, while real users will send millions.

This paper proposed using **one language model to red-team another**: generate test questions automatically, run them through the target model, and use a classifier to flag harmful replies. Applied to DeepMind's 280-billion-parameter chatbot, it surfaced tens of thousands of failures - offensive replies, leaked training data, real phone numbers and email addresses - that manual testing had not found.

- **Scale.** Hundreds of thousands of test cases instead of a few thousand.
- **A menu of generation methods**, from simple prompting to reinforcement learning, trading off diversity against difficulty.
- **New categories of harm** beyond offensiveness, including privacy leakage and distributional bias.
- **It became the template** for automated red teaming, now standard practice before frontier model releases.

**The insight:** the capabilities that make a language model dangerous - fluency, breadth, creativity - also make it a good generator of test cases. Point one model at another and you can search the space of inputs far more widely than people can.

---

## The Setup

```
  red LM  --generates-->  test questions
                              |
                              v
                        target LM (Dialogue-Prompted Gopher, 280B)
                              |
                              v
                     replies  --->  classifier: harmful?  --->  failures found
```

- **Target:** Dialogue-Prompted Gopher (DPG), a 280B-parameter model prompted to act as a chatbot.
- **Red LM:** also a Gopher model.
- **Classifier:** a 1.4B-parameter Gopher fine-tuned to detect offensive replies, trained on the Bot-Adversarial Dialogue dataset.

---

## Four Ways to Generate Test Cases

1. **Zero-shot:** prompt the red LM with something like "List of questions to ask someone:" and sample many completions. Diverse but mostly easy.
2. **Stochastic few-shot:** put previously successful test cases into the prompt as examples, sampled in proportion to how harmful they proved. More likely to find failures.
3. **Supervised learning:** fine-tune the red LM on successful test cases.
4. **Reinforcement learning:** train the red LM to maximise the classifier's harm score, with a KL penalty to keep it close to its starting distribution so it does not collapse to a single trick.

There is a clear **trade-off**: methods that find more failures produce less diverse test cases (the authors measure diversity with Self-BLEU). Both matter - diversity finds new kinds of failure; difficulty finds more of the same kind.

---

## Key Results

### Offensive replies
- **Zero-shot:** 0.5 million generated questions produced offensive replies **3.7% of the time**, which is **18,444 failures**.
- **Reinforcement learning:** with a weak KL penalty, over **40%** of test cases elicited offensive replies - but diversity collapsed; in one setting, 78% of test cases contained the word "invisible", showing the red LM had found one narrow exploit.
- **Coverage beyond human data:** of the top 100 noun phrases most associated with offensive replies, **37 did not appear** in the human-written Bot-Adversarial Dialogue data - automated red teaming found topics humans had not probed.

### Data leakage
- **1,709 replies** reproduced 13-word sequences verbatim from the training data. Of those, **821** were 13-grams that appear only once in the training corpus - evidence of memorisation, not common phrases.

### Contact information
- Among 100,000 replies, **3,206 contained phone numbers** (2,790 unique), of which **479 appeared in the training data**.
- **11,683 replies contained email addresses**, of which **749** were copied verbatim from training data.

### Other findings
- **Distributional bias:** the target model produced more negative replies about some demographic groups than others.
- **Multi-turn red teaming:** in generated conversations, offensive replies became more likely once earlier turns had gone offensive - harm compounds over a dialogue.

---

## Why This Was Revolutionary

- **Red teaming became scalable and repeatable.** A method, not just a team.
- **It broadened what "harm" means in testing**: privacy leakage and memorisation alongside toxicity.
- **It named the dynamics** - the diversity-difficulty trade-off, and the offence-defence asymmetry, where attackers need one success and defenders must cover everything.

---

## Real-World Impact

- **Pre-release testing** at major labs now mixes human experts with automated generation of attacks, a practice this paper helped establish.
- **Automated attack research** grew from it, including gradient-based methods like [GCG](../127-gcg-adversarial-attacks/summary.md).
- **Constitutional Classifiers** (Anthropic, January 2025, [arXiv 2501.18837](https://arxiv.org/abs/2501.18837)) show the defence side at scale: input and output classifiers trained on synthetic data generated from a written "constitution" of allowed and disallowed content. Across more than 3,000 estimated hours of human red teaming, no universal jailbreak was found that extracted detail comparable to an unguarded model across most target queries - at a cost of a 0.38 percentage-point rise in refusals on production traffic and 23.7% extra inference compute. A 2026 follow-up ([arXiv 2601.04603](https://arxiv.org/abs/2601.04603)) reported large compute reductions using cascaded classifiers and internal-activation probes.
- **Safety frameworks and regulation** increasingly require documented red teaming before release; see [frontier safety frameworks](../../../explainers/policy/frontier-safety-frameworks.md) and the [EU AI Act](../../../explainers/policy/eu-ai-act.md).

---

## Key Takeaways for Practitioners

1. **Automate the breadth, keep humans for depth.** Generated tests cover the space; experts find subtle, high-severity failures.
2. **Measure diversity as well as hit rate.** A red-teaming method that finds one bug a thousand times is not doing its job.
3. **Test for leakage, not just toxicity.** Memorised personal data is a real, measurable risk.
4. **Test conversations, not only single prompts.** Failures compound across turns.
5. **Your classifier is part of the system under test.** Its blind spots become the red team's blind spots.

---

## Limitations & Future Directions

- **Bounded by the classifier.** Harms the classifier cannot detect go unfound.
- **RL red teams collapse** onto narrow exploits without careful regularisation.
- **Single target model.** Results describe one 2022 chatbot.
- **Discovery, not repair.** Finding failures is the first step; fixing them without new failures is harder.

---

## Further Reading

- **Original Paper:** [arxiv.org/abs/2202.03286](https://arxiv.org/abs/2202.03286)
- **Constitutional Classifiers:** [arxiv.org/abs/2501.18837](https://arxiv.org/abs/2501.18837)
- **In this collection:** [GCG Adversarial Attacks](../127-gcg-adversarial-attacks/summary.md), [Llama Guard](../../language-models/96-llama-guard/summary.md), [Constitutional AI](../../language-models/14-constitutional-ai/summary.md), [Sleeper Agents](../83-sleeper-agents/summary.md), [Weak-to-Strong Generalization](../128-weak-to-strong/summary.md)

## Citation

```bibtex
@inproceedings{perez2022red,
  title={Red Teaming Language Models with Language Models},
  author={Perez, Ethan and Huang, Saffron and Song, Francis and Cai, Trevor and Ring, Roman and Aslanides, John and Glaese, Amelia and McAleese, Nat and Irving, Geoffrey},
  booktitle={Proceedings of the 2022 Conference on Empirical Methods in Natural Language Processing},
  pages={3419--3448},
  year={2022}
}
```

<!-- related:start -->

---

## Related in This Collection

- [Constitutional AI: Harmlessness from AI Feedback](../../language-models/14-constitutional-ai/summary.md)
- [Sleeper Agents: Training Deceptive LLMs that Persist Through Safety Training](../../techniques/83-sleeper-agents/summary.md)
- [Llama Guard: LLM-based Input-Output Safeguard for Human-AI Conversations](../../language-models/96-llama-guard/summary.md)
- [Universal and Transferable Adversarial Attacks on Aligned Language Models (GCG)](../../techniques/127-gcg-adversarial-attacks/summary.md)
- [Weak-to-Strong Generalization: Eliciting Strong Capabilities With Weak Supervision (Weak-to-Strong)](../../techniques/128-weak-to-strong/summary.md)

<!-- related:end -->
