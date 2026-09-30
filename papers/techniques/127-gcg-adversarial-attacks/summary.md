---
title: "Universal and Transferable Adversarial Attacks on Aligned Language Models (GCG)"
slug: "127-gcg-adversarial-attacks"
number: 127
category: "techniques"
authors: "Andy Zou, Zifan Wang, Nicholas Carlini, Milad Nasr, J. Zico Kolter, Matt Fredrikson (Carnegie Mellon University, Center for AI Safety, Google DeepMind, Bosch Center for AI)"
published: "July 2023 (arXiv)"
year: 2023
url: "https://arxiv.org/abs/2307.15043"
tags: ["safety", "alignment", "evaluation"]
---

# Universal and Transferable Adversarial Attacks on Aligned Language Models (GCG)

**Authors:** Andy Zou, Zifan Wang, Nicholas Carlini, Milad Nasr, J. Zico Kolter, Matt Fredrikson (Carnegie Mellon University, Center for AI Safety, Google DeepMind, Bosch Center for AI)
**Published:** July 2023 (arXiv)
**Paper:** [arxiv.org/abs/2307.15043](https://arxiv.org/abs/2307.15043)

---

## Why This Paper Matters

By mid-2023, chat models had been trained with [RLHF](../../language-models/05-instructgpt-rlhf/summary.md) and similar methods to refuse harmful requests. "Jailbreaks" existed, but they were hand-written tricks - role-play scenarios, elaborate framings - that spread on social media and were patched one by one. The implicit hope was that safety training was robust and the jailbreaks were edge cases.

This paper undermined that hope. It showed that **an automated search can find short strings of seemingly random tokens that, appended to a request, make aligned models comply** - and that strings found on small open models **transfer** to large closed models the attacker never had access to.

- **Automated, not artisanal.** Attacks are found by optimisation, so they can be generated in bulk.
- **Universal.** A single suffix worked across many different harmful requests.
- **Transferable.** Suffixes optimised on open Vicuna and Guanaco models worked, at varying rates, on GPT-3.5, GPT-4, PaLM-2 and Claude.
- **It reframed jailbreaks as an adversarial-robustness problem**, linking LLM safety to a decade of research on adversarial examples in vision, where no complete defence has ever been found.

This summary explains the idea and its consequences at a conceptual level. It deliberately omits attack strings and implementation detail.

---

## The Problem: Is Refusal Training Robust?

Alignment training teaches a model to respond to certain requests with a refusal. Adversarial-examples research in computer vision had shown that neural networks can be pushed to wrong outputs by small, carefully chosen input changes that look meaningless to humans. Earlier attempts to do the same to language models had limited success, because text is discrete: you cannot nudge a word by a tiny amount the way you can nudge a pixel.

---

## The Core Idea: Optimise a Suffix

At a high level, the method (named **Greedy Coordinate Gradient**, GCG) works like this:

```
 [user request]  +  [adversarial suffix: a short sequence of tokens]
                              ^
                              |  repeatedly adjusted, one token at a time,
                              |  to make an affirmative reply more likely
```

1. **Choose a target.** Rather than aiming for a full harmful answer, aim for the model to *begin* its reply affirmatively ("Sure, here is..."). Once a model has started complying, it tends to continue.
2. **Use gradients to shortlist edits.** The model's gradients indicate which token swaps at each position of the suffix would most increase the probability of that affirmative start.
3. **Evaluate and keep the best.** Try a batch of candidate swaps with ordinary forward passes and keep whichever helps most. Repeat.
4. **Optimise across many prompts and models at once.** Training a single suffix against many different requests and several open models produces a suffix that is universal and more likely to transfer.

The method builds on earlier discrete-optimisation work (AutoPrompt, HotFlip) but searches over all suffix positions rather than one at a time, which the authors found much more effective.

---

## Key Results

Evaluated on **AdvBench**, a set the authors built of 500 harmful strings and 500 harmful behaviours.

### White-box (attacking the model the suffix was optimised on)

| Setting | Vicuna-7B | Llama-2-7B-Chat |
|---|---|---|
| Harmful behaviours, GCG | 100% | 88% |
| Harmful behaviours, prior method (AutoPrompt) | 96% | 36% |
| Harmful strings (exact match), GCG | 88% | 57% |
| Harmful strings, prior method | 25% | 3% |

### Transfer (suffixes optimised on open models, tested on others)
Using an ensemble of suffixes optimised on Vicuna and Guanaco models, the attack success rates reported were:

| Target model | Success rate |
|---|---|
| GPT-3.5 | 86.6% |
| PaLM-2 | 66.0% |
| Claude-1 | 47.9% |
| GPT-4 | 46.9% |
| Claude-2 | 2.1% |

The authors also observed that optimising for too long could **overfit** to the source models and reduce transfer.

### Disclosure
The authors shared their results with OpenAI, Google, Meta and Anthropic before publication.

---

## Why This Was Revolutionary

- **Jailbreaking became systematic.** Anyone with an open model and a GPU could generate attacks.
- **Open models became attack laboratories** for closed ones, via transfer.
- **It set expectations.** The authors drew the analogy to vision, where adversarial examples have persisted for a decade despite many proposed defences - suggesting that "patch each jailbreak" would not be a complete strategy.

---

## Defences That Followed

The paper triggered a wave of defensive research:

- **Perplexity filtering** - GCG suffixes look like gibberish, so a detector for unusually high-perplexity input catches many of them (Jain et al., 2023, "Baseline Defenses for Adversarial Attacks Against Aligned Language Models", which also tested paraphrasing and retokenisation).
- **SmoothLLM** (Robey et al., 2023) - randomly perturb characters in the input several times and aggregate the responses; adversarial suffixes are brittle to such perturbations.
- **Input and output classifiers** - separate models that screen prompts and responses, such as [Llama Guard](../../language-models/96-llama-guard/summary.md).
- **Circuit Breakers** (Zou et al., 2024) - training that interrupts the model's internal representations associated with harmful outputs, rather than relying only on refusals.
- **Constitutional Classifiers** (Anthropic, 2025) - classifiers trained on synthetic data generated from a written constitution, stress-tested with thousands of hours of red teaming; see [Red Teaming Language Models](../129-red-teaming-lms/summary.md).
- **Adversarial training** - including attack examples during safety training.

Attackers adapted too, producing more fluent automated jailbreaks that evade perplexity filters. The contest continues.

---

## Real-World Impact

- **Security framing for LLMs.** Jailbreaks and prompt injection became standard categories in AI security and red-teaming practice. The sibling repo covers the practitioner side in [AI threat modelling](https://github.com/PatrickWiloak/cloud-data-ai-security-zero-to-hero/blob/main/learn/concepts/ai-threat-modeling.md) and [guardrails and safety](https://github.com/PatrickWiloak/cloud-data-ai-security-zero-to-hero/blob/main/learn/concepts/guardrails-and-safety.md).
- **AdvBench** became a common evaluation set for attacks and defences.
- **Defence in depth** - model training plus classifiers plus monitoring - became the norm for production deployments.

---

## Key Takeaways for Practitioners

1. **Do not rely on refusal training alone.** Layer input and output filtering on top of the model.
2. **Assume transfer.** An attack developed on a public model may work on yours.
3. **Monitor for anomalous input**, such as long runs of unusual tokens.
4. **Scope what a jailbroken model could do.** Limit tools, data access and permissions so a successful jailbreak has little to act on - especially for agents.

---

## Limitations & Future Directions

- **Detectable.** Gibberish suffixes are easy to flag with simple filters.
- **Uneven transfer.** Rates varied widely by target, and some models (Claude-2 in the paper) were largely unaffected.
- **Compute cost.** Optimising suffixes requires gradient access to open models and GPU time.
- **Harm measurement.** Early evaluation often judged success by whether the model began complying, which can overstate how useful the resulting outputs are to an attacker.

---

## Further Reading

- **Original Paper:** [arxiv.org/abs/2307.15043](https://arxiv.org/abs/2307.15043)
- **Baseline Defenses:** Jain et al. 2023, [arxiv.org/abs/2309.00614](https://arxiv.org/abs/2309.00614)
- **SmoothLLM:** Robey et al. 2023, [arxiv.org/abs/2310.03684](https://arxiv.org/abs/2310.03684)
- **Circuit Breakers:** Zou et al. 2024, [arxiv.org/abs/2406.04313](https://arxiv.org/abs/2406.04313)
- **In this collection:** [Llama Guard](../../language-models/96-llama-guard/summary.md), [Constitutional AI](../../language-models/14-constitutional-ai/summary.md), [Red Teaming Language Models](../129-red-teaming-lms/summary.md), [Sleeper Agents](../83-sleeper-agents/summary.md)
- **Explainer:** [Frontier safety frameworks](../../../explainers/policy/frontier-safety-frameworks.md)

## Citation

```bibtex
@article{zou2023universal,
  title={Universal and Transferable Adversarial Attacks on Aligned Language Models},
  author={Zou, Andy and Wang, Zifan and Carlini, Nicholas and Nasr, Milad and Kolter, J. Zico and Fredrikson, Matt},
  journal={arXiv preprint arXiv:2307.15043},
  year={2023}
}
```

<!-- related:start -->

---

## Related in This Collection

- [Language Models are Few-Shot Learners (GPT-3)](../../language-models/04-gpt3-few-shot-learners/summary.md)
- [Training Language Models to Follow Instructions with Human Feedback (InstructGPT)](../../language-models/05-instructgpt-rlhf/summary.md)
- [Constitutional AI: Harmlessness from AI Feedback](../../language-models/14-constitutional-ai/summary.md)
- [GPT-4 Technical Report](../../language-models/36-gpt4/summary.md)
- [Sleeper Agents: Training Deceptive LLMs that Persist Through Safety Training](../../techniques/83-sleeper-agents/summary.md)
- [PaLM: Scaling Language Modeling with Pathways](../../language-models/94-palm/summary.md)
- [Llama Guard: LLM-based Input-Output Safeguard for Human-AI Conversations](../../language-models/96-llama-guard/summary.md)
- [Red Teaming Language Models with Language Models (LM Red Teaming)](../../techniques/129-red-teaming-lms/summary.md)

<!-- related:end -->
