---
title: "Emergent Abilities of Large Language Models (and the Mirage Rebuttal)"
slug: "81-emergent-abilities"
number: 81
category: "techniques"
authors: "Jason Wei, Yi Tay, Rishi Bommasani, Colin Raffel, Barret Zoph, Sebastian Borgeaud, Dani Yogatama, Maarten Bosma, Denny Zhou, Donald Metzler, Ed H. Chi, Tatsunori Hashimoto, Oriol Vinyals, Percy Liang, Jeff Dean, William Fedus (Google Research, Stanford, UNC Chapel Hill, DeepMind); rebuttal by Rylan Schaeffer, Brando Miranda, Sanmi Koyejo (Stanford)"
published: "June 2022 (TMLR 2022); rebuttal April 2023 (NeurIPS 2023, Outstanding Paper)"
year: 2022
url: "https://arxiv.org/abs/2206.07682"
tags: ["scaling", "evaluation", "reasoning"]
---

# Emergent Abilities of Large Language Models (and the Mirage Rebuttal)

**Authors:** Jason Wei, Yi Tay, Rishi Bommasani, Colin Raffel, Barret Zoph, Sebastian Borgeaud, Dani Yogatama, Maarten Bosma, Denny Zhou, Donald Metzler, Ed H. Chi, Tatsunori Hashimoto, Oriol Vinyals, Percy Liang, Jeff Dean, William Fedus (Google Research, Stanford, UNC Chapel Hill, DeepMind); rebuttal by Rylan Schaeffer, Brando Miranda, Sanmi Koyejo (Stanford)
**Published:** June 2022 (TMLR 2022); rebuttal April 2023 (NeurIPS 2023, Outstanding Paper)
**Papers:** [Emergent Abilities arxiv.org/abs/2206.07682](https://arxiv.org/abs/2206.07682) | [Are Emergent Abilities a Mirage? arxiv.org/abs/2304.15004](https://arxiv.org/abs/2304.15004)

---

## Why This Matters

This pair of papers is **the field's central argument about whether scaling produces surprises**. The first documented capabilities that appear abruptly at scale and are absent below it. The second argued the abruptness is largely an artifact of how we measure. Both are right about important things, and the disagreement is worth understanding because it determines how you plan, budget, and worry about AI systems.

- **[Scaling laws](../12-scaling-laws/summary.md) predict loss smoothly.** They do not obviously predict *capabilities*. Emergence is the claim that specific abilities switch on.
- **If emergence is real and unpredictable**, you cannot know what the next model will be able to do before training it - a serious problem for safety planning and for business forecasting.
- **If it is a metric artifact**, capability growth is smooth and predictable after all, and the surprises are ours, not the model's.
- **The practical answer sits in between**, and knowing which parts are which is genuinely useful.

**The insight (paper 1):** plot benchmark accuracy against training compute and some tasks show a flat line at random chance across three orders of magnitude, then a sharp rise. **The insight (paper 2):** those flat-then-sharp curves appear on **discontinuous metrics** like exact-match accuracy. Switch to a continuous metric on the same model outputs and the same "emergent" ability shows smooth, gradual improvement all along.

---

## The Problem: Scaling Laws Do Not Tell You What a Model Can Do

Kaplan's [scaling laws](../12-scaling-laws/summary.md) and [Chinchilla](../18-chinchilla/summary.md) predict cross-entropy loss as a function of compute, with remarkable precision. Loss is not what anyone cares about. Nobody funds a training run to reduce perplexity by 0.02; they fund it because the model will be able to do things.

The mapping from loss to capability was, and largely still is, unknown. The emergence paper made this concrete by documenting cases where the mapping looks discontinuous.

---

## The Core Innovation

### Paper 1: Documenting Emergence

Define an ability as **emergent** if it is not present in smaller models but is present in larger ones, and if the transition is not extrapolable from the smaller-model trend. The paper cataloged more than a hundred such cases across GPT-3, LaMDA, PaLM, and Chinchilla model families.

```
Accuracy on 3-digit arithmetic (schematic)

  50% |                                  ______
      |                                 /
      |                                /
      |                               /
  25% |                              /
      |                             /
   0% |__________________________ /
      +----------------------------------------
        10^20      10^22      10^23      10^24
                 training FLOPs

  Random chance for three orders of magnitude, then a rise.
  Extrapolating the flat part predicts nothing.
```

Canonical examples: multi-digit arithmetic, word unscrambling, transliteration, multi-task language understanding (MMLU), truthfulness on TruthfulQA, and - importantly - **the ability to benefit from [chain-of-thought prompting](../09-chain-of-thought/summary.md) at all**. Below a certain scale, CoT prompting makes models *worse*; above it, dramatically better. Instruction tuning shows the same pattern ([FLAN](../80-flan/summary.md) hurt small models, helped large ones).

### Paper 2: The Metric Explanation

Schaeffer et al. observed that the emergent tasks overwhelmingly used **discontinuous, all-or-nothing metrics**. Consider 5-digit addition scored by exact match:

```
Model must get ALL 5 digits right to score anything.

If per-digit accuracy improves smoothly:
  per-digit 0.70  ->  exact-match 0.70^5 = 0.17
  per-digit 0.80  ->  exact-match 0.80^5 = 0.33
  per-digit 0.90  ->  exact-match 0.90^5 = 0.59
  per-digit 0.95  ->  exact-match 0.95^5 = 0.77

Smooth linear improvement in the underlying skill
produces a curve that looks like a sudden switch.
```

Their evidence:
- Replacing exact match with **token edit distance** or **per-token accuracy** on the *same model outputs* turns sharp curves into smooth ones.
- Emergence claims cluster in a small number of metrics (exact string match, multiple choice grade) across BIG-Bench; tasks scored with continuous metrics almost never show emergence.
- They **induced apparent emergence in vision models** - architectures nobody claims are emergent - simply by scoring them with a discontinuous metric. This is the strongest part of the argument.

Their conclusion: emergence is often a property of the researcher's choice of metric, not of the model.

---

## Key Components Explained

### 1. What Both Papers Agree On
Underlying capability improves **smoothly** with scale. Neither paper disputes this. The disagreement is about whether the *observable* jumps carry independent information.

### 2. What Survives the Rebuttal
The rebuttal explains the *shape* of the curve; it does not eliminate the *practical* problem. If a model is at 15 percent exact-match on a task, it cannot do that task in production. When it crosses to 85 percent, it can. That transition is real, is what users experience, and remains hard to forecast even if the underlying per-token accuracy was smooth all along. **Smooth underlying progress plus a threshold is still a threshold.**

### 3. What Remains Genuinely Unexplained
Some phenomena are awkward for a pure metric explanation:
- **Chain-of-thought prompting reverses sign** with scale - it actively hurts small models and helps large ones. That is not a scoring artifact.
- **Instruction tuning reverses sign** the same way.
- **In-context learning itself** appears to arise with the formation of specific internal circuits (induction heads), which mechanistic interpretability work has tied to a visible phase change in the training loss curve. That is a mechanistic discontinuity, not a metric one.

### 4. The Forecasting Problem
GPT-4's technical report demonstrated that some capabilities can be predicted from smaller runs (they predicted coding performance from models 1,000x smaller) and that others - notably a task called Hindsight Neglect - moved in the opposite direction from the extrapolation. The honest state of the art: some capabilities are forecastable with careful metric design, some are not yet.

---

## Key Results

- **Paper 1:** cataloged 100+ tasks showing emergent behavior across BIG-Bench and standard benchmarks, in multiple model families, at broadly consistent compute thresholds (frequently around 10^22 to 10^24 FLOPs).
- **Paper 2:** showed that changing the metric removes emergence in the large majority of examined cases; that emergence claims concentrate in a few discontinuous metrics; and that apparent emergence can be manufactured in non-LLM systems by choosing such a metric.
- **Paper 2 won a NeurIPS 2023 Outstanding Paper award**, and its central methodological point - that your metric shapes your scientific claim - is now widely accepted.
- **Neither result** provided a method for predicting when a specific downstream capability will become usable, which remains open.

---

## Why This Matters for the Field

- **Safety planning.** If dangerous capabilities can appear abruptly, pre-deployment evaluation must probe for capabilities the current model does not have. Frontier safety frameworks and responsible scaling policies are built on this premise, and they are robust to the metric critique: a threshold crossing is still a threshold crossing.
- **Evaluation methodology changed.** The rebuttal is why serious evaluations now report continuous metrics, log-probabilities, and partial credit alongside exact match. "Measure with a continuous metric if you want to see the trend" is standard advice now.
- **Investment logic.** The case for very expensive training runs partly rests on the expectation of new capabilities, not just lower loss. How you read this debate changes how you read that case.
- **Scientific hygiene.** This is one of the cleanest examples in ML of a striking empirical claim being substantially explained by measurement choice, and it is worth internalizing as a general caution.

---

## Real-World Impact

- **Benchmark design.** BIG-Bench and successors added continuous scoring; evaluation harnesses report multiple metrics by default.
- **Frontier model reports** (GPT-4, Claude, Gemini) include scaling predictions and explicitly discuss which capabilities were and were not forecastable.
- **Pre-deployment dangerous-capability evaluations** at major labs test for abilities the model is not yet believed to have, precisely because of the emergence concern.
- **Mechanistic interpretability** picked up the unexplained residue: if capabilities arise from circuits forming, then finding those circuits ([sparse autoencoders](../82-sparse-autoencoders/summary.md), induction-head analysis) is the way to understand emergence properly.

---

## Key Takeaways for Practitioners

1. **Always report a continuous metric alongside your headline one.** Exact match alone hides progress and will make your own model's improvement look like nothing until it looks like magic.
2. **A flat benchmark does not mean no learning.** Check per-token accuracy, log-probability of the correct answer, or edit distance before concluding a model cannot do the task.
3. **Threshold effects are real for products** even when underlying progress is smooth. A 40 percent success rate and an 85 percent success rate are different products.
4. **Do not extrapolate capability from a flat line**, in either direction. Flat may mean "not learning" or "learning below the metric's resolution."
5. **Test the next model on tasks the current one fails.** Whatever the mechanism, capability thresholds get crossed between model generations.
6. **Beware benchmark saturation as the mirror image.** Once a metric saturates it stops distinguishing models, and progress becomes invisible again for the opposite reason.

---

## Limitations & Future Directions

- **Neither paper gives a predictive theory** mapping loss to downstream capability. This remains one of the most valuable open problems in the field.
- **Compute thresholds are family-specific.** Data quality, architecture, and post-training shift where transitions occur, so "10^23 FLOPs" is not a law.
- **Post-training complicates everything.** Modern capability jumps come substantially from RL and reasoning training, not raw pretraining scale, so scaling-curve framing captures less of the picture than it did in 2022.
- **The mechanistic account is nascent.** Tying capability onset to identifiable circuits is promising but has been demonstrated for only a few narrow abilities.

---

## Further Reading

- **Emergent Abilities of Large Language Models:** [arxiv.org/abs/2206.07682](https://arxiv.org/abs/2206.07682)
- **Are Emergent Abilities of Large Language Models a Mirage?:** [arxiv.org/abs/2304.15004](https://arxiv.org/abs/2304.15004)
- **In-context Learning and Induction Heads:** [transformer-circuits.pub/2022/in-context-learning-and-induction-heads](https://transformer-circuits.pub/2022/in-context-learning-and-induction-heads/index.html)
- **Beyond the Imitation Game (BIG-Bench):** [arxiv.org/abs/2206.04615](https://arxiv.org/abs/2206.04615)

## Citation

```bibtex
@article{wei2022emergent,
  title={Emergent Abilities of Large Language Models},
  author={Wei, Jason and Tay, Yi and Bommasani, Rishi and Raffel, Colin and Zoph, Barret and Borgeaud, Sebastian and Yogatama, Dani and Bosma, Maarten and Zhou, Denny and Metzler, Donald and others},
  journal={Transactions on Machine Learning Research},
  year={2022}
}

@inproceedings{schaeffer2023emergent,
  title={Are Emergent Abilities of Large Language Models a Mirage?},
  author={Schaeffer, Rylan and Miranda, Brando and Koyejo, Sanmi},
  booktitle={Advances in Neural Information Processing Systems (NeurIPS)},
  year={2023}
}
```

<!-- related:start -->

---

## Related in This Collection

- [Language Models are Few-Shot Learners (GPT-3)](../../language-models/04-gpt3-few-shot-learners/summary.md)
- [Chain-of-Thought Prompting Elicits Reasoning in Large Language Models](../../techniques/09-chain-of-thought/summary.md)
- [Scaling Laws for Neural Language Models](../../techniques/12-scaling-laws/summary.md)
- [Training Compute-Optimal Large Language Models (Chinchilla)](../../techniques/18-chinchilla/summary.md)
- [GPT-4 Technical Report](../../language-models/36-gpt4/summary.md)
- [FLAN: Finetuned Language Models Are Zero-Shot Learners (Instruction Tuning)](../../techniques/80-flan/summary.md)
- [Sparse Autoencoders and Monosemanticity: Reading the Features Inside a Model](../../techniques/82-sparse-autoencoders/summary.md)

<!-- related:end -->
