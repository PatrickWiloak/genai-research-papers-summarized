---
title: "In-context Learning and Induction Heads (Induction Heads)"
slug: "126-induction-heads"
number: 126
category: "techniques"
authors: "Catherine Olsson, Nelson Elhage, Neel Nanda, Nicholas Joseph, Nova DasSarma, Tom Henighan, Ben Mann, Amanda Askell, Yuntao Bai, Anna Chen, Tom Conerly, Dawn Drain, Deep Ganguli, Zac Hatfield-Dodds, Danny Hernandez, Scott Johnston, Andy Jones, Jackson Kernion, Liane Lovitt, Kamal Ndousse, Dario Amodei, Tom Brown, Jack Clark, Jared Kaplan, Sam McCandlish, Chris Olah (Anthropic)"
published: "March 2022 (Transformer Circuits Thread; arXiv September 2022)"
year: 2022
url: "https://arxiv.org/abs/2209.11895"
tags: ["interpretability", "attention", "transformers"]
---

# In-context Learning and Induction Heads (Induction Heads)

**Authors:** Catherine Olsson, Nelson Elhage, Neel Nanda, Nicholas Joseph, Nova DasSarma, Tom Henighan, Ben Mann, Amanda Askell, Yuntao Bai, Anna Chen, Tom Conerly, Dawn Drain, Deep Ganguli, Zac Hatfield-Dodds, Danny Hernandez, Scott Johnston, Andy Jones, Jackson Kernion, Liane Lovitt, Kamal Ndousse, Dario Amodei, Tom Brown, Jack Clark, Jared Kaplan, Sam McCandlish, Chris Olah (Anthropic)
**Published:** March 2022 (Transformer Circuits Thread; arXiv September 2022)
**Paper:** [arxiv.org/abs/2209.11895](https://arxiv.org/abs/2209.11895)

---

## Why This Paper Matters

[GPT-3](../../language-models/04-gpt3-few-shot-learners/summary.md) made **in-context learning** famous: show a model a few examples in its prompt and it picks up the pattern, with no change to its weights. It is arguably the most useful thing language models do - and in 2022 nobody could say *how* it worked inside the network.

This paper offered the first concrete mechanistic answer. It identified a small circuit, the **induction head**, that looks back for an earlier occurrence of the current token and copies what came after it. It then gathered six lines of evidence that induction heads are responsible for a large share of in-context learning, from tiny attention-only models up to 13B-parameter Transformers.

- **A capability traced to a mechanism.** Most claims about what models "learn" are behavioural. This one points at specific attention heads.
- **A phase change you can see.** Early in training, every model the authors tested undergoes an abrupt transition: induction heads form, in-context learning jumps, and a visible bump appears in the loss curve.
- **A founding result for mechanistic interpretability**, the research programme that later produced [sparse autoencoders](../82-sparse-autoencoders/summary.md) and circuit tracing.

**The insight:** in-context learning at its simplest is pattern completion - "last time I saw A, B came next, so predict B". That operation can be built from two attention heads working together, and it generalises from literal copying to fuzzy, abstract pattern matching.

---

## The Mechanism: Two Heads Working Together

An induction head completes the pattern:

```
... [A] [B] ... [A]  ->  predict [B]
```

It needs two properties:
1. **Prefix matching:** from the current token `A`, attend to the token that came *right after* an earlier `A`.
2. **Copying:** raise the probability of whatever token it attended to (`B`).

A single head cannot do prefix matching on its own, because it needs to know "what came before each earlier token". So an induction head relies on a **previous-token head** in an earlier layer, which writes "my predecessor was A" into each position. The induction head then searches for positions whose predecessor matches the current token.

```
Layer 1: previous-token head   - at each position, record "the token before me"
Layer 2: induction head        - find positions whose "token before me" == current token,
                                  attend there, copy that token forward
```

This composition is why **one-layer attention-only models never form induction heads**, and two-layer ones do. The groundwork came from the companion paper, *A Mathematical Framework for Transformer Circuits* (Elhage et al., 2021).

---

## Measuring In-Context Learning

The authors needed a model-agnostic number. They defined an **in-context learning score**: the loss on the 500th token of a context minus the average loss on the 50th token. If a model uses context well, later tokens are easier to predict, so the score is negative, and more negative means more in-context learning. The authors note the choice of token indices is arbitrary but convenient.

---

## The Evidence

They studied **34 models** and ran more than **50,000 head ablations**:
- small models with 1 to 6 layers, attention-only and with MLPs, trained on about 10B tokens;
- a series of full-scale models from 4 layers and 13M parameters up to 40 layers and 13B parameters;
- modified "smeared key" architectures designed to make induction heads easier to form.

### The phase change
Somewhere around **2.5 to 5 billion tokens** into training, every multi-layer model underwent a sudden change: induction heads appeared, and the in-context learning score jumped from under 0.15 nats to about 0.4 nats. After that it stayed roughly **constant regardless of model size** - an "unexplained curiosity" in the authors' words. The phase change shows up as a bump in the training loss, described as the only place in training where the loss curve is not convex.

### Six arguments
1. **Co-occurrence:** induction heads and the jump in in-context learning appear at the same point in training, in every model.
2. **Co-perturbation:** changing the architecture so induction heads can form in one layer ("smeared keys") moves the in-context learning jump with them, including into one-layer models.
3. **Direct ablation:** in small attention-only models, removing induction heads removes "almost all" in-context learning. Removing other heads does not.
4. **Generality:** the same heads also do fuzzy and abstract pattern completion - nearest-neighbour matching, and translating a repeated phrase across languages.
5. **Mechanistic plausibility:** the circuit can be read directly off the weights in small models, and it does what the theory says.
6. **Continuity:** the behaviour of small models, where the evidence is causal, carries smoothly into large ones.

### How strong is the evidence?
The authors are careful: for small attention-only models the evidence is **causal**; for large models with MLPs it is **correlational**. They describe it as "only the beginnings of evidence" for the large-model claim.

---

## Why This Was Revolutionary

- **It connected a training-dynamics phenomenon to a circuit.** The loss bump, the capability jump and the mechanism line up.
- **It showed interpretability could explain something important**, not just toy behaviours.
- **It gave the field a vocabulary** - previous-token heads, induction heads, phase changes - used throughout later work.

---

## Real-World Impact

- **Mechanistic interpretability** as a field built on this and the Mathematical Framework paper; [sparse autoencoders](../82-sparse-autoencoders/summary.md) extended the approach from heads to features.
- **Emergence debates.** The paper's phase change is often cited alongside [Emergent Abilities](../81-emergent-abilities/summary.md) as an example of a sudden capability change with an identifiable cause.
- **Follow-up work** found in-context learning can be *transient*: Singh et al. (2023, "The Transient Nature of Emergent In-Context Learning in Transformers") showed some training setups lose it with longer training, and that regularisation can preserve it.

---

## Key Takeaways for Practitioners

1. **Repetition in the prompt is powerful.** Induction-style copying is why few-shot examples and consistent formatting work so well.
2. **It is also a failure mode.** The same mechanism makes models copy mistakes, echo earlier text and get stuck in loops.
3. **Capabilities can arrive suddenly in training.** Smooth average loss can hide abrupt internal changes.
4. **Interpretability claims need the right evidence level.** Distinguish causal results in small models from correlations in large ones.

---

## Limitations & Future Directions

- **Mostly correlational at scale.** The strongest evidence is in small models.
- **In-context learning is broader than induction.** Sophisticated in-context behaviour in large models likely involves many other mechanisms.
- **The constant-size puzzle** - why in-context learning after the phase change barely depends on model size - is unexplained.
- **The score is a proxy.** Loss improvement from token 50 to 500 is not the same as few-shot task performance.

---

## Further Reading

- **Original Paper:** [transformer-circuits.pub/2022/in-context-learning-and-induction-heads](https://transformer-circuits.pub/2022/in-context-learning-and-induction-heads/index.html) and [arxiv.org/abs/2209.11895](https://arxiv.org/abs/2209.11895)
- **A Mathematical Framework for Transformer Circuits:** [transformer-circuits.pub/2021/framework](https://transformer-circuits.pub/2021/framework/index.html)
- **The Transient Nature of Emergent In-Context Learning:** [arxiv.org/abs/2311.08360](https://arxiv.org/abs/2311.08360)
- **In this collection:** [GPT-3](../../language-models/04-gpt3-few-shot-learners/summary.md), [Emergent Abilities](../81-emergent-abilities/summary.md), [Sparse Autoencoders](../82-sparse-autoencoders/summary.md), [Attention Is All You Need](../../architectures/01-attention-is-all-you-need/summary.md)
- **Explainer:** [Do LLMs reason?](../../../explainers/open-questions/do-llms-reason.md)

## Citation

```bibtex
@article{olsson2022context,
  title={In-context Learning and Induction Heads},
  author={Olsson, Catherine and Elhage, Nelson and Nanda, Neel and Joseph, Nicholas and DasSarma, Nova and Henighan, Tom and Mann, Ben and Askell, Amanda and Bai, Yuntao and Chen, Anna and others},
  journal={Transformer Circuits Thread},
  year={2022}
}
```

<!-- related:start -->

---

## Related in This Collection

- [Attention Is All You Need](../../architectures/01-attention-is-all-you-need/summary.md)
- [Language Models are Few-Shot Learners (GPT-3)](../../language-models/04-gpt3-few-shot-learners/summary.md)
- [Emergent Abilities of Large Language Models (and the Mirage Rebuttal)](../../techniques/81-emergent-abilities/summary.md)
- [Sparse Autoencoders and Monosemanticity: Reading the Features Inside a Model](../../techniques/82-sparse-autoencoders/summary.md)

<!-- related:end -->
