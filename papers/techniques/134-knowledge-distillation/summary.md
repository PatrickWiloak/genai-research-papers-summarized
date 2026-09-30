---
title: "Distilling the Knowledge in a Neural Network (Knowledge Distillation)"
slug: "134-knowledge-distillation"
number: 134
category: "techniques"
authors: "Geoffrey Hinton, Oriol Vinyals, Jeff Dean (Google)"
published: "March 2015 (NIPS 2014 Deep Learning Workshop)"
year: 2015
url: "https://arxiv.org/abs/1503.02531"
tags: ["efficiency", "distillation", "transfer-learning"]
---

# Distilling the Knowledge in a Neural Network (Knowledge Distillation)

**Authors:** Geoffrey Hinton, Oriol Vinyals, Jeff Dean (Google)
**Published:** March 2015 (NIPS 2014 Deep Learning Workshop)
**Paper:** [arxiv.org/abs/1503.02531](https://arxiv.org/abs/1503.02531)

---

## Why This Paper Matters

The best-performing model is often too big, too slow or too expensive to deploy. An ensemble of ten networks beats any one of them, but nobody wants to run ten networks per request. This short paper gave the standard answer: **train a small "student" model to imitate the full output distribution of the big "teacher", not just the correct labels.**

The idea is older - Caruana and colleagues compressed ensembles in 2006 - but Hinton, Vinyals and Dean gave it the name, the temperature trick and the clearest explanation of *why* it works. Nearly a decade later it underpins how the industry ships small models:

- **Small open models are routinely distilled from large ones.** The six distilled DeepSeek-R1 models are the most famous recent example.
- **"Dark knowledge" became a core concept.** A teacher's wrong-answer probabilities carry information about how classes relate, which hard labels throw away.
- **The temperature-softened softmax** from this paper is still the default distillation loss.

**The insight:** a trained model's output is richer than its top answer. When a digit classifier says an image is "2" with 0.9 probability, the fact that it gives "3" a little more than "7" tells you something about what that particular 2 looks like. Training on those soft probabilities transfers the teacher's way of generalising, not just its answers.

---

## The Problem: Training and Deployment Want Different Models

The authors compare models to insects that have a larval form suited to eating and an adult form suited to travelling. Training favours huge models and ensembles, because they extract the most structure from the data. Deployment favours small, fast models, because latency and cost dominate. The question is how to move what the big model learned into the small one.

Training the small model directly on the original labels does not work as well: it lacks the capacity to discover the same structure from hard labels alone.

---

## The Core Innovation: Soft Targets at High Temperature

### Softmax with temperature
A classifier turns logits `z` into probabilities with a softmax. Add a temperature `T`:

```
q_i = exp(z_i / T) / sum_j exp(z_j / T)

T = 1   ->  sharp: [0.98, 0.015, 0.005, ...]   almost a one-hot label
T = 5   ->  soft:  [0.55, 0.25,  0.12,  ...]   relative similarities visible
```

At normal temperature the teacher's small probabilities are so close to zero that they barely affect a loss. Raising `T` spreads probability mass so those relationships become a usable training signal.

### The distillation loss
Train the student with two terms:

```
loss = alpha * CE(student_soft(T), teacher_soft(T)) * T^2
     + (1 - alpha) * CE(student(T=1), true_label)
```

- The **soft term** matches the teacher's softened distribution at the same high temperature.
- The **hard term** keeps the student anchored to the true labels.
- Multiplying the soft term by **T squared** keeps its gradients on the same scale as the hard term when you change the temperature.

The authors also show that in the high-temperature limit, matching soft targets reduces to matching the logits themselves - connecting the method to earlier logit-matching approaches.

### Specialist models
The paper also proposes training **specialist** networks that each focus on a confusable subset of classes, alongside one generalist - a precursor to later mixture-of-experts thinking, tested on a very large internal image dataset.

---

## Key Results

**MNIST**
- A large, heavily regularised network made **67 test errors**.
- A small network trained normally made **146 errors**.
- The same small network trained on the large network's soft targets made **74 errors** - most of the gap closed.
- Striking demonstration: when all examples of the digit **3** were removed from the transfer set, the distilled student still classified **98.6% of test 3s correctly** once a bias term was adjusted. It had learned what a 3 is purely from how the teacher's probabilities for *other* digits resembled 3s.

**Speech recognition (Android voice search acoustic model)**
- An ensemble of 10 models improved frame accuracy from **58.9% to 61.1%**.
- A single distilled model reached **60.8%** - most of the ensemble's gain in one model's cost.
- Trained on only **3% of the data**, a model using soft targets reached **57.0%** frame accuracy against **44.5%** with hard labels, and did not overfit - soft targets act as a strong regulariser.

---

## Why This Was Revolutionary

- **It separated "the model you train" from "the model you ship".**
- **It named and explained dark knowledge**, reframing model outputs as a transferable asset.
- **It is simple.** One extra loss term and a temperature. It spread quickly because anyone could try it.

---

## Real-World Impact

- **DistilBERT** (2019) kept most of BERT's performance at a fraction of the size, making distillation standard for [BERT](../../language-models/03-bert/summary.md)-era deployment.
- **LLM distillation today** often means training the student on text the teacher generated (sequence-level distillation) rather than matching probabilities token by token. [DeepSeek-R1](../../language-models/26-deepseek-r1/summary.md) fine-tuned Qwen and Llama models on about **800,000 teacher-generated samples**, with no reinforcement learning stage for the students. DeepSeek-R1-Distill-Qwen-32B scored **72.6% on AIME 2024**, against 47.0% for a same-size model trained with RL alone - evidence that for small models, distilling a strong reasoner beats discovering reasoning from scratch.
- **Logit distillation in pretraining** returned at scale: Google's Gemma 2 smaller models were trained with a teacher's token distributions as targets.
- **Model providers** commonly ship smaller, cheaper tiers that are widely understood to be distilled from their flagship models, and API terms of service commonly forbid using outputs to train competing models - a sign of how effective distillation is.

---

## Key Takeaways for Practitioners

1. **A good teacher is the best dataset.** If you have access to a strong model, its outputs (or probabilities) usually beat raw labels for training a smaller one.
2. **Tune the temperature.** Too low and the soft signal vanishes; too high and everything looks equally likely. Values between 2 and 10 are common starting points.
3. **Distillation inherits flaws.** The student copies the teacher's mistakes and biases, and cannot exceed it on what it imitates.
4. **Check the licence.** Many model licences and API terms restrict training other models on their outputs.
5. For the hands-on side, the sibling repo covers [quantisation and distillation](https://github.com/PatrickWiloak/cloud-data-ai-security-zero-to-hero/blob/main/learn/concepts/quantization-and-distillation.md).

---

## Limitations & Future Directions

- **The student is bounded by the teacher** on the distilled task, though students sometimes generalise better thanks to regularisation.
- **Capacity gaps matter.** A tiny student learns poorly from a vastly larger teacher; intermediate "teacher assistant" models can help.
- **Sequence-level distillation of LLMs** is closer to supervised fine-tuning on synthetic data, which raises its own questions - see [Model Collapse](../140-model-collapse/summary.md) and the [synthetic data explainer](../../../explainers/open-questions/synthetic-data-and-model-collapse.md).

---

## Further Reading

- **Original Paper:** [arxiv.org/abs/1503.02531](https://arxiv.org/abs/1503.02531)
- **Model Compression** (Bucila, Caruana, Niculescu-Mizil, 2006) - the earlier ensemble-compression work
- **DistilBERT:** Sanh et al. 2019, [arxiv.org/abs/1910.01108](https://arxiv.org/abs/1910.01108)
- **In this collection:** [DeepSeek-R1](../../language-models/26-deepseek-r1/summary.md), [phi-1](../../language-models/135-phi-1-textbooks/summary.md), [Self-Instruct](../79-self-instruct/summary.md), [GPTQ/AWQ quantisation](../86-gptq-awq-quantization/summary.md)

## Citation

```bibtex
@article{hinton2015distilling,
  title={Distilling the Knowledge in a Neural Network},
  author={Hinton, Geoffrey and Vinyals, Oriol and Dean, Jeff},
  journal={arXiv preprint arXiv:1503.02531},
  year={2015}
}
```

<!-- related:start -->

---

## Related in This Collection

- [BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](../../language-models/03-bert/summary.md)
- [DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning](../../language-models/26-deepseek-r1/summary.md)
- [Qwen3: Technical Report](../../language-models/28-qwen3/summary.md)
- [Self-Instruct: Aligning Language Models with Self-Generated Instructions](../../techniques/79-self-instruct/summary.md)
- [GPTQ and AWQ: Post-Training Quantization for Large Language Models](../../techniques/86-gptq-awq-quantization/summary.md)
- [Textbooks Are All You Need (phi-1)](../../language-models/135-phi-1-textbooks/summary.md)
- [AI Models Collapse When Trained on Recursively Generated Data (Model Collapse)](../../techniques/140-model-collapse/summary.md)

<!-- related:end -->
