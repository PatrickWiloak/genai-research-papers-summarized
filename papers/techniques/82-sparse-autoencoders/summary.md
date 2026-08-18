---
title: "Sparse Autoencoders and Monosemanticity: Reading the Features Inside a Model"
slug: "82-sparse-autoencoders"
number: 82
category: "techniques"
authors: "Trenton Bricken, Adly Templeton, Joshua Batson, Brian Chen, Adam Jermyn, Tom Conerly, Nick Turner, Cem Anil, Carson Denison, Amanda Askell, Chris Olah et al. (Anthropic); superposition groundwork by Nelson Elhage, Tristan Hume, Catherine Olsson et al. (Anthropic); scaling work by Leo Gao et al. (OpenAI)"
published: "May 2024 (Scaling Monosemanticity, Claude 3 Sonnet); earlier: September 2022 (Toy Models of Superposition), October 2023 (Towards Monosemanticity); concurrent: June 2024 (OpenAI, Scaling and Evaluating Sparse Autoencoders)"
year: 2024
url: "https://transformer-circuits.pub/2024/scaling-monosemanticity/index.html"
tags: ["interpretability", "safety"]
---

# Sparse Autoencoders and Monosemanticity: Reading the Features Inside a Model

**Authors:** Trenton Bricken, Adly Templeton, Joshua Batson, Brian Chen, Adam Jermyn, Tom Conerly, Nick Turner, Cem Anil, Carson Denison, Amanda Askell, Chris Olah et al. (Anthropic); superposition groundwork by Nelson Elhage, Tristan Hume, Catherine Olsson et al. (Anthropic); scaling work by Leo Gao et al. (OpenAI)
**Published:** May 2024 (Scaling Monosemanticity, Claude 3 Sonnet); earlier: September 2022 (Toy Models of Superposition), October 2023 (Towards Monosemanticity); concurrent: June 2024 (OpenAI, Scaling and Evaluating Sparse Autoencoders)
**Paper:** [Scaling Monosemanticity](https://transformer-circuits.pub/2024/scaling-monosemanticity/index.html) (see Further Reading for the full sequence)

---

## Why This Matters

This line of work is **the most credible progress to date on the interpretability problem**: figuring out what is actually happening inside a large language model, in terms a human can read. It is the closest thing the field has to a debugger for neural networks.

- **Individual neurons are uninterpretable.** One neuron fires for Chinese text, HTTP requests, and DNA sequences. This has blocked interpretability for a decade.
- **Sparse autoencoders decompose activations into interpretable features** - millions of them, each corresponding to a recognizable concept.
- **Demonstrated on a frontier model.** Scaling Monosemanticity extracted tens of millions of features from Claude 3 Sonnet, including safety-relevant ones: deception, sycophancy, dangerous code, bias.
- **Features are causal, not just correlational.** Clamping a feature changes model behavior in the predicted way. The famous demonstration is "Golden Gate Claude," where amplifying one feature made the model relate everything to the Golden Gate Bridge.

**The insight:** models represent far more concepts than they have neurons, by storing them as **directions in activation space** that overlap - a phenomenon called superposition. So do not look at neurons. Learn an overcomplete, sparse dictionary of directions instead, using an autoencoder with many more hidden units than the layer has neurons and a sparsity penalty forcing each input to activate only a few of them.

---

## The Problem: Polysemantic Neurons

The natural approach to understanding a network is to look at what each neuron responds to. In vision models this partially worked - there are curve detectors and dog-head detectors. In language models it fails.

A typical neuron in a transformer's MLP layer fires on an incoherent grab-bag of inputs. The reason is **superposition**, characterized in Toy Models of Superposition:

```
The model needs to represent ~100,000 concepts.
The layer has ~10,000 neurons.

Solution the model finds: represent concepts as DIRECTIONS
in the 10,000-dimensional space, not as individual neurons.
With near-orthogonal directions you can pack far more than
10,000 concepts into 10,000 dimensions, at the price of a
little interference.

Consequence: each neuron participates in MANY concepts,
and each concept is spread across MANY neurons.
Looking at one neuron shows you a slice of dozens of concepts.
```

The toy-models paper showed superposition arises predictably when features are **sparse** (most concepts are absent from any given input), which is exactly the regime language lives in. It also showed models arrange these directions in geometric structures (antipodal pairs, pentagons, tetrahedra) depending on sparsity - superposition is an organized strategy, not noise.

This is why probing single neurons was never going to work, and why the field needed a change of basis.

---

## The Core Innovation

Train a **sparse autoencoder** on the model's internal activations. Its hidden layer is much wider than the input, and a sparsity penalty forces only a handful of hidden units to be active at once.

```
   model activations x           reconstruction x_hat
   (e.g. 4,096 dims)             (4,096 dims)
          |                              ^
          |    encode                    |  decode
          v                              |
   +--------------------------------------------+
   |  SAE hidden layer: 1,000,000+ features      |
   |  but only ~20-100 active for any input      |
   +--------------------------------------------+

   Loss = || x - x_hat ||^2   +   lambda * ||f||_1
          reconstruction           sparsity penalty

   Wide + sparse => each feature is forced to specialize.
   The DECODER's columns are the learned directions:
   a dictionary of concepts the model uses.
```

Because the dictionary is overcomplete (far more features than dimensions) and activation is sparse, the autoencoder cannot cheat by spreading a concept across many features. The pressure is toward **monosemantic** features: one feature, one recognizable concept.

The result is a change of coordinates from the model's compressed, entangled basis into a human-legible one.

---

## Key Components Explained

### 1. Overcompleteness
**What it does:** Gives superposed concepts room to separate.
**How it works:** If the model packs 100,000 concepts into 4,096 dimensions, a 4,096-unit autoencoder cannot untangle them. Making the dictionary 8x, 100x, or 1000x wider than the layer provides one slot per concept. Feature count is the main scaling knob, and more features consistently reveal finer-grained concepts.

### 2. Sparsity Penalty
**What it does:** Forces specialization.
**How it works:** An L1 penalty (or, in later work, a hard top-k constraint) on the hidden activations means the autoencoder pays for every active feature. The cheapest way to reconstruct an input while activating few features is for each feature to mean something specific. Top-k SAEs, introduced in OpenAI's scaling work, fix the number of active features exactly and remove the tricky L1-coefficient tuning.

### 3. Feature Interpretation and Validation
**What it does:** Establishes that features mean what they appear to mean.
**How it works:** For each feature, collect the inputs that activate it most strongly and have humans or an LLM describe the pattern. Then validate causally: **clamp** the feature to a high or zero value and observe the behavior change. Correlation plus intervention is the standard of evidence, and it is what distinguishes this work from earlier "this neuron looks like X" claims.

### 4. Scaling to a Production Model
**What it does:** Moves from toy demonstrations to a deployed frontier model.
**How it works:** Scaling Monosemanticity applied SAEs to the residual stream of Claude 3 Sonnet, training dictionaries with roughly 1M, 4M, and 34M features. Findings:
- Features are **abstract and multilingual/multimodal** - a "code error" feature fires on bugs in many programming languages; a concept feature fires on the concept in text and images alike.
- Features are **compositional and organized** - a feature's nearest neighbors in the dictionary are conceptually related, so the dictionary has semantic geometry.
- **Safety-relevant features exist and are findable**: deception, sycophancy, bias, power-seeking framing, dangerous or malicious code, secrecy.
- **Feature steering works.** Clamping the Golden Gate Bridge feature high produced "Golden Gate Claude," a model that steers every conversation to the bridge and identifies as it. Anthropic released it publicly for a day, which did more to communicate interpretability to non-specialists than any paper.

### 5. Scaling Laws for Dictionaries
**What it does:** Makes SAE training a predictable engineering process.
**How it works:** OpenAI's work trained a 16-million-feature SAE on GPT-4 activations and established clean scaling relationships between compute, dictionary size, sparsity, and reconstruction quality - along with metrics for feature quality beyond reconstruction loss. Interpretability became something you can budget for.

---

## Key Results

- **Towards Monosemanticity (2023):** on a one-layer transformer, SAEs extracted thousands of features that were interpretable where the raw neurons were not - the existence proof.
- **Scaling Monosemanticity (2024):** millions of interpretable features from Claude 3 Sonnet, with causal steering demonstrations and identification of safety-relevant features.
- **OpenAI (2024):** 16M features from GPT-4, top-k SAEs, and scaling laws for dictionary learning.
- **Feature steering** reliably changes model behavior in the direction the feature's interpretation predicts, including inducing behaviors (writing insecure code, expressing a particular bias) that the model otherwise refuses or avoids.

---

## Why This Was Revolutionary

- **Broke the polysemanticity barrier** that had blocked mechanistic interpretability since it began.
- **Turned interpretability from analysis into intervention.** Features are handles, not just labels.
- **Provided a route to auditing.** If you can find a "deception" feature and watch whether it fires, you have something closer to a lie detector than any behavioral test provides.
- **Scaled to a real deployed model**, which is the bar interpretability work had historically failed to clear.
- **Made the case concrete for the public.** Golden Gate Claude communicated "we can see and adjust internal concepts" better than any explanation.

---

## Real-World Impact

- **Safety monitoring.** Feature activation as a runtime signal for jailbreak attempts, deceptive framing, or harmful-content generation - complementary to output classifiers, because it inspects the process rather than the product.
- **Steering as a control surface.** Adjusting behavior by clamping features, without fine-tuning. Still research-grade, but a genuinely different lever from prompting or RLHF.
- **Debugging model failures.** Which features fired when the model produced a bad answer is a far more actionable diagnostic than attention maps ever were.
- **Open ecosystem.** Gemma Scope (open SAEs for Gemma models), SAELens, and Neuronpedia made SAE research accessible outside frontier labs, and there is now a substantial independent research community.
- **Connection to [Constitutional AI](../../language-models/14-constitutional-ai/summary.md) and alignment.** Behavioral alignment methods shape outputs; interpretability aims to verify what is happening underneath, which matters exactly when a model has an incentive to look aligned - see [Sleeper Agents](../83-sleeper-agents/summary.md).

---

## Key Takeaways for Practitioners

1. **Neuron-level interpretation is a dead end** for language models. If you are inspecting individual activations, change basis first.
2. **Sparsity plus overcompleteness is the recipe**, and it generalizes: the same dictionary-learning idea applies to embeddings, vision features, and other dense representations you want to understand.
3. **Always validate causally.** A feature that correlates with a concept but does not change behavior when clamped may be an artifact of the dictionary, not a mechanism in the model.
4. **Feature count is the main dial.** More features find rarer, finer concepts; too few produce blended, polysemantic features all over again.
5. **Expect this to reach production tooling.** Feature-based monitoring is a plausible near-term component of safety stacks, and worth tracking even if you do not do interpretability research.

---

## Limitations & Future Directions

- **Coverage is incomplete.** Even 34M features do not capture everything; the papers estimate a long tail of concepts not represented in any given dictionary. There is a persistent gap between the model's behavior and what the dictionary explains.
- **Reconstruction is lossy.** Replacing activations with their SAE reconstruction degrades model performance, which means the dictionary is missing real signal.
- **Features are not circuits.** Knowing which concepts exist is not knowing how they are composed into computations. Attribution graphs and circuit tracing are the follow-on effort, and are much harder.
- **Expensive.** Training a high-quality SAE on a frontier model costs a meaningful fraction of a training run.
- **Feature interpretation is subjective.** Human or LLM labeling of what a feature "means" is a soft step in an otherwise quantitative pipeline, and automated interpretability has its own failure modes.
- **Adversarial robustness unknown.** Whether a model could learn to route sensitive computation around monitored features is an open and important question.

---

## Further Reading

- **Toy Models of Superposition:** [transformer-circuits.pub/2022/toy_model](https://transformer-circuits.pub/2022/toy_model/index.html)
- **Towards Monosemanticity:** [transformer-circuits.pub/2023/monosemantic-features](https://transformer-circuits.pub/2023/monosemantic-features/index.html)
- **Scaling Monosemanticity:** [transformer-circuits.pub/2024/scaling-monosemanticity](https://transformer-circuits.pub/2024/scaling-monosemanticity/index.html)
- **Scaling and Evaluating Sparse Autoencoders (OpenAI):** [arxiv.org/abs/2406.04093](https://arxiv.org/abs/2406.04093)
- **Gemma Scope (open SAEs):** [arxiv.org/abs/2408.05147](https://arxiv.org/abs/2408.05147)
- **In-context Learning and Induction Heads:** [transformer-circuits.pub/2022/in-context-learning-and-induction-heads](https://transformer-circuits.pub/2022/in-context-learning-and-induction-heads/index.html)

## Citation

```bibtex
@article{templeton2024scaling,
  title={Scaling Monosemanticity: Extracting Interpretable Features from Claude 3 Sonnet},
  author={Templeton, Adly and Conerly, Tom and Marcus, Jonathan and Lindsey, Jack and Bricken, Trenton and Chen, Brian and Pearce, Adam and Citro, Craig and others},
  journal={Transformer Circuits Thread},
  year={2024}
}

@article{bricken2023monosemanticity,
  title={Towards Monosemanticity: Decomposing Language Models With Dictionary Learning},
  author={Bricken, Trenton and Templeton, Adly and Batson, Joshua and Chen, Brian and Jermyn, Adam and Conerly, Tom and Turner, Nick and others},
  journal={Transformer Circuits Thread},
  year={2023}
}
```

<!-- related:start -->

---

## Related in This Collection

- [Training Language Models to Follow Instructions with Human Feedback (InstructGPT)](../../language-models/05-instructgpt-rlhf/summary.md)
- [Scaling Laws for Neural Language Models](../../techniques/12-scaling-laws/summary.md)
- [Constitutional AI: Harmlessness from AI Feedback](../../language-models/14-constitutional-ai/summary.md)
- [GPT-4 Technical Report](../../language-models/36-gpt4/summary.md)
- [Sleeper Agents: Training Deceptive LLMs that Persist Through Safety Training](../../techniques/83-sleeper-agents/summary.md)

<!-- related:end -->
