# Scaling Monosemanticity: Extracting Interpretable Features from Claude 3 Sonnet

**Authors:** Adly Templeton, Tom Conerly, Jonathan Marcus, Jack Lindsey, Trenton Bricken, Brian Chen, Adam Pearce, Craig Citro, Emmanuel Ameisen, Andy Jones, Hoagy Cunningham, Nicholas L. Turner, Callum McDougall, Monte MacDiarmid, C. Daniel Freeman, Theodore R. Sumers, Edward Rees, Joshua Batson, Adam Jermyn, Shan Carter, Chris Olah, Tom Henighan (Anthropic)
**Published:** May 2024 (Anthropic's *Transformer Circuits Thread*)
**Paper:** [transformer-circuits.pub/2024/scaling-monosemanticity/](https://transformer-circuits.pub/2024/scaling-monosemanticity/)

---

## Why This Paper Matters

This paper is a landmark in mechanistic interpretability — the project of reverse-engineering what is actually happening inside a neural network. Until 2024, most interpretability work had been done on small toy models. "Scaling Monosemanticity" took the techniques to a production-scale frontier model, Anthropic's Claude 3 Sonnet, and showed that **millions of human-interpretable features** could be extracted from its internal activations using sparse autoencoders. Among those features were a feature for the Golden Gate Bridge, a feature for code with security vulnerabilities, a feature for sycophantic praise, and features for concepts as abstract as "deception" or "the experience of inner conflict."

The paper matters for three reasons. First, it provided strong evidence that the **superposition hypothesis** — the idea that neural networks represent many more concepts than they have neurons by storing them as overlapping linear combinations — is correct and tractable at scale. Second, it demonstrated that interpretability research can produce safety-relevant insights about real, deployed models. Third, it set off a wave of follow-up work (OpenAI's GPT-4 SAE paper, Google DeepMind's Gemma Scope, dozens of academic projects) that made sparse autoencoders one of the central tools of modern interpretability.

---

## The Problem

A neural network is a black box. Given a prompt, it produces output, and somewhere in the middle billions of floating-point numbers shuffle through matrix multiplications. We do not, in any natural sense, *understand* what the model is doing. For safety, debugging, and basic scientific curiosity, we would like to.

A natural starting place: maybe individual neurons correspond to individual concepts. Inspect a neuron, find what makes it fire, label it. This worked sometimes — vision networks famously had "cat-detector" and "edge-detector" neurons. But on language models, the project largely failed: most neurons were **polysemantic**, firing on a confusing mix of unrelated concepts. One neuron might activate on the word "the" in legal documents, the symbol pi, and pictures of dogs. There seemed to be no clean concepts to find.

The leading hypothesis to explain this was **superposition**. Anthropic's earlier work proposed that networks pack many more features than they have neurons by representing each feature as a particular *direction in activation space*, and allowing those directions to overlap. Any one neuron is a projection across many features and so looks polysemantic. The features themselves, if you could find them in the right basis, would be cleanly interpretable.

The question: can we *find* those features in real, large models?

---

## The Core Innovation

The technique is a **sparse autoencoder (SAE)**: a shallow neural network that learns to reconstruct a model's internal activations using a much larger but sparsely-activating hidden layer. The idea is:

- Take the residual stream activations of a transformer at some layer.
- Train an autoencoder with a hidden layer of, say, 1 million or 30 million features.
- Add a sparsity penalty so that for any given input, only a small number of features (often dozens) are non-zero.
- The features the autoencoder learns are candidate "concepts" — directions in activation space that the model is using to represent meaningful things.

```
    Claude activations  ---->  Encoder  ---->  Sparse features (very high-D, ~99.9% zeros)
    (d_model = 4096)            (linear)         |
                                                  |
    Reconstructed activations  <----  Decoder <--+
    (with reconstruction loss + sparsity penalty)
```

The conceptual move: instead of trying to interpret the *neurons*, build a *dictionary* of directions and interpret the dictionary entries. Each entry is one feature. With enough features, each entry can correspond to a single, semantically clean concept rather than a polysemantic mush.

This idea was not new. Anthropic and others had used SAEs on small models. The contribution of "Scaling Monosemanticity" was showing it works on a frontier production model and scaling it to extract tens of millions of features.

---

## How It Works

### Training the SAE

Anthropic trained sparse autoencoders on the residual stream activations of Claude 3 Sonnet at a middle layer. They trained three SAEs of different sizes: 1M, 4M, and 34M features. Each SAE was trained on billions of activation vectors collected from a large text corpus run through Sonnet, using a standard reconstruction loss plus an L1 sparsity penalty.

The architecture is simple — a linear encoder, ReLU, linear decoder, with the constraint that decoder rows have unit norm — but training at this scale required enormous compute and careful engineering.

### Interpreting Features

Once trained, each of the millions of features is a candidate concept. Researchers interpret a feature by:

1. Finding the top text inputs from a held-out corpus that maximally activate it.
2. Looking for a common theme.
3. Naming and validating the feature.

Many features had very clean themes. For example:

- **Golden Gate Bridge feature** — activated on text mentioning the bridge, images of the bridge, the bridge in many languages, and even oblique references to its color or location.
- **Code with security vulnerabilities** — activated on code with buffer overflows, SQL injections, hardcoded credentials.
- **Sycophantic praise** — activated on text full of excessive flattery.
- **Inner conflict** — activated on text describing a person torn between two desires.
- **Brain science** — activated on neuroscience discussions across multiple languages.
- Many features for specific people, places, languages, programming languages, emotional tones, and rhetorical patterns.

### Causal Tests

Critically, the paper goes beyond labeling features and demonstrates that they are *causally* used by the model. By **clamping** a feature — artificially turning its activation up or down during inference — researchers could systematically change Sonnet's behavior:

- Clamping the Golden Gate Bridge feature to a very high value produced "Golden Gate Claude," a version of the model that worked the Golden Gate Bridge into nearly every response. (Anthropic publicly deployed this as a demonstration.)
- Clamping the sycophancy feature up made the model more flattering; clamping it down made it more direct.
- Clamping safety-relevant features (e.g., scam-detection) produced predictable changes in how the model handled relevant prompts.

These interventions are the proof that the SAE features are not just decorative — they are real components of the model's computation.

---

## Key Results

The paper reports several headline findings:

- **Tens of millions of features**, the vast majority of which appear meaningfully interpretable on inspection.
- **Multilingual and multimodal features** — many features activate on the same concept across English, French, Chinese, and even on images of the concept fed through Sonnet's vision pathway.
- **Abstract features** — beyond concrete entities, features exist for abstract concepts like "deception," "code containing a bug," "the experience of being praised," and "self-reference."
- **Causal interventions work** — clamping features predictably changes model behavior, confirming they are real internal variables.
- **Safety-relevant features exist** — features for scams, dangerous biological capabilities, manipulative behavior, and racist or biased content can be located and studied directly.
- **Feature scale matters** — larger SAEs find rarer, more specific concepts. There appears to be no clear ceiling: more features keep finding more meaningful structure.

The paper is candid about limitations: there are also "uninterpretable" features, the 34M dictionary likely still does not contain everything Sonnet knows, and SAEs may distort the representations they extract.

---

## Impact and Legacy

"Scaling Monosemanticity" reshaped the interpretability landscape essentially overnight:

- **Field consensus shifted.** Sparse autoencoders went from a niche technique to the default tool for studying transformer internals.
- **Follow-up at every major lab.** OpenAI released an SAE study on GPT-4 weeks later. Google DeepMind released Gemma Scope, a large public set of SAEs on Gemma models. Academic groups produced dozens of follow-up papers.
- **Open tools and datasets.** SAELens, Neuronpedia, and other community resources emerged to make SAE-based interpretability more accessible.
- **A roadmap for AI safety.** The paper made concrete the idea that we might be able to *detect* dangerous internal states (e.g., a "deception" feature firing) in deployed models, opening practical research directions for monitoring and steering.
- **Golden Gate Claude as cultural moment.** Anthropic briefly deployed the feature-clamped variant publicly, giving the broader public a tangible demonstration of interpretability research.

Mechanistic interpretability is still far from a complete science. Features alone do not yet give us *circuits* — the connected computational pathways that perform reasoning, planning, or other complex behaviors. But this paper made interpretability of frontier models feel like a tractable research program rather than a distant aspiration, which is itself a major accomplishment.

---

## Connections to Other Papers

- **Earlier Anthropic Circuits work** — "Toy Models of Superposition" and "Towards Monosemanticity" laid the conceptual and methodological foundation. "Scaling Monosemanticity" is the scale-up of those ideas to a production model.
- **Constitutional AI (#14)** — Anthropic's complementary safety work. SAEs identify features inside the model; Constitutional AI shapes the model's behavior. Together they support a two-pronged safety strategy: shape outputs, monitor internals.
- **RLHF / InstructGPT (#5)** — RLHF changes model behavior without offering insight into *why* the model behaves differently. SAEs offer a way to inspect what changed inside a model after RLHF, opening a path to mechanistic understanding of alignment training.
- **Transformer (foundational)** — SAEs operate on the residual stream, which is the central data structure of transformer architectures. Their efficacy depends on properties (linear superposition, additive structure) specific to transformers.
- **LoRA (#10) and QLoRA (#22)** — both involve manipulating low-rank structures inside trained models. SAEs are a different kind of structural analysis of the same models.
- **Scaling Laws (#12) and Chinchilla (#18)** — the empirical regularity that "bigger models behave smoother and richer" makes SAE-style analysis tractable: larger models have cleaner feature geometry.
- **DPO (#19) and DeepSeek-R1 (#26)** — emerging work uses SAE features to study what changes in models post-finetuning or post-RL, linking interpretability to alignment.
- **AlphaFold 2 (#87)** — both papers exemplify the value of *targeted, principled analysis* of large neural networks: one to understand biology, the other to understand cognition.

---

## Key Takeaways

1. **Sparse autoencoders find interpretable features inside production LLMs.** Anthropic extracted tens of millions of features from Claude 3 Sonnet, the vast majority of which appear human-interpretable on inspection.
2. **The superposition hypothesis is right and tractable.** Concepts are stored as overlapping directions in activation space, and we can recover those directions with the right tools at scale.
3. **Features are causally real.** Clamping a feature changes the model's behavior in predictable ways — proving the features are not just decorative correlations.
4. **A path to interpretable safety monitoring.** Features for deception, dangerous capabilities, sycophancy, and bias can be located, studied, and used to monitor or steer model behavior — a concrete research agenda for AI safety.
5. **Interpretability of frontier models is now feasible.** Before this paper, mechanistic interpretability felt like a small-models-only project. After it, every major lab is doing SAE work on frontier-scale models, and the field has a clear shared methodology.
