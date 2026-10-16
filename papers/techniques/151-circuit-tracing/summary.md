---
title: "Circuit Tracing and On the Biology of a Large Language Model (Attribution Graphs)"
slug: "151-circuit-tracing"
number: 151
category: "techniques"
authors: "Emmanuel Ameisen, Jack Lindsey, Adam Pearce, Wes Gurnee, Nicholas L. Turner, Brian Chen, Craig Citro, and 19 further Anthropic colleagues including Chris Olah and Joshua Batson (Anthropic). The companion paper \"On the Biology of a Large Language Model\" is led by Jack Lindsey with the same team."
published: "March 2025 (March 27, 2025; transformer-circuits.pub)"
year: 2025
url: "https://transformer-circuits.pub/2025/attribution-graphs/methods.html"
tags: ["interpretability", "safety"]
---

# Circuit Tracing and On the Biology of a Large Language Model (Attribution Graphs)

**Authors:** Emmanuel Ameisen, Jack Lindsey, Adam Pearce, Wes Gurnee, Nicholas L. Turner, Brian Chen, Craig Citro, and 19 further Anthropic colleagues including Chris Olah and Joshua Batson (Anthropic). The companion paper "On the Biology of a Large Language Model" is led by Jack Lindsey with the same team.
**Published:** March 2025 (March 27, 2025; transformer-circuits.pub)
**Paper Link:** [transformer-circuits.pub/2025/attribution-graphs/methods.html](https://transformer-circuits.pub/2025/attribution-graphs/methods.html) (companion: [biology.html](https://transformer-circuits.pub/2025/attribution-graphs/biology.html))

---

## Why This Paper Matters

Interpretability asks what a neural network is actually computing. For years the best tools gave either a bag of labels or a narrow, hand-dissected circuit. Early circuit work found things like [induction heads](../126-induction-heads/summary.md), small attention-based mechanisms in toy models. [Sparse autoencoders](../82-sparse-autoencoders/summary.md) then gave a dictionary of human-readable "features" inside a model, but a dictionary is not a story: it tells you which concepts are present, not how one concept led to the next and finally to the output word.

These two papers, released together by Anthropic, connect the features into a graph. The first (the methods paper, titled "Circuit Tracing: Revealing Computational Graphs in Language Models") describes how to build, for one specific prompt, a diagram of which features pushed which other features on the way to the answer. The second ("On the Biology of a Large Language Model") uses it on Claude 3.5 Haiku, a production model, across a set of case studies: multi-step reasoning, planning ahead in poetry, hallucination, refusals, jailbreaks and more. The title borrows from biology on purpose: the authors treat the model like an organism whose internals are found by dissection and experiment, not designed in advance.

Anthropic also published a plain-language companion post, "Tracing the thoughts of a large language model," the same day. Together they are the clearest public example of looking inside a frontier model and seeing intermediate steps that its text output never mentions.

## The Core Innovation

Replace the hard-to-read parts of the model with a sparse, interpretable stand-in, then trace the influence of each part on the output for one prompt.

```
prompt --> model internals --> next token
              |
              v  (swap MLPs for a "cross-layer transcoder")
        replacement model with sparse features
              |
              v  (freeze attention + normalization for this prompt, add error terms)
        local replacement model (reproduces the real output exactly)
              |
              v  (linear effects between features, then prune)
        attribution graph: prompt tokens -> features -> ... -> output logits
```

Analogy: a standard circuit diagram of a chip tells you every wire; an attribution graph is a trace of which wires actually carried current while the chip computed one particular sum. It is specific to the input, and small enough to read.

## Key Components Explained

### 1. Cross-layer transcoders (CLTs)
**What it does:** Provides interpretable features that stand in for the model's MLP (feed-forward) layers.
**How it works:** Like a sparse autoencoder, a transcoder has a wide layer of features that are mostly zero. Unlike an autoencoder that reconstructs its own input, a transcoder predicts what an MLP outputs. The cross-layer twist is that a feature reads from the residual stream at one layer and can write to the outputs of its own layer and every later layer, with separate decoder weights per target layer. All layers train jointly with a reconstruction loss plus a sparsity penalty. The effect is that a computation spread over several layers can be represented by one feature, which makes graphs shorter (the paper reports an example average path length of 2.3 for the CLT versus 3.7 for per-layer transcoders). Sizes range from 300K to 10M features for an 18-layer research model, and 300K to 30M for Claude 3.5 Haiku.

### 2. The replacement model and the local replacement model
**What it does:** Lets the researchers run the model with its MLPs swapped for CLT features.
**How it works:** Attention layers run as usual. The largest 18-layer CLT matches the original model's next-token prediction on about 50% of test prompts, better than per-layer transcoders. For analysis of a single prompt, the authors build a "local" version: attention patterns and normalization are frozen from the original run, and error terms are added so the output matches the real model exactly. The paper is careful to say matching outputs does not prove the same mechanism, which is why it tests faithfulness separately.

### 3. Attribution graphs
**What it does:** Shows how information flowed for one prompt.
**How it works:** Nodes are the output logits, active CLT features at each token position, token embeddings and error nodes. Edges are direct linear effects, computed with backward Jacobians that stop gradients through nonlinearities, so each feature's activity is exactly the sum of its incoming edges. Raw graphs are enormous, so they are pruned: typically about 10x fewer nodes while losing about 20% of the explained behavior. Pruned graphs often still have hundreds of nodes. Researchers group features into supernodes by meaning and read the result as a diagram.

### 4. Intervention experiments
**What it does:** Checks that the graph is causal, not just a picture.
**How it works:** The researchers suppress or boost the features the graph names and see whether the model's output changes as predicted. Edits are applied to the original model, which is the check that matters because the graph came from the replacement.

## Key Results

The biology paper's case studies (all on Claude 3.5 Haiku):

- **Multi-step reasoning.** Asked for the capital of the state containing Dallas, the model internally activates a "Texas" concept before "Austin." Suppressing the Texas features weakens Austin, and swapping in California features makes it answer Sacramento. So the model does two hops inside one forward pass, with no written chain of thought.
- **Planning in poems.** When writing a rhyming couplet, the model appears to choose a candidate end-of-line word (the example is "rabbit") before writing the line, then composes the line to reach it. Injecting an alternative planned word ("green" or "rabbit" in the paper's experiment) in 25 poems made the line end with the injected word in 70% of cases. This is evidence of forward planning, not just word-by-word guessing.
- **Hallucination.** Refusing to answer is the default circuit. "Known entity" features suppress that default when the model recognizes a name. A made-up name (Michael Batkin) keeps the refusal. A name that is only partly recognized (Andrej Karpathy, in the paper's example) can partly suppress the refusal and the model fabricates, for instance a paper citation. Hallucination looks like a misfire of the familiarity check.
- **Addition.** For 36 + 59, the model runs parallel paths, a rough-magnitude estimate and a precise ones-digit calculation. Asked how it did it, the model gives the standard carry-the-one explanation, which does not match what the graph shows.
- **Multilingual circuits.** Antonym prompts in English, French and Chinese share language-independent concept features, with language-specific parts at the input and output. Claude 3.5 Haiku shares a larger proportion of features across languages than a smaller model (the companion post says more than twice), so shared circuitry appears to grow with scale.
- **Refusals and jailbreaks.** Specific harm features combine into a general "harmful request" feature that drives refusal. In an acrostic jailbreak the model spells out a harmful word without recognizing the request until it begins responding, and its refusal tends to wait until a sentence ends.
- **Medical diagnosis and others.** The paper also covers a diagnosis example, chain-of-thought faithfulness (reasoning that is real, made up, or worked backward from a supplied answer) and a model with a hidden goal. I did not read those sections in full, so treat this summary of them as incomplete.

## Why This Was Revolutionary

- Turned features into end-to-end mechanisms on a production-scale model, not a toy.
- Made "does the model plan ahead?" an empirical question with a causal test.
- Showed that stated explanations (the addition story) can differ from the internal computation, which is concrete evidence for the gap between what a model says and what it does.
- Gave the field a reusable method and a reusable vocabulary (attribution graph, replacement model, supernode).

## Real-World Impact

The method is the natural next step from sparse autoencoders: same dictionary idea, now wired into a computational graph. It reframes the debate in [Do LLMs Reason?](../../../explainers/open-questions/do-llms-reason.md) with direct internal evidence of intermediate steps and planning, and its faithfulness material connects to [Reasoning Models Don't Always Say What They Think](../152-reasoning-models-dont-say-what-they-think/summary.md). Public alignment work now routinely cites interpretability evidence of this kind, for example in checking whether a model's stated reasoning matches its actual computation.

## Key Takeaways for Practitioners

1. A model's output text is not a reliable window into its computation; internal evidence can disagree with the model's own explanation.
2. Hallucination can be understood as a failure of a "do I know this?" check, which suggests why calibration and refusal training interact.
3. Treat the graphs as hypotheses. Always validate with interventions on the real model.
4. Findings are per-prompt. Do not assume a single example reveals the model-wide mechanism.

## Limitations & Future Directions

- Attention patterns are frozen, so the method misses how attention chooses what to read; induction-style mechanisms ([induction heads](../126-induction-heads/summary.md)) are a named blind spot.
- The replacement model only partly captures the original: reconstruction error remains (normalized error about 11.5% for the largest 18-layer CLT and 21.7% for the largest Haiku CLT), and error nodes are not interpretable.
- The biology paper says the graphs give useful insight on about a quarter of the prompts tried, and the case studies are selected, not a random sample.
- Inactive features and inhibition are hard to explain, so the graphs cannot say why a behavior did not occur.
- Reading even a short prompt's circuit takes a few hours of human effort, so scaling to long chains of reasoning needs better tooling, possibly AI-assisted.
- Feature splitting means features may not sit at the most useful level of abstraction.

## Further Reading

- **Methods paper:** https://transformer-circuits.pub/2025/attribution-graphs/methods.html
- **Biology paper:** https://transformer-circuits.pub/2025/attribution-graphs/biology.html
- **Anthropic post:** https://www.anthropic.com/research/tracing-thoughts-language-model
- **Related summaries:** [Sparse Autoencoders](../82-sparse-autoencoders/summary.md), [Induction Heads](../126-induction-heads/summary.md), [Reasoning Models Don't Always Say What They Think](../152-reasoning-models-dont-say-what-they-think/summary.md)

## Citation

```bibtex
@article{ameisen2025circuit,
  title={Circuit Tracing: Revealing Computational Graphs in Language Models},
  author={Ameisen, Emmanuel and Lindsey, Jack and Pearce, Adam and Gurnee, Wes and Turner, Nicholas L. and Chen, Brian and Citro, Craig and others},
  journal={Transformer Circuits Thread},
  year={2025},
  url={https://transformer-circuits.pub/2025/attribution-graphs/methods.html}
}

@article{lindsey2025biology,
  title={On the Biology of a Large Language Model},
  author={Lindsey, Jack and Gurnee, Wes and Ameisen, Emmanuel and Chen, Brian and Pearce, Adam and Turner, Nicholas L. and Citro, Craig and others},
  journal={Transformer Circuits Thread},
  year={2025},
  url={https://transformer-circuits.pub/2025/attribution-graphs/biology.html}
}
```

<!-- related:start -->

---

## Related in This Collection

- [Chain-of-Thought Prompting Elicits Reasoning in Large Language Models](../../techniques/09-chain-of-thought/summary.md)
- [Claude 3.5 Sonnet: Computer Use and Enhanced Capabilities](../../language-models/30-claude-3.5-sonnet/summary.md)
- [Sparse Autoencoders and Monosemanticity: Reading the Features Inside a Model](../../techniques/82-sparse-autoencoders/summary.md)
- [In-context Learning and Induction Heads (Induction Heads)](../../techniques/126-induction-heads/summary.md)
- [Reasoning Models Don't Always Say What They Think (CoT Faithfulness)](../../techniques/152-reasoning-models-dont-say-what-they-think/summary.md)

<!-- related:end -->
