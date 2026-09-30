---
title: "The Scaling Hypothesis"
slug: "112-scaling-hypothesis"
number: 112
category: "essays"
authors: "Gwern Branwen (independent researcher)"
published: "May 2020 (gwern.net; revised through 2022, with later editorial notes)"
year: 2020
url: "https://gwern.net/scaling-hypothesis"
tags: ["essay", "scaling"]
---

# The Scaling Hypothesis

**Authors:** Gwern Branwen (independent researcher)
**Published:** May 2020 (gwern.net; revised through 2022, with later editorial notes)
**Essay:** [gwern.net/scaling-hypothesis](https://gwern.net/scaling-hypothesis)

---

## Why This Matters

Written in response to [GPT-3](../../language-models/04-gpt3-few-shot-learners/summary.md), this long essay made the strongest public case that the path to general AI might simply be *bigger neural networks trained on more data with more compute*, and that most of the field was refusing to see it. At the time this was a minority view. Within a few years it was the operating strategy of every frontier lab.

- **It named and sharpened the idea.** Gwern distinguishes a *strong* scaling hypothesis (scale a general architecture and sophisticated behaviour emerges) from a *weak* one (scale matters, but you still need to invent brain-like modules one by one).
- **It explained why prediction might be enough.** The section "Why Does Pretraining Work?" argues that driving next-token loss toward human level forces a model to learn grammar, facts, reasoning and theory of mind, because those are what the remaining errors are made of.
- **It predicted an arms race** and argued that the bottleneck was conviction, not hardware.
- **It reads, in hindsight, like a forecast of 2021-2025,** and the living page now carries an editorial note answering one of its own speculations with "Roughly, yes."

**The insight:** "blessings of scale". For deep learning, "hard problems are easier to solve than easy problems": bigger models learn faster, more stably and more generally, and qualitatively new abilities (like in-context learning) appear without being designed in.

---

## The Problem It Addressed

In mid-2020, GPT-3 showed that a 175-billion-parameter model (about 117x larger than GPT-2) could perform many tasks just from examples in its prompt. Gwern notes it was released "to remarkably little interest from researchers". Most of the field viewed scaling as brute force, likely to hit diminishing returns, and not real research. Gwern had expected closer to 100 billion parameters to be the practical ceiling and was surprised that "this vast increase in size did not run into diminishing or negative returns".

The question was whether GPT-3 was a curiosity or a sign that simply scaling an old, simple approach would keep producing more general intelligence.

---

## The Argument

### 1. GPT-3 as proof of concept

Gwern stresses how unoptimised GPT-3 was: "a magnificently obsolete architecture from early 2018", trained "in the dumbest way possible (unidirectional prediction of next text token)". Its meta-learning (learning a new task from a few examples in the prompt) was not engineered; it emerged. If that is what the simplest version does, there is obviously room above it.

### 2. Blessings of scale

The observation: as networks, data and compute grow, problems that plague small networks tend to go away. Larger models generalise better, are more stable and start to meta-learn. Gwern's explanation draws on the idea that a big network contains an enormous number of sub-models; "neural nets are lazy" and grab the easiest shortcut (memorisation, surface features) unless the data and model are large and varied enough that the only way to keep lowering loss is a general solution.

### 3. The hypothesis, stated

> "The strong scaling hypothesis is that, once we find a scalable architecture like self-attention or convolutions, which like the brain can be applied fairly uniformly ... we can simply train ever larger NNs and ever more sophisticated behavior will emerge naturally as the easiest way to optimize for all the tasks & data."

He contrasts this with a *weak* scaling hypothesis that he attributes to DeepMind at the time: AGI needs the "right algorithms", assembled module by module, with compute as "a more powerful tool with which to hunt for the right algorithms".

### 4. Why pretraining works: the last bits are deepest

The central explanation walks through what a character-level model learns as its loss (bits per character) falls:

```
bits/char   what the model has learned
---------   --------------------------------------------------
   8        nothing (uniform guess over bytes)
  ~5        letter and space frequencies
  3-4       real words, punctuation, plurals
  <3        which words co-occur (topics, associations)
  ~2        sentences start to make sense
  1-2       human-sounding for a few sentences
 ~0.7       human level (estimate Gwern cites for humans
            predicting text)

Each step down is smaller in absolute terms and harder to get.
What is left in the final gap is reasoning, consistency,
causality and theory of mind.
```

His example: getting "he" versus "she" right in "Janelle ate some ice cream, because he/she likes sweet things like ice cream" matters enormously to a reader but barely moves the average loss. As a model approaches human-level prediction, almost all remaining error comes from exactly these hard cases. So "nothing less than true understanding will suffice for ideal prediction." (The table is a paraphrase of Gwern's illustrative numbers, not measurements.)

He also admits the counter-argument that nagged him: the pretraining thesis felt "too much like a magic trick", and it could have failed through lack of data, models too small, or the wrong architecture. His conclusion is that "apparently, it would've worked fine"; it just needed more compute and data than anyone had been willing to risk.

### 5. A quantitative guess

Using the scaling formula he fits from the GPT-3 paper, Gwern estimates GPT-2 at roughly 3.3 validation loss, GPT-3 at roughly 1.73, and a hypothetical GPT-4 with 100-1000x more compute at about 1.24. He is explicit that nobody knows which capabilities that would buy.

### 6. Prospects and the sociology of labs

The "Prospects" section argues that the limit was "less hardware than human": organisations had the compute but not the conviction. He predicts that others would eventually have to follow OpenAI, and criticises Google Brain and DeepMind for not believing in scaling. He quotes Ilya Sutskever that besides data and compute a third thing was needed: "conviction".

### 7. Critiquing the critics

The final part argues that AI experts had no coherent model of progress and did not update when their predictions failed. Gwern calls their forecasts "an emperor sans garments".

---

## Key Claims

1. **Scale produces qualitatively new abilities**, not just better scores on old ones.
2. **Simple, uniform architectures plus scale beat intricate, hand-designed ones**, echoing [The Bitter Lesson](../111-bitter-lesson/summary.md).
3. **Next-token prediction is a path to general capability**, because the hard residual errors require understanding.
4. **The binding constraint is willingness to spend**, since the budgets required were small compared with other big-science projects.
5. **Mainstream forecasts were not tracking reality**, so outsiders should weight them less.

---

## How It Has Aged

As of September 2026, most of the essay's directional predictions came true, faster than almost anyone expected in 2020. Some of its mechanisms and specifics did not hold.

**What came true:**

- **The arms race happened.** Gwern noted that as of October 2020 no model had exceeded Turing-NLG's 17 billion parameters apart from GPT-3. Within two years Google's [PaLM](../../language-models/94-palm/summary.md) reached 540 billion, and DeepMind (whose lack of scaling vision the essay criticised) released Gopher and [Chinchilla](../../techniques/18-chinchilla/summary.md). Scaling became the explicit strategy of every major lab.
- **Scale kept buying capability.** [GPT-4](../../language-models/36-gpt4/summary.md) (2023) and its successors delivered large jumps in exactly the areas Gwern said the "last bits" were about: reasoning, consistency and knowledge. [Emergent abilities](../../techniques/81-emergent-abilities/summary.md) became a named research topic.
- **The multimodal agent picture.** The essay asked whether we would soon discuss a large multimodal Transformer as "the backbone of a MuZero-like learning+planning DRL agent" running on many tasks such as coding. The living page now answers: "[Roughly, yes. -Editor 2025-10-19]". Reasoning models trained with reinforcement learning on top of pretrained models ([o1](../../language-models/31-openai-o1/summary.md), [DeepSeek-R1](../../language-models/26-deepseek-r1/summary.md)) and coding agents fit that description.
- **Its sequels.** The essay's own header now points readers to later works in the same line, including [Situational Awareness](../113-situational-awareness/summary.md).

**What did not, or needed revision:**

- **The specific scaling law was revised.** Gwern's loss extrapolations built on the Kaplan et al. [scaling laws](../../techniques/12-scaling-laws/summary.md). [Chinchilla](../../techniques/18-chinchilla/summary.md) (2022) showed that those laws underweighted data: compute-optimal models should be smaller and trained on far more tokens. The direction held; the recipe changed.
- **Pretraining alone was not the whole story.** Much of the progress after 2022 came from post-training ([RLHF](../../language-models/05-instructgpt-rlhf/summary.md), [RLVR](../../techniques/39-rlvr/summary.md)) and [test-time compute](../../techniques/50-test-time-compute/summary.md), not only from lowering next-token loss. Ilya Sutskever's NeurIPS 2024 remark that "pre-training as we know it will unquestionably end" because data is finite marks the shift. The strong hypothesis survives if you count RL and inference as scaling; the narrow "just pretrain bigger" reading does not.
- **Data became the constraint** in a way the essay mostly treated as solvable. Quality filtering ([FineWeb](../../techniques/133-fineweb/summary.md)), synthetic data and the risks of training on generated text ([Model Collapse](../../techniques/140-model-collapse/summary.md)) are now central.
- **Efficiency innovations mattered.** Gwern suggested Transformers "seem mostly be about efficiency" and even RNNs might have worked. Efficiency techniques such as [Mixture of Experts](../../architectures/37-mixture-of-experts/summary.md) and [FlashAttention](../../techniques/16-flash-attention/summary.md) turned out to be a large part of what made scale affordable.

---

## Criticisms

- **Unfalsifiable in practice.** Any failure can be blamed on insufficient scale, so the hypothesis is hard to refute in advance. Gwern acknowledges it is "difficult to prove in advance rather than as a fait accompli".
- **Loss is not capability.** Lower perplexity does not guarantee reliable reasoning, truthfulness or agency. Critics point to persistent hallucination and brittle out-of-distribution behaviour in very large models (see [ARC](../../techniques/138-arc-agi/summary.md)).
- **Tone and targeting.** The essay's attacks on named labs and on "the critics" as a class were polemical, and some specific judgements (for example, about which organisations would matter) were overtaken quickly.
- **Safety framing.** Gwern treats scaling as both likely and dangerous. Readers who accept the first half and ignore the second have used the essay to justify racing, which is not its conclusion.
- **Source type.** It is a long, frequently revised web essay with extensive footnotes, not a peer-reviewed paper. Quotes and numbers should be checked against the current page, since it changes.

---

## Key Takeaways for Practitioners

1. **Test whether your problem gets easier with scale** before investing in clever specialised methods. If a bigger model or more data fixes it, the clever method is temporary.
2. **Loss curves are informative but not the product.** Track capability evals alongside loss; the last small improvements in loss can carry most of the useful behaviour.
3. **Scaling has several axes:** parameters, data, pretraining compute, RL and inference-time compute. The binding one changes over time.
4. **Beware extrapolating one scaling law.** Chinchilla showed a widely used law was miscalibrated; re-derive for your setting.

---

## Further Reading

- **Original essay:** [gwern.net/scaling-hypothesis](https://gwern.net/scaling-hypothesis)
- **Kaplan et al., "Scaling Laws for Neural Language Models" (2020):** [arxiv.org/abs/2001.08361](https://arxiv.org/abs/2001.08361)
- **Brown et al., "Language Models are Few-Shot Learners" (2020):** [arxiv.org/abs/2005.14165](https://arxiv.org/abs/2005.14165)
- **In this collection:** [GPT-3](../../language-models/04-gpt3-few-shot-learners/summary.md), [Scaling Laws](../../techniques/12-scaling-laws/summary.md), [Chinchilla](../../techniques/18-chinchilla/summary.md), [The Bitter Lesson](../111-bitter-lesson/summary.md), [Emergent Abilities](../../techniques/81-emergent-abilities/summary.md), [Situational Awareness](../113-situational-awareness/summary.md)

## Citation

```bibtex
@misc{branwen2020scaling,
  title={The Scaling Hypothesis},
  author={Branwen, Gwern},
  year={2020},
  howpublished={\url{https://gwern.net/scaling-hypothesis}}
}
```

<!-- related:start -->

---

## Related in This Collection

- [Language Models are Few-Shot Learners (GPT-3)](../../language-models/04-gpt3-few-shot-learners/summary.md)
- [Training Language Models to Follow Instructions with Human Feedback (InstructGPT)](../../language-models/05-instructgpt-rlhf/summary.md)
- [Scaling Laws for Neural Language Models](../../techniques/12-scaling-laws/summary.md)
- [FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness](../../techniques/16-flash-attention/summary.md)
- [Training Compute-Optimal Large Language Models (Chinchilla)](../../techniques/18-chinchilla/summary.md)
- [DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning](../../language-models/26-deepseek-r1/summary.md)
- [OpenAI o1: Learning to Reason with Reinforcement Learning](../../language-models/31-openai-o1/summary.md)
- [GPT-4 Technical Report](../../language-models/36-gpt4/summary.md)

<!-- related:end -->
