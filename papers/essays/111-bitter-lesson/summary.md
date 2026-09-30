---
title: "The Bitter Lesson"
slug: "111-bitter-lesson"
number: 111
category: "essays"
authors: "Richard S. Sutton (University of Alberta; DeepMind)"
published: "March 2019 (essay, incompleteideas.net)"
year: 2019
url: "http://www.incompleteideas.net/IncIdeas/BitterLesson.html"
tags: ["essay", "scaling"]
---

# The Bitter Lesson

**Authors:** Richard S. Sutton (University of Alberta; DeepMind)
**Published:** March 2019 (essay, incompleteideas.net)
**Essay:** [incompleteideas.net/IncIdeas/BitterLesson.html](http://www.incompleteideas.net/IncIdeas/BitterLesson.html)

---

## Why This Matters

At about 1,100 words, this may be the most influential page in modern AI. Sutton, a founder of reinforcement learning, argued that 70 years of AI history teach one lesson: methods that exploit ever-growing computation beat methods that encode human knowledge, and "by a large margin". Researchers keep relearning this, and it keeps hurting.

- **It named the pattern.** Chess, Go, speech and vision all followed the same arc: hand-crafted knowledge helps at first, then plateaus, then loses to general methods scaled up.
- **It gave scaling a philosophy.** The essay became the standard justification for building bigger models on more data rather than cleverer, more specialised ones.
- **It identified the two methods that scale:** search and learning.
- **It is still argued about,** including by Sutton himself, who in 2025 said large language models do not fully live up to it.

**The insight:** the cost of computation keeps falling exponentially, so any method whose performance grows with computation will eventually overtake any method whose performance is capped by what humans put in.

---

## The Problem It Addressed

Most AI research, Sutton says, "has been conducted as if the computation available to the agent were constant". Under that assumption, injecting human knowledge is one of the few ways to get better. But over a period only slightly longer than a typical research project, far more computation becomes available, and the knowledge-heavy approach turns into a liability, because it "tends to complicate methods in ways that make them less suited to taking advantage of general methods leveraging computation."

The problem is partly psychological. Building in your understanding of a domain is satisfying and gives quick wins. Waiting for compute to catch up does not.

---

## The Argument

### 1. The historical cases

| Field | Knowledge-based approach | What won |
|---|---|---|
| Chess (1997) | Methods using human understanding of chess structure | "Massive, deep search" with special hardware (Deep Blue beat Kasparov) |
| Go (about 20 years later) | Avoiding search by exploiting human knowledge of the game | Search at scale plus learning a value function by self-play |
| Speech (from the 1970s DARPA competition) | Knowledge of words, phonemes, the vocal tract | Statistical hidden Markov models, then deep learning |
| Computer vision | Edges, generalised cylinders, SIFT features | Deep networks using "only the notions of convolution and certain kinds of invariances" |

In chess, the losing researchers said "brute force" search "was not a general strategy, and anyway it was not how people played chess". Sutton's point is that neither objection mattered.

### 2. The four-step pattern

The essay spells out the lesson as a cycle:

```
1. Researchers build knowledge into their agents.
2. It helps in the short term, and is satisfying.
3. In the long run it plateaus, and even inhibits progress.
4. Breakthroughs come from the opposite approach:
   scaling computation through search and learning.

   The win "is tinged with bitterness, and often incompletely
   digested, because it is success over a favored,
   human-centric approach."
```

### 3. Two lessons to take away

**First, general methods that scale.** "The two methods that seem to scale arbitrarily in this way are search and learning."

**Second, do not build in the contents of minds.** "The actual contents of minds are tremendously, irredeemably complex". Space, objects, multiple agents, symmetries are part of the arbitrary outside world. Instead, "we should build in only the meta-methods that can find and capture this arbitrary complexity." The closing line: "We want AI agents that can discover like we can, not which contain what we have discovered."

---

## Key Claims

1. **Compute is the long-run driver**, via the continued exponential fall in its cost (Moore's law, generalised).
2. **Human knowledge and compute scaling tend to conflict in practice**: time, attention and design choices spent on one are not spent on the other.
3. **Search and learning** are the two families that keep improving with more compute.
4. **Build meta-methods, not content.** Let the system discover structure; do not hand it our discoveries.
5. **The field has not learned this yet** and keeps making "the same kind of mistakes".

---

## How It Has Aged

As of September 2026, the essay is one of the most cited touchstones in AI, and its core prediction has largely held. How to read it is still debated.

**What came true:**

- **Scale won in language.** Within about a year of the essay, [GPT-3](../../language-models/04-gpt3-few-shot-learners/summary.md) and [scaling laws](../../techniques/12-scaling-laws/summary.md) showed smooth gains from simply training larger models on more data, and [The Scaling Hypothesis](../112-scaling-hypothesis/summary.md) cited the Bitter Lesson directly. General architectures such as the [Transformer](../../architectures/01-attention-is-all-you-need/summary.md) and the [Vision Transformer](../../architectures/11-vision-transformer/summary.md), which build in less structure than earlier models, became standard across language, vision and audio.
- **Search and learning, together, at test time.** The reasoning-model wave ([OpenAI o1](../../language-models/31-openai-o1/summary.md), [DeepSeek-R1](../../language-models/26-deepseek-r1/summary.md), [test-time compute](../../techniques/50-test-time-compute/summary.md)) adds compute spent searching and reasoning at inference on top of learning, which is Sutton's two methods combined. Earlier, [AlphaZero](../../techniques/102-alphazero/summary.md) had already shown the recipe in games with no human game knowledge beyond the rules.
- **Recognition for its author.** Sutton and Andrew Barto received the 2024 ACM A.M. Turing Award (announced March 2025) for the foundations of reinforcement learning.

**Where it is contested:**

- **Sutton does not think LLMs are the full answer.** In a September 2025 interview on Dwarkesh Patel's podcast, Sutton argued that large language models are not really "bitter-lesson-pilled": they learn mostly from human-written data in a separate training phase rather than from their own experience while in use. He argued that systems which learn continually from experience will eventually replace them. The same view is laid out in [Welcome to the Era of Experience](../116-era-of-experience/summary.md) (Silver and Sutton, 2025). So the most famous pro-scaling essay is, by its own author, not an endorsement of the current paradigm as it stands.
- **Data limits.** In his NeurIPS 2024 talk, Ilya Sutskever said "pre-training as we know it will unquestionably end" because data is not growing like compute. Compute alone does not scale a method that needs fresh human data (see [Model Collapse](../../techniques/140-model-collapse/summary.md)). This is consistent with Sutton's preference for learning from experience, but it complicates the simple "more compute wins" reading.
- **Human knowledge moved rather than vanished.** The most effective recent gains often come from careful data curation ([FineWeb](../../techniques/133-fineweb/summary.md)), human preference data ([RLHF](../../language-models/05-instructgpt-rlhf/summary.md)) and efficiency engineering ([FlashAttention](../../techniques/16-flash-attention/summary.md), [Mixture of Experts](../../architectures/37-mixture-of-experts/summary.md)). Some read this as human ingenuity in service of scaling, which the essay allows; others read it as evidence that the dichotomy is too clean.

---

## Criticisms

- **"A Better Lesson" (Rodney Brooks, March 2019).** Published within a week, it argues that the celebrated successes are full of human insight. Convolution builds in translation invariance; architectures and training regimes are designed by people. Humans are, in Brooks's words, being asked "to pour their intelligence into the algorithms in a different place and form." He also pointed to data inefficiency, energy use (he contrasts about 2,500 watts for self-driving car computers with about 20 watts for a human brain) and the slowing of Moore's law.
- **Survivorship and timescale.** The essay picks domains where scale eventually won. In the meantime, knowledge-based methods delivered working systems for decades, and "eventually" can be longer than a company, a career, or a product cycle.
- **It can be used to justify anything expensive.** "Just scale it" is not a plan when data, energy or money are the binding constraint, and the essay gives no guidance on *which* general method to scale.
- **Inductive bias is not the enemy.** Some built-in structure (convolutions, attention, positional encodings) is exactly what made scaling work. The real question is which priors generalise, not whether to have any.

---

## Key Takeaways for Practitioners

1. **Prefer methods whose performance improves with compute and data.** If a trick only works at your current scale, expect it to be obsolete soon.
2. **Be suspicious of hand-built features and rules** in any component you expect to keep for years. They are often the first thing a larger model makes unnecessary.
3. **But ship on today's compute.** Human knowledge is legitimate when it buys results now; just do not let it lock you out of the scaling path.
4. **Search and learning compose.** Spending compute at inference (sampling, verifying, searching) is a second axis of scaling next to training.
5. **Watch the binding constraint.** When data or energy, not compute, is the limit, "more compute" is not the lesson.

---

## Further Reading

- **Original essay:** [incompleteideas.net/IncIdeas/BitterLesson.html](http://www.incompleteideas.net/IncIdeas/BitterLesson.html)
- **Rodney Brooks, "A Better Lesson" (2019):** [rodneybrooks.com/a-better-lesson](https://rodneybrooks.com/a-better-lesson/)
- **Dwarkesh Patel, interview with Richard Sutton (2025):** [dwarkesh.com/p/richard-sutton](https://www.dwarkesh.com/p/richard-sutton)
- **In this collection:** [Computing Machinery and Intelligence](../108-computing-machinery-and-intelligence/summary.md), [Software 2.0](../110-software-2/summary.md), [The Scaling Hypothesis](../112-scaling-hypothesis/summary.md), [Welcome to the Era of Experience](../116-era-of-experience/summary.md), [AlphaZero](../../techniques/102-alphazero/summary.md), [Scaling Laws](../../techniques/12-scaling-laws/summary.md)

## Citation

```bibtex
@misc{sutton2019bitter,
  title={The Bitter Lesson},
  author={Sutton, Richard S.},
  year={2019},
  month={March},
  howpublished={\url{http://www.incompleteideas.net/IncIdeas/BitterLesson.html}}
}
```

<!-- related:start -->

---

## Related in This Collection

- [Language Models are Few-Shot Learners (GPT-3)](../../language-models/04-gpt3-few-shot-learners/summary.md)
- [Training Language Models to Follow Instructions with Human Feedback (InstructGPT)](../../language-models/05-instructgpt-rlhf/summary.md)
- [An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale (Vision Transformer)](../../architectures/11-vision-transformer/summary.md)
- [Scaling Laws for Neural Language Models](../../techniques/12-scaling-laws/summary.md)
- [FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness](../../techniques/16-flash-attention/summary.md)
- [DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning](../../language-models/26-deepseek-r1/summary.md)
- [OpenAI o1: Learning to Reason with Reinforcement Learning](../../language-models/31-openai-o1/summary.md)
- [Mixtral of Experts (and the Mixture-of-Experts Architecture)](../../architectures/37-mixture-of-experts/summary.md)

<!-- related:end -->
