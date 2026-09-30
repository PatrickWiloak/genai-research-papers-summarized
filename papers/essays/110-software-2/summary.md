---
title: "Software 2.0"
slug: "110-software-2"
number: 110
category: "essays"
authors: "Andrej Karpathy (Tesla)"
published: "November 2017 (Medium)"
year: 2017
url: "https://karpathy.medium.com/software-2-0-a64152b37c35"
tags: ["essay", "code"]
---

# Software 2.0

**Authors:** Andrej Karpathy (Tesla)
**Published:** November 2017 (Medium)
**Essay:** [karpathy.medium.com/software-2-0-a64152b37c35](https://karpathy.medium.com/software-2-0-a64152b37c35)

---

## Why This Matters

"Software 2.0" gave the industry a name for a shift it was already living through. Karpathy, then Director of AI at Tesla, argued that neural networks are not "just another tool in your machine learning toolbox" but a new way of writing software: instead of a programmer writing instructions, an optimiser writes the program (the weights), guided by data and a goal.

- **It reframed ML as programming.** Code is written by gradient descent; the dataset is the source code.
- **It predicted the job split.** "2.0 programmers" curate and label data; "1.0 programmers" build the tools, infrastructure and training code around it.
- **It listed concrete engineering advantages** (constant runtime and memory, portability, easy to put in silicon, can trade speed for accuracy by resizing) that still explain why hardware and deployment evolved as they did.
- **It asked for tooling that did not yet exist:** "Software 2.0 IDEs" and a "Software 2.0 GitHub" for datasets.
- **Its author later extended it** to "Software 3.0", where the program is a natural-language prompt to a large language model.

**The insight:** for many problems "it is significantly easier to collect the data (or more generally, identify a desirable behavior) than to explicitly write the program". When that is true, you should specify the behaviour and let optimisation search for the program.

---

## The Problem It Addressed

By 2017 deep learning had won in vision, speech and games, but most engineers still thought of neural nets as a classifier you bolt onto a conventional system. That framing, Karpathy argues, "completely misses the forest for the trees": it hides the trend that entire pieces of software were being replaced, not assisted.

---

## The Argument

### 1. Two ways to write a program

```
Software 1.0                          Software 2.0
------------                          ------------
Human writes explicit instructions    Human specifies a goal
  (Python, C++ ...)                     (dataset of input/output pairs,
                                         "win at Go", ...)
                                      Human writes a rough skeleton
                                        (a network architecture)
Each line picks one point in          Optimiser searches that subset
  "program space"                       of program space
                                        (backprop + SGD)
Output: source code                   Output: weights
```

"Software 2.0 is written in much more abstract, human unfriendly language, such as the weights of a neural network." Nobody writes weights directly because there are millions of them, and "coding directly in weights is kind of hard (I tried)."

### 2. The transition is already happening

Karpathy walks through areas where writing explicit code lost to learned systems:

- **Visual recognition:** engineered features plus a bit of ML, replaced by ConvNets trained on ImageNet, and increasingly by searching over architectures too.
- **Speech recognition:** hidden Markov models and Gaussian mixtures, replaced "almost entirely" by neural nets. He quotes the line attributed to Fred Jelinek: "Every time I fire a linguist, the performance of our speech recognition system goes up".
- **Speech synthesis:** stitching methods replaced by WaveNet-style ConvNets.
- **Machine translation:** phrase-based statistics giving way to neural models.
- **Games:** AlphaGo Zero, a ConvNet reading the raw board, "by far the strongest player".
- **Databases:** "The Case for Learned Index Structures" replaced B-trees with neural models, reporting up to 70 percent faster lookups with an order of magnitude less memory.

### 3. Why prefer the 2.0 stack

Beyond "it works better", Karpathy lists properties of a network compared to a production C++ codebase:

| Property | Why it matters |
|---|---|
| Computationally homogeneous | Mostly matrix multiply and ReLU, so correctness and performance are easier to guarantee |
| Simple to bake into silicon | A tiny instruction set suits custom chips |
| Constant running time | Same FLOPs every forward pass; no infinite loops |
| Constant memory use | No dynamic allocation, no leaks |
| Highly portable | A sequence of matrix multiplies runs almost anywhere |
| Very agile | Halve the channels and retrain for twice the speed, or add capacity when data grows |
| Modules meld | Separately trained pieces can be backpropagated through jointly |
| "It is better than you" | In images, video, sound and speech, the learned program beats hand-written ones |

### 4. The costs

He is explicit about the downsides:

- **Opacity:** "we'll be left with a choice of using a 90% accurate model we understand, or 99% accurate model we don't."
- **Silent failure:** networks can adopt biases from their data in ways that are hard to audit.
- **Strange failure modes:** adversarial examples show how unintuitive the stack is.

### 5. What programming becomes

"Software 1.0 is code we write. Software 2.0 is code written by the optimization based on an evaluation criterion." When the network fails on a rare case, you fix it by adding labelled examples, not by writing code. That implies new tools: an IDE that surfaces likely mislabelled examples by per-example loss, suggests what to label next by model uncertainty, and a GitHub where "repositories are datasets and commits are made up of additions and edits of the labels."

The prediction: 2.0 spreads to any domain "where repeated evaluation is possible and cheap, and where the algorithm itself is difficult to design explicitly", and "when we develop AGI, it will certainly be written in Software 2.0."

---

## Key Claims

1. Neural networks are a **new programming paradigm**, not a classifier.
2. **Data is code**: labelling and curation are programming.
3. The 2.0 stack has **systems advantages** (fixed compute and memory, portability, hardware friendliness) beyond accuracy.
4. **Anything cheap to evaluate and hard to specify** will move to 2.0.
5. The field needs **new tools** built around datasets rather than source files.

---

## How It Has Aged

As of September 2026:

**What came true:**

- **Data work became the core of ML engineering.** Dataset curation, labelling pipelines and evaluation sets are now treated as the main lever on model quality. Large-scale data filtering (see [FineWeb](../../techniques/133-fineweb/summary.md)) and deliberately designed training data (see [phi-1](../../language-models/135-phi-1-textbooks/summary.md)) are exactly "programming by dataset".
- **The tooling appeared, roughly as sketched.** Labelling platforms, dataset versioning, model and dataset hubs, and loss-based data-cleaning tools are now standard infrastructure.
- **"Cheap to evaluate" was the right filter.** The fastest-moving area of 2024-2026 is reinforcement learning on tasks with automatic checkers, such as maths answers and code that must pass tests (see [RLVR](../../techniques/39-rlvr/summary.md) and [SWE-bench](../../techniques/84-swe-bench/summary.md)). That is Karpathy's criterion nearly word for word.
- **Silicon followed homogeneity.** Accelerators built around dense matrix multiplication now dominate AI compute, as the "simple to bake into silicon" argument predicted.
- **The author extended the frame.** In a June 2025 talk at Y Combinator's AI Startup School, Karpathy described "Software 3.0": prompts to large language models as a third way of programming, with English as the programming language. He also coined "vibe coding" in early 2025 for building software largely by prompting.

**What did not, or not simply:**

- **1.0 did not shrink.** Hand-written software kept growing, and the systems around models (serving, orchestration, agents, tool integrations such as [MCP](../../techniques/59-model-context-protocol/summary.md)) are mostly conventional code. In practice the layers stack rather than replace each other.
- **The "Software 2.0 IDE" did not arrive as one product.** Its pieces are spread across many tools.
- **"Modules meld" was less important than expected.** The winning pattern was one huge pretrained model adapted to many tasks (fine-tuning, prompting), not many separately trained modules backpropagated together.
- **Opacity got more important, not less.** The 90-versus-99 percent tradeoff now shows up as questions of safety, auditability and regulation, not just engineering taste.

---

## Criticisms

- **It undersells the cost of 2.0.** Data collection, labelling and compute are expensive, and many problems do not have cheap evaluation. For those, 1.0 remains cheaper and more reliable.
- **Determinism and guarantees matter.** Much software must be exactly right (billing, cryptography, safety interlocks). A learned program that is "better than you" on average but occasionally, silently wrong is unacceptable there.
- **It is a lens, not a theory.** Critics note the essay renames machine learning more than it adds a new technical claim. Its value is in the reframing and the predictions about workflow.
- **"Better than you"** was true in perception in 2017, but overgeneralised when read as a claim about all software.

---

## Key Takeaways for Practitioners

1. **Ask whether it is easier to show the behaviour than to specify it.** If yes, and you can evaluate cheaply, consider a learned component.
2. **Treat datasets like code.** Version them, review changes, write tests (held-out evals) and track regressions.
3. **Fix failures with data** where you can: targeted examples of the failing case, not special-case code around the model.
4. **Keep a 1.0 guardrail** where correctness must be guaranteed.

---

## Further Reading

- **Original essay:** [karpathy.medium.com/software-2-0-a64152b37c35](https://karpathy.medium.com/software-2-0-a64152b37c35)
- **Karpathy, "Software Is Changing (Again)" (YC AI Startup School, June 2025):** [ycombinator.com/library/MW-andrej-karpathy-software-is-changing-again](https://www.ycombinator.com/library/MW-andrej-karpathy-software-is-changing-again)
- **Kraska et al., "The Case for Learned Index Structures":** [arxiv.org/abs/1712.01208](https://arxiv.org/abs/1712.01208)
- **In this collection:** [The Unreasonable Effectiveness of RNNs](../109-unreasonable-effectiveness-of-rnns/summary.md), [The Bitter Lesson](../111-bitter-lesson/summary.md), [AlphaZero](../../techniques/102-alphazero/summary.md), [Codex](../../language-models/56-codex/summary.md)

## Citation

```bibtex
@misc{karpathy2017software,
  title={Software 2.0},
  author={Karpathy, Andrej},
  year={2017},
  month={November},
  howpublished={\url{https://karpathy.medium.com/software-2-0-a64152b37c35}}
}
```

<!-- related:start -->

---

## Related in This Collection

- [RLVR: Reinforcement Learning from Verifiable Rewards](../../techniques/39-rlvr/summary.md)
- [Codex: Evaluating Large Language Models Trained on Code](../../language-models/56-codex/summary.md)
- [Model Context Protocol (MCP): An Open Standard for AI Tool Integration](../../techniques/59-model-context-protocol/summary.md)
- [SWE-bench: Can Language Models Resolve Real-World GitHub Issues?](../../techniques/84-swe-bench/summary.md)
- [Mastering Chess and Shogi by Self-Play with a General Reinforcement Learning Algorithm](../../techniques/102-alphazero/summary.md)
- [The Bitter Lesson](../../essays/111-bitter-lesson/summary.md)
- [The FineWeb Datasets: Decanting the Web for the Finest Text Data at Scale (FineWeb)](../../techniques/133-fineweb/summary.md)
- [Textbooks Are All You Need (phi-1)](../../language-models/135-phi-1-textbooks/summary.md)

<!-- related:end -->
