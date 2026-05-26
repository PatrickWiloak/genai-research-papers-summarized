# Self-Refine: Iterative Refinement with Self-Feedback

**Authors:** Aman Madaan, Niket Tandon, Prakhar Gupta, Skyler Hallinan, Luyu Gao, Sarah Wiegreffe, Uri Alon, Nouha Dziri, Shrimai Prabhumoye, Yiming Yang, Shashank Gupta, Bodhisattwa Prasad Majumder, Katherine Hermann, Sean Welleck, Amir Yazdanbakhsh, Peter Clark (CMU, Allen AI, University of Washington, NVIDIA, UC San Diego, Google Research)
**Published:** March 2023 (NeurIPS 2023)
**Paper:** [arxiv.org/abs/2303.17651](https://arxiv.org/abs/2303.17651)

---

## Why This Paper Matters

Self-Refine introduced one of the simplest and most influential ideas in the modern LLM toolkit: a single language model can dramatically improve its own outputs by acting as its own critic. There is no extra training, no second model, no reward signal — just the same model prompted three different ways (generate, critique, refine) in a loop. The technique improved GPT-4 outputs on tasks ranging from code optimization to dialogue response generation, with average gains of around 20% over a single-shot baseline.

More importantly, Self-Refine codified the now-ubiquitous **"LLM-as-critic"** pattern. This pattern underlies the iterative reasoning behavior of o1-style models, the verifier loops in agents, the self-correction steps in modern coding assistants, and the synthetic data pipelines used to train post-RLHF models. If Chain-of-Thought taught the field that language models reason better when they think out loud, Self-Refine taught the field that they reason even better when they read their own homework.

---

## The Problem

LLMs frequently produce outputs that are *mostly* right but contain fixable flaws: a buggy line in otherwise correct code, an awkward phrase in a translation, a confusing sentence in an explanation, a weak metaphor in a poem. The model often *knows* the output is suboptimal — it can readily point out the problem if asked — but its single forward pass committed to the flawed version anyway.

Humans don't work this way. A skilled writer drafts, rereads, edits, and revises. A skilled programmer writes a function, runs it through their head, spots an off-by-one error, and fixes it. Could we get language models to do the same thing?

Previous approaches to this problem usually required:

- A **separate critic model** trained on human feedback (expensive)
- A **reward model** like in RLHF (requires preference data)
- **External tools** like code interpreters or search (adds infrastructure)
- **Fine-tuning** the base model to produce better outputs first (slow)

The question Madaan et al. asked: can a single off-the-shelf LLM do all the work itself, with no extra training and no external help?

---

## The Core Innovation

Self-Refine is a three-step loop using the same underlying model:

1. **Generate** an initial output for the task
2. **Feedback** — prompt the model to critique its own output
3. **Refine** — prompt the model to produce a new output that addresses the critique

Repeat steps 2 and 3 until the model declares the output acceptable, or a maximum iteration count is reached.

The crucial insight is that these three roles can be elicited from the same frozen model with different prompts. The model carrying "critic hat" can detect flaws the same model wearing "generator hat" produced, because the critic is not constrained by the autoregressive commitments the generator already made.

```
Task input
   |
   v
[Generator prompt] -> Output v1
   |
   v
[Critic prompt + Output v1] -> Feedback
   |
   v
[Refiner prompt + Output v1 + Feedback] -> Output v2
   |
   v
... repeat until stop ...
```

No gradient updates. No second model. No labeled training data. Just three prompts and a loop.

---

## How It Works

### Generate

A standard task prompt. For code optimization:

```
Task: Optimize this Python code for runtime.

Code:
def sum_of_squares(n):
    total = 0
    for i in range(n):
        total += i * i
    return total
```

The model returns an initial attempt.

### Feedback

The model is given the original task and its output, and asked to critique:

```
You are an expert Python programmer. Below is a piece of code
and an optimization. Identify any inefficiencies, bugs, or
opportunities for improvement. Be specific.

Original code: ...
Proposed optimization: ...

Feedback:
```

The model might respond:

```
The loop-based approach is still O(n). There is a closed-form
formula for the sum of squares from 0 to n-1:
    n * (n-1) * (2n-1) / 6
This would be O(1) and significantly faster for large n.
```

### Refine

The model receives the original task, its prior output, and the feedback, and produces a new attempt:

```
def sum_of_squares(n):
    return n * (n - 1) * (2 * n - 1) // 6
```

A simple stop condition is included in the feedback prompt — for example, asking the model to output "STOP" if the answer is already good — so refinement halts when no further improvements are identified.

### Few-shot Examples

For each task, the authors supply a handful of demonstration examples showing the format of feedback and refinement. The model uses these to learn what kind of critique is appropriate for the domain (style for poetry, correctness for code, persuasiveness for arguments, etc.).

---

## Key Results

The authors evaluated Self-Refine across seven diverse tasks with GPT-3.5 and GPT-4:

| Task | Base GPT-4 | Self-Refine GPT-4 | Improvement |
|------|------------|-------------------|-------------|
| Dialogue Response | 25.4% | 49.2% | +23.8 |
| Code Optimization | 27.3% | 36.0% | +8.7 |
| Code Readability | 27.4% | 56.2% | +28.8 |
| Math Reasoning (GSM-style) | 92.9% | 93.1% | +0.2 |
| Sentiment Reversal | 3.8% | 36.2% | +32.4 |
| Acronym Generation | 30.4% | 56.0% | +25.6 |
| Constrained Generation | 15.0% | 45.0% | +30.0 |

The average absolute improvement was roughly 20 percentage points. The gains were largest on open-ended tasks (writing, sentiment, generation) where there is a clear notion of "better" but no single correct answer. They were smallest on math, where the model's first answer was either right or had a subtle reasoning error that the critic couldn't always catch.

One striking observation: stronger base models benefited *more* from self-refinement, not less. GPT-4 used the critique more effectively than GPT-3.5, suggesting that the technique scales with model capability rather than substituting for it.

---

## Impact and Legacy

Self-Refine became the canonical reference for the "LLM-as-critic" pattern, and its influence shows up across modern AI systems:

- **Reasoning models like OpenAI o1 and DeepSeek-R1** internalize the self-critique loop during training. The model is rewarded for spending tokens noticing its own mistakes ("Wait, that's not right...") and correcting them — essentially Self-Refine, but baked into the weights via reinforcement learning.
- **Coding agents** (Cursor, Aider, Claude Code) routinely use self-critique loops: write code, run tests, read errors, revise.
- **Synthetic data pipelines** for post-training use critic models (often the same model in a different role) to filter and improve training examples.
- **Constitutional AI** (Anthropic) generalizes the pattern with explicit principles guiding the critique step.
- **Agent frameworks** like Reflexion, CRITIC, and many LangChain agents implement variants of generate-critique-refine.

The simplicity of Self-Refine is also its legacy. Before this paper, "self-improvement" in LLMs usually meant elaborate training procedures. Self-Refine showed that meaningful gains were available at inference time, with no training at all, just by asking the model to read its own work.

---

## Connections to Other Papers

- **Chain-of-Thought (#9)** — Self-Refine extends CoT from "think before answering" to "think, answer, critique, then answer again." Both work by giving the model more compute and structure at inference time.
- **ReAct (#21)** — Self-Refine is a sibling pattern. ReAct alternates thinking and acting on the external world; Self-Refine alternates generating and critiquing the model's own output.
- **Tree of Thoughts (#25)** — ToT explores many parallel reasoning branches with self-evaluation at each node. Self-Refine is the linear special case: a single trajectory revised in place.
- **Constitutional AI (#14)** — Anthropic's RLAIF method uses LLM self-critique guided by a list of principles. Self-Refine's prompt-only loop is the inference-time counterpart of CAI's training-time loop.
- **InstructGPT (#5)** — RLHF requires a separate reward model trained on human preferences. Self-Refine asks: what if the generator model is already a good enough preference judge to skip the reward model?
- **DeepSeek-R1 (#26) and o1 (#31)** — Modern reasoning models can be viewed as "Self-Refine trained into the weights." The model learns to do the critique-and-revise loop autonomously inside its chain of thought.
- **DPO (#19)** — DPO trains a model directly on preference pairs. Self-Refine generates such pairs (original vs. refined) for free at inference time, useful for bootstrapping training data.
- **Generative Agents (#58)** — Multi-agent systems often use one agent's output as another agent's input for critique. Self-Refine is the single-agent special case.

---

## Key Takeaways

1. **Same model, three roles.** A single frozen LLM can generate, critique, and revise its own outputs using only prompt engineering. No training, no second model.
2. **Critics catch what generators commit.** The autoregressive nature of generation forces early commitments. A second pass dedicated to critique is free from those commitments and finds fixable flaws.
3. **Open-ended tasks benefit most.** Writing, dialogue, and constrained generation see 20-30 point gains. Mathematical reasoning, where errors are subtle and binary, sees smaller gains.
4. **Stronger models, stronger gains.** Self-Refine compounds with model capability rather than replacing it. GPT-4 uses critique better than GPT-3.5.
5. **A blueprint for modern reasoning.** The Self-Refine loop foreshadowed o1-style models, agent self-correction, Constitutional AI, and the broader test-time-compute paradigm of trading inference cycles for quality.
