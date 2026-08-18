---
title: "Reflexion: Language Agents with Verbal Reinforcement Learning"
slug: "78-reflexion"
number: 78
category: "techniques"
authors: "Noah Shinn, Federico Cassano, Edward Berman, Ashwin Gopinath, Karthik Narasimhan, Shunyu Yao (Northeastern University, MIT, Princeton University)"
published: "March 2023 (NeurIPS 2023)"
year: 2023
url: "https://arxiv.org/abs/2303.11366"
tags: ["agents", "reasoning", "tool-use", "prompting"]
---

# Reflexion: Language Agents with Verbal Reinforcement Learning

**Authors:** Noah Shinn, Federico Cassano, Edward Berman, Ashwin Gopinath, Karthik Narasimhan, Shunyu Yao (Northeastern University, MIT, Princeton University)
**Published:** March 2023 (NeurIPS 2023)
**Paper:** [arxiv.org/abs/2303.11366](https://arxiv.org/abs/2303.11366)

---

## Why This Matters

Reflexion is **the paper that taught agents to learn from their own failures without any gradient updates**. When an agent fails a task, it writes down *why* it failed in plain English, stores that note in memory, and tries again with the note in context. Accuracy jumps.

- **Reported 91 percent pass@1 on HumanEval** with GPT-4, against roughly 80 percent for GPT-4 alone at the time.
- **No fine-tuning.** The "policy update" is a paragraph of English appended to the prompt.
- **The retry loop every coding agent uses.** Claude Code, Cursor, SWE-agent, and Devin-style systems all run some version of act, observe failure, reflect, retry.
- **Made "self-correction" concrete.** It specified what a reflection is, where it is stored, and how it re-enters the loop, rather than vaguely asking the model to "check its work."

**The insight:** reinforcement learning updates a policy with a scalar reward. A language model's policy is its prompt, and language is a far richer feedback channel than a scalar. Instead of "reward = 0," tell the agent "you failed because you assumed the list was sorted." Then put that sentence in its context next attempt. Verbal feedback carries the *diagnosis*, not just the *verdict*.

---

## The Problem: Agents Repeat Their Mistakes

[ReAct](../21-react/summary.md) had shown that interleaving reasoning and actions makes capable agents. But those agents had no memory across attempts. Fail a task, retry, and the agent walks into the same wall - often identically, because the reasoning that produced the failure is the reasoning it will produce again.

The classic fix is reinforcement learning: run the episode, compute a reward, update the weights. For LLM agents that is impractical for three reasons:

1. **Cost.** Fine-tuning a frontier model per task is absurd, and most agents run behind an API where weights are inaccessible.
2. **Sample inefficiency.** RL needs many episodes; agentic tasks are slow and expensive per episode.
3. **Credit assignment.** A scalar reward over a 30-step trajectory does not say *which* step was wrong. The agent must infer the diagnosis from a single number, over many episodes.

---

## The Core Innovation

Three cooperating components and an episodic memory:

```
        +--------------------------------------------------+
        |                                                  |
        v                                                  |
    +--------+     trajectory      +-----------+           |
    | ACTOR  | ------------------> | EVALUATOR |           |
    | (LLM,  |                     | (tests /  |           |
    | ReAct- |                     |  heuristic|           |
    |  style)|                     |  / LLM)   |           |
    +--------+                     +-----------+           |
        ^                                |                 |
        |                                | scalar or       |
        | prompt includes                | binary reward   |
        | recent reflections             v                 |
        |                        +----------------+        |
        |                        | SELF-REFLECTION|        |
        |                        | MODEL (LLM):   |        |
        |                        | "why did this  |        |
        |                        |  fail, and what|        |
        |                        |  to do next"   |        |
        |                        +----------------+        |
        |                                |                 |
        |                                v                 |
        |                     +---------------------+      |
        +---------------------| EPISODIC MEMORY     |------+
                              | (last k reflections)|
                              +---------------------+
```

1. **Actor** - the agent that does the task, typically ReAct-style reasoning plus tool calls.
2. **Evaluator** - produces a reward signal. For code, run the unit tests. For QA, exact match. For open-ended tasks, an LLM judge.
3. **Self-reflection model** - reads the failed trajectory *and* the evaluator's signal, and writes a short natural-language post-mortem: what went wrong, and what to do differently.

The reflection is appended to episodic memory, capped at the most recent few entries so context does not explode. The next attempt begins with those reflections in the prompt. This is the "gradient update," expressed in English.

---

## Key Components Explained

### 1. Verbal Reinforcement
**What it does:** Replaces a scalar gradient with a semantic critique.
**How it works:** A scalar reward of 0 says "wrong." A reflection says "the function crashed on the empty-list input because I indexed before checking length." The second contains the credit assignment already solved. This is the paper's central claim, and it is why one retry often suffices where RL would need hundreds of episodes.

### 2. Episodic Memory With a Sliding Window
**What it does:** Persists lessons without flooding the context.
**How it works:** Keep the last k reflections (small, often 1 to 3). Long memories degrade performance - old reflections about different failure modes act as distractors. This is a recurring lesson in agent design: memory needs eviction policy, not just storage.

### 3. The Evaluator Is the Bottleneck
**What it does:** Determines whether the whole loop can work.
**How it works:** Reflexion needs a reliable signal that the attempt failed. Code has unit tests, which is why the code results are the strongest. Tasks without automatic verification need an LLM judge, which is noisier and can produce reflections about failures that did not occur. The dependence on verifiable feedback is the same constraint that shapes [RLVR](../39-rlvr/summary.md).

### 4. Self-Generated Tests
**What it does:** Manufactures an evaluator when none is given.
**How it works:** For programming tasks, the paper has the model write its own unit tests before writing the solution, then uses those tests as the evaluator. Imperfect tests still catch a large share of real bugs, and this trick is now standard in coding agents.

---

## Key Results

| Task | Baseline | Reflexion | Notes |
|---|---|---|---|
| HumanEval (Python code) | 80.1% (GPT-4) | **91.0%** | pass@1 |
| HumanEval (Python) | 71.0% (GPT-3.5 + CoT) | 88.0% | |
| ALFWorld (embodied tasks) | ~75% (ReAct) | **97%** | over 12 trials |
| HotPotQA (multi-hop QA) | ~30% (ReAct) | ~51% | absolute gain of about 20 points |
| MBPP, Leetcode Hard | varies | consistent gains | including on other languages |

- Gains appear within a handful of trials, most of the benefit arriving in the first one or two retries.
- Ablations confirmed that the *verbal* reflection is doing the work: simply retrying with the same prompt, or retrying with only a scalar failure flag, captured a small fraction of the gain.

---

## Why This Was Revolutionary

- **Made iteration a first-class agent primitive.** Before Reflexion, an agent's answer was its first attempt. After, the retry loop with a diagnosis is the default architecture.
- **Showed language is a legitimate learning channel.** No weight updates, yet behavior improves across episodes. This blurred the line between prompting and training in a way that still shapes agent research.
- **Cheap and model-agnostic.** Any API model, any task with a checkable outcome.
- **Named the components.** Actor/Evaluator/Self-Reflection is a vocabulary the field adopted for describing agent loops.

---

## Real-World Impact

- **Coding agents.** Write code, run tests, read the failure, fix, repeat - the core loop of SWE-agent, Aider, Claude Code, Cursor's agent mode, and the systems that push [SWE-bench](../84-swe-bench/summary.md) scores. Reflexion is the paper that formalized it.
- **CI-driven agents.** Agents that fix failing builds consume the test output as the evaluator signal, exactly as specified here.
- **Self-healing pipelines.** Data pipelines and API integrations where an agent reads the error, reflects, and adjusts its call.
- **Evaluation frames.** "Pass@1 with reflection" versus "pass@k" became a distinction people report, because the two measure different things.

---

## Key Takeaways for Practitioners

1. **Give your agent a real evaluator or the loop is theater.** Unit tests, type checks, linters, schema validation, HTTP status codes. Without ground truth, reflections drift.
2. **Keep the reflection short and actionable.** "Next time, check for the empty case before indexing" beats a page of self-analysis. Prompt for one or two sentences.
3. **Cap memory at a few reflections.** More is worse. Evict aggressively.
4. **Two or three retries is the practical ceiling.** Gains flatten fast and cost is linear; if three attempts fail, the task usually needs a different approach or a human.
5. **Have the model write tests first.** Even mediocre self-written tests turn an unverifiable task into a verifiable one.
6. **Log reflections.** They are an excellent diagnostic dataset for understanding where your agent actually struggles.

---

## Limitations & Future Directions

- **Self-correction without external feedback is unreliable.** Follow-up work (notably "Large Language Models Cannot Self-Correct Reasoning Yet," 2023) found that models asked to critique their own answers *without* a ground-truth signal often change correct answers to wrong ones. Reflexion works because the evaluator is external. This distinction is frequently lost when people cite the paper.
- **Reflection quality is capability-bound.** A model that could not diagnose the bug cannot write a useful reflection about it.
- **Memory does not transfer across tasks** in the original design; each task starts fresh. Cross-task reflection libraries are a natural extension and an open problem.
- **Cost multiplies.** Each retry is a full episode, and episodes are the expensive unit in agentic work.
- **Largely absorbed into models.** Modern reasoning models do some of this internally during extended thinking, though the external test-driven loop remains strictly more reliable because it involves real execution.

---

## Further Reading

- **Original Paper:** [arxiv.org/abs/2303.11366](https://arxiv.org/abs/2303.11366)
- **Code:** [github.com/noahshinn/reflexion](https://github.com/noahshinn/reflexion)
- **Self-Refine (concurrent related work):** [arxiv.org/abs/2303.17651](https://arxiv.org/abs/2303.17651)
- **LLMs Cannot Self-Correct Reasoning Yet (the counterpoint):** [arxiv.org/abs/2310.01798](https://arxiv.org/abs/2310.01798)
- **In this collection:** [ReAct](../21-react/summary.md), [Generative Agents](../58-generative-agents/summary.md), [SWE-bench](../84-swe-bench/summary.md)

## Citation

```bibtex
@inproceedings{shinn2023reflexion,
  title={Reflexion: Language Agents with Verbal Reinforcement Learning},
  author={Shinn, Noah and Cassano, Federico and Berman, Edward and Gopinath, Ashwin and Narasimhan, Karthik and Yao, Shunyu},
  booktitle={Advances in Neural Information Processing Systems (NeurIPS)},
  year={2023}
}
```

<!-- related:start -->

---

## Related in This Collection

- [Language Models are Few-Shot Learners (GPT-3)](../../language-models/04-gpt3-few-shot-learners/summary.md)
- [Chain-of-Thought Prompting Elicits Reasoning in Large Language Models](../../techniques/09-chain-of-thought/summary.md)
- [ReAct: Synergizing Reasoning and Acting in Language Models](../../techniques/21-react/summary.md)
- [GPT-4 Technical Report](../../language-models/36-gpt4/summary.md)
- [RLVR: Reinforcement Learning from Verifiable Rewards](../../techniques/39-rlvr/summary.md)
- [Generative Agents: Interactive Simulacra of Human Behavior](../../techniques/58-generative-agents/summary.md)
- [SWE-bench: Can Language Models Resolve Real-World GitHub Issues?](../../techniques/84-swe-bench/summary.md)
- [Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena](../../techniques/85-llm-as-judge/summary.md)

<!-- related:end -->
