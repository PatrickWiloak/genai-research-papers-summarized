# Reflexion: Language Agents with Verbal Reinforcement Learning

**Authors:** Noah Shinn, Federico Cassano, Edward Berman, Ashwin Gopinath, Karthik Narasimhan, Shunyu Yao (Northeastern University, MIT, Princeton)

**Published:** March 2023 (NeurIPS 2023)

**Paper Link:** https://arxiv.org/abs/2303.11366

---

## Why This Paper Matters

Reflexion was one of the foundational papers of the agentic-LLM era. It showed that a language model agent could **dramatically improve on a task across multiple attempts by writing down — in natural language — what went wrong and how to do better next time**. No gradient updates, no weight changes. Just text in a memory buffer.

This "verbal reinforcement learning" idea now sits behind almost every modern agent loop: AutoGPT, BabyAGI, OpenAI's o1-style self-critique, Anthropic's tool-use agents, Devin, SWE-agent, and the entire wave of coding agents that fix bugs across multiple iterations. Reflexion is the paper that turned the static prompt-response interaction into a **dynamic, learning-across-trials loop**, and it did so without requiring any access to model internals.

It also delivered an eye-catching empirical result: on HumanEval (Python code generation), Reflexion lifted GPT-4 from 80.1% to 91.0% pass@1 — a substantial jump on a benchmark people had assumed was nearly saturated.

---

## The Problem Before

By early 2023, ReAct (#21) had established that LLM agents could interleave reasoning, action, and observation. But ReAct agents were **memoryless across attempts**: when an agent failed a task, it had no built-in way to learn from that failure and try again differently. Other approaches:

- **Traditional RL** would update model weights from rewards, but that requires gradient access, large amounts of compute, and many trajectories — impractical with frontier LLMs accessed via API.
- **Self-consistency / Chain-of-Thought (#09)** sample multiple reasoning paths and vote, but each sample is independent — no learning.
- **Tree of Thoughts (#25)** explores branches at decision time, but doesn't persist learning across full task attempts.

The gap: a way for an LLM agent to *improve across episodes* without weight updates, using only its own natural-language self-reflection as the learning signal.

---

## The Core Innovation: Three Specialized LLM Roles + a Reflection Memory

Reflexion decomposes an agent into three cooperating LLM roles, plus an external memory.

### 1. Actor

Generates actions and reasoning given the current task, history, and **any reflections from previous attempts**. This is essentially a ReAct agent — its prompt now includes a "Lessons from past attempts" section drawn from the reflection memory.

### 2. Evaluator

Looks at the trajectory the Actor produced and returns a reward signal. The signal can come from:
- **Ground truth** (does the code pass unit tests? does the answer match?).
- **An LLM judge** (does this answer seem correct?).
- **Heuristics** (did the agent reach the goal state?).

The Evaluator only needs to distinguish "success" from "failure" — Reflexion is robust to sparse, even binary, reward.

### 3. Self-Reflector

When the Evaluator says the attempt failed, this role generates a **natural-language reflection**: a written explanation of what went wrong and how to do better next time. For example:

> *"I assumed `s.split()` would handle multiple consecutive spaces correctly, but it returned an empty string in the result. Next time I should test edge cases with consecutive whitespace and use `s.split()` without args, which collapses them automatically."*

This reflection is stored in the **reflection memory** (a small buffer, typically the last 1-3 reflections to fit in context).

### The loop

```
Reflection memory: []

For trial = 1, 2, ..., max_trials:
    trajectory = Actor.run(task, reflection_memory)
    reward = Evaluator.evaluate(trajectory)
    if reward == success:
        return trajectory
    reflection = SelfReflector.reflect(task, trajectory, reward)
    reflection_memory.append(reflection)
```

The crucial insight: the "policy update" happens in **textual context**, not in model weights. The agent's behavior changes across trials because its prompt now contains hard-won lessons from previous failures.

---

## Why "Verbal RL"?

Reflexion frames this as a kind of reinforcement learning:

- The **policy** is the LLM itself, parameterized by context (prompt + memory).
- The **reward** is the Evaluator's success signal.
- The **policy update** is the generated reflection appended to memory.
- The **episode** is one full attempt at the task.

It is not RL in the gradient-descent sense, but it has the same structural feedback loop: reward shapes behavior across attempts. The mechanism that performs the update is the LLM's own ability to generate useful natural-language critiques — language replaces gradients.

---

## Key Results

Reflexion was evaluated across three quite different task types:

- **HumanEval (Python coding):** GPT-4 baseline 80.1% pass@1; **Reflexion-GPT-4 reached 91.0% pass@1**, beating then-state-of-the-art. The Evaluator used the public unit tests as the reward signal; the agent iteratively reflected on test failures.
- **HotPotQA (multi-hop QA):** Reflexion improved accuracy by ~17 absolute points over a ReAct baseline by reflecting on retrieval failures and refining its search strategy.
- **AlfWorld (text-based household tasks):** lifted success rate from 75% to 90%+ on tasks requiring sequential decision-making; agents learned across episodes to plan more efficiently and avoid past mistakes.

A consistent observation: **most of the gains came in the first 1-3 retries.** Reflexion's benefit comes from being able to correct categorical mistakes (misreading the problem, using the wrong tool, missing an edge case) — once those are fixed, returns diminish.

---

## A Worked Example: Coding Agent on a Failing Test

Imagine a coding task: "Write a function `count_vowels(s)` that returns the number of vowels in string `s`, treating 'y' as a vowel."

**Trial 1.** The Actor writes:
```python
def count_vowels(s):
    return sum(1 for c in s if c in "aeiou")
```
The Evaluator runs the test suite. Test `count_vowels("rhythm")` expects 1 (because of the 'y') but got 0. Failure.

**Self-Reflector** receives the trajectory and the test failure. It writes:
```
The function counted only "aeiou" but the spec says 'y' should also be treated
as a vowel. The test "rhythm" expects 1. Next attempt should include 'y' in the
vowel set, and also probably handle uppercase letters since the spec doesn't
restrict case.
```

This reflection joins the memory.

**Trial 2.** The Actor now sees the reflection in its prompt and writes:
```python
def count_vowels(s):
    return sum(1 for c in s.lower() if c in "aeiouy")
```
Tests pass. Success.

Notice: nothing about the model changed. Only the *prompt context* changed — a new lesson was added. The model is in some sense "smarter on this task" because its context now contains task-specific learnings.

---

## What Counts as a Good Reflection?

A non-obvious finding from the paper: not all reflections are useful, and the format matters. The best reflections:

- **Identify the specific failure** ("the test expected 1 but I returned 0") rather than vague self-criticism ("I should be more careful").
- **Propose a concrete next action** ("include 'y' in the vowel set") rather than only diagnosing.
- **Generalize one step beyond the immediate failure** ("...and handle uppercase too") to prevent fixing one bug while introducing another.

Reflexion prompts the Self-Reflector with structured instructions that encourage this format. When the reflector role is given vague or unstructured prompts, the reflections degrade into generic self-help and don't actually shift behavior on retry.

This is essentially the same insight that emerges in process reward modeling and Meta-CoT — **specific, actionable critique is much more valuable than generic feedback**, both for humans and for LLM agents.

---

## Impact and Legacy

Reflexion's design pattern — actor + evaluator + reflector + memory — became a template for the agentic-LLM ecosystem. Its influences:

- **Coding agents**: SWE-agent, AutoCodeRover, OpenDevin, Cursor's iterative repair, Aider — all use a reflect-and-retry loop with test failures as the reward signal, structurally identical to Reflexion on HumanEval.
- **General-purpose agent frameworks**: AutoGPT, BabyAGI, LangChain agents, and Generative Agents (#58) all incorporate self-reflection / memory-of-failures patterns.
- **Self-improving reasoning**: Reflexion-style verbal RL bridges to o1 (#31) and R1 (#26), which can be viewed as compressing the reflect-and-retry loop into a single internal reasoning pass. Reflexion does it across episodes; o1 does it within a single response.
- **Tool use / Model Context Protocol (#59)**: The "agent uses a tool, observes result, reflects, retries" loop is now the standard interaction model for production agents.
- **Process supervision (#51)**: Reflections are essentially per-trajectory critiques; PRMs generalize this to per-step critiques used for training.
- **Voyager** (Minecraft agent) directly extends Reflexion with a skill library: successful reflections become reusable skills, not just one-off lessons.

Reflexion also shifted how the community thinks about RL with LLMs. Before, RL meant RLHF (#05) on millions of preference pairs. Reflexion showed there is a *cheap, prompt-time* form of reinforcement that doesn't require any gradient access — vital for closed-source frontier models where you can't fine-tune.

---

## Limitations

- **Costs scale with retries.** Each trial is a full LLM rollout; reflective agents are 3-10x more expensive than single-shot agents.
- **Requires a reliable evaluator.** On tasks where success is hard to check, the reward signal is noisy and reflections can be misleading.
- **Memory is fragile.** Long reflection chains can drift or contradict each other; most implementations keep only the last few.
- **Doesn't update weights.** Lessons learned in one task don't transfer to future tasks unless you incorporate them into prompts manually.

---

## Connections to Other Papers

- **ReAct (#21):** The Actor in Reflexion is essentially a ReAct agent. Reflexion adds the across-episode learning loop ReAct lacks.
- **Chain-of-Thought (#09):** CoT improves a single response; Reflexion improves across multiple responses by reflecting on outcomes.
- **Tree of Thoughts (#25):** Sibling approach — both expand the single-pass paradigm. ToT searches in width within one attempt; Reflexion iterates in depth across attempts.
- **STaR (#81) / Quiet-STaR (#82):** Use successful reasoning traces to train the model; Reflexion uses failed traces to update the prompt context. Different mechanisms, same overarching philosophy of learning from rollouts.
- **OpenAI o1 (#31) / DeepSeek-R1 (#26):** Internalize the reflect-and-retry pattern into a single inference pass via long hidden chains of thought.
- **Generative Agents (#58):** Use memory streams and reflection in a more social/simulation-oriented way; conceptual cousin of Reflexion.
- **RLHF / InstructGPT (#05):** Traditional weight-updating RL on LLMs; Reflexion shows a complementary in-context RL that needs no gradients.
- **Process Reward Models (#51):** Provide step-level critique signals — more granular than Reflexion's trajectory-level reflection.
- **Test-Time Compute (#50):** Reflexion is one of the canonical examples of trading more compute at inference for better answers.

---

## Key Takeaways

- **Verbal reinforcement learning**: an agent can improve across attempts by writing natural-language reflections on its failures and conditioning future attempts on them — no gradient updates needed.
- **Three-role decomposition** (Actor, Evaluator, Self-Reflector) cleanly separates *doing*, *judging*, and *learning*, making the loop modular and inspectable.
- **The reward signal can be sparse and binary** (pass/fail unit tests, task complete or not) — the LLM's reflection ability fills in the credit-assignment gap.
- **+10% pass@1 on HumanEval for GPT-4** demonstrated that even frontier models leave significant performance on the table when used single-shot.
- **Template for the agentic era**: coding agents, web agents, planning agents, and o1-style reasoning models all carry Reflexion's reflect-evaluate-retry DNA.
