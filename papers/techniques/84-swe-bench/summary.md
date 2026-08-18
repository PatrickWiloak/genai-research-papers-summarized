---
title: "SWE-bench: Can Language Models Resolve Real-World GitHub Issues?"
slug: "84-swe-bench"
number: 84
category: "techniques"
authors: "Carlos E. Jimenez, John Yang, Alexander Wettig, Shunyu Yao, Kexin Pei, Ofir Press, Karthik Narasimhan (Princeton University, University of Chicago)"
published: "October 2023 (ICLR 2024)"
year: 2023
url: "https://arxiv.org/abs/2310.06770"
tags: ["evaluation", "benchmarks", "agents", "code"]
---

# SWE-bench: Can Language Models Resolve Real-World GitHub Issues?

**Authors:** Carlos E. Jimenez, John Yang, Alexander Wettig, Shunyu Yao, Kexin Pei, Ofir Press, Karthik Narasimhan (Princeton University, University of Chicago)
**Published:** October 2023 (ICLR 2024)
**Paper:** [arxiv.org/abs/2310.06770](https://arxiv.org/abs/2310.06770)

---

## Why This Matters

SWE-bench is **the benchmark that defines the coding-agent era**. When a model announcement quotes a percentage for software engineering, it is almost always this. It replaced toy function-completion benchmarks with a task that looks like the actual job: read an issue, navigate a large unfamiliar repository, change the right files, and pass the maintainers' tests.

- **Real issues from real repositories.** 2,294 issue and pull-request pairs from 12 widely used Python projects including Django, scikit-learn, SymPy, and Matplotlib.
- **Verified by the project's own tests**, not by string matching or a judge model.
- **It was brutally hard at launch.** The best systems in the original paper resolved under 2 percent of instances.
- **It became the industry's headline metric.** Progress from roughly 2 percent to the 70-80 percent range across 2024 and 2025 is the clearest available measure of agentic capability growth.

**The insight:** benchmarks like HumanEval ask a model to write one self-contained function from a docstring. Real software engineering is mostly the opposite: understanding a large existing codebase well enough to make a small, correct, non-breaking change. Build the benchmark from artifacts that already exist - merged pull requests that closed issues and came with tests - and you get a hard, realistic, automatically gradable task at scale.

---

## The Problem: Coding Benchmarks Were Not Measuring Software Engineering

By late 2023, [Codex](../../language-models/56-codex/summary.md)-style models were scoring high on HumanEval and MBPP, and those numbers meant progressively less:

- **Self-contained.** One function, no repository, no dependencies, no other files.
- **Small.** A few lines of output from a clear specification.
- **Saturating.** Scores in the 80s and 90s left little headroom.
- **Contaminated.** These problems are all over GitHub and therefore all over pretraining data.

None of it resembled the work. A developer's day is spent reading code they did not write, locating where a symptom originates, and making a change that does not break the other 200,000 lines.

---

## The Core Innovation

Mine merged pull requests that (a) close an issue and (b) add or modify tests. That gives a task with a natural specification and a natural grader.

```
For each instance:

  INPUT to the model:
    - the issue text (a real bug report or feature request)
    - the repository at the commit just BEFORE the fix
      (thousands of files, often 100k+ lines)

  The model must produce:
    - a patch (diff) against the repository

  GRADING (fully automatic):
    - apply the patch
    - run FAIL_TO_PASS tests: tests added by the real PR that
      failed before and must pass now  -> did you fix it?
    - run PASS_TO_PASS tests: tests that passed before and must
      still pass                        -> did you break anything?
    - both must pass, or the instance is failed

  No partial credit. No judge model. No string comparison.
```

Every design choice serves realism:
- **Issue text, not a specification.** Real issues are vague, sometimes wrong, occasionally include a stack trace, and rarely say which file to edit.
- **Whole repository as context.** Far beyond any context window at the time, so retrieval and navigation are part of the task.
- **The regression tests are the honest part.** PASS_TO_PASS is what prevents "delete the failing assertion" strategies from scoring.
- **Multi-file edits are common**, as real fixes often are.

---

## Key Components Explained

### 1. FAIL_TO_PASS and PASS_TO_PASS
**What it does:** Encodes "fix the bug without breaking anything" as an automatic check.
**How it works:** The real PR's added tests define success; the pre-existing suite defines non-regression. This is a genuinely well-posed automatic grader for an open-ended task, which is rare and is much of why the benchmark works.

### 2. SWE-bench Lite and SWE-bench Verified
**What it does:** Fixes cost and quality problems with the full set.
**How it works:**
- **Lite** (300 instances) is a cheaper subset for iteration.
- **Verified** (500 instances, released August 2024 in collaboration with OpenAI) is the important one: professional developers reviewed instances and removed those that were **underspecified** (the issue does not contain enough information to know what fix is wanted) or had **broken or unfair tests**. A meaningful fraction of the original set was unsolvable in principle. Verified is now the standard reporting set, and comparing a Verified number to an original-set number is a common and serious error when reading model announcements.

### 3. The Agent Scaffold Matters as Much as the Model
**What it does:** Turns a model into a system that can attempt the task.
**How it works:** SWE-agent (from the same group) showed that giving the model a purpose-built interface - a file viewer with sensible chunking, a search tool, an editor with syntax checking on write - roughly tripled resolution rates over naive retrieval-and-patch approaches. The lesson generalized far beyond this benchmark: **the tool interface an agent sees is a first-class design problem**, not plumbing. Every coding agent since has invested heavily here.

### 4. Contamination Pressure
**What it does:** Limits the benchmark's shelf life.
**How it works:** The issues and their real fixes are public on GitHub and enter pretraining corpora. Newer models may have seen the actual patches. This is why SWE-bench Multimodal, SWE-bench Pro, and continuously refreshed variants with recent issues exist, and why any single number should be read with the model's training cutoff in mind.

---

## Key Results

- **At publication (2023):** the best configuration resolved **1.96 percent** of instances. Claude 2 with retrieval was the strongest evaluated model and still under 5 percent on the easier subsets. The gap between "writes good functions" and "does software engineering" was enormous.
- **2024:** SWE-agent scaffolding plus GPT-4 reached the low teens; [Claude 3.5 Sonnet](../../language-models/30-claude-3.5-sonnet/summary.md) reached roughly 49 percent on Verified, which was a step change.
- **2025:** frontier models reported roughly 70-80 percent on SWE-bench Verified - [Claude 4 family](../../language-models/43-claude4/summary.md) and [GPT-5](../../language-models/42-gpt5/summary.md) are both in this range.
- Going from about 2 percent to about 80 percent in roughly two years on a benchmark designed to be hard is one of the most striking capability trajectories on record.

---

## Why This Was Revolutionary

- **Made the benchmark match the job.** Realistic input, realistic context size, realistic grading.
- **Created enough headroom to be useful for years.** A benchmark that starts at 2 percent measures progress for a long time, unlike one that starts at 80.
- **Automatic grading without a judge.** Test execution is objective, cheap, and not gameable by writing persuasively - properties most agentic benchmarks lack.
- **Drove the agent-scaffold research program.** SWE-agent's finding that the interface matters as much as the model reshaped how agent systems are built.
- **Provided the training signal, not just the score.** Test-verified outcomes are exactly the reward function that [RLVR](../39-rlvr/summary.md)-style training needs, and agentic coding RL is now a major post-training investment across labs.

---

## Real-World Impact

- **The standard coding metric** in model announcements from Anthropic, OpenAI, Google, DeepSeek, and Qwen.
- **Product decisions follow it.** The jump into the 40-50 percent range is roughly when autonomous coding agents (Devin, Claude Code, Cursor agent mode, OpenAI's coding agents) became commercially viable rather than demos.
- **Shaped agent architecture** industry-wide: repository navigation, file-view chunking, edit validation, and test-run loops are all standard because of what worked here.
- **Inspired a family of benchmarks** - SWE-bench Multimodal, SWE-Lancer (economically valued tasks), Aider's polyglot benchmark, TerminalBench, and internal enterprise variants built on private repositories.
- **Fed the training loop.** Verified-outcome tasks are used as RL environments, which is part of why scores moved so fast.

---

## Key Takeaways for Practitioners

1. **Read which variant is being quoted.** Verified, Lite, and full SWE-bench are different numbers. Verified is the meaningful one; the others are not comparable to it.
2. **Scaffolding is a large share of the score.** A weaker model with a good agent harness routinely beats a stronger model with a bad one. If you are building agents, invest in the tool interface.
3. **80 percent on SWE-bench is not 80 percent on your codebase.** These are well-tested open-source Python projects with clear issues. Your repository has less test coverage, more implicit context, and vaguer tickets.
4. **The grading design is worth copying.** Fixed-it tests plus did-not-break-it tests is an excellent template for evaluating agents on your own tasks.
5. **Watch for contamination.** A model released long after the benchmark may have seen the fixes. Recent-issue variants are the honest check.
6. **Test coverage is now infrastructure for AI.** Repositories with good tests are ones agents can work in safely; repositories without them are not.

---

## Limitations & Future Directions

- **Python only, 12 repositories**, mostly libraries. Not representative of application code, frontend work, or other language ecosystems.
- **Bug fixes, not feature development.** Real engineering includes design, ambiguity resolution, and stakeholder negotiation, none of which appear.
- **Passing tests is not the same as being correct.** A patch can satisfy the test suite while being poor code, or correct only by coincidence. Human review of high-scoring patches consistently finds quality issues the tests do not catch.
- **Contamination worsens over time**, which limits the lifespan of any fixed set.
- **Approaching saturation.** With frontier models in the 70-80 percent range, the remaining instances are increasingly the ambiguous or unfair ones, and headroom is running out. Harder successors are already appearing.

---

## Further Reading

- **Original Paper:** [arxiv.org/abs/2310.06770](https://arxiv.org/abs/2310.06770)
- **Leaderboard:** [swebench.com](https://www.swebench.com/)
- **SWE-agent (the scaffolding paper):** [arxiv.org/abs/2405.15793](https://arxiv.org/abs/2405.15793)
- **SWE-bench Verified:** [openai.com/index/introducing-swe-bench-verified](https://openai.com/index/introducing-swe-bench-verified/)
- **In this collection:** [Codex](../../language-models/56-codex/summary.md), [ReAct](../21-react/summary.md), [Reflexion](../78-reflexion/summary.md), [RLVR](../39-rlvr/summary.md)

## Citation

```bibtex
@inproceedings{jimenez2024swebench,
  title={SWE-bench: Can Language Models Resolve Real-World GitHub Issues?},
  author={Jimenez, Carlos E. and Yang, John and Wettig, Alexander and Yao, Shunyu and Pei, Kexin and Press, Ofir and Narasimhan, Karthik},
  booktitle={International Conference on Learning Representations},
  year={2024}
}
```

<!-- related:start -->

---

## Related in This Collection

- [ReAct: Synergizing Reasoning and Acting in Language Models](../../techniques/21-react/summary.md)
- [Qwen3: Technical Report](../../language-models/28-qwen3/summary.md)
- [Claude 3.5 Sonnet: Computer Use and Enhanced Capabilities](../../language-models/30-claude-3.5-sonnet/summary.md)
- [GPT-4 Technical Report](../../language-models/36-gpt4/summary.md)
- [RLVR: Reinforcement Learning from Verifiable Rewards](../../techniques/39-rlvr/summary.md)
- [GPT-5: Unified Intelligence](../../language-models/42-gpt5/summary.md)
- [Claude 4 Family: The Agentic AI Leader](../../language-models/43-claude4/summary.md)
- [Codex: Evaluating Large Language Models Trained on Code](../../language-models/56-codex/summary.md)

<!-- related:end -->
