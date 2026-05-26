# SWE-bench: Can Language Models Resolve Real-World GitHub Issues?

**Authors:** Carlos E. Jimenez, John Yang, Alexander Wettig, Shunyu Yao, Kexin Pei, Ofir Press, Karthik Narasimhan (Princeton, University of Chicago)
**Published:** October 2023 (ICLR 2024)
**Paper:** [arxiv.org/abs/2310.06770](https://arxiv.org/abs/2310.06770)

---

## Why This Paper Matters

SWE-bench changed how the field measures coding ability. Before it, benchmarks like HumanEval and MBPP asked models to write short, self-contained functions from a docstring. SWE-bench instead gave models a real bug report or feature request from a real open-source Python project and asked: can you produce a patch that makes the project's existing test suite pass? That single shift — from toy functions to entire repositories — exposed how far synthetic benchmarks were overestimating model usefulness, and it became the yardstick the whole industry now uses for "agentic" coding systems.

Within a year of release, SWE-bench was the benchmark every frontier lab competed on. Devin, Claude (Anthropic), GPT-4 / o-series, Cursor, Cognition, Aider, OpenDevin, and SWE-agent all reported SWE-bench numbers. OpenAI's curated "SWE-bench Verified" subset became the de facto standard for comparing coding agents. The benchmark's design — multi-file edits, large context, real failure modes — drove an entire research agenda on long-context retrieval, repo navigation, and autonomous agent loops.

---

## The Problem with Previous Coding Benchmarks

By 2023, models like GPT-4 were scoring 80%+ on HumanEval. That suggested coding was nearly solved. But anyone who actually used these models on real projects knew that wasn't true. The benchmarks were misleading because:

**1. Functions in isolation, not systems.** HumanEval gave a short docstring and asked for one function. Real software work means navigating thousands of files and understanding how they interact.

**2. No real bugs.** Synthetic tasks rarely matched the kind of subtle, context-dependent issues that actually appear in production code.

**3. Tiny context.** A 30-line problem cannot test whether a model can find the right file out of 500.

**4. Contamination risk.** Popular benchmarks leak into pretraining data. By 2023, HumanEval-style problems were probably memorized.

The field needed a benchmark that looked like actual software engineering work: read an issue, find the relevant code in a large repo, write a patch, verify it doesn't break anything.

---

## The Core Innovation

SWE-bench's idea is elegantly simple: **mine real pull requests from real GitHub repositories, and use the merged PR's tests as the success criterion.**

The construction process:

```
For each popular Python repo (e.g. django, sympy, scikit-learn):
    For each merged PR that fixes an issue:
        1. Take the issue text -> this is the "input"
        2. Take the repo state BEFORE the PR -> this is the codebase
        3. Take the tests added/modified in the PR -> these are the graders
        4. The model must produce a patch that:
              - Makes the new "fail-to-pass" tests pass
              - Does NOT break existing "pass-to-pass" tests
```

A model is given the issue and the entire repository. It must output a unified diff. The diff is applied, the test suite is executed in a Docker container, and success is binary: all required tests pass, or the attempt fails.

---

## How It Works

### Dataset construction

The authors scraped 12 large Python repositories: `django`, `sympy`, `scikit-learn`, `matplotlib`, `flask`, `requests`, `pylint`, `pytest`, `seaborn`, `astropy`, `sphinx`, and `xarray`. From thousands of merged PRs, they filtered for ones that:

- Resolved an issue (linked via "fixes #123")
- Modified at least one test file (so there's a verifiable check)
- Could be reproduced in a fresh environment

This yielded **2,294 task instances**. Each instance contains:

| Field | Content |
|-------|---------|
| `repo` | e.g. `django/django` |
| `base_commit` | SHA of the codebase before the fix |
| `problem_statement` | The original GitHub issue text |
| `patch` | The gold human patch (held out) |
| `test_patch` | The tests that determine correctness |
| `FAIL_TO_PASS` | Tests that fail before, pass after |
| `PASS_TO_PASS` | Tests that must keep passing |

### The evaluation loop

```
1. Spin up Docker image at repo's base_commit
2. Give the model: issue text + access to the repo
3. Model produces a .patch (unified diff)
4. Apply the patch with `git apply`
5. Run the FAIL_TO_PASS and PASS_TO_PASS tests
6. Pass if and only if ALL required tests pass
```

There is no partial credit. A patch either resolves the issue or it doesn't.

### Why this is hard

A typical SWE-bench instance involves:

- **Repos with 100K+ lines of code.** The model can't read everything.
- **Multi-file edits.** ~20% of fixes touch 2+ files.
- **Long context.** Average repo has thousands of files; relevant ones must be found.
- **Subtle bugs.** Off-by-one errors, edge cases, regressions from previous fixes.
- **API conventions.** Patches must match the project's style and idioms.

---

## Key Results

### The 2023 baseline was brutal

When the paper was published, the strongest models barely registered:

| Model | SWE-bench Resolve Rate |
|-------|------------------------|
| Claude 2 | 1.96% |
| GPT-4 | 1.74% |
| ChatGPT-3.5 | 0.17% |
| SWE-Llama 13B | 0.70% |

GPT-4, the best model on every other benchmark, solved fewer than 2 in 100 real bugs. This was a wake-up call: agentic coding was nowhere near solved.

### The progression (2023 to 2025)

SWE-bench scores tracked the rise of true coding agents:

| System | Approximate Resolve Rate |
|--------|--------------------------|
| GPT-4 (Oct 2023, no agent) | ~2% |
| SWE-agent + GPT-4 (early 2024) | ~12% |
| Devin (Mar 2024, full Verified) | ~14% |
| Claude 3.5 Sonnet + agent (mid 2024) | ~49% |
| Claude 3.5 Sonnet (new, late 2024) | ~49% on Verified |
| Frontier agents (2025) | 60-70%+ on Verified |

In two years the field went from 2% to 70% — one of the steepest improvement curves in any AI benchmark.

### SWE-bench Verified

OpenAI, working with the original authors, released **SWE-bench Verified** in August 2024: 500 instances that human engineers confirmed were unambiguous, solvable, and had appropriate test coverage. Verified became the new gold standard because the original full benchmark had a long tail of underspecified or unfair instances. Every major lab now reports Verified numbers.

### SWE-bench Lite, Multimodal, Multilingual

The success of the original spawned a family:

- **SWE-bench Lite:** 300 simpler instances for quick iteration
- **SWE-bench Multimodal:** instances involving UI screenshots
- **Multi-SWE-bench:** extends to Java, Go, Rust, TypeScript, etc.

---

## Impact and Legacy

SWE-bench is responsible for crystallizing the modern "coding agent" as a research target. Before it, coding research was largely about single-turn code generation. After it, every serious system involves:

- **Repo-level retrieval.** Finding which files matter.
- **Multi-turn execution.** Edit, run tests, observe output, iterate.
- **Tool use.** Shell commands, file editors, search.
- **Long context.** Often 100K+ tokens of code.

The SWE-agent paper (also from Princeton, 2024) introduced the **Agent-Computer Interface** — a structured terminal designed for LLMs — and its design was downstream of SWE-bench evaluation needs. Devin's launch in March 2024 was framed almost entirely around its SWE-bench number. Anthropic's coding-focused releases (Claude 3.5 Sonnet, Claude 4) all cite Verified prominently.

It also shifted the commercial AI story. SWE-bench made it concrete that LLMs could actually do paid software-engineering work, not just autocomplete snippets — fueling the rise of Cursor, Cognition, Replit Agent, Codeium, and Anthropic's own Claude Code.

---

## Connections to Other Papers

- **Codex (#56):** First LLM-from-GitHub-code paper; established that pretraining on code works. SWE-bench measures the next frontier — using that code competence in real repos.
- **ReAct (#21):** The reason/act loop is the template for every SWE-bench agent.
- **Reflexion (#83) and Self-Refine (#84):** Iteration strategies used by leading SWE-bench agents.
- **Voyager (#86):** Skill-acquisition agent in Minecraft — a sibling line of work showing agents can operate in complex environments.
- **Tree of Thoughts (#25):** Search strategies that some agent systems use over candidate patches.
- **Generative Agents (#58):** Earlier demonstration that LLMs can drive complex multi-step behavior.
- **InstructGPT (#5) and DPO (#19):** Alignment methods that make models useful enough to act as agents at all.
- **Test-Time Compute (#50):** SWE-bench rewards spending more tokens — many top agents scale heavily at inference.
- **Model Context Protocol (#59):** Standardized tool interface that many SWE-bench agents now use.

---

## Key Takeaways

1. **Real beats synthetic.** Mining actual PRs gave a benchmark with real difficulty distribution and natural held-out tests.
2. **Binary, verifiable grading.** Tests pass or they don't — no human judgment required, which makes the benchmark trustworthy at scale.
3. **It exposed agentic gaps.** Models that ace HumanEval failed SWE-bench, proving repo-scale work is a fundamentally different skill.
4. **It drove an entire research wave.** Coding agents, long-context models, and tool-use APIs all advanced largely because SWE-bench made progress measurable.
5. **Verified is the standard.** When comparing modern coding systems, SWE-bench Verified is the number that matters.
