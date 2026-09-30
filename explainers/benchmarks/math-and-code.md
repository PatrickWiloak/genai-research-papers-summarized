# Math and Code Benchmarks

**In one line:** Maths and coding benchmarks are the easiest to grade automatically and the fastest to go stale, so trust the fresh, held-out and actively maintained ones (competition problems released after training, FrontierMath's holdout, versioned agent suites) over the famous static ones.
**Last reviewed:** 2026-09-30

---

## The short version

- Maths and code are popular benchmark domains because answers can be **checked by a machine**: a number either matches, a program either passes its tests.
- The early generation (**GSM8K**, **MATH**, **HumanEval**) is saturated and contaminated. They still appear in tables but no longer separate frontier models.
- The replacements take one of three routes: **fresh problems** (AIME each year, LiveCodeBench, MathArena), **secret problems** (FrontierMath's holdout set), or **realistic long tasks** (SWE-bench, Terminal-Bench).
- The realistic tasks turned out to be **fragile**. In 2026 OpenAI stopped reporting SWE-bench Verified (February) and then withdrew its endorsement of SWE-bench Pro (July) after audits found large shares of broken tasks. Epoch AI now rates both "Flawed".
- For coding agents, the **harness** (the scaffold, tools, time limits and number of attempts wrapped around the model) can move a score as much as the model does. Always read the footnote.

## Why maths and code dominate launch tables

A language model's output is text. For most tasks, judging whether that text is good needs a human or another model. Maths and code are the exceptions:

```
Maths:   model's final answer  ==  answer key ?          (exact match)
Code:    model's program       ->  run hidden tests  ->  pass / fail
Agent:   model edits a repo    ->  run the repo's tests -> resolved / not
```

That makes them cheap, objective and repeatable, and it also makes them ideal training signals. Reinforcement learning with verifiable rewards (see [RLVR](../../papers/techniques/39-rlvr/summary.md) and [DeepSeek-R1](../../papers/language-models/26-deepseek-r1/summary.md)) trains on exactly this kind of checkable problem, which is one reason scores on these benchmarks rose so fast after 2024. It is also why a model can be tuned toward a benchmark's style without getting better at the real skill.

## Maths benchmarks

### GSM8K (2021)

**Measures:** multi-step arithmetic reasoning in plain language. **Format:** 8,500 grade-school word problems, each needing several steps of arithmetic. **Scoring:** exact match on the final number.

OpenAI built it to study training **verifiers** that rank sampled solutions. It became the standard "can it do word problems" test and is now saturated.

**Known flaws.** Contamination is well documented. Scale AI's **GSM1k** (2024) wrote 1,000 new problems matched to GSM8K's difficulty and found accuracy drops of up to 8 points, with some model families showing systematic overfitting, and a correlation between how likely a model was to generate GSM8K text and how much it dropped. Apple's **GSM-Symbolic** (2024) generated variants from templates and found that performance fell when only the numbers were changed and fell further as irrelevant clauses were added, which suggests pattern matching rather than robust reasoning. See [Do LLMs Reason?](../open-questions/do-llms-reason.md) for both sides of that argument.

### MATH and MATH-500 (2021, 2023)

**Measures:** high-school competition mathematics. **Format:** 12,500 competition problems, each with a full step-by-step solution. **Scoring:** exact match of the final answer, after normalising LaTeX.

When released, the authors wrote that "scaling is not currently solving MATH". It was solved anyway, mostly by chain-of-thought and then reasoning models. Most modern reports use **MATH-500**: OpenAI's "Let's Verify Step by Step" (2023) moved 4,500 of the MATH test problems into training and evaluated only on the remaining 500, chosen at random. See [Process Reward Models](../../papers/techniques/51-process-reward-models/summary.md).

**Known flaws.** Saturated; answer normalisation is fiddly (is `1/2` the same as `0.5` or `\frac{1}{2}`?), and different graders give different scores for the same outputs.

### AIME

**Measures:** hard high-school competition maths. **Format:** the American Invitational Mathematics Examination; each exam has 15 problems whose answers are integers from 0 to 999, and there are two exams (I and II) each year. **Scoring:** exact match, usually averaged over many samples because 30 questions per year is a tiny test.

AIME became the headline maths number for reasoning models because each year brings **new, unseen problems**.

**Known flaws.** Last year's AIME is on the internet by the time the next model trains. The **MathArena** project (ETH Zurich, 2025) evaluates models on competitions as soon as they are held and "found strong signs of contamination in AIME 2024". So "AIME 2024" in a 2026 table is a much weaker test than the current year's exam. The small size also means one question is worth over 3 points.

### FrontierMath (2024 to 2026)

**Measures:** research-level mathematics. **Format:** originally about 350 unpublished problems written by mathematicians: Tiers 1-3 (300 problems, ranging from olympiad-style to early PhD research) and Tier 4 (50 problems, far harder). Answers are closed-form objects (a number, a matrix, a formula) that can be checked automatically and are hard to guess. Models get a Python environment. **Scoring:** accuracy.

Built by Epoch AI; at launch in November 2024, state-of-the-art models solved under 2 percent.

**Known flaws.**

- **Funding and access.** OpenAI commissioned and funded FrontierMath and has access to most problems. Epoch keeps a **holdout set** for clean evaluation. For Tier 4, OpenAI has access to 30 of the 50 problems and Epoch holds out 20 (the January 2026 evaluation used 48: 28 OpenAI-accessible, 20 held out).
- **Errors.** On June 12, 2026 Epoch released **FrontierMath v2**, after an audit found errors in 42 percent of problems. Scores rose across the board and v1 and v2 numbers are not comparable.

**Where the frontier stands.** In January 2026, Epoch reported a manual GPT-5.2 Pro run at 31 percent on Tier 4 (15 of 48 problems), scoring 50 percent on Epoch's held-out problems against 18 percent on the ones OpenAI had seen, which Epoch read as "no evidence of over-fitting". As of the v2 release in June 2026, Epoch's newsletter said Anthropic's Fable 5 topped its FrontierMath leaderboards. Check Epoch's live pages for current figures; they move monthly.

## Code benchmarks

### HumanEval (2021)

**Measures:** writing a single Python function from a docstring. **Format:** 164 hand-written problems, each with unit tests. **Scoring:** **pass@k**, the chance that at least one of k sampled programs passes all tests; pass@1 is the usual headline.

Introduced with [Codex](../../papers/language-models/56-codex/summary.md), which solved 28.8 percent at pass@1 and 70.2 percent with 100 samples.

**Known flaws.** Too few tests per problem: **EvalPlus** (2023) added 80 times more tests (HumanEval+) and found pass rates dropped by up to 19.3-28.9 percent, and that weak tests had mis-ranked models. It is also tiny, self-contained, heavily contaminated and saturated.

### LiveCodeBench (2024 onward)

**Measures:** competitive-programming ability, plus self-repair, code execution and test-output prediction. **Format:** problems continuously collected from LeetCode, AtCoder and Codeforces, each tagged with its release date. **Scoring:** pass@1 on hidden tests.

The key idea is the **time window**: you score a model only on problems published after its training cutoff. The paper used this to show contamination, reporting some models only on problems after August 2023. The dataset grew from 400 problems (v1, May 2023 to March 2024) to 1,055 (v6, to April 2025).

**Known flaws.** Competition puzzles are not everyday software engineering, and because every lab picks its own window, numbers from different reports often cover different problems.

### SWE-bench, SWE-bench Verified and SWE-bench Pro (2023 to 2026)

**Measures:** resolving real GitHub issues in large Python repositories. **Format:** the model gets an issue and the repository before the fix, and must produce a patch. **Scoring:** the repository's own tests, including ones added by the real fix, must pass. The full story is in the [SWE-bench summary](../../papers/techniques/84-swe-bench/summary.md).

**SWE-bench Verified** (August 2024) was a 500-task subset that OpenAI had human engineers screen for solvable, fairly-tested problems. It became the industry's headline coding number.

**What went wrong.**

- **February 23, 2026:** OpenAI said it would stop reporting SWE-bench Verified. It audited 138 tasks (27.6 percent of the set) that o3 failed to solve consistently over 64 runs; 59.4 percent had flawed tests that rejected functionally correct fixes, a floor of 16.4 percent of the whole benchmark. It also found signs of contamination: frontier models could reproduce details of the reference fixes. Top scores had moved only from about 75 to 81 percent in six months.
- OpenAI recommended **SWE-bench Pro** (Scale AI, 1,865 tasks: 731 public, 276 private, 858 held out) instead.
- **July 8, 2026:** OpenAI published "Separating signal from noise in coding evaluations", estimating about 30 percent of Pro's public tasks were broken (misleading prompts, overly strict tests, underspecified prompts, low-coverage tests), and withdrew its recommendation. Independent audits by Jonathan Gabor (February 2026) and Datacurve (May 2026) had also found widespread issues.
- As of its September 1, 2026 review, **Epoch AI rates both SWE-bench Verified and SWE-bench Pro "Flawed"**, its label for benchmarks where at least 20 percent of an inspected sample contains errors, or where an issue corrupts grading at scale.

**Harness sensitivity.** SWE-bench scores depend heavily on the agent scaffold. Anthropic's Sonnet 4.5 launch reported 77.2 percent averaged over 10 trials with a two-tool scaffold and a specific prompt addition, and 82.0 percent with parallel attempts plus a selection model. Same model, two honest numbers.

### Terminal-Bench (2025 onward)

**Measures:** getting real work done in a command-line environment: software engineering, security, scientific computing, data science, debugging, system administration. **Format:** each task is a container with a starting environment, a human-written reference solution and tests. **Scoring:** fraction of tasks where the tests pass.

Built by Stanford and the Laude Institute. Terminal-Bench 2.0 (November 2025, published at ICLR 2026) had 89 tasks each verified by three reviewers.

It is also the clearest example of a **continuous benchmark**. In 2026 the team shipped 2.1 (May, fixing 28 tasks), 3.0 (July 30, 74 new harder tasks across 7 domains, best models about 34 percent at launch) and 4.0 (August 28, which removed 8 tasks, including 2 for saturation and 2 because solutions had become public, and fixed 19). They now version it like software.

**Known flaws.** In April 2026 the maintainers reported **cheating and reward hacking** on the leaderboard: one submitter had stored encrypted solutions inside its agent, another uploaded the test files, and one agent sometimes downloaded solutions from the internet. They now require full trajectories for every passing trial and run an agent judge over them. The harness matters here too: Anthropic's Opus 4.8 launch footnote notes that GPT-5.5 scores 83.4 percent on Terminal-Bench 2.1 with the Codex CLI harness, a different number from the one in the table, which used the common Terminus-2 harness.

## What to watch

- **Who builds the next coding benchmark.** OpenAI's July 2026 post argued for benchmarks designed from scratch by experienced developers rather than mined from GitHub history. Watch for a successor to SWE-bench that labs actually adopt.
- **Continuous benchmarks.** If Terminal-Bench's version-and-prune model spreads, headline numbers will come with version tags (for example "TB 4.0") and will not be comparable across versions.
- **FrontierMath v2 and its holdout gap.** The difference between scores on held-out and seen problems is one of the few direct contamination checks available.
- **Proof-based maths.** Final-answer benchmarks cannot grade proofs. MathArena's proof evaluations and formal-proof work (see [AlphaGeometry](../../papers/techniques/61-alphageometry/summary.md)) point where maths evaluation goes next.

Launch-table reading tips are collected in [Knowledge and Reasoning](knowledge-and-reasoning.md#how-to-read-a-model-launchs-benchmark-table).

## Read next

- [SWE-bench](../../papers/techniques/84-swe-bench/summary.md), [Codex](../../papers/language-models/56-codex/summary.md), [RLVR](../../papers/techniques/39-rlvr/summary.md), [rStar-Math](../../papers/techniques/35-rstar-math/summary.md), [Process Reward Models](../../papers/techniques/51-process-reward-models/summary.md)
- Explainers: [Agents and Computer Use](agents-and-computer-use.md), [Contamination and Saturation](contamination-and-saturation.md), [Reasoning Models](../concepts/reasoning-models.md)
- Sibling repo: [Evals for LLMs](https://github.com/PatrickWiloak/cloud-data-ai-security-zero-to-hero/blob/main/learn/concepts/evals-for-llms.md)

## Sources

- Cobbe et al., "Training Verifiers to Solve Math Word Problems" (GSM8K, 2021): https://arxiv.org/abs/2110.14168
- Zhang et al., "A Careful Examination of Large Language Model Performance on Grade School Arithmetic" (GSM1k, 2024): https://arxiv.org/abs/2405.00332
- Mirzadeh et al., "GSM-Symbolic" (2024): https://arxiv.org/abs/2410.05229
- Hendrycks et al., "Measuring Mathematical Problem Solving With the MATH Dataset" (2021): https://arxiv.org/abs/2103.03874
- Lightman et al., "Let's Verify Step by Step" (2023) and PRM800K README (MATH-500 split): https://arxiv.org/abs/2305.20050, https://github.com/openai/prm800k
- Balunović et al., "MathArena: Evaluating LLMs on Uncontaminated Math Competitions" (2025): https://arxiv.org/abs/2505.23281
- Glazer et al., "FrontierMath" (2024): https://arxiv.org/abs/2411.04872
- Epoch AI, FrontierMath about page (tiers, funding, access): https://epoch.ai/frontiermath/tiers-1-4/about
- Epoch AI, "New record on FrontierMath Tier 4" (January 23, 2026): https://epochai.substack.com/p/new-record-on-frontiermath-tier-4
- Epoch AI, "The Epoch Brief - June 12, 2026" (FrontierMath v2): https://epochai.substack.com/p/the-epoch-brief-june-12-2026
- Chen et al., "Evaluating Large Language Models Trained on Code" (HumanEval, 2021): https://arxiv.org/abs/2107.03374
- Liu et al., "Is Your Code Generated by ChatGPT Really Correct?" (EvalPlus, 2023): https://arxiv.org/abs/2305.01210
- Jain et al., "LiveCodeBench" (2024) and repository README (release versions): https://arxiv.org/abs/2403.07974, https://github.com/LiveCodeBench/LiveCodeBench
- Jimenez et al., "SWE-bench" (2023): https://arxiv.org/abs/2310.06770
- OpenAI, "Why SWE-bench Verified no longer measures frontier coding capabilities" (February 23, 2026): https://openai.com/index/why-we-no-longer-evaluate-swe-bench-verified/
- OpenAI, "Separating signal from noise in coding evaluations" (July 8, 2026): https://openai.com/index/separating-signal-from-noise-coding-evaluations/
- Epoch AI, SWE-bench Verified benchmark review: https://epoch.ai/benchmarks/swe-bench-verified/review
- Epoch AI, SWE-Bench Pro benchmark review (reviewed September 1, 2026): https://epoch.ai/benchmarks/swe-bench-pro/review
- Pebblous, "SWE-bench Verified Retired: What OpenAI Found" (secondary summary of the February audit): https://blog.pebblous.ai/blog/swe-bench-verified-retired/en/
- Anthropic, "Introducing Claude Sonnet 4.5" (SWE-bench methodology footnote): https://www.anthropic.com/news/claude-sonnet-4-5
- Anthropic, "Introducing Claude Opus 4.8" (May 28, 2026; Terminal-Bench 2.1 footnote): https://www.anthropic.com/news/claude-opus-4-8
- Merrill et al., "Terminal-Bench: Benchmarking Agents on Hard, Realistic Tasks in Command Line Interfaces" (ICLR 2026): https://arxiv.org/abs/2601.11868
- Terminal-Bench team posts: "Leaderboard Integrity Update" (April 19, 2026), "Terminal-Bench 3.0" and "Continuous Benchmarks" (July 30, 2026), "Terminal-Bench 4.0" (August 28, 2026): https://www.tbench.ai/news/leaderboard-integrity-update, https://www.tbench.ai/news/terminal-bench-3-0, https://www.tbench.ai/news/continuous-benchmarks, https://www.tbench.ai/news/terminal-bench-4-0
