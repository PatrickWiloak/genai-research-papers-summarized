# Agents and Computer-Use Benchmarks

**In one line:** Agent benchmarks score whether a model can finish a multi-step job in a live environment (a desktop, a website, a customer-service conversation), and the most useful single trend line is METR's "time horizon": the length of human task an AI can complete half the time.
**Last reviewed:** 2026-09-30

---

## The short version

- An **agent benchmark** gives a model tools and an environment, lets it act for many steps, then checks the **end state**: is the file saved, the order refunded, the database correct? No one grades the prose.
- The main families: **computer use** (OSWorld), **web tasks** (WebArena), **tool use with a simulated customer** (tau-bench), **general assistant questions** (GAIA), and **task length** (METR time horizons). Coding agents (SWE-bench, Terminal-Bench) are covered in [Math and Code](math-and-code.md).
- Progress has been fast. OSWorld went from 12 percent for the best model in April 2024 to scores above the paper's 72 percent human baseline by 2026, which is why **OSWorld 2.0** (June 2026) was built with 108 tasks that take a skilled human about 1.6 hours each. The best agent at launch completed 20.6 percent.
- METR's measurements show the length of task AI agents can complete doubling roughly **every seven months since 2019**, faster since 2024. By May 2026 the best measured model was beyond what METR's task suite can reliably measure (above 16 hours).
- The recurring flaws: **environments break**, **graders are too lenient or too strict**, **the harness matters as much as the model**, and **one success is not reliability**. tau-bench's pass^k metric exists because agents that succeed once often fail the same task on retry.

## What makes agent benchmarks different

A chat benchmark asks one question and grades one answer. An agent benchmark runs a loop:

```
 goal: "Export the Q3 sheet as PDF and email it to Sam"
   |
   v
 [model] --action--> [environment: VM, browser, API, simulated user]
    ^                        |
    +------observation-------+      (repeat up to N steps)
   |
   v
 checker inspects final state: PDF exists? email sent to right address?
   -> success / fail  (or partial credit per checkpoint)
```

This is closer to real work, but it adds moving parts that ordinary benchmarks do not have: a **step budget**, a **harness** (the prompts, tools and memory tricks around the model), a live environment that can crash or change, and a checker script that has to anticipate every valid way to succeed. Each is a source of noise or gaming. For the design side of building agents, see [Building Effective Agents](../../papers/essays/115-building-effective-agents/summary.md) and [ReAct](../../papers/techniques/21-react/summary.md); for tool calling basics, the sibling repo's [Tool Use and Function Calling](https://github.com/PatrickWiloak/cloud-data-ai-security-zero-to-hero/blob/main/learn/concepts/tool-use-and-function-calling.md).

## The benchmarks

### OSWorld, OSWorld-Verified and OSWorld 2.0

**Measures:** operating a real computer through the screen, mouse and keyboard. **Format:** a virtual machine running real applications (browser, office suite, code editor, file manager). The original 2024 benchmark had 369 tasks, each with a set-up configuration and a custom checking script; tasks include workflows that span several apps. **Scoring:** execution-based: the checker inspects files, settings or application state after the agent stops.

At release the best model succeeded on 12.24 percent of tasks versus 72.36 percent for humans, mostly failing on **GUI grounding** (finding the right thing on screen to click). Full summary: [OSWorld](../../papers/techniques/139-osworld/summary.md).

| Date | Event |
|---|---|
| April 2024 | OSWorld released: best model 12.24%, humans 72.36% |
| July 28, 2025 | **OSWorld-Verified**: an in-place upgrade fixing community-reported task problems, improving grading and infrastructure |
| September 2025 | Anthropic reports Claude Sonnet 4.5 at 61.4%, up from 42.2% for Sonnet 4 four months earlier (100-step limit, averaged over 4 runs) |
| May 2026 | Anthropic's Opus 4.8 launch notes it "made changes to how we run the OSWorld-Verified evaluation" and restates Opus 4.7 at 82.3%, above the original human baseline |
| June 26, 2026 | **OSWorld 2.0** released |

**OSWorld 2.0** is a different kind of test: 108 long-horizon workflows across seven professional domains, using 31 self-hosted websites plus desktop apps. A skilled human takes a median of about 1.6 hours per task, and 69.6 percent take more than an hour. Tasks are graded against an average of 27.25 checkpoints, giving both a strict **binary completion** score and a **partial** score. At launch, Claude Opus 4.8 with maximum thinking completed 20.6 percent (partial score 54.8 percent) at a 500-step limit, and GPT-5.5 plateaued near 14 percent while using far fewer tokens. Completion fell with task length: above 163 minutes, every model scored zero.

**Known flaws.** The 72.36 percent human baseline is a 2024 measurement, not a ceiling. Lab-reported scores depend on step limits, screenshot resolution and harness choices, and the Opus 4.8 footnote shows that a lab changing how it runs the evaluation can move a previously published number. Checkers can reject valid alternative solutions or accept wrong ones.

### WebArena (2023)

**Measures:** completing realistic tasks on websites. **Format:** self-hosted, fully functional copies of sites from four domains: e-commerce, a social forum, collaborative software development and content management, plus tools such as a map and manuals. **Scoring:** functional correctness: did the task's effect actually happen (the right item ordered, the right post edited)?

At release, the best GPT-4-based agent succeeded on 14.41 percent of tasks, against 78.24 percent for humans. Self-hosting is the key design choice: live websites change daily, so a benchmark built on them cannot be rerun.

**Known flaws.** Some tasks are ambiguous and some checkers are brittle (string matching on answers that could be phrased several ways). The frozen sites grow less representative of the modern web over time. Scores across papers are often not comparable because agents differ in how they observe the page (screenshots, accessibility tree or raw HTML).

### tau-bench and tau2-bench (2024, 2025)

**Measures:** whether a customer-service agent follows company policy while using tools and talking to a user. **Format:** an LLM simulates the customer; the agent gets domain APIs (for example retail orders or airline bookings) and a written policy. **Scoring:** the final database state is compared with the annotated goal state, so politeness does not score points and an unauthorised refund loses them.

Built by Sierra. Its most important contribution is the metric **pass^k**: the probability that the agent succeeds on **all k** independent attempts at the same task. This is the opposite of pass@k in coding, which rewards succeeding once in k tries.

```
pass@k : succeed at least once in k tries   (rises with k: capability)
pass^k : succeed every time in k tries      (falls with k: reliability)
```

At release, gpt-4o succeeded on under 50 percent of tasks, and pass^8 in retail was under 25 percent. **tau2-bench** (June 2025) added a telecom domain with **dual control**, where the simulated user also has tools and the agent must guide them through actions (for example, troubleshooting a phone); scores dropped sharply compared with the single-control domains.

**Known flaws.** The simulated user is itself an LLM and can behave unrealistically or make errors that fail the task. Some tasks have policy ambiguities with more than one defensible outcome. Labs often report pass^1 only, which hides the reliability problem the benchmark was designed to expose.

### GAIA (2023)

**Measures:** general assistant ability: questions that are simple for people but need browsing, file handling, multimodal understanding and several steps of tool use. **Format:** 466 questions with short, unambiguous answers; answers to 300 of them are withheld to power a leaderboard hosted on Hugging Face. **Scoring:** exact match of the final answer.

GAIA inverts the usual approach. Instead of questions that are hard for humans, it asks ones that are "conceptually simple for humans yet challenging for most advanced AIs": human respondents scored 92 percent, GPT-4 with plugins 15 percent. The authors argued that robustness on everyday tasks, not superhuman exam scores, is the milestone to watch.

**Known flaws.** Web-dependent answers can change or disappear. Only the final answer is checked, so lucky guesses and lookups of leaked answers count. The public validation set has been online since 2023, which invites contamination.

### METR's time-horizon measurements (2025 onward)

**Measures:** how long a task (measured in skilled-human time) an AI agent can complete. **Format:** a suite of mostly software and machine-learning engineering tasks, from seconds-long to many hours long, each timed with human experts. **Scoring:** for each model, fit a curve of success rate against human task length; the **50 percent time horizon** is the task length at which the model is predicted to succeed half the time. METR also reports an 80 percent horizon.

```
success
 100% |****
      |     ***
  50% |- - - - -*  <- 50% time horizon (e.g. "4 hours")
      |           ***
   0% |______________****___
        1 min   1 hr   1 day      human task length (log scale)
```

The March 2025 paper found Claude 3.7 Sonnet's 50 percent horizon was about 50 minutes and that the frontier horizon had been **doubling roughly every seven months since 2019**. In January 2026 METR released Time Horizon 1.1, growing the suite from 170 to 228 tasks; it kept the long-run seven-month doubling but estimated faster recent progress: 131 days since 2023 and 89 days since 2024. Under TH1.1, Claude Opus 4.5 measured 320 minutes and GPT-5 214 minutes.

**Where the frontier stands (as of May 2026).** On its May 8, 2026 update, METR added an early Claude Mythos Preview, marked "likely at least 16 hrs", and added the warning that "measurements above 16 hrs are unreliable with our current task suite". METR's May 2026 frontier risk report, covering February to March 2026, put the best publicly available models at about 12 hours (95 percent interval 5 to 61 hours) at the 50 percent threshold and about 1.5 hours at 80 percent. The most capable model labs shared with METR before release had point estimates between 16 and 20 hours at 50 percent and 3 to 4 hours at 80 percent. Several recent models (Claude Opus 4.7, GPT-5.5, Grok 4.3) had no published horizon at that update.

**Known flaws, in METR's own words.** A January 2026 note by lead author Thomas Kwa stresses that:

- **It is not how long the AI works.** It is how much serial human labour it replaces at a 50 percent success rate; AIs usually finish much faster than humans.
- **It is not precise.** Error bars are historically about a factor of 2 in each direction and wider as the suite saturates. Opus 4.5's original estimate of 4 hours 49 minutes had a 95 percent interval of 1 hour 49 minutes to 20 hours 25 minutes.
- **It is domain-specific.** The tasks are mostly software and research engineering, which are well-specified and automatically checkable. Messier real-world work may have much shorter horizons.

The 50 percent threshold is also a choice: a job you would delegate usually needs far more than a coin-flip chance of success, which is why the 80 percent horizon is much shorter.

## Reading agent results in a launch table

The general checklist is in [Knowledge and Reasoning](knowledge-and-reasoning.md#how-to-read-a-model-launchs-benchmark-table). For agent rows, also check:

1. **Step or time budget.** 100 steps and 500 steps are different tests (OSWorld-Verified launch numbers typically use 100 steps; OSWorld 2.0's headline figure uses 500).
2. **Harness.** Is it the benchmark's reference agent, or the lab's own product (a CLI, a browser extension)? Scores are not comparable across harnesses.
3. **Averaged or best run.** Anthropic's OSWorld numbers, for example, are averaged over several runs; a single best run is higher.
4. **pass@1 or pass^k.** For anything customer-facing, reliability over repeated attempts matters more than one success.
5. **Version.** OSWorld versus OSWorld-Verified versus OSWorld 2.0; tau-bench versus tau2-bench.

## What to watch

- **OSWorld 2.0 and long-horizon suites.** The gap between short-task scores (near or above human) and long-workflow completion (about 20 percent in June 2026) is where computer-use progress will show.
- **METR's next task suite.** Until METR adds longer tasks, frontier time horizons can only be reported as "at least 16 hours". A new suite would reset the frontier estimate and test whether the fast post-2024 doubling time holds.
- **Reliability metrics becoming standard.** Whether labs start publishing pass^k and 80 percent horizons alongside the flattering single-success numbers.
- **Safety evaluations built on the same tools.** Agent benchmarks now double as dangerous-capability tests; see [Frontier Safety Frameworks](../policy/frontier-safety-frameworks.md).

## Read next

- [OSWorld](../../papers/techniques/139-osworld/summary.md), [ReAct](../../papers/techniques/21-react/summary.md), [Toolformer](../../papers/techniques/24-toolformer/summary.md), [Voyager](../../papers/techniques/100-voyager/summary.md), [Building Effective Agents](../../papers/essays/115-building-effective-agents/summary.md), [SWE-bench](../../papers/techniques/84-swe-bench/summary.md)
- Explainers: [Math and Code](math-and-code.md), [Agent Protocols](../ecosystem/agent-protocols.md), [Scaling Limits](../open-questions/scaling-limits.md), [Contamination and Saturation](contamination-and-saturation.md)
- Sibling repo: [Tool Use and Function Calling](https://github.com/PatrickWiloak/cloud-data-ai-security-zero-to-hero/blob/main/learn/concepts/tool-use-and-function-calling.md), [MCP Explained](https://github.com/PatrickWiloak/cloud-data-ai-security-zero-to-hero/blob/main/learn/concepts/mcp-explained.md)

## Sources

- Xie et al., "OSWorld: Benchmarking Multimodal Agents for Open-Ended Tasks in Real Computer Environments" (2024): https://arxiv.org/abs/2404.07972
- OSWorld project site (OSWorld-Verified, July 28, 2025; OSWorld 2.0 announcement): https://osworld-v1.xlang.ai/
- OSWorld 2.0 site and paper (released June 26, 2026): https://osworld-v2.xlang.ai/ and https://github.com/xlang-ai/OSWorld-V2
- Anthropic, "Introducing Claude Sonnet 4.5" (OSWorld figures and methodology): https://www.anthropic.com/news/claude-sonnet-4-5
- Anthropic, "Introducing Claude Sonnet 4.6" (February 17, 2026; OSWorld-Verified note): https://www.anthropic.com/news/claude-sonnet-4-6
- Anthropic, "Introducing Claude Opus 4.8" (May 28, 2026; OSWorld-Verified footnote): https://www.anthropic.com/news/claude-opus-4-8
- Zhou et al., "WebArena: A Realistic Web Environment for Building Autonomous Agents" (2023): https://arxiv.org/abs/2307.13854
- Yao et al., "tau-bench: A Benchmark for Tool-Agent-User Interaction in Real-World Domains" (2024): https://arxiv.org/abs/2406.12045
- Barres et al., "tau2-Bench: Evaluating Conversational Agents in a Dual-Control Environment" (2025): https://arxiv.org/abs/2506.07982
- Mialon et al., "GAIA: a benchmark for General AI Assistants" (2023): https://arxiv.org/abs/2311.12983
- Kwa et al., "Measuring AI Ability to Complete Long Software Tasks" (METR, 2025): https://arxiv.org/abs/2503.14499
- METR, "Time Horizon 1.1" (January 29, 2026): https://metr.org/blog/2026-1-29-time-horizon-1-1/
- METR, "Clarifying limitations of time horizon" (Thomas Kwa, January 22, 2026): https://metr.org/notes/2026-01-22-time-horizon-limitations/
- METR, Task-Completion Time Horizons of Frontier AI Models (updated May 8, 2026): https://metr.org/time-horizons/
- METR, "Frontier Risk Report (February to March 2026)" (May 19, 2026): https://metr.org/blog/2026-05-19-frontier-risk-report/
