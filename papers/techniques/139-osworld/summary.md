---
title: "OSWorld: Benchmarking Multimodal Agents for Open-Ended Tasks in Real Computer Environments (OSWorld)"
slug: "139-osworld"
number: 139
category: "techniques"
authors: "Tianbao Xie, Danyang Zhang, Jixuan Chen, Xiaochuan Li, Siheng Zhao, Ruisheng Cao, Toh Jing Hua, Zhoujun Cheng, Dongchan Shin, Fangyu Lei, Yitao Liu, Yiheng Xu, Shuyan Zhou, Silvio Savarese, Caiming Xiong, Victor Zhong, Tao Yu (University of Hong Kong, Carnegie Mellon University, Salesforce Research, University of Waterloo)"
published: "April 2024 (NeurIPS 2024 Datasets and Benchmarks Track)"
year: 2024
url: "https://arxiv.org/abs/2404.07972"
tags: ["evaluation", "benchmarks", "agents"]
---

# OSWorld: Benchmarking Multimodal Agents for Open-Ended Tasks in Real Computer Environments (OSWorld)

**Authors:** Tianbao Xie, Danyang Zhang, Jixuan Chen, Xiaochuan Li, Siheng Zhao, Ruisheng Cao, Toh Jing Hua, Zhoujun Cheng, Dongchan Shin, Fangyu Lei, Yitao Liu, Yiheng Xu, Shuyan Zhou, Silvio Savarese, Caiming Xiong, Victor Zhong, Tao Yu (University of Hong Kong, Carnegie Mellon University, Salesforce Research, University of Waterloo)
**Published:** April 2024 (NeurIPS 2024 Datasets and Benchmarks Track)
**Paper:** [arxiv.org/abs/2404.07972](https://arxiv.org/abs/2404.07972)

---

## Why This Paper Matters

Most of what people do at work happens inside ordinary desktop software: a spreadsheet, an email client, a browser, an image editor, a terminal. An AI that could operate a computer the way a person does - looking at the screen, moving the mouse, typing - could in principle do any of it. By 2024 that was the stated goal of "computer use" agents. What was missing was a fair way to measure progress.

OSWorld supplied one: **369 real tasks on a real Ubuntu desktop** (plus 43 on Windows), running in virtual machines, each with a script that checks the final state of the machine. At launch it was humbling. **Humans completed 72.36% of tasks; the best model managed 12.24%.**

- **Real applications, not simulations**: LibreOffice, Chrome, VS Code, GIMP, Thunderbird, VLC, the file manager and terminal.
- **Execution-based grading**: 134 evaluation functions inspect files, settings and application state - there is no partial credit for plausible-looking actions.
- **Multi-app workflows** and deliberately **infeasible tasks** the agent should recognise and refuse.
- **It became the standard computer-use benchmark.** Anthropic's first computer-use release in October 2024 led with its OSWorld score, and lab announcements since routinely report it.

**The insight:** web-only benchmarks like WebArena covered one application in controlled sites. Real computer use crosses applications, deals with dialogs and file systems, and has many correct ways to finish. Put the agent in an actual operating system, let it do anything a person could, and judge only the outcome.

---

## The Problem: Agent Benchmarks Were Too Narrow

Earlier agent benchmarks tested slices of the problem:
- **Web navigation** in sandboxed sites (WebArena, Mind2Web).
- **Mobile** app control (Android environments).
- **Code or API tools** with no graphical interface.

None captured the everyday reality of switching between a spreadsheet, a browser and a file manager, or of starting from a messy real desktop state. And many evaluated by comparing action sequences to a reference, which penalises valid alternative solutions.

---

## The Core Innovation: A Real Computer as the Test Harness

### The environment
Each task runs in a **virtual machine** snapshot. The agent observes the screen (a screenshot, the accessibility tree, or both) and acts through **pyautogui** - mouse moves, clicks, typing, hotkeys - plus three special actions: `WAIT`, `FAIL` (declare the task impossible) and `DONE`. The default budget in the paper was **15 steps**.

```
  task instruction  --->  agent (VLM / LLM)
                             |   observe: screenshot and/or accessibility tree
                             v
                     pyautogui actions  --->  real Ubuntu VM
                                                   |
                          custom checker inspects final machine state  --->  pass / fail
```

### The tasks
- **369 Ubuntu tasks**: 268 single-application, 101 multi-application workflows.
- **30 infeasible tasks**, where the right answer is to recognise the request cannot be done.
- **84 tasks adapted** from earlier benchmarks; the rest written fresh.
- **302 distinct initial states** and **134 evaluation functions**.
- **43 Windows tasks** as an additional set.
- Built with about **1,800 annotator-hours** by nine of the authors.

### Execution-based evaluation
Each task has a checker that looks at what actually changed: is the spreadsheet cell correct, was the email sent with the attachment, is the VS Code setting saved. Any sequence of actions that produces the right end state passes.

---

## Key Results (at launch, 2024)

| Agent | Observation | Success |
|---|---|---|
| Humans | screen | **72.36%** |
| GPT-4 | accessibility tree | **12.24%** (best model) |
| GPT-4V | screenshot + set-of-marks | 11.77% |
| GPT-4V | screenshot only | 5.26% |

For comparison, humans score about 88% on WebArena, so OSWorld is harder for people too - but the human-model gap was vastly larger.

**Where agents failed:** imprecise mouse coordinates (clicking slightly off target), not understanding GUI conventions, getting lost in multi-step workflows, and being unable to recover from their own mistakes. Higher screen resolution and more history helped only modestly.

---

## What Happened Next

The benchmark tracked a steep climb:
- **October 2024:** Anthropic's Claude 3.5 Sonnet computer-use beta scored **14.9%** screenshot-only (next best 7.8%), and **22.0%** with more steps. See the [Claude 3.5 Sonnet summary](../../language-models/30-claude-3.5-sonnet/summary.md).
- **July 2025:** the maintainers launched **OSWorld-Verified**, fixing task and checker issues and standardising the evaluation, with runs commonly allowed up to 100 steps.
- **September 2025:** Anthropic reported **61.4%** for Claude Sonnet 4.5, up from 42.2% for Sonnet 4.
- **As of 2026-09-30**, the official OSWorld-Verified results sheet listed agent systems above 90% and general-purpose models in the mid-80s at 100 steps - above the original human baseline, though that baseline was measured on the original task set rather than the verified one.
- **June 2026:** OSWorld 2.0 ([arXiv 2606.29537](https://arxiv.org/abs/2606.29537)) introduced 108 long workflows with a median human completion time of about 1.6 hours, on which the best reported model scored around 21% - resetting the frontier the way the original did.

---

## Why This Was Revolutionary

- **It defined "computer use" as a measurable capability.**
- **Execution-based checking in a real OS** made scores meaningful and hard to game by imitation.
- **It showed the bottleneck was grounding** - turning intent into precise GUI actions - which is what model developers then trained on.

---

## Real-World Impact

- **Computer-use products** from Anthropic, OpenAI and Google report OSWorld scores as a headline agent metric.
- **Screen-grounding research** (models trained to locate UI elements precisely) grew directly out of the failure analysis.
- **Benchmark family:** OSWorld sits beside [SWE-bench](../84-swe-bench/summary.md) for code and WebArena for the web in most agent evaluations. See the [agents and computer use benchmarks explainer](../../../explainers/benchmarks/agents-and-computer-use.md).

---

## Key Takeaways for Practitioners

1. **Accessibility trees beat screenshots** for current models when available; pixels alone demand strong visual grounding.
2. **Step budgets change scores a lot.** Compare results at the same budget (15, 50 or 100 steps).
3. **Check which version** - original OSWorld and OSWorld-Verified are not directly comparable.
4. **High scores on short tasks do not imply reliability on long ones**, as OSWorld 2.0 shows.

---

## Limitations & Future Directions

- **Mostly Linux desktop software.** Enterprise applications and web-heavy workflows are under-represented.
- **Short tasks.** Most take a person minutes, not hours - the gap OSWorld 2.0 targets.
- **Checker coverage.** Some tasks admit valid outcomes that a script misjudges; OSWorld-Verified fixed many.
- **Safety is out of scope.** The benchmark measures whether agents can act, not whether they should, which matters once agents touch real accounts and files.

---

## Further Reading

- **Original Paper:** [arxiv.org/abs/2404.07972](https://arxiv.org/abs/2404.07972)
- **Project and leaderboard:** [os-world.github.io](https://os-world.github.io/)
- **In this collection:** [SWE-bench](../84-swe-bench/summary.md), [ReAct](../21-react/summary.md), [Claude 3.5 Sonnet](../../language-models/30-claude-3.5-sonnet/summary.md), [GPT-4V](../../multimodal/23-gpt4v/summary.md)
- **Explainer:** [Agents and computer use benchmarks](../../../explainers/benchmarks/agents-and-computer-use.md)

## Citation

```bibtex
@inproceedings{xie2024osworld,
  title={OSWorld: Benchmarking Multimodal Agents for Open-Ended Tasks in Real Computer Environments},
  author={Xie, Tianbao and Zhang, Danyang and Chen, Jixuan and Li, Xiaochuan and Zhao, Siheng and Cao, Ruisheng and Hua, Toh Jing and Cheng, Zhoujun and Shin, Dongchan and Lei, Fangyu and others},
  booktitle={Advances in Neural Information Processing Systems Datasets and Benchmarks Track},
  year={2024}
}
```

<!-- related:start -->

---

## Related in This Collection

- [ReAct: Synergizing Reasoning and Acting in Language Models](../../techniques/21-react/summary.md)
- [GPT-4V(ision): System Card](../../multimodal/23-gpt4v/summary.md)
- [Claude 3.5 Sonnet: Computer Use and Enhanced Capabilities](../../language-models/30-claude-3.5-sonnet/summary.md)
- [GPT-4 Technical Report](../../language-models/36-gpt4/summary.md)
- [SWE-bench: Can Language Models Resolve Real-World GitHub Issues?](../../techniques/84-swe-bench/summary.md)

<!-- related:end -->
