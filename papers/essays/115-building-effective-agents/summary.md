---
title: "Building Effective Agents (Building Effective Agents)"
slug: "115-building-effective-agents"
number: 115
category: "essays"
authors: "Erik Schluntz, Barry Zhang (Anthropic)"
published: "December 2024 (Anthropic engineering blog)"
year: 2024
url: "https://www.anthropic.com/engineering/building-effective-agents"
tags: ["essay", "agents", "tool-use"]
---

# Building Effective Agents (Building Effective Agents)

**Authors:** Erik Schluntz, Barry Zhang (Anthropic)
**Published:** December 2024 (Anthropic engineering blog)
**Post:** [anthropic.com/engineering/building-effective-agents](https://www.anthropic.com/engineering/building-effective-agents)

---

## Why This Matters

"Building Effective Agents" is the most widely cited practical guide to building LLM agents. It appeared on December 19, 2024, just as "agents" had become the industry's favourite buzzword and developers were piling on heavy frameworks. Its message was deflationary and useful: most successful teams "weren't using complex frameworks or specialized libraries," but were building with "simple, composable patterns."

- **It gave the field a shared vocabulary.** Prompt chaining, routing, parallelization, orchestrator-workers and evaluator-optimizer are now the standard names for these patterns.
- **It drew a clear line between workflows and agents,** which settles a lot of confused debate about what an "agent" even is.
- **It argued for restraint.** Start with the simplest thing that works and add autonomy only when it pays for itself.
- **It made tool design a first-class problem.** Its "agent-computer interface" idea, and the report that the team spent more time on tools than on the main prompt, changed how people build agents.

---

## The Problem It Addresses

By late 2024, [ReAct](../../techniques/21-react/summary.md)-style loops, tool use and multi-agent frameworks were everywhere. Teams building agents faced a confusing landscape: frameworks that hid the actual prompts, "agents" that were really fixed pipelines, and autonomous systems that cost a lot, went off the rails and were hard to debug. Anthropic, drawing on work with "dozens of teams building large language model (LLM) agents across industries," set out to describe what actually worked.

---

## The Argument

### Workflows vs agents

The post starts with a definition that does a lot of work:

```
AGENTIC SYSTEMS
 |
 +-- WORKFLOWS: "LLMs and tools are orchestrated through
 |              predefined code paths"
 |              -> the developer decides the steps
 |
 +-- AGENTS:    "LLMs dynamically direct their own processes
                and tool usage, maintaining control over how
                they accomplish tasks"
                -> the model decides the steps
```

Workflows give predictability and consistency for well-defined tasks. Agents give flexibility for open-ended problems "where it's difficult or impossible to predict the required number of steps". Both trade latency and cost for better task performance, so the first question is whether you need either. The guidance: "Start with simple prompts, optimize them with comprehensive evaluation, and add multi-step agentic systems only when simpler solutions fall short."

### On frameworks

The original post named several frameworks (LangGraph, Amazon Bedrock's AI Agent framework, Rivet and Vellum; the live page has since been updated with newer ones). It grants that they make it easy to get started, but warns that they "often create extra layers of abstraction that can obscure the underlying prompts and responses, making them harder to debug." The advice is to start by using LLM APIs directly, since "many patterns can be implemented in a few lines of code", and to understand the underlying code if you do use a framework.

### The building block: the augmented LLM

Every pattern is built from one unit: an LLM augmented with **retrieval, tools and memory**. The post points to the [Model Context Protocol](../../techniques/59-model-context-protocol/summary.md), released a month earlier, as one way to plug an LLM into a growing ecosystem of third-party tools.

```
          +-----------+
 query -> |    LLM    | -> response
          +-----------+
           |    |    |
     retrieval tools memory
```

---

## Key Components Explained: The Five Workflow Patterns

### 1. Prompt chaining
**What it does:** Splits a task into a fixed sequence of LLM calls, each working on the output of the previous one.
**How it works:** You can add programmatic checks ("gates") between steps to make sure things are on track. It trades latency for accuracy by making each call's job easier.
**When to use it:** When a task "can be easily and cleanly decomposed into fixed subtasks."
**Examples from the post:** Writing marketing copy, then translating it. Writing an outline, checking it against criteria, then writing the document.

```
input -> [LLM 1] -> gate -> [LLM 2] -> [LLM 3] -> output
                     |
                    fail -> exit
```

### 2. Routing
**What it does:** Classifies an input and sends it to a specialised follow-up prompt, tool or model.
**How it works:** Keeps concerns separate, so that optimising for one kind of input does not hurt another.
**Examples:** Sending customer-service queries (general questions, refund requests, technical support) to different processes. Sending easy questions to a small, cheap model and hard ones to a more capable model.

### 3. Parallelization
**What it does:** Runs LLM calls at the same time and combines their outputs programmatically. It comes in two forms.
- **Sectioning:** break the task into independent subtasks and run them in parallel. Examples: one model answers the user while another screens for inappropriate content (a guardrail), or each call evaluates a different aspect of performance in an automated eval.
- **Voting:** run the same task several times and aggregate. Examples: several prompts review code for vulnerabilities, or content is judged inappropriate using different vote thresholds to balance false positives and negatives.
**When to use it:** When the subtasks can run in parallel for speed, or when multiple attempts or perspectives give more confident results. (Voting is a cousin of [self-consistency](../../techniques/77-self-consistency/summary.md).)

### 4. Orchestrator-workers
**What it does:** A central LLM breaks the task down on the fly, hands pieces to worker LLMs, and synthesises their results.
**How it differs from parallelization:** The subtasks are not fixed in advance. The orchestrator decides them based on the specific input.
**Examples:** Coding products that make complex changes to several files. Search tasks that gather and analyse information from many sources.

```
               +--> [worker A] --+
input -> [orchestrator] -> [worker B] -> [synthesizer] -> output
               +--> [worker C] --+
       (subtasks chosen at runtime)
```

### 5. Evaluator-optimizer
**What it does:** One LLM produces a response, a second evaluates it and gives feedback, and the loop repeats.
**When to use it:** When there are clear evaluation criteria and iterative refinement gives measurable gains, much as a human writer improves with an editor's feedback. It is the workflow version of the self-critique loop in [Reflexion](../../techniques/78-reflexion/summary.md) and [Self-Refine](../../techniques/99-self-refine/summary.md).
**Examples:** Literary translation with nuances a first pass might miss. Complex search that needs several rounds, where the evaluator decides whether more searching is needed.

### Agents

Agents, the post says, are "typically just LLMs using tools based on environmental feedback in a loop." They work from a human command or discussion, plan and act on their own, get "ground truth" from the environment at each step (tool results, code execution), may pause for human feedback, and stop on completion or at a limit such as a maximum number of iterations.

```
human task --> [LLM] --action--> environment
                 ^                    |
                 +------feedback------+
          (repeat until done or a stop condition)
```

Their autonomy makes them good for scaling tasks in trusted environments, but it "means higher costs, and the potential for compounding errors." The post recommends extensive testing in sandboxed environments with guardrails. Examples given: a coding agent resolving [SWE-bench](../../techniques/84-swe-bench/summary.md) tasks, and Anthropic's computer-use reference implementation.

The patterns are building blocks, not prescriptions. "The key to success, as with any LLM features, is measuring performance and iterating on implementations."

---

## Key Claims

1. **Simple, composable patterns beat heavy frameworks** for most production use.
2. **Workflows and agents are different things**: predefined code paths versus model-directed control.
3. **Use the least autonomy that works.** Agents cost more and compound errors, so justify them.
4. **Three core principles:** keep the design simple, make the agent's planning steps transparent, and carefully craft the agent-computer interface (ACI) through tool documentation and testing.
5. **Invest in tools as much as in UIs.** "Think about how much effort goes into human-computer interfaces (HCI), and plan to invest just as much effort in creating good agent-computer interfaces (ACI)."
6. **Coding and customer support are natural fits** because success is measurable: tests pass, or the issue is resolved.

---

## The Tool-Design Appendix

The second appendix, "Prompt engineering your tools", has been as influential as the patterns. Its advice:

- Give the model enough tokens to "think" before it writes itself into a corner.
- Keep formats close to what the model has seen in natural text (for example, code in markdown rather than escaped inside JSON).
- Avoid formatting overhead, such as keeping accurate line counts for diffs.
- Write tool descriptions as you would a docstring for a junior developer, with examples and edge cases.
- **Poka-yoke your tools** (a Japanese manufacturing term for "mistake-proofing"): change arguments so mistakes are harder to make.

Its best-known anecdote comes from the team's SWE-bench agent: "we actually spent more time optimizing our tools than the overall prompt." The model made mistakes with relative file paths after changing directories, so they changed the tool to always require absolute paths, and the model then "used this method flawlessly."

---

## How It Has Aged (as of September 2026)

Well. The vocabulary stuck, and the advice to start simple remains the default recommendation.

- **The pattern names became standard.** "Orchestrator-workers", "evaluator-optimizer" and "prompt chaining" are used across vendors, frameworks and courses.
- **Anthropic built on it directly.** "How we built our multi-agent research system" (June 2025) describes an orchestrator-worker design in production. "Writing effective tools for agents" (September 2025) expanded the ACI appendix. "Effective context engineering for AI agents" (September 2025) developed the idea that what goes into the context is the main design lever.
- **MCP took off.** The post's pointer to the [Model Context Protocol](../../techniques/59-model-context-protocol/summary.md) turned out to be early; MCP became a widely adopted standard for connecting tools to models.
- **Coding agents validated the "measurable domains first" point.** Coding became the flagship agent use case, driven by test-verified feedback, as the post argued.
- **Frameworks did not disappear.** Anthropic itself later released an agent SDK (the Claude Agent SDK), and the live page now lists it among the frameworks. The advice shifted from "avoid frameworks" towards "use thin ones and understand what they do", which is consistent with the original caveat.
- **Some parts are dated.** The model names in the examples have been updated on the live page, and newer models handle longer autonomous runs than the post assumed. That shifts the workflow-versus-agent trade-off towards agents for more tasks, but it does not change the method.

---

## Criticisms

- **It is a practitioner's taxonomy, not a study.** The patterns come from experience, not controlled comparisons, and the post gives no benchmarks showing when one pattern beats another.
- **The categories overlap.** Orchestrator-workers with a loop is close to an agent, and evaluator-optimizer is a special case of chaining. The boundaries are useful but fuzzy.
- **Vendor perspective.** It comes from a model provider whose models power these systems. Its scepticism of frameworks is well argued, but it also favours direct API use.
- **Light on operations.** It says little about evaluation infrastructure, observability, cost control, security (prompt injection through tools) or multi-agent failure modes, which later writing covered in more depth.

---

## Real-World Impact

- **The default reference for teams building agents,** widely linked from framework docs, courses and conference talks.
- **Shaped agent products.** Coding agents, research agents and customer-support agents commonly use the orchestrator-worker and evaluator-optimizer structures it named.
- **Moved attention to tools and context.** The ACI idea helped make tool design and context engineering recognised specialities.

---

## Key Takeaways for Practitioners

1. **Try a single well-prompted call with retrieval first.** Many "agent" problems are solved there.
2. **Prefer workflows when the steps are knowable.** They are cheaper, faster and easier to test.
3. **Use agents when the path cannot be predicted and success can be checked.** Tests, resolution status or a clear evaluator make autonomy safe to iterate on.
4. **Spend real time on tools.** Clear names, good descriptions, mistake-proof arguments, and testing how the model actually uses them.
5. **Keep the loop observable.** Show the plan, log every step, and set stopping conditions.
6. **Measure, then add complexity.** Every added LLM call needs a measured improvement to justify it.

---

## Limitations & Future Directions

The post predates the big improvements in long-horizon reasoning models and long-context agents in 2025-2026. It does not cover multi-agent coordination at scale, memory across sessions, or security threats like prompt injection through tool outputs. Later Anthropic engineering posts and the wider agent literature filled some of these gaps. For the research roots of the patterns, see [ReAct](../../techniques/21-react/summary.md) (the reason-act loop), [Toolformer](../../techniques/24-toolformer/summary.md) (learned tool use), [Reflexion](../../techniques/78-reflexion/summary.md) (self-critique loops), and [Generative Agents](../../techniques/58-generative-agents/summary.md) (memory and planning).

---

## Further Reading

- **Original post:** [anthropic.com/engineering/building-effective-agents](https://www.anthropic.com/engineering/building-effective-agents)
- **Cookbook implementations:** [platform.claude.com/cookbook/patterns-agents-basic-workflows](https://platform.claude.com/cookbook/patterns-agents-basic-workflows)
- **Multi-agent research system (June 2025):** [anthropic.com/engineering/multi-agent-research-system](https://www.anthropic.com/engineering/multi-agent-research-system)
- **Writing effective tools for agents (September 2025):** [anthropic.com/engineering/writing-tools-for-agents](https://www.anthropic.com/engineering/writing-tools-for-agents)
- **Effective context engineering (September 2025):** [anthropic.com/engineering/effective-context-engineering-for-ai-agents](https://www.anthropic.com/engineering/effective-context-engineering-for-ai-agents)
- **In this collection:** [ReAct](../../techniques/21-react/summary.md), [Model Context Protocol](../../techniques/59-model-context-protocol/summary.md), [Reflexion](../../techniques/78-reflexion/summary.md), [SWE-bench](../../techniques/84-swe-bench/summary.md), [Claude 3.5 Sonnet](../../language-models/30-claude-3.5-sonnet/summary.md)

## Citation

```bibtex
@misc{schluntz2024agents,
  title={Building Effective Agents},
  author={Schluntz, Erik and Zhang, Barry},
  year={2024},
  month={December},
  howpublished={Anthropic Engineering Blog, \url{https://www.anthropic.com/engineering/building-effective-agents}}
}
```

<!-- related:start -->

---

## Related in This Collection

- [ReAct: Synergizing Reasoning and Acting in Language Models](../../techniques/21-react/summary.md)
- [Toolformer: Language Models Can Teach Themselves to Use Tools](../../techniques/24-toolformer/summary.md)
- [Claude 3.5 Sonnet: Computer Use and Enhanced Capabilities](../../language-models/30-claude-3.5-sonnet/summary.md)
- [Generative Agents: Interactive Simulacra of Human Behavior](../../techniques/58-generative-agents/summary.md)
- [Model Context Protocol (MCP): An Open Standard for AI Tool Integration](../../techniques/59-model-context-protocol/summary.md)
- [Self-Consistency Improves Chain of Thought Reasoning in Language Models](../../techniques/77-self-consistency/summary.md)
- [Reflexion: Language Agents with Verbal Reinforcement Learning](../../techniques/78-reflexion/summary.md)
- [SWE-bench: Can Language Models Resolve Real-World GitHub Issues?](../../techniques/84-swe-bench/summary.md)

<!-- related:end -->
