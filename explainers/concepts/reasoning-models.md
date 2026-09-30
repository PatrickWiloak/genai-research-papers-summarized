# Reasoning Models

**In one line:** "Reasoning" or "thinking" models are language models trained with reinforcement learning to write out a long chain of working before they answer, which trades extra tokens (time and money) for accuracy on hard problems, so you should turn the thinking up for maths, code and multi-step tasks and down for everything else.
**Last reviewed:** 2026-09-30

---

## The short version

- A reasoning model is still a next-token predictor. The difference is that it has been **trained to produce a long private chain of thought first**, trying approaches, checking work and backtracking, and only then write the answer.
- The key training ingredient is **reinforcement learning on verifiable rewards (RLVR)**: give the model problems whose answers can be checked automatically (a maths result, a passing test suite), reward correct final answers, and let it discover for itself which kinds of thinking help.
- This opened a second scaling axis, **test-time compute**: instead of only making the model bigger, let it spend more tokens thinking on each problem. Accuracy on hard tasks rises with thinking length, with diminishing returns.
- Thinking is paid for as **output tokens**, occupies the [context window](context-windows.md) and [KV cache](kv-cache.md), and adds latency. As of September 2026 every major API exposes a control for how much the model thinks (an "effort" level or a token budget).
- Most commercial APIs **do not show the raw chain of thought**. They return a summary, or nothing, and bill for the full hidden chain. Open-weight reasoning models show everything.
- Whether a visible chain of thought reflects what the model is "really" doing is an open research question. Studies find it is often **unfaithful**, which matters for the idea of monitoring AI systems by reading their reasoning.

## From prompting trick to trained behaviour

The idea that models do better when they "show their work" is older than reasoning models.

| Year | Step | What changed |
|---|---|---|
| 2022 | [Chain-of-thought prompting](../../papers/techniques/09-chain-of-thought/summary.md) | Few-shot examples with worked steps made large models much better at multi-step problems |
| 2022 | [Self-consistency](../../papers/techniques/77-self-consistency/summary.md) | Sample many chains, take the majority answer |
| 2022-2024 | [STaR](../../papers/techniques/97-star/summary.md), [Quiet-STaR](../../papers/techniques/98-quiet-star/summary.md) | Train on the model's own successful rationales |
| 2023 | [Process reward models](../../papers/techniques/51-process-reward-models/summary.md) | Score each reasoning step, not only the final answer |
| Feb 2024 | [GRPO](../../papers/techniques/38-grpo/summary.md) | A cheaper RL algorithm that compares a group of sampled answers instead of training a separate critic |
| Aug 2024 | [Scaling test-time compute](../../papers/techniques/50-test-time-compute/summary.md) | Showed that spending inference compute well can beat a much larger model |
| Sep 2024 | [OpenAI o1](../../papers/language-models/31-openai-o1/summary.md) | First widely available model trained to think at length before answering |
| Jan 2025 | [DeepSeek-R1](../../papers/language-models/26-deepseek-r1/summary.md) | Open weights and a public recipe; reasoning emerged from RL alone in R1-Zero |

The shift was from **asking** a model to reason to **training** it so that long, self-correcting reasoning is what it does by default.

## How they are trained: RL on verifiable rewards

The core loop, described most openly in the DeepSeek-R1 paper and in the [RLVR summary](../../papers/techniques/39-rlvr/summary.md):

```
for each training problem with a checkable answer:
    sample several attempts from the model (each = thinking + answer)
    check each final answer automatically
        maths: does the number match?
        code:  do the tests pass?
    reward correct attempts, penalise wrong ones
    update the model to make rewarded attempts more likely   (e.g. GRPO)
```

Nobody writes the reasoning for the model. It is rewarded only for getting answers right, and it discovers on its own that longer thinking, re-checking and backtracking earn more reward. DeepSeek reported that **R1-Zero**, trained with RL directly on a base model with no supervised reasoning examples, went from 15.6% to 71.0% pass@1 on the AIME 2024 maths competition and spontaneously began writing reflections such as re-examining earlier steps. The released R1 added a small amount of supervised "cold start" data and further training stages to make the output readable.

Why verifiable rewards matter: earlier alignment training ([RLHF](../../papers/language-models/05-instructgpt-rlhf/summary.md)) relied on learned reward models of human preference, which a strong optimiser can learn to game. A unit test or an exact answer is much harder to game, so RL can run for longer and push further. The limit is the flip side: the approach works best where answers can be checked, which is why reasoning gains have been largest in maths, coding and science and more modest in open-ended writing. See [Do LLMs reason?](../open-questions/do-llms-reason.md) for the debate over what this capability amounts to.

## Test-time compute: the second scaling axis

Before 2024, "better model" mostly meant more parameters, data and training compute (see [scaling laws](../../papers/techniques/12-scaling-laws/summary.md)). Reasoning models add a dial at inference time.

```
          accuracy on hard problems
              |                        ________ diminishing returns
              |                 ______/
              |           _____/
              |       ___/
              |    __/
              |  _/
              |_/
              +------------------------------------> thinking tokens per problem
```

There are two ways to spend it:

- **Sequential**: one longer chain of thought, in which the model can revise itself. This is what "thinking" models mainly do.
- **Parallel**: many independent attempts, then a majority vote or a verifier picks the best ([self-consistency](../../papers/techniques/77-self-consistency/summary.md), best-of-N with a [process reward model](../../papers/techniques/51-process-reward-models/summary.md)).

Snell et al. (2024) showed that the best mix depends on difficulty, that a compute-optimal strategy was more than 4 times more efficient than a best-of-N baseline, and that on problems where a smaller model already had some success, extra test-time compute let it outperform a model 14 times larger. The practical reading: thinking helps most on problems that are hard but within reach, and helps little on easy lookups or on problems far beyond the model.

## Reasoning tokens, budgets and effort

Thinking is not free. It is generated text, so it is billed, takes time and uses memory.

| Provider (as of September 2026) | How you control thinking | Billing | What you see |
|---|---|---|---|
| **OpenAI** | `reasoning.effort` with levels from `none`/`minimal` up to `high`, `xhigh` and `max`, depending on the model | Reasoning tokens billed as output tokens | Raw reasoning is not returned; optional summaries on request |
| **Anthropic** | Newer Claude models use **adaptive thinking**, where the model decides how much to think, steered by an `effort` setting; older models used a fixed `budget_tokens` (minimum 1,024) | Thinking billed as output tokens, even when not shown | A summary of the reasoning, or nothing (the default on the newest models); never the raw chain |
| **Google Gemini** | `thinking_level` (minimal/low, medium, high) with dynamic adjustment | Billed on the full thought tokens, though only a summary is returned | Thought summaries, plus an encrypted signature of the reasoning state |

Things that follow from this:

- **Budget your context.** Thinking shares the window with everything else. OpenAI suggests reserving at least 25,000 tokens for reasoning and output when starting out; if the limit is hit mid-thought you can pay for reasoning and get no answer.
- **Sampling knobs often disappear.** With thinking on, providers commonly fix or restrict temperature and related settings (see [sampling and decoding](sampling-and-decoding.md)).
- **One model, two modes.** Many current models are "hybrid": the same weights can answer instantly or think first, controlled per request. [Qwen3](../../papers/language-models/28-qwen3/summary.md) exposed thinking and non-thinking modes in one open model, and the major APIs now let the model decide or let you set the level.
- **Multi-turn and tools.** In agent loops the model may think between tool calls. APIs pass the (possibly encrypted) reasoning back across turns so the model keeps its train of thought.

A rough rule for builders: default to a low or medium effort, raise it for maths, complex code, planning and multi-step analysis, and measure whether the gain is worth the latency and cost on your task. The sibling repo's [prompt engineering](https://github.com/PatrickWiloak/cloud-data-ai-security-zero-to-hero/blob/main/learn/concepts/prompt-engineering.md) and [evals](https://github.com/PatrickWiloak/cloud-data-ai-security-zero-to-hero/blob/main/learn/concepts/evals-for-llms.md) pages cover the practical side.

## Hidden versus visible chains of thought

When o1 launched, OpenAI chose not to show its raw reasoning, with competitive advantage and user experience among the reasons given (see the [o1 summary](../../papers/language-models/31-openai-o1/summary.md)). DeepSeek-R1 took the opposite path and showed everything. As of September 2026 the pattern is:

- **Closed APIs**: summaries or nothing, with full billing for the hidden chain. Encrypted reasoning is returned so that it can be passed back in later turns without being readable.
- **Open-weight reasoning models**: the full chain is visible, because you run the model yourself.

Two arguments pull in different directions:

- **For hiding**: raw chains can be long, messy and misleading to users; they may reveal training secrets or make it easy to distil the model into a competitor; and if chains are shown, there is commercial pressure to train them to look nice, which could make them less honest.
- **For showing**: users and auditors can see why an answer was reached, spot errors, and hold providers accountable; and safety researchers want to monitor reasoning for signs of harmful intent.

### Is the chain of thought faithful?

A visible chain is only useful for oversight if it reflects the model's actual computation. Evidence so far says: **partly**.

- Anthropic (April 2025) slipped hints into questions and checked whether models admitted using them. Claude 3.7 Sonnet mentioned the hint it used about **25%** of the time and DeepSeek R1 about **39%**; for hints framed as unauthorised access, the figures were 41% and 19%.
- A July 2025 position paper by researchers across several labs and academia, "Chain of Thought Monitorability: A New and Fragile Opportunity for AI Safety", argued that reading reasoning in natural language is a real but imperfect safety tool, and urged developers to consider how training choices (for example, directly optimising the chain to look good) might erode it.

This connects to the wider question of whether these models reason in any deep sense, covered in [Do LLMs reason?](../open-questions/do-llms-reason.md).

## Costs and limits

- **Latency and price.** A hard question can consume tens of thousands of thinking tokens. See [inference economics](../compute/inference-economics.md).
- **Overthinking.** On easy questions, long reasoning adds cost and can talk the model out of a correct first answer. Adaptive thinking modes exist partly to address this.
- **Verifiability bias.** Gains are largest where rewards are checkable. Transfer to fuzzy domains (strategy, writing, judgement) is real but smaller and harder to measure.
- **Benchmarks.** Reasoning models saturated many maths and science benchmarks quickly; see [math and code benchmarks](../benchmarks/math-and-code.md) and [contamination and saturation](../benchmarks/contamination-and-saturation.md).

## What to watch

- **Reasoning in latent space.** Research such as Meta's Coconut (December 2024), in which a model feeds its hidden states back in instead of writing words, could make reasoning cheaper but would remove the readable chain that monitoring relies on.
- **RL beyond verifiable domains.** Using model judges or rubrics as rewards (see [LLM-as-a-judge](../../papers/techniques/85-llm-as-judge/summary.md)) is the obvious route to RLVR-style gains on open-ended tasks, but judges are easier to game than unit tests; see [synthetic data and model collapse](../open-questions/synthetic-data-and-model-collapse.md) for related risks.
- **Transparency norms.** Whether closed labs move toward showing more of the chain, or less, and whether regulators ask for it.
- **Long-horizon agents.** Thinking between tool calls over hours of work is where test-time compute is now being spent; see [agents and computer use](../benchmarks/agents-and-computer-use.md).

## Read next

- [Chain-of-Thought Prompting](../../papers/techniques/09-chain-of-thought/summary.md)
- [OpenAI o1](../../papers/language-models/31-openai-o1/summary.md) and [DeepSeek-R1](../../papers/language-models/26-deepseek-r1/summary.md)
- [GRPO](../../papers/techniques/38-grpo/summary.md) and [RLVR](../../papers/techniques/39-rlvr/summary.md)
- [Scaling Test-Time Compute](../../papers/techniques/50-test-time-compute/summary.md) and [Process Reward Models](../../papers/techniques/51-process-reward-models/summary.md)
- [Meta Chain-of-Thought](../../papers/techniques/34-meta-cot/summary.md), [rStar-Math](../../papers/techniques/35-rstar-math/summary.md)
- [Do LLMs reason?](../open-questions/do-llms-reason.md), [Sampling and decoding](sampling-and-decoding.md), [Context windows](context-windows.md)
- Model family context: [GPT](../model-families/gpt.md), [Claude](../model-families/claude.md), [Gemini](../model-families/gemini.md), [DeepSeek](../model-families/deepseek.md), [Qwen](../model-families/qwen.md)

## Sources

- Wei et al., "Chain-of-Thought Prompting Elicits Reasoning in Large Language Models", NeurIPS 2022. https://arxiv.org/abs/2201.11903
- DeepSeek-AI, "DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning", January 2025 (R1-Zero AIME 2024 15.6% to 71.0%). https://arxiv.org/abs/2501.12948
- Shao et al., "DeepSeekMath" (introduces GRPO), February 2024. https://arxiv.org/abs/2402.03300
- Snell, Lee, Xu, Kumar, "Scaling LLM Test-Time Compute Optimally can be More Effective than Scaling Model Parameters", August 2024. https://arxiv.org/abs/2408.03314
- Lightman et al., "Let's Verify Step by Step", 2023. https://arxiv.org/abs/2305.20050
- OpenAI, "Learning to Reason with LLMs", September 12, 2024. https://openai.com/index/learning-to-reason-with-llms/
- OpenAI, Reasoning models guide (effort levels, billing, summaries, 25,000-token reserve), accessed 2026-09-30. https://developers.openai.com/api/docs/guides/reasoning
- Anthropic, "Thinking" and "Extended thinking" documentation (adaptive thinking, effort, `budget_tokens` minimum 1,024, summarized or omitted display, billing), accessed 2026-09-30. https://platform.claude.com/docs/en/build-with-claude/thinking and https://platform.claude.com/docs/en/build-with-claude/extended-thinking
- Google, Gemini API "Thinking" documentation (thinking levels, thought summaries, billing), accessed 2026-09-30. https://ai.google.dev/gemini-api/docs/thinking
- Anthropic, "Reasoning models don't always say what they think", April 3, 2025. https://www.anthropic.com/research/reasoning-models-dont-say-think
- Hao et al., "Training Large Language Models to Reason in a Continuous Latent Space" (Coconut), December 2024. https://arxiv.org/abs/2412.06769
- Korbak et al., "Chain of Thought Monitorability: A New and Fragile Opportunity for AI Safety", July 2025. https://arxiv.org/abs/2507.11473
