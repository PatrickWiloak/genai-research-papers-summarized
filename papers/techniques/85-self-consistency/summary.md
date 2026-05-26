# Self-Consistency Improves Chain of Thought Reasoning in Language Models

**Authors:** Xuezhi Wang, Jason Wei, Dale Schuurmans, Quoc Le, Ed Chi, Sharan Narang, Aakanksha Chowdhery, Denny Zhou (Google Research, Brain Team)
**Published:** March 2022 (ICLR 2023)
**Paper:** [arxiv.org/abs/2203.11171](https://arxiv.org/abs/2203.11171)

---

## Why This Paper Matters

Self-Consistency is one of the cheapest, simplest, and most broadly applicable techniques in the entire LLM playbook. The recipe is: sample many chain-of-thought solutions for the same problem at non-zero temperature, then take the answer that appears most often. That's it. Applied to math word problems, this trick lifted PaLM-540B's GSM8K accuracy from 56.5% to 74.4% — a roughly 18-point absolute gain — with no model changes whatsoever.

The paper is foundational because it established the **"majority-over-samples"** template for test-time compute. Every modern reasoning approach that talks about "scaling inference" — best-of-N sampling, verifier-guided search, process reward models, o1-style long-thought generation — descends conceptually from Self-Consistency. It also reframed the relationship between sampling and reasoning: instead of treating each sample as a fragile guess, treat the distribution of samples as a probabilistic vote on the correct answer.

---

## The Problem

Chain-of-Thought (CoT) prompting had recently shown that large LLMs solve math and reasoning problems much better when prompted to "think step by step" before answering. But CoT used greedy decoding: one reasoning trace, one answer. This had two issues:

1. **Brittleness.** A single token sampled wrong early in the trace could derail the entire answer. Greedy decoding committed to that path with no recourse.
2. **Underuse of the model's distribution.** A well-calibrated language model assigns probability mass to many reasoning paths. Greedy decoding throws all of that information away by following only the single most likely next token.

It was already known that sampling multiple completions could help — for example, with self-verification or rerankers — but those approaches required either a trained verifier or expensive matching procedures.

The Self-Consistency question: what if the model's own probability distribution is *already* expressive enough to vote on the right answer, with no extra machinery?

---

## The Core Innovation

The key insight is a simple observation about reasoning tasks: **there are many valid reasoning paths to a correct answer, but they tend to converge on the same final number, while incorrect reasoning paths diverge in many different directions.**

If you sample 40 chains of thought for a math problem, you might see:

- 22 traces arrive at the answer 18
- 8 traces arrive at 12
- 5 traces arrive at 24
- 5 traces arrive at various other wrong answers

The right answer (18) is the modal answer, even though no single sample dominates probabilistically. Marginalizing over reasoning paths — summing the model's confidence in each *answer* across all the paths that produced it — gives a much sharper estimate of the correct answer than any single greedy decode.

Formally, instead of computing:

```
answer = argmax_y P(y | x)
```

Self-Consistency computes:

```
answer = argmax_y  Sum_z  P(y, z | x)
```

where `z` is the latent reasoning path. The sum is approximated by sampling N reasoning paths and counting how often each final answer appears.

```
                       +--> "The answer is 18"  
                       +--> "The answer is 18"  
   Prompt --LLM-->     +--> "The answer is 12"     --> majority vote --> 18
                       +--> "The answer is 18"  
                       +--> "The answer is 24"  
                       (N samples, T=0.7)
```

---

## How It Works

The full procedure has three steps:

1. **Prompt with Chain-of-Thought.** Use a standard few-shot CoT prompt that elicits step-by-step reasoning ending in a final answer.
2. **Sample N traces with temperature.** Run the model N times with `temperature` around 0.5-0.7 and/or top-k or top-p sampling. Each sample produces an independent reasoning trace.
3. **Extract and majority-vote.** Parse out the final answer from each trace. Return whichever answer appears most often.

Pseudocode:

```python
def self_consistent_answer(prompt, model, N=40, temperature=0.7):
    samples = [model.generate(prompt, temperature=temperature)
               for _ in range(N)]
    answers = [extract_final_answer(s) for s in samples]
    return Counter(answers).most_common(1)[0][0]
```

### Why It Works Where Greedy Fails

Consider GSM8K problem: "Janet's ducks lay 16 eggs per day. She eats 3 for breakfast and bakes muffins with 4. She sells the rest at the farmers' market for $2 per egg. How much does she make every day?"

- A greedy decode might write "16 - 3 - 4 = 9, so 9 * 2 = $18" (correct).
- Another might write "She has 16 eggs. After eating 3, she has 13. She bakes with 4, leaving 9. But she only sold 8 because one broke" (incorrect — invented detail).
- Yet another might forget to multiply by $2 and answer "$9".

Over 40 samples, the *correct* arithmetic path is the most common stable trajectory because the model has learned the underlying procedure well — even though any individual sample can derail. Wrong answers spread across many different failure modes; right answers concentrate.

### Cost vs. Benefit

The cost is exactly N times one generation. The benefit, as the paper shows, follows a clear curve: most of the improvement comes from the first 10-20 samples, with diminishing returns beyond 40.

---

## Key Results

The paper evaluated Self-Consistency on four LLM families (UL2-20B, GPT-3, LaMDA-137B, PaLM-540B) across arithmetic, commonsense, and symbolic reasoning benchmarks.

Headline gains on PaLM-540B:

| Benchmark | CoT (greedy) | CoT + Self-Consistency | Gain |
|-----------|--------------|-----------------------|------|
| GSM8K (math word problems) | 56.5% | 74.4% | +17.9 |
| SVAMP (math) | 79.0% | 86.6% | +7.6 |
| AQuA (algebra) | 35.8% | 48.3% | +12.5 |
| StrategyQA (commonsense) | 75.3% | 81.6% | +6.3 |
| ARC-Challenge (science) | 85.2% | 88.7% | +3.5 |

Additional findings:

- The technique helps across all model sizes, but gains are larger for stronger models because their sampled traces are more often *correct enough* for majority vote to work.
- Robustness to prompt variations improves: with greedy decoding, small CoT prompt rewrites can swing accuracy by several points; with Self-Consistency, the noise is averaged out.
- It also works for tasks with non-numeric answers (e.g., multiple choice), as long as answers can be canonicalized for vote-counting.
- It pairs naturally with any prompting method — vanilla, CoT, ReAct, Tree of Thoughts.

---

## Impact and Legacy

Self-Consistency is the conceptual ancestor of essentially every "scale test-time compute" technique in use today:

- **Best-of-N sampling** — sample N answers, pick the one a verifier scores highest. Self-Consistency is the special case where "verifier" = "majority vote."
- **Process reward models** — instead of voting on final answers, score each *step* of each trace and aggregate. A learned-judge upgrade to Self-Consistency.
- **Tree of Thoughts** — explicitly searches a tree of reasoning steps rather than independently sampling whole traces. Same underlying philosophy of exploring multiple paths.
- **o1 and DeepSeek-R1** — large reasoning models internalize "try multiple approaches, see which checks out" inside a single long chain of thought, but the operational principle is identical.
- **Test-time compute scaling** — papers like "Scaling LLM Test-Time Compute Optimally" build directly on the Self-Consistency framing of trading inference cost for quality.
- **rStar-Math and RLVR** — training-time methods that rely on sampling many traces and scoring them; Self-Consistency provides both the sampling regime and the scoring baseline.

The paper also normalized a now-standard evaluation practice: when reporting LLM accuracy on hard reasoning benchmarks, papers often report both greedy and self-consistency (or majority@N) numbers.

---

## Connections to Other Papers

- **Chain-of-Thought (#9)** — direct parent paper. CoT taught models to reason; Self-Consistency taught us to sample many CoT reasonings and vote.
- **Tree of Thoughts (#25)** — generalizes Self-Consistency from independent samples to structured search over a tree of partial thoughts.
- **Process Reward Models (#51)** — replaces majority vote with a learned step-level scorer.
- **Test-Time Compute (#50)** — formalizes the trade-off Self-Consistency first demonstrated.
- **Self-Refine (#84)** — a complementary inference-time technique. Self-Refine improves a single trajectory iteratively; Self-Consistency aggregates over many parallel trajectories.
- **rStar-Math (#35)** — combines Self-Consistency-style multi-sampling with MCTS and process verification.
- **RLVR (#39) and DeepSeek-R1 (#26)** — RL training pipelines that depend on sampling many completions and scoring them; Self-Consistency is the simplest scoring rule.
- **AlphaZero (#89)** — different domain, but the same philosophy: search broadly, aggregate signals, exploit the structure of the answer space.
- **Meta-CoT (#34) and o1 (#31)** — modern long-thought reasoning systems can be seen as Self-Consistency executed *within* a single rollout, with the model itself doing the aggregation.

---

## Key Takeaways

1. **Sample, then vote.** Take N CoT samples at non-zero temperature and return the most common answer. That single change reliably adds 5-20 points on hard reasoning benchmarks.
2. **The math is marginalization.** Self-Consistency approximates `argmax_y Sum_z P(y, z | x)`, summing over latent reasoning paths instead of greedily picking one. This is what the model's distribution *wants* to do; greedy decoding was just throwing the signal away.
3. **Right answers concentrate; wrong answers disperse.** Many valid reasoning paths arrive at the correct number, while errors fan out in many directions. Majority vote exploits that asymmetry.
4. **Costs N-times compute; usually worth it.** Most of the gain shows up in the first 10-20 samples, with diminishing returns after ~40.
5. **The blueprint for test-time compute.** Self-Consistency is the simplest member of a family that now includes best-of-N, process reward models, ToT, and o1-style reasoning models. If you understand Self-Consistency, you understand the core idea of "spend more inference, get smarter answers."
