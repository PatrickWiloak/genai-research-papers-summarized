# STaR: Bootstrapping Reasoning With Reasoning (Self-Taught Reasoner)

**Authors:** Eric Zelikman, Yuhuai Wu, Jesse Mu, Noah D. Goodman (Stanford University, Google Research)

**Published:** March 2022 (NeurIPS 2022)

**Paper Link:** https://arxiv.org/abs/2203.14465

---

## Why This Paper Matters

STaR is the quiet conceptual ancestor of OpenAI's o1 and DeepSeek-R1. Years before "reasoning models" became the dominant paradigm, STaR proposed a simple but profound idea: **let a language model bootstrap its own reasoning ability by keeping the chains of thought that lead to correct answers, then fine-tuning on them.**

This is the core feedback loop of modern reasoning models — generate reasoning traces, filter by outcome correctness, train on the survivors, repeat. STaR did it in 2022 with supervised fine-tuning on a small model. DeepSeek-R1 (#26) and o1 (#31) do essentially the same thing at vastly larger scale with RL (GRPO #38, RLVR #39) and longer context. The conceptual debt is direct.

STaR also introduced **rationalization** — when the model gets a question wrong, give it the correct answer and ask it to invent a reasoning chain that would have led there. This "backward justification" trick made the bootstrap dramatically more sample-efficient and foreshadows process-reward (#51) and Meta-CoT (#34) ideas where reasoning traces are themselves the supervision signal.

---

## The Problem Before

By early 2022, Chain-of-Thought prompting (#09) had shown that asking a model to "think step by step" before answering substantially improved performance on math and reasoning. But CoT had limits:

- **Few-shot prompting only.** You needed humans to hand-craft a small number of example reasoning chains per task.
- **No improvement loop.** The base model's reasoning ability didn't actually get better — you were just prompting more cleverly.
- **Expensive data.** Datasets of step-by-step solutions (like GSM8K's annotated rationales) were scarce and labor-intensive to create.
- **Small reasoning datasets capped performance.** Fine-tuning on 7,500 human-written rationales for GSM8K helped, but you couldn't scale it without paying humans to write more.

The question STaR asks: **can the model generate its own training data of reasoning traces, supervised only by whether the final answer is correct?**

---

## The Core Innovation: A Bootstrap Loop

STaR's algorithm is striking in its simplicity.

```
Start with a pretrained LM and a dataset of (question, correct answer) pairs
(NO ground-truth reasoning chains needed).

Repeat:
  1. For each question, prompt the model to produce
        rationale -> answer
  2. KEEP only the (question, rationale, answer) triples where
        the answer matches the ground truth.
  3. For questions the model got WRONG:
        give the model the correct answer
        ask it to produce a rationale that would have led there
        (this is "rationalization")
  4. Fine-tune the model on ALL kept (question, rationale, answer) triples.
  5. Use the improved model to generate the next batch.
```

Each iteration the model:
- Solves more questions correctly.
- Generates more training data.
- Becomes better at producing useful reasoning traces.

It is **outcome-supervised self-improvement** — the only signal needed is "is the final answer right or wrong," which for math, code, and many factual tasks comes for free.

### Why rationalization matters

Without rationalization, the bootstrap stalls. The model produces correct reasoning for problems it already happens to solve, but never sees training examples for problems it gets wrong. Hard problems stay forever out of reach.

Rationalization fixes this. By telling the model the correct answer in advance and asking it to "explain why this is the answer," the model can produce plausible reasoning for problems it couldn't solve from scratch. These reverse-engineered rationales are surprisingly useful training data — they're not just memorization, because the model still has to construct an actual chain of reasoning ending in the given answer.

Conceptually, rationalization is a kind of **hindsight relabeling**: turn failures into successes by changing what you're conditioning on.

---

## Key Results

STaR was tested on three tasks:

- **Arithmetic** (multi-digit addition): STaR boosted accuracy from 76.3% to 89.5% on a 6B-parameter GPT-J, beating much larger models trained on the same data without bootstrapping.
- **CommonsenseQA**: STaR-trained 6B model reached **72.5%** accuracy, comparable to a 30x larger 137B GPT-3 model fine-tuned conventionally on the same dataset.
- **GSM8K** (grade-school math): STaR matched or beat fine-tuning on the human-annotated rationales — without using *any* human rationales, only the final answers.

A key finding: **most of the gains came from rationalization.** Without it, the bootstrap loop plateaus quickly because the model can never escape its initial competence ceiling. With it, the loop keeps producing meaningful training data on harder problems.

---

## A Worked Example

Suppose the training problem is:

> *Q: A train leaves Chicago at 60 mph. Another leaves New York at 80 mph, 800 miles away, two hours later. When do they meet?*
>
> *A: 7 hours after the first train leaves.*

A pretrained 6B model might struggle on this. STaR proceeds as follows:

1. **Sample with CoT prompting.** The model generates ten attempts. Two get the right final answer; eight don't.
2. **Keep the two correct (question, rationale, answer) triples** — these are now training data.
3. **For one of the failed attempts**, present the correct answer and ask: "Given that the answer is 7 hours, what's the reasoning?" The model now produces a plausible step-by-step explanation. This rationalized triple also joins the training set.
4. **Fine-tune.** The model is updated on the combined set.
5. **Re-sample.** Next iteration, the model gets the question right 4 out of 10 times. The pool of training examples grows. Harder problems become solvable.

Over a handful of iterations, the model that was initially correct on 5% of problems is correct on 60%+ — purely from its own (filtered) outputs.

---

## Why It Works (and Where Naive Versions Fail)

The non-obvious insight is that **filtering by correctness is a form of supervision**, even though the rationale text itself isn't labeled. The argument:

- Random rationales rarely lead to correct answers by accident.
- Therefore, rationales that *do* lead to correct answers are *biased toward containing valid reasoning*.
- Training on this biased set pushes the model toward producing valid-reasoning-like text more often.

This is a kind of **rejection sampling fine-tuning** — a technique that reappears in RLHF (#05) and in modern RL pipelines as a strong baseline before adding full PPO/GRPO.

Pitfalls to avoid:
- **Shortcut exploitation.** If the dataset has spurious patterns (e.g., the answer is always near the end of the problem), the model can find them without doing real reasoning. Diverse datasets matter.
- **Mode collapse.** Without enough diversity in sampled rationales, the model can lock into a single solution style and fail to generalize.
- **Rationalization hallucination.** When the model invents a rationale for a hard problem given the answer, it can produce confident-sounding but wrong reasoning. The paper mitigates this by giving rationalized samples lower weight or fewer iterations of training.

---

## Impact and Legacy

STaR was undervalued at the time — it was a relatively small paper at NeurIPS 2022, and its 6B-parameter setting felt modest. In hindsight, almost every major reasoning model since 2024 implements a version of its core loop:

- **OpenAI o1 (#31)** is reported to use large-scale RL on reasoning traces, keeping the ones that lead to verifiable correct answers — STaR's bootstrap loop, scaled up and with RL replacing SFT.
- **DeepSeek-R1 (#26)** explicitly describes a pipeline of: sample reasoning traces, filter by correctness using verifiable rewards, fine-tune (or RL), repeat. R1-Zero variant is particularly close to STaR in spirit — RL-only bootstrap from outcome rewards.
- **rStar-Math (#35)** uses Monte Carlo Tree Search to generate and filter reasoning traces, then trains a small model on the filtered traces — a tree-augmented STaR.
- **Meta-CoT (#34)** explicitly cites STaR as a foundation for self-improving reasoning systems.
- **Process Reward Models (#51)** can be viewed as adding step-level supervision on top of STaR's outcome-level supervision.
- **GRPO (#38) and RLVR (#39)** provide the RL machinery that lets STaR's loop run at scale and with longer reasoning chains than supervised fine-tuning can support.

STaR also reframed how people thought about training data: **reasoning data can be generated, not just collected.** This is now central to the field — frontier labs are spending enormous compute on generating, filtering, and training on synthetic reasoning traces.

---

## Limitations

- **Requires verifiable answers.** STaR only works on tasks where you can check correctness automatically. Open-ended generation (creative writing, summarization) doesn't fit.
- **No supervision over the reasoning itself.** A model can produce a wrong rationale that happens to land on the right answer — and STaR will train on it. This noise is tolerable in aggregate but adds inefficiency. Process Reward Models (#51) address this by scoring individual steps.
- **Plateaus on hard problems.** Without rationalization, the bootstrap can't escape the base model's competence ceiling. Even with rationalization, problems where the model never gets close are unreachable.
- **Small-scale demonstration.** STaR's results were on a 6B model and three tasks. Whether the loop scales cleanly to frontier-size models and broader task distributions wasn't shown until much later work (R1-Zero in particular).
- **Distribution shift between iterations.** Fine-tuning shifts what the model samples next iteration; if the new distribution is too narrow, the loop can collapse onto a single style of reasoning rather than diversifying.

---

## Connections to Other Papers

- **Chain-of-Thought (#09):** STaR's foundation. CoT showed that reasoning traces help; STaR showed you can train *on* reasoning traces, generated by the model itself.
- **DeepSeek-R1 (#26):** The most direct large-scale realization of the STaR philosophy — outcome-supervised RL on self-generated reasoning traces.
- **OpenAI o1 (#31):** Same conceptual core, scaled up and combined with test-time search.
- **Meta-CoT (#34):** Generalizes STaR-style bootstrapping into the broader paradigm of reasoning-trace generation as supervision.
- **rStar-Math (#35):** Tree-search-augmented version of STaR's bootstrap loop for math problems.
- **GRPO (#38) / RLVR (#39):** The RL algorithms used by modern systems to scale STaR-like loops beyond what SFT supports.
- **Process Reward Models (#51):** Add per-step supervision; STaR uses only outcome supervision.
- **Test-Time Compute (#50):** Complementary axis — STaR trains better reasoning; test-time compute uses better reasoning at inference.
- **Tree of Thoughts (#25):** Search over reasoning traces; can be combined with STaR-style training.

---

## Key Takeaways

- **Outcome supervision is enough**: a model can teach itself to reason given only (question, correct answer) pairs, no human-written rationales required.
- **The bootstrap loop**: generate rationales -> filter by correctness -> fine-tune -> repeat. This single recipe is the conceptual core of modern reasoning models.
- **Rationalization breaks plateaus**: giving the model the answer and asking it to produce supporting reasoning lets it learn from problems it couldn't initially solve.
- **A 6B model trained with STaR matched a 137B model on CommonsenseQA** — a strikingly early hint that reasoning training is more valuable than raw scale.
- **Direct intellectual ancestor of o1, R1, rStar-Math, and the modern reasoning-model paradigm** — STaR articulated the loop the entire field is now running at massive scale.
