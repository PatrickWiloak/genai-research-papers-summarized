# Knowledge and Reasoning Benchmarks

**In one line:** MMLU and its successors measure how much a model knows and how well it reasons on exam-style questions, and the useful ones in 2026 are the ones that are still hard: treat any score above about 90 percent as a solved test, not a ranking.
**Last reviewed:** 2026-09-30

---

## The short version

- Knowledge-and-reasoning benchmarks are **closed-ended exams**: a question, a single correct answer, and automatic grading. That makes them cheap, reproducible and easy to compare, which is why every model launch quotes them.
- They come in **generations**. MMLU (2020) was replaced by MMLU-Pro (2024), GPQA Diamond (2023) and Humanity's Last Exam (2025) as each earlier test saturated. HLE itself now has a cleaned 1,000-question subset, HLE-Diamond (September 2026).
- **ARC-AGI is the odd one out.** It tests learning a new rule from a few examples rather than recalling knowledge, and it has stayed hard for longer. Its third version (March 2026) is an interactive game benchmark where frontier models scored under 1 percent at launch.
- **Every one of these datasets contains wrong answers.** Audits found errors in an estimated 6.5 percent of MMLU and in a large share of HLE's chemistry and biology questions. Near the top of a leaderboard, part of the gap between models is label noise.
- **Read the settings, not just the number.** "With tools" versus "no tools", reasoning effort, and which question subset was used can move a score by tens of points.

## What these benchmarks are for

A language model is trained to predict text, so the first question anyone asks is "what does it know, and can it reason with it?" The cheapest way to find out is to give it an exam. The format is almost always the same:

```
question  ->  model answers  ->  compare to answer key  ->  accuracy %
              (letter A-J, or a short exact answer)
```

Because the grading is mechanical, two labs can run the same test and (in principle) get comparable numbers. The weakness is the flip side of the strength: the questions are fixed and public, so they leak into training data, and a model can learn the test rather than the subject. That problem gets its own page in [Contamination and Saturation](contamination-and-saturation.md).

## The benchmarks

### MMLU (2020)

**What it measures:** breadth of academic and professional knowledge. **Format:** four-option multiple choice across 57 subjects, from elementary maths and US history to law and medicine, about 14,000 test questions. **Scoring:** plain accuracy; random guessing gets 25 percent.

When Hendrycks and colleagues released it, most models were near chance and the largest GPT-3 was about 20 points above it. MMLU became the standard "general intelligence" number for four years. See the full summary: [MMLU](../../papers/techniques/137-mmlu/summary.md).

**Known flaws.**

- **Saturated.** By January 2025 the HLE authors could write that LLMs "now achieve over 90% accuracy on popular benchmarks like MMLU". A benchmark where every frontier model scores in the 90s no longer separates them.
- **Wrong answer keys.** The 2024 "Are We Done with MMLU?" audit re-annotated 5,700 questions (MMLU-Redux) and estimated that 6.49 percent of MMLU questions contain errors, including 57 percent of the analysed Virology questions.
- **Sensitive to prompt format.** How the options are laid out and whether the model answers with a letter or a sentence changes scores by several points.

### MMLU-Pro (2024)

**What it measures:** the same breadth as MMLU, with more reasoning. **Format:** ten answer options instead of four, harder reasoning-focused questions, trivial and noisy MMLU items removed. **Scoring:** accuracy; random guessing drops to 10 percent.

The authors reported accuracy falling 16 to 33 points versus MMLU for the same models, and score swings from prompt wording shrinking from 4-5 points to about 2. Chain-of-thought reasoning helps on MMLU-Pro where it barely helped on MMLU, which is evidence it tests more than recall. It is the usual stand-in when a launch table still wants a "general knowledge" line.

### GPQA and GPQA Diamond (2023)

**What it measures:** graduate-level science that cannot be looked up. **Format:** 448 four-option questions in biology, physics and chemistry written by domain experts; the **Diamond** subset is the 198 highest-quality questions and is the one everyone reports. **Scoring:** accuracy.

The design is clever: "Google-proof" means skilled non-experts with unrestricted web access and over 30 minutes per question reached only 34 percent, while PhD-level experts in the field reached 65 percent (74 percent discounting mistakes they later acknowledged). The best GPT-4 baseline in the paper scored 39 percent.

**Known flaws.** It is small (198 questions means each question is worth about half a point, so differences of one or two points are noise), it is multiple choice (so elimination and guessing help), and it is now near its ceiling: Epoch AI's benchmark hub notes a PhD-expert baseline of 69.7 percent measured by OpenAI, and frontier models score well above that. It remains useful mainly as a floor check.

### Humanity's Last Exam (2025)

**What it measures:** the frontier of expert knowledge and reasoning across dozens of subjects. **Format:** 2,500 questions (finalised April 2025), a mix of multiple-choice and exact short answers, some with images, written by subject-matter experts worldwide. Questions were kept only if frontier models of the time failed them. **Scoring:** accuracy, graded by checking the final answer against the key; calibration error is reported too.

Built by the Center for AI Safety and Scale AI, HLE was explicitly designed as "the final closed-ended academic benchmark of its kind". It was published in *Nature* in January 2026.

**Known flaws.**

- **Adversarial construction breeds bad questions.** Because a question was accepted only if models got it wrong, some accepted questions were wrong or ambiguous. FutureHouse (July 2025) found that 29 percent (plus or minus 3.7) of the text-only chemistry and biology answers conflicted with peer-reviewed literature. The HLE team's own three-expert follow-up found about 18 percent of a bio/chem subset problematic. The team responded with **HLE-Rolling** (a continuously revised fork, October 2025) and **HLE-Diamond** (1,000 cleaned questions, 500 reasoning and 500 knowledge, September 22, 2026).
- **"With tools" and "no tools" are different tests.** With web search and code execution, a model can look things up. Anthropic's launch notes, for example, specify the full tool set and a domain blocklist used "to decontaminate eval results" when reporting HLE with tools. Never compare a with-tools number against a no-tools one.
- **Subset choice.** Independent evaluator Artificial Analysis runs only the 2,158 text-only questions, so its numbers do not line up exactly with lab-reported full-set numbers.

**Where the frontier stands (as of September 2026).** On HLE-Diamond without tools, the HLE team's launch results put GPT-6 Astra top at 59.9 percent and Claude Opus 5.5 at 54.6 percent. On the full text-only set without tools, Artificial Analysis's leaderboard in September 2026 showed Claude Opus 5.5 highest at 61.4 percent. For scale, FutureHouse noted that as of July 2025 the top model scored 26.9 percent without tools and 44 percent with tools and internet access.

### BIG-Bench, BIG-Bench Hard and BIG-Bench Extra Hard

**What it measures:** a grab bag of reasoning skills. **Format:** the original BIG-bench (2022) was 204 tasks from 450 authors across 132 institutions, covering linguistics, logic, maths, social bias, common sense and more. **BIG-Bench Hard** (BBH, 2022) picked the 23 tasks where models still lost to the average human rater. **Scoring:** per-task accuracy or exact match, averaged.

BBH is best known as the benchmark that showed **chain-of-thought prompting** unlocking reasoning: with it, Codex beat average human raters on 17 of the 23 tasks (see [Chain-of-Thought](../../papers/techniques/09-chain-of-thought/summary.md)). BIG-bench was also the source for much of the "emergent abilities" debate (see [Emergent Abilities](../../papers/techniques/81-emergent-abilities/summary.md)).

**Known flaws.** BBH is saturated: Google DeepMind's **BIG-Bench Extra Hard** (February 2025) replaced every BBH task with a harder one probing the same skill, because "state-of-the-art models achieve near-perfect scores on many tasks in BBH". At BBEH's release the best general model scored 9.8 percent (harmonic mean) and the best reasoning model 44.8 percent. BIG-bench also pioneered the **canary string**, a unique marker in every task file so model builders can filter it out of training data; it only works if they actually do.

### ARC-AGI (2019 to 2026)

**What it measures:** fluid intelligence, meaning learning a new rule from very little data. **Format:** in ARC-AGI-1 and 2, a few example pairs of coloured grids, each showing an input transformed into an output; the model must infer the rule and produce the output grid for a new input. **Scoring:** exact match on the whole grid, usually allowing two attempts.

François Chollet's 2019 paper "On the Measure of Intelligence" defined intelligence as **skill-acquisition efficiency** and built ARC so that memorised knowledge does not help. See the summary: [ARC](../../papers/techniques/138-arc-agi/summary.md).

| Version | Released | What happened |
|---|---|---|
| ARC-AGI-1 | 2019 | Hard for LLMs for five years. In December 2024 OpenAI's o3 scored 75.7% on the semi-private set at about $26 per task and 87.5% at a far larger compute setting. That o3 had been trained on 75% of the public training set. |
| ARC-AGI-2 | March 24, 2025 | Every task solved by at least 2 humans in 2 attempts or fewer. At launch, pure LLMs scored 0% and o3-preview (low) about 4%. The ARC Prize 2025 Kaggle winner (NVARC) reached 24.03% on the private set at about $0.20 per task. |
| ARC-AGI-3 | March 25, 2026 | Hundreds of hand-built, turn-based game environments with no instructions or stated goals: the agent must explore, work out the rules and what "winning" means. Humans solved every environment; frontier AI scored 0.51% at launch. |

**Known flaws and debates.** ARC's own organisers now report **cost per task** alongside score, because a brute-force search can buy accuracy with compute (the o3 result is the standard example). Training on the public ARC training set is allowed but blurs what "novel" means. And there is a live debate about whether a grid-puzzle benchmark measures "general" intelligence or one specific visual skill; see [Do LLMs Reason?](../open-questions/do-llms-reason.md).

**Where it stands (as of September 2026).** The ARC Prize 2026 competition on ARC-AGI-3 runs until November 2, 2026, with a $700,000 grand prize for the first agent to score 100 percent on the private set, and no one has claimed it.

## How to read a model launch's benchmark table

Every launch post has a table like the one below. Here is how to read it without being misled. The same checklist applies to the maths, code and agent pages.

```
Benchmark             Model A   Model B   Model C
GPQA Diamond          91.2%     90.8%     88.5%     <- ceiling: gaps are noise
HLE (no tools)        41.0%     35.2%     --        <- still discriminating
HLE (with tools)      53.0%     --        45.1%     <- different test from row above
SWE-bench Verified    80.9%*    79.4%     77.0%     <- * = footnote: read it
ARC-AGI-2             68.8%     --        52.9%     <- check cost per task
```

(Illustrative numbers, not real results.)

1. **Is the benchmark saturated?** If the top scores are within a few points of each other and near the known label-error rate or human ceiling (MMLU, GPQA Diamond), the row tells you almost nothing about ranking.
2. **Is every column run the same way?** Look for "with tools" versus "no tools", reasoning effort ("max", "high"), number of attempts (pass@1 versus best-of-N), and averaging over several runs. Anthropic's Claude Sonnet 4.5 launch, for instance, footnoted that its SWE-bench Verified 77.2% was averaged over 10 trials with a specific prompt addition, and that a parallel test-time-compute setup reached 82.0%. Both numbers are honest; they measure different things.
3. **Who ran the competitors' numbers?** Many tables mix "our run of our model" with "their published number for their model", often using different harnesses. Footnotes usually say so.
4. **Which version and subset?** HLE full set versus text-only versus HLE-Diamond; SWE-bench Verified versus Pro; ARC-AGI-1 versus 2 versus 3. Versions are not comparable.
5. **What is missing?** Labs choose which benchmarks to show. A benchmark that everyone else reports and this table omits is information.
6. **Is the difference bigger than the noise?** With a 198-question test, two points is four questions. Look for confidence intervals; if there are none, assume a few points are within noise.
7. **Is there an independent replication?** Evaluators such as Epoch AI, Artificial Analysis and the benchmark maintainers' own leaderboards rerun models under fixed settings. Their numbers are often lower than launch numbers and more comparable across labs.

For the practical side of running your own evals, see the sibling repo's [Evals for LLMs](https://github.com/PatrickWiloak/cloud-data-ai-security-zero-to-hero/blob/main/learn/concepts/evals-for-llms.md).

## What to watch

- **HLE-Diamond adoption.** If labs switch their launch tables from full HLE to HLE-Diamond (released September 22, 2026), expect a reset in reported numbers and fewer disputed questions.
- **ARC Prize 2026 results**, due December 4, 2026. A large jump on ARC-AGI-3's private set would be the first strong sign that agents can learn unfamiliar interactive rules efficiently.
- **The end of static exams.** HLE was billed as the last closed-ended academic benchmark of its kind. The field is moving toward maintained, versioned benchmarks (HLE-Rolling) and agentic tasks; see [Agents and Computer Use](agents-and-computer-use.md).

## Read next

- [MMLU](../../papers/techniques/137-mmlu/summary.md) and [ARC / On the Measure of Intelligence](../../papers/techniques/138-arc-agi/summary.md)
- [Chain-of-Thought Prompting](../../papers/techniques/09-chain-of-thought/summary.md) and [Emergent Abilities](../../papers/techniques/81-emergent-abilities/summary.md)
- [OpenAI o1](../../papers/language-models/31-openai-o1/summary.md) and [Test-Time Compute](../../papers/techniques/50-test-time-compute/summary.md), which explain why reasoning models jumped on GPQA and ARC
- Explainers: [Math and Code](math-and-code.md), [Human-Preference Arenas](human-preference-arenas.md), [Contamination and Saturation](contamination-and-saturation.md), [Reasoning Models](../concepts/reasoning-models.md), [Do LLMs Reason?](../open-questions/do-llms-reason.md)

## Sources

- Hendrycks et al., "Measuring Massive Multitask Language Understanding" (2020): https://arxiv.org/abs/2009.03300
- Gema et al., "Are We Done with MMLU?" (2024): https://arxiv.org/abs/2406.04127
- Polo et al., "tinyBenchmarks" (2024), for MMLU's ~14K test size: https://arxiv.org/abs/2402.14992
- Wang et al., "MMLU-Pro" (2024): https://arxiv.org/abs/2406.01574
- Rein et al., "GPQA: A Graduate-Level Google-Proof Q&A Benchmark" (2023): https://arxiv.org/abs/2311.12022
- Epoch AI, GPQA Diamond benchmark page (198 questions, 69.7% expert baseline): https://epoch.ai/benchmarks/gpqa-diamond
- Phan et al., "Humanity's Last Exam" (2025): https://arxiv.org/abs/2501.14249
- Humanity's Last Exam site and news (finalised April 3, 2025; HLE-Rolling October 8, 2025; *Nature* January 28, 2026): https://lastexam.ai/
- CAIS and Scale AI, "Introducing HLE-Diamond" (September 22, 2026): https://lastexam.ai/blog/hle-diamond
- FutureHouse, "About 30% of Humanity's Last Exam chemistry/biology answers are likely wrong" (July 23, 2025, updated September 16, 2025): https://www.futurehouse.org/research-announcements/hle-exam
- Artificial Analysis, Humanity's Last Exam leaderboard (viewed September 2026): https://artificialanalysis.ai/evaluations/humanitys-last-exam
- Anthropic, "Introducing Claude Opus 4.6" (February 5, 2026), HLE methodology footnote: https://www.anthropic.com/news/claude-opus-4-6
- Anthropic, "Introducing Claude Sonnet 4.5", SWE-bench methodology footnote: https://www.anthropic.com/news/claude-sonnet-4-5
- Srivastava et al., "Beyond the Imitation Game" (BIG-bench, 2022): https://arxiv.org/abs/2206.04615
- Suzgun et al., "Challenging BIG-Bench Tasks and Whether Chain-of-Thought Can Solve Them" (2022): https://arxiv.org/abs/2210.09261
- Kazemi et al., "BIG-Bench Extra Hard" (2025): https://arxiv.org/abs/2502.19187
- BIG-bench repository README (canary strings): https://github.com/google/BIG-bench
- Chollet, "On the Measure of Intelligence" (2019): https://arxiv.org/abs/1911.01547
- ARC Prize, "OpenAI o3 Breakthrough High Score on ARC-AGI-Pub" (December 20, 2024): https://arcprize.org/blog/oai-o3-pub-breakthrough
- ARC Prize, "Announcing ARC-AGI-2 and ARC Prize 2025" (March 24, 2025): https://arcprize.org/blog/announcing-arc-agi-2-and-arc-prize-2025
- ARC Prize, "ARC Prize 2025 Results and Analysis": https://arcprize.org/blog/arc-prize-2025-results-analysis
- ARC Prize, "Announcing ARC-AGI-3" (March 25, 2026): https://arcprize.org/blog/arc-agi-3-launch
- ARC Prize 2026 competition pages (key dates, prizes): https://arcprize.org/competitions/2026 and https://arcprize.org/competitions/2026/arc-agi-3
