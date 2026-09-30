# Human-Preference Arenas and LLM Judges

**In one line:** Arena leaderboards rank models by which answer people prefer in blind side-by-side votes, which captures "feels better to use" better than any exam, but preference is not correctness, style sways votes, and the 2025 "Leaderboard Illusion" dispute showed how the rules of the arena shape who wins.
**Last reviewed:** 2026-09-30

---

## The short version

- **Chatbot Arena** (later LMArena, now **Arena**) shows a user two anonymous model answers to their own prompt, asks which is better, and turns millions of such votes into a rating. It became the most-watched public ranking of chat models.
- The ratings are "Elo-style" but since December 2023 are fitted with the **Bradley-Terry** model, a statistical version of the same idea. A rating difference translates into a predicted win rate; the absolute number means nothing on its own.
- **Style biases votes.** Longer, more formatted answers win more often. Since August 2024 Arena has offered **style control**, a regression that separates style from substance, and it became the default view in May 2025.
- **LLM-as-a-judge** benchmarks (MT-Bench, AlpacaEval, Arena-Hard-Auto) replace the human voter with a strong model to get cheap, fast approximations of Arena rankings. They inherit and add biases: position, length, self-preference.
- **The Leaderboard Illusion** (Cohere Labs and academic co-authors, April 2025) argued that private pre-release testing, unequal sampling and unequal data access let a few large labs tune for Arena. Arena disputed several numbers but changed some policies. Both sides agree the ranking depends on the rules as well as on the models.

## Why preference arenas exist

Exams like MMLU (see [Knowledge and Reasoning](knowledge-and-reasoning.md)) have one right answer. Most real use of a chat model does not: "rewrite this email", "explain this error", "plan a trip". No answer key exists, and automatic text-similarity metrics correlate poorly with what people think is good. The Arena idea, introduced by the LMSYS group at UC Berkeley in April 2023, is simple:

```
user types any prompt
        |
   +----+----+
   v         v
 Model ?   Model ?          (anonymous, randomly paired)
   |         |
 answer A  answer B
   \        /
    user votes: A better / B better / tie / both bad
        |
 identities revealed; vote added to the pool
        |
 fit Bradley-Terry model over all votes -> rating per model
```

Because prompts come from real users, the benchmark cannot be memorised in the usual way, and it keeps moving as people ask about new things. The paper behind it, together with MT-Bench, is summarised in [Judging LLM-as-a-Judge](../../papers/techniques/85-llm-as-judge/summary.md).

## How the rating works: Elo and Bradley-Terry

**Elo** comes from chess. Each player has a rating; the difference between two ratings predicts the probability that one beats the other, and every game nudges both ratings. On the usual scale, a 100-point gap means the higher-rated side is expected to win about 64 percent of the time, and a 400-point gap about 91 percent.

```
P(A beats B) = 1 / (1 + 10^((R_B - R_A) / 400))
```

Online Elo has a flaw for this use: the result depends on the **order** games were played in, and recent games count more. Chess players change over time, so that is a feature. A fixed model checkpoint does not change, so it is noise. In December 2023 LMSYS moved to the **Bradley-Terry** model, which fits one set of ratings to all votes at once, gives the same answer regardless of order, and produces **confidence intervals**. Scores are still reported on an Elo-like scale, which is why people still say "Elo".

**How to read an Arena score:**

- Only **differences** matter. "1,450" means nothing alone; "30 points above model X" means an expected win rate of roughly 54 percent against it.
- Check the **confidence interval**. Models whose intervals overlap are statistically tied.
- Check the **category**. Arena reports separate boards (hard prompts, for example), and results often differ from the overall ranking.

## Style control

In August 2024 the Arena team asked "does style matter?" and found it does. They added style features (answer length, and counts of markdown headers, bold text and list items) as extra variables in the Bradley-Terry regression, so each model's score estimates how often it would win **if both answers had the same style**.

The effect on rankings was immediate: GPT-4o-mini and Grok-2-mini dropped below most frontier models, while Claude 3.5 Sonnet, Claude 3 Opus and Llama-3.1-405B rose substantially. In May 2025, after a community survey, style control became the **default** leaderboard view, and the team said it planned to add controls for sentiment and sycophancy.

Style control is a correction, not a cure. It only removes the style features it measures, and a model can still win votes by being confident, agreeable or flattering in ways no simple feature captures.

## The benchmarks

### Chatbot Arena / LMArena / Arena

**Measures:** aggregate human preference on real user prompts. **Format:** anonymous pairwise battles on text, with image support added in 2024 and video by early 2026. **Scoring:** Bradley-Terry ratings with confidence intervals, overall and per category.

| Date | Event |
|---|---|
| April 2023 | Chatbot Arena launched by LMSYS (UC Berkeley) |
| December 2023 | Switch from online Elo to Bradley-Terry |
| March 2024 | Paper published: over 240,000 votes collected by then |
| August 2024 | Style control introduced |
| September 2024 | Moved to its own domain as **LMArena** |
| April to May 2025 | Incorporated as a company; $100 million seed round at a $600 million valuation |
| May 2025 | Style control becomes the default view |
| January 2026 | $150 million Series A at about $1.7 billion; rebrand to **Arena** on January 28 |

**Known flaws.**

- **Preference is not correctness.** A voter asking about a topic they do not know cannot tell a confident wrong answer from a right one. Arena is weak evidence about factual accuracy, maths or safety.
- **Who votes, and on what.** Voters are self-selected visitors to an AI comparison site, and their prompts may not look like yours.
- **Submitted variant versus released model.** In April 2025 Meta's Llama 4 Maverick appeared high on the leaderboard using a variant named "Llama-4-Maverick-03-26-Experimental", which Meta said was "optimized for conversationality". The released model, once tested, ranked 32nd as of April 11, 2025. The Arena maintainers apologised and changed their policies. See [Llama 4](../../papers/language-models/41-llama4/summary.md).
- **Incentives.** Since 2025 the arena has been a venture-funded company ranking the products of the industry it serves. That does not show bias, but it is a structural fact worth knowing.

### The Leaderboard Illusion (2025)

In April 2025 Shivalika Singh, Sara Hooker and co-authors (Cohere Labs with academic collaborators) published a critique of Chatbot Arena. Their main claims:

- **Private testing and selective disclosure.** Some providers test many private variants before release and publish only the best score. They identified 27 private variants tested by Meta before Llama 4. Picking the maximum of many noisy scores inflates the result even if no single variant is better.
- **Unequal sampling.** Proprietary models were sampled in more battles, and fewer were removed. They estimated Google and OpenAI each received about 19 to 20 percent of all Arena data, while 83 open-weight models together received 29.7 percent.
- **Data access enables overfitting.** Battle data flows back to the providers whose models are sampled. They reported that training on Arena data gave relative gains of up to 112 percent on the Arena distribution.
- **Silent deprecation.** Many models were retired without notice, which can distort the rankings of those that remain.

**Arena's response (May 2025)** disputed several figures. It said open models made up 40.9 percent of its data by its own April 2025 statistics, that the paper's 100-plus-point boost from private testing came from a simulation rather than Arena data, and that the real boost was about 11 points after 50 tests and shrinks as fresh votes arrive. It said its pre-release testing policy had been public since March 2024 and open to all providers. It also committed to changes: explicitly allowing every provider to test multiple variants, clearly marking retired models, and labelling scores "provisional" until enough fresh post-release votes arrive when many variants were tested.

**How to weigh it.** Both sides agree on the mechanism: taking the best of many private tries biases the published score upward, and more data from a distribution helps a model do well on it. They disagree on the size of the effect and whether the rules were fair. For a reader, the practical lesson is to treat small Arena gaps between frontier models as uninformative and to be suspicious of any model that debuts at the top and then falls.

### MT-Bench (2023)

**Measures:** multi-turn conversation quality. **Format:** 80 hand-written two-turn questions across 8 categories (writing, roleplay, extraction, reasoning, maths, coding, STEM, humanities). **Scoring:** a strong model (originally GPT-4) grades each answer, or compares two answers.

Its paper showed GPT-4's judgements agreed with human preferences about 80 percent of the time, roughly the rate at which two humans agree, and it catalogued the judge's biases: favouring the first answer shown (**position bias**), favouring longer answers (**verbosity bias**), favouring its own outputs (**self-enhancement bias**), and grading maths poorly.

**Known flaws.** Eighty questions is small, it has been public since 2023, and frontier models now score near the top, so it no longer separates them.

### AlpacaEval and length-controlled AlpacaEval (2023, 2024)

**Measures:** instruction following on everyday requests. **Format:** 805 instructions; a judge model compares the evaluated model's answer with a fixed reference model's answer. **Scoring:** **win rate** against the reference.

AlpacaEval is fast and cheap (its maintainers quote under $10 and under three minutes per run), but it strongly favoured longer outputs. **Length-controlled AlpacaEval** (April 2024) fits a regression to predict what the judge would prefer if both answers were the same length. This made the metric harder to game by padding and raised its Spearman rank correlation with Chatbot Arena from 0.94 to 0.98. It is now AlpacaEval's default metric.

**Known flaws.** A 0.98 rank correlation is measured across a set of models at one point in time; it does not mean every individual comparison matches human judgement. The judge is itself a model from one lab, which raises self-preference questions, and the instruction set is public.

### Arena-Hard-Auto

The Arena team also maintains **Arena-Hard-Auto**: 500 challenging prompts and 250 creative-writing prompts drawn from Arena traffic, graded by LLM judges, and designed to predict Arena rankings offline. It is widely used in open-model papers. The same caution applies as for any judge-graded benchmark: check which judge model was used and whether style control was applied.

## Reading preference results in a launch table

The general checklist is in [Knowledge and Reasoning](knowledge-and-reasoning.md#how-to-read-a-model-launchs-benchmark-table). For preference rows, also check:

1. **Is it style-controlled?** Since May 2025 this is Arena's default, but launch posts sometimes quote whichever view looks best.
2. **Which board and which date?** Arena ratings move as new models and votes arrive. A launch-day rank is provisional.
3. **Was the tested model the released model?** Look for the exact model identifier.
4. **For LLM-judge benchmarks, which judge?** Scores from different judges are not comparable, and a judge from the same lab as the model under test is a conflict of interest.

## What to watch

- **Whether Arena's 2025 reforms hold up** as it grows as a company, and whether independent audits of sampling rates and provisional labels appear.
- **Beyond style control.** Arena has said it wants to control for sentiment and sycophancy. A model that wins by flattering the voter is the failure mode to watch.
- **Specialised boards and expert raters.** Preference rankings are increasingly split by domain, which gives more useful but less headline-friendly numbers.

## Read next

- [Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena](../../papers/techniques/85-llm-as-judge/summary.md)
- [InstructGPT / RLHF](../../papers/language-models/05-instructgpt-rlhf/summary.md) and [DPO](../../papers/language-models/19-dpo/summary.md): human preferences as a training signal, not just an evaluation
- [Llama 4](../../papers/language-models/41-llama4/summary.md)
- Explainers: [Contamination and Saturation](contamination-and-saturation.md), [Llama](../model-families/llama.md), [Labs Landscape](../ecosystem/labs-landscape.md)
- Sibling repo: [Evals for LLMs](https://github.com/PatrickWiloak/cloud-data-ai-security-zero-to-hero/blob/main/learn/concepts/evals-for-llms.md)

## Sources

- Zheng et al., "Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena" (2023): https://arxiv.org/abs/2306.05685
- Chiang et al., "Chatbot Arena: An Open Platform for Evaluating LLMs by Human Preference" (2024): https://arxiv.org/abs/2403.04132
- LMSYS, "Chatbot Arena: New models & Elo system update" (December 7, 2023; Bradley-Terry transition): https://www.lmsys.org/blog/2023-12-07-leaderboard/
- Li, Angelopoulos and Chiang, "Does style matter? Disentangling style and substance in Chatbot Arena" (August 29, 2024): https://www.lmsys.org/blog/2024-08-28-style-control/
- Arena (@arena) on X, style control becomes the default view (May 16, 2025): https://x.com/arena/status/1923398953468678529
- Wikipedia, "LMArena" / "Arena (AI platform)" (launch date, rebrands, funding rounds, modalities): https://en.wikipedia.org/wiki/LMArena
- TechCrunch, "Meta's vanilla Maverick AI model ranks below rivals on a popular chat benchmark" (April 11, 2025): https://techcrunch.com/2025/04/11/metas-vanilla-maverick-ai-model-ranks-below-rivals-on-a-popular-chat-benchmark
- Singh et al., "The Leaderboard Illusion" (April 29, 2025): https://arxiv.org/abs/2504.20879
- Arena, "Our response to 'The Leaderboard Illusion'" (May 9, 2025, updated June 13, 2025): https://arena.ai/blog/our-response/
- Dubois et al., "Length-Controlled AlpacaEval: A Simple Way to Debias Automatic Evaluators" (2024): https://arxiv.org/abs/2404.04475
- AlpacaEval repository README (805 instructions, cost, length-controlled default): https://github.com/tatsu-lab/alpaca_eval
- Arena-Hard-Auto repository: https://github.com/lm-sys/arena-hard-auto
