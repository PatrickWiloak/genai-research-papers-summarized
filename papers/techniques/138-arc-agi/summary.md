---
title: "On the Measure of Intelligence (ARC)"
slug: "138-arc-agi"
number: 138
category: "techniques"
authors: "François Chollet (Google)"
published: "November 2019 (arXiv preprint)"
year: 2019
url: "https://arxiv.org/abs/1911.01547"
tags: ["evaluation", "benchmarks", "reasoning"]
---

# On the Measure of Intelligence (ARC)

**Authors:** François Chollet (Google)
**Published:** November 2019 (arXiv preprint)
**Paper:** [arxiv.org/abs/1911.01547](https://arxiv.org/abs/1911.01547)

---

## Why This Matters

This 64-page essay does two things. It argues that the AI field has been **measuring the wrong thing**: skill at particular tasks rather than the ability to acquire new skills. And it introduces a benchmark built on that argument, the **Abstraction and Reasoning Corpus (ARC)**, a set of small coloured-grid puzzles that most people can solve and that AI systems failed at for five years.

- **A definition of intelligence you can operationalise:** "The intelligence of a system is a measure of its skill-acquisition efficiency over a scope of tasks, with respect to priors, experience, and generalization difficulty."
- **A benchmark that resisted scale.** While language models went from GPT-2 to GPT-4, ARC stayed largely unsolved; the best Kaggle score in 2020 was about 20 percent, and the state of the art was about 33 percent in early 2024.
- **A public test for AGI claims.** The ARC Prize (from 2024) turned it into a million-dollar competition, and ARC results became a headline in frontier model launches.
- **A marker for the reasoning-model shift.** In December 2024, OpenAI's o3 preview scored 75.7 to 87.5 percent on ARC-AGI-1, the clearest early sign that test-time reasoning changed what models could do.

**The insight:** if you can buy any level of skill with enough training data or built-in knowledge, then skill says nothing about intelligence. As the abstract puts it, "unlimited priors or unlimited training data allow experimenters to 'buy' arbitrary levels of skills for a system, in a way that masks the system's own generalization power." Measure instead how efficiently a system turns a small amount of experience into skill on problems it has never seen.

---

## The Argument

### Skill is not intelligence
Chollet surveys a century of attempts to define intelligence in psychology and AI and finds two camps: intelligence as a **collection of skills**, and intelligence as **general learning ability**. AI research had followed the first, building benchmarks such as chess, Go, Atari and ImageNet, where a system can reach superhuman skill through massive training or hand-engineering without becoming more general. [AlphaZero](../102-alphazero/summary.md)'s Go play is extraordinary skill, but it says little about acquiring an unfamiliar skill quickly.

### A spectrum of generalization

```
  Local generalization     handle new inputs from a known task      (robustness)
        |                  e.g. a classifier on new photos
        v
  Broad generalization     handle new tasks within a domain         (flexibility)
        |                  e.g. a self-driving car in a new city
        v
  Extreme generalization   handle entirely new tasks sharing only   (generality)
                           abstract structure with past experience
```

Humans show extreme generalization. Most machine learning, then and now, is very good at local generalization. Chollet argues that progress toward general AI must be measured at the broad and extreme end.

### Control for priors and experience
A fair comparison between humans and machines must hold constant what each is given. Chollet grounds ARC in the **Core Knowledge** priors from developmental psychology, which humans have from infancy:

- **Objectness and elementary physics** (objects cohere, persist, and interact by contact)
- **Agentness and goal-directedness**
- **Natural numbers and elementary arithmetic** (small quantities, counting, comparison)
- **Elementary geometry and topology** (distance, orientation, inside and outside)

ARC tasks assume only these, and no language, cultural or learned knowledge.

---

## The Benchmark: ARC

```
  One ARC task (illustrative)

  Demonstration pairs (usually ~3):
    input grid  ->  output grid
    input grid  ->  output grid
    input grid  ->  output grid

  Test:
    input grid  ->  ???      the solver must construct the output
                            (size, colours and every cell) from scratch

  Grids: 1x1 to 30x30, 10 colours
  Up to 3 attempts per test input; exact match only
```

- **Every task is a new rule.** Fill enclosed regions, extend a pattern, count objects and draw a bar of that length, mirror a shape. The solver must infer the rule from about three examples.
- **Sizes (2019):** 400 training tasks and 600 evaluation tasks, split into 400 public and 200 private. By the ARC Prize era, ARC-AGI-1 was described as 400 public training, 400 public evaluation, 100 semi-private and 100 private evaluation tasks.
- **The private set is the point.** Neither the solver nor its developer should have seen the test tasks ("developer-aware generalization"), which a hidden set enforces.
- **Hard to game with data.** Tasks are deliberately unlike one another and hard to generate synthetically in bulk, so memorising or scaling training data should not help.

---

## Key Claims and How They Held Up

### "Current deep learning cannot solve ARC"
For about five years this held. A 2020 Kaggle competition's top score was 20 percent, and by early 2024 the private-set state of the art had only reached 33 percent. Large language models, despite huge gains elsewhere, did poorly when given the grids as text.

### The ARC Prize (2024)
Chollet and Mike Knoop launched the ARC Prize on 11 June 2024, with a grand prize for the first team to reach 85 percent on the private set under compute limits and with open-sourced code. By the end of the 2024 competition, according to the [ARC Prize 2024 Technical Report](https://arxiv.org/abs/2412.04604):

- The private-set state of the art rose **from 33 percent to 55.5 percent**.
- "The ARChitects" won first place at **53.5 percent**. MindsAI scored 55.5 percent but was not eligible for prizes because it chose not to open-source its solution.
- The winning approaches were **deep-learning-guided program synthesis** and **test-time training** (fine-tuning a model on each task's demonstration pairs at inference time), and combinations of the two.
- The report also flagged weaknesses: the private set had been unchanged since 2019, and an ensemble of 2020 competition entries had solved 49 percent of it by brute-force search, so it was more vulnerable than hoped.

### OpenAI o3 (December 2024)
On 20 December 2024 the ARC Prize team [reported](https://arcprize.org/blog/oai-o3-pub-breakthrough) results for an OpenAI o3 preview system on ARC-AGI-1:

| Configuration | Semi-private eval | Public eval | Cost per task |
|---|---|---|---|
| High-efficiency (within the $10k public leaderboard limit) | **75.7%** | 82.8% | about $26 |
| Low-efficiency (172x the compute) | **87.5%** | 91.5% | about $4,560 |

Chollet called it "a surprising and important step-function increase in AI capabilities" but added: "I don't think o3 is AGI yet. o3 still fails on some very easy tasks, indicating fundamental differences with human intelligence." The result is widely treated as a milestone for [test-time compute](../50-test-time-compute/summary.md) and [reasoning models](../../language-models/31-openai-o1/summary.md). One caveat: the publicly released o3 of April 2025 was a different configuration, and the ARC Prize team [reported](https://arcprize.org/blog/analyzing-o3-with-arc-agi) it at 41 percent (low) and 53 percent (medium) on ARC-AGI-1.

### ARC-AGI-2 (March 2025)
ARC-AGI-2 was [announced](https://arcprize.org/blog/announcing-arc-agi-2-and-arc-prize-2025) on 24 March 2025 and described in a [May 2025 paper](https://arxiv.org/abs/2505.11831) (Chollet, Knoop, Kamradt, Landers, Pinkard). It keeps the grid format but targets what reasoning systems still found hard: **symbolic interpretation** (giving symbols meaning beyond their visual pattern), **compositional reasoning** (several interacting rules at once) and **contextual rule application** (applying a rule differently depending on context).

- Every task was solved by at least two people in at most two attempts in testing with over 400 members of the public; the human panel averaged 60 percent.
- At launch, pure LLMs scored 0 percent, and the o3 preview in its low-compute setting scored about 4 percent.
- **ARC Prize 2025** offered a $700,000 grand prize for 85 percent. According to the organisers' [results post](https://arcprize.org/blog/arc-prize-2025-results-analysis) (5 December 2025), the top Kaggle entry (NVARC) reached **24 percent** on the private set at about $0.20 per task. Among verified frontier systems the post lists Claude Opus 4.5 (Thinking, 64k) at **37.6 percent** for $2.20 per task, and a Poetiq refinement system on Gemini 3 Pro at **54 percent** for $31 per task. The grand prize went unclaimed.

### ARC-AGI-3 (2026)
ARC Prize 2026 adds **ARC-AGI-3**, an interactive benchmark of game-like environments that tests exploration, planning, memory and goal acquisition rather than static puzzles. The competition opened on 25 March 2026 with deadlines in November 2026. Results were not in as of this writing.

---

## Why This Was Revolutionary

- **Reframed the goal.** "Skill-acquisition efficiency" gave researchers a precise alternative to "does it beat humans at X".
- **Built a benchmark that survived the scaling era.** Few benchmarks from 2019 were still unsaturated in 2024; ARC was.
- **Made efficiency part of the score.** ARC Prize leaderboards report cost per task alongside accuracy, a practice other benchmarks are starting to copy.
- **Pushed research into new methods.** Test-time training and program synthesis guided by neural networks got serious attention largely because of ARC.

---

## Criticisms

- **Visual puzzles in a text model's clothing.** LLMs receive grids as strings of numbers, a representation humans would also find hard. Some argue poor early LLM scores measured perception and format as much as reasoning.
- **Is it measuring intelligence or a narrow puzzle skill?** ARC tests one family of abstract grid transformations. Doing well does not obviously mean general intelligence, and Chollet himself says solving ARC would not mean AGI.
- **The formal definition is hard to compute.** The algorithmic-information-theory definition in the paper cannot practically be calculated for real systems; ARC is a practical stand-in for it, not an implementation.
- **Compute can still buy skill.** o3's high-compute score used 172 times the compute of its efficient setting, which is partly why cost per task is now reported alongside scores.
- **Small, ageing hidden sets.** The ARC Prize team found that the 2019 private set was more vulnerable to brute-force search than intended, one reason ARC-AGI-2 exists.

---

## Key Takeaways for Practitioners

1. **Always ask which ARC.** ARC-AGI-1 and ARC-AGI-2 scores are not comparable, and public, semi-private and private sets differ too.
2. **Read cost per task alongside accuracy.** A score bought with thousands of dollars of inference per puzzle means something different from a cheap one.
3. **Test-time adaptation works.** Fine-tuning on a task's own examples at inference, or iterating a candidate program against them, is a general technique worth borrowing for few-example problems.
4. **Benchmark design lesson:** hold priors and experience fixed, hide the test set, and make tasks hard to generate in bulk if you want a benchmark that resists memorisation.

---

## Limitations & Future Directions

- **Narrow modality.** Small 2D grids with 10 colours; no language, no real-world perception.
- **Static tasks.** ARC-AGI-1 and 2 do not test acting in an environment, which ARC-AGI-3 aims to address.
- **Human baselines vary** with who is tested and how; the ARC-AGI-2 study was the first large calibrated one.
- **Open question as of 2026:** whether ARC-AGI-2 falls to frontier models plus refinement loops the way ARC-AGI-1 did, and whether that says anything general about intelligence.

---

## Further Reading

- **Original Paper:** [arxiv.org/abs/1911.01547](https://arxiv.org/abs/1911.01547)
- **ARC data:** [github.com/fchollet/ARC](https://github.com/fchollet/ARC)
- **ARC Prize 2024 Technical Report:** [arxiv.org/abs/2412.04604](https://arxiv.org/abs/2412.04604)
- **o3 on ARC-AGI (December 2024):** [arcprize.org/blog/oai-o3-pub-breakthrough](https://arcprize.org/blog/oai-o3-pub-breakthrough)
- **ARC-AGI-2 paper:** [arxiv.org/abs/2505.11831](https://arxiv.org/abs/2505.11831)
- **ARC Prize 2025 results:** [arcprize.org/blog/arc-prize-2025-results-analysis](https://arcprize.org/blog/arc-prize-2025-results-analysis)
- **In this collection:** [OpenAI o1](../../language-models/31-openai-o1/summary.md), [Test-Time Compute](../50-test-time-compute/summary.md), [AlphaZero](../102-alphazero/summary.md), [The Bitter Lesson](../../essays/111-bitter-lesson/summary.md), [Computing Machinery and Intelligence](../../essays/108-computing-machinery-and-intelligence/summary.md), [MMLU](../137-mmlu/summary.md)

## Citation

```bibtex
@article{chollet2019measure,
  title={On the Measure of Intelligence},
  author={Chollet, Fran{\c{c}}ois},
  journal={arXiv preprint arXiv:1911.01547},
  year={2019}
}
```

<!-- related:start -->

---

## Related in This Collection

- [OpenAI o1: Learning to Reason with Reinforcement Learning](../../language-models/31-openai-o1/summary.md)
- [GPT-4 Technical Report](../../language-models/36-gpt4/summary.md)
- [Gemini 3: Google's Most Capable AI Model](../../multimodal/47-gemini3/summary.md)
- [Scaling LLM Test-Time Compute: The Theoretical Foundation for Reasoning Models](../../techniques/50-test-time-compute/summary.md)
- [Language Models are Unsupervised Multitask Learners (GPT-2)](../../language-models/64-gpt2/summary.md)
- [Mastering Chess and Shogi by Self-Play with a General Reinforcement Learning Algorithm](../../techniques/102-alphazero/summary.md)
- [Computing Machinery and Intelligence (The Imitation Game / Turing Test)](../../essays/108-computing-machinery-and-intelligence/summary.md)
- [The Bitter Lesson](../../essays/111-bitter-lesson/summary.md)

<!-- related:end -->
