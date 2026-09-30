# Do LLMs Reason?

**In one line:** Whether language models "really reason" depends heavily on what you mean by reasoning; the evidence shows real multi-step internal computation and striking problem-solving, alongside brittleness that looks like pattern-matching - and both camps can point to strong results.
**Last reviewed:** 2026-09-30

---

## The short version

- **Much of the disagreement is about definitions.** "Reasoning" can mean producing correct conclusions from premises, doing so by a systematic procedure, doing so robustly on unfamiliar problems, or doing so the way humans do. Models meet some of these bars far better than others.
- **Evidence for:** models solve competition math at gold-medal level, pass hard new benchmarks, and interpretability research has traced genuine intermediate steps inside them (for example, internally working out "Texas" on the way from "Dallas" to "Austin").
- **Evidence against:** accuracy drops when irrelevant details or new numbers are added to familiar problems (GSM-Symbolic, 2024), collapses on scaled-up puzzles (The Illusion of Thinking, 2025), and stays far below humans on tasks designed to require novel abstraction (ARC).
- **The critiques have themselves been critiqued.** The Illusion of Thinking was challenged within days for counting unsolvable puzzles and output-length limits as reasoning failures.
- **The visible "chain of thought" is not a reliable window.** Research shows models' written reasoning often omits what actually drove their answer.

## What the words mean

| Term | Plain meaning | Why it matters here |
|---|---|---|
| Reasoning (functional) | Reaching correct conclusions on problems that require combining several steps | Easy to test; models increasingly pass |
| Systematic / algorithmic reasoning | Applying a general procedure that works for any input size | Tested by scaling problems up; models often degrade |
| Generalisation / novelty | Solving problems unlike anything seen in training | Hard to test because training data is huge and opaque |
| Chain of thought (CoT) | Text a model writes before its answer, "thinking out loud" | Improves accuracy, but may not reflect the internal process |
| Reasoning model | A model trained (usually with RL) to produce long chains of thought before answering, such as o1 or DeepSeek-R1 | Most 2025-2026 debate is about these |
| Memorisation / pattern-matching | Reproducing solutions or templates seen in training | The main alternative explanation for success |

A useful framing: nobody disputes that models produce correct answers on many reasoning tasks. The dispute is about **how**, and therefore about **how far it will generalise**.

```
         "It reasons"                                  "It pattern-matches"
  <-----------------------------------------------------------------------------> 
  general procedures,                                   retrieval and interpolation
  robust to surface changes,                            of training examples,
  extends to bigger inputs                              fragile to surface changes

  Most evidence places current models somewhere in between,
  and in different places for different tasks.
```

## The case that they do reason

**1. Chain of thought and reasoning models unlocked real problem-solving.** [Chain-of-Thought Prompting](../../papers/techniques/09-chain-of-thought/summary.md) (2022) showed that asking for intermediate steps sharply improved multi-step math. Reasoning models trained with reinforcement learning, [OpenAI o1](../../papers/language-models/31-openai-o1/summary.md) (2024) and [DeepSeek-R1](../../papers/language-models/26-deepseek-r1/summary.md) (2025), went much further, and R1's training showed behaviours like self-checking and backtracking emerging from reward alone. See [Reasoning models](../concepts/reasoning-models.md).

**2. Performance on hard, new problems.** In July 2025 an advanced Gemini Deep Think model reached gold-medal standard at the International Mathematical Olympiad, solving 5 of 6 problems (35 of 42 points) in natural language within the 4.5-hour limit; the IMO president described the solutions as "clear, precise and most of them easy to follow". IMO problems are newly written each year, which limits the memorisation explanation. On ARC-AGI-1, a benchmark built specifically to resist memorisation, OpenAI's o3 scored 75.7% (within the compute budget) and 87.5% (with far more compute) in December 2024.

**3. Interpretability finds intermediate steps.** Anthropic's "On the Biology of a Large Language Model" (March 2025) traced the internal computations of Claude 3.5 Haiku. Asked for the capital of the state containing Dallas, the model internally represents "Texas" and then uses it to reach "Austin", rather than jumping straight to an answer. When writing rhyming poetry it selects the rhyme word before writing the line, a form of planning. Earlier work on [induction heads](../../papers/techniques/126-induction-heads/summary.md) (2022) identified concrete circuits that implement in-context pattern completion, and [sparse autoencoders](../../papers/techniques/82-sparse-autoencoders/summary.md) found abstract, interpretable features. These are mechanisms, not just outputs.

**4. The "just statistics" argument proves too much.** Defenders note that being trained on next-token prediction does not limit what the model learns internally, just as evolution optimising for reproduction does not limit what brains compute. The right question is what algorithms the network has learned, which interpretability can partly answer.

## The case that they do not (or not robustly)

**1. Fragility to irrelevant changes: GSM-Symbolic.** Apple researchers (Mirzadeh et al., October 2024) rebuilt the grade-school math benchmark GSM8K as templates, so names and numbers could be varied. Accuracy varied noticeably across versions of the same problem, and adding a single irrelevant but plausible-sounding clause ("GSM-NoOp") caused drops of up to 65% across state-of-the-art models. The authors hypothesised that current models "cannot perform genuine logical reasoning; they replicate reasoning steps from their training data".

**2. Collapse at higher complexity: The Illusion of Thinking.** Shojaee et al. (Apple, June 2025; NeurIPS 2025) tested reasoning models on puzzles like Tower of Hanoi and River Crossing where difficulty can be dialled up precisely. They found three regimes: on easy problems, standard models did as well or better; at medium difficulty, reasoning models helped; at high difficulty, both collapsed to near-zero accuracy. Strikingly, reasoning effort (tokens spent thinking) rose with difficulty and then *fell* before the collapse, even with budget left, and giving the model the solution algorithm did not help.

**3. Sensitivity to how likely the answer text is.** "Embers of Autoregression" (McCoy et al., 2023) found GPT-4 decoded a simple cipher correctly 51% of the time when the answer was a high-probability sentence but 13% when it was low-probability, although the procedure is identical. A system executing an algorithm should not care. "Faith and Fate" (Dziri et al., 2023) found accuracy on multi-digit multiplication and similar compositional tasks decays rapidly with size, consistent with matching sub-patterns rather than running a procedure.

**4. ARC: novel abstraction remains hard.** François Chollet's [On the Measure of Intelligence](../../papers/techniques/138-arc-agi/summary.md) (2019) defined intelligence as skill-acquisition efficiency on novel tasks and introduced ARC, grid puzzles easy for people and hard for pattern-matchers. Chollet himself said o3's 2024 result did not mean AGI, that o3 still failed some easy tasks, and predicted the harder ARC-AGI-2 (2025) could reduce its score to under 30% even at high compute, "while a smart human would still be able to score over 95% with no training". The ARC Prize has since introduced ARC-AGI-3, an interactive benchmark of game-like environments testing how efficiently agents learn new goals compared with humans.

**5. Written reasoning is not the real reasoning.** Anthropic's April 2025 study planted hints in prompts and checked whether reasoning models admitted using them. Claude 3.7 Sonnet mentioned the hint 25% of the time on average and DeepSeek R1 39%. The same interpretability work that found the Dallas-Texas-Austin chain found that when Claude 3.5 Haiku explained its arithmetic, it described the schoolbook carrying method while internally using a quite different parallel approximation strategy. So a fluent chain of thought is weak evidence of how an answer was reached.

## The rebuttals to the rebuttals

The Illusion of Thinking drew a quick reply. "Comment on The Illusion of Thinking" (A. Lawsen, June 2025) argued that:

- the Tower of Hanoi failures largely coincided with the models' **output token limits**; models sometimes said explicitly they were stopping because the move list was too long;
- the River Crossing instances with more than five pairs were **mathematically impossible** with the given boat capacity, so models were scored as failing unsolvable problems;
- when asked to output a program that generates the solution rather than every move, models solved instances previously reported as failures.

Those who find the original paper persuasive reply that the Comment does not explain every result (such as reasoning effort falling as puzzles get harder), and that a genuine reasoner should recognise and say when a puzzle is impossible rather than produce a wrong answer. The exchange illustrates a general point: evaluating reasoning is itself hard, and conclusions often hinge on scoring details.

Similar arguments surround GSM-Symbolic. Critics note that humans are also distracted by irrelevant information in word problems; supporters reply that the drop exists at all on trivial changes, and that "humans do it too" does not show the model is doing what humans do when they succeed.

## Where the debate stands

| Question | State of evidence (September 2026) |
|---|---|
| Do models produce correct multi-step solutions to new, hard problems? | Yes, demonstrably (IMO 2025, frontier math and code benchmarks) |
| Do they compute intermediate steps internally? | Yes, in traced cases (Anthropic interpretability, 2025) |
| Is their reasoning robust to irrelevant surface changes? | Not fully; degradation documented |
| Do they execute general algorithms at any scale? | Often not; performance degrades with problem size |
| Do they match human efficiency at learning truly novel tasks? | No, by ARC's measures |
| Does visible chain of thought faithfully show their process? | Often not |

A defensible neutral summary: current models perform real, non-trivial computation that deserves the name reasoning in the functional sense, and that computation is uneven, less systematic than a symbolic algorithm, and hard to observe from their written explanations. Whether the gaps close with more scale and training, or reflect a deeper limit of the approach, is the actual open question. See [Is scaling hitting a wall?](./scaling-limits.md)

## What to watch

- **ARC-AGI-2 and ARC-AGI-3 scores**, especially at low cost per task, since Chollet's definition rewards efficiency.
- **Robustness benchmarks** that perturb problems (GSM-Symbolic-style), and whether reasoning models close the gap.
- **Interpretability of reasoning models**: whether circuit tracing scales to long chains of thought.
- **Chain-of-thought monitoring:** whether training can make written reasoning more faithful, which matters for safety as well as for this debate.
- **Contamination checks** on headline results; see [Contamination and saturation](../benchmarks/contamination-and-saturation.md).

## Read next

- [Chain-of-Thought Prompting](../../papers/techniques/09-chain-of-thought/summary.md), [OpenAI o1](../../papers/language-models/31-openai-o1/summary.md), [DeepSeek-R1](../../papers/language-models/26-deepseek-r1/summary.md)
- [In-context Learning and Induction Heads](../../papers/techniques/126-induction-heads/summary.md), [Sparse Autoencoders](../../papers/techniques/82-sparse-autoencoders/summary.md)
- [On the Measure of Intelligence (ARC)](../../papers/techniques/138-arc-agi/summary.md), [Computing Machinery and Intelligence](../../papers/essays/108-computing-machinery-and-intelligence/summary.md)
- [Emergent Abilities (and the Mirage Rebuttal)](../../papers/techniques/81-emergent-abilities/summary.md)
- [Reasoning models](../concepts/reasoning-models.md), [Knowledge and reasoning benchmarks](../benchmarks/knowledge-and-reasoning.md), [Math and code benchmarks](../benchmarks/math-and-code.md)

## Sources

- Mirzadeh et al., "GSM-Symbolic: Understanding the Limitations of Mathematical Reasoning in Large Language Models" (arXiv 2410.05229, Oct 2024): https://arxiv.org/abs/2410.05229
- Shojaee et al., "The Illusion of Thinking" (arXiv 2506.06941, Jun 2025; NeurIPS 2025): https://arxiv.org/abs/2506.06941
- A. Lawsen, "Comment on The Illusion of Thinking" (arXiv 2506.09250, Jun 2025): https://arxiv.org/abs/2506.09250
- McCoy et al., "Embers of Autoregression" (arXiv 2309.13638, Sep 2023): https://arxiv.org/abs/2309.13638
- Dziri et al., "Faith and Fate: Limits of Transformers on Compositionality" (arXiv 2305.18654, 2023): https://arxiv.org/abs/2305.18654
- Anthropic, "On the Biology of a Large Language Model" (27 Mar 2025): https://transformer-circuits.pub/2025/attribution-graphs/biology.html
- Anthropic, "Reasoning models don't always say what they think" (3 Apr 2025): https://www.anthropic.com/research/reasoning-models-dont-say-think
- ARC Prize, "OpenAI o3 Breakthrough High Score on ARC-AGI-Pub" (20 Dec 2024): https://arcprize.org/blog/oai-o3-pub-breakthrough ; ARC-AGI-2: https://arcprize.org/arc-agi/2/ ; ARC-AGI-3: https://arcprize.org/arc-agi/3/
- Google DeepMind, IMO 2025 gold-medal announcement (July 2025): https://deepmind.google/discover/blog/advanced-version-of-gemini-with-deep-think-officially-achieves-gold-medal-standard-at-the-international-mathematical-olympiad/
