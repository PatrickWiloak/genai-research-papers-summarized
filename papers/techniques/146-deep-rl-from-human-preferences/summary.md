---
title: "Deep Reinforcement Learning from Human Preferences"
slug: "146-deep-rl-from-human-preferences"
number: 146
category: "techniques"
authors: "Paul Christiano, Jan Leike, Tom B. Brown, Miljan Martic, Shane Legg, Dario Amodei (OpenAI, DeepMind)"
published: "June 2017 (arXiv v1, 12 June 2017; NIPS 2017)"
year: 2017
url: "https://arxiv.org/abs/1706.03741"
tags: ["alignment", "reinforcement-learning", "rlhf"]
---

# Deep Reinforcement Learning from Human Preferences

**Authors:** Paul Christiano, Jan Leike, Tom B. Brown, Miljan Martic, Shane Legg, Dario Amodei (OpenAI, DeepMind)
**Published:** June 2017 (arXiv v1, 12 June 2017; NIPS 2017)
**Paper Link:** [arxiv.org/abs/1706.03741](https://arxiv.org/abs/1706.03741)

---

## Why This Paper Matters

Reinforcement learning (RL) needs a reward function: a number that says how well the agent just did. For a game with a score, that is easy. For most things people actually care about ("be helpful", "drive politely", "do a nice backflip") nobody can write the formula. Hand-written rewards also get exploited: the agent finds a loophole that scores high while missing the point, a failure called **reward hacking**.

This paper's answer: **do not write the reward, learn it from human comparisons.** Show a person two short clips of the agent behaving and ask "which is better?". Train a model to predict those choices, then use the learned model as the reward for ordinary deep RL. People are far better at judging "this looks closer to a backflip" than at specifying a backflip mathematically.

It is the origin of what is now called **RLHF** (reinforcement learning from human feedback). Everything that makes chat assistants follow instructions politely descends from this recipe: collect comparisons, fit a reward model, optimize against it with RL. The 2017 paper did it on Atari games and simulated robots; [InstructGPT](../../language-models/05-instructgpt-rlhf/summary.md) did the same with language models five years later.

The analogy: instead of giving a dog trainer a rulebook, you let the trainer say "better" or "worse" between two attempts, and the dog gradually works out what you want.

---

## The Core Innovation

Three processes run at the same time, asynchronously:

```
        +----------------------+
        |  Policy (the agent)  |  acts in the environment,
        |  trained with RL     |  gets reward from the predictor
        +----------+-----------+
                   | trajectory segments (1-2 s clips)
                   v
        +----------------------+
        |  Human (or oracle)   |  "left clip is better"
        +----------+-----------+
                   | preference pairs
                   v
        +----------------------+
        |  Reward predictor    |  supervised learning on
        |  (small ensemble)    |  the human comparisons
        +----------------------+
```

The agent never sees the true reward. It only ever optimizes the learned one, and the human keeps supplying fresh comparisons as the agent's behaviour changes.

---

## Key Components Explained

### 1. Preferences over short trajectory segments
**What it does:** Makes human feedback cheap and consistent.
**How it works:** Each query shows two trajectory segments of 1 to 2 seconds. The human picks the better one (ties and "cannot tell" are also allowed). The authors found comparisons much easier for people to give consistently than absolute scores, especially for continuous control and the qualitative tasks.

### 2. The reward predictor
**What it does:** Turns comparisons into a reward signal.
**How it works:** It follows the Bradley-Terry model of pairwise choice: the probability that you prefer segment A over B is a function of the summed predicted reward over each segment.

```
P(A preferred over B) = exp(sum r(A)) / (exp(sum r(A)) + exp(sum r(B)))

Train r by cross-entropy against the human's actual choices.
```

An **ensemble** of predictors (3 by default) is trained on resampled data, and each is normalized. Regularization (L2 and dropout) is used to limit overfitting.

### 3. Query selection
**What it does:** Spends scarce human labels where they teach the most.
**How it works:** Sample many pairs of segments and ask about the ones where the ensemble members disagree most. This is a crude approximation of "uncertainty" sampling and the ablations show it helps.

### 4. Ordinary deep RL on the learned reward
**What it does:** Optimizes the behaviour.
**How it works:** The policy is trained with off-the-shelf algorithms: A2C (a synchronous variant of A3C) for Atari and TRPO for MuJoCo robotics. The paper predates [PPO](../63-ppo/summary.md), the algorithm that later became the default optimizer in RLHF pipelines.

---

## Key Results

- **Feedback efficiency:** the abstract reports human feedback on **under 1%** of the agent's interactions with the environment, which the authors describe as cutting the interaction complexity by roughly 3 orders of magnitude relative to labelling everything.
- **Simulated robotics (MuJoCo):** with 700 queries to a human rater, the method nearly matched RL on the true reward on most tasks. By 1,400 synthetic labels it was slightly better than RL given the true reward on some tasks, likely because the learned reward is better shaped.
- **Atari:** with 5,500 queries, results varied by game. BeamRider and Pong matched or came close to RL with synthetic labels; SpaceInvaders and Breakout never matched. On Qbert the method failed to beat the first level with real human feedback, possibly because short clips are confusing there.
- **Enduro:** human labellers beat the A3C baseline, since they reward any progress towards passing cars, effectively shaping the reward.
- **Time cost:** contractors answered an average query in 3 to 5 seconds, so real-feedback experiments took between 30 minutes and 5 hours of human time.
- **Novel behaviours (no reward function exists):**
  - A simulated Hopper doing consecutive backflips: about 900 queries, in under an hour.
  - Half-cheetah moving on one leg: about 800 queries in under an hour.
  - Staying alongside other cars in Enduro: roughly 1,300 queries and 4 million frames.
  These were trained with feedback from the authors themselves.
- **Cost:** the appendix estimates about $25 of compute and, for 5,000 labels, about 5 hours of human labour (roughly $36 at US minimum wage).
- **Ablations:** removing online query collection hurt badly. Offline reward training produced bizarre behaviour, e.g. on Pong the agent sometimes learned to avoid losing points without scoring, leading to endless volleys. Lesson: **human feedback must be interleaved with RL, not collected once.**

---

## Why This Was Revolutionary

- It showed preference learning **scales to deep RL** with neural networks and non-expert labellers, which the authors call the first evidence this can be economically scaled to state-of-the-art RL systems.
- It established the **reward model** as a reusable component.
- It made the **safety framing** concrete: specifying what we want is the hard part, and learning it from people is a path to aligned systems.
- It involved authors who went on to lead alignment work at OpenAI, DeepMind and Anthropic.

---

## Real-World Impact

- **Language-model RLHF.** The recipe was applied to text in follow-up work and then to instruction following in [InstructGPT](../../language-models/05-instructgpt-rlhf/summary.md), the direct ancestor of ChatGPT-style assistants.
- **Optimizer choice.** Later RLHF pipelines swapped the 2017 RL algorithms for [PPO](../63-ppo/summary.md), and more recently for critic-free variants such as [GRPO](../38-grpo/summary.md).
- **Skipping the reward model.** [DPO](../../language-models/19-dpo/summary.md) showed you can train directly on preference pairs without an explicit reward model or RL loop.
- **Verifiable rewards.** For tasks with checkable answers, [RLVR](../39-rlvr/summary.md) replaces human comparisons with automatic checks, a different answer to the same "where does the reward come from?" question.

---

## Key Takeaways for Practitioners

1. **Comparisons beat scores.** Humans are more consistent saying "A is better than B" than rating on a scale.
2. **Keep the loop online.** A reward model trained once can be exploited; keep gathering labels as the policy shifts.
3. **Use uncertainty to pick queries.** Ensemble disagreement is a cheap way to spend labels well.
4. **Expect reward hacking.** A learned reward is a proxy; the optimizer will find its weak spots.
5. **Keep segments short and clear.** Clips that are hard to judge (as in Qbert) give noisy labels.

---

## Limitations & Future Directions

- **Proxy rewards are exploitable.** The paper shows partial rewards leading to odd behaviour; this problem grew with scale and is why modern RLHF uses KL penalties and careful reward-model refreshes.
- **Human error and inconsistency.** Real feedback was typically slightly worse than synthetic oracle feedback, sometimes comparable to synthetic with about 40% fewer labels.
- **Narrow scope.** Short clips in simulated environments with 1 to 2 second judgments, not open-ended language or long horizons.
- **Labeller pipeline.** The authors note outsourcing labels could be improved; scaling this to language took years of additional work.
- **Venue:** the arXiv page lists no venue; NIPS 2017 is from general knowledge and not re-checked against the proceedings.

---

## Further Reading

- **Original Paper:** https://arxiv.org/abs/1706.03741
- **Follow-ups in this repo:** [InstructGPT](../../language-models/05-instructgpt-rlhf/summary.md), [PPO](../63-ppo/summary.md), [DPO](../../language-models/19-dpo/summary.md), [GRPO](../38-grpo/summary.md), [RLVR](../39-rlvr/summary.md)

## Citation

```bibtex
@inproceedings{christiano2017deep,
  title={Deep Reinforcement Learning from Human Preferences},
  author={Christiano, Paul and Leike, Jan and Brown, Tom B. and Martic, Miljan and Legg, Shane and Amodei, Dario},
  booktitle={Advances in Neural Information Processing Systems},
  year={2017}
}
```

<!-- related:start -->

---

## Related in This Collection

- [Training Language Models to Follow Instructions with Human Feedback (InstructGPT)](../../language-models/05-instructgpt-rlhf/summary.md)
- [Direct Preference Optimization (DPO): Your Language Model is Secretly a Reward Model](../../language-models/19-dpo/summary.md)
- [GRPO: Group Relative Policy Optimization](../../techniques/38-grpo/summary.md)
- [RLVR: Reinforcement Learning from Verifiable Rewards](../../techniques/39-rlvr/summary.md)
- [Proximal Policy Optimization Algorithms (PPO)](../../techniques/63-ppo/summary.md)

<!-- related:end -->
