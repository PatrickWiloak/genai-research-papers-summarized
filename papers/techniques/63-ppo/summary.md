---
title: "Proximal Policy Optimization Algorithms (PPO)"
slug: "63-ppo"
number: 63
category: "techniques"
authors: "John Schulman, Filip Wolski, Prafulla Dhariwal, Alec Radford, Oleg Klimov (OpenAI)"
published: "July 2017"
year: 2017
url: "https://arxiv.org/abs/1707.06347"
tags: ["reinforcement-learning", "alignment"]
---

# Proximal Policy Optimization Algorithms (PPO)

**Authors:** John Schulman, Filip Wolski, Prafulla Dhariwal, Alec Radford, Oleg Klimov (OpenAI)

**Published:** July 2017

**Paper Link:** https://arxiv.org/abs/1707.06347

---

## Why This Paper Matters

PPO is the reinforcement learning algorithm that makes ChatGPT behave like ChatGPT. When OpenAI trained InstructGPT - and later ChatGPT - to be helpful, harmless, and honest, the final fine-tuning step used PPO to optimize the model against a learned reward signal. Every major aligned LLM of the 2022-2024 era relied on it.

The paper itself is a masterclass in pragmatic ML engineering: it took a powerful-but-painful predecessor (TRPO) and made it simpler, faster, and almost as reliable - by replacing a constrained optimization problem with a single clipped objective function. The result became the default RL algorithm for the LLM alignment era.

---

## A Minimal RL Primer

If you know LLMs but not RL, here is all the vocabulary you need:

- **Agent** - the model being trained (the LLM)
- **Environment** - the context the agent acts in (a prompt)
- **Action** - what the agent does (generates tokens, forming a response)
- **Reward** - a scalar score for how good the action was (from a reward model, or from a rule like "did the code pass tests?")
- **Policy** - the agent's strategy: given a prompt, what is the probability of each possible response? The policy is the LLM's weights.
- **Goal** - update the policy so expected reward goes up, without breaking everything else

The challenge: gradient steps in RL can be large and unpredictable. One bad update can collapse a policy that took days to train.

---

## The Problem PPO Solves

### Why Policy Gradient Methods Are Fragile

In standard policy gradient training, you sample a response, observe a reward, and nudge the model's weights toward responses that scored well. The problem is **step size**: too small a step and training is painfully slow; too large a step and you can accidentally destroy the policy - the model swings to some degenerate output distribution that is hard to recover from.

### TRPO: Correct but Complex

The predecessor, **Trust Region Policy Optimization (TRPO, 2015)**, solved this with a hard constraint: the new policy must stay within a "trust region" measured by KL divergence from the old policy. It works well, but the implementation requires computing second-order derivatives (a conjugate gradient solve and a line search), which is expensive and tricky to implement correctly in large models.

```
TRPO: maximize expected reward
      subject to KL(old_policy, new_policy) <= delta
      -> Correct, but complex second-order optimization
```

### PPO's Answer

PPO achieves a similar effect with a **clipped first-order objective** - no second-order math required, no constrained optimization solver, no line search. Just a modified loss function you can drop into any standard optimizer.

---

## Core Innovation: The Clipped Surrogate Objective

### The Probability Ratio

The key quantity is the ratio of the new policy's probability for an action to the old policy's probability:

```
r_t(theta) = pi_theta(a_t | s_t) / pi_theta_old(a_t | s_t)
```

- If `r > 1`: the new policy assigns higher probability to this action than the old one did
- If `r < 1`: the new policy assigns lower probability to this action

### The Clipped Objective

PPO clips this ratio so it cannot move too far from 1.0 in either direction:

```
L_CLIP(theta) = E_t [ min(
    r_t(theta) * A_t,
    clip(r_t(theta), 1 - epsilon, 1 + epsilon) * A_t
) ]

Where:
  A_t       = advantage estimate (how much better/worse was this action
               compared to what was expected?)
  epsilon   ~ 0.2  (the clip range, a hyperparameter)
```

**Intuition with an analogy:** Imagine you are adjusting an employee's bonus based on performance. If they did well (positive advantage), you want to reward them - but you cap the raise so one lucky day does not wildly distort their compensation. If they did poorly (negative advantage), you want to reduce their reward - but again with a floor, so one bad day does not destroy them. The clip is that cap-and-floor.

```
Case 1: Action was GOOD (A_t > 0)
  - We want to increase probability of this action
  - But clip prevents r_t from exceeding 1 + epsilon
  - Stops us from making huge updates because one rollout went well

Case 2: Action was BAD (A_t < 0)
  - We want to decrease probability of this action
  - But clip prevents r_t from going below 1 - epsilon
  - Stops us from collapsing the policy after one bad sample
```

The `min` in the objective ensures we always take the more conservative (lower) estimate - which means the clipping is always active as a ceiling, not just in favorable cases.

### Visualized

```
Policy update size (ratio r_t):

        No benefit from going further right
                          |
  [--------------------[CLIP]
  0.8                  1.0                  1.2
  [CLIP]--------------------]
     |
  No benefit from going further left

epsilon = 0.2 means the policy can only shift ~20% in probability ratio
```

---

## Key Components

### 1. Advantage Estimation (GAE)

PPO uses **Generalized Advantage Estimation (GAE)** to compute `A_t` - a measure of how much better an action was than the "baseline" expected value. GAE blends two extreme estimators:

```
TD(0):        Low variance, high bias   (looks one step ahead)
Monte Carlo:  Low bias, high variance   (looks to the end of the episode)

GAE(lambda): weighted average controlled by lambda in [0, 1]
  lambda = 0  ->  TD(0)
  lambda = 1  ->  Monte Carlo
  lambda ~ 0.95 in practice (mostly Monte Carlo, slightly smoothed)
```

High-quality advantage estimates mean the policy updates in the right direction reliably.

### 2. Multiple Epochs per Rollout

A key efficiency gain: PPO reuses each batch of collected experience for **multiple gradient updates** (typically 3-10 epochs), rather than discarding it after one update. The clipping constraint is what makes this safe - it prevents any single epoch from moving the policy too far.

```
Collect rollout data
  -> Run K gradient epochs on the same data (clipping makes this safe)
  -> Collect new rollout data
  -> Repeat
```

TRPO could only safely do one update per rollout because it had no analogous mechanism to prevent over-updating.

### 3. On-Policy Actor-Critic Architecture

PPO is an **on-policy** algorithm: data is collected using the current policy, used to update the policy, then discarded. It uses an actor-critic setup:

```
Actor  - the policy network (which action to take)
Critic - the value network (how good is this state?)

Both often share early layers (parameter sharing) with separate output heads
```

The full training objective combines three terms:

```
L_total = L_CLIP          (policy improvement)
        - c1 * L_VF       (value function loss - trains the critic)
        + c2 * H          (entropy bonus - encourages exploration)
```

The entropy bonus `H` discourages the policy from collapsing to always picking one response, which would kill exploration.

---

## Key Results

PPO was evaluated on continuous control tasks (MuJoCo locomotion) and Atari games - the standard RL benchmarks in 2017.

**Continuous control (MuJoCo):**
- PPO matched or exceeded TRPO on most tasks
- Significantly outperformed simpler methods like A2C and REINFORCE
- Half-Cheetah, Hopper, Walker2D: competitive with or better than TRPO

**Atari games (49 games):**
- PPO outperformed A2C on the majority of games
- Competitive with ACER (a more complex method with replay buffers)

**Training efficiency:**
- Wall-clock time substantially lower than TRPO (no second-order solve)
- Simpler to implement correctly
- Stable across a wide range of hyperparameters

---

## Why It Was Revolutionary

### 1. Simplicity at Scale

TRPO required conjugate gradient descent and a line search. PPO needs only standard stochastic gradient descent with one modified loss function. That difference is the gap between "works in a research lab" and "ships in production."

### 2. Reliable Stability

The clip constraint acts as a soft trust region - keeping updates in a safe range without expensive enforcement. Policy collapse became rare.

### 3. Sample Efficiency from Replay

Multiple epochs per rollout meant PPO extracted more learning from each experience batch - critical when "environment" means calling an expensive inference endpoint or waiting for human feedback.

### 4. Became the Default

Within the RL community, PPO rapidly displaced more complex algorithms as the baseline to beat. Its simplicity meant fewer implementation bugs in research, and it transferred naturally to the LLM setting.

---

## Real-World Impact: PPO Powers RLHF

This is the connection that makes PPO essential reading for anyone in LLMs.

### The RLHF Pipeline

InstructGPT (OpenAI, 2022) - the direct ancestor of ChatGPT - uses a three-step process:

```
Step 1: Supervised Fine-Tuning (SFT)
  Human labelers write ideal responses; model fine-tunes on them.

Step 2: Reward Model Training
  Humans rank pairs of model outputs; a separate "reward model" (RM)
  learns to predict human preferences as a scalar score.

Step 3: RL Fine-Tuning with PPO
  The LLM (policy) generates responses.
  The reward model scores them.
  PPO updates the LLM to generate responses the RM scores higher.
  A KL penalty keeps the LLM from drifting too far from the SFT model.
```

PPO is the engine of Step 3. The LLM is the policy; the reward model is the environment; token generation is the action space. The clipped objective prevents the LLM from collapsing to reward-hacking behavior during the RL phase.

```
Objective in RLHF with PPO:
  maximize: reward_model(response) - beta * KL(policy || sft_model)

  beta ~ 0.02: keeps the model from drifting into nonsense
               that happens to score high with the reward model
```

### Connection to Papers in This Collection

**[InstructGPT/RLHF (paper 05)](../../language-models/05-instructgpt-rlhf/summary.md)** - PPO is the RL algorithm that makes RLHF work. The InstructGPT paper describes the full three-step pipeline; PPO is the mechanism underlying Step 3.

**[DPO (paper 19)](../../language-models/19-dpo/summary.md)** - Direct Preference Optimization (2023) sidesteps PPO entirely. It shows mathematically that the reward model is implicit in the policy, and you can optimize preferences directly with supervised learning - no RL loop needed. DPO became popular for open-source alignment precisely because it removes the complexity of running PPO against a reward model.

**[GRPO (paper 38)](../38-grpo/summary.md)** - Group Relative Policy Optimization (DeepSeek, 2024) is a PPO variant that eliminates the critic (value) network by using group-relative advantage estimation instead. It powers DeepSeek-R1's reasoning training. GRPO keeps PPO's clipped objective but replaces the expensive critic model with in-group statistics, reducing memory by roughly 50%.

### The Alignment Algorithm Family Tree

```
TRPO (2015) -- correct but complex second-order optimization
    |
    v
PPO (2017) -- same stability, first-order simplicity
    |
    +---> RLHF / InstructGPT (2022) -- PPO + reward model --> ChatGPT
    |
    +---> DPO (2023) -- removes the RL loop entirely
    |
    +---> GRPO (2024) -- removes the critic; powers DeepSeek-R1
```

---

## Key Takeaways

1. **The clip is the key insight** - constraining the probability ratio to `[1-epsilon, 1+epsilon]` prevents destructive updates without any second-order math
2. **On-policy with multi-epoch replay** - reusing rollouts safely is what makes PPO compute-efficient
3. **Advantage estimation matters** - GAE with lambda ~ 0.95 gives stable, low-variance training signal
4. **Simpler than TRPO, comparably reliable** - the paper's whole contribution is this tradeoff
5. **Foundation of LLM alignment** - understanding PPO is prerequisite knowledge for understanding how ChatGPT and every major RLHF-trained model was built

---

## Limitations and Future Directions

### Limitations

- **On-policy data inefficiency** - each batch of rollouts is discarded after a few epochs; off-policy algorithms can reuse data more aggressively
- **Reward hacking** - if the reward model has weaknesses, PPO will find and exploit them; the KL penalty helps but does not eliminate this
- **Credit assignment over long sequences** - assigning credit to individual tokens in a 2000-token response is noisy; the critic's value estimates get increasingly inaccurate for long-horizon tasks
- **Four models in memory** - for LLM RLHF, you need the policy, reference model, reward model, and critic simultaneously; expensive at scale
- **Sensitive to reward scaling** - requires careful hyperparameter tuning, especially the KL penalty coefficient

### What Came After

- **DPO (2023)** - eliminates the RL loop entirely; works for offline preference datasets
- **GRPO (2024)** - eliminates the critic model using group baselines; scales better for reasoning tasks
- **RLVR** - using verifiable rewards (correct/incorrect on math, code) instead of learned reward models; powers the reasoning model generation (DeepSeek-R1, QwQ)
- **Process Reward Models** - giving per-step rewards rather than a single terminal reward; addresses credit assignment for long reasoning chains

---

## Further Reading

- **Original Paper:** https://arxiv.org/abs/1707.06347
- **OpenAI Blog Post (2017):** https://openai.com/index/openai-baselines-ppo/
- **TRPO (predecessor):** https://arxiv.org/abs/1502.05477
- **GAE (advantage estimation):** https://arxiv.org/abs/1506.02438
- **The 37 Implementation Details of PPO:** https://iclr-blog-track.github.io/2022/03/25/ppo-implementation-details/
- **HuggingFace RLHF Tutorial:** https://huggingface.co/blog/rlhf

---

## Citation

```bibtex
@article{schulman2017proximal,
  title={Proximal policy optimization algorithms},
  author={Schulman, John and Wolski, Filip and Dhariwal, Prafulla and Radford, Alec and Klimov, Oleg},
  journal={arXiv preprint arXiv:1707.06347},
  year={2017}
}
```

<!-- related:start -->

---

## Related in This Collection

- [Training Language Models to Follow Instructions with Human Feedback (InstructGPT)](../../language-models/05-instructgpt-rlhf/summary.md)
- [Direct Preference Optimization (DPO): Your Language Model is Secretly a Reward Model](../../language-models/19-dpo/summary.md)
- [DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning](../../language-models/26-deepseek-r1/summary.md)
- [GRPO: Group Relative Policy Optimization](../../techniques/38-grpo/summary.md)
- [RLVR: Reinforcement Learning from Verifiable Rewards](../../techniques/39-rlvr/summary.md)
- [Let's Verify Step by Step: Process Reward Models](../../techniques/51-process-reward-models/summary.md)

<!-- related:end -->
