# Mastering Diverse Domains through World Models (DreamerV3)

**Authors:** Danijar Hafner, Jurgis Pasukonis, Jimmy Ba, Timothy Lillicrap (University of Toronto / DeepMind)
**Published:** January 2023 (Nature, 2025 for journal version)
**Paper:** [arxiv.org/abs/2301.04104](https://arxiv.org/abs/2301.04104)

---

## Why This Paper Matters

DreamerV3 was the first reinforcement-learning algorithm to **collect diamonds in Minecraft from scratch, with a fixed set of hyperparameters, with no human demonstrations and no curriculum**. That single sentence is what made it famous — collecting diamonds had been a long-standing open challenge that prior systems either failed at entirely or solved only with extensive human data (the MineRL competitions).

What made DreamerV3 even more important than the headline result was the way it got there: **one algorithm, one set of hyperparameters, more than 150 tasks across continuous control, Atari, DMLab, ProcGen, Crafter, and Minecraft, beating specialized baselines on essentially all of them**. Before DreamerV3, "general-purpose RL" meant retuning your algorithm for every new environment. After it, the dream of a single learning rule that just works across domains looked real for the first time.

It is also one of the cleanest demonstrations of the **model-based RL bet**: learn a world model, then plan and learn in imagination. This idea is now central to embodied AI, robotics, and (in a different form) to the entire foundation-world-model program represented by Genie.

---

## The Problem

Reinforcement learning has historically been notoriously sample-inefficient and brittle:

**1. Per-environment tuning.** SOTA on Atari, continuous control, and Minecraft each required completely different algorithms with different hyperparameters. Try to use the same algorithm on a new domain and it often diverges or learns nothing.

**2. Sample inefficiency.** Model-free RL (PPO, SAC, IMPALA) needs hundreds of millions of environment steps. In real-world or expensive simulators that's a deal-breaker.

**3. Long-horizon credit assignment.** Sparse rewards (like Minecraft's diamond, which requires dozens of correct sub-actions over many minutes) defeat most algorithms.

**4. Mixed observation/action types.** Pixel observations, vector observations, discrete actions, continuous actions, dense rewards, sparse rewards — handling all of these uniformly is unusually hard.

DreamerV3's pitch: one algorithm, fixed hyperparameters, robust across all of these.

---

## The Core Innovation

DreamerV3 is the third iteration of the Dreamer family of model-based RL agents. Its architecture is built around three networks that all learn together:

```
World Model:    learn to predict the future in latent space
Actor:          choose actions to maximize predicted future reward
Critic:         estimate expected future reward of states
```

The critical move is that **the actor and critic are trained entirely inside the world model's imagination**. The agent dreams thousands of rollouts in latent space using the world model, then optimizes its policy on those dreams. Real environment steps are only used to update the world model itself.

What DreamerV3 added on top of DreamerV2:

1. **Symlog predictions.** Transform all targets through `symlog(x) = sign(x) * log(|x| + 1)` to handle reward and value scales that vary by orders of magnitude across tasks.
2. **Two-hot encoded value targets.** Predict reward and value as categorical distributions over symlog-spaced bins instead of regression — far more stable across scales.
3. **KL balancing and free bits.** Stabilizes the world-model latent dynamics so it doesn't collapse or diverge regardless of domain.
4. **Percentile-based return normalization.** Actor objectives are normalized using rolling 5th/95th percentile returns, making the optimizer's job equally hard regardless of whether rewards are 0/1 or 0/100,000.

Together, these tricks remove the per-task tuning that plagued prior agents.

---

## How It Works

### The Recurrent State-Space Model (RSSM)

The world model is an RSSM — a recurrent network with both a deterministic hidden state `h_t` and a stochastic latent `z_t`:

```
h_t = f(h_{t-1}, z_{t-1}, a_{t-1})       # deterministic recurrence
z_t ~ p(z_t | h_t)                        # stochastic latent (prior)
z_t ~ q(z_t | h_t, o_t)                   # posterior given observation
o_hat_t = decoder(h_t, z_t)               # reconstruct observation
r_hat_t = reward_head(h_t, z_t)           # predict reward
gamma_hat_t = continue_head(h_t, z_t)     # predict episode continuation
```

Training the world model is variational — minimize a reconstruction loss for observations and rewards, plus a KL between prior and posterior dynamics.

### Imagination training

Once the world model exists, the actor and critic learn entirely in imagined trajectories:

```
1. Sample a starting latent state from real experience.
2. Roll the world model forward for H=15 steps:
       for t = 1..H:
           a_t  = actor(h_t, z_t)
           z_t  = prior(h_t)              # imagined, no observation
           h_t+1, ...
           r_t  = reward_head(h_t, z_t)
3. Compute imagined returns using critic bootstraps.
4. Update actor to maximize returns; update critic to predict them.
```

The agent never sees the real environment during this loop. All gradient signal comes from inside the world model's dream.

### Fixed hyperparameters

Every experiment in the paper — Atari, ProcGen, DMC, DMLab, Crafter, Minecraft — uses the **same** learning rates, batch sizes, network widths, and discounts. The agent simply scales naturally with model size; the paper releases "S", "M", "L", and "XL" models all with identical recipes.

---

## Key Results

### The Minecraft diamond

The headline: DreamerV3 collected its first diamond after roughly 100 million environment steps, **with no human data and no curriculum**, using the same hyperparameters as every other task in the paper. Diamonds require chopping wood, crafting tools, building furnaces, mining iron, smelting iron, and eventually mining the diamond — dozens of correct steps with sparse rewards. Previous winners of the MineRL diamond challenge all used human demonstrations.

### Atari, ProcGen, DMC, DMLab

DreamerV3 set new state of the art on:

- **Atari-100k** (sample-efficient benchmark)
- **Atari-200M** (asymptotic)
- **ProcGen** (procedurally generated levels — tests generalization)
- **DeepMind Control Suite** (continuous control)
- **DMLab-30** (3D navigation)
- **Crafter** (2D Minecraft-like)

On most of these it beat algorithms specifically tuned for that domain (PPO, IMPALA, Rainbow, DQN, MuZero, EfficientZero, SAC).

### Predictable scaling

Performance improved smoothly with model size and gradient steps per environment step. This is the same scaling-curve story the language-model field has been riding — and it had never been shown this cleanly for RL.

### Robustness without retuning

The paper's experiments ablate showing that without symlog, two-hot, KL balancing, or percentile normalization, the algorithm fails on at least one major domain. With all four, the same hyperparameters carry through all 150+ tasks. This is the practical contribution that mattered most to the field.

---

## Impact and Legacy

DreamerV3 became the default baseline for general-purpose RL. Its influence shows up in three concrete ways:

**1. Algorithmic templates.** The combination of symlog targets, two-hot critics, and percentile return normalization is now standard in modern RL libraries. They show up in subsequent agents (e.g. several follow-ups from the same authors and from OpenAI's Mujoco/Isaac work).

**2. World models go mainstream.** DreamerV3 was the strongest argument yet that learning a world model and acting through it is competitive with — and often superior to — model-free RL. This thesis is the load-bearing assumption behind the entire foundation-world-model wave (Genie, GAIA-1, World Labs, DIAMOND).

**3. Path to general embodied agents.** A single algorithm with single hyperparameters mastering Minecraft, Atari, and continuous control suggests "RL across domains" can be a real research target rather than an aspiration. This influenced DeepMind's SIMA agent (a single agent across 3D games) and broader generalist-agent research.

The paper was later extended in a Nature 2025 publication, formalizing the result for a wider scientific audience.

---

## Connections to Other Papers

- **AlphaZero (#89):** The original "learn a model and plan inside it" success — but with a known, perfect model (game rules). DreamerV3 extends the idea to **learned** world models from pixels.
- **Genie (#94):** Sister line of world-model research. Genie learns a generative world model from internet video for interactive generation; Dreamer learns one from agent experience for planning. Both are foundation-world-model precursors.
- **Voyager (#86):** Another Minecraft milestone — but driven by an LLM with hand-built skills. DreamerV3 reaches Minecraft mastery purely through model-based RL from pixels.
- **MuZero (predecessor to AlphaZero family):** Closest spiritual ancestor — also learns the model and acts through it — but tuned per domain. DreamerV3 generalizes this across all domains with fixed hyperparameters.
- **Scaling Laws (#12):** DreamerV3 reproduces a clean scaling curve for RL, importing the language-model paradigm into reinforcement learning.
- **Generative Agents (#58):** A different agent paradigm using LLM cognition. DreamerV3 represents the pure-RL alternative to embodied intelligence.

---

## Key Takeaways

1. **One algorithm, many domains, fixed hyperparameters.** This was the practical breakthrough. RL stopped needing per-environment tuning.
2. **Symlog and two-hot targets.** A small set of numerical-stability tricks lets the same network handle rewards from 0/1 to 0/1,000,000 without re-tuning.
3. **Imagination training is competitive.** Training the actor and critic entirely in latent rollouts of a learned world model matched or beat model-free RL across the board.
4. **First diamonds in Minecraft from scratch.** A long-standing benchmark of generalization and long-horizon credit assignment fell to a model-based RL agent with no human data.
5. **World models as the path forward.** DreamerV3 is the strongest evidence pre-Genie that learning a world model is not a side path but a core ingredient for general embodied intelligence.
