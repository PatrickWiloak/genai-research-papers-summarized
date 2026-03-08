# GRPO: Group Relative Policy Optimization

**Authors:** Zhihong Shao, Peiyi Wang, et al. (DeepSeek-AI)
**Published:** February 5, 2024 (in DeepSeekMath paper)
**Paper:** [arxiv.org/abs/2402.03300](https://arxiv.org/abs/2402.03300)

---

## Why This Matters

GRPO is **the RL algorithm that enabled the reasoning revolution**:

- 🧠 **Powers DeepSeek-R1** - The algorithm behind the most important open reasoning model
- 🚫 **No critic model needed** - Eliminates the most expensive part of PPO
- 📊 **Group-based comparison** - Simpler, more stable than reward models
- 💰 **50% less memory** - No value function to store and train
- 🌍 **Industry standard** - Adopted across the field for reasoning model training

**Real-world impact:**
- Enabled DeepSeek-R1 to match OpenAI o1
- Made RL training accessible to smaller teams
- Foundation of RLVR (Reinforcement Learning from Verifiable Rewards)
- Replaced PPO as the go-to algorithm for LLM reasoning training

**The insight:** **You don't need a separate critic model to do RL.** Just sample multiple outputs, compare them within the group, and reward the better ones.

---

## The Problem GRPO Solves

### Why RL for LLMs?

**Supervised fine-tuning (SFT) alone is limited:**
```
SFT: "Here are correct answers, learn to copy them"
- Model mimics training data
- Ceiling = quality of training data
- Can't discover new strategies
```

**RL goes further:**
```
RL: "Here's a reward signal, figure out how to maximize it"
- Model explores solutions
- Can surpass training data quality
- Discovers emergent strategies (self-reflection, backtracking)
```

### The Problem with PPO

**PPO (Proximal Policy Optimization) was the standard RL algorithm for LLMs:**

```
PPO requires 4 models in memory simultaneously:
1. Policy model (the LLM being trained)
2. Reference model (frozen copy for KL penalty)
3. Reward model (scores outputs)
4. Critic/Value model (estimates expected reward)

Memory: ~4x the LLM size
Training: Complex, unstable
```

**This is incredibly expensive:**
- A 70B parameter LLM means ~280B parameters in memory
- Critic model training is unstable and noisy
- Reward model can be gamed (reward hacking)

### GRPO's Solution

**GRPO eliminates the critic model entirely:**

```
GRPO requires only 3 models:
1. Policy model (the LLM being trained)
2. Reference model (frozen copy for KL penalty)
3. NO reward model needed for verifiable tasks
   (just check if the answer is correct)

Memory: ~2x the LLM size (50% reduction!)
Training: Simpler, more stable
```

---

## How GRPO Works

### The Algorithm

**Step 1: Sample a group of outputs**
```
For each question Q:
  Generate K responses: {o1, o2, o3, ..., oK}
  (typically K = 16-64)
```

**Step 2: Score each response**
```
For verifiable tasks (math, code):
  score(oi) = 1 if correct, 0 if wrong

For general tasks:
  score(oi) = reward_model(oi)
```

**Step 3: Compute group-relative advantage**
```
For each response oi:
  advantage(oi) = (score(oi) - mean(scores)) / std(scores)

This normalizes within the group:
- Better-than-average responses get positive advantage
- Worse-than-average responses get negative advantage
```

**Step 4: Update policy**
```
Increase probability of high-advantage responses
Decrease probability of low-advantage responses
With KL penalty to stay close to reference model
```

### Mathematical Formulation

```python
# Simplified GRPO algorithm
def grpo_step(model, ref_model, questions, K=16):
    for question in questions:
        # 1. Sample K responses
        responses = [model.generate(question) for _ in range(K)]

        # 2. Score responses
        scores = [verify_answer(r, ground_truth) for r in responses]

        # 3. Compute group-relative advantages
        mean_score = mean(scores)
        std_score = std(scores) + 1e-8  # avoid division by zero
        advantages = [(s - mean_score) / std_score for s in scores]

        # 4. Compute policy gradient with clipping
        for response, advantage in zip(responses, advantages):
            # Ratio between new and old policy
            ratio = model.prob(response) / old_model.prob(response)

            # Clipped objective (like PPO)
            clipped_ratio = clip(ratio, 1-epsilon, 1+epsilon)
            loss = -min(ratio * advantage, clipped_ratio * advantage)

            # KL penalty
            kl = kl_divergence(model, ref_model, response)
            total_loss = loss + beta * kl

            total_loss.backward()

    optimizer.step()
```

### Key Difference from PPO

```
PPO:
  advantage = reward - value_estimate(state)
  ↑ Requires trained critic model to estimate value
  ↑ Critic is expensive and noisy

GRPO:
  advantage = (reward - group_mean) / group_std
  ↑ No critic needed!
  ↑ Group statistics serve as baseline
  ↑ Naturally normalized
```

---

## GRPO vs Other RL Methods

### Comparison Table

| Aspect | PPO | GRPO | DPO | REINFORCE |
|--------|-----|------|-----|-----------|
| **Critic model** | Required | Not needed | Not needed | Not needed |
| **Reward model** | Required | Optional | Not needed | Required |
| **Memory** | 4x model | 2x model | 2x model | 2x model |
| **Stability** | Moderate | High | High | Low |
| **For reasoning** | Works | **Best** | Limited | Unstable |
| **Online learning** | Yes | Yes | No (offline) | Yes |
| **Sample efficiency** | Moderate | Moderate | High | Low |

### Why GRPO Beats PPO for Reasoning

```
PPO critic problems:
1. Hard to train critic for long reasoning chains
2. Credit assignment over 1000+ tokens is noisy
3. Critic often wrong -> bad training signal

GRPO advantages:
1. Binary correctness is a clean signal
2. Group comparison is intuitive and stable
3. No noisy critic to corrupt training
```

### Why Not Just Use DPO?

```
DPO limitations:
- Offline: Uses pre-collected preference pairs
- Can't explore: Doesn't generate new responses
- Fixed distribution: Doesn't adapt during training

GRPO advantages:
- Online: Generates fresh responses each step
- Explores: Discovers new reasoning strategies
- Adaptive: Distribution shifts as model improves
```

---

## GRPO in DeepSeek-R1

### The Training Pipeline

```
1. Base model (DeepSeek-V3, 671B MoE)
   ↓
2. Small SFT warmup (~1000 reasoning examples)
   - Teaches basic format
   - Prevents language mixing
   ↓
3. GRPO reinforcement learning
   - Millions of math/code problems
   - Binary reward: correct/incorrect
   - No reward model!
   ↓
4. DeepSeek-R1 (matches OpenAI o1)
```

### Emergent Behaviors from GRPO

**GRPO training produced emergent reasoning without being taught:**

```
Self-reflection: "Wait, let me reconsider..."
Self-verification: "Let me check this answer..."
Backtracking: "This approach isn't working, let me try..."
Strategy selection: "I'll use method X for this type of problem..."
```

**These emerged because GRPO rewards correctness** - the model discovered that these strategies lead to more correct answers.

### The "Aha Moment"

**DeepSeek-R1-Zero (pure GRPO, no SFT):**
```
AIME 2024 accuracy over training:
- Start: 15.6%
- After GRPO: 71.0% (4.5x improvement!)

The model taught itself to reason using only
"right/wrong" feedback on math problems.
```

---

## Known Issues and Improvements

### Length Bias

**Problem:** GRPO can incentivize longer responses
```
Longer responses → more reasoning steps → higher chance of getting right answer
But: Unnecessarily verbose
Result: Model learns to pad with extra reasoning
```

### Dr. GRPO (2025)

**Fix for length bias:**
```
Standard GRPO:
  advantage = (reward - mean) / std
  Problem: Longer correct answers get same reward as short correct answers

Dr. GRPO:
  Normalizes reward by response length
  Penalizes unnecessary verbosity
  Result: Same accuracy, 30-40% shorter responses
```

### Other Variants

- **GSPO (Group Sequence Policy Optimization):** Sequence-level clipping
- **Token-Regulated GRPO:** Per-token optimization
- **Training-Free GRPO:** Applies group comparison at inference time

---

## Practical Usage

### Training with GRPO (using TRL)

```python
from trl import GRPOTrainer, GRPOConfig

# Configure GRPO training
config = GRPOConfig(
    output_dir="./grpo-model",
    num_generations=16,  # K responses per question
    max_new_tokens=2048,
    learning_rate=1e-6,
    per_device_train_batch_size=4,
    kl_coef=0.05,  # KL penalty weight
    num_train_epochs=1,
)

# Define reward function
def reward_fn(completions, ground_truths):
    """Binary reward: 1 if correct, 0 if wrong"""
    rewards = []
    for completion, truth in zip(completions, ground_truths):
        answer = extract_answer(completion)
        rewards.append(1.0 if answer == truth else 0.0)
    return rewards

# Create trainer
trainer = GRPOTrainer(
    model=model,
    config=config,
    train_dataset=math_dataset,
    reward_funcs=reward_fn,
    tokenizer=tokenizer,
)

# Train
trainer.train()
```

### When to Use GRPO

**Perfect for:**
- Math reasoning (verifiable answers)
- Code generation (test cases verify correctness)
- Logic puzzles (definitive solutions)
- Any task with binary correct/incorrect signal

**Less ideal for:**
- Creative writing (no clear "correct" answer)
- Open-ended conversation (need reward model)
- Subjective tasks (can't verify automatically)

---

## Key Takeaways

1. **No critic model needed** - 50% memory savings vs PPO
2. **Group comparison** - Simple, stable advantage estimation
3. **Powers reasoning models** - Behind DeepSeek-R1's success
4. **Works with binary rewards** - Perfect for RLVR (verifiable tasks)
5. **Industry standard** - Adopted as the go-to RL algorithm for reasoning

**Bottom line:** GRPO made RL training for LLMs practical and efficient. By eliminating the critic model and using group-relative comparisons, it enabled the reasoning revolution that started with DeepSeek-R1.

---

## Further Reading

### Original Paper
- **DeepSeekMath (introduces GRPO):** https://arxiv.org/abs/2402.03300
- **DeepSeek-R1 (GRPO at scale):** https://arxiv.org/abs/2501.12948

### Explainers
- **GRPO Deep Dive:** https://cameronrwolfe.substack.com/p/grpo
- **GRPO Illustrated:** https://epichka.com/blog/2025/grpo/

### Improvements
- **Dr. GRPO (fixing length bias):** https://arxiv.org/abs/2503.20783

### Related Work
- **PPO (Schulman et al.):** https://arxiv.org/abs/1707.06347
- **DPO:** https://arxiv.org/abs/2305.18290
- **RLHF/InstructGPT:** https://arxiv.org/abs/2203.02155

---

**Published:** February 5, 2024 (in DeepSeekMath)
**Impact:** 🔥🔥🔥🔥🔥 **CRITICAL** - Enabled the reasoning model revolution
**Adoption:** Industry standard for reasoning model training
**Current Relevance:** THE algorithm for RLVR training
**Legacy:** Made RL training for LLMs practical and accessible

**Modern Status (March 2026):** GRPO has become the dominant RL algorithm for training reasoning models. Nearly every open-source reasoning model uses GRPO or a variant. Dr. GRPO and other improvements continue to refine the approach.
