---
title: "Direct Preference Optimization (DPO): Your Language Model is Secretly a Reward Model"
slug: "19-dpo"
number: 19
category: "language-models"
authors: "Rafael Rafailov, Archit Sharma, et al. (Stanford)"
published: "May 2023 (NeurIPS 2023)"
year: 2023
url: "https://arxiv.org/abs/2305.18290"
tags: [language-models]
---

# Direct Preference Optimization (DPO): Your Language Model is Secretly a Reward Model

**Authors:** Rafael Rafailov, Archit Sharma, et al. (Stanford)
**Published:** May 2023 (NeurIPS 2023)
**Paper:** [arxiv.org/abs/2305.18290](https://arxiv.org/abs/2305.18290)

---

## Why This Paper Matters

Getting a language model to follow instructions and behave helpfully is not just a matter of training on more text. A pretrained model learns to predict tokens, not to be useful. Alignment - making the model's outputs match human intent and values - requires a separate step, and for years the dominant method was Reinforcement Learning from Human Feedback (RLHF).

RLHF works, but it is expensive and fragile. You need three active models at training time (a policy, a reference policy, and a reward model), a Proximal Policy Optimization (PPO) loop with dozens of sensitive hyperparameters, and careful reward hacking mitigation. InstructGPT and early ChatGPT used exactly this pipeline.

DPO's central insight is that the entire RL loop is unnecessary. The RLHF objective has a **closed-form optimal solution** that expresses the reward function in terms of the policy itself - which means you never have to train a reward model at all, and you never run RL. Instead, you optimize a simple binary cross-entropy loss directly on preference pairs. The result is a stable, cheap, single-phase fine-tune that matches or beats PPO-based RLHF.

DPO did not just offer a shortcut - it offered a different conceptual frame: the language model *is* the reward model. That insight reshaped how the entire open-source ecosystem approaches alignment.

---

## The Problem with RLHF

### The Standard Three-Stage Pipeline

The classic RLHF recipe (as used in InstructGPT - see `papers/language-models/05-instructgpt-rlhf/summary.md`) has three stages:

```
Stage 1 - Supervised Fine-Tuning (SFT):
  Fine-tune the pretrained LM on high-quality demonstration data.
  Produces: SFT model (used as starting point and reference).

Stage 2 - Reward Model Training:
  Collect human preference data: for each prompt x, annotators
  pick which of two completions (y_w, y_l) they prefer.
  Train a separate reward model r_phi(x, y) to predict preferences
  using the Bradley-Terry model.

Stage 3 - RL Fine-Tuning with PPO:
  Use PPO to optimize the SFT model against r_phi,
  while penalizing KL divergence from the SFT reference.
```

This works, but creates several practical problems:

- **Three model copies in memory at once**: the live policy, the frozen reference policy, and the reward model. For 7B+ parameter models this is prohibitive on typical hardware.
- **RL training instability**: PPO's clipped surrogate objective has many sensitive hyperparameters (clip ratio, value function coefficient, entropy bonus, learning rate schedules for policy and value heads). Small changes cause divergence or reward hacking.
- **Reward model inaccuracy**: The reward model is a proxy for human preferences, not preferences themselves. Overoptimizing against it (Goodhart's Law) causes the policy to find adversarial completions the reward model scores highly but humans find bad.
- **Iterative data collection**: In the most principled version, you regenerate preference data on-policy. In practice most labs freeze the preference dataset, which introduces distribution mismatch.

---

## The Core Innovation: The Implicit Reward

### The Key Mathematical Insight

Standard RLHF maximizes expected reward while penalizing divergence from a reference policy:

```
max_pi  E_{x~D, y~pi}[r(x,y)] - beta * KL[pi(·|x) || pi_ref(·|x)]
```

where `beta` controls how far the policy is allowed to move from the reference. This is a constrained optimization problem. Rafailov et al. show that it has a **closed-form optimal solution**:

```
pi*(y|x) = (1/Z(x)) * pi_ref(y|x) * exp(r(x,y) / beta)
```

where `Z(x)` is a per-prompt normalizing constant (the partition function). This is the Boltzmann/softmax distribution over completions weighted by reward.

The key move: **rearrange this equation to express the reward in terms of the policy**:

```
r(x,y) = beta * log(pi*(y|x) / pi_ref(y|x)) + beta * log Z(x)
```

Because `Z(x)` does not depend on `y`, it cancels when you take the difference between rewards for two completions from the same prompt. This means you can express the *preference probability* between completion `y_w` (preferred/chosen) and `y_l` (rejected) purely in terms of the policy ratio:

```
p*(y_w > y_l | x) = sigmoid(
    beta * log(pi*(y_w|x) / pi_ref(y_w|x))
  - beta * log(pi*(y_l|x) / pi_ref(y_l|x))
)
```

### The Bradley-Terry Preference Model

The Bradley-Terry model is a classical pairwise comparison model: the probability that item A is preferred to item B is proportional to the "strength" of A relative to B. In the RLHF context this means:

```
p(y_w > y_l | x) = sigmoid(r(x, y_w) - r(x, y_l))
```

Human annotators implicitly generate samples from this distribution when they mark one completion as better. DPO substitutes the implicit reward expression into this model, removing `r` entirely and leaving only the policy ratios.

### The DPO Loss

Because the preference probability is now expressed entirely in terms of `pi` and `pi_ref`, you can maximize the log-likelihood of the observed human preferences directly:

```
L_DPO(pi; pi_ref) = -E_{(x, y_w, y_l) ~ D}[
    log sigmoid(
        beta * log(pi(y_w|x) / pi_ref(y_w|x))
      - beta * log(pi(y_l|x) / pi_ref(y_l|x))
    )
]
```

This is a binary cross-entropy loss over preference pairs. There is no reward model, no value network, no PPO rollout, no advantage estimation. You just update `pi` with gradient descent while keeping `pi_ref` frozen.

Intuitively, the loss increases the relative probability of the chosen completion and decreases the relative probability of the rejected completion - *relative to the reference model*. The `beta` term controls how aggressively the policy is allowed to deviate: a larger beta keeps the policy closer to the SFT reference.

---

## Key Components Explained

### 1. The Reference Model (pi_ref)

The reference model is the SFT checkpoint, held frozen throughout DPO training. It serves two roles:

- **Anchor**: the KL penalty is implicitly enforced through the log-ratio terms. A completion that is very unlikely under `pi_ref` gets heavily penalized even if it scores well on preferences.
- **Normalization baseline**: the ratio `pi(y|x) / pi_ref(y|x)` measures how much the policy has moved from the SFT distribution for a specific completion. This makes the loss scale-invariant to absolute token probabilities.

Because `pi_ref` is frozen, you only need it for inference (a forward pass to get log-probs). In practice you can often quantize or offload it to reduce memory pressure.

### 2. The beta Hyperparameter

`beta` is the single most important hyperparameter in DPO. It controls the strength of the implicit KL penalty:

- **Low beta (e.g., 0.05-0.1)**: the policy is allowed to move far from the reference. Higher alignment signal but risk of distribution shift and incoherence.
- **High beta (e.g., 0.5-1.0)**: the policy stays conservative, close to the SFT reference. Safer but potentially less aligned.

In practice, beta = 0.1 is a common default for 7B-scale models.

### 3. Preference Data Format

DPO consumes triplets `(x, y_w, y_l)`:
- `x`: the prompt
- `y_w`: the preferred/chosen completion
- `y_l`: the rejected completion

This is the same data format used to train the reward model in RLHF - so existing preference datasets (Anthropic HH-RLHF, OpenAI WebGPT comparisons, etc.) work with DPO without modification.

### 4. The Two-Phase Recipe

```
Phase 1 - SFT:
  Fine-tune pretrained model on demonstration data.
  Save checkpoint as both pi_theta (trainable) and pi_ref (frozen).

Phase 2 - DPO:
  Load preference dataset of (prompt, chosen, rejected) triplets.
  Compute log-probs under pi_theta and pi_ref for both completions.
  Compute DPO loss and update pi_theta only.
  Done.
```

Compare to RLHF's three-stage pipeline: DPO collapses stages 2 and 3 into a single supervised-style pass with no RL.

---

## Key Results

The paper evaluates DPO on three tasks, comparing against PPO and several other baselines:

### Sentiment Control
On the IMDb sentiment task (generate positive movie reviews):
- DPO achieves **71.3% positive** completions while maintaining coherence
- PPO achieves 66.5% positive
- DPO Pareto-dominates PPO on the reward-KL frontier across most beta values

### Single-Turn Dialogue (Anthropic HH Dataset)
Using the Anthropic Helpful and Harmless dataset, GPT-4 was used as a judge to evaluate generated responses:
- DPO-trained model is **preferred over PPO** approximately 61% of the time
- DPO is preferred over the SFT baseline roughly 80% of the time

### Summarization (TL;DR Reddit)
- DPO achieves comparable win rates to PPO against human-written summaries
- DPO is substantially cheaper to train, requiring no reward model or RL loop

### Stability
A qualitative but important result: PPO training curves routinely show reward hacking spikes, collapsed KL, and sudden degradation. DPO training curves look like standard supervised fine-tuning - smooth loss curves, stable validation metrics. This is a major practical advantage when shipping models.

---

## Why This Was Revolutionary

### 1. Complexity Reduction
The alignment pipeline went from requiring RL expertise (understanding PPO clipping, value function training, advantage normalization) to requiring only standard supervised fine-tuning expertise. Any team that could run SFT could now run alignment.

### 2. Hardware Accessibility
Eliminating the reward model and the RL rollout model meant alignment training became feasible on a single machine. This unlocked alignment for the open-source community - labs running Zephyr, Mistral, Llama 2 fine-tunes, and hundreds of HuggingFace models adopted DPO within weeks of the paper's release.

### 3. The Conceptual Reframe
The paper's subtitle - "Your Language Model is Secretly a Reward Model" - is more than marketing. It revealed that the policy and reward model are not separate objects but dual representations of the same underlying preference distribution. This insight seeded a wave of follow-on work rethinking what alignment optimization is actually doing.

### 4. Theoretical Justification for Practice
Before DPO, practitioners often skipped the full RLHF pipeline because it was too expensive, settling for SFT on chosen responses only. DPO gave a principled alternative with theoretical grounding - not just an approximation but the exact optimum of the RLHF objective.

---

## Comparison with Related Work

### vs. PPO-based RLHF (InstructGPT)
See `papers/techniques/63-ppo/summary.md` for PPO details. The comparison at a high level:

| Dimension | PPO-RLHF | DPO |
|---|---|---|
| Models needed | 3 (policy, reference, reward) | 2 (policy, reference) |
| Training phases | 3 (SFT, RM, RL) | 2 (SFT, DPO) |
| Optimization | RL (on-policy rollouts) | Supervised (offline preference data) |
| Stability | Sensitive to hyperparameters | Stable like SFT |
| Reward hacking | Possible | Implicit KL mitigates it |
| Theoretical basis | KL-constrained RL | Closed-form solution to same objective |

PPO remains relevant for cases requiring on-policy data generation or very large-scale training where its sample efficiency advantages matter. For the vast majority of practical alignment work, DPO's simplicity wins.

### vs. GRPO (Group Relative Policy Optimization)
GRPO (used in DeepSeek-R1 and reasoning-focused models) takes a different path: it keeps the RL framework but eliminates the value/critic network by normalizing rewards within a group of sampled responses. Where DPO eliminates RL entirely via a closed-form derivation, GRPO simplifies RL by removing one of its moving parts. GRPO is better suited to tasks where generating multiple samples and scoring them on-policy is feasible (e.g., math with verifiable answers). DPO is better suited to general preference alignment where offline preference data already exists.

### vs. SFT on Chosen Only
A naive baseline is to supervised fine-tune on the chosen (preferred) completions, ignoring the rejected ones. This is strictly weaker than DPO because it provides no signal about what to avoid. DPO's loss explicitly decreases the probability of rejected completions relative to the reference, which the naive SFT approach cannot do.

---

## Real-World Impact and Descendants

### Immediate Adoption (2023)
- **Zephyr-7B** (HuggingFace, Oct 2023): one of the first widely-used open models trained with DPO. Achieved GPT-3.5-level performance on MT-Bench at 7B parameters.
- **Mistral fine-tune ecosystem**: the majority of Mistral-7B-based instruction models on HuggingFace Hub use DPO.
- **TRL library** (HuggingFace): `DPOTrainer` became one of the most-used training classes in the library, lowering the implementation barrier to a handful of lines.

### DPO Variants and Successors (2023-2024)
The paper's theoretical framework spawned a family of preference optimization objectives:

- **IPO (Identity Preference Optimization)**: replaces the sigmoid with an identity link to avoid overfitting on deterministic preferences. Addresses a theoretical concern that the Bradley-Terry model can be overfit when preferences are near-deterministic.
- **KTO (Kahneman-Tversky Optimization)**: removes the requirement for paired preferences. Instead of (chosen, rejected) pairs, KTO works with individual (prompt, completion, label) triplets, making it usable with unpaired feedback.
- **ORPO (Odds Ratio Preference Optimization)**: folds the SFT and DPO losses into a single training step, eliminating even the two-phase recipe.
- **SimPO (Simple Preference Optimization)**: removes the reference model entirely by using average log-probability as an implicit length-normalized reward, further reducing memory requirements.
- **SPIN (Self-Play Fine-Tuning)**: uses the model's own generations as the rejected completions, removing the need for human-annotated rejected responses.

### Influence on Production Alignment
By 2024, DPO or a close variant had become the default preference optimization step for most open-weight model releases. Llama 3, Gemma, Phi-3, and many others use DPO or DPO-inspired objectives in their post-training pipelines. Even labs that use PPO at scale incorporate offline preference optimization steps that share DPO's spirit.

---

## Key Takeaways

1. **The reward model is redundant**: the RLHF objective's optimal policy already encodes the reward implicitly through the log-ratio of policy to reference. DPO makes this explicit.

2. **RL is not required for preference alignment**: a simple binary cross-entropy loss on preference pairs achieves the same theoretical objective as PPO-based RLHF, more stably and at lower cost.

3. **The beta hyperparameter is the only major knob**: it controls how far the policy moves from SFT. In practice, values between 0.05 and 0.5 cover most use cases.

4. **Data quality matters more than method complexity**: DPO is sensitive to the quality of preference annotations. A small set of high-quality comparisons outperforms a large set of noisy ones.

5. **The framework is general**: the closed-form derivation applies to any KL-constrained RL objective, not just language modeling. DPO variants have been applied to image generation, code generation, and multimodal alignment.

6. **Simplicity won**: the history of DPO's adoption is a case study in the value of reducing engineering complexity. A theoretically equivalent but dramatically simpler method displaced a more complex one within a year.

---

## Limitations and Future Directions

### Known Limitations

- **Offline data assumption**: DPO optimizes against a fixed preference dataset. If the policy moves far from the SFT reference (low beta), the data distribution no longer matches the policy distribution - the same distribution shift problem that offline RL faces. On-policy or iterative DPO variants address this at the cost of added complexity.

- **Preference data quality sensitivity**: the Bradley-Terry model assumes consistent, transitive preferences. Real human annotation is noisy, inconsistent, and context-dependent. Poor-quality preference data can cause the model to learn spurious patterns.

- **Length bias**: DPO models tend to favor longer completions because longer sequences have more tokens over which the log-ratio can accumulate. Several variants (SimPO, length-normalized DPO) address this by normalizing by sequence length.

- **Reward hacking in a different form**: while DPO avoids the explicit reward model, the implicit reward can still be gamed. Models sometimes learn to produce outputs that increase the log-ratio through surface features rather than genuine quality improvement.

- **No on-policy exploration**: because DPO is purely offline, it cannot discover high-quality completions outside the support of the preference dataset. For tasks with complex, sparse reward signals (e.g., multi-step reasoning, code correctness), on-policy methods like GRPO retain an advantage.

### Active Research Directions

- **Online/iterative DPO**: regenerating preference data on-policy during training, combining DPO's simplicity with PPO's on-policy benefits.
- **Constitutional AI integration**: using AI-generated preference labels (rather than human annotation) at scale to feed DPO pipelines.
- **Multimodal preference optimization**: extending DPO to image-text and video-text models.
- **Theoretical analysis of implicit reward quality**: understanding when the implicit reward from DPO generalizes vs. overfits to annotation artifacts.

---

## Further Reading

- **Original DPO Paper:** https://arxiv.org/abs/2305.18290
- **InstructGPT (RLHF baseline):** https://arxiv.org/abs/2203.02155
- **Zephyr - first major DPO open model:** https://arxiv.org/abs/2310.16944
- **IPO - Identity Preference Optimization:** https://arxiv.org/abs/2310.12036
- **KTO - unpaired preference optimization:** https://arxiv.org/abs/2402.01306
- **SimPO - reference-free DPO variant:** https://arxiv.org/abs/2405.14734
- **The Illustrated RLHF (Hugging Face blog):** https://huggingface.co/blog/rlhf
- **Yannic Kilcher DPO walkthrough (video):** https://www.youtube.com/watch?v=XZLc09hkMwA
- **TRL DPOTrainer docs:** https://huggingface.co/docs/trl/dpo_trainer

---

## Citation

```bibtex
@inproceedings{rafailov2023direct,
  title={Direct Preference Optimization: Your Language Model is Secretly a Reward Model},
  author={Rafailov, Rafael and Sharma, Archit and Mitchell, Eric and Manning, Christopher D and Ermon, Stefano and Finn, Chelsea},
  booktitle={Advances in Neural Information Processing Systems},
  volume={36},
  year={2023}
}
```

<!-- related:start -->

---

## Related in This Collection

- [Language Models are Few-Shot Learners (GPT-3)](../../language-models/04-gpt3-few-shot-learners/summary.md)
- [Training Language Models to Follow Instructions with Human Feedback (InstructGPT)](../../language-models/05-instructgpt-rlhf/summary.md)
- [Constitutional AI: Harmlessness from AI Feedback](../../language-models/14-constitutional-ai/summary.md)
- [LLaMA 2: Open Foundation and Fine-Tuned Chat Models](../../language-models/17-llama2/summary.md)
- [DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning](../../language-models/26-deepseek-r1/summary.md)
- [LLaMA 3.3: Matching 405B Performance with 70B Parameters](../../language-models/33-llama3.3/summary.md)
- [GPT-4 Technical Report](../../language-models/36-gpt4/summary.md)
- [GRPO: Group Relative Policy Optimization](../../techniques/38-grpo/summary.md)

<!-- related:end -->
