---
title: "DAPO: An Open-Source LLM Reinforcement Learning System at Scale (DAPO)"
slug: "155-dapo"
number: 155
category: "techniques"
authors: "Qiying Yu, Zheng Zhang, Ruofei Zhu, Yufeng Yuan, Xiaochen Zuo, Yu Yue, et al. (ByteDance Seed, Tsinghua University AIR, The University of Hong Kong; project lead Qiying Yu, senior authors include Hao Zhou and Mingxuan Wang)"
published: "March 2025 (arXiv v1, 18 March 2025; revised May 2025)"
year: 2025
url: "https://arxiv.org/abs/2503.14476"
tags: ["reinforcement-learning", "reasoning", "alignment"]
---

# DAPO: An Open-Source LLM Reinforcement Learning System at Scale (DAPO)

**Authors:** Qiying Yu, Zheng Zhang, Ruofei Zhu, Yufeng Yuan, Xiaochen Zuo, Yu Yue, et al. (ByteDance Seed, Tsinghua University AIR, The University of Hong Kong; project lead Qiying Yu, senior authors include Hao Zhou and Mingxuan Wang)
**Published:** March 2025 (arXiv v1, 18 March 2025; revised May 2025)
**Paper Link:** [arxiv.org/abs/2503.14476](https://arxiv.org/abs/2503.14476)

---

## Why This Paper Matters

In early 2025, reinforcement learning (RL) on verifiable answers was clearly the technique behind "reasoning" models such as OpenAI o1 and DeepSeek-R1 (see [DeepSeek-R1](../../language-models/26-deepseek-r1/summary.md)). But the recipes were only partly public. The DAPO authors say the key technical details of the state-of-the-art systems were concealed, and that the community struggled to reproduce the results. Their own first attempt is the telling data point: running plain [GRPO](../38-grpo/summary.md) on the Qwen2.5-32B base model reached only 30 points on AIME 2024 (a hard US high-school math competition), far below the 47 points DeepSeek reported for the same base model.

DAPO is the paper that closes that gap in the open. It names four concrete failure modes of naive GRPO on long chain-of-thought (CoT) outputs, fixes each one, and releases the algorithm, the training code (built on the verl framework) and the dataset. The result is 50 points on AIME 2024 with Qwen2.5-32B, beating DeepSeek-R1-Zero-Qwen-32B (47) using, per the paper, 50% of the training steps.

For a practitioner, the value is less the headline number than the diagnosis. "Why does my RL run's entropy collapse, my response lengths explode, or my gradients vanish?" now has named causes and cheap remedies.

## The Core Innovation

DAPO stands for Decoupled Clip and Dynamic sAmpling Policy Optimization. It starts from GRPO (no value network; advantages come from normalizing rewards within a group of sampled answers to the same question) and changes four things, plus one removal.

```
GRPO (baseline)                      DAPO
------------------------------       ----------------------------------
one clip range [1-e, 1+e]            separate e_low = 0.2, e_high = 0.28
keep all prompts in the batch        drop prompts where all answers are
                                     right or all wrong; keep sampling
loss averaged per sample             loss averaged per token
truncated answer = punished          soft, length-aware penalty near the
                                     length limit
KL penalty to the start model        no KL term
```

The reward itself is simple and rule-based: +1 if the final answer is equivalent to the ground truth, -1 otherwise. No learned reward model, which avoids reward hacking (see [RLVR](../39-rlvr/summary.md)).

## Key Components Explained

### 1. Clip-Higher
**What it does:** Stops the policy from becoming deterministic too early (entropy collapse).
**How it works:** PPO-style clipping limits how far a token's probability can rise in one update relative to the old policy. With the usual clip of 0.2, a token at probability 0.01 can rise only to 0.012, while a token at 0.9 can rise to the cap of its ratio (the paper's example gives 1.08, i.e. no practical limit). So "exploit" tokens grow easily and rare "explore" tokens barely move. DAPO decouples the two bounds and raises only the upper one: e_low = 0.2, e_high = 0.28. The lower bound is left alone because raising it would push token probabilities to zero and shrink the sampling space.
**Analogy:** a teacher who lets students double down on what already works but caps praise for any new idea at a trivial amount. Clip-Higher lifts that cap.

### 2. Dynamic Sampling
**What it does:** Keeps every batch full of useful training signal.
**How it works:** In GRPO, if all sampled answers to a prompt are correct (or all wrong), every answer gets the same reward, so the advantage is zero and the gradient is zero. As training proceeds, more prompts become "all correct", so the effective batch shrinks and gradients get noisier. DAPO over-samples and filters out prompts with accuracy exactly 0 or 1, continuing to sample until the batch is full of prompts with mixed outcomes. The paper reports that, even though more rollouts are generated, the model reaches the same performance faster because fewer training steps are needed.

### 3. Token-Level Policy Gradient Loss
**What it does:** Gives long responses their fair weight.
**How it works:** GRPO averages the loss within each response and then across responses, so a 10,000-token answer counts the same as a 200-token one and each of its tokens counts far less. That hurts in two ways: good long reasoning chains are under-learned, and bad long outputs (gibberish, repetition) are under-penalized, which lets entropy and length grow unhealthily. DAPO sums over all tokens in the batch and divides by the total token count, so every token carries equal weight. The paper notes this adds fewer accuracy points than the other tricks but makes training stabler and length growth healthier.

### 4. Overlong Reward Shaping
**What it does:** Removes noise from truncated answers.
**How it works:** Generation has a maximum length. By default, a truncated answer is punished, but it may have been correct reasoning that was simply long, which confuses the model. DAPO first tries Overlong Filtering (mask the loss for truncated samples), which already stabilizes training. It then adds Soft Overlong Punishment, a length-aware penalty:

```
R_length(y) = 0                         if |y| <= L_max - L_cache
            = ((L_max - L_cache) - |y|) / L_cache
                                        if L_max - L_cache < |y| <= L_max
            = -1                        if |y| > L_max
```

The penalty is added to the correctness reward. In the paper's setup the expected maximum is 16,384 tokens plus a 4,096-token soft-penalty buffer, for 20,480 tokens total.

### 5. Removing the KL term
In RLHF, a KL penalty keeps the model near its starting point so it stays a good assistant. For long-CoT reasoning on verifiable tasks the model is expected to drift far from the base model, so DAPO drops the penalty.

### Data: DAPO-Math-17K
Competition answers come in messy formats (expressions, fractions, radicals), which are hard to grade by rule. The authors transform problems so the answer is an integer (for example, asking for a+b+c instead of a form with a radical). The resulting set has 17K prompts, each with an integer answer.

## Key Results

Training details reported: Qwen2.5-32B base, AdamW with constant learning rate 1e-6, 512 prompts per rollout batch with 16 responses each, evaluation as avg@32 on AIME 2024 (32 repeats) at temperature 1.0 and top-p 0.7.

| Configuration | AIME 2024 avg@32 |
|---|---|
| DeepSeek-R1-Zero-Qwen-32B (reported) | 47 |
| Naive GRPO | 30 |
| + Overlong Filtering | 36 |
| + Clip-Higher | 38 |
| + Soft Overlong Punishment | 41 |
| + Token-level Loss | 42 |
| + Dynamic Sampling (full DAPO) | 50 |

- Each technique adds points, and Dynamic Sampling contributes the largest last step (42 to 50).
- Training behavior matters as much as scores. The paper recommends watching response length, reward, entropy and mean probability together. Training reward often correlates poorly with validation accuracy, which signals overfitting.
- The authors observe reflection and backtracking behavior emerging during training that was nearly absent early on.

## Why This Was Revolutionary

- A fully open, reproducible recipe (algorithm, code, dataset) for reasoning-model RL at a time when the details were withheld.
- Turned vague "RL is unstable" folklore into four named, individually ablated fixes.
- Showed that small objective-level changes (an asymmetric clip, a loss normalization) are worth many benchmark points.
- Made entropy and length monitoring a standard part of RL practice.

## Real-World Impact

DAPO's changes are now common options in open RL training frameworks and are widely cited as a baseline when comparing GRPO variants. It builds directly on [GRPO](../38-grpo/summary.md) and the verifiable-reward setup of [RLVR](../39-rlvr/summary.md), and it is the open counterpart to the recipe described in [DeepSeek-R1](../../language-models/26-deepseek-r1/summary.md). I have not verified any specific downstream adoption figures, so treat "widely used" as a qualitative statement.

## Key Takeaways for Practitioners

1. If entropy falls fast, the policy is collapsing. Try a higher upper clip bound before anything exotic.
2. Filter out prompts that are all-correct or all-wrong; they carry no gradient.
3. Normalize the loss by token, not by sample, when responses are long and vary a lot in length.
4. Do not punish truncated answers harshly; mask or softly shape them.
5. Track length, entropy and validation accuracy together, not training reward alone.
6. Verifiable, easily parsed answers (integers here) make rule-based rewards reliable.

## Limitations & Future Directions

- Evaluated on math only (AIME 2024); the authors say the method should transfer to other tasks but do not show it.
- A single base model (Qwen2.5-32B) and a single headline benchmark; the 50 is an avg@32 figure on one test set.
- The progressive ablation table adds techniques cumulatively, so each row's gain depends on the order.
- Training dynamics remain sensitive: the paper itself stresses that small changes can have unexpected effects.
- The origin of emergent reflection behavior is left for future work.

## Further Reading

- **Original Paper:** https://arxiv.org/abs/2503.14476
- **Project page:** https://dapo-sia.github.io/
- **Training framework (verl):** https://github.com/volcengine/verl
- **Related summaries:** [GRPO](../38-grpo/summary.md), [RLVR](../39-rlvr/summary.md), [DeepSeek-R1](../../language-models/26-deepseek-r1/summary.md)
- **Concept page:** [Reasoning models](../../../explainers/concepts/reasoning-models.md)

## Citation

```bibtex
@article{yu2025dapo,
  title={DAPO: An Open-Source LLM Reinforcement Learning System at Scale},
  author={Yu, Qiying and Zhang, Zheng and Zhu, Ruofei and others},
  journal={arXiv preprint arXiv:2503.14476},
  year={2025}
}
```

<!-- related:start -->

---

## Related in This Collection

- [Training Language Models to Follow Instructions with Human Feedback (InstructGPT)](../../language-models/05-instructgpt-rlhf/summary.md)
- [Chain-of-Thought Prompting Elicits Reasoning in Large Language Models](../../techniques/09-chain-of-thought/summary.md)
- [DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning](../../language-models/26-deepseek-r1/summary.md)
- [Qwen3: Technical Report](../../language-models/28-qwen3/summary.md)
- [OpenAI o1: Learning to Reason with Reinforcement Learning](../../language-models/31-openai-o1/summary.md)
- [GRPO: Group Relative Policy Optimization](../../techniques/38-grpo/summary.md)
- [RLVR: Reinforcement Learning from Verifiable Rewards](../../techniques/39-rlvr/summary.md)
- [Proximal Policy Optimization Algorithms (PPO)](../../techniques/63-ppo/summary.md)

<!-- related:end -->
