---
title: "Tulu 3: Pushing Frontiers in Open Language Model Post-Training (Tulu 3)"
slug: "148-tulu-3"
number: 148
category: "language-models"
authors: "Nathan Lambert, Jacob Morrison, Valentina Pyatkin, Shengyi Huang, Hamish Ivison, Faeze Brahman, Lester James V. Miranda, Alisa Liu, Nouha Dziri, Shane Lyu, Yuling Gu, Saumya Malik, Victoria Graf, Jena D. Hwang, Jiangjiang Yang, Ronan Le Bras, Oyvind Tafjord, Chris Wilhelm, Luca Soldaini, Noah A. Smith, Yizhong Wang, Pradeep Dasigi, Hannaneh Hajishirzi (Allen Institute for AI and University of Washington)"
published: "November 2024"
year: 2024
url: "https://arxiv.org/abs/2411.15124"
tags: ["alignment", "reinforcement-learning", "open-models"]
---

# Tulu 3: Pushing Frontiers in Open Language Model Post-Training (Tulu 3)

**Authors:** Nathan Lambert, Jacob Morrison, Valentina Pyatkin, Shengyi Huang, Hamish Ivison, Faeze Brahman, Lester James V. Miranda, Alisa Liu, Nouha Dziri, Shane Lyu, Yuling Gu, Saumya Malik, Victoria Graf, Jena D. Hwang, Jiangjiang Yang, Ronan Le Bras, Oyvind Tafjord, Chris Wilhelm, Luca Soldaini, Noah A. Smith, Yizhong Wang, Pradeep Dasigi, Hannaneh Hajishirzi (Allen Institute for AI and University of Washington)
**Published:** November 2024
**Paper Link:** [arxiv.org/abs/2411.15124](https://arxiv.org/abs/2411.15124)

---

## Why This Paper Matters

A pretrained base model can predict text but cannot follow instructions, refuse harmful requests or hold a conversation. Turning it into a useful assistant is **post-training**, and in 2024 the best post-training recipes were closely guarded. Open-weight models like Llama 3.1 Instruct shipped the weights but not the data, code or training details. The public recipes that did exist (such as the earlier Tulu 2 and several community mixes) lagged well behind.

Tulu 3, from the Allen Institute for AI (Ai2), publishes the whole thing: data, code, evaluation suite, training recipes and every intermediate checkpoint, built on Llama 3.1 base models at 8B, 70B and 405B parameters. The authors report that Tulu 3 outperforms the instruct versions of Llama 3.1, Qwen 2.5 and Mistral at the same sizes, and at 70B matches closed models such as Claude 3.5 Haiku and GPT-4o mini; at 405B it is competitive with DeepSeek v3 and GPT-4o.

It also matters for one new term. Tulu 3 named and popularized **RLVR** (Reinforcement Learning with Verifiable Rewards): RL where the reward is a programmatic check (is the math answer right? did the response obey the constraint?) rather than a learned reward model. A few months later, DeepSeek-R1 scaled the same idea to reasoning models, making RLVR the dominant late-stage training method.

## The Core Innovation

A three-stage recipe, each stage with its own data type, plus the engineering around it:

```
Llama 3.1 base
   |
   v  Stage 1: SFT       prompt + good answer pairs (curated, decontaminated)
Tulu 3 SFT
   |
   v  Stage 2: DPO       preference pairs, on-policy, length-normalized
Tulu 3 DPO
   |
   v  Stage 3: RLVR      reward only if the answer is verifiably correct
Tulu 3 (final)
```

The thesis is that post-training is an engineering discipline: pick target skills, build evaluations before data, iterate on data mixes per skill, and keep the evaluation data out of training. The paper is as much a guide to doing that as a model release.

## Key Components Explained

### 1. Target skills and Tulu 3 Eval
**What it does:** Defines what "good" means before training.
**How it works:** The team targets core skills: knowledge recall, reasoning, math, coding, instruction following, general chat and safety. They split evaluations into a development suite used to make decisions and an unseen suite held out to check overfitting, and release an open evaluation toolkit (also released as OLMES) that standardizes benchmark settings. They also built a new evaluation, IFEval-OOD, with out-of-distribution constraints, to check that instruction following generalizes beyond the training constraints.

### 2. Data curation and decontamination
**What it does:** Builds prompts and answers while keeping test sets out.
**How it works:** Prompts are a mix of public datasets and synthetic ones (for example persona-driven prompts for math, code and instruction following). A decontamination tool checks training prompts for overlap with the evaluation benchmarks and removes matches; several existing public datasets were modified because they overlapped. The paper also drops data with unclear licensing.

### 3. Supervised fine-tuning (SFT)
**What it does:** Teaches format and the basic skills.
**How it works:** The team built skill-specific data mixtures, kept the ones that helped, and repeatedly adjusted the combined mix to fix lagging skills, such as adding math-specific data to lift GSM8K and MATH. The final SFT mix beats the earlier Tulu 2 mix at both 8B and 70B.

### 4. Preference tuning with DPO
**What it does:** Makes answers better and more helpful by contrasting good and bad answers.
**How it works:** Preference data is generated "on-policy" in part: completions from the Tulu 3 SFT model are compared with completions from other models, and an LLM judge picks the winner (an extension of the UltraFeedback pipeline). After careful multi-skill selection, the combined preference data comes to 354,192 instances, and the final 8B and 70B mixes each exceed 270,000 pairs. For the algorithm they tried several variants and found **length-normalized DPO** worked best; plain DPO tends to favor longer answers. See [DPO](../../language-models/19-dpo/summary.md). They also ran a controlled comparison with PPO against a reward model: PPO got similar, slightly lower average scores in an untuned setup and cost more (roughly 28 hours on two nodes), so they used DPO for preference tuning and kept PPO for RLVR.

### 5. RLVR
**What it does:** Pushes specific skills with a clean, hard-to-game reward.
**How it works:** Take prompts that come with a verifier. If the completion is correct, reward alpha; otherwise 0. Train with PPO and a KL penalty to the reference model:

```
v(x, y) = alpha  if the answer is verifiably correct
          0      otherwise
objective: maximize E[ v(x, y) - beta * KL(policy || reference) ]
alpha = 10 (set from pilot runs, not tuned further)
```

This is the standard RLHF objective with a deterministic check in place of a learned reward model. Training prompts come from three sources: the GSM8K training set, the MATH training set, and IFEval-style prompts built from Tulu 2 SFT instructions plus constraints that each have a verification function. Together, roughly 30,000 prompts with ground-truth labels. Findings include that initializing the PPO value function from a general reward model worked best. Related summaries: [RLVR](../../techniques/39-rlvr/summary.md), [PPO](../../techniques/63-ppo/summary.md), and [GRPO](../../techniques/38-grpo/summary.md), the critic-free variant that later became the common choice.

### 6. Infrastructure
**What it does:** Makes RL on large models practical.
**How it works:** An asynchronous setup runs generation in vLLM while learners apply gradient updates at the same time. The codebase scales to 70B and 405B policies.

## Key Results

- **Open-weight leaders at each size.** Tulu 3 beats Llama 3.1 Instruct, Qwen 2.5 Instruct and Mistral-Instruct at comparable sizes on the paper's evaluation suite. At 70B it surpasses GPT-3.5-Turbo and GPT-4o mini, and approaches Claude 3.5 Haiku. Exact per-benchmark averages are in the paper's tables and are not reproduced here.
- **RLVR improves targeted skills.** Training with verifiable rewards raised test performance on GSM8K, MATH and IFEval, in each case beating the starting checkpoint on that benchmark. Training reward rose steadily, while spending more KL budget did not always mean better results on math.
- **Scaling to 405B.** For the 405B model, GSM8K was dropped because SFT and DPO had already saturated it, and the IFEval data did not help much in early runs, so RLVR used only MATH prompts. MATH performance improved by over 5 points in as few as 25 RLVR steps. Only 75 steps were run in total, fewer than for the smaller models, because of the difficulty of scaling the asynchronous RL infrastructure.
- **Fully open.** Released: SFT, DPO and RLVR checkpoints at all sizes, the SFT and preference datasets, the RLVR prompt sets, decontamination tooling and evaluation code.

## Why This Was Revolutionary

- First fully open recipe that matched the best open-weight post-training from big labs.
- Named RLVR and demonstrated it as a distinct, simple stage after DPO.
- Showed how much of post-training quality comes from data mixing and decontamination discipline, not exotic algorithms.
- Reported negative and neutral results (what did not reliably help), which is rare in model reports.
- Gave academics and small teams a reproducible baseline.

## Real-World Impact

RLVR spread quickly. [DeepSeek-R1](../../language-models/26-deepseek-r1/summary.md) used rule-based rewards with [GRPO](../../techniques/38-grpo/summary.md) at scale to elicit long reasoning, and RLVR is now the default for math, code and agentic training. Tulu 3's data, recipes and eval tooling became a common starting point for open post-training research. Its DPO stage rests on [DPO](../../language-models/19-dpo/summary.md), and the preference-then-RL lineage traces back to [InstructGPT](../../language-models/05-instructgpt-rlhf/summary.md).

## Key Takeaways for Practitioners

1. Build the evaluation suite first, with a held-out set, and decontaminate training data against it.
2. Add stages in order of cost: SFT, then DPO, then RLVR for skills with checkable answers.
3. Use length-normalized DPO unless you have a reason not to; watch for length inflation.
4. RLVR needs a verifier, not a reward model. Math answers, code tests and format constraints qualify; open-ended chat does not.
5. Initialize the PPO value model from a general reward model; do not skip KL control.
6. Check that gains on a trained constraint type generalize (IFEval-OOD exists for this).

## Limitations & Future Directions

- RLVR was demonstrated only on three verifiable domains (math and precise instruction following); the authors leave richer verifiers, such as code execution, to future work.
- PPO-based RLVR carries a value model and heavy infrastructure; GRPO-style methods later reduced that.
- Evaluations rely on benchmarks that can saturate or leak; the paper treats contamination seriously but cannot eliminate it.
- Uses Llama 3.1 base models under Llama's license, so the recipe is open but the base is not fully open in the strictest sense.
- Tulu 3 targets English-centric assistant behavior; multilingual and long-context post-training are left open.

## Further Reading

- **Original Paper:** https://arxiv.org/abs/2411.15124
- **Models, data and code:** Ai2 Tulu 3 collections on Hugging Face and the `allenai/open-instruct` repository
- **Related summaries:** [RLVR](../../techniques/39-rlvr/summary.md), [GRPO](../../techniques/38-grpo/summary.md), [DPO](../../language-models/19-dpo/summary.md), [DeepSeek-R1](../../language-models/26-deepseek-r1/summary.md)

## Citation

```bibtex
@article{lambert2024tulu3,
  title={Tulu 3: Pushing Frontiers in Open Language Model Post-Training},
  author={Lambert, Nathan and Morrison, Jacob and Pyatkin, Valentina and Huang, Shengyi and Ivison, Hamish and Brahman, Faeze and Miranda, Lester James V. and Liu, Alisa and Dziri, Nouha and Lyu, Shane and Gu, Yuling and Malik, Saumya and Graf, Victoria and Hwang, Jena D. and Yang, Jiangjiang and Le Bras, Ronan and Tafjord, Oyvind and Wilhelm, Chris and Soldaini, Luca and Smith, Noah A. and Wang, Yizhong and Dasigi, Pradeep and Hajishirzi, Hannaneh},
  journal={arXiv preprint arXiv:2411.15124},
  year={2024}
}
```

<!-- related:start -->

---

## Related in This Collection

- [Language Models are Few-Shot Learners (GPT-3)](../../language-models/04-gpt3-few-shot-learners/summary.md)
- [Training Language Models to Follow Instructions with Human Feedback (InstructGPT)](../../language-models/05-instructgpt-rlhf/summary.md)
- [Direct Preference Optimization (DPO): Your Language Model is Secretly a Reward Model](../../language-models/19-dpo/summary.md)
- [DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning](../../language-models/26-deepseek-r1/summary.md)
- [Qwen3: Technical Report](../../language-models/28-qwen3/summary.md)
- [Claude 3.5 Sonnet: Computer Use and Enhanced Capabilities](../../language-models/30-claude-3.5-sonnet/summary.md)
- [LLaMA 3.3: Matching 405B Performance with 70B Parameters](../../language-models/33-llama3.3/summary.md)
- [GPT-4 Technical Report](../../language-models/36-gpt4/summary.md)

<!-- related:end -->
