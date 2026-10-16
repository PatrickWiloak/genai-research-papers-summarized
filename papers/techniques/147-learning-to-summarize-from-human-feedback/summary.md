---
title: "Learning to Summarize from Human Feedback"
slug: "147-learning-to-summarize-from-human-feedback"
number: 147
category: "techniques"
authors: "Nisan Stiennon, Long Ouyang, Jeff Wu, Daniel M. Ziegler, Ryan Lowe, Chelsea Voss, Alec Radford, Dario Amodei, Paul Christiano (OpenAI)"
published: "September 2020 (NeurIPS 2020)"
year: 2020
url: "https://arxiv.org/abs/2009.01325"
tags: ["alignment", "reinforcement-learning", "rlhf"]
---

# Learning to Summarize from Human Feedback

**Authors:** Nisan Stiennon, Long Ouyang, Jeff Wu, Daniel M. Ziegler, Ryan Lowe, Chelsea Voss, Alec Radford, Dario Amodei, Paul Christiano (OpenAI)
**Published:** September 2020 (NeurIPS 2020)
**Paper Link:** [arxiv.org/abs/2009.01325](https://arxiv.org/abs/2009.01325)

---

## Why This Paper Matters

Before this paper, language models were trained to imitate text and graded with proxy metrics. For summarization the recipe was: fine-tune on human-written reference summaries (supervised learning), then score with ROUGE, a measure of word overlap with those references. Both are rough stand-ins for what people actually want, which is a summary that is accurate, covers the important points and reads well. A model that copies the style of its training data is capped at the quality of that data, and ROUGE rewards overlap, not quality.

Stiennon et al. showed a different route: collect human judgments of which of two summaries is better, train a reward model to predict those judgments, then fine-tune the summarizer with reinforcement learning to maximize the predicted score. On a filtered version of the Reddit TL;DR dataset, the resulting models produced summaries that human labelers preferred over the human-written reference summaries. A 1.3B-parameter model trained this way beat a supervised model ten times its size.

Historically, this is the bridge in the RLHF story. Christiano et al. (2017) showed that preference feedback could train agents in games and simulated robotics, with small policies. InstructGPT (2022) used the same recipe to make a general assistant. This paper is the first large-scale demonstration that the recipe works on real language, with big pretrained transformers, careful labeler management and a proper analysis of what breaks. Much of the InstructGPT pipeline (and several of its authors) came directly from here.

## The Core Innovation

Treat human preference as the training signal, and make it scalable by learning a model of it.

```
Step 1: Collect comparisons
  post + two candidate summaries --> human picks the better one

Step 2: Train a reward model (RM)
  RM(post, summary) --> scalar score
  trained so that the preferred summary scores higher

Step 3: Optimize a policy against the RM with RL (PPO)
  policy writes summary --> RM scores it --> update policy
  (repeat the whole loop with fresh comparisons from the new policy)
```

A useful analogy: instead of asking an editor to grade every draft a writer produces (too slow), you train a junior editor on the senior editor's past choices, then let the writer practice against the junior editor all day. The paper's details are about keeping the junior editor honest.

## Key Components Explained

### 1. The data and the baselines
**What it does:** Defines the task and gives a strong reference point.
**How it works:** The authors use the TL;DR dataset of Reddit posts with the original poster's own summary. They filter it to a whitelist of subreddits understandable to a general audience and keep only posts whose human summary is 24 to 48 tokens long, leaving 123,169 posts (about 5% held out for validation). The length filter limits length as a confound. Models are GPT-3-style transformer decoders at 1.3B and 6.7B parameters, first fine-tuned supervised on the reference summaries.

### 2. Human comparison data and quality control
**What it does:** Produces the preference labels.
**How it works:** Labelers (recruited via Upwork and two labeling services, Scale and Lionbridge) pick the better of two summaries. The release includes 64,832 summary comparisons on TL;DR. The authors argue that an earlier attempt (Ziegler et al., 2019) suffered because labelers optimized for something other than the researchers' notion of quality. Their fixes: move to a batch setting, where large batches of comparisons are collected, models retrained, and new batches collected from the new policy; and keep a hands-on relationship with labelers (detailed instructions, answering questions, continuous agreement monitoring). Labelers agreed with researchers 77% (+/- 2%) of the time, close to researcher-to-researcher agreement of 73% (+/- 4%).

### 3. The reward model
**What it does:** Predicts which summary a human would prefer.
**How it works:** A transformer initialized from the supervised model outputs a score, trained so that the log-odds of "summary A beats summary B" follow the score difference. The paper's scaling ablation trained seven reward models from 160M to 13B parameters on 8k to 64k comparisons: doubling the data raised validation accuracy by about 1.1%, doubling the model size by about 1.8%.

### 4. RL with a KL penalty
**What it does:** Improves the summarizer against the reward model without letting it wander into nonsense.
**How it works:** The policy is trained with PPO (see [PPO](../../techniques/63-ppo/summary.md)). The reward is the RM score minus a penalty on the KL divergence from the supervised model:

```
R(x, y) = RM(x, y) - beta * log( pi_RL(y|x) / pi_SFT(y|x) )
```

The penalty has two jobs: it acts as an entropy bonus that stops the policy collapsing to one output, and it keeps outputs close to the region where the reward model has seen data. The value function uses separate parameters from the policy (initialized from the RM) so value updates do not damage the pretrained policy early on.

## Key Results

- **Beats much larger supervised models.** The 1.3B human-feedback model was preferred to reference summaries 61% of the time, versus 43% for a supervised model ten times larger. The 6.7B model improved further, so human feedback also benefits from scale.
- **Beats the human references.** Both human-feedback models were judged better than the reference summaries in the dataset. The models learned to write longer summaries, which confounds the comparison. After controlling for length the advantage drops by about 5%, and the 6.7B model is still preferred about 65% of the time.
- **Better on every quality axis.** On 7-point Likert ratings (coverage, accuracy, coherence, overall), human-feedback models beat the supervised baselines everywhere, most of all on coverage. The 6.7B model got a perfect 7/7 overall score 45% of the time, versus 20% for its supervised baseline and 23% for the reference summaries.
- **Transfers to news.** Without any news-specific fine-tuning, the TL;DR-trained 6.7B model produced CNN/DailyMail summaries nearly as good as a model fine-tuned on CNN/DM references, despite summaries about half as long on average.
- **Reward models beat ROUGE as an objective.** Optimizing against the learned RM gave better summaries (per humans) than optimizing ROUGE. ROUGE agreement with labelers fell to about 50% on samples from the human-feedback model, while the 6.7B RM still agreed with labelers on CNN/DM comparisons 66.5% of the time, close to inter-labeler agreement of 66.9%.
- **Over-optimization is real.** Pushing harder against the RM (a weaker KL penalty) improved summaries at first, then made them worse by human judgment. This is the empirical origin of "reward hacking" as a concern for RLHF, and the same overfitting appeared with ROUGE.

## Why This Was Revolutionary

- Showed the preference-then-RL recipe works on natural language with large pretrained models.
- Established the standard three-stage pipeline: supervised fine-tune, learn a reward model from comparisons, optimize with PPO plus a KL penalty.
- Proved that human-feedback models can exceed the quality of the human demonstrations they started from, which pure imitation cannot do.
- Measured the failure mode (over-optimization) instead of hiding it, giving later work a baseline.
- Treated labeler management as part of the method, a lesson InstructGPT carried forward.

## Real-World Impact

This is the direct ancestor of [InstructGPT](../../language-models/05-instructgpt-rlhf/summary.md) and thus of ChatGPT-style assistants. The pipeline (comparisons, reward model, PPO with KL penalty) is reused almost unchanged; InstructGPT broadens the task from summaries to arbitrary instructions. The RL step relies on [PPO](../../techniques/63-ppo/summary.md). Later work attacked the cost and fragility of the reward-model stage: [DPO](../../language-models/19-dpo/summary.md) removes the explicit reward model and the RL loop, and [RLVR](../../techniques/39-rlvr/summary.md) replaces the learned reward with programmatic checks wherever answers can be verified, which sidesteps over-optimization of a learned proxy.

## Key Takeaways for Practitioners

1. A learned preference model can be a better training target than an automatic metric, but only if the labels are consistent. Measure labeler-researcher agreement continuously.
2. Always control for length when comparing outputs; RL against a reward model tends to produce longer text.
3. Keep a KL penalty to the starting model. Without it the policy exploits the reward model.
4. Expect a peak: there is a point where more optimization against the proxy makes true quality worse. Track true quality with fresh human evaluation, not just the proxy score.
5. Reward model quality scales with both data and model size, so budget for both.

## Limitations & Future Directions

- The study covers English summarization of Reddit and news; the authors point to broader tasks as future work, which InstructGPT then took on.
- Human labeling is expensive and the reward model is biased: the 6.7B RM favored shorter-summary edits only 62.6% of the time versus 76.4% for humans, so it leans toward length.
- Models were at most 6.7B parameters; behavior at larger scale was left to follow-ups.
- Optimizing a single scalar preference ignores disagreement among people and makes the policy vulnerable to over-optimization.
- Follow-ups reduce cost or the proxy problem: [DPO](../../language-models/19-dpo/summary.md) (no explicit RM), [GRPO](../../techniques/38-grpo/summary.md) and [RLVR](../../techniques/39-rlvr/summary.md) (verifiable rewards, no critic for GRPO).

## Further Reading

- **Original Paper:** https://arxiv.org/abs/2009.01325
- **Predecessor (Christiano et al., 2017), Deep RL from human preferences:** https://arxiv.org/abs/1706.03741
- **Earlier attempt (Ziegler et al., 2019), Fine-tuning language models from human preferences:** https://arxiv.org/abs/1909.08593
- **Successor, InstructGPT:** https://arxiv.org/abs/2203.02155
- **Related summaries:** [InstructGPT](../../language-models/05-instructgpt-rlhf/summary.md), [PPO](../../techniques/63-ppo/summary.md), [DPO](../../language-models/19-dpo/summary.md)

## Citation

```bibtex
@inproceedings{stiennon2020learning,
  title={Learning to summarize from human feedback},
  author={Stiennon, Nisan and Ouyang, Long and Wu, Jeff and Ziegler, Daniel M. and Lowe, Ryan and Voss, Chelsea and Radford, Alec and Amodei, Dario and Christiano, Paul},
  booktitle={Advances in Neural Information Processing Systems},
  year={2020}
}
```

<!-- related:start -->

---

## Related in This Collection

- [Language Models are Few-Shot Learners (GPT-3)](../../language-models/04-gpt3-few-shot-learners/summary.md)
- [Training Language Models to Follow Instructions with Human Feedback (InstructGPT)](../../language-models/05-instructgpt-rlhf/summary.md)
- [Direct Preference Optimization (DPO): Your Language Model is Secretly a Reward Model](../../language-models/19-dpo/summary.md)
- [GRPO: Group Relative Policy Optimization](../../techniques/38-grpo/summary.md)
- [RLVR: Reinforcement Learning from Verifiable Rewards](../../techniques/39-rlvr/summary.md)
- [Proximal Policy Optimization Algorithms (PPO)](../../techniques/63-ppo/summary.md)
- [Deep Reinforcement Learning from Human Preferences](../../techniques/146-deep-rl-from-human-preferences/summary.md)

<!-- related:end -->
