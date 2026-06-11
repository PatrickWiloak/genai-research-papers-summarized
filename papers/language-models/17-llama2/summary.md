---
title: "LLaMA 2: Open Foundation and Fine-Tuned Chat Models"
slug: "17-llama2"
number: 17
category: "language-models"
authors: "Hugo Touvron, Louis Martin, et al. (Meta AI)"
published: "July 2023"
year: 2023
url: "https://arxiv.org/abs/2307.09288"
tags: ["language-model", "alignment", "rlhf"]
---

# LLaMA 2: Open Foundation and Fine-Tuned Chat Models

**Authors:** Hugo Touvron, Louis Martin, et al. (Meta AI)
**Published:** July 2023
**Paper:** [arxiv.org/abs/2307.09288](https://arxiv.org/abs/2307.09288)

---

## Why This Paper Matters

When Meta released [LLaMA 1](../15-llama/summary.md) in February 2023, it was research-only - you had to apply for access and couldn't build commercial products with it. LLaMA 2, released in July 2023, changed the equation entirely: it came with a permissive community license allowing commercial use for most organizations, it trained on substantially more data (2 trillion tokens vs 1.4 trillion), it doubled the context window, and it shipped a production-ready instruction-tuned variant - Llama-2-Chat - built via a full RLHF pipeline with detailed safety work.

The result was the most consequential open-weights release in LLM history up to that point. Within months, LLaMA 2 became the default backbone for open-source LLM work: fine-tuners, quantization projects (GGUF/llama.cpp), application builders, and research groups all converged on it. It demonstrated that a well-resourced lab could publish both the weights and the alignment methodology transparently - a direct challenge to the closed approach taken by OpenAI with GPT-4 and Anthropic with Claude.

For practitioners the paper is as important as a technical manual as it is as a research artifact. The 77-page document describes the full RLHF training pipeline, the dual-reward-model approach, rejection sampling, Ghost Attention, safety red-teaming, and benchmark methodology in unusual depth. Understanding it is essentially understanding how ChatGPT-class models are actually built.

---

## What Changed from LLaMA 1

LLaMA 2 is not a ground-up redesign. The base architecture is the same decoder-only Transformer with the improvements introduced in LLaMA 1 - RoPE positional embeddings, RMSNorm in place of LayerNorm, and SwiGLU activations. What changed is significant but largely quantitative and procedural:

**Training data and context:** The pretraining corpus grew from ~1.4T tokens to ~2T tokens (roughly 40% more), drawn from publicly available sources with no Meta user data. The context length doubled from 2,048 to 4,096 tokens, enabling longer coherent generations and better document-level understanding.

**Grouped-query attention (GQA):** The 70B model (and the unreleased 34B) uses grouped-query attention rather than standard multi-head attention. GQA is a middle ground between multi-head attention (one set of K/V projections per head) and multi-query attention (one shared K/V pair for all heads). It groups heads so that each group shares a single K/V projection, reducing the KV-cache memory footprint at inference time without a meaningful quality penalty. The 7B and 13B models use standard multi-head attention.

**Chat models via RLHF:** LLaMA 1 shipped only base pretrained weights. LLaMA 2 ships both the base models and the Llama-2-Chat variants, which go through supervised fine-tuning followed by iterative RLHF. This is the core addition - the alignment work is the paper's main contribution, not the base model architecture.

**Commercial license:** The LLaMA 2 Community License permits commercial use for most organizations. The one exception is companies with more than 700 million monthly active users (targeting large platforms) who must seek a separate agreement from Meta.

---

## Key Components Explained

### Pretraining Architecture

All three released sizes - 7B, 13B, and 70B - use the same decoder-only Transformer skeleton with these design choices:

- **RMSNorm** applied pre-layer (not post-layer) for training stability without centering normalization
- **SwiGLU** activation functions in the feed-forward blocks, which empirically outperform ReLU and GELU at scale
- **Rotary Positional Embeddings (RoPE)** rather than learned absolute positions, enabling better generalization to longer sequences at inference
- **Grouped-query attention** at 70B scale (see above); 7B and 13B use standard multi-head attention

Pretraining used approximately 1.7 million GPU-hours on A100-80GB GPUs, running with a global batch size of 4M tokens.

### Supervised Fine-Tuning (SFT)

The first alignment stage after pretraining is supervised fine-tuning on high-quality demonstration data. Meta collected 27,540 human-annotated instruction-response pairs, deliberately prioritizing quality over quantity. The paper makes an explicit argument that a small set of high-quality examples outperforms a large set of noisier data - a position that influenced subsequent work on data curation.

SFT runs for 2 epochs over this dataset. The loss is computed only on assistant responses, not on the prompt tokens, so the model learns to produce completions rather than memorize input patterns.

### Reward Modeling

After SFT, two separate reward models are trained - one for helpfulness and one for safety. This two-reward-model design is a deliberate choice: a single reward model must trade off helpfulness against safety, and gradients from the two objectives can conflict. Keeping them separate lets the training pipeline balance them as explicit constraints rather than implicitly through a single scalar.

Both reward models are initialized from the final SFT checkpoint. They are trained on binary preference data: annotators see two completions for the same prompt and choose which is better (for helpfulness) or safer (for safety). The training objective is a binary cross-entropy loss over these preference pairs. The paper reports that the reward models reach competitive accuracy on held-out preference data before the RLHF loop begins.

### Iterative RLHF with Rejection Sampling and PPO

The RLHF stage does not run a single training pass. Instead, Meta used an iterative approach over multiple rounds (five total in the final model):

1. **Rejection sampling fine-tuning:** Sample K completions from the current policy for each prompt; score them with the reward model; keep only the highest-scoring completion and fine-tune on it. This is a simple but effective method for extracting high-quality outputs without the instability of policy gradient.
2. **PPO (Proximal Policy Optimization):** After rejection sampling stabilizes the policy, PPO applies gradient-based updates directly on reward model scores, with a KL penalty against the SFT model to prevent the policy from drifting too far.

The iterative design matters: each round of RLHF produces a better policy, which is used to collect new preference data, which trains better reward models, which improve the next RLHF round. This compound improvement is why the final Llama-2-Chat models outperform what a single-pass RLHF run would achieve.

The paper references [InstructGPT/RLHF](../05-instructgpt-rlhf/summary.md) as the foundational alignment framework being applied here, and the LLaMA 2 work can be read as a detailed public case study of that pipeline executed at scale.

### Ghost Attention (GAtt)

One practical failure mode of multi-turn RLHF-trained chat models is "instruction amnesia": the model follows the system prompt faithfully in turn 1 but gradually ignores it as the conversation grows. A user who sets a persona or behavioral constraint in the system prompt will find it eroding after a few exchanges.

Ghost Attention is Meta's fix. During SFT, the system prompt is artificially concatenated to every user message in the training sequence - it is "ghost" because the loss is zeroed out on those repeated tokens, so the model never sees it as something to predict, only as persistent context. This conditions the model to treat the system prompt as a standing constraint across all turns rather than a one-time prefix.

The paper demonstrates GAtt's effect with attention visualization: without it, attention to system-prompt tokens drops sharply after the first turn; with it, attention remains roughly constant throughout the conversation.

### Safety Work

The safety component of LLaMA 2 is more extensive than any prior open model release. Key elements:

- **Red-teaming:** Hundreds of adversarial prompts across illicit/criminal content, hateful and harmful speech, and unqualified professional advice categories
- **Context distillation:** Safety instructions are prepended to prompts at fine-tuning time and then distilled into model weights, reducing reliance on system-prompt-level safety guardrails at inference
- **Safety reward model:** Trained in parallel with the helpfulness reward model; used as a constraint in PPO to prevent reward hacking on safety-related content
- **Violation rate benchmarking:** Meta reports ~0.1% policy violation rate on red-team prompts for Llama-2-Chat 70B, lower than comparable closed models at the time

---

## Key Results

### Base Model Benchmarks

| Model | MMLU | HumanEval | BBH | GSM8k |
|-------|------|-----------|-----|-------|
| GPT-3.5 | 70.0 | 48.1 | ~50 | 57.1 |
| LLaMA 2 70B | 68.9 | 29.9 | ~50 | 56.8 |
| LLaMA 2 13B | 54.8 | 18.3 | 39.4 | 28.7 |
| LLaMA 2 7B | 45.3 | 12.8 | 32.6 | 14.6 |

The 70B model matches GPT-3.5 closely on MMLU (general knowledge) and GSM8k (grade-school math). The gap on HumanEval (code) is larger, which motivated the follow-on Code Llama release.

### Chat Model Performance

Llama-2-Chat 70B was rated as preferred or tied against ChatGPT (GPT-3.5-turbo) in independent human evaluations. On the MT-Bench multi-turn dialogue benchmark it scored 6.27 vs GPT-3.5's 7.94, a meaningful gap that still placed it well ahead of other open-source models available at the time. On safety evaluations, Llama-2-Chat scored higher than GPT-3.5 on the TruthfulQA benchmark and showed lower violation rates on adversarial prompts.

---

## Why This Was Revolutionary

**Transparency of the alignment pipeline.** Closed-source labs had been running RLHF since InstructGPT (2022), but the exact procedures were not public. LLaMA 2's paper documented two reward models, iterative rejection sampling, PPO configuration, and safety methodology in reproducible detail. It became the standard reference for understanding how production chat models are actually built.

**Permissive licensing for commercial use.** LLaMA 1 existed behind a research wall. LLaMA 2's license change was not incremental - it created a new category of "open, deployable, commercial-grade" LLM that had not previously existed. Every serious open-source LLM project that followed either used LLaMA 2 weights directly or justified its own work against LLaMA 2 as the baseline.

**Demonstrated safety without sacrificing capability.** Prior open releases (GPT-J, BLOOM, Falcon) had minimal safety work. LLaMA 2 showed that safety training could be applied rigorously to open weights without collapsing model capability - a proof-of-concept that open and safe were not in tension.

**Efficiency at multiple scales.** Shipping 7B, 13B, and 70B simultaneously, with quantized versions available almost immediately via llama.cpp/GGUF, meant LLaMA 2 ran on consumer hardware from a MacBook Air to a multi-GPU server. This range covered essentially every deployment context outside of hyperscale.

---

## Real-World Impact and Descendants

LLaMA 2 became the backbone of the open LLM ecosystem for 2023-2024. A partial list of what it directly produced or enabled:

- **Code Llama** (Meta, August 2023): Fine-tuned on code-heavy data; 7B/13B/34B; fills the HumanEval gap of the base models
- **Mistral 7B** (Mistral AI, September 2023): New architecture but benchmarked directly against LLaMA 2 7B; introduced sliding window attention; showed a smaller model could exceed LLaMA 2 13B
- **Vicuna, WizardLM, Orca 2, Zephyr, OpenHermes:** Fine-tunes from the LLaMA 2 base addressing specific use cases or alignment approaches
- **llama.cpp / GGUF:** Quantized inference library that made LLaMA 2 runnable on CPU-only hardware; 30M+ downloads on Hugging Face as of late 2023
- **Ollama, LM Studio, Jan:** Consumer-facing local LLM products built on llama.cpp and LLaMA 2 weights
- **LLaMA 3** (Meta, April 2024): Direct successor; 8B/70B/400B; trained on 15T tokens; dramatically improved code and reasoning; maintained the same license structure LLaMA 2 established

The paper's detailed RLHF documentation also influenced academic work on alignment - teams studying preference learning, reward hacking, and Constitutional AI approaches all cite LLaMA 2 as a practical implementation reference alongside the theoretical InstructGPT paper.

---

## Key Takeaways for Practitioners

1. **Two reward models beat one.** Separating helpfulness and safety reward models avoids gradient conflict and gives explicit control over the helpfulness/safety tradeoff during PPO. This is worth replicating in any RLHF pipeline where those objectives diverge.

2. **Rejection sampling before PPO stabilizes training.** Running rejection sampling fine-tuning first gives PPO a better starting policy and reduces the instability common in early PPO iterations.

3. **Quality of SFT data matters more than quantity.** The 27,540-example SFT set deliberately prioritized annotation quality over volume. Subsequent work (LIMA, Alpaca) confirmed this finding: a few thousand well-curated examples can match fine-tuning on millions of noisier samples.

4. **GQA is now a standard efficiency tool.** Grouped-query attention, used in the 70B model, has been adopted by nearly every subsequent large model (Mistral, LLaMA 3, Gemma, Falcon 2). It is the default for any model where KV-cache size is a deployment constraint.

5. **Ghost Attention is a simple fix for instruction following across turns.** If you are fine-tuning a chat model and seeing system-prompt erosion, GAtt-style training (concatenate the system prompt to every turn during SFT, zero the loss on it) is a straightforward remedy.

6. **Open weights shift the baseline.** Once LLaMA 2 existed, every closed API had to compete with a free, self-hostable alternative. This changed the economics of LLM deployment and the expectations for what "table stakes" capability looks like.

---

## Limitations and Future Directions

**Code generation gap.** The 70B base model scores 29.9 on HumanEval vs GPT-3.5's 48.1 - a substantial gap. Meta addressed this with Code Llama but the base LLaMA 2 is not a strong general-purpose coding model.

**Context length still limited.** 4,096 tokens was an improvement over LLaMA 1 but remained well below what was needed for long-document tasks. GPT-4 shipped with 8K (later 128K) context, and Claude 2 had 100K. LLaMA 2's 4K ceiling was a real deployment constraint for document analysis use cases.

**English-centric.** The pretraining corpus is overwhelmingly English. Multilingual performance is poor compared to models trained with explicit multilingual data, limiting applicability in non-English markets.

**License is not fully open.** The community license is permissive but not OSI-approved open source. The 700M-user threshold and the requirement to attribute Meta limit what "open" means in practice. Fully open models (Falcon, Mistral, later OLMo) have used Apache 2.0 or similar licenses without such restrictions.

**RLHF data is not released.** The paper describes the preference dataset methodology but the actual annotation data is not public, making it impossible to fully reproduce the alignment work from scratch.

**Overrefusal and sycophancy.** Like all RLHF-tuned models, Llama-2-Chat exhibits overrefusal on benign prompts that pattern-match to sensitive topics, and sycophantic agreement with user assertions. These are emergent failure modes of the preference optimization process that remain active research problems.

---

## Further Reading

- **LLaMA 2 paper:** https://arxiv.org/abs/2307.09288
- **LLaMA 1 (predecessor):** [../15-llama/summary.md](../15-llama/summary.md)
- **InstructGPT / RLHF (foundational alignment method):** [../05-instructgpt-rlhf/summary.md](../05-instructgpt-rlhf/summary.md)
- **Code Llama:** https://arxiv.org/abs/2308.12950
- **Llama 3 technical report:** https://arxiv.org/abs/2407.21783
- **Grouped-query attention paper:** https://arxiv.org/abs/2305.13245
- **llama.cpp (quantized inference):** https://github.com/ggerganov/llama.cpp
- **Mistral 7B (immediate successor in the ecosystem):** https://arxiv.org/abs/2310.06825
- **Andrej Karpathy's LLM walkthrough (highly relevant context):** https://www.youtube.com/watch?v=kCc8FmEb1nY

---

## Citation

```bibtex
@article{touvron2023llama2,
  title={Llama 2: Open Foundation and Fine-Tuned Chat Models},
  author={Touvron, Hugo and Martin, Louis and Stone, Kevin and Albert, Peter and Almahairi, Amjad and Babaei, Yasmine and Bashlykov, Nikolay and Batra, Soumya and Bhargava, Prajjwal and Bhosale, Shruti and others},
  journal={arXiv preprint arXiv:2307.09288},
  year={2023}
}
```

<!-- related:start -->

---

## Related in This Collection

- [Language Models are Few-Shot Learners (GPT-3)](../../language-models/04-gpt3-few-shot-learners/summary.md)
- [Training Language Models to Follow Instructions with Human Feedback (InstructGPT)](../../language-models/05-instructgpt-rlhf/summary.md)
- [Constitutional AI: Harmlessness from AI Feedback](../../language-models/14-constitutional-ai/summary.md)
- [LLaMA: Open and Efficient Foundation Language Models](../../language-models/15-llama/summary.md)
- [LLaMA 3.3: Matching 405B Performance with 70B Parameters](../../language-models/33-llama3.3/summary.md)
- [GPT-4 Technical Report](../../language-models/36-gpt4/summary.md)
- [RoFormer: Enhanced Transformer with Rotary Position Embedding (RoPE)](../../techniques/54-rope-rotary-position-embedding/summary.md)
- [Proximal Policy Optimization Algorithms (PPO)](../../techniques/63-ppo/summary.md)

<!-- related:end -->
