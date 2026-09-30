---
title: "Muon: An Optimizer for Hidden Layers, and Muon is Scalable for LLM Training (Muon)"
slug: "144-muon"
number: 144
category: "techniques"
authors: "Keller Jordan, with Yuchen Jin, Vlado Boza, Jiacheng You, Franz Cesista, Laker Newhouse, Jeremy Bernstein (independent researchers and academic collaborators) - Muon post; Jingyuan Liu, Jianlin Su, Xingcheng Yao, Zhejun Jiang, Guokun Lai, Yulun Du, Yidao Qin, Weixin Xu, Enzhe Lu, Junjie Yan, Yanru Chen, Huabin Zheng, Yibo Liu, Shaowei Liu, Bohong Yin, Weiran He, Han Zhu, Yuzhi Wang, Jianzhou Wang, Mengnan Dong, Zheng Zhang, Yongsheng Kang, Hao Zhang, Xinran Xu, Yutao Zhang, Yuxin Wu, Xinyu Zhou, Zhilin Yang (Moonshot AI, UCLA) - scaling paper"
published: "December 2024 (blog post); February 2025 (arXiv preprint)"
year: 2024
url: "https://arxiv.org/abs/2502.16982"
tags: ["optimization", "training", "efficiency"]
---

# Muon: An Optimizer for Hidden Layers, and Muon is Scalable for LLM Training (Muon)

**Authors:** Keller Jordan, with Yuchen Jin, Vlado Boza, Jiacheng You, Franz Cesista, Laker Newhouse, Jeremy Bernstein (independent researchers and academic collaborators) - Muon post; Jingyuan Liu, Jianlin Su, Xingcheng Yao, Zhejun Jiang, Guokun Lai, Yulun Du, Yidao Qin, Weixin Xu, Enzhe Lu, Junjie Yan, Yanru Chen, Huabin Zheng, Yibo Liu, Shaowei Liu, Bohong Yin, Weiran He, Han Zhu, Yuzhi Wang, Jianzhou Wang, Mengnan Dong, Zheng Zhang, Yongsheng Kang, Hao Zhang, Xinran Xu, Yutao Zhang, Yuxin Wu, Xinyu Zhou, Zhilin Yang (Moonshot AI, UCLA) - scaling paper
**Published:** December 2024 (blog post); February 2025 (arXiv preprint)
**Paper:** [arxiv.org/abs/2502.16982](https://arxiv.org/abs/2502.16982)
**Post:** [kellerjordan.github.io/posts/muon](https://kellerjordan.github.io/posts/muon/)

---

## Why This Matters

For a decade, nothing reliably displaced [Adam and AdamW](../142-adam/summary.md) for training large neural networks. Many proposed optimizers beat Adam in their papers and faded in practice. **Muon** is the first new optimizer in years to go from a hobbyist benchmark to frontier-scale training in about a year:

- **December 2024:** Keller Jordan's blog post introduces Muon, which had set speed records in the community "NanoGPT speedrun" (training a small GPT-2-style model to a target loss as fast as possible).
- **February 2025:** Moonshot AI's "Muon is Scalable for LLM Training" reports about **2x compute efficiency over AdamW** in scaling-law experiments and trains **Moonlight**, a mixture-of-experts model, on 5.7T tokens with Muon.
- **July-August 2025:** Moonshot's Kimi K2 (1T total parameters) was pretrained with a Muon variant, MuonClip, and Zhipu's [GLM-4.5 technical report](https://arxiv.org/abs/2508.06471) states that it "employed the Muon optimizer for all parameters except word embedding, bias, and weights for RMSNorm."

**The insight:** Adam treats every number in a weight matrix independently. But a weight matrix is a linear map, and its gradient updates tend to be dominated by a few directions. Muon takes the momentum update for each weight matrix and **orthogonalizes** it, setting all its singular values to 1, so that every direction the update touches moves by the same amount. The rare directions that Adam under-weights get a fair share of each step.

---

## The Problem: Updates Are Nearly Low-Rank

Jordan's post reports an empirical observation: for the 2D weight matrices in transformers, "updates produced by both SGD-momentum and Adam ... typically have very high condition number." A high condition number means the update matrix is dominated by a few large singular directions, making it close to low-rank. The many smaller directions, which may matter for learning, barely move.

```
  An update matrix G, decomposed by SVD:  G = U S V^T

  singular values S (typical):   [ 50, 12, 3, 0.4, 0.1, 0.02, ... ]
                                   \_____ dominates the step _____/

  Muon replaces G with U V^T:     all singular values = 1
                                  -> every direction moves equally
```

"Orthogonalizing" means replacing G with the nearest semi-orthogonal matrix, `U V^T`: same input and output directions, but with the magnitudes flattened. The post puts it this way: orthogonalization "effectively increases the scale of other 'rare directions' which have small magnitude in the update but are nevertheless important."

---

## The Core Innovation

Muon stands for **MomentUm Orthogonalized by Newton-Schulz**. For each 2D weight matrix:

```
  Muon step for a weight matrix W  (per step t)

  G   = gradient of the loss w.r.t. W
  B   = mu * B + G                      # momentum buffer (Nesterov-style by default)
  O   = NewtonSchulz5(B)                # approximately U V^T, the orthogonalized update
  W   = W - lr * O
```

Computing an exact SVD every step is too slow on GPUs, so Muon approximates `U V^T` with **five iterations of a Newton-Schulz polynomial**, which uses only matrix multiplications and runs well in bfloat16:

```
  NewtonSchulz5(G):
    X = G / ||G||_F                     # normalize so singular values <= 1
    repeat 5 times:
      A = X X^T
      X = a*X + (b*A + c*A*A) X         # odd polynomial a*x + b*x^3 + c*x^5
                                        # applied to the singular values
    return X

  tuned coefficients:  a = 3.4445,  b = -4.7750,  c = 2.0315
```

The coefficients were tuned to push singular values toward 1 quickly rather than exactly. The result is only approximately orthogonal (singular values end up roughly between 0.7 and 1.3), which the post reports does not hurt.

### What Muon is not applied to
Muon is only for **hidden-layer 2D weight matrices**. Embeddings, the output classifier head, and scalar or vector parameters (biases, normalization gains) are trained with AdamW. The post reports that input and output layers behave differently and do better with AdamW.

### Cost
The post bounds the extra work at `T*m/B` of the model's FLOPs (T = 5 Newton-Schulz steps, m = model width, B = batch size in tokens), estimating about 0.7 percent for the NanoGPT speedrun and 0.5 percent for a Llama-405B-scale run.

### Theory
The post links the method to **steepest descent under the spectral norm**: if you ask for the update that most reduces the loss for a fixed spectral-norm step size, the answer is the orthogonalized gradient. Jeremy Bernstein and collaborators developed this "modular norm" view, and Muon is also related to the **Shampoo** optimizer: Shampoo's preconditioner without accumulation reduces to the same orthogonalized update.

---

## Key Results from the Original Post

As reported by Keller Jordan (December 2024):

- **CIFAR-10 speedrun:** training to 94 percent accuracy dropped from 3.3 to 2.6 A100-seconds.
- **NanoGPT speedrun:** Muon improved the record for reaching 3.28 validation loss on FineWeb by a factor of 1.35.
- **1.5B-parameter transformer:** reached GPT-2 XL-level HellaSwag performance in 10 hours on an 8xH100 node, versus 13.3 hours with AdamW.

The post was explicit about what was unknown: whether Muon would work for runs of 20B+ parameters, how to distribute it efficiently, and whether it would help beyond pretraining. "At the time of writing, I don't know the answers to these questions."

---

## "Muon is Scalable for LLM Training" (Moonshot AI, 2025)

The Moonshot paper set out to answer those questions. It identifies **two changes** needed at scale:

### 1. Add weight decay
In long runs, weights and layer outputs trained with plain Muon kept growing, which hurt performance and raised numerical issues. The fix is AdamW-style decoupled weight decay:

```
  W = W - lr * (O + lambda * W)
```

### 2. Match the update size to AdamW
An orthogonal matrix's RMS (root-mean-square entry size) depends on its shape, so different matrices would get different effective learning rates. The paper rescales each update by `0.2 * sqrt(max(A, B))` for an A x B matrix, making Muon's update RMS match the typical RMS of AdamW updates (they cite 0.2 to 0.4). The practical benefit: **you can reuse AdamW's learning rate and weight decay** and switch optimizers without a new hyperparameter search.

### Results
- **Scaling laws:** across compute-optimal training runs, Muon needed about **52 percent of AdamW's training FLOPs** to reach the same loss, the "~2x computational efficiency" in the abstract.
- **Moonlight:** a mixture-of-experts model described in the abstract as 3B activated / 16B total parameters (the paper's detailed table gives 2.24B activated and 15.29B total), in the style of [DeepSeek-V3](../../language-models/27-deepseek-v3/summary.md), trained on 5.7T tokens with Muon. The paper reports it beating similar-size models on its benchmark suite, for example MMLU 70.0 against 54.7 for Llama 3.2-3B and 65.6 for Qwen2.5-3B, and GSM8K 77.4 against 34.0 for Llama 3.2-3B. These comparisons involve different training data and token counts, so they show the recipe works at scale rather than isolating the optimizer's effect.
- **Distributed Muon:** an open-source implementation built on ZeRO-1 sharding (see [ZeRO](../76-zero-megatron/summary.md)). Orthogonalization needs the full matrix, so the sharded gradient is gathered, orthogonalized and re-partitioned. Muon keeps one momentum buffer per matrix instead of Adam's two, so its optimizer state is about half of AdamW's.
- **An optimizer mismatch effect:** models pretrained with Muon did best when also fine-tuned with Muon, and mixing optimizers between pretraining and fine-tuning gave weaker gains. The paper lists this as an open problem.

### Later adoption
- **Kimi K2** ([arXiv 2507.20534](https://arxiv.org/abs/2507.20534), July 2025), with 32B activated and 1T total parameters, was pretrained on 15.5T tokens with **MuonClip**, which adds a "QK-clip" technique that rescales query and key projection weights to prevent exploding attention logits. The report says it trained without loss instability.
- **GLM-4.5** (August 2025) used Muon with 5 Newton-Schulz steps, momentum 0.95 and update RMS scaled to 0.2, and reported that it "can accelerate convergence and tolerate larger batch sizes."

---

## Why This Was Revolutionary

- **Broke Adam's decade-long hold** on large-scale pretraining, at least for some labs.
- **Matrix-aware by design.** It treats weight matrices as linear maps rather than bags of numbers, a principled shift in how optimizers are designed.
- **Cheap enough to deploy.** Under 1 percent FLOP overhead and lower optimizer memory than AdamW.
- **Showed the open research path works.** An optimizer developed in public speedrun competitions and blog posts reached trillion-parameter production training within about a year.

---

## Key Takeaways for Practitioners

1. **Use Muon for hidden 2D weights only**; keep AdamW for embeddings, the output head, norms and biases.
2. **At scale, add weight decay and RMS matching.** The Moonshot recipe lets you reuse AdamW hyperparameters.
3. **Watch attention logits** in large runs; MuonClip's QK-clip exists because orthogonalized updates can make them grow.
4. **Be careful mixing optimizers** between pretraining and fine-tuning; the Moonshot paper saw weaker gains when they differed.
5. **Verify the gain on your own setup.** A 2x efficiency claim from one lab's scaling study is strong evidence, not a guarantee for every architecture and data mix.

---

## Limitations & Future Directions

- **Not all parameters.** Muon still needs AdamW for non-matrix parameters, and the paper lists folding them in as future work.
- **Communication cost.** Orthogonalization needs whole matrices, which complicates tensor-parallel and fully sharded training more than element-wise Adam does.
- **Independent replication is still building.** As of 2026, several labs report using Muon in production, but head-to-head comparisons against well-tuned AdamW at the largest scales remain limited, and results depend on tuning. Many variants (AdaMuon, NorMuon and others) are being proposed.
- **Theory is partial.** Steepest descent under the spectral norm motivates the update, but why it helps so much in practice is not fully explained.
- **Fine-tuning and RL** are less studied than pretraining.

---

## Further Reading

- **Muon blog post (Keller Jordan, December 2024):** [kellerjordan.github.io/posts/muon](https://kellerjordan.github.io/posts/muon/)
- **Muon is Scalable for LLM Training:** [arxiv.org/abs/2502.16982](https://arxiv.org/abs/2502.16982)
- **Kimi K2 (MuonClip):** [arxiv.org/abs/2507.20534](https://arxiv.org/abs/2507.20534)
- **GLM-4.5 technical report:** [arxiv.org/abs/2508.06471](https://arxiv.org/abs/2508.06471)
- **In this collection:** [Adam and AdamW](../142-adam/summary.md), [Mixed Precision Training](../143-mixed-precision-training/summary.md), [ZeRO and Megatron-LM](../76-zero-megatron/summary.md), [DeepSeek-V3](../../language-models/27-deepseek-v3/summary.md), [Mixture of Experts](../../architectures/37-mixture-of-experts/summary.md)

## Citation

```bibtex
@misc{jordan2024muon,
  author={Keller Jordan and Yuchen Jin and Vlado Boza and Jiacheng You and Franz Cesista and Laker Newhouse and Jeremy Bernstein},
  title={Muon: An optimizer for hidden layers in neural networks},
  year={2024},
  url={https://kellerjordan.github.io/posts/muon/}
}

@article{liu2025muon,
  title={Muon is Scalable for LLM Training},
  author={Liu, Jingyuan and Su, Jianlin and Yao, Xingcheng and Jiang, Zhejun and Lai, Guokun and Du, Yulun and Qin, Yidao and Xu, Weixin and Lu, Enzhe and Yan, Junjie and Chen, Yanru and Zheng, Huabin and Liu, Yibo and Liu, Shaowei and Yin, Bohong and He, Weiran and Zhu, Han and Wang, Yuzhi and Wang, Jianzhou and Dong, Mengnan and Zhang, Zheng and Kang, Yongsheng and Zhang, Hao and Xu, Xinran and Zhang, Yutao and Wu, Yuxin and Zhou, Xinyu and Yang, Zhilin},
  journal={arXiv preprint arXiv:2502.16982},
  year={2025}
}
```

<!-- related:start -->

---

## Related in This Collection

- [DeepSeek-V3 Technical Report](../../language-models/27-deepseek-v3/summary.md)
- [Qwen3: Technical Report](../../language-models/28-qwen3/summary.md)
- [LLaMA 3.3: Matching 405B Performance with 70B Parameters](../../language-models/33-llama3.3/summary.md)
- [Mixtral of Experts (and the Mixture-of-Experts Architecture)](../../architectures/37-mixture-of-experts/summary.md)
- [Language Models are Unsupervised Multitask Learners (GPT-2)](../../language-models/64-gpt2/summary.md)
- [ZeRO and Megatron-LM: How Trillion-Parameter Models Are Actually Trained](../../techniques/76-zero-megatron/summary.md)
- [The FineWeb Datasets: Decanting the Web for the Finest Text Data at Scale (FineWeb)](../../techniques/133-fineweb/summary.md)
- [Measuring Massive Multitask Language Understanding (MMLU)](../../techniques/137-mmlu/summary.md)

<!-- related:end -->
