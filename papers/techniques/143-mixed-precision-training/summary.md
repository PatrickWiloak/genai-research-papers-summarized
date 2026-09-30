---
title: "Mixed Precision Training"
slug: "143-mixed-precision-training"
number: 143
category: "techniques"
authors: "Paulius Micikevicius, Sharan Narang, Jonah Alben, Gregory Diamos, Erich Elsen, David Garcia, Boris Ginsburg, Michael Houston, Oleksii Kuchaiev, Ganesh Venkatesh, Hao Wu (NVIDIA, Baidu Research)"
published: "October 2017 (ICLR 2018)"
year: 2017
url: "https://arxiv.org/abs/1710.03740"
tags: ["efficiency", "training", "systems"]
---

# Mixed Precision Training

**Authors:** Paulius Micikevicius, Sharan Narang, Jonah Alben, Gregory Diamos, Erich Elsen, David Garcia, Boris Ginsburg, Michael Houston, Oleksii Kuchaiev, Ganesh Venkatesh, Hao Wu (NVIDIA, Baidu Research)
**Published:** October 2017 (ICLR 2018)
**Paper:** [arxiv.org/abs/1710.03740](https://arxiv.org/abs/1710.03740)

---

## Why This Matters

Every large model trained since about 2018 has used some form of **mixed precision**: doing most of the arithmetic in a small, fast number format while keeping the few numbers that need accuracy in a larger one. This paper, from NVIDIA and Baidu, gave the recipe that made 16-bit training match 32-bit accuracy **without changing models or hyperparameters**. It is a large part of why GPU training got several times faster per dollar, and its ideas lead directly to today's BF16 and FP8 training.

- **About half the memory.** Weights, activations and gradients are stored in 16 bits instead of 32.
- **Much faster arithmetic.** NVIDIA's Volta Tensor Cores ran FP16 matrix maths far faster than FP32; the paper cites 2-6x speedups for memory- or arithmetic-bound operations.
- **Three simple techniques**: an FP32 master copy of the weights, loss scaling, and FP32 accumulation.
- **Worked across model types**: image classifiers, object detectors, speech recognition, translation, language models and GANs.

**The insight:** only a few steps in training actually need 32-bit precision. Small weight updates and the sums inside matrix multiplications need it; the bulk storage and multiplications do not. Keep precision where it matters and drop it everywhere else.

---

## The Problem: FP16 Is Fast but Fragile

A floating-point number stores a sign, an **exponent** (which sets the range) and a **mantissa** (which sets the precision).

```
  Format   bits   exponent  mantissa   max value    smallest subnormal
  FP32      32       8         23      ~3.4e38        ~1.4e-45
  FP16      16       5         10      65,504         ~6e-8  (2^-24)
  BF16      16       8          7      ~3.4e38        (same range as FP32)
  FP8 E4M3   8       4          3      448
  FP8 E5M2   8       5          2      57,344
```

FP16 has two failure modes in training:

1. **Underflow.** Gradients are often tiny. Anything smaller than about 2^-24 becomes exactly zero in FP16. In one of the paper's speech models, about 5 percent of weight-gradient values were that small.
2. **Swamping.** Adding a tiny update to a large weight can leave the weight unchanged, because FP16's 10-bit mantissa cannot represent the difference. If a weight is more than about 2,048 times larger than its update, the update can vanish.

Earlier low-precision work (binary or quantized networks) lost accuracy on large models or kept gradients in full precision, so it saved little compute during training.

---

## The Core Innovation

```
  One mixed-precision training step

         FP32 master weights
                |
                | cast to FP16
                v
        FP16 weights --> forward pass (FP16 storage,
                |        FP16 x FP16 products summed in FP32)
                v
             loss (x S)          <-- loss scaling by factor S
                |
                v
        backward pass in FP16  -->  FP16 gradients (scaled by S)
                |
                | cast to FP32, divide by S,
                | skip step if any inf/NaN
                v
        optimizer update on FP32 master weights
```

### 1. FP32 Master Copy of Weights
Keep the authoritative weights in FP32 and cast a copy to FP16 each iteration for the forward and backward passes. The update `lr * gradient` is applied in FP32, so small updates are not lost to underflow or swamping. The paper shows a Mandarin speech model that matched FP32 accuracy with a master copy but lost 80 percent relative accuracy when FP16 weights were updated directly. The extra copy adds 50 percent to weight memory, but activations dominate training memory, and those are halved.

### 2. Loss Scaling
Multiply the loss by a constant S before backpropagation. By the chain rule every gradient is multiplied by S too, lifting tiny values into FP16's representable range. Divide by S before the weight update. In a Multibox SSD object detector, 67 percent of activation-gradient values were zero and much of FP16's range went unused; scaling by just 8 was enough to match FP32. The paper used factors from 8 to 32K. It also noted that overflow can be detected by checking for infinities or NaNs and the step skipped, and suggested adjusting the factor automatically. That became **dynamic loss scaling**, the default in PyTorch's `torch.cuda.amp` and similar tools.

### 3. FP32 Accumulation
Multiply FP16 numbers, but add up the products in FP32 and round to FP16 only when writing to memory. Volta Tensor Cores do this in hardware. Without it, some models did not match baseline accuracy. Reductions such as batch-normalization statistics and softmax sums also stay in FP32.

---

## Key Results

**ImageNet classification (top-1 accuracy, no loss scaling needed):**

| Model | FP32 baseline | Mixed precision |
|---|---|---|
| AlexNet | 56.77% | 56.93% |
| VGG-D | 65.40% | 65.43% |
| GoogLeNet | 68.33% | 68.43% |
| Inception v2 | 70.03% | 70.02% |
| Inception v3 | 73.85% | 74.13% |
| ResNet-50 | 75.92% | 76.04% |

**Object detection (Pascal VOC mAP):**

| Model | FP32 | Mixed, no loss scaling | Mixed, with loss scaling |
|---|---|---|---|
| Faster R-CNN | 69.1% | 68.6% | 69.7% |
| Multibox SSD | 76.9% | diverges | 77.1% |

- **Speech (DeepSpeech 2, up to 215M parameters):** mixed precision matched or slightly beat FP32 character error rate; the authors speculate half-precision storage acts as a mild regularizer.
- **Language modeling (bigLSTM on 1 billion words):** diverged without loss scaling after about 300K iterations; a scale factor of 128 matched FP32 perplexity.
- **Translation and DCGAN** also matched FP32 results.

The key claim held across all of them: **no hyperparameter changes**, same accuracy.

---

## Later Context: BF16 and FP8

### BF16: dropping loss scaling
**bfloat16** ("brain floating point", from Google Brain) keeps FP32's 8-bit exponent and cuts the mantissa to 7 bits. It has the same range as FP32, so gradients rarely underflow and **loss scaling is usually unnecessary**. "[A Study of BFLOAT16 for Deep Learning Training](https://arxiv.org/abs/1905.12322)" (Kalamkar et al., Intel and Facebook, 2019) reported that BF16 training matched FP32 across domains without hyperparameter changes. After TPUs, NVIDIA's A100 (2020) supported BF16, and it became the default for LLM pretraining: [Llama](../../language-models/15-llama/summary.md) and the [Switch Transformer](../../architectures/67-switch-transformer/summary.md) describe BF16 training, and ZeRO-era recipes recommend it (see [ZeRO and Megatron-LM](../76-zero-megatron/summary.md)). The master-weights idea from this paper carried over unchanged: optimizer states and master weights stay in FP32.

### FP8: the next halving
"[FP8 Formats for Deep Learning](https://arxiv.org/abs/2209.05433)" (Micikevicius et al., NVIDIA, Arm and Intel, 2022), led by the same first author, proposed two 8-bit formats: **E4M3** (more precision, used for weights and activations) and **E5M2** (more range, suited to gradients). It reported FP8 training matching 16-bit results on models up to 175B parameters. NVIDIA's Hopper GPUs (H100) added FP8 Tensor Cores.

### DeepSeek-V3: FP8 at frontier scale
[DeepSeek-V3](../../language-models/27-deepseek-v3/summary.md) (December 2024) is the best-documented case of FP8 pretraining at scale. It reported being the first to "validate the feasibility and effectiveness of FP8 training on an extremely large-scale model." Its recipe is this paper's philosophy taken further:

- **Most matrix multiplications in FP8** (the forward pass and both backward-pass multiplications), using E4M3 for all tensors.
- **Fine-grained scaling instead of one loss scale**: a scale factor per 1x128 tile of activations and per 128x128 block of weights, so outliers in one region do not ruin precision elsewhere. This is loss scaling's idea applied locally.
- **Higher-precision accumulation**: partial sums are promoted to FP32 on CUDA cores every 128 elements, a direct descendant of FP32 accumulation here.
- **Sensitive parts kept in higher precision**: embeddings, the output head, MoE gating, normalization and attention.
- **Master weights, weight gradients and optimizer states in higher precision** (some optimizer states in BF16).
- Reported relative loss error versus a BF16 baseline stayed **below 0.25 percent**.

The three techniques of 2017, a high-precision master copy, scaling to fit the format's range, and high-precision accumulation, are all still visible in 2024 FP8 training.

---

## Why This Was Revolutionary

- **Made 16-bit training a default, not a research project.** No hyperparameter changes meant anyone could turn it on.
- **Co-designed with hardware.** Tensor Cores and this recipe arrived together, and every NVIDIA generation since has added lower-precision formats on the same principles.
- **Roughly doubled the model size** that fit on a given GPU for training and substantially raised throughput.
- **Set the pattern for every later precision step**: find which operations are sensitive, keep those in high precision, and scale values into the format's range.

---

## Real-World Impact

- **Built into every framework**: PyTorch `torch.amp` (autocast plus a gradient scaler), TensorFlow mixed precision, JAX dtype policies, NVIDIA Apex (historically), DeepSpeed and Megatron-LM.
- **Foundational to LLM training economics.** BF16 made the 2020-2024 generation of large models affordable; FP8 is doing the same for the next.
- **Precedent for quantized inference.** The same thinking about range, scaling and sensitive layers informs [GPTQ, AWQ](../86-gptq-awq-quantization/summary.md) and [QLoRA](../22-qlora/summary.md), though those target inference and fine-tuning rather than full training.

---

## Key Takeaways for Practitioners

1. **Use BF16 if your hardware supports it** (A100, H100, TPUs and later). It needs no loss scaling and is the modern default.
2. **On FP16-only hardware, use dynamic loss scaling** via your framework's automatic mixed precision tools; do not hand-pick a constant.
3. **Keep master weights and optimizer state in FP32.** This is the part of the recipe that has never changed.
4. **Keep reductions in FP32**: softmax sums, normalization statistics, loss computation.
5. **FP8 is real but needs care.** Use fine-grained scaling and a well-tested library such as NVIDIA Transformer Engine, and watch for loss divergence compared with a BF16 baseline.

---

## Limitations & Future Directions

- **Loss scaling is a workaround** for FP16's narrow range; BF16 made it largely unnecessary.
- **Master weights cost memory**, which is part of why optimizer state sharding ([ZeRO](../76-zero-megatron/summary.md)) was needed.
- **Evaluated on 2017-scale models.** The largest here were about 200M parameters; stability at billions of parameters needed further work (BF16, careful normalization, gradient clipping).
- **FP8 and lower.** Formats below 8 bits (FP6, FP4 and microscaling formats in newer hardware) are active research as of 2026; the open question is how much of training, not just inference, can move there.

---

## Further Reading

- **Original Paper:** [arxiv.org/abs/1710.03740](https://arxiv.org/abs/1710.03740)
- **A Study of BFLOAT16 for Deep Learning Training:** [arxiv.org/abs/1905.12322](https://arxiv.org/abs/1905.12322)
- **FP8 Formats for Deep Learning:** [arxiv.org/abs/2209.05433](https://arxiv.org/abs/2209.05433)
- **DeepSeek-V3 Technical Report (FP8 training):** [arxiv.org/abs/2412.19437](https://arxiv.org/abs/2412.19437)
- **In this collection:** [DeepSeek-V3](../../language-models/27-deepseek-v3/summary.md), [ZeRO and Megatron-LM](../76-zero-megatron/summary.md), [Adam](../142-adam/summary.md), [FlashAttention](../16-flash-attention/summary.md), [GPTQ and AWQ](../86-gptq-awq-quantization/summary.md)

## Citation

```bibtex
@inproceedings{micikevicius2018mixed,
  title={Mixed Precision Training},
  author={Micikevicius, Paulius and Narang, Sharan and Alben, Jonah and Diamos, Gregory and Elsen, Erich and Garcia, David and Ginsburg, Boris and Houston, Michael and Kuchaiev, Oleksii and Venkatesh, Ganesh and Wu, Hao},
  booktitle={International Conference on Learning Representations},
  year={2018}
}
```

<!-- related:start -->

---

## Related in This Collection

- [Generative Adversarial Networks (GANs)](../../image-generation/02-generative-adversarial-networks/summary.md)
- [FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness](../../techniques/16-flash-attention/summary.md)
- [QLoRA: Efficient Finetuning of Quantized LLMs](../../techniques/22-qlora/summary.md)
- [DeepSeek-V3 Technical Report](../../language-models/27-deepseek-v3/summary.md)
- [Mixtral of Experts (and the Mixture-of-Experts Architecture)](../../architectures/37-mixture-of-experts/summary.md)
- [Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity](../../architectures/67-switch-transformer/summary.md)
- [Deep Residual Learning for Image Recognition (ResNet)](../../architectures/73-resnet/summary.md)
- [ZeRO and Megatron-LM: How Trillion-Parameter Models Are Actually Trained](../../techniques/76-zero-megatron/summary.md)

<!-- related:end -->
