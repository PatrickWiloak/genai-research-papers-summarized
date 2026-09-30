---
title: "Adam: A Method for Stochastic Optimization (Adam)"
slug: "142-adam"
number: 142
category: "techniques"
authors: "Diederik P. Kingma, Jimmy Lei Ba (University of Amsterdam and OpenAI; University of Toronto)"
published: "December 2014 (ICLR 2015)"
year: 2014
url: "https://arxiv.org/abs/1412.6980"
tags: ["optimization", "training"]
---

# Adam: A Method for Stochastic Optimization (Adam)

**Authors:** Diederik P. Kingma, Jimmy Lei Ba (University of Amsterdam and OpenAI; University of Toronto)
**Published:** December 2014 (ICLR 2015)
**Paper:** [arxiv.org/abs/1412.6980](https://arxiv.org/abs/1412.6980)

---

## Why This Matters

Adam is **the default way neural networks learn**. Almost every large model in this collection, from [BERT](../../language-models/03-bert/summary.md) and GPT-3 to [Llama](../../language-models/15-llama/summary.md) and [DeepSeek-V3](../../language-models/27-deepseek-v3/summary.md), was trained with Adam or its weight-decay fix, **AdamW**. When ICLR gave Adam its 2025 Test of Time award, the organisers noted that Adam or AdamW is mentioned in over half of that year's ICLR papers.

- **Works out of the box.** The default settings (learning rate 0.001, beta1 = 0.9, beta2 = 0.999, epsilon = 1e-8) are good enough for a remarkable range of problems.
- **Per-parameter step sizes.** Each weight gets its own effective learning rate, adapted from the history of its gradients.
- **Cheap.** Two extra numbers stored per parameter and a few element-wise operations per step.
- **Robust to noisy, sparse and non-stationary gradients**, which describes most deep learning.

**The insight:** combine two ideas that each fixed half the problem. **Momentum** smooths the direction of travel by averaging recent gradients. **RMSProp**-style scaling divides each parameter's step by a running estimate of its gradient size, so steep and shallow directions move at comparable speeds. Adam does both, and adds a **bias correction** so the early steps are the right size.

---

## The Problem: One Learning Rate Does Not Fit All Weights

Plain stochastic gradient descent (SGD) updates every weight by the same learning rate times its gradient:

```
  theta  <-  theta - lr * g
```

That fails in predictable ways:

- **Mismatched scales.** Some parameters get large gradients and some tiny ones. A learning rate small enough to be stable for the first is painfully slow for the second.
- **Noise.** Minibatch gradients are noisy estimates, so steps jitter.
- **Sparse features.** In language and recommendation models, some parameters get non-zero gradients rarely and need bigger steps when they do.

Earlier adaptive methods each solved part of this. **AdaGrad** (2011) scaled steps by the accumulated sum of squared gradients, good for sparse data but its steps shrink toward zero over time. **RMSProp** (Tieleman and Hinton, 2012, from a lecture rather than a paper) used an exponential moving average instead, fixing the shrinkage, but had no principled bias correction.

---

## The Core Innovation

Keep two exponential moving averages per parameter: the **first moment** (mean of the gradient) and the **second moment** (mean of the squared gradient). Step in the direction of the first, scaled down by the square root of the second.

```
  Adam, one step for every parameter (element-wise)

  g   = gradient at this step
  m   = beta1 * m + (1 - beta1) * g           # momentum: average direction
  v   = beta2 * v + (1 - beta2) * g^2         # average squared size

  m_hat = m / (1 - beta1^t)                   # bias correction
  v_hat = v / (1 - beta2^t)                   #   (m and v start at zero)

  theta = theta - lr * m_hat / (sqrt(v_hat) + eps)

  Defaults: lr = 0.001, beta1 = 0.9, beta2 = 0.999, eps = 1e-8
```

The ratio `m_hat / sqrt(v_hat)` is roughly a signal-to-noise ratio. When gradients consistently point the same way, it is near 1 and the parameter moves about `lr` per step. When gradients flip sign randomly, the average `m_hat` is small relative to `sqrt(v_hat)` and the step shrinks automatically. The paper calls this a natural form of step-size annealing.

---

## Key Components Explained

### 1. Bias Correction
**What it does:** Fixes the underestimate at the start of training.
**How it works:** `m` and `v` start at zero, so for the first steps they are biased toward zero, especially `v` with beta2 = 0.999 (it takes about 1,000 steps to warm up). Dividing by `1 - beta^t` removes the bias exactly for a constant gradient. Without it, `v` is too small early, steps are too large, and training can blow up. The paper shows this matters most when beta2 is close to 1, as in its variational autoencoder experiments.

### 2. Invariance to Gradient Scale
**What it does:** Makes the learning rate meaningful across problems.
**How it works:** Multiplying all gradients by a constant cancels in `m_hat / sqrt(v_hat)`. The effective step is bounded by roughly `lr`, so the learning rate sets a trust region in parameter space. This is much of why the defaults transfer so well.

### 3. AdaMax
**What it does:** A variant in the same paper using the infinity norm instead of the L2 norm for the second moment.
**How it works:** Replace `v` with a running maximum of `|g|`. It is simpler and needs no bias correction for that term. It is rarely used today, but the idea of swapping the norm reappears in later optimizers (see [Muon](../144-muon/summary.md)).

### 4. The Convergence Analysis
**What it does:** Gives a theoretical guarantee in the online convex setting.
**How it works:** The paper proves an O(sqrt(T)) regret bound. In 2018, Reddi, Kale and Kumar ("[On the Convergence of Adam and Beyond](https://arxiv.org/abs/1904.09237)", ICLR 2018) showed a simple convex problem on which Adam does not converge, pointing to a flaw in the original proof, and proposed AMSGrad as a fix. In practice AMSGrad rarely beats Adam, and the gap between Adam's theory and its practice has been an active research topic since.

---

## Key Results

The experiments are modest by today's standards and span the main model types of 2014:

- **Logistic regression** on MNIST and on sparse IMDB bag-of-words features: Adam matched SGD with Nesterov momentum on the dense problem and matched AdaGrad on the sparse one, showing it handles both regimes.
- **Multilayer neural networks** on MNIST (with dropout): Adam converged faster than the other methods tested, including the quasi-Newton SFO optimizer, which was also 5-10x slower per iteration.
- **Convolutional networks** on CIFAR-10: Adam converged faster than AdaGrad and matched or beat SGD with momentum.
- **Bias-correction ablation** on a variational autoencoder: removing bias correction caused instability with beta2 near 1.

None of these is a headline benchmark. The paper's influence came from how reliably the method worked when others tried it.

---

## AdamW: The Variant Actually Used Today

"[Decoupled Weight Decay Regularization](https://arxiv.org/abs/1711.05101)" (Ilya Loshchilov and Frank Hutter, University of Freiburg, ICLR 2019) found a subtle bug in how Adam was used with regularization.

**Weight decay** shrinks every weight a little each step, pulling the model toward simpler solutions. With plain SGD, adding an L2 penalty `lambda * ||theta||^2` to the loss is mathematically the same as weight decay. **With Adam it is not.** The L2 gradient `lambda * theta` goes into `m` and `v` and is then divided by `sqrt(v_hat)`, so weights with large gradient history are barely regularized and weights with small gradients are over-regularized.

```
  Adam + L2 (what libraries did):
    g = grad(loss) + lambda * theta           # decay mixed into the gradient
    ... usual Adam update with g ...          # then rescaled per parameter

  AdamW (decoupled):
    g = grad(loss)                            # gradient of the loss only
    ... usual Adam update with g ...
    theta = theta - lr * lambda * theta       # decay applied directly, unscaled
```

Decoupling makes the best weight decay value roughly independent of the learning rate and improved Adam's generalization on image classification, closing much of the gap with SGD. PyTorch and TensorFlow added AdamW implementations, and it became the standard for transformers. A typical LLM pretraining recipe as of 2024-2025 is AdamW with beta1 = 0.9, beta2 = 0.95 (lower than Adam's default, for stability at scale), weight decay around 0.1, a warmup period and cosine or similar decay, and gradient clipping; Llama, for example, reports AdamW with beta2 = 0.95.

---

## Why This Was Revolutionary

- **Made optimization mostly a solved default.** Researchers could change architectures without re-tuning the optimizer from scratch.
- **Enabled the transformer era.** Transformers are notoriously hard to train with plain SGD; Adam-family optimizers (with warmup) made them practical.
- **Simple enough to be everywhere.** A few lines of code, in every framework, with defaults that work.
- **Set the baseline every new optimizer must beat.** A decade of proposed replacements mostly failed to displace it at scale, which is why [Muon](../144-muon/summary.md)'s results in 2024-2025 drew attention.

---

## Real-World Impact

- **Standard in large-scale training**: GPT-style models, [Llama](../../language-models/15-llama/summary.md), [CLIP](../../multimodal/08-clip/summary.md), diffusion models and most fine-tuning recipes use AdamW.
- **Memory is a design constraint.** Adam stores two extra values per parameter, usually in 32-bit precision. For large models the optimizer state is bigger than the weights, which is what [ZeRO](../76-zero-megatron/summary.md) sharding and 8-bit optimizer states (used in [QLoRA](../22-qlora/summary.md)-style fine-tuning) attack.
- **Interacts with precision.** [Mixed precision training](../143-mixed-precision-training/summary.md) keeps Adam's state and master weights in higher precision for stability.

---

## Key Takeaways for Practitioners

1. **Use AdamW, not Adam with L2**, whenever you want weight decay. In PyTorch that is `torch.optim.AdamW`.
2. **For large transformers, lower beta2 to about 0.95** and use warmup; the 0.999 default can be unstable at scale.
3. **Do not apply weight decay to everything.** Biases, normalization parameters and often embeddings are usually excluded.
4. **Budget memory for optimizer state**: roughly two extra copies of the parameters, plus a master copy in mixed precision.
5. **Tune the learning rate first.** Adam's defaults are robust, but the learning rate and its schedule still matter most.

---

## Limitations & Future Directions

- **Generalization gap.** In some vision settings, SGD with momentum generalizes better than Adam; AdamW narrowed but did not always close this.
- **Memory cost** of two state tensors per parameter.
- **Theory lags practice.** The original convergence proof was flawed, and why Adam works so well on deep networks is still only partly understood.
- **Element-wise, not matrix-aware.** Adam treats every weight independently, ignoring the structure of weight matrices. Matrix-aware optimizers such as Shampoo and [Muon](../144-muon/summary.md) exploit that structure and reported meaningful speedups in 2024-2025.

---

## Further Reading

- **Original Paper:** [arxiv.org/abs/1412.6980](https://arxiv.org/abs/1412.6980)
- **AdamW (Decoupled Weight Decay Regularization):** [arxiv.org/abs/1711.05101](https://arxiv.org/abs/1711.05101)
- **On the Convergence of Adam and Beyond (AMSGrad):** [arxiv.org/abs/1904.09237](https://arxiv.org/abs/1904.09237)
- **ICLR 2025 Test of Time announcement:** [blog.iclr.cc/2025/04/14/announcing-the-test-of-time-award-winners-from-iclr-2015](https://blog.iclr.cc/2025/04/14/announcing-the-test-of-time-award-winners-from-iclr-2015/)
- **In this collection:** [Muon](../144-muon/summary.md), [Mixed Precision Training](../143-mixed-precision-training/summary.md), [ZeRO and Megatron-LM](../76-zero-megatron/summary.md), [QLoRA](../22-qlora/summary.md)

## Citation

```bibtex
@inproceedings{kingma2015adam,
  title={Adam: A Method for Stochastic Optimization},
  author={Kingma, Diederik P. and Ba, Jimmy},
  booktitle={International Conference on Learning Representations},
  year={2015}
}

@inproceedings{loshchilov2019decoupled,
  title={Decoupled Weight Decay Regularization},
  author={Loshchilov, Ilya and Hutter, Frank},
  booktitle={International Conference on Learning Representations},
  year={2019}
}
```

<!-- related:start -->

---

## Related in This Collection

- [BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](../../language-models/03-bert/summary.md)
- [Language Models are Few-Shot Learners (GPT-3)](../../language-models/04-gpt3-few-shot-learners/summary.md)
- [Learning Transferable Visual Models From Natural Language Supervision (CLIP)](../../multimodal/08-clip/summary.md)
- [QLoRA: Efficient Finetuning of Quantized LLMs](../../techniques/22-qlora/summary.md)
- [DeepSeek-V3 Technical Report](../../language-models/27-deepseek-v3/summary.md)
- [Auto-Encoding Variational Bayes (VAE)](../../image-generation/57-vae/summary.md)
- [ZeRO and Megatron-LM: How Trillion-Parameter Models Are Actually Trained](../../techniques/76-zero-megatron/summary.md)
- [Mixed Precision Training](../../techniques/143-mixed-precision-training/summary.md)

<!-- related:end -->
