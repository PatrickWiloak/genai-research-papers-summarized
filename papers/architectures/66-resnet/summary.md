# Deep Residual Learning for Image Recognition (ResNet)

**Authors:** Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun (Microsoft Research)

**Published:** December 2015 (CVPR 2016 — Best Paper Award)

**Paper Link:** https://arxiv.org/abs/1512.03385

---

## Why This Paper Matters

ResNet is one of the most cited papers in the history of deep learning. Before ResNet, training very deep neural networks didn't work: adding more layers actually made models perform **worse**, even on training data. ResNet introduced a deceptively simple idea — **skip connections** that let signals bypass layers — that made it possible to train networks of unprecedented depth (152 layers, and later over 1000). The paper won ImageNet 2015 in a near-clean sweep, dropped error rates dramatically, and forever changed how neural networks are built.

The same residual connections it introduced are now found in **every Transformer block** (#1, #3, #4, #11, #68), every modern CNN, every diffusion U-Net (#6, #7), and basically every deep model trained today. If you've used a deep neural network in the last decade, you've almost certainly used ResNet's core idea.

---

## The Problem: Deep Networks Got Worse

By 2015, deep learning had been making rapid progress with networks like AlexNet (8 layers), VGG (19 layers), and GoogLeNet (22 layers). The accepted wisdom was: **deeper is better**. More layers = more capacity = better learned features.

But there was a strange empirical phenomenon. When the authors tried training a plain 56-layer CNN, it performed **worse** than a 20-layer version — not just on test data (which would suggest overfitting) but on the **training data itself**.

```
Plain CNN:
  20 layers: ~9% training error
  56 layers: ~12% training error   ← worse on training set!
```

This is the **degradation problem**. It is NOT overfitting (which would show better training error but worse test error). The deeper model simply could not fit the training data as well as the shallower one.

### Why Did This Happen?

In theory, a 56-layer network can always represent at least what the 20-layer network does — just set the extra 36 layers to be identity functions. So adding layers shouldn't hurt.

In practice, **optimizers struggle to find this solution**. The extra layers can't easily learn to be identity functions. Gradients get noisy or vanish across many layers. The optimization landscape becomes treacherous.

---

## The Core Innovation: Residual Learning

Instead of asking each layer to learn the desired underlying mapping `H(x)` directly, ask it to learn the **residual** `F(x) = H(x) - x`, and then add `x` back.

```
Standard block:    output = F(x)
Residual block:    output = F(x) + x
```

The little `+ x` is called a **skip connection** or **identity shortcut**. The signal `x` bypasses the layers and is added to their output.

### Diagram

```
        x
        │
        ├─────────────┐
        ▼             │ identity skip
   [Conv 3x3]         │
   [BatchNorm]        │
   [ReLU]             │
   [Conv 3x3]         │
   [BatchNorm]        │
        │             │
        ▼             │
       (+)◄───────────┘
        │
        ▼
      ReLU
        │
        ▼
    output = F(x) + x
```

### Why This Helps

1. **Identity becomes the default.** If the optimal thing for a block to do is "nothing," it just needs to push `F(x)` toward zero — which is easy, since weights are initialized small. Without the skip, the block would have to learn weights that exactly reconstruct the identity function — hard.

2. **Gradients flow freely.** During backprop, gradients can travel through the skip connection unchanged. Even if `F`'s gradient vanishes, the skip carries the gradient backward. This addresses vanishing gradients in deep nets.

3. **Easier optimization landscape.** Empirically, residual networks have smoother loss landscapes than their plain counterparts, which makes SGD-style optimizers more effective.

---

## Architecture Details

### Basic Block vs Bottleneck Block

For shallower ResNets (18, 34 layers), each block has two 3×3 convolutions:

```
3×3 conv, F filters
3×3 conv, F filters
```

For deeper ResNets (50, 101, 152 layers), the **bottleneck block** is used to save compute:

```
1×1 conv, F filters         (reduce dimension)
3×3 conv, F filters         (do work)
1×1 conv, 4F filters        (expand dimension back)
```

### ResNet-50 / 101 / 152

| Model | Layers | Top-1 ImageNet Error |
|-------|--------|----------------------|
| ResNet-18 | 18 | 30.43% |
| ResNet-34 | 34 | 26.73% |
| ResNet-50 | 50 | 24.7% |
| ResNet-101 | 101 | 23.6% |
| **ResNet-152** | **152** | **23.0%** |

The **152-layer ResNet** was the headline model — 8× deeper than VGG, yet **still trainable** thanks to residual connections.

### Handling Dimension Changes

When a block changes the number of filters or spatial size, the skip connection can't just be the identity. The paper offers two options:
- **Zero-pad** the extra channels (parameter-free)
- **1×1 convolution** to project to the right shape (more flexible, slightly more parameters)

Both work; the projection version usually wins by a small margin.

---

## Key Results

### ImageNet 2015 Sweep
- **ImageNet Classification:** ResNet-152 ensemble achieved **3.57% top-5 error** — a huge improvement over VGG (7.3%) and GoogLeNet (6.7%).
- **ImageNet Detection (COCO):** Won 1st place with 28% relative improvement over the previous year.
- **ImageNet Localization:** Won 1st place.
- **COCO Detection and Segmentation:** Won 1st place in both.

A clean sweep of **all five major tracks** of the 2015 ImageNet/COCO competitions.

### The Degradation Problem, Solved
Plain network vs ResNet at increasing depths:
- 18-layer plain: 27.94% training error → 18-layer ResNet: 27.88%
- 34-layer plain: 28.54% training error → 34-layer ResNet: **25.03%** (better, not worse!)

The deeper ResNet outperforms the shallower one — adding layers now helps.

### CIFAR-10 with Ridiculous Depth
The paper also experimented with very deep ResNets on CIFAR-10, training networks of **110 layers** and even **1202 layers**. Performance kept improving up to ~110 layers, with diminishing returns beyond. Before ResNet, training a 100-layer network was simply impossible.

---

## Why This Was Revolutionary

### 1. Made Depth Practical
Before ResNet, "very deep" meant ~20-30 layers. After ResNet, hundreds of layers became routine, and the field could finally test the hypothesis that depth matters.

### 2. A Tiny Idea, Universally Applicable
The skip connection adds no parameters and almost no compute. It is a structural change to network topology, not a new layer type. Because it is so general, it was immediately applied everywhere:
- **DenseNet (2016):** Skip connections from every layer to every later layer.
- **U-Net (2015):** Skip connections across the encoder-decoder bridge (used in diffusion models).
- **Transformer (2017):** Residual connections around every self-attention and feedforward sub-layer.
- **Highway Networks, ResNeXt, Wide ResNets:** Many variants.

### 3. Suddenly Optimization Was Tractable
Residual learning reframed the role of each layer: not "compute everything from scratch" but "compute a small correction to what we already have." This perspective made deep networks behave like ensembles of shallower paths, which empirically optimize more gracefully.

### 4. Unblocked the Field
Many subsequent breakthroughs implicitly required depths only achievable with residual connections. Without ResNet, the Transformer's 96+ stacked attention layers in GPT-3, the U-Net depths used in diffusion models, and modern computer vision pipelines simply could not have existed.

---

## Residual Connections in the Transformer

This is the most consequential downstream use. Every Transformer block looks like:

```
x → MultiHeadAttention(LayerNorm(x))   + x      # residual
x → FeedForward(LayerNorm(x))           + x      # residual
```

The two `+ x` operations are direct lifts from ResNet. Without them, stacking 96 attention layers in GPT-3 or 175 in GPT-4-scale models would not converge. Every time you use ChatGPT, Claude, or any modern LLM, ResNet's skip connections are silently doing their job inside every layer.

---

## Limitations and Subsequent Refinements

### 1. Batch Normalization Dependency
The original ResNet relied heavily on Batch Normalization. Later work (LayerNorm, GroupNorm, "Fixup initialization") explored ways to remove this dependency, particularly for small-batch or sequence-modeling regimes.

### 2. Block Design Wasn't Optimal
- **Pre-activation ResNet** (He et al., 2016): Put BatchNorm + ReLU **before** the convolutions for slightly better gradient flow.
- **ResNeXt:** Use grouped convolutions inside each block.
- **Wide ResNet:** Make blocks wider, not deeper.

### 3. Still a CNN
ResNet was eventually surpassed on ImageNet by Vision Transformers (#11), which retain the residual connection idea but replace convolutions with self-attention.

---

## Impact and Legacy

### Direct Use
ResNets remained the **default image classification backbone** from 2015 until Vision Transformers caught up around 2021. Even today, ResNet-50 is a common baseline and a workhorse model for transfer learning, object detection, and feature extraction.

### Citations
ResNet is one of the **most-cited papers in all of science** — over 200,000 citations as of 2025, ranking it among the most cited papers of the 21st century in any field.

### Architectural DNA
Residual connections appear in:
- **All Transformers** (#1, #3, #4, #11, #15, #68, #69, etc.)
- **U-Nets** for diffusion (#6, #7) and image segmentation
- **DenseNet, Inception-ResNet, MobileNetV2, EfficientNet**
- **AlphaFold and AlphaFold 2** (protein structure prediction)
- **Whisper, Wav2Vec** (speech models)
- **MuZero, AlphaZero** (reinforcement learning)

It is hard to find a modern deep learning paper that does not implicitly use this idea.

---

## Connections to Other Papers

- **Attention Is All You Need (#1):** Every Transformer block uses ResNet-style residual connections around its attention and feedforward sub-layers — the depth of modern LLMs is only possible because of this.
- **Vision Transformer (#11):** Also uses residual connections in every block; eventually replaced ResNet as the state-of-the-art image classifier.
- **Diffusion Models (#6) and Stable Diffusion (#7):** The U-Net backbone of diffusion models is built from residual blocks.
- **BERT (#3) and GPT-3 (#4):** All Transformer-based LLMs rely on residual connections to train deep stacks.
- **LLaMA (#15):** Same residual-block-based architecture as all modern LLMs, with various tweaks (RMSNorm, etc.) layered on top.
- **GPT-1 (#69):** First decoder-only Transformer, already using ResNet-style residual connections inherited from the Transformer.
- **MAE (#74):** ResNet-style residual blocks underpin the ViT encoder used by Masked Autoencoders.

---

## Key Takeaways

1. **The degradation problem is real:** Naively adding layers to a deep network makes it harder to train, not just to generalize.
2. **Residual learning reframes the task:** Learning `F(x) = H(x) - x` is easier than learning `H(x)` directly, especially when the right answer is close to the identity.
3. **Skip connections are essentially free:** Zero parameters, negligible compute, massive improvement in trainability.
4. **Depth finally became useful:** ResNet made networks of 100+ layers practical, opening the door to today's massive Transformers.
5. **A universal building block:** Residual connections are now found in nearly every modern deep architecture, from image classifiers to language models to diffusion models to protein folders.
