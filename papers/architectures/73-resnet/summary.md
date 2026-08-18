---
title: "Deep Residual Learning for Image Recognition (ResNet)"
slug: "73-resnet"
number: 73
category: "architectures"
authors: "Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun (Microsoft Research)"
published: "December 2015 (CVPR 2016, Best Paper Award)"
year: 2015
url: "https://arxiv.org/abs/1512.03385"
tags: ["architecture", "vision", "computer-vision"]
---

# Deep Residual Learning for Image Recognition (ResNet)

**Authors:** Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun (Microsoft Research)
**Published:** December 2015 (CVPR 2016, Best Paper Award)
**Paper:** [arxiv.org/abs/1512.03385](https://arxiv.org/abs/1512.03385)

---

## Why This Matters

ResNet is **the reason deep networks can be deep**. Every [Transformer](../01-attention-is-all-you-need/summary.md) block, every diffusion [U-Net](../74-unet/summary.md) block, every layer of every model in this collection contains a residual connection, and they all trace back to this paper. It is the single most-cited paper in deep learning.

- **Trained 152 layers** when the previous state of the art topped out around 20, and won ILSVRC 2015 with 3.57 percent top-5 ImageNet error - below the commonly cited human benchmark.
- **Swept every 2015 competition** it entered: ImageNet classification, detection, and localization, plus COCO detection and segmentation.
- **The `x + f(x)` pattern is now universal.** GPT, Claude, Llama, Stable Diffusion, AlphaFold, Whisper - all of them are stacks of residual blocks.
- **It changed what "depth" costs.** Before ResNet, adding layers eventually made models *worse*. After ResNet, depth became a dial you could turn.

**The insight:** if a deeper network is at least as expressive as a shallower one, the extra layers only need to learn the identity function to break even - and neural networks are surprisingly bad at learning the identity. So restructure the layer to compute a *residual* on top of an identity that is wired in for free.

---

## The Problem: The Degradation Problem

By 2015 the field knew depth mattered - AlexNet at 8 layers, VGG at 19, GoogLeNet at 22 - and it knew that stacking more layers stopped helping. The obvious suspects were vanishing gradients and overfitting, and both had been partially addressed (better initialization, batch normalization). Yet the problem persisted, and it was stranger than overfitting:

```
Plain networks on CIFAR-10:

  20 layers  ->  TRAINING error: lower
  56 layers  ->  TRAINING error: HIGHER

The deeper network is worse on the TRAINING set.
This is not overfitting. It is an optimization failure.
```

This is the **degradation problem**. It is logically absurd on its face: a 56-layer network can express everything a 20-layer network can, by setting the extra 36 layers to the identity. A solution exists; gradient descent just cannot find it. The paper's contribution is a reparameterization that makes that solution trivially findable.

---

## The Core Innovation

Instead of asking a block of layers to learn a target mapping `H(x)`, ask it to learn the *difference* from the input, and add the input back:

```
Plain block:                Residual block:

   x                            x -----------+
   |                            |            |
 [conv]                       [conv]         | identity
   |                            |            | shortcut
 [relu]                       [relu]         |
   |                            |            |
 [conv]                       [conv]         |
   |                            |            |
   v                            + <----------+
  H(x)                          |
                                v
                            F(x) + x
```

Now `H(x) = F(x) + x`, so the block learns `F(x) = H(x) - x`, the residual. If the optimal behavior for this block is "do nothing," the network only has to drive the weights of `F` toward zero - which is exactly what weight decay and standard initialization already pull it toward. The identity is the default, not something to be discovered.

**Why the gradient story matters too.** During backpropagation, the addition node routes gradient to both branches unchanged. That means every residual connection provides a direct gradient path from the loss to early layers, bypassing all intervening weights:

```
d(loss)/dx  =  d(loss)/d(out) * ( 1 + dF/dx )
                                   ^
                                   the "1" is the shortcut.
                                   Even if dF/dx vanishes,
                                   gradient still flows.
```

This is why 100+ layer networks became trainable at all, and it is the same reason 100+ layer transformers train stably today.

---

## Key Components Explained

### 1. Identity Shortcut Connections
**What it does:** Adds the block's input to its output, with no parameters and no extra compute.
**How it works:** A plain element-wise addition. The paper tested learned (projection) shortcuts against pure identity ones and found identity shortcuts as good or better, and free. When dimensions change (at downsampling stages), a 1x1 convolution projects the shortcut to match - the only place parameters enter the skip path.

### 2. Bottleneck Blocks
**What it does:** Makes very deep networks affordable.
**How it works:** For ResNet-50 and deeper, each block is 1x1 conv (reduce channels) then 3x3 conv (the work) then 1x1 conv (restore channels). The 1x1 layers squeeze the expensive 3x3 convolution into a lower-dimensional space. ResNet-152 has roughly 8 times VGG-19's depth at *lower* computational cost. The same bottleneck idea reappears in [LoRA](../../techniques/10-lora/summary.md)'s low-rank decomposition and in transformer feed-forward design.

### 3. Batch Normalization Everywhere
**What it does:** Keeps activations well-scaled through great depth.
**How it works:** Every convolution is followed by batch norm before the nonlinearity. Residual connections and normalization are complementary: the shortcut fixes the optimization landscape, normalization fixes the activation statistics. Transformers keep both, substituting layer normalization for batch norm.

### 4. Pre-activation ResNet (the 2016 follow-up)
**What it does:** Makes the identity path completely clean.
**How it works:** The follow-up paper moved normalization and activation *before* the convolutions, so the shortcut path is pure addition end to end with nothing in the way. This enabled 1,000+ layer networks and is the layout modern transformers use (pre-norm residual blocks: `x + Attn(LN(x))`).

---

## Key Results

| Model | Layers | ImageNet top-5 error |
|---|---|---|
| VGG-16 | 16 | 8.4% |
| GoogLeNet | 22 | 7.9% |
| ResNet-34 | 34 | 5.7% |
| ResNet-152 | 152 | 4.5% (single model) |
| ResNet ensemble | - | **3.57% (ILSVRC 2015 winner)** |

- On CIFAR-10 the authors trained a 1,202-layer network successfully - it overfit, but it *trained*, which was the point.
- First place in all five main tracks of ILSVRC 2015 and COCO 2015.
- Deeper ResNets were consistently better on training error, confirming the degradation problem was optimization, not capacity.

---

## Why This Was Revolutionary

- **Turned depth from a liability into a resource.** The scaling story of the last decade - bigger, deeper models keep getting better - starts here.
- **Identified degradation as an optimization problem** rather than a capacity or regularization problem, and fixed it with a structural change instead of a training trick.
- **Gave every subsequent architecture its skeleton.** The residual block is the atom that transformers, U-Nets, and state-space models like [Mamba](../20-mamba/summary.md) are all built from.
- **Set the template for "make the easy thing the default."** ControlNet's zero convolutions, adapter layers, and gated architectures all apply the same design philosophy: initialize so that the new component starts as a no-op.

---

## Real-World Impact

- **Inside every model in this collection.** A transformer block is `x + Attention(Norm(x))` followed by `x + FFN(Norm(x))`. That is two residual blocks per layer, ResNet's idea verbatim.
- **[Vision Transformer](../11-vision-transformer/summary.md)** replaced ResNet's convolutions but kept its residual structure and, for years, was compared against ResNet baselines as the standard.
- **Diffusion models** are built from residual blocks inside a U-Net; [DiT](../../image-generation/44-sora-dit/summary.md) is residual blocks in a transformer.
- **[AlphaFold](../../techniques/68-alphafold/summary.md)**, [Whisper](../../multimodal/49-whisper/summary.md), [CLIP](../../multimodal/08-clip/summary.md) (whose original image encoder options included ResNets), and essentially every production vision system.
- **ResNet-50 is still the default vision backbone** for transfer learning, embedded deployment, and as the baseline every new architecture must beat.

---

## Key Takeaways for Practitioners

1. **If your deep model will not train, check that you have residual connections and normalization.** This fixes more instability than any learning-rate schedule.
2. **Initialize new modules to be near-identity.** Zero-initialized gates, small-scale output projections, and LoRA's zero-initialized B matrix are all "start as a no-op" applications of ResNet's logic.
3. **Pre-norm over post-norm for very deep stacks.** It keeps the residual highway clean and is why modern LLMs train stably at 100+ layers.
4. **Bottlenecks buy depth cheaply.** Reducing dimension, doing the expensive operation, then restoring dimension is a reliable compute/quality trade.
5. **ResNet-50 remains an excellent baseline.** Before reaching for a transformer on a small vision dataset, try a pretrained ResNet - it often wins on limited data.

---

## Limitations & Future Directions

- **Convolutional inductive bias.** ResNet assumes locality and translation equivariance, which is a strength on small data and a ceiling on large data. ViT surpassed it once datasets got big enough.
- **Depth has diminishing returns.** Past roughly 150 layers, accuracy gains largely stop; later work (Wide ResNet, EfficientNet) showed width and resolution scaling matter as much as depth.
- **The theory is still debated.** Why residual connections help is explained variously as gradient flow, ensemble-of-shallow-paths behavior, and loss-landscape smoothing. All have evidence; none is the complete story.
- **Residual streams create their own problems at scale**, including activation growth through depth, which motivated normalization placement changes and techniques like QK-norm in large transformers.

---

## Further Reading

- **Original Paper:** [arxiv.org/abs/1512.03385](https://arxiv.org/abs/1512.03385)
- **Identity Mappings in Deep Residual Networks (pre-activation):** [arxiv.org/abs/1603.05027](https://arxiv.org/abs/1603.05027)
- **Residual Networks Behave Like Ensembles of Shallow Networks:** [arxiv.org/abs/1605.06431](https://arxiv.org/abs/1605.06431)
- **Batch Normalization:** [arxiv.org/abs/1502.03167](https://arxiv.org/abs/1502.03167)

## Citation

```bibtex
@inproceedings{he2016deep,
  title={Deep Residual Learning for Image Recognition},
  author={He, Kaiming and Zhang, Xiangyu and Ren, Shaoqing and Sun, Jian},
  booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
  pages={770--778},
  year={2016}
}
```

<!-- related:start -->

---

## Related in This Collection

- [High-Resolution Image Synthesis with Latent Diffusion Models (Stable Diffusion)](../../image-generation/07-stable-diffusion/summary.md)
- [Learning Transferable Visual Models From Natural Language Supervision (CLIP)](../../multimodal/08-clip/summary.md)
- [LoRA: Low-Rank Adaptation of Large Language Models](../../techniques/10-lora/summary.md)
- [An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale (Vision Transformer)](../../architectures/11-vision-transformer/summary.md)
- [Mamba: Linear-Time Sequence Modeling with Selective State Spaces](../../architectures/20-mamba/summary.md)
- [Sora and Diffusion Transformers (DiT): Video Generation as World Simulation](../../image-generation/44-sora-dit/summary.md)
- [Whisper: Robust Speech Recognition via Large-Scale Weak Supervision](../../multimodal/49-whisper/summary.md)
- [Highly Accurate Protein Structure Prediction with AlphaFold (AlphaFold 2)](../../techniques/68-alphafold/summary.md)

<!-- related:end -->
