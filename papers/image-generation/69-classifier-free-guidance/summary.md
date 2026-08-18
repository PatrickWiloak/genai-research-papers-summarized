---
title: "Classifier-Free Diffusion Guidance"
slug: "69-classifier-free-guidance"
number: 69
category: "image-generation"
authors: "Jonathan Ho, Tim Salimans (Google Research, Brain Team)"
published: "December 2021 (NeurIPS 2021 Workshop on Deep Generative Models); arXiv July 2022"
year: 2021
url: "https://arxiv.org/abs/2207.12598"
tags: ["image-generation", "diffusion", "guidance"]
---

# Classifier-Free Diffusion Guidance

**Authors:** Jonathan Ho, Tim Salimans (Google Research, Brain Team)
**Published:** December 2021 (NeurIPS 2021 Workshop on Deep Generative Models); arXiv July 2022
**Paper:** [arxiv.org/abs/2207.12598](https://arxiv.org/abs/2207.12598)

---

## Why This Matters

Classifier-free guidance is **the two-line trick that made text-to-image generation actually follow the prompt**. Every image model you have used - Stable Diffusion, DALL-E 3, Midjourney, Imagen, Flux - runs it at every sampling step. The "guidance scale" or "CFG scale" slider in every image UI is this paper.

- **Prompt adherence** - Without guidance, a diffusion model conditioned on text produces plausible images that only loosely match the prompt. With guidance, it locks on.
- **A quality/diversity dial** - One scalar knob trades sample diversity for fidelity to the condition, tunable at inference with no retraining.
- **No extra model** - It replaced classifier guidance, which required training a separate noise-aware classifier.
- **Universal** - The same trick now steers video (Sora), audio, and 3D diffusion models, and reappears in autoregressive image models.

**The insight:** you do not need a classifier to push samples toward a condition. A single diffusion model can be trained to be both conditional and unconditional, and the *difference* between its two predictions is itself a direction in noise space that points toward "more of this condition."

---

## The Problem: Guidance Used to Need a Second Model

[DDPM](../06-diffusion-models/summary.md) models learn to denoise. Conditioning them on a label or caption is easy - just feed the condition in - but the resulting samples are disappointingly weak on that condition. Diffusion models spread probability mass generously, so a "a corgi wearing sunglasses" prompt yields a lot of corgis with no sunglasses.

Dhariwal and Nichol's **classifier guidance** (2021) fixed this by training a separate image classifier on *noisy* images, then nudging each denoising step along the classifier's gradient toward the target class. It worked, and it is what let diffusion beat [GANs](../02-generative-adversarial-networks/summary.md) on ImageNet. But it has three problems:

1. You must train a second model, and on noisy data, which no off-the-shelf classifier provides.
2. It only works for conditions a classifier can score. Free-form text captions do not fit.
3. It is arguably cheating on the benchmark - improving Inception Score and FID using a classifier gradient, when those metrics are themselves classifier-based.

---

## The Core Innovation

Train **one** diffusion model that is conditional *and* unconditional, then take the difference at sampling time.

**Training.** During training, randomly replace the conditioning signal with a null token (an empty caption, a learned "no condition" embedding) some percentage of the time - typically 10 to 20 percent. The same network now learns two things at once: `eps(x_t, t, c)` when a condition is present and `eps(x_t, t, null)` when it is not.

**Sampling.** At each denoising step, run the model twice and extrapolate:

```
eps_guided = eps(x_t, t, null) + w * ( eps(x_t, t, c) - eps(x_t, t, null) )

where:
  eps(x_t, t, c)     = noise prediction given the prompt
  eps(x_t, t, null)  = noise prediction with no prompt
  w                  = guidance scale (the "CFG scale" in your UI)

w = 0   -> pure unconditional generation (ignores the prompt)
w = 1   -> plain conditional generation (no guidance)
w > 1   -> extrapolate PAST the conditional prediction, away from
           the unconditional one: exaggerate whatever the prompt added
```

The vector `eps(c) - eps(null)` is "everything the prompt changed about the prediction." Scaling it up amplifies exactly the prompt's contribution and nothing else. It is a direction, not a magnitude - which is why the same scale works across prompts.

---

## Key Components Explained

### 1. The Null Condition
**What it does:** Gives the model a well-defined "no information" input so the unconditional branch is a real, trained mode rather than an accident.
**How it works:** For text models, the null is usually the embedding of the empty string. Stable Diffusion caches this once and reuses it at every step. Because the null branch is trained with real data, it stays a competent denoiser - a broken unconditional branch produces a garbage guidance direction and visible artifacts.

### 2. Condition Dropout During Training
**What it does:** Trains both branches in one network with one loss.
**How it works:** Each training example independently drops its condition with probability p (10 percent is standard). Too low and the unconditional branch is undertrained; too high and conditional quality suffers. The cost is essentially zero: no extra parameters, no extra pass.

### 3. The Guidance Scale
**What it does:** Trades diversity for prompt fidelity at inference time.
**How it works:** Low scales (1 to 3) produce varied, natural, sometimes off-prompt images. Mid scales (5 to 9) are the sweet spot for most text-to-image models; Stable Diffusion defaults to 7.5. High scales (15+) produce oversaturated, high-contrast, blown-out images because the extrapolation pushes the latent outside the data distribution. Later work added scale schedules, dynamic thresholding (Imagen), and rescaling tricks to push usable scales higher.

### 4. The 2x Inference Cost
**What it does:** Nothing good - it is the price.
**How it works:** Every step needs two forward passes. In practice they are batched together as one batch of size 2, so wall-clock cost is under 2x but memory doubles. This is a big share of why image generation is slower than it looks on paper, and it motivated distillation work (guidance distillation, [consistency models](https://arxiv.org/abs/2303.01469)) that bakes guidance into a single pass.

---

## Key Results

- On class-conditional ImageNet 64x64 and 128x128, classifier-free guidance matched or beat classifier guidance on the FID/Inception Score trade-off curve, using one model instead of two.
- The paper made explicit that guidance is a **trade-off curve, not a single point**: increasing `w` monotonically improves Inception Score (fidelity to class) while worsening FID past a point (loss of diversity). Every image model since reports results at a chosen guidance scale for this reason.
- Adoption is the real result. [Stable Diffusion](../07-stable-diffusion/summary.md), Imagen, DALL-E 2 and 3, Midjourney, Flux, and video models including [Sora](../44-sora-dit/summary.md) all ship classifier-free guidance as the default sampler behavior.

---

## Why This Was Revolutionary

- **Deleted a whole model from the pipeline.** No noisy classifier to train, maintain, or match to each new dataset.
- **Unlocked free-form text conditioning.** You cannot build a classifier over arbitrary captions, but you can drop a caption. This is the direct enabler of prompt-driven image generation.
- **Made quality a user-facing dial.** Guidance scale is one of the few inference-time knobs in generative modeling that meaningfully changes output character without retraining.
- **Generalized far past images.** Any conditional diffusion or flow model - audio, video, molecules, robot trajectories - gets the same trick for free.

---

## Real-World Impact

- **Every image UI.** The "CFG scale," "guidance," or "prompt strength" slider in Automatic1111, ComfyUI, Diffusers, Midjourney, and Firefly is `w` in the equation above.
- **Negative prompts.** A widely used extension: replace the null condition with a *negative* prompt embedding. The guidance direction then becomes "away from blurry, watermark, extra fingers" and toward the real prompt. This is pure classifier-free guidance with a different second branch, and it was not in the original paper.
- **Distillation targets.** Guidance-distilled and few-step models (LCM, SDXL Turbo, Flux Schnell) exist largely to remove the 2x cost this paper introduced.
- **Beyond diffusion.** Guidance-style extrapolation has been applied to autoregressive image models and to language model decoding (contrastive decoding, context-aware decoding) with the same "amplify the difference between two conditionings" logic.

---

## Key Takeaways for Practitioners

1. **7.5 is a default, not a law.** Photoreal prompts often want 4 to 7; illustration and heavily stylized prompts tolerate more. If your images look burnt and oversaturated, your guidance is too high before your prompt is wrong.
2. **Negative prompts are free guidance.** They cost nothing extra because you are already running the second branch. Use the slot.
3. **Guidance interacts with step count and scheduler.** Fewer sampling steps ([DDIM](../70-ddim/summary.md) at 20 steps) generally needs slightly lower guidance to avoid artifacts.
4. **Distilled few-step models often want guidance near 1.** Turbo/LCM-style models have guidance baked in; stacking more on top breaks them.
5. **If you fine-tune, keep dropping the condition.** Fine-tuning a model without condition dropout degrades its unconditional branch and quietly ruins guidance quality.

---

## Limitations & Future Directions

- **2x compute per step**, permanently, unless distilled away.
- **High scales leave the data manifold**, producing saturation and contrast artifacts. Imagen's dynamic thresholding and later "CFG rescale" methods are patches on this.
- **One global scale for the whole image and all timesteps** is crude. Later work varies guidance across timesteps (strong early, weak late) and across image regions.
- **It is a heuristic, not a principled sampler.** The guided distribution is not the true conditional distribution; it is a sharpened caricature of it. Work on flow-based samplers and [rectified flow](../72-flow-matching-sd3/summary.md) revisits this.

---

## Further Reading

- **Original Paper:** [arxiv.org/abs/2207.12598](https://arxiv.org/abs/2207.12598)
- **Classifier guidance (the predecessor):** [Diffusion Models Beat GANs on Image Synthesis](https://arxiv.org/abs/2105.05233)
- **Imagen (dynamic thresholding for high guidance):** [arxiv.org/abs/2205.11487](https://arxiv.org/abs/2205.11487)
- **In this collection:** [DDPM](../06-diffusion-models/summary.md), [Stable Diffusion](../07-stable-diffusion/summary.md), [DDIM](../70-ddim/summary.md), [ControlNet](../71-controlnet/summary.md)

## Citation

```bibtex
@article{ho2022classifier,
  title={Classifier-Free Diffusion Guidance},
  author={Ho, Jonathan and Salimans, Tim},
  journal={arXiv preprint arXiv:2207.12598},
  year={2022}
}
```

<!-- related:start -->

---

## Related in This Collection

- [Generative Adversarial Networks (GANs)](../../image-generation/02-generative-adversarial-networks/summary.md)
- [Denoising Diffusion Probabilistic Models (DDPM)](../../image-generation/06-diffusion-models/summary.md)
- [High-Resolution Image Synthesis with Latent Diffusion Models (Stable Diffusion)](../../image-generation/07-stable-diffusion/summary.md)
- [Sora and Diffusion Transformers (DiT): Video Generation as World Simulation](../../image-generation/44-sora-dit/summary.md)
- [DALL-E 3: Improving Image Generation with Better Captions](../../image-generation/48-dalle3/summary.md)
- [Denoising Diffusion Implicit Models (DDIM)](../../image-generation/70-ddim/summary.md)
- [ControlNet: Adding Conditional Control to Text-to-Image Diffusion Models](../../image-generation/71-controlnet/summary.md)
- [Flow Matching and Rectified Flow: The New Default for Image Generation (Stable Diffusion 3)](../../image-generation/72-flow-matching-sd3/summary.md)

<!-- related:end -->
