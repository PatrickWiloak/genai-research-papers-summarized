---
title: "Consistency Models (Consistency Models)"
slug: "120-consistency-models"
number: 120
category: "image-generation"
authors: "Yang Song, Prafulla Dhariwal, Mark Chen, Ilya Sutskever (OpenAI)"
published: "March 2023 (ICML 2023)"
year: 2023
url: "https://arxiv.org/abs/2303.01469"
tags: ["image-generation", "diffusion", "inference-optimization"]
---

# Consistency Models (Consistency Models)

**Authors:** Yang Song, Prafulla Dhariwal, Mark Chen, Ilya Sutskever (OpenAI)
**Published:** March 2023 (ICML 2023)
**Paper:** [arxiv.org/abs/2303.01469](https://arxiv.org/abs/2303.01469)

---

## Why This Matters

[Diffusion models](../06-diffusion-models/summary.md) produce excellent images, but slowly: they build an image by removing noise over tens to hundreds of steps, each one a full network pass. The paper notes this typically costs 10 to 2,000 times more compute than a one-shot generator like a GAN. Consistency models are **a new family of generative models built to jump from noise to data in a single step**, while keeping the option to take a few more steps for better quality.

- **One-step generation with diffusion-level quality,** at the time a new record: FID 3.55 on CIFAR-10 and 6.20 on ImageNet 64x64 with a single network evaluation (FID measures how close generated images are to real ones; lower is better).
- **Two ways to train:** distil a pre-trained diffusion model, or train from scratch as a standalone generative model.
- **Keeps diffusion's flexibility:** zero-shot inpainting, colourisation, super-resolution and editing without task-specific training.
- **Became the basis of fast image generation in practice.** Latent Consistency Models and LCM-LoRA brought 2 to 4 step generation to Stable Diffusion within months.

**The insight:** every point on a diffusion model's deterministic denoising path leads to the same clean image. Train a network to map *any* point on that path straight to its endpoint, and enforce that by requiring its outputs for neighbouring points on the same path to agree. That is the "consistency" in the name.

---

## The Problem: Diffusion Sampling Is Slow

A diffusion model can be viewed as defining a **probability flow ODE** (an ordinary differential equation whose solution smoothly transforms noise into data). Sampling means numerically solving that ODE from pure noise at time T back to clean data near time 0. [DDIM](../70-ddim/summary.md) and later solvers cut the step count, but, per the paper, still needed more than 10 steps for competitive samples.

The main alternative was **distillation**: train a student to imitate many teacher steps with fewer steps. Most methods needed an expensive pre-generated dataset of teacher samples. Progressive distillation (Salimans and Ho, 2022) avoided that by halving the step count repeatedly, but required many rounds of training and still lost quality at one step.

---

## The Core Innovation

Define a **consistency function** that maps any point on an ODE trajectory to the trajectory's origin:

```
f(x_t, t) = x_epsilon       for every t in [epsilon, T]

   noise                                       data
   x_T  ----->  x_t2  ----->  x_t1  ----->  x_epsilon
     \            \             \              ^
      \------------\-------------\-------------|
         f maps every point on the path to the same endpoint

Self-consistency:   f(x_t, t) = f(x_t', t')   for any t, t' on one path
Boundary condition: f(x_epsilon, epsilon) = x_epsilon
```

A network f_theta trained to have this property is a **consistency model**. Sampling is one line:

```
x_T ~ Normal(0, T^2 I)
image = f_theta(x_T, T)        # one network evaluation
```

For better quality, alternate: generate, add some noise back, and map to data again (multistep consistency sampling). Each extra step costs one more network evaluation.

The boundary condition is built into the architecture rather than learned:

```
f_theta(x, t) = c_skip(t) * x + c_out(t) * F_theta(x, t)

with c_skip(epsilon) = 1 and c_out(epsilon) = 0
```

This mirrors the skip parameterisation of the EDM diffusion models (Karras et al., 2022), so standard diffusion architectures can be reused.

---

## Key Components Explained

### 1. Consistency Distillation (CD)
**What it does:** Turns a pre-trained diffusion model into a one-step generator.
**How it works:** Take a training image, add noise to reach a point x at time t(n+1) on the path. Use the pre-trained diffusion model to take **one ODE solver step** back to time t(n), giving an estimate of the neighbouring point. Train the network so that its outputs at the two neighbouring points match:

```
loss = d( f_theta(x_{t(n+1)}, t(n+1)),  f_theta_minus(x_hat_{t(n)}, t(n)) )

theta_minus = exponential moving average (EMA) of theta, a slowly
              updated "target network", as in reinforcement learning
d           = distance function; LPIPS (a perceptual image distance) worked best
solver      = Heun's method worked best
```

Because agreement between neighbours propagates along the whole path, the model learns to jump from any point to the end. No teacher sample dataset is needed; each step uses only one teacher call.

### 2. Consistency Training (CT)
**What it does:** Trains a consistency model with no pre-trained diffusion model at all.
**How it works:** The teacher's ODE step is replaced by an unbiased estimate built from the training image itself: both neighbouring points are made by adding the same noise to the same clean image at two noise levels. The paper proves that this approximates the distillation loss as the step count grows, and uses a schedule that increases the number of discretisation steps during training. This makes consistency models a standalone family of generative models, not just a distillation trick.

### 3. Zero-shot editing
**What it does:** Keeps diffusion's inverse-problem abilities.
**How it works:** Multistep sampling alternates "denoise to data" and "re-noise," so you can inject constraints between steps, for example resetting known pixels (inpainting) or known low-resolution content (super-resolution). The paper shows inpainting, colourisation, super-resolution, denoising, interpolation and stroke-guided editing on a model trained only for generation.

---

## Key Results

```
Consistency Distillation (CD)       1 step    2 steps
----------------------------------  --------  --------
CIFAR-10 FID                        3.55      2.93
ImageNet 64x64 FID                  6.20      4.70

Consistency Training (CT, no teacher)
----------------------------------  --------  --------
CIFAR-10 FID                        8.70      5.83
```

- **CD beat progressive distillation** at one and few steps on CIFAR-10, ImageNet 64x64 and LSUN 256x256 (Bedroom and Cat), with both distilled from the same EDM teachers.
- **CT beat other one-step non-adversarial models** (VAEs and normalising flows) on CIFAR-10 and matched one-step progressive distillation without using any teacher. It still trailed the best GANs.
- Samples from CT and from the EDM diffusion model share structure when started from the same noise, which the authors read as evidence that CT does not suffer from mode collapse.

---

## Why This Was Revolutionary

- **Reframed fast sampling as learning a map, not solving an ODE faster.** This shifted the fast-generation research agenda.
- **Unified distillation and standalone training** under one objective.
- **Kept multistep flexibility.** Unlike GANs, you can trade compute for quality at inference time.
- **Brought real-time image generation into reach** once combined with latent diffusion.

---

## Real-World Impact

- **Latent Consistency Models (LCM, October 2023)** applied consistency distillation in Stable Diffusion's latent space, giving good images in 2 to 4 steps, and **LCM-LoRA (November 2023)** packaged it as a plug-in adapter for existing Stable Diffusion checkpoints. These powered a wave of near-real-time and interactive generation tools.
- **Improved Consistency Training (Song and Dhariwal, 2023)** made CT much stronger, removing the EMA teacher and replacing LPIPS with a Pseudo-Huber loss, closing much of the gap to distillation.
- **Continuous-time consistency models (sCM, Lu and Song, 2024)** simplified and stabilised training and scaled it to much larger image models.
- **Few-step distillation became standard.** As of 2026, most deployed image and video generators offer distilled few-step variants, using consistency-style objectives, adversarial distillation (as in SDXL Turbo) or combinations. The idea also sits close to one-step methods built on [flow matching](../72-flow-matching-sd3/summary.md), whose straighter paths are easier to shortcut.

---

## Key Takeaways for Practitioners

1. **If inference cost matters, distil.** A consistency-distilled model at 1 to 4 steps is often good enough and orders of magnitude cheaper than a 50-step sampler.
2. **Use 2 steps, not 1, when quality matters.** The drop from 3.55 to 2.93 FID on CIFAR-10 for one extra evaluation is typical of the trade-off.
3. **LCM-LoRA is the low-effort route** for existing Stable Diffusion pipelines.
4. **Be careful with LPIPS as a loss** if you then evaluate on ImageNet-like data; LPIPS uses ImageNet-trained features, and later work moved away from it.
5. **Distilled models copy the teacher's flaws.** Prompt following and failure modes carry over; distillation does not add capability.

---

## Limitations & Future Directions

- **One-step quality still trails multistep diffusion** and, for consistency training in the original paper, the best GANs.
- **Consistency training was fragile and weaker than distillation** in the original, which Improved Consistency Training and sCM largely addressed.
- **Sensitive to design choices:** distance metric, discretisation schedule, EMA rate and ODE solver all mattered in the ablations.
- **Evaluated on small images** (CIFAR-10, ImageNet 64x64, LSUN 256x256); scaling to text-to-image came through follow-up work such as LCM.
- **Distillation inherits teacher costs:** you still need a strong diffusion model to distil from.

---

## Further Reading

- **Original Paper:** [arxiv.org/abs/2303.01469](https://arxiv.org/abs/2303.01469)
- **Latent Consistency Models:** [arxiv.org/abs/2310.04378](https://arxiv.org/abs/2310.04378)
- **LCM-LoRA:** [arxiv.org/abs/2311.05556](https://arxiv.org/abs/2311.05556)
- **Improved Techniques for Training Consistency Models:** [arxiv.org/abs/2310.14189](https://arxiv.org/abs/2310.14189)
- **Continuous-time consistency models (sCM):** [arxiv.org/abs/2410.11081](https://arxiv.org/abs/2410.11081)
- **In this collection:** [Diffusion Models](../06-diffusion-models/summary.md), [DDIM](../70-ddim/summary.md), [Flow Matching and SD3](../72-flow-matching-sd3/summary.md), [Stable Diffusion](../07-stable-diffusion/summary.md), [Knowledge Distillation](../../techniques/134-knowledge-distillation/summary.md)

## Citation

```bibtex
@inproceedings{song2023consistency,
  title={Consistency Models},
  author={Song, Yang and Dhariwal, Prafulla and Chen, Mark and Sutskever, Ilya},
  booktitle={International Conference on Machine Learning (ICML)},
  year={2023}
}
```

<!-- related:start -->

---

## Related in This Collection

- [Generative Adversarial Networks (GANs)](../../image-generation/02-generative-adversarial-networks/summary.md)
- [High-Resolution Image Synthesis with Latent Diffusion Models (Stable Diffusion)](../../image-generation/07-stable-diffusion/summary.md)
- [Auto-Encoding Variational Bayes (VAE)](../../image-generation/57-vae/summary.md)
- [Denoising Diffusion Implicit Models (DDIM)](../../image-generation/70-ddim/summary.md)
- [Flow Matching and Rectified Flow: The New Default for Image Generation (Stable Diffusion 3)](../../image-generation/72-flow-matching-sd3/summary.md)
- [Distilling the Knowledge in a Neural Network (Knowledge Distillation)](../../techniques/134-knowledge-distillation/summary.md)

<!-- related:end -->
