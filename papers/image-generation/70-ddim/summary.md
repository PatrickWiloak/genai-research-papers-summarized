---
title: "Denoising Diffusion Implicit Models (DDIM)"
slug: "70-ddim"
number: 70
category: "image-generation"
authors: "Jiaming Song, Chenlin Meng, Stefano Ermon (Stanford University)"
published: "October 2020 (ICLR 2021)"
year: 2020
url: "https://arxiv.org/abs/2010.02502"
tags: ["image-generation", "diffusion", "sampling", "inference-optimization"]
---

# Denoising Diffusion Implicit Models (DDIM)

**Authors:** Jiaming Song, Chenlin Meng, Stefano Ermon (Stanford University)
**Published:** October 2020 (ICLR 2021)
**Paper:** [arxiv.org/abs/2010.02502](https://arxiv.org/abs/2010.02502)

---

## Why This Matters

DDIM is **the paper that took diffusion sampling from 1,000 steps to 20** without retraining anything. [DDPM](../06-diffusion-models/summary.md) proved diffusion models could beat GANs on quality; DDIM made them usable.

- **10 to 50x faster sampling** - Comparable quality in 20 to 100 steps instead of 1,000.
- **Zero retraining** - It is a different *sampler* for an already-trained DDPM. Drop it in.
- **Deterministic generation** - The same seed and prompt give the same image, every time.
- **A real latent space** - Because sampling is deterministic, the input noise becomes a meaningful latent you can interpolate in and invert into. This is the foundation of image editing with diffusion.

**The insight:** DDPM's slow sampling comes from its forward process being Markovian - each step depends only on the last, so you have to walk every step back. But the DDPM *training objective* does not actually require the forward process to be Markovian. Swap in a non-Markovian process with the same marginals, and you can take long jumps at sampling time with the exact same trained network.

---

## The Problem: Diffusion Was Beautiful and Unusable

DDPM's results in 2020 were stunning and its sampling was hopeless. Generating one 256x256 image required 1,000 sequential network evaluations. On the hardware of the day that was minutes per image, versus milliseconds for a [GAN](../02-generative-adversarial-networks/summary.md). Nobody was going to build a product on that.

The obvious fix - "just take fewer steps" - does not work for DDPM. Its reverse process is derived from a specific Markov chain; skipping steps breaks the math and produces noise-flecked garbage. The step count was baked into the model at training time.

---

## The Core Innovation

DDIM observes that DDPM's training loss only depends on the **marginal** distributions `q(x_t | x_0)` - how noisy an image looks at time t given the original - and not on the joint forward trajectory. Many different forward processes share those same marginals. DDIM constructs a family of non-Markovian ones, parameterized by a noise level `sigma`.

```
DDIM update step (predict x_0, then re-noise to the next level):

  1. Predict the clean image from the current noisy one:
       x0_hat = ( x_t - sqrt(1 - a_t) * eps(x_t, t) ) / sqrt(a_t)

  2. Point back toward the next (less noisy) level:
       x_{t-1} = sqrt(a_{t-1}) * x0_hat
                 + sqrt(1 - a_{t-1} - sigma^2) * eps(x_t, t)
                 + sigma * z          (z ~ N(0, I))

  sigma chosen to match DDPM  -> stochastic, DDPM behavior
  sigma = 0                   -> fully DETERMINISTIC: no z, no randomness
```

Setting `sigma = 0` is the DDIM sampler. Every step becomes a deterministic function, the whole generation is an ordinary differential equation solve, and - critically - you can evaluate that ODE on a **subsequence** of timesteps. Take t = 1000, 950, 900, ... 0 instead of every integer. Twenty evaluations instead of a thousand.

---

## Key Components Explained

### 1. Non-Markovian Forward Processes
**What it does:** Decouples "how the model was trained" from "how many steps you sample."
**How it works:** DDIM defines the forward process to condition on both the previous step and the original image `x_0`. This keeps the per-step marginals identical to DDPM (so the trained network is still valid) while allowing the reverse process to skip.

### 2. The `eta` Parameter
**What it does:** Interpolates continuously between DDIM and DDPM.
**How it works:** Implementations expose `eta` where `eta = 0` is deterministic DDIM and `eta = 1` reproduces DDPM's stochastic sampler. Values in between add controlled randomness. In most diffusion libraries `DDIMScheduler(eta=0.0)` is the default.

### 3. Deterministic Sampling and the Latent Space
**What it does:** Turns the initial noise `x_T` into a genuine latent code for the image.
**How it works:** With no randomness in the loop, `x_T -> x_0` is a bijection-like map. Two consequences follow immediately:
- **Interpolation.** Spherically interpolate (slerp) between two noise vectors and you get a smooth semantic morph between the two images, the way GAN latent walks work. DDPM cannot do this - its randomness destroys the correspondence.
- **Inversion.** Run the ODE *backward* from a real image to recover the noise that would generate it. This is **DDIM inversion**, and it is the mechanism behind prompt-to-prompt editing, null-text inversion, and most training-free image editing methods.

### 4. Consistency Across Step Counts
**What it does:** Lets you preview cheaply and render expensively.
**How it works:** Because the trajectory approximates the same ODE, the image produced from a given seed at 20 steps is *recognizably the same image* as at 100 steps, just less refined. You can iterate on prompts at low step counts and then do a final high-step render.

---

## Key Results

- On CIFAR-10 and CelebA, DDIM at 50 steps matched DDPM's 1,000-step FID; at 20 steps it was close, while DDPM at 20 steps was badly degraded.
- Speedups of **10x to 50x** in wall-clock sampling, from the same checkpoint, with no additional training.
- Demonstrated semantically meaningful interpolation in the noise space - a property diffusion models were previously assumed to lack relative to GANs and [VAEs](../57-vae/summary.md).

---

## Why This Was Revolutionary

- **Reframed sampling as ODE solving.** Once generation is an ODE, the entire numerical-methods literature applies. DPM-Solver, DPM-Solver++, UniPC, Heun, and the Euler/DPM samplers in every image UI are direct descendants that push quality sampling down to 10 to 20 steps.
- **Decoupled training from inference cost.** This is now taken for granted; in 2020 it was not.
- **Created the editing toolkit.** Deterministic invertibility is what makes "change the cat to a dog, keep everything else" possible without retraining.
- **Made diffusion products viable.** Combined with latent-space operation ([Stable Diffusion](../07-stable-diffusion/summary.md)), few-step sampling is why you get an image in seconds instead of minutes.

---

## Real-World Impact

- **Every diffusion library ships it.** Hugging Face Diffusers, ComfyUI, Automatic1111 - the sampler dropdown is a list of DDIM's descendants, and DDIM itself is still there.
- **Image-to-image and inpainting** start from a partially noised real image, which relies on the deterministic correspondence DDIM established.
- **Editing methods.** Prompt-to-prompt, Null-text Inversion, InstructPix2Pix training data, and most "edit this real photo" pipelines use DDIM inversion.
- **Distillation.** Progressive distillation and [consistency models](https://arxiv.org/abs/2303.01469) distill the DDIM ODE trajectory into 1 to 4 step samplers. You cannot distill a trajectory that is not deterministic, so DDIM is a prerequisite.
- **Video.** [Sora and DiT](../44-sora-dit/summary.md)-family video models sample with DDIM-style solvers; at video scale, a 50x step reduction is the difference between feasible and not.

---

## Key Takeaways for Practitioners

1. **Step count is a quality/latency dial, not a model property.** Start at 20 to 30 steps with a modern solver; go higher only if you can see the difference.
2. **Use `eta = 0` when you need reproducibility.** Same seed, same prompt, same image - required for A/B testing prompts, for regression tests, and for any editing workflow.
3. **Use a small `eta` when outputs look too "clean" or repetitive.** A little stochasticity restores variety.
4. **DDIM inversion is lossy at high guidance.** Inverting an image generated with [classifier-free guidance](../69-classifier-free-guidance/summary.md) at scale 7.5 does not round-trip cleanly; null-text inversion exists specifically to fix this.
5. **Prefer DPM-Solver++ or UniPC in production.** They are DDIM's better-conditioned successors and hit good quality at 10 to 20 steps.

---

## Limitations & Future Directions

- **Still multi-step.** 20 steps is far better than 1,000 but still 20x the cost of a single-pass generator. Consistency models, rectified flow, and adversarial distillation attack this.
- **Quality drops below about 10 steps** with the plain DDIM solver; higher-order solvers extend the usable range.
- **Inversion accumulates error** over the reverse trajectory, especially with strong guidance.
- **Superseded conceptually by flow matching.** [Rectified flow and Stable Diffusion 3](../72-flow-matching-sd3/summary.md) train models whose ODE trajectories are straight by construction, so few-step sampling works without the extra machinery.

---

## Further Reading

- **Original Paper:** [arxiv.org/abs/2010.02502](https://arxiv.org/abs/2010.02502)
- **DPM-Solver++ (fast higher-order sampler):** [arxiv.org/abs/2211.01095](https://arxiv.org/abs/2211.01095)
- **Score-Based Generative Modeling through SDEs (the ODE/SDE view):** [arxiv.org/abs/2011.13456](https://arxiv.org/abs/2011.13456)
- **Prompt-to-Prompt (editing via inversion):** [arxiv.org/abs/2208.01626](https://arxiv.org/abs/2208.01626)

## Citation

```bibtex
@inproceedings{song2021denoising,
  title={Denoising Diffusion Implicit Models},
  author={Song, Jiaming and Meng, Chenlin and Ermon, Stefano},
  booktitle={International Conference on Learning Representations},
  year={2021}
}
```

<!-- related:start -->

---

## Related in This Collection

- [Generative Adversarial Networks (GANs)](../../image-generation/02-generative-adversarial-networks/summary.md)
- [Denoising Diffusion Probabilistic Models (DDPM)](../../image-generation/06-diffusion-models/summary.md)
- [High-Resolution Image Synthesis with Latent Diffusion Models (Stable Diffusion)](../../image-generation/07-stable-diffusion/summary.md)
- [Sora and Diffusion Transformers (DiT): Video Generation as World Simulation](../../image-generation/44-sora-dit/summary.md)
- [Auto-Encoding Variational Bayes (VAE)](../../image-generation/57-vae/summary.md)
- [Classifier-Free Diffusion Guidance](../../image-generation/69-classifier-free-guidance/summary.md)
- [Flow Matching and Rectified Flow: The New Default for Image Generation (Stable Diffusion 3)](../../image-generation/72-flow-matching-sd3/summary.md)

<!-- related:end -->
