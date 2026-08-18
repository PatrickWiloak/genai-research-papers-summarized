---
title: "Flow Matching and Rectified Flow: The New Default for Image Generation (Stable Diffusion 3)"
slug: "72-flow-matching-sd3"
number: 72
category: "image-generation"
authors: "Yaron Lipman, Ricky T. Q. Chen, Heli Ben-Hamu, Maximilian Nickel, Matt Le (Meta AI) - Flow Matching; Xingchao Liu, Chengyue Gong, Qiang Liu (UT Austin) - Rectified Flow; Patrick Esser, Sumith Kulal, Andreas Blattmann, Rahim Entezari, Jonas Muller et al. (Stability AI) - Stable Diffusion 3"
published: "October 2022 (Flow Matching, ICLR 2023); September 2022 (Rectified Flow, ICLR 2023); March 2024 (Stable Diffusion 3, ICML 2024)"
year: 2022
url: "https://arxiv.org/abs/2210.02747"
tags: ["image-generation", "diffusion", "flow-matching", "architecture"]
---

# Flow Matching and Rectified Flow: The New Default for Image Generation (Stable Diffusion 3)

**Authors:** Yaron Lipman, Ricky T. Q. Chen, Heli Ben-Hamu, Maximilian Nickel, Matt Le (Meta AI) - Flow Matching; Xingchao Liu, Chengyue Gong, Qiang Liu (UT Austin) - Rectified Flow; Patrick Esser, Sumith Kulal, Andreas Blattmann, Rahim Entezari, Jonas Muller et al. (Stability AI) - Stable Diffusion 3
**Published:** October 2022 (Flow Matching, ICLR 2023); September 2022 (Rectified Flow, ICLR 2023); March 2024 (Stable Diffusion 3, ICML 2024)
**Papers:** [Flow Matching arxiv.org/abs/2210.02747](https://arxiv.org/abs/2210.02747) | [Rectified Flow arxiv.org/abs/2209.03003](https://arxiv.org/abs/2209.03003) | [SD3 arxiv.org/abs/2403.03206](https://arxiv.org/abs/2403.03206)

---

## Why This Matters

Flow matching is **what replaced diffusion as the training objective for frontier image and video models**. Stable Diffusion 3, Flux, and a growing share of video generators are rectified-flow models, not [DDPM](../06-diffusion-models/summary.md)-style diffusion models. If you learned diffusion in 2022 and stopped, this is the update.

- **Straight paths instead of curved ones** - Diffusion's noise-to-image trajectory is curved, which is why it needs many small sampling steps. Flow matching trains the path to be as straight as possible, so few steps suffice.
- **Simpler objective** - No noise schedules, no variance-preserving versus variance-exploding bookkeeping, no signal-to-noise ratio juggling. Just: predict the velocity from noise to data.
- **Better few-step quality** - Usable images at 4 to 20 steps without distillation tricks.
- **It generalizes** - Flow matching connects any two distributions, not just Gaussian noise to data, which opens up image-to-image, cross-modal, and protein/molecule generation with the same machinery.

**The insight:** generation is transport. You are moving points from a noise distribution to a data distribution. Diffusion picks one particular (and rather winding) path for that transport. Flow matching says: pick the path yourself, and the straightest choice - a literal straight line between a noise sample and a data sample - is both easy to train and cheap to sample.

---

## The Problem: Diffusion's Path Is Curved

A trained diffusion model defines an ODE from noise to image ([DDIM](../70-ddim/summary.md) made this explicit). Solving that ODE numerically is what sampling is. The cost of solving an ODE accurately depends on how curved its trajectories are: a straight line can be solved exactly in one Euler step, while a curved path needs many small steps or a high-order solver.

Diffusion's trajectories are curved, for a structural reason. The forward process adds noise on a schedule that is not a straight interpolation, and the reverse process must average over many possible clean images at high noise levels, which bends the path. Hence 20 to 50 steps as the practical floor, plus a small industry of solvers and distillation methods to work around it.

The other accumulated problem was complexity. By 2023, "diffusion model" meant a stack of choices - schedule (linear, cosine, sigmoid), parameterization (epsilon, x0, v), variance type, SNR weighting - each of which mattered and none of which was principled.

---

## The Core Innovation

### Flow Matching: train a velocity field along a chosen path

Pick an interpolation between a noise sample `x_0 ~ N(0, I)` and a data sample `x_1`. The simplest choice is a straight line:

```
Path:      x_t = (1 - t) * x_0  +  t * x_1        for t in [0, 1]

Velocity:  dx_t/dt = x_1 - x_0                    (constant! a straight line)

Training:  minimize  || v_theta(x_t, t)  -  (x_1 - x_0) ||^2

Sampling:  start at x_0 ~ N(0, I), integrate
           x_{t+dt} = x_t + v_theta(x_t, t) * dt
```

That is the whole method. The network learns the average velocity field. The loss is a plain regression - no ELBO, no score matching derivation, no schedule.

**Rectified flow** is the same idea framed as straightening: train on straight-line couplings, then optionally "reflow" by regenerating pairs with the trained model and retraining, which makes the learned trajectories progressively straighter and pushes the model toward genuine one-step generation.

### Why straightness buys you steps

If the true velocity field were perfectly constant along a trajectory, one Euler step from t=0 to t=1 would be exact. It is not perfectly constant - the learned field averages over many possible endpoints - but it is far straighter than a diffusion trajectory, so error accumulates much more slowly as you reduce step count.

### Stable Diffusion 3: flow matching at scale, plus a new backbone

SD3 combined rectified flow with two other changes:

1. **MMDiT (Multimodal Diffusion Transformer)** - a [DiT](../44-sora-dit/summary.md)-style transformer where text and image tokens each get their **own set of weights** for attention projections and MLPs, but attend to each other in a joint attention operation. Previous models pushed text in through cross-attention only; giving text its own stream substantially improved prompt following and, notably, text rendering inside images.
2. **Timestep sampling weighted toward the middle** - the paper found that sampling `t` uniformly is suboptimal for rectified flow, because the hard part of the trajectory is in the middle. A logit-normal weighting that concentrates training on mid-trajectory timesteps improved results consistently.

SD3 also uses three text encoders (two CLIP variants plus T5-XXL), with T5 droppable at inference to save memory at some cost in prompt fidelity.

---

## Key Components Explained

### 1. Conditional Flow Matching
**What it does:** Makes the objective tractable.
**How it works:** The true "marginal" velocity field is intractable, but the *conditional* one - the velocity given a specific `(x_0, x_1)` pair - is trivially `x_1 - x_0`. Flow matching's key theorem is that regressing on the conditional velocity yields the correct marginal field in expectation. This is the same trick that makes denoising score matching work, applied to transport.

### 2. Resolution-Dependent Timestep Shifting
**What it does:** Fixes a scaling problem at high resolution.
**How it works:** At higher resolutions, a given noise level destroys relatively less signal (there is more redundancy in a big image), so the same timestep is "easier." SD3 shifts the timestep schedule as a function of resolution. This is a small detail with a large practical effect and is why flow-based models transfer across resolutions more gracefully.

### 3. The Reflow Procedure
**What it does:** Straightens trajectories further, toward one-step generation.
**How it works:** Generate `(noise, image)` pairs from the trained model, then retrain on those pairs. Since the pairs now come from the model's own (already fairly straight) map, the retrained model's paths get straighter. Repeat. InstaFlow used this to produce one-step text-to-image models.

### 4. What Stayed the Same
Latent space operation (a VAE encoder/decoder, from [Stable Diffusion](../07-stable-diffusion/summary.md)), [classifier-free guidance](../69-classifier-free-guidance/summary.md), and transformer backbones all carry over unchanged. Flow matching swaps the training objective and sampler, not the system architecture.

---

## Key Results

- SD3's scaling study found **smooth, predictable improvement with model size** from 800M to 8B parameters, with no sign of saturation - the same kind of clean [scaling law](../../techniques/12-scaling-laws/summary.md) behavior that made LLM investment predictable.
- In the paper's human evaluation, SD3 was preferred over contemporary open and closed models on prompt following, typography, and visual aesthetics.
- The systematic comparison in the SD3 paper tested many diffusion and flow formulations under matched conditions and found **rectified flow with the shifted timestep sampling to be the best performer**, which is the empirical basis for the field's shift.
- Flow matching reaches good sample quality at meaningfully fewer function evaluations than equivalent diffusion training, before any distillation.

---

## Why This Was Revolutionary

- **Simplified the objective to a regression.** A generative model you can specify in three lines is easier to reason about, tune, and extend than a schedule-laden diffusion stack.
- **Made few-step sampling a property of training, not a post-hoc distillation hack.**
- **Unified generative transport.** Flow matching does not require the source distribution to be Gaussian, so the same framework covers image-to-image translation, cross-modal generation, and non-Euclidean data (proteins, molecules, on manifolds).
- **MMDiT fixed text rendering.** Legible text in generated images went from a running joke to routine, largely because text got its own weight stream rather than being a second-class cross-attention input.

---

## Real-World Impact

- **Stable Diffusion 3 and 3.5**, and **Flux** (from the original Stable Diffusion authors' new lab), are rectified-flow models. Flux in particular became the open-weights quality leader on release.
- **Video generation** increasingly uses flow matching, where the step-count savings matter most - a video model's cost is per-frame times steps.
- **Audio and speech** models (including several TTS systems) adopted flow matching for the same reasons.
- **Science.** Flow matching on manifolds is used in protein backbone generation and molecular conformer generation, adjacent to what [AlphaFold](../../techniques/68-alphafold/summary.md) does for structure prediction.
- **Library support.** Hugging Face Diffusers ships flow-matching schedulers as first-class citizens alongside the diffusion ones.

---

## Key Takeaways for Practitioners

1. **If you are training a generative image or video model today, start with rectified flow**, not DDPM. Fewer knobs, better few-step behavior.
2. **Step counts differ.** Flow models are typically run at 20 to 28 steps for quality and behave acceptably at 4 to 8; do not carry over your diffusion step-count intuitions.
3. **Guidance behaves differently.** Flow models often want lower guidance scales than SD 1.5 did; Flux-family models use distilled guidance where the scale means something different again.
4. **Do not skip timestep weighting.** Uniform `t` sampling is the single most common mistake when people implement flow matching themselves.
5. **The velocity parameterization is the "v-prediction" you may already know.** If you used v-prediction diffusion, you were already most of the way here.

---

## Limitations & Future Directions

- **One-step generation still needs reflow or distillation** for top quality; straight-by-construction training gets close but not all the way.
- **Guidance remains a heuristic** layered on top, with the same 2x cost and saturation issues.
- **Training cost is unchanged.** Flow matching saves inference steps, not pretraining compute.
- **Theory outpaced practice on couplings.** Optimal-transport couplings (pairing each noise sample with a *nearby* data sample rather than a random one) straighten paths further, but computing them at scale is expensive and remains an open engineering problem.

---

## Further Reading

- **Flow Matching for Generative Modeling:** [arxiv.org/abs/2210.02747](https://arxiv.org/abs/2210.02747)
- **Rectified Flow:** [arxiv.org/abs/2209.03003](https://arxiv.org/abs/2209.03003)
- **Scaling Rectified Flow Transformers (SD3):** [arxiv.org/abs/2403.03206](https://arxiv.org/abs/2403.03206)
- **Stochastic Interpolants (a closely related framework):** [arxiv.org/abs/2303.08797](https://arxiv.org/abs/2303.08797)

## Citation

```bibtex
@inproceedings{lipman2023flow,
  title={Flow Matching for Generative Modeling},
  author={Lipman, Yaron and Chen, Ricky T. Q. and Ben-Hamu, Heli and Nickel, Maximilian and Le, Matt},
  booktitle={International Conference on Learning Representations},
  year={2023}
}

@inproceedings{esser2024scaling,
  title={Scaling Rectified Flow Transformers for High-Resolution Image Synthesis},
  author={Esser, Patrick and Kulal, Sumith and Blattmann, Andreas and Entezari, Rahim and M{\"u}ller, Jonas and others},
  booktitle={International Conference on Machine Learning},
  year={2024}
}
```

<!-- related:start -->

---

## Related in This Collection

- [Denoising Diffusion Probabilistic Models (DDPM)](../../image-generation/06-diffusion-models/summary.md)
- [High-Resolution Image Synthesis with Latent Diffusion Models (Stable Diffusion)](../../image-generation/07-stable-diffusion/summary.md)
- [Learning Transferable Visual Models From Natural Language Supervision (CLIP)](../../multimodal/08-clip/summary.md)
- [Sora and Diffusion Transformers (DiT): Video Generation as World Simulation](../../image-generation/44-sora-dit/summary.md)
- [Auto-Encoding Variational Bayes (VAE)](../../image-generation/57-vae/summary.md)
- [Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer (T5)](../../language-models/65-t5/summary.md)
- [Highly Accurate Protein Structure Prediction with AlphaFold (AlphaFold 2)](../../techniques/68-alphafold/summary.md)
- [Classifier-Free Diffusion Guidance](../../image-generation/69-classifier-free-guidance/summary.md)

<!-- related:end -->
