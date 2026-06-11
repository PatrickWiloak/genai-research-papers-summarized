---
title: "Denoising Diffusion Probabilistic Models (DDPM)"
slug: "06-diffusion-models"
number: 6
category: "image-generation"
authors: "Jonathan Ho, Ajay Jain, Pieter Abbeel (UC Berkeley)"
published: "June 2020 (NeurIPS 2020)"
year: 2020
url: "https://arxiv.org/abs/2006.11239"
tags: ["image-generation", "diffusion"]
---

# Denoising Diffusion Probabilistic Models (DDPM)

**Authors:** Jonathan Ho, Ajay Jain, Pieter Abbeel (UC Berkeley)

**Published:** June 2020 (NeurIPS 2020)

**Paper Link:** https://arxiv.org/abs/2006.11239

---

## Why This Paper Matters

This paper established **diffusion models** as a powerful alternative to GANs for image generation. While earlier diffusion work existed, DDPM made them practical and showed they could match or exceed GAN quality. This work laid the foundation for Stable Diffusion, DALL-E 2, Imagen, and the entire text-to-image revolution. Diffusion models are now the dominant approach for image generation.

---

## The Core Idea: Learning to Denoise

### The Intuition

Imagine you have a photo and you:
1. Gradually add noise to it until it's pure static
2. Learn to reverse this process step-by-step
3. Start from random noise → gradually remove noise → get a new image

**This is exactly what diffusion models do!**

### The Process

```
┌────────────────────────────────────────────────────┐
│  FORWARD PROCESS (Diffusion) - FIXED              │
│  Clean Image → Gradually Add Noise → Pure Noise   │
└────────────────────────────────────────────────────┘
  x₀ → x₁ → x₂ → ... → x_T
  🖼️ → 📊 → ▓▓ → ░░ → ▒▒ (pure noise)

┌────────────────────────────────────────────────────┐
│  REVERSE PROCESS (Denoising) - LEARNED            │
│  Pure Noise → Gradually Remove Noise → New Image  │
└────────────────────────────────────────────────────┘
  x_T → ... → x₂ → x₁ → x₀
  ▒▒ → ░░ → ▓▓ → 📊 → 🖼️ (new image)
```

---

## How It Works (Step-by-Step)

### Forward Process (Diffusion)

**Add Gaussian noise gradually over T steps (e.g., T=1000)**

Mathematical form:
```
q(x_t | x_{t-1}) = N(x_t; √(1-β_t) x_{t-1}, β_t I)

Where:
- x_t: Image at step t
- β_t: Noise schedule (how much noise to add at step t)
- N: Normal distribution
```

**In simple terms:**
- Take previous image
- Scale it down slightly
- Add a bit of noise
- Repeat 1000 times

**Key insight:** After enough steps, the image becomes pure Gaussian noise (looks random).

### Reverse Process (Denoising)

**Learn to remove noise step-by-step**

```
p_θ(x_{t-1} | x_t) = N(x_{t-1}; μ_θ(x_t, t), Σ_θ(x_t, t))

Where:
- θ: Neural network parameters
- μ_θ: Predicted mean (where to move to)
- Σ_θ: Predicted variance (usually fixed)
```

**Neural Network predicts:**
- What the slightly less noisy image should look like
- OR equivalently: what noise to subtract

---

## The Neural Network Architecture

### U-Net with Attention

**Architecture:**
```
Noisy Image (x_t) + Time Step (t)
          ↓
┌─────────────────────────────┐
│       Encoder               │
│  [Conv + Downsample]        │
│  [ResNet Blocks]            │
│  [Self-Attention]           │
└─────────────────────────────┘
          ↓
    Bottleneck
          ↓
┌─────────────────────────────┐
│       Decoder               │
│  [ResNet Blocks]            │
│  [Self-Attention]           │
│  [Conv + Upsample]          │
│  [Skip Connections]         │
└─────────────────────────────┘
          ↓
   Predicted Noise (ε_θ)
```

**Key Components:**

1. **Time Embedding:**
   - Encodes which timestep t we're at
   - Uses sinusoidal position encoding
   - Tells network how much noise to expect

2. **ResNet Blocks:**
   - Multiple residual blocks per resolution
   - Helps gradient flow
   - Includes time embedding

3. **Self-Attention:**
   - Capture long-range dependencies
   - Applied at lower resolutions (16×16, 8×8)
   - Expensive but important for quality

4. **Skip Connections:**
   - Connect encoder to decoder (U-Net style)
   - Preserve fine details
   - Help gradient flow

---

## Training Process

### Objective: Learn to Predict Noise

**Algorithm:**
```
For each training step:
1. Sample real image x₀ from dataset
2. Sample random timestep t ~ Uniform(1, T)
3. Sample noise ε ~ N(0, I)
4. Create noisy image: x_t = √(ᾱ_t) x₀ + √(1-ᾱ_t) ε
5. Predict noise: ε_θ(x_t, t)
6. Compute loss: ||ε - ε_θ(x_t, t)||²
7. Update network with gradient descent
```

**Training Objective (Simplified):**
```
Loss = MSE(actual_noise, predicted_noise)
```

**Why this works:**
- If model can predict the noise...
- ...it can subtract it to denoise
- ...repeat to gradually recover clean image

---

## Sampling (Generation) Process

### Generating New Images

**Algorithm:**
```
1. Start with pure random noise: x_T ~ N(0, I)

2. For t = T, T-1, ..., 1:
   a. Predict noise: ε_θ(x_t, t)
   b. Compute denoised image:
      x_{t-1} = (x_t - √(1-ᾱ_t) ε_θ(x_t, t)) / √(ᾱ_t)
   c. Add small amount of noise (except at t=1)

3. Return x₀ (final denoised image)
```

**Time Required:**
- 1000 steps typical
- ~1-5 seconds per image on GPU
- Much slower than GANs (single forward pass)

### Visual Representation

```
Step 1000: ▓▓▓▓▓▓ (pure noise)
Step 900:  ░░▓▓░░ (mostly noise)
Step 700:  ▒▒░░▒▒ (vague shapes)
Step 500:  📊📊📊 (rough structure)
Step 300:  🖼️📊🖼️ (recognizable)
Step 100:  🖼️🖼️🖼️ (mostly clear)
Step 1:    🖼️🖼️🖼️ (final image)
```

---

## Key Innovations

### 1. **Simplified Training Objective**
- Previous diffusion papers had complex objectives
- DDPM: just predict the noise
- Simpler, more stable training

### 2. **Variance Schedule**
- Careful design of β_t (noise schedule)
- Start small (β₁ = 0.0001), end large (β_T = 0.02)
- Linear or cosine schedule

### 3. **Connection to Score-Based Models**
- Showed diffusion is equivalent to score matching
- Theoretical foundation

### 4. **Reweighted Loss**
- Different timesteps contribute differently
- Clever weighting improves results

---

## Results

### Image Quality (CIFAR-10)

**Inception Score:**
- DDPM: 9.46
- Best GAN: 9.63
- Nearly matching GANs!

**FID (Fréchet Inception Distance, lower is better):**
- DDPM: 3.17
- Best GAN: 2.92
- Competitive with state-of-the-art

### Advantages Over GANs

**1. Training Stability:**
- No mode collapse
- No adversarial dynamics
- Smooth, predictable training

**2. Sample Diversity:**
- Better mode coverage
- Doesn't miss rare examples
- More diverse outputs

**3. Likelihood Estimation:**
- Can compute probabilities
- Useful for anomaly detection
- Better for scientific applications

### Disadvantages

**1. Slow Sampling:**
- 1000 steps vs. 1 GAN step
- 100-1000× slower
- Later work addresses this (DDIM, latent diffusion)

**2. Computational Cost:**
- Expensive training
- Expensive inference
- Requires significant GPU resources

---

## Mathematics Explained Simply

### Forward Process

**Adding noise with schedule:**
```
x_t = √(ᾱ_t) x₀ + √(1-ᾱ_t) ε

Where:
- ᾱ_t = ∏(1 - β_i) for i=1 to t
- ε ~ N(0, I) is random noise
- As t increases, first term shrinks, second grows
```

**Interpretation:**
- Weighted average of original image and noise
- t=0: 100% image, 0% noise
- t=T: ~0% image, 100% noise

### Reverse Process

**Removing noise:**
```
x_{t-1} = (x_t - √(1-ᾱ_t) ε_θ(x_t, t)) / √(ᾱ_t) + σ_t z

Where:
- First term: predicted clean image
- σ_t z: small noise for stochasticity
```

**Interpretation:**
- Use neural network to estimate noise
- Subtract estimated noise
- Add tiny noise to maintain distribution
- Repeat to gradually denoise

---

## Comparison: Diffusion vs. Other Generative Models

| Aspect | Diffusion | GANs | VAEs |
|--------|-----------|------|------|
| **Sample Quality** | Very High | Very High | Medium |
| **Sample Diversity** | Excellent | Medium | Excellent |
| **Training Stability** | Excellent | Poor | Excellent |
| **Sampling Speed** | Slow | Very Fast | Fast |
| **Likelihood** | Yes | No | Yes |
| **Mode Coverage** | Excellent | Medium | Excellent |

---

## Impact and Applications

### Direct Descendants:

**DDIM (2020):**
- Deterministic sampling
- 10-50× faster (100 steps instead of 1000)
- Enables interpolation

**Improved DDPM (2021):**
- Better noise schedules
- Learned variance
- Higher quality

**Score-Based Generative Models:**
- Continuous time diffusion
- Theoretical improvements
- State-of-the-art results

### Latent Diffusion Models (2022):
- Run diffusion in latent space (compressed)
- 10-100× faster than pixel diffusion
- **This became Stable Diffusion!**

### Guided Diffusion:
- Classifier guidance
- Classifier-free guidance
- Enables text-to-image (DALL-E 2, Imagen)

---

## Real-World Applications

### Image Generation:
- **Stable Diffusion:** Text-to-image synthesis
- **DALL-E 2:** OpenAI's text-to-image
- **Imagen:** Google's text-to-image
- **Midjourney:** Art generation

### Other Domains:
- **Audio:** WaveGrad, DiffWave (speech synthesis)
- **Video:** Video diffusion models
- **3D:** Point cloud generation
- **Molecules:** Drug discovery
- **Protein Design:** Scientific research

### Image Editing:
- Inpainting (fill missing regions)
- Super-resolution (enhance quality)
- Style transfer
- Image-to-image translation

---

## Improvements and Variants

### Faster Sampling:
- **DDIM:** Deterministic, 10-50× faster
- **DPM-Solver:** 10-20 steps
- **UniPC:** 5-10 steps
- Still active research area

### Better Quality:
- **Classifier Guidance:** Use classifier to guide generation
- **Classifier-Free Guidance:** Guide without classifier (better)
- **Cascaded Diffusion:** Multiple models at different resolutions

### Efficiency:
- **Latent Diffusion:** Work in compressed space
- **Consistency Models:** One-step generation
- **Progressive Distillation:** Teacher-student speedup

---

## Training Tips and Tricks

### 1. **Noise Schedule**
- Linear works OK
- Cosine schedule often better (less noise early, more late)
- Tune β_start and β_end

### 2. **Architecture**
- U-Net is standard
- More attention = better quality but slower
- Larger model = better quality

### 3. **Training Stability**
- Very stable compared to GANs
- EMA (Exponential Moving Average) of weights helps
- Batch size matters (larger is often better)

### 4. **Evaluation**
- FID for quality
- IS for quality + diversity
- Human evaluation for perceptual quality

---

## Limitations

### 1. **Sampling Speed**
- 1000 steps is slow
- Limits real-time applications
- Ongoing research to accelerate

### 2. **Computational Cost**
- Training requires significant compute
- Inference more expensive than GANs
- Environmental concerns

### 3. **Limited Control**
- Base diffusion has no control
- Need additional conditioning (text, class, etc.)
- Guidance mechanisms add complexity

### 4. **Resolution**
- Original DDPM: 32×32 or 64×64
- Higher resolution requires more compute
- Cascaded or latent diffusion for high-res

---

## Key Concepts

### 1. **Noise Schedule (β_t)**
How much noise to add at each step. Critical hyperparameter.

### 2. **Variance Preserving**
Forward process designed so variance stays constant.

### 3. **Denoising Score Matching**
Connection to estimating gradients of data distribution.

### 4. **Markov Chain**
Each step only depends on previous step (not full history).

### 5. **Reverse SDE/ODE**
Can view reverse process as solving differential equation.

---

## Why Diffusion Models Won

### Over GANs:
- **More stable training** (no adversarial dynamics)
- **Better mode coverage** (less likely to miss examples)
- **Easier to scale** (predictable training)

### Over VAEs:
- **Sharper samples** (no blurriness)
- **Better quality** (state-of-the-art results)

### Over Autoregressive Models:
- **Parallel sampling** (all pixels simultaneously at each step)
- **Easier conditioning** (text, class, etc.)

**Result:** Diffusion became dominant for image generation by 2022-2023.

---

## Key Takeaways

1. **Diffusion = gradual noising + learned denoising**
2. **Training is stable and predictable**
3. **Sample quality rivals or exceeds GANs**
4. **Slow sampling is main drawback** (but improving)
5. **Theoretical foundations are solid** (score matching, SDEs)
6. **Highly extensible** (text, audio, video, 3D)
7. **Foundation for modern image generation** (Stable Diffusion, DALL-E 2)

---

## Further Reading

- **Original Paper:** https://arxiv.org/abs/2006.11239
- **Lillian Weng's Blog:** https://lilianweng.github.io/posts/2021-07-11-diffusion-models/
- **Annotated Diffusion:** https://huggingface.co/blog/annotated-diffusion
- **DDIM Paper:** https://arxiv.org/abs/2010.02502
- **Score-Based Models:** https://arxiv.org/abs/2011.13456

---

## Citation

```bibtex
@article{ho2020denoising,
  title={Denoising diffusion probabilistic models},
  author={Ho, Jonathan and Jain, Ajay and Abbeel, Pieter},
  journal={Advances in neural information processing systems},
  volume={33},
  pages={6840--6851},
  year={2020}
}
```

<!-- related:start -->

---

## Related in This Collection

- [Generative Adversarial Networks (GANs)](../../image-generation/02-generative-adversarial-networks/summary.md)
- [High-Resolution Image Synthesis with Latent Diffusion Models (Stable Diffusion)](../../image-generation/07-stable-diffusion/summary.md)
- [DALL-E 3: Improving Image Generation with Better Captions](../../image-generation/48-dalle3/summary.md)
- [Auto-Encoding Variational Bayes (VAE)](../../image-generation/57-vae/summary.md)

<!-- related:end -->
