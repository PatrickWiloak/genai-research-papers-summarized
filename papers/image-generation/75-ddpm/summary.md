# Denoising Diffusion Probabilistic Models (DDPM)

**Authors:** Jonathan Ho, Ajay Jain, Pieter Abbeel (UC Berkeley)

**Published:** June 2020 (arXiv 2006.11239), NeurIPS 2020

**Paper Link:** https://arxiv.org/abs/2006.11239

---

## Why This Paper Matters

DDPM is **the** paper that made diffusion models work. The diffusion framework had existed since Sohl-Dickstein et al. in 2015 as a theoretical curiosity, but no one had produced compelling images from it. Ho, Jain, and Abbeel reformulated the training objective into something dramatically simpler — predict the noise added to an image — and produced samples on CIFAR-10 and CelebA that rivaled or beat the best GANs of the time.

Within two years, this paper's recipe powered DALL-E 2, Imagen, Stable Diffusion, Midjourney, and essentially every successful image generator. The "predict the noise" loss it introduced is still the foundation of almost all modern diffusion models, from text-to-image to text-to-video to protein design. If GANs (#2) ruled image generation from 2014 to 2020, DDPM ended that era.

---

## The Problem Before DDPM

By 2020, generative image modeling had a clear leader and several promising alternatives:

- **GANs** (BigGAN, StyleGAN2) produced the sharpest images but were notoriously unstable, suffered from mode collapse, and were hard to evaluate (no likelihood)
- **VAEs** were stable and gave likelihoods but produced blurry images
- **Autoregressive models** (PixelCNN, PixelRNN) had good likelihoods but were extremely slow to sample
- **Normalizing flows** were elegant but required restricted architectures
- **Score-based models** (Song & Ermon, 2019) showed promise but were complex

**Diffusion models**, introduced by Sohl-Dickstein et al. in 2015, were a beautiful theoretical idea — slowly corrupt data with noise, then learn to reverse it — but produced poor samples and had complex variational losses that were hard to optimize.

The open question: could diffusion models actually generate competitive images? And could anyone make them simple enough to train?

---

## The Core Innovation: A Simpler Training Objective

DDPM's main contribution is identifying that the messy variational lower bound on the original diffusion training objective can be drastically simplified — without losing performance — to:

> **Train a neural network to predict the noise that was added to an image at a given timestep.**

That's it. The full training loop is:

```python
def train_step(image_x0, model):
    # 1. Pick a random timestep
    t = uniform(1, T)

    # 2. Add noise according to the schedule
    noise = randn_like(image_x0)
    x_t = sqrt(alpha_bar[t]) * image_x0 + sqrt(1 - alpha_bar[t]) * noise

    # 3. Predict the noise
    predicted_noise = model(x_t, t)

    # 4. Simple MSE loss
    loss = mse(noise, predicted_noise)
    return loss
```

This "simple" loss (Lsimple in the paper) is a reweighted version of the variational bound that empirically produces dramatically better samples than the unweighted bound. The connection to score matching (predicting grad log p(x)) is shown explicitly, unifying DDPM with score-based models.

---

## How DDPM Works

### The Forward Process: Gradually Add Noise

Given a clean image x_0, we define a Markov chain that adds Gaussian noise at each step:

```
x_0 -> x_1 -> x_2 -> ... -> x_T  (T = 1000)

x_t = sqrt(1 - beta_t) * x_{t-1} + sqrt(beta_t) * noise
```

The beta_t values follow a fixed schedule (linear from 1e-4 to 0.02 in the paper). After T=1000 steps, x_T is essentially pure Gaussian noise — all information about the original image is destroyed.

A beautiful property: because Gaussians compose nicely, you can jump directly to any timestep:

```
x_t = sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * noise
where alpha_bar_t = product of (1 - beta_s) for s=1..t
```

This means training doesn't need to simulate the full chain — just sample a random t and a random noise, and you can compute x_t directly.

### The Reverse Process: Learn to Denoise

A neural network epsilon_theta(x_t, t) is trained to predict the noise that was added. At sampling time, we start from pure noise x_T ~ N(0, I) and gradually denoise:

```
For t = T, T-1, ..., 1:
    predicted_noise = epsilon_theta(x_t, t)
    x_{t-1} = (1 / sqrt(alpha_t)) * (x_t - (beta_t / sqrt(1 - alpha_bar_t)) * predicted_noise) + sigma_t * z
    where z ~ N(0, I) (and z=0 at t=1)
```

After 1000 denoising steps, you have a generated image.

### Architecture: U-Net with Attention

The denoising network epsilon_theta is a **U-Net** — an encoder-decoder with skip connections, originally from biomedical segmentation. DDPM's U-Net has:

- Residual blocks at each resolution
- Self-attention at lower resolutions (16x16, 8x8)
- **Timestep embedding** injected into every block via sinusoidal embedding + MLP (similar to Transformer positional encoding)
- Group normalization throughout
- Symmetric down/up sampling

The same network handles all 1000 timesteps — the time embedding tells it which noise level to denoise.

```python
class DenoisingUNet:
    def forward(self, x_t, t):
        t_emb = mlp(sinusoidal_embedding(t))  # timestep embedding
        h = downsample_path(x_t, t_emb)        # encoder
        h = bottleneck(h, t_emb)               # attention here
        out = upsample_path(h, t_emb, skips)   # decoder with skip connections
        return out  # predicted noise, same shape as input
```

---

## The Key Reformulation in Detail

The 2015 diffusion paper trained on the full variational lower bound:

```
L_VLB = E[Sum_t KL(q(x_{t-1}|x_t, x_0) || p_theta(x_{t-1}|x_t))] + ...
```

This is principled but messy. DDPM showed that:

1. The KL between two Gaussians has a closed form
2. By choosing to parameterize the model in terms of **predicted noise** rather than **predicted mean**, the per-timestep loss becomes a simple MSE
3. Dropping the per-timestep weighting (using L_simple instead of L_VLB) actually helps sample quality, even though it's no longer a valid lower bound on log-likelihood

```
L_simple = E_{t, x_0, eps} [|| eps - epsilon_theta(x_t, t) ||^2]
```

This is dramatically easier to implement, more stable, and produces better samples. It's basically "denoising score matching with a particular noise schedule."

---

## Key Results

### Sample Quality

On CIFAR-10 (32x32 images):
- **DDPM:** FID 3.17 — better than the best GAN at the time (StyleGAN2-ADA: 3.26)
- **Inception Score:** 9.46 (state of the art)

On LSUN Bedrooms, LSUN Cats, LSUN Church (256x256):
- Sample quality competitive with or better than StyleGAN2
- No mode collapse — clear coverage of the full data distribution

The samples were genuinely sharp and diverse, not the blurry outputs that had plagued earlier likelihood-based generative models.

### Likelihood

DDPM achieved competitive (though not state-of-the-art) log-likelihoods on CIFAR-10, with the caveat that the variational bound is not the tightest possible likelihood estimate.

### Compute Cost

The big downside of DDPM: sampling requires 1000 forward passes through the U-Net, making generation **orders of magnitude slower** than GANs. A single 256x256 image takes minutes. This sampling cost spawned an entire subfield of follow-up work to accelerate it.

---

## Why DDPM Worked Where Earlier Diffusion Didn't

Several engineering choices were crucial:

1. **The L_simple objective** — much easier to optimize than the full variational bound
2. **Predicting noise rather than mean** — a better parameterization with cleaner gradients
3. **U-Net architecture with attention and time embedding** — much more capacity than the simple networks used in the 2015 paper
4. **Long noise schedule** — 1000 timesteps with carefully chosen beta schedule
5. **Sufficient compute** — diffusion benefits from large models and long training

None of these are conceptual breakthroughs in isolation. Together, they turned a failed idea into the dominant image generation paradigm.

---

## Impact and Legacy

### The image generation revolution

DDPM directly enabled the diffusion-model wave:

- **Improved DDPM** (Nichol & Dhariwal, 2021): Cosine schedule, learned variance, better sampling
- **Classifier guidance** (Dhariwal & Nichol, 2021): "Diffusion Models Beat GANs on Image Synthesis"
- **Classifier-free guidance** (Ho & Salimans, 2022): Powerful conditional generation
- **GLIDE** (OpenAI, 2021): First major text-to-image diffusion model
- **DALL-E 2** (2022): CLIP + diffusion
- **Imagen** (2022): Cascaded text-to-image diffusion from Google
- **Stable Diffusion** (2022): Latent diffusion — DDPM in compressed latent space, runs on consumer GPUs
- **DiT, Sora, Veo:** Transformer-based diffusion for video

### Beyond images

The DDPM recipe transferred to almost every continuous-data domain:
- **Audio:** WaveGrad, DiffWave, AudioLM
- **3D:** DreamFusion, point cloud diffusion
- **Molecules and proteins:** AlphaFold-style structure generation
- **Robotics policies:** Diffusion Policy
- **Decision making:** Diffuser

### Conceptual contributions

- Established **noise prediction** as the canonical diffusion training objective
- Showed that **simple MSE** can outperform a principled VLB
- Provided the explicit **link to score-based models**, unifying two frameworks
- Made **U-Net + time embedding** the standard architecture
- Demonstrated that **stable, mode-covering generative modeling** is possible without adversarial training

DDPM is the rare paper that's both theoretically elegant and a practical recipe — and it changed an entire field overnight.

---

## Limitations

- **Slow sampling:** 1000 steps is impractical; spawned huge follow-up work (DDIM, DPM-Solver, distillation, consistency models)
- **Pixel-space training is expensive:** Mitigated by latent diffusion (Stable Diffusion)
- **Fixed noise schedule:** Hand-designed; later work showed schedule matters a lot
- **No conditional generation in original paper:** Class-conditional and text-conditional generation came later
- **Mode coverage trades off with sample quality:** Guidance techniques (CFG) make this trade-off explicit and controllable
- **Compute cost remains substantial** for training large diffusion models

---

## Connections to Other Papers

- **Generative Adversarial Networks (#2):** DDPM's competitor and predecessor as the dominant image generator. Where GANs use adversarial training, DDPM uses denoising — more stable, better mode coverage
- **VAE (#67):** DDPM is sometimes called a hierarchical VAE with a fixed encoder. Both are likelihood-based, but DDPM produces much sharper samples
- **Diffusion Models (#6):** The general framework paper — DDPM is the breakthrough specific instantiation that made the framework work
- **Stable Diffusion (#7):** Built directly on DDPM by moving the diffusion process into the latent space of a VAE, making generation practical on consumer hardware
- **Attention Is All You Need (#1):** DDPM's U-Net uses self-attention at low resolutions; later diffusion models (DiT) replace the U-Net with a full Transformer
- **Sora / DiT (#44):** Transformer-based diffusion models — same DDPM training objective, different backbone, scaled to video
- **DALL-E 3 (#48):** Text-to-image generation built on a DDPM-style backbone with CLIP-style text conditioning
- **VQ-VAE (#76):** An alternative approach to generative modeling via discrete tokens; diffusion (continuous) and VQ-VAE (discrete) define the two main axes of modern generative modeling

---

## Key Takeaways

1. **Predict the noise, not the mean** — DDPM's parameterization makes the training loss a simple MSE
2. **The "simple" loss beats the principled one** — dropping per-timestep weighting from the variational bound improves samples
3. **Diffusion can match or beat GANs** — DDPM ended the GAN era for image generation
4. **Stable and mode-covering** — diffusion training has none of the instability or mode collapse that plagued GANs
5. **A general recipe for generative modeling** — the DDPM blueprint now powers text-to-image, video, audio, 3D, molecules, and policies; almost every modern generative model uses some descendant of this approach
