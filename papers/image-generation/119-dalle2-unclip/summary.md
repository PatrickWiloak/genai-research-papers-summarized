---
title: "Hierarchical Text-Conditional Image Generation with CLIP Latents (DALL-E 2 / unCLIP)"
slug: "119-dalle2-unclip"
number: 119
category: "image-generation"
authors: "Aditya Ramesh, Prafulla Dhariwal, Alex Nichol, Casey Chu, Mark Chen (OpenAI)"
published: "April 2022"
year: 2022
url: "https://arxiv.org/abs/2204.06125"
tags: ["image-generation", "diffusion", "text-to-image"]
---

# Hierarchical Text-Conditional Image Generation with CLIP Latents (DALL-E 2 / unCLIP)

**Authors:** Aditya Ramesh, Prafulla Dhariwal, Alex Nichol, Casey Chu, Mark Chen (OpenAI)
**Published:** April 2022
**Paper:** [arxiv.org/abs/2204.06125](https://arxiv.org/abs/2204.06125)

---

## Why This Matters

DALL-E 2 was **the moment text-to-image generation went from research curiosity to public phenomenon**. Announced in April 2022, its 1024x1024 images of things like "an astronaut riding a horse" circulated everywhere, and within months Midjourney and [Stable Diffusion](../07-stable-diffusion/summary.md) had followed. The paper behind it, which the authors call **unCLIP**, introduced a distinctive two-step recipe.

- **Generate the idea first, then the pixels.** A "prior" turns the caption into a [CLIP](../../multimodal/08-clip/summary.md) image embedding (a compact vector summarising what an image shows), and a [diffusion](../06-diffusion-models/summary.md) "decoder" turns that embedding into an image.
- **Better diversity at the same realism.** In human evaluations against OpenAI's earlier GLIDE model, unCLIP was preferred for diversity 70.5 percent of the time, with photorealism roughly tied.
- **State-of-the-art zero-shot score.** A zero-shot FID (a measure of how close generated images are to real ones; lower is better) of 10.39 on MS-COCO, against about 28 for the original DALL-E.
- **New editing abilities for free:** image variations, blending two images, and editing an image by changing a text description.

**The insight:** CLIP had already learned a space where images and text sit near each other when they mean the same thing. If you can decode a CLIP image embedding back into pixels, then generating an image reduces to generating the right point in that space.

---

## Background: DALL-E 1 (2021)

The first DALL-E ("Zero-Shot Text-to-Image Generation", Ramesh, Pavlov, Goh, Gray, Voss, Radford, Chen and Sutskever, [arXiv 2102.12092](https://arxiv.org/abs/2102.12092), February 2021, ICML 2021) took a pure language-model approach:

```
Stage 1: a discrete VAE compresses each 256x256 image into a
         32 x 32 grid of tokens, each one of 8,192 codebook entries
         (the same idea as VQ-VAE)

Stage 2: a 12-billion-parameter sparse Transformer models
         [ up to 256 text tokens ][ 1,024 image tokens ]
         as one stream, predicting the next token, like GPT

Sampling: generate many candidates, rerank them with CLIP (N = 512)
```

It was trained on 250 million text-image pairs from the internet. Human raters preferred its samples over the prior best model (DF-GAN) on MS-COCO captions 93 percent of the time for caption match and 90 percent for realism. It showed that scale plus a simple autoregressive model could compose concepts ("an armchair in the shape of an avocado"), but images were low resolution and often blurry. The token-sequence approach links to [VQ-VAE](../89-vq-vae/summary.md) and [VQ-GAN](../90-vq-gan/summary.md). DALL-E 2 dropped it in favour of diffusion.

---

## The Problem: Fidelity Versus Diversity

By late 2021, diffusion models conditioned on text (notably OpenAI's GLIDE) produced much better images than DALL-E 1. Their quality depended on **[classifier-free guidance](../69-classifier-free-guidance/summary.md)**, a sampling trick that pushes outputs harder toward the caption. Guidance has a known cost: as you turn it up, images get more realistic but **less diverse**. For the same prompt, every sample converges on the same camera angle, colour and layout.

The unCLIP paper asks whether an explicit intermediate representation can separate the two: decide the *content* of the image (with diversity) first, then render it at high fidelity.

---

## The Core Innovation

A two-stage generative stack on top of a frozen CLIP model:

```
caption y
   |
   v
[ CLIP text encoder ] --> text embedding z_t
   |
   v
[ PRIOR ]  P(z_i | y)          generates a CLIP IMAGE embedding z_i
   |                           (autoregressive or diffusion)
   v
[ DECODER ] P(x | z_i, y)      diffusion model: embedding -> 64x64 image
   |
   v
[ UPSAMPLER 1 ] 64 -> 256      diffusion upsamplers
[ UPSAMPLER 2 ] 256 -> 1024
   |
   v
image x
```

The decoder is the "inverse" of CLIP's image encoder, hence the name unCLIP. It is non-deterministic: decoding the same embedding twice gives two different images that share the same semantics and style but differ in details CLIP does not capture.

---

## Key Components Explained

### 1. The decoder
**What it does:** Turns a CLIP image embedding into an image.
**How it works:** It is a modified version of the 3.5-billion-parameter GLIDE diffusion model. The CLIP embedding is projected into the diffusion timestep embedding and also into four extra tokens of context alongside the caption tokens. To enable classifier-free guidance, the CLIP embedding is dropped 10 percent of the time during training and the caption 50 percent of the time. Two further diffusion models upsample 64x64 to 256x256 and then to 1024x1024.

### 2. The prior
**What it does:** Bridges from text to a CLIP image embedding.
**How it works:** Two options were compared:
- **Autoregressive prior:** compresses the embedding with PCA, quantises it into discrete codes and predicts them one at a time, like a language model.
- **Diffusion prior:** a decoder-only Transformer that directly denoises the embedding vector, conditioned on the caption and the CLIP text embedding. At sampling time it draws two candidates and keeps the one with the higher dot product with the text embedding.

The diffusion prior matched or beat the autoregressive one at lower compute and became the default.

### 3. Why not skip the prior?
**What it does:** Justifies the extra stage.
**How it works:** The authors tried feeding the decoder only the caption, or the CLIP *text* embedding in place of an image embedding. Both worked less well than using the prior, in both FID and human preference. Text embeddings and image embeddings live in the same space but are not interchangeable.

### 4. Editing in CLIP space
**What it does:** Gives manipulation abilities without extra training.
**How it works:**
- **Variations:** encode an image with CLIP, decode it several times to get new images with the same content and style.
- **Interpolations:** spherically interpolate between two images' embeddings to blend them.
- **Text diffs:** move an image's embedding in the direction of (new caption embedding minus old caption embedding), for example turning "a photo of a cat" toward "an anime drawing of a cat."

---

## Key Results

- **Zero-shot MS-COCO FID** (the model never trained on COCO): DALL-E about 28, GLIDE 12.24, unCLIP with autoregressive prior 10.63, with diffusion prior **10.39**.
- **Human evaluation vs GLIDE** (share of comparisons where unCLIP with diffusion prior won): photorealism 48.9 percent, caption similarity 45.3 percent, **diversity 70.5 percent**. So photorealism was roughly tied, caption matching slightly worse, diversity much better.
- **Guidance keeps diversity.** As guidance increases, GLIDE's samples collapse onto one composition while unCLIP's stay varied, because the content is already fixed in the sampled embedding before the decoder is guided.
- **Training data:** the CLIP encoder used about 650 million images; the prior, decoder and upsamplers were trained on the DALL-E dataset of about 250 million images.

---

## Why This Was Revolutionary

- **Took text-to-image to 1024x1024 photographic quality** in a public product, which set expectations for the whole field.
- **Showed that a learned multimodal embedding can be the backbone of a generator,** not just a classifier or reranker.
- **Made the diversity-fidelity trade-off an explicit design lever.**
- **Popularised image variations and embedding arithmetic** as user-facing features.

---

## Real-World Impact

- **Public release.** OpenAI opened DALL-E 2 to a research preview in April 2022 and to the public later in 2022, alongside an API. For many people it was their first encounter with generative image models.
- **Competition.** Google's [Imagen](../91-imagen/summary.md) arrived weeks later and argued that a large frozen text encoder (T5) mattered more than a CLIP prior, and [Stable Diffusion](../07-stable-diffusion/summary.md) brought open weights. The prior-plus-decoder design lived on in some open models (the Kandinsky series, for example), but most later systems conditioned directly on text encoders.
- **Successors at OpenAI.** [DALL-E 3](../48-dalle3/summary.md) (2023) addressed unCLIP's weakest point, prompt following, by retraining on much better synthetic captions. In 2025 ChatGPT's image generation moved to natively multimodal models in the [GPT-4o](../../language-models/40-gpt4o/summary.md) line.
- **Policy precedent.** DALL-E 2's staged access, content filters and published "risks and limitations" document became a template for how labs released image generators.

---

## Key Takeaways for Practitioners

1. **Separating "what" from "how" is a useful design pattern.** Sampling a semantic embedding first and rendering second gives you diversity control and editing hooks.
2. **CLIP embeddings lose information.** Spatial relationships, counting, attribute binding and exact text are weak in CLIP, and anything built on CLIP inherits those weaknesses.
3. **Guidance is a trade-off, not free quality.** Measure diversity as well as fidelity.
4. **FID alone is not enough.** The paper leaned on human evaluations of photorealism, caption match and diversity, which remains best practice.

---

## Limitations & Future Directions

- **Attribute binding.** unCLIP is worse than GLIDE at prompts like "a red cube on top of a blue cube," often swapping colours. The authors attribute this to CLIP embeddings not binding attributes to objects.
- **Text in images.** It struggles to render coherent text, likely because CLIP embeddings do not encode spelling precisely and BPE tokenisation hides spelling from the model.
- **Detail in complex scenes** is often low.
- **Risks.** The paper notes that better realism raises the risk of deceptive and harmful content, and points to the separate DALL-E 2 risks and limitations document. Mitigations included filtered training data and restricted access at launch.
- **What fixed them:** DALL-E 3's recaptioning for prompt following; T5-style or LLM text encoders (Imagen, SD3) for text rendering and composition; see [Flow Matching and SD3](../72-flow-matching-sd3/summary.md).

---

## Further Reading

- **Original Paper:** [arxiv.org/abs/2204.06125](https://arxiv.org/abs/2204.06125)
- **DALL-E 1 (Zero-Shot Text-to-Image Generation):** [arxiv.org/abs/2102.12092](https://arxiv.org/abs/2102.12092)
- **GLIDE:** [arxiv.org/abs/2112.10741](https://arxiv.org/abs/2112.10741)
- **DALL-E 2 risks and limitations:** [github.com/openai/dalle-2-preview](https://github.com/openai/dalle-2-preview/)
- **In this collection:** [CLIP](../../multimodal/08-clip/summary.md), [Diffusion Models](../06-diffusion-models/summary.md), [Imagen](../91-imagen/summary.md), [DALL-E 3](../48-dalle3/summary.md), [Classifier-Free Guidance](../69-classifier-free-guidance/summary.md), [VQ-VAE](../89-vq-vae/summary.md)

## Citation

```bibtex
@article{ramesh2022hierarchical,
  title={Hierarchical Text-Conditional Image Generation with CLIP Latents},
  author={Ramesh, Aditya and Dhariwal, Prafulla and Nichol, Alex and Chu, Casey and Chen, Mark},
  journal={arXiv preprint arXiv:2204.06125},
  year={2022}
}
```

<!-- related:start -->

---

## Related in This Collection

- [High-Resolution Image Synthesis with Latent Diffusion Models (Stable Diffusion)](../../image-generation/07-stable-diffusion/summary.md)
- [Learning Transferable Visual Models From Natural Language Supervision (CLIP)](../../multimodal/08-clip/summary.md)
- [GPT-4 Technical Report](../../language-models/36-gpt4/summary.md)
- [GPT-4o: The First Omni Model](../../language-models/40-gpt4o/summary.md)
- [DALL-E 3: Improving Image Generation with Better Captions](../../image-generation/48-dalle3/summary.md)
- [Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer (T5)](../../language-models/65-t5/summary.md)
- [Classifier-Free Diffusion Guidance](../../image-generation/69-classifier-free-guidance/summary.md)
- [Flow Matching and Rectified Flow: The New Default for Image Generation (Stable Diffusion 3)](../../image-generation/72-flow-matching-sd3/summary.md)

<!-- related:end -->
