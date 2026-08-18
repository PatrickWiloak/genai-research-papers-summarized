---
title: "ControlNet: Adding Conditional Control to Text-to-Image Diffusion Models"
slug: "71-controlnet"
number: 71
category: "image-generation"
authors: "Lvmin Zhang, Anyi Rao, Maneesh Agrawala (Stanford University)"
published: "February 2023 (ICCV 2023, Marr Prize / Best Paper)"
year: 2023
url: "https://arxiv.org/abs/2302.05543"
tags: ["image-generation", "diffusion", "controllable-generation", "fine-tuning"]
---

# ControlNet: Adding Conditional Control to Text-to-Image Diffusion Models

**Authors:** Lvmin Zhang, Anyi Rao, Maneesh Agrawala (Stanford University)
**Published:** February 2023 (ICCV 2023, Marr Prize / Best Paper)
**Paper:** [arxiv.org/abs/2302.05543](https://arxiv.org/abs/2302.05543)

---

## Why This Matters

ControlNet is **the paper that turned text-to-image models from slot machines into tools**. Before it, your only control over composition was the prompt, and prompts are terrible at specifying "this pose, this camera angle, this exact layout." ControlNet lets you hand the model a sketch, a depth map, or a stick figure and say: match this structure, fill in the style from the prompt.

- **Spatial control** - Edges, depth, human pose, segmentation maps, scribbles, normal maps, line art.
- **Trains on small data** - Useful ControlNets have been trained on 50,000 image pairs on a single consumer GPU. The paper reports quality results well before million-scale datasets.
- **Never breaks the base model** - The original [Stable Diffusion](../07-stable-diffusion/summary.md) weights stay frozen. A ControlNet is an add-on you can attach and detach.
- **Made AI image generation professional** - Storyboarding, product visualization, architectural rendering, and consistent character work all depend on structural control.

**The insight:** to add a new control signal to a large pretrained diffusion model without destroying it, clone the model's encoder, train only the clone, and connect it back through layers initialized to **zero** - so at step one the modified model is mathematically identical to the original, and control is learned from there.

---

## The Problem: Prompts Cannot Describe Geometry

By early 2023 Stable Diffusion could render almost anything, and controlling *where* things went was nearly impossible. Practitioners tried:

- **Prompt engineering.** "a man standing with his left arm raised, three-quarter view" produces a man with some arms in some position.
- **Fine-tuning the whole model** on paired data. This works but needs enormous datasets; with small datasets the model catastrophically forgets and quality collapses.
- **[LoRA](../../techniques/10-lora/summary.md) adapters.** Excellent for style and subject, weak for per-image spatial layout, because the control signal is a whole image, not a few weights.

The specific technical hazard was catastrophic forgetting: any naive attempt to fine-tune a billion-parameter model on 50k pairs degrades it badly, and you have destroyed the thing that made it worth using.

---

## The Core Innovation

Copy the encoder, freeze the original, and connect with **zero convolutions**.

```
        input latent x_t + timestep + text prompt
                     |
        +------------+------------------+
        |                               |
   FROZEN SD encoder            TRAINABLE COPY of
   (weights locked)             the SD encoder
        |                               |
        |                        condition image c
        |                        (edges/depth/pose)
        |                        enters here through
        |                        a zero-conv
        |                               |
        |                        zero convolution
        |                               |
   FROZEN SD decoder  <---- added to skip connections
                     |
                  output
```

**Zero convolution** is a 1x1 convolution whose weights and bias are initialized to zero. Two properties follow:

1. **At training step 0, the ControlNet contributes exactly nothing.** The combined model produces bit-identical output to stock Stable Diffusion. There is no "damage window" at the start of training where random new weights scramble the base model.
2. **Gradients are still non-zero.** A zero-initialized convolution has zero output but its gradient with respect to its weights depends on its input, which is non-zero. So it learns, starting from a perfectly safe initial condition and growing control gradually.

The frozen original is the safety net: whatever the ControlNet learns, the base model's knowledge is untouched and always retrievable.

---

## Key Components Explained

### 1. The Locked Copy / Trainable Copy Pair
**What it does:** Preserves the pretrained model while learning new conditioning.
**How it works:** The trainable copy starts as an exact duplicate of the base model's encoder blocks, so it begins with full knowledge of natural images rather than random initialization. This is why it converges on small datasets - it is not learning to see, only learning to route a new signal.

### 2. Zero Convolutions
**What it does:** Provides a damage-free connection between the copy and the frozen network.
**How it works:** Placed both at the ControlNet's input (where the condition image enters) and at every output that rejoins the frozen decoder's skip connections. Because they start at zero, the network smoothly interpolates from "ignore the control" to "obey the control" over training.

### 3. Condition Types
**What it does:** One architecture, many controls - each a separately trained ControlNet.
**How it works:** The condition image is just an image, so anything renderable as an image works:

| Control | Extracted with | Typical use |
|---|---|---|
| Canny edges | Canny edge detector | Preserve exact outlines of a reference |
| Depth | MiDaS depth estimator | Preserve 3D layout, re-style a scene |
| Human pose | OpenPose keypoints | Put a character in a specific pose |
| Segmentation | Semantic segmenter | Lay out sky/building/road regions |
| Scribble | Hand drawing | Turn a rough doodle into a finished image |
| Normal map | Normal estimator | Preserve surface geometry |
| Line art / MLSD | Line detectors | Architecture, anime, technical drawing |

### 4. Composability
**What it does:** Stack multiple controls, and combine with other adapters.
**How it works:** Multiple ControlNets can be applied simultaneously with per-net weights (for example depth for layout plus pose for the figure). ControlNets also compose with LoRAs, textual inversions, and IP-Adapters because none of them modify the same weights.

### 5. Control Strength and Guidance Windows
**What it does:** Dials how strictly the structure is obeyed.
**How it works:** Implementations expose a control weight (0 to 2) and a start/end step range. Applying control only during the first 60 percent of sampling fixes composition early and lets the model add free detail late - a standard trick for avoiding stiff, traced-looking output.

---

## Key Results

- Trained on datasets ranging from about 50,000 to 3 million image/condition pairs, with useful results reported at the small end - a single consumer GPU regime.
- Human raters preferred ControlNet's conditional fidelity over prior conditioning approaches and over fine-tuned baselines across the tested control types.
- The base model's output distribution is provably unchanged at initialization, and empirically the frozen model shows no quality regression after ControlNet training - the failure mode that killed naive fine-tuning.
- Won the Marr Prize (best paper) at ICCV 2023.

---

## Why This Was Revolutionary

- **Solved conditioning without catastrophic forgetting**, using an idea (zero initialization) simple enough to explain in a sentence.
- **Turned image generation into a production pipeline step.** Storyboard artists could lock composition; product teams could lock the product's shape and vary the scene.
- **Established the adapter pattern for diffusion.** T2I-Adapter, IP-Adapter, InstantID, and the entire SDXL/Flux control ecosystem follow ControlNet's frozen-base plus trained-side-network template.
- **Democratized control.** Because training is cheap, hundreds of community ControlNets exist for niche conditions nobody at a lab would have prioritized.

---

## Real-World Impact

- **Standard in every serious image workflow.** ComfyUI and Automatic1111 ship ControlNet as a first-class feature; most professional Stable Diffusion and Flux pipelines use at least one.
- **Consistent characters and scenes** across a sequence of images, which is the hard requirement for comics, storyboards, and marketing sets.
- **Architecture and interior design** - feed a depth or line render of a CAD model, restyle in a hundred ways without changing the geometry.
- **Video.** Applying the same control per frame (with temporal modules) is a core technique in AnimateDiff-style video pipelines and informs how controllability is being added to [DiT-based video models](../44-sora-dit/summary.md).
- **The zero-init idea spread.** Zero-initialized gating shows up in adapter designs across modalities, including in LLM adapter layers.

---

## Key Takeaways for Practitioners

1. **Match the control to the job.** Canny preserves too much (you will see the traced edges); depth preserves layout while letting the model reinterpret surfaces; pose preserves the figure only.
2. **Turn control off before the end.** Ending control at 60 to 80 percent of steps usually produces more natural images than full-strength control throughout.
3. **Preprocessing quality dominates.** A bad depth map or a noisy Canny threshold degrades output more than any sampler setting.
4. **Stack sparingly.** Two well-chosen ControlNets is usually better than four fighting each other; sum of weights above roughly 1.5 tends to produce rigid output.
5. **The frozen-base pattern generalizes.** When you need to add a capability to a large pretrained model on limited data, cloning a branch and gating it with a zero-initialized connection is a reliably safe recipe.

---

## Limitations & Future Directions

- **One ControlNet per condition type**, each a few hundred megabytes to a couple of gigabytes. Multi-condition unified models (and Flux's unified control variants) address this.
- **Memory and latency cost** - roughly a 30 to 50 percent overhead per active ControlNet.
- **Control can over-constrain.** Strong edge control produces images that look like a coloring-book fill rather than a photograph.
- **Requires a preprocessor** for real inputs, so the pipeline has a dependency on an external depth or pose estimator whose failures propagate.
- **Retraining per base model.** A Stable Diffusion 1.5 ControlNet does not work on SDXL or Flux; the ecosystem re-trains for each new base.

---

## Further Reading

- **Original Paper:** [arxiv.org/abs/2302.05543](https://arxiv.org/abs/2302.05543)
- **Code:** [github.com/lllyasviel/ControlNet](https://github.com/lllyasviel/ControlNet)
- **T2I-Adapter (lighter alternative):** [arxiv.org/abs/2302.08453](https://arxiv.org/abs/2302.08453)
- **IP-Adapter (image prompt conditioning):** [arxiv.org/abs/2308.06721](https://arxiv.org/abs/2308.06721)

## Citation

```bibtex
@inproceedings{zhang2023adding,
  title={Adding Conditional Control to Text-to-Image Diffusion Models},
  author={Zhang, Lvmin and Rao, Anyi and Agrawala, Maneesh},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
  year={2023}
}
```

<!-- related:start -->

---

## Related in This Collection

- [High-Resolution Image Synthesis with Latent Diffusion Models (Stable Diffusion)](../../image-generation/07-stable-diffusion/summary.md)
- [LoRA: Low-Rank Adaptation of Large Language Models](../../techniques/10-lora/summary.md)
- [Sora and Diffusion Transformers (DiT): Video Generation as World Simulation](../../image-generation/44-sora-dit/summary.md)

<!-- related:end -->
