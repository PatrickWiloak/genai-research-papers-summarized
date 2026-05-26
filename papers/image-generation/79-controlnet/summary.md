# Adding Conditional Control to Text-to-Image Diffusion Models (ControlNet)

**Authors:** Lvmin Zhang, Anyi Rao, Maneesh Agrawala (Stanford University)

**Published:** February 2023 (ICCV 2023, best paper honorable mention)

**Paper Link:** https://arxiv.org/abs/2302.05543

---

## Why This Paper Matters

Stable Diffusion gave anyone with a GPU a text-to-image model. But text prompts alone are a frustrating interface: you can describe "a knight in armor running through a forest," but you can't easily say "make him in *this exact pose* with *that exact composition*." ControlNet is the paper that solved this. It introduced a clean, almost surgical way to add structural conditioning — edges, depth, human poses, segmentation maps, scribbles — to a frozen pretrained diffusion model without breaking it.

The release of ControlNet (with code and pretrained models for ~8 control modalities simultaneously) instantly turned Stable Diffusion from a slot-machine generator into a real creative tool. It also established a design pattern — **trainable copy + zero convolutions** — that now appears throughout the adapter / LoRA (#10) / IP-Adapter / T2I-Adapter ecosystem. Modern Stable Diffusion workflows in tools like ComfyUI, Automatic1111, and InvokeAI typically stack multiple ControlNets per generation.

---

## The Problem Before

Stable Diffusion (#07) produces stunning images, but control is loose:

- **Prompts** describe content but not layout. "A cat on the left, a dog on the right" frequently swaps positions, merges animals, or ignores the spatial instruction.
- **Img2img / inpainting** give some spatial control but can't condition on *abstract structure* like a stick-figure pose or a depth map.
- **Full fine-tuning** of the entire 860M-parameter U-Net on a new task is expensive, requires lots of paired data, and tends to forget the original capabilities (catastrophic forgetting).

What was needed: a way to inject structured control signals (edges, poses, depth) into a fixed diffusion model *additively* — preserving everything the base model already knows while adding new conditioning channels.

---

## The Core Innovation: Trainable Copy + Zero Convolutions

ControlNet's design has two key pieces.

### 1. The trainable copy

The pretrained Stable Diffusion U-Net encoder is **cloned**. The original copy stays frozen; the clone is made trainable. The clone receives the new control input (e.g., a Canny edge map or a depth image) plus the noisy latent, and learns to produce per-layer feature adjustments. The decoder of the U-Net stays the original and adds these adjustments to its skip connections.

This means the model never has to *forget* anything: the base U-Net is bitwise unchanged, and the trainable copy starts from the same weights, so it begins with all of Stable Diffusion's visual knowledge intact rather than learning from scratch.

```
Noisy latent z_t  +  Text prompt
        |
        v
   Frozen U-Net encoder ---> skip connections ---> Frozen U-Net decoder ---> denoised z_{t-1}
        ^                          ^
        |                          |  (add ControlNet outputs here)
        |                          |
   Trainable copy of encoder       |
        ^                          |
        |          zero convs -----+
        |
   Control image (edges, depth, pose, ...)
```

### 2. Zero convolutions

The connection between the trainable copy and the frozen decoder goes through **1x1 convolutions whose weights are initialized to zero**. At step 0 of training:

- Zero convolutions output zero.
- The control branch therefore contributes nothing.
- The model's output is identical to the original Stable Diffusion.

This is critical. Most fine-tuning methods inject random noise into a pretrained model at step 0, briefly destroying its capabilities and forcing recovery. With zero convs, training **only adds capability** — gradients flow only in directions that help the new control signal, never directions that hurt the base model.

Both the weights and the biases of these connection layers start at zero. Despite producing zero outputs, gradients still flow because of how the chain rule interacts with the residual structure — so the layer can learn to be nonzero where useful, but never accidentally degrades the base model along the way.

---

## How It's Trained

For each control type (Canny edges, HED soft edges, depth, normal maps, segmentation, scribble, OpenPose skeletons, M-LSD lines, etc.):

1. Take a dataset of (image, text caption) pairs.
2. Automatically extract the corresponding control signal (e.g., run Canny on the image).
3. Train the ControlNet branch to predict noise from (noisy latent, prompt, control image), with the base U-Net frozen.

Training is remarkably cheap by 2023 standards: ~600 hours on a single A100 for many of the control types, and 50k-300k image-control pairs are enough. This makes it accessible to small labs and hobbyists.

A small but important detail: during training, the text prompt is dropped 50% of the time. This forces the control image alone to carry meaningful information, which in turn improves robustness when prompts are weak or contradictory.

---

## Key Results

- **Eight distinct controls** released in the original paper: Canny edges, HED soft edges, user scribbles, M-LSD straight lines, HED edges, depth (MiDaS), normal maps, semantic segmentation (ADE20K), and OpenPose skeletons. All work with the same base Stable Diffusion 1.5.
- **No quality regression** on prompts the controls don't touch — the base model is preserved exactly.
- **Composable controls**: multiple ControlNets can be stacked at inference (e.g., pose + depth) by summing their contributions.
- **Robust to weak data**: even small datasets (50k pairs) produced usable models for some control types, thanks to starting from the strong pretrained encoder.

In user studies, generations with ControlNet matched the structural intent of the control image while still respecting the text prompt — something pure prompt-engineering could not achieve.

---

## Why Zero Convolutions Actually Train

A natural worry: if the output is zero, isn't the gradient zero too? Not quite. Consider a zero-initialized 1x1 conv layer `y = W*x + b` with `W=0` and `b=0`. Its forward pass outputs zero, so it contributes nothing — but its gradient with respect to W is **`dL/dW = x * dL/dy`**, which is generally nonzero. So during the first backward pass, W gets updated based on how the loss responds to the layer's output. After the first step, W is no longer zero, the layer starts producing nonzero outputs, and training proceeds normally.

The elegant property: the model **never had to undo damage**. If you'd initialized W randomly (the conventional choice), the layer would immediately corrupt the frozen decoder's input, hurting performance briefly while the model recovered. Zero init means the training trajectory starts from "exactly the original model" and moves monotonically toward "original model + useful new conditioning."

This is closely related to the **identity-preserving initialization** trick used in many residual architectures and to **LoRA**'s convention of initializing one of its two low-rank matrices to zero.

---

## Practical Workflow Notes

A few things every Stable Diffusion practitioner discovered about ControlNet within weeks of release:

- **The control image quality matters enormously.** Garbage in, garbage out — bad depth maps produce bad geometry. Tools quickly emerged to preview/edit control images before generation.
- **Conditioning strength is tunable.** A `controlnet_conditioning_scale` parameter (typically 0.5-1.5) lets users dial the control's influence: low for "loose suggestion," high for "rigid adherence."
- **Stacking is the killer feature.** Combining OpenPose (for a character's pose) + Canny (for a background's structure) + depth (for 3D consistency) gives almost cinematic-level control over composition.
- **ControlNet doesn't replace prompts** — it complements them. The prompt still controls style, content, identity; the control image controls geometry and layout.
- **Inference is roughly 2x more expensive** than vanilla Stable Diffusion because the trainable copy runs alongside the frozen encoder. Multi-ControlNet stacks scale linearly in cost.

---

## Limitations

ControlNet is powerful but not perfect:

- **One control modality per model.** Each ControlNet is trained for a specific type of input; you can't ask a single net to handle both "edges" and "depth." Stacking multiple ControlNets solves this at runtime cost.
- **Quality is upper-bounded by the base model.** ControlNet adds control, not capability — if Stable Diffusion can't render hands well, ControlNet can't fix that.
- **Conflicting controls cause artifacts.** Combining a pose that says "arm raised" with a depth map that puts the arm down produces visually broken results.
- **Training data leakage.** Some ControlNets memorize their training distribution (e.g., a depth ControlNet trained on indoor scenes may struggle with outdoor depth maps).

---

## Impact and Legacy

ControlNet was the **largest single jump** in usable controllability for diffusion models. Its consequences:

- **The Stable Diffusion ecosystem exploded.** Pose-conditioned character generation, depth-controlled environments, sketch-to-image, room redecoration from a segmentation map — all became standard workflows within months.
- **Adapter pattern became the default.** Subsequent works — T2I-Adapter, IP-Adapter (image-conditioned), ReferenceNet, LoRA (#10) for diffusion — all follow some variant of "freeze the base, add a small trainable branch." ControlNet legitimized this approach as the production pattern, not just a research curiosity.
- **Zero-initialization tricks** proliferated. The "start at zero so you can't hurt the base model" idea now appears in many fine-tuning techniques and is closely related to LoRA's zero-initialized B matrix.
- **Influence on video and 3D.** AnimateDiff, video ControlNets, and 3D-aware diffusion all build on the same trainable-copy template.
- **Conceptually**, ControlNet helped solidify the worldview that **pretrained generative models are foundation models** — you adapt them, you don't retrain them.

---

## Connections to Other Papers

- **Stable Diffusion (#07):** The base model ControlNet conditions. The trainable encoder is literally a copy of SD's U-Net encoder.
- **DDPM (#06):** ControlNet operates on the same denoising objective; only the inputs to the noise predictor change.
- **LoRA (#10):** Sibling technique for parameter-efficient adaptation. LoRA modifies attention weights in-place with low-rank deltas; ControlNet adds a parallel encoder branch. Both rely on zero (or near-zero) initialization to preserve base-model behavior.
- **DreamBooth (#80):** Complementary — DreamBooth personalizes the *content* (specific subjects); ControlNet controls the *structure* (poses, layouts).
- **DALL-E 3 (#48):** A different approach to controllability — improve the language understanding rather than add structural conditioning. The two strategies are now combined in production systems.

---

## Key Takeaways

- **Trainable copy of the encoder + zero convolutions** = add new conditioning to a pretrained diffusion model without ever degrading it.
- **Zero-initialized connection layers** guarantee training begins as a no-op, so it can only improve, never hurt — a powerful general principle for adapting foundation models.
- **A single architecture handles many control types** (edges, depth, pose, segmentation, scribbles) — each is just a different trained ControlNet on the same frozen base.
- **Made diffusion controllable enough for professional creative work**, transforming Stable Diffusion from a curiosity into a production tool.
- **Established the adapter pattern** that now dominates fine-tuning across image, video, and 3D generation — and inspired downstream work like LoRA for diffusion, IP-Adapter, and AnimateDiff.
