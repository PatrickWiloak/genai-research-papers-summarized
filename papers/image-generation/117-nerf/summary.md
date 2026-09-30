---
title: "NeRF: Representing Scenes as Neural Radiance Fields for View Synthesis (NeRF)"
slug: "117-nerf"
number: 117
category: "image-generation"
authors: "Ben Mildenhall, Pratul P. Srinivasan, Matthew Tancik, Jonathan T. Barron, Ravi Ramamoorthi, Ren Ng (UC Berkeley, Google Research, UC San Diego)"
published: "March 2020 (ECCV 2020, oral; Best Paper Honorable Mention)"
year: 2020
url: "https://arxiv.org/abs/2003.08934"
tags: ["3d", "image-generation", "computer-vision"]
---

# NeRF: Representing Scenes as Neural Radiance Fields for View Synthesis (NeRF)

**Authors:** Ben Mildenhall, Pratul P. Srinivasan, Matthew Tancik, Jonathan T. Barron, Ravi Ramamoorthi, Ren Ng (UC Berkeley, Google Research, UC San Diego)
**Published:** March 2020 (ECCV 2020, oral; Best Paper Honorable Mention)
**Paper:** [arxiv.org/abs/2003.08934](https://arxiv.org/abs/2003.08934)

---

## Why This Matters

NeRF is **the paper that made "store a 3D scene inside a neural network" work photorealistically**. Give it a few dozen ordinary photos of a scene with known camera positions, and it learns a representation from which you can render brand-new viewpoints, with reflections, fine geometry and soft transparency, that look like real photographs.

- **A tiny network holds a whole scene.** The trained weights take about 5 MB, which the paper points out is less than the input photos themselves.
- **No 3D supervision.** No depth sensor, no meshes, no ground-truth geometry. Just 2D images and their camera poses.
- **A large jump in quality.** On the paper's realistic synthetic benchmark it scored 31.01 dB PSNR against 26.05 dB for the best prior method, a gap you can see with the naked eye.
- **It started a field.** "Neural radiance fields" became one of the most active research areas in computer vision for several years, and its image-formation model is inherited directly by [3D Gaussian Splatting](../118-3d-gaussian-splatting/summary.md), which replaced it in most practical use.

**The insight:** treat the scene as a continuous function that can be queried at any point, and render it with the classic, fully differentiable volume rendering equation. Because every step from "network weights" to "pixel colour" is differentiable, you can train the network with nothing more than "make the rendered pixel match the photographed pixel."

---

## The Problem: Photos In, New Views Out

**View synthesis** means taking a set of photos of a scene and producing an image from a camera position nobody photographed. It is the core of virtual tours, visual effects, VR capture and robot mapping.

Before NeRF, the main approaches each had a clear weakness:

- **Meshes and textures** (classic photogrammetry) struggle with thin structures, transparency, and shiny, view-dependent surfaces. Optimising a mesh by gradient descent is awkward because mesh topology is discrete.
- **Voxel grids** (3D arrays of colour and opacity, such as Neural Volumes) are easy to optimise, but memory grows with the cube of resolution, so detail is capped. Neural Volumes used a 128^3 grid.
- **Multiplane images** (stacks of semi-transparent layers, such as LLFF) are quick to compute but only suit roughly forward-facing captures and produce huge per-scene storage. The paper measured over 15 GB for LLFF on one synthetic scene.
- **Earlier neural scene networks** (such as Scene Representation Networks) represented only a single opaque surface and gave over-smoothed results.

The field needed a representation that was continuous (no resolution cap), compact, handled transparency and view-dependent effects, and could be optimised end to end from photos.

---

## The Core Innovation

Represent the scene as a function that maps a **5D input** (a 3D position plus a 2D viewing direction) to a **colour and a density**, and parameterise that function with a plain multilayer perceptron (MLP), a stack of fully connected layers.

```
F_theta : (x, y, z, theta, phi)  ->  (r, g, b, sigma)

  x, y, z      where in space
  theta, phi   which direction you are looking from
  r, g, b      colour emitted toward the viewer from that point
  sigma        density: how much "stuff" is there (opacity per unit length)
```

To render a pixel, shoot a ray from the camera through that pixel, sample points along it, ask the network for colour and density at each point, and blend them with the volume rendering equation:

```
Colour of ray  ~=  sum over samples i of   T_i * (1 - exp(-sigma_i * delta_i)) * c_i

  where T_i = exp( - sum over j < i of sigma_j * delta_j )

  delta_i                        distance to the next sample
  (1 - exp(-sigma_i * delta_i))  how opaque this short segment is
  T_i                            transmittance: how much light survives
                                 to reach sample i without being blocked
```

Each sample contributes its colour in proportion to how solid it is and how visible it is from the camera. A solid wall early on the ray blocks everything behind it; fog lets some light through.

Training is then very simple:

```
repeat:
    pick a batch of 4,096 pixels (rays) from the training photos
    render each ray through the network
    loss = squared error between rendered colour and true colour
    backpropagate into the MLP weights
```

One network is trained per scene. There is no dataset of scenes and no generalisation across scenes: the network *is* the scene.

---

## Key Components Explained

### 1. Density depends on position only; colour depends on position and direction
**What it does:** Keeps geometry consistent across views while allowing shiny, view-dependent appearance.
**How it works:** The position passes through 8 fully connected layers (256 channels each, ReLU, with a skip connection that feeds the input back in at the fifth layer). That produces the density and a feature vector. Only then is the viewing direction added, followed by one more 128-channel layer that outputs RGB. So the *shape* of the scene cannot change with viewpoint but its *colour* can, which is how specular highlights behave. The paper's ablations show that without view dependence, reflections such as those on the microphone stand are lost.

### 2. Positional encoding
**What it does:** Lets a small MLP represent sharp edges and fine texture.
**How it works:** Neural networks are biased toward learning smooth, low-frequency functions, so raw coordinates give blurry results. NeRF first maps each coordinate through sines and cosines at increasing frequencies:

```
gamma(p) = ( sin(2^0 * pi * p), cos(2^0 * pi * p),
             ...
             sin(2^(L-1) * pi * p), cos(2^(L-1) * pi * p) )

L = 10 for position (x, y, z)
L = 4  for viewing direction
```

The paper notes the resemblance to the positional encoding in the [Transformer](../../architectures/01-attention-is-all-you-need/summary.md) but stresses the different purpose: here it lets the network fit high-frequency detail rather than marking token order. In the ablation study, removing positional encoding and removing view dependence were the two largest quality losses.

### 3. Hierarchical (coarse-to-fine) sampling
**What it does:** Spends network queries where the scene actually is.
**How it works:** Most of a ray passes through empty air or the inside of solid objects, where samples are wasted. NeRF trains two networks. A "coarse" network is evaluated at 64 stratified points along each ray; its blending weights form a rough probability distribution over where visible content lies. A further 128 points are drawn from that distribution, and a "fine" network is evaluated at all 192. This is similar in spirit to importance sampling: concentrate effort near surfaces.

### 4. Nothing but posed photographs
**What it does:** Makes capture practical.
**How it works:** Camera positions and intrinsics for real scenes come from COLMAP, a standard structure-from-motion tool. The real scenes in the paper were captured handheld with a phone, 20 to 62 images each.

---

## Key Results

From Table 1 of the paper (PSNR in dB, higher is better; each column uses the baselines that could run on it):

```
Dataset                          Best prior            NeRF
-------------------------------  --------------------  ------
Diffuse Synthetic 360 (4 objs)   34.38 (LLFF)          40.15
Realistic Synthetic 360 (8)      26.05 (Neural Vol.)   31.01
Real Forward-Facing (8 scenes)   24.13 (LLFF)          26.50
```

- NeRF beat the other per-scene methods (Neural Volumes, SRN) on every metric and beat LLFF on all but one (LLFF had slightly better LPIPS on the real scenes; the authors argue NeRF's multi-view consistency is visibly better in video).
- **With only 25 input images**, NeRF still beat all baselines that were given 100 images on the synthetic set.
- **Storage:** about 5 MB of weights per scene, roughly 3,000 times smaller than LLFF's output.
- **Cost:** training took 100k to 300k iterations, about 1 to 2 days per scene on one NVIDIA V100. Rendering one frame took 150 to 200 million network queries, about 30 seconds per frame on a V100. That slowness is the method's defining weakness.

---

## Why This Was Revolutionary

- **Photorealistic quality from casual captures.** Ship rigging, gear teeth, shiny materials and occlusions were reproduced far better than by any prior view synthesis method.
- **Differentiable rendering as the whole training signal.** A classic graphics equation, used as a differentiable layer, was enough supervision on its own. That idea underlies almost all later inverse-rendering work.
- **Coordinate networks went mainstream.** "An MLP from coordinates to values, plus a frequency encoding" spread to images, audio, signed distance fields and medical imaging.
- **A standard benchmark.** The "Realistic Synthetic 360" objects (Lego, Ship, Microphone, Materials and others) became the default test set for years.

---

## Real-World Impact

- **An explosion of follow-ups:** anti-aliasing (Mip-NeRF), unbounded 360-degree scenes (Mip-NeRF 360), dynamic scenes, relighting, few-view inputs and city-scale capture.
- **The speed race.** Plenoxels (voxel grids with no network at all) and Instant-NGP (multiresolution hash encodings, NVIDIA 2022) cut training from days to minutes. Instant-NGP showed the big MLP was not the essential ingredient; the volume rendering formulation was.
- **Replaced in practice by [3D Gaussian Splatting](../118-3d-gaussian-splatting/summary.md) (2023),** which keeps NeRF's image-formation model but swaps the neural field for millions of explicit 3D Gaussians and reaches real-time rendering. As of 2026 most practical capture pipelines use splatting, which descends directly from this paper's formulation.
- **Text-to-3D.** DreamFusion (Google, 2022) optimised a NeRF so that its renders looked right to a frozen 2D text-to-image [diffusion model](../06-diffusion-models/summary.md) ([Imagen](../91-imagen/summary.md)), turning a 2D generator into a 3D one.

---

## Key Takeaways for Practitioners

1. **The rendering equation is the durable part.** MLPs, voxel grids, hash tables and Gaussians have all been swapped in; alpha-compositing along rays with transmittance stayed.
2. **Use a frequency encoding whenever an MLP takes coordinates as input.** Without one, expect blur.
3. **Camera poses are the hidden dependency.** Quality collapses with bad poses; many "NeRF failed" cases are really "pose estimation failed."
4. **Per-scene optimisation is not a model you download.** Each scene is a fresh training run, which shapes cost and product design.
5. **If you need real-time rendering, start with Gaussian splatting**; NeRF-style fields remain useful when you want a compact, continuous representation.

---

## Limitations & Future Directions

- **Very slow:** 1 to 2 days of training and about 30 seconds per frame in the original. Addressed by Plenoxels and Instant-NGP, then by 3D Gaussian Splatting.
- **Static scenes only.** Anything that moves between photos turns into ghostly "floaters." Later work added time as an input.
- **Needs accurate camera poses** from an external tool, and fairly dense views.
- **Baked-in lighting.** The network learns emitted colour, not materials and lights, so scenes cannot easily be relit or edited.
- **Aliasing and unbounded scenes.** Rays were treated as infinitely thin lines and scenes as bounded or forward-facing; Mip-NeRF and Mip-NeRF 360 addressed both.
- **Geometry is implicit.** Extracting a clean mesh from a density field is possible but lossy.

---

## Further Reading

- **Original Paper:** [arxiv.org/abs/2003.08934](https://arxiv.org/abs/2003.08934)
- **Mip-NeRF 360:** [arxiv.org/abs/2111.12077](https://arxiv.org/abs/2111.12077)
- **Instant-NGP (hash encodings):** [arxiv.org/abs/2201.05989](https://arxiv.org/abs/2201.05989)
- **DreamFusion (text-to-3D with NeRF):** [arxiv.org/abs/2209.14988](https://arxiv.org/abs/2209.14988)
- **In this collection:** [3D Gaussian Splatting](../118-3d-gaussian-splatting/summary.md), [Attention Is All You Need](../../architectures/01-attention-is-all-you-need/summary.md), [Diffusion Models](../06-diffusion-models/summary.md)

## Citation

```bibtex
@inproceedings{mildenhall2020nerf,
  title={NeRF: Representing Scenes as Neural Radiance Fields for View Synthesis},
  author={Mildenhall, Ben and Srinivasan, Pratul P. and Tancik, Matthew and Barron, Jonathan T. and Ramamoorthi, Ravi and Ng, Ren},
  booktitle={European Conference on Computer Vision (ECCV)},
  year={2020}
}
```

<!-- related:start -->

---

## Related in This Collection

- [Attention Is All You Need](../../architectures/01-attention-is-all-you-need/summary.md)
- [Photorealistic Text-to-Image Diffusion Models with Deep Language Understanding (Imagen)](../../image-generation/91-imagen/summary.md)
- [3D Gaussian Splatting for Real-Time Radiance Field Rendering (3D Gaussian Splatting)](../../image-generation/118-3d-gaussian-splatting/summary.md)

<!-- related:end -->
