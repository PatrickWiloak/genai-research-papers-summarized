---
title: "3D Gaussian Splatting for Real-Time Radiance Field Rendering (3D Gaussian Splatting)"
slug: "118-3d-gaussian-splatting"
number: 118
category: "image-generation"
authors: "Bernhard Kerbl, Georgios Kopanas, Thomas Leimkühler, George Drettakis (Inria and Université Côte d'Azur, Max-Planck-Institut für Informatik)"
published: "August 2023 (SIGGRAPH 2023, ACM Transactions on Graphics 42(4); Best Paper Award)"
year: 2023
url: "https://arxiv.org/abs/2308.04079"
tags: ["3d", "computer-vision", "efficiency"]
---

# 3D Gaussian Splatting for Real-Time Radiance Field Rendering (3D Gaussian Splatting)

**Authors:** Bernhard Kerbl, Georgios Kopanas, Thomas Leimkühler, George Drettakis (Inria and Université Côte d'Azur, Max-Planck-Institut für Informatik)
**Published:** August 2023 (SIGGRAPH 2023, ACM Transactions on Graphics 42(4); Best Paper Award)
**Paper:** [arxiv.org/abs/2308.04079](https://arxiv.org/abs/2308.04079)

---

## Why This Matters

3D Gaussian Splatting (3DGS) is **the method that made radiance fields real-time**. [NeRF](../117-nerf/summary.md) showed that you could turn photos into a photorealistic, explorable 3D scene, but the best NeRF variants took hours or days to train and rendered at well under one frame per second. 3DGS matched the best NeRF quality, trained in minutes, and rendered at over 100 frames per second at 1080p.

- **Real-time at state-of-the-art quality.** On the Mip-NeRF 360 benchmark it rendered at 134 fps against 0.06 fps for Mip-NeRF 360, with similar image quality.
- **Fast training.** About 40 minutes for full quality, or about 6 minutes for a result comparable to Instant-NGP, versus up to 48 hours for Mip-NeRF 360.
- **No neural network in the scene.** The scene is an explicit cloud of millions of small, coloured, semi-transparent 3D blobs, optimised directly by gradient descent.
- **It displaced NeRF in practice.** Within about two years, "splats" had become the default output of 3D capture apps and a popular research target.

**The insight:** keep NeRF's image-formation model (alpha-blend colours along each ray) but change the representation. Instead of querying a network at hundreds of points per ray, describe the scene with 3D Gaussians that can be projected onto the screen and blended with a fast, sort-based GPU rasteriser, the kind of operation graphics hardware is built for.

---

## The Problem: Quality Or Speed, Pick One

By 2022 radiance fields had split into two camps:

- **High quality, very slow.** Mip-NeRF 360 gave the best images on unbounded real scenes but needed up to 48 hours of training and rendered at a fraction of a frame per second, because every pixel requires many network queries along its ray.
- **Fast, lower quality.** Plenoxels (sparse voxel grids) and Instant-NGP (hash-grid features plus a tiny network) trained in minutes, but lost quality on large unbounded scenes and still rendered at only about 10 fps.

Both camps share a cost: rays must be marched through space, including empty space, and sampled many times. For complete scenes at 1080p, no method reached real-time display rates (30 fps or more). That rules out VR, games, and most interactive use.

---

## The Core Innovation

Represent the scene as a set of **anisotropic 3D Gaussians** (ellipsoid-shaped fuzzy blobs, stretched differently along different axes), each with:

```
position      (x, y, z)          where it sits
covariance    rotation + scale   its size, shape and orientation
opacity       alpha              how solid it is
colour        spherical-harmonic coefficients (colour that varies
              with viewing direction, for shine and reflections)
```

To render, project every Gaussian onto the image (a 3D Gaussian projects to a 2D Gaussian "splat"), sort them by depth, and alpha-blend front to back. This is the same image-formation rule NeRF uses, but over a sorted list of splats rather than hundreds of samples per ray:

```
NeRF:   for each pixel -> march a ray -> query an MLP ~200 times -> blend
3DGS:   for each Gaussian -> project once -> sort by depth
        for each pixel -> blend the few splats that cover it
```

Because projection, sorting and blending are all differentiable, the positions, shapes, opacities and colours of all Gaussians can be optimised by comparing renders against the training photos, just like NeRF.

---

## Key Components Explained

### 1. Initialisation from structure-from-motion points
**What it does:** Gives the optimisation a sensible starting point.
**How it works:** Camera poses for the input photos come from a structure-from-motion tool (COLMAP), which also produces a sparse 3D point cloud as a by-product. Each point becomes an initial Gaussian. The ablations show random initialisation noticeably hurts quality on real scenes.

### 2. Anisotropic covariance, parameterised safely
**What it does:** Lets a few large, flat or thin Gaussians cover surfaces and fine structures efficiently.
**How it works:** A covariance matrix must stay valid (positive semi-definite) during gradient descent, which raw matrix entries do not guarantee. The paper stores each Gaussian's shape as a scale vector and a rotation quaternion and builds the covariance from them. The ablations show that forcing Gaussians to be spheres (isotropic) lowers quality.

### 3. Adaptive density control: clone, split, prune
**What it does:** Grows detail where the scene needs it and removes waste.
**How it works:** Every 100 iterations, Gaussians with large position gradients (a sign the region is not yet well explained) are densified:

```
small Gaussian in an under-covered region  -> CLONE it (copy, nudge along gradient)
large Gaussian covering too much detail    -> SPLIT it into two,
                                              scale divided by 1.6
nearly transparent Gaussian                -> PRUNE it
every 3,000 iterations                     -> reset opacities near zero,
                                              so useless ones fade and get pruned
```

Scenes end up with roughly 1 to 5 million Gaussians. The loss is a mix of per-pixel L1 error and a structural similarity (D-SSIM) term.

### 4. Tile-based differentiable rasteriser
**What it does:** Makes both rendering and training fast.
**How it works:** The screen is divided into 16x16-pixel tiles. Each projected Gaussian is assigned to the tiles it overlaps, with a key combining tile ID and depth, and all of them are sorted in a single GPU radix sort. Each tile then blends its own depth-sorted list, stopping early once pixels become fully opaque. Unlike earlier point-based renderers, there is no fixed cap on how many splats can receive gradients per pixel, which matters for quality.

---

## Key Results

Table 1 of the paper, Mip-NeRF 360 dataset (A6000 GPU except where noted; Mip-NeRF 360 numbers are copied from its paper):

```
Method              PSNR   SSIM   LPIPS  Train    FPS    Memory
------------------  -----  -----  -----  -------  -----  ------
Plenoxels           23.08  0.626  0.463  25m49s    6.79  2.1 GB
Instant-NGP Base    25.30  0.671  0.371   5m37s   11.7   13 MB
Instant-NGP Big     25.59  0.699  0.331   7m30s    9.43  48 MB
Mip-NeRF 360        27.69  0.792  0.237  48h       0.06  8.6 MB
3DGS (7K iters)     25.60  0.770  0.279   6m25s  160     523 MB
3DGS (30K iters)    27.21  0.815  0.214  41m33s  134     734 MB
```

- **Quality:** at 30K iterations 3DGS roughly matches Mip-NeRF 360 (slightly lower PSNR, better SSIM and LPIPS).
- **Speed:** about 2,000 times faster rendering than Mip-NeRF 360 and over 10 times faster than Instant-NGP.
- **The cost is memory:** hundreds of megabytes per scene, against megabytes for NeRF-style models.
- Similar patterns held on the Tanks and Temples and Deep Blending datasets.

---

## Why This Was Revolutionary

- **Broke the quality-versus-speed trade-off** that had defined radiance field research.
- **Brought radiance fields back into the graphics pipeline.** Splats are explicit primitives that can be moved, deleted, combined and streamed, which neural fields cannot easily do.
- **Showed that the neural network was optional.** The lasting contribution of NeRF turned out to be differentiable volumetric rendering from photos, not the MLP.
- **Practical code.** The authors released a CUDA implementation and viewer, which the community built on quickly.

---

## Real-World Impact

- **Rapid adoption in capture tools.** Consumer and professional 3D capture apps, game-engine plugins and web viewers added Gaussian splat support within months of publication.
- **A large research family:** compression of splat scenes, dynamic and 4D Gaussians for video, 2D Gaussian Splatting (2024) for more accurate surfaces, feed-forward models that predict Gaussians from a few images without per-scene optimisation, and text-to-3D pipelines that output splats instead of NeRFs.
- **Standardisation.** By 2025, industry groups such as the Metaverse Standards Forum were holding workshops on a common interchange format for splats.
- **Robotics and mapping.** Splat-based SLAM and scene maps have become an active line of work because they render fast enough for closed-loop use.

---

## Key Takeaways for Practitioners

1. **Default to splatting for interactive viewing** of captured scenes; it is what current tooling supports best.
2. **Good poses and a decent point cloud still matter.** 3DGS inherits NeRF's dependence on structure-from-motion.
3. **Budget memory, not just time.** Hundreds of megabytes per scene is normal, and peak training memory can exceed 20 GB in the original implementation.
4. **Seven thousand iterations is often enough** for a usable preview; the paper shows quality at 7K is already close on many scenes.
5. **Watch for view-dependent artefacts.** Shiny or poorly observed regions are where splats misbehave first.

---

## Limitations & Future Directions

- **Memory-hungry.** Scenes take hundreds of megabytes, and the authors report that peak GPU memory during training can exceed 20 GB in their unoptimised prototype. Compression research followed quickly.
- **Artefacts in poorly observed regions:** elongated or "splotchy" Gaussians where the capture is thin, which other methods also struggle with.
- **Popping.** When large Gaussians change sort order between frames, they can visibly pop; the authors attribute this partly to their simple culling and depth-sorting approach.
- **Geometry is approximate.** Gaussians are not surfaces, so meshes and normals extracted from them are noisy; 2D Gaussian Splatting and related work target this.
- **Still per-scene and still static** in the original; dynamic scenes and generalising models came later.

---

## Further Reading

- **Original Paper:** [arxiv.org/abs/2308.04079](https://arxiv.org/abs/2308.04079)
- **Project page and code:** [repo-sam.inria.fr/fungraph/3d-gaussian-splatting](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/)
- **Mip-NeRF 360:** [arxiv.org/abs/2111.12077](https://arxiv.org/abs/2111.12077)
- **Instant-NGP:** [arxiv.org/abs/2201.05989](https://arxiv.org/abs/2201.05989)
- **2D Gaussian Splatting:** [arxiv.org/abs/2403.17888](https://arxiv.org/abs/2403.17888)
- **In this collection:** [NeRF](../117-nerf/summary.md)

## Citation

```bibtex
@article{kerbl3Dgaussians,
  title={3D Gaussian Splatting for Real-Time Radiance Field Rendering},
  author={Kerbl, Bernhard and Kopanas, Georgios and Leimk{\"u}hler, Thomas and Drettakis, George},
  journal={ACM Transactions on Graphics},
  volume={42},
  number={4},
  year={2023}
}
```

<!-- related:start -->

---

## Related in This Collection

- [NeRF: Representing Scenes as Neural Radiance Fields for View Synthesis (NeRF)](../../image-generation/117-nerf/summary.md)

<!-- related:end -->
