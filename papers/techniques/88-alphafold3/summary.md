# Accurate Structure Prediction of Biomolecular Interactions with AlphaFold 3

**Authors:** Josh Abramson, Jonas Adler, Jack Dunger, Richard Evans, Tim Green, Alexander Pritzel, Olaf Ronneberger, Lindsay Willmore, Andrew J. Ballard, Joshua Bambrick, Sebastian W. Bodenstein, David A. Evans, Chia-Chun Hung, Michael O'Neill, David Reiman, Kathryn Tunyasuvunakool, Zachary Wu, Akvile Zemgulyte, Eirini Arvaniti, Charles Beattie, Ottavia Bertolli, Alex Bridgland, Alexey Cherepanov, Miles Congreve, Alexander I. Cowen-Rivers, Andrew Cowie, Michael Figurnov, Fabian B. Fuchs, Hannah Gladman, Rishub Jain, Yousuf A. Khan, Caroline M. R. Low, Kuba Perlin, Anna Potapenko, Pascal Savy, Sukhdeep Singh, Adrian Stecula, Ashok Thillaisundaram, Catherine Tong, Sergei Yakneen, Ellen D. Zhong, Michal Zielinski, Augustin Zidek, Victor Bapst, Pushmeet Kohli, Max Jaderberg, Demis Hassabis, John M. Jumper (DeepMind and Isomorphic Labs)
**Published:** May 2024 (Nature, vol. 630, pp. 493-500)
**Paper:** [nature.com/articles/s41586-024-07487-w](https://www.nature.com/articles/s41586-024-07487-w) — DOI: 10.1038/s41586-024-07487-w

---

## Why This Paper Matters

AlphaFold 2 predicted the shape of single proteins. AlphaFold 3 predicts the shape of essentially anything biology cares about — proteins interacting with other proteins, with small-molecule drugs, with DNA, with RNA, with ions, with chemical modifications. It is a unified model for **biomolecular complex prediction**. In doing so, it generalized the achievement of AlphaFold 2 from "one protein" to "the cellular machinery itself," covering most molecular interactions relevant to biology and pharmacology.

This matters because biology is fundamentally about interactions: drugs work by binding proteins, transcription factors work by binding DNA, ribosomes work by coordinating proteins and RNA, signaling pathways work by stitching together many complexes. AlphaFold 3 lets researchers ask, for the first time at planetary scale, "what does the protein-ligand-DNA complex look like?" with reasonable accuracy and near-zero cost. For drug discovery in particular, the ability to predict how a candidate small molecule will sit in a protein binding pocket — without having to crystallize the complex — is a major step toward computational medicine. The same year AlphaFold 3 was published, AlphaFold 2's authors received the Nobel Prize, and Isomorphic Labs (Google DeepMind's drug discovery spinout) began partnering with major pharmaceutical companies to put AlphaFold 3 into pipelines.

---

## The Problem

AlphaFold 2 had two important limitations:

1. **Mostly proteins.** AlphaFold 2 was designed for protein chains. Extensions (AlphaFold-Multimer) handled protein-protein complexes, but DNA, RNA, ligands, ions, and post-translational modifications required separate, less accurate tools.

2. **Coordinate regression.** The Structure Module deterministically refined an explicit coordinate frame. This worked beautifully for single proteins but struggled with the inherent flexibility and multiple valid conformations seen in many biological complexes.

Specialized tools (AutoDock, Glide, RoseTTAFold-NA) existed for each category of complex, each with their own pipelines, performance ceilings, and idiosyncrasies. A practicing scientist studying a drug-protein-DNA system might have to use three different programs and reconcile their outputs.

The question for AlphaFold 3: can a single model handle **all** biomolecular types and their interactions, with comparable or better accuracy than each specialized tool?

---

## The Core Innovation

AlphaFold 3 makes two major architectural changes from AlphaFold 2:

1. **A token-based, all-atom representation.** Where AlphaFold 2 thought in terms of "residues with predefined chemistry," AlphaFold 3 represents every molecular component — amino acids, nucleotides, ligand atoms, ions, modifications — as a uniform "token" with associated atoms. This lets the same architecture process anything.

2. **A diffusion-based structure module.** Instead of regressing coordinates, AlphaFold 3 *samples* them by reversing a diffusion process. Starting from random Gaussian noise on atom positions, the model iteratively denoises toward a plausible structure. This is the same conceptual mechanism that powers image generators like DALL-E 3 and Stable Diffusion, but applied to 3D atomic coordinates.

The diffusion module replaces AlphaFold 2's Structure Module entirely. The Evoformer trunk is also simplified into a "Pairformer" that focuses primarily on the pair representation rather than equally on the MSA. The pipeline becomes:

```
Inputs: protein sequences, RNA/DNA sequences, ligand SMILES,
        ion identities, modifications, optional templates
                       |
                       v
                Tokenization
                       |
                       v
        MSA + template processing
                       |
                       v
                  Pairformer
                       |
                       v
        Diffusion Module (denoising)
                       |
                       v
        All-atom 3D structure + confidence
```

### Why Diffusion?

Diffusion has three properties that matter here:

- **Probabilistic outputs.** Many biomolecular structures have inherent multiplicity (a ligand may bind in two orientations; a flexible loop may sample multiple conformations). A diffusion sampler can generate multiple plausible structures, not just one.
- **Architectural simplicity.** The denoising network does not need to be SE(3)-equivariant by construction; the symmetry is learned from data augmentation. This simplifies the model.
- **Stronger gradients on hard targets.** Diffusion training penalizes the model on a continuum of noise levels, which empirically improves generalization to unfamiliar molecules.

---

## How It Works

### Inputs

AlphaFold 3 accepts a heterogeneous specification of the system to be predicted: any number of protein chains (with optional MSAs and templates), nucleic acid chains, ligands (as SMILES strings), ions, and chemical modifications. Each "entity" is tokenized into the same uniform representation.

### The Pairformer Trunk

The Pairformer is a simplified Evoformer that operates primarily on the pair representation (token x token). The MSA contributes its information through a single, lighter-weight processing stack rather than being a co-equal tensor. This reduces compute and reflects the empirical finding that for AlphaFold 3's broader scope, the pair representation does most of the structural work.

### The Diffusion Module

Given the Pairformer's output, the diffusion module:

1. Initializes random Gaussian noise on all atom positions.
2. Runs ~20-200 denoising steps. Each step is a learned neural network that, conditioned on the Pairformer output, predicts a less-noisy set of coordinates.
3. Outputs an all-atom structure for the complete complex.

To get diverse samples, the procedure can be repeated with different random seeds. The model also outputs a confidence head producing pLDDT-style per-atom confidence and a Predicted Aligned Error (PAE) matrix between tokens.

### Training

AlphaFold 3 was trained on the entire Protein Data Bank, including all multimers, ligands, nucleic acids, and modifications. A novel "cross-distillation" step uses AlphaFold 2 predictions as additional training data for protein-only inputs to maintain protein accuracy while expanding the model's repertoire.

---

## Key Results

The paper reports state-of-the-art performance across nearly every biomolecular interaction category:

- **Protein-ligand interactions.** On the PoseBusters benchmark, AlphaFold 3 achieves a success rate substantially better than classical docking tools (AutoDock Vina, Gold) and competitive with the best deep-learning docking methods, despite not being specialized for docking.
- **Protein-protein complexes.** Outperforms AlphaFold-Multimer 2.3 on a hard, recent PDB-derived benchmark.
- **Protein-nucleic acid complexes.** Outperforms RoseTTAFold-NA and other specialized tools.
- **Antibody-antigen interactions.** Improved success rate, though this category remains hard because antibody-antigen interfaces are evolutionarily constrained in unusual ways.
- **Covalent modifications and exotic chemistry.** First general-purpose model to handle these uniformly.

The model also produces realistic confidence calibration: high pLDDT regions are nearly always correct; low pLDDT regions accurately flag uncertain predictions. This is critical for the model to be useful in drug discovery, where a confident wrong prediction is more dangerous than an explicitly uncertain one.

A web server (AlphaFold Server) and downloadable weights (under a non-commercial license at release) made the system broadly accessible to the research community.

---

## Impact and Legacy

AlphaFold 3's release crystallized a few important shifts in computational biology:

- **Generalist > specialist.** A single deep learning model now beats most specialized pipelines for protein-ligand docking, RNA structure prediction, and complex assembly. This is reminiscent of the LLM story: one model trained on many tasks outperforms many models trained on one task.
- **Diffusion for structure.** The success of diffusion in AlphaFold 3 has accelerated work on generative protein design (RFdiffusion, Chroma) and on diffusion models for chemistry more broadly.
- **Drug discovery acceleration.** Isomorphic Labs (DeepMind's drug discovery spinout) partnered with Eli Lilly and Novartis, putting AlphaFold 3 into industrial drug pipelines.
- **The "computational biology stack" gets simpler.** A working scientist can now answer many structural questions with one tool and one API call.
- **Open questions remain.** AlphaFold 3 still struggles with very flexible molecules, highly novel chemotypes, and induced-fit binding modes. It also does not yet model molecular dynamics — only static endpoints — leaving room for future generations.

The paper also intensified a debate about reproducibility and openness in AI for science. Unlike AlphaFold 2, AlphaFold 3 was initially released only via a web server with usage limits and without code. Code and weights were later released under a more permissive arrangement after community pressure, but the episode foreshadowed ongoing tension between commercial AI development and open scientific practice.

---

## Connections to Other Papers

- **AlphaFold 2 (#87)** — direct predecessor. AlphaFold 3 generalizes its scope and swaps the Structure Module for a diffusion sampler.
- **AlphaZero (#89)** — sibling DeepMind program. Both demonstrate the pattern of "domain-appropriate deep learning + strong inductive biases" applied to scientific or game problems.
- **AlphaGeometry (#61) and AlphaEvolve (#62)** — co-members of DeepMind's "AI for science" portfolio, all building on the lesson that ML can produce genuinely new scientific contributions.
- **Diffusion models (DDPM and Score-based generative models)** — AlphaFold 3 imports the denoising-diffusion machinery from generative modeling of images and applies it to 3D coordinates.
- **Transformer (foundational)** — Pairformer is descended from attention-based architectures, with biology-specific operations.
- **RAG (#13)** — MSA and template search remain a retrieval-augmented component of the pipeline: pull related sequences and structures, then condition the network on them.
- **Constitutional AI (#14) / RLHF (#5)** — AlphaFold 3's release sparked debates about openness in commercial AI labs that parallel the alignment-and-deployment tensions seen in language models.
- **Self-Consistency (#85) and Self-Refine (#84)** — the diffusion sampler's ability to draw multiple samples and rank them by confidence echoes the "sample many, pick the best" pattern that has become standard in modern AI.

---

## Key Takeaways

1. **One model for nearly all of biology.** AlphaFold 3 predicts complexes of proteins with proteins, ligands, DNA, RNA, ions, and modifications using a single architecture. The "specialist tool per molecule type" era is ending.
2. **Diffusion replaces deterministic regression.** Instead of refining coordinates by attention iteration, AlphaFold 3 denoises Gaussian noise into atomic positions. This handles multimodality and improves generalization.
3. **Drug discovery has a new default tool.** Predicting how a candidate drug binds its target was previously slow and expensive. AlphaFold 3 makes it fast, free, and scalable, transforming hit-to-lead pipelines.
4. **Confidence calibration is a feature.** AlphaFold 3 tells you which parts of its predictions to trust, which is essential for downstream scientific use.
5. **A continued template for AI in science.** AlphaFold 3 reinforces the recipe of "task-appropriate generative architecture + comprehensive curated data + domain-aware inductive biases," now applied across an even broader scientific frontier.
