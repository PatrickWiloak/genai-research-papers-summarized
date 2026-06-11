---
title: "Highly Accurate Protein Structure Prediction with AlphaFold (AlphaFold 2)"
slug: "68-alphafold"
number: 68
category: "techniques"
authors: "John Jumper, Richard Evans, Alexander Pritzel, Tim Green, Michael Figurnov, Olaf Ronneberger, Kathryn Tunyasuvunakool, Russ Bates, Augustin Zidek, Anna Potapenko, Alex Bridgland, Clemens Meyer, Simon A. A. Kohl, Andrew J. Ballard, Andrew Cowie, Bernardino Romera-Paredes, Stanislav Nikolov, Rishub Jain, Jonas Adler, Trevor Back, Stig Petersen, David Reiman, Ellen Clancy, Michal Zielinski, Martin Steinegger, Michalina Pacholska, Tamas Berghammer, Sebastian Bodenstein, David Silver, Oriol Vinyals, Andrew W. Senior, Koray Kavukcuoglu, Pushmeet Kohli, Demis Hassabis (DeepMind)"
published: "2021 (Nature, volume 596)"
year: 2021
url: "https://www.nature.com/articles/s41586-021-03819-2"
tags: ["science", "attention"]
---

# Highly Accurate Protein Structure Prediction with AlphaFold (AlphaFold 2)

**Authors:** John Jumper, Richard Evans, Alexander Pritzel, Tim Green, Michael Figurnov, Olaf Ronneberger, Kathryn Tunyasuvunakool, Russ Bates, Augustin Zidek, Anna Potapenko, Alex Bridgland, Clemens Meyer, Simon A. A. Kohl, Andrew J. Ballard, Andrew Cowie, Bernardino Romera-Paredes, Stanislav Nikolov, Rishub Jain, Jonas Adler, Trevor Back, Stig Petersen, David Reiman, Ellen Clancy, Michal Zielinski, Martin Steinegger, Michalina Pacholska, Tamas Berghammer, Sebastian Bodenstein, David Silver, Oriol Vinyals, Andrew W. Senior, Koray Kavukcuoglu, Pushmeet Kohli, Demis Hassabis (DeepMind)

**Published:** 2021 (Nature, volume 596)

**Paper Link:** https://www.nature.com/articles/s41586-021-03819-2

---

## Why This Paper Matters

AlphaFold 2 solved one of the most significant open problems in biology: predicting a protein's 3D structure from its amino-acid sequence alone. The protein-folding problem had been open for over 50 years. At CASP14 (2020), AlphaFold achieved near-experimental accuracy, with a median GDT score above 92 out of 100, roughly doubling the accuracy of the previous best system.

For this collection, AlphaFold is a landmark proof that the attention-based Transformer toolkit - the same machinery behind GPT and BERT - generalizes far beyond language into fundamental natural science. It is arguably the most consequential single application of deep learning to date.

---

## The Problem: Protein Folding in Plain English

A protein is a chain of amino acids, like a very long necklace with 20 possible bead types. The sequence of beads is encoded in DNA and is relatively easy to read. What is not straightforward is the *shape* the chain folds into once it is assembled.

Shape determines function. A misfolded protein can cause Alzheimer's disease, cancer, or cystic fibrosis. Knowing the correct 3D shape unlocks drug design, enzyme engineering, and fundamental biology.

Determining shape experimentally (X-ray crystallography, cryo-EM) is slow and expensive - each protein can take months and cost hundreds of thousands of dollars. There are roughly 200 million known protein sequences but, before AlphaFold, only about 170,000 experimentally determined structures.

The computational challenge: the number of possible 3D conformations for even a small protein is astronomically large (Levinthal's paradox - random search is hopeless). Researchers had worked on this since the 1970s.

---

## Core Innovation

AlphaFold 2 replaces the fragmented, multi-stage pipelines of earlier approaches with a single end-to-end differentiable neural network built around two key inventions:

1. **Evoformer** - an attention-based module that reasons jointly over evolutionary sequence data and residue-residue spatial relationships.
2. **Structure Module** - a module that translates those abstract representations into 3D atomic coordinates, respecting the rigid geometry of protein backbones.

---

## Key Components Explained

### 1. Multiple Sequence Alignments (MSA) - The Evolutionary Lens

Before AlphaFold runs its neural network, it searches genetic databases for *homologous* proteins - related proteins found in other organisms that evolved from the same ancestor.

Think of it like studying an English word by looking at its cognates in French, Spanish, and Italian. The family resemblance reveals which parts of the word are structurally essential.

Residues (amino-acid positions) that tend to *co-evolve* across thousands of species are almost certainly in physical contact in the 3D structure - a mutation at one position is often compensated by a mutation at a paired position to preserve function. This co-evolutionary signal is the key prior that AlphaFold exploits.

The MSA is represented as a 2D grid: rows are homologous sequences, columns are residue positions.

### 2. The Evoformer - Attention Over Two Representations

The Evoformer is a stack of attention-based blocks (see [Attention Is All You Need](../../architectures/01-attention-is-all-you-need/summary.md)) that maintains and refines *two* coupled representations simultaneously:

- **MSA representation** - a matrix capturing which residues co-vary across homologous sequences (rows = sequences, columns = residue positions)
- **Pair representation** - a matrix capturing relationships between every pair of residues in the target sequence (a residue-by-residue grid encoding distances, orientations, and contacts)

The crucial insight: these two representations are *not* processed independently. Evoformer blocks pass information bidirectionally between them at every layer.

```
MSA Representation (S sequences x L residues)
          |
          v  axial attention + outer product mean update
          |
Pair Representation (L residues x L residues)
          |
          v  triangle attention + triangle multiplicative update
          |
(back to MSA representation for next block)
```

**Axial attention** keeps computation tractable: instead of full attention over the entire MSA grid, it applies attention row-wise (across residue positions for one sequence) and column-wise (across sequences for one residue position) in alternation.

**Outer product mean:** information flows from MSA to pair by computing outer products of MSA row vectors and averaging them, effectively asking "for every pair of positions (i, j), how do the sequence-level features at i and j co-vary?"

**Triangle attention and multiplicative updates:** information flows within the pair representation using triangle-aware operations. If residues A and B are close, and B and C are close, the model should infer something about A-C proximity. Triangle operations enforce this geometric consistency by attending over the third vertex of each residue triplet.

After 48 Evoformer blocks, the pair representation encodes a rich, geometrically consistent picture of the structure - before any 3D coordinates have been produced.

### 3. Structure Module - From Representation to 3D Coordinates

The structure module takes the per-residue embeddings from the final Evoformer layer and outputs explicit 3D positions for every backbone atom (and ultimately all side-chain atoms).

Key ideas:

- **Frames (rigid bodies):** Each residue is represented as a local coordinate frame - a rotation and a translation in 3D space - rather than raw x, y, z coordinates. This encodes the rigid geometry of peptide bonds naturally.
- **Invariant Point Attention (IPA):** A specialized attention mechanism that operates on 3D frames. Attention weights are computed using both the abstract pair representation *and* actual 3D distances between frames, making the whole computation equivariant to global rotation and translation.
- **Iterative refinement:** The structure module is applied across multiple recycling iterations. The predicted 3D structure is fed back into the Evoformer as additional input, allowing the model to progressively refine its predictions.

```
Evoformer output (per-residue embeddings + pair representation)
    |
    v
Structure Module
    |- Frames (rotation + translation per residue)
    |- Invariant Point Attention (IPA)
    |- Side-chain torsion angle prediction
    |
    v
Full 3D atomic coordinates
```

A **per-residue confidence score (pLDDT, 0-100)** is predicted alongside the structure. Regions with pLDDT below 50 are intrinsically disordered and should not be interpreted as having a fixed shape.

### 4. Training Signal

AlphaFold is trained end-to-end on the Protein Data Bank (PDB) - experimentally determined structures. The main loss is **Frame-Aligned Point Error (FAPE):** the average distance between predicted and true atom positions, computed in local reference frames to ensure rotation/translation invariance.

A **distogram loss** on the predicted pair representation (predicting residue-residue distance distributions) provides an auxiliary signal that helps the Evoformer learn good pair representations even early in training.

---

## Key Results

### CASP14 (2020 Critical Assessment of Protein Structure Prediction)

CASP is a biennial blind prediction competition - structures are solved experimentally and competitors predict them without knowing the answer.

| System | Median GDT Score |
|--------|-----------------|
| Best prior system (CASP13 winner) | ~43 |
| AlphaFold 2 (CASP14) | **92.4** |
| Experimental accuracy threshold | ~90+ |

**GDT (Global Distance Test):** percentage of residues whose predicted position falls within a few angstroms of the true position, averaged across distance thresholds. A score above 90 is considered experimental quality.

AlphaFold achieved GDT scores above 90 on two-thirds of targets, including many proteins with no close homologs. The CASP14 assessors described it as effectively solving the protein-folding problem for single-chain proteins.

### AlphaFold Protein Structure Database

In 2021, DeepMind and EMBL-EBI released the **AlphaFold Protein Structure Database**, providing predicted structures for:
- All ~20,000 human proteins
- The proteomes of 47 other organisms
- Ultimately expanded to over 200 million protein structures, covering nearly the entire known protein universe

All structures are freely available at https://alphafold.ebi.ac.uk.

---

## Why This Was Revolutionary

### 1. Speed - from months to seconds
Experimental structure determination: weeks to years per protein. AlphaFold 2: seconds to minutes per protein on a standard GPU.

### 2. Coverage
Structural coverage of the known proteome jumped from roughly 17% to over 35% almost immediately after the database launch, with coverage of well-studied organisms approaching 100%.

### 3. Generalization across all of life
AlphaFold works across the tree of life - bacteria, archaea, fungi, plants, animals. It was not tuned for any specific organism.

### 4. Proof of concept for deep learning in science
AlphaFold demonstrated that the architecture family behind language models - attention, learned representations, end-to-end training - could solve century-scale scientific problems, not just NLP benchmarks. It validated the research direction for applying ML to physics, chemistry, and biology.

---

## Real-World Impact

- **Drug discovery:** Structural biology is the foundation of rational drug design. AlphaFold structures are embedded in early-stage discovery pipelines at virtually every major pharmaceutical company and research institution.
- **Neglected tropical diseases:** Open access to structures of parasitic proteins has accelerated research on diseases like malaria, schistosomiasis, and Chagas disease where commercial incentives are low.
- **Enzyme engineering:** Designing enzymes for industrial applications (biofuels, plastics degradation) relies on structural knowledge. AlphaFold is now standard in enzyme design pipelines.
- **Antibody and vaccine design:** Predicting antigen structures accelerates rational vaccine development.
- **Nobel Prize:** Demis Hassabis and John Jumper received the 2024 Nobel Prize in Chemistry for AlphaFold, alongside David Baker for his work on computational protein design.

---

## Key Takeaways

1. **Attention is not just for language.** The Transformer's ability to model pairwise relationships over arbitrary sequences transfers directly to biological sequences.
2. **Co-evolutionary signal is powerful.** Billions of years of evolution have run billions of natural experiments; mining those experiments statistically gives rich geometric information for free.
3. **Joint reasoning over multiple representations pays off.** The Evoformer's bidirectional coupling of MSA and pair representations is a key architectural innovation - neither alone is sufficient.
4. **Geometric inductive biases matter.** Representing residues as rigid frames and using invariant point attention significantly outperforms naive coordinate regression.
5. **Confidence estimation is essential for science.** The pLDDT score lets users know *which parts* of a prediction to trust, making the tool practically useful rather than just accurate on average.
6. **Open release multiplies impact.** Making 200 million structures freely available turned a research result into infrastructure for all of biology.

---

## Limitations and Future Directions

### Limitations of AlphaFold 2

- **Single-chain focus:** The original system was designed for individual protein chains. Predicting multi-protein complexes required significant additional work (addressed partly by AlphaFold-Multimer, 2021).
- **Static structures:** AlphaFold predicts a single lowest-energy conformation. Many proteins are dynamic, switching between multiple conformations - capturing conformational ensembles requires separate approaches.
- **No small molecules or nucleic acids:** The original system cannot model protein-DNA, protein-RNA, or protein-ligand interactions natively.
- **Intrinsically disordered regions:** Proteins with no stable fold (flagged by low pLDDT) are common and biologically important, but AlphaFold provides limited insight into them.
- **Overconfident predictions on orphan sequences:** For sequences with no close homologs, the model can produce confident-looking predictions that are incorrect. pLDDT is a useful but imperfect warning signal.

### AlphaFold 3 (2024)

DeepMind released AlphaFold 3 in 2024, extending the approach to:
- **Molecular complexes** involving proteins, DNA, RNA, and small-molecule ligands together
- A **diffusion-based structure module** replacing the frame-based approach of AlphaFold 2
- Broader coverage of the full molecular biology landscape

The architecture evolution from AlphaFold 2 to 3 mirrors broader ML trends: from autoregressive/frame-based decoding toward diffusion-based generation.

---

## Further Reading

- **AlphaFold 2 paper:** https://www.nature.com/articles/s41586-021-03819-2
- **AlphaFold Protein Structure Database:** https://alphafold.ebi.ac.uk
- **AlphaFold-Multimer (protein complexes):** https://www.biorxiv.org/content/10.1101/2021.10.04.463034v2
- **AlphaFold 3 paper:** https://www.nature.com/articles/s41586-024-07487-w
- **Yannic Kilcher video walkthrough:** https://www.youtube.com/watch?v=uQ1uVbrIv-Q
- **Attention Is All You Need (foundation architecture):** ../../architectures/01-attention-is-all-you-need/summary.md

---

## Citation

```bibtex
@article{jumper2021highly,
  title={Highly accurate protein structure prediction with {AlphaFold}},
  author={Jumper, John and Evans, Richard and Pritzel, Alexander and Green, Tim and Figurnov, Michael and Ronneberger, Olaf and Tunyasuvunakool, Kathryn and Bates, Russ and Zidek, Augustin and Potapenko, Anna and others},
  journal={Nature},
  volume={596},
  number={7873},
  pages={583--596},
  year={2021},
  publisher={Nature Publishing Group}
}
```

<!-- related:start -->

---

## Related in This Collection

- [Attention Is All You Need](../../architectures/01-attention-is-all-you-need/summary.md)
- [BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](../../language-models/03-bert/summary.md)

<!-- related:end -->
