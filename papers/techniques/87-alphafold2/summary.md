# Highly Accurate Protein Structure Prediction with AlphaFold

**Authors:** John Jumper, Richard Evans, Alexander Pritzel, Tim Green, Michael Figurnov, Olaf Ronneberger, Kathryn Tunyasuvunakool, Russ Bates, Augustin Zidek, Anna Potapenko, Alex Bridgland, Clemens Meyer, Simon A. A. Kohl, Andrew J. Ballard, Andrew Cowie, Bernardino Romera-Paredes, Stanislav Nikolov, Rishub Jain, Jonas Adler, Trevor Back, Stig Petersen, David Reiman, Ellen Clancy, Michal Zielinski, Martin Steinegger, Michalina Pacholska, Tamas Berghammer, Sebastian Bodenstein, David Silver, Oriol Vinyals, Andrew W. Senior, Koray Kavukcuoglu, Pushmeet Kohli, Demis Hassabis (DeepMind)
**Published:** July 2021 (Nature, vol. 596, pp. 583-589)
**Paper:** [nature.com/articles/s41586-021-03819-2](https://www.nature.com/articles/s41586-021-03819-2) — DOI: 10.1038/s41586-021-03819-2

---

## Why This Paper Matters

AlphaFold 2 solved a 50-year-old grand challenge of biology: predicting the three-dimensional structure of a protein from its amino acid sequence with near-experimental accuracy. At the CASP14 blind structure prediction competition in late 2020, AlphaFold 2's median backbone accuracy was ~0.96 Å on competition targets — roughly the width of a single atom, and comparable to the noise between two different experimental measurements of the same protein. Competing methods sat above 2.8 Å. The biology community immediately described the result as a "solution" to the protein folding problem.

The impact has been seismic. DeepMind and EMBL-EBI released predicted structures for over 200 million proteins — essentially every catalogued protein on Earth — through the AlphaFold Protein Structure Database, free and open. Drug discovery, enzyme design, basic biological research, and structural biology pedagogy were all reshaped within months. In 2024, Demis Hassabis and John Jumper were awarded the Nobel Prize in Chemistry for this work (shared with David Baker for related computational design contributions), making AlphaFold one of the most consequential pieces of AI research in any scientific domain.

---

## The Problem

A protein is a chain of amino acids that folds into a specific three-dimensional shape, and that shape determines what the protein does — whether it catalyzes a reaction, binds a virus, transports oxygen, or signals between cells. For decades, biologists could determine these shapes experimentally using X-ray crystallography, NMR, or cryo-EM, but each structure could take months or years and cost hundreds of thousands of dollars. Of the hundreds of millions of known protein sequences, only ~170,000 had experimentally-determined structures by 2020.

The protein folding problem asked: given just the sequence (a string of letters from a 20-letter alphabet), can you computationally predict the 3D structure? It is hard because:

1. The number of physically possible conformations is astronomical (Levinthal's paradox).
2. Folding depends on subtle long-range interactions between residues that are sequence-distant but spatially close.
3. There is no obvious closed-form mapping from sequence to structure.

The biennial CASP (Critical Assessment of Structure Prediction) competition, running since 1994, had measured slow but steady progress for 25 years. The original AlphaFold (2018) was the best in CASP13 but still produced structures too inaccurate for most practical uses. CASP14 in 2020 was the breakthrough.

---

## The Core Innovation

AlphaFold 2 is a deep learning system with two stages: an **Evoformer** that builds rich representations of the sequence and its evolutionary context, and a **Structure Module** that converts those representations into explicit 3D coordinates. The architecture has several specific design choices that depart from earlier work:

1. **Multiple Sequence Alignment (MSA) as input.** Evolution is the original protein structure predictor. Residues that are spatially close tend to co-evolve: if one mutates, its partner often mutates too. AlphaFold 2 starts from an MSA — hundreds of related sequences from other organisms — to extract these co-evolutionary signals.

2. **Evoformer trunk with dual representations.** The Evoformer maintains *two* tensors simultaneously: an MSA representation (sequences x residues) and a pair representation (residue x residue). Each block updates each tensor, with information flowing between them via triangle attention and outer-product updates. This is the key architectural innovation: structure is fundamentally a pairwise relationship between residues, and the pair representation makes it first-class in the network.

3. **Equivariant structure module.** The Structure Module takes the Evoformer's outputs and predicts 3D coordinates directly, using an SE(3)-equivariant attention mechanism. It places each residue's local frame (the backbone N, Cα, C atoms) in 3D space, then iteratively refines positions in a way that respects rotation and translation symmetries.

4. **Recycling.** The output of the Structure Module is fed back as input to the Evoformer, several times. This lets the network revise its understanding of the structure based on its own provisional predictions — a kind of internal self-refinement loop.

5. **End-to-end training with auxiliary losses.** The whole system is trained jointly with losses on the final structure, the predicted pairwise distances, the MSA reconstruction, and per-residue confidence (pLDDT). The model also predicts its own confidence, which turns out to be remarkably well-calibrated.

```
Sequence
   |
   v
MSA search (UniRef, BFD)  +  template search (PDB)
   |                          |
   v                          v
   +---- Evoformer (48 blocks) ----+
                 |
                 v
        Structure Module (8 iterations)
                 |
                 v
        3D coordinates + per-residue confidence
                 |
                 +---- (recycle) -> back to Evoformer
```

---

## How It Works

### Inputs

For a target sequence, AlphaFold 2 first runs sequence search tools (Jackhmmer, HHblits) against large protein databases (UniRef90, BFD, MGnify) to build an MSA. It also searches the PDB for structural templates of related proteins, although templates are not required.

### The Evoformer

Each Evoformer block has the following sub-modules:

- **Row-wise attention over the MSA** — each sequence's residues attend to one another.
- **Column-wise attention** — each residue position attends across sequences (capturing co-evolution).
- **Outer product mean** — produces an update to the pair representation from the MSA.
- **Triangle multiplicative updates and triangle attention** — operate on the pair representation, enforcing the geometric constraint that pairwise distances must form valid triangles. This is the conceptual breakthrough: structure must satisfy triangle inequalities, and the network bakes that in.

After 48 blocks, the MSA representation summarizes evolutionary context, and the pair representation encodes a rich prediction of which residues are close to which.

### The Structure Module

The Structure Module starts with all residues in a "black hole" configuration at the origin and iteratively places them in 3D space. Each iteration:

1. Computes an SE(3)-equivariant attention over residues, using the pair representation as a bias.
2. Updates each residue's local frame (rotation and translation).
3. Predicts side-chain torsion angles.

After 8 iterations, the output is a complete all-atom structure. Crucially, the network also outputs a per-residue **pLDDT** (predicted local distance difference test) score from 0 to 100, which experimentally correlates very well with actual accuracy.

### Recycling

The whole forward pass is repeated up to 3 times, with the previous prediction fed back as input. Each cycle lets the model reconsider its predictions in light of its own provisional structure. Recycling improves accuracy on hard targets substantially.

---

## Key Results

At CASP14:

- **Median GDT-TS across all targets: 92.4** (out of 100). A GDT-TS of 90 has historically been considered roughly experimental quality.
- **Median backbone RMSD: 0.96 Å** on the easiest category and ~1.5 Å overall.
- AlphaFold 2 was the only group that consistently produced experiment-comparable models across nearly all target categories, including the hard "free modeling" targets where no homologous structures exist.

Beyond CASP, the AlphaFold Protein Structure Database (with EMBL-EBI) has released:

- ~200 million predicted structures as of 2024 (essentially every catalogued protein).
- Per-residue confidence scores allowing users to know which parts of a structure to trust.
- Free, open access — used by hundreds of thousands of researchers.

Downstream impact includes:

- Antibiotic discovery (Halicin and similar leads accelerated via predicted structures).
- Neglected tropical disease research where target structures were previously unavailable.
- Cryo-EM structure determination, where AlphaFold models serve as starting templates.
- Functional annotation of "dark proteome" regions previously uncharacterized.

---

## Impact and Legacy

AlphaFold 2's impact extends well beyond structural biology:

- **Open data release.** DeepMind chose to release the AlphaFold Protein Structure Database freely, an unusual move for a commercial AI lab and an enormous gift to global science.
- **Nobel Prize in Chemistry 2024.** Hassabis and Jumper shared the Nobel with David Baker for computational protein design. AlphaFold is the first AI system to underlie a Nobel Prize.
- **A template for AI-for-science.** AlphaFold demonstrated that domain-specialized deep learning can solve grand-challenge scientific problems, inspiring efforts in materials, weather (GraphCast), fusion plasma control, mathematics (AlphaGeometry), and algorithm discovery (AlphaEvolve, FunSearch).
- **Methodological influence.** The Evoformer's joint MSA+pair representation, triangle-attention operations, and SE(3)-equivariant structure modules have all been adapted into many subsequent biological models (OmegaFold, ESMFold, RoseTTAFold).
- **Industry transformation.** Drug discovery pipelines now routinely include AlphaFold predictions in their first steps. Whole companies (Isomorphic Labs, Generate Biomedicines) have built strategies around AlphaFold-class models.

AlphaFold also clarified an important methodological lesson: the most impactful AI systems often combine large-scale learning with strong domain-appropriate inductive biases (here, triangle inequalities, SE(3) equivariance, evolutionary co-variation). It is not "pure scale" but "scale + the right structural prior."

---

## Connections to Other Papers

- **Transformer (#1, foundational)** — Evoformer is fundamentally an attention-based architecture, descended from the Transformer family but with biology-specific operations (triangle attention, MSA attention).
- **AlphaFold 3 (#88)** — direct successor. Generalizes the structure module to a diffusion-based prediction over arbitrary biomolecular complexes.
- **AlphaZero (#89)** — sibling DeepMind program. Both demonstrate that domain-appropriate deep learning + strong inductive biases can solve apparently intractable problems, AlphaZero via self-play search and AlphaFold via evolutionary signals.
- **AlphaGeometry (#61) and AlphaEvolve (#62)** — sequel "AI for science" projects from DeepMind, taking AlphaFold's blueprint (deep learning + domain knowledge) into mathematics and algorithm design.
- **Scaling Laws (#12) and Chinchilla (#18)** — AlphaFold is comparatively small (~93M parameters) by LLM standards. Its success reminds the field that bigger is not always better; the right architecture and data matter enormously.
- **RAG (#13)** — AlphaFold's MSA search is conceptually a retrieval step: pull relevant evolutionary "documents" (homologous sequences) before reasoning about structure.
- **Flash Attention (#16)** — modern reproductions and extensions of AlphaFold use efficient attention kernels, including Flash-style implementations, to handle the very long pair-representation tensors.

---

## Key Takeaways

1. **A 50-year grand-challenge problem was largely solved.** AlphaFold 2 brought single-chain protein structure prediction to near-experimental accuracy, with median backbone error of about one atom.
2. **The architecture is its own contribution.** The Evoformer (with its dual MSA + pair representations and triangle attention) and the SE(3)-equivariant structure module encode biology-specific inductive biases that "just" scaling generic models could not have produced.
3. **Evolution is the original structure predictor.** AlphaFold's central data source is the MSA of evolutionarily related sequences. Co-evolution tells you which residues sit close in space.
4. **Open release amplified scientific impact.** The AlphaFold Protein Structure Database (200M+ structures) is freely available, and its existence has changed daily practice across biology.
5. **A blueprint for AI in science.** AlphaFold motivated a generation of "AI for science" efforts (AlphaFold 3, AlphaGeometry, AlphaEvolve, GraphCast) and demonstrated that AI can produce Nobel-worthy contributions to the natural sciences.
