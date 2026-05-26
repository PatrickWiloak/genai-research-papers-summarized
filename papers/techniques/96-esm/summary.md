# Evolutionary-Scale Prediction of Atomic-Level Protein Structure with a Language Model (ESM-2 / ESMFold)

**Authors:** Zeming Lin, Halil Akin, Roshan Rao, Brian Hie, Zhongkai Zhu, Wenting Lu, Nikita Smetanin, Robert Verkuil, Ori Kabeli, Yaniv Shmueli, Allan dos Santos Costa, Maryam Fazel-Zarandi, Tom Sercu, Salvatore Candido, Alexander Rives (Meta AI, Fundamental AI Research)
**Published:** March 2023, **Science** vol. 379, issue 6637 ([DOI: 10.1126/science.ade2574](https://doi.org/10.1126/science.ade2574))
**Paper:** [biorxiv.org/content/10.1101/2022.07.20.500902](https://www.biorxiv.org/content/10.1101/2022.07.20.500902)

---

## Why This Paper Matters

ESM-2 is the moment biology got its GPT. Meta's Fundamental AI Research lab trained a 15-billion-parameter language model **on protein sequences instead of human text**, and the model spontaneously learned the structural and functional grammar of proteins — well enough that a small head on top (ESMFold) could predict 3D atomic structure directly from a single sequence, without the multiple sequence alignments (MSAs) that AlphaFold 2 had relied on. ESMFold was up to **60x faster than AlphaFold 2** at comparable accuracy on well-folded proteins, and the team used it to compute and release the **ESM Metagenomic Atlas** — predicted structures for over **617 million proteins** from environmental sequencing data.

The deeper significance: ESM-2 demonstrated that **the scaling-laws story works for biology**. Larger protein language models, trained on more sequences with more compute, learn richer biophysical structure inside their representations. That single empirical claim turned protein language models into a foundation of computational biology and seeded a wave of "foundation models for science" — for DNA, RNA, cells, small molecules, and beyond.

---

## The Problem

For 50 years, predicting a protein's 3D structure from its amino-acid sequence was the central open problem in biology. AlphaFold 2 (2021) largely cracked it for well-studied proteins, but with a major dependency: it needed an **MSA** — a stack of evolutionarily related sequences aligned to the target. Building MSAs requires:

- A database search (HHblits, MMseqs2) over hundreds of millions of sequences
- Minutes to hours per protein
- A reasonable number of homologs to actually exist

For two huge classes of proteins this is fatal:

**1. Orphan proteins.** Many proteins — especially from rarely sequenced organisms — have few or no detectable homologs. No MSA, no AlphaFold 2.

**2. Metagenomic sequences.** The vast majority of life on Earth is microbial and most of it has never been cultured. Metagenomic sequencing produces hundreds of millions of protein fragments, often with no close relatives in databases.

The community needed a structure predictor that worked **from a single sequence**, and that could scale to hundreds of millions of proteins in reasonable time. ESM was Meta's answer.

---

## The Core Innovation

ESM-2 applies the language-model recipe — transformer + masked-token pretraining + massive scale — directly to the **alphabet of amino acids**.

```
A protein:  M K V L L S T I L G ... (sequence of 20 possible "tokens")
ESM-2:      a transformer that masks tokens and predicts them
            (just like BERT, but on protein sequences)
Training:   ~65 million unique sequences from UniRef
Scale:      8M, 35M, 150M, 650M, 3B, 15B parameters
```

The model is trained only to fill in masked amino acids. No structure labels, no functional labels, no MSAs. **But because evolution conserves residues that are physically or functionally important, the masked-prediction signal contains rich structural information.**

Two empirical claims fall out:

1. As the model gets larger, its internal representations encode **more accurate structural information** — measured by the linear probability that pairs of residues are in contact.
2. With sufficient scale, you can decode a 3D structure from those representations directly — **no MSA required**.

ESMFold is the structure decoder built on top.

---

## How It Works

### ESM-2: the protein language model

Standard transformer encoder, masked language modeling, attention over the full sequence. The model sees ~15% of tokens masked and must reconstruct them.

```
Input:   M K V L L [MASK] T I L G [MASK] ...
Target:  ...predict S at position 5, A at position 10...
```

Training data is UniRef50/UniRef90 — clustered protein sequences chosen to span the known diversity of life. Six scale variants are released, from 8M to 15B parameters.

### Emergent structure in attention

When the model is large enough, the attention maps between residues begin to align with the **physical contact map** of the folded protein. Residues that touch in 3D get attended to by each other in the transformer. This was first observed in earlier ESM papers and gets sharper with scale.

### ESMFold: a single-sequence structure predictor

ESMFold uses the frozen ESM-2 15B model as the input encoder, then attaches a folding head adapted from AlphaFold 2's structure module:

```
Sequence  --> ESM-2 (15B, frozen)  --> token + pair embeddings
                                                 |
                                  Folding head (AF2-style)
                                                 |
                                    3D atomic coordinates
```

The crucial difference from AlphaFold 2: the input is **just one sequence**. No MSA construction, no template search, no homology database lookup. Everything AF2 was getting from the MSA, ESMFold gets from ESM-2's learned representations.

### The Metagenomic Atlas

With ESMFold's speed (~14 seconds for a 384-residue protein on a single A100), the team ran inference on **617 million metagenomic protein sequences** — most of which had never had a structure predicted before. The resulting database, the ESM Metagenomic Atlas, was released publicly with bulk download. It roughly doubled the total number of predicted protein structures known to science overnight.

---

## Key Results

### Scaling laws for biology

The paper's most influential plot: as ESM-2 grows from 8M to 15B parameters, contact-prediction accuracy improves smoothly and predictably — the same kind of clean power-law curve seen in language models. Bigger model -> better internal structure -> better downstream folding.

| ESM-2 size | Contact prediction (long-range P@L) |
|------------|-------------------------------------|
| 8M | ~ 0.17 |
| 150M | ~ 0.30 |
| 650M | ~ 0.39 |
| 3B | ~ 0.47 |
| 15B | ~ 0.52 |

This was the first crisp scaling law in computational biology.

### ESMFold vs AlphaFold 2

On the CAMEO benchmark of recent structures:

- **Comparable accuracy** to AlphaFold 2 on proteins where AF2 already does well.
- **Up to 60x faster** end-to-end (no MSA construction).
- **Strictly better on orphan proteins** — sequences with few homologs, where AF2's MSAs are sparse and degrade quality.

### The Metagenomic Atlas

617 million predicted structures, including hundreds of millions of proteins from previously uncharted regions of sequence space. Of these, ~225 million were predicted with high confidence (pLDDT > 70). Released publicly via the ESM Atlas web interface and bulk downloads — instantly the largest structural database in existence.

---

## Impact and Legacy

ESM-2 / ESMFold launched the **protein foundation model** as a category, and inspired an explosion of follow-on work:

- **ESM-3** (EvolutionaryScale, 2024): a generative multimodal protein model that jointly reasons over sequence, structure, and function, used to design novel fluorescent proteins.
- **AlphaFold 3** (2024): extended structure prediction to complexes including ligands, nucleic acids, and post-translational modifications.
- **DNA / RNA language models** (Nucleotide Transformer, Evo, RNA-FM): the same scaling recipe applied to other biological sequences.
- **Cell language models** (Geneformer, scGPT): transformers over gene-expression "sentences."
- **Boltz** and other open-source structure predictors: continued the push toward fast, single-sequence prediction.

It also marked the founding of **EvolutionaryScale**, the spinout that the ESM team formed in 2024 to continue this research line outside Meta — one of the first true "biology-foundation-model" companies.

Beyond proteins, ESM-2 was a load-bearing piece of evidence for the general claim that **the foundation-model paradigm is not specific to language**. Combined with results in vision (CLIP, Sora/DiT), audio (Whisper), and world models (Genie, DreamerV3), it helped establish that scale + self-supervised pretraining + transformer is a recipe that works across modalities.

---

## Connections to Other Papers

- **BERT (#3):** ESM-2's masked-language-modeling objective is BERT, applied to amino acids. The architectural recipe is unchanged; only the alphabet differs.
- **Scaling Laws (#12):** ESM-2 demonstrates the Chinchilla-style scaling story in biology — bigger models on more sequences yield predictably better representations.
- **Chinchilla (#18):** Informs the data/parameter scaling choices made for ESM-2's largest models.
- **AlphaFold 2 / 3 (#87 / #88):** ESMFold is the alternative paradigm — single-sequence, foundation-model-driven — to AlphaFold's MSA-centric approach. The two together define modern computational structural biology.
- **AlphaZero (#89), AlphaGeometry (#61), AlphaEvolve (#62):** DeepMind's "AlphaX" line targets specific scientific or game domains; ESM is Meta's equivalent bet that **the foundation-model paradigm itself** is the right tool for science.
- **GPT-3 (#4) and successors:** Same recipe (transformer + scale + self-supervised pretraining), different alphabet. ESM is the most direct biological analog of GPT-style scaling.

---

## Key Takeaways

1. **Foundation models work for biology.** A transformer trained only to fill in masked amino acids learns representations rich enough to decode 3D structure.
2. **Scaling laws apply.** Contact-prediction accuracy improves as a clean function of parameters and compute, exactly as in language models — the first really clean such curve in computational biology.
3. **No MSAs needed.** ESMFold predicts atomic structure from a single sequence, up to 60x faster than AlphaFold 2 and strictly better on orphan and metagenomic proteins.
4. **Scale opened a new database.** The ESM Metagenomic Atlas — 617M predicted structures — was made possible only because ESMFold is fast enough to run on hundreds of millions of sequences.
5. **A template for foundation models in science.** ESM-2 helped establish that the scaling + self-supervised pretraining recipe generalizes far beyond language, seeding the wave of foundation models for DNA, RNA, cells, and chemistry.
