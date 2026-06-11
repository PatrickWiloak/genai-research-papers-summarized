---
title: "Sequence to Sequence Learning with Neural Networks (Seq2Seq)"
slug: "55-seq2seq"
number: 55
category: "architectures"
authors: "Ilya Sutskever, Oriol Vinyals, Quoc V. Le (Google)"
published: "September 2014 (NeurIPS 2014)"
year: 2014
url: "https://arxiv.org/abs/1409.3215"
tags: [architectures]
---

# Sequence to Sequence Learning with Neural Networks (Seq2Seq)

**Authors:** Ilya Sutskever, Oriol Vinyals, Quoc V. Le (Google)

**Published:** September 2014 (NeurIPS 2014)

**Paper Link:** https://arxiv.org/abs/1409.3215

---

## Why This Paper Matters

Before Seq2Seq, neural networks could only map a fixed-size input to a fixed-size output. That ruled out machine translation, summarization, and question answering - tasks where input and output lengths vary independently. This paper introduced the **encoder-decoder architecture**: compress a variable-length input into a single vector, then expand that vector into a variable-length output. The design became the template for all sequence-to-sequence learning and directly motivated the attention mechanism and the Transformer that followed.

---

## The Core Innovation: Encoder-Decoder

### The Problem
- A sentence in English might be 7 words; its French translation might be 9.
- A standard neural network needs fixed-size inputs and outputs - you cannot bolt a "translate this" layer onto the end.

### The Solution
Split the problem into two networks connected by a bottleneck vector:

1. **Encoder** - reads the input sequence word by word and produces a single fixed-length vector (the "context vector" or "thought vector") that summarizes the entire input.
2. **Decoder** - reads that vector and generates the output sequence one word at a time, feeding each generated word back in as input for the next step.

**Analogy:** The encoder is a person who reads a document and writes a dense one-paragraph summary. The decoder is a different person who reads only that summary and writes a translation. All the meaning must pass through the summary - nothing else is shared.

---

## Key Components Explained

### 1. Multilayer LSTMs

Both encoder and decoder are **Long Short-Term Memory (LSTM)** networks stacked 4 layers deep. LSTMs solve the vanishing-gradient problem that made plain RNNs forget early words by the time they reached the end of a long sentence.

```
Input word -> [LSTM layer 1] -> [LSTM layer 2] -> [LSTM layer 3] -> [LSTM layer 4]
                                                                           |
                                                                   hidden state h_t
```

Each layer passes its hidden state up to the next layer and forward to the next time step. After processing the last input token, the top-layer hidden state becomes the context vector.

### 2. The Context Vector

The entire input sequence is compressed into a single vector of 1000 numbers (the hidden state of the final encoder step). The decoder is initialized with this vector and then autoregressively generates output tokens:

```
Input:  "The cat sat"
         |   |   |
       [ENC][ENC][ENC]  ->  [ context vector ]
                                    |
                           [DEC] -> "Le"
                           [DEC] -> "chat"
                           [DEC] -> "s'assit"
                           [DEC] -> <END>
```

### 3. Reversing the Source Sentence

One of the paper's key tricks: **reverse the word order of the input before encoding it**.

Why it helps: reversing moves the first source word close in time to the first target word. In a normal forward pass, the encoder finishes reading the full input before the decoder starts - putting the first source word far from the start of decoding. Reversal shortens the "communication path" for the most important early words, giving the decoder a stronger gradient signal at the start of generation.

```
Normal:   "The cat sat" -> encode -> decode -> "Le chat s'assit"
                                               (encoder saw "The" 3 steps ago)

Reversed: "sat cat The" -> encode -> decode -> "Le chat s'assit"
                                               (encoder just saw "The" - 1 step ago)
```

This one trick improved BLEU scores substantially without any architectural changes.

### 4. Architecture Overview

```
SOURCE (reversed):  "." "sat" "cat" "The"
                      |    |    |    |
                  +------------------------------+
                  |  ENCODER (4-layer LSTM)      |
                  |  Reads input left to right   |
                  |  Final hidden state =        |
                  |  context vector  c           |
                  +-------------+----------------+
                                |  c (1000-dim vector)
                                |
                  +-------------+----------------+
                  |  DECODER (4-layer LSTM)      |
                  |  Initialized with c          |
                  |  Generates one token/step    |
                  |  Feeds output back as input  |
                  +------------------------------+
                      |    |      |       |
TARGET:              "Le" "chat" "s'assit" "."
```

---

## Key Results

**WMT'14 English-to-French translation:**
- **34.81 BLEU** - first purely neural system to outperform a phrase-based SMT baseline on a large-scale task
- Added to an ensemble, reached **36.5 BLEU** - surpassing the best phrase-based system at the time

**Long sentences:** Earlier neural MT systems degraded sharply as sentence length grew. Seq2Seq (with reversal) held performance much more consistently on longer inputs.

**Sentence clustering:** When the authors visualized context vectors, sentences with similar meaning clustered together regardless of surface form - evidence the encoder was learning semantic structure, not just memorizing word patterns.

---

## Why This Was Revolutionary

### 1. Variable-length inputs and outputs
For the first time, a single end-to-end trainable neural network could handle arbitrary-length sequences on both sides. No hand-crafted feature alignment, no phrase tables.

### 2. A reusable template
Encoder + context vector + decoder is a general pattern. Within months, researchers applied it to summarization, parsing, image captioning (CNN encoder, LSTM decoder), and speech recognition.

### 3. End-to-end training
The whole system is trained jointly with backpropagation through time. The encoder learns to produce useful context vectors because the decoder needs them - no separate supervised signal for the intermediate representation.

### 4. Proved LSTMs could scale
4-layer LSTMs on 160,000-word vocabularies trained on 12 million sentence pairs. This showed deep LSTMs were practical at real-world scale.

---

## Real-World Impact

### Direct descendants:
- **Bahdanau Attention** (2014) - replaced the fixed vector with a dynamic weighted sum over all encoder states; see Limitations below
- **Luong Attention** (2015) - simplified and generalized Bahdanau's mechanism
- **Transformer** (Vaswani et al., 2017) - replaced the LSTM entirely with self-attention, but kept the encoder-decoder skeleton that Seq2Seq defined
- **Google Neural Machine Translation** (2016) - production system built directly on this architecture, serving billions of translations
- **Pointer Networks** (Vinyals et al., 2015) - extended the decoder to point to input tokens, enabling copy mechanisms

### Still visible today:
- Encoder-decoder is the backbone of T5, BART, mT5, and most seq2seq fine-tuning setups
- The "context vector" concept became the "hidden state" that attention mechanisms improve upon

---

## Key Takeaways for Practitioners

1. **Separate encoding from decoding** - one network summarizes the input, another generates the output; this division of labor is still the dominant paradigm
2. **Reversal is a cheap win** - a simple data preprocessing step gave a significant BLEU gain; always look for structural tricks before reaching for more parameters
3. **Depth matters** - 4-layer LSTMs outperformed 1-layer LSTMs substantially; stacking recurrent layers was the 2014 equivalent of scaling
4. **The bottleneck is a feature and a bug** - the fixed context vector forces compact representations that generalize well, but it also caps capacity for long sequences
5. **End-to-end training is powerful** - joint optimization of encoder and decoder means the two halves co-adapt; no intermediate supervision needed

---

## Limitations & Future Directions

### The fixed-vector bottleneck

The fundamental weakness of Seq2Seq is that every input sentence - whether 5 words or 50 - must be compressed into the same fixed-size vector. For short sentences this is fine. For long sentences, the encoder must overwrite early information to accommodate later words, and translation quality degrades.

**The fix:** Dzmitry Bahdanau, Kyunghyun Cho, and Yoshua Bengio proposed attention in "Neural Machine Translation by Jointly Learning to Align and Translate" (September 2014, arxiv 1409.0473) - published the same month as Seq2Seq. Instead of a single context vector, the decoder at each step computes a weighted sum over all encoder hidden states. The weights (the "alignment") are learned - the model looks back at the full input and decides which parts to focus on for each output word. This removed the information bottleneck entirely.

**The next step:** The Transformer (Vaswani et al., 2017) generalized attention further, replacing the LSTM recurrence with self-attention throughout. The encoder-decoder structure from Seq2Seq survived unchanged; only the internal mechanism changed. In that sense, every Transformer is a descendant of this paper.

### Sequential processing
LSTMs process tokens one at a time - you cannot parallelize across the sequence length. This made training slow and limited context length. Attention-based models fixed this.

### No explicit alignment
The model learns translation implicitly. Before attention, there was no way to inspect which source words influenced which target words. Attention made the alignment interpretable.

---

## Further Reading

- **Bahdanau Attention (the direct follow-up):** https://arxiv.org/abs/1409.0473
- **Attention Is All You Need (Transformer):** https://arxiv.org/abs/1706.03762
- **The Illustrated Seq2Seq with Attention (Jay Alammar):** https://jalammar.github.io/visualizing-neural-machine-translation-mechanics-of-seq2seq-models-with-attention/
- **Original paper:** https://arxiv.org/abs/1409.3215

---

## Citation

```bibtex
@article{sutskever2014sequence,
  title={Sequence to sequence learning with neural networks},
  author={Sutskever, Ilya and Vinyals, Oriol and Le, Quoc V},
  journal={Advances in neural information processing systems},
  volume={27},
  year={2014}
}
```

<!-- related:start -->

---

## Related in This Collection

- [Attention Is All You Need](../../architectures/01-attention-is-all-you-need/summary.md)
- [Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer (T5)](../../language-models/65-t5/summary.md)
- [Neural Machine Translation by Jointly Learning to Align and Translate (Bahdanau Attention)](../../architectures/66-bahdanau-attention/summary.md)

<!-- related:end -->
