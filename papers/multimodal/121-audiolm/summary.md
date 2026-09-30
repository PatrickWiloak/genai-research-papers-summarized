---
title: "AudioLM: a Language Modeling Approach to Audio Generation (AudioLM)"
slug: "121-audiolm"
number: 121
category: "multimodal"
authors: "Zalán Borsos, Raphaël Marinier, Damien Vincent, Eugene Kharitonov, Olivier Pietquin, Matt Sharifi, Dominik Roblek, Olivier Teboul, David Grangier, Marco Tagliasacchi, Neil Zeghidour (Google Research)"
published: "September 2022 (IEEE/ACM Transactions on Audio, Speech, and Language Processing, 2023)"
year: 2022
url: "https://arxiv.org/abs/2209.03143"
tags: ["audio", "language-model", "discrete-representation"]
---

# AudioLM: a Language Modeling Approach to Audio Generation (AudioLM)

**Authors:** Zalán Borsos, Raphaël Marinier, Damien Vincent, Eugene Kharitonov, Olivier Pietquin, Matt Sharifi, Dominik Roblek, Olivier Teboul, David Grangier, Marco Tagliasacchi, Neil Zeghidour (Google Research)
**Published:** September 2022 (IEEE/ACM Transactions on Audio, Speech, and Language Processing, 2023)
**Paper:** [arxiv.org/abs/2209.03143](https://arxiv.org/abs/2209.03143)

---

## Why This Matters

AudioLM showed that **the GPT recipe works for raw audio**: turn sound into a sequence of discrete tokens, then train a Transformer to predict the next token. Given just 3 seconds of someone speaking, it continues in the same voice, with the same accent and recording conditions, producing speech that is grammatical and mostly sensible, **without ever seeing a transcript**.

- **Near-indistinguishable from real speech.** Human raters trying to tell AudioLM continuations from real recordings were right 51.2 percent of the time, statistically no better than a coin flip.
- **No text, no labels.** Trained on 60,000 hours of unlabelled English audiobook speech.
- **Not just speech.** The same framework, trained on piano recordings, produced coherent piano continuations with no sheet music or MIDI.
- **A template for what followed.** Its split into "semantic" and "acoustic" tokens shaped MusicLM, [VALL-E](../122-vall-e/summary.md), and much later spoken-dialogue model design.

**The insight:** audio carries structure at two very different scales. Long-range structure (words, syntax, melody) and fine acoustic detail (voice timbre, room sound) need different kinds of token. Model the coarse structure first, then fill in the detail.

---

## The Problem: Quality Or Coherence

Before AudioLM, generating audio without strong conditioning had a clear split:

- **Waveform models** such as WaveNet produce very realistic sound locally, but left to themselves they **babble**: the audio sounds like speech but says nothing coherent over more than a moment.
- **Textless spoken language models** such as GSLM ran a language model over tokens from self-supervised speech models (HuBERT). They captured some linguistic structure but **lost the speaker's voice** and recording quality, and were largely limited to resynthesis in one voice.

The trade-off comes from tokenisation. Tokens good for reconstruction carry every acoustic detail, making sequences long and structure hard to learn. Tokens good for structure discard the detail you need to sound real.

---

## The Core Innovation

Use **two tokenisers** and generate in a **hierarchy**:

```
                       audio waveform
                  /                      \
   [ w2v-BERT, k-means ]            [ SoundStream codec ]
   SEMANTIC tokens, 25 Hz           ACOUSTIC tokens, 50 Hz
   (what is said, melody)           (voice, timbre, room)
   poor reconstruction              high-quality reconstruction
```

Generation happens in three stages, each a separate decoder-only Transformer:

```
Stage 1  Semantic modelling
         predict semantic tokens         -> long-term structure

Stage 2  Coarse acoustic modelling
         predict coarse acoustic tokens  -> speaker identity,
         conditioned on semantic tokens     recording conditions

Stage 3  Fine acoustic modelling
         predict fine acoustic tokens    -> removes compression
         conditioned on coarse ones         artefacts, adds detail

Then the SoundStream decoder turns the acoustic tokens back into a waveform.
```

To continue a prompt, AudioLM tokenises the 3-second prompt with both tokenisers and runs each stage forward from it.

---

## Key Components Explained

### 1. Acoustic tokens from SoundStream
**What it does:** Provides a compact, high-fidelity discrete representation of sound.
**How it works:** SoundStream is a neural audio codec (Google, 2021). A convolutional encoder produces one embedding every 20 ms (50 Hz), and **residual vector quantisation (RVQ)** turns each embedding into a stack of codes: the first quantiser captures a coarse approximation, each subsequent one encodes what is left over. With 4 quantisers of 1,024 entries, that is about 2,000 bits per second. Coarse quantisers carry speaker and recording properties; fine ones carry detail. This is the same discrete-codebook idea as [VQ-VAE](../../image-generation/89-vq-vae/summary.md), applied in layers.

### 2. Semantic tokens from w2v-BERT
**What it does:** Captures linguistic and melodic content.
**How it works:** w2v-BERT (a 0.6-billion-parameter self-supervised speech model, trained with masked prediction like BERT) produces embeddings at 25 Hz. AudioLM takes an intermediate layer, clusters the embeddings with k-means and uses the cluster index as the token. The paper measured that these tokens are much better at distinguishing phonemes than acoustic tokens but reconstruct audio poorly, confirming the need for both.

### 3. The hierarchy and why it helps
**What it does:** Keeps sequences short enough to model while preserving both properties.
**How it works:** Semantic tokens are assumed to depend only on past semantic tokens, not on acoustic detail, so stage 1 is a short sequence. Stage 3 assumes fine detail is local, so it runs on non-overlapping 3-second chunks and can ignore semantic tokens. Each stage is a 12-layer Transformer (16 heads, width 1,024) with about 0.3 billion parameters.

### 4. A detector for its own output
**What it does:** A mitigation against misuse.
**How it works:** Because AudioLM continuations fooled human listeners, the authors trained a classifier to detect AudioLM-generated speech. It reached 98.6 percent accuracy, even when compared against real audio passed through the same codec.

---

## Key Results

- **Human indistinguishability:** 10 raters, 100 samples of 10 seconds (3 s real prompt plus 7 s continuation, or 10 s of real audio passed through the codec). Correct identification was **51.2 percent**, not significantly different from chance (p = 0.23).
- **Linguistic knowledge without text:** on the ZeroResource 2021 tests sWUGGY (does the model prefer real words over non-words?) and sBLIMP (does it prefer grammatical sentences?), AudioLM scored highest among systems without text supervision, improving sBLIMP by 8 percent relative over the previous best.
- **Voice preservation:** a speaker classifier identified the prompt's speaker in AudioLM continuations 92.6 percent of the time.
- **Intelligibility:** word error rate of about 6 percent when its continuations were transcribed by a speech recogniser, as later reported in the [VALL-E](../122-vall-e/summary.md) comparison.
- **Piano:** retrained on an internal dataset of 40,000 hours of piano music, AudioLM was preferred over an acoustic-tokens-only model in 83.3 percent of paired comparisons, because only the full hierarchy kept consistent melody and rhythm.

---

## Why This Was Revolutionary

- **Language modelling became a general audio generator.** No transcripts, no MIDI, no hand-designed features.
- **The semantic/acoustic token split** was a clean, reusable solution to the long-standing trade-off between coherence and fidelity.
- **Neural codecs became the standard tokeniser** for generative audio, the way BPE is for text.
- **Safety built into the paper.** Pairing a capability this strong with a released-in-paper detector was an early example of shipping a mitigation alongside a generative result.

---

## Real-World Impact

- **MusicLM (Google, January 2023)** extended the hierarchy to text-conditioned music generation.
- **[VALL-E](../122-vall-e/summary.md) (Microsoft, January 2023)** took the codec-token language model and conditioned it on text, turning AudioLM-style continuation into zero-shot text-to-speech with 3-second voice cloning.
- **SoundStorm (Google, 2023)** replaced the slow autoregressive acoustic stages with parallel decoding.
- **Speech-native dialogue models.** Later full-duplex spoken assistants operate on audio tokens directly. Kyutai's Moshi (2024), for example, uses a codec whose first level is distilled from a self-supervised speech model, a direct descendant of the semantic-plus-acoustic idea. OpenAI has not published architectural details for [GPT-4o](../../language-models/40-gpt4o/summary.md)'s voice mode.
- **Complement to recognition.** Where [Whisper](../49-whisper/summary.md) turns speech into text, AudioLM-style models generate speech; modern voice systems combine both directions.

---

## Key Takeaways for Practitioners

1. **Tokenisation is the design decision.** For audio, choose tokens by what you need: structure (self-supervised clusters) or fidelity (codec codes). Often you need both.
2. **Codec bitrate trades against sequence length.** More RVQ levels mean better audio and longer sequences; hierarchical or parallel decoding manages that.
3. **Prompted continuation is voice cloning.** Any model that continues audio in the prompt's voice needs a misuse policy and ideally a detector.
4. **Short prompts go a long way.** Three seconds was enough to carry voice identity and recording conditions.

---

## Limitations & Future Directions

- **No text control.** AudioLM continues audio; it cannot be told what to say. VALL-E and SPEAR-TTS added text conditioning.
- **Slow.** Three autoregressive stages over long token sequences; SoundStorm addressed this.
- **English audiobook speech only** for the speech model, mostly reading style.
- **Semantic content is plausible, not meaningful.** Continuations are grammatical and topical over short spans but can drift over longer ones.
- **Misuse risks.** The authors name impersonation and biometric spoofing, and note that bias in training data can affect accents and dialects. Their detector is specific to AudioLM and would not by itself catch other generators.

---

## Further Reading

- **Original Paper:** [arxiv.org/abs/2209.03143](https://arxiv.org/abs/2209.03143)
- **Audio examples:** [google-research.github.io/seanet/audiolm/examples](https://google-research.github.io/seanet/audiolm/examples/)
- **SoundStream:** [arxiv.org/abs/2107.03312](https://arxiv.org/abs/2107.03312)
- **w2v-BERT:** [arxiv.org/abs/2108.06209](https://arxiv.org/abs/2108.06209)
- **MusicLM:** [arxiv.org/abs/2301.11325](https://arxiv.org/abs/2301.11325)
- **In this collection:** [VALL-E](../122-vall-e/summary.md), [Whisper](../49-whisper/summary.md), [VQ-VAE](../../image-generation/89-vq-vae/summary.md), [GPT-4o](../../language-models/40-gpt4o/summary.md)

## Citation

```bibtex
@article{borsos2023audiolm,
  title={AudioLM: A Language Modeling Approach to Audio Generation},
  author={Borsos, Zal{\'a}n and Marinier, Rapha{\"e}l and Vincent, Damien and Kharitonov, Eugene and Pietquin, Olivier and Sharifi, Matt and Roblek, Dominik and Teboul, Olivier and Grangier, David and Tagliasacchi, Marco and Zeghidour, Neil},
  journal={IEEE/ACM Transactions on Audio, Speech, and Language Processing},
  volume={31},
  pages={2523--2533},
  year={2023}
}
```

<!-- related:start -->

---

## Related in This Collection

- [BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](../../language-models/03-bert/summary.md)
- [GPT-4 Technical Report](../../language-models/36-gpt4/summary.md)
- [GPT-4o: The First Omni Model](../../language-models/40-gpt4o/summary.md)
- [Whisper: Robust Speech Recognition via Large-Scale Weak Supervision](../../multimodal/49-whisper/summary.md)
- [Neural Discrete Representation Learning (VQ-VAE)](../../image-generation/89-vq-vae/summary.md)
- [Neural Codec Language Models are Zero-Shot Text to Speech Synthesizers (VALL-E)](../../multimodal/122-vall-e/summary.md)
- [Neural Machine Translation of Rare Words with Subword Units (BPE)](../../techniques/136-bpe-subword-units/summary.md)

<!-- related:end -->
