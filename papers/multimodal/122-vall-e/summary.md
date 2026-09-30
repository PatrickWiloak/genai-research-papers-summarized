---
title: "Neural Codec Language Models are Zero-Shot Text to Speech Synthesizers (VALL-E)"
slug: "122-vall-e"
number: 122
category: "multimodal"
authors: "Chengyi Wang, Sanyuan Chen, Yu Wu, Ziqiang Zhang, Long Zhou, Shujie Liu, Zhuo Chen, Yanqing Liu, Huaming Wang, Jinyu Li, Lei He, Sheng Zhao, Furu Wei (Microsoft)"
published: "January 2023"
year: 2023
url: "https://arxiv.org/abs/2301.02111"
tags: ["audio", "language-model", "safety"]
---

# Neural Codec Language Models are Zero-Shot Text to Speech Synthesizers (VALL-E)

**Authors:** Chengyi Wang, Sanyuan Chen, Yu Wu, Ziqiang Zhang, Long Zhou, Shujie Liu, Zhuo Chen, Yanqing Liu, Huaming Wang, Jinyu Li, Lei He, Sheng Zhao, Furu Wei (Microsoft)
**Published:** January 2023
**Paper:** [arxiv.org/abs/2301.02111](https://arxiv.org/abs/2301.02111)

---

## Why This Matters

VALL-E is **the paper that turned text-to-speech into a language-modelling problem and made 3-second voice cloning a research baseline**. Give it a sentence and a 3-second recording of someone it has never heard, and it speaks the sentence in that person's voice, keeping their emotion and even the acoustics of the room they recorded in.

- **Scale over engineering.** Trained on 60,000 hours of English speech from over 7,000 speakers, which the authors describe as hundreds of times more than typical TTS training data.
- **Clear gains on zero-shot TTS.** Against the previous best zero-shot system (YourTTS) on LibriSpeech, speaker similarity rose from 0.337 to 0.580 and human similarity ratings from 3.45 to 4.38 out of 5.
- **In-context learning for speech.** Like GPT-3 with a prompt, VALL-E adapts to a new voice from the prompt alone, with no fine-tuning.
- **It made the misuse problem concrete.** A convincing clone from 3 seconds of audio, which anyone can get from a voicemail or a video clip, is the core of phone scams and political deepfakes. The paper names this risk.

**The insight:** take a neural audio codec that turns speech into discrete tokens, as [AudioLM](../121-audiolm/summary.md) did, and treat TTS as "predict the codec tokens that follow this phoneme sequence and this voice prompt." Then scale the data the way text language models did.

---

## The Problem: TTS Did Not Generalise to New Voices

Standard TTS systems in 2022 used a cascade: text to phonemes, phonemes to a mel spectrogram (an acoustic model), spectrogram to waveform (a vocoder). They were trained on small amounts of clean studio recordings. That produced excellent quality for the voices they were trained on and poor results for anyone else:

- **Zero-shot quality collapsed.** For unseen speakers, speaker similarity and naturalness dropped sharply.
- **Adaptation was heavy.** Speaker adaptation needed fine-tuning; speaker-encoder approaches needed carefully designed features and architectures.
- **Internet-scale data did not help** because noisy, varied audio degraded models built for clean studio input.

Meanwhile text language models had shown that more, messier data plus a simple next-token objective produces in-context learning. VALL-E asked whether speech could follow the same path.

---

## The Core Innovation

Replace "phonemes to spectrogram" with "phonemes to discrete codec codes," modelled by a language model:

```
Old cascade:   text -> phonemes -> mel spectrogram -> vocoder -> waveform
VALL-E:        text -> phonemes -> codec tokens    -> codec decoder -> waveform

Inputs at inference:
   phonemes of the text you want spoken
   + phonemes of the 3-second prompt's transcript
   + codec tokens of the 3-second prompt
Output:
   codec tokens that continue the prompt, saying the new text in the same voice
```

The codec is Meta's **EnCodec**, used off the shelf. At 24 kHz it produces embeddings at 75 Hz, each quantised by **8 residual vector quantisers** with 1,024 entries each (the 6 kbps setting). The first quantiser carries the coarse content and most speaker identity; later ones add finer detail.

---

## Key Components Explained

### 1. Autoregressive model for the first quantiser
**What it does:** Decides content, timing and prosody.
**How it works:** A decoder-only Transformer generates the first-level codes one frame at a time, conditioned on the phonemes and the prompt's first-level codes. Autoregressive generation naturally decides how long each sound lasts, so there is no separate duration model. Decoding uses sampling rather than beam search, which the authors found could send the model into infinite loops. Sampling also makes outputs diverse: the same text and prompt give different renditions.

### 2. Non-autoregressive model for quantisers 2 to 8
**What it does:** Fills in acoustic detail quickly.
**How it works:** A second Transformer predicts all frames of quantiser j at once, given the phonemes, the prompt and all codes from quantisers 1 to j-1. It is called seven times, once per remaining quantiser, with greedy decoding. This keeps inference cost far below generating all 8 x 75 codes per second autoregressively. Both models have 12 layers, 16 attention heads and width 1,024.

```
frame:       1   2   3   4   5  ...
quantiser 1  AR  AR  AR  AR  AR       <- one frame at a time
quantiser 2  [ all frames at once ]   <- NAR pass 1
...
quantiser 8  [ all frames at once ]   <- NAR pass 7
```

### 3. Training data: LibriLight, transcribed by machine
**What it does:** Provides scale.
**How it works:** LibriLight is 60,000 hours of audiobook speech with no transcripts. The authors trained a speech recogniser on 960 hours of labelled LibriSpeech and used it to produce phoneme transcriptions for all of LibriLight. The labels are imperfect, and the paper argues the scale more than compensates.

### 4. In-context prompting
**What it does:** Conditions on a new voice without training.
**How it works:** Two modes: **VALL-E** takes a 3-second clip from another utterance of the target speaker and speaks new text; **VALL-E-continual** uses the first 3 seconds of the utterance itself and continues it. The model also carries over the prompt's emotion and acoustic environment (for example reverberation), which conventional TTS could not do.

---

## Key Results

LibriSpeech test-clean (speakers unseen in training), from the paper's Table 2 and Table 3:

```
System              WER (%)   Speaker similarity (WavLM, -1 to 1)
------------------  -------   ----------------------------------
Ground truth        2.2       0.754
YourTTS             7.7       0.337
VALL-E              5.9       0.580
VALL-E-continual    3.8       0.508
(AudioLM, reported  6.0       n/a)

Human ratings (40 speakers):
                    SMOS (1-5)    CMOS vs VALL-E
YourTTS             3.45          -0.12
VALL-E              4.38           0.00
Ground truth        4.5           +0.17
```

- **WER** is word error rate when a speech recogniser transcribes the output (lower means more intelligible). **SMOS** is human-rated similarity to the target voice; **CMOS** is human-rated comparative naturalness.
- **VCTK** (a multi-accent dataset): VALL-E beat YourTTS by +0.11 SMOS and +0.23 CMOS, and scored +0.04 CMOS against ground truth, meaning listeners judged its naturalness on par with real recordings.

---

## Why This Was Revolutionary

- **Recast TTS as language modelling** over codec tokens, bringing scaling and in-context learning to speech synthesis.
- **Made zero-shot voice cloning from 3 seconds a standard benchmark setting,** following AudioLM's 3-second prompts.
- **Preserved emotion and acoustic environment,** showing that the prompt carries far more than a speaker embedding.
- **Established the AR-plus-NAR codec recipe** that many later TTS systems adopted or reacted against.

---

## Real-World Impact

- **A Microsoft series:** VALL-E X (March 2023) added cross-lingual synthesis (speak another language in your own voice), and **VALL-E 2 (June 2024)** claimed "human parity" zero-shot TTS on LibriSpeech and VCTK, using repetition-aware sampling and grouped code modelling.
- **An industry pattern.** Codec language models, and later hybrids with diffusion or flow-matching decoders, became a dominant approach for zero-shot TTS across academic and open-source projects through 2024 and 2025.
- **Commercial voice cloning** services that clone from seconds of audio became widely available from several companies, whether or not they used VALL-E's exact design.

---

## The Misuse Problem, Honestly

VALL-E's capability is exactly what a fraudster wants: a convincing copy of a specific person's voice from a few seconds of public audio.

- **What the paper says.** Its broader-impacts paragraph acknowledges risks "such as spoofing voice identification or impersonating a specific speaker," suggests that a detection model could be built, and says Microsoft will apply its AI Principles. Unlike [AudioLM](../121-audiolm/summary.md), which trained and reported a detector (98.6 percent accuracy), the VALL-E paper only proposes one.
- **What Microsoft did.** Microsoft published demo samples, not a downloadable model. For VALL-E 2 its project page states that it has "no plans to incorporate VALL-E 2 into a product or expand access to the public," and frames it as research. Open-source reimplementations of the VALL-E design appeared anyway, and comparable capability became available from other sources.
- **What happened in the world.** In January 2024, New Hampshire voters received robocalls with an AI-generated voice imitating President Biden telling them not to vote in the primary. In February 2024 the US Federal Communications Commission ruled unanimously that AI-generated voices count as "artificial" under the Telephone Consumer Protection Act, making such robocalls illegal without consent. Voice-clone scams impersonating relatives or executives have been widely reported.
- **Why mitigations are hard.** Detectors trained on one generator often miss others. Watermarks only work if the generator applies them and the audio is not re-encoded. Voice biometrics used by banks are directly threatened. Consent and provenance rules help legitimate providers but do not bind bad actors running open models.
- **The legitimate uses are real too:** restoring voices for people with ALS or aphasia (explicitly cited in the VALL-E 2 paper), dubbing, accessibility and personalised assistants. The tension between those benefits and impersonation risk has not been resolved as of 2026.

---

## Key Takeaways for Practitioners

1. **Codec tokens plus a language model is a strong TTS baseline;** the quality of the codec caps the quality of the output.
2. **Autoregressive decoding brings robustness problems:** skipped, repeated or garbled words. Measure WER, not just similarity.
3. **Treat any voice-cloning capability as dual-use.** Require consent for target voices, log usage, and watermark outputs where you can.
4. **Do not rely on voice as an authentication factor.** Three seconds of public audio can be enough to clone it.
5. **Reading-style data gives reading-style speech.** Audiobook-trained models sound like audiobooks; conversational use needs conversational data.

---

## Limitations & Future Directions

- **Synthesis robustness.** The authors report that words can be unclear, missed or duplicated, caused by unstable attention alignment in the autoregressive model. VALL-E 2's repetition-aware sampling targets this.
- **Data coverage.** Even 60,000 hours does not cover every voice, especially accented speakers (results on VCTK were worse than on LibriSpeech), and audiobook data is mostly reading style.
- **Two-model design.** The authors suggest a single universal model, or a fully non-autoregressive one for speed, as future directions.
- **English only** in the original; VALL-E X added cross-lingual support.
- **Misuse.** See above. The paper's mitigation is a suggestion, not a released tool.

---

## Further Reading

- **Original Paper:** [arxiv.org/abs/2301.02111](https://arxiv.org/abs/2301.02111)
- **Demo page:** [aka.ms/valle](https://aka.ms/valle)
- **VALL-E X (cross-lingual):** [arxiv.org/abs/2303.03926](https://arxiv.org/abs/2303.03926)
- **VALL-E 2:** [arxiv.org/abs/2406.05370](https://arxiv.org/abs/2406.05370)
- **EnCodec:** [arxiv.org/abs/2210.13438](https://arxiv.org/abs/2210.13438)
- **FCC ruling on AI voices in robocalls:** [fcc.gov/document/fcc-makes-ai-generated-voices-robocalls-illegal](https://www.fcc.gov/document/fcc-makes-ai-generated-voices-robocalls-illegal)
- **In this collection:** [AudioLM](../121-audiolm/summary.md), [Whisper](../49-whisper/summary.md), [GPT-3](../../language-models/04-gpt3-few-shot-learners/summary.md) (in-context learning), [VQ-VAE](../../image-generation/89-vq-vae/summary.md)

## Citation

```bibtex
@article{wang2023valle,
  title={Neural Codec Language Models are Zero-Shot Text to Speech Synthesizers},
  author={Wang, Chengyi and Chen, Sanyuan and Wu, Yu and Zhang, Ziqiang and Zhou, Long and Liu, Shujie and Chen, Zhuo and Liu, Yanqing and Wang, Huaming and Li, Jinyu and He, Lei and Zhao, Sheng and Wei, Furu},
  journal={arXiv preprint arXiv:2301.02111},
  year={2023}
}
```

<!-- related:start -->

---

## Related in This Collection

- [Language Models are Few-Shot Learners (GPT-3)](../../language-models/04-gpt3-few-shot-learners/summary.md)
- [Whisper: Robust Speech Recognition via Large-Scale Weak Supervision](../../multimodal/49-whisper/summary.md)
- [Neural Discrete Representation Learning (VQ-VAE)](../../image-generation/89-vq-vae/summary.md)
- [AudioLM: a Language Modeling Approach to Audio Generation (AudioLM)](../../multimodal/121-audiolm/summary.md)

<!-- related:end -->
