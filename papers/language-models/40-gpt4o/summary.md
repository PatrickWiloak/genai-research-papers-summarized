# GPT-4o: The First Omni Model

**Authors:** OpenAI
**Published:** May 13, 2024
**System Card:** [cdn.openai.com/gpt-4o-system-card.pdf](https://cdn.openai.com/gpt-4o-system-card.pdf)

---

## Why This Matters

GPT-4o was **the first truly native multimodal model**:

- 🎯 **"Omni" = everything** - Text, audio, image, and video in ONE model
- 🗣️ **Real-time voice** - 232ms average response time (human-like)
- ⚡ **2x faster than GPT-4 Turbo** - Same quality, half the latency
- 💰 **50% cheaper** - Made frontier AI more accessible
- 🌍 **Multimodal first** - Not bolted-on, natively trained across modalities

**Real-world impact:**
- Made voice AI feel natural for the first time
- Set the standard for multimodal integration
- GPT-4o mini became the most cost-effective frontier model
- Forced competitors (Google, Anthropic) to accelerate multimodal plans

**The insight:** **Stop building separate models for text, vision, and audio.** Train one model that understands all modalities natively, and the modalities reinforce each other.

---

## The Breakthrough

### Before GPT-4o: Separate Models

```
Previous ChatGPT voice pipeline:
  User speaks → Whisper (speech-to-text) → GPT-4 (text processing) → TTS (text-to-speech)
  Latency: 2.8-5.4 seconds
  Lost: Tone, emotion, background sounds, non-verbal cues

GPT-4o pipeline:
  User speaks → GPT-4o (processes audio directly) → GPT-4o (generates audio directly)
  Latency: 232ms average (320ms worst case)
  Preserved: Tone, emotion, laughter, singing, emphasis
```

### Native Multimodality

**What "native" means:**

```
Bolted-on multimodal (GPT-4V):
  Image → Separate vision encoder → Text representation → LLM
  Audio → Separate speech model → Text → LLM
  Problem: Loses cross-modal information

Native multimodal (GPT-4o):
  Image + Audio + Text → Single unified model → Any output modality
  Advantage: Modalities inform each other directly
```

**Example of cross-modal understanding:**
```
Input: [Audio of someone saying "I'm fine" in a sad tone]

Bolted-on: Transcribes "I'm fine" → Responds to text "I'm fine"
GPT-4o: Detects sad tone + words → "You say you're fine,
but you sound like you might be having a tough day.
Want to talk about it?"
```

---

## Performance

### Text Benchmarks

**MMLU (Language Understanding):**

| Model | Score |
|-------|-------|
| GPT-4 | 86.4% |
| Claude 3 Opus | 86.8% |
| Gemini 1.5 Pro | 85.9% |
| **GPT-4o** | **87.2%** |

**Matches or exceeds GPT-4 on every text benchmark.**

### Vision Benchmarks

**MMMU (Multimodal Understanding):**

| Model | Score |
|-------|-------|
| GPT-4V | 56.8% |
| Gemini 1.5 Pro | 58.5% |
| **GPT-4o** | **69.1%** |

**Significant jump in visual reasoning.**

### Audio Capabilities

**Speech recognition (Whisper-level accuracy):**
```
GPT-4o matches Whisper v3 on English
Dramatically better on low-resource languages
Can understand tone, emotion, multiple speakers
```

**Voice response time:**

| System | Average Latency |
|--------|----------------|
| GPT-3.5 voice | 2.8s |
| GPT-4 voice | 5.4s |
| **GPT-4o voice** | **232ms** |

**24x faster than GPT-4 voice!**

### Multilingual Performance

GPT-4o dramatically improved non-English performance:
```
Tokenizer efficiency improvement:
- Hindi: 4.4x fewer tokens
- Arabic: 2.0x fewer tokens
- Chinese: 1.5x fewer tokens

Result: Better quality AND cheaper for non-English
```

---

## How It Works

### Unified Architecture

**Single model, all modalities:**

```
                    ┌─────────────────┐
  Text input   ───>│                 │──> Text output
  Image input  ───>│   GPT-4o       │──> Audio output
  Audio input  ───>│   (unified)    │──> Image understanding
  Video input  ───>│                 │──> Cross-modal reasoning
                    └─────────────────┘

Key: All modalities share the same transformer weights
Cross-modal attention: Audio tokens attend to image tokens, etc.
```

### End-to-End Audio

**The biggest innovation:**

```
Previous: Audio → Text → Process → Text → Audio (pipeline)
GPT-4o:   Audio → Process → Audio (end-to-end)

This preserves:
- Emotional tone and inflection
- Singing ability
- Laughter and non-verbal sounds
- Accents and speaking styles
- Background audio context
```

### Speed and Efficiency

**How GPT-4o is faster AND cheaper:**
```
1. Single model (not 3 models in a pipeline)
2. Optimized architecture (likely distillation from GPT-4)
3. Better tokenizer (fewer tokens for same content)
4. Efficient serving infrastructure

Result:
- 2x faster than GPT-4 Turbo
- 50% cheaper per token
- Same or better quality
```

---

## GPT-4o Mini

**Released July 2024 - the efficiency breakthrough:**

| Aspect | GPT-4o | GPT-4o Mini |
|--------|--------|-------------|
| **MMLU** | 87.2% | 82.0% |
| **Speed** | Fast | Faster |
| **Input price** | $5/1M tokens | $0.15/1M tokens |
| **Output price** | $15/1M tokens | $0.60/1M tokens |

**33x cheaper than GPT-4o for input!**

**GPT-4o mini vs competitors:**

| Model | MMLU | Price (input) |
|-------|------|--------------|
| GPT-4o mini | 82.0% | $0.15/1M |
| Gemini Flash | 77.9% | $0.075/1M |
| Claude Haiku | 73.8% | $0.25/1M |

**Best quality-per-dollar at launch.**

---

## Real-World Capabilities

### Voice Conversations

```
GPT-4o can:
- Have natural real-time conversations
- Detect and respond to emotional tone
- Sing songs and generate music-like audio
- Handle interruptions mid-sentence
- Switch between languages fluidly
- Tell stories with dramatic inflection
```

### Vision Tasks

```
GPT-4o can:
- Solve math problems from photos of handwriting
- Read and analyze charts/graphs
- Understand memes and visual humor
- Assist with real-world navigation (photos)
- Analyze medical images (with caveats)
- Code from screenshots/mockups
```

### Cross-Modal Tasks

```
Novel capabilities:
- "Describe this image, then discuss it vocally with emotion"
- "Look at this math equation photo and explain it step by step"
- "Translate this sign in the photo and speak the translation"
- Real-time visual assistance (live camera feed + voice)
```

---

## Practical Usage

### API Access

```python
from openai import OpenAI

client = OpenAI()

# Text + Image
response = client.chat.completions.create(
    model="gpt-4o",
    messages=[
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "What's happening in this image?"},
                {
                    "type": "image_url",
                    "image_url": {"url": "https://example.com/photo.jpg"}
                }
            ]
        }
    ]
)

print(response.choices[0].message.content)
```

### Audio API

```python
# Real-time audio conversation
from openai import OpenAI

client = OpenAI()

# Using the Realtime API for voice
# (WebSocket-based for low-latency streaming)
response = client.audio.speech.create(
    model="gpt-4o-audio-preview",
    voice="alloy",
    input="Hello! How can I help you today?"
)

# Stream audio response
response.stream_to_file("output.mp3")
```

---

## Impact on the Field

### Changed Multimodal Expectations

```
Before GPT-4o:
  "Multimodal = text model + vision encoder"
  Separate models stitched together

After GPT-4o:
  "Multimodal = single model, all modalities natively"
  End-to-end, no pipelines
```

### Forced Industry Response

- **Google:** Accelerated Gemini's native multimodal capabilities
- **Anthropic:** Expanded Claude's vision, began audio work
- **Meta:** Designed Llama 4 as natively multimodal
- **Open source:** LLaVA and others improved multimodal integration

### Voice AI Revolution

```
Before GPT-4o: Voice AI felt robotic (pipeline delay, no emotion)
After GPT-4o:  Voice AI felt human (instant, emotional, natural)

Spawned: New voice AI startups, real-time translation tools,
         voice-first interfaces, accessibility applications
```

---

## Limitations

### 1. Audio Generation Risks
```
Can generate realistic voices
Risk of deepfakes and impersonation
OpenAI restricted to preset voices initially
```

### 2. Not All Modalities Equal
```
Text: Excellent (GPT-4 level)
Vision: Very good (state-of-the-art)
Audio: Good but still developing
Video: Limited (frame-by-frame, not true video understanding)
```

### 3. Still Hallucinates
```
Multimodal doesn't fix hallucination
Can "see" things that aren't in images
Audio transcription not perfect
```

---

## Key Takeaways

1. **Native multimodal > Pipeline** - One model for all modalities beats stitching separate models together
2. **Voice breakthrough** - 232ms latency made voice AI feel human for the first time
3. **Faster AND cheaper** - 2x speed, 50% cost reduction over GPT-4 Turbo
4. **GPT-4o mini** - Made frontier-quality AI affordable for everyone
5. **Set the standard** - Every competitor now building native multimodal models

**Bottom line:** GPT-4o proved that the future of AI is natively multimodal. By training a single model across text, audio, and vision, it achieved better results than pipelines of specialized models - faster, cheaper, and with richer cross-modal understanding.

---

## Further Reading

### Official Resources
- **GPT-4o System Card:** https://cdn.openai.com/gpt-4o-system-card.pdf
- **Announcement:** https://openai.com/index/hello-gpt-4o/
- **GPT-4o mini:** https://openai.com/index/gpt-4o-mini-advancing-cost-efficient-intelligence/

### Related Work
- **GPT-4 Technical Report:** https://arxiv.org/abs/2303.08774
- **Gemini (native multimodal):** https://arxiv.org/abs/2312.11805
- **CLIP (vision-language bridge):** https://arxiv.org/abs/2103.00020

---

**Published:** May 13, 2024
**Impact:** 🔥🔥🔥🔥 **HIGH** - Defined native multimodal AI
**Adoption:** Massive - default ChatGPT model, most-used frontier model globally
**Current Relevance:** Superseded by GPT-4.1 and GPT-5, but established the omni paradigm
**Legacy:** Made native multimodal the standard, proved voice AI could feel human

**Modern Status (March 2026):** GPT-4o has been succeeded by GPT-4.1 and GPT-5, but the "omni" approach it pioneered is now the norm. Every major model is natively multimodal. GPT-4o mini remains widely used as a cost-effective option.
