# GPT-4V(ision): System Card

**Authors:** OpenAI
**Published:** September 2023
**Paper:** [cdn.openai.com/papers/GPTV_System_Card.pdf](https://cdn.openai.com/papers/GPTV_System_Card.pdf)

---

## Why This Matters

GPT-4V brought **vision capabilities to the most powerful LLM**:

- 👁️ **Multimodal GPT-4** - Same model, now understands images
- 🎯 **State-of-the-art vision-language** - Best VQA, OCR, reasoning
- 💡 **Real-world utility** - Accessibility, education, analysis
- 🔥 **Sparked multimodal race** - Gemini, Claude 3, LLaVA follow
- 🚀 **New AI paradigm** - From text-only to truly multimodal

**Real-world impact:**
- Be My Eyes partnership (accessibility for blind users)
- Educational tutoring with diagrams/charts
- Code from screenshots/mockups
- Scientific diagram analysis
- Medical imaging assistance (controversial)

**The insight:** GPT-4's reasoning + vision = powerful new capabilities. The future is multimodal.

---

## What is GPT-4V?

### GPT-4 with Vision

**GPT-4 (text-only):**
```
Input:  Text prompt
Output: Text response
```

**GPT-4V (multimodal):**
```
Input:  Text prompt + images
Output: Text response about the images
```

**Key capability:** Understand visual content with GPT-4's reasoning power

---

## Capabilities

### 1. Visual Question Answering (VQA)

**Example:**
```
Image: [Photo of a busy street scene]
User: "How many red cars are in this image?"
GPT-4V: "I can see 3 red cars: one sedan near the traffic light,
         one SUV in the middle lane, and one compact car parked on
         the right side."
```

**Performance:**
- MMMU (Multimodal Massive Understanding): 56.8% (vs 35% for other models)
- VQAv2: State-of-the-art accuracy

### 2. OCR and Document Understanding

**Example:**
```
Image: [Screenshot of a receipt]
User: "Extract the items and prices from this receipt"
GPT-4V: "
Items:
- Coffee: $4.50
- Croissant: $3.25
- Orange juice: $5.00
Total: $12.75
Tip: $2.00
Grand Total: $14.75
"
```

**Can handle:**
- Receipts, invoices
- Forms, documents
- Handwritten text
- Multi-language text

### 3. Chart and Graph Analysis

**Example:**
```
Image: [Complex line graph showing stock prices]
User: "What trends do you see?"
GPT-4V: "The graph shows three main trends:
         1. Stock A peaked in March at ~$150
         2. Stock B has been declining since July
         3. Stock C shows high volatility with a recent uptick"
```

**Capabilities:**
- Quantitative reasoning
- Trend analysis
- Multi-chart comparison

### 4. Meme and Humor Understanding

**Example:**
```
Image: [Distracted boyfriend meme]
User: "Explain this meme"
GPT-4V: "This is the 'distracted boyfriend' meme. It's being used to
         show [context-specific interpretation based on labels]"
```

**Impressive:** Cultural context, visual metaphors, wordplay

### 5. Scientific Diagrams

**Example:**
```
Image: [Cell biology diagram]
User: "Label the parts of this cell"
GPT-4V: "This appears to be a eukaryotic cell. I can see:
         - Nucleus (labeled 1)
         - Mitochondria (labeled 2)
         - Endoplasmic reticulum (labeled 3)
         ..."
```

### 6. Spatial Reasoning

**Example:**
```
Image: [Room layout]
User: "Will this couch fit through that doorway?"
GPT-4V: "Based on the image, the couch appears to be approximately
         7-8 feet long, while the doorway looks to be about 3 feet wide.
         The couch would need to be rotated vertically, and even then
         it might be a tight fit. I'd recommend measuring both to be sure."
```

### 7. Code from Images

**Example:**
```
Image: [Hand-drawn UI mockup]
User: "Generate HTML/CSS for this design"
GPT-4V: [Generates working code matching the mockup]
```

**Capabilities:**
- Screenshot to code
- Diagram to implementation
- Mockup to prototype

---

## Technical Approach

### Architecture (Inferred)

**Not publicly disclosed, but likely:**
```
┌─────────────────────────────────────┐
│         Vision Encoder              │
│    (Based on CLIP or similar)       │
│                                     │
│  Image → Vision Transformer → Embedding
└─────────────────┬───────────────────┘
                  │
                  ▼
┌─────────────────────────────────────┐
│       Projection Layer              │
│  (Maps vision to language space)    │
└─────────────────┬───────────────────┘
                  │
                  ▼
┌─────────────────────────────────────┐
│          GPT-4 LLM                  │
│  (Processes text + vision tokens)   │
│                                     │
│  [Text tokens] + [Image tokens] → Response
└─────────────────────────────────────┘
```

**Similar to:**
- LLaVA (vision encoder + LLM)
- Flamingo (cross-attention)
- BLIP-2 (Q-Former)

**But much larger scale and better training**

### Training (Speculated)

**Phase 1: Pre-training**
- Image-text pairs (billions)
- Captioning, VQA datasets
- Web-scale data

**Phase 2: Instruction Tuning**
- High-quality vision-language tasks
- Human feedback (RLHF)
- Safety training

**Phase 3: Alignment**
- Refusal training (harmful content)
- Bias mitigation
- Safety evaluations

---

## Benchmarks

### Visual Question Answering

| Benchmark | GPT-4V | Best Previous | Improvement |
|-----------|--------|---------------|-------------|
| **MMMU** | 56.8% | 35.0% | +62% |
| **VQAv2** | SOTA | - | New SOTA |
| **TextVQA** | SOTA | - | New SOTA |

### Document Understanding

| Task | GPT-4V | Previous Best |
|------|--------|---------------|
| **DocVQA** | SOTA | - |
| **InfoVQA** | SOTA | - |
| **ChartQA** | SOTA | - |

### Multimodal Reasoning

| Benchmark | GPT-4V | GPT-4 (text) | Gain from Vision |
|-----------|--------|--------------|------------------|
| **ScienceQA** (with images) | 85.7% | N/A | Enabled |
| **MathVista** | 58.1% | - | Best published |

---

## Real-World Applications

### 1. Accessibility (Be My Eyes)

**Partnership announced September 2023:**

**Use case:**
```
Blind user takes photo of:
- Product labels (read ingredients)
- Navigation (describe surroundings)
- Documents (read forms)
- Objects (identify items)

GPT-4V provides detailed descriptions
```

**Impact:** Transformative for visually impaired users

### 2. Education

**Tutoring with images:**
```
Student: [Photo of math problem]
        "I don't understand step 3"

GPT-4V: "In step 3, we're applying the quadratic formula.
         Let me break it down:
         [Explains with reference to image]"
```

**Capabilities:**
- Handwritten work analysis
- Diagram explanation
- Error identification

### 3. Medical Imaging (Controversial)

**Potential:**
```
Image: [X-ray]
GPT-4V: Can identify potential issues
```

**But OpenAI restricts this:**
- Not approved for medical diagnosis
- Can hallucinate
- Liability concerns

### 4. Programming

**Screenshot to code:**
```
Developer: [Screenshot of app]
           "Generate React code"

GPT-4V: [Produces working React components matching UI]
```

**Use cases:**
- Mockup to code
- Bug identification in UI
- Reverse engineering

### 5. Content Creation

**Image analysis for writing:**
```
Input: [Photo of landscape]
User: "Write a story inspired by this"

GPT-4V: [Generates creative content based on visual details]
```

---

## Safety and Limitations

### Refused Capabilities

**OpenAI explicitly blocks:**

**1. People Identification**
```
User: "Who is this person?"
GPT-4V: "I can't identify specific people in images"
```

**2. Medical Diagnosis**
```
User: "What disease is this?"
GPT-4V: "I'm not able to diagnose medical conditions.
         Please consult a healthcare professional."
```

**3. Captcha Solving**
```
User: "Solve this captcha"
GPT-4V: "I can't help with that"
```

### Known Limitations

**1. Hallucinations**
- Can describe things not in image
- May miscount objects
- Spatial reasoning errors

**Example:**
```
User: "How many birds?"
GPT-4V: "I see 7 birds"
[Actual: 5 birds]
```

**2. Text Recognition Errors**
- Especially for handwriting
- Low-resolution text
- Non-English scripts (varies)

**3. Fine-Grained Details**
```
Struggles with:
- Very small objects
- Subtle differences
- Low-quality images
```

**4. Temporal Understanding**
```
Cannot:
- Understand video well (treats as frames)
- Reason about motion
- Understand sequences reliably
```

**5. Depth Perception**
```
Challenges:
- 3D reasoning from 2D images
- Distance estimation
- Occlusion reasoning
```

### Safety Mitigations

**1. Refusal Training**
- Won't identify people
- Won't generate harmful content
- Careful with sensitive topics

**2. Bias Mitigation**
- Tested for gender/race bias
- Evaluation on fairness benchmarks
- Ongoing monitoring

**3. Usage Policies**
- Terms of service
- Rate limiting
- Content filtering

---

## Comparison with Alternatives

### GPT-4V vs Gemini Vision

| Aspect | GPT-4V | Gemini Vision |
|--------|--------|---------------|
| **Release** | Sep 2023 | Dec 2023 |
| **Benchmark Performance** | SOTA (at release) | Claimed better |
| **Context Length** | Limited | Longer (claimed) |
| **Availability** | ChatGPT Plus, API | Limited access |

### GPT-4V vs Claude 3 Vision

| Aspect | GPT-4V | Claude 3 Opus |
|--------|--------|---------------|
| **Release** | Sep 2023 | Mar 2024 |
| **Performance** | Strong | Competitive |
| **Image Count** | Multiple | Up to 5 (Pro) |
| **Use Case** | General | Analysis-heavy |

### GPT-4V vs LLaVA

| Aspect | GPT-4V | LLaVA |
|--------|--------|-------|
| **Accessibility** | Closed (API) | Open source |
| **Performance** | Better | Good |
| **Cost** | API pricing | Self-host |
| **Customization** | Limited | Full control |

---

## Practical Usage

### Using GPT-4V via API

```python
from openai import OpenAI
import base64

client = OpenAI(api_key="your-api-key")

# Method 1: URL
response = client.chat.completions.create(
    model="gpt-4-vision-preview",
    messages=[
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "What's in this image?"
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": "https://example.com/image.jpg"
                    }
                }
            ]
        }
    ],
    max_tokens=300
)

print(response.choices[0].message.content)


# Method 2: Base64
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

base64_image = encode_image("path/to/image.jpg")

response = client.chat.completions.create(
    model="gpt-4-vision-preview",
    messages=[
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "Describe this image in detail"
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{base64_image}"
                    }
                }
            ]
        }
    ]
)

print(response.choices[0].message.content)
```

### Multiple Images

```python
response = client.chat.completions.create(
    model="gpt-4-vision-preview",
    messages=[
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Compare these two images"},
                {
                    "type": "image_url",
                    "image_url": {"url": "https://example.com/image1.jpg"}
                },
                {
                    "type": "image_url",
                    "image_url": {"url": "https://example.com/image2.jpg"}
                }
            ]
        }
    ]
)
```

### Detail Level Control

```python
# Low detail (faster, cheaper)
{
    "type": "image_url",
    "image_url": {
        "url": "https://example.com/image.jpg",
        "detail": "low"  # 512x512 max
    }
}

# High detail (better quality, more expensive)
{
    "type": "image_url",
    "image_url": {
        "url": "https://example.com/image.jpg",
        "detail": "high"  # 2048x2048 tiles
    }
}
```

---

## Use Case Examples

### 1. Chart Analysis

```python
response = client.chat.completions.create(
    model="gpt-4-vision-preview",
    messages=[{
        "role": "user",
        "content": [
            {"type": "text", "text": """
                Analyze this sales chart:
                1. What are the key trends?
                2. Which product is performing best?
                3. Any anomalies?
            """},
            {"type": "image_url", "image_url": {"url": chart_url}}
        ]
    }]
)
```

### 2. UI to Code

```python
response = client.chat.completions.create(
    model="gpt-4-vision-preview",
    messages=[{
        "role": "user",
        "content": [
            {"type": "text", "text": """
                Generate HTML/CSS/JS code that recreates this UI design.
                Make it responsive and modern.
            """},
            {"type": "image_url", "image_url": {"url": mockup_url}}
        ]
    }],
    max_tokens=4096
)
```

### 3. Document Extraction

```python
response = client.chat.completions.create(
    model="gpt-4-vision-preview",
    messages=[{
        "role": "user",
        "content": [
            {"type": "text", "text": """
                Extract all information from this invoice in JSON format:
                {
                  "vendor": "",
                  "date": "",
                  "items": [],
                  "total": ""
                }
            """},
            {"type": "image_url", "image_url": {"url": invoice_url}}
        ]
    }]
)
```

### 4. Educational Tutoring

```python
response = client.chat.completions.create(
    model="gpt-4-vision-preview",
    messages=[
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "I got this math problem wrong. Can you explain where I made a mistake?"},
                {"type": "image_url", "image_url": {"url": homework_url}}
            ]
        }
    ]
)
```

---

## Pricing

**GPT-4V API Pricing** (as of 2024):

| Detail Level | Input (per image) | Output (per token) |
|--------------|-------------------|-------------------|
| **Low** | $0.01275 | $0.03/1K tokens |
| **High** | $0.0255+ | $0.03/1K tokens |

**Token calculation:**
- Low detail: ~85 tokens
- High detail: 170 tokens × number of 512px tiles

**Example cost:**
```
1 high-res image (1024x1024):
- 4 tiles × 170 = 680 tokens
- Cost: ~$0.02 per image

Plus text tokens as usual
```

---

## Impact on the Field

### Multimodal Race

**GPT-4V sparked competition:**

**2023:**
- GPT-4V (September)
- Google Gemini announced (December)

**2024:**
- Claude 3 with vision (March)
- LLaVA 1.5 (open source)
- Gemini 1.5 (1M context with images)

**Trend:** All frontier models now multimodal

### Research Directions

**Enabled new research:**
- Better vision-language training
- Multimodal reasoning
- Open-source alternatives (LLaVA, etc.)
- Specialized applications

### Industry Impact

**New applications possible:**
- Visual assistants
- Accessibility tools
- Education platforms
- Content moderation

---

## Future Directions

### Current Limitations to Address

**1. Video Understanding**
- Current: Frame-by-frame
- Future: Native video comprehension

**2. 3D Understanding**
- Current: 2D images only
- Future: 3D scene understanding

**3. Real-time Vision**
- Current: Static images
- Future: Live camera feeds

**4. Fine-Grained Control**
- Current: General vision
- Future: Domain-specific expertise

### Speculation on GPT-5

**Likely improvements:**
- Better spatial reasoning
- Fewer hallucinations
- Video native support
- Higher resolution handling
- More modalities (audio?)

---

## Key Takeaways

1. **Multimodal is the future** - Text-only is limiting
2. **GPT-4's reasoning + vision** - Powerful combination
3. **Real-world utility** - Accessibility, education, productivity
4. **Safety-first approach** - Refuses risky capabilities
5. **Sparked competition** - Every major lab now multimodal

**Bottom line:** GPT-4V brought GPT-4's reasoning to images, creating powerful new applications and defining the multimodal AI era.

---

## Further Reading

### Official Resources
- **System Card:** https://cdn.openai.com/papers/GPTV_System_Card.pdf
- **API Documentation:** https://platform.openai.com/docs/guides/vision
- **Safety Evaluations:** In system card

### Related Papers
- **CLIP (foundation):** https://arxiv.org/abs/2103.00020
- **LLaVA (open alternative):** https://arxiv.org/abs/2304.08485
- **Flamingo (similar approach):** https://arxiv.org/abs/2204.14198

### Comparisons
- **Gemini Vision:** Google's technical report
- **Claude 3 Vision:** Anthropic's announcement
- **Open alternatives:** LLaVA, CogVLM

### Use Cases
- **Be My Eyes partnership:** Blog post
- **Educational applications:** Case studies
- **Developer showcase:** Community examples

---

**Published:** September 2023
**Impact:** 🔥🔥🔥🔥🔥 **CRITICAL** - Defined multimodal LLM era
**Adoption:** Millions of users (ChatGPT Plus, API)
**Current Relevance:** State-of-the-art multimodal (competing with Gemini, Claude 3)
**Legacy:** Made multimodal AI mainstream

**Modern Status (2024/2025):** GPT-4V is still among the best vision-language models, though Gemini 1.5 and Claude 3 Opus are competitive. Multimodal is now table stakes for frontier models. GPT-4V showed the way.

**The Impact:** Every AI assistant is now expected to handle images. GPT-4V made that the new normal.
