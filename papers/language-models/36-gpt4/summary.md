# GPT-4 Technical Report

**Authors:** OpenAI
**Published:** March 14, 2023
**Paper:** [arxiv.org/abs/2303.08774](https://arxiv.org/abs/2303.08774)

---

## Why This Matters

GPT-4 was **the model that defined the frontier era**:

- 🔥 **Massive leap over GPT-3.5** - Qualitative and quantitative jump in capability
- 🎓 **Passes the bar exam** - Top 10% of test takers, proving expert-level reasoning
- 🖼️ **First multimodal GPT** - Accepts both text and image inputs
- 📈 **Predictable scaling** - Loss predicted in advance from smaller models
- 🔒 **Most secretive report** - Disclosed almost no architecture or training details

**Real-world impact:**
- Powered ChatGPT Plus, Microsoft Copilot, and hundreds of products
- Set the benchmark that every subsequent model has been measured against
- Demonstrated AI could pass professional exams (bar, SAT, GRE, AP tests)
- Sparked massive industry investment in AI
- Ignited debate about openness - OpenAI released the least "open" report ever

**The insight:** **Scaling continued to deliver massive capability gains**, but OpenAI decided the details were now too dangerous (and too competitively valuable) to share.

---

## The Breakthrough

### From ChatGPT to GPT-4

**The jump was enormous:**

```
GPT-3.5 (ChatGPT) → GPT-4:
- Bar exam: 10th percentile → 90th percentile
- MMLU: 70.0% → 86.4%
- HumanEval (coding): 48.1% → 67.0%
- SAT Math: 70th percentile → 89th percentile
```

**What changed:** Not just better scores - GPT-4 showed qualitatively different reasoning, following complex multi-step instructions, handling nuance, and maintaining coherence over long conversations.

### Professional Exam Performance

GPT-4 was the first AI to convincingly pass professional-grade examinations:

| Exam | GPT-3.5 Percentile | GPT-4 Percentile |
|------|-------------------|------------------|
| Uniform Bar Exam | ~10th | ~90th |
| SAT Math | ~70th | ~89th |
| SAT Reading/Writing | ~87th | ~93rd |
| GRE Quantitative | ~25th | ~80th |
| GRE Verbal | ~63rd | ~99th |
| AP Biology | 2/5 | 5/5 |
| AP Chemistry | 4/5 | 4/5 |
| AP US History | 4/5 | 5/5 |

**Not just memorization** - GPT-4 solved novel problems requiring multi-step reasoning, legal analysis, and scientific deduction.

---

## Performance

### Language Understanding

**MMLU (Massive Multitask Language Understanding):**

| Model | Score |
|-------|-------|
| GPT-3.5 | 70.0% |
| Claude 1 | 75.6% |
| PaLM 2 | 78.3% |
| **GPT-4** | **86.4%** |
| Human expert | ~89.8% |

**Approaching human expert performance.**

**Multilingual MMLU:**
GPT-4 surpassed the English state-of-the-art in **24 of 26 languages tested**, even for low-resource languages like Latvian, Welsh, and Swahili.

### Coding

**HumanEval (Python code generation):**

| Model | Pass@1 |
|-------|--------|
| GPT-3.5 | 48.1% |
| PaLM 2 | 36.6% |
| **GPT-4** | **67.0%** |

### Reasoning

**HellaSwag (Commonsense reasoning):**

| Model | Score |
|-------|-------|
| GPT-3.5 | 85.5% |
| LLaMA 65B | 84.2% |
| **GPT-4** | **95.3%** |

---

## How It Works (What We Know)

### The Secrecy Problem

**What OpenAI disclosed:**
- It's a Transformer-based model
- Pre-trained to predict next tokens
- Fine-tuned with RLHF
- Accepts text and image inputs

**What OpenAI refused to disclose:**
- Model size (parameters)
- Architecture details (layers, heads, dimensions)
- Training data composition
- Training compute
- Hardware used
- Dataset construction methods

**Their justification:**
> "Given both the competitive landscape and the safety implications of large-scale models like GPT-4, this report contains no further details about the architecture (including model size), hardware, training compute, dataset construction, training method, or similar."

**Community reaction was divided:**
- Safety researchers: Supported caution
- Open-source advocates: Called it hypocritical ("Open" AI)
- Competitors: Noted the competitive advantage of secrecy
- Leaked info: Rumored to be ~1.8T parameters, 8-way MoE (unconfirmed)

### RLHF and Safety

**GPT-4 improved safety significantly over GPT-3.5:**
- 82% less likely to respond to disallowed content
- 40% more likely to produce factual responses
- Trained with adversarial "red team" testing (50+ experts)

**Safety training pipeline:**
```
Pre-training (next-token prediction)
↓
Supervised fine-tuning (human demonstrations)
↓
RLHF (reward model from human preferences)
↓
Rule-based reward model (safety-specific)
↓
Final model
```

### Predictable Scaling

**One of the paper's most important contributions:**

OpenAI showed they could predict GPT-4's performance on benchmarks using much smaller models (1000x-10000x less compute), fitted to a scaling law.

```
1. Train small models at various scales
2. Fit power law to performance vs compute
3. Predict large model performance
4. GPT-4 matched predictions closely

Example: MMLU prediction was within 2% of actual
```

**Why this matters:** Organizations can predict if a training run will be worthwhile before spending $100M+.

### Multimodal Capabilities

**Image input (text output):**

GPT-4 could process images alongside text:
```
Input: [Photo of a refrigerator interior]
Question: "What can I make with these ingredients?"

GPT-4: "Based on what I see - eggs, cheese, butter,
milk, and some vegetables - you could make:
1. A vegetable omelette
2. A cheese quiche
3. Scrambled eggs with sauteed vegetables..."
```

**Visual reasoning examples:**
- Explaining why a meme is funny
- Reading charts and graphs
- Solving physics problems from diagrams
- Understanding screenshots and UI elements

**Note:** Image input was initially limited (via ChatGPT Plus) and rolled out gradually.

---

## Technical Details

### Context Length

```
GPT-4: 8,192 tokens
GPT-4-32k: 32,768 tokens (4x longer)
GPT-3.5: 4,096 tokens

Later update (GPT-4 Turbo, Nov 2023): 128,000 tokens
```

### Training

**What we know:**
- Finished training in August 2022 (6+ months before release)
- Extensive red-teaming period
- Used Microsoft Azure infrastructure
- Training data cutoff: September 2021 (later updated)

**Estimated costs:** $50-100M+ (industry estimates)

---

## Practical Usage

### API Access

```python
from openai import OpenAI

client = OpenAI()

# Text-only
response = client.chat.completions.create(
    model="gpt-4",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Explain the Riemann hypothesis in simple terms."}
    ]
)

print(response.choices[0].message.content)
```

### With Image Input

```python
response = client.chat.completions.create(
    model="gpt-4-vision-preview",
    messages=[
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "What's in this image?"},
                {
                    "type": "image_url",
                    "image_url": {"url": "https://example.com/image.jpg"}
                }
            ]
        }
    ]
)
```

---

## Impact on the Field

### Set the Frontier Bar

**Before GPT-4:**
- "AI can pass some tests" (novelty)

**After GPT-4:**
- "AI outperforms most humans on professional exams" (paradigm shift)
- Every new model measured against GPT-4
- Sparked hundreds of billions in AI investment

### Spawned the Catch-Up Race

**Models that explicitly benchmarked against GPT-4:**
- Claude 2/3 (Anthropic)
- Gemini 1.0/1.5 (Google)
- LLaMA 2/3 (Meta) - open-source alternative
- DeepSeek-V3 - matched GPT-4 for $5.76M
- Mistral/Mixtral - efficient alternatives

### Changed How We Think About AI

```
Before GPT-4: "Can AI do X?"
After GPT-4:  "How well can AI do X compared to humans?"
```

---

## Limitations

### 1. Hallucinations
```
Still confidently generates incorrect information
GPT-4 is better than GPT-3.5 (40% improvement)
But still unreliable for factual accuracy
```

### 2. Knowledge Cutoff
```
Training data cutoff: September 2021
No knowledge of events after that date
Addressed later with browsing and retrieval tools
```

### 3. Reasoning Errors
```
Can fail on novel reasoning tasks
Performance drops on problems not in training distribution
Still struggles with some spatial and temporal reasoning
```

### 4. Context Window
```
8K tokens initially (vs 100K+ today)
Struggled with very long documents
Addressed with GPT-4 Turbo (128K) later
```

### 5. No Audio or Video
```
Text and image input only
No audio understanding or generation
No video processing
Addressed later with GPT-4o
```

---

## Comparison with Predecessors and Successors

### GPT-4 in Context

| Aspect | GPT-3.5 | GPT-4 | GPT-4o | GPT-5 |
|--------|---------|-------|--------|-------|
| **MMLU** | 70.0% | 86.4% | 87.2% | ~90%+ |
| **Bar Exam** | ~10th %ile | ~90th %ile | ~90th %ile | - |
| **HumanEval** | 48.1% | 67.0% | 90.2% | - |
| **Multimodal** | Text only | Text + Image | Text + Image + Audio | Full omni |
| **Context** | 4K | 8K/32K | 128K | 1M+ |
| **Open weights** | No | No | No | No |
| **Release** | Nov 2022 | Mar 2023 | May 2024 | Aug 2025 |

---

## Key Takeaways

1. **Massive capability jump** - GPT-4 crossed the threshold into expert-level performance
2. **Professional exam mastery** - Top 10% on bar exam, proving real-world reasoning
3. **Predictable scaling** - Performance can be forecast from smaller models
4. **Multimodal foundation** - First GPT with image understanding
5. **Secrecy precedent** - Set the trend of closed technical reports

**Bottom line:** GPT-4 defined what a frontier model means. Every model since has been measured against it. While it disclosed almost nothing about how it was built, its capabilities changed the industry's understanding of what AI can do.

---

## Further Reading

### Original Paper
- **GPT-4 Technical Report:** https://arxiv.org/abs/2303.08774

### System Card
- **GPT-4 System Card:** https://cdn.openai.com/papers/gpt-4-system-card.pdf

### Analysis
- **OpenAI Blog Post:** https://openai.com/index/gpt-4-research/
- **Sparks of AGI Paper:** https://arxiv.org/abs/2303.12712 (Microsoft's analysis)

### Related Work
- **GPT-3:** https://arxiv.org/abs/2005.14165
- **InstructGPT:** https://arxiv.org/abs/2203.02155
- **GPT-4V System Card:** https://cdn.openai.com/papers/GPTV_System_Card.pdf

---

**Published:** March 14, 2023
**Impact:** 🔥🔥🔥🔥🔥 **CRITICAL** - Defined the frontier model era
**Citations:** 10,000+ (as of 2025)
**Adoption:** Universal - powered ChatGPT Plus, Microsoft Copilot, and thousands of applications
**Current Relevance:** Superseded by GPT-4o and GPT-5, but established the benchmarks everyone uses
**Legacy:** The model that made the world take AI seriously

**Modern Status (March 2026):** GPT-4 has been superseded by GPT-4o, GPT-4.1, and GPT-5, but its impact is permanent. It set the standard for what a frontier model should be capable of and launched the era of AI as a serious professional tool.
