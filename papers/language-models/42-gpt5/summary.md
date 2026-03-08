# GPT-5: Unified Intelligence

**Authors:** OpenAI
**Published:** August 7, 2025
**System Card:** [cdn.openai.com/gpt-5-system-card.pdf](https://cdn.openai.com/gpt-5-system-card.pdf)
**Announcement:** [openai.com/index/introducing-gpt-5/](https://openai.com/index/introducing-gpt-5/)

---

## Why This Matters

GPT-5 is **OpenAI's unified model that merged the GPT and o-series lines**:

- 🧠 **Unified architecture** - One model that routes between fast and deep reasoning
- 🎯 **94.6% on AIME 2025** - State-of-the-art math without tools
- 💻 **74.9% SWE-bench Verified** - Near-human software engineering
- 🔍 **80% fewer hallucinations** - Dramatic factual accuracy improvement
- ⚡ **50-80% fewer tokens** - More efficient than o3 at same quality

**Real-world impact:**
- Ended the confusing GPT-4/o1/o3 product split
- Set new benchmarks across math, coding, and multimodal
- Reduced hallucination to lowest levels ever measured
- Became the model everything else is benchmarked against in 2025

**The insight:** **Don't make users choose between fast and smart.** Build one model with a router that automatically decides when to think deeply and when to respond quickly.

---

## The Breakthrough

### Unifying GPT and o-Series

**The problem before GPT-5:**
```
GPT-4o: Fast, good at general tasks, bad at hard reasoning
o1/o3:  Slow, excellent at reasoning, expensive, overkill for simple tasks

Users had to choose: "Is this question hard enough for o3?"
Wrong choice = wasted money or bad answers
```

**GPT-5's solution:**
```
Single model with internal router:
  Simple question → Fast path (GPT-level speed)
  Complex question → Deep thinking (o-series reasoning)
  Ambiguous       → Router decides based on complexity

User experience: Just ask. The model figures out how hard to think.
```

### Architecture

```
GPT-5 internal structure:

User query → Router
              ├→ Fast model (most queries, instant response)
              └→ Thinking model (hard problems, extended reasoning)

The router considers:
- Conversation type
- Problem complexity
- Tool requirements
- User's explicit intent ("think carefully about this")
```

---

## Performance

### Mathematics

**AIME 2025 (Competition Math):**

| Model | Score |
|-------|-------|
| GPT-4o | ~20% |
| Claude 3.5 Sonnet | ~25% |
| OpenAI o3 | 88.9% |
| **GPT-5** | **94.6%** |
| **GPT-5 (with tools)** | **96.4%** |

**State-of-the-art math reasoning.**

### Coding

**SWE-bench Verified (Real-world software engineering):**

| Model | Score |
|-------|-------|
| GPT-4o | 38.4% |
| Claude 3.5 Sonnet | 49.0% |
| OpenAI o3 | 69.1% |
| **GPT-5** | **74.9%** |

**Aider Polyglot (Multi-language coding):**

| Model | Score |
|-------|-------|
| GPT-4o | ~65% |
| Claude 3.5 Sonnet | ~70% |
| **GPT-5** | **88.0%** |

### Multimodal

**MMMU (Multimodal Understanding):**

| Model | Score |
|-------|-------|
| GPT-4o | 69.1% |
| Gemini 2.5 Pro | 75.2% |
| **GPT-5** | **84.2%** |

### Hallucination Reduction

**The most impactful improvement:**

```
With web search enabled:
  GPT-5 vs GPT-4o: 45% fewer factual errors

With thinking enabled:
  GPT-5 vs o3: 80% fewer factual errors

This is the biggest single-generation reduction
in hallucination rate ever measured.
```

### Efficiency

**GPT-5 thinking vs o3:**
```
Same or better quality
50-80% fewer output tokens
= Much cheaper to run
= Much faster responses

GPT-5 learned when NOT to think deeply
```

---

## Key Features

### Adaptive Thinking

```
User: "What's the capital of France?"
GPT-5: [Fast path] "Paris."
(No thinking needed, instant response)

User: "Prove that the square root of 2 is irrational."
GPT-5: [Thinking path]
<thinking>
I'll use proof by contradiction...
Assume sqrt(2) = p/q where p,q are integers with no common factors...
Then 2 = p²/q², so p² = 2q²...
This means p² is even, so p is even...
Let p = 2k, then 4k² = 2q², so q² = 2k²...
This means q is also even, contradicting our assumption.
</thinking>
"Here's a proof by contradiction..."
```

### Tool Use

**GPT-5 can use tools when helpful:**
```
- Web search (for current information)
- Code execution (for computation)
- File analysis (for documents)
- Image generation (via DALL-E)

The model decides WHEN to use tools
No explicit instruction needed
```

### Health and Safety

```
HealthBench Hard: 46.2%
(Evaluates medical question answering)

GPT-5 shows improved performance on sensitive domains
Better at acknowledging uncertainty
More careful with medical/legal/financial advice
```

---

## GPT-5.2 (Late 2025)

**Incremental improvement released months later:**

```
Key improvements:
- Better reasoning efficiency
- Improved code generation
- Enhanced multilingual capabilities
- Faster inference
- Lower cost

GPT-5.2 became the default ChatGPT model
```

---

## Practical Usage

### API Access

```python
from openai import OpenAI

client = OpenAI()

# GPT-5 automatically routes between fast and thinking modes
response = client.chat.completions.create(
    model="gpt-5",
    messages=[
        {"role": "user", "content": "Solve this optimization problem..."}
    ]
)

print(response.choices[0].message.content)
```

### With Thinking Control

```python
# Force deep thinking
response = client.chat.completions.create(
    model="gpt-5",
    messages=[
        {"role": "user", "content": "Think carefully: prove Fermat's Last Theorem for n=3"}
    ],
    reasoning_effort="high"  # Force deep thinking
)

# Quick response
response = client.chat.completions.create(
    model="gpt-5",
    messages=[
        {"role": "user", "content": "What's 2+2?"}
    ],
    reasoning_effort="low"  # Skip deep thinking
)
```

### Pricing

```
GPT-5:
  Input: ~$10/1M tokens
  Output: ~$30/1M tokens
  Thinking tokens: Varies

GPT-5 mini:
  Significantly cheaper
  Good for most tasks
```

---

## Impact on the Field

### Ended the Model Confusion

```
Before GPT-5:
  "Should I use GPT-4o or o1 or o3 or o3-mini?"
  Confusing product lineup

After GPT-5:
  "Just use GPT-5"
  One model, automatic routing
```

### New Benchmark Standard

```
GPT-5 became the new bar:
- Every new model compared to GPT-5
- Claude 4.5 Opus, Gemini 3, DeepSeek targeted GPT-5 level
- SWE-bench Verified became the key coding benchmark
```

### Hallucination Progress

```
The 80% reduction in hallucination (vs o3) was
the most significant safety improvement in a generation.

Showed that reasoning + retrieval can dramatically
reduce factual errors.
```

---

## Limitations

### 1. Still Hallucinates
```
80% fewer errors is impressive but not zero
Still can't be fully trusted for critical decisions
Verification still required
```

### 2. Expensive for Thinking
```
When GPT-5 decides to think deeply:
- Uses many more tokens
- Significantly more expensive
- Slower response time
- Not always necessary
```

### 3. Closed Source
```
No weights, no architecture details
Can't fine-tune (only through OpenAI's API)
Dependent on OpenAI's infrastructure and pricing
```

### 4. Router Imperfect
```
Sometimes thinks too deeply on simple questions
Sometimes doesn't think enough on hard questions
Users can override, but default routing isn't perfect
```

---

## Comparison with Competitors

### GPT-5 vs Claude Opus 4.5 vs Gemini 3

| Aspect | GPT-5 | Claude Opus 4.5 | Gemini 3 Pro |
|--------|-------|-----------------|-------------|
| **SWE-bench** | 74.9% | **80.9%** | 76.2% |
| **AIME 2025** | **94.6%** | ~85% | ~88% |
| **MMMU** | **84.2%** | ~78% | ~80% |
| **Hallucination** | **Lowest** | Low | Low |
| **Agentic tasks** | Good | **Best** | Good |
| **Context** | 1M+ | 200K | 1M+ |
| **Open source** | No | No | No |

**Each model leads in different areas.**

---

## Key Takeaways

1. **Unified model** - Merged fast (GPT) and deep (o-series) into one system
2. **Adaptive routing** - Automatically decides how hard to think
3. **94.6% AIME** - State-of-the-art mathematical reasoning
4. **80% fewer hallucinations** - Biggest factual accuracy improvement ever
5. **50-80% more efficient** - Better than o3 with far fewer tokens

**Bottom line:** GPT-5 ended the era of choosing between "fast" and "smart" models. Its unified architecture with adaptive routing set a new standard for what frontier AI should look like - one model that knows when to think quickly and when to think deeply.

---

## Further Reading

### Official Resources
- **Announcement:** https://openai.com/index/introducing-gpt-5/
- **System Card:** https://cdn.openai.com/gpt-5-system-card.pdf
- **Developer Guide:** https://openai.com/index/introducing-gpt-5-for-developers/
- **GPT-5.2:** https://openai.com/index/introducing-gpt-5-2/

### Related Work
- **GPT-4 Technical Report:** https://arxiv.org/abs/2303.08774
- **OpenAI o1:** https://openai.com/index/learning-to-reason-with-llms/
- **GPT-4o:** https://openai.com/index/hello-gpt-4o/

---

**Published:** August 7, 2025
**Impact:** 🔥🔥🔥🔥🔥 **CRITICAL** - Unified frontier model, set new benchmarks
**Adoption:** Massive - default ChatGPT model, primary API model
**Current Relevance:** Current frontier model (with GPT-5.2/5.4 updates)
**Legacy:** Merged reasoning and general intelligence into one system

**Modern Status (March 2026):** GPT-5 remains OpenAI's flagship, with GPT-5.2 and GPT-5.4 incremental updates. It faces strong competition from Claude Opus 4.5/4.6 (leading in coding/agentic tasks) and Gemini 3 (leading in multimodal). The unified fast/thinking architecture has been widely adopted.
