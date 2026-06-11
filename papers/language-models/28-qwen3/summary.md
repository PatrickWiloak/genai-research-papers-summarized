---
title: "Qwen3: Technical Report"
slug: "28-qwen3"
number: 28
category: "language-models"
authors: "Qwen Team (Alibaba Cloud)"
published: "May 14, 2025"
year: 2025
url: "https://arxiv.org/abs/2505.09388"
tags: [language-models]
---

# Qwen3: Technical Report

**Authors:** Qwen Team (Alibaba Cloud)
**Published:** May 14, 2025
**Paper:** [arxiv.org/abs/2505.09388](https://arxiv.org/abs/2505.09388)

---

## Why This Matters

Qwen3 solved the **thinking vs non-thinking dilemma**:

- 🧠 **Unified model** - Thinking AND fast modes in ONE model
- 🎯 **Thinking budget** - User controls computation time
- 🚀 **SOTA performance** - Beats o1/DeepSeek-R1 on many benchmarks
- 📈 **235B flagship** - Largest model with thinking capability
- 🌍 **Fully open source** - Apache 2.0 license

**Real-world impact:**
- Eliminates need for separate chat/reasoning models
- Dynamic mode switching based on query complexity
- Best of both worlds: fast when possible, deep when needed

**The insight:** **One model should adapt** - not make users choose between fast chat model and slow reasoning model.

---

## The Problem Qwen3 Solved

**Before Qwen3:**
```
Simple question: "What's 2+2?"
- GPT-4o: Instant "4"
- o1: [30 seconds of thinking] → "4"

Complex question: "Prove Fermat's Last Theorem"  
- GPT-4o: Quick but wrong
- o1: [Long thinking] → Correct proof

User problem: Which model to use?
```

**Qwen3 solution:**
```
One model automatically:
- Fast mode for "2+2" → Instant
- Thinking mode for theorem → Deep reasoning
- User can override with thinking budget
```

---

## Key Innovations

### 1. Integrated Thinking/Non-Thinking Modes

**Architecture:**
```
Input query
    ↓
Difficulty assessment (learned)
    ↓
    ├─→ Simple? → Fast generation
    └─→ Complex? → Thinking mode

Or user specifies:
- thinking_budget='low' → Fast
- thinking_budget='high' → Deep reasoning
```

### 2. Thinking Budget Mechanism

```python
# User controls compute allocation
response = qwen3.generate(
    "Solve complex physics problem",
    thinking_budget="high"  # low, medium, high
)

# High budget:
- More reasoning tokens
- Multiple solution attempts  
- Self-verification
- Longer latency

# Low budget:
- Direct answer
- Minimal reasoning
- Fast response
```

### 3. Model Sizes

**Dense models:**
- Qwen3-0.6B
- Qwen3-1.8B
- Qwen3-4B
- Qwen3-14B
- Qwen3-32B

**MoE flagship:**
- **Qwen3-235B-A22B** (235B total, 22B active)
- Best performance
- Efficient inference

---

## Benchmarks

### Mathematical Reasoning

**AIME 2024:**

| Model | Score |
|-------|-------|
| GPT-4o | 9.3% |
| DeepSeek-R1 | 79.8% |
| **Qwen3-235B** | **85.7%** |

**AIME 2025:**

| Model | Score |
|-------|-------|
| Gemini 2.5 Pro | SOTA |
| **Qwen3-235B** | **81.5%** |

**MATH-500:**
- Qwen3-32B (thinking): 92.8%
- Qwen3-235B: **96.5%+**

### Coding

**LiveCodeBench v5:**

| Model | Pass@1 |
|-------|--------|
| GPT-4o | 37.5% |
| Claude 3.5 | 49.3% |
| DeepSeek-R1 | 66.8% |
| **Qwen3-235B** | **70.7%** |

**Codeforces:**
- **Rating: 2,056** (Expert level)
- 97th percentile of competitive programmers

### Function Calling

**BFCL v3 (Berkeley Function Calling Leaderboard):**
- **Qwen3-235B: 70.8%**
- Best open-source model for tool use

### Comparison with DeepSeek-V3

**Head-to-head on 15 benchmarks:**
- **Qwen3-235B wins: 14/15**
- Only loss: Specific multilingual task
- Clear SOTA for open models

---

## Technical Details

### Architecture

**MoE Design (235B model):**
```
Total parameters: 235B
Active per token: ~22B
Experts: 64
Top-K routing: 8 experts per token

Efficiency:
- 10× capacity of dense 22B model
- Inference cost of 22B model
```

**Context length:**
- 32K tokens (standard)
- Extended to 128K with rope scaling

### Training

**Phase 1: Pre-training**
- Massive multilingual corpus
- Code, math, scientific data
- Multi-turn conversations

**Phase 2: Post-training**
```
1. Supervised fine-tuning (SFT)
   - High-quality examples
   - Thinking and non-thinking data

2. Reinforcement learning
   - Group Relative Policy Optimization (GRPO)
   - Learns when to think
   - Optimizes thinking strategies

3. Safety alignment
   - Harmlessness training
   - Bias mitigation
```

**Mode switching training:**
```
Model learns to detect:
- Query complexity
- Whether thinking needed
- Optimal thinking budget

Training data includes:
- Simple questions (fast mode target)
- Complex problems (thinking mode target)
- Mixed difficulty (adaptive mode)
```

---

## Practical Usage

### Basic Usage

```python
from transformers import AutoModel Tokenizer, AutoModelForCausalLM

# Load model
model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen3-32B",
    device_map="auto",
    trust_remote_code=True
)
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-32B")

# Simple query (automatic fast mode)
response = model.chat(
    tokenizer,
    "What is the capital of France?"
)
# Fast response: "Paris"

# Complex query (automatic thinking mode)
response = model.chat(
    tokenizer,
    "Prove that there are infinitely many prime numbers"
)
# Engages thinking mode automatically
```

### Thinking Budget Control

```python
# Explicit thinking budget
response = model.chat(
    tokenizer,
    "Solve: x^3 - 6x^2 + 11x - 6 = 0",
    thinking_budget="high"  # Force deep thinking
)

# Low budget for speed
response = model.chat(
    tokenizer,
    "Summarize this text briefly",
    thinking_budget="low"  # Fast mode
)

# Medium for balanced
response = model.chat(
    tokenizer,
    "Debug this code",
    thinking_budget="medium"
)
```

### API Usage

```python
import openai

# Qwen provides OpenAI-compatible API
client = openai.OpenAI(
    api_key="your-qwen-api-key",
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
)

# Automatic mode selection
response = client.chat.completions.create(
    model="qwen3-235b",
    messages=[
        {"role": "user", "content": "Calculate 2+2"}  # Fast
    ]
)

response = client.chat.completions.create(
    model="qwen3-235b",
    messages=[
        {"role": "user", "content": "Solve USAMO problem..."}  # Thinking
    ]
)
```

---

## Comparison with Competitors

### Qwen3 vs DeepSeek-R1

| Aspect | DeepSeek-R1 | Qwen3 |
|--------|-------------|-------|
| **Thinking mode** | Always on | Adaptive |
| **Speed** | Slow (reasoning overhead) | Fast when possible |
| **Performance** | Excellent | Slightly better |
| **Efficiency** | Wastes compute on simple queries | Optimal allocation |
| **User experience** | Always waits | Instant when appropriate |

### Qwen3 vs Gemini 2.5

| Aspect | Gemini 2.5 | Qwen3 |
|--------|------------|-------|
| **Availability** | API only | Open source |
| **Multimodal** | Native | Text only |
| **Context** | 1M tokens | 128K tokens |
| **Math** | SOTA | SOTA (comparable) |
| **Cost** | Higher | Free (self-host) |

### Qwen3 vs o1

| Aspect | OpenAI o1 | Qwen3 |
|--------|-----------|-------|
| **Mode switching** | No (always thinks) | Yes (adaptive) |
| **Open source** | No | Yes |
| **Coding** | Good | Better |
| **Math** | Excellent | Excellent |
| **Practical usability** | Slow for all queries | Fast when appropriate |

---

## Real-World Applications

### 1. Coding Assistant

```python
# Simple syntax question (fast mode)
"How do I reverse a list in Python?"
→ Instant: "my_list.reverse() or my_list[::-1]"

# Complex debugging (thinking mode)
"Why does this recursive algorithm cause stack overflow?"
[Code with subtle bug]
→ Engages deep reasoning, traces execution, finds issue
```

### 2. Educational Tutor

```python
# Fact recall (fast)
"Who wrote Hamlet?"
→ Instant: "William Shakespeare"

# Complex explanation (thinking)
"Explain the themes in Hamlet and how they relate to Elizabethan politics"
→ Deep analysis with historical context
```

### 3. Data Analysis

```python
# Simple stats (fast)
"What's the mean of [1,2,3,4,5]?"
→ Instant: "3"

# Complex analysis (thinking)
"Given this dataset, determine if there's a statistically significant
correlation between X and Y, accounting for confounding variables"
→ Multi-step statistical reasoning
```

---

## Limitations

### 1. **Not Multimodal**
- Text only (no images, audio, video)
- For multimodal: Use Gemini 2.5 or Qwen-VL

### 2. **Smaller Context than Gemini**
- 128K vs Gemini's 1M
- Still large, but not for entire codebases

### 3. **Mode Detection Can Fail**
- Occasionally uses thinking mode when not needed
- Or fast mode when thinking would help
- User can override with thinking_budget

### 4. **Requires Good Hardware**
- 235B model needs significant GPU memory
- Use smaller models (32B, 14B) for consumer hardware

---

## Key Takeaways

1. **Unified adaptive model** - No need for separate chat/reasoning
2. **User-controllable thinking** - Budget mechanism for flexibility
3. **SOTA performance** - Beats competitors on most benchmarks
4. **Fully open source** - Apache 2.0, free to use and modify
5. **Efficient** - Only thinks when needed

**Bottom line:** Qwen3 solved the "when to think" problem by making one model that adapts automatically, delivering the best of both worlds: speed when possible, depth when necessary.

---

## Further Reading

- **Paper:** https://arxiv.org/abs/2505.09388
- **Models:** https://huggingface.co/Qwen
- **Documentation:** https://qwenlm.github.io/
- **API:** https://help.aliyun.com/zh/model-studio/

**Published:** May 14, 2025
**Impact:** 🔥🔥🔥🔥🔥 **CRITICAL** - Unified thinking/non-thinking paradigm
**Adoption:** Massive in open-source community
**Current Relevance:** Leading open-source model (October 2025)

<!-- related:start -->

---

## Related in This Collection

- [DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning](../../language-models/26-deepseek-r1/summary.md)
- [DeepSeek-V3 Technical Report](../../language-models/27-deepseek-v3/summary.md)
- [Gemini 2.5: Pushing the Frontier with Advanced Reasoning, Multimodality, Long Context, and Next Generation Agentic Capabilities](../../multimodal/29-gemini-2.5/summary.md)
- [Claude 3.5 Sonnet: Computer Use and Enhanced Capabilities](../../language-models/30-claude-3.5-sonnet/summary.md)
- [OpenAI o1: Learning to Reason with Reinforcement Learning](../../language-models/31-openai-o1/summary.md)
- [GPT-4 Technical Report](../../language-models/36-gpt4/summary.md)
- [Mixtral of Experts (and the Mixture-of-Experts Architecture)](../../architectures/37-mixture-of-experts/summary.md)
- [GPT-4o: The First Omni Model](../../language-models/40-gpt4o/summary.md)

<!-- related:end -->
