# OpenAI o1: Learning to Reason with Reinforcement Learning

**Authors:** OpenAI
**Published:** September 12, 2024
**Announcement:** [openai.com/index/learning-to-reason-with-llms](https://openai.com/index/learning-to-reason-with-llms/)

---

## Why This Matters

OpenAI o1 **started the reasoning model era**:

- 🧠 **Thinks before answering** - Extended chain-of-thought
- 🎯 **PhD-level performance** - 89th percentile on Codeforces
- 🏆 **Breakthrough on hard problems** - 83% on AIME (vs 13% GPT-4o)
- ⏱️ **Spends time thinking** - Uses "reasoning tokens"
- 🔒 **Reasoning mostly hidden** - Privacy/competitive reasons

**Real-world impact:**
- Proved reasoning > scale for hard problems
- Sparked reasoning model race (DeepSeek-R1, Qwen3, etc.)
- Changed AI evaluation (need harder benchmarks)
- Showed RL works for reasoning

**The insight:** **Let the model think longer** - performance scales with reasoning time, not just model size.

---

## How o1 Works

### Reinforcement Learning for Reasoning

**Traditional approach:**
```
Train on examples: Q: "2+2" A: "4"
Model learns patterns
Limited by training data quality
```

**o1 approach:**
```
Train to explore reasoning strategies
Reward correct final answers
Learn what thinking patterns work
Can solve novel problems
```

**The chain-of-thought:**
```
Problem: "Find x: x² + 5x + 6 = 0"

GPT-4o: "x = -2 or x = -3"

o1: [Thinking for 30 seconds]
"Let me think through this step by step.
I'll try factoring: (x+2)(x+3) = 0
This gives x = -2 or x = -3
Let me verify: (-2)² + 5(-2) + 6 = 4 - 10 + 6 = 0 ✓
And (-3)² + 5(-3) + 6 = 9 - 15 + 6 = 0 ✓
Therefore x = -2 or x = -3"
```

---

## Performance

### Mathematical Reasoning

**AIME (American Invitational Mathematics Examination):**

| Model | Pass@1 |
|-------|--------|
| GPT-4o | 13.4% |
| **o1-preview** | **83.3%** |
| **o1-mini** | **70.0%** |

**6× better than GPT-4o!**

**IMO (International Mathematics Olympiad):**
- o1: 49% problems solved
- Previous models: <10%

### Coding Competitions

**Codeforces:**

| Model | Percentile |
|-------|------------|
| GPT-4o | 11th percentile |
| **o1** | **89th percentile** |

**o1 beats 89% of competitive programmers!**

**USACO (USA Computing Olympiad):**
- o1: Gold division level
- GPT-4: Bronze/Silver

### Science (GPQA Diamond)

| Model | Accuracy |
|-------|----------|
| GPT-4o | 49.9% |
| **o1-preview** | **78.3%** |
| PhD experts | ~70% |

**o1 exceeds human PhD performance!**

---

## Technical Details

### Reasoning Tokens

**How it works:**
```
User asks question
    ↓
o1 generates internal "thinking"
(Hidden from user, thousands of tokens)
    ↓
Refines answer through reasoning
    ↓
Returns final answer
```

**Why hidden:**
- Competitive advantage
- Privacy (may contain sensitive patterns)
- User experience (too verbose)

### Training Approach

**Reinforcement learning:**
```
1. Model explores different reasoning approaches
2. Gets reward for correct final answers
3. Learns which reasoning patterns work
4. Improves over time (like AlphaGo)

Not supervised on reasoning examples!
Discovers strategies on its own
```

**Chain-of-thought:**
```
Encouraged to:
- Break down problems
- Try multiple approaches
- Self-correct mistakes
- Verify answers
```

---

## Practical Usage

### Using o1 via API

```python
from openai import OpenAI

client = OpenAI(api_key="your-api-key")

# o1-preview (most capable, slower, expensive)
response = client.chat.completions.create(
    model="o1-preview",
    messages=[
        {
            "role": "user",
            "content": "Solve: If f(x) = x³ - 6x² + 11x - 6, find all zeros"
        }
    ]
)

print(response.choices[0].message.content)

# o1-mini (faster, cheaper, still good at reasoning)
response = client.chat.completions.create(
    model="o1-mini",
    messages=[
        {
            "role": "user",
            "content": "Write a Python function to find longest palindromic substring"
        }
    ]
)
```

### Best Practices

**Good use cases:**
```python
# Complex math
"Prove that √2 is irrational"

# Hard coding problems
"Implement A* pathfinding with custom heuristic"

# Scientific reasoning
"Explain why superconductors work at quantum level"

# Multi-step logic
"Design database schema for complex social network"
```

**Bad use cases (use GPT-4o instead):**
```python
# Simple questions
"What is the capital of France?"  # Overkill, expensive

# Creative writing
"Write a poem about autumn"  # GPT-4o is better

# General chat
"Tell me about your day"  # Not worth the cost

# Real-time applications
"Quick answer needed"  # Too slow (thinking takes time)
```

---

## Limitations

### 1. **Slow**
```
Thinking time: 10-60+ seconds
GPT-4o: <1 second
Trade-off: Quality vs speed
```

### 2. **Expensive**
```
Pricing (per 1M tokens):
- Input: $15 (o1-preview), $3 (o1-mini)
- Output: $60 (o1-preview), $12 (o1-mini)

vs GPT-4o:
- Input: $2.50
- Output: $10

o1 is 6× more expensive!
```

### 3. **Hidden Reasoning**
```
Can't see how it thinks
Harder to debug
Less transparent than DeepSeek-R1
```

### 4. **No Streaming**
```
Must wait for complete answer
Can't see partial progress
Frustrating for long answers
```

### 5. **Limited Multimodal**
```
Text-only for reasoning
No vision (use GPT-4o for images)
```

---

## Comparison with Successors

### o1 vs DeepSeek-R1 (Jan 2025)

| Aspect | OpenAI o1 | DeepSeek-R1 |
|--------|-----------|-------------|
| **Performance** | Excellent | Matches o1 |
| **Open source** | No | **Yes** |
| **Reasoning visible** | No | **Yes** |
| **Cost** | High | **Very low** |
| **Release** | Sept 2024 | Jan 2025 (4 months later) |

DeepSeek-R1 caught up and surpassed in accessibility!

### o1 vs Gemini 2.5 Deep Think (Jul 2025)

| Aspect | o1 | Gemini 2.5 |
|--------|-----|------------|
| **Reasoning** | Dedicated | **Integrated** |
| **Multimodal** | No | **Yes** |
| **Context** | 128K | **1M** |
| **Adaptive** | No (always thinks) | **Yes** (thinks when needed) |

Gemini improved on o1's concept!

---

## Impact on the Field

### Sparked Reasoning Model Race

**Timeline:**
```
Sept 2024: OpenAI o1 released
Oct 2024: Claude 3.5 Sonnet improvements
Dec 2024: DeepSeek-V3 (foundation)
Jan 2025: DeepSeek-R1 (matches o1, open source)
Mar 2025: Gemini 2.5 (integrated thinking)
May 2025: Qwen3 (adaptive thinking)

o1 started it all!
```

### Changed Evaluation

**Before o1:**
```
MMLU, HumanEval, etc.
Models saturating benchmarks
Need harder tests
```

**After o1:**
```
Focus on:
- AIME (olympiad math)
- GPQA (PhD-level science)
- Codeforces (competitive programming)
- IMO, USAMO (hardest problems)
```

### Proved RL > Scaling

**Old paradigm:**
```
Better AI = Bigger models
Just add more parameters
```

**New paradigm (thanks to o1):**
```
Better AI = Smarter training
RL for reasoning
Quality > quantity
```

---

## Key Takeaways

1. **Started reasoning era** - First model to "think" before answering
2. **6× better on hard math** - 83% AIME vs 13% GPT-4o
3. **RL for reasoning** - Learns strategies, not just patterns
4. **Trade-offs** - Slow + expensive, but solves hard problems
5. **Inspired open alternatives** - DeepSeek-R1, Qwen3 followed

**Bottom line:** o1 proved that giving AI time to think unlocks new capabilities, starting the reasoning model revolution that defines AI in 2025.

---

## Further Reading

- **Announcement:** https://openai.com/index/learning-to-reason-with-llms/
- **System Card:** OpenAI documentation
- **API Docs:** https://platform.openai.com/docs/models/o1

**Published:** September 12, 2024
**Impact:** 🔥🔥🔥🔥🔥 **CRITICAL** - Started reasoning model era
**Legacy:** Inspired entire category of AI models
