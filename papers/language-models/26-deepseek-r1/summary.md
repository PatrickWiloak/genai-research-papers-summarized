# DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning

**Authors:** DeepSeek-AI
**Published:** January 20, 2025
**Paper:** [arxiv.org/abs/2501.12948](https://arxiv.org/abs/2501.12948)

---

## Why This Matters

DeepSeek-R1 is **the most important open-source breakthrough of 2025**:

- 🔥 **Matches OpenAI o1** - First open model to reach o1-level reasoning
- 💡 **Pure RL discovery** - Reasoning emerges from RL alone (no human reasoning data!)
- 🚀 **Fully open source** - All model weights released (1.5B to 671B)
- 💰 **Incredibly efficient** - Trained for <$6M (vs $100M+ for competitors)
- 🌍 **Game changer** - Proves world-class AI doesn't require massive budgets

**Real-world impact:**
- Democratized reasoning AI (anyone can run/fine-tune)
- Sparked open reasoning model ecosystem (Kimi k1.5, QwQ, etc.)
- Proved RL > supervised learning for reasoning
- Shifted industry from "bigger is better" to "smarter is better"

**The insight:** **Reasoning capabilities emerge naturally from reinforcement learning** - you don't need to show the model how to reason, just reward it for correct answers.

---

## The Breakthrough

### DeepSeek-R1-Zero: Pure RL Magic

**The experiment that shocked everyone:**

```
Take a base LLM (no reasoning ability)
↓
Apply pure RL (no supervised fine-tuning)
↓
AIME 2024: 15.6% → 71.0% (4.5× improvement!)
```

**What emerged spontaneously:**
- ✅ Chain-of-thought reasoning
- ✅ Self-reflection ("wait, let me reconsider...")
- ✅ Self-verification (double-checking answers)
- ✅ Backtracking (undoing wrong steps)

**No human ever showed it how to do these things!**

### The Problem with R1-Zero

Despite amazing reasoning, R1-Zero had issues:
- ❌ Poor readability (messy thoughts)
- ❌ Language mixing (random Chinese in English responses)
- ❌ Format inconsistency

### DeepSeek-R1: The Production Model

**Solution:** Add minimal supervised fine-tuning before RL

**Training pipeline:**
```
1. Base model (DeepSeek-V3)
2. SFT on small amount of reasoning data (~1000 examples)
3. RL with Group Relative Policy Optimization (GRPO)
4. Result: Clean, reliable, world-class reasoning
```

---

## Performance

### Mathematical Reasoning

**AIME 2024 (American Invitational Mathematics Examination):**

| Model | Pass@1 | With Majority Voting |
|-------|--------|---------------------|
| GPT-4o | 9.3% | - |
| Claude 3.5 Sonnet | 16.0% | - |
| DeepSeek-V3 | 39.2% | - |
| **DeepSeek-R1-Zero** | **71.0%** | **86.7%** |
| **DeepSeek-R1** | **79.8%** | **90.0%** |
| OpenAI o1-1217 | 79.2% | - |

**R1 matches o1!**

**MATH-500 (Competition Math):**

| Model | Accuracy |
|-------|----------|
| GPT-4o | 76.6% |
| Claude 3.5 Sonnet | 78.3% |
| Gemini 2.0 Flash | 86.9% |
| **DeepSeek-R1** | **97.3%** |
| OpenAI o1 | 96.4% |

**R1 beats o1!**

### Coding

**Codeforces Rating:**

| Model | Rating |
|-------|--------|
| GPT-4o | 808 |
| Claude 3.5 Sonnet | 1,169 |
| **DeepSeek-R1** | **2,029** |
| OpenAI o1 | 1,891 |

**R1 achieves 96th percentile of human competitive programmers!**

**LiveCodeBench (Real-world Coding):**

| Model | Pass@1 |
|-------|--------|
| GPT-4o | 37.5% |
| Claude 3.5 Sonnet | 49.3% |
| **DeepSeek-R1** | **66.8%** |
| OpenAI o1 | 58.7% |

### Knowledge & Reasoning

**GPQA Diamond (Expert-level Science):**

| Model | Accuracy |
|-------|----------|
| GPT-4o | 49.9% |
| Claude 3.5 Sonnet | 59.4% |
| **DeepSeek-R1** | **71.5%** |
| OpenAI o1 | 78.3% |

**MMLU-Pro (Professional Knowledge):**

| Model | Accuracy |
|-------|----------|
| GPT-4o | 72.6% |
| Claude 3.5 Sonnet | 78.0% |
| **DeepSeek-R1** | **81.2%** |
| OpenAI o1 | 85.5% |

---

## How It Works

### Group Relative Policy Optimization (GRPO)

**DeepSeek's custom RL algorithm:**

**Standard RLHF (e.g., PPO):**
```
1. Train reward model on human preferences
2. Use reward model to score outputs
3. Update policy to maximize reward
Problem: Reward model can be inaccurate
```

**GRPO (DeepSeek's approach):**
```
1. Sample multiple outputs for each question
2. Get ground-truth correctness (binary: right/wrong)
3. Compare outputs in groups (relative ranking)
4. Update policy based on which outputs are better
Advantage: No reward model needed, more stable
```

**Algorithm:**
```python
# Simplified GRPO
for batch in dataset:
    # Generate K responses per question
    responses = [model.generate(question) for _ in range(K)]

    # Score each response (0 or 1 for correctness)
    scores = [verify_answer(r, ground_truth) for r in responses]

    # Compute advantage (how much better than average)
    baseline = mean(scores)
    advantages = [score - baseline for score in scores]

    # Update policy
    for response, advantage in zip(responses, advantages):
        if advantage > 0:
            loss = -log_prob(response) * advantage
            loss.backward()

    optimizer.step()
```

**Key insight:** Comparing outputs is easier than absolute scoring

### Emergent Reasoning Patterns

**Self-reflection:**
```
Question: What is 2+2×2?

R1's reasoning:
"Let me compute this step by step.
2 + 2 × 2
Following order of operations (PEMDAS):
First multiply: 2 × 2 = 4
Then add: 2 + 4 = 6

Wait, let me double-check this...
Multiplication first: 2 × 2 = 4
Then addition: 2 + 4 = 6
Yes, the answer is 6."
```

**Self-verification:**
```
"Let me verify: if 2+2×2 = 6, then working backwards:
6 - 2 = 4
4 / 2 = 2 ✓
This confirms my answer."
```

**Backtracking:**
```
"Hmm, my approach of solving X first doesn't seem to work.
Let me try a different method: solve for Y instead.
[proceeds with alternative approach]
This works! The answer is..."
```

**All emerged from RL without being explicitly taught!**

---

## Distilled Models

**DeepSeek released 6 distilled models** - knowledge transferred from R1 to smaller models:

| Model | Base | Parameters | AIME 2024 | MATH-500 |
|-------|------|------------|-----------|----------|
| R1-Distill-Qwen-1.5B | Qwen2.5-1.5B | 1.5B | 26.7% | 75.6% |
| R1-Distill-Qwen-7B | Qwen2.5-7B | 7B | 49.5% | 87.1% |
| R1-Distill-Qwen-14B | Qwen2.5-14B | 14B | 58.7% | 90.0% |
| R1-Distill-Qwen-32B | Qwen2.5-32B | 32B | 67.9% | 92.8% |
| R1-Distill-Llama-8B | Llama-3.3-8B | 8B | 48.1% | 85.5% |
| R1-Distill-Llama-70B | Llama-3.3-70B | 70B | 69.4% | 93.9% |

**Tiny 1.5B model achieves 75.6% on MATH-500!** (GPT-4o gets 76.6%)

**How distillation works:**
```
1. Generate reasoning traces with DeepSeek-R1
2. Fine-tune smaller model to mimic R1's reasoning
3. Result: Small model inherits reasoning capability
```

---

## Technical Details

### Architecture

**Base model:** DeepSeek-V3
- 671B total parameters
- 37B active parameters (MoE)
- 128K context length

**R1 additions:**
- No architecture changes!
- Just RL training on top of V3

### Training

**Phase 1: Cold start data**
- ~1000 high-quality reasoning examples
- Teaches basic format and style
- Prevents language mixing

**Phase 2: Reinforcement Learning**
- Millions of math/code problems
- GRPO algorithm
- Reward: correctness (binary)
- ~6 weeks on 2048 H800 GPUs

**Training cost:** <$6 million (vs $100M+ for competitors)

### Data

**RL training data:**
- Mathematics: competition problems, theorem proving
- Coding: programming contests, algorithm challenges
- Science: physics, chemistry, biology problems
- Logic: puzzles, reasoning tasks

**No reasoning traces in training data** - model learns to reason on its own!

---

## Practical Usage

### Using DeepSeek-R1

```python
from openai import OpenAI

# DeepSeek provides OpenAI-compatible API
client = OpenAI(
    api_key="your-deepseek-api-key",
    base_url="https://api.deepseek.com"
)

# Use R1 for reasoning tasks
response = client.chat.completions.create(
    model="deepseek-reasoner",
    messages=[
        {"role": "user", "content": "Solve: If x^2 + 5x + 6 = 0, what is x?"}
    ]
)

print(response.choices[0].message.content)
```

**Output:**
```
Let me solve this quadratic equation step by step.

Given: x^2 + 5x + 6 = 0

I'll use factoring:
x^2 + 5x + 6 = (x + 2)(x + 3) = 0

This means either:
x + 2 = 0  →  x = -2
OR
x + 3 = 0  →  x = -3

Let me verify:
For x = -2: (-2)^2 + 5(-2) + 6 = 4 - 10 + 6 = 0 ✓
For x = -3: (-3)^2 + 5(-3) + 6 = 9 - 15 + 6 = 0 ✓

Therefore, x = -2 or x = -3
```

### Local Deployment

```python
# Using Hugging Face Transformers
from transformers import AutoTokenizer, AutoModelForCausalLM

# Load distilled model (smaller, faster)
model_name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",
    torch_dtype="auto"
)

# Generate with reasoning
prompt = "What is the derivative of x^3 + 2x^2 - 5x + 1?"
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

outputs = model.generate(
    **inputs,
    max_new_tokens=2048,
    temperature=0.6,
    do_sample=True
)

response = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(response)
```

### Using with vLLM (Fast Inference)

```bash
# Install vLLM
pip install vllm

# Run inference server
python -m vllm.entrypoints.openai.api_server \
    --model deepseek-ai/DeepSeek-R1-Distill-Llama-70B \
    --tensor-parallel-size 4

# Use OpenAI-compatible API
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "deepseek-ai/DeepSeek-R1-Distill-Llama-70B",
    "messages": [{"role": "user", "content": "Solve: 2x + 5 = 13"}]
  }'
```

---

## Impact on the Field

### Democratization of Reasoning AI

**Before R1:**
- Reasoning AI = OpenAI o1 only
- Closed source, expensive API
- $15-$60 per million tokens

**After R1:**
- Multiple open reasoning models
- Free to run locally
- Fine-tuneable for specific domains
- $0.55 per million tokens (DeepSeek API)

### Proved RL > Supervised Learning

**Traditional approach:**
```
Collect human reasoning examples (expensive)
↓
Fine-tune model on examples (limited by data)
↓
Model mimics human reasoning (ceiling = human performance)
```

**R1 approach:**
```
Just provide problems + answers
↓
Let RL discover reasoning strategies (unlimited exploration)
↓
Model surpasses human reasoning patterns
```

### Sparked Open Reasoning Ecosystem

**Models inspired by/building on R1:**
- **Kimi k1.5:** Alibaba's reasoning model
- **QwQ-32B:** Qwen team's reasoning model
- **Skywork-o1-Open:** Chinese open-source reasoning
- **Many fine-tunes:** Domain-specific reasoning models

---

## Comparison with Competitors

### R1 vs OpenAI o1

| Aspect | OpenAI o1 | DeepSeek-R1 |
|--------|-----------|-------------|
| **Performance** | SOTA (Dec 2024) | Matches o1 (Jan 2025) |
| **Availability** | API only | Fully open source |
| **Cost** | $15-60/1M tokens | $0.55/1M tokens (API) or free (self-host) |
| **Size** | Unknown | 671B total, 37B active |
| **Training cost** | $100M+ (estimated) | <$6M |
| **Fine-tuning** | Not possible | Fully possible |

### R1 vs Claude 3.5 Sonnet

| Aspect | Claude 3.5 | DeepSeek-R1 |
|--------|------------|-------------|
| **Math (AIME)** | 16.0% | **79.8%** |
| **Coding (Codeforces)** | 1,169 | **2,029** |
| **General reasoning** | Excellent | Excellent |
| **Creative writing** | **Better** | Good |
| **Instruction following** | **Better** | Good |

**Claude still better for:** Writing, chat, general assistance
**R1 dominates:** Math, coding, logical reasoning

### R1 vs Gemini 2.0

| Aspect | Gemini 2.0 Flash | DeepSeek-R1 |
|--------|------------------|-------------|
| **Math** | 86.9% (MATH-500) | **97.3%** |
| **Multimodal** | **Native** | Text only |
| **Speed** | **Faster** | Slower (long reasoning) |
| **Cost** | Low | **Lower** |

---

## Limitations

### 1. **Verbose Reasoning**
```
Problem: R1 generates long reasoning traces
Impact: Slower response time, higher token cost
Mitigation: Use distilled models, or hide reasoning from user
```

### 2. **Not Always Needed**
```
Simple question: "What is the capital of France?"
R1: [500 tokens of reasoning] → "Paris"
GPT-4: "Paris" (immediate)

Use R1 only for complex reasoning tasks
```

### 3. **Text-Only**
```
R1 doesn't support images, audio, video
For multimodal: Use Gemini/GPT-4V/Claude
```

### 4. **Hallucination in Reasoning**
```
R1 can confidently reason to wrong conclusion
More reasoning ≠ always correct
Always verify critical outputs
```

### 5. **Formatting Issues (R1-Zero)**
```
R1-Zero still has language mixing
Use full R1 or distilled models for production
```

---

## When to Use DeepSeek-R1

### ✅ Perfect For

**1. Mathematics**
- Competition math
- Theorem proving
- Complex calculations
- Step-by-step solutions

**2. Coding**
- Algorithmic problems
- Debugging complex code
- Optimization challenges
- Code reasoning/explanation

**3. Logic Puzzles**
- Brain teasers
- Constraint satisfaction
- Planning problems
- Strategic reasoning

**4. Scientific Reasoning**
- Physics problems
- Chemistry calculations
- Biology analysis
- Multi-step scientific questions

### ❌ Not Ideal For

**1. Simple Queries**
- Factual lookup ("Who is X?")
- Simple definitions
- Basic conversational responses

**2. Creative Tasks**
- Story writing
- Poetry
- Marketing copy
- Brainstorming

**3. Multimodal Tasks**
- Image understanding
- Video analysis
- Audio processing

**4. Real-time Applications**
- Chatbots (too slow)
- Low-latency systems
- Quick responses needed

---

## Key Innovations

### 1. **Pure RL Reasoning**
First demonstration that reasoning emerges from RL without supervised reasoning data

### 2. **GRPO Algorithm**
More stable and efficient than PPO for LLM training

### 3. **Massive Efficiency**
World-class model for <$6M training cost

### 4. **Open Source Everything**
Full model weights, code, training details

### 5. **Successful Distillation**
Small models (1.5B) achieve GPT-4 level reasoning

---

## Key Takeaways

1. **RL discovers reasoning** - Don't need human reasoning examples
2. **Matches o1 performance** - First open model to reach this level
3. **Incredibly efficient** - 10-20× cheaper to train than competitors
4. **Fully open source** - Democratizes reasoning AI
5. **Distillation works** - Small models inherit reasoning capability

**Bottom line:** DeepSeek-R1 is the most important open-source AI breakthrough of 2025. It proved that world-class reasoning AI can be built efficiently and openly, changing the economics and accessibility of advanced AI forever.

---

## Further Reading

### Original Paper
- **DeepSeek-R1:** https://arxiv.org/abs/2501.12948

### Model Downloads
- **DeepSeek-R1:** https://huggingface.co/deepseek-ai/DeepSeek-R1
- **Distilled Models:** https://huggingface.co/collections/deepseek-ai/deepseek-r1-distill
- **API:** https://platform.deepseek.com/

### Analysis
- **Nature Coverage:** https://www.nature.com/articles/s41586-025-09422-z
- **Technical Analysis:** AI Papers Academy
- **Community Discussion:** Hugging Face Papers

### Related Work
- **OpenAI o1:** System card (limited details)
- **DeepSeek-V3:** https://arxiv.org/abs/2412.19437
- **GRPO Algorithm:** In R1 paper appendix

---

**Published:** January 20, 2025
**Impact:** 🔥🔥🔥🔥🔥 **CRITICAL** - Democratized reasoning AI
**Citations:** 300+ (in 9 months)
**Adoption:** Massive - spawned entire reasoning model ecosystem
**Current Relevance:** Industry standard for open reasoning models
**Legacy:** Proved advanced AI doesn't require massive budgets

**Modern Status (October 2025):** DeepSeek-R1 remains the gold standard for open-source reasoning models. Sparked a wave of innovation in RL-based training and efficient model development. The 2025 "ChatGPT moment" for reasoning AI.
