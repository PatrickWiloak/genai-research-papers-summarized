# Speculative Decoding: Fast Inference from Transformers

**Authors:** Yaniv Leviathan, Matan Kalman, Yossi Matias (Google Research)
**Published:** November 2022 (ICML 2023)
**Paper:** [arxiv.org/abs/2211.17192](https://arxiv.org/abs/2211.17192)

---

## Why This Matters

Speculative decoding is **the most important inference optimization technique for LLMs**:

- ⚡ **2-3x faster inference** - Without changing model outputs AT ALL
- 🎯 **Mathematically exact** - Identical distribution to normal decoding
- 🔧 **No retraining needed** - Works with any existing model
- 🏗️ **Universal technique** - Used by OpenAI, Google, Meta, Anthropic, and others
- 💰 **Free speedup** - Same quality, just faster

**Real-world impact:**
- Deployed in every major LLM serving system
- Dramatically reduced latency for end users
- Enabled real-time applications (voice AI, coding assistants)
- Key enabler of the reasoning model era (long outputs need speed)

**The insight:** **Use a small, fast model to "guess" what the large model would say, then have the large model verify multiple tokens in parallel.** Verification is much faster than generation.

---

## The Problem

### Why LLMs Are Slow

**Autoregressive decoding is fundamentally sequential:**

```
Standard LLM generation:
  Token 1 → Token 2 → Token 3 → ... → Token N
  Each token requires a full forward pass through the model
  Can't generate Token 2 until Token 1 is done

For a 70B model generating 500 tokens:
  500 sequential forward passes
  Each takes ~50ms
  Total: ~25 seconds

The GPU is underutilized:
  During generation, the model is memory-bound
  GPU compute sits mostly idle
  Batch size = 1 for each step
```

**The bottleneck is sequential dependency, not compute.**

---

## How Speculative Decoding Works

### The Core Idea

```
Instead of:
  Large model generates 1 token at a time (slow)

Do this:
  1. Small model quickly generates K tokens (fast guesses)
  2. Large model verifies all K tokens in ONE forward pass
  3. Accept correct guesses, reject wrong ones
  4. Always get at least 1 token, often get K tokens
```

### Step by Step

```
Setup:
  Target model: Large (e.g., 70B) - slow but accurate
  Draft model: Small (e.g., 7B) - fast but less accurate

Step 1: Draft
  Small model generates K=5 draft tokens quickly:
  "The cat sat on the [mat] [and] [looked] [at] [the]"
  Time: ~5ms (5 tokens from small model)

Step 2: Verify
  Large model processes ALL draft tokens in ONE forward pass:
  Computes probabilities for each position simultaneously
  Time: ~50ms (one forward pass, but processes 5 positions)

Step 3: Accept/Reject
  For each draft token, compare probabilities:
  Token 1 "mat" - Large model agrees (p_large ≈ p_small) → ACCEPT
  Token 2 "and" - Large model agrees → ACCEPT
  Token 3 "looked" - Large model agrees → ACCEPT
  Token 4 "at" - Large model disagrees → REJECT
  Token 5 "the" - Skipped (after rejection)

  Result: 3 tokens accepted + 1 new token from large model = 4 tokens

Step 4: Repeat
  Continue from the last accepted position
```

### Why Verification Is Faster Than Generation

```
Generation (standard):
  For each token: Full forward pass, one token out
  5 tokens = 5 forward passes = 250ms

Verification (speculative):
  One forward pass processes ALL positions in parallel
  5 tokens verified = 1 forward pass = 50ms
  (Same computation as generating 1 token!)

Why? Transformer self-attention naturally processes all positions.
The GPU is already computing attention over the full sequence.
Adding a few more positions is nearly free.
```

---

## The Mathematics

### Acceptance Criterion

**The key guarantee: output distribution is IDENTICAL to the target model.**

```python
# Simplified speculative sampling
def speculative_decode(target_model, draft_model, prompt, K=5):
    tokens = prompt

    while not done:
        # 1. Draft: generate K tokens with small model
        draft_tokens = []
        draft_probs = []
        for i in range(K):
            p_draft = draft_model.get_probs(tokens + draft_tokens)
            token = sample(p_draft)
            draft_tokens.append(token)
            draft_probs.append(p_draft[token])

        # 2. Verify: get target model probs for all positions
        target_probs = target_model.get_probs_batch(
            tokens, draft_tokens
        )  # One forward pass!

        # 3. Accept/reject each draft token
        accepted = []
        for i, token in enumerate(draft_tokens):
            p_target = target_probs[i][token]
            p_draft = draft_probs[i]

            # Accept with probability min(1, p_target / p_draft)
            if random() < min(1.0, p_target / p_draft):
                accepted.append(token)
            else:
                # Reject: sample from adjusted distribution
                adjusted = max(0, target_probs[i] - draft_probs[i])
                adjusted = adjusted / sum(adjusted)
                new_token = sample(adjusted)
                accepted.append(new_token)
                break  # Stop at first rejection

        tokens.extend(accepted)
```

### Why It's Exact

```
The acceptance probability min(1, p_target/p_draft) ensures:

If draft model agrees with target: Accept (fast!)
If draft model disagrees: Reject and resample from corrected distribution

The corrected distribution = target - draft (normalized)
This mathematically guarantees the final output follows
EXACTLY the target model's distribution.

Not approximate. Not "close enough." EXACT.
```

### Expected Speedup

```
Speedup depends on draft model quality:
  α = average acceptance rate (how often draft matches target)

Expected tokens per step = 1/(1-α)
  α = 0.5 → 2 tokens per step (2x speedup)
  α = 0.7 → 3.3 tokens per step
  α = 0.8 → 5 tokens per step
  α = 0.9 → 10 tokens per step

But also need to account for draft model cost:
  Effective speedup = tokens_per_step / (1 + draft_cost/target_cost)
```

---

## Practical Considerations

### Choosing a Draft Model

```
Good draft models:
- Same family, smaller size (70B target → 7B draft)
- Distilled versions of the target
- Quantized versions of the target
- Fine-tuned small models on target's outputs

Key trade-off:
- Too small: Low acceptance rate, minimal speedup
- Too large: High acceptance rate, but draft is expensive
- Sweet spot: ~10-20x smaller than target
```

### Real-World Speedups

| Setup | Speedup | Acceptance Rate |
|-------|---------|----------------|
| Llama 70B + Llama 7B draft | 2.0-2.5x | ~60-70% |
| PaLM 540B + PaLM 62B draft | 2.0-3.0x | ~65-75% |
| GPT-4 + internal draft | ~2x | Undisclosed |
| Gemini + internal draft | ~2-3x | Undisclosed |

### Variants and Improvements

**Self-speculative decoding:**
```
No separate draft model needed!
Use early layers of the target model as the "draft"
Skip later layers for draft, use all layers for verification
Simpler deployment, slightly lower acceptance rate
```

**Medusa (Multi-head speculation):**
```
Add extra prediction heads to the target model
Each head predicts a future token
Verify tree of possibilities in parallel
Can achieve 2-3x speedup with no draft model
```

**Staged speculative decoding:**
```
Chain of models: Tiny → Small → Large
Tiny drafts K tokens very fast
Small verifies and extends
Large does final verification
3-5x speedups possible
```

**Speculative cascades:**
```
Combine small, fast model with efficient verification
Large model quickly checks if guesses match
Optimized for production serving
```

---

## Deployment

### Using with vLLM

```python
from vllm import LLM, SamplingParams

# Enable speculative decoding
llm = LLM(
    model="meta-llama/Llama-3.1-70B-Instruct",
    speculative_model="meta-llama/Llama-3.1-8B-Instruct",
    num_speculative_tokens=5,
    use_v2_block_manager=True
)

sampling_params = SamplingParams(temperature=0.7, max_tokens=500)
output = llm.generate("Explain quantum computing", sampling_params)
print(output[0].outputs[0].text)
```

### Using with Hugging Face

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load target and draft models
target = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3.1-70B-Instruct",
    device_map="auto",
    torch_dtype="auto"
)
draft = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3.1-8B-Instruct",
    device_map="auto",
    torch_dtype="auto"
)
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-70B-Instruct")

inputs = tokenizer("Explain quantum computing", return_tensors="pt")

# Generate with speculative decoding
outputs = target.generate(
    **inputs,
    max_new_tokens=200,
    assistant_model=draft,  # Speculative decoding!
    do_sample=True,
    temperature=0.7
)

print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

---

## When Speculative Decoding Helps Most

### Best For
```
+ Long outputs (more tokens to speculate on)
+ Predictable text (higher acceptance rate)
+ Latency-sensitive applications
+ Single-request serving (not high-throughput batching)
+ Reasoning models (long chain-of-thought outputs)
```

### Less Helpful For
```
- Short outputs (overhead not worth it)
- High-throughput batch serving (GPU already saturated)
- Very creative/unpredictable outputs (low acceptance rate)
- When draft model is unavailable
```

---

## Limitations

### 1. Memory Overhead
```
Need to store both target AND draft model
Draft model adds 10-15% to memory requirements
```

### 2. Diminishing Returns with Batching
```
With large batch sizes, GPU is already compute-bound
Speculative decoding helps less
Best for low-batch, latency-sensitive scenarios
```

### 3. Variable Speedup
```
Speedup depends on:
- Draft model quality
- Text predictability
- Sampling temperature (high temp = less predictable)
Worst case: No speedup (but never slower!)
```

---

## Key Takeaways

1. **2-3x faster, identical output** - The rare "free lunch" in ML
2. **Draft and verify** - Small model guesses, large model checks in parallel
3. **Mathematically exact** - Not an approximation, provably identical distribution
4. **Universally deployed** - Used by every major LLM provider
5. **No retraining needed** - Works with any existing autoregressive model

**Bottom line:** Speculative decoding is one of the most elegant algorithms in modern ML. By exploiting the asymmetry between generation (sequential) and verification (parallel), it achieves significant speedups without any quality loss. It's deployed everywhere and is essential for making large models practical.

---

## Further Reading

### Original Papers
- **Speculative Decoding (Leviathan et al.):** https://arxiv.org/abs/2211.17192
- **Speculative Sampling (Chen et al.):** https://arxiv.org/abs/2302.01318

### Variants
- **Medusa:** https://arxiv.org/abs/2401.10774
- **Eagle (Speculative Sampling++):** https://arxiv.org/abs/2401.15077
- **Staged Speculative Decoding:** https://arxiv.org/abs/2308.04623

### Implementations
- **vLLM:** https://docs.vllm.ai/en/latest/features/spec_decode.html
- **Hugging Face (assistant_model):** https://huggingface.co/blog/assisted-generation

---

**Published:** November 2022 (ICML 2023)
**Impact:** 🔥🔥🔥🔥🔥 **CRITICAL** - Most important inference optimization
**Adoption:** Universal - used by every major LLM provider
**Current Relevance:** Standard technique in all production LLM systems
**Legacy:** Proved you can speed up generation without sacrificing quality

**Modern Status (March 2026):** Speculative decoding and its variants (Medusa, Eagle, self-speculation) are standard in every production LLM serving system. Research continues on improving draft model quality, tree-based speculation, and hardware-aware optimization.
