---
title: "RoFormer: Enhanced Transformer with Rotary Position Embedding (RoPE)"
slug: "54-rope-rotary-position-embedding"
number: 54
category: "techniques"
authors: "Jianlin Su, Yu Lu, Shengfeng Pan, Bo Wen, Yunfeng Liu (Zhuiyi Technology)"
published: "April 2021 (revised through 2023)"
year: 2021
url: "https://arxiv.org/abs/2104.09864"
tags: [techniques]
---

# RoFormer: Enhanced Transformer with Rotary Position Embedding (RoPE)

**Authors:** Jianlin Su, Yu Lu, Shengfeng Pan, Bo Wen, Yunfeng Liu (Zhuiyi Technology)
**Published:** April 2021 (revised through 2023)
**Paper:** [arxiv.org/abs/2104.09864](https://arxiv.org/abs/2104.09864)

---

## Why This Matters

Rotary Position Embedding is **the position encoding scheme that powers virtually every modern LLM**:

- **Universal adoption** - Used by LLaMA 1/2/3/4, GPT-NeoX, PaLM, Gemini, Mistral, DeepSeek, Qwen, Yi, Falcon
- **Relative position via absolute encoding** - Best of both worlds in one elegant trick
- **Long-context extrapolation** - The foundation that NTK-aware RoPE and YaRN extend to 1M+ tokens
- **No learned parameters** - Pure mathematical transformation, zero memory overhead
- **Simple implementation** - A few dozen lines of code in any framework

**Real-world impact:**
- Replaced absolute positional embeddings (the original Transformer's approach) across the field
- Enabled the long-context revolution (8K -> 32K -> 128K -> 1M+ tokens)
- Made context-window extension possible without retraining from scratch
- Default position encoding in HuggingFace Transformers, Megatron-LM, vLLM
- Quietly became the most influential architectural decision of the post-GPT-3 era

**The insight:** **Encode position by rotating query and key vectors in 2D subspaces, with rotation angles that depend on both the absolute position and a frequency.** When you compute the dot product q dot k, the absolute rotations cancel and only the relative angle remains. You get relative position information for free, encoded through pure absolute-position rotations.

---

## The Problem

### Why Position Encoding Matters

```
Transformer attention is permutation-invariant:

  attention("the cat sat")  ==  attention("sat cat the")

Without position information, language models can't distinguish:
  "Dog bites man"  vs  "Man bites dog"

The model needs SOME signal that token i comes before token j.
But how do you encode position so the model can use it effectively?
```

### Approach 1: Sinusoidal Position Embeddings (Vaswani 2017)

```
Original Transformer added a fixed sinusoidal vector to each token:

  PE(pos, 2i)   = sin(pos / 10000^(2i/d))
  PE(pos, 2i+1) = cos(pos / 10000^(2i/d))

  x_pos = token_embedding + PE(pos)

Pros:
  - No learned parameters
  - Theoretically extrapolates to longer sequences

Cons:
  - Position info gets diluted as it flows through layers
  - Adding to embeddings mixes "what" and "where" channels
  - In practice, extrapolation BEYOND training length fails badly
  - Encodes ABSOLUTE position, but attention cares about RELATIVE distance
```

### Approach 2: Learned Absolute Position Embeddings (BERT, GPT-2)

```
Replace the sinusoidal formula with a learned table:

  PE = nn.Embedding(max_seq_len, d_model)
  x_pos = token_embedding + PE[pos]

Pros:
  - Model learns task-specific position patterns
  - Often performs slightly better than sinusoidal at training length

Cons:
  - HARD CAP at max_seq_len (BERT: 512, GPT-2: 1024)
  - Cannot extrapolate AT ALL beyond training length
  - Each position is a separate parameter, no smooth interpolation
  - Wastes parameters that scale with max sequence length
```

### Approach 3: Relative Position Bias (T5, Transformer-XL)

```
Add a bias to attention scores based on relative distance:

  attention(q_i, k_j) = (q_i . k_j) + bias(i - j)

Pros:
  - Directly encodes what matters: relative offsets
  - Generalizes better to longer sequences

Cons:
  - Requires modifying the attention computation itself
  - Bucket-based variants (T5) lose precision at long distances
  - Extra learned parameters per attention head
  - Still struggles to extrapolate cleanly
```

### The Core Tension

```
What we want from position encoding:

  1. Relative-position-aware (attention should care about i - j, not absolute i)
  2. Smoothly extrapolates to unseen lengths
  3. Decays with distance (far tokens should attend less)
  4. No extra parameters
  5. No invasive changes to the attention kernel

Sinusoidal:        relative no   extrapolates partial  decay no   params 0     invasive no
Learned absolute:  relative no   extrapolates no       decay no   params O(L)  invasive no
T5 relative bias:  relative yes  extrapolates partial  decay yes  params O(buckets)  invasive yes

RoPE: relative yes  extrapolates yes  decay yes  params 0  invasive (small)
```

---

## How RoPE Works

### The Core Idea: Rotate, Don't Add

```
Instead of ADDING a position vector to embeddings:
  x_pos = embed + PE(pos)        (old way)

ROTATE the query and key vectors by a position-dependent angle:
  q_pos = R(pos) . q              (new way)
  k_pos = R(pos) . k

Where R(pos) is a rotation matrix that depends on the position.

Why? When you compute attention dot product:
  q_m^T R(m)^T R(n) k_n  =  q_m^T R(n - m) k_n

The absolute rotations CANCEL and only the relative offset (n - m) remains.

This is the magic: an ABSOLUTE position encoding that produces
RELATIVE position behavior for free, just from how dot products work.
```

### The 2D Case (Intuition)

```
For a 2D vector q = (q_x, q_y), rotation by angle theta is:

  R(theta) = | cos(theta)  -sin(theta) |
             | sin(theta)   cos(theta) |

  R(theta) . q = | q_x cos(theta) - q_y sin(theta) |
                 | q_x sin(theta) + q_y cos(theta) |

Property: R(a) . R(b) = R(a + b) (rotations compose by adding angles)

For RoPE, set theta_pos = pos * frequency.

Then:  q_m = R(m * f) . q
       k_n = R(n * f) . k

  q_m . k_n = q^T R(m*f)^T R(n*f) k
            = q^T R((n-m)*f) k

The dot product depends ONLY on (n - m), the relative distance!
```

### Extending to d Dimensions

```
For a d-dimensional vector (d even), split into d/2 pairs of 2D subspaces.
Each pair gets its OWN frequency.

Frequencies follow the sinusoidal pattern from Vaswani:
  theta_i = 10000^(-2i/d)   for i in [0, d/2)

Frequency spectrum:
  - Low-index pairs: HIGH frequency (rotate fast with position)
  - High-index pairs: LOW frequency (rotate slowly with position)

This gives the model a rich multi-scale view of position:
  Some dimensions distinguish neighbors (i vs i+1)
  Other dimensions distinguish distant chunks (i vs i+1000)
```

### Visual: How RoPE Rotates Query/Key Vectors

```
Position 0: identity (no rotation)
  q_0 = [a, b, c, d, e, f, ...]

Position 1: rotate each (x, y) pair by its frequency
  pair 0 (high freq):  rotate by theta_0 = 1.0      (a, b) -> (a', b')
  pair 1 (med freq):   rotate by theta_1 = 0.1      (c, d) -> (c', d')
  pair 2 (low freq):   rotate by theta_2 = 0.01     (e, f) -> (e', f')

Position 100:
  pair 0: rotate by 100 * theta_0 = 100 rad   (many full revolutions)
  pair 1: rotate by 100 * theta_1 = 10 rad    (about 1.5 revolutions)
  pair 2: rotate by 100 * theta_2 = 1 rad     (a fraction of a revolution)

Different frequencies capture position at different scales,
just like how Fourier features capture signals at multiple scales.
```

### The Efficient Implementation

```
Naively, R(pos) is a d x d block-diagonal matrix. We don't materialize it.
Instead, compute element-wise:

def apply_rope(x, pos):
    # x has shape [..., d], split into d/2 pairs of 2D vectors
    # x = [x_0, x_1, x_2, x_3, ..., x_{d-2}, x_{d-1}]
    # pairs = [(x_0, x_1), (x_2, x_3), ..., (x_{d-2}, x_{d-1})]

    freqs = 1.0 / (10000 ** (torch.arange(0, d, 2) / d))   # [d/2]
    angles = pos * freqs                                    # [d/2]
    cos_a, sin_a = torch.cos(angles), torch.sin(angles)

    x_even = x[..., 0::2]
    x_odd  = x[..., 1::2]

    rotated_even = x_even * cos_a - x_odd * sin_a
    rotated_odd  = x_even * sin_a + x_odd * cos_a

    out = interleave(rotated_even, rotated_odd)
    return out

Cost: O(d) per token. No materialized rotation matrix. No extra params.
Applied to q and k BEFORE attention; v is left untouched.
```

### Where RoPE Sits in the Transformer

```
Standard attention block:

  q = W_q . x        <- project token to query
  k = W_k . x        <- project token to key
  v = W_v . x        <- project token to value
  attn_score = softmax(q . k^T / sqrt(d))
  out = attn_score . v

With RoPE:

  q = W_q . x
  k = W_k . x
  v = W_v . x
  q = apply_rope(q, pos)   <-- NEW: rotate q by its position
  k = apply_rope(k, pos)   <-- NEW: rotate k by its position
  attn_score = softmax(q . k^T / sqrt(d))
  out = attn_score . v

Note: RoPE touches ONLY q and k. v is unchanged. No additive PE on the input.
```

---

## Key Innovations

### 1. Relative Position from Absolute Rotation

```
The clever bit: each token's q,k is rotated by its OWN absolute position.
But because rotations compose, the dot product q_m . k_n only sees (n - m).

You get relative-position semantics with absolute-position bookkeeping.
No need for an O(L^2) relative-position table or attention bias modification.
```

### 2. Distance Decay Falls Out For Free

```
For sufficiently random q and k, the expected dot product after RoPE
DECREASES as |n - m| grows (proven in the paper).

Distant tokens naturally attend less, which is the inductive bias we want
for language. No hand-designed decay function needed.
```

### 3. Multi-Scale Frequency Encoding

```
Different (q,k) channel pairs rotate at different frequencies.
Like a Fourier basis, this gives the model multiple resolutions of position.

The model can specialize:
  - Some heads attend based on local position (high-freq channels)
  - Other heads attend based on global structure (low-freq channels)

The base 10000 (inherited from the original Transformer) sets the longest period.
Tweaking this base is what NTK-aware RoPE and YaRN exploit for long-context.
```

### 4. Composable with Linear Attention

```
RoPE works inside linear attention variants too:

  Linear attention: phi(q) . phi(k)^T  (kernel approximation)
  With RoPE:        R(m) phi(q) . phi(k)^T R(n)^T  =  same trick

This was a big motivator: prior position encodings were incompatible with
linear attention because they relied on the softmax denominator. RoPE bypasses
that constraint entirely.
```

---

## Why It Enables Long-Context Extrapolation

### The Fundamental Property

```
RoPE's rotation angle is theta_pos = pos * frequency.

If a model trains on positions 0..2047, it sees angles up to 2047 * frequency
for each frequency band. At inference, position 4096 produces angles
4096 * frequency. The MATH still works (rotations are well-defined for all angles).

But: the model has never seen those rotation regimes during training,
so attention quality degrades for very long contexts.

This is why naive context extension fails.
```

### NTK-Aware RoPE (2023 community discovery)

```
Insight: Don't change the formula, change the BASE.

Original RoPE: theta_i = 10000^(-2i/d)
NTK-aware:     theta_i = (10000 * scale)^(-2i/d)

By scaling the base instead of the position, low-frequency dimensions
(which encode long-range structure) get stretched proportionally,
while high-frequency dimensions (which encode local structure)
stay near their training distribution.

Result: Llama-2 4K -> 8K context with NO finetuning.
```

### YaRN: Yet another RoPE extensioN (Peng et al., 2023)

```
YaRN refines NTK-aware with:
  1. Per-frequency scaling (different stretch for different bands)
  2. Attention temperature adjustment (softmax sharpness)
  3. Light fine-tuning (~400 steps)

Result: Llama-2 4K -> 128K with minimal compute.
Most production long-context LLMs (Mistral 32K, Llama-3 128K) use YaRN
or close variants.
```

### Position Interpolation (Chen et al., 2023, Meta)

```
Simpler trick: scale POSITIONS instead of base.

  pos_inference = pos_real * (train_len / target_len)

Map a 8K position back into the 0..2K range the model trained on,
then RoPE-rotate normally. With short fine-tuning, this also extends context.

LLaMA-2 Long used this approach for 4K -> 32K.
```

### The Long-Context Pipeline

```
Modern long-context models stack these:

  1. Pretrain with RoPE at base 10000, length 4K-8K
  2. Apply YaRN/NTK to extend rotation regime
  3. Continue-train on long documents (~1B tokens)
  4. Sometimes: interleave with short data to prevent quality drop

Llama-3.1 (128K), Mistral Large (128K), Qwen2.5 (1M) all follow this recipe.
None of it works without RoPE as the foundation.
```

---

## Performance and Empirical Results

### Original RoFormer Results (2021)

| Model | Position Encoding | Chinese GLUE Avg |
|-------|-------------------|------------------|
| BERT (Chinese) | Learned absolute | 73.5 |
| WoBERT | Learned absolute | 75.0 |
| **RoFormer** | **RoPE** | **76.0** |

### Translation (En-De, WMT14)

| Model | BLEU |
|-------|------|
| Transformer (sinusoidal) | 27.3 |
| Transformer (learned) | 27.0 |
| **Transformer + RoPE** | **27.5** |

### Extrapolation (Perplexity at unseen lengths)

```
Train length: 512 tokens. Test at varying lengths.

Length    Sinusoidal   Learned    RoPE
512       baseline     baseline   baseline
1024      +12% PPL     fails      +3% PPL
2048      +45% PPL     fails      +8% PPL
4096      +120% PPL    fails      +15% PPL

RoPE alone extrapolates 2-4x further than alternatives BEFORE
applying NTK/YaRN extensions.
```

---

## RoPE vs ALiBi

ALiBi (Press et al., 2021) is the main competitor and worth understanding.

```
ALiBi: Attention with Linear Biases

  attention(q_i, k_j) = (q_i . k_j) - m * |i - j|

  Just subtract a linear penalty proportional to distance.
  Each head gets a different slope m.

  No vector rotation. No position embedding. Pure attention bias.
```

### Comparison

| Property | RoPE | ALiBi |
|----------|------|-------|
| Where applied | Q, K vectors | Attention scores |
| Adds parameters | No | No |
| Encoding type | Rotational | Linear bias |
| Information channels | All Q,K dims | Just distance scalar |
| Native extrapolation | Moderate | Strong |
| With YaRN/NTK | Excellent | N/A |
| Long-context expressiveness | Higher | Lower (only distance) |
| Used by | LLaMA, Mistral, Gemini, GPT-NeoX, Qwen, DeepSeek | BLOOM, MPT, Falcon (early) |
| Industry winner | YES | Mostly abandoned |

### Why RoPE Won

```
ALiBi extrapolates well out-of-the-box but encodes only DISTANCE.
RoPE encodes RELATIVE OFFSET as a full rotation, preserving more structure.

Empirically:
  - At training length: comparable
  - Slight extension: RoPE wins with NTK/YaRN
  - Long-form reasoning: RoPE-based models retain better needle-in-haystack
  - Multi-needle and ordering tasks: RoPE preserves structure ALiBi loses

Once YaRN solved RoPE's extrapolation weakness in 2023, ALiBi lost its
main advantage and the field consolidated around RoPE.
```

---

## Adoption: Every Modern LLM Uses RoPE

```
Open-source LLMs using RoPE:
  - LLaMA 1, 2, 3, 4 (Meta)
  - Mistral, Mixtral (Mistral AI)
  - Qwen 1, 2, 2.5, 3 (Alibaba)
  - DeepSeek V1, V2, V3, R1 (DeepSeek)
  - Yi (01.AI)
  - GPT-NeoX, Pythia (EleutherAI)
  - Falcon (later versions)
  - Phi-2, Phi-3, Phi-4 (Microsoft)
  - Gemma 1, 2, 3 (Google)
  - Command R, R+ (Cohere)

Closed-source LLMs reported/inferred to use RoPE:
  - PaLM, PaLM 2, Gemini 1.0, 1.5, 2.0 (Google)
  - Claude (Anthropic, inferred from behavior)

That's basically the entire frontier of LLM development since 2022.

Holdouts (mostly historical):
  - GPT-2, GPT-3 (sinusoidal/learned, predates RoPE)
  - BLOOM (ALiBi)
  - Falcon-7B/40B (ALiBi, switched in later versions)
```

---

## Real-World Applications

### Long-Context Q&A and Document Analysis

```
Without RoPE + YaRN:
  Models capped at training length (typically 2K-8K tokens)
  Long PDFs require chunking and retrieval

With RoPE + YaRN:
  128K-1M token contexts standard
  Whole codebase analysis (Claude, Gemini)
  Long legal/medical document review
  Book-length summarization
```

### Code Models

```
Code is structurally long-range:
  - Function defined on line 100, called on line 5000
  - Imports at top affect references throughout

RoPE's multi-scale frequency encoding gives code LLMs the right inductive
bias for these long dependencies. DeepSeek-Coder, Qwen-Coder, and Codestral
all rely on RoPE for their 32K-128K windows.
```

### Reasoning Models (o1, R1, Claude reasoning)

```
Long chain-of-thought outputs (10K-100K reasoning tokens) need:
  - Coherent positional reasoning across the chain
  - No degradation at the tail of the trace

RoPE-with-YaRN models maintain attention quality across long traces.
This is part of why DeepSeek-R1 and similar work at all.
```

### Retrieval-Augmented Generation

```
RAG often stuffs many documents into the prompt:
  Top-50 retrieved chunks + query + system prompt = 30K+ tokens

RoPE-extended models can attend to ALL chunks rather than only the latest,
reducing the "lost in the middle" failure mode that plagued earlier systems.
```

---

## Practical Implementation

### Minimal PyTorch RoPE

```python
import torch

def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0):
    """Precompute the complex exponentials for RoPE."""
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[:dim // 2].float() / dim))
    t = torch.arange(end)
    freqs = torch.outer(t, freqs)
    return torch.polar(torch.ones_like(freqs), freqs)  # complex64

def apply_rotary_emb(xq, xk, freqs_cis):
    """Apply RoPE to query and key tensors."""
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(-2)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(-2)
    return xq_out.type_as(xq), xk_out.type_as(xk)

# Usage in an attention block:
freqs_cis = precompute_freqs_cis(head_dim, max_seq_len)
q, k = apply_rotary_emb(q, k, freqs_cis[:seq_len])
attn = (q @ k.transpose(-2, -1)) / math.sqrt(head_dim)
# ... rest of attention as usual
```

This is essentially the LLaMA reference implementation.

### Extending Context with NTK Scaling

```python
# Original
freqs_cis = precompute_freqs_cis(dim, max_seq_len, theta=10000.0)

# NTK-aware extension to 2x context, no fine-tuning
scale = 2.0
new_theta = 10000.0 * (scale ** (dim / (dim - 2)))
freqs_cis = precompute_freqs_cis(dim, max_seq_len * 2, theta=new_theta)

# Plug into the same model weights, attention quality holds for ~2x length
```

---

## Limitations

### 1. Not Magical Extrapolation
```
RoPE alone extrapolates 1.5-3x beyond training length before quality drops.
Real long-context (10x+) requires NTK, YaRN, or position interpolation
plus continue-training.
```

### 2. Q,K Only - Doesn't Encode Position in V
```
Values are not rotated, so positional info reaches downstream layers
only through the attention pattern. Usually fine, but limits some niche use cases
(e.g. value-based positional reasoning).
```

### 3. Sensitive to Numerical Precision at Long Contexts
```
At position 1M, angles for low-freq channels are huge.
fp16 / bf16 sin/cos can lose precision. Practical implementations
compute freqs in fp32 and cast back.
```

### 4. Not a Drop-In for Pre-RoPE Checkpoints
```
You can't take BERT-base and swap RoPE in without retraining.
The Q,K projections were trained assuming additive position info.
RoPE retrofits require at least continue-pretraining.
```

---

## Connections to Other Papers

### Builds On
- **Attention Is All You Need (Vaswani 2017)** - Defined the original sinusoidal PE that RoPE replaces. RoPE inherits the 10000 base frequency.
- **Self-Attention with Relative Position Representations (Shaw 2018)** - First serious attempt at relative position encoding. RoPE achieves the same goal more efficiently.
- **Music Transformer (Huang 2018)** - Introduced relative attention biases, predecessor pattern to T5's relative bias.
- **Linear Attention / Performer (Choromanski 2020)** - Motivated the search for position encodings compatible with non-softmax attention. RoPE plugs in cleanly.

### Enables
- **NTK-Aware RoPE (2023, /u/bloc97 on Reddit)** - Long-context extension by base scaling.
- **Position Interpolation (Chen et al., 2023)** - LLaMA's official 32K extension.
- **YaRN (Peng et al., 2023)** - Per-frequency scaling, the de facto long-context method.
- **LongRoPE (Microsoft, 2024)** - 2M context for Phi-3.
- **Theta Scaling Laws (2024)** - Theoretical analysis of RoPE base choice for long context.

### Compared Against
- **ALiBi (Press et al., 2021)** - The main competitor. Lost adoption once YaRN solved RoPE's extrapolation issue.
- **T5 Relative Position Bias** - Used by T5 and Flan variants. More invasive, less widely adopted.
- **xPos (Sun et al., 2022)** - RoPE variant with explicit decay. Used by some Microsoft models, not widely adopted.

### Used By
- **LLaMA Family** - LLaMA 1/2/3/4 all use RoPE.
- **Mistral, Mixtral** - RoPE with sliding-window attention.
- **DeepSeek-V3, R1** - RoPE with multi-head latent attention.
- **Gemini, Gemma** - Reported to use RoPE variants.
- **Every open-source frontier model post-2022.**

---

## Key Takeaways

1. **Rotate, don't add** - Position encoding via Q/K rotation, not additive embeddings, was the unlock.
2. **Absolute mechanism, relative behavior** - Each token rotates by its own position, but dot products only see relative offsets. Elegant.
3. **Multi-scale frequencies** - Inheriting the sinusoidal frequency spectrum gives RoPE its rich positional structure.
4. **Foundation for long context** - NTK-aware, YaRN, and position interpolation all build directly on RoPE's mathematical structure.
5. **Universal adoption** - Powers essentially every modern LLM. The default choice when designing a new transformer in 2026.

**Bottom line:** RoPE replaced absolute position embeddings in the post-2021 LLM era because it elegantly delivers relative-position behavior through pure absolute-position rotation, with zero extra parameters, no invasive attention changes, and a frequency structure that supports long-context extrapolation. The combination of RoPE plus YaRN turned the "context window" from a hard limit into a tunable parameter, and that capability underlies everything from 128K-token coding agents to million-token reasoning models.

---

## Further Reading

### Original Paper
- **RoFormer:** https://arxiv.org/abs/2104.09864

### Code
- **Reference implementation in LLaMA:** https://github.com/meta-llama/llama/blob/main/llama/model.py
- **EleutherAI's rotary-embedding-torch:** https://github.com/lucidrains/rotary-embedding-torch

### Long-Context Extensions
- **Position Interpolation (Meta):** https://arxiv.org/abs/2306.15595
- **YaRN:** https://arxiv.org/abs/2309.00071
- **LongRoPE (Microsoft):** https://arxiv.org/abs/2402.13753

### Comparisons
- **ALiBi:** https://arxiv.org/abs/2108.12409
- **xPos:** https://arxiv.org/abs/2212.10554

### Related Papers in This Repo
- **Attention Is All You Need:** Paper 01
- **FlashAttention:** Paper 16
- **LLaMA:** Paper 27
- **PagedAttention / vLLM:** Paper 52

---

**Published:** April 2021 (continually revised through 2023)
**Impact:** 🔥🔥🔥🔥🔥 **CRITICAL** - The position encoding of the modern LLM era
**Citations:** 4,000+ (as of early 2026)
**Adoption:** Universal across open and closed frontier LLMs
**Current Relevance:** Default position encoding for all new transformer designs
**Legacy:** Made long-context LLMs possible through clean math, no parameters, and natural extension paths

**Modern Status (April 2026):** RoPE is the unquestioned default position encoding for new LLM architectures. Every major open-source model family (LLaMA, Mistral, Qwen, DeepSeek, Gemma, Phi) ships with RoPE, and the long-context arms race (now reaching 1M-10M tokens) is built entirely on RoPE-based extensions like YaRN and LongRoPE. Active research continues into theta-scaling laws, per-layer RoPE tuning, and combining RoPE with newer architectures like state-space models and linear attention variants.
