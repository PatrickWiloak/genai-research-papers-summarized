# FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness

**Authors:** Tri Dao, Daniel Y. Fu, Stefano Ermon, Atri Rudra, Christopher Ré (Stanford, CMU)
**Published:** May 2022 (NeurIPS 2022)
**Paper:** [arxiv.org/abs/2205.14135](https://arxiv.org/abs/2205.14135)

---

## Why This Matters

FlashAttention made long-context Transformers practically feasible by solving the memory bottleneck. This breakthrough:

- **10-20× faster** than standard attention
- **Enables 64k+ context lengths** (vs 2k typical at the time)
- **No approximation** - mathematically identical to standard attention
- **Powers GPT-4, Claude 2+** and all modern long-context LLMs
- **Critical infrastructure** - every major AI lab uses this

**Real-world impact:**
- GPT-4's 32k-128k context windows
- Claude's 100k-200k context
- Gemini's 1M context
- Makes AI practical for long documents, codebases, conversations

**The insight:** The problem isn't FLOPs, it's memory access. Optimize for GPU memory hierarchy.

---

## The Problem

**Standard attention is memory-bound, not compute-bound:**

### Memory Bottleneck (Not Compute)
```
Sequence length: n = 4096 tokens
Attention matrix: n × n = 16M values
Memory needed: 16M × 4 bytes = 64MB just for attention scores

Problem: Reading/writing this to/from GPU memory is SLOW
- GPU compute: ~300 TFLOPS
- GPU memory bandwidth: ~1.5 TB/s
- Attention is bottlenecked by memory, not math
```

### Quadratic Memory Growth
```
n = 512   → 1MB attention matrix
n = 2048  → 16MB
n = 4096  → 64MB
n = 8192  → 256MB
n = 16384 → 1GB  (single attention layer!)
```

**At 16k tokens:** Can't even fit attention matrix in GPU memory for large models.

### Before FlashAttention
- Most models: 2k-4k context max
- Longer sequences: Quadratic memory explosion
- Solutions: Approximate attention (sacrifice quality)
  - Sparse attention
  - Linear attention
  - Low-rank approximations
  - All lose information!

**The question:** Can we get exact attention with better memory efficiency?

---

## Core Innovation

### IO-Aware Algorithm Design

**Key insight:** Modern GPUs have a memory hierarchy:
```
Fast but small:  SRAM (on-chip) - ~20MB, very fast
Slow but large:  HBM (off-chip) - ~40GB, 10-20× slower

Standard attention: Constantly moves data between HBM ↔ SRAM
FlashAttention: Minimize HBM accesses, maximize SRAM reuse
```

### The Algorithm

**Standard Attention (memory-intensive):**
```python
# Load Q, K, V from HBM to SRAM
Q, K, V = load_from_HBM()

# Compute attention scores (write to HBM)
S = Q @ K.T  # n×n matrix, write to HBM
save_to_HBM(S)

# Compute softmax (read from HBM, write back)
P = softmax(S)  # Read S from HBM, write P to HBM
save_to_HBM(P)

# Compute output (read P from HBM)
O = P @ V
save_to_HBM(O)

# Total HBM accesses: O(n²) reads/writes
```

**FlashAttention (IO-optimized):**
```python
# Divide Q, K, V into blocks that fit in SRAM
# Process blocks in "tiles"

for each block of queries:
    for each block of keys/values:
        # Load small blocks into SRAM
        Q_block = load_small_block(Q)
        K_block = load_small_block(K)
        V_block = load_small_block(V)

        # Compute attention ENTIRELY in SRAM
        S_block = Q_block @ K_block.T
        P_block = softmax(S_block)
        O_block = P_block @ V_block

        # Only save final output to HBM
        save_to_HBM(O_block)

# Total HBM accesses: O(n) - MUCH better!
```

### Tiling and Recomputation

**The trick:** Never materialize the full n×n attention matrix

1. **Tiling:** Process attention in small blocks
2. **Online softmax:** Compute softmax incrementally without storing full scores
3. **Recomputation:** Recompute attention scores in backward pass (trade compute for memory)

**Mathematics:**
- Softmax can be computed in one pass with running statistics
- No need to store intermediate attention matrix
- Backward pass recomputes forward values on-the-fly

---

## Technical Details

### Block-Sparse Tiling

**Divide sequence into blocks:**
```
Sequence length: n = 4096
Block size: B = 256
Number of blocks: 4096/256 = 16

Instead of 4096×4096 attention:
Process 16×16 blocks of 256×256 each
Each block fits in SRAM!
```

### Online Softmax Algorithm

**Standard softmax requires two passes:**
```python
# Pass 1: Find max for numerical stability
max_val = max(scores)

# Pass 2: Compute softmax
exp_scores = exp(scores - max_val)
softmax = exp_scores / sum(exp_scores)
```

**FlashAttention's online softmax (one pass):**
```python
# Maintain running statistics
running_max = -inf
running_sum = 0

for each block:
    new_max = max(running_max, block_max)
    # Rescale previous sum
    running_sum = running_sum * exp(running_max - new_max)
    # Add new block
    running_sum += sum(exp(block - new_max))
    running_max = new_max

# Final softmax without storing all scores
```

### Memory Complexity

**Standard Attention:**
```
Memory: O(n² + n·d)
- O(n²): Attention matrix
- O(n·d): Q, K, V matrices

For n=4096, d=128:
- Attention: 64MB
- QKV: 6MB
- Total: ~70MB per layer
```

**FlashAttention:**
```
Memory: O(n·d)
- Only store Q, K, V, O
- No attention matrix!
- Blocks processed in SRAM

For n=4096, d=128:
- QKV: 6MB
- Total: ~6MB per layer (10× reduction!)
```

---

## Results and Impact

### Speed Improvements

| Sequence Length | Standard Attention | FlashAttention | Speedup |
|----------------|-------------------|----------------|---------|
| 512 | 1.0× | 2.1× | 2.1× |
| 1024 | 1.0× | 3.8× | 3.8× |
| 2048 | 1.0× | 7.6× | 7.6× |
| 4096 | 1.0× | 15.2× | 15.2× |
| 8192 | OOM | 1.0× | ∞ (enables) |

**Speedup increases with sequence length!**

### Memory Savings

| Model | Standard (max length) | FlashAttention (max length) | Improvement |
|-------|----------------------|----------------------------|-------------|
| GPT-2 | 1024 | 4096 | 4× |
| BERT-Large | 512 | 8192 | 16× |
| GPT-3 size | 2048 | 16384 | 8× |

### Quality: Identical

**FlashAttention is exact, not approximate:**
- Numerical error: < 10^-6 (floating point precision)
- No quality loss
- Same outputs as standard attention

---

## Real-World Applications

### Long-Context LLMs (2023-2024)

**GPT-4:**
- 32k context: FlashAttention
- 128k context: FlashAttention 2

**Claude:**
- 100k context: FlashAttention
- 200k context (Claude 3): FlashAttention 2

**Gemini 1.5:**
- 1M context: FlashAttention + custom optimizations

### Training Speedups

**LLaMA Training:**
- 15% faster training with FlashAttention
- Enables longer context in pre-training

**Stable Diffusion:**
- 2× faster image generation
- Enables higher resolution

### Cost Savings

**Production inference:**
- 40-50% cost reduction (less GPU time)
- Enables longer contexts without more hardware
- Better GPU utilization (less idle memory bandwidth)

---

## FlashAttention 2 (2023)

**Even faster:** [arxiv.org/abs/2307.08691](https://arxiv.org/abs/2307.08691)

### Improvements
- **2× faster** than FlashAttention 1
- Better GPU utilization (75% → 90%)
- Optimized for H100/A100 GPUs
- Better parallelization

### What Changed
```
FlashAttention 1: Block size optimized for A100
FlashAttention 2:
  - Better work partitioning across GPU threads
  - Reduced non-matmul FLOPs
  - Better occupancy (more work in flight)
```

**Result:**
- GPT-4: Uses FlashAttention 2
- Most new models: Built with FA2

---

## Limitations

### 1. **Requires Careful Implementation**
- Complex CUDA kernels
- GPU-specific optimization
- Hard to extend to new operations

### 2. **Not All Attention Patterns**
- Works best for dense attention
- Sparse patterns may need different optimizations
- Block structure matters

### 3. **Backward Pass Still Expensive**
- Recomputes attention scores
- Trades compute for memory
- Training still slower than inference

### 4. **Hardware Specific**
- Optimized for NVIDIA GPUs (CUDA)
- Different GPUs need different tuning
- Not as fast on CPUs or older GPUs

---

## Practical Usage

### Installation

```bash
pip install flash-attn
# Requires CUDA 11.6+, PyTorch 1.12+
```

### Usage (PyTorch)

```python
from flash_attn import flash_attn_func

# Your standard attention:
# attn = softmax(Q @ K.T / sqrt(d)) @ V

# Replace with FlashAttention:
output = flash_attn_func(
    q,  # (batch, seqlen, nheads, headdim)
    k,  # (batch, seqlen, nheads, headdim)
    v,  # (batch, seqlen, nheads, headdim)
    dropout_p=0.1,
    softmax_scale=1.0 / math.sqrt(headdim),
    causal=True  # For autoregressive models
)

# That's it! 10-20× faster, same results
```

### Hugging Face Integration

```python
from transformers import AutoModel

# Many models now use FlashAttention by default
model = AutoModel.from_pretrained(
    "meta-llama/Llama-2-7b",
    attn_implementation="flash_attention_2"  # Use FA2
)

# Or force it:
model = model.to_bettertransformer()  # Auto-uses FlashAttention
```

### Training with FlashAttention

```python
# Just replace standard attention in your model
class TransformerBlock(nn.Module):
    def __init__(self):
        # Instead of:
        # self.attn = nn.MultiheadAttention(...)

        # Use:
        from flash_attn.modules.mha import MHA
        self.attn = MHA(
            embed_dim=768,
            num_heads=12,
            use_flash_attn=True
        )
```

---

## Impact on Field

### Enabled Long Context Era

**Before FlashAttention (2021):**
- Most models: 2k context
- GPT-3: 2k context
- Long context = expensive approximations

**After FlashAttention (2022+):**
- GPT-4: 128k context
- Claude 3: 200k context
- Gemini 1.5: 1M context
- Exact attention at scale!

### Infrastructure Standard

**Adoption:**
- Hugging Face Transformers: Built-in support
- PyTorch 2.0: SDPA (uses FlashAttention under hood)
- Every major AI lab: Uses FlashAttention
- LLaMA 2/3: Trained with FlashAttention

### Research Impact

**Enabled new research:**
- Retrieval over full books
- Multi-document reasoning
- Long-form conversation
- Code generation (full repositories)
- RAG with long contexts

---

## Key Takeaways

1. **Memory bandwidth is the bottleneck**, not compute
2. **IO-aware algorithms** dramatically outperform naive implementations
3. **Exact ≠ slow** - can have both speed and accuracy
4. **Tiling + recomputation** trades cheap compute for expensive memory
5. **Enabled the long-context revolution** in 2023-2024

**Bottom line:** FlashAttention is **critical infrastructure** for modern AI. Every long-context model uses it.

---

## Further Reading

### Original Papers
- **FlashAttention:** https://arxiv.org/abs/2205.14135
- **FlashAttention-2:** https://arxiv.org/abs/2307.08691

### Code
- **Official Implementation:** https://github.com/Dao-AILab/flash-attention
- **PyTorch SDPA:** Built into PyTorch 2.0+
- **Hugging Face:** Integrated in Transformers library

### Related Work
- **Paged Attention (vLLM):** Memory optimization for inference
- **Ring Attention:** Distributed long-context
- **FlashDecoding:** Optimized for generation

### Tutorials
- **Tri Dao's blog:** https://tridao.me/blog/
- **Hugging Face guide:** FlashAttention integration
- **CUDA tutorial:** Understanding GPU memory hierarchy

---

**Published:** May 2022
**Impact:** 🔥🔥🔥🔥🔥 **CRITICAL** - Powers all modern long-context LLMs
**Citations:** 1000+ (and growing rapidly)
**Adoption:** Universal in production LLMs
**Legacy:** Made long-context AI practically feasible

**Current Status (2024/2025):** FlashAttention 2 is the default for all new models. FlashAttention 3 in development for even longer contexts.
