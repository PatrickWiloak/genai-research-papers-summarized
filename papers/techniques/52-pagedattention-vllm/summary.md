# PagedAttention: Efficient LLM Serving with vLLM

**Authors:** Woosuk Kwon, Zhuohan Li, Sicheng Zhuang, et al. (UC Berkeley)
**Published:** September 2023 (SOSP 2023)
**Paper:** [arxiv.org/abs/2309.06180](https://arxiv.org/abs/2309.06180)

---

## Why This Matters

PagedAttention is **the infrastructure breakthrough that made LLM serving practical**:

- **24x higher throughput** - Serves 24x more requests than HuggingFace Transformers
- **Near-zero memory waste** - Reduces KV-cache memory waste from 60-80% to <4%
- **Virtual memory for AI** - Applies OS-level memory management to GPU memory
- **vLLM** - Open-source serving engine now used by most of the industry
- **Enabled the LLM API economy** - Made it economically viable to serve LLMs at scale

**Real-world impact:**
- vLLM became the standard LLM serving framework
- Used by Databricks, Anyscale, and dozens of AI companies
- Enabled affordable API pricing for LLM providers
- Made local LLM deployment practical for small teams
- Key infrastructure behind the reasoning model era (long outputs need efficient serving)

**The insight:** **LLM serving wastes 60-80% of GPU memory on fragmented KV-cache storage.** Borrow the operating system's virtual memory technique - paging - to eliminate this waste and dramatically increase throughput.

---

## The Problem

### Why Serving LLMs Is Hard

**The KV-cache bottleneck:**

```
During generation, LLMs cache key-value pairs for attention:

Input: "Tell me about quantum computing"
  Each token creates K,V vectors at every layer
  For Llama-70B: ~2MB per token per request
  For a 2048-token sequence: ~4GB of KV-cache per request

With 40GB GPU memory:
  Model weights: ~35GB (quantized)
  Available for KV-cache: ~5GB
  Maximum concurrent requests: ~1

That's terrible! One user at a time!
```

### The Memory Waste Problem

```
Traditional KV-cache allocation:

Request 1: "Hi" → Allocates 2048 tokens of KV-cache space
  Actually uses: 50 tokens
  Wasted: 2000 tokens (97.5% waste!)

Why? Must pre-allocate maximum possible length
  Don't know in advance how long the response will be
  Can't move the cache after allocation (GPU memory is contiguous)

Three types of waste:
1. Internal fragmentation: Pre-allocated but unused space
2. External fragmentation: Small gaps between allocations
3. Reservation waste: Reserved for potential future tokens

Total waste: 60-80% of KV-cache memory is wasted!
```

### The Impact of Memory Waste

```
With 60-80% memory waste:
  Can serve 2-3 requests concurrently

With near-zero waste:
  Can serve 10-20 requests concurrently

  That's 5-10x more throughput from the SAME hardware
  Directly translates to 5-10x lower cost per query
```

---

## How PagedAttention Works

### The Virtual Memory Analogy

```
Operating systems solved this exact problem decades ago:

Problem (1960s): Programs need contiguous memory
  Program A needs 1GB, Program B needs 2GB
  Physical RAM: 4GB
  But after A and B run, memory is fragmented
  Can't fit Program C (1.5GB) even though 1.5GB is free!

Solution: Virtual Memory + Paging
  Divide memory into fixed-size pages (4KB)
  Program sees contiguous virtual addresses
  OS maps virtual pages to ANY physical location
  No fragmentation!

PagedAttention: Same idea for KV-cache on GPUs
  Divide KV-cache into fixed-size blocks
  Each block holds KV pairs for a fixed number of tokens
  Blocks can be stored ANYWHERE in GPU memory
  A page table maps logical positions to physical blocks
```

### The PagedAttention Algorithm

```
Block size: 16 tokens (typical)

Request arrives: "Write a poem about cats"

Step 1: Allocate first block (tokens 1-16)
  Physical block #47 on GPU ← Wherever there's space!

Step 2: Generate tokens, fill block
  "Soft" "paws" "on" "moonlit" "floors" ... (16 tokens)

Step 3: Block full? Allocate next block
  Physical block #123 ← Doesn't need to be adjacent!

Step 4: Page table tracks mapping
  Logical block 0 → Physical block #47
  Logical block 1 → Physical block #123
  ...

Step 5: Attention computation uses page table
  When computing attention, gather KV pairs from
  scattered blocks using the page table
  GPU handles this gather efficiently
```

### Memory Efficiency

```
Traditional:
  Request: Pre-allocate 2048 tokens = 4GB
  Uses 100 tokens = 200MB
  Waste: 3.8GB (95%)

PagedAttention:
  Request: Allocate blocks on demand
  Uses 100 tokens = 7 blocks x 16 tokens = 112 tokens allocated
  Waste: 12 tokens in last block (only ~12%)

  And those 7 blocks are scattered wherever there's space
  No external fragmentation AT ALL
```

---

## vLLM: The Serving Engine

### Architecture

```
vLLM = PagedAttention + Continuous Batching + Scheduling

1. PagedAttention: Efficient memory management (see above)
2. Continuous batching: Don't wait for a batch to complete
   - New requests join immediately
   - Finished requests leave immediately
   - GPU is always busy
3. Smart scheduling: Prioritize requests intelligently
   - Preempt long requests for short ones
   - Swap KV-cache to CPU when GPU is full
   - Resume swapped requests later
```

### How Continuous Batching Works

```
Traditional batching:
  Batch = [Req1, Req2, Req3, Req4]
  Req1 finishes at step 50
  Req2 finishes at step 200
  GPU waits for ALL to finish before accepting new requests
  Req1's slot wasted for 150 steps!

Continuous batching (vLLM):
  Step 50: Req1 finishes → Req5 joins immediately
  Step 80: Req3 finishes → Req6 joins immediately
  GPU never sits idle waiting
  Throughput increase: 2-3x over traditional batching
```

---

## Performance

### Throughput Comparison

| System | Throughput (req/s) | vs. HuggingFace |
|--------|-------------------|-----------------|
| HuggingFace Transformers | 1x (baseline) | 1x |
| Text Generation Inference | 3.5x | 3.5x |
| **vLLM** | **14-24x** | **14-24x** |

### Memory Efficiency

```
KV-cache memory waste:
  HuggingFace: 60-80%
  FasterTransformer: 20-40%
  vLLM (PagedAttention): <4%

Practical impact on Llama-13B (A100 40GB):
  HuggingFace: ~4 concurrent requests
  vLLM: ~40+ concurrent requests
  Same GPU, 10x more users served
```

### Real-World Benchmarks

```
Serving Llama-70B on 4x A100-80GB:

Metric              | Without vLLM | With vLLM
--------------------|-------------|----------
Concurrent users    | 8           | 80+
Throughput (tok/s)  | 200         | 2000+
Latency (p50)       | 2s          | 0.5s
Cost per 1M tokens  | $15         | $1.50

10x cost reduction from the same hardware.
```

---

## Advanced Features

### KV-Cache Sharing (Copy-on-Write)

```
Multiple requests with the same system prompt:

Request 1: [System prompt] + "What is AI?"
Request 2: [System prompt] + "Explain gravity"

Without sharing:
  System prompt KV-cache computed and stored TWICE
  Wastes memory and compute

With PagedAttention copy-on-write:
  System prompt blocks are SHARED between requests
  Only divergent tokens get new blocks
  Like fork() in Unix - shared pages until write

For chat applications with standard system prompts:
  Saves 30-50% memory
  Supports 2x more concurrent requests
```

### Prefix Caching

```
Common prefix pattern:

All requests start with:
  "[System: You are a helpful assistant. You follow instructions...]"

vLLM caches the KV blocks for this prefix:
  First request: Compute and cache prefix KV
  All subsequent requests: Reuse cached KV blocks
  Skip recomputing the prefix entirely

Speedup: 2-5x for requests with long shared prefixes
```

### Speculative Decoding Integration

```
vLLM + Speculative Decoding + PagedAttention:
  Draft model KV-cache managed by PagedAttention
  Target model KV-cache managed by PagedAttention
  Both benefit from zero-waste memory management
  Combined speedup: 3-5x over naive serving
```

---

## Practical Usage

### Basic vLLM Server

```python
from vllm import LLM, SamplingParams

# Start serving
llm = LLM(
    model="meta-llama/Llama-3.1-70B-Instruct",
    tensor_parallel_size=4,  # Across 4 GPUs
    gpu_memory_utilization=0.90,  # Use 90% of GPU memory
)

# Generate
sampling = SamplingParams(temperature=0.7, max_tokens=512)
outputs = llm.generate(
    ["Explain quantum computing", "Write a haiku about AI"],
    sampling
)

for output in outputs:
    print(output.outputs[0].text)
```

### OpenAI-Compatible API Server

```bash
# Launch vLLM as an OpenAI-compatible server
python -m vllm.entrypoints.openai.api_server \
    --model meta-llama/Llama-3.1-70B-Instruct \
    --tensor-parallel-size 4 \
    --port 8000

# Use with any OpenAI SDK client
curl http://localhost:8000/v1/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "meta-llama/Llama-3.1-70B-Instruct",
        "prompt": "Explain quantum computing",
        "max_tokens": 256
    }'
```

### With OpenAI Python Client

```python
from openai import OpenAI

# Point to local vLLM server
client = OpenAI(base_url="http://localhost:8000/v1", api_key="unused")

response = client.chat.completions.create(
    model="meta-llama/Llama-3.1-70B-Instruct",
    messages=[{"role": "user", "content": "Explain quantum computing"}],
    max_tokens=256
)
print(response.choices[0].message.content)
```

---

## Impact on the Industry

### Made LLM APIs Affordable

```
Before PagedAttention/vLLM:
  Serving Llama-70B: ~$15 per million tokens
  Only large companies could afford to serve LLMs
  High latency, low concurrency

After PagedAttention/vLLM:
  Serving Llama-70B: ~$1.50 per million tokens
  Startups and researchers can serve models affordably
  Low latency, high concurrency

This 10x cost reduction enabled:
  - Together AI, Anyscale, and other API providers
  - Affordable open-source model hosting
  - The "open-source LLM API" economy
```

### Adoption

```
Who uses vLLM:
  - Databricks (Model Serving)
  - Anyscale (Ray Serve)
  - Together AI
  - Replicate
  - Dozens of startups
  - Most open-source LLM deployments

vLLM GitHub: 40,000+ stars
  Most popular LLM serving framework
  Active development community
```

---

## Limitations

### 1. GPU Memory Still Finite
```
PagedAttention eliminates waste but can't create memory
Very long contexts (100K+ tokens) still challenge GPU memory
Need techniques like offloading, compression alongside
```

### 2. Gather Operations Add Overhead
```
Scattered KV blocks require gather operations
Small overhead per attention computation
~5% overhead vs. contiguous memory (well worth the trade-off)
```

### 3. Block Size Trade-off
```
Larger blocks: Less overhead, more internal waste
Smaller blocks: More overhead, less waste
Typical sweet spot: 16 tokens per block
```

### 4. Model Loading Time
```
vLLM has higher startup time than simple inference
Loading model + initializing page tables + warming up
Best for persistent serving, not one-off inference
```

---

## Key Takeaways

1. **Virtual memory for GPUs** - Applied the OS paging concept to KV-cache, eliminating 60-80% memory waste
2. **24x throughput** - Same hardware serves 24x more requests than naive approaches
3. **10x cost reduction** - Made LLM serving economically viable for everyone
4. **vLLM became the standard** - 40K+ GitHub stars, used by most LLM serving deployments
5. **Enabling technology** - Without efficient serving, the LLM API economy wouldn't exist

**Bottom line:** PagedAttention solved the memory management crisis that made LLM serving prohibitively expensive. By applying virtual memory concepts to GPU KV-cache, it eliminated the massive memory waste that limited concurrent request handling. vLLM, the serving engine built on PagedAttention, became the industry standard and made it economically viable to serve LLMs to millions of users.

---

## Further Reading

### Original Paper
- **PagedAttention:** https://arxiv.org/abs/2309.06180

### Code
- **vLLM GitHub:** https://github.com/vllm-project/vllm
- **Documentation:** https://docs.vllm.ai

### Related Work
- **Speculative Decoding:** Paper 45 in this repo
- **FlashAttention:** Paper 16 in this repo
- **Continuous Batching (Orca):** https://arxiv.org/abs/2210.07669

### Infrastructure Papers
- **SGLang:** https://arxiv.org/abs/2312.07104
- **TensorRT-LLM:** https://github.com/NVIDIA/TensorRT-LLM

---

**Published:** September 2023 (SOSP 2023 Best Paper)
**Impact:** 🔥🔥🔥🔥🔥 **CRITICAL** - Made LLM serving practical and affordable
**Citations:** 1,500+ (as of early 2026)
**Adoption:** Universal - vLLM is the most popular LLM serving framework
**Current Relevance:** Standard infrastructure for LLM deployment
**Legacy:** Applied OS virtual memory concepts to AI, enabling the LLM API economy

**Modern Status (March 2026):** vLLM remains the dominant open-source LLM serving framework with 40K+ GitHub stars. PagedAttention's core ideas have been adopted by competing frameworks (SGLang, TensorRT-LLM) and influenced proprietary serving systems at OpenAI, Google, and Anthropic. The framework continues active development with support for newer features like speculative decoding, multi-modal serving, and disaggregated inference.
