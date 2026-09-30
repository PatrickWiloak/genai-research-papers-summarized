# The KV Cache

**In one line:** When a language model generates text it keeps a running memory of every earlier token (the key-value, or KV, cache), and the size of that memory, not the model's weights, is often what limits context length, batch size and serving cost.
**Last reviewed:** 2026-09-30

---

## The short version

- A Transformer generates one token at a time, and each new token has to "look back" at every earlier token through attention. Recomputing everything for every step would be wasteful, so the model stores two vectors per earlier token per layer, the **key** and the **value**. That store is the **KV cache**.
- The cache grows **linearly with the number of tokens** and has to live in fast accelerator memory for the whole request. For long contexts or many simultaneous users it can be larger than the model itself.
- A simple formula tells you its size: `2 x layers x KV heads x head size x bytes per number`, per token. For Llama 3.3 70B that is about **320 KiB per token**, or **40 GiB for one full 128K-token conversation**.
- Most of the architecture and serving tricks of 2023-2025 exist to shrink or manage this cache: **grouped-query attention** (fewer KV heads), **multi-head latent attention** (compress the cache), **PagedAttention** (stop wasting the memory you have), plus cache quantization, sliding windows and prefix reuse.
- Generation speed is usually limited by **reading** the cache and weights from memory, not by arithmetic. A smaller cache means faster tokens as well as more of them.

## What the cache is

In a Transformer's attention layer, every token is turned into three vectors: a **query** ("what am I looking for?"), a **key** ("what do I contain?") and a **value** ("what do I pass on if selected?"). A token attends to earlier tokens by comparing its query with their keys, then taking a weighted mix of their values. The hands-on [Transformer architecture](https://github.com/PatrickWiloak/cloud-data-ai-security-zero-to-hero/blob/main/learn/concepts/transformer-architecture.md) page walks through this; the original design is in [Attention Is All You Need](../../papers/architectures/01-attention-is-all-you-need/summary.md).

During generation, the keys and values of past tokens never change. So instead of recomputing them at every step, the model computes them once and stores them:

```
Prompt: "The cat sat on the"            Generating the next token:

layer 1:  K,V for [The][cat][sat][on][the]   <- stored once
layer 2:  K,V for [The][cat][sat][on][the]   <- stored once
 ...
layer N:  K,V for [The][cat][sat][on][the]   <- stored once

step 1: new token "mat"  -> compute its Q,K,V; attend over cache; append its K,V
step 2: new token "."    -> compute its Q,K,V; attend over cache; append its K,V
```

This splits inference into two phases:

- **Prefill**: the whole prompt is processed in parallel and its keys and values are written into the cache. This phase is compute-heavy and fast per token.
- **Decode**: tokens are produced one at a time. Each step does little arithmetic but must read the entire cache (and the weights) from memory. This phase is **memory-bandwidth bound**, which is why the cache's size matters for speed and not only for capacity.

## The memory maths, with a worked example

Per token, the cache holds one key and one value vector for every KV head in every layer:

```
KV bytes per token = 2 (K and V)
                   x number of layers
                   x number of KV heads
                   x head dimension
                   x bytes per number (2 for 16-bit formats like BF16)

KV bytes per request = KV bytes per token x tokens in context
KV bytes on the GPU  = sum over all concurrent requests
```

**Worked example: Llama 3.3 70B.** Its published configuration has 80 layers, 64 query heads but only 8 key-value heads, and a head dimension of 128, stored in BF16 (2 bytes).

```
per token:  2 x 80 x 8 x 128 x 2 bytes  = 327,680 bytes  = 320 KiB

one 8K-token chat:       320 KiB x 8,192    = 2.5 GiB
one full 128K context:   320 KiB x 131,072  = 40 GiB
ten users at 32K each:   320 KiB x 32,768 x 10 = 100 GiB
```

For comparison, the weights of a 70-billion-parameter model in BF16 are roughly 140 GB. So one user at full context adds a cache worth over a quarter of the model, and a handful of long-context users need more memory for their caches than for the model. This is why serving providers price long contexts carefully and why "the model supports 128K" does not mean a given deployment will give every user 128K at once.

**The same model without GQA.** If all 64 heads kept their own keys and values (standard multi-head attention), the per-token figure would be 8 times larger: 2.5 MiB per token, or 320 GiB for a single 128K context. That is more than the combined memory of four 80 GB accelerators, for one conversation. The next section is about why nobody ships it that way.

These figures cover the cache alone and ignore implementation overhead; real systems also need working memory for activations.

## Why GQA, MLA and PagedAttention exist

Three different attacks on the same problem.

### Fewer KV heads: MQA and GQA

**Multi-query attention** (Shazeer, 2019) let all query heads share a single key-value head, shrinking the cache by the number of heads but costing quality. **Grouped-query attention** (Ainslie et al., 2023) is the compromise: query heads are split into groups, and each group shares one KV head. With 8 KV heads serving 64 query heads, the cache is 8 times smaller with little measurable quality loss. That is the setting in the Llama 3.3 example above, and it is now the default in most open models. See the [GQA summary](../../papers/architectures/75-grouped-query-attention/summary.md).

```
Multi-head (MHA):   Q Q Q Q Q Q Q Q      each query head has its own K,V
                    K K K K K K K K
Grouped (GQA):      Q Q Q Q | Q Q Q Q    each group shares one K,V
                       K    |    K
Multi-query (MQA):  Q Q Q Q Q Q Q Q      everyone shares one K,V
                           K
```

### Compress the cache: MLA

DeepSeek's **multi-head latent attention**, introduced with DeepSeek-V2 in 2024, takes a different route: instead of storing full keys and values, it stores one small compressed **latent vector** per token per layer and reconstructs keys and values from it when needed. DeepSeek reported that this cut the KV cache by 93.3% relative to their earlier 67B dense model and raised maximum generation throughput 5.76 times. See the [MLA summary](../../papers/architectures/141-multi-head-latent-attention/summary.md).

A rough comparison using DeepSeek-V3's published configuration (61 layers, a 512-dimensional compressed latent plus a 64-dimensional positional part per layer), assuming BF16 storage:

```
per token:  61 x (512 + 64) x 2 bytes = 70,272 bytes  = about 69 KiB
```

That is under a quarter of Llama 3.3 70B's 320 KiB per token, for a model with far more total parameters (671B, of which 37B are active per token; see [DeepSeek-V3](../../papers/language-models/27-deepseek-v3/summary.md)). The trade-off is extra computation to decompress and a more complex attention implementation.

### Stop wasting memory: PagedAttention

Early serving systems reserved one contiguous block of memory per request, sized for the maximum possible output. Most of it sat empty, and gaps between blocks could not be reused. The vLLM authors measured 60-80% of KV cache memory being wasted this way. **PagedAttention** (Kwon et al., 2023) borrows virtual memory from operating systems: the cache is stored in small fixed-size blocks that can live anywhere in memory, with a table mapping each request's logical positions to physical blocks. Waste drops to under 4%, more requests fit in the same GPU, and throughput rises accordingly. See the [PagedAttention summary](../../papers/techniques/52-pagedattention-vllm/summary.md) and the hands-on [inference servers](https://github.com/PatrickWiloak/cloud-data-ai-security-zero-to-hero/blob/main/learn/concepts/inference-servers.md) page.

Paging also makes **sharing** easy. If many requests start with the same system prompt or document, their cache blocks can be computed once and reused. vLLM's automatic prefix caching does this by hashing each block together with everything before it. The API-level version of the same idea is **prompt caching**, which providers sell at a discount for repeated prefixes; see the sibling repo's [prompt caching](https://github.com/PatrickWiloak/cloud-data-ai-security-zero-to-hero/blob/main/learn/concepts/prompt-caching.md) page.

### Other levers

| Technique | What it does to the cache | Trade-off |
|---|---|---|
| **KV cache quantization** (for example FP8) | Stores keys and values in fewer bits, fitting more tokens in memory; vLLM supports this directly | Small accuracy risk; see [quantization](https://github.com/PatrickWiloak/cloud-data-ai-security-zero-to-hero/blob/main/learn/concepts/quantization-and-distillation.md) |
| **Sliding-window attention** | Each layer only attends to the last W tokens, so the cache stops growing past W | Distant tokens are only reachable indirectly; see [Longformer](../../papers/architectures/131-longformer/summary.md) and [Mistral 7B](../../papers/language-models/95-mistral-7b/summary.md) |
| **Recurrent / state-space layers** | Replace the growing cache with a fixed-size state | Weaker exact recall of long contexts; see [Mamba](../../papers/architectures/20-mamba/summary.md) |
| **Efficient attention kernels** | Do not shrink the cache but read it faster | Engineering only; see [FlashAttention](../../papers/techniques/16-flash-attention/summary.md) |

## How the cache connects to everything else

- **Long context is a memory problem as much as a modelling one.** Extending position encodings (see [context windows](context-windows.md)) makes a model able to handle 1M tokens; the KV cache decides whether you can afford to serve it.
- **Reasoning models make it worse.** Thinking tokens are generated tokens, and each one adds to the cache. A model that thinks for 20,000 tokens before answering holds a 20,000-token cache for the whole request (see [reasoning models](reasoning-models.md)).
- **Speculative decoding** uses a small draft model to propose several tokens that the big model checks in one pass, which amortises the cost of reading the cache and weights across several tokens (see [sampling and decoding](sampling-and-decoding.md) and the [speculative decoding summary](../../papers/techniques/45-speculative-decoding/summary.md)).
- **Pricing.** Input tokens that hit a prompt cache are billed at a fraction of the normal rate by major providers (as of September 2026 Anthropic lists cache reads at 10% of base input price for most models), which is a direct pass-through of KV cache reuse. See [inference economics](../compute/inference-economics.md).

## What to watch

- **Which cache-shrinking design wins.** GQA is the safe default; MLA-style compression has also been used outside DeepSeek (Moonshot's Kimi K2 configuration, for example, carries the same 512-dimensional compressed KV latent). Watch new open model configurations for `num_key_value_heads` and latent dimensions.
- **Hybrid architectures.** Models that mix attention layers with fixed-state layers cut the cache for most of the network while keeping some full attention for recall.
- **Cache offloading and disaggregated serving.** Moving caches between GPU memory, CPU memory and storage, and splitting prefill and decode onto different machines, are active areas in serving engines.

## Read next

- [GQA: Grouped-Query Attention](../../papers/architectures/75-grouped-query-attention/summary.md)
- [Multi-head Latent Attention (DeepSeek-V2)](../../papers/architectures/141-multi-head-latent-attention/summary.md)
- [PagedAttention and vLLM](../../papers/techniques/52-pagedattention-vllm/summary.md)
- [Speculative Decoding](../../papers/techniques/45-speculative-decoding/summary.md)
- [FlashAttention](../../papers/techniques/16-flash-attention/summary.md)
- [Context windows](context-windows.md), [Sampling and decoding](sampling-and-decoding.md), [Inference economics](../compute/inference-economics.md), [AI hardware landscape](../compute/ai-hardware-landscape.md)
- Hands-on: [GPUs for AI](https://github.com/PatrickWiloak/cloud-data-ai-security-zero-to-hero/blob/main/learn/concepts/gpus-for-ai.md), [Inference servers](https://github.com/PatrickWiloak/cloud-data-ai-security-zero-to-hero/blob/main/learn/concepts/inference-servers.md)

## Sources

- Llama 3.3 70B Instruct configuration (80 layers, 64 attention heads, 8 KV heads, head_dim 128, BF16, 131,072 max positions). https://huggingface.co/meta-llama/Llama-3.3-70B-Instruct (mirrored config read from https://huggingface.co/unsloth/Llama-3.3-70B-Instruct/raw/main/config.json)
- DeepSeek-V3 configuration (61 layers, `kv_lora_rank` 512, `qk_rope_head_dim` 64). https://huggingface.co/deepseek-ai/DeepSeek-V3/raw/main/config.json
- DeepSeek-AI, "DeepSeek-V2: A Strong, Economical, and Efficient Mixture-of-Experts Language Model", 2024 (93.3% KV cache reduction, 5.76x throughput). https://arxiv.org/abs/2405.04434
- Ainslie et al., "GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints", EMNLP 2023. https://arxiv.org/abs/2305.13245
- Shazeer, "Fast Transformer Decoding: One Write-Head is All You Need", 2019. https://arxiv.org/abs/1911.02150
- Kwon et al., "Efficient Memory Management for Large Language Model Serving with PagedAttention", SOSP 2023 (60-80% waste, under 4% with paging). https://arxiv.org/abs/2309.06180
- Moonshot AI, Kimi K2 Instruct configuration (`kv_lora_rank` 512). https://huggingface.co/moonshotai/Kimi-K2-Instruct/raw/main/config.json
- Anthropic, Models overview (prompt cache read pricing), accessed 2026-09-30. https://platform.claude.com/docs/en/about-claude/models/overview
- vLLM documentation, "Automatic Prefix Caching" design notes. https://docs.vllm.ai/en/latest/design/prefix_caching.html
- vLLM documentation, "Quantized KV Cache" (FP8). https://docs.vllm.ai/en/latest/features/quantization/quantized_kvcache.html
