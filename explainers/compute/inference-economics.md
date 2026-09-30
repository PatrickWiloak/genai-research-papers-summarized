# Inference Economics

**In one line:** The price of a given level of AI capability has been falling by roughly 10x or more per year, but your actual bill depends on which tokens you buy (input, output, cached, reasoning), so read a price sheet as four prices, not one.
**Last reviewed:** 2026-09-30

---

## The short version

- **Inference** is running a trained model to answer requests. Over a model's life it usually costs far more in total than training it, because it happens billions of times.
- **Price for constant capability has collapsed.** a16z measured about 10x per year (2021-2024); Epoch AI found declines of 9x to 900x per year depending on the task (as of March 2025).
- **Output tokens cost about 5x input tokens** on the main price lists, because generating text is sequential and memory-bound, while reading a prompt is parallel.
- **Cached input is the cheapest token.** Reusing a prompt prefix costs 10% of the normal input price or less on current price lists.
- **Reasoning models change the arithmetic.** Their hidden "thinking" is billed as output, so the per-token price can fall while the per-task cost rises.
- **Serving efficiency is the engine underneath:** batching, KV-cache management, quantization, speculative decoding and better chips all lower cost per token.

## The mental model: prefill is cheap, decode is expensive

Every request has two phases:

```
PREFILL (read the prompt)                 DECODE (write the answer)
------------------------------------      ------------------------------------
all prompt tokens processed at once       one token at a time, each one
  -> big matrix multiplies                  depending on the last
  -> chip is compute-bound                -> must re-read all model weights
  -> high utilisation, cheap per token       for every token
                                          -> chip is memory-bandwidth-bound
                                          -> low utilisation, expensive per token
```

That asymmetry is why output tokens are priced several times higher than input tokens. It is also why most of the engineering in serving goes into making decode cheaper.

The trick that makes decode affordable is **batching**: while the weights are streaming through the chip for one user's next token, the chip can compute the next token for many other users almost for free. Memory bandwidth is spent once; the arithmetic is shared. The more users served per weight read, the lower the cost per token, up to the point where memory for each user's conversation state runs out.

That per-user state is the **KV cache** (the stored attention keys and values for every token so far; see [KV Cache](../concepts/kv-cache.md)). Long contexts and many simultaneous users both grow it, and when it fills the chip's memory, batching stops helping.

## How providers price tokens

As of September 2026, the three largest API providers all publish separate prices for at least input, cached input and output.

### Anthropic (Claude API list prices, per million tokens)

| Model | Input | Cache write (5 min) | Cache hit | Output |
|---|---|---|---|---|
| Claude Opus 5.5 | $4 | $5 | $0.20 | $20 |
| Claude Sonnet 5.5 | $2 | $2.50 | $0.20 | $10 |
| Claude Haiku 4.5 | $1 | $1.25 | $0.10 | $5 |
| Claude Opus 4.1 (older, for comparison) | $15 | $18.75 | $1.50 | $75 |

Batch processing (results within a day rather than immediately) is 50% off input and output. Source: Claude pricing page, read 2026-09-30.

### OpenAI (API list prices, per million tokens, standard tier, short context)

| Model | Input | Cached input | Cache writes | Output |
|---|---|---|---|---|
| gpt-6-astra | $10.00 | $1.00 | $12.50 | $50.00 |
| gpt-6.1-sol | $2.00 | $0.10 | $2.50 | $10.00 |
| gpt-6-luna | $0.10 | $0.01 | $0.125 | $0.50 |

OpenAI also sells Batch and Flex tiers (discounted, slower) and Fast and Ultrafast tiers (premium, lower latency). Long-context requests cost more. Source: OpenAI API pricing page, read 2026-09-30.

### Google (Gemini API paid tier, per million tokens)

| Model | Input | Output (including thinking tokens) | Context caching |
|---|---|---|---|
| Gemini 3.1 Pro Preview | $2.00 (up to 200K-token prompts), $4.00 above | $12.00, $18.00 above 200K | listed separately |
| Gemini 3.8 Flash | $0.75 until 2026-12-31, then $1.50 | $3.75 until 2026-12-31, then $7.50 | $0.075, then $0.15 |
| Gemini 3.5 Flash | $1.50 | $9.00 | $0.15 |

Google also charges an hourly storage fee for explicitly cached content. Source: Gemini API pricing page, read 2026-09-30.

### Reading these tables

- **Output is 5x input** across the flagship rows above (Anthropic and OpenAI) and 6x for Gemini 3.1 Pro. That ratio reflects the prefill/decode asymmetry.
- **Cached input is 90% or more off** normal input (between 90% and 97.5% off in the tables above and Anthropic's full list). For an agent or chatbot that resends the same long system prompt and history every turn, caching is usually the single largest saving. The sibling repo's [Prompt Caching](https://github.com/PatrickWiloak/cloud-data-ai-security-zero-to-hero/blob/main/learn/concepts/prompt-caching.md) page covers how to use it.
- **Tokens are not the same size across vendors or even model versions.** Anthropic notes that its tokenizer for Claude 4.7 and later produces about 30% more tokens for the same text than the earlier one. A lower price per token can be partly offset by more tokens per page. See [Tokenization](../concepts/tokenization.md).
- **Within one family, prices drop between generations.** Claude's top tier went from $15/$75 (Opus 4.1) to $5/$25 (Opus 4.5 through Opus 5) to $4/$20 (Opus 5.5), per Anthropic's price list.

## Why the price per token keeps falling

Two independent measurements, using different methods, agree on the direction:

- **a16z, "LLMflation" (November 2024):** for a model of equivalent performance, cost fell about 10x per year over three years. A model scoring what GPT-3 scored cost $60 per million tokens in November 2021; by November 2024 the cheapest model reaching that score (Llama 3.2 3B) cost $0.06, a 1,000x drop.
- **Epoch AI (March 2025):** measuring the price to reach fixed benchmark thresholds, declines ranged from 9x to 900x per year depending on the task, with the fastest drops most recent. Epoch cautions that the fastest rates may not persist.

The causes stack on top of each other:

| Lever | What it does | Where to read more |
|---|---|---|
| Better chips | More memory bandwidth and lower-precision arithmetic per dollar | [The AI Hardware Landscape](ai-hardware-landscape.md) |
| Smaller models, same quality | Distillation and better data let a small model match an old large one | [Knowledge Distillation](../../papers/techniques/134-knowledge-distillation/summary.md), [phi-1](../../papers/language-models/135-phi-1-textbooks/summary.md) |
| Mixture-of-experts | Only a fraction of parameters run per token | [Mixture of Experts](../../papers/architectures/37-mixture-of-experts/summary.md) |
| Quantization | Store weights in 8 or 4 bits instead of 16, halving or quartering memory traffic | [GPTQ and AWQ](../../papers/techniques/86-gptq-awq-quantization/summary.md) |
| Attention kernels | Fewer trips to slow memory during attention | [FlashAttention](../../papers/techniques/16-flash-attention/summary.md) |
| KV-cache paging and continuous batching | Pack more users onto one GPU without wasting memory | [PagedAttention / vLLM](../../papers/techniques/52-pagedattention-vllm/summary.md) |
| Smaller KV cache | Share or compress keys and values | [Grouped-Query Attention](../../papers/architectures/75-grouped-query-attention/summary.md), [Multi-head Latent Attention](../../papers/architectures/141-multi-head-latent-attention/summary.md) |
| Speculative decoding | A small draft model proposes several tokens; the big model checks them in one pass | [Speculative Decoding](../../papers/techniques/45-speculative-decoding/summary.md) |
| Competition | Open-weight models served by many hosts compress margins | [Open vs Closed Weights](../concepts/open-vs-closed-weights.md) |

Two of these deserve a closer look.

**Continuous batching and paged KV cache.** Early servers batched requests in fixed groups and reserved memory for each request's maximum length, wasting much of it. [vLLM's PagedAttention](../../papers/techniques/52-pagedattention-vllm/summary.md) stores the KV cache in small pages, like an operating system's virtual memory, and lets new requests join a running batch as soon as others finish. The paper reported 2-4x higher throughput than the previous state of the art at the same latency. Every major serving engine now does some version of this.

**Disaggregated serving.** Because prefill and decode stress the hardware differently, large deployments now run them on separate pools of GPUs and move the KV cache between them. NVIDIA's open-source Dynamo (March 2025) and the llm-d project (May 2025) are built around this idea. See [Open-Source Stack](../ecosystem/open-source-stack.md).

## Reasoning tokens change the picture

[Reasoning models](../concepts/reasoning-models.md) such as [OpenAI o1](../../papers/language-models/31-openai-o1/summary.md) and [DeepSeek-R1](../../papers/language-models/26-deepseek-r1/summary.md) think before answering by generating long hidden chains of thought. Providers bill those as output tokens:

- OpenAI's reasoning guide: reasoning tokens "are not visible via the API" but "occupy space in the model's context window and are billed as output tokens", and a model may generate "a few hundred to tens of thousands" of them per request.
- Google's price list labels output price as "including thinking tokens".
- Anthropic reports the thinking share of billed output tokens in a separate usage field.

The consequence, in rough terms:

```
Non-reasoning answer:   500 input  +    300 output tokens
Reasoning answer:       500 input  +  8,000 reasoning  +  300 output tokens
                                      (billed as output)
```

With output priced at 5x input, the reasoning answer can cost more than 20 times as much as the plain one, even on a model with a lower per-token price. This is the core tension of 2025-2026 inference economics: **price per token fell, tokens per task rose.** Agents amplify it further, because each step of an agent loop re-sends the growing context (input, hopefully cached) and generates new reasoning (output).

The practical levers are the ones providers now expose: effort or budget settings that cap thinking, choosing a smaller model for easy steps, and caching the stable parts of long agent contexts. [Test-time compute](../../papers/techniques/50-test-time-compute/summary.md) research explains why spending more tokens on hard problems can be worth it, and why spending them on easy ones is waste.

## A worked cost comparison

For a support chatbot handling one million conversations a month, each with a 3,000-token system prompt and history, 200 new user tokens and 300 output tokens, on a model priced at $2 input, $0.20 cache hit and $10 output per million tokens (Claude Sonnet 5.5's list price, September 2026):

```
Without caching:
  input   3,200 x 1M = 3.2B tokens x $2    = $6,400
  output    300 x 1M = 0.3B tokens x $10   = $3,000
  total                                    = $9,400

With the 3,000-token prefix cached (ignoring cache-write cost):
  cached  3,000 x 1M = 3.0B x $0.20        =   $600
  fresh     200 x 1M = 0.2B x $2           =   $400
  output                                   = $3,000
  total                                    = $4,000
```

Caching more than halves the bill, and output becomes three-quarters of what is left. Real numbers depend on cache hit rates and write costs; the point is which line items dominate.

## What to watch

- **Rubin-generation hardware (from H2 2026).** NVIDIA claims up to 10x lower inference token cost versus Blackwell; independent measurements will show how much reaches API prices.
- **Promotional pricing ending.** Google's Gemini 3.8 Flash list price doubles on 2027-01-01 per its pricing page.
- **Reasoning token growth.** Whether effort controls and better training reduce thinking length faster than agents increase task length.
- **Cache pricing competition.** Cache-hit discounts have deepened (Anthropic's top tiers price hits below the standard 10% of input: 2.5% for Fable 5.1, 5% for Opus 5.5); watch whether this becomes the main axis of price competition.
- **Epoch AI's price-trend updates.** The best public check on whether the 10x-per-year pattern still holds.

## Read next

- [PagedAttention / vLLM](../../papers/techniques/52-pagedattention-vllm/summary.md) - the serving breakthrough behind cheap batching
- [FlashAttention](../../papers/techniques/16-flash-attention/summary.md) - memory-aware attention
- [Speculative Decoding](../../papers/techniques/45-speculative-decoding/summary.md) - faster decode without changing output
- [GPTQ and AWQ](../../papers/techniques/86-gptq-awq-quantization/summary.md) - quantization for inference
- [KV Cache](../concepts/kv-cache.md), [Context Windows](../concepts/context-windows.md), [Reasoning Models](../concepts/reasoning-models.md)
- [The Cost of Training](cost-of-training.md) and [The AI Hardware Landscape](ai-hardware-landscape.md)
- Sibling repo: [Inference Servers](https://github.com/PatrickWiloak/cloud-data-ai-security-zero-to-hero/blob/main/learn/concepts/inference-servers.md), [Prompt Caching](https://github.com/PatrickWiloak/cloud-data-ai-security-zero-to-hero/blob/main/learn/concepts/prompt-caching.md), [Quantization and Distillation](https://github.com/PatrickWiloak/cloud-data-ai-security-zero-to-hero/blob/main/learn/concepts/quantization-and-distillation.md)

## Sources

- Anthropic, Claude API pricing (read 2026-09-30): https://platform.claude.com/docs/en/about-claude/pricing
- Anthropic, extended thinking documentation (thinking tokens in usage): https://platform.claude.com/docs/en/build-with-claude/extended-thinking
- OpenAI, API pricing (read 2026-09-30): https://developers.openai.com/api/docs/pricing
- OpenAI, reasoning models guide (reasoning tokens billed as output): https://developers.openai.com/api/docs/guides/reasoning
- Google, Gemini API pricing (read 2026-09-30): https://ai.google.dev/gemini-api/docs/pricing
- a16z, "Welcome to LLMflation - LLM inference cost is going down fast" (November 12, 2024): https://a16z.com/llmflation-llm-inference-cost/
- Epoch AI, "LLM inference prices have fallen rapidly but unequally across tasks" (March 12, 2025): https://epoch.ai/data-insights/llm-inference-price-trends
- Kwon et al., "Efficient Memory Management for Large Language Model Serving with PagedAttention" (2023): https://arxiv.org/abs/2309.06180
- NVIDIA, "NVIDIA Dynamo Open-Source Library Accelerates and Scales AI Reasoning Models" (March 18, 2025): https://www.globenewswire.com/news-release/2025/03/18/3044894/0/en/NVIDIA-Dynamo-Open-Source-Library-Accelerates-and-Scales-AI-Reasoning-Models.html
- Red Hat, "Red Hat Launches the llm-d Community" (May 20, 2025): https://www.redhat.com/en/about/press-releases/red-hat-launches-llm-d-community-powering-distributed-gen-ai-inference-scale
- NVIDIA Newsroom, Rubin platform (January 5, 2026): https://nvidianews.nvidia.com/news/rubin-platform-ai-supercomputer
