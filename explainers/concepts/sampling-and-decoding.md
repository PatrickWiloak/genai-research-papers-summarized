# Sampling and Decoding

**In one line:** A language model only outputs a probability for every possible next token; the decoding strategy (greedy, temperature, top-k, top-p, min-p, beam search, constrained decoding) is the separate rule that turns those probabilities into actual text, and it changes the output as much as many model upgrades do.
**Last reviewed:** 2026-09-30

---

## The short version

- At each step the model produces a score (a **logit**) for every token in its vocabulary. A softmax turns the scores into probabilities. **Decoding** is whatever rule picks the next token from that distribution. The model itself does not choose.
- **Greedy** decoding always takes the most likely token. It is predictable but tends to be dull and repetitive in open-ended writing.
- **Sampling** picks randomly in proportion to probability. **Temperature** sharpens or flattens the distribution; **top-k**, **top-p (nucleus)** and **min-p** cut off the unreliable long tail before sampling.
- **Beam search** keeps several candidate sequences and returns the most probable one overall. It suits translation and other tasks with one right answer, not chat.
- **Constrained (structured) decoding** masks out any token that would break a required format such as a JSON schema or grammar, so the output is guaranteed to parse.
- **Speculative decoding** does not change what gets generated; it changes how fast, by letting a small model draft tokens that the big model verifies in bulk.

## From logits to a token

```
context: "The capital of France is"

model -> logits over ~100,000+ tokens
softmax -> probabilities:
    " Paris"   0.92
    " the"     0.03
    " a"       0.01
    " located" 0.01
    ... tens of thousands more, each tiny

decoding rule -> pick one -> append -> repeat
```

The same probabilities can yield very different text depending on the rule. That is why the same model can feel crisp in one app and rambling in another, and why API parameters like `temperature` and `top_p` exist at all. For the model side of this picture, see the sibling repo's [LLM basics](https://github.com/PatrickWiloak/cloud-data-ai-security-zero-to-hero/blob/main/learn/concepts/llm-basics.md); for what counts as a "token", see [tokenization](tokenization.md).

## Greedy decoding

Take the highest-probability token every time. It is deterministic in principle and works well when there is one right answer (short factual replies, classification, code completion with tight context).

Its weakness shows up in longer free text. Holtzman et al. (2019) documented what they called **neural text degeneration**: maximising likelihood at decoding time produces text that is "bland and strangely repetitive", often looping on the same phrase. High-probability text is not the same as human-like text; people routinely choose words that are not the single most likely one.

## Temperature

Temperature divides the logits before the softmax.

```
p_i = exp(logit_i / T) / sum_j exp(logit_j / T)

T < 1  -> distribution sharpens; top tokens get even more likely (more focused)
T = 1  -> the model's own distribution
T > 1  -> distribution flattens; unlikely tokens get a real chance (more varied, more errors)
T -> 0 -> equivalent to greedy
```

Low temperature suits extraction, coding and factual answers; higher temperature suits brainstorming and fiction. Temperature alone has a flaw: raising it to get variety also raises the odds of picking from the tens of thousands of junk tokens in the tail. That is what the truncation methods below are for.

**A note on determinism.** Temperature 0 does not guarantee identical outputs from a hosted API. Thinking Machines Lab (September 2025) traced most of this to a lack of **batch invariance**: a server's numerical results can change with how many other requests are batched alongside yours, and load varies from moment to moment. They showed that kernels designed to be batch-invariant restore reproducibility.

## Truncation: top-k, top-p, min-p

All three do the same thing in different ways: throw away the unlikely tail, renormalise, then sample from what is left.

| Method | Rule | Strength | Weakness |
|---|---|---|---|
| **Top-k** | Keep the k most likely tokens | Simple | A fixed k is too many when the model is sure and too few when it is not |
| **Top-p (nucleus)** | Keep the smallest set of tokens whose probabilities add up to p (for example 0.9) | Adapts: a confident step keeps 1-2 tokens, an open step keeps many | At high temperature the flattened tail can still sneak into the nucleus |
| **Min-p** | Keep tokens whose probability is at least `min_p x` the top token's probability | Scales the cut-off with the model's confidence; stays coherent at higher temperatures | Newer, less studied than top-p |

```
Confident step:                 Uncertain step:
" Paris" 0.92                   " quickly" 0.12
" the"   0.03                   " slowly"  0.10
...                             " then"    0.09 ... (flat)

top-k=40:  keeps 40 either way (bad in the confident case)
top-p=0.9: keeps 1 token        | keeps dozens
min-p=0.1: keeps tokens >= 0.092 | keeps tokens >= 0.012
```

**Top-p** was introduced as **nucleus sampling** by Holtzman et al. in "The Curious Case of Neural Text Degeneration" (ICLR 2020) and is exposed as the `top_p` parameter by most chat APIs. **Min-p** was proposed by Nguyen et al. and presented as an oral at ICLR 2025; the authors report gains across models from 1B to 123B parameters, and it is implemented in Hugging Face Transformers and vLLM.

In practice most applications leave these at the provider's defaults and adjust temperature. Common advice is to change either temperature or top-p, not both, because they interact.

Two other knobs you will see:

- **Repetition / frequency / presence penalties** lower the score of tokens that have already appeared, a blunt fix for looping.
- **Logit bias** adds a fixed boost or penalty to chosen token IDs, for example to ban a word.

## Beam search

Beam search keeps the **B** best partial sequences (the "beam") at each step, extends each by every possible token, and keeps the B best of the results, scored by total probability.

```
B = 2
step 1:  [The]  [A]
step 2:  [The cat] [The dog]            (best 2 of all extensions)
step 3:  [The cat sat] [The dog ran]
...      return the highest-probability complete sequence
```

It finds sequences with higher overall likelihood than greedy decoding, which is why it was the standard for machine translation and summarisation in the [sequence-to-sequence](../../papers/architectures/55-seq2seq/summary.md) era. For open-ended generation it inherits, and often amplifies, the degeneration problem: the most likely text is generic and repetitive. It also costs B times the compute and memory. Modern chat models rarely use it.

A related family of ideas spends extra inference compute on **multiple full samples** instead of a beam: sample several answers and take the majority ([self-consistency](../../papers/techniques/77-self-consistency/summary.md)), or score candidates with a verifier ([process reward models](../../papers/techniques/51-process-reward-models/summary.md)). This is part of the test-time compute story covered in [reasoning models](reasoning-models.md).

## Structured and constrained decoding

Often the output must be machine-readable: valid JSON, a function call matching a schema, a SQL query, one of five labels. Asking nicely in the prompt works most of the time. **Constrained decoding** makes it work every time.

The idea: before sampling each token, compute which tokens are allowed given the format and everything generated so far, and set every other token's probability to zero.

```
Schema: {"name": string, "age": integer}
Generated so far:  {"name": "Ada", "age":
Allowed next tokens:  digits, whitespace, "-"      (no letters, no quotes)
```

Willard and Louf (2023) showed this can be done efficiently by compiling a regular expression or grammar into a finite-state machine and pre-indexing which vocabulary tokens are valid in each state. Their open-source library, **Outlines**, and similar grammar features in other engines brought this to local models. On the API side, OpenAI launched **Structured Outputs** in August 2024, combining a model trained to follow schemas with constrained decoding, and reported 100% schema adherence on its complex-schema evaluation versus under 40% for an older model without it. Other providers followed; Anthropic, for example, documents structured outputs that use constrained sampling with compiled grammars to guarantee schema-valid JSON. See also the sibling repo's [tool use and function calling](https://github.com/PatrickWiloak/cloud-data-ai-security-zero-to-hero/blob/main/learn/concepts/tool-use-and-function-calling.md).

The trade-off: a valid format is not a correct answer, and constraints can hurt reasoning. Tam et al. (2024), in "Let Me Speak Freely?", found that forcing formats like JSON during generation measurably reduced reasoning performance on some tasks compared with free-form answers. A common pattern is to let the model reason freely first and constrain only the final answer.

## Speculative decoding: faster, not different

Generation is slow mostly because every step must read the whole model and its [KV cache](kv-cache.md) from memory to produce just one token. **Speculative decoding** (Leviathan, Kalman and Matias, 2022) uses a small, fast **draft** model to guess several tokens ahead, then runs the large model once over all of them in parallel to check.

```
draft model proposes:  [the] [cat] [sat] [on] [a]
big model checks all 5 in ONE forward pass:
    accept [the] [cat] [sat] [on], reject [a] -> resample "the"
result: 5 tokens for about the cost of 1 big-model step (+ cheap drafts)
```

A modified accept/reject rule guarantees that the output follows exactly the same distribution the large model would have produced on its own, so quality is unchanged. The speed-up depends on how often the draft is right. Variants replace the separate draft model with extra prediction heads or with the big model's own earlier layers, and some open-weight releases now ship with a matching drafter (Meta's Muse Glimmer did in August 2026). See the [speculative decoding summary](../../papers/techniques/45-speculative-decoding/summary.md).

## Choosing settings

| Task | Typical starting point |
|---|---|
| Extraction, classification, code edits | Low temperature (0 to 0.3), or constrained output |
| Chat and explanation | Provider defaults |
| Brainstorming, fiction | Higher temperature with top-p or min-p to keep it coherent |
| Machine-readable output | Structured outputs / constrained decoding, reason first if the task is hard |
| Reasoning models | Follow the provider's guidance; sampling controls are often fixed (see below) |

Reasoning and "thinking" models increasingly take these knobs away. As of September 2026, Anthropic's API returns an error for any non-default `temperature`, `top_p` or `top_k` on its newest Claude models, and on older models restricts them whenever thinking is on. The provider tunes decoding for the long internal chain and you steer with an effort setting instead (see [reasoning models](reasoning-models.md)).

These are starting points, not rules. Measure on your own task; the hands-on [evals for LLMs](https://github.com/PatrickWiloak/cloud-data-ai-security-zero-to-hero/blob/main/learn/concepts/evals-for-llms.md) page covers how.

## What to watch

- **Reproducibility.** Batch-invariant kernels make bit-identical outputs possible; watch whether hosted APIs start offering them.
- **Decoding for reasoning.** As models do more of their work in long hidden chains, the choice between one long chain, many parallel samples and verifier-guided search is becoming a decoding question as much as a training one.
- **Grammar-constrained generation everywhere.** Schema-constrained output has gone from a local-model trick to a standard API feature; the open question is how to constrain without hurting reasoning.

## Read next

- [Speculative Decoding](../../papers/techniques/45-speculative-decoding/summary.md)
- [Self-Consistency](../../papers/techniques/77-self-consistency/summary.md) and [Tree of Thoughts](../../papers/techniques/25-tree-of-thoughts/summary.md) - spending decoding compute on search
- [GPT-2](../../papers/language-models/64-gpt2/summary.md) - open-ended generation where sampling choices first became visible to the public
- [Tokenization](tokenization.md), [KV cache](kv-cache.md), [Reasoning models](reasoning-models.md)
- Hands-on: [Prompt engineering](https://github.com/PatrickWiloak/cloud-data-ai-security-zero-to-hero/blob/main/learn/concepts/prompt-engineering.md), [Tool use and function calling](https://github.com/PatrickWiloak/cloud-data-ai-security-zero-to-hero/blob/main/learn/concepts/tool-use-and-function-calling.md)

## Sources

- Holtzman, Buys, Du, Forbes, Choi, "The Curious Case of Neural Text Degeneration", ICLR 2020. https://arxiv.org/abs/1904.09751
- Nguyen et al., "Turning Up the Heat: Min-p Sampling for Creative and Coherent LLM Outputs", ICLR 2025. https://arxiv.org/abs/2407.01082
- Willard, Louf, "Efficient Guided Generation for Large Language Models", 2023. https://arxiv.org/abs/2307.09702
- OpenAI, "Introducing Structured Outputs in the API", August 2024. https://openai.com/index/introducing-structured-outputs-in-the-api/
- Tam et al., "Let Me Speak Freely? A Study on the Impact of Format Restrictions on Performance of Large Language Models", 2024. https://arxiv.org/abs/2408.02442
- He and Thinking Machines Lab, "Defeating Nondeterminism in LLM Inference", September 2025. https://thinkingmachines.ai/blog/defeating-nondeterminism-in-llm-inference/
- Anthropic, "Thinking" documentation (sampling parameter restrictions), accessed 2026-09-30. https://platform.claude.com/docs/en/build-with-claude/thinking
- Anthropic, "Structured outputs" documentation, accessed 2026-09-30. https://platform.claude.com/docs/en/build-with-claude/structured-outputs
- Hugging Face, "Meta is back with Muse Glimmer", August 2026 (ships with a speculative-decoding drafter). https://huggingface.co/blog/muse-glimmer
- Leviathan, Kalman, Matias, "Fast Inference from Transformers via Speculative Decoding", ICML 2023. https://arxiv.org/abs/2211.17192
