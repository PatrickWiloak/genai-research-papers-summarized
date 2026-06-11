---
title: "Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity"
slug: "67-switch-transformer"
number: 67
category: "architectures"
authors: "William Fedus, Barret Zoph, Noam Shazeer (Google)"
published: "January 2021 (JMLR 2022)"
year: 2021
url: "https://arxiv.org/abs/2101.03961"
tags: ["moe", "architecture", "scaling"]
---

# Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity

**Authors:** William Fedus, Barret Zoph, Noam Shazeer (Google)

**Published:** January 2021 (JMLR 2022)

**Paper Link:** https://arxiv.org/abs/2101.03961

---

## Why This Paper Matters

The Switch Transformer cracked open a door that was previously too expensive to walk through: it showed that you can build a **trillion-parameter model** while keeping the compute per token roughly the same as a model ten to a hundred times smaller. It did this by making each token activate only a tiny slice of the network's parameters - a technique called **sparse Mixture-of-Experts (MoE)** - and it simplified earlier MoE designs dramatically to make them stable and fast enough to actually train at scale. Every major sparse LLM that came after (Mixtral, DeepSeek-V3, GPT-4's rumored MoE structure) owes its DNA to ideas sharpened here.

---

## The Core Innovation: Top-1 Routing (The "Switch")

### The Problem Switch Transformers Solved

Dense Transformers activate **every single parameter** for every token. That is fine when your model has 1 billion parameters, but scaling to trillions would make inference and training prohibitively expensive. The classic solution was Mixture-of-Experts, where the model has multiple "expert" feed-forward networks and each token only uses a few of them. But prior MoE work (like Shazeer et al. 2017) used **top-2 routing** - each token picks its two best experts. This added complexity, communication overhead, and training instability.

The Switch Transformer's key insight was: **why not just use one expert?**

### Top-1 Routing - The "Switch"

The name "Switch" comes from the router behaving like an electrical switch: each token is routed to **exactly one expert** and no others.

```
Token arrives at a Transformer layer
         |
  Router (small linear layer + softmax)
         |
  Scores computed for each of N experts
         |
  Token sent to the single HIGHEST-scoring expert
         |
  That expert processes the token and returns output
  (All other experts ignored for this token)
```

This is simpler than top-2, cheaper to compute, and - somewhat surprisingly - works just as well or better in practice.

---

## Key Components Explained

### 1. The Router

The router is a small learned linear layer sitting at the entrance of each MoE layer. For a token with hidden state `x` and `N` available experts, it computes:

```
Router scores = softmax(x · W_r)   [W_r is a learned N-dim weight matrix]
Chosen expert = argmax(scores)
```

Think of it like a hotel concierge who has learned, from millions of check-ins, to match each guest's profile to exactly the right room specialist - a fluent French speaker goes to the French-language desk, a business traveler goes to the corporate desk, and so on. Each guest (token) gets one desk (expert), and that expert has become very good at their particular slice of requests.

### 2. Expert Parallelism

The real payoff of sparsity is in hardware. With a dense model you must fit all parameters on one device or use expensive tensor parallelism. With Switch Transformers you can spread experts across many devices:

```
Device 1: Experts 1,  2,  3,  4
Device 2: Experts 5,  6,  7,  8
Device 3: Experts 9,  10, 11, 12
Device 4: Experts 13, 14, 15, 16
```

Each device only processes the tokens routed to its experts. Communication cost is modest: tokens travel to their expert's device, get processed, and come back. This lets parameter count grow with device count almost for free in terms of FLOPs per token.

```
+--------------------------------------------------------+
|  Switch Transformer Layer (replaces one FFN sublayer)  |
|                                                        |
|  Tokens:  T1  T2  T3  T4  T5  T6  T7  T8             |
|             |   |   |   |   |   |   |   |             |
|           [         Router          ]                  |
|           /    |    |    \    ...                      |
|         E1    E2   E3    E4   ... EN                   |
|  (each expert is a full FFN on its own device)         |
|                                                        |
|  T1->E3,  T2->E1,  T3->E3,  T4->E7, ...               |
+--------------------------------------------------------+
```

### 3. Load Balancing and the Auxiliary Loss

Left to its own devices, the router collapses: it learns to send nearly all tokens to one or two popular experts, leaving the rest idle (called "expert collapse"). This wastes capacity and creates bottlenecks on the overloaded expert's device.

The fix is an **auxiliary load-balancing loss** added to the training objective:

```
L_aux = alpha * N * sum_i (f_i * P_i)

where:
  N     = number of experts
  f_i   = fraction of tokens actually dispatched to expert i
  P_i   = mean router probability assigned to expert i
  alpha = small hyperparameter (e.g. 0.01)
```

This penalizes imbalance: if expert i receives a disproportionate share of tokens, both `f_i` and `P_i` are high, making the product large and raising the loss. The router is nudged to spread load more evenly while still being free to specialize.

### 4. Capacity Factor and Token Dropping

Even with the auxiliary loss, some experts get overloaded in a given batch. The **capacity factor** (CF) controls a hard buffer:

```
Expert capacity = (tokens per batch / num experts) * capacity_factor
```

A CF of 1.0 means each expert gets exactly its fair share of tokens. A CF of 1.25 gives each expert a 25% overflow buffer. Tokens that arrive at a full expert are **dropped** - they skip that layer entirely and pass through unchanged via a residual connection.

Token dropping sounds alarming, but in practice a well-tuned auxiliary loss keeps drops rare, and the residual bypass means the model degrades gracefully rather than crashing.

### 5. Selective Precision and Training Stability

Sparse routing introduces instability that dense models do not face: routing decisions are discrete-like (argmax), and early in training the router has not yet found a good assignment. The paper introduces several tricks to stabilize training:

- **BFloat16 for most computations** - faster and more memory efficient than FP32
- **FP32 for the router itself** - the routing softmax and loss terms need the extra precision to avoid instability from tiny probability differences
- **Router z-loss** (a small penalty on the log-partition function of the router) - discourages very large router logits that cause overflow
- **Careful initialization** - smaller initial weights for the router reduce early routing chaos

---

## Key Results

| Model | Parameters | FLOPs/token | Speed vs T5-Base |
|-------|-----------|-------------|-----------------|
| T5-Base (dense) | 223M | baseline | 1x |
| Switch-Base (128 experts) | 7.4B | same as T5-Base | **7x faster** to same quality |
| Switch-XXL | ~395B | same as T5-XXL | strong improvements |
| Switch-C | ~1.6T | similar to T5-XXL | best-in-class quality |

Key benchmarks (SuperGLUE, multilingual translation, fine-tuning tasks):
- Switch-Base reached T5-Base quality **7x faster** in pre-training
- Switch models consistently outperformed dense baselines given **equal compute budgets**
- Scaling experts from 2 to 2048 per layer continued to improve quality log-linearly

---

## Why This Was Revolutionary

### 1. Decoupled Parameters from Compute

Before Switch Transformers, "more parameters" directly meant "more FLOPs per token." Switch broke that coupling: you can have 100x more parameters while spending only slightly more compute per token (routing overhead). This is a fundamental shift in how to think about model scale.

### 2. Simpler than Prior MoE

Earlier MoE work used top-2 routing, noisy gating, and complex routing schemes. Top-1 routing proved that the simplest possible routing - "just pick one" - was not only sufficient but often better due to reduced communication cost and easier load balancing.

### 3. Practical Trillion-Parameter Training

Prior to this paper, trillion-parameter models existed only as theoretical proposals. Switch-C at 1.6T parameters was actually trained and benchmarked, demonstrating this was an engineering reality, not just a scaling law thought experiment.

### 4. Fixed Compute Budget Comparison

By controlling for FLOPs per token rather than parameter count, the paper showed sparse models beat dense models on a level playing field. This changed the framing for subsequent scaling research.

---

## Impact and Descendants

Switch Transformers established the template that later MoE LLMs refined and deployed at production scale:

- **Mixtral 8x7B / 8x22B (Mistral AI, 2023-2024)** - sparse MoE decoder with top-2 routing, the first widely deployed open MoE LLM. See: [Mixture of Experts and Mixtral summary](../../architectures/37-mixture-of-experts/summary.md)
- **DeepSeek-V3 (2024)** - pushes sparse MoE further with fine-grained expert splitting, shared experts, and auxiliary-loss-free load balancing; achieves frontier quality at dramatically lower training cost
- **GPT-4** - widely believed (though not officially confirmed) to use a sparse MoE architecture inspired by this line of work
- **ST-MoE (Google, 2022)** - directly extended Switch Transformer with stability improvements and achieved state-of-the-art on SuperGLUE
- **GLaM (Google, 2021)** - sparse MoE language model using similar principles, 1.2T parameters, 3x more efficient than GPT-3

---

## Key Takeaways for Practitioners

1. **Top-1 routing is sufficient** - you do not need top-2 or more to get the benefits of MoE; simpler is often better
2. **Load balancing is non-negotiable** - without the auxiliary loss, expert collapse will ruin your model
3. **Capacity factor is a tuning knob** - higher CF means less token dropping but more memory; 1.0-1.25 is a good starting range
4. **Use FP32 for the router even when training in BF16** - the routing computation is sensitive to numerical precision
5. **Expert parallelism scales cleanly** - adding devices to host more experts costs little extra compute per token
6. **Sparse models beat dense at fixed compute** - if you have a compute budget and want maximum quality, MoE is worth the engineering complexity

---

## Limitations and Future Directions

### Limitations

- **Communication overhead** - routing tokens across devices requires all-to-all communication, which dominates latency at large expert counts on slower interconnects
- **Token dropping** - dropped tokens receive no expert processing; rare but non-zero; harder to reason about model behavior
- **Training instability** - more brittle than dense Transformers; requires careful tuning of auxiliary loss weight, capacity factor, and initialization
- **Fine-tuning challenges** - sparse pre-trained models can be harder to fine-tune; later work (ST-MoE) addressed this with improved routing
- **Expert load imbalance at inference** - with small batch sizes (common at inference), load balancing can degrade significantly
- **Reproducibility and debugging** - non-deterministic routing makes debugging and reproducibility harder than dense models

### Open Questions at the Time (Now Partly Answered)

- Can MoE be applied effectively to decoder-only generative models? (Yes - Mixtral, GPT-4)
- How many experts is optimal? (Depends on hardware topology; 8-64 is a common sweet spot for deployed models)
- Can you eliminate token dropping entirely? (DeepSeek-V3's auxiliary-loss-free approach moves in this direction)

---

## Further Reading

- **Original Paper:** https://arxiv.org/abs/2101.03961
- **Mixtral of Experts (Mistral AI, 2024):** https://arxiv.org/abs/2401.04088
- **ST-MoE: Designing Stable and Transferable Sparse Expert Models:** https://arxiv.org/abs/2202.08906
- **GLaM: Efficient Scaling of Language Models with Mixture-of-Experts:** https://arxiv.org/abs/2112.06905
- **DeepSeek-V3 Technical Report:** https://arxiv.org/abs/2412.19437
- **Original Mixture-of-Experts (Shazeer et al. 2017):** https://arxiv.org/abs/1701.06538
- **The Illustrated Switch Transformer (Jay Alammar):** http://jalammar.github.io/illustrated-switch-transformer/

---

## Citation

```bibtex
@article{fedus2022switch,
  title={Switch transformers: Scaling to trillion parameter models with simple and efficient sparsity},
  author={Fedus, William and Zoph, Barret and Shazeer, Noam},
  journal={Journal of Machine Learning Research},
  volume={23},
  number={120},
  pages={1--39},
  year={2022}
}
```

<!-- related:start -->

---

## Related in This Collection

- [Language Models are Few-Shot Learners (GPT-3)](../../language-models/04-gpt3-few-shot-learners/summary.md)
- [DeepSeek-V3 Technical Report](../../language-models/27-deepseek-v3/summary.md)
- [GPT-4 Technical Report](../../language-models/36-gpt4/summary.md)
- [Mixtral of Experts (and the Mixture-of-Experts Architecture)](../../architectures/37-mixture-of-experts/summary.md)
- [Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer (T5)](../../language-models/65-t5/summary.md)

<!-- related:end -->
