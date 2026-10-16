---
title: "Mamba-3: Improved Sequence Modeling using State Space Principles (Mamba-3)"
slug: "156-mamba-3"
number: 156
category: "architectures"
authors: "Aakash Lahoti, Kevin Y. Li, Berlin Chen, Caitlin Wang, Aviv Bick, J. Zico Kolter, Tri Dao, Albert Gu (Carnegie Mellon University, Princeton University, Together AI, Cartesia AI)"
published: "March 2026 (arXiv v1, 16 March 2026; the arXiv listing marks it ICLR 2026)"
year: 2026
url: "https://arxiv.org/abs/2603.15569"
tags: ["architecture", "state-space", "efficiency"]
---

# Mamba-3: Improved Sequence Modeling using State Space Principles (Mamba-3)

**Authors:** Aakash Lahoti, Kevin Y. Li, Berlin Chen, Caitlin Wang, Aviv Bick, J. Zico Kolter, Tri Dao, Albert Gu (Carnegie Mellon University, Princeton University, Together AI, Cartesia AI)
**Published:** March 2026 (arXiv v1, 16 March 2026; the arXiv listing marks it ICLR 2026)
**Paper Link:** [arxiv.org/abs/2603.15569](https://arxiv.org/abs/2603.15569)

---

## Why This Paper Matters

Transformers dominate language modeling, but at inference time they pay for it: attention costs compute that grows quadratically with context, and the KV cache (the stored keys and values of every earlier token, see [KV cache](../../../explainers/concepts/kv-cache.md)) grows linearly in memory. Sub-quadratic models such as state space models (SSMs) and linear attention instead compress the past into a fixed-size state, so each new token costs the same no matter how long the context is.

The Mamba-3 authors argue that the field's attention has moved from training speed to inference efficiency, because test-time compute (long chains of thought, agentic loops) now takes a large share of total cost. From that angle they point out three weaknesses of current linear models: some trade quality for efficiency; they fail on state-tracking tasks such as determining the parity of a bit string; and their decoding is hardware-inefficient in practice, with low arithmetic intensity (few arithmetic operations per byte moved from memory), leaving much of the GPU idle during generation.

Mamba-3 is their "inference-first" redesign of the Mamba layer ([Mamba](../20-mamba/summary.md), and its successor Mamba-2). It is by the original Mamba authors (Albert Gu and Tri Dao). Note that this is a 2026 paper; everything below comes from the paper text itself, and independent reproductions or later follow-ups are not covered.

## The Core Innovation

Three changes on top of Mamba-2, all derived from the SSM viewpoint rather than the linear-attention viewpoint:

```
Mamba-2                          Mamba-3
---------------------------      ------------------------------------------
Euler-like discretization        exponential-trapezoidal discretization
real-valued state transition     complex-valued state (data-dependent RoPE)
outer-product state update       matrix-multiply update (MIMO option)
short causal convolution         no short conv (biases + trapezoid replace it)
```

## Key Components Explained

### 1. Exponential-trapezoidal discretization
**What it does:** Turns the continuous-time SSM into a token-by-token recurrence more accurately.
**How it works:** An SSM is defined in continuous time and must be discretized to run on tokens. The paper gives a framework for time-varying (selective) SSMs, shows that the heuristic used in Mamba-1 and Mamba-2 is one instance ("exponential-Euler"), and proposes a more expressive instance. The new rule uses both the current and previous input in each update, which acts like an implicit width-2 convolution on the SSM input. Together with learnable biases on B and C, the authors find this makes the short causal convolution that earlier Mamba-style and other recurrent models relied on unnecessary.

### 2. Complex-valued state
**What it does:** Gives the model the ability to track state, such as counting or parity.
**How it works:** Viewing the SSM as complex-valued adds a rotation to the state update. The paper shows this is equivalent to a data-dependent rotary embedding (RoPE) and can be computed cheaply. On formal-language tasks (scaled accuracy), Mamba-3 reaches 100.00 on parity and 98.51 on modular arithmetic without brackets, while Mamba-2 scores 0.90 and 47.81 and Mamba-3 without RoPE scores 2.27 and 1.49. With brackets Mamba-3 gets 87.75 versus 0.88 for Mamba-2. A Gated DeltaNet variant with an extended eigenvalue range scores 100.00, 99.25 and 93.50, so Mamba-3 is competitive, not best.

### 3. Multi-input, multi-output (MIMO)
**What it does:** Spends the idle compute during decoding on a better model.
**How it works:** Decoding is limited by memory traffic, not arithmetic. Switching from a single-input single-output (SISO) outer-product update to a rank-R matrix-multiply update raises arithmetic intensity roughly in proportion to R without enlarging the state, so more FLOPs are done in the time already spent. The paper uses R = 4 and shrinks the MLP width slightly (for 1.5B, 4096 to 3824, about 6.6%) to keep parameter counts comparable. It reports up to 4x more decoding FLOPs than Mamba-2 at fixed state size with similar wall-clock decode latency. The cost is slower training and a moderate prefill overhead.

### 4. Architecture refinements
The block alternates Mamba-3 and SwiGLU layers with pre-norm, Llama style. Additions: RMS normalization after the B and C projections (BCNorm, analogous to QK-norm in Transformers), which lets pure Mamba-3 models drop Mamba-2's post-gate RMSNorm; and learnable head-specific, channel-wise biases on B and C.

## Key Results

All models were trained on 100B FineWeb-Edu tokens at 2K context with the Llama-3.1 tokenizer, at 180M, 440M, 880M and 1.5B parameters.

| 1.5B model | FineWeb-Edu ppl | Avg. downstream acc. |
|---|---|---|
| Transformer | 10.51 | 55.4 |
| Gated DeltaNet | 10.45 | 55.8 |
| Mamba-2 | 10.47 | 55.7 |
| Mamba-3 SISO | 10.35 | 56.4 |
| Mamba-3 MIMO | 10.24 | 57.6 |

- SISO beats the next best model (Gated DeltaNet) by 0.6 points at 1.5B; MIMO adds another 1.2, for 1.8 over Gated DeltaNet. The paper states Mamba-3 SISO beats Mamba-2 at every scale tested.
- **State size:** across state-size sweeps (440M, 2x Chinchilla tokens) Mamba-3 matches Mamba-2 perplexity with half the state size, and the MIMO variant at state 64 matches Mamba-2 at 128.
- **Ablation (440M):** removing both the biases and the trapezoidal rule raises perplexity from 15.72 to 16.68; adding the short convolution back does not help (15.85).
- **Retrieval:** weaker than a Transformer on real-world extraction from semi-structured data (SWDE, FDA), competitive on question answering, and better than Mamba-2 on out-of-distribution-length needle-in-a-haystack. The authors expect linear layers to be used mainly in hybrids with attention.
- **Speed:** on an H100 (batch 128, 1.5B-scale), the paper reports Mamba-3 SISO decode kernels as the fastest of the compared kernels (Mamba-2, Gated DeltaNet), and MIMO as close to Mamba-2. The authors release training and inference kernels.

## Why This Was Revolutionary

- Frames model design around decode hardware utilization, not just parameter count and training throughput.
- Gives the old Mamba discretization a principled footing and a stronger replacement.
- Restores state tracking to an SSM with a cheap complex/RoPE trick.
- Shows the short convolution, long assumed essential, can be removed.

(Whether it proves "revolutionary" in practice is for follow-up work to show; the paper's own claim is a better performance-efficiency Pareto frontier.)

## Real-World Impact

At the time of writing there is little to report beyond the paper: the code and kernels are released in the state-spaces/mamba repository. The paper cites recent hybrid models that mix linear layers with attention (including Kimi Linear and Nemotron-H) as the setting where such layers are used. For the lineage, see [Mamba](../20-mamba/summary.md); for why decode cost matters, see [KV cache](../../../explainers/concepts/kv-cache.md), [context windows](../../../explainers/concepts/context-windows.md) and [inference economics](../../../explainers/compute/inference-economics.md).

## Key Takeaways for Practitioners

1. Decoding is memory-bound; extra arithmetic per byte of state is nearly free, which is what MIMO exploits.
2. A fixed-size state still limits retrieval; plan on hybrids with some attention layers for recall-heavy workloads.
3. Check state-tracking behavior (parity, modular arithmetic) when evaluating a linear model; language-model perplexity does not reveal it.
4. Results are at 180M to 1.5B parameters and 100B tokens; do not extrapolate to frontier scale without evidence.

## Limitations & Future Directions

- Scale: only up to 1.5B parameters, 100B training tokens and 2K context.
- Retrieval is below the Transformer, notably on SWDE and FDA; the best normalization choice for hybrids is described as unclear.
- MIMO trains more slowly and adds prefill overhead.
- Benchmarks are the authors' own, using their kernels; independent replication is not yet reflected here.

## Further Reading

- **Original Paper:** https://arxiv.org/abs/2603.15569
- **Code and kernels:** https://github.com/state-spaces/mamba
- **Related summaries:** [Mamba](../20-mamba/summary.md)
- **Explainers:** [KV cache](../../../explainers/concepts/kv-cache.md), [Inference economics](../../../explainers/compute/inference-economics.md)

## Citation

```bibtex
@article{lahoti2026mamba3,
  title={Mamba-3: Improved Sequence Modeling using State Space Principles},
  author={Lahoti, Aakash and Li, Kevin Y. and Chen, Berlin and Wang, Caitlin and Bick, Aviv and Kolter, J. Zico and Dao, Tri and Gu, Albert},
  journal={arXiv preprint arXiv:2603.15569},
  year={2026}
}
```

<!-- related:start -->

---

## Related in This Collection

- [Attention Is All You Need](../../architectures/01-attention-is-all-you-need/summary.md)
- [Training Compute-Optimal Large Language Models (Chinchilla)](../../techniques/18-chinchilla/summary.md)
- [Mamba: Linear-Time Sequence Modeling with Selective State Spaces](../../architectures/20-mamba/summary.md)
- [Scaling LLM Test-Time Compute: The Theoretical Foundation for Reasoning Models](../../techniques/50-test-time-compute/summary.md)
- [RoFormer: Enhanced Transformer with Rotary Position Embedding (RoPE)](../../techniques/54-rope-rotary-position-embedding/summary.md)
- [GQA: Grouped-Query Attention (and Multi-Query Attention)](../../architectures/75-grouped-query-attention/summary.md)
- [RULER: What's the Real Context Size of Your Long-Context Language Models? (RULER)](../../techniques/132-ruler/summary.md)
- [The FineWeb Datasets: Decanting the Web for the Finest Text Data at Scale (FineWeb)](../../techniques/133-fineweb/summary.md)

<!-- related:end -->
