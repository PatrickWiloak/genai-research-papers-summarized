---
title: "ZeRO and Megatron-LM: How Trillion-Parameter Models Are Actually Trained"
slug: "76-zero-megatron"
number: 76
category: "techniques"
authors: "Samyam Rajbhandari, Jeff Rasley, Olatunji Ruwase, Yuxiong He (Microsoft) - ZeRO; Mohammad Shoeybi, Mostofa Patwary, Raul Puri, Patrick LeGresley, Jared Casper, Bryan Catanzaro (NVIDIA) - Megatron-LM"
published: "October 2019 (ZeRO, SC 2020); September 2019 (Megatron-LM)"
year: 2019
url: "https://arxiv.org/abs/1910.02054"
tags: ["distributed-training", "systems", "scaling", "efficiency"]
---

# ZeRO and Megatron-LM: How Trillion-Parameter Models Are Actually Trained

**Authors:** Samyam Rajbhandari, Jeff Rasley, Olatunji Ruwase, Yuxiong He (Microsoft) - ZeRO; Mohammad Shoeybi, Mostofa Patwary, Raul Puri, Patrick LeGresley, Jared Casper, Bryan Catanzaro (NVIDIA) - Megatron-LM
**Published:** October 2019 (ZeRO, SC 2020); September 2019 (Megatron-LM)
**Papers:** [ZeRO arxiv.org/abs/1910.02054](https://arxiv.org/abs/1910.02054) | [Megatron-LM arxiv.org/abs/1909.08053](https://arxiv.org/abs/1909.08053)

---

## Why This Matters

Every paper in this collection that trains a large model depends on this work, and almost none of them explain it. [GPT-3](../../language-models/04-gpt3-few-shot-learners/summary.md), [LLaMA](../../language-models/15-llama/summary.md), [DeepSeek-V3](../../language-models/27-deepseek-v3/summary.md), and every frontier model are trained with some combination of ZeRO and Megatron-style parallelism. **This is the systems layer that makes the scaling story physically possible.**

- **A 70B model does not fit on any GPU.** Weights, gradients, and optimizer states for 70B parameters in mixed precision need roughly 1.1 TB. The largest GPU has 192 GB.
- **ZeRO shards the training state** across GPUs instead of replicating it, removing memory redundancy without changing the math.
- **Megatron-LM splits individual layers** across GPUs, so a single layer too large for one device still runs.
- **Together they are "3D parallelism"** - data, tensor, and pipeline parallel - the standard recipe for training anything above about 10B parameters.

**The insight:** standard data parallelism replicates the entire model state on every GPU, which is enormously wasteful. If N GPUs are going to hold N copies of the same optimizer state, you can instead hold 1/N of it each and communicate the pieces when needed. Memory drops linearly with GPU count; communication rises modestly.

---

## The Problem: The Memory Arithmetic Does Not Work

Training memory is dominated not by the weights but by the *optimizer state*. With Adam in mixed precision:

```
Per parameter, training a model in mixed precision with Adam:

  FP16 weights                  2 bytes
  FP16 gradients                2 bytes
  FP32 master weights           4 bytes   \
  FP32 Adam momentum            4 bytes    > "optimizer states": 12 bytes
  FP32 Adam variance            4 bytes   /
  ------------------------------------
  TOTAL                        16 bytes per parameter

  1B parameters   ->  16 GB
  70B parameters  ->  1.1 TB
  175B parameters ->  2.8 TB

  ... plus activations, which for long sequences can exceed all of the above.
```

Classic data parallelism gives every GPU a full copy of all 16 bytes per parameter and differs only in which data batch it sees. A 1,000-GPU cluster therefore stores 1,000 identical copies of the optimizer state. The parallelism scales throughput and does nothing at all for memory, so model size is capped by a single device.

---

## The Core Innovation

### ZeRO: eliminate redundancy, keep data parallelism

ZeRO (Zero Redundancy Optimizer) partitions the training state across data-parallel workers in three progressive stages:

```
                        Per-GPU memory for a model with P parameters
                        across N data-parallel GPUs

Baseline (DDP)          16P            everything replicated
ZeRO-1 (optimizer)      4P + 12P/N     shard optimizer states
ZeRO-2 (+ gradients)    2P + 14P/N     shard gradients too
ZeRO-3 (+ parameters)   16P/N          shard everything; gather layer by layer
```

- **Stage 1** shards the 12 bytes of optimizer state. Each GPU updates only its slice of the weights, then all-gathers the updated weights. Nearly free: no extra communication volume compared to standard all-reduce.
- **Stage 2** additionally shards gradients. Instead of all-reducing full gradients, use reduce-scatter so each GPU ends up with only the gradient slice it needs. Also communication-neutral.
- **Stage 3** shards the parameters themselves. Each GPU stores 1/N of the weights and all-gathers a layer's parameters just before using it, then discards them. This adds roughly 50 percent more communication but makes per-GPU memory scale linearly with cluster size - the model can be arbitrarily large given enough GPUs.

**ZeRO-Offload and ZeRO-Infinity** extend this to CPU RAM and NVMe, trading bandwidth for capacity. This is how hobbyists fine-tune models much larger than their GPU.

### Megatron-LM: split the layer itself

Tensor parallelism splits individual matrix multiplications across GPUs. For a transformer MLP `Y = GeLU(X * A) * B`:

```
Split A by COLUMNS:     A = [A1 | A2]
  GPU1 computes GeLU(X*A1),  GPU2 computes GeLU(X*A2)
  No communication needed - GeLU is element-wise.

Split B by ROWS:        B = [B1]
                            [B2]
  GPU1 computes GeLU(X*A1)*B1,  GPU2 computes GeLU(X*A2)*B2
  Then ONE all-reduce sums the partial results.

Result: one all-reduce per MLP block. Same for attention,
splitting by attention head.
```

This is why tensor parallelism is normally confined **within a node**: it needs an all-reduce twice per layer, which requires NVLink-class bandwidth. Across nodes it stalls on the network.

### Pipeline parallelism: split by layer

Put layers 1-20 on GPU group A, 21-40 on group B, and so on. The problem is the "pipeline bubble" - group B idles while waiting for A. GPipe and later PipeDream/interleaved schedules split each batch into micro-batches so groups stay busy, shrinking but never fully eliminating the bubble.

### 3D parallelism: use all three

Real frontier training combines them:

```
Example layout for a large model on 1,024 GPUs:

  Tensor parallel   = 8    (within a node, over NVLink)
  Pipeline parallel = 8    (across nodes, layer groups)
  Data parallel     = 16   (replicas of the whole pipeline,
                            with ZeRO sharding their optimizer state)

  8 x 8 x 16 = 1,024
```

Plus **sequence/context parallelism** (splitting the sequence dimension) and **expert parallelism** (routing [MoE](../../architectures/37-mixture-of-experts/summary.md) experts to different GPUs) in modern stacks.

---

## Key Components Explained

### 1. Activation Checkpointing
**What it does:** Trades compute for memory on activations, which ZeRO does not address.
**How it works:** Discard intermediate activations during the forward pass and recompute them during the backward pass. Costs roughly 30 percent extra compute, saves a large multiple of activation memory. Standard in every large training run and how long-sequence training is made feasible.

### 2. Mixed Precision Training
**What it does:** Halves memory and roughly doubles throughput.
**How it works:** Compute in FP16 or BF16, keep an FP32 master copy of weights for stable updates. BF16 has largely won because its wider exponent range avoids the loss-scaling machinery FP16 requires. FP8 training is now in use on newer hardware, notably in DeepSeek-V3.

### 3. Communication Overlap
**What it does:** Hides the cost of all the extra communication ZeRO-3 introduces.
**How it works:** Prefetch the next layer's parameters while computing the current one. Well-implemented ZeRO-3 achieves high overlap; poorly configured ZeRO-3 is dominated by communication stalls. This is the main reason ZeRO-3 sometimes underperforms in practice.

### 4. FSDP: ZeRO-3 in PyTorch
**What it does:** Brings ZeRO-3 into the standard framework.
**How it works:** PyTorch's Fully Sharded Data Parallel is ZeRO-3 by another name, natively integrated. For most teams today, "use FSDP" is the practical form of this paper.

---

## Key Results

- ZeRO enabled training models over **100B parameters** on 400 GPUs, roughly 8x larger than what was feasible with model parallelism alone at the time, with super-linear throughput scaling in some regimes (larger effective batch per GPU improves efficiency).
- Megatron-LM trained an **8.3B parameter** transformer with 76 percent scaling efficiency on 512 GPUs, when 1.5B was the prior public frontier.
- Megatron-Turing NLG 530B, trained with 3D parallelism combining both systems, demonstrated the approach at frontier scale.
- Both are open source (DeepSpeed and Megatron-LM), and both are still actively used.

---

## Why This Was Revolutionary

- **Removed the single-GPU memory ceiling** on model size, converting "how big a model can we train" from a hardware question into a cluster-size question.
- **Made the [scaling laws](../12-scaling-laws/summary.md) actionable.** Knowing that bigger is better is useless if you cannot fit bigger. This is the infrastructure that let the field act on Kaplan and [Chinchilla](../18-chinchilla/summary.md).
- **Established the parallelism vocabulary** - data, tensor, pipeline, sequence, expert - that every training-infrastructure discussion now uses.
- **Open sourced the frontier stack.** DeepSpeed and Megatron-LM are why organizations outside the largest labs can train large models at all.

---

## Real-World Impact

- **DeepSpeed** (ZeRO) and **Megatron-LM** are used, directly or via derivatives (Megatron-DeepSpeed, NeMo, FSDP, Accelerate), in nearly every large training run.
- **[DeepSeek-V3](../../language-models/27-deepseek-v3/summary.md)'s remarkable training cost** came from co-designing the parallelism strategy, the MoE routing, and FP8 precision - an advanced application of exactly these ideas.
- **Fine-tuning at home.** ZeRO-Offload plus [QLoRA](../22-qlora/summary.md) is why a single consumer GPU can fine-tune a 70B model.
- **Cost transparency.** Because these systems' efficiency is measurable (model FLOPs utilization), training-cost claims in model reports can be sanity-checked.

---

## Key Takeaways for Practitioners

1. **Pick the simplest tier that fits.** ZeRO-2 for models that fit with sharded gradients; ZeRO-3/FSDP only when you must, since it costs communication.
2. **Keep tensor parallelism inside a node.** Crossing the node boundary with TP is the classic way to destroy throughput.
3. **Activation checkpointing first.** It is often the single largest memory win and the easiest to enable.
4. **Watch MFU (model FLOPs utilization), not just loss.** Below about 30 percent MFU on modern hardware, your parallelism configuration is wrong, not your model.
5. **Use BF16.** Unless you are on hardware that only supports FP16, BF16 removes an entire category of loss-scaling failures.

---

## Limitations & Future Directions

- **Communication-bound at scale.** Past a certain cluster size, interconnect bandwidth, not FLOPs, sets the ceiling. This drives investment in NVLink, InfiniBand, and topology-aware scheduling.
- **Configuration is a dark art.** The 3D parallelism search space is large, and the optimal point depends on model shape, sequence length, and cluster topology. Auto-tuners exist but are imperfect.
- **Fault tolerance.** A thousand-GPU run that loses one node must restart from a checkpoint; elastic and fault-tolerant training remains an active area.
- **Long sequences break the assumptions.** Activation memory scales with sequence length, which is why context and sequence parallelism (Ring Attention and relatives) became necessary additions.

---

## Further Reading

- **ZeRO:** [arxiv.org/abs/1910.02054](https://arxiv.org/abs/1910.02054)
- **Megatron-LM:** [arxiv.org/abs/1909.08053](https://arxiv.org/abs/1909.08053)
- **ZeRO-Infinity:** [arxiv.org/abs/2104.07857](https://arxiv.org/abs/2104.07857)
- **Reducing Activation Recomputation (sequence parallelism):** [arxiv.org/abs/2205.05198](https://arxiv.org/abs/2205.05198)
- **PyTorch FSDP:** [arxiv.org/abs/2304.11277](https://arxiv.org/abs/2304.11277)

## Citation

```bibtex
@inproceedings{rajbhandari2020zero,
  title={ZeRO: Memory Optimizations Toward Training Trillion Parameter Models},
  author={Rajbhandari, Samyam and Rasley, Jeff and Ruwase, Olatunji and He, Yuxiong},
  booktitle={SC20: International Conference for High Performance Computing, Networking, Storage and Analysis},
  year={2020}
}

@article{shoeybi2019megatron,
  title={Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism},
  author={Shoeybi, Mohammad and Patwary, Mostofa and Puri, Raul and LeGresley, Patrick and Casper, Jared and Catanzaro, Bryan},
  journal={arXiv preprint arXiv:1909.08053},
  year={2019}
}
```

<!-- related:start -->

---

## Related in This Collection

- [Language Models are Few-Shot Learners (GPT-3)](../../language-models/04-gpt3-few-shot-learners/summary.md)
- [Scaling Laws for Neural Language Models](../../techniques/12-scaling-laws/summary.md)
- [Training Compute-Optimal Large Language Models (Chinchilla)](../../techniques/18-chinchilla/summary.md)
- [QLoRA: Efficient Finetuning of Quantized LLMs](../../techniques/22-qlora/summary.md)
- [DeepSeek-V3 Technical Report](../../language-models/27-deepseek-v3/summary.md)
- [Mixtral of Experts (and the Mixture-of-Experts Architecture)](../../architectures/37-mixture-of-experts/summary.md)

<!-- related:end -->
