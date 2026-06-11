---
title: "QLoRA: Efficient Finetuning of Quantized LLMs"
slug: "22-qlora"
number: 22
category: "techniques"
authors: "Tim Dettmers, Artidoro Pagnoni, et al. (University of Washington)"
published: "May 2023 (NeurIPS 2023)"
year: 2023
url: "https://arxiv.org/abs/2305.14314"
tags: [techniques]
---

# QLoRA: Efficient Finetuning of Quantized LLMs

**Authors:** Tim Dettmers, Artidoro Pagnoni, et al. (University of Washington)
**Published:** May 2023 (NeurIPS 2023)
**Paper:** [arxiv.org/abs/2305.14314](https://arxiv.org/abs/2305.14314)

---

## Why This Paper Matters

Before QLoRA, fine-tuning a large language model was a privilege reserved for organizations with serious GPU clusters. A 65-billion-parameter model like LLaMA-65B required roughly 780 GB of GPU memory in full 16-bit precision - roughly 10 high-end A100 80GB cards running in parallel. Even with LoRA (see [../10-lora/summary.md](../10-lora/summary.md)), which reduces the number of trainable parameters dramatically, you still had to load the full base model weights in memory to compute gradients. The base model alone for a 65B network weighs about 130 GB in 16-bit. That is not a research-lab laptop or a rented single-GPU cloud instance.

QLoRA changed the economics entirely. By freezing the base model in 4-bit precision and training only small LoRA adapter weights kept in 16-bit, it cut the memory footprint for 65B fine-tuning to just 48 GB - a single NVIDIA A40 or consumer-grade RTX 4090 (24 GB at 33B, 48 GB at 65B). Critically, the authors showed **no measurable quality loss** compared to full 16-bit fine-tuning. The Guanaco family of instruction-tuned models trained with QLoRA reached 99.3% of ChatGPT-3.5 performance on the Vicuna benchmark with a single 24-hour training run on one GPU.

That combination - 16x memory reduction, no quality loss, and commodity hardware access - triggered the open-source fine-tuning boom of 2023. QLoRA is now the de-facto default technique any time someone fine-tunes an open model.

---

## Background: What QLoRA Builds On

QLoRA is a tight integration of two prior ideas. Understanding each individually makes the combination clearer.

**Quantization** compresses model weights from their training-time precision (usually 16-bit or 32-bit floats) into lower-bit integers. At 4 bits you store each weight in one of 16 discrete levels rather than 65,536. The obvious risk is accuracy loss - cramming a continuous distribution into 16 buckets throws away information. Prior 4-bit quantization schemes (GPTQ, round-to-nearest integer quantization) accepted some degradation, especially for fine-tuning workloads where gradients need to flow through the model.

**LoRA** (Low-Rank Adaptation) keeps the base model frozen and injects small trainable matrices into the attention layers. Instead of updating all 65B parameters, you update perhaps 50-100 million in the adapters - a 600-1000x reduction in trainable parameters. The savings are real but incomplete: you still need the base model in memory to run the forward pass, and prior LoRA work assumed 16-bit base weights.

QLoRA's core insight is that these two ideas are **composable**: freeze the base model in 4-bit (saving memory), keep the LoRA adapters in 16-bit (preserving gradient quality), and dequantize only the small slice of base weights you need at each forward-pass step. The result is a system that is almost as cheap as pure quantized inference but supports full task-specific learning.

---

## Key Components Explained

### 1. 4-bit NormalFloat (NF4) - An Information-Theoretically Optimal Quantization Type

Standard 4-bit quantization maps weights to evenly-spaced integer bins. That is fine for uniform distributions, but neural network weights are not uniform - they are approximately Gaussian (bell-curve shaped) after training. Most weights cluster near zero, with far fewer at the extremes.

If you allocate your 16 quantization levels evenly, you waste precision on the sparse extremes and cram too many common near-zero weights into too few buckets. The result is systematic rounding error at the dense center of the distribution.

NF4 fixes this by placing quantization levels at **quantiles** of the standard normal distribution rather than at evenly-spaced intervals. Each of the 16 levels covers an equal share of the probability mass. Conceptually:

```
Standard 4-bit (uniform bins):
|--|----|----|----|----|----|----|----|--|
 ^  lots of empty space at extremes

NF4 (quantile bins):
|--|--|--|--|--|--|--|--|--|--|--|--|--|--|
 ^  tight near zero (where most weights live)
```

This is the information-theoretically optimal quantization for a normal distribution - it minimizes the expected quantization error given a fixed number of bits. The paper proves this formally using properties of quantile functions.

In practice, NF4 produces consistently better results than standard int4 or float4 for fine-tuning tasks without any computational overhead at inference time.

### 2. Double Quantization - Quantizing the Quantization Constants

Standard blockwise quantization (the typical method for handling the fact that a layer's weights have varying ranges) works by dividing weight tensors into blocks and storing one quantization constant per block. For example, with block size 64 and 32-bit quantization constants, you pay about 32/64 = 0.5 extra bits per parameter just to store those constants.

The QLoRA authors observed that quantization constants themselves are just another set of floating-point numbers - and they, too, can be quantized. **Double quantization** applies a second round of quantization to the first-round constants:

- First quantization: weights to NF4, block size 64, constants stored as float32
- Second quantization: those float32 constants to float8, block size 256

Net memory saving: roughly 0.37 bits per parameter. For a 65B model that adds up to about 3 GB of savings - meaningful when you are already trying to squeeze into 48 GB. More importantly it establishes a principle: every constant you store is a potential compression target.

### 3. Paged Optimizers - Handling Memory Spikes with NVIDIA Unified Memory

Even with the base model quantized, training still requires optimizer states for the adapter parameters. For Adam, that is two copies of the gradient (first and second moment) per trainable parameter. With 50-100M trainable LoRA parameters in bfloat16, that is a manageable few hundred MB.

The harder problem is gradient checkpointing. PyTorch's gradient checkpointing trades memory for compute: instead of storing all intermediate activations during the forward pass, it recomputes them during the backward pass. This typically reduces activation memory significantly, but it also causes **memory spikes** - brief moments during the backward pass when both the stored checkpoints and the just-recomputed activations coexist in GPU memory simultaneously. These spikes can push an otherwise-fine training run over the VRAM limit and trigger an out-of-memory (OOM) crash.

Paged optimizers use NVIDIA's **unified memory** feature, which automatically pages GPU memory to CPU RAM when the GPU runs out. Think of it like virtual memory in an operating system: when a program needs more RAM than physically available, the OS silently swaps pages to disk. Unified memory does the same thing between GPU VRAM and CPU DRAM.

The optimizer states for the LoRA adapters are stored in this pageable memory pool. During normal steps they live on the GPU. During OOM-spike moments in gradient checkpointing, the CUDA runtime automatically evicts them to CPU RAM and pages them back in when the spike has passed. The result is that training completes reliably even on GPUs with tight margins, at the cost of slightly slower steps when paging occurs.

### 4. Frozen 4-bit Base + 16-bit LoRA Adapters - The Core Training Loop

The computational graph during QLoRA training looks like this:

```
Input tokens
     |
[Base model weights] -- stored NF4 (4-bit), frozen
     |  (dequantize to bfloat16 on the fly for each forward step)
     |
[Layer activations] -- bfloat16
     |
[LoRA adapter A, B matrices] -- bfloat16, trainable
     |
     +-- LoRA delta = (A @ B) * alpha/r, added to layer output
     |
Loss
     |
Gradients flow back through LoRA adapters only
     (base model weights receive no gradient update)
```

The key subtlety: dequantization happens at compute time, not storage time. Base weights stay in NF4 on disk and in VRAM. Only the small block of weights needed for the current matrix multiplication is temporarily expanded to bfloat16, multiplied, and discarded. This keeps the memory footprint at 4-bit levels while keeping arithmetic numerically sound.

---

## Key Results

### Memory Comparison Across Methods

```
Method                  | 65B model VRAM needed
------------------------|----------------------
Full fine-tuning        | ~780 GB  (10+ A100s)
LoRA (16-bit base)      | ~360 GB  (5+ A100s)
QLoRA (NF4 base)        | ~48 GB   (1x A40/A6000)
QLoRA (NF4 base, 33B)   | ~24 GB   (1x RTX 3090/4090)
```

### Guanaco Models

The paper introduced the Guanaco model family - QLoRA fine-tuned versions of LLaMA trained on the OASST1 instruction-following dataset. Key results:

- **Guanaco 65B**: 99.3% of ChatGPT performance on the Vicuna evaluation set (human rater preference), trained in 24 hours on a single 48GB GPU
- **Guanaco 33B**: 97.8% of ChatGPT performance, trainable on a single consumer RTX 3090
- **Guanaco 7B**: Competitive with prior models that required 5-10x the compute

The headline claim - 99.3% ChatGPT parity - was measured with GPT-4 as a judge rating head-to-head responses. It is worth noting this benchmark has ceiling effects and the evaluation methodology has known biases, but the result was directionally robust and independently reproduced across many community fine-tunes in the months following publication.

### Quality Preservation Under Quantization

The paper ran systematic ablations comparing full 16-bit fine-tuning (gold standard), 16-bit LoRA, 8-bit LoRA, and 4-bit LoRA with various quantization types (int4, float4, NF4). NF4 with double quantization matched 16-bit LoRA within noise across MMLU, coding, and reasoning benchmarks. Standard int4 and float4 showed visible degradation. This validated NF4 as the right choice for normally-distributed weight tensors.

---

## Why This Was Revolutionary

### 1. It Decoupled Model Size from Fine-Tuning Accessibility

Before QLoRA, there was a rough rule: you need roughly 2 bytes of VRAM per parameter to fine-tune (in 16-bit). A 7B model needed ~14 GB, a 65B needed ~130 GB just for weights - plus activations, gradients, and optimizer states. QLoRA broke this rule. Model size no longer determined whether fine-tuning was feasible on your hardware.

### 2. It Enabled Rapid Iteration at Scale

A 24-hour fine-tuning run on a single GPU is qualitatively different from a week-long multi-GPU job. It means researchers can run multiple experiments per day, try different datasets, hyperparameters, and task formulations, and observe results quickly. The cost dropped from thousands of dollars to tens of dollars per run.

### 3. It Legitimized 4-bit Quantization for Training

Prior to QLoRA, 4-bit quantization was used for inference (deploying already-trained models on edge devices or reducing serving costs). Nobody used 4-bit for training because the conventional wisdom was that gradients would be too noisy. QLoRA proved the conventional wisdom wrong - provided you use the right quantization scheme (NF4) and keep adapter weights in higher precision.

### 4. It Created a Template for Future Efficiency Research

The double quantization idea - quantize everything you can, including metadata - became a recurring theme in subsequent efficiency papers. The paged-optimizer approach demonstrated that unified memory could be used productively in training loops, not just as a crash-prevention afterthought.

---

## Real-World Impact and Descendants

QLoRA's release triggered an explosion of community fine-tuning work in 2023:

- **Guanaco** (direct output of the paper) - first demonstration that a single-GPU run could produce ChatGPT-competitive results
- **WizardLM, Orca, OpenHermes, Dolphin** - all used QLoRA pipelines on LLaMA and Mistral base models
- **HuggingFace PEFT library** - integrated QLoRA (via bitsandbytes) within weeks of the paper; `load_in_4bit=True` became a one-line API
- **LLaMA.cpp and GGUF format** - while separate from QLoRA, the same community interest in 4-bit models drove the gguf quantization ecosystem for inference
- **AutoGPTQ and GPTQ-for-LLaMA** - complementary inference-focused quantization; many production stacks combine GPTQ inference with QLoRA-derived adapter weights
- **Axolotl** - open-source fine-tuning framework that made QLoRA accessible with a config file rather than raw Python

The broader effect was structural: QLoRA made it economically viable for small teams and individual researchers to produce domain-specific models competitive with much larger commercial offerings. Biomedical, legal, coding, and customer-service LLMs proliferated in 2023-2024 largely because QLoRA removed the compute barrier.

---

## Key Takeaways for Practitioners

1. **Use QLoRA as your default fine-tuning setup unless you have a reason not to.** The quality parity with 16-bit fine-tuning is well-established. Start with NF4 quantization, double quantization enabled, and paged Adam optimizer.

2. **Match LoRA rank to task complexity.** Rank 16-64 is typical for instruction fine-tuning. Higher ranks (128-256) are sometimes used for tasks requiring more expressive adaptation but at greater memory cost.

3. **Target the right layers.** The paper fine-tunes all linear layers in the transformer, not just the attention projection matrices. Targeting query/value projections only (the original LoRA recommendation) sometimes leaves quality on the table.

4. **Monitor GPU memory with `nvidia-smi dmon` during training.** Paged optimizers handle spikes gracefully but if you are seeing frequent paging events your batch size may be too aggressive.

5. **bitsandbytes + HuggingFace PEFT is the standard stack.** `BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16)` followed by `prepare_model_for_kbit_training` and a `LoraConfig` gets you to a functional QLoRA setup in under 20 lines.

6. **Consider compute dtype separately from storage dtype.** Weights are stored in NF4 but computations run in bfloat16. On Ampere and later GPUs, bfloat16 uses tensor cores and is faster than float32 for matrix multiplications - so you are not paying a compute penalty for the quantization.

---

## Limitations and Future Directions

### Limitations

- **4-bit arithmetic is not universally lossless.** NF4 is optimal for normally distributed weights, which covers most transformer layers. But some layers - particularly embedding tables and the final language model head - have different distributions. QLoRA typically keeps these in 16-bit, which is the right call but means the memory savings are not uniform across the network.

- **Inference still requires dequantization overhead.** While training in QLoRA is memory-efficient, deploying the resulting merged model is a separate question. You can merge LoRA weights back into the base model (now in 16-bit) or run the NF4 base with adapters applied at inference - each has trade-offs in speed and memory.

- **Adapter expressiveness ceiling.** For very large distributional shifts (training a general model to be a narrow domain expert from scratch), the low-rank constraint of LoRA can be a bottleneck. Full fine-tuning, if you can afford it, is still the ceiling.

- **Paged optimizer paging is slow.** When OOM spikes actually trigger paging to CPU RAM, that step is significantly slower. This usually does not matter for throughput because spikes are rare, but it can confuse profiling and make step-time estimates unreliable.

### Future Directions

- **2-bit and 1-bit quantization for inference** (BitNet, 1.58-bit models from Microsoft) push the quantization frontier further, though training in 2-bit is still an open problem.
- **QLoRA for vision and multimodal models** - the technique transfers naturally to vision transformers and multimodal architectures, and has been applied to LLaVA and similar systems.
- **Quantization-aware training** - training from scratch with quantization in the loop (rather than post-training quantization) may eventually close the remaining gap between quantized and full-precision models.
- **Better adapter architectures** - LoRA+ (asymmetric learning rates), DoRA (weight decomposition), and IA3 offer alternatives to standard LoRA that may interact better with 4-bit quantization.

---

## Further Reading

- **Original Paper:** https://arxiv.org/abs/2305.14314
- **HuggingFace QLoRA blog post (with code walkthrough):** https://huggingface.co/blog/4bit-transformers-bitsandbytes
- **Tim Dettmers' blog on quantization fundamentals:** https://timdettmers.com/2022/08/17/llm-int8-and-emergent-features/
- **The Illustrated QLoRA (Hatem Haddad):** https://huggingface.co/blog/haddadalwi/qlora-illustrated
- **bitsandbytes library:** https://github.com/TimDettmers/bitsandbytes
- **HuggingFace PEFT library:** https://github.com/huggingface/peft
- **LoRA paper (prerequisite):** [../10-lora/summary.md](../10-lora/summary.md)
- **Axolotl fine-tuning framework:** https://github.com/OpenAccess-AI-Collective/axolotl

---

## Citation

```bibtex
@article{dettmers2023qlora,
  title={QLoRA: Efficient Finetuning of Quantized LLMs},
  author={Dettmers, Tim and Pagnoni, Artidoro and Holtzman, Ari and Zettlemoyer, Luke},
  journal={Advances in Neural Information Processing Systems},
  volume={36},
  year={2023},
  url={https://arxiv.org/abs/2305.14314}
}
```

<!-- related:start -->

---

## Related in This Collection

- [LoRA: Low-Rank Adaptation of Large Language Models](../../techniques/10-lora/summary.md)
- [GPT-4 Technical Report](../../language-models/36-gpt4/summary.md)
- [LLaVA: Visual Instruction Tuning](../../multimodal/46-llava/summary.md)

<!-- related:end -->
