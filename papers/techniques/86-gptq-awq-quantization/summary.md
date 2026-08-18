---
title: "GPTQ and AWQ: Post-Training Quantization for Large Language Models"
slug: "86-gptq-awq-quantization"
number: 86
category: "techniques"
authors: "Elias Frantar, Saleh Ashkboos, Torsten Hoefler, Dan Alistarh (IST Austria, ETH Zurich, Neural Magic) - GPTQ; Ji Lin, Jiaming Tang, Haotian Tang, Shang Yang, Xingyu Dang, Chuang Gan, Song Han (MIT, SJTU, MIT-IBM Watson AI Lab) - AWQ"
published: "October 2022 (GPTQ, ICLR 2023); June 2023 (AWQ, MLSys 2024 Best Paper)"
year: 2022
url: "https://arxiv.org/abs/2210.17323"
tags: ["quantization", "efficiency", "inference-optimization"]
---

# GPTQ and AWQ: Post-Training Quantization for Large Language Models

**Authors:** Elias Frantar, Saleh Ashkboos, Torsten Hoefler, Dan Alistarh (IST Austria, ETH Zurich, Neural Magic) - GPTQ; Ji Lin, Jiaming Tang, Haotian Tang, Shang Yang, Xingyu Dang, Chuang Gan, Song Han (MIT, SJTU, MIT-IBM Watson AI Lab) - AWQ
**Published:** October 2022 (GPTQ, ICLR 2023); June 2023 (AWQ, MLSys 2024 Best Paper)
**Papers:** [GPTQ arxiv.org/abs/2210.17323](https://arxiv.org/abs/2210.17323) | [AWQ arxiv.org/abs/2306.00978](https://arxiv.org/abs/2306.00978)

---

## Why This Matters

These two papers are **why you can run a serious model on your own machine**. Post-training quantization compresses a trained model from 16 bits per weight to 4, cutting memory roughly 4x and speeding up generation, with quality loss small enough that most users do not notice.

- **A 70B model goes from about 140 GB to about 35 GB** - from "needs a multi-GPU server" to "runs on one high-end GPU," and 4-bit 7B and 13B models run on laptops.
- **No retraining.** Post-training quantization takes a finished checkpoint and a small calibration set. Minutes to hours, not a training run.
- **Nearly every local model you download is quantized** with these methods or their descendants (GGUF k-quants in llama.cpp, EXL2, bitsandbytes NF4).
- **They also speed things up.** Generation is memory-bandwidth-bound, so moving 4x fewer bytes per token is a direct throughput win.

**The insight:** neural network weights carry far more numeric precision than they need, but not uniformly - a small number of weights matter enormously and most matter very little. Naive rounding destroys quality because it treats them alike. Both papers exploit this asymmetry, GPTQ by compensating for rounding error as it goes, AWQ by protecting the weights that measurably matter.

---

## The Problem: Models Do Not Fit

```
Memory for weights alone, by precision:

Model    FP16      INT8      INT4
7B       14 GB     7 GB      3.5 GB
13B      26 GB     13 GB     6.5 GB
70B      140 GB    70 GB     35 GB
405B     810 GB    405 GB    202 GB

Consumer GPUs: 8-24 GB.
A single datacenter GPU: 80-192 GB.
```

At FP16, a 13B model does not fit on most consumer hardware and a 70B needs multiple datacenter GPUs. Add the [KV cache](../../architectures/75-grouped-query-attention/summary.md) and the picture is worse.

Naive round-to-nearest quantization at 4 bits was known to break large language models - perplexity degrades sharply, and the degradation is not graceful. The reason is **outlier activations**: in large transformers, a few feature dimensions carry activations orders of magnitude larger than the rest, and the weights that interact with them are disproportionately important. Uniform rounding treats those weights as ordinary and the model falls apart. (This is the same phenomenon LLM.int8() addressed by keeping outlier dimensions in higher precision.)

---

## The Core Innovation

### GPTQ: quantize one weight at a time, compensating as you go

GPTQ processes each layer's weight matrix column by column. After rounding one weight, it **updates the remaining unquantized weights** to absorb the error that rounding just introduced.

```
For each column j in the weight matrix:
  1. Quantize column j to 4 bits (round to the nearest grid point)
  2. Measure the error this introduced
  3. Adjust the REMAINING columns (j+1, j+2, ...) to compensate,
     using second-order (Hessian) information about how much each
     weight affects the layer's output
  4. Move to column j+1

Result: the layer's OUTPUT stays close to the original,
even though individual weights have moved a lot.
```

The second-order information comes from a small calibration set (typically 128 random sequences) run through the model to estimate the Hessian of the layer-wise reconstruction error. The optimization is done efficiently with Cholesky decomposition and batched updates, which is what makes it fast: **OPT-175B quantized in about four GPU hours**, when comparable methods needed days.

The key framing: do not minimize weight error, minimize **output** error. Which weights moved is irrelevant if the layer computes the same function.

### AWQ: protect the weights that matter, by scaling

AWQ starts from a sharper observation: **roughly 1 percent of weights are salient**, and preserving just those preserves quality. But keeping 1 percent in FP16 means mixed-precision storage, which is slow and awkward on real hardware.

AWQ's trick avoids mixed precision entirely:

```
Identify salient weight channels by looking at ACTIVATION
magnitude, not weight magnitude. (Which weights see the big
inputs is what matters, not which weights are big.)

Then, before quantizing, apply a per-channel scale s:
    W' = W * s        and        X' = X / s
    W' X' = W X       (mathematically equivalent)

Scaling up a salient channel's weights makes them occupy more
of the quantization grid, so rounding costs them relatively
less. Everything stays uniform INT4 - no mixed precision,
no special kernels for outliers.
```

Choosing scales by activation statistics rather than weight statistics is the paper's central empirical finding, and it is a little counterintuitive. AWQ also needs no backpropagation and does not overfit the calibration set, so it generalizes better across domains than reconstruction-based methods.

---

## Key Components Explained

### 1. Calibration Data
**What it does:** Tells the method which weights matter.
**How it works:** A small sample (commonly 128 sequences) of representative text. **The domain matters**: calibrating on generic web text and deploying on code or a non-English language costs measurable quality. If you serve a specialized domain, calibrate on it. AWQ is less sensitive to this than GPTQ, which is one of its practical advantages.

### 2. Group Size
**What it does:** Trades compression against accuracy.
**How it works:** Rather than one scale factor per tensor, use one per group of weights (commonly 128). Smaller groups mean better accuracy and slightly more overhead. `4-bit, group size 128` is the standard configuration you see in model names.

### 3. Where the Speedup Comes From
**What it does:** Explains why 4-bit is faster, not just smaller.
**How it works:** Single-stream generation is memory-bandwidth-bound - the GPU spends its time waiting for weights to arrive, not computing. Quartering the bytes read per token gives a near-proportional speedup on decode. At large batch sizes the workload becomes compute-bound and the advantage shrinks, which is why quantization helps local single-user inference more than high-throughput serving.

### 4. The Format Landscape
**What it does:** Orients you in a confusing ecosystem.
**How it works:**
- **GPTQ** - widely supported, strong accuracy, calibration-sensitive.
- **AWQ** - comparable or better accuracy, better generalization, fast kernels, now common in vLLM deployments.
- **GGUF k-quants (llama.cpp)** - the CPU and Apple Silicon ecosystem, with mixed bit widths per tensor type; the reason local inference on a laptop works at all.
- **bitsandbytes NF4** - the format [QLoRA](../22-qlora/summary.md) uses, optimized for fine-tuning rather than serving.
- **FP8 and INT8** - lighter compression with near-zero loss, native support on recent datacenter GPUs, preferred where memory is not the binding constraint.

---

## Key Results

- **GPTQ:** 3-4 bit quantization of OPT-175B and BLOOM-176B with small perplexity increases, completed in roughly 4 GPU hours; enabled 175B inference on a single high-memory GPU where FP16 needed several.
- **AWQ:** better perplexity than GPTQ at matched bit width across LLaMA-family and instruction-tuned models, with stronger results on multimodal and domain-shifted evaluation; efficient kernels delivering large speedups over FP16 on both desktop and mobile-class hardware. **MLSys 2024 Best Paper.**
- Both methods show 4-bit quality loss small enough for general use, with degradation becoming clear at 3 bits and severe at 2.

---

## Why This Was Revolutionary

- **Democratized inference.** The entire local-model ecosystem - Ollama, LM Studio, llama.cpp, text-generation-webui - exists because 4-bit quantization works.
- **Cut serving costs across the industry**, including for providers who quantize models they host.
- **Established output-error minimization as the right objective** for compression, displacing weight-error heuristics.
- **AWQ's activation-based saliency** reframed the outlier problem in a way that later work (SmoothQuant, activation-aware KV cache quantization) built on directly.
- **Made "which quantization" a normal product decision** rather than a research question.

---

## Real-World Impact

- **Hugging Face is full of `-GPTQ`, `-AWQ`, and `-GGUF` variants** of every popular open model; for many users these are the only versions they ever run.
- **vLLM, TensorRT-LLM, SGLang, and llama.cpp** all ship quantized inference paths as standard.
- **On-device AI.** Phone and laptop assistants depend on 4-bit or lower quantization.
- **[QLoRA](../22-qlora/summary.md)** combined 4-bit quantization with LoRA fine-tuning, letting a single consumer GPU fine-tune a 65B model - the most-used application of these ideas.
- **KV cache quantization** applies the same reasoning to the cache rather than the weights, and stacks with [GQA](../../architectures/75-grouped-query-attention/summary.md) for long-context serving.

---

## Key Takeaways for Practitioners

1. **4-bit is the sweet spot.** Use it by default for local inference. 3-bit is visibly worse; 2-bit is for experiments.
2. **AWQ or GPTQ for GPU serving; GGUF for CPU, Apple Silicon, and mixed setups.** AWQ is the safer default when both are available.
3. **Calibrate on your domain** if you serve something unusual - code, legal text, a non-English language, or a specific chat format.
4. **Expect the biggest wins on single-stream decoding.** At high batch sizes, quantization saves memory more than time.
5. **Benchmark on your task, not on perplexity.** Perplexity is a poor proxy for the degradations users notice; quantized models often lose long-context recall and instruction-following precision before they lose fluency.
6. **A quantized larger model usually beats an FP16 smaller one** at equal memory - a 4-bit 70B generally outperforms an FP16 13B.

---

## Limitations & Future Directions

- **Quality loss is real, if small.** It concentrates in long-context tasks, multi-step reasoning, and rare knowledge rather than in obvious fluency.
- **Below 4 bits degrades sharply** without additional techniques; 1-bit and ternary approaches (BitNet and relatives) require training from scratch rather than post-training conversion.
- **Kernel support fragments across hardware.** A format that is fast on one GPU generation may be slow or unsupported on another.
- **Quantization interacts badly with fine-tuning.** Fine-tuning a quantized model requires care; QLoRA works by keeping the base frozen and training higher-precision adapters.
- **Reasoning models may be more sensitive.** Long chains compound small errors, and there is evidence that heavily quantized models lose more on extended reasoning than short-form benchmarks suggest.
- **Native low precision is the trend.** FP8 training ([DeepSeek-V3](../../language-models/27-deepseek-v3/summary.md)) and low-bit-native architectures aim to avoid post-hoc conversion entirely.

---

## Further Reading

- **GPTQ:** [arxiv.org/abs/2210.17323](https://arxiv.org/abs/2210.17323)
- **AWQ:** [arxiv.org/abs/2306.00978](https://arxiv.org/abs/2306.00978)
- **LLM.int8() (the outlier problem):** [arxiv.org/abs/2208.07339](https://arxiv.org/abs/2208.07339)
- **SmoothQuant:** [arxiv.org/abs/2211.10438](https://arxiv.org/abs/2211.10438)
- **In this collection:** [QLoRA](../22-qlora/summary.md), [PagedAttention/vLLM](../52-pagedattention-vllm/summary.md), [Speculative Decoding](../45-speculative-decoding/summary.md)

## Citation

```bibtex
@inproceedings{frantar2023gptq,
  title={GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers},
  author={Frantar, Elias and Ashkboos, Saleh and Hoefler, Torsten and Alistarh, Dan},
  booktitle={International Conference on Learning Representations},
  year={2023}
}

@inproceedings{lin2024awq,
  title={AWQ: Activation-aware Weight Quantization for On-Device LLM Compression and Acceleration},
  author={Lin, Ji and Tang, Jiaming and Tang, Haotian and Yang, Shang and Dang, Xingyu and Gan, Chuang and Han, Song},
  booktitle={Proceedings of Machine Learning and Systems (MLSys)},
  year={2024}
}
```

<!-- related:start -->

---

## Related in This Collection

- [LoRA: Low-Rank Adaptation of Large Language Models](../../techniques/10-lora/summary.md)
- [QLoRA: Efficient Finetuning of Quantized LLMs](../../techniques/22-qlora/summary.md)
- [DeepSeek-V3 Technical Report](../../language-models/27-deepseek-v3/summary.md)
- [Speculative Decoding: Fast Inference from Transformers](../../techniques/45-speculative-decoding/summary.md)
- [PagedAttention: Efficient LLM Serving with vLLM](../../techniques/52-pagedattention-vllm/summary.md)
- [GQA: Grouped-Query Attention (and Multi-Query Attention)](../../architectures/75-grouped-query-attention/summary.md)

<!-- related:end -->
