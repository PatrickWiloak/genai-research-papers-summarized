# The AI Hardware Landscape

**In one line:** Frontier AI runs on a handful of accelerator families (NVIDIA GPUs, Google TPUs, AWS Trainium, AMD Instinct, and a few inference specialists), and the real limits are memory bandwidth, interconnect and chip supply, not raw arithmetic.
**Last reviewed:** 2026-09-30

---

## The short version

- **NVIDIA still sets the pace.** Its data centre GPUs went A100 (2020) to H100/H200 (Hopper) to B200/GB200 and GB300 (Blackwell), and the next generation, Rubin, is due from partners in the second half of 2026.
- **The hyperscalers build their own chips.** Google's TPUs (now the seventh-generation Ironwood) and AWS Trainium (now Trainium3) exist to lower cost and reduce dependence on one supplier. Anthropic, for example, trains and serves on both.
- **AMD is the main merchant alternative to NVIDIA.** Its Instinct MI300 and MI350 parts compete on memory capacity; the MI400 series with HBM4 is announced for 2026.
- **Specialist inference chips trade flexibility for speed.** Cerebras (wafer-scale) and Groq (LPU) serve tokens very fast; in December 2025 NVIDIA licensed Groq's technology and hired its leadership.
- **Arithmetic is not the bottleneck.** High-bandwidth memory (HBM) and the links between chips decide how fast a model actually runs, and HBM supply is sold out well ahead.
- **Export controls shape who gets what.** US rules have restricted advanced AI chips to China since 2022; the policy has shifted several times since, most recently in January 2026.

This page is the industry picture. For the practitioner view (which GPU to rent, how much memory a model needs), see the sibling repo's [GPUs for AI](https://github.com/PatrickWiloak/cloud-data-ai-security-zero-to-hero/blob/main/learn/concepts/gpus-for-ai.md).

## The mental model: a factory, a warehouse and the roads between them

An AI accelerator is three things bolted together:

```
   +-------------------+        +------------------+
   |  compute cores    | <----> |  HBM (on-package |    "the warehouse"
   |  (tensor units)   |  fast  |  memory stacks)  |
   |  "the factory"    |        +------------------+
   +-------------------+
            ^
            |  interconnect (NVLink, ICI, NeuronLink, Infinity Fabric)
            v                                            "the roads"
   +-------------------+   +-------------------+   +-------------------+
   |   accelerator 2   |---|   accelerator 3   |---|   ... up to 72+   |
   +-------------------+   +-------------------+   +-------------------+
```

- **Compute** is how many multiply-adds per second the chip can do, quoted in FLOPS (floating-point operations per second). Vendors quote peak numbers at low precision (FP8, FP4) and often "with sparsity", which doubles the headline. Real workloads get a fraction of peak.
- **Memory** is HBM (high-bandwidth memory): DRAM chips stacked vertically and placed right next to the processor. Two numbers matter: capacity (can the model fit?) and bandwidth (how fast can weights be streamed into the cores?).
- **Interconnect** is how chips talk to each other. No frontier model fits on one chip, so the model is split across many, and every training step or generated token requires them to exchange data.

A useful rule of thumb: **training is mostly compute-bound, generating text is mostly memory-bound.** When a model writes one token, it must read every active weight from memory once, but it does very little arithmetic per byte read. So a chip's memory bandwidth sets a hard ceiling on tokens per second for a single user. As a rough example, a 70-billion-parameter model stored at 8 bits per weight is about 70 GB; an H100 with 3.35 TB/s of bandwidth can stream that at most about 48 times a second, so roughly 48 tokens per second at batch size one, before any overhead. Batching many users together (see [Inference Economics](inference-economics.md)) is how providers escape this ceiling.

## NVIDIA: the reference platform

NVIDIA's advantage is not only the chips. It is CUDA (the programming platform almost all AI software targets), NVLink (its chip-to-chip interconnect), and rack-scale systems sold as one unit.

| Generation | Part (as sold) | HBM per GPU | Memory bandwidth | Chip-to-chip link | Notes |
|---|---|---|---|---|---|
| Ampere (2020) | A100 80GB SXM | 80 GB | 2.0 TB/s | NVLink 600 GB/s | Trained GPT-3-era and early GPT-4-era models |
| Hopper (2022) | H100 SXM | 80 GB HBM3 | 3.35 TB/s | NVLink 900 GB/s | The workhorse of 2023-2025; FP8 support |
| Hopper refresh | H200 SXM | 141 GB | 4.8 TB/s | NVLink 900 GB/s | Same compute as H100, more and faster memory |
| Blackwell (2024-25) | GB200 NVL72 rack | about 186 GB | 576 TB/s per 72-GPU rack | NVLink 5, 1.8 TB/s per GPU | Two-die GPU; 72 GPUs act as one NVLink domain |
| Blackwell Ultra (2025) | GB300 NVL72 rack | about 278 GB | 576 TB/s per rack | 1.8 TB/s per GPU | More memory per GPU for long-context reasoning |
| Rubin (H2 2026) | Vera Rubin NVL72 | up to 288 GB HBM4 | up to 22 TB/s per GPU | NVLink 6, 3.6 TB/s per GPU | 50 PFLOPS NVFP4 inference per GPU (vendor figure) |

All figures are from NVIDIA's product pages and announcements listed in Sources, as of September 2026.

Three trends stand out in that table:

1. **Memory grows faster than compute matters.** Per-GPU memory went from 80 GB to up to 288 GB in six years, and bandwidth from about 2 TB/s to up to 22 TB/s. The H200 is the clearest proof that memory is the bottleneck: it has the same compute as the H100, yet NVIDIA quotes up to 1.9x faster Llama 2 70B inference purely from memory.
2. **Precision keeps dropping.** Hopper added FP8; Blackwell added FP4 (NVIDIA's NVFP4 format). Halving the bits per number roughly doubles throughput and halves memory, if the model tolerates it. The training side of this story starts with [Mixed Precision Training](../../papers/techniques/143-mixed-precision-training/summary.md); the inference side is [quantization](../../papers/techniques/86-gptq-awq-quantization/summary.md).
3. **The unit of sale became the rack.** A GB200 NVL72 rack links 72 GPUs so tightly that software can treat them almost as one giant GPU. Rubin continues this: NVIDIA says the Vera Rubin NVL72 rack has 260 TB/s of scale-up bandwidth. (The rack was first announced as "NVL144", counting dies; it is now named by package count.)

NVIDIA claims Rubin will deliver up to a 10x reduction in inference token cost and a 4x reduction in GPUs needed to train mixture-of-experts models versus Blackwell. These are vendor claims made before general availability; treat them as upper bounds until independent benchmarks exist.

## Google TPUs

Google has designed its own Tensor Processing Units since 2015 and trained its Gemini models on them. TPUs are available only through Google Cloud.

- **Ironwood (TPU7x)**, the seventh generation, became generally available in 2026. Per chip: 192 GiB of HBM, about 7.4 TB/s of bandwidth, 4,614 TFLOPS of FP8, and 1.2 TB/s of inter-chip interconnect (ICI). Google says a full superpod links 9,216 chips for 42.5 FP8 exaflops.
- **Scale-up is the TPU's strength.** Google's optical switching lets thousands of chips share one high-bandwidth domain, where NVIDIA's NVLink domain is a single rack.
- **The eighth generation splits in two.** At Cloud Next in April 2026, Google previewed a training chip (TPU 8t) and a separate, cheaper inference chip (TPU 8i), both on TSMC's 2 nm process, with availability expected in late 2027.
- **Outside customers are growing.** In October 2025 Anthropic announced access to up to one million TPUs, with well over a gigawatt of capacity coming online in 2026.

## AWS Trainium and Inferentia

Amazon's Annapurna Labs designs two lines: Inferentia (inference) and Trainium (training, and increasingly inference too). They are programmed through the AWS Neuron SDK.

- **Trainium3**, announced generally available at re:Invent in December 2025, is AWS's first 3 nm chip: 2.52 PFLOPS of FP8 and 144 GB of HBM3e at 4.9 TB/s per chip. A Trn3 UltraServer links up to 144 chips (362 FP8 PFLOPS).
- AWS claims up to 4.4x more compute and 4x better energy efficiency than Trainium2 UltraServers.
- **Trainium4** was announced at the same event with at least 3x the FP8 compute and 4x the memory bandwidth of Trainium3; no ship date was given.
- Anthropic is the anchor customer: AWS named it among the first Trainium3 users.

## AMD Instinct

AMD is the only other company selling merchant data centre GPUs at frontier scale. Its pitch has consistently been **more memory per GPU**.

- **MI355X** (MI350 series, 2025): 288 GB of HBM3E at 8 TB/s.
- **MI455X** (MI400 series, 2026): 432 GB of HBM4 at 23.3 TB/s, about 40 PFLOPS MXFP4, per AMD's Hot Chips presentation in August 2026. AMD's "Helios" rack puts 72 of them together with 31 TB of HBM4. AMD says it expects to ship these systems; general availability had not been announced as of September 2026.
- AMD's software stack, ROCm, is the main gap versus CUDA. It has narrowed (vLLM and PyTorch support AMD officially), but most new techniques still land on NVIDIA first.

## Inference specialists

A different design school argues that GPUs carry too much baggage for serving a trained model. The main bets:

- **Cerebras** builds a single processor from a whole silicon wafer (WSE-3), keeping far more memory on-chip as SRAM, which is much faster than HBM. It sells mostly fast inference. It has a multi-year agreement with OpenAI to deploy 750 MW of its systems in stages starting in 2026, and it listed on Nasdaq in May 2026.
- **Groq** built the LPU (language processing unit), also SRAM-heavy and deterministic. In December 2025 NVIDIA agreed a non-exclusive licence to Groq's inference technology, reported at about $20 billion, and hired its founder Jonathan Ross and other leaders; Groq continues as an independent company. NVIDIA's March 2026 Vera Rubin materials already include a "Groq 3 LPX" rack of 256 LPUs for low-latency inference.
- Others exist (SambaNova, Etched, Tenstorrent and more). They are left out of the tables here because their public, verifiable specifications change often; check each vendor directly.

The trade-off: SRAM is fast but small, so these systems need many chips to hold a large model, and they are less flexible for training.

## The real bottlenecks

### HBM supply

HBM is made by three companies: SK hynix, Samsung and Micron. It takes several times the wafer area of ordinary DRAM per bit, and every accelerator above needs a lot of it. SK hynix said its 2026 HBM output was sold out, and industry reporting through September 2026 describes a broad memory shortage. When people say "GPU shortage", they often mean an HBM and advanced-packaging shortage.

### Advanced packaging

Putting HBM stacks next to a processor needs 2.5D packaging (TSMC's CoWoS is the dominant version). Packaging capacity, not chip fabrication, has repeatedly capped how many accelerators can ship.

### Interconnect

Large models are split across chips in several ways (data, tensor, pipeline and expert parallelism; see [ZeRO and Megatron-LM](../../papers/techniques/76-zero-megatron/summary.md)). Each split adds communication. That is why vendors now compete on scale-up domains (72 GPUs on NVLink, 144 Trainium3 chips, 9,216 Ironwood chips) as much as on per-chip FLOPS. [Mixture-of-experts](../../papers/architectures/37-mixture-of-experts/summary.md) models, which route each token to a few experts that may sit on different chips, lean on interconnect especially hard.

### Power

A frontier cluster is now measured in gigawatts, not chips. See [The Cost of Training](cost-of-training.md) for the energy side.

## Export controls

Since October 2022 the US has restricted exports of advanced AI chips and chipmaking tools to China, tightening the rules in 2023 and 2024. Chip vendors responded with cut-down China versions (H800, then H20). The policy has moved several times since:

| Date | Change |
|---|---|
| January 15, 2025 | The outgoing administration published the "AI Diffusion" rule, a tiered worldwide licensing system for AI chips. |
| April 2025 | The US required a licence for NVIDIA's H20 sales to China. |
| May 12-13, 2025 | Commerce's Bureau of Industry and Security (BIS) announced it would rescind the AI Diffusion rule before it took effect, and issued guidance instead (including on using Huawei Ascend chips). |
| January 13-15, 2026 | BIS moved licence review for H200, AMD MI325X and similar chips to China from "presumption of denial" to case-by-case, with conditions: no reduction in capacity available to US customers, customer screening, and independent US testing. A proclamation the same week imposed a 25% tariff on such chips under Section 232. |

The effect on the research picture is visible in the paper record: [DeepSeek-V3](../../papers/language-models/27-deepseek-v3/summary.md) was trained on H800s, export-compliant Hopper chips with reduced interconnect, and much of its engineering (FP8 training, communication overlap) works around exactly that constraint. Meanwhile China is scaling domestic chips such as Huawei's Ascend line. Both the rules and their enforcement remain in flux; see [US AI Policy](../policy/us-ai-policy.md).

## What to watch

- **Rubin in the field (H2 2026).** Whether independent benchmarks confirm NVIDIA's claimed inference cost reductions.
- **AMD MI400 and Helios availability (2026).** The first real test of whether HBM4 capacity plus ROCm wins frontier customers.
- **TPU 8t and 8i (expected late 2027).** Whether splitting training and inference silicon becomes the industry norm.
- **Trainium4 timing.** Not yet dated by AWS.
- **HBM supply.** Memory makers have warned of shortage into later years; any easing would lower accelerator costs across the board.
- **Export policy.** Licence volumes under the January 2026 H200 rule, and any replacement for the rescinded diffusion framework.
- **NVIDIA and Groq integration.** Whether LPU-style inference racks become a standard part of NVIDIA deployments.

## Read next

- [The Cost of Training](cost-of-training.md) and [Inference Economics](inference-economics.md) - what this hardware costs to use
- [Mixed Precision Training](../../papers/techniques/143-mixed-precision-training/summary.md) - why lower precision formats matter
- [ZeRO and Megatron-LM](../../papers/techniques/76-zero-megatron/summary.md) - how models are split across chips
- [FlashAttention](../../papers/techniques/16-flash-attention/summary.md) - an algorithm designed around the memory hierarchy
- [DeepSeek-V3](../../papers/language-models/27-deepseek-v3/summary.md) - frontier training on export-restricted hardware
- [KV Cache](../concepts/kv-cache.md) - why inference memory grows with context
- [Labs Landscape](../ecosystem/labs-landscape.md) - who owns which compute
- Sibling repo: [GPUs for AI](https://github.com/PatrickWiloak/cloud-data-ai-security-zero-to-hero/blob/main/learn/concepts/gpus-for-ai.md) and [Inference Servers](https://github.com/PatrickWiloak/cloud-data-ai-security-zero-to-hero/blob/main/learn/concepts/inference-servers.md)

## Sources

- NVIDIA, A100 product page: https://www.nvidia.com/en-us/data-center/a100/
- NVIDIA, H100 product page: https://www.nvidia.com/en-us/data-center/h100/
- NVIDIA, H200 product page: https://www.nvidia.com/en-us/data-center/h200/
- NVIDIA, GB200 NVL72 product page: https://www.nvidia.com/en-us/data-center/gb200-nvl72/
- NVIDIA, GB300 NVL72 product page: https://www.nvidia.com/en-us/data-center/gb300-nvl72/
- NVIDIA Newsroom, "NVIDIA Kicks Off the Next Generation of AI With Rubin" (January 5, 2026): https://nvidianews.nvidia.com/news/rubin-platform-ai-supercomputer
- NVIDIA Technical Blog, "Inside NVIDIA Rubin GPU Architecture" (HBM4 capacity and bandwidth): https://developer.nvidia.com/blog/inside-nvidia-rubin-gpu-architecture-powering-the-era-of-agentic-ai/
- NVIDIA Technical Blog, "NVIDIA Vera Rubin POD" (March 16, 2026; NVL72 and Groq 3 LPX): https://developer.nvidia.com/blog/nvidia-vera-rubin-pod-seven-chips-five-rack-scale-systems-one-ai-supercomputer/
- TechPowerUp, on the NVL144 to NVL72 naming change: https://www.techpowerup.com/342049/nvidia-vera-rubin-nvl144-servers-set-for-2026-volume-production
- Google Cloud, TPU7x (Ironwood) documentation: https://docs.cloud.google.com/tpu/docs/tpu7x
- Google, "Ironwood: The first Google TPU for the age of inference": https://blog.google/innovation-and-ai/infrastructure-and-cloud/google-cloud/ironwood-tpu-age-of-inference/
- The Next Web, Ironwood GA and eighth-generation TPU preview (April 22, 2026): https://thenextweb.com/news/google-ironwood-tpu-inference-cloud-next
- Google Cloud Press Corner, "Anthropic to Expand Use of Google Cloud TPUs and Services" (October 23, 2025): https://www.googlecloudpresscorner.com/2025-10-23-Anthropic-to-Expand-Use-of-Google-Cloud-TPUs-and-Services
- About Amazon, "Trainium3 UltraServers now available": https://www.aboutamazon.com/news/aws/trainium-3-ultraserver-faster-ai-training-lower-cost
- AWS What's New, "Announcing Amazon EC2 Trn3 UltraServers" (December 2025): https://aws.amazon.com/about-aws/whats-new/2025/12/amazon-ec2-trn3-ultraservers/
- AMD, Instinct MI350 Series: https://www.amd.com/en/products/accelerators/instinct/mi350.html
- ServeTheHome, "AMD MI400 GPU at Hot Chips 2026" (August 24, 2026): https://www.servethehome.com/amd-mi400-gpu-at-hot-chips-2026/
- Cerebras, "OpenAI partners with Cerebras": https://www.cerebras.ai/blog/openai-partners-with-cerebras-to-bring-high-speed-inference-to-the-mainstream
- CNBC, Cerebras IPO (May 14, 2026): https://www.cnbc.com/2026/05/14/cerebras-ipo-mints-two-billionaires-sets-stage-for-potential-ai-wave.html
- Groq, "Groq and Nvidia Enter Non-Exclusive Inference Technology Licensing Agreement": https://groq.com/newsroom/groq-and-nvidia-enter-non-exclusive-inference-technology-licensing-agreement-to-accelerate-ai-inference-at-global-scale
- CNBC, on the reported $20 billion value (December 24, 2025): https://www.cnbc.com/2025/12/24/nvidia-buying-ai-chip-startup-groq-for-about-20-billion-biggest-deal.html
- TechSpot, SK hynix sells out 2026 memory capacity: https://www.techspot.com/news/110058-sk-hynix-completely-sells-out-semiconductor-supply-ai.html
- BIS, "Department of Commerce Announces Rescission of Biden-Era Artificial Intelligence Diffusion Rule" (May 2025): https://www.bis.gov/press-release/department-commerce-announces-rescission-biden-era-artificial-intelligence-diffusion-rule-strengthens
- BIS, "Department of Commerce Revises License Review Policy for Semiconductors Exported to China" (January 13, 2026): https://www.bis.gov/press-release/department-commerce-revises-license-review-policy-semiconductors-exported-china
- Congressional Research Service, "U.S. Export Controls and China: Advanced Semiconductors" (R48642): https://www.congress.gov/crs-product/R48642
- CNAS, "Unpacking the H200 Export Policy" (tariff and conditions): https://www.cnas.org/publications/cnas-insights/cnas-insights-unpacking-the-h200-export-policy
- DeepSeek-AI, "DeepSeek-V3 Technical Report" (H800 cluster): https://arxiv.org/abs/2412.19437
