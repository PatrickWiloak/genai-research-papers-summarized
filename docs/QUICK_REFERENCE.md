# Quick Reference Guide - All Papers at a Glance

One line on what each of the **107 papers** contributed and why it mattered, grouped by
category and ordered by paper number. This is the lookup table; for a card view with relevance
badges see [BROWSE.md](../BROWSE.md), for the generated listings see [INDEX.md](../INDEX.md) and
[TAGS.md](../TAGS.md), and for a guided order see [ROADMAP.md](./ROADMAP.md).

---

## Architecture Papers

| # | Paper | Year | Key Contribution | Impact |
|---|-------|------|------------------|--------|
| 1 | **[Attention Is All You Need](../papers/architectures/01-attention-is-all-you-need/summary.md)** | 2017 | Self-attention and the Transformer; no recurrence, fully parallel | The architecture underneath essentially every model in this collection |
| 11 | **[Vision Transformer (ViT)](../papers/architectures/11-vision-transformer/summary.md)** | 2020 | Treat image patches as a token sequence | Removed the CNN prior; the vision encoder in modern multimodal models |
| 20 | **[Mamba](../papers/architectures/20-mamba/summary.md)** | 2023 | Selective state-space model with linear-time sequence scaling | The most credible non-attention alternative; now used in hybrids |
| 37 | **[Mixtral & Mixture-of-Experts](../papers/architectures/37-mixture-of-experts/summary.md)** | 2024 | Sparse mixture of experts, top-2 of 8 routing | MoE became the default frontier architecture |
| 55 | **[Seq2Seq](../papers/architectures/55-seq2seq/summary.md)** | 2014 | LSTM encoder-decoder mapping variable-length sequences | Established encoder-decoder; its bottleneck motivated attention |
| 66 | **[Bahdanau Attention](../papers/architectures/66-bahdanau-attention/summary.md)** | 2014 | Learned soft alignment over all encoder states | Invented attention; direct ancestor of self-attention |
| 67 | **[Switch Transformer](../papers/architectures/67-switch-transformer/summary.md)** | 2021 | Top-1 expert routing at trillion-parameter scale | Made sparse MoE stable enough to actually train |
| 73 | **[ResNet](../papers/architectures/73-resnet/summary.md)** | 2015 | Identity skip connections make very deep networks trainable | Residual connections sit in every Transformer block |
| 74 | **[U-Net](../papers/architectures/74-unet/summary.md)** | 2015 | Symmetric encoder-decoder with skip connections | The denoiser backbone of the entire diffusion era |
| 75 | **[Grouped-Query Attention](../papers/architectures/75-grouped-query-attention/summary.md)** | 2023 | Share key/value heads across groups of query heads | Shrinks the KV cache; standard in modern LLM serving |
| 88 | **[Masked Autoencoders (MAE)](../papers/architectures/88-mae/summary.md)** | 2021 | Mask 75% of patches, reconstruct with an asymmetric decoder | Scalable self-supervised pretraining for vision |

---

## Language Model Papers

| # | Paper | Year | Key Contribution | Impact |
|---|-------|------|------------------|--------|
| 3 | **[BERT](../papers/language-models/03-bert/summary.md)** | 2018 | Bidirectional masked-language-model pretraining | Made pretrain-then-fine-tune standard; still runs in search and retrieval |
| 4 | **[GPT-3](../papers/language-models/04-gpt3-few-shot-learners/summary.md)** | 2020 | 175B parameters; in-context few-shot learning | Prompting displaced fine-tuning; the direct run-up to ChatGPT |
| 5 | **[InstructGPT (RLHF)](../papers/language-models/05-instructgpt-rlhf/summary.md)** | 2022 | Reinforcement learning from human feedback on a base model | Turned raw LLMs into assistants; the recipe behind ChatGPT and Claude |
| 14 | **[Constitutional AI](../papers/language-models/14-constitutional-ai/summary.md)** | 2022 | Model critiques itself against written principles (RLAIF) | Cut human labelling out of alignment; the method behind Claude |
| 15 | **[LLaMA](../papers/language-models/15-llama/summary.md)** | 2023 | Compute-optimal training; smaller models trained far longer | Kicked off the open-weight ecosystem |
| 17 | **[LLaMA 2](../papers/language-models/17-llama2/summary.md)** | 2023 | Open weights with a fully documented RLHF chat pipeline | Made commercially usable open chat models mainstream |
| 19 | **[DPO](../papers/language-models/19-dpo/summary.md)** | 2023 | Optimise preferences directly, with no reward model or RL loop | Replaced PPO-based RLHF for most open fine-tunes |
| 26 | **[DeepSeek-R1](../papers/language-models/26-deepseek-r1/summary.md)** | 2025 | Reasoning learned by pure RL on verifiable rewards | Open model matching o1, and published what OpenAI kept closed |
| 27 | **[DeepSeek-V3](../papers/language-models/27-deepseek-v3/summary.md)** | 2024 | 671B MoE, 37B active, trained for under $6M | Frontier quality at a fraction of Western training budgets |
| 28 | **[Qwen3](../papers/language-models/28-qwen3/summary.md)** | 2025 | Hybrid thinking / non-thinking modes in a single model | Leading open-weight multilingual family |
| 30 | **[Claude 3.5 Sonnet](../papers/language-models/30-claude-3.5-sonnet/summary.md)** | 2024 | Computer use: driving a GUI from screenshots | Opened the desktop-automation agent category |
| 31 | **[OpenAI o1](../papers/language-models/31-openai-o1/summary.md)** | 2024 | RL training to spend inference compute on hidden reasoning | Created the reasoning-model category |
| 33 | **[LLaMA 3.3](../papers/language-models/33-llama3.3/summary.md)** | 2024 | Distillation and post-training recover 405B quality at 70B | Frontier-class open models on modest hardware |
| 36 | **[GPT-4](../papers/language-models/36-gpt4/summary.md)** | 2023 | Large multimodal model with predictable scaling, disclosure withheld | The capability jump that made LLMs a general product platform |
| 40 | **[GPT-4o](../papers/language-models/40-gpt4o/summary.md)** | 2024 | One network natively trained across text, audio and vision | Made real-time conversational voice practical |
| 41 | **[Llama 4](../papers/language-models/41-llama4/summary.md)** | 2025 | Open-weight MoE, natively multimodal, 10M-token context | Brought frontier architecture choices into open weights |
| 42 | **[GPT-5](../papers/language-models/42-gpt5/summary.md)** | 2025 | A unified model routing between fast answers and deep reasoning | Collapsed the confusing model-picker era into one system |
| 43 | **[Claude 4 Family](../papers/language-models/43-claude4/summary.md)** | 2025 | Extended thinking plus long-horizon agentic tool use | State of the art on real software-engineering work |
| 56 | **[Codex](../papers/language-models/56-codex/summary.md)** | 2021 | GPT fine-tuned on public code; HumanEval and pass@k | Produced GitHub Copilot and the whole code-model lineage |
| 64 | **[GPT-2](../papers/language-models/64-gpt2/summary.md)** | 2019 | A 1.5B-parameter LM performing tasks zero-shot | Established scale-plus-prompting, and the staged-release precedent |
| 65 | **[T5](../papers/language-models/65-t5/summary.md)** | 2019 | Cast every NLP task as text-to-text, with systematic ablations | Standardised the framing; its encoder still powers image models |
| 93 | **[GPT-1](../papers/language-models/93-gpt1/summary.md)** | 2018 | Generative pretraining, then discriminative fine-tuning | The original GPT recipe, at 117M parameters |
| 94 | **[PaLM](../papers/language-models/94-palm/summary.md)** | 2022 | 540B dense model trained across two TPU pods via Pathways | The practical ceiling for dense models; strong CoT results |
| 95 | **[Mistral 7B](../papers/language-models/95-mistral-7b/summary.md)** | 2023 | GQA and sliding-window attention in a small Apache-2.0 model | Made open weights genuinely competitive |
| 96 | **[Llama Guard](../papers/language-models/96-llama-guard/summary.md)** | 2023 | An LLM fine-tuned to classify prompt and response safety | The open reference for production content moderation |

---

## Image & Video Generation Papers

| # | Paper | Year | Key Contribution | Impact |
|---|-------|------|------------------|--------|
| 2 | **[GANs](../papers/image-generation/02-generative-adversarial-networks/summary.md)** | 2014 | Generator and discriminator trained against each other | Launched modern generative modelling; dominant for images until diffusion |
| 6 | **[Diffusion Models (DDPM)](../papers/image-generation/06-diffusion-models/summary.md)** | 2020 | Learn to reverse a fixed noising process | Made diffusion practical and displaced GANs for image generation |
| 7 | **[Stable Diffusion](../papers/image-generation/07-stable-diffusion/summary.md)** | 2022 | Run diffusion in a compressed VAE latent space | 10-100x cheaper generation; put text-to-image on consumer GPUs |
| 44 | **[Sora & Diffusion Transformers](../papers/image-generation/44-sora-dit/summary.md)** | 2022-2024 | Replace the U-Net with a Transformer; video as spacetime patches | Launched text-to-video and the DiT backbone |
| 48 | **[DALL-E 3](../papers/image-generation/48-dalle3/summary.md)** | 2023 | Retrain on synthetic, highly descriptive captions | Fixed prompt following and text rendering in images |
| 57 | **[Variational Autoencoder (VAE)](../papers/image-generation/57-vae/summary.md)** | 2013 | Amortised variational inference via the reparameterisation trick | Gave generative models a usable latent space; the encoder in latent diffusion |
| 69 | **[Classifier-Free Guidance](../papers/image-generation/69-classifier-free-guidance/summary.md)** | 2021 | Interpolate conditional and unconditional predictions | The guidance-scale dial and negative prompts in every image model |
| 70 | **[DDIM](../papers/image-generation/70-ddim/summary.md)** | 2020 | Deterministic non-Markovian sampling of a trained diffusion model | Cut sampling from 1000 steps to ~20 with no retraining |
| 71 | **[ControlNet](../papers/image-generation/71-controlnet/summary.md)** | 2023 | A trainable encoder copy injecting spatial conditioning | Precise pose, depth and edge control over image generation |
| 72 | **[Flow Matching / SD3](../papers/image-generation/72-flow-matching-sd3/summary.md)** | 2022-2024 | Learn straight probability paths instead of a noising chain | The formulation behind SD3 and Flux |
| 89 | **[VQ-VAE](../papers/image-generation/89-vq-vae/summary.md)** | 2017 | Quantise latents against a learned codebook | Made images and audio tokenisable for autoregressive models |
| 90 | **[VQ-GAN](../papers/image-generation/90-vq-gan/summary.md)** | 2020 | VQ-VAE tokens with adversarial and perceptual losses | The compress-then-model recipe behind high-resolution synthesis |
| 91 | **[Imagen](../papers/image-generation/91-imagen/summary.md)** | 2022 | A frozen large text encoder plus cascaded diffusion | Showed text understanding, not the image model, was the bottleneck |
| 92 | **[DreamBooth](../papers/image-generation/92-dreambooth/summary.md)** | 2022 | Bind a rare token to a subject with a prior-preservation loss | Created the personalisation and fine-tune-sharing ecosystem |

---

## Multimodal Papers

| # | Paper | Year | Key Contribution | Impact |
|---|-------|------|------------------|--------|
| 8 | **[CLIP](../papers/multimodal/08-clip/summary.md)** | 2021 | Contrastive image-text pretraining on 400M pairs | Zero-shot classification, and the text conditioning inside image generators |
| 23 | **[GPT-4V(ision)](../papers/multimodal/23-gpt4v/summary.md)** | 2023 | Production vision-language system card | Brought image understanding into mainstream assistant products |
| 29 | **[Gemini 2.5](../papers/multimodal/29-gemini-2.5/summary.md)** | 2025 | Long-context multimodal reasoning with thinking budgets | Pushed production context windows past 1M tokens |
| 32 | **[SAM 2](../papers/multimodal/32-sam2/summary.md)** | 2024 | Promptable segmentation unified across images and video | Foundation-model segmentation with memory across frames |
| 46 | **[LLaVA](../papers/multimodal/46-llava/summary.md)** | 2023 | Connect a frozen vision encoder to an LLM with a projection layer | Made open multimodal models buildable by anyone |
| 47 | **[Gemini 3](../papers/multimodal/47-gemini3/summary.md)** | 2025 | Frontier multimodal reasoning with a Deep Think mode | First model past 1500 LMArena Elo |
| 49 | **[Whisper](../papers/multimodal/49-whisper/summary.md)** | 2022 | Weakly supervised ASR on 680k hours across 99 languages | The default open speech model; robust with no fine-tuning |

---

## Technique & Method Papers

| # | Paper | Year | Key Contribution | Impact |
|---|-------|------|------------------|--------|
| 9 | **[Chain-of-Thought](../papers/techniques/09-chain-of-thought/summary.md)** | 2022 | Prompt the model to show intermediate steps | Unlocked multi-step reasoning; ancestor of every reasoning model |
| 10 | **[LoRA](../papers/techniques/10-lora/summary.md)** | 2021 | Freeze the base model, train low-rank update matrices | Made fine-tuning affordable; the default adapter format |
| 12 | **[Scaling Laws](../papers/techniques/12-scaling-laws/summary.md)** | 2020 | Loss falls as a power law in compute, data and parameters | Made frontier training budgets predictable rather than speculative |
| 13 | **[RAG](../papers/techniques/13-rag/summary.md)** | 2020 | Retrieve documents at inference and condition generation on them | The default way production systems ground answers in private data |
| 16 | **[FlashAttention](../papers/techniques/16-flash-attention/summary.md)** | 2022 | IO-aware tiled attention that never materialises the N x N matrix | Longer contexts and faster training on the same hardware |
| 18 | **[Chinchilla](../papers/techniques/18-chinchilla/summary.md)** | 2022 | For a fixed compute budget, scale data alongside parameters | Corrected GPT-3-era over-parameterisation; every lab retuned to it |
| 21 | **[ReAct](../papers/techniques/21-react/summary.md)** | 2023 | Interleave reasoning traces with tool-use actions | The loop underneath most LLM agent frameworks |
| 22 | **[QLoRA](../papers/techniques/22-qlora/summary.md)** | 2023 | LoRA on top of a 4-bit frozen base model | 65B fine-tuning on a single consumer GPU |
| 24 | **[Toolformer](../papers/techniques/24-toolformer/summary.md)** | 2023 | Model teaches itself when to call APIs, self-supervised | Early proof that tool use can be learned rather than prompted |
| 25 | **[Tree of Thoughts](../papers/techniques/25-tree-of-thoughts/summary.md)** | 2023 | Search over branching reasoning paths with backtracking | Generalised CoT into deliberate search; precursor to test-time compute |
| 34 | **[Meta-CoT](../papers/techniques/34-meta-cot/summary.md)** | 2025 | Model the search process itself, not just the final chain | Theoretical framing for System 2 reasoning in LLMs |
| 35 | **[rStar-Math](../papers/techniques/35-rstar-math/summary.md)** | 2025 | Small models self-evolve via MCTS with process rewards | 7B models matching o1-preview on competition maths |
| 38 | **[GRPO](../papers/techniques/38-grpo/summary.md)** | 2024 | Group-relative advantages replace PPO's learned critic | Made large-scale reasoning RL cheap enough to run; powers R1 |
| 39 | **[RLVR](../papers/techniques/39-rlvr/summary.md)** | 2024-2025 | Reward verifiable correctness instead of learned human preference | The training signal behind the reasoning-model generation |
| 45 | **[Speculative Decoding](../papers/techniques/45-speculative-decoding/summary.md)** | 2022 | A draft model proposes tokens, the target model verifies | 2-3x faster serving with mathematically identical outputs |
| 50 | **[Scaling Test-Time Compute](../papers/techniques/50-test-time-compute/summary.md)** | 2024 | Trade inference compute against parameter count, optimally | The theory that justified o1-style reasoning models |
| 51 | **[Process Reward Models](../papers/techniques/51-process-reward-models/summary.md)** | 2023 | Supervise each reasoning step, not just the final answer | How reasoning models verify their chains; PRM800K released |
| 52 | **[PagedAttention & vLLM](../papers/techniques/52-pagedattention-vllm/summary.md)** | 2023 | Paged, non-contiguous KV-cache allocation | 24x serving throughput; vLLM is the industry default |
| 53 | **[Word2Vec](../papers/techniques/53-word2vec/summary.md)** | 2013 | Dense embeddings learned from context prediction | Started representation learning for language |
| 54 | **[RoPE](../papers/techniques/54-rope-rotary-position-embedding/summary.md)** | 2021 | Encode relative position by rotating queries and keys | The positional scheme in nearly every modern LLM |
| 58 | **[Generative Agents](../papers/techniques/58-generative-agents/summary.md)** | 2023 | Memory stream plus reflection and planning | The reference architecture for long-running agent simulations |
| 59 | **[Model Context Protocol (MCP)](../papers/techniques/59-model-context-protocol/summary.md)** | 2024 | An open client-server protocol for model-tool integration | Collapsed M x N custom integrations; adopted industry-wide |
| 60 | **[GraphRAG](../papers/techniques/60-graph-rag/summary.md)** | 2024 | LLM-extracted knowledge graph with community summaries | Answers corpus-level questions that vector RAG cannot |
| 61 | **[AlphaGeometry](../papers/techniques/61-alphageometry/summary.md)** | 2024 | Neuro-symbolic prover: LM proposes constructions, solver deduces | Olympiad-medallist geometry with no human proof data |
| 62 | **[AlphaEvolve](../papers/techniques/62-alphaevolve/summary.md)** | 2025 | LLM-guided evolutionary search over programs | Beat a 56-year-old matrix-multiplication record; deployed at Google |
| 63 | **[PPO](../papers/techniques/63-ppo/summary.md)** | 2017 | Clipped surrogate objective for stable policy updates | The RL algorithm inside RLHF |
| 68 | **[AlphaFold 2](../papers/techniques/68-alphafold/summary.md)** | 2021 | Evoformer plus a structure module over multiple sequence alignments | Solved protein structure prediction; 200M+ structures released |
| 76 | **[ZeRO & Megatron-LM](../papers/techniques/76-zero-megatron/summary.md)** | 2019 | Shard optimiser state, gradients and parameters across devices | How trillion-parameter training physically happens |
| 77 | **[Self-Consistency](../papers/techniques/77-self-consistency/summary.md)** | 2022 | Sample many chains and take the majority answer | The cheapest reliable accuracy gain on reasoning tasks |
| 78 | **[Reflexion](../papers/techniques/78-reflexion/summary.md)** | 2023 | Agent writes verbal self-critique into episodic memory | Let agents learn from failure without weight updates |
| 79 | **[Self-Instruct](../papers/techniques/79-self-instruct/summary.md)** | 2022 | Bootstrap instruction data from the model itself | Made instruction tuning cheap; produced Alpaca and its descendants |
| 80 | **[FLAN](../papers/techniques/80-flan/summary.md)** | 2021 | Multi-task instruction fine-tuning for zero-shot transfer | Established instruction tuning as a standard training stage |
| 81 | **[Emergent Abilities](../papers/techniques/81-emergent-abilities/summary.md)** | 2022-2023 | Do capabilities appear discontinuously with scale? | Paired with the Mirage rebuttal; a lesson in metric choice |
| 82 | **[Sparse Autoencoders](../papers/techniques/82-sparse-autoencoders/summary.md)** | 2022-2024 | Decompose activations into sparse, interpretable features | The leading interpretability method; enabled feature steering |
| 83 | **[Sleeper Agents](../papers/techniques/83-sleeper-agents/summary.md)** | 2024 | Backdoors that survive safety fine-tuning | Showed current alignment training can fail to remove deception |
| 84 | **[SWE-bench](../papers/techniques/84-swe-bench/summary.md)** | 2023 | Resolve real GitHub issues in real repositories | The benchmark frontier coding agents are measured on |
| 85 | **[LLM-as-a-Judge](../papers/techniques/85-llm-as-judge/summary.md)** | 2023 | Use a strong model as evaluator; MT-Bench and Chatbot Arena | How most model comparisons are now run |
| 86 | **[GPTQ & AWQ](../papers/techniques/86-gptq-awq-quantization/summary.md)** | 2022-2023 | Post-training 4-bit weight quantisation | 70B models on one consumer GPU; why local inference exists |
| 87 | **[Dense Retrieval](../papers/techniques/87-dense-retrieval/summary.md)** | 2019-2020 | DPR, ColBERT and Sentence-BERT embedding retrieval | The retriever layer under every RAG system |
| 97 | **[STaR](../papers/techniques/97-star/summary.md)** | 2022 | Keep self-generated rationales that reach the right answer, retrain | The self-improvement loop behind modern reasoning training |
| 98 | **[Quiet-STaR](../papers/techniques/98-quiet-star/summary.md)** | 2024 | Learn latent rationales before every token | Generalised STaR to arbitrary text; predates o1 |
| 99 | **[Self-Refine](../papers/techniques/99-self-refine/summary.md)** | 2023 | The same model generates, critiques and revises | The LLM-as-critic pattern used across agents and reasoning |
| 100 | **[Voyager](../papers/techniques/100-voyager/summary.md)** | 2023 | Lifelong agent that writes and stores reusable skills as code | The first convincing open-ended LLM agent |
| 101 | **[AlphaFold 3](../papers/techniques/101-alphafold3/summary.md)** | 2024 | Diffusion-based prediction over proteins, DNA, RNA and ligands | Extended structure prediction to biomolecular complexes |
| 102 | **[AlphaZero](../papers/techniques/102-alphazero/summary.md)** | 2017 | Self-play RL with MCTS from the rules alone | The ancestor of every self-improvement training loop |
| 103 | **[KTO](../papers/techniques/103-kto/summary.md)** | 2024 | Prospect-theoretic loss on unpaired binary feedback | Alignment without expensive paired preference data |
| 104 | **[Genie](../papers/techniques/104-genie/summary.md)** | 2024 | Latent action model learned unsupervised from video | First foundation world model; playable worlds from one image |
| 105 | **[DreamerV3](../papers/techniques/105-dreamerv3/summary.md)** | 2023 | Model-based RL that learns inside an imagined world model | One hyperparameter set across 150+ tasks; Minecraft diamonds |
| 106 | **[ESM-2 / ESMFold](../papers/techniques/106-esm/summary.md)** | 2023 | Protein language model predicting structure without MSAs | Scaling laws transfer to biology; 617M structures released |
| 107 | **[CICERO](../papers/techniques/107-cicero/summary.md)** | 2022 | Language model fused with a strategic planning engine | Human-level Diplomacy, negotiation included |

---

## Browsing by Topic or Year

These views are generated from [`papers.json`](../papers.json), so they never go stale:

- **By topic** - [TAGS.md](../TAGS.md) groups every paper under its topic tags
  (retrieval, alignment, efficiency, agents, reasoning, interpretability, and the rest).
- **By year** - the By Year block in the [README](../README.md) is regenerated on every
  build; [INDEX.md](../INDEX.md) has the full clickable list by category.

This page deliberately does not repeat those groupings by hand - a second, hand-typed copy is
the thing that drifts.

---

## Numbers Worth Remembering

Figures taken from the summaries themselves, not from memory.

| Paper | Scale | Note |
|-------|-------|------|
| [BERT](../papers/language-models/03-bert/summary.md) | 110M / 340M params | Base and Large; tiny by current standards |
| [GPT-1](../papers/language-models/93-gpt1/summary.md) | 117M params | Where the recipe starts |
| [GPT-2](../papers/language-models/64-gpt2/summary.md) | 1.5B params | The "too dangerous to release" model |
| [GPT-3](../papers/language-models/04-gpt3-few-shot-learners/summary.md) | 175B params, 300B tokens | Under-trained by Chinchilla's rule |
| [Chinchilla](../papers/techniques/18-chinchilla/summary.md) | 70B params, 1.4T tokens | ~20 tokens per parameter; beat 280B Gopher |
| [LLaMA](../papers/language-models/15-llama/summary.md) | 65B params, 1.4T tokens | Chinchilla-optimal, public data only |
| [PaLM](../papers/language-models/94-palm/summary.md) | 540B params | Largest dense Transformer of its era |
| [Mixtral](../papers/architectures/37-mixture-of-experts/summary.md) | 47B total, 13B active | Sparse MoE, top-2 of 8 experts |
| [DeepSeek-V3](../papers/language-models/27-deepseek-v3/summary.md) | 671B total, 37B active | Frontier quality under $6M of compute |
| [CLIP](../papers/multimodal/08-clip/summary.md) | 400M image-text pairs | Scraped, not labelled |
| [Whisper](../papers/multimodal/49-whisper/summary.md) | 680k hours, 99 languages | Weak supervision at scale |

---

## Problem → Solution Mapping

**Modelling and architecture**

| Problem | Papers |
|---------|--------|
| Sequences can't be processed in parallel | [Transformer](../papers/architectures/01-attention-is-all-you-need/summary.md) (1) |
| The encoder-decoder bottleneck loses information | [Bahdanau Attention](../papers/architectures/66-bahdanau-attention/summary.md) (66), [Seq2Seq](../papers/architectures/55-seq2seq/summary.md) (55) |
| Deep networks won't train | [ResNet](../papers/architectures/73-resnet/summary.md) (73) |
| Attention is quadratic in sequence length | [FlashAttention](../papers/techniques/16-flash-attention/summary.md) (16), [Mamba](../papers/architectures/20-mamba/summary.md) (20) |
| Capacity is too expensive to serve | [Mixtral / MoE](../papers/architectures/37-mixture-of-experts/summary.md) (37), [Switch Transformer](../papers/architectures/67-switch-transformer/summary.md) (67) |
| Position needs encoding without hurting extrapolation | [RoPE](../papers/techniques/54-rope-rotary-position-embedding/summary.md) (54) |
| The KV cache dominates serving memory | [GQA](../papers/architectures/75-grouped-query-attention/summary.md) (75), [PagedAttention](../papers/techniques/52-pagedattention-vllm/summary.md) (52) |

**Training and alignment**

| Problem | Papers |
|---------|--------|
| No idea how much compute or data a run needs | [Scaling Laws](../papers/techniques/12-scaling-laws/summary.md) (12), [Chinchilla](../papers/techniques/18-chinchilla/summary.md) (18) |
| The model is capable but not helpful | [InstructGPT](../papers/language-models/05-instructgpt-rlhf/summary.md) (5), [FLAN](../papers/techniques/80-flan/summary.md) (80) |
| RLHF needs a reward model and an RL loop | [DPO](../papers/language-models/19-dpo/summary.md) (19), [GRPO](../papers/techniques/38-grpo/summary.md) (38) |
| Preference data must be collected in pairs | [KTO](../papers/techniques/103-kto/summary.md) (103) |
| Human preference labels are expensive | [Constitutional AI](../papers/language-models/14-constitutional-ai/summary.md) (14), [RLVR](../papers/techniques/39-rlvr/summary.md) (39) |
| Instruction data is expensive | [Self-Instruct](../papers/techniques/79-self-instruct/summary.md) (79) |
| A model won't fit in memory to train | [ZeRO & Megatron-LM](../papers/techniques/76-zero-megatron/summary.md) (76) |
| Fine-tuning the whole model costs too much | [LoRA](../papers/techniques/10-lora/summary.md) (10), [QLoRA](../papers/techniques/22-qlora/summary.md) (22) |
| The RL algorithm itself is unstable | [PPO](../papers/techniques/63-ppo/summary.md) (63) |

**Reasoning**

| Problem | Papers |
|---------|--------|
| The model answers hard questions in one shot | [Chain-of-Thought](../papers/techniques/09-chain-of-thought/summary.md) (9) |
| A single chain is unreliable | [Self-Consistency](../papers/techniques/77-self-consistency/summary.md) (77), [Tree of Thoughts](../papers/techniques/25-tree-of-thoughts/summary.md) (25) |
| Only the final answer is supervised | [Process Reward Models](../papers/techniques/51-process-reward-models/summary.md) (51) |
| Reasoning data has to be written by hand | [STaR](../papers/techniques/97-star/summary.md) (97), [Quiet-STaR](../papers/techniques/98-quiet-star/summary.md) (98) |
| Bigger models are the only way to improve | [Test-Time Compute](../papers/techniques/50-test-time-compute/summary.md) (50), [OpenAI o1](../papers/language-models/31-openai-o1/summary.md) (31) |
| The first draft is wrong | [Self-Refine](../papers/techniques/99-self-refine/summary.md) (99), [Reflexion](../papers/techniques/78-reflexion/summary.md) (78) |

**Knowledge, tools and agents**

| Problem | Papers |
|---------|--------|
| The model hallucinates or lacks private data | [RAG](../papers/techniques/13-rag/summary.md) (13), [Dense Retrieval](../papers/techniques/87-dense-retrieval/summary.md) (87) |
| Retrieval can't answer corpus-level questions | [GraphRAG](../papers/techniques/60-graph-rag/summary.md) (60) |
| The model can't act on the world | [ReAct](../papers/techniques/21-react/summary.md) (21), [Toolformer](../papers/techniques/24-toolformer/summary.md) (24) |
| Every tool needs a bespoke integration | [MCP](../papers/techniques/59-model-context-protocol/summary.md) (59) |
| Agents forget across long horizons | [Generative Agents](../papers/techniques/58-generative-agents/summary.md) (58), [Voyager](../papers/techniques/100-voyager/summary.md) (100) |

**Generation**

| Problem | Papers |
|---------|--------|
| Need photorealistic images | [GANs](../papers/image-generation/02-generative-adversarial-networks/summary.md) (2), [DDPM](../papers/image-generation/06-diffusion-models/summary.md) (6) |
| Diffusion is too slow and too expensive | [Stable Diffusion](../papers/image-generation/07-stable-diffusion/summary.md) (7), [DDIM](../papers/image-generation/70-ddim/summary.md) (70), [Flow Matching](../papers/image-generation/72-flow-matching-sd3/summary.md) (72) |
| Output ignores the prompt | [CFG](../papers/image-generation/69-classifier-free-guidance/summary.md) (69), [DALL-E 3](../papers/image-generation/48-dalle3/summary.md) (48), [Imagen](../papers/image-generation/91-imagen/summary.md) (91) |
| Need control over composition | [ControlNet](../papers/image-generation/71-controlnet/summary.md) (71) |
| Need a specific person, pet or product | [DreamBooth](../papers/image-generation/92-dreambooth/summary.md) (92) |
| Images need to be modelled as tokens | [VQ-VAE](../papers/image-generation/89-vq-vae/summary.md) (89), [VQ-GAN](../papers/image-generation/90-vq-gan/summary.md) (90) |
| Need video, not stills | [Sora / DiT](../papers/image-generation/44-sora-dit/summary.md) (44), [Genie](../papers/techniques/104-genie/summary.md) (104) |

**Deployment and evaluation**

| Problem | Papers |
|---------|--------|
| Inference is too slow | [Speculative Decoding](../papers/techniques/45-speculative-decoding/summary.md) (45), [PagedAttention](../papers/techniques/52-pagedattention-vllm/summary.md) (52) |
| The model won't fit on the GPU you have | [GPTQ & AWQ](../papers/techniques/86-gptq-awq-quantization/summary.md) (86), [QLoRA](../papers/techniques/22-qlora/summary.md) (22) |
| Benchmarks don't reflect real work | [SWE-bench](../papers/techniques/84-swe-bench/summary.md) (84), [LLM-as-a-Judge](../papers/techniques/85-llm-as-judge/summary.md) (85) |
| Benchmark jumps may be an artefact | [Emergent Abilities](../papers/techniques/81-emergent-abilities/summary.md) (81) |
| Nobody knows what the model is doing | [Sparse Autoencoders](../papers/techniques/82-sparse-autoencoders/summary.md) (82) |
| Safety training may not have worked | [Sleeper Agents](../papers/techniques/83-sleeper-agents/summary.md) (83), [Llama Guard](../papers/language-models/96-llama-guard/summary.md) (96) |

---

## Paper Dependencies

If a summary assumes something you don't have yet, it is almost always one of these.

```
Attention (66) ──► Transformer (1) ──► everything else
                        │
                        ├──► BERT (3) ──────────► encoder models, retrieval (87)
                        ├──► GPT-1 (93) ► GPT-2 (64) ► GPT-3 (4) ► GPT-4 (36) ► GPT-5 (42)
                        ├──► ViT (11) ──────────► CLIP (8) ──► LLaVA (46), multimodal
                        └──► MoE (67, 37) ──────► DeepSeek-V3 (27), Llama 4 (41)

Scaling Laws (12) ──► Chinchilla (18) ──► LLaMA (15) ──► Mistral (95), open weights

VAE (57) ──► DDPM (6) ──► Stable Diffusion (7) ──► SD3 / Flow Matching (72)
   │            │              ▲
   │            └──► DDIM (70) │
   └──► VQ-VAE (89) ► VQ-GAN (90)
        U-Net (74) ───────────┘

PPO (63) ──► InstructGPT (5) ──► DPO (19) / KTO (103)
   │                              │
   └──► GRPO (38) ──► RLVR (39) ──┴──► DeepSeek-R1 (26)

CoT (9) ──► Self-Consistency (77) ──► Tree of Thoughts (25) ──► Test-Time Compute (50) ──► o1 (31)
   └──► STaR (97) ──► Quiet-STaR (98)
```

**Read first, in this order, if you are starting cold:** Transformer (1) → GPT-3 (4) →
InstructGPT (5) → Chain-of-Thought (9) → RAG (13). Everything else has a path back to those five.

---

## When to Reference Each Paper

| If you are... | Read |
|---------------|------|
| Building a chatbot or assistant | 4, 5, 14, 13, 43 |
| Adding retrieval over your own documents | 13, 87, 60 |
| Fine-tuning on a budget | 10, 22, 86 |
| Building an agent that uses tools | 21, 24, 59, 78, 100 |
| Serving a model cheaply at scale | 52, 45, 86, 75 |
| Training a model from scratch | 12, 18, 76, 1, 54 |
| Improving reasoning quality | 9, 77, 51, 50, 26 |
| Choosing an alignment method | 5, 19, 103, 38, 39 |
| Generating images | 6, 7, 69, 70, 71, 92 |
| Generating video | 44, 104 |
| Working with speech | 49 |
| Building multimodal features | 8, 11, 46, 23 |
| Evaluating models honestly | 84, 85, 81 |
| Auditing a model for safety | 82, 83, 96, 14 |
| Applying this outside language | 68, 101, 106, 102, 105, 107 |

---

## Reading Time

Measured from the summaries in this repository, not estimated:

| | |
|---|---|
| Summaries | 107 |
| Words across all summaries | ~203,000 |
| Average summary | ~1,900 words (about 8-10 minutes) |
| Whole collection, cover to cover | ~17 hours at 200 words/minute |

The Quick Stats table in the [README](../README.md) carries the canonical figures, including how
much source material the summaries compress. Those numbers are regenerated by
`scripts/build_manifest.py`, so they cannot drift from the papers themselves.

---

**Papers:** 107 · **See also:** [BROWSE.md](../BROWSE.md) · [INDEX.md](../INDEX.md) · [TAGS.md](../TAGS.md) · [ROADMAP.md](./ROADMAP.md) · [COMPARISONS.md](./COMPARISONS.md)
