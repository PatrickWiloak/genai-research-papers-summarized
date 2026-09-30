# The Open-Source AI Stack

**In one line:** Open AI tooling sorts into five layers (get a model, train it, serve it at scale, run it locally, build an app on it), and knowing which layer a tool lives in tells you what it is for and what it replaces.
**Last reviewed:** 2026-09-30

---

## The short version

- **Hugging Face Hub is where open models live**, and the `transformers` library is the reference code that defines them. Almost every other tool reads models from the Hub.
- **PyTorch is the foundation.** Nearly all open training and serving tools are built on it. Hugging Face `transformers` v5 dropped TensorFlow and Flax to go PyTorch-only.
- **Serving at scale means vLLM or SGLang** (open, multi-vendor) or NVIDIA's TensorRT-LLM (NVIDIA-only, highly tuned). Orchestrators such as NVIDIA Dynamo and llm-d spread them across clusters.
- **Running on your own machine means llama.cpp and its GGUF format**, usually through a friendlier wrapper such as Ollama or LM Studio. On Apple silicon, MLX is the native alternative.
- **Fine-tuning** is mostly TRL (Hugging Face's post-training library), Unsloth (speed and low memory) or Axolotl (config-driven).
- **Application frameworks** (LangChain/LangGraph, LlamaIndex and others) sit on top and wire models to tools, data and each other.
- **The OpenAI-compatible API is the glue.** Most servers and local runners expose it, so apps can switch between them by changing a URL. See [Agent Protocols](agent-protocols.md).

## The mental model: five layers

```
  +---------------------------------------------------------------+
  | 5. APPLICATION   LangChain / LangGraph, LlamaIndex, ...        |
  |                  (agents, RAG, tool calling, orchestration)    |
  +---------------------------------------------------------------+
  | 4. LOCAL         Ollama, LM Studio  ->  llama.cpp (GGUF), MLX  |
  |    (laptop,      (one user, consumer hardware)                 |
  |     desktop)                                                   |
  +---------------------------------------------------------------+
  | 3. SERVING       vLLM, SGLang, TensorRT-LLM                    |
  |    (data centre) + cluster layer: NVIDIA Dynamo, llm-d         |
  +---------------------------------------------------------------+
  | 2. TRAINING /    TRL, Unsloth, Axolotl (+ PEFT for LoRA)       |
  |    FINE-TUNING                                                 |
  +---------------------------------------------------------------+
  | 1. MODELS AND    Hugging Face Hub (weights, datasets)          |
  |    FOUNDATION    transformers (model definitions), PyTorch     |
  +---------------------------------------------------------------+
                         hardware: GPUs, TPUs, Apple silicon, CPUs
```

A typical path: download an open-weight model from the Hub (layer 1), fine-tune it with LoRA using Unsloth or TRL (layer 2), then either serve it to many users with vLLM (layer 3) or convert it to GGUF and run it on a laptop with Ollama (layer 4), and build the product with an application framework (layer 5).

## The map in one table

| Tool | Layer | What it is for | Who is behind it | Licence |
|---|---|---|---|---|
| **Hugging Face Hub** | Models | Hosting and versioning open model weights, datasets and demos ("Spaces") | Hugging Face | Service (models carry their own licences) |
| **transformers** | Models | Reference Python definitions of hundreds of model architectures; load, run and train them | Hugging Face | Apache-2.0 |
| **PyTorch** | Foundation | The deep learning framework underneath almost everything here | PyTorch Foundation (Linux Foundation) | BSD-style |
| **vLLM** | Serving | High-throughput serving engine built on PagedAttention; runs on NVIDIA, AMD, TPU, Trainium and more | Started at UC Berkeley; PyTorch Foundation project since May 2025 | Apache-2.0 |
| **SGLang** | Serving | High-performance serving with fast prefix reuse and structured generation | Started as LMSYS research; now the sgl-project community | Apache-2.0 |
| **TensorRT-LLM** | Serving | NVIDIA's compiler-style inference library, tuned per NVIDIA GPU | NVIDIA | Open source (see repo) |
| **NVIDIA Dynamo** | Serving (cluster) | Orchestrates vLLM, SGLang or TensorRT-LLM across many GPUs, with prefill and decode split apart | NVIDIA (open-sourced March 2025) | Open source (see repo) |
| **llm-d** | Serving (cluster) | Kubernetes-native distributed inference built on vLLM with KV-cache-aware routing | Red Hat, Google Cloud, IBM Research, CoreWeave, NVIDIA; CNCF sandbox | Apache-2.0 |
| **llama.cpp** | Local | C/C++ inference that runs quantized models on CPUs, Apple silicon and consumer GPUs | ggml-org; ggml.ai joined Hugging Face in February 2026 | MIT |
| **GGUF** | Local (format) | Single-file model format with built-in quantization, used by llama.cpp and its wrappers | ggml project | Open format |
| **Ollama** | Local | One-command download and run of models, with a local API (native and OpenAI-compatible) | Ollama | MIT |
| **LM Studio** | Local | Desktop app with a GUI, model browser and local server; runs llama.cpp (GGUF) and, on Macs, MLX | LM Studio | Free app; CLI is MIT |
| **MLX** | Local / research | Array framework for Apple silicon, with an ecosystem for running and fine-tuning LLMs on Macs | Apple machine learning research | MIT |
| **TRL** | Fine-tuning | Supervised fine-tuning and preference/RL methods (DPO, GRPO and others) on top of transformers | Hugging Face | Apache-2.0 |
| **Unsloth** | Fine-tuning | Faster, lower-memory LoRA and full fine-tuning with hand-written kernels; exports to GGUF | Unsloth AI | Apache-2.0 |
| **Axolotl** | Fine-tuning | YAML-config-driven fine-tuning that wraps the Hugging Face stack for multi-GPU runs | Axolotl AI | Apache-2.0 |
| **LangChain / LangGraph** | Application | Agent building blocks (LangChain) on a stateful graph runtime (LangGraph); both reached 1.0 in October 2025 | LangChain Inc. | MIT |
| **LlamaIndex** | Application | Connecting LLMs to documents: ingestion, parsing, indexing, retrieval (RAG) | LlamaIndex Inc. | MIT |

Licences and descriptions are from each project's repository as of September 2026. Always check a model's own licence separately: the tool being open source says nothing about the weights you run in it (see [Open vs Closed Weights](../concepts/open-vs-closed-weights.md)).

## Layer by layer

### 1. Models and foundation: Hugging Face and PyTorch

The **Hugging Face Hub** is to open models what GitHub is to code. Labs such as Meta, Alibaba (Qwen), Mistral, DeepSeek and Google publish open weights there, alongside a very large number of community fine-tunes and quantized versions.

**transformers** is the library that turns a checkpoint on the Hub into running PyTorch code. With v5 (release candidate announced December 2025), Hugging Face described it as the model-definition layer that other tools (vLLM, SGLang, llama.cpp, MLX, Unsloth, Axolotl) align to. Hugging Face reported more than 400 model architectures and about 3 million daily installs at that time. v5 also dropped TensorFlow and Flax to go PyTorch-only.

**PyTorch** is governed by the PyTorch Foundation under the Linux Foundation. In 2025 the foundation expanded into an umbrella for other projects, with vLLM among the first.

### 2. Fine-tuning: TRL, Unsloth, Axolotl

Most people do not train models from scratch; they adapt an existing open model. The dominant technique is [LoRA](../../papers/techniques/10-lora/summary.md) (training small add-on matrices instead of all weights), often with [QLoRA](../../papers/techniques/22-qlora/summary.md) (doing it on a 4-bit quantized base to save memory).

- **TRL** is the Hugging Face library for post-training: supervised fine-tuning, and preference and reinforcement methods such as [DPO](../../papers/language-models/19-dpo/summary.md) and [GRPO](../../papers/techniques/38-grpo/summary.md).
- **Unsloth** focuses on doing the same work faster and in less GPU memory, which makes single-GPU and free-notebook fine-tuning practical. It can export straight to GGUF for local use.
- **Axolotl** wraps the same Hugging Face components behind a single YAML config file, which suits reproducible and multi-GPU runs.

The choice is mostly ergonomic: TRL for flexibility and new methods, Unsloth for small hardware, Axolotl for config-driven pipelines. The sibling repo's [Fine-tuning vs RAG](https://github.com/PatrickWiloak/cloud-data-ai-security-zero-to-hero/blob/main/learn/concepts/fine-tuning-vs-rag.md) covers when to fine-tune at all.

### 3. Serving: vLLM, SGLang, TensorRT-LLM

Serving engines exist because naive inference wastes most of the GPU (see [Inference Economics](../compute/inference-economics.md)).

- **vLLM** came out of the [PagedAttention paper](../../papers/techniques/52-pagedattention-vllm/summary.md) and is the most widely used open serving engine. Its strength is breadth: many model architectures and many hardware backends (NVIDIA, AMD, Google TPU, AWS Neuron, Intel and others).
- **SGLang** grew out of research on structured LLM programs. Its RadixAttention reuses shared prompt prefixes across requests, which suits agents and multi-turn chat. It joined the PyTorch ecosystem in March 2025.
- **TensorRT-LLM** is NVIDIA's own engine. It is built to get the most out of NVIDIA hardware, at the cost of being NVIDIA-only.

On top of single-node engines sits a newer **cluster layer**. NVIDIA **Dynamo** (announced March 2025 as the successor to its Triton server) and **llm-d** (launched by Red Hat and partners in May 2025, now a CNCF sandbox project) both split prefill and decode onto separate GPU pools and route requests to wherever the relevant KV cache already sits.

vLLM and SGLang expose an OpenAI-compatible API, and `transformers` v5 added a `transformers serve` command that does too. The sibling repo's [Inference Servers](https://github.com/PatrickWiloak/cloud-data-ai-security-zero-to-hero/blob/main/learn/concepts/inference-servers.md) page is the hands-on companion.

### 4. Local: llama.cpp, GGUF, Ollama, LM Studio, MLX

Running a model on your own machine is a different problem: one user, limited memory, often no data centre GPU.

- **llama.cpp**, started by Georgi Gerganov in March 2023, runs models in plain C/C++ with aggressive quantization (see [GPTQ and AWQ](../../papers/techniques/86-gptq-awq-quantization/summary.md) for the ideas). It is the engine underneath much of local AI. In February 2026 ggml.ai, the company behind it, joined Hugging Face; the projects stay MIT-licensed and community-run.
- **GGUF** is llama.cpp's model file format. One file holds the weights (quantized to 8, 5, 4 or fewer bits), the tokenizer and metadata. A file name ending in something like `Q4_K_M.gguf` is a 4-bit GGUF build.
- **Ollama** wraps this into `ollama run <model>`: it downloads, caches and serves models, exposing its own API plus OpenAI-compatible Chat Completions and Responses endpoints on `localhost:11434`.
- **LM Studio** is a desktop app with a model browser and chat UI. It runs GGUF models through llama.cpp on Mac, Windows and Linux, and MLX models on Apple silicon; it can also act as an MCP client.
- **MLX** is Apple's array framework for Apple silicon, released in late 2023. Macs share one pool of memory between CPU and GPU, so a Mac with a lot of RAM can run models that would not fit on a consumer graphics card. The `mlx-lm` package handles running and fine-tuning LLMs.

### 5. Application frameworks

These libraries do not run models; they call them (local or hosted) and connect them to everything else.

- **LangChain and LangGraph** reached 1.0 in October 2025. LangGraph is the low-level runtime for stateful, long-running agents; LangChain is now a higher-level API on top of it, centred on a `create_agent` helper.
- **LlamaIndex** specialises in the data side: parsing documents, building indexes and retrieval for [RAG](../../papers/techniques/13-rag/summary.md). The sibling repo's [RAG Explained](https://github.com/PatrickWiloak/cloud-data-ai-security-zero-to-hero/blob/main/learn/concepts/rag-explained.md) is the hands-on version.
- Many others fill the same layer, including Hugging Face's smolagents, CrewAI, Pydantic AI and the agent SDKs from OpenAI, Anthropic and Google. They change quickly; pick on fit rather than popularity.

An honest caveat: for simple applications, many teams skip frameworks and call the model API directly, because an extra abstraction layer can make debugging harder. Anthropic's [Building Effective Agents](../../papers/essays/115-building-effective-agents/summary.md) makes this argument explicitly.

## What connects the layers

Three conventions make this stack interchangeable:

1. **Hugging Face model repos** as the common source of weights and configs.
2. **GGUF** as the common local format.
3. **The OpenAI-compatible HTTP API** as the common serving interface, now joined by [MCP](../../papers/techniques/59-model-context-protocol/summary.md) for tools. Both are covered in [Agent Protocols](agent-protocols.md).

## What to watch

- **transformers v5 settling.** How quickly downstream tools align to it as the single source of model definitions.
- **llama.cpp under Hugging Face.** Whether faster same-day GGUF support for new models follows, as announced.
- **The cluster serving layer.** Dynamo, llm-d and engine-native features overlap heavily; expect consolidation.
- **Responses-style APIs in open servers.** Ollama and others already expose them; see [Agent Protocols](agent-protocols.md).
- **Non-NVIDIA backends.** vLLM and SGLang support for AMD MI400 and new TPU generations will decide whether open serving stays hardware-neutral in practice.

## Read next

- [PagedAttention / vLLM](../../papers/techniques/52-pagedattention-vllm/summary.md), [LoRA](../../papers/techniques/10-lora/summary.md), [QLoRA](../../papers/techniques/22-qlora/summary.md), [GPTQ and AWQ](../../papers/techniques/86-gptq-awq-quantization/summary.md)
- [Agent Protocols](agent-protocols.md) and [Labs Landscape](labs-landscape.md)
- [Inference Economics](../compute/inference-economics.md) and [The AI Hardware Landscape](../compute/ai-hardware-landscape.md)
- [Open vs Closed Weights](../concepts/open-vs-closed-weights.md), [Llama](../model-families/llama.md), [Qwen](../model-families/qwen.md), [Mistral](../model-families/mistral.md)
- Sibling repo: [Inference Servers](https://github.com/PatrickWiloak/cloud-data-ai-security-zero-to-hero/blob/main/learn/concepts/inference-servers.md), [Quantization and Distillation](https://github.com/PatrickWiloak/cloud-data-ai-security-zero-to-hero/blob/main/learn/concepts/quantization-and-distillation.md), [Fine-tuning vs RAG](https://github.com/PatrickWiloak/cloud-data-ai-security-zero-to-hero/blob/main/learn/concepts/fine-tuning-vs-rag.md)

## Sources

- Hugging Face, "Transformers v5: Simple model definitions powering the AI ecosystem" (December 1, 2025): https://huggingface.co/blog/transformers-v5
- Hugging Face transformers repository: https://github.com/huggingface/transformers
- PyTorch Foundation, "PyTorch Foundation Welcomes vLLM as a Hosted Project" (May 2025): https://pytorch.org/blog/pytorch-foundation-welcomes-vllm/
- vLLM repository: https://github.com/vllm-project/vllm
- SGLang repository: https://github.com/sgl-project/sglang
- Zheng et al., "SGLang: Efficient Execution of Structured Language Model Programs" (2023): https://arxiv.org/abs/2312.07104
- TensorRT-LLM repository: https://github.com/NVIDIA/TensorRT-LLM
- NVIDIA, "NVIDIA Dynamo Open-Source Library Accelerates and Scales AI Reasoning Models" (March 18, 2025): https://www.globenewswire.com/news-release/2025/03/18/3044894/0/en/NVIDIA-Dynamo-Open-Source-Library-Accelerates-and-Scales-AI-Reasoning-Models.html
- NVIDIA Dynamo repository: https://github.com/ai-dynamo/dynamo
- Red Hat, "Red Hat Launches the llm-d Community" (May 20, 2025): https://www.redhat.com/en/about/press-releases/red-hat-launches-llm-d-community-powering-distributed-gen-ai-inference-scale
- llm-d repository: https://github.com/llm-d/llm-d
- llama.cpp repository: https://github.com/ggml-org/llama.cpp
- Hugging Face, "GGML and llama.cpp join HF to ensure the long-term progress of Local AI" (February 2026): https://huggingface.co/blog/ggml-joins-hf
- Ollama repository: https://github.com/ollama/ollama
- Ollama, OpenAI compatibility documentation: https://docs.ollama.com/api/openai-compatibility
- LM Studio documentation: https://lmstudio.ai/docs/app
- LM Studio CLI repository: https://github.com/lmstudio-ai/lms
- MLX repository: https://github.com/ml-explore/mlx
- TRL repository: https://github.com/huggingface/trl
- Unsloth repository: https://github.com/unslothai/unsloth
- Axolotl repository: https://github.com/axolotl-ai-cloud/axolotl
- LangChain, "LangChain and LangGraph Agent Frameworks Reach v1.0 Milestones" (October 2025): https://www.langchain.com/blog/langchain-langgraph-1dot0
- LlamaIndex repository: https://github.com/run-llama/llama_index
