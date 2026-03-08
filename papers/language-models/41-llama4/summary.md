# Llama 4: Natively Multimodal Open-Source AI

**Authors:** Meta AI
**Published:** April 5, 2025
**Blog:** [ai.meta.com/blog/llama-4-multimodal-intelligence/](https://ai.meta.com/blog/llama-4-multimodal-intelligence/)

---

## Why This Matters

Llama 4 is **the first open-source natively multimodal MoE model family**:

- 🔓 **Fully open source** - Open weights under Llama license
- 🏗️ **MoE architecture** - Meta's first Mixture-of-Experts models
- 📚 **10M token context** - Scout has the longest context window of any production model
- 🖼️ **Natively multimodal** - Text, image, and video from the ground up
- 🏆 **Competitive with GPT-4o** - Maverick beats GPT-4o on multiple benchmarks

**Real-world impact:**
- Brought MoE + native multimodality to open source
- 10M token context enables entirely new use cases
- Competitive with proprietary models at a fraction of the cost
- Meta's strongest statement yet in the open-source AI race

**The insight:** **Combine MoE efficiency with native multimodality** to create open models that compete with the best closed-source systems.

---

## Model Family

### Two Models at Launch

| Aspect | Llama 4 Scout | Llama 4 Maverick |
|--------|--------------|-----------------|
| **Total params** | 109B | 400B+ |
| **Active params** | 17B | 17B |
| **Experts** | 16 | 128 |
| **Shared experts** | 1 | 1 |
| **Routing** | Top-1 + shared | Top-1 + shared |
| **Context window** | **10M tokens** | 1M tokens |
| **Multimodal** | Text + Image | Text + Image + Video |
| **Hardware** | Single H100 (int4) | Single H100 host |

**Both models have 17B active parameters** - despite very different total sizes. This is the MoE efficiency at work.

### Llama 4 Scout

**The long-context specialist:**
```
10M token context window:
- Read entire codebases
- Process thousands of documents
- Analyze weeks of conversation history
- Handle massive knowledge bases

For comparison:
- GPT-4: 128K tokens
- Claude 3.5: 200K tokens
- Gemini 2.5: 1M tokens
- Llama 4 Scout: 10M tokens (10x Gemini!)
```

**Fits on a single H100 GPU** with int4 quantization - making it remarkably accessible.

### Llama 4 Maverick

**The quality leader:**
```
128 experts (vs Scout's 16):
- More specialized knowledge
- Better at complex reasoning
- Stronger multimodal performance
- Video understanding

Benchmarks vs GPT-4o:
- Beats GPT-4o across multiple benchmarks
- Matches DeepSeek-V3 on reasoning and coding
- At less than half the active parameters
```

---

## Architecture

### MoE Design

**Alternating dense and MoE layers:**

```
Layer 1: Dense (standard transformer block)
Layer 2: MoE (router + 128 experts + shared expert)
Layer 3: Dense
Layer 4: MoE
...

Why alternating?
- Dense layers provide shared computation
- MoE layers provide specialized computation
- Balance between generalization and specialization
```

### Expert Routing

**Top-1 routing with shared expert:**

```
For each token at each MoE layer:
1. Shared expert ALWAYS processes the token (common knowledge)
2. Router selects 1 additional expert (specialized knowledge)
3. Outputs are combined

Total active: Shared expert + 1 routed expert
Much more efficient than top-2 routing (Mixtral)
```

```python
# Simplified Llama 4 MoE routing
def moe_forward(token, experts, shared_expert, router):
    # Shared expert always runs
    shared_output = shared_expert(token)

    # Route to 1 specialized expert
    scores = router(token)  # [128 scores]
    best_expert = argmax(scores)
    expert_output = experts[best_expert](token)

    # Combine
    weight = softmax(scores)[best_expert]
    output = shared_output + weight * expert_output
    return output
```

### Native Multimodality

**Trained from scratch with visual data:**

```
Input processing:
  Text  → Token embedding → Transformer
  Image → Vision encoder → Patch tokens → Transformer
  Video → Frame-by-frame → Temporal tokens → Transformer

All modalities share the same transformer backbone
Cross-attention between modalities
```

---

## Performance

### vs Closed-Source Models

**Llama 4 Maverick vs competitors:**

| Benchmark | Maverick | GPT-4o | Gemini 2.0 Flash | DeepSeek-V3 |
|-----------|----------|--------|-------------------|-------------|
| **LMSYS ELO** | 1417 | ~1400 | ~1380 | ~1410 |
| **Reasoning** | Competitive | Good | Good | **Best** |
| **Coding** | Competitive | Good | Good | **Best** |
| **Multilingual** | Good | Good | Good | Competitive |
| **Multimodal** | **Strong** | Strong | Strong | Text only |

**Maverick achieves GPT-4o level with open weights and 17B active params.**

### Llama 4 Scout vs Competitors

| Benchmark | Scout | Gemma 3 | Gemini 2.0 Flash-Lite | Mistral 3.1 |
|-----------|-------|---------|----------------------|-------------|
| **Overall** | **Best** | Good | Good | Good |
| **Context length** | **10M** | 128K | 1M | 128K |
| **Active params** | 17B | 12B | - | 24B |

**Best in class for models in its size range.**

### Long-Context Performance

```
Llama 4 Scout at 10M context:
- Needle-in-a-haystack: Near perfect retrieval
- Multi-document QA: State-of-the-art
- Code repository understanding: Excellent
- Extended conversation: Maintains coherence

Use cases unlocked:
- Entire codebase analysis (millions of lines)
- Legal document review (thousands of pages)
- Research paper synthesis (hundreds of papers)
- Long-running agent conversations
```

---

## What Changed from Llama 3

### Architecture Shift

| Aspect | Llama 3 | Llama 4 |
|--------|---------|---------|
| **Architecture** | Dense | MoE |
| **Multimodal** | Text only (mostly) | Native multimodal |
| **Largest active** | 405B | 17B |
| **Context** | 128K | 10M |
| **Experts** | None | 16-128 |

### Training Innovations

```
Llama 4 training improvements:
1. Native multimodal pre-training (not post-hoc)
2. MoE training with load balancing
3. Extended context training (progressive lengthening)
4. Improved data quality and curation
5. Better multilingual data coverage
```

---

## Practical Usage

### Running Locally

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "meta-llama/Llama-4-Scout-17B-16E-Instruct"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",
    device_map="auto",
    load_in_4bit=True  # Fits on single H100
)

messages = [
    {"role": "user", "content": "Explain the MoE architecture in Llama 4"}
]

inputs = tokenizer.apply_chat_template(messages, return_tensors="pt")
outputs = model.generate(inputs.to(model.device), max_new_tokens=500)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

### With Vision

```python
from transformers import AutoProcessor

processor = AutoProcessor.from_pretrained(model_name)

messages = [
    {
        "role": "user",
        "content": [
            {"type": "image", "url": "https://example.com/chart.png"},
            {"type": "text", "text": "Analyze this chart and summarize the trends."}
        ]
    }
]

inputs = processor(messages, return_tensors="pt").to(model.device)
outputs = model.generate(**inputs, max_new_tokens=500)
print(processor.decode(outputs[0], skip_special_tokens=True))
```

### API Access

Available through major cloud providers:
```
- Meta AI (direct)
- Together AI
- AWS Bedrock
- Azure AI
- Google Cloud
- Groq, Fireworks, etc.
```

---

## Reception and Controversy

### Mixed Initial Reception

```
Praise:
+ First open-source natively multimodal MoE
+ 10M context is genuinely industry-leading
+ Fits on accessible hardware
+ Strong benchmark performance

Criticism:
- Some benchmarks appeared cherry-picked
- Early users reported quality inconsistencies
- Maverick underperformed expectations on some tasks
- Community debated whether benchmarks matched real-world use
```

### The Benchmark Debate

```
Meta reported strong LMSYS ELO scores
Community noted discrepancies between:
- Official benchmarks (very competitive)
- User experience (sometimes underwhelming)

This reflects a broader industry issue:
benchmark gaming vs real-world performance
```

---

## Limitations

### 1. Memory Requirements
```
Despite 17B active params, total model is large:
- Scout: 109B total (all weights in memory)
- Maverick: 400B+ total
Need significant VRAM even with quantization
```

### 2. MoE Complexity
```
MoE models are harder to:
- Fine-tune (expert balancing)
- Quantize (per-expert calibration)
- Deploy efficiently (expert parallelism)
```

### 3. Video Understanding
```
Video support is early-stage
Frame-by-frame processing, not true temporal understanding
Limited compared to Gemini's video capabilities
```

---

## Key Takeaways

1. **MoE goes open source** - First major open model family to adopt Mixture-of-Experts
2. **10M token context** - Orders of magnitude beyond competitors
3. **17B active params** - Frontier quality from efficient inference
4. **Native multimodal** - Text, image, and video from the ground up
5. **Open competition** - Matches GPT-4o level with open weights

**Bottom line:** Llama 4 represents Meta's most ambitious open-source release, combining MoE, native multimodality, and extreme context lengths. While reception was mixed, it pushed the boundary of what's available as open-source AI.

---

## Further Reading

### Official Resources
- **Blog Post:** https://ai.meta.com/blog/llama-4-multimodal-intelligence/
- **Model Page:** https://www.llama.com/models/llama-4/

### Model Downloads
- **Hugging Face:** https://huggingface.co/meta-llama
- **Scout:** https://huggingface.co/meta-llama/Llama-4-Scout-17B-16E-Instruct
- **Maverick:** https://huggingface.co/meta-llama/Llama-4-Maverick-17B-128E-Instruct

### Related Work
- **Llama 3:** https://arxiv.org/abs/2407.21783
- **Mixtral (MoE inspiration):** https://arxiv.org/abs/2401.04088
- **DeepSeek-V3 (advanced MoE):** https://arxiv.org/abs/2412.19437

---

**Published:** April 5, 2025
**Impact:** 🔥🔥🔥🔥 **HIGH** - Open-source MoE + multimodal milestone
**Adoption:** Growing - available across major cloud platforms
**Current Relevance:** Latest open-source frontier model family
**Legacy:** Brought MoE architecture and native multimodality to open source

**Modern Status (March 2026):** Llama 4 Scout and Maverick are widely deployed, particularly for long-context use cases. The 10M token context window remains industry-leading. Meta is expected to release additional Llama 4 variants including a larger "Behemoth" model.
