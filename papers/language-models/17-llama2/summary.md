# LLaMA 2: Open Foundation and Fine-Tuned Chat Models

**Authors:** Hugo Touvron, Louis Martin, et al. (Meta AI)
**Published:** July 2023
**Paper:** [arxiv.org/abs/2307.09288](https://arxiv.org/abs/2307.09288)

---

## Why This Matters

LLaMA 2 is the **most important open-source LLM** for production use. This release:

- 🔥 **Fully open license** - Free for commercial use (unlike LLaMA 1)
- 🎯 **Competitive with ChatGPT** - Matches GPT-3.5 on many benchmarks
- 📚 **Open safety practices** - Detailed RLHF methodology shared
- 🌍 **Democratized AI** - Anyone can deploy ChatGPT-quality models
- 💰 **Saved millions** - Companies can self-host instead of API costs

**Real-world impact:**
- Powers countless startups and products
- Basis for hundreds of fine-tuned models
- Standard for open-source AI in 2023-2024

---

## Key Improvements Over LLaMA 1

### 1. Commercial License
**LLaMA 1:** Research-only
**LLaMA 2:** Free commercial use (under 700M users)

### 2. Better Training
- **2 trillion tokens** (vs 1.4T in LLaMA 1)
- 40% more data
- Longer context: 4k tokens (vs 2k)

### 3. Chat Models with RLHF
- **LLaMA 2-Chat** - Instruction-tuned versions
- Full RLHF training
- Safety improvements

### 4. Model Sizes
- 7B, 13B, 70B parameters
- 34B trained but not released

---

## Training Details

### Pre-training
- **Data:** 2T tokens from public sources
- **Context:** 4096 tokens
- **Compute:** ~1.7M GPU hours (A100)
- **Architecture:** Same as LLaMA 1 (RoPE, RMSNorm, SwiGLU)

### Fine-tuning (Chat Models)
**Stage 1: Supervised Fine-Tuning (SFT)**
- 27,540 high-quality examples
- Focused on helpfulness and safety

**Stage 2: RLHF**
- Reward modeling on human preferences
- Iterative RLHF (multiple rounds)
- Rejection sampling + PPO

**Innovation: Ghost Attention (GAtt)**
- Maintains context across conversation turns
- Prevents forgetting system prompt

---

## Performance

### Benchmarks vs Competitors

| Model | MMLU | HumanEval | BBH | GSM8k |
|-------|------|-----------|-----|-------|
| GPT-3.5 | 70.0 | 48.1 | ~50 | 57.1 |
| **LLaMA 2 70B** | **68.9** | **29.9** | ~50 | **56.8** |
| LLaMA 2 13B | 54.8 | 18.3 | 39.4 | 28.7 |
| LLaMA 2 7B | 45.3 | 12.8 | 32.6 | 14.6 |

### Chat Model Performance
**LLaMA 2-Chat 70B:**
- Competitive with ChatGPT on helpfulness
- Better than ChatGPT on safety benchmarks
- Preferred over ChatGPT in 36% of comparisons

---

## Safety & Alignment

### Red Teaming
- 350+ prompts tested
- Violation rate: ~0.1% (vs 1-2% for competitors)

### Safety Categories
- Illicit & Criminal
- Hateful & Harmful
- Unqualified Advice

### Context Distillation
- Prepends safety instructions
- Then distills into model weights
- Removes need for system prompt overhead

---

## Practical Usage

```python
from transformers import AutoTokenizer, AutoModelForCausalLM

# Load model
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b-chat-hf",
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf")

# Chat format
messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "What is the capital of France?"}
]

inputs = tokenizer.apply_chat_template(messages, return_tensors="pt")
outputs = model.generate(inputs, max_new_tokens=100)
print(tokenizer.decode(outputs[0]))
```

---

## Impact

**Ecosystem spawned:**
- Code Llama (code generation)
- Llama-2-GGUF (quantized versions)
- Mistral (inspired by LLaMA 2)
- Hundreds of fine-tunes

**Status:** 🔥 **CRITICAL** - Most deployed open-source LLM

---

**Published:** July 2023
**Downloads:** 30M+ on Hugging Face
**Current Relevance:** 🔥🔥🔥🔥🔥 Still the standard for open LLMs
