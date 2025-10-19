# QLoRA: Efficient Finetuning of Quantized LLMs

**Authors:** Tim Dettmers, Artidoro Pagnoni, et al. (University of Washington)
**Published:** May 2023 (NeurIPS 2023)
**Paper:** [arxiv.org/abs/2305.14314](https://arxiv.org/abs/2305.14314)

---

## Why This Matters

QLoRA made **fine-tuning 65B+ models possible on a single GPU**:

- 🚀 **Fine-tune 65B on 48GB GPU** (vs 780GB needed before)
- 💰 **16× memory reduction** - Accessible to everyone
- 🎯 **No quality loss** - Matches full 16-bit fine-tuning
- 🔥 **Democratized LLMs** - Anyone can customize large models

**Real-world impact:**
- Guanaco models (competitive with ChatGPT)
- Standard for LLaMA fine-tuning
- Enabled open-source AI boom

**Current Status:** 🔥 **CRITICAL** - Standard for efficient fine-tuning

---

## The Innovation

**QLoRA = Quantization + LoRA**

**4-bit quantization:**
- Model weights in 4-bit (vs 16-bit)
- 75% memory reduction

**+ LoRA:**
- Train small adapters only  
- Already efficient

**= QLoRA:**
- 4-bit base model + 16-bit LoRA adapters
- Best of both worlds!

---

## Key Techniques

### 1. 4-bit NormalFloat (NF4)
- Optimized for normally distributed weights
- Better than standard 4-bit

### 2. Double Quantization  
- Quantize the quantization constants
- Saves ~0.4 bits per parameter

### 3. Paged Optimizers
- Use CPU RAM for optimizer states
- Prevents OOM errors

---

## Results

**Guanaco 65B** (QLoRA-finetuned):
- 99.3% of ChatGPT performance
- Trained on single 48GB GPU
- 24 hours training time

**Memory requirements:**
```
Full fine-tuning 65B: 780GB
LoRA fine-tuning 65B: 360GB  
QLoRA fine-tuning 65B: 48GB  ✅
```

---

## Usage

```python
from transformers import AutoModelForCausalLM, BitsAndBytesConfig
from peft import prepare_model_for_kbit_training, LoraConfig

# Load in 4-bit
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)

model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-70b-hf",
    quantization_config=bnb_config,
)

# Prepare for QLoRA
model = prepare_model_for_kbit_training(model)

# Add LoRA adapters
lora_config = LoraConfig(r=64, lora_alpha=16)
model = get_peft_model(model, lora_config)

# Train as normal!
```

---

## Impact

**Enabled:**
- Fine-tuning on consumer GPUs
- Rapid experimentation
- Personalized models
- Research accessibility

**Popular models:**
- Guanaco series
- QLoRA-finetuned LLaMAs
- Mistral QLoRA variants

---

## Key Takeaways

1. **16× memory reduction** - 65B fits on 48GB
2. **No quality loss** - Matches full precision
3. **Combines quantization + LoRA** - Best of both
4. **Democratized fine-tuning** - Anyone can do it
5. **Now standard practice** - Default for open models

**Status:** Critical infrastructure for accessible AI

---

**Published:** May 2023
**Downloads:** Used by thousands daily
**Current Relevance:** 🔥🔥🔥🔥🔥 Essential technique
