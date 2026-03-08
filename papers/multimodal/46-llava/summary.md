# LLaVA: Visual Instruction Tuning

**Authors:** Haotian Liu, Chunyuan Li, Qingyang Wu, Yong Jae Lee
**Published:** April 2023 (NeurIPS 2023 Oral)
**Paper:** [arxiv.org/abs/2304.08485](https://arxiv.org/abs/2304.08485)

---

## Why This Matters

LLaVA **created the blueprint for open-source multimodal models**:

- 🔓 **Open-source multimodal** - First practical open vision-language model
- 🧩 **Simple architecture** - Just connect a vision encoder to an LLM with a projection layer
- 📊 **GPT-4 generated data** - Used language-only GPT-4 to create visual instruction data
- 🎯 **85.1% of GPT-4V quality** - Remarkably close with much less compute
- 🌊 **Spawned an ecosystem** - LLaVA-1.5, LLaVA-NeXT, and dozens of derivatives

**Real-world impact:**
- Democratized multimodal AI (anyone could build a vision-language model)
- Established the "vision encoder + projection + LLM" architecture pattern
- Inspired virtually every open-source multimodal model
- Showed that instruction tuning transfers to vision tasks

**The insight:** **You don't need to train a multimodal model from scratch.** Connect a frozen CLIP vision encoder to a frozen LLM with a trainable projection layer, then instruction-tune - and you get a surprisingly capable visual assistant.

---

## The Architecture

### Elegant Simplicity

```
LLaVA architecture (3 components):

Image → [CLIP ViT-L/14] → Visual features
                              ↓
                     [Projection Matrix W]  ← Only this is trained initially!
                              ↓
                        Visual tokens
                              ↓
Text  → [Tokenizer] → Text tokens + Visual tokens → [Vicuna LLM] → Response
```

**That's it.** A vision encoder, a projection layer, and an LLM.

### Component Details

| Component | Model | Parameters | Trainable? |
|-----------|-------|-----------|------------|
| Vision encoder | CLIP ViT-L/14 | 304M | Frozen (pre-trained) |
| Projection | Linear layer | ~5M | **Yes** |
| Language model | Vicuna 13B | 13B | Fine-tuned |

### Why This Works

```
CLIP already understands images (trained on 400M image-text pairs)
Vicuna already understands language (fine-tuned LLaMA)

The projection layer just needs to learn:
"Translate CLIP's visual representation into Vicuna's token space"

This is a much simpler problem than training from scratch!
```

---

## Training

### Two-Stage Process

**Stage 1: Feature Alignment (Pre-training)**
```
Goal: Teach the projection to map visual features to language space
Data: 595K image-caption pairs (CC3M filtered)
What trains: Only the projection matrix W
Duration: ~4 hours on 8 A100 GPUs

Input:  [Image of a dog in a park]
Target: "A golden retriever playing fetch in a sunny park"

The LLM and vision encoder stay frozen.
Only W learns to connect them.
```

**Stage 2: Visual Instruction Tuning (Fine-tuning)**
```
Goal: Teach the model to follow visual instructions
Data: 158K visual instruction-following examples
What trains: Projection W + LLM (end-to-end)
Duration: ~10 hours on 8 A100 GPUs

Input:  [Image] + "Describe what's happening in detail"
Target: Detailed description following the instruction
```

### The Clever Data Generation

**Problem:** No visual instruction-following datasets existed.

**LLaVA's solution: Use GPT-4 (text-only) to generate them!**

```
1. Take an image from COCO dataset
2. Feed GPT-4 the image's captions and bounding box descriptions
   (text representations of what's in the image)
3. Ask GPT-4 to generate:
   - Conversations about the image
   - Detailed descriptions
   - Complex reasoning questions

GPT-4 never "sees" the image - it works from text descriptions
But it generates high-quality instruction-following data!
```

**Three types of generated data:**

| Type | Count | Example |
|------|-------|---------|
| Conversation | 58K | Multi-turn Q&A about an image |
| Detailed description | 23K | "Describe this image in detail" |
| Complex reasoning | 77K | "What might happen next?" |

---

## Performance

### vs GPT-4V

**Relative score on visual instruction following:**

| Model | Relative to GPT-4V |
|-------|-------------------|
| BLIP-2 | 46.4% |
| InstructBLIP | 60.6% |
| **LLaVA** | **85.1%** |
| GPT-4V | 100% (reference) |

**85% of GPT-4V quality with a 13B model!**

### Science QA

| Model | Accuracy |
|-------|----------|
| GPT-3.5 (CoT) | 75.2% |
| LLaVA | 90.9% |
| **LLaVA + GPT-4 (judge)** | **92.5%** |

### Qualitative Examples

```
Image: [Photo of a messy desk with coffee, papers, laptop]

User: "What can you tell about the person who works at this desk?"

LLaVA: "Based on the image, the person appears to be a busy
professional or student. There's a laptop open suggesting they
do computer-based work. The coffee cup indicates they might
work long hours. The scattered papers suggest they're in the
middle of a project. The overall messiness indicates they
prioritize productivity over organization..."
```

---

## LLaVA-1.5 and Beyond

### LLaVA-1.5 (October 2023)

**Simple improvements, big gains:**

| Change | Why |
|--------|-----|
| MLP projection (2-layer) | Better feature mapping than linear |
| Higher resolution (336px) | More visual detail |
| More training data | 665K total |
| Vicuna-v1.5 | Better base LLM |

**Result:** State-of-the-art on 11 of 12 benchmarks, matching or beating models 10x its training cost.

### The LLaVA Family

```
LLaVA (Apr 2023) → Visual instruction tuning concept
LLaVA-1.5 (Oct 2023) → Simple but effective improvements
LLaVA-NeXT (Jan 2024) → Dynamic high resolution
LLaVA-OneVision (2024) → Unified image/video understanding
LLaVA-Video (2024) → Video understanding capabilities
```

---

## Why LLaVA Changed the Field

### The Open-Source Multimodal Blueprint

**Before LLaVA:**
```
Building a multimodal model required:
- Massive compute budget
- Custom architecture design
- Millions of image-text pairs
- Months of training
- Only big labs could do it
```

**After LLaVA:**
```
Building a multimodal model requires:
- A pre-trained vision encoder (CLIP - free)
- A pre-trained LLM (LLaMA - free)
- A projection layer (tiny, train in hours)
- Visual instruction data (generate with GPT-4)
- One day of fine-tuning on 8 GPUs
- Anyone with a few GPUs can do it!
```

### Models Inspired by LLaVA

```
Direct derivatives:
- LLaVA-1.5, LLaVA-NeXT, LLaVA-OneVision

Same architecture pattern:
- InternVL (Shanghai AI Lab)
- Qwen-VL (Alibaba)
- CogVLM (Tsinghua)
- MiniGPT-4 (KAUST)
- ShareGPT4V
- Bunny, TinyLLaVA, and many more

The "CLIP + Projection + LLM" pattern became THE standard
for open-source multimodal models.
```

---

## Practical Usage

### Running LLaVA

```python
from llava.model.builder import load_pretrained_model
from llava.conversation import conv_templates
from llava.utils import process_images
from PIL import Image

# Load model
tokenizer, model, image_processor, context_len = load_pretrained_model(
    model_path="liuhaotian/llava-v1.5-13b",
    model_base=None,
    model_name="llava-v1.5-13b"
)

# Process image
image = Image.open("photo.jpg")
image_tensor = process_images([image], image_processor, model.config)

# Generate response
prompt = "Describe this image in detail."
conv = conv_templates["v1"].copy()
conv.append_message(conv.roles[0], f"<image>\n{prompt}")
conv.append_message(conv.roles[1], None)

input_ids = tokenizer(conv.get_prompt(), return_tensors="pt")
output = model.generate(input_ids, images=image_tensor, max_new_tokens=500)
print(tokenizer.decode(output[0], skip_special_tokens=True))
```

### Using with Hugging Face

```python
from transformers import AutoProcessor, LlavaForConditionalGeneration
from PIL import Image

model = LlavaForConditionalGeneration.from_pretrained(
    "llava-hf/llava-1.5-13b-hf",
    torch_dtype="auto",
    device_map="auto"
)
processor = AutoProcessor.from_pretrained("llava-hf/llava-1.5-13b-hf")

image = Image.open("photo.jpg")
prompt = "USER: <image>\nWhat's in this image?\nASSISTANT:"

inputs = processor(text=prompt, images=image, return_tensors="pt")
output = model.generate(**inputs, max_new_tokens=200)
print(processor.decode(output[0], skip_special_tokens=True))
```

---

## Limitations

### 1. Resolution Constraints
```
Original LLaVA: 224x224 (very low)
LLaVA-1.5: 336x336 (better)
Still loses fine details in high-res images
LLaVA-NeXT addressed this with dynamic resolution
```

### 2. Hallucination
```
Can describe objects that aren't in the image
Especially with leading questions
"Is there a cat in this image?" → May say yes even if no cat
```

### 3. No Video (Original)
```
Image-only in original LLaVA
LLaVA-Video added video later
Frame-by-frame, not true temporal understanding
```

### 4. Base LLM Limitations
```
Reasoning quality limited by base LLM (Vicuna 13B)
Smaller than proprietary alternatives
Struggles with complex multi-step visual reasoning
```

---

## Key Takeaways

1. **Simple architecture works** - Vision encoder + projection + LLM is enough
2. **Data generation innovation** - GPT-4 can create visual instruction data from text descriptions
3. **Efficient training** - Hours, not months; 8 GPUs, not thousands
4. **85% of GPT-4V** - Remarkably close to proprietary models
5. **Created the template** - Every open-source multimodal model follows this pattern

**Bottom line:** LLaVA proved that multimodal AI doesn't require massive resources. Its simple, elegant architecture became the standard blueprint for open-source vision-language models, democratizing multimodal AI research.

---

## Further Reading

### Original Papers
- **LLaVA:** https://arxiv.org/abs/2304.08485
- **LLaVA-1.5:** https://arxiv.org/abs/2310.03744
- **LLaVA-NeXT:** https://llava-vl.github.io/blog/2024-01-30-llava-next/

### Code and Models
- **GitHub:** https://github.com/haotian-liu/LLaVA
- **Models:** https://huggingface.co/liuhaotian

### Related Work
- **CLIP (vision encoder):** https://arxiv.org/abs/2103.00020
- **MiniGPT-4:** https://arxiv.org/abs/2304.10592
- **InstructBLIP:** https://arxiv.org/abs/2305.06500

---

**Published:** April 2023 (NeurIPS 2023 Oral)
**Impact:** 🔥🔥🔥🔥 **HIGH** - Blueprint for open-source multimodal models
**Citations:** 5,000+ (as of early 2026)
**Adoption:** Massive - spawned dozens of derivatives
**Current Relevance:** Architecture pattern still dominant in open-source multimodal
**Legacy:** Democratized multimodal AI, made it accessible to everyone

**Modern Status (March 2026):** The LLaVA architecture pattern (vision encoder + projection + LLM) remains the dominant approach for open-source multimodal models. LLaVA-OneVision and LLaVA-Video extend the concept to unified vision understanding. The original paper has over 5,000 citations.
