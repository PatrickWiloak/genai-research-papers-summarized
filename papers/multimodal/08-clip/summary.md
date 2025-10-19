# Learning Transferable Visual Models From Natural Language Supervision (CLIP)

**Authors:** Alec Radford, Jong Wook Kim, Chris Hallacy, et al. (OpenAI)

**Published:** February 2021 (ICML 2021)

**Paper Link:** https://arxiv.org/abs/2103.00020

---

## Why This Paper Matters

CLIP (Contrastive Language-Image Pre-training) revolutionized how AI connects vision and language. Instead of training on labeled datasets (cat, dog, car), CLIP learns from **400 million text-image pairs** scraped from the internet. This enables **zero-shot transfer**: CLIP can recognize objects it was never explicitly trained to classify. CLIP became essential infrastructure for Stable Diffusion, DALL-E 2, and countless multimodal applications.

---

## The Core Innovation: Learning from Image-Text Pairs

### Traditional Computer Vision

**Old Approach:**
```
1. Collect labeled images: "cat", "dog", "car"
2. Train classifier on fixed set of classes
3. Model only knows those specific classes
4. Can't generalize to new categories
```

**Limitation:**
- ImageNet has 1,000 classes
- Real world has millions of concepts
- Expensive to label everything

### CLIP's Approach

**New Paradigm:**
```
1. Collect image-text pairs from internet:
   - Image of cat + Text: "a fluffy orange cat"
   - Image of sunset + Text: "beautiful sunset over mountains"
2. Learn to match images with their descriptions
3. Can recognize anything described in text!
```

**Breakthrough:**
- No manual labeling needed
- Works on any concept expressible in language
- Learns rich, nuanced representations

---

## How CLIP Works

### Architecture Overview

```
┌──────────────────────────────────────────────────┐
│              Image Encoder                       │
│  (Vision Transformer or ResNet)                  │
│                                                  │
│  Image → [Feature Vector 512-D]                 │
└──────────────────────────────────────────────────┘
                    ↓
              Image Embedding

┌──────────────────────────────────────────────────┐
│              Text Encoder                        │
│  (Transformer)                                   │
│                                                  │
│  Text → [Feature Vector 512-D]                  │
└──────────────────────────────────────────────────┘
                    ↓
              Text Embedding

            [Contrastive Learning]
            Match corresponding pairs!
```

### Contrastive Learning Objective

**Training Process:**

```
Batch of N image-text pairs:
(I₁, T₁), (I₂, T₂), ..., (Iₙ, Tₙ)

1. Encode all images → image embeddings
2. Encode all texts → text embeddings
3. Compute similarity matrix (N×N):
   - High similarity for matching pairs (I₁, T₁)
   - Low similarity for non-matching (I₁, T₂)
4. Update both encoders
```

**Similarity Matrix Example (N=4):**
```
         T₁    T₂    T₃    T₄
    ┌─────────────────────────┐
I₁  │  1.0   0.1   0.2   0.1 │ ← I₁ should match T₁
I₂  │  0.1   1.0   0.1   0.2 │ ← I₂ should match T₂
I₃  │  0.2   0.1   1.0   0.1 │ ← I₃ should match T₃
I₄  │  0.1   0.2   0.1   1.0 │ ← I₄ should match T₄
    └─────────────────────────┘

Goal: Maximize diagonal, minimize off-diagonal
```

### Loss Function

**InfoNCE (Contrastive) Loss:**
```
For each image I_i with matching text T_i:

L_image = -log(exp(sim(I_i, T_i) / τ) / Σ_j exp(sim(I_i, T_j) / τ))

Where:
- sim(I, T) = cosine similarity between embeddings
- τ = temperature parameter (learnable, typically 0.07)
- Σ_j = sum over all texts in batch
```

**Symmetric loss:**
- Image-to-text loss (above)
- Text-to-image loss (same but swapped)
- Total loss = average of both

**Intuition:** Each image should have high similarity with its text and low similarity with other texts.

---

## Model Architecture Details

### Image Encoder Options

**1. Vision Transformer (ViT):**
```
Image (224×224) → Patches (16×16)
→ Linear Embedding
→ Transformer Encoder (12-24 layers)
→ Global Average Pooling
→ Projection to 512-D
```

**Variants:**
- ViT-B/32: Base model, 32×32 patches
- ViT-B/16: Base model, 16×16 patches (higher res)
- ViT-L/14: Large model, 14×14 patches

**2. ResNet:**
```
Image (224×224)
→ Convolutional layers
→ Residual blocks
→ Attention pooling (instead of avg pool)
→ Projection to 512-D
```

**Variants:**
- ResNet-50
- ResNet-101

### Text Encoder

**Transformer:**
```
Text (max 77 tokens)
→ Token Embedding
→ Positional Embedding
→ Transformer (12 layers)
→ Take [EOS] token embedding
→ Projection to 512-D
```

**Architecture:**
- Similar to GPT-2
- Causal self-attention
- 12 layers, 512 hidden dim, 8 attention heads

---

## Training Details

### Dataset: WIT (WebImageText)

**Collection Process:**
1. Search internet for image-text pairs
2. Filter for quality and diversity
3. Result: **400 million pairs**

**Sources:**
- Websites, blogs, social media
- Image captions, alt-text
- Diverse topics and languages (primarily English)

**No manual labeling!** All data naturally occurring on web.

### Training Configuration

**Largest Model (ViT-L/14):**
- Batch size: 32,768 (very large!)
- Training steps: 32 epochs
- Compute: 256 V100 GPUs × 12 days
- Total: ~3,000 GPU-days
- Cost: ~$600K in compute

**Key Hyperparameters:**
- Learning rate: 5e-4 with cosine decay
- Weight decay: 0.2
- AdamW optimizer
- Mixed precision training

---

## Zero-Shot Classification

### How It Works

**Traditional classifier:**
```
Image → Model → [Probabilities for cat, dog, bird, ...]
```

**CLIP zero-shot:**
```
1. Create text prompts for each class:
   - "a photo of a cat"
   - "a photo of a dog"
   - "a photo of a bird"

2. Encode all text prompts → text embeddings

3. Encode image → image embedding

4. Compare image to all text embeddings

5. Highest similarity = prediction
```

**Example:**
```
Image: [picture of a cat]

Similarities:
- "a photo of a cat": 0.85 ← Highest!
- "a photo of a dog": 0.23
- "a photo of a bird": 0.19

Prediction: Cat ✓
```

### Prompt Engineering Matters

**Different prompts give different results:**

**Simple:**
```
"cat", "dog", "bird"
```

**Better:**
```
"a photo of a cat", "a photo of a dog", "a photo of a bird"
```

**Best (with ensembling):**
```
"a photo of a cat"
"a picture of a cat"
"an image of a cat"
"a rendering of a cat"
...

Average similarity across all templates
```

---

## Key Results

### Zero-Shot ImageNet Classification

**CLIP Performance:**
- **ViT-L/14:** 76.2% top-1 accuracy
- No ImageNet training data used!

**Comparison:**
- ResNet-50 (trained on ImageNet): 76.3%
- CLIP matches supervised baseline zero-shot!

**Revolutionary:** Zero-shot performance = supervised training!

### Robustness to Distribution Shift

**ImageNet variants (different conditions):**

| Dataset | Supervised ResNet | CLIP ViT-L/14 |
|---------|-------------------|---------------|
| ImageNet | 76.3% | 76.2% |
| ImageNetV2 | 60.8% | 70.1% |
| ImageNet-Sketch | 34.5% | 60.2% |
| ImageNet-R | 36.5% | 73.5% |

**CLIP is much more robust!** Better generalization to new conditions.

### Transfer to Other Datasets (Zero-Shot)

**Object recognition:**
- Food101: 83.9%
- CIFAR-10: 90.3%
- CIFAR-100: 65.1%
- STL-10: 97.2%

**Scene recognition:**
- SUN397: 62.8%
- Scenes: 86.8%

**Action recognition:**
- UCF101: 66.6%
- Kinetics700: 56.0%

**Impressive across diverse domains without fine-tuning!**

---

## Real-World Applications

### 1. Text-to-Image Generation

**DALL-E 2, Stable Diffusion:**
- Use CLIP to guide image generation
- CLIP ensures generated images match text
- Text encoder from CLIP provides conditioning

**Mechanism:**
```
Text → CLIP Text Encoder → Guide diffusion model
```

### 2. Image Search

**Search images using natural language:**
```
Query: "a sunset over mountains"
→ Encode query with CLIP
→ Compare to encoded image database
→ Return most similar images
```

**No need for manual tags!**

### 3. Content Moderation

**Detect inappropriate content:**
```
Image + Prompts:
- "safe for work content"
- "violent content"
- "explicit content"
→ Classify based on similarity
```

### 4. Visual Question Answering

**Answer questions about images:**
```
Image + Question: "What color is the car?"
Possible answers: "red", "blue", "green"
→ Encode image with question
→ Compare to answer embeddings
→ Select best match
```

### 5. Image Captioning

**Generate descriptions:**
- Use CLIP to rank candidate captions
- Ensemble with language models

### 6. Object Detection

**Open-vocabulary detection:**
- Detect any object described in text
- Not limited to pre-trained classes

---

## Prompt Engineering for CLIP

### Template Design

**Domain-specific templates:**

**For natural photos:**
```
"a photo of a {object}"
```

**For satellite imagery:**
```
"a satellite photo of a {landmark}"
```

**For art:**
```
"a painting of a {subject}"
```

**For OCR/text:**
```
"a text saying '{text}'"
```

### Ensemble Prompts

**Multiple templates, average results:**
```
templates = [
    "a photo of a {}",
    "a cropped photo of a {}",
    "a close-up photo of a {}",
    "a photo of one {}",
    ...
]

For each class:
    embeddings = [encode(t.format(class)) for t in templates]
    final_embedding = mean(embeddings)
```

**Improves robustness and accuracy!**

---

## Limitations and Challenges

### 1. **Fine-Grained Classification**

**Struggles with similar categories:**
- Dog breeds (Husky vs. Malamute)
- Car models (Honda Civic vs. Accord)
- Flower species

**Reason:** Training data has coarse descriptions, not fine details.

### 2. **Abstract or Complex Tasks**

**Challenges:**
- Counting objects
- Spatial relationships ("left of", "above")
- Abstract concepts
- Complex reasoning

### 3. **Data Quality and Bias**

**Web data issues:**
- Noisy captions (don't always match image)
- Geographic bias (Western-centric)
- Demographic bias (gender, race, age)
- NSFW content filtering imperfect

### 4. **Computational Cost**

**Requires significant resources:**
- Training: Thousands of GPU-days
- Inference: Large models slower
- Not accessible for small organizations

### 5. **Text Length Limitation**

**77 token maximum:**
- Limits complex descriptions
- Can't process long documents
- Needs concise prompts

---

## CLIP Variants and Successors

### OpenCLIP (Open-Source)

**Community recreation:**
- Multiple model sizes
- Different training datasets (LAION)
- Open weights and training code
- Used in Stable Diffusion 2.0+

### ALIGN (Google, 2021)

**Similar to CLIP:**
- 1.8 billion image-text pairs (4× more)
- Simpler data filtering
- Better performance on some tasks

### Florence (Microsoft, 2021)

**Larger scale:**
- 900 million image-text pairs
- Multiple pretraining tasks
- Improved transfer learning

### BLIP / BLIP-2 (Salesforce, 2022-2023)

**Better understanding:**
- Bidirectional training
- Captioning and retrieval
- More efficient architectures

### EVA-CLIP (2023)

**State-of-the-art:**
- Larger models (5 billion params)
- Better training techniques
- Higher accuracy

---

## Impact on AI Research

### Paradigm Shifts:

**1. Supervision from Language:**
- Language as universal interface
- No manual annotation needed
- Scalable to any concept

**2. Zero-Shot Learning:**
- Models work out-of-the-box
- Generalize to unseen categories
- More flexible deployment

**3. Multimodal Representations:**
- Unified vision-language space
- Enables cross-modal tasks
- Foundation for generative models

### Research Directions Enabled:

- Open-vocabulary object detection
- Visual prompt learning
- Multimodal large language models
- Compositional reasoning
- Vision-language pretraining methods

---

## Practical Usage Tips

### 1. **Choosing Model Size**

**Trade-offs:**
- ViT-B/32: Fast, decent accuracy, small
- ViT-B/16: Slower, better accuracy
- ViT-L/14: Best accuracy, slowest, largest

**Recommendation:** Start with ViT-B/32 for prototyping.

### 2. **Prompt Engineering**

**Best practices:**
- Use domain-specific templates
- Ensemble multiple prompts
- Include context ("a photo of", "a painting of")
- Experiment with phrasing

### 3. **Fine-Tuning**

**Options:**
- Zero-shot: Use as-is
- Linear probe: Train classifier on top
- Full fine-tuning: Update all weights
- Prompt tuning: Learn prompt embeddings

**Usually:** Zero-shot or linear probe sufficient!

### 4. **Preprocessing**

**Image preprocessing:**
- Resize to 224×224
- Normalize with CLIP stats
- Center crop for best results

**Text preprocessing:**
- Truncate to 77 tokens
- Lowercase not necessary (model handles)

---

## Comparison: CLIP vs. Traditional Vision Models

| Aspect | CLIP | ResNet/ImageNet |
|--------|------|-----------------|
| **Training Data** | 400M image-text pairs | 1.2M labeled images |
| **Classes** | Open-vocabulary | 1,000 fixed classes |
| **Zero-Shot** | Yes (76% ImageNet) | No |
| **Robustness** | High | Medium |
| **Fine-Tuning** | Optional | Required |
| **Multimodal** | Yes | No |

---

## Key Concepts

### 1. **Contrastive Learning**
Learning by contrasting positive pairs against negative pairs.

### 2. **Zero-Shot Transfer**
Performing tasks without task-specific training data.

### 3. **Multimodal Embedding**
Shared space where images and text are comparable.

### 4. **Natural Language Supervision**
Using text as supervision signal instead of labels.

### 5. **Distribution Robustness**
Generalizing beyond training distribution.

---

## Ethical Considerations

### Bias Concerns:

**Training data biases:**
- Geographic (Western-centric)
- Demographic (representation)
- Cultural (contexts and norms)

**Impact:**
- Perpetuates stereotypes
- Unequal performance across groups
- Fairness issues in applications

### Dual Use:

**Positive uses:**
- Accessibility tools
- Content moderation
- Research and education

**Potential harms:**
- Surveillance
- Profiling
- Misinformation detection (and creation)

**Paper includes bias analysis and discussion.**

---

## Key Takeaways

1. **Natural language enables zero-shot learning** for vision tasks
2. **Contrastive learning** is powerful for multimodal representations
3. **Web-scale data** (400M pairs) enables strong generalization
4. **Zero-shot CLIP rivals supervised models** on ImageNet
5. **Robustness improves** with language-supervised training
6. **Foundation for text-to-image** generation (DALL-E 2, Stable Diffusion)
7. **Prompt engineering** significantly affects performance

---

## Further Reading

- **Original Paper:** https://arxiv.org/abs/2103.00020
- **OpenAI Blog:** https://openai.com/blog/clip
- **OpenCLIP:** https://github.com/mlfoundations/open_clip
- **LAION Dataset:** https://laion.ai/
- **CLIP Interrogator:** https://github.com/pharmapsychotic/clip-interrogator

---

## Citation

```bibtex
@inproceedings{radford2021learning,
  title={Learning transferable visual models from natural language supervision},
  author={Radford, Alec and Kim, Jong Wook and Hallacy, Chris and Ramesh, Aditya and Goh, Gabriel and Agarwal, Sandhini and Sastry, Girish and Askell, Amanda and Mishkin, Pamela and Clark, Jack and others},
  booktitle={International conference on machine learning},
  pages={8748--8763},
  year={2021},
  organization={PMLR}
}
```
