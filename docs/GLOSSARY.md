# GenAI Glossary - Key Terms Explained

Comprehensive glossary of technical terms used across the 15 foundational papers.

---

## Table of Contents
- [A](#a) | [B](#b) | [C](#c) | [D](#d) | [E](#e) | [F](#f) | [G](#g) | [H](#h) | [I](#i) | [K](#k) | [L](#l) | [M](#m) | [N](#n) | [O](#o) | [P](#p) | [Q](#q) | [R](#r) | [S](#s) | [T](#t) | [U](#u) | [V](#v) | [W](#w) | [Z](#z)

---

## A

### Activation Function
Mathematical function applied to neuron outputs to introduce non-linearity.
- **Examples:** ReLU, GELU, SwiGLU, Sigmoid, Tanh
- **Why needed:** Without non-linearity, neural networks would just be linear regression
- **Paper:** Transformers (GELU), LLaMA (SwiGLU)

### Adapter Layers
Small neural network modules inserted between layers for efficient fine-tuning.
- **Related:** LoRA (low-rank variant)
- **Advantage:** Only train adapter weights, not full model
- **Paper:** LoRA

### Adversarial Training
Training where two models compete (generator vs discriminator).
- **Key idea:** Competition drives both to improve
- **Paper:** GANs
- **Challenge:** Training instability (mode collapse, oscillation)

### Alignment
Making AI systems behave according to human values and intentions.
- **Methods:** RLHF, Constitutional AI, reward modeling
- **Papers:** InstructGPT, Constitutional AI
- **Goal:** Helpful, harmless, honest systems

### Attention Mechanism
Allows models to focus on relevant parts of input when processing.
- **Types:** Self-attention, cross-attention, multi-head attention
- **Formula:** Attention(Q, K, V) = softmax(QK^T / √d_k)V
- **Papers:** Transformers (introduced), BERT, GPT-3, ViT, CLIP

### Attention Heads
Parallel attention mechanisms that learn different relationships.
- **Multi-head attention:** 8-96 heads typical
- **Purpose:** Each head can specialize (syntax, semantics, etc.)
- **Paper:** Transformers

### Autoencoder
Neural network that learns to compress and reconstruct data.
- **Parts:** Encoder (compress), latent space, decoder (reconstruct)
- **Uses:** Dimensionality reduction, denoising
- **Paper:** Stable Diffusion (VAE variant)

### Autoregressive Model
Model that predicts next element based on previous elements.
- **Example:** GPT predicting next word given previous words
- **Formula:** P(x) = P(x₁) × P(x₂|x₁) × P(x₃|x₁,x₂) × ...
- **Papers:** GPT-3, LLaMA
- **Contrast:** Non-autoregressive (parallel generation)

---

## B

### Backpropagation
Algorithm for computing gradients to update neural network weights.
- **Key idea:** Chain rule applied backwards through network
- **Used in:** Every paper (fundamental training algorithm)

### Batch Normalization
Normalizing layer inputs to improve training stability.
- **Alternative:** Layer normalization (used in Transformers)
- **Benefits:** Faster training, higher learning rates possible

### Batch Size
Number of examples processed together before updating weights.
- **Small batch:** More noise, can escape local minima
- **Large batch:** More stable, better GPU utilization
- **Typical:** 32-2048 for language models, higher for vision

### BERT (Bidirectional Encoder Representations from Transformers)
Transformer encoder trained with masked language modeling.
- **Key innovation:** Bidirectional context (unlike GPT's left-to-right)
- **Paper:** #3 BERT
- **Uses:** Classification, NER, semantic search

### Bias (Model)
Systematic errors in model predictions.
- **Types:** Social bias, sampling bias, inductive bias
- **Papers:** All models have biases from training data
- **Mitigation:** Diverse data, debiasing techniques, auditing

### BPE (Byte-Pair Encoding)
Tokenization algorithm that builds vocabulary from frequent subwords.
- **Example:** "unhappiness" → "un" + "happiness"
- **Benefits:** Handles rare words, multilingual, compact
- **Papers:** GPT-3, BERT, LLaMA (all use variants)

---

## C

### Causal Attention
Attention that only looks at previous tokens (not future).
- **Also called:** Masked attention
- **Used in:** GPT-3, LLaMA (decoder-only models)
- **Purpose:** Prevent "cheating" during autoregressive generation
- **Implementation:** Mask out future positions with -∞

### Chain-of-Thought (CoT)
Prompting technique to elicit step-by-step reasoning.
- **Example:** "Let's think step by step..."
- **Paper:** #9 Chain-of-Thought
- **Benefit:** Improves complex reasoning tasks

### Checkpoint
Saved model state during training.
- **Contents:** Model weights, optimizer state, training step
- **Purpose:** Resume training, rollback if needed, evaluation
- **Frequency:** Every N steps or epochs

### Chinchilla Scaling
Optimal allocation of compute between model size and training tokens.
- **Key finding:** Most models undertrained, should use more tokens
- **Paper:** Scaling Laws, applied in LLaMA
- **Formula:** N_opt ∝ C^0.5, D_opt ∝ C^0.5

### CLIP (Contrastive Language-Image Pre-training)
Model that learns vision-language alignment through contrastive learning.
- **Paper:** #8 CLIP
- **Training:** Match image-text pairs, separate mismatches
- **Uses:** Zero-shot classification, text-to-image guidance

### Conditioning
Providing additional input to guide model output.
- **Examples:** Text for image generation, class label for generation
- **Papers:** Stable Diffusion (text conditioning), CLIP
- **Methods:** Concatenation, cross-attention, adaptive normalization

### Constitutional AI
Alignment method using AI self-critique guided by principles.
- **Paper:** #14 Constitutional AI
- **Key idea:** Explicit principles > implicit human preferences
- **Stages:** SL-CAI (supervised), RL-CAI (reinforcement learning)

### Context Length / Context Window
Maximum number of tokens the model can process at once.
- **Examples:** GPT-3 (2048), GPT-4 (32k-128k), BERT (512)
- **Limitation:** Quadratic attention cost
- **Solutions:** Sparse attention, sliding window, retrieval

### Contrastive Learning
Learning by contrasting positive and negative examples.
- **Formula:** Pull similar pairs together, push dissimilar apart
- **Papers:** CLIP (image-text), SimCLR (images)
- **Loss:** InfoNCE, NT-Xent

### Convolution
Operation that applies a filter across spatial dimensions.
- **Used in:** CNNs (ResNet, etc.)
- **Replaced by:** Self-attention in ViT
- **Benefits:** Translation invariance, local patterns

### Cross-Attention
Attention between two different sequences (e.g., text and image).
- **Example:** Decoder attending to encoder in translation
- **Papers:** Original Transformers, Stable Diffusion
- **Contrast:** Self-attention (within same sequence)

---

## D

### Decoder
Part of model that generates output sequence.
- **Transformer decoder:** Uses causal attention
- **Papers:** Transformers, GPT-3, LLaMA
- **Components:** Self-attention, cross-attention, FFN

### Denoising
Removing noise from data, used in training.
- **Denoising Autoencoders:** Reconstruct clean from noisy
- **Diffusion Models:** Iteratively denoise from pure noise
- **Papers:** DDPM, Stable Diffusion

### Diffusion Model
Generative model that learns to reverse a noise process.
- **Forward:** Gradually add noise to data
- **Reverse:** Learn to remove noise step-by-step
- **Papers:** #6 DDPM, #7 Stable Diffusion

### Discriminator
Model that tries to distinguish real from fake data.
- **Paper:** GANs
- **Training:** Binary classification (real vs generated)
- **Purpose:** Provides signal to train generator

### Distillation
Training a smaller "student" model to mimic larger "teacher" model.
- **Benefits:** Smaller, faster model with similar performance
- **Methods:** Match outputs, intermediate layers, or both
- **Papers:** Used in many production deployments

### Dropout
Randomly dropping connections during training to prevent overfitting.
- **Typical rate:** 0.1-0.5 (10-50% dropped)
- **At inference:** All connections active (but scaled)
- **Papers:** Used in all modern models

---

## E

### Embedding
Dense vector representation of discrete objects (words, images, etc.).
- **Example:** "cat" → [0.2, -0.5, 0.1, ..., 0.3] (dim 768)
- **Properties:** Similar concepts have similar vectors
- **Papers:** All models use embeddings

### Encoder
Part of model that processes input into representations.
- **Transformer encoder:** Uses bidirectional attention
- **Papers:** BERT (encoder-only), Transformers (encoder-decoder)
- **Components:** Self-attention, FFN, layer norm

### Epoch
One complete pass through entire training dataset.
- **Example:** 10 epochs = seeing every example 10 times
- **Modern LLMs:** Often <1 epoch (so much data)

### Evaluation Metrics
Measures of model performance.
- **Language:** Perplexity, BLEU, ROUGE, accuracy
- **Vision:** FID, Inception Score, accuracy
- **Alignment:** Human preferences, safety benchmarks

---

## F

### Feed-Forward Network (FFN)
Simple neural network applied after attention.
- **Architecture:** Linear → Activation → Linear
- **Size:** Usually 4× hidden dimension
- **Papers:** All Transformer-based models
- **Purpose:** Non-linear transformation of representations

### Few-Shot Learning
Learning from just a few examples.
- **0-shot:** No examples, just instruction
- **1-shot:** One example
- **5-shot:** Five examples
- **Paper:** GPT-3 (emergent few-shot ability)

### Fine-Tuning
Further training a pre-trained model on specific task.
- **Full fine-tuning:** Update all weights
- **Parameter-efficient:** Update subset (LoRA, adapters)
- **Papers:** BERT (pre-train then fine-tune), LoRA (efficient)

### FLOP (Floating Point Operation)
Basic unit of computation.
- **Training cost:** Measured in petaFLOPs (10^15)
- **Example:** GPT-3 training ~3,640 petaFLOP-days
- **Paper:** Scaling Laws (relates FLOPs to performance)

---

## G

### GAN (Generative Adversarial Network)
Generator and discriminator competing in a game.
- **Paper:** #2 GANs
- **Formula:** min_G max_D V(D,G) = E[log D(x)] + E[log(1-D(G(z)))]
- **Challenge:** Training instability

### Generator
Model that creates new data samples.
- **In GANs:** Tries to fool discriminator
- **In general:** Any generative model
- **Papers:** GANs, Diffusion, GPT-3

### Gradient
Direction and rate of steepest increase in loss.
- **Descent:** Move opposite to gradient (decrease loss)
- **Clipping:** Limit gradient magnitude for stability
- **Accumulation:** Sum gradients over multiple batches

### Greedy Decoding
Choosing most likely token at each step.
- **Simple but:** Can get stuck in suboptimal sequences
- **Alternative:** Beam search, sampling

---

## H

### Hallucination
Model generating false information confidently.
- **Cause:** No grounding in facts, just pattern matching
- **Mitigation:** RAG, fact-checking, retrieval, Constitutional AI
- **Papers:** RAG reduces hallucinations

### Hidden Dimension
Size of internal representations.
- **Examples:** BERT-base (768), GPT-3 (12,288), LLaMA-7B (4,096)
- **Larger:** More capacity, but more compute
- **Also called:** d_model, embedding dimension

### Hybrid Model
Combining multiple approaches.
- **Examples:** RLHF + Constitutional AI, CNN + Transformer
- **Papers:** Many production systems use hybrids

---

## I

### In-Context Learning
Learning from examples in the prompt without weight updates.
- **Example:** Provide 3 examples of translation, then model translates new text
- **Paper:** GPT-3 (discovered as emergent ability)
- **Mechanism:** Not fully understood

### Inference
Using trained model to make predictions.
- **Contrast:** Training (updating weights)
- **Cost:** Usually much cheaper than training
- **Optimization:** Quantization, distillation, pruning

### InstructGPT
GPT-3 fine-tuned with RLHF to follow instructions.
- **Paper:** #5 InstructGPT
- **Led to:** ChatGPT
- **Method:** SFT → Reward Model → PPO

---

## K

### KL Divergence (Kullback-Leibler)
Measure of difference between two probability distributions.
- **Used in:** RLHF (KL penalty to prevent drift)
- **Formula:** D_KL(P||Q) = Σ P(x) log(P(x)/Q(x))
- **Not symmetric:** D_KL(P||Q) ≠ D_KL(Q||P)

---

## L

### Latent Space
Compressed representation space.
- **Example:** Stable Diffusion compresses 512×512 image to 64×64 latent
- **Benefits:** Smaller, semantically meaningful
- **Papers:** Stable Diffusion (VAE latent space)

### Layer Normalization
Normalizing across features for each example.
- **Used in:** Transformers (instead of batch norm)
- **Papers:** All Transformer models
- **Benefit:** Independent of batch size

### Learning Rate
Step size for weight updates during training.
- **Too high:** Unstable, diverges
- **Too low:** Slow training, stuck in local minima
- **Typical:** 1e-4 to 3e-4 for LLMs
- **Schedule:** Warmup, then cosine decay

### LLaMA (Large Language Model Meta AI)
Efficient open-source language models.
- **Paper:** #15 LLaMA
- **Key:** Compute-optimal training (more tokens, not just params)
- **Impact:** Spawned open-source ecosystem (Alpaca, Vicuna)

### LoRA (Low-Rank Adaptation)
Efficient fine-tuning using low-rank matrices.
- **Paper:** #10 LoRA
- **Formula:** W' = W + BA (where B, A are low-rank)
- **Reduction:** 10,000× fewer trainable parameters

### Loss Function
Measures how wrong model's predictions are.
- **Examples:** Cross-entropy, MSE, contrastive loss
- **Goal:** Minimize during training
- **Papers:** Each uses task-specific loss

---

## M

### Masked Language Modeling (MLM)
Training by predicting masked tokens.
- **Example:** "The [MASK] sat on the mat" → predict "cat"
- **Paper:** BERT
- **Benefit:** Learn bidirectional context

### Multi-Head Attention
Running multiple attention mechanisms in parallel.
- **Why:** Each head can focus on different aspects
- **Typical:** 8-96 heads
- **Papers:** Transformers (introduced), all Transformer models

### Multimodal
Processing multiple types of data (text, images, audio).
- **Papers:** CLIP (text-image), Stable Diffusion (text-to-image)
- **Challenge:** Aligning different modalities

---

## N

### Neural Network
Computational model inspired by biological neurons.
- **Components:** Layers of connected neurons with weights
- **Training:** Backpropagation + gradient descent
- **Papers:** All papers use neural networks

### Normalization
Scaling values to standard range.
- **Types:** Batch norm, layer norm, RMS norm
- **Purpose:** Stabilize training, faster convergence
- **Papers:** All modern models use normalization

---

## O

### Optimizer
Algorithm for updating weights based on gradients.
- **Adam:** Adaptive learning rate (most common)
- **SGD:** Stochastic gradient descent
- **AdamW:** Adam with weight decay (common in LLMs)
- **Papers:** All use optimizers for training

### Overfitting
Model memorizes training data instead of learning patterns.
- **Signs:** Great on training, poor on validation
- **Prevention:** Dropout, regularization, more data
- **Papers:** All papers address overfitting

---

## P

### Parameter
Learnable weight in neural network.
- **Count:** GPT-3 (175B), BERT-base (110M), LLaMA-7B (7B)
- **More parameters:** More capacity, but more compute
- **Papers:** Scaling Laws relates parameters to performance

### Patch
Small region of an image treated as a token.
- **Example:** 16×16 pixel patch in ViT
- **Paper:** Vision Transformer
- **Benefit:** Makes images compatible with Transformers

### Perplexity
Measure of how surprised model is by test data.
- **Formula:** exp(cross-entropy loss)
- **Lower is better:** Less surprised = better model
- **Papers:** Common metric for language models

### Positional Encoding
Adding position information to token embeddings.
- **Why needed:** Attention has no inherent order sense
- **Types:** Sinusoidal (Transformers), learned, RoPE (LLaMA)
- **Papers:** Transformers, ViT, LLaMA

### Pre-training
Initial training on large general dataset.
- **Then:** Fine-tune on specific task
- **Examples:** BERT on Wikipedia, GPT-3 on web text
- **Papers:** BERT (introduced paradigm), GPT-3, LLaMA

### Prompt
Input text given to language model.
- **Types:** Zero-shot, few-shot, instruction
- **Engineering:** Crafting prompts for best results
- **Papers:** GPT-3 (popularized prompting)

### PPO (Proximal Policy Optimization)
Reinforcement learning algorithm.
- **Used in:** RLHF for aligning language models
- **Papers:** InstructGPT, Constitutional AI
- **Benefit:** Stable policy updates

---

## Q

### Quantization
Reducing precision of weights/activations.
- **Example:** FP32 → INT8 (32-bit to 8-bit)
- **Benefit:** 4× smaller, faster inference
- **Cost:** Slight quality loss
- **Papers:** Applied to all models for deployment

### Query, Key, Value (Q, K, V)
Three matrices in attention mechanism.
- **Query:** What am I looking for?
- **Key:** What do I contain?
- **Value:** What information do I carry?
- **Formula:** Attention(Q,K,V) = softmax(QK^T/√d)V
- **Papers:** Transformers (introduced)

---

## R

### RAG (Retrieval-Augmented Generation)
Combining retrieval with generation.
- **Paper:** #13 RAG
- **Process:** Retrieve relevant docs → generate answer using them
- **Benefit:** Grounded in facts, less hallucination

### Reinforcement Learning (RL)
Learning from rewards/penalties.
- **In AI alignment:** Learn from human preferences
- **Papers:** InstructGPT (RLHF), Constitutional AI (RLAIF)
- **Algorithm:** PPO (Proximal Policy Optimization)

### ReLU (Rectified Linear Unit)
Activation function: f(x) = max(0, x)
- **Simple:** Just zero out negatives
- **Variants:** LeakyReLU, GeLU, SwiGLU
- **Papers:** Used in many models

### Residual Connection
Shortcut that adds input to output of layer.
- **Formula:** Output = Layer(x) + x
- **Benefit:** Easier gradient flow, enables deep networks
- **Papers:** ResNet (vision), Transformers (all)

### Reward Model
Model that scores outputs for quality.
- **Training:** On human preferences
- **Used in:** RLHF to guide RL training
- **Papers:** InstructGPT, Constitutional AI

### RLHF (Reinforcement Learning from Human Feedback)
Training with human preferences as rewards.
- **Paper:** #5 InstructGPT
- **Stages:** SFT → Reward Model → PPO
- **Result:** Aligned, helpful models

### RoPE (Rotary Position Embedding)
Position encoding using rotation matrices.
- **Benefits:** Better extrapolation to longer sequences
- **Paper:** LLaMA, GPT-NeoX
- **Alternative to:** Learned or sinusoidal positions

---

## S

### Sampling
Randomly selecting next token based on probabilities.
- **Temperature:** Controls randomness (higher = more random)
- **Top-k:** Sample from k most likely tokens
- **Top-p (nucleus):** Sample from cumulative probability p
- **Papers:** Used in all generative models

### Scaling Laws
Predictable relationships between size, data, compute, and performance.
- **Paper:** #12 Scaling Laws
- **Formula:** L(N) ∝ N^(-α)
- **Impact:** Justified massive investments in scaling

### Self-Attention
Attention mechanism within a single sequence.
- **Example:** Each word attends to all other words
- **Papers:** Transformers (introduced), all Transformer models
- **Complexity:** O(n²) in sequence length

### Seq2Seq (Sequence-to-Sequence)
Model that maps input sequence to output sequence.
- **Architecture:** Encoder-decoder
- **Examples:** Translation, summarization
- **Papers:** Original Transformers

### Softmax
Converts logits to probability distribution.
- **Formula:** softmax(x_i) = exp(x_i) / Σ exp(x_j)
- **Properties:** Outputs sum to 1, all between 0-1
- **Used in:** Attention, output layers

### Stable Diffusion
Efficient diffusion in latent space.
- **Paper:** #7 Stable Diffusion
- **Key:** Run diffusion on compressed latent representation
- **Speed:** 10-100× faster than pixel-space diffusion

### Supervised Learning
Learning from labeled examples.
- **Example:** Image + label, text + answer
- **Contrast:** Unsupervised (no labels), self-supervised
- **Papers:** Most fine-tuning is supervised

### SwiGLU
Activation function used in LLaMA.
- **Formula:** SwiGLU(x) = Swish(xW) ⊙ xV
- **Paper:** LLaMA (from PaLM)
- **Benefit:** Better than ReLU for language models

---

## T

### Temperature
Parameter controlling randomness in sampling.
- **Low (0.1):** More deterministic, focused
- **High (1.0+):** More random, creative
- **Zero:** Greedy decoding (always most likely)
- **Papers:** All generative models

### Token
Basic unit of text for models.
- **Examples:** Word, subword, character
- **Typical:** Subwords via BPE
- **Count:** GPT-3 vocab = 50k tokens
- **Papers:** All language models use tokens

### Tokenization
Breaking text into tokens.
- **Methods:** BPE, WordPiece, SentencePiece
- **Example:** "unhappiness" → ["un", "happiness"]
- **Papers:** All language models

### Transformer
Architecture based on self-attention.
- **Paper:** #1 Attention Is All You Need
- **Components:** Self-attention, FFN, layer norm, residual
- **Impact:** Replaced RNNs, now dominant architecture

### Transfer Learning
Applying knowledge from one task to another.
- **Example:** Pre-train on Wikipedia, fine-tune for medical QA
- **Papers:** BERT (paradigm), GPT-3, all modern models
- **Benefit:** Less data needed for downstream tasks

---

## U

### Unsupervised Learning
Learning from unlabeled data.
- **Examples:** Clustering, autoencoders
- **In LLMs:** Next-token prediction is self-supervised (label = next word)
- **Papers:** Most pre-training is self-supervised

---

## V

### VAE (Variational Autoencoder)
Autoencoder with probabilistic latent space.
- **Used in:** Stable Diffusion (for compression)
- **Benefits:** Smooth latent space, generation capability
- **Components:** Encoder, latent space, decoder

### Vision Transformer (ViT)
Transformer applied to images using patches.
- **Paper:** #11 Vision Transformer
- **Key:** Treat image patches as tokens
- **Impact:** Unified architecture for vision and language

---

## W

### Weight
Learnable parameter in neural network.
- **Training:** Adjust weights to minimize loss
- **Initialization:** Important for convergence
- **Decay:** Regularization technique (penalize large weights)

### Weight Decay
Regularization that penalizes large weights.
- **Also called:** L2 regularization
- **Formula:** Loss = Task Loss + λ||W||²
- **Benefit:** Prevents overfitting

---

## Z

### Zero-Shot Learning
Performing task without any examples.
- **Example:** "Classify this sentiment: [text]" (no examples given)
- **Papers:** GPT-3 (in-context), CLIP (image classification)
- **Challenge:** Harder than few-shot

---

## Common Abbreviations

| Abbrev | Full Term | Meaning |
|--------|-----------|---------|
| **LLM** | Large Language Model | Billion+ parameter language model (GPT-3, LLaMA) |
| **NLP** | Natural Language Processing | Field of AI dealing with text |
| **CV** | Computer Vision | Field of AI dealing with images |
| **ML** | Machine Learning | Learning from data |
| **DL** | Deep Learning | ML with deep neural networks |
| **RL** | Reinforcement Learning | Learning from rewards |
| **SFT** | Supervised Fine-Tuning | Fine-tuning with labeled data |
| **FFN** | Feed-Forward Network | Simple neural network layer |
| **MLP** | Multi-Layer Perceptron | Stacked feed-forward layers |
| **CNN** | Convolutional Neural Network | Network with convolution layers |
| **RNN** | Recurrent Neural Network | Network with recurrence (replaced by Transformers) |
| **LSTM** | Long Short-Term Memory | Type of RNN (also replaced) |
| **GAN** | Generative Adversarial Network | Generator vs discriminator |
| **VAE** | Variational Autoencoder | Probabilistic autoencoder |
| **BPE** | Byte-Pair Encoding | Subword tokenization |
| **BLEU** | Bilingual Evaluation Understudy | Translation quality metric |
| **FID** | Fréchet Inception Distance | Image quality metric |
| **MAE** | Mean Absolute Error | Loss function |
| **MSE** | Mean Squared Error | Loss function |

---

## Mathematical Notation

| Symbol | Meaning | Example |
|--------|---------|---------|
| **d_model** | Hidden dimension size | 768, 4096, 12288 |
| **d_k** | Key/Query dimension | Often d_model / num_heads |
| **d_v** | Value dimension | Usually same as d_k |
| **n** | Sequence length | Number of tokens |
| **h** | Number of attention heads | 8, 12, 96 |
| **N** | Number of parameters | 7B, 175B |
| **D** | Dataset size (tokens) | 300B, 1.4T |
| **C** | Compute (FLOPs) | Training budget |
| **α** | Scaling exponent | In power laws |
| **β** | Optimizer parameter | Adam β₁, β₂ |
| **λ** | Regularization strength | Weight decay coefficient |
| **∝** | Proportional to | L ∝ N^(-α) |
| **⊙** | Element-wise multiplication | Hadamard product |
| **⊕** | Element-wise addition | - |

---

## Units

| Unit | Meaning | Example |
|------|---------|---------|
| **M** | Million (10^6) | 110M parameters |
| **B** | Billion (10^9) | 175B parameters |
| **T** | Trillion (10^12) | 1.4T tokens |
| **FLOP** | Floating Point Operation | Basic compute unit |
| **petaFLOP** | 10^15 FLOPs | GPT-3: 3,640 petaFLOP-days |
| **GPU-hour** | 1 GPU for 1 hour | Training cost measure |
| **Token** | Basic text unit | Subword piece |
| **Step** | One weight update | Training progress |
| **Epoch** | Full pass through data | Usually <1 for LLMs |

---

## Recommended Reading Order

**Start here (basics):**
- Token, Embedding, Attention, Transformer
- Self-Attention, Multi-Head Attention
- Encoder, Decoder, Seq2Seq

**Then core concepts:**
- Pre-training, Fine-Tuning, Transfer Learning
- Parameter, FLOP, Scaling Laws
- Prompt, Few-Shot, In-Context Learning

**Then techniques:**
- LoRA, RAG, Chain-of-Thought
- RLHF, Constitutional AI
- Diffusion, GAN, Latent Space

**Advanced:**
- KL Divergence, PPO, Reward Model
- Quantization, Distillation
- RoPE, SwiGLU, specific optimizations

---

**Last Updated:** 2025-10-19
**Terms Covered:** 150+
**Papers Referenced:** All 15
