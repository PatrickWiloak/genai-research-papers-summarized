# Learning Roadmap - From Beginner to Expert

A structured path through the collection's 107 papers, chosen by your background and goals.

Each path is a curated subset, not the whole library - the point is an order that builds on itself.
When you finish one, [BROWSE.md](../BROWSE.md) and [INDEX.md](../INDEX.md) have everything else,
and [READING_GUIDE.md](./READING_GUIDE.md) says which of it is still worth your time.

---

## Choose Your Path

### Path 1: Complete Beginner (No ML Background)
**Goal:** Understand what modern AI is and how it works
**Time:** 20-30 hours
**Prerequisites:** None

### Path 2: Software Engineer
**Goal:** Build AI applications
**Time:** 15-20 hours
**Prerequisites:** Programming experience

### Path 3: ML Student/Researcher
**Goal:** Deep technical understanding
**Time:** 30-40 hours
**Prerequisites:** Linear algebra, calculus, basic ML

### Path 4: AI Product Manager
**Goal:** Understand capabilities and trade-offs
**Time:** 10-15 hours
**Prerequisites:** None

### Path 5: Reasoning & Agents (2024-2026)
**Goal:** Catch up on everything that happened after the "essential papers" lists were written
**Time:** 12-18 hours
**Prerequisites:** Path 2, or comfort with Transformers, RLHF and Chain-of-Thought

---

## Path 1: Complete Beginner

### Week 1: Foundations
**Goal:** Understand the basic architecture that powers everything

**Day 1-2: Start with Transformers**
- Read: [Transformers summary](../papers/architectures/01-attention-is-all-you-need/summary.md)
- Focus on: "Why This Matters" and "Core Innovation" sections
- Skip: Mathematical formulas on first read
- Watch: "The Illustrated Transformer" (linked in summary)
- **Key takeaway:** Self-attention lets models process all words at once

**Day 3: Understanding Language Models**
- Read: [GPT-3 summary](../papers/language-models/04-gpt3-few-shot-learners/summary.md)
- Focus on: Few-shot learning, in-context learning
- Try: Experiment with ChatGPT using few-shot examples
- **Key takeaway:** Large models can learn from examples in prompts

**Day 4-5: Why Training Matters**
- Read: [Scaling Laws summary](../papers/techniques/12-scaling-laws/summary.md)
- Focus on: The three scaling laws (simple version)
- Read: [LLaMA summary](../papers/language-models/15-llama/summary.md)
- **Key takeaway:** Training longer on more data > just making bigger models

**Day 6-7: Review and Explore**
- Re-read any confusing sections
- Check [Glossary](./GLOSSARY.md) for terms you don't understand
- Watch related YouTube videos (Two Minute Papers, etc.)

### Week 2: Image Generation
**Goal:** Understand how AI creates images

**Day 8-9: Basic Image Generation**
- Read: [GANs summary](../papers/image-generation/02-generative-adversarial-networks/summary.md)
- Focus on: Generator vs discriminator game
- **Key takeaway:** Two models competing makes both better

**Day 10-11: Modern Image Generation**
- Read: [Diffusion Models summary](../papers/image-generation/06-diffusion-models/summary.md)
- Focus on: Iterative denoising process
- Read: [Stable Diffusion summary](../papers/image-generation/07-stable-diffusion/summary.md)
- Try: Generate images with Stable Diffusion online demo
- **Key takeaway:** Modern models denoise step-by-step

**Day 12-13: Connecting Text and Images**
- Read: [CLIP summary](../papers/multimodal/08-clip/summary.md)
- Focus on: How models learn image-text relationships
- **Key takeaway:** Contrastive learning aligns vision and language

**Day 14: Review Week 2**
- Use [Quick Reference](./QUICK_REFERENCE.md) to compare approaches
- Try different text-to-image tools to see concepts in action

### Week 3: Making AI Helpful
**Goal:** Understand alignment and practical techniques

**Day 15-16: Making AI Follow Instructions**
- Read: [InstructGPT summary](../papers/language-models/05-instructgpt-rlhf/summary.md)
- Focus on: RLHF process (simplified)
- **Key takeaway:** Human feedback shapes model behavior

**Day 17-18: Practical Improvements**
- Read: [Chain-of-Thought summary](../papers/techniques/09-chain-of-thought/summary.md)
- Try: Use "let's think step by step" in ChatGPT
- Read: [RAG summary](../papers/techniques/13-rag/summary.md)
- **Key takeaway:** Techniques that make models more useful

**Day 19-20: Efficient Adaptation**
- Read: [LoRA summary](../papers/techniques/10-lora/summary.md) - Focus on "Why This Matters"
- **Key takeaway:** Can customize models without retraining everything

**Day 21: Final Review**
- Read [Comparisons](./COMPARISONS.md) - "When to Use Which"
- Reflect on how all pieces fit together

---

## Path 2: Software Engineer

### Sprint 1: Architecture (3-5 days)
**Goal:** Understand the underlying architectures

1. **Transformers** - The foundation
   - Focus on: Architecture details, code examples
   - Implement: Basic attention mechanism
   - Resources: Hugging Face tutorial

2. **Vision Transformer** - Extending to images
   - Focus on: How patches work, unified architecture
   - Implement: Patch embedding
   - **Key:** Same architecture works for text and images

3. **Scaling Laws** - Planning resources
   - Focus on: Practical implications
   - **Use this:** When choosing model size for your project

### Sprint 2: Practical Techniques (5-7 days)
**Goal:** Learn tools for building applications

1. **RAG** - Most important for applications
   - Focus on: Implementation guide, code examples
   - Implement: Basic RAG with LangChain
   - Try: LlamaIndex tutorials
   - **This is crucial:** 80% of production LLM apps use RAG

2. **LoRA** - Efficient fine-tuning
   - Focus on: Implementation, when to use
   - Implement: Fine-tune a small model with LoRA
   - Resources: PEFT library examples

3. **Chain-of-Thought** - Better prompting
   - Focus on: Prompt engineering techniques
   - Implement: CoT prompts in your app
   - **Quick win:** Improves reasoning immediately

### Sprint 3: Deployment Decisions (2-3 days)
**Goal:** Choose the right approach for your use case

1. **LLaMA** - Open-source options
   - Focus on: Model sizes, deployment costs
   - Compare: LLaMA vs GPT API for your use case
   - Resources: llama.cpp for local deployment

2. **InstructGPT vs Constitutional AI** - Alignment
   - Focus on: Which alignment approach fits your needs
   - Read: [Comparisons](./COMPARISONS.md) - Alignment section

3. **Review [Quick Reference](./QUICK_REFERENCE.md)**
   - Section: "When to Use Which Paper's Techniques"
   - Build: Decision tree for your projects

### Sprint 4: What Production Actually Runs On (4-6 days)
**Goal:** The layer between "it works in a notebook" and "it serves users"

1. **[Dense Retrieval](../papers/techniques/87-dense-retrieval/summary.md)** - the retriever under your RAG
   - Focus on: why pure vector search misses IDs, error codes and proper nouns
   - Build: hybrid BM25 + dense retrieval with rank fusion
   - **This is the single most common RAG bug**

2. **[PagedAttention / vLLM](../papers/techniques/52-pagedattention-vllm/summary.md)** - serving
   - Focus on: KV-cache waste, continuous batching
   - Try: serve a model with vLLM, compare throughput to raw Transformers

3. **[GPTQ & AWQ](../papers/techniques/86-gptq-awq-quantization/summary.md)** - fitting the model
   - Focus on: what 4-bit costs you in quality
   - Try: run a 70B quantised model locally

4. **[MCP](../papers/techniques/59-model-context-protocol/summary.md)** - tool integration
   - Focus on: the M x N problem it removes
   - Build: a small MCP server exposing one of your own tools

5. **[LLM-as-a-Judge](../papers/techniques/85-llm-as-judge/summary.md)** - knowing if it works
   - Focus on: position bias, verbosity bias, self-preference
   - Build: an eval set for your own app before you need one

### Hands-On Project Ideas
- **RAG chatbot:** company docs + hybrid retrieval + a local model, served with vLLM
- **Fine-tuned classifier:** LoRA or QLoRA on domain-specific data
- **Text-to-image app:** Stable Diffusion with ControlNet for layout and a subject LoRA
- **Coding agent:** ReAct loop + MCP tools, scored against a slice of SWE-bench
- **Reasoning assistant:** Chain-of-Thought with self-consistency, measured against a single pass

---

## Path 3: ML Student/Researcher

### Phase 1: Theoretical Foundations (1-2 weeks)
**Goal:** Deep understanding of core innovations

1. **Transformers** - Study in depth
   - Read: Full paper + summary
   - Implement: Full Transformer from scratch
   - Understand: All mathematical details
   - Study: Attention visualization, positional encodings

2. **Scaling Laws** - Mathematical foundations
   - Read: Full paper
   - Understand: Power law derivations
   - Study: Chinchilla revisions
   - Apply: Predict performance for your compute budget

3. **Vision Transformer** - Architecture generalization
   - Read: Full paper + summary
   - Compare: CNN inductive biases vs Transformer learned patterns
   - Implement: Patch embedding, position interpolation
   - Study: Attention patterns in different layers

### Phase 2: Training Methods (2 weeks)
**Goal:** Understand different training paradigms

1. **BERT** - Masked language modeling
   - Implement: MLM from scratch
   - Understand: Bidirectional training
   - Compare: vs autoregressive (GPT)

2. **GPT-3** - Autoregressive scaling
   - Read: Full paper (especially Broader Impacts section)
   - Understand: Few-shot learning emergence
   - Study: In-context learning mechanisms (active research area)

3. **LLaMA** - Compute-optimal training
   - Read: Full paper
   - Understand: How they applied Chinchilla laws
   - Study: Training optimizations (Flash Attention, etc.)
   - Compare: Data mixtures, preprocessing

### Phase 3: Generative Models (2 weeks)
**Goal:** Master generative modeling approaches

1. **GANs** - Adversarial training
   - Read: Full paper + all variants (DCGAN, StyleGAN, etc.)
   - Implement: Basic GAN, understand training dynamics
   - Study: Mode collapse, training stabilization

2. **Diffusion Models (DDPM)** - Denoising approach
   - Read: Full paper + mathematical derivations
   - Implement: DDPM from scratch
   - Understand: Reverse process, score matching, SDE formulation

3. **Stable Diffusion** - Latent diffusion
   - Read: Full paper
   - Understand: VAE compression, latent space diffusion
   - Implement: Text conditioning with cross-attention

4. **CLIP** - Contrastive multimodal
   - Read: Full paper
   - Implement: Contrastive loss, dual encoder
   - Study: Zero-shot capabilities, prompt engineering

### Phase 4: Alignment & Efficiency (1-2 weeks)
**Goal:** Understand alignment and efficient training

1. **InstructGPT (RLHF)** - Human feedback
   - Read: Full paper
   - Understand: PPO for language models, reward modeling
   - Study: SFT → RM → RL pipeline

2. **Constitutional AI** - AI feedback
   - Read: Full paper
   - Compare: vs RLHF (trade-offs)
   - Study: Critique-revision process

3. **LoRA** - Parameter-efficient fine-tuning
   - Read: Full paper
   - Understand: Low-rank decomposition mathematics
   - Implement: LoRA from scratch
   - Compare: vs other PEFT methods (prefix tuning, adapters)

4. **RAG** - Retrieval-augmented
   - Read: Full paper
   - Understand: End-to-end training, marginalization
   - Implement: Dense retrieval + generation

5. **Chain-of-Thought** - Reasoning
   - Read: Full paper
   - Study: Why it works, when it emerges
   - Research: Latest CoT variants (Tree of Thoughts, etc.)

### Phase 5: The Modern Frontier (2-3 weeks)
**Goal:** Reach the current research edge

1. **Efficient architecture** - [RoPE](../papers/techniques/54-rope-rotary-position-embedding/summary.md),
   [GQA](../papers/architectures/75-grouped-query-attention/summary.md),
   [FlashAttention](../papers/techniques/16-flash-attention/summary.md),
   [Mixtral / MoE](../papers/architectures/37-mixture-of-experts/summary.md),
   [Mamba](../papers/architectures/20-mamba/summary.md)
   - Implement: rotary embeddings and grouped-query attention from scratch
   - Understand: why every frontier model is now sparse

2. **Training systems** - [ZeRO & Megatron-LM](../papers/techniques/76-zero-megatron/summary.md)
   - Understand: data, tensor, pipeline and expert parallelism, and where each breaks down
   - This is the phase most theory-first researchers skip and later regret

3. **Alignment, current generation** - [DPO](../papers/language-models/19-dpo/summary.md),
   [KTO](../papers/techniques/103-kto/summary.md),
   [GRPO](../papers/techniques/38-grpo/summary.md),
   [RLVR](../papers/techniques/39-rlvr/summary.md)
   - Derive: DPO's closed form from the RLHF objective; this is the key exercise
   - Compare: what each method removes from the PPO pipeline, and what it costs

4. **Reasoning** - [STaR](../papers/techniques/97-star/summary.md),
   [Process Reward Models](../papers/techniques/51-process-reward-models/summary.md),
   [Quiet-STaR](../papers/techniques/98-quiet-star/summary.md),
   [Test-Time Compute](../papers/techniques/50-test-time-compute/summary.md),
   [DeepSeek-R1](../papers/language-models/26-deepseek-r1/summary.md)
   - Understand: outcome vs process supervision, and the compute-optimal trade
   - R1 is the one to read in full: it documents what o1 did not

5. **Interpretability and safety** -
   [Sparse Autoencoders](../papers/techniques/82-sparse-autoencoders/summary.md),
   [Sleeper Agents](../papers/techniques/83-sleeper-agents/summary.md),
   [Emergent Abilities](../papers/techniques/81-emergent-abilities/summary.md)
   - Implement: a sparse autoencoder on a small model's activations
   - Read Emergent Abilities *with* the Mirage rebuttal - it is a lesson in metric design

6. **Generative modelling, current generation** -
   [DDIM](../papers/image-generation/70-ddim/summary.md),
   [Classifier-Free Guidance](../papers/image-generation/69-classifier-free-guidance/summary.md),
   [Flow Matching](../papers/image-generation/72-flow-matching-sd3/summary.md),
   [DiT](../papers/image-generation/44-sora-dit/summary.md)
   - Derive: flow matching's straight-path objective, compare to the DDPM ELBO

### Research Project Ideas
- **Reproduce results:** Pick a paper, replicate key experiments
- **Ablation studies:** Remove components, measure impact
- **Novel combinations:** LoRA + Constitutional AI, etc.
- **Scaling experiments:** Validate scaling laws on your domain
- **Analysis:** Interpretability of attention patterns, emergent capabilities

### Advanced Resources
- Read all "Further Reading" sections
- Study follow-up papers
- Join research discussions (r/MachineLearning, Twitter/X)
- Read criticism and rebuttals
- Implement from scratch (no libraries)

---

## Path 4: AI Product Manager

### Week 1: Understand Capabilities (6-8 hours)
**Goal:** Know what's possible and impossible

**Day 1: Foundation (2 hours)**
- Read: Transformers - "Why This Matters" only
- Read: Scaling Laws - Focus on practical implications
- **Key question:** How does model size affect capabilities?

**Day 2: Language Models (2 hours)**
- Read: GPT-3 - Few-shot learning section
- Read: LLaMA - Comparison tables
- **Key question:** What can current models do without fine-tuning?

**Day 3: Image Generation (2 hours)**
- Read: Stable Diffusion - Practical applications section
- Read: CLIP - Use cases section
- Try: Generate some images to understand quality/speed
- **Key question:** What's the cost/quality trade-off?

### Week 2: Practical Deployment (6-8 hours)
**Goal:** Understand implementation trade-offs

**Day 4: Making Models Useful (2 hours)**
- Read: RAG - "Why This Matters" and limitations
- Read: Chain-of-Thought - When it helps
- **Key question:** How to reduce hallucinations?

**Day 5: Customization (2 hours)**
- Read: LoRA - Cost comparison
- Read: [Comparisons](./COMPARISONS.md) - RAG vs Fine-tuning section
- **Key question:** When to fine-tune vs use RAG?

**Day 6: Alignment & Safety (2 hours)**
- Read: InstructGPT - Why alignment matters
- Read: Constitutional AI - Transparency benefits
- **Key question:** How to ensure safe, aligned behavior?

### Week 3: Decision Framework (4-6 hours)
**Goal:** Make informed product decisions

**Day 7: Read [Comparisons](./COMPARISONS.md) fully (2-3 hours)**
- Focus on: "When to Use Which", decision matrices
- **Output:** Decision tree for your product

**Day 8: Cost-Benefit Analysis (2-3 hours)**
- Read: [Quick Reference](./QUICK_REFERENCE.md) - Cost comparisons
- Compare: API vs self-hosted
- **Output:** Cost model for your use case

**Day 9: Roadmap Planning**
- Review: What's possible today vs future
- Identify: Which papers' techniques apply to your product
- **Output:** Technical feasibility assessment

### Product Decision Cheat Sheet

**For chatbots:**
- Use: InstructGPT/GPT-4 or Constitutional AI (Claude)
- Add: RAG for knowledge grounding
- Enhance: Chain-of-Thought for complex queries

**For content generation:**
- Text: GPT-3/GPT-4 or LLaMA fine-tuned
- Images: Stable Diffusion
- Both: CLIP for text-image alignment

**For classification/search:**
- Use: BERT-based models or CLIP (multimodal)
- Customize: LoRA fine-tuning if needed

**For domain-specific:**
- Base: LLaMA (open, cost-effective)
- Customize: LoRA fine-tuning
- Knowledge: RAG with domain docs

**Key Metrics to Track:**
- Latency: RAG adds overhead, quantization helps
- Cost: API vs self-hosted, model size
- Quality: Accuracy, hallucination rate
- Safety: Alignment approach (RLHF vs CAI)

---

## Path 5: Reasoning & Agents (2024-2026)

Most "essential AI papers" lists were written before late 2024. This path covers what changed
after them: models that are trained to reason rather than prompted to, alignment that shed its
machinery, and agents that survive more than one turn.

**Assumed:** you already know Transformers, RLHF and Chain-of-Thought. If not, do Path 2 first.

### Stage 1: How Reasoning Became a Training Target (4-5 hours)

The chain of ideas here is unusually clean - read them in order and each one answers a question the
previous one raised.

1. [Self-Consistency](../papers/techniques/77-self-consistency/summary.md) - one chain is
   unreliable, so sample many and vote. *Raises: this is expensive.*
2. [STaR](../papers/techniques/97-star/summary.md) - keep the chains that reached the right answer
   and train on them. *Raises: what if the right answer came from bad reasoning?*
3. [Process Reward Models](../papers/techniques/51-process-reward-models/summary.md) - grade every
   step, not just the answer. *Raises: step labels are expensive.*
4. [Test-Time Compute](../papers/techniques/50-test-time-compute/summary.md) - the theory: when is
   thinking longer better than being bigger? *This is the conceptual centre of the path.*
5. [OpenAI o1](../papers/language-models/31-openai-o1/summary.md) - the first product built on it,
   with the method withheld.
6. [DeepSeek-R1](../papers/language-models/26-deepseek-r1/summary.md) - the same capability, with
   the method published. **Read this one in full.**

### Stage 2: The Training Machinery (3-4 hours)

1. [PPO](../papers/techniques/63-ppo/summary.md) - the baseline everything else is defined against
   (skim if you know it)
2. [DPO](../papers/language-models/19-dpo/summary.md) - drop the reward model
3. [KTO](../papers/techniques/103-kto/summary.md) - drop the requirement for paired data
4. [GRPO](../papers/techniques/38-grpo/summary.md) - drop the critic
5. [RLVR](../papers/techniques/39-rlvr/summary.md) - drop learned rewards entirely, where a
   verifier exists

**The exercise that makes this stick:** for each method, write down what it removed from the
previous pipeline and what that cost. The whole line is a story of subtraction.

### Stage 3: Agents That Last More Than One Turn (3-4 hours)

1. [ReAct](../papers/techniques/21-react/summary.md) - the base loop
2. [Reflexion](../papers/techniques/78-reflexion/summary.md) - learning from failure without
   touching the weights
3. [Self-Refine](../papers/techniques/99-self-refine/summary.md) - the critic pattern, in its
   simplest form
4. [Generative Agents](../papers/techniques/58-generative-agents/summary.md) - memory, reflection
   and planning as an architecture
5. [Voyager](../papers/techniques/100-voyager/summary.md) - skills stored as reusable code
6. [MCP](../papers/techniques/59-model-context-protocol/summary.md) - how the tools get connected
7. [SWE-bench](../papers/techniques/84-swe-bench/summary.md) - how you find out whether any of it
   worked

### Stage 4: Knowing What You Have (2-3 hours)

Reasoning and agent claims are unusually easy to overstate. This stage is the antidote.

1. [LLM-as-a-Judge](../papers/techniques/85-llm-as-judge/summary.md) - and the biases it brings
2. [Emergent Abilities](../papers/techniques/81-emergent-abilities/summary.md) - read it together
   with the Mirage rebuttal
3. [Sparse Autoencoders](../papers/techniques/82-sparse-autoencoders/summary.md) - what
   interpretability can actually deliver today
4. [Sleeper Agents](../papers/techniques/83-sleeper-agents/summary.md) - why "we safety-tuned it"
   is not proof of anything

### Project Ideas

- **Reasoning eval harness:** take a task you care about, and measure single-pass vs
  chain-of-thought vs self-consistency at N=5, 10, 40. Plot accuracy against token spend. The
  crossover point is the most useful number you will produce this month.
- **Verifier-first pipeline:** find a task in your domain with checkable answers, then build the
  verifier before the model. RLVR only works where this exists.
- **Agent with a real scoreboard:** a ReAct loop over MCP tools, scored on a held-out slice of your
  own issue tracker rather than a benchmark.
- **Judge calibration:** have an LLM judge rank outputs you have already ranked yourself, and
  measure the disagreement. Do this before trusting a judge anywhere.

### Where to Go Next

The frontier is moving fastest in world models
([Genie](../papers/techniques/104-genie/summary.md),
[DreamerV3](../papers/techniques/105-dreamerv3/summary.md)) and in applying the same toolkit outside
language ([AlphaFold 3](../papers/techniques/101-alphafold3/summary.md),
[ESM-2](../papers/techniques/106-esm/summary.md),
[AlphaEvolve](../papers/techniques/62-alphaevolve/summary.md)). See
[docs/GAPS.md](./GAPS.md) for what this collection does not cover yet.

---

## General Tips for All Paths

### Active Learning Strategies
1. **Take notes:** Write summaries in your own words
2. **Draw diagrams:** Visualize architectures and processes
3. **Code along:** Implement concepts (even simplified versions)
4. **Teach others:** Explain concepts to solidify understanding
5. **Ask questions:** Use issues or discussions

### When You Get Stuck
1. Check [Glossary](./GLOSSARY.md) for unfamiliar terms
2. Re-read "Why This Matters" section
3. Watch video explanations (linked in summaries)
4. Skip math details on first read, return later
5. Ask in community forums (r/MachineLearning, Discord servers)

### Maximizing Understanding
- **First read:** Focus on concepts, skip equations
- **Second read:** Understand architecture and flow
- **Third read:** Work through mathematics
- **Apply:** Implement or use in a project

### Tracking Progress
- [ ] Completed Week/Sprint/Phase 1
- [ ] Implemented at least one concept
- [ ] Can explain key innovations to someone else
- [ ] Understand when to use each technique
- [ ] Built a small project applying concepts

---

## After Completing Your Path

### Next Steps

**For Beginners:**
- Build a simple project using one concept
- Take an ML course (fast.ai, Coursera)
- Re-read papers with more technical depth

**For Engineers:**
- Build production applications
- Contribute to open-source projects
- Experiment with latest models
- Share your implementations

**For Researchers:**
- Read latest papers (2024-2025)
- Identify research gaps
- Propose novel combinations
- Submit to conferences

**For Product Managers:**
- Prototype AI features
- Evaluate vendor solutions
- Plan AI product roadmap
- Stay updated on new capabilities

### Staying Current

**Follow these resources:**
- ArXiv daily (cs.AI, cs.CL, cs.CV)
- Papers with Code (trending)
- Hugging Face blog
- OpenAI, Anthropic, Google AI blogs
- Twitter/X: AI researchers
- YouTube: Two Minute Papers, Yannic Kilcher

**New papers to watch (2024-2025):**
- Multimodal models (GPT-4V, Gemini)
- Longer context (100k+ tokens)
- Efficient training (MoE, sparse models)
- Better alignment (DPO, RRHF)
- Reasoning improvements (Tree of Thoughts, etc.)

---

## Customizing Your Path

Mix and match based on your specific interests:

**Vision**
- ResNet → U-Net → ViT → MAE → CLIP → SAM 2

**Language models, in lineage order**
- Transformers → GPT-1 → GPT-2 → GPT-3 → Chinchilla → LLaMA → Mistral → Mixtral → DeepSeek-V3

**Image and video generation**
- VAE → GANs → DDPM → DDIM → Classifier-Free Guidance → Stable Diffusion → ControlNet →
  DreamBooth → Flow Matching / SD3 → Sora / DiT

**Alignment and safety**
- InstructGPT → Constitutional AI → DPO → KTO → GRPO → RLVR → Llama Guard → Sleeper Agents →
  Sparse Autoencoders

**Reasoning**
- Chain-of-Thought → Self-Consistency → Tree of Thoughts → STaR → Process Reward Models →
  Test-Time Compute → o1 → DeepSeek-R1

**Efficiency and serving**
- Scaling Laws → Chinchilla → LoRA → QLoRA → FlashAttention → RoPE → GQA → GPTQ & AWQ →
  PagedAttention → Speculative Decoding → Mixture of Experts

**Retrieval and agents**
- RAG → Dense Retrieval → GraphRAG → ReAct → Toolformer → Reflexion → Generative Agents →
  Voyager → MCP

**Multimodal**
- Transformers → ViT → CLIP → LLaVA → Whisper → GPT-4V → Gemini 3

**Science and world models**
- AlphaZero → AlphaFold 2 → AlphaFold 3 → ESM-2 → AlphaGeometry → AlphaEvolve →
  DreamerV3 → Genie → CICERO

Each of these is a complete thread through the collection. [TAGS.md](../TAGS.md) has the generated
version of the same idea, by topic tag.

---

## Estimated Time Investment

| Path | Quick Pass | Thorough | Deep Study |
|------|-----------|----------|------------|
| **Beginner** | 15 hours | 30 hours | 50 hours |
| **Engineer** | 10 hours | 20 hours | 35 hours |
| **Researcher** | 25 hours | 50 hours | 100+ hours |
| **Product Manager** | 8 hours | 15 hours | 25 hours |
| **Reasoning & Agents** | 12 hours | 18 hours | 40 hours |

**Quick pass:** Skim summaries, focus on key sections
**Thorough:** Read all summaries carefully, some code
**Deep study:** Read papers, implement, experiment

---

## Success Criteria

**You've succeeded when you can:**

✅ Explain the key innovation of each paper to a non-expert
✅ Choose the right technique for a given problem
✅ Understand trade-offs between different approaches
✅ Read new AI papers and understand them
✅ Build or deploy an AI application
✅ Critically evaluate AI products and claims

---

**Remember:** Everyone learns differently. Adjust the pace and depth to match your needs. The goal is understanding, not speed!

**Questions?** Open an issue or check [Contributing](../CONTRIBUTING.md).

**Last updated:** 2026-08-20 · Paths draw on all 107 papers; see [INDEX.md](../INDEX.md) for the full list.
