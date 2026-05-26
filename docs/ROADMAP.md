# Learning Roadmap - From Beginner to Expert

A structured path through the 94 papers based on your background and goals.

> **Note:** The original beginner / engineer / researcher / PM tracks below were designed around the original 15 papers. They remain a valid foundation. The **Extended Tracks** section at the bottom adds modern reasoning, agents, scientific AI, and the wider open-source ecosystem.

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

---

## Path 1: Complete Beginner

### Week 1: Foundations
**Goal:** Understand the basic architecture that powers everything

**Day 1-2: Start with Transformers**
- Read: [Transformers summary](./01-attention-is-all-you-need/summary.md)
- Focus on: "Why This Matters" and "Core Innovation" sections
- Skip: Mathematical formulas on first read
- Watch: "The Illustrated Transformer" (linked in summary)
- **Key takeaway:** Self-attention lets models process all words at once

**Day 3: Understanding Language Models**
- Read: [GPT-3 summary](./04-gpt3-few-shot-learners/summary.md)
- Focus on: Few-shot learning, in-context learning
- Try: Experiment with ChatGPT using few-shot examples
- **Key takeaway:** Large models can learn from examples in prompts

**Day 4-5: Why Training Matters**
- Read: [Scaling Laws summary](./12-scaling-laws/summary.md)
- Focus on: The three scaling laws (simple version)
- Read: [LLaMA summary](./15-llama/summary.md)
- **Key takeaway:** Training longer on more data > just making bigger models

**Day 6-7: Review and Explore**
- Re-read any confusing sections
- Check [Glossary](./GLOSSARY.md) for terms you don't understand
- Watch related YouTube videos (Two Minute Papers, etc.)

### Week 2: Image Generation
**Goal:** Understand how AI creates images

**Day 8-9: Basic Image Generation**
- Read: [GANs summary](./02-generative-adversarial-networks/summary.md)
- Focus on: Generator vs discriminator game
- **Key takeaway:** Two models competing makes both better

**Day 10-11: Modern Image Generation**
- Read: [Diffusion Models summary](./06-diffusion-models/summary.md)
- Focus on: Iterative denoising process
- Read: [Stable Diffusion summary](./07-stable-diffusion/summary.md)
- Try: Generate images with Stable Diffusion online demo
- **Key takeaway:** Modern models denoise step-by-step

**Day 12-13: Connecting Text and Images**
- Read: [CLIP summary](./08-clip/summary.md)
- Focus on: How models learn image-text relationships
- **Key takeaway:** Contrastive learning aligns vision and language

**Day 14: Review Week 2**
- Use [Quick Reference](./QUICK_REFERENCE.md) to compare approaches
- Try different text-to-image tools to see concepts in action

### Week 3: Making AI Helpful
**Goal:** Understand alignment and practical techniques

**Day 15-16: Making AI Follow Instructions**
- Read: [InstructGPT summary](./05-instructgpt-rlhf/summary.md)
- Focus on: RLHF process (simplified)
- **Key takeaway:** Human feedback shapes model behavior

**Day 17-18: Practical Improvements**
- Read: [Chain-of-Thought summary](./09-chain-of-thought/summary.md)
- Try: Use "let's think step by step" in ChatGPT
- Read: [RAG summary](./13-rag/summary.md)
- **Key takeaway:** Techniques that make models more useful

**Day 19-20: Efficient Adaptation**
- Read: [LoRA summary](./10-lora/summary.md) - Focus on "Why This Matters"
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

### Hands-On Project Ideas
- **RAG chatbot:** Company docs + LLaMA + LangChain
- **Fine-tuned classifier:** LoRA on domain-specific data
- **Text-to-image app:** Stable Diffusion API integration
- **Reasoning assistant:** GPT with Chain-of-Thought prompts

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

**Interested in vision?**
- Prioritize: ViT → CLIP → GANs → Diffusion → Stable Diffusion

**Interested in language only?**
- Prioritize: Transformers → BERT → GPT-3 → Scaling Laws → LLaMA → InstructGPT

**Interested in alignment/safety?**
- Prioritize: InstructGPT → Constitutional AI → Chain-of-Thought → RAG

**Interested in efficiency?**
- Prioritize: Scaling Laws → LoRA → LLaMA → RAG

**Interested in multimodal?**
- Prioritize: Transformers → ViT → CLIP → Stable Diffusion

---

## Estimated Time Investment

| Path | Quick Pass | Thorough | Deep Study |
|------|-----------|----------|------------|
| **Beginner** | 15 hours | 30 hours | 50 hours |
| **Engineer** | 10 hours | 20 hours | 35 hours |
| **Researcher** | 25 hours | 50 hours | 100+ hours |
| **Product Manager** | 8 hours | 15 hours | 25 hours |

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

**Questions?** Open an issue or check [Contributing](./CONTRIBUTING.md).

---

## 🆕 Extended Tracks (adds papers beyond the original 15)

### Track A: Pre-Transformer Foundations (4 papers, ~4 hours)
For learners who want the *full* historical arc before diving into the Transformer.

1. **Word2Vec** (#63) — meaning lives in vector geometry
2. **Seq2Seq** (#64) — encoder-decoder paradigm invented
3. **Bahdanau Attention** (#65) — attention invented; ancestor of the Transformer
4. **ResNet** (#66) — skip connections; in every Transformer block

### Track B: The GPT Lineage (5 papers, ~5 hours)
The clearest narrative arc in modern AI.

1. **GPT-1** (#69) — decoder-only Transformer, the pretrain-finetune recipe
2. **GPT-2** (#70) — zero-shot transfer at 1.5B params
3. **GPT-3** (#4) — few-shot in-context learning at 175B
4. **InstructGPT** (#5) — RLHF turns GPT into ChatGPT
5. **GPT-4 / GPT-4o / GPT-5** (#36, #40, #42) — modern frontier

### Track C: Modern Reasoning Models (6 papers, ~7 hours)
How models learned to think.

1. **Chain-of-Thought** (#9) — "Let's think step by step"
2. **Self-Consistency** (#85) — majority vote over samples
3. **STaR** (#81) — bootstrap reasoning
4. **Process Reward Models** (#51) — step-level verification
5. **Quiet-STaR** (#82) — internal thoughts at every token
6. **OpenAI o1 / DeepSeek-R1** (#31, #26) — RL-trained reasoning

### Track D: Agents & Tool Use (6 papers, ~6 hours)
1. **ReAct** (#21) — reasoning + acting
2. **Toolformer** (#24) — self-taught tools
3. **Reflexion** (#83) — verbal RL for agents
4. **Self-Refine** (#84) — LLM as critic
5. **Voyager** (#86) — Minecraft agent w/ skill library
6. **SWE-bench** (#91) — the coding-agent benchmark

### Track E: Image Generation Deep Dive (8 papers, ~10 hours)
1. **VAE** (#67) — latent variable generators
2. **GANs** (#2) — adversarial training
3. **DDPM** (#75) — diffusion that beats GANs
4. **VQ-VAE / VQ-GAN** (#76, #77) — discrete tokens for images
5. **Stable Diffusion** (#7) — latent diffusion
6. **Imagen** (#78) — frozen T5 text encoder
7. **ControlNet / DreamBooth** (#79, #80) — control + personalize
8. **Sora / DiT** (#44) — diffusion Transformers for video

### Track F: Scientific & World-Model AI (7 papers, ~8 hours)
For learners interested in AI for science and embodied intelligence.

1. **AlphaZero** (#89) — self-play RL
2. **AlphaFold 2 & 3** (#87, #88) — solved protein folding
3. **ESM-2** (#96) — protein language model
4. **DreamerV3** (#95) — world-model RL
5. **Genie** (#94) — foundation world model
6. **CICERO** (#97) — Diplomacy AI
7. **AlphaGeometry / AlphaEvolve** (#61, #62) — discovery-engine AI

### Track G: Open-Source LLM Ecosystem (6 papers, ~6 hours)
1. **LLaMA / LLaMA 2** (#15, #17)
2. **Mistral 7B** (#72)
3. **Mixtral 8x7B** (#73)
4. **DeepSeek-V3** (#27)
5. **Qwen3** (#28)
6. **Llama Guard** (#92) — safety classifier

### Track H: Interpretability & Safety (4 papers, ~4 hours)
1. **Constitutional AI** (#14)
2. **DPO / KTO** (#19, #93)
3. **Llama Guard** (#92)
4. **Sparse Autoencoders / Scaling Monosemanticity** (#90)

---

**Last Updated:** 2026-05-26
