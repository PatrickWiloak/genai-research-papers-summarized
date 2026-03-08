# RLVR: Reinforcement Learning from Verifiable Rewards

**Authors:** Multiple research groups (paradigm, not a single paper)
**Emerged:** 2024-2025
**Key Papers:** DeepSeek-R1, DeepSeekMath, OpenAI o1

---

## Why This Matters

RLVR is **the training paradigm that replaced RLHF for reasoning**:

- 🔄 **Paradigm shift** - From human preferences to verifiable correctness
- 🚫 **No reward model** - Use ground truth instead of learned preferences
- 🧠 **Emergent reasoning** - Models discover reasoning strategies on their own
- 📈 **Scales better** - Unlimited verifiable data vs expensive human labels
- 🎯 **Cleaner signal** - Binary correct/incorrect vs noisy human preferences

**Real-world impact:**
- Enabled DeepSeek-R1 to match OpenAI o1
- Spawned the reasoning model ecosystem (R1, QwQ, Kimi k1.5)
- Changed how the industry thinks about training LLMs
- Made "inference-time compute" a new scaling axis

**The insight:** **For tasks with verifiable answers, you don't need humans in the loop.** Just let the model explore and reward it for being correct.

---

## RLHF vs RLVR

### The Evolution of LLM Training

```
Stage 1: Pre-training (predict next token)
  → Raw capability, but unaligned

Stage 2: SFT (supervised fine-tuning)
  → Follows instructions, but limited by data quality

Stage 3a: RLHF (2022-2024 dominant)
  → Aligned to human preferences
  → Powers ChatGPT, Claude, etc.

Stage 3b: RLVR (2024-2025 emerging)
  → Optimized for correctness on verifiable tasks
  → Powers reasoning models (o1, R1, etc.)
```

### Side-by-Side Comparison

| Aspect | RLHF | RLVR |
|--------|------|------|
| **Reward source** | Learned reward model (from human preferences) | Verifiable ground truth |
| **Signal quality** | Noisy (humans disagree) | Clean (binary correct/incorrect) |
| **Scalability** | Limited by human labeling | Unlimited (auto-generate problems) |
| **Cost** | Expensive (human annotators) | Cheap (automated verification) |
| **What it optimizes** | "What do humans prefer?" | "Is the answer correct?" |
| **Best for** | Chat, writing, general assistance | Math, code, logic, reasoning |
| **Risk** | Reward hacking, sycophancy | Verbose reasoning, narrow focus |
| **Key algorithm** | PPO, DPO | GRPO, PPO |

### How RLHF Works (Review)

```
1. Collect pairs of model outputs
2. Human annotators rank: "Output A is better than Output B"
3. Train reward model on these preferences
4. Use RL to optimize model against reward model

Problems:
- Humans are expensive ($15-50/hour)
- Humans disagree (inter-annotator agreement ~70%)
- Reward model can be wrong
- Model learns to game reward model (reward hacking)
```

### How RLVR Works

```
1. Collect problems with known answers (math, code, logic)
2. Model generates solution
3. Verify: Is the answer correct? (automated)
4. Use RL to reinforce correct solutions

Advantages:
- Verification is free (automated)
- Signal is perfect (correct is correct)
- Infinite data (generate problems programmatically)
- No reward hacking (can't game ground truth)
```

---

## How RLVR Works in Practice

### The Training Loop

```python
# Simplified RLVR training loop
def rlvr_training_step(model, problem_batch):
    for problem, ground_truth in problem_batch:
        # 1. Generate multiple solutions
        solutions = [model.generate(problem) for _ in range(K)]

        # 2. Verify each solution (automated!)
        rewards = []
        for solution in solutions:
            extracted_answer = extract_final_answer(solution)
            reward = 1.0 if extracted_answer == ground_truth else 0.0
            rewards.append(reward)

        # 3. Apply GRPO (or PPO)
        advantages = compute_group_advantages(rewards)
        update_model(model, solutions, advantages)
```

### Verification Methods

**Mathematics:**
```python
def verify_math(solution, ground_truth):
    answer = extract_boxed_answer(solution)  # Extract from \boxed{}
    return float(answer == ground_truth)     # Exact match
```

**Code:**
```python
def verify_code(solution, test_cases):
    try:
        exec(solution)
        results = [run_test(solution, tc) for tc in test_cases]
        return float(all(results))  # All tests pass
    except:
        return 0.0  # Runtime error
```

**Logic/Reasoning:**
```python
def verify_logic(solution, ground_truth):
    answer = extract_answer(solution)
    return float(answer.strip().lower() == ground_truth.strip().lower())
```

### What Data is Used?

```
Mathematics:
- Competition problems (AMC, AIME, IMO)
- Textbook exercises
- Programmatically generated problems
- Theorem proving tasks

Code:
- Programming contest problems (Codeforces, LeetCode)
- Unit test verification
- Algorithm challenges

Logic:
- Formal logic puzzles
- Constraint satisfaction problems
- Planning tasks
```

---

## Why RLVR Produces Reasoning

### The Emergent Behavior Discovery

**DeepSeek-R1-Zero showed that RLVR produces reasoning without being taught:**

```
Training signal: "Is the answer correct?" (binary)
No examples of reasoning provided!

Yet the model spontaneously developed:
- Chain-of-thought reasoning
- Self-reflection ("Wait, let me reconsider...")
- Self-verification ("Let me check my work...")
- Backtracking ("This approach isn't working...")
- Strategy selection ("I'll try method X first...")
```

### Why This Happens

```
Intuition:
1. Random solutions have low accuracy
2. Solutions with intermediate steps have higher accuracy
3. RL reinforces whatever leads to correct answers
4. Model discovers: "Showing work → more often correct"
5. Over time, develops sophisticated reasoning strategies

It's not magic - it's optimization pressure:
"Figure out whatever helps you get right answers"
→ The model discovers that reasoning helps
```

### The Inference-Time Compute Insight

**RLVR created a new scaling axis:**

```
Traditional scaling:
  More training compute → Better model → Better answers

RLVR scaling (inference-time):
  Same model → More thinking time → Better answers

This means you can trade:
  Inference cost (tokens generated) ↔ Answer quality

Like a student who can think longer on a test
```

---

## The RLVR Ecosystem

### Models Trained with RLVR

| Model | Lab | Year | RLVR Details |
|-------|-----|------|-------------|
| OpenAI o1 | OpenAI | 2024 | Undisclosed (likely RLVR) |
| DeepSeek-R1 | DeepSeek | 2025 | GRPO with math/code verification |
| QwQ-32B | Alibaba | 2025 | RLVR-trained reasoning |
| Kimi k1.5 | Moonshot | 2025 | RLVR-inspired training |
| OpenAI o3 | OpenAI | 2025 | Advanced RLVR |
| Gemini 2.5 | Google | 2025 | Integrated thinking mode |

### The RLVR + SFT Recipe

**Most production models combine both:**

```
Phase 1: Pre-training (next token prediction)
  → Raw language ability

Phase 2: SFT (supervised fine-tuning)
  → Instruction following, format, safety

Phase 3: RLHF (human preference optimization)
  → General helpfulness, chat quality

Phase 4: RLVR (verifiable reward optimization)
  → Reasoning, math, code, logic

Result: Model that's both helpful AND capable of deep reasoning
```

---

## Limitations

### 1. Only Works for Verifiable Tasks
```
Verifiable: Math, code, logic, factual QA
NOT verifiable: Creative writing, style, humor, empathy

For non-verifiable tasks, still need RLHF or DPO
```

### 2. Verbose Reasoning
```
RLVR models tend to "think out loud" extensively
Even simple questions get long reasoning chains
Wastes tokens and increases latency

Fix: Dr. GRPO, length penalties, thinking budgets
```

### 3. Narrow Optimization
```
Optimizing for math correctness doesn't improve:
- Writing quality
- Emotional intelligence
- Common sense reasoning
- Creative problem-solving

Need balanced training across objectives
```

### 4. Reward Sparsity
```
Binary correct/incorrect is sparse:
- Model gets 0 reward for 95% of attempts early in training
- Slow initial learning
- Need curriculum (easy → hard problems)

Fix: Partial credit, process rewards, curriculum learning
```

---

## RLVR vs Other Paradigms

### When to Use What

```
RLHF: General assistant behavior, chat, safety alignment
  → "Be helpful and harmless"

RLVR: Reasoning, math, code, verifiable tasks
  → "Get the right answer"

DPO: Quick alignment, preference tuning, style matching
  → "Be more like this, less like that"

SFT: Basic instruction following, format learning
  → "Follow this template"

Best practice: Use ALL of them in sequence
```

---

## Key Takeaways

1. **Verifiable > Preferences** - Ground truth beats human opinions for reasoning tasks
2. **Reasoning emerges** - Models discover reasoning strategies from correctness signal alone
3. **New scaling axis** - Inference-time compute matters as much as training compute
4. **GRPO is the algorithm** - Most RLVR uses GRPO for efficient training
5. **Complementary to RLHF** - Best models use both RLHF (helpfulness) and RLVR (reasoning)

**Bottom line:** RLVR is the paradigm shift that enabled reasoning models. By replacing noisy human preferences with clean verification signals, it unlocked emergent reasoning capabilities that define the current generation of AI.

---

## Further Reading

### Key Papers
- **DeepSeek-R1 (RLVR at scale):** https://arxiv.org/abs/2501.12948
- **DeepSeekMath (GRPO for RLVR):** https://arxiv.org/abs/2402.03300
- **OpenAI o1 (reasoning via RL):** https://openai.com/index/learning-to-reason-with-llms/

### Analysis
- **RLVR Survey:** Cameron Wolfe's deep dive on GRPO
- **1-Shot RLVR:** Shows even 1 training example can unlock reasoning

### Related Work
- **RLHF/InstructGPT:** https://arxiv.org/abs/2203.02155
- **DPO:** https://arxiv.org/abs/2305.18290
- **PPO:** https://arxiv.org/abs/1707.06347

---

**Emerged:** 2024-2025
**Impact:** 🔥🔥🔥🔥🔥 **CRITICAL** - New training paradigm for reasoning
**Adoption:** Universal for reasoning models
**Current Relevance:** THE paradigm for training reasoning capabilities
**Legacy:** Proved that reasoning emerges from correctness optimization

**Modern Status (March 2026):** RLVR is now a standard training stage alongside pre-training, SFT, and RLHF. Every serious reasoning model uses some form of RLVR. The paradigm continues to evolve with better algorithms (Dr. GRPO), richer reward signals (process rewards), and broader task coverage.
