# Let's Verify Step by Step: Process Reward Models

**Authors:** Hunter Lightman, Vineet Kosaraju, Yura Burda, et al. (OpenAI)
**Published:** May 2023 (ICLR 2024 Oral)
**Paper:** [arxiv.org/abs/2305.20050](https://arxiv.org/abs/2305.20050)

---

## Why This Matters

This paper proved that **rewarding correct reasoning steps beats rewarding correct final answers**:

- **Process > Outcome** - Verifying each step outperforms just checking the final answer
- **78.2% on MATH** - State-of-the-art at time of publication using process supervision
- **PRM800K dataset** - Released 800,000 step-level human labels for math reasoning
- **Foundation for o1** - Process reward models are core to how reasoning models work
- **Scalable oversight** - A path toward verifying reasoning in superhuman AI systems

**Real-world impact:**
- Core technique behind OpenAI o1's reasoning verification
- Adopted by DeepSeek-R1 (via process-based rewards)
- Foundation for "verifier-guided search" in reasoning models
- Key component of test-time compute scaling strategies
- Shaped how the field thinks about AI alignment via process supervision

**The insight:** **Don't just check if the answer is right - check if every step of the reasoning is right.** This simple change dramatically improves mathematical reasoning and provides better training signal for reward models.

---

## The Core Problem

### Two Ways to Reward Reasoning

**Outcome Reward Models (ORMs):**
```
Only check the final answer:

Question: "What is 23 x 47?"
Solution: "23 x 47 = 23 x 40 + 23 x 7
          = 920 + 161
          = 1081"  ← Check: Is 1081 correct? YES → Reward

Problem: The reasoning could be wrong even when answer is right
         "23 x 47 = 1081 because I memorized it" → Also rewarded
         Doesn't teach the model to reason well
```

**Process Reward Models (PRMs):**
```
Check every single step:

Question: "What is 23 x 47?"
Solution:
  Step 1: "23 x 47 = 23 x 40 + 23 x 7"  ← Correct decomposition? YES
  Step 2: "= 920 + 161"                   ← 23x40=920? YES. 23x7=161? YES
  Step 3: "= 1081"                        ← 920+161=1081? YES

Every step verified independently
Catches errors even when final answer happens to be right
```

### Why Process Supervision Is Harder

```
Outcome supervision is easy:
  Just check the final answer against ground truth
  Can be automated (for math, coding, etc.)
  No human annotation needed per step

Process supervision is expensive:
  Need to verify EVERY reasoning step
  Requires human experts or very capable models
  Much more annotation per problem

But the investment pays off enormously.
```

---

## How Process Reward Models Work

### The Architecture

```
Standard reward model:
  Input: Question + Full Solution → Score (single number)

Process reward model:
  Input: Question + Solution
  Output: Score for EACH step

  Step 1: "Let x = 3y + 2"        → 0.95 (correct)
  Step 2: "Then 2x = 6y + 4"      → 0.97 (correct)
  Step 3: "So x + y = 10y + 6"    → 0.12 (WRONG! Should be 4y + 2)
  Step 4: "Therefore y = 0.4"     → 0.08 (wrong, follows from error)

  The PRM catches the exact step where reasoning goes wrong.
```

### Training the PRM

```
1. Take a math problem
2. Have the model generate a step-by-step solution
3. Human annotators label each step:
   - Positive: Step is correct given previous steps
   - Negative: Step contains an error
   - Neutral: Ambiguous or just restating

4. Train the reward model on these step-level labels

PRM800K dataset:
  - 800,000 step-level labels
  - 75,000 solutions to 12,000 problems
  - From MATH dataset (competition math)
  - Human expert annotations
```

---

## The Key Results

### Process vs. Outcome Supervision

**On MATH benchmark (competition-level math):**

| Method | MATH Accuracy |
|--------|--------------|
| Majority voting (no verifier) | 69.6% |
| Outcome Reward Model (ORM) | 72.4% |
| **Process Reward Model (PRM)** | **78.2%** |

**Process supervision wins by a significant margin.**

### How Best-of-N with PRM Works

```
Step 1: Generate N solutions to a math problem
        (N = 1860 in the paper's best result)

Step 2: Score each solution with the PRM
        PRM assigns a score to each step
        Overall score = product of step scores (or min)

Step 3: Select the highest-scoring solution

Result:
  N=1:    ~50% accuracy (single attempt)
  N=100:  ~72% accuracy (best of 100 with PRM)
  N=1860: ~78% accuracy (best of 1860 with PRM)

  vs. ORM at N=1860: ~72% (much worse selection)
```

### Why PRM Selects Better Solutions

```
ORM failure mode:
  Solution with wrong reasoning but lucky right answer:
    "2^10 = 1024, and 1024 mod 7 = 2" (correct answer)
    But reasoning skipped crucial steps
    ORM: "Answer is right!" → High score

  Solution with right reasoning but arithmetic slip:
    Detailed, correct approach, typo in final step
    ORM: "Answer is wrong!" → Low score

PRM catches this:
  Evaluates reasoning quality, not just final answer
  Rewards correct process even with minor slips
  Penalizes lucky guesses that skip reasoning
```

---

## Connection to Reasoning Models

### PRMs Enable Test-Time Compute Scaling

```
From "Scaling LLM Test-Time Compute" (Paper 50):

Strategy 1: Search against a verifier
  The verifier IS a process reward model
  PRM scores candidate solutions step-by-step
  Enables selecting the best reasoning chain

Without PRMs:
  Generate 100 solutions → Pick randomly? By length? → Poor

With PRMs:
  Generate 100 solutions → PRM scores each step → Pick best → Great
```

### How o1 and R1 Use Process Rewards

```
OpenAI o1:
  - Trained with process-level reward signals
  - Internal chain-of-thought is verified step by step
  - Model learns to self-correct when a step is wrong
  - PRM800K from this paper likely used in training

DeepSeek-R1:
  - Uses GRPO with verifiable rewards
  - Process-level feedback emerges during RL training
  - Model develops self-verification behaviors
  - "Wait, let me check that step..." ← Process supervision in action

The connection:
  PRMs provide the training signal → Model internalizes step-by-step verification
  → Reasoning models that verify their own work
```

---

## The PRM800K Dataset

### Dataset Details

```
Scale:
  - 12,000 math problems from MATH dataset
  - 75,000 model-generated solutions
  - 800,000 step-level human annotations

Annotation process:
  - Expert mathematicians (not crowdworkers)
  - Each step labeled: positive, negative, or neutral
  - Focus on mathematical correctness
  - ~10 steps per solution on average

Distribution:
  - ~75% positive steps (correct reasoning)
  - ~15% negative steps (errors)
  - ~10% neutral steps (ambiguous/restating)

Released publicly - enabled widespread research on PRMs
```

### What Makes Good Step-Level Labels

```
Positive step:
  "Since x^2 + 6x + 9 = (x+3)^2, we can substitute..."
  Mathematically correct transformation ✓

Negative step:
  "Since x^2 + 6x + 9 = (x+2)^2, we can substitute..."
  Factoring error: should be (x+3)^2, not (x+2)^2 ✗

Neutral step:
  "We need to find the value of x."
  Not wrong, but not a reasoning step - just restating the problem
```

---

## Broader Implications

### Alignment and Scalable Oversight

```
The alignment connection:

If AI systems become superhuman:
  - We can't verify their final answers (too complex)
  - But we CAN verify individual reasoning steps
  - Process supervision = checking each step is valid
  - Even if we can't see the destination, we can check the path

This is "scalable oversight":
  Complex problem → Break into simple steps → Verify each step
  Humans can verify steps even if they can't solve the full problem
```

### Beyond Math

```
PRMs work for any domain with verifiable steps:

Mathematics: Check each algebraic manipulation
Coding: Check each line/function for correctness
Logic: Verify each deductive step
Science: Verify each experimental reasoning step
Legal: Check each argument in a legal chain

Less suitable for:
Creative writing (no "correct" steps)
Opinion-based tasks (subjective)
Simple factual recall (no reasoning chain)
```

---

## Practical Implementation

### Training a PRM

```python
# Simplified PRM training
from transformers import AutoModelForSequenceClassification

class ProcessRewardModel:
    def __init__(self, base_model="meta-llama/Llama-3.1-8B"):
        # PRM outputs a score per step
        self.model = AutoModelForSequenceClassification.from_pretrained(
            base_model, num_labels=1  # Score per step
        )

    def score_solution(self, question, solution_steps):
        """Score each step in a solution."""
        scores = []
        context = question
        for step in solution_steps:
            context += f"\n{step}"
            # Score this step given all previous context
            score = self.model(tokenize(context))
            scores.append(score)
        return scores

    def select_best(self, question, solutions, method="min"):
        """Select best solution from candidates."""
        best_score = -float('inf')
        best_solution = None
        for solution in solutions:
            steps = split_into_steps(solution)
            scores = self.score_solution(question, steps)
            # Use minimum step score (weakest link)
            overall = min(scores) if method == "min" else prod(scores)
            if overall > best_score:
                best_score = overall
                best_solution = solution
        return best_solution
```

### Using PRMs for Best-of-N

```python
def solve_with_prm(model, prm, question, n=100):
    """Generate N solutions and select best using PRM."""
    # Generate N candidate solutions
    solutions = []
    for _ in range(n):
        solution = model.generate(
            question,
            temperature=0.7,  # Some randomness for diversity
            max_tokens=1024
        )
        solutions.append(solution)

    # Score each with PRM
    best = prm.select_best(question, solutions)
    return best

# This simple approach achieves 78.2% on MATH
# vs 69.6% with majority voting (no PRM)
```

---

## Limitations

### 1. Annotation Cost
```
800,000 step-level labels required expert mathematicians
Much more expensive than outcome labels
Not easily scalable to all domains
Automated PRM training (from o1-like models) is an active area
```

### 2. Domain Specificity
```
PRM800K is math-only
PRMs for coding, science, etc. need separate datasets
Transfer between domains is limited
```

### 3. Step Granularity
```
What counts as a "step" is ambiguous
Too fine-grained: Each word is a step (too noisy)
Too coarse: Each paragraph (misses errors)
Optimal granularity varies by domain
```

### 4. Reward Hacking
```
Models can learn to generate "PRM-friendly" steps
Correct-looking but vacuous reasoning
Gaming the step scores without genuine reasoning
Ongoing challenge in reward model research
```

---

## Key Takeaways

1. **Process > Outcome** - Verifying each reasoning step beats just checking final answers (78.2% vs 72.4% on MATH)
2. **Step-level verification** - PRMs catch exactly WHERE reasoning goes wrong, not just IF it's wrong
3. **Enables search** - PRMs are the verifier that makes test-time compute scaling work
4. **Foundation for reasoning models** - o1, R1, and other reasoning models build on process supervision
5. **Scalable oversight** - A path toward verifying superhuman AI reasoning

**Bottom line:** "Let's Verify Step by Step" proved that the granularity of supervision matters enormously. By shifting from outcome-level to process-level rewards, it achieved a 6-point accuracy gain on competition math and established the verification framework that reasoning models like o1 and R1 rely on. The insight that process supervision produces better reasoners has become foundational to modern AI.

---

## Further Reading

### Original Paper
- **Let's Verify Step by Step:** https://arxiv.org/abs/2305.20050
- **PRM800K Dataset:** https://github.com/openai/prm800k

### Related Work
- **Test-Time Compute Scaling:** Paper 50 in this repo
- **OpenAI o1 (Reasoning model):** Paper 31 in this repo
- **GRPO (RL for reasoning):** Paper 38 in this repo
- **RLVR (Verifiable rewards):** Paper 39 in this repo

### Follow-up Papers
- **Math-Shepherd (Auto PRM):** https://arxiv.org/abs/2312.08935
- **OmegaPRM (Monte Carlo PRM):** https://arxiv.org/abs/2406.06592

---

**Published:** May 2023 (ICLR 2024 Oral)
**Impact:** 🔥🔥🔥🔥🔥 **CRITICAL** - Foundation for reasoning model verification
**Citations:** 800+ (as of early 2026)
**Adoption:** Universal - core technique in reasoning models
**Current Relevance:** Foundational to o1, R1, and all reasoning model verification
**Legacy:** Proved process supervision beats outcome supervision

**Modern Status (March 2026):** Process reward models are a core component of every reasoning model. The insight from this paper - that step-level verification dramatically outperforms outcome-level verification - has been validated at scale by o1, R1, and their successors. Automated PRM training (using stronger models to generate step labels) has largely replaced human annotation, making the approach scalable. PRM800K remains widely used for research.
