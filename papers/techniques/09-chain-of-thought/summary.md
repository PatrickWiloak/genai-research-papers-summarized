# Chain-of-Thought Prompting Elicits Reasoning in Large Language Models

**Authors:** Jason Wei, Xuezhi Wang, Dale Schuurmans, et al. (Google Research, Brain Team)

**Published:** January 2022 (NeurIPS 2022)

**Paper Link:** https://arxiv.org/abs/2201.11903

---

## Why This Paper Matters

This paper introduced **Chain-of-Thought (CoT) prompting**, a simple yet powerful technique that dramatically improves reasoning in large language models. By prompting models to show their work step-by-step (like humans do), CoT enables models to solve complex problems they otherwise fail at. This became a fundamental technique used in ChatGPT, GPT-4, and virtually all modern LLM applications requiring reasoning.

---

## The Core Insight: Show Your Work

### Traditional Prompting (Direct Answer)

```
Q: Roger has 5 tennis balls. He buys 2 more cans of tennis balls.
   Each can has 3 tennis balls. How many tennis balls does he have now?

A: 11
```

**Problem:** Model gives answer without reasoning. If wrong, we don't know why.

### Chain-of-Thought Prompting

```
Q: Roger has 5 tennis balls. He buys 2 more cans of tennis balls.
   Each can has 3 tennis balls. How many tennis balls does he have now?

A: Roger started with 5 balls.
   2 cans of 3 balls each is 2 × 3 = 6 balls.
   5 + 6 = 11.
   The answer is 11.
```

**Benefit:** Step-by-step reasoning improves accuracy and interpretability!

---

## How Chain-of-Thought Works

### The Technique

**Simple idea:** Add examples with reasoning steps in the prompt.

**Standard Few-Shot:**
```
Q: [Example 1 Question]
A: [Example 1 Answer]

Q: [Example 2 Question]
A: [Example 2 Answer]

Q: [Your Question]
A:
```

**Chain-of-Thought Few-Shot:**
```
Q: [Example 1 Question]
A: [Example 1 Step-by-step reasoning]
   The answer is [Example 1 Answer].

Q: [Example 2 Question]
A: [Example 2 Step-by-step reasoning]
   The answer is [Example 2 Answer].

Q: [Your Question]
A: [Model generates reasoning steps]
   The answer is [Model's Answer].
```

**Key:** Include intermediate reasoning steps in the examples.

---

## Detailed Example: Math Word Problem

### Without Chain-of-Thought

```
Q: A juggler can juggle 16 balls. Half of the balls are golf balls,
   and half of the golf balls are blue. How many blue golf balls are there?

Standard Prompting → Answer: 8 ❌

(Model incorrectly calculates half of 16)
```

### With Chain-of-Thought

```
Q: A juggler can juggle 16 balls. Half of the balls are golf balls,
   and half of the golf balls are blue. How many blue golf balls are there?

CoT Prompting:
Let me break this down step by step:
1. The juggler has 16 balls total
2. Half of the balls are golf balls: 16 ÷ 2 = 8 golf balls
3. Half of the golf balls are blue: 8 ÷ 2 = 4 blue golf balls
The answer is 4. ✓

(Correct!)
```

---

## Key Results

### Arithmetic Reasoning (GSM8K)

**Dataset:** Grade school math problems

| Model | Standard Prompting | Chain-of-Thought |
|-------|-------------------|------------------|
| GPT-3 (350M) | 2.0% | 2.5% |
| GPT-3 (1.3B) | 3.0% | 3.8% |
| GPT-3 (6.7B) | 7.0% | 9.2% |
| GPT-3 (175B) | 17.0% | **58.1%** |
| PaLM (540B) | 18.1% | **74.4%** |

**Massive improvements at scale!** 3-4× better with CoT.

### Commonsense Reasoning (CSQA)

**CommonsenseQA dataset:**

| Model | Standard | CoT |
|-------|----------|-----|
| GPT-3 (175B) | 70.8% | 72.5% |
| PaLM (540B) | 79.0% | 83.4% |

**Consistent but smaller improvements** (already high baseline).

### Symbolic Reasoning

**Last letter concatenation task:**
```
Q: Take the last letters of "Elon Musk" and concatenate them.
```

| Model | Standard | CoT |
|-------|----------|-----|
| GPT-3 (175B) | 20% | 58% |

**Nearly 3× improvement!**

---

## Why Chain-of-Thought Works

### Hypothesis 1: Decomposition

**Complex problems → Simple sub-problems**

```
Complex: "Multi-step math problem"
    ↓
Decomposed:
    Step 1: Calculate intermediate value A
    Step 2: Calculate intermediate value B
    Step 3: Combine A and B for final answer
```

Each step is easier than the whole problem.

### Hypothesis 2: Additional Computation

**More "thinking time":**
- Standard: One forward pass
- CoT: Multiple reasoning steps
- More tokens = more computation = better results

**Analogy:** Quick mental math vs. writing work on paper.

### Hypothesis 3: Attention and Context

**Better information flow:**
- Intermediate results kept in context
- Model can reference earlier steps
- Mimics human working memory

### Hypothesis 4: Pattern Matching

**Training data contains reasoning:**
- Web text includes step-by-step solutions
- CoT prompting activates these patterns
- Model "remembers" how to reason

---

## When Chain-of-Thought Helps Most

### Strong Benefits:

**1. Multi-step Problems**
- Math word problems
- Logical reasoning
- Planning tasks

**2. Complex Reasoning**
- Multiple constraints
- Conditional logic
- Temporal reasoning

**3. Arithmetic**
- Calculations
- Number manipulation
- Quantitative reasoning

### Limited Benefits:

**1. Simple Tasks**
- Single-step problems
- Pattern matching
- Retrieval

**2. Small Models**
- < 10B parameters see minimal gains
- Reasoning emerges at scale

---

## Variants and Extensions

### 1. Zero-Shot Chain-of-Thought

**Paper:** "Let's think step by step" (Kojima et al., 2022)

**Technique:**
```
Q: [Question]
A: Let's think step by step.

(Model automatically generates reasoning!)
```

**No examples needed!** Just add magic phrase.

**Results:**
- Surprisingly effective
- Easier to use (no example crafting)
- Slightly lower performance than few-shot CoT

### 2. Self-Consistency

**Paper:** "Self-Consistency Improves Chain of Thought" (Wang et al., 2022)

**Technique:**
1. Generate multiple reasoning paths (sample several times)
2. Take majority vote on final answers

```
Sample 1: [reasoning] → Answer: 42
Sample 2: [reasoning] → Answer: 42
Sample 3: [reasoning] → Answer: 41
Sample 4: [reasoning] → Answer: 42
Sample 5: [reasoning] → Answer: 42

Majority: 42 ✓
```

**Results:** Significant improvement (often 5-15%) over standard CoT.

### 3. Least-to-Most Prompting

**Technique:**
1. Decompose problem into sub-problems
2. Solve sub-problems sequentially
3. Use earlier solutions for later sub-problems

**Example:**
```
Problem: "What is the capital of the country where the Eiffel Tower is located?"

Sub-problem 1: "Where is the Eiffel Tower located?"
→ France

Sub-problem 2: "What is the capital of France?"
→ Paris

Answer: Paris
```

### 4. Tree of Thoughts

**Technique:**
- Generate multiple reasoning steps at each stage
- Evaluate which are most promising
- Explore like a search tree

**More powerful but more expensive.**

### 5. Program-of-Thoughts

**Technique:**
- Generate Python code for calculations
- Execute code for precise results
- Combine reasoning with computation

**Example:**
```
Q: Calculate 347 × 892

CoT: Let me write code:
result = 347 * 892
print(result)

Execute: 309524

Answer: 309524 ✓
```

---

## Practical Examples

### Example 1: Math Problem

**Prompt:**
```
Q: There are 15 trees in the grove. Grove workers will plant trees in the grove today.
   After they are done, there will be 21 trees. How many trees did the grove workers plant?

A: There are originally 15 trees. After planting, there are 21 trees.
   So the workers planted 21 - 15 = 6 trees. The answer is 6.

Q: If there are 3 cars in the parking lot and 2 more arrive, how many cars are there?

A: There are originally 3 cars. 2 more arrive. 3 + 2 = 5. The answer is 5.

Q: A shop has 8 apples. They sell 3 and get a delivery of 12 more. How many apples do they have?

A:
```

**Model Output:**
```
The shop starts with 8 apples. They sell 3, so 8 - 3 = 5 apples left.
Then they receive 12 more, so 5 + 12 = 17 apples. The answer is 17.
```

### Example 2: Logical Reasoning

**Prompt:**
```
Q: All birds can fly. Penguins are birds. Can penguins fly?

A: The first statement says all birds can fly. The second says penguins are birds.
   Following the logic, if all birds can fly and penguins are birds,
   then penguins can fly. However, in reality this is false - the first
   premise is incorrect. Based on the given premises: Yes.
   Based on reality: No, penguins cannot fly.

Q: All mammals are warm-blooded. Whales are mammals. Are whales warm-blooded?

A:
```

**Model Output:**
```
The first statement says all mammals are warm-blooded.
The second says whales are mammals.
If all mammals are warm-blooded, and whales are mammals,
then whales must be warm-blooded.
The answer is Yes, whales are warm-blooded.
```

### Example 3: Commonsense Reasoning

**Prompt:**
```
Q: If you drop a glass on the floor, what will happen?

A: Glass is fragile. When fragile things are dropped on hard surfaces like floors,
   they typically break. So if you drop a glass on the floor, it will likely
   shatter into pieces. The answer is: the glass will break.

Q: If you put an ice cube in hot water, what happens?

A:
```

**Model Output:**
```
Ice is frozen water. Hot water has high temperature.
When ice is exposed to heat, it melts and turns into liquid water.
So if you put an ice cube in hot water, the ice will melt.
The answer is: the ice cube will melt.
```

---

## Limitations and Failure Cases

### 1. **Faulty Reasoning**

**Problem:** Model generates plausible-sounding but incorrect reasoning.

**Example:**
```
Q: I have 3 apples. I give away 5. How many do I have?

A: I started with 3 apples. I gave away 5.
   So 3 - 5 = -2 apples. The answer is -2 apples.

(Logical answer mathematically, but doesn't make real-world sense!)
```

### 2. **Reasoning Shortcuts**

**Problem:** Model skips steps or uses pattern matching instead of reasoning.

### 3. **Hallucination in Steps**

**Problem:** Intermediate steps contain false information.

**Example:**
```
Q: What is the capital of France?

A: France is a country in Europe. The capital of European countries
   is Brussels. Therefore, the capital of France is Brussels.

(False premise leads to wrong answer!)
```

### 4. **Length and Verbosity**

**Problem:** Longer outputs cost more (tokens) and take more time.

**Trade-off:** Better accuracy vs. higher cost.

### 5. **Model Size Dependency**

**Problem:** Doesn't work well on smaller models (< 10B parameters).

**Limitation:** Requires large, expensive models.

---

## Impact on AI Applications

### Direct Applications:

**1. Educational Tools:**
- Math tutoring systems
- Step-by-step problem solving
- Explanatory feedback

**2. Code Generation:**
- Breaking down complex tasks
- Explaining code logic
- Debugging assistance

**3. Data Analysis:**
- Multi-step data queries
- Analytical reasoning
- Report generation

**4. Customer Support:**
- Troubleshooting procedures
- Decision trees
- Complex query resolution

### Research Impact:

**Spawned research directions:**
- Zero-shot reasoning
- Self-consistency
- Tree-of-thought reasoning
- Program-guided reasoning
- Reasoning evaluation benchmarks

**Established paradigm:** Prompting > fine-tuning for many tasks.

---

## Best Practices for Using CoT

### 1. **Craft Good Examples**

**Quality over quantity:**
- 3-5 well-crafted examples often enough
- Ensure examples are correct
- Cover diverse reasoning patterns

### 2. **Format Consistency**

**Structure examples clearly:**
```
Q: [Question]
A: [Step 1 explanation]
   [Step 2 explanation]
   ...
   The answer is [Answer].
```

### 3. **Explicit Final Answer**

**Always end with clear answer:**
- "The answer is X"
- "Therefore, X"
- "So the result is X"

**Why:** Helps parsing and extraction.

### 4. **Balance Detail**

**Not too brief:**
```
Bad: 3 + 2 = 5 ❌
Good: Start with 3, add 2, so 3 + 2 = 5 ✓
```

**Not too verbose:**
```
Bad: We begin by considering the first number, which is 3.
     Then we examine the second number, 2. We must perform
     addition, which is a mathematical operation... ❌
```

### 5. **Use Self-Consistency for Critical Tasks**

**For high-stakes decisions:**
- Generate 5-10 reasoning paths
- Take majority vote
- Review any inconsistencies

---

## Measuring Reasoning Quality

### Accuracy Metrics:

**1. Final Answer Accuracy:**
- Is the final answer correct?
- Standard evaluation metric

**2. Reasoning Step Accuracy:**
- Are intermediate steps correct?
- More nuanced evaluation

**3. Faithfulness:**
- Does reasoning actually lead to answer?
- Or is reasoning post-hoc rationalization?

### Challenges:

**Evaluating reasoning is hard:**
- Multiple valid reasoning paths
- Subjective judgment needed
- Expensive (human evaluation)

---

## Chain-of-Thought vs. Other Techniques

| Technique | Accuracy | Cost | Interpretability | Ease of Use |
|-----------|----------|------|------------------|-------------|
| Standard Prompting | Low | Low | Low | Easy |
| Chain-of-Thought | High | Medium | High | Medium |
| Self-Consistency CoT | Highest | High | High | Medium |
| Fine-Tuning | High | Very High | Low | Hard |
| Tool Use (Code) | Highest | Medium | Medium | Hard |

---

## Key Takeaways

1. **Chain-of-thought dramatically improves reasoning** in large models
2. **Simple technique:** Just add reasoning steps to examples
3. **Emergent ability:** Only works at scale (>10B params)
4. **Interpretability bonus:** See how model reaches conclusions
5. **Variants improve further:** Zero-shot, self-consistency, tree-of-thoughts
6. **Trade-off:** Better accuracy vs. more tokens (cost/time)
7. **Fundamental technique:** Used in ChatGPT, GPT-4, and most LLM apps

---

## Future Directions

### Active Research:

**1. Making reasoning more reliable:**
- Verifying reasoning steps
- Detecting faulty logic
- Combining with external tools

**2. Efficiency improvements:**
- Shorter reasoning chains
- Learned reasoning (fine-tuning)
- Distilling reasoning ability to small models

**3. Complex reasoning:**
- Multi-hop reasoning
- Long-form reasoning
- Combining multiple reasoning types

**4. Evaluation:**
- Better benchmarks
- Automated reasoning evaluation
- Faithfulness metrics

---

## Further Reading

- **Original Paper:** https://arxiv.org/abs/2201.11903
- **Zero-Shot CoT:** https://arxiv.org/abs/2205.11916
- **Self-Consistency:** https://arxiv.org/abs/2203.11171
- **Tree of Thoughts:** https://arxiv.org/abs/2305.10601
- **Least-to-Most Prompting:** https://arxiv.org/abs/2205.10625
- **Prompt Engineering Guide:** https://www.promptingguide.ai/

---

## Citation

```bibtex
@article{wei2022chain,
  title={Chain-of-thought prompting elicits reasoning in large language models},
  author={Wei, Jason and Wang, Xuezhi and Schuurmans, Dale and Bosma, Maarten and Xia, Fei and Chi, Ed and Le, Quoc V and Zhou, Denny and others},
  journal={Advances in Neural Information Processing Systems},
  volume={35},
  pages={24824--24837},
  year={2022}
}
```
