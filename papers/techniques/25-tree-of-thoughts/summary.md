# Tree of Thoughts: Deliberate Problem Solving with Large Language Models

**Authors:** Shunyu Yao, Dian Yu, Jeffrey Zhao, et al. (Princeton, Google DeepMind)
**Published:** May 2023 (NeurIPS 2023)
**Paper:** [arxiv.org/abs/2305.10601](https://arxiv.org/abs/2305.10601)

---

## Why This Matters

Tree of Thoughts (ToT) extends Chain-of-Thought reasoning with **search and backtracking**:

- 🌳 **Explore multiple reasoning paths** - Not just one linear chain
- 🔍 **Self-evaluate and backtrack** - Undo bad decisions
- 🎯 **Solves harder problems** - Creative tasks, planning, search
- 📈 **74% → 74%** on Game of 24 (vs 4% for CoT)
- 🧠 **Closer to human reasoning** - We explore alternatives too

**Real-world impact:**
- Advanced reasoning in GPT-4 and Claude
- Planning and strategy tasks
- Creative problem solving (writing, design)
- Mathematical problem solving

**The insight:** LLMs should explore a **tree** of reasoning paths, not just follow a single **chain**.

---

## The Problem with Chain-of-Thought

### CoT is Linear - Can't Backtrack

**Chain-of-Thought (linear):**
```
Problem: Use 4, 9, 10, 13 to make 24

Thought 1: 13 - 10 = 3
Thought 2: 3 × 9 = 27
Thought 3: 27 - 4 = 23
Result: Failed! (Can't backtrack)
```

**Limitations:**
- ❌ One wrong step → entire chain fails
- ❌ Can't explore alternative paths
- ❌ No way to undo decisions
- ❌ Greedy, not optimal

### Human Reasoning is Tree-Like

**How humans solve it:**
```
Let me try different combinations:
- Path 1: (13-10) × 9 - 4 = 23 ❌ Close but wrong
- Path 2: (10-4) × (13-9) = 24 ✅ This works!
- Backtrack from path 1 when stuck
- Explore path 2 instead
```

**What we need:**
- ✅ Try multiple approaches
- ✅ Evaluate each path
- ✅ Backtrack when stuck
- ✅ Search for solution

---

## Core Innovation

### Tree of Thoughts Framework

**Instead of linear chain, build a search tree:**

```
                    Problem
                       |
          ┌────────────┼────────────┐
     Thought 1a    Thought 1b    Thought 1c
          |            |             |
    ┌────┼────┐   ┌───┼───┐     ┌───┼───┐
  T2a  T2b  T2c  T2a  T2b ...   ...
   |    |    |
  [Evaluate each path, prune bad ones, continue good ones]
```

**Key components:**
1. **Thought decomposition** - Break problem into steps
2. **Thought generation** - Generate multiple candidates per step
3. **State evaluation** - Score how promising each path is
4. **Search algorithm** - BFS or DFS to explore tree

---

## How It Works

### 1. Thought Decomposition

**Define what a "thought" is for your problem:**

**Example: Game of 24**
- Problem: Use 4, 9, 10, 13 to make 24
- Thought = one equation that combines 2 numbers
- Each thought leaves 3 numbers remaining

**Example: Creative Writing**
- Problem: Write a mystery story
- Thought = one paragraph
- Each thought advances the plot

**Example: Crossword**
- Problem: Fill crossword puzzle
- Thought = filling one word
- Each thought constrains remaining words

### 2. Thought Generation

**For each state, generate k candidate next thoughts:**

**Approach 1: Sample multiple thoughts**
```python
# Generate 3 different next steps
thought_1 = llm.generate("Next step: ...")  # Independent sample
thought_2 = llm.generate("Next step: ...")  # Independent sample
thought_3 = llm.generate("Next step: ...")  # Independent sample
```

**Approach 2: Propose multiple in one call**
```python
# Ask for 3 alternatives at once
thoughts = llm.generate("""
Propose 3 different next steps:
1.
2.
3.
""")
```

### 3. State Evaluation

**Evaluate how promising each path is:**

**Method 1: Value each state independently**
```python
for thought in candidate_thoughts:
    score = llm.evaluate(f"""
    On a scale of 1-10, how likely is this path to lead to the solution?
    Current state: {current_state}
    Next thought: {thought}
    Score (1-10):
    """)
```

**Method 2: Vote across states**
```python
prompt = f"""
Which of these paths is most promising?
A) {thought_a}
B) {thought_b}
C) {thought_c}
Vote (A/B/C):
"""
votes = [llm.vote(prompt) for _ in range(5)]  # Get multiple votes
best = most_common(votes)
```

### 4. Search Algorithm

**Choose how to explore the tree:**

**Breadth-First Search (BFS):**
- Explore all options at depth d before depth d+1
- Good when solution is shallow
- More memory intensive

**Depth-First Search (DFS):**
- Explore one path deeply before trying others
- Good when solution is deep
- Less memory, may get stuck

**Beam Search:**
- Keep top-k paths at each level
- Balance between BFS and DFS
- Most practical for LLMs

---

## Example: Game of 24

### Problem
**Use 4, 9, 10, 13 to make 24 using +, -, ×, ÷**

### Tree of Thoughts Solution

**Level 1: Generate 3 possible first steps**
```
A) 13 - 10 = 3   → Remaining: [4, 9, 3]
B) 10 - 4 = 6    → Remaining: [13, 9, 6]
C) 13 - 9 = 4    → Remaining: [10, 4, 4]

Evaluate each:
A) "sure" (keeping high numbers 4, 9)
B) "likely" (created 6, factor of 24)
C) "impossible" (two 4s, hard to make 24)

Prune C, keep A and B
```

**Level 2a: Continue from A [4, 9, 3]**
```
A1) 9 × 3 = 27   → Remaining: [4, 27]
A2) 4 × 3 = 12   → Remaining: [9, 12]
A3) 9 - 3 = 6    → Remaining: [4, 6]

Evaluate:
A1) "impossible" (27 and 4 can't make 24)
A2) "maybe" (12 and 9 hard)
A3) "likely" (6 × 4 = 24!)

Select A3
```

**Level 3: From A3 [4, 6]**
```
6 × 4 = 24 ✓

Solution: (13-10=3), (9-3=6), (6×4=24)
Full: 4 × (9 - (13-10)) = 24
```

**Level 2b: Continue from B [13, 9, 6]**
```
B1) 13 - 9 = 4   → Remaining: [6, 4]
B2) 9 + 6 = 15   → Remaining: [13, 15]

Evaluate:
B1) "likely" (6 × 4 = 24!)
B2) "impossible" (can't make 24)

Select B1 → 6 × 4 = 24 ✓

Solution: (10-4=6), (13-9=4), (6×4=24)
```

**Found 2 solutions by exploring tree!**

### Comparison

| Method | Success Rate (Game of 24) |
|--------|---------------------------|
| IO Prompting | 4% |
| Chain-of-Thought | 4% |
| CoT-Self-Consistency (40 samples) | 9% |
| **Tree of Thoughts (BFS)** | **74%** |

**18× better than CoT!**

---

## Algorithm Pseudocode

### BFS Version

```python
def tree_of_thoughts_bfs(problem, b=3, max_depth=5):
    """
    Tree of Thoughts with Breadth-First Search.

    Args:
        problem: Initial problem state
        b: Branching factor (thoughts per state)
        max_depth: Maximum search depth
    """
    # Initialize with root
    current_level = [problem]

    for depth in range(max_depth):
        next_level = []

        for state in current_level:
            # Check if this state is a solution
            if is_solution(state):
                return state

            # Generate b candidate next thoughts
            thoughts = generate_thoughts(state, num_thoughts=b)

            # Evaluate each thought
            values = [evaluate_state(state, thought) for thought in thoughts]

            # Create new states
            for thought, value in zip(thoughts, values):
                new_state = apply_thought(state, thought)
                next_level.append((new_state, value))

        # Keep top-k most promising states
        next_level.sort(key=lambda x: x[1], reverse=True)
        current_level = [s for s, v in next_level[:b]]

        if not current_level:
            return None  # No solution found

    return None  # Max depth reached


def generate_thoughts(state, num_thoughts):
    """Generate multiple candidate thoughts."""
    prompt = f"""
    Current state: {state}
    Propose {num_thoughts} different next steps:
    """
    response = llm.generate(prompt)
    return parse_thoughts(response)


def evaluate_state(state, thought):
    """Evaluate how promising a state is."""
    prompt = f"""
    State: {state}
    Next thought: {thought}

    Rate the likelihood this leads to solution (1-10):
    """
    score = llm.generate(prompt)
    return int(score)
```

### DFS Version

```python
def tree_of_thoughts_dfs(state, depth=0, max_depth=5):
    """
    Tree of Thoughts with Depth-First Search.
    """
    # Base cases
    if is_solution(state):
        return state
    if depth >= max_depth:
        return None

    # Generate and evaluate thoughts
    thoughts = generate_thoughts(state, num_thoughts=3)
    values = [evaluate_state(state, t) for t in thoughts]

    # Sort by value (best first)
    sorted_thoughts = sorted(zip(thoughts, values),
                            key=lambda x: x[1],
                            reverse=True)

    # Try each thought (DFS)
    for thought, value in sorted_thoughts:
        # Prune obviously bad paths
        if value < threshold:
            continue

        new_state = apply_thought(state, thought)
        result = tree_of_thoughts_dfs(new_state, depth+1, max_depth)

        if result is not None:
            return result  # Found solution

    return None  # No solution in this subtree
```

---

## Results Across Tasks

### Game of 24

| Method | Success Rate |
|--------|--------------|
| IO | 4.0% |
| CoT | 4.0% |
| CoT-SC (100 samples) | 9.0% |
| **ToT (b=1)** | **45%** |
| **ToT (b=5)** | **74%** |

**Key insight:** More branching → better performance

### Creative Writing

**Task:** Write a coherent passage with 4 random words

| Method | Coherence Score |
|--------|----------------|
| IO | 6.2/10 |
| CoT | 6.9/10 |
| **ToT** | **7.6/10** |

**Why better:** Can revise and explore different narrative directions

### 5×5 Mini Crosswords

| Method | Success Rate |
|--------|--------------|
| IO | 0% |
| CoT | 16% |
| **ToT** | **78%** |

**Why much better:** Can backtrack when words don't fit

---

## Real-World Applications

### 1. Mathematical Problem Solving

**Complex multi-step math:**
```python
# ToT can explore different solution strategies
problem = "Prove that sqrt(2) is irrational"

# Strategy 1: Proof by contradiction
# Strategy 2: Use fundamental theorem of arithmetic
# Strategy 3: Continued fractions

# Evaluate which approach is most promising
# Pursue best strategy
# Backtrack if stuck
```

### 2. Code Debugging

**Explore different hypotheses about bugs:**
```
Bug: Function returns wrong output

Hypothesis 1: Off-by-one error in loop
  → Test: Add print statements
  → Result: Loop indices are correct ❌

Hypothesis 2: Wrong operator (< vs <=)
  → Test: Check comparison operators
  → Result: Found it! Should be <= ✓
```

### 3. Creative Writing

**Plot development with alternatives:**
```
Chapter 3: Detective investigates

Branch A: Detective finds fingerprints
  → Evaluate: "Cliché, but moves plot forward" (7/10)

Branch B: Detective interviews witness
  → Evaluate: "Interesting character development" (8/10)

Branch C: Plot twist - detective is suspect
  → Evaluate: "Too early for this twist" (4/10)

Choose B, continue story
```

### 4. Strategic Planning

**Business strategy exploration:**
```
Goal: Increase revenue by 20%

Strategy A: Expand to new markets
  Cost: High | Risk: High | Upside: Very high
  → Explore substrategy: Which market?

Strategy B: Improve existing products
  Cost: Medium | Risk: Low | Upside: Medium
  → More conservative but reliable

Strategy C: Acquire competitor
  Cost: Very high | Risk: Very high
  → May not be feasible

Evaluate all paths, choose best
```

---

## Implementation: LangChain

```python
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate

class TreeOfThoughts:
    def __init__(self, llm, branching_factor=3):
        self.llm = llm
        self.b = branching_factor

    def solve(self, problem, max_depth=5):
        """Solve problem using Tree of Thoughts."""
        return self._bfs(problem, max_depth)

    def _bfs(self, initial_state, max_depth):
        """Breadth-first search."""
        states = [{"state": initial_state, "history": []}]

        for depth in range(max_depth):
            next_states = []

            for current in states:
                # Check if solved
                if self._is_solution(current["state"]):
                    return current

                # Generate thoughts
                thoughts = self._generate_thoughts(current["state"])

                # Evaluate and expand
                for thought in thoughts:
                    value = self._evaluate(current["state"], thought)

                    if value > 5:  # Threshold
                        new_state = self._apply_thought(
                            current["state"], thought
                        )
                        next_states.append({
                            "state": new_state,
                            "history": current["history"] + [thought],
                            "value": value
                        })

            # Keep top-k
            next_states.sort(key=lambda x: x["value"], reverse=True)
            states = next_states[:self.b]

            if not states:
                return None

        return None

    def _generate_thoughts(self, state):
        """Generate candidate next thoughts."""
        prompt = f"""
        Current state: {state}

        Propose {self.b} different next steps to solve this problem.
        Each step should be distinct and promising.

        Steps:
        1.
        2.
        3.
        """
        response = self.llm(prompt)
        return self._parse_thoughts(response)

    def _evaluate(self, state, thought):
        """Evaluate thought value."""
        prompt = f"""
        Current state: {state}
        Proposed next step: {thought}

        On a scale of 1-10, how likely is this step to lead to the solution?
        Consider:
        - Does it make progress toward the goal?
        - Is it a valid step?
        - Does it avoid dead ends?

        Rating (1-10):
        """
        response = self.llm(prompt)
        try:
            return int(response.strip())
        except:
            return 5  # Default

    def _parse_thoughts(self, response):
        """Parse thoughts from LLM response."""
        lines = response.strip().split('\n')
        thoughts = []
        for line in lines:
            if line.strip() and line[0].isdigit():
                # Remove numbering
                thought = line.split('.', 1)[1].strip()
                thoughts.append(thought)
        return thoughts[:self.b]

    def _apply_thought(self, state, thought):
        """Apply thought to state."""
        return f"{state}\n{thought}"

    def _is_solution(self, state):
        """Check if state is a solution."""
        prompt = f"""
        State: {state}

        Is this a complete solution to the problem? (yes/no)
        """
        response = self.llm(prompt).strip().lower()
        return 'yes' in response


# Usage
llm = OpenAI(temperature=0.7, model="gpt-4")
tot = TreeOfThoughts(llm, branching_factor=3)

problem = """
Use the numbers 4, 9, 10, 13 exactly once each with operations +, -, ×, ÷ to make 24.
"""

solution = tot.solve(problem, max_depth=3)
if solution:
    print("Solution found!")
    print("\n".join(solution["history"]))
else:
    print("No solution found")
```

---

## Comparison with Other Methods

### ToT vs Chain-of-Thought

| Aspect | CoT | ToT |
|--------|-----|-----|
| **Search** | Linear (greedy) | Tree (systematic) |
| **Backtracking** | No | Yes |
| **Exploration** | Single path | Multiple paths |
| **Evaluation** | None | Explicit state evaluation |
| **Best for** | Direct problems | Search problems |
| **Cost** | Low (one path) | High (many paths) |

### ToT vs Self-Consistency CoT

| Aspect | Self-Consistency | ToT |
|--------|-----------------|-----|
| **Paths** | Independent samples | Explored tree |
| **Evaluation** | Vote at end | During search |
| **Efficiency** | Generate all, then pick | Prune bad paths early |
| **Backtracking** | No | Yes |

### ToT vs ReAct

| Aspect | ReAct | ToT |
|--------|-------|-----|
| **Tools** | External actions | Internal reasoning |
| **Exploration** | Linear with tools | Tree search |
| **Best for** | Real-world actions | Complex reasoning |
| **Combination** | Can combine! | Can combine! |

**Combined ReAct + ToT:**
- Use ToT to explore action sequences
- Use ReAct to execute actions
- Best of both worlds!

---

## Limitations

### 1. **High Token Cost**
- Explores many paths → many LLM calls
- b=5, depth=4 → up to 5^4 = 625 paths!
- **Mitigation:** Aggressive pruning, lower branching factor

### 2. **Latency**
- Must explore sequentially (can't parallelize fully)
- Deep trees take time
- **Mitigation:** Iterative deepening, time limits

### 3. **Evaluation Quality**
- LLM self-evaluation not always accurate
- Bad evaluation → wrong paths chosen
- **Mitigation:** Multiple votes, external validators

### 4. **Not All Problems Need This**
- Simple Q&A: CoT is enough
- Only helps on problems requiring search
- **When to use:** Planning, creativity, optimization

### 5. **Prompt Engineering**
- Needs good thought decomposition
- Evaluation prompts critical
- Task-specific setup required

---

## When to Use Tree of Thoughts

### ✅ Good Fit

**1. Search Problems**
- Game of 24, Sudoku, puzzles
- Multiple possible solutions
- Need to explore options

**2. Planning Tasks**
- Route planning
- Resource allocation
- Strategy games

**3. Creative Tasks**
- Story writing (explore plot directions)
- Design (explore alternatives)
- Brainstorming

**4. Optimization**
- Find best solution among many
- Multi-objective problems

### ❌ Bad Fit

**1. Simple Q&A**
- "What is the capital of France?"
- CoT is sufficient

**2. Factual Retrieval**
- RAG is better
- No need for search

**3. Time-Sensitive**
- Need answer quickly
- Can't afford search

**4. Limited Budget**
- High token cost
- Use simpler methods

---

## Modern Developments

### 1. **GoT (Graph of Thoughts, 2023)**

**Extension:** Not just trees, but graphs
- Thoughts can merge (not just split)
- Can revisit earlier states
- More flexible than strict tree

### 2. **RAP (Reasoning via Planning, 2023)**

**Combination:** ToT + Monte Carlo Tree Search
- Uses world model to simulate outcomes
- More efficient search
- Better for long-horizon planning

### 3. **Self-Taught Reasoner (STaR)**

**Training approach:**
- Generate reasoning with ToT
- Use successful paths to fine-tune
- Model learns to reason better

### 4. **ToT in GPT-4**

**Speculated internal use:**
- GPT-4 may use ToT-like search internally
- Would explain better reasoning
- Not confirmed by OpenAI

---

## Implementation Tips

### 1. Start Simple
```python
# Don't use full ToT for everything
# Start with CoT, upgrade if needed

if is_simple_problem(task):
    return chain_of_thought(task)
elif needs_search(task):
    return tree_of_thoughts(task)
else:
    return direct_prompting(task)
```

### 2. Tune Branching Factor
```python
# Higher b = better quality, higher cost
# Lower b = faster, may miss solutions

easy_tasks: b = 2
medium_tasks: b = 3
hard_tasks: b = 5
```

### 3. Use Pruning Aggressively
```python
# Don't keep bad paths
if evaluate(thought) < threshold:
    continue  # Skip this branch

# Prune percentage at each level
keep_top_percent = 0.5  # Keep top 50%
```

### 4. Cache Evaluations
```python
# Don't re-evaluate same states
evaluation_cache = {}

def evaluate_cached(state, thought):
    key = hash(state + thought)
    if key not in evaluation_cache:
        evaluation_cache[key] = evaluate(state, thought)
    return evaluation_cache[key]
```

---

## Key Takeaways

1. **Trees > Chains** for problems requiring search
2. **Exploration + evaluation** = better solutions
3. **Backtracking** unlocks new problem classes
4. **Trade-off:** Quality vs cost vs latency
5. **Not universal** - use when appropriate

**Bottom line:** Tree of Thoughts extends LLM reasoning from linear chains to systematic search, enabling solutions to harder problems at the cost of more computation.

---

## Further Reading

### Original Paper
- **Tree of Thoughts:** https://arxiv.org/abs/2305.10601

### Extensions
- **Graph of Thoughts:** https://arxiv.org/abs/2308.09687
- **RAP (Reasoning via Planning):** https://arxiv.org/abs/2305.14992
- **Self-Taught Reasoner:** https://arxiv.org/abs/2203.14465

### Related Work
- **Chain-of-Thought:** https://arxiv.org/abs/2201.11903
- **Self-Consistency:** https://arxiv.org/abs/2203.11171
- **ReAct:** https://arxiv.org/abs/2210.03629

### Implementations
- **Official Code:** https://github.com/princeton-nlp/tree-of-thought-llm
- **LangChain Discussion:** Community implementations
- **Guidance Library:** Microsoft's structured generation

---

**Published:** May 2023
**Impact:** 🔥🔥🔥🔥 **HIGH** - Advanced reasoning paradigm
**Citations:** 400+
**Adoption:** Research stage, some production use for complex tasks
**Current Relevance:** Important for hard reasoning problems
**Legacy:** Showed LLMs can do deliberate search, not just greedy generation

**Modern Status (2024/2025):** Active research area. Not yet mainstream due to cost, but important for problems where CoT fails. May be used internally in frontier models.
