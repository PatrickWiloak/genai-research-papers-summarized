---
title: "AlphaEvolve: A Gemini-Powered Coding Agent for Designing Advanced Algorithms"
slug: "62-alphaevolve"
number: 62
category: "techniques"
authors: "Alexander Novikov, Ngan Vu, Marvin Eisenberger, et al. (Google DeepMind)"
published: "May 2025"
year: 2025
url: "https://deepmind.google/discover/blog/alphaevolve-a-gemini-powered-coding-agent-for-designing-advanced-algorithms/"
tags: [techniques]
---

# AlphaEvolve: A Gemini-Powered Coding Agent for Designing Advanced Algorithms

**Authors:** Alexander Novikov, Ngan Vu, Marvin Eisenberger, et al. (Google DeepMind)
**Published:** May 2025
**Blog:** [deepmind.google/discover/blog/alphaevolve](https://deepmind.google/discover/blog/alphaevolve-a-gemini-powered-coding-agent-for-designing-advanced-algorithms/)
**Whitepaper:** [AlphaEvolve.pdf](https://storage.googleapis.com/deepmind-media/DeepMind.com/Blog/alphaevolve-a-gemini-powered-coding-agent-for-designing-advanced-algorithms/AlphaEvolve.pdf)

---

## Why This Matters

AlphaEvolve is **the first AI system to discover novel algorithms that outperform anything humans have produced in 50+ years**:

- **Broke a 56-year-old record** - First improvement to Strassen's matrix multiplication algorithm for 4x4 complex matrices since 1969
- **Already running in Google production** - Deployed in data center scheduling, TPU hardware design, and Gemini training kernels
- **0.7% recovery of Google's compute fleet** - Real, measurable savings across all of Google's data centers
- **Generalized scientific discovery agent** - Not narrow like AlphaTensor; works across math, scheduling, hardware, kernels
- **LLM + evolution = research** - Demonstrates a recipe for AI systems doing genuine scientific work

**Real-world impact (May 2025):**
- New matrix multiplication algorithms applicable to ML/scientific computing
- Improved Borg cluster scheduler running across all of Google
- TPU hardware optimizations contributing to next-gen chips
- Faster Gemini training (FlashAttention kernel sped up 32%)
- 75% of cases match human SOTA on 50+ open math problems; 20% improve on it

**The insight:** **An LLM that can write code, paired with an evolutionary search loop and automatic fitness evaluation, can discover algorithms that no human has ever found.** Gemini proposes mutations; the evaluator scores them; the population evolves; repeat at scale. The result is an autonomous algorithm-discovery engine.

---

## The Problem

### Why Algorithm Discovery Is Hard

```
Designing a new algorithm requires:
  1. Deep mathematical intuition
  2. Patience to test thousands of variants
  3. Skill to verify correctness
  4. Creativity to escape local optima

Strassen (1969) found that 2x2 matrix mul needs 7 (not 8) multiplications.
For 56 years, nobody improved 4x4 complex-valued matrix multiplication.
  Best known: 49 scalar multiplications (Strassen recursive)
  Lower bound (theoretical): unknown
  Search space: astronomically large
  Verification: exact algebraic identity must hold
```

### Why Pure LLMs Fail

```
Ask Gemini: "Design a faster matrix multiplication algorithm."

Problems:
  - LLM hallucinates incorrect identities
  - Cannot verify its own arithmetic
  - One-shot generation gets stuck on textbook answers
  - No mechanism to refine through failure
  - No memory of which paths were dead ends

Even reasoning models (o1, Gemini Thinking) plateau quickly:
  Single-shot reasoning doesn't explore the search space
  Cannot accumulate insight across thousands of attempts
```

### Why Pure Search Fails

```
Classical evolutionary algorithms:
  - Random mutations rarely produce valid programs
  - Need handcrafted DSLs (Domain Specific Languages)
  - Cannot leverage prior knowledge
  - Scale poorly to complex algorithm spaces

AlphaTensor (2022): Used RL on a custom tensor game.
  Worked, but limited to one specific problem.
  Each new problem = new game = new RL training run.
```

### The Gap AlphaEvolve Fills

```
Need: A general-purpose algorithm discovery engine that
  1. Writes real code (not toy DSLs)
  2. Evolves over thousands of iterations
  3. Verifies correctness automatically
  4. Generalizes across problem domains
  5. Improves on human-authored SOTA
```

---

## How AlphaEvolve Works

### The Evolutionary Loop

```
                    +---------------------------+
                    |   Problem Specification   |
                    |  (code skeleton + eval)   |
                    +-------------+-------------+
                                  |
                                  v
        +---------------------------------------------------+
        |              Population Database                  |
        |  (programs, scores, lineage, prompt history)      |
        +---------------------------------------------------+
              |                                ^
              | sample parents                 | insert child
              v                                |
        +-----------+   prompt    +------------+----+
        |  Prompt   +-----------> |  Gemini Flash   |
        | Sampler   |             |  + Gemini Pro   |
        +-----------+             +----------+------+
                                             |
                                             | new code
                                             v
                                  +----------+--------+
                                  |  Evaluator Pool   |
                                  |  (run tests, get  |
                                  |   numerical score)|
                                  +----------+--------+
                                             |
                                             v
                                       insert into DB
```

### The Two-Model Recipe

```
Gemini 2.0 Flash:
  - Fast, cheap, high throughput
  - Generates the bulk of mutation proposals
  - Used for breadth: explore many directions

Gemini 2.0 Pro:
  - Slower, more capable, higher cost
  - Used sparingly for hard cases and refinement
  - Used for depth: produce careful, complex edits

The ensemble works because evolution rewards quantity:
  Most Flash mutations fail; cheap to discard.
  Pro mutations succeed more often when stuck.
  Combined cost-per-discovery is dominated by Flash.
```

### The Code Diff Trick

```
AlphaEvolve does NOT ask the LLM to rewrite full programs.

Instead, it requests structured diffs:

  <<<<<<< SEARCH
  def matmul_block(A, B):
      C = np.zeros((4, 4))
      for i in range(4):
          for j in range(4):
              for k in range(4):
                  C[i,j] += A[i,k] * B[k,j]
      return C
  =======
  def matmul_block(A, B):
      # 48-multiplication scheme discovered by AlphaEvolve
      m1 = (A[0,0] + A[1,1]) * (B[0,0] + B[1,1])
      ...
      return assemble(m1, m2, ..., m48)
  >>>>>>> REPLACE

Benefits:
  - Localizes mutation to one region
  - Preserves correct surrounding code
  - LLM stays focused
  - Diffs apply atomically; failures are easy to revert
```

### The Prompt Sampler

```
For each new generation, the sampler builds a prompt with:

  1. Problem description (static)
  2. Best programs found so far (top-k by score)
  3. Recent diverse programs (for exploration)
  4. Inspirations: human-authored references when available
  5. Critic feedback: what failed last time
  6. Few-shot examples of successful diffs

This is essentially RAG over the program archive.
The prompt itself evolves as the population evolves.
```

### Automatic Evaluation

```
Every problem requires a programmable scorer:

  evaluate(program) -> float (higher = better)

Examples:
  - Matrix mul: count scalar multiplications, verify identity
  - Sorting: average comparisons over benchmark inputs
  - Scheduler: simulate cluster, measure cost-per-job
  - Kernel: run on TPU, measure wall-clock latency
  - Hardware: synthesize circuit, report area + power

The scorer must be:
  - Deterministic (or low-variance)
  - Fast (called millions of times)
  - Correctness-checking (reject invalid programs)
  - Numerical (so evolution can rank)
```

---

## Headline Result: 4x4 Complex Matrix Multiplication

### Strassen's 56-Year Wall

```
Standard 4x4 matrix multiply: 64 scalar multiplications
Strassen recursive (1969):    49 scalar multiplications

For complex-valued 4x4 matrices, 49 stood as the record
for 56 years. No human improved it.

AlphaEvolve discovered: 48 scalar multiplications.
```

### Comparison to AlphaTensor (2022)

```
AlphaTensor:
  - Specialized RL agent
  - Trained on a custom tensor decomposition game
  - Found 47 multiplications for 4x5 by 5x5 mod 2
  - Did NOT improve 4x4 complex case
  - Required separate training for each problem

AlphaEvolve:
  - General-purpose evolutionary code agent
  - Same system that does scheduling and kernels
  - Found 48 multiplications for 4x4 complex
  - Beats AlphaTensor on 14 of the matrix-mul targets it tested
  - No retraining per problem
```

### Why This Was Possible

```
Key advantages over AlphaTensor:
  1. Operates on real code, not abstract tensors
  2. Can express continuous-valued algorithms
       (AlphaTensor was restricted to specific finite fields)
  3. Leverages Gemini's prior knowledge of math
  4. Evolution explores larger neighborhoods than RL search
  5. Self-supervised by exact algebraic verification
```

---

## Production Deployments at Google

### Data Center Scheduling (Borg)

```
Problem: Bin-pack jobs onto Google's machines.
  Google's existing scheduler is decades-mature.
  Any improvement compounds across millions of machines.

AlphaEvolve produced: a new heuristic function for Borg.
  Continuously running in production since rollout.
  Recovers ~0.7% of Google's worldwide compute capacity.
  Translates to thousands of machines worth of headroom.
  Code is human-readable and was reviewed before deployment.
```

### TPU Hardware Design

```
Problem: Optimize Verilog circuits for TPU arithmetic units.
  Hand-tuned by hardware engineers for years.

AlphaEvolve found: a circuit-level rewrite of a key arithmetic block.
  Verified by hardware engineers as correct.
  Integrated into an upcoming TPU.
  First time an LLM-derived design has shipped in Google silicon.
```

### Gemini Training Kernels

```
Problem: Speed up the matrix-multiply kernel that
         dominates Gemini training cost.

AlphaEvolve generated tiling and pipelining variants:
  - 23% speedup on the target kernel
  - 1% reduction in total Gemini training time
  - Translates to days of accelerator time saved per training run

For FlashAttention specifically:
  - 32.5% speedup on the FlashAttention kernel
  - Discovered novel block-size and pipelining schedule
```

### Improved Sorting Networks

```
AlphaEvolve generated faster sorting for small arrays
(the hot path inside libstdc++-style sort routines).

Better than the human-tuned versions used in production code.
Now compiled into Google's internal libraries.
```

### Math Benchmark Sweep

```
DeepMind tested AlphaEvolve on 50+ open problems in
mathematics and computer science.

Results:
  ~75% of problems: matched the best human-known solution
  ~20% of problems: BEAT the best known solution

Examples:
  - Kissing number bounds in high dimensions
  - Erdos minimum-overlap problem
  - Sums-of-squares decompositions
  - Ramsey-type constructions
```

---

## Key Innovations

### 1. Code as the Substrate

```
FunSearch (2023, predecessor):
  - Operated on a single Python function (the "priority function")
  - Limited to problems expressible as scoring a function
  - Required heavy human framing of each problem

AlphaEvolve:
  - Evolves arbitrary blocks of arbitrary languages
       (Python, C++, JAX, Verilog, even pseudo-code)
  - Multi-file evolution: can mutate several locations at once
  - Scales from one-line tweaks to whole-module rewrites
```

### 2. Asynchronous Distributed Evolution

```
Traditional EA:
  - Sequential generations
  - Wait for whole population to evaluate
  - Wastes time on slow evaluations

AlphaEvolve:
  - Continuous insertion: any worker can submit any time
  - Diverse sampling: avoid premature convergence
  - Map-elites style archive: keep best per niche
  - Scales to thousands of parallel evaluators
```

### 3. LLM-Guided Mutation

```
Naive mutation: edit a random token.
  -> Almost always produces broken code.

AlphaEvolve mutation: ask Gemini for a targeted diff.
  -> Almost always parses, often compiles, sometimes scores higher.
  -> The LLM imports decades of programmer intuition for free.

This is the core wager of the paper:
  "Evolutionary search becomes tractable when the mutation
   operator already knows how programs work."
```

### 4. Cascade Evaluation

```
Cheap pre-screen: compile + tiny test
  -> Discard 80% of mutations in milliseconds.

Medium screen: standard test suite
  -> Discard another 15%.

Full evaluation: real benchmark
  -> Run only 5% of candidates at full cost.

This 100x's the effective search budget.
```

### 5. No Per-Problem Training

```
The same Gemini model is used across:
  - Pure math (matrix multiplication)
  - Discrete optimization (scheduling)
  - Hardware (Verilog circuits)
  - Software (training kernels)
  - Combinatorics (sorting)

Specialization happens entirely in the prompt and evaluator.
No fine-tuning, no RLHF, no per-domain training run.
```

---

## Comparison to FunSearch and AlphaTensor

```
                  | FunSearch (2023) | AlphaTensor (2022) | AlphaEvolve (2025)
------------------+------------------+--------------------+--------------------
Substrate         | one Py function  | tensor decomposition| arbitrary code
LLM               | Codey (small)    | none (RL only)     | Gemini Flash + Pro
Search            | evolutionary     | RL with MCTS       | evolutionary
Domains           | math + combin.   | matrix mul only    | math, sched, hw, sw
Production use    | none             | none               | Yes (Google scale)
Beat human SOTA?  | sometimes        | yes (some cases)   | yes (many cases)
Per-problem train | no               | yes                | no
Code generated    | tiny snippets    | n/a                | full programs/diffs
```

---

## Architecture: The Full Stack

```
+-------------------------------------------------------------+
|                    Problem Specification                    |
|  - Initial program (code skeleton)                          |
|  - Evaluator (programmable scorer)                          |
|  - Optional: human references, hints, constraints           |
+-------------------------------------------------------------+
                              |
                              v
+-------------------------------------------------------------+
|                  Distributed Controller                     |
|  - Maintains population database                            |
|  - Schedules workers across thousands of cores              |
|  - Handles checkpointing, retries, observability            |
+-------------------------------------------------------------+
                |                            |
                v                            v
+----------------------------+    +-----------------------------+
|       Mutation Workers     |    |     Evaluation Workers      |
|  - Sample parents          |    |  - Run cascade (cheap->full)|
|  - Build prompt            |    |  - Verify correctness       |
|  - Call Gemini Flash/Pro   |    |  - Compute fitness score    |
|  - Apply diffs             |    |  - Report back              |
+----------------------------+    +-----------------------------+
                |                            |
                +------------+---------------+
                             v
                  +---------------------+
                  | Population Archive  |
                  | (best-per-niche +   |
                  |  diverse sample)    |
                  +---------------------+
```

---

## Practical Considerations

### What You Need to Use AlphaEvolve

```
Even though AlphaEvolve is internal to DeepMind, the recipe
is reproducible. To build something similar:

1. A frontier code-capable LLM (Gemini, Claude, GPT-4-class)
2. A programmable, deterministic evaluator
3. An archive that samples by score AND by diversity
4. Diff-based mutation prompts
5. Significant compute (at least 10K-100K LLM calls per problem)

Cost rough order:
  Hard math problems: ~$10K-$100K of LLM compute per discovery
  Production deployment: amortized over years of savings
  Net: extremely positive ROI for Google-scale problems
```

### When AlphaEvolve Excels

```
+ Problem has a fast, deterministic evaluator
+ Solutions are short-to-medium length code
+ Existing human solutions exist as starting point
+ Improvements are continuous (not one-shot binary)
+ Domain has rich structure the LLM has seen before

Example: Matrix mul, kernels, schedulers, circuits.
```

### When It Struggles

```
- No clean evaluator (e.g. "is this proof elegant?")
- Solutions require novel mathematical concepts not in training data
- Search space is purely combinatorial with no smooth gradient
- Evaluator is expensive (>1 minute per run)
- Problem requires real-world experiments
```

---

## Open Questions

### Is AlphaEvolve "Superhuman" at Algorithm Design?

```
For: It beat 50+ year old records held by top mathematicians.
Against: It still requires human-authored evaluators and skeletons.

Most accurate framing:
  AlphaEvolve is a force multiplier on human algorithmic intuition,
  not (yet) a replacement. The human picks the problem and writes
  the evaluator. The system explores the space at superhuman scale.
```

### Can It Discover Concepts, Not Just Optimizations?

```
Current: AlphaEvolve refines and recombines existing ideas.
Open:    Can it invent genuinely new mathematical concepts?
         Strassen's 1969 paper introduced the idea of bilinear
         algorithms. Could AlphaEvolve introduce a new framework?

So far: All discoveries fit within existing frameworks.
```

### The "Verifier Bottleneck"

```
AlphaEvolve only works where verification is easy.
Most of science is the opposite: hypotheses are easy, verification is hard.

For domains without cheap evaluators (biology, social science, physics
experiments), AlphaEvolve's recipe does not directly apply.

This is the fundamental limit on LLM-driven science:
  We can only optimize what we can grade.
```

### Implications for AI Doing Research

```
AlphaEvolve is a proof of concept that AI can do real research:
  - Original contributions
  - Beat human SOTA on long-standing problems
  - Generate code that ships in production

Five-year question:
  How much of mathematics, computer science, and engineering
  research can be automated this way?

DeepMind's bet: a lot. The recipe is general.
```

---

## Limitations

### 1. Requires a Programmable Evaluator
```
Many important problems lack one.
"Make this UX better" or "prove this theorem elegantly"
have no automatic scoring function.
```

### 2. Compute-Hungry
```
Tens of thousands of LLM calls per discovery.
Evaluation can dominate cost (e.g. running TPU benchmarks).
Only worth it for high-value problems.
```

### 3. Local-Search Bias
```
Like all evolution, AlphaEvolve hill-climbs from current population.
Can miss radically different solutions that require
crossing fitness valleys.
```

### 4. Closed Source (For Now)
```
DeepMind has not released AlphaEvolve as a tool.
Open-source efforts (OpenEvolve, others) are reproducing
the recipe in 2025-2026.
```

### 5. Human-in-the-Loop Still Required
```
Every production deployment was reviewed by a human expert
before shipping. AlphaEvolve does not yet self-deploy.
```

---

## Connections to Other Papers

### AlphaTensor (DeepMind, 2022)
```
Direct predecessor on matrix multiplication.
AlphaTensor: RL agent on a single problem.
AlphaEvolve: General agent that beats AlphaTensor on the same problems.
```

### FunSearch (DeepMind, 2023)
```
Direct predecessor on LLM + evolution.
FunSearch: small LLM, single function, narrow.
AlphaEvolve: frontier LLM, full programs, broad domains.
```

### AlphaGeometry (DeepMind, 2024) - paper 61 in this repo
```
Sibling project for Olympiad-level geometry proofs.
Both use neural + symbolic loop. AlphaGeometry is theorem-focused;
AlphaEvolve is algorithm-focused.
```

### Voyager (NVIDIA, 2023)
```
LLM agent that learns Minecraft skills via code.
Same idea - LLM writes code, environment evaluates -
applied to game-playing instead of algorithm discovery.
```

### Tree of Thoughts (Princeton, 2023) - paper 25 in this repo
```
Single-prompt search through reasoning steps.
AlphaEvolve generalizes this to multi-day population-level search.
```

### Reflexion (Northeastern, 2023) - paper 57 in this repo
```
Self-reflection feedback loop for LLM agents.
AlphaEvolve uses similar idea but at population scale instead of
single-trajectory.
```

### Self-Refine (CMU, 2023)
```
Iterative critique-and-improve with LLMs.
AlphaEvolve replaces single-thread refine with parallel evolution.
```

---

## Real-World Applications

### Where the Recipe Generalizes

```
Any field with code + automatic evaluation:

- Compiler optimization passes
- Database query planners
- Solver heuristics (SAT, MILP, constraint programming)
- Network routing
- Game-playing engines
- Cryptographic primitive design
- Numerical method optimization (ODE solvers, PDE schemes)
- Biology pipeline tuning (where simulators exist)
- Robotics controllers (in simulation)
```

### Industry Implications

```
Hyperscalers (Google, Meta, Microsoft, Amazon) have:
  - Frontier LLMs in-house
  - Massive compute
  - Proprietary problems with built-in evaluators
       (training kernels, ad ranking, scheduling, hardware)

Expect every hyperscaler to build an AlphaEvolve clone for
their own production stack within 12-18 months of this paper.
```

### Startup Implications

```
Open-source clones (OpenEvolve, evolve-py) lower the barrier.
Verticals with strong evaluators (compilers, solvers, EDA tools)
become candidates for "AI does the algorithm design" companies.

Defensibility comes from:
  - Owning the evaluator (proprietary benchmark + verifier)
  - Owning the problem distribution (real customer workloads)
  - Owning the deployment surface (where the algorithm runs)
```

---

## Key Takeaways

1. **LLM + evolution = research engine** - Frontier code LLMs combined with population-level evolutionary search produce genuine scientific discoveries.
2. **First post-Strassen improvement in 56 years** - 48-multiplication algorithm for 4x4 complex matrices, a result no human had found.
3. **Already paying for itself at Google** - 0.7% data center efficiency, 23% kernel speedups, hardware design contributions, faster Gemini training.
4. **General, not narrow** - One system tackles math, scheduling, hardware, and software. No per-problem retraining.
5. **The verifier is the bottleneck** - The recipe only works where you can grade output. Building good evaluators is the new "writing good prompts."

**Bottom line:** AlphaEvolve is the most concrete demonstration to date that AI can do original research that beats human experts. By pairing Gemini Flash and Pro with an evolutionary loop, structured code diffs, and automatic evaluation, DeepMind built a system that discovers algorithms in domains where humans had been stuck for decades. The recipe generalizes to any problem with a programmable evaluator, which means we should expect a wave of AlphaEvolve-style systems across compilers, solvers, hardware, and ML infrastructure over the next few years.

---

## Further Reading

### Original Sources
- **Blog:** https://deepmind.google/discover/blog/alphaevolve-a-gemini-powered-coding-agent-for-designing-advanced-algorithms/
- **Whitepaper:** https://storage.googleapis.com/deepmind-media/DeepMind.com/Blog/alphaevolve-a-gemini-powered-coding-agent-for-designing-advanced-algorithms/AlphaEvolve.pdf

### Predecessors
- **AlphaTensor (Nature 2022):** https://www.nature.com/articles/s41586-022-05172-4
- **FunSearch (Nature 2023):** https://www.nature.com/articles/s41586-023-06924-6
- **AlphaGeometry (Nature 2024):** Paper 61 in this repo

### Related Work
- **Tree of Thoughts:** Paper 25 in this repo
- **Reflexion:** Paper 57 in this repo
- **Self-Refine:** https://arxiv.org/abs/2303.17651
- **Voyager:** https://arxiv.org/abs/2305.16291

### Open-Source Reimplementations
- **OpenEvolve:** community reproduction of the AlphaEvolve loop
- **evolve-py:** lightweight evolutionary code search libraries

---

**Published:** May 2025
**Impact:** 🔥🔥🔥🔥🔥 **LANDMARK** - First convincing demonstration of AI doing original algorithm research at superhuman level
**Production Status:** Deployed across Google (Borg, TPU, Gemini training)
**Citations:** Rapidly accumulating; already cited in dozens of follow-up papers within months
**Current Relevance:** Defines the template for LLM-driven scientific discovery agents
**Legacy:** Likely to be remembered as the moment AI started contributing original mathematics and algorithms back to humanity, not just consuming them.

**Modern Status (April 2026):** Open-source reimplementations have appeared (OpenEvolve and others) and the AlphaEvolve recipe has become a standard pattern: frontier LLM + diff-based mutation + programmable evaluator + population-level evolution. Several hyperscalers have launched internal versions targeting their own infrastructure problems. The biggest open question remains how to extend the approach to domains without cheap evaluators - which is, ultimately, most of science.
