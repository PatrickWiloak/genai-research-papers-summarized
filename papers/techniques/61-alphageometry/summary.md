---
title: "AlphaGeometry: Solving Olympiad Geometry Without Human Demonstrations"
slug: "61-alphageometry"
number: 61
category: "techniques"
authors: "Trieu H. Trinh, Yuhuai Wu, Quoc V. Le, He He, Thang Luong (Google DeepMind, NYU)"
published: "January 2024 (Nature, vol. 625)"
year: 2024
url: "https://www.nature.com/articles/s41586-023-06747-5"
tags: ["reasoning", "science"]
---

# AlphaGeometry: Solving Olympiad Geometry Without Human Demonstrations

**Authors:** Trieu H. Trinh, Yuhuai Wu, Quoc V. Le, He He, Thang Luong (Google DeepMind, NYU)
**Published:** January 2024 (Nature, vol. 625)
**Paper:** [nature.com/articles/s41586-023-06747-5](https://www.nature.com/articles/s41586-023-06747-5)
**Follow-up:** [AlphaProof / AlphaGeometry 2 - IMO 2024 Silver](https://deepmind.google/discover/blog/ai-solves-imo-problems-at-silver-medal-level/)

---

## Why This Matters

AlphaGeometry is **the first AI system to approach gold-medalist performance on International Mathematical Olympiad (IMO) geometry problems**, and a milestone for neuro-symbolic reasoning:

- **25/30 IMO geometry problems solved** - within 1 problem of the average IMO gold medalist (25.9), more than 2x the previous best AI baseline (10/30)
- **Zero human proofs in training** - learned entirely from 100 million synthetically generated theorems and proofs
- **Neuro-symbolic architecture** - language model proposes auxiliary constructions; symbolic engine performs deductive closure
- **Human-readable proofs** - outputs natural geometric arguments mathematicians can verify, not opaque tactic scripts
- **Foundation for AlphaProof** - the architectural template that won silver at IMO 2024 (Lean-based, 4/6 problems)
- **Terence Tao's commentary:** Called it "astonishing" and "beyond what I would have predicted"

**Real-world impact:**
- Re-opened the door on automated theorem proving as a viable AI research frontier
- Direct lineage to AlphaProof (silver medal, IMO 2024) and the autoformalization wave of 2024-2025
- Inspired the "LLM proposes, verifier checks" pattern that became central to o1, RLVR, and rStar-Math
- Demonstrated that **synthetic data generation with formal verification can replace human demonstrations** in domains where ground truth is checkable
- Shifted expectations: olympiad math went from "decades away" to "next year" almost overnight

**The insight:** **Olympiad geometry is hard for LLMs because it requires creative auxiliary constructions (drawing helper points/lines) that don't follow from the goal.** Symbolic engines can do mechanical deduction perfectly but can't invent these creative leaps. Combine them: let an LLM do the creative part, let a symbolic engine do the rigorous part, and synthesize 100M training examples by running the symbolic engine in reverse.

---

## The Problem

### Why Olympiad Geometry Is Hard

```
A typical IMO geometry problem:

  "Let ABC be an acute triangle with AB < AC. Let O be its circumcenter
   and let D be the foot of the altitude from A. Points E and F lie on AB
   and AC respectively such that BDEF is cyclic. Prove that OE = OF."

What humans do:
  1. Draw the figure (no obvious approach yet)
  2. Try the obvious: angle chasing, similar triangles - stuck
  3. CONSTRUCT something not in the problem:
       - Drop perpendicular from O to AB, call it M
       - Reflect D over the midpoint of EF
       - Draw the circle through B, D, E, F
  4. Now the problem becomes mechanical
  5. Write up the proof

The hard step is step 3: the auxiliary construction.
There are infinite possible constructions. Most don't help.
The right one often requires "seeing" the hidden structure.
```

### Why Symbolic Solvers Alone Fail

```
Classical geometry theorem provers (Wu's method, Grobner bases, GEX):
  + Can verify any provable statement in elementary geometry
  + Decision procedures exist for the algebraic fragment
  - CANNOT invent auxiliary points
  - Search space explodes when you add construction steps
  - On IMO 2000-2022 geometry: solve only ~10/30 problems
  - The 20 unsolved ones all require creative constructions

The bottleneck isn't deduction - it's INVENTION.
```

### Why LLMs Alone Fail

```
GPT-4 on IMO geometry (2023 baseline):
  - Knows definitions, common lemmas, standard tricks
  - Hallucinates "obviously" steps that don't follow
  - Cannot reliably do multi-step deductive chains
  - No way to verify its own proofs
  - Solves 0-2 IMO geometry problems out of 30

LLMs are creative but unreliable.
Symbolic solvers are reliable but uncreative.
```

### The Auxiliary Construction Bottleneck

```
For 30 hard IMO geometry problems:
  - 10 solvable by pure symbolic deduction (no auxiliary points)
  - 20 require constructing 1-3 auxiliary points/lines
  - Of those 20:
      ~15 require well-known constructions (midpoint, foot, reflection)
      ~5 require genuinely creative constructions

If you could automate auxiliary construction, you could solve ~25/30.
That's the AlphaGeometry hypothesis.
```

---

## How AlphaGeometry Works

### The Neuro-Symbolic Loop

```
Input: A geometry problem stated in formal language
       (premises + goal: e.g., "perp(AD, BC), midpoint(M, BC) -> EQ(MA, MB)")

       +-------------------------------------------------+
       |  SYMBOLIC ENGINE (DD + AR)                      |
       |  - Deductive Database: forward chains 9 rules   |
       |  - Algebraic Reasoning: Gaussian elimination    |
       |    over angle/ratio/distance equations          |
       |  - Computes the deductive closure of premises   |
       +---------------------+---------------------------+
                             |
                  Goal proven? ---- Yes ---> Output proof
                             |
                            No
                             |
                             v
       +-------------------------------------------------+
       |  LANGUAGE MODEL (transformer, ~150M params)     |
       |  - Sees current proof state                     |
       |  - Proposes ONE auxiliary construction          |
       |    e.g., "Let M be midpoint of BC"              |
       |  - Adds new point/line to the figure            |
       +---------------------+---------------------------+
                             |
                             v
                   (loop back to symbolic engine)

Loop until proof found or budget exhausted (typically 1-128 attempts).
```

### Key Architectural Choice

```
The LM and symbolic engine speak the SAME formal language.

Domain language (truncated):
  Predicates:  coll(A,B,C)    perp(AB,CD)    para(AB,CD)
               cong(AB,CD)    eqangle(AB,CD,EF,GH)    cyclic(A,B,C,D)
  Constructs:  midpoint(M,A,B)    foot(F,A,BC)    intersection(P,L1,L2)
               on_circle(P,O,r)   reflection(P', P, line)

The LM is fine-tuned to output ONLY constructs (not full proofs).
The symbolic engine handles all predicate-level deduction.

Division of labor:
  LM = "What helper object should I draw?"
  Symbolic = "Given these objects, what follows?"
```

### The Symbolic Engine: DD + AR

```
DD (Deductive Database):
  Forward-chains 9 geometric rules until fixed point
  Examples:
    - If perp(AB,CD) and perp(EF,CD), then para(AB,EF)
    - If midpoint(M,A,B) and midpoint(N,A,C), then para(MN,BC)
  Operates on a database of facts; adds new facts each iteration
  Terminates when no new facts derived

AR (Algebraic Reasoning):
  Some facts are equational, not deducible by rule chaining:
    - Sums of angles in a triangle = 180
    - Ratios from similar triangles
    - Power of a point
  AR runs Gaussian elimination over the linear system of these equations
  Discovers angle/length equalities DD alone misses

Together: DD + AR is a complete decision procedure
          for a large fragment of elementary Euclidean geometry.
```

---

## The Synthetic Data Pipeline

### 100 Million Theorems From Nothing

```
The training data problem:
  - Human-proved IMO problems: ~hundreds globally
  - Annotated proofs in formal language: thousands at most
  - Need: millions of (problem, auxiliary_construction) pairs

The trick: GENERATE them by running the symbolic engine FORWARD
          and then EXTRACTING auxiliary constructions BACKWARD.

Pipeline:
  1. Sample a random geometric premise:
       "Triangle ABC with M = midpoint(BC), H = foot(A, BC), ..."
       (~10-30 random points/lines/circles)

  2. Run symbolic engine to deductive closure:
       Derives THOUSANDS of facts about the figure
       e.g., "MA = MB = MC", "OH || BC", "angle(OAB) = angle(OCA)", ...

  3. Pick a fact F from the closure as the "goal"
       (Now we have a synthetic problem: premise -> F)

  4. Trace BACKWARD through the proof tree:
       Find the MINIMAL set of premise points needed
       Identify points used in the proof but NOT in the minimal premise
       Those are the AUXILIARY constructions

  5. Training example:
       Input:  minimal premise + goal F
       Output: auxiliary points + their constructions
```

### Why This Works

```
Key insight: The symbolic engine generates the proof for free.
            We just need to teach the LM what auxiliary constructions
            are typically useful for what goals.

Statistics from the 100M-theorem corpus:
  - Average premise size: 4-6 points
  - Average auxiliary constructions: 0-5 (most have 1-2)
  - Average proof length: 30-100 deduction steps
  - Coverage: ~9 million unique theorem types

After deduplication and filtering:
  100M raw -> ~9M training examples
  Trained 151M-parameter transformer (Meliad framework)
```

### Diversity Without Human Data

```
The corpus contains theorems that:
  - Re-discover classical results (Euler line, nine-point circle, ...)
  - Match IMO-style structures (cyclic quadrilaterals, radical axes, ...)
  - Include genuinely novel synthetic theorems
  - Span configurations from 5 to 30+ points

Critically: ZERO human-written proofs were used for pretraining.
The LM learns the "language of auxiliary constructions" purely
from the synthetic distribution generated by the symbolic engine.
```

---

## The Neuro-Symbolic Inference Loop

### A Worked Example

```
Problem (IMO 2008 P1, simplified):
  "H is the orthocenter of acute triangle ABC. The circle with center
   midpoint(BC) passing through H meets line BC at A1, A2.
   Define B1, B2, C1, C2 similarly.
   Prove: A1, A2, B1, B2, C1, C2 are concyclic."

Iteration 1:
  Symbolic engine runs DD+AR on premises.
  Derives ~4000 facts. Goal not among them.
  LM sees state, proposes: "Let O be circumcenter of ABC"

Iteration 2:
  Symbolic engine adds O, runs DD+AR.
  Derives ~6500 facts. Closer but not done.
  LM proposes: "Let R = OA (circumradius marker)"

Iteration 3:
  Symbolic engine derives that |OA1| = sqrt(R^2 - something)
  All six points equidistant from O. Goal proven.

Output: human-readable proof in natural geometry language
        with the constructed O explicitly named.
```

### Beam Search and Compute Budget

```
At inference time:
  - LM generates K candidate constructions (beam = 512)
  - Each candidate is fed to the symbolic engine
  - Parallel search across candidates
  - Depth limit: typically 4 auxiliary constructions

Compute per problem:
  - 1 to 4 hours of CPU time on the symbolic engine
  - Single GPU for LM inference
  - Total: comparable to a human olympiad participant's time budget

The system fails gracefully: most failed proofs are still informative
partial deductions (useful for hint generation, lemma extraction).
```

---

## Results: IMO 2000-2022 Benchmark

### The 30-Problem Test Set

```
30 olympiad-level geometry problems from IMOs 2000-2022,
translated into AlphaGeometry's formal language.

System                              Solved
----------------------------------  ------
Wu's method (1978)                  10/30
Full angle method                   3/30
Deductive Database (DD)             7/30
Deductive Database + AR (no LM)     14/30
GPT-4 (zero-shot)                   0/30
GPT-4 (with chain-of-thought)       0/30 (rigorous)
AlphaGeometry                       25/30  <-- DeepMind result
Average IMO gold medalist           25.9/30 (estimated)
```

### Qualitative Wins

```
- AlphaGeometry's proofs are HUMAN-READABLE
  - "Let M be the midpoint of BC. Then OM perp BC..."
  - Mathematicians can verify line by line
  - Contrast with Lean tactic scripts that are verifiable but opaque

- Found a SIMPLER proof than the official solution on IMO 2004 P1
  - Didn't use the auxiliary construction the official guide suggested
  - Discovered a shorter alternative

- Generalized one IMO problem
  - Solved a strictly stronger statement than was asked
  - Suggested a translation invariance the original missed
```

---

## AlphaProof and AlphaGeometry 2 (IMO 2024)

### Silver Medal at IMO 2024

```
At IMO 2024 (July 2024, Bath, UK), DeepMind announced:

System: AlphaProof + AlphaGeometry 2
Result: 4 of 6 problems solved (28/42 points)
        Silver medal threshold: 29 points
        Score: 28 - one point below gold

Problems solved:
  P1 (algebra)        - AlphaProof
  P2 (number theory)  - AlphaProof
  P4 (geometry)       - AlphaGeometry 2 (in seconds)
  P6 (combinatorics)  - AlphaProof (the "hardest" problem)

Problems missed:
  P3, P5 (combinatorics) - both unsolved

Time per problem: up to 3 days on TPUs (vs human limit: 4.5 hours)
But: P4 was solved in seconds.
```

### AlphaGeometry 2

```
Improvements over AlphaGeometry 1:
  - Larger formal language (2x more predicates)
  - Faster symbolic engine (rewrite in C++)
  - Multiple LM agents in parallel with knowledge sharing
  - Handles motion and locus problems
  - Can solve 83% of IMO geometry problems 2000-2023
    (vs 53% for AlphaGeometry 1)
```

### AlphaProof: From Geometry to All of Math

```
The big leap: generalize beyond geometry.

Architecture:
  - Replace custom symbolic engine with LEAN 4 theorem prover
  - Replace custom formal language with Lean's mathlib
  - LM proposes Lean tactics (not auxiliary constructions)
  - Lean kernel verifies every step

Training:
  - 1 million informal math problems formalized via LM (autoformalization)
  - AlphaZero-style RL: LM proposes proofs, Lean verifies, success -> training signal
  - Self-play: harder problems generated as easier ones are solved

Key innovation: AUTOFORMALIZATION at scale
  - Take natural-language problem
  - LM converts to Lean statement (often multiple candidates)
  - Lean verifies the statements are well-formed
  - Whichever statement gets proven is taken as ground truth

This dissolved the "formal data scarcity" problem in one stroke.
mathlib went from ~150K theorems to effectively unlimited.
```

### Why This Matters: The Lean Bridge

```
Before AlphaProof:
  - Formal math: niche, painful, low-data
  - LLM math: impressive but unverifiable
  - Bridge between them: nonexistent

After AlphaProof:
  - LLMs autoformalize at scale
  - Lean kernel provides cryptographic ground truth
  - Reinforcement learning has a verifier
  - The "AI mathematician" research program is now plausible

Direct lineage:
  - DeepSeek-Prover (2024-2025)
  - InternLM-Math
  - Lean Copilot
  - Hundreds of "LLM + theorem prover" papers in 2024-2025
```

---

## Terence Tao's Commentary

```
Fields Medalist Terence Tao on AlphaGeometry (Mastodon, Jan 2024):

  "This is impressive work. ... I would not have expected a tool of
   this nature to be developed so soon. ... The architectural choice
   of using a language model to suggest auxiliary constructions, and
   a symbolic engine to verify, seems like the right separation of
   concerns. ... I expect we'll see this neuro-symbolic pattern
   replicated across many domains of formal reasoning."

On AlphaProof (July 2024):

  "Several mathematician colleagues did say the quality of the
   generated proofs is at the level of a strong IMO contestant.
   This is not a complete game-changer, but it is a significant
   step. The right way to think of these tools is as 'mathematician
   accelerators', not replacements."

Key Tao prediction (Aug 2024):
  "Within a few years, AI systems will be able to handle
   research-level mathematics for narrow problem types,
   particularly in formal verification of conjectures."
```

---

## Connections to Other Papers

### Process Reward Models (Paper #51)

```
PRMs and AlphaGeometry share an architectural insight:
  - Don't reward final answers; reward correct intermediate steps
  - AlphaGeometry: every symbolic deduction is a verified step
  - PRMs: train a verifier per step

AlphaProof took this further:
  - Lean's kernel IS a perfect process reward model
  - Every tactic application either type-checks or doesn't
  - No need to train a separate verifier; the prover IS the verifier

This pattern - "use a verifier as the reward signal" - became RLVR
  (Reinforcement Learning with Verifiable Rewards), which underpins
  o1, DeepSeek-R1, and the entire 2024-2025 reasoning-model era.
```

### OpenAI o1 / DeepSeek-R1

```
The conceptual pipeline:
  AlphaGeometry (Jan 2024)
      "LM proposes, symbolic verifies, train on success"
            |
            v
  AlphaProof (Jul 2024)
      "Same idea but with Lean for general math"
            |
            v
  o1 (Sep 2024)
      "Same idea but with general verifiers (math/code/etc.)
       and chain-of-thought as the proposal mechanism"
            |
            v
  DeepSeek-R1 (Jan 2025)
      "Same idea, open-source, RLVR is the recipe"

AlphaGeometry is the existence proof that this loop closes
in a hard reasoning domain. Everything since has been scaling it.
```

### rStar-Math (Paper #35)

```
rStar-Math (Microsoft, Jan 2025) explicitly cites AlphaGeometry's
synthetic-data + verifier loop:
  - MCTS over math reasoning steps (instead of LM beam search)
  - Process reward model verifies each step
  - Self-improvement loop: success -> training data -> better policy

The "small model with strong verifier beats big model" finding
in rStar-Math echoes AlphaGeometry's 151M-parameter LM matching
gold-medalist humans in geometry.
```

### Autoformalization Wave

```
AlphaProof's autoformalization at scale inspired:
  - LLM-AutoFormalization (Wu et al., 2024)
  - DeepSeek-Prover-V2 (2025)
  - Lean Copilot in mathlib
  - The Equational Theories Project (Tao et al., crowdsourced + AI)

Pattern: use LLMs to translate informal -> formal,
         filter via the prover's type checker,
         scale formal corpora 10-100x.
```

---

## Real-World Applications

### 1. Mathematical Discovery

```
- Co-pilot for working mathematicians (Lean Copilot, mathlib bots)
- Automated lemma generation for ongoing formalization projects
- Conjecture verification at scale
- Equational Theories Project (Tao 2024) used similar AI assistance
```

### 2. Math Education

```
- Hint generation for students stuck on geometry problems
- Multiple-proof generation (different approaches to same problem)
- Solution explanation in natural geometric language
- Khan Academy / Brilliant integration prototypes (rumored 2025)
```

### 3. Formal Verification Beyond Math

```
- Hardware verification (the same "creative step + mechanical check" pattern)
- Smart contract verification
- Cryptographic protocol proofs
- Distributed systems proofs (TLA+/Lean hybrid)
```

### 4. Scientific Reasoning

```
- AlphaProof's autoformalization template applied to physics, chemistry
- Symbolic regression with LLM-proposed forms, numerical verification
- Materials science: LLM proposes structures, DFT verifies stability
```

---

## Limitations

### 1. Domain-Specific Symbolic Engine
```
AlphaGeometry 1's DD+AR engine is hand-built for plane geometry.
Doesn't generalize to other math.
AlphaProof solved this by switching to Lean - but that brought
Lean's own complexity.
```

### 2. Compute Cost
```
IMO 2024 problems took up to 3 DAYS on TPUs per problem.
Humans had 4.5 hours total for 3 problems.
"Silver medal" comes with a serious asterisk on time-to-solution.
```

### 3. Coverage Gaps
```
AlphaGeometry: didn't solve 5/30 IMO problems
  - Mostly involve inequalities or motion
  - The formal language doesn't cover everything
AlphaProof: didn't solve P3 and P5 (combinatorics) at IMO 2024
  - Combinatorics resists formalization more than geometry/algebra
```

### 4. Not a "General Mathematician"
```
The system can prove statements you give it.
It cannot decide what statements are interesting.
Cannot read a research paper, identify open questions, formalize them,
generalize them, or write expository math.
The "creative mathematician" gap is much wider than the IMO gap.
```

### 5. Synthetic Data Distribution Shift
```
The 100M synthetic theorems may not match the distribution of
human-interesting theorems. AlphaGeometry's proofs sometimes feel
"alien" - technically correct but stylistically unusual.
```

---

## Key Takeaways

1. **Neuro-symbolic separation of concerns wins** - LMs propose creative steps, symbolic engines verify rigor. Neither alone is sufficient; the combination is dramatically more capable than either.

2. **Synthetic data with formal verification can replace human demonstrations** - 100M theorems generated from a symbolic engine, zero human proofs, gold-medalist performance. This template generalizes.

3. **Auxiliary construction is the bottleneck in olympiad geometry** - Identifying the bottleneck and targeting the LM specifically at that step (not full proofs) is what made the system work.

4. **AlphaProof generalized the recipe with Lean** - Replacing the custom symbolic engine with a general-purpose theorem prover and adding autoformalization unlocked all of math, not just geometry.

5. **This is the architecture behind modern reasoning models** - The "propose, verify, train on success" loop is now the dominant paradigm via RLVR, o1, DeepSeek-R1, and rStar-Math. AlphaGeometry was the first proof point that it scales to olympiad-level reasoning.

**Bottom line:** AlphaGeometry showed that a small language model paired with a symbolic verifier and trained on synthetic data can solve problems that defeated both pure LLMs and pure symbolic systems for decades. Its successor AlphaProof generalized the approach to all of math via Lean and autoformalization, earning silver at IMO 2024. The architectural pattern - LM as creative proposer, formal system as verifier, synthetic data via the verifier itself - has since become the dominant recipe for AI reasoning systems.

---

## Further Reading

### Original Papers
- **AlphaGeometry (Nature 2024):** https://www.nature.com/articles/s41586-023-06747-5
- **AlphaProof / AlphaGeometry 2 blog:** https://deepmind.google/discover/blog/ai-solves-imo-problems-at-silver-medal-level/

### Code and Models
- **AlphaGeometry GitHub:** https://github.com/google-deepmind/alphageometry
- **mathlib (Lean math library):** https://github.com/leanprover-community/mathlib4

### Related Work in This Repo
- **Process Reward Models:** Paper #51 (verification as reward signal)
- **rStar-Math:** Paper #35 (small model + verifier + synthetic data)
- **PagedAttention / vLLM:** Paper #52 (efficient serving for long-CoT reasoning)

### Background on Theorem Proving
- **Lean Theorem Prover:** https://leanprover.github.io/
- **mathlib overview:** https://leanprover-community.github.io/mathlib-overview.html
- **Tao on AI in math:** https://terrytao.wordpress.com/category/mathematics/

### Follow-up Research (2024-2025)
- **DeepSeek-Prover-V2:** https://arxiv.org/abs/2408.08152
- **Lean Copilot:** https://github.com/lean-dojo/LeanCopilot
- **Autoformalization Survey:** https://arxiv.org/abs/2406.06777
- **Equational Theories Project (Tao):** https://teorth.github.io/equational_theories/

---

**Published:** January 2024 (Nature)
**Impact:** 🔥🔥🔥🔥🔥 **LANDMARK** - First AI to reach gold-medalist level on IMO geometry
**Citations:** 800+ (as of early 2026)
**Adoption:** Architectural template for AlphaProof, o1-class reasoning, RLVR
**Current Relevance:** Foundational - the neuro-symbolic + synthetic-data + verifier recipe

**Modern Status (April 2026):** AlphaGeometry's neuro-symbolic recipe has become the dominant paradigm for AI reasoning. AlphaProof's silver-medal result at IMO 2024 was matched and exceeded by GPT-5 and Gemini 2.5 systems on IMO 2025 (gold-level, multiple labs). The autoformalization + Lean verification approach pioneered here now drives a thriving ecosystem of formal math tools, with mathlib growing 3x in size and major mathematicians (Tao, Buzzard, Scholze) actively integrating AI assistants into their workflow. The pattern - propose with an LM, verify with a formal system, train on verified successes - is the foundation of modern reasoning models including o3, Claude Sonnet 4.5, and Gemini 2.5 Deep Think.

<!-- related:start -->

---

## Related in This Collection

- [DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning](../../language-models/26-deepseek-r1/summary.md)
- [Gemini 2.5: Pushing the Frontier with Advanced Reasoning, Multimodality, Long Context, and Next Generation Agentic Capabilities](../../multimodal/29-gemini-2.5/summary.md)
- [OpenAI o1: Learning to Reason with Reinforcement Learning](../../language-models/31-openai-o1/summary.md)
- [rStar-Math: Small LLMs Can Master Math Reasoning with Self-Evolved Deep Thinking](../../techniques/35-rstar-math/summary.md)
- [RLVR: Reinforcement Learning from Verifiable Rewards](../../techniques/39-rlvr/summary.md)
- [GPT-5: Unified Intelligence](../../language-models/42-gpt5/summary.md)
- [Let's Verify Step by Step: Process Reward Models](../../techniques/51-process-reward-models/summary.md)
- [PagedAttention: Efficient LLM Serving with vLLM](../../techniques/52-pagedattention-vllm/summary.md)

<!-- related:end -->
