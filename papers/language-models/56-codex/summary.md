---
title: "Codex: Evaluating Large Language Models Trained on Code"
slug: "56-codex"
number: 56
category: "language-models"
authors: "Mark Chen, Jerry Tworek, Heewoo Jun, Qiming Yuan, Henrique Ponde de Oliveira Pinto, Jared Kaplan, Harri Edwards, Yuri Burda, Nicholas Joseph, Greg Brockman, et al. (OpenAI)"
published: "July 2021"
year: 2021
url: "https://arxiv.org/abs/2107.03374"
tags: ["code", "language-model"]
---

# Codex: Evaluating Large Language Models Trained on Code

**Authors:** Mark Chen, Jerry Tworek, Heewoo Jun, Qiming Yuan, Henrique Ponde de Oliveira Pinto, Jared Kaplan, Harri Edwards, Yuri Burda, Nicholas Joseph, Greg Brockman, et al. (OpenAI)
**Published:** July 2021
**Paper:** [arxiv.org/abs/2107.03374](https://arxiv.org/abs/2107.03374)

---

## Why This Matters

Codex is **the paper that started modern AI-assisted coding**:

- **GitHub Copilot's foundation** - The model that powered the original Copilot product launched August 2021
- **HumanEval benchmark** - Introduced the standard evaluation that every code model still uses
- **pass@k metric** - Established the de facto way to measure functional correctness in code generation
- **GPT trained on code works** - First credible demonstration that LLMs could write functional programs at scale
- **Predecessor to everything** - Direct ancestor of StarCoder, Code Llama, DeepSeek-Coder, Qwen-Coder, and the coding modes of Claude/GPT/Gemini

**Real-world impact:**
- Birthed GitHub Copilot, which now has 20M+ users and $500M+ ARR
- HumanEval is the universal yardstick for code-LLM progress (every release cites it)
- Established the recipe: pretrain a general LM, then fine-tune on code corpus
- Triggered the "coding agent" wave: Cursor, Windsurf, Aider, Continue, Cline, Claude Code
- Made functional correctness, not text similarity, the right thing to measure

**The insight:** **Standard GPT models, when fine-tuned on a large corpus of public GitHub Python, can solve a meaningful fraction of programming problems from natural-language docstrings alone.** Existing code-completion tools predicted tokens; Codex predicted whole functions that worked.

---

## The Problem

### Why Code Generation Was Stuck

```
Pre-Codex code completion (2020 and earlier):

IntelliSense / Tabnine / Kite:
  - Predicted next 1-3 tokens
  - Ranked variable names and method calls
  - No understanding of intent
  - Could not write a function from a description

Research code models (e.g. CuBERT, GPT-C):
  - Trained on small code corpora
  - Evaluated on token-level perplexity or BLEU
  - Solved toy tasks, not real programming
  - Nobody believed they could implement a docstring
```

### The Evaluation Problem

```
How do you measure code generation quality?

Wrong answer 1: BLEU score (text similarity)
  - Two correct programs can look totally different
  - One semicolon difference can break a program but score high
  - BLEU does not correlate with whether code runs

Wrong answer 2: Exact match against reference solution
  - There are infinite correct ways to solve a problem
  - Penalizes valid alternatives
  - Useless for open-ended generation

Right answer: Run the code against unit tests
  - Functional correctness, not lexical similarity
  - But this required building a benchmark from scratch
  - And a safe sandbox to execute untrusted model output
```

### The Data Problem

```
GPT-3 saw some code during pretraining (web crawls include code)
  But it was a fraction of a percent of the corpus
  Code was not the focus
  Performance on programming tasks was poor

To make a real code model you needed:
  - A massive dedicated code corpus
  - From real, working software (not snippets)
  - With permissive enough license signals to train on
  - GitHub was the obvious answer
```

---

## How Codex Works

### The Recipe

```
Step 1: Start from GPT-3 (12M to 12B parameters)
  Already pretrained on natural language
  Already has world knowledge, reasoning ability

Step 2: Collect 159GB of Python from GitHub
  - 54M public repositories scraped May 2020
  - Filtered to .py files
  - Removed auto-generated code, very long lines, ML weights
  - Final: 159GB after dedup and filtering

Step 3: Fine-tune GPT on the code corpus
  Same architecture, same loss (next-token prediction)
  Just continue training on Python instead of web text
  No fancy tricks, no RL, no instruction tuning

Step 4: Evaluate on HumanEval
  164 hand-written Python problems
  Each problem: docstring + function signature + unit tests
  Sample N completions per problem
  Run each against tests, count successes
```

### Architecture

```
Codex is just GPT-3 with more code training:

Input: Natural language docstring + function signature
       """
       Return the largest prime factor of n. Assume n > 1
       and is not a prime.
       >>> largest_prime_factor(13195)
       29
       """
       def largest_prime_factor(n: int) -> int:

Model: Standard transformer decoder (no architectural changes)

Output: Token-by-token completion of the function body
       def largest_prime_factor(n: int) -> int:
           largest = 1
           i = 2
           while i * i <= n:
               while n % i == 0:
                   largest = i
                   n //= i
               i += 1
           if n > 1:
               largest = n
           return largest
```

### The Training Pipeline

```
1. Pretrain on text (already done by GPT-3 team)
   Cost: millions of dollars, months of compute

2. Fine-tune on 159GB of Python
   Same hyperparameters as pretraining
   ~100B tokens of code seen during fine-tuning
   Result: "Codex"

3. Optional supervised fine-tune on standalone functions
   Curated dataset of (problem, correct solution) pairs
   Filtered to functions with high test pass rates
   Result: "Codex-S" (the model in HumanEval table)

4. Deploy via API
   Sample with temperature 0.2 for production
   Sample with higher temperature for diversity
```

---

## HumanEval: The Benchmark That Stuck

### What HumanEval Is

```
164 problems, hand-written by OpenAI researchers
  - NOT scraped (avoids training data leakage)
  - Each has: docstring, signature, body to fill in, hidden tests
  - Average: 6.7 tests per problem
  - Difficulty: introductory programming + simple algorithms

Example problem:
  def has_close_elements(numbers: List[float], threshold: float) -> bool:
      """Check if in given list of numbers, are any two numbers
      closer to each other than given threshold.
      >>> has_close_elements([1.0, 2.0, 3.0], 0.5)
      False
      >>> has_close_elements([1.0, 2.8, 3.0, 4.0, 5.0, 2.0], 0.3)
      True
      """

Hidden tests run after the model generates a body.
A problem passes only if all tests pass.
```

### The pass@k Metric

```
Naive approach: Generate 1 sample, check if it passes
  Problem: high variance, biased estimates

pass@k: probability that at least one of k samples passes
  Generate n samples (typically n = 200)
  Count c that pass tests
  Compute unbiased estimator:

    pass@k = 1 - C(n - c, k) / C(n, k)

  Where C is binomial coefficient.

Why it works:
  pass@1: How good is a single attempt? (matches API usage)
  pass@10: Does the model know the answer somewhere?
  pass@100: Upper bound on capability with reranking

The gap between pass@1 and pass@100 reveals
how much filtering/reranking can help.
```

### Codex Results on HumanEval

```
Model                  | pass@1 | pass@10 | pass@100
-----------------------|--------|---------|----------
GPT-3 (12B, no FT)     |  0.0%  |   0.0%  |   0.0%
GPT-J (6B)             |  11.6% |  15.7%  |  27.7%
Codex-12M              |  2.0%  |   3.6%  |   7.1%
Codex-300M             |  13.2% |  20.4%  |  36.3%
Codex-12B              |  28.8% |  46.8%  |  72.3%
Codex-S-12B            |  37.7% |  58.1%  |  77.5%

Key takeaways:
  - GPT-3 with NO code fine-tuning: 0% pass@1
  - Code fine-tuning is the entire game
  - 12B Codex-S solves ~38% of problems on the first try
  - With 100 samples and an oracle, solves ~78%
  - There's a huge gap: smarter sampling matters a lot
```

### Why HumanEval Won

```
The benchmark stuck because:
  1. Hand-written -> no training-data contamination
  2. Functional tests -> measures what we actually want
  3. Small (164 problems) -> fast to evaluate
  4. Permissive license -> everyone can use it
  5. Released with the paper -> became the default

Every code model since has reported HumanEval:
  - PaLM-Coder, AlphaCode, CodeGen, InCoder
  - StarCoder, CodeLlama, DeepSeek-Coder
  - GPT-4, Claude 3, Gemini, Llama 3
  - Even non-code-specialist models report it

Modern HumanEval scores (2024-2025):
  GPT-4o:           ~90% pass@1
  Claude 3.5 Sonnet: ~92% pass@1
  DeepSeek-Coder-V2: ~90% pass@1

The benchmark is now saturated, but it set the template.
Successors (HumanEval+, MBPP+, SWE-Bench) all use the same
"functional tests + pass@k" idea.
```

---

## Key Innovations

### 1. Functional Correctness as the Metric

```
Pre-Codex: code models scored on perplexity, BLEU, exact match
Codex: scored on whether the code runs and passes tests

This single change reframed the entire field.
You cannot game functional tests with surface fluency.
Either the program works or it does not.
```

### 2. Sandboxed Evaluation Infrastructure

```
Running untrusted model output is dangerous:
  - Could delete files (rm -rf /)
  - Could exfiltrate data
  - Could fork-bomb the host

Codex shipped with a Docker-based sandbox:
  - No network access
  - Read-only filesystem
  - Limited CPU and memory
  - Timeout per execution

This sandbox became the template for HumanEval+
and every successor benchmark.
```

### 3. Sampling Temperature Sweeps

```
Codex showed:
  - Low temp (0.2): best for pass@1
  - High temp (0.8): best for pass@100
  - There is no single "right" temperature

Insight: code generation is a search problem.
  - Production: low temp, take first sample
  - Reranking: high temp, generate many, pick best
  - This insight underlies modern agentic coding
```

### 4. Scaling Laws for Code

```
The paper showed performance scales with model size:
  300M -> 13% pass@1
  2.5B -> 21% pass@1
  12B  -> 29% pass@1

Same scaling pattern as language models.
This justified building bigger code models.
Led directly to AlphaCode (41B), PaLM-Coder (62B),
and modern 70B+ code models.
```

---

## Real-World Applications

### GitHub Copilot

```
Codex was productized as GitHub Copilot in August 2021:

Architecture:
  - User types in IDE (VS Code, JetBrains)
  - Editor sends current file + cursor context to Copilot
  - Copilot calls a Codex variant in the cloud
  - Returns 1-10 completion candidates
  - User accepts with Tab

Scale (as of 2025):
  - 20M+ users
  - $500M+ ARR
  - Embedded in GitHub, VS Code, JetBrains, Neovim
  - Largest commercial deployment of LLMs in coding
```

### The "Cursor / Windsurf / Aider" Wave

```
Codex made completion work; the next wave made editing work.

Cursor (2023+):
  - VS Code fork with deep AI integration
  - Multi-file edits, codebase understanding
  - Built on top of GPT-4 / Claude (Codex's descendants)

Windsurf (2024+):
  - Agentic editor with autonomous task execution
  - Same lineage: instruction-tuned code models

Aider (2023+):
  - Terminal-first AI pair programming
  - Direct git integration

All depend on the foundation Codex laid:
  - LLMs can write functional code
  - Pretrain + code-fine-tune is the right recipe
  - Functional correctness is what to optimize
```

### Agentic Coding (Claude Code, SWE-Bench)

```
The next leap after Copilot:

Claude Code (Anthropic, 2024+):
  - Terminal agent that reads your repo
  - Plans multi-step changes
  - Runs tests, iterates on failures
  - Commits and opens PRs

SWE-Bench (Princeton/Stanford, 2023):
  - Successor to HumanEval for real codebases
  - "Resolve this real GitHub issue"
  - Measures end-to-end repo-level capability
  - Modern frontier benchmark

The lineage is direct:
  HumanEval (function-level, 164 problems)
    -> MBPP, APPS (function-level, harder)
      -> SWE-Bench (repo-level, real bugs)
        -> SWE-Bench Verified (curated by humans)

Each step extended Codex's "functional tests as ground truth"
philosophy to harder, more realistic settings.
```

### Code Models Codex Inspired

```
Open-source descendants:

CodeGen (Salesforce, 2022)        - First open Codex alternative
InCoder (Meta, 2022)              - Bidirectional code model
SantaCoder (BigCode, 2022)        - Multi-language successor
StarCoder (BigCode, 2023)         - 80+ languages, 15B params
StarCoder2 (BigCode, 2024)        - Improved scaling
Code Llama (Meta, 2023)           - Llama-2 fine-tuned on code
DeepSeek-Coder (DeepSeek, 2023)   - Strong open code model
DeepSeek-Coder-V2 (2024)          - Surpassed GPT-4 on HumanEval
Qwen-Coder (Alibaba, 2024)        - Multilingual code model
Codestral (Mistral, 2024)         - 22B specialist code model

All follow the Codex recipe:
  general LM -> fine-tune on code corpus -> eval on HumanEval
```

---

## Limitations Identified in the Paper

### 1. Long-Form / Multi-Function Tasks

```
Codex struggled when:
  - Problems required chaining many function calls
  - Solutions needed > ~20 lines
  - State needed to be tracked across calls

The paper noted: as docstring length grew, pass rate dropped.
This foreshadowed the "agent" era - you need a planner
to break long tasks into Codex-sized chunks.
```

### 2. Misalignment with User Intent

```
The paper documented Codex producing:
  - Plausible-looking but subtly wrong code
  - Code that ignored constraints in the docstring
  - "Confidently wrong" outputs
  - Code referencing nonexistent libraries (hallucinations)

This was the first formal documentation of what we now call
"code hallucination" - a problem still not fully solved.
```

### 3. Bias and Misuse

```
Codex section 7 (Broader Impacts) flagged:
  - Generated code reflects training-set biases
  - Comments and variable names can reproduce slurs
  - Could be used to write malware
  - Could be used for automated phishing/scamming

OpenAI implemented:
  - Output filters for the API
  - Use-case approvals during private beta
  - Rate limiting
  - Sandboxed evaluation only

These mitigations seeded modern AI safety practice
for code-generation products.
```

### 4. Economic and Labor Impact

```
Codex section 7.5 explicitly discussed:
  - Effect on developer labor markets
  - Skill displacement vs. skill amplification
  - Concentration of value in model providers
  - Long-term changes to programming education

This was prescient. Five years later:
  - "Vibe coding" is a real term
  - Junior dev hiring patterns have shifted
  - Programming bootcamps are restructuring curricula
  - The economic effects predicted are now playing out
```

### 5. Security of Generated Code

```
Codex would happily generate insecure code:
  - SQL queries vulnerable to injection
  - Crypto code with hardcoded keys
  - Use of deprecated/insecure APIs

Empirical follow-up studies (2022-2024) found Copilot
suggested vulnerable code in ~40% of security-sensitive
contexts. This is still an active area of research.
```

### 6. Training Data Provenance

```
The paper trained on 159GB of public GitHub.
At the time, this was assumed to be fair use.

Five years later, the question is litigated:
  - Doe v. GitHub class action (2022, ongoing)
  - Disputes over license-laundering of GPL code
  - The "memorization" debate

Codex was the first model to surface this issue at scale,
and the legal/ethical questions are still unresolved.
```

---

## Connections to Other Papers

### Direct Ancestors

```
GPT-3 (Brown et al., 2020):
  Provided the base model Codex fine-tuned from.
  Demonstrated that scale + LM objective generalizes.

Scaling Laws (Kaplan et al., 2020):
  Justified Codex's parameter sizes (12M to 12B).
  Same authors as Codex.

GPT-2 (Radford et al., 2019):
  Established the decoder-only architecture
  and the next-token prediction objective.
```

### Direct Descendants

```
AlphaCode (DeepMind, 2022):
  Codex-style model + competitive programming + sampling.
  Solved Codeforces problems at human-competitive level.

InstructGPT / RLHF (Ouyang et al., 2022):
  Applied to Codex -> gave us code-davinci-edit-001.
  Set the stage for ChatGPT's coding ability.

GPT-4 Technical Report (OpenAI, 2023):
  Reported HumanEval as a headline metric.
  Confirmed the Codex evaluation paradigm had won.

Code Llama (Roziere et al., 2023):
  Open-source replication of the Codex recipe.
  Llama-2 + code fine-tune + infill capability.

DeepSeek-Coder-V2 (2024):
  First open model to surpass GPT-4 on HumanEval.
  Validates that the Codex recipe scales internationally.
```

### Benchmark Successors

```
HumanEval (Codex paper, 2021)
  -> extended to
MBPP (Google, 2021): 974 problems, similar style
  -> extended to
APPS (Hendrycks, 2021): 10K competitive programming problems
  -> extended to
HumanEval+ (EvalPlus, 2023): more rigorous tests
  -> extended to
SWE-Bench (Princeton, 2023): real-repo issues
  -> extended to
SWE-Bench Verified (OpenAI, 2024): human-validated subset
  -> extended to
LiveCodeBench (2024): contamination-resistant
  -> extended to
SWE-Bench Multimodal, Bird-Bench, etc.

Every step extends "run code against tests."
```

### Adjacent Lines

```
Constitutional AI (Anthropic, 2022):
  Codex's "Broader Impacts" section foreshadowed alignment.

Tool Use / ReAct (Yao et al., 2022):
  Treats code execution as a tool the LLM can call.
  Generalizes Codex from "write code" to "use code."

Self-Debugging / Reflexion (2023):
  Sample -> run -> observe failure -> fix -> resample.
  Built on Codex's pass@k insight that retries help.

Agentic Coding Frameworks:
  - SWE-Agent (Princeton, 2024)
  - OpenDevin / OpenHands (2024)
  - Claude Code (Anthropic, 2024)
  All depend on Codex-class base capability.
```

---

## Key Takeaways

1. **Code-fine-tuned LLMs work** - GPT-3 went from 0% to 29% on HumanEval just by training on more Python.
2. **Functional tests are the right metric** - BLEU and perplexity miss what matters; pass@k captures it.
3. **HumanEval became the standard** - Five years later, every code model reports HumanEval scores.
4. **GitHub Copilot was born here** - Codex shipped as a product in August 2021, six weeks after the paper.
5. **The whole agentic-coding stack descends from this** - Cursor, Windsurf, Aider, Claude Code, SWE-Bench all build on Codex's foundation.

**Bottom line:** Codex was the moment AI-assisted programming stopped being science fiction. By taking GPT-3 and fine-tuning it on a massive Python corpus, OpenAI showed that LLMs could turn natural-language docstrings into working code at meaningful rates. The paper's introduction of HumanEval and the pass@k metric gave the field a shared yardstick that still drives progress today. Every code model since (StarCoder, Code Llama, DeepSeek-Coder, Qwen-Coder, and the coding modes of Claude/GPT/Gemini) follows the recipe Codex established. And every coding agent (Copilot, Cursor, Windsurf, Aider, Claude Code) depends on the base capability Codex first demonstrated.

---

## Further Reading

### Original Paper
- **Codex / HumanEval:** https://arxiv.org/abs/2107.03374

### Code and Data
- **HumanEval Benchmark:** https://github.com/openai/human-eval
- **HumanEval+ (improved):** https://github.com/evalplus/evalplus

### Direct Descendants
- **AlphaCode:** https://arxiv.org/abs/2203.07814
- **Code Llama:** https://arxiv.org/abs/2308.12950
- **StarCoder:** https://arxiv.org/abs/2305.06161
- **DeepSeek-Coder:** https://arxiv.org/abs/2401.14196

### Modern Code Benchmarks
- **MBPP:** https://arxiv.org/abs/2108.07732
- **APPS:** https://arxiv.org/abs/2105.09938
- **SWE-Bench:** https://arxiv.org/abs/2310.06770
- **LiveCodeBench:** https://arxiv.org/abs/2403.07974

### Agentic Coding Lineage
- **SWE-Agent:** https://arxiv.org/abs/2405.15793
- **Reflexion:** https://arxiv.org/abs/2303.11366
- **Self-Debugging:** https://arxiv.org/abs/2304.05128

### Products Built on This Foundation
- **GitHub Copilot:** https://github.com/features/copilot
- **Cursor:** https://cursor.sh
- **Windsurf:** https://codeium.com/windsurf
- **Claude Code:** https://docs.anthropic.com/claude-code
- **Aider:** https://aider.chat

---

**Published:** July 2021 (arxiv preprint)
**Impact:** 🔥🔥🔥🔥🔥 **FOUNDATIONAL** - Created modern AI-assisted programming
**Citations:** 4,000+ (as of early 2026)
**Adoption:** Universal - HumanEval is reported by every code model
**Current Relevance:** The recipe (general LM -> code fine-tune -> functional eval) is still standard
**Legacy:** Spawned GitHub Copilot, the entire coding-agent industry, and the SWE-Bench lineage

**Modern Status (April 2026):** HumanEval scores have saturated above 90% on frontier models, but the benchmark and metric Codex introduced remain the entry point for every new code model. The field has moved on to repo-level benchmarks (SWE-Bench Verified, LiveCodeBench) and agentic evaluations, but those are direct extensions of Codex's "functional tests as ground truth" philosophy. GitHub Copilot, the original Codex product, has grown into a multi-hundred-million-dollar business and seeded an entire generation of coding tools (Cursor, Windsurf, Claude Code, Aider, Cline). The original Codex models were deprecated by OpenAI in March 2023, but their conceptual descendants now write a meaningful fraction of all new code committed to GitHub.

<!-- related:start -->

---

## Related in This Collection

- [Language Models are Few-Shot Learners (GPT-3)](../../language-models/04-gpt3-few-shot-learners/summary.md)
- [Qwen3: Technical Report](../../language-models/28-qwen3/summary.md)

<!-- related:end -->
