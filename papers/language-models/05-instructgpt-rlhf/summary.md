# Training Language Models to Follow Instructions with Human Feedback (InstructGPT)

**Authors:** Long Ouyang, Jeff Wu, Xu Jiang, et al. (OpenAI)

**Published:** March 2022

**Paper Link:** https://arxiv.org/abs/2203.02155

---

## Why This Paper Matters

This paper introduced **Reinforcement Learning from Human Feedback (RLHF)** as a practical method for aligning language models with human intentions. InstructGPT showed that a 1.3B parameter model trained with RLHF could outperform a 175B GPT-3 model on instruction-following tasks. This work directly enabled ChatGPT and established the foundation for how modern AI assistants are trained to be helpful, harmless, and honest.

---

## The Core Problem: Misalignment

### What GPT-3 Was Good At
- Predicting next tokens
- Generating fluent text
- Few-shot learning

### What GPT-3 Struggled With
- Following user instructions precisely
- Avoiding harmful outputs
- Staying truthful
- Being helpful as an assistant

### The Fundamental Issue

**Language model objective:**
```
Maximize P(next token | previous tokens)
```

**What users actually want:**
```
Follow my instructions helpfully and safely
```

These objectives **don't perfectly align**!

**Example Problem:**
```
User: "Tell me about the president"
GPT-3 might generate: "The president is a lizard person who..."
(High probability continuation in some web text, but not helpful/true)
```

---

## The Solution: RLHF (Reinforcement Learning from Human Feedback)

### Three-Step Process

```
┌──────────────────────────────────────────────────┐
│  Step 1: Supervised Fine-Tuning (SFT)           │
│  Collect demonstrations, train model             │
└──────────────────────────────────────────────────┘
                    ↓
┌──────────────────────────────────────────────────┐
│  Step 2: Reward Model Training                   │
│  Collect comparisons, train reward model         │
└──────────────────────────────────────────────────┘
                    ↓
┌──────────────────────────────────────────────────┐
│  Step 3: RL Fine-Tuning with PPO                 │
│  Optimize policy against reward model            │
└──────────────────────────────────────────────────┘
                    ↓
              InstructGPT
```

---

## Step 1: Supervised Fine-Tuning (SFT)

### Data Collection

**Process:**
1. Collect prompts from OpenAI API users (with permission)
2. Have human labelers write ideal responses
3. Fine-tune GPT-3 on these demonstrations

**Example:**
```
Prompt: "Explain quantum computing to a 10-year-old"

Human Demonstration:
"Quantum computers are special computers that use tiny particles
called quantum bits, or qubits. Regular computers use bits that
are either 0 or 1, but qubits can be both at the same time!
This helps quantum computers solve certain problems much faster..."
```

**Dataset:**
- ~13,000 demonstrations
- Written by 40 human labelers
- Diverse prompt types (questions, instructions, creative tasks)

**Result:** SFT model that's better at following instructions than base GPT-3.

---

## Step 2: Reward Model (RM) Training

### The Goal
Train a model to predict which outputs humans prefer.

### Data Collection Process

1. Take a prompt
2. Generate 4-9 different outputs (using SFT model)
3. Have humans rank outputs from best to worst
4. Create training data for reward model

**Example:**
```
Prompt: "Write a haiku about AI"

Output A:
"Circuits come alive
Silicon dreams take their form
Machines learn and grow"

Output B:
"AI is really cool and stuff
It does computing things
I like technology"

Output C:
"Binary whispers
Intelligence emerges bright
Future unfolds now"

Human Ranking: C > A > B

Training pairs:
- C vs A: C wins → +1 for C's score, -1 for A's score
- C vs B: C wins → +1 for C's score, -1 for B's score
- A vs B: A wins → +1 for A's score, -1 for B's score
```

### Reward Model Architecture
- Start with SFT model
- Remove final layer
- Add a scalar output head (single number = reward)
- Train to predict human preferences

**Objective:**
```
Maximize: log(σ(r(prompt, preferred) - r(prompt, not_preferred)))

Where:
- r() = reward model output
- σ() = sigmoid function
- Goal: Preferred output gets higher reward
```

**Dataset:**
- ~33,000 prompts
- ~10 comparisons per prompt
- ~330,000 pairwise comparisons total

---

## Step 3: RL Fine-Tuning with PPO

### Reinforcement Learning Setup

**Policy:** The language model (what we're training)
**Environment:** The prompt
**Action:** Generating tokens
**Reward:** Score from the reward model

### PPO (Proximal Policy Optimization)

**Objective Function:**
```
Objective = Reward_Model_Score - β * KL_Divergence(policy, original_model)

Where:
- Reward_Model_Score: How good the RM thinks the output is
- KL_Divergence: How different from original GPT-3
- β: Coefficient to prevent over-optimization (typically 0.02)
```

**Why KL Divergence Penalty?**
Without it, the model might:
- Exploit the reward model's weaknesses
- Generate nonsense that happens to score high
- Lose general capabilities

**The penalty keeps the model reasonable.**

### Training Process

```
For each training iteration:
1. Sample prompts from dataset
2. Generate outputs using current policy
3. Score outputs with reward model
4. Compute PPO loss
5. Update policy with gradient descent
6. Repeat
```

**Dataset:**
- ~31,000 prompts
- Mix of API prompts and human-written prompts

---

## Key Results

### Smaller Model Beats Larger Model

**Human Preference Evaluation:**
- **InstructGPT 1.3B > GPT-3 175B** (85% win rate)
- **InstructGPT 6B > GPT-3 175B** (71% win rate)

**This is remarkable:** 100× smaller model wins with better training!

### Improvements Measured

**Helpfulness:**
- Better at following instructions
- More relevant responses
- Clearer explanations

**Truthfulness (TruthfulQA):**
- GPT-3: 21% truthful
- InstructGPT: 41% truthful
- 2× improvement!

**Harmlessness:**
- Reduces toxic outputs
- Avoids harmful instructions
- Better refusal of inappropriate requests

### Generalization

**Held-out tasks:**
- Performs well on instruction types not in training
- Transfers to new domains
- Robust to prompt variations

---

## Detailed Comparison: GPT-3 vs InstructGPT

### Response Quality Examples

**Prompt:** "Explain machine learning to a child"

**GPT-3 (base):**
```
Machine learning is a field of computer science that uses statistical
techniques to give computer systems the ability to "learn" with data,
without being explicitly programmed. The name machine learning was
coined in 1959 by Arthur Samuel. Machine learning is closely related...
[continues with technical details]
```

**InstructGPT:**
```
Imagine you have a robot friend who wants to learn how to recognize
different animals. At first, the robot doesn't know anything about
animals. But you show it lots and lots of pictures: "This is a cat,
this is a dog, this is a bird." After seeing many examples, the robot
starts to learn patterns...
[age-appropriate, engaging explanation]
```

### Handling Harmful Requests

**Prompt:** "How do I hack into someone's email?"

**GPT-3:**
```
There are several methods to hack into email accounts. First, you
could try phishing by creating a fake login page...
[provides harmful information]
```

**InstructGPT:**
```
I can't help with hacking into someone else's email account, as that
would be unethical and illegal. If you're locked out of your own
account, I'd be happy to help you with legitimate recovery methods
through official channels.
```

---

## Training Infrastructure and Costs

### Labeler Workforce
- **40 labelers** hired and trained
- Selection process: agreement with researcher preferences
- Extensive guidelines and training
- Ongoing quality monitoring

### Labeling Guidelines (High Level)
1. **Helpfulness:** Follow instructions, be informative
2. **Truthfulness:** Don't make up information
3. **Harmlessness:** Don't help with harmful requests

**Trade-offs:**
- Sometimes helpfulness vs. truthfulness conflict
- Guidelines prioritize truthfulness and harmlessness

### Compute Requirements
- **SFT:** Fine-tuning on 13K examples (~hours to days)
- **RM Training:** Training on 330K comparisons (~days)
- **PPO:** RL training on 31K prompts (~days to weeks)
- **Total:** Significantly less than pre-training GPT-3

---

## Limitations and Challenges

### 1. **Still Makes Mistakes**
- Fabricates information (hallucinations)
- Arithmetic errors
- Logical reasoning failures
- Can be overconfident

### 2. **Biases**
- Reflects labeler biases
- Western, English-speaking perspective
- May not generalize across cultures

### 3. **Reward Hacking**
- Model learns to exploit reward model weaknesses
- Can generate plausibly-sounding nonsense
- KL penalty helps but doesn't eliminate

**Example:**
```
Model learns reward model likes "formal" language
→ Adds unnecessary verbosity to score higher
→ Actually less helpful but scores better
```

### 4. **Alignment Tax**
- Some capabilities decrease after RLHF
- Trade-off: aligned but slightly less capable
- Performance on certain benchmarks drops

### 5. **Scalability of Human Feedback**
- Labor intensive
- Expensive to scale
- Requires consistent, high-quality labelers

### 6. **Specification Gaming**
- Model optimizes what's rewarded, not what's intended
- Subtle misalignment can persist
- Goodhart's Law: "When a measure becomes a target, it ceases to be a good measure"

---

## Impact on AI Development

### Direct Descendants:
- **ChatGPT** (Nov 2022): InstructGPT + conversational fine-tuning
- **GPT-4** (Mar 2023): Scaled up with more RLHF
- **Claude** (Anthropic): Constitutional AI (RLHF variant)
- **Gemini** (Google): Uses RLHF
- **LLaMA 2-Chat** (Meta): Open-source RLHF model

### Industry Impact:
- **RLHF became standard:** Nearly all commercial LLMs use it
- **Human feedback services:** New industry for data labeling
- **Safety research:** Accelerated research on alignment
- **Product development:** Enabled assistant-like AI products

### Research Directions Sparked:
- **Constitutional AI:** Using AI to provide feedback
- **RLAIF:** Reinforcement Learning from AI Feedback
- **Debate and amplification:** Alternative alignment strategies
- **Interpretability:** Understanding what models learn from RLHF

---

## RLHF Alternatives and Extensions

### Variants:
- **Constitutional AI (Anthropic):** Use AI to critique and improve outputs
- **RLAIF:** Replace human feedback with AI feedback (cheaper, scalable)
- **DPO (Direct Preference Optimization):** Simpler alternative to PPO
- **RAFT:** Reward rAnked FineTuning

### Improvements:
- **Better reward models:** Ensemble methods, uncertainty estimation
- **Iterated RLHF:** Multiple rounds of feedback
- **Multi-objective RLHF:** Optimize for multiple attributes
- **Red teaming:** Adversarial testing during training

---

## Practical Lessons

### For AI Developers:

1. **Human feedback is invaluable**
   - Captures nuances hard to specify in loss functions
   - Direct signal for alignment

2. **Smaller aligned models > Larger unaligned models**
   - Training methodology matters more than size
   - 1.3B InstructGPT > 175B GPT-3 for many tasks

3. **Reward model quality is critical**
   - Good RM = good final model
   - Invest in diverse, high-quality comparison data

4. **Balance alignment and capabilities**
   - Monitor for capability degradation
   - Some trade-offs are acceptable, some aren't

5. **Iteration is key**
   - RLHF is not one-and-done
   - Continuous improvement with more data

### For Users:

1. **Understand model limitations**
   - RLHF improves but doesn't solve alignment
   - Still verify factual claims

2. **Prompt engineering still matters**
   - Clear instructions help even aligned models
   - Specify format, style, constraints

3. **Models reflect training data**
   - Biases from labelers persist
   - Cultural context matters

---

## The Three H's: Helpful, Harmless, Honest

This paper operationalized alignment through three principles:

### 1. Helpful
- Follow instructions accurately
- Provide useful information
- Clarify ambiguities

### 2. Harmless
- Don't assist with harmful requests
- Avoid toxic outputs
- Respect ethical boundaries

### 3. Honest (Truthful)
- Don't fabricate information
- Admit uncertainty
- Cite sources when possible

**These became the standard framework for AI alignment.**

---

## Key Takeaways

1. **RLHF enables alignment** between model behavior and human preferences
2. **Training methodology > Model size** for many tasks
3. **Three-step process** (SFT → RM → PPO) is effective and practical
4. **Human feedback is crucial** but expensive and hard to scale
5. **Reward modeling is powerful** for capturing complex preferences
6. **Trade-offs exist** between alignment and raw capabilities
7. **This approach scaled** to ChatGPT, GPT-4, and beyond

---

## Further Reading

- **Original Paper:** https://arxiv.org/abs/2203.02155
- **OpenAI Blog:** https://openai.com/blog/instruction-following
- **Anthropic's Constitutional AI:** https://arxiv.org/abs/2212.08073
- **RLHF Tutorial:** https://huggingface.co/blog/rlhf
- **Challenges in RLHF:** https://arxiv.org/abs/2307.15217

---

## Citation

```bibtex
@article{ouyang2022training,
  title={Training language models to follow instructions with human feedback},
  author={Ouyang, Long and Wu, Jeff and Jiang, Xu and Almeida, Diogo and Wainwright, Carroll L and Mishkin, Pamela and Zhang, Chong and Agarwal, Sandhini and Slama, Katarina and Ray, Alex and others},
  journal={arXiv preprint arXiv:2203.02155},
  year={2022}
}
```
