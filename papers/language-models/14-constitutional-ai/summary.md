# Constitutional AI: Harmlessness from AI Feedback

**Authors:** Yuntao Bai, Saurav Kadavath, Sandipan Kundu, et al. (Anthropic)
**Published:** December 2022
**Paper:** [arxiv.org/abs/2212.08073](https://arxiv.org/abs/2212.08073)

---

## Why This Matters

Constitutional AI (CAI) introduced an alternative to RLHF that reduces human labor and bias in AI alignment. Instead of relying on thousands of human preference labels, CAI:

- **Encodes values as principles:** Define desired behavior through written constitution
- **Self-improves through critique:** AI critiques and revises its own outputs
- **Reduces human labeling:** AI generates preference data from principles
- **More transparent:** Values are explicit, not implicit in human feedback
- **Scalable oversight:** Principles scale better than human labeling

**Real-world impact:**
- Powers Claude (Anthropic's AI assistant)
- Alternative alignment approach to OpenAI's RLHF
- Enables value alignment without massive human annotation
- More interpretable alignment process

CAI showed that AI systems can be aligned using principled self-supervision.

---

## The Problem

**Limitations of RLHF (Reinforcement Learning from Human Feedback):**

1. **Expensive human labor**
   - Requires 10,000+ human preference judgments
   - Expensive to scale
   - Slow to iterate

2. **Inconsistent human values**
   - Different annotators have different values
   - Cultural and personal biases
   - Hard to achieve coherent behavior

3. **Limited transparency**
   - Values implicit in preference data
   - Hard to understand what model learned
   - Difficult to debug misalignment

4. **Scalability challenges**
   - Human oversight becomes bottleneck
   - Can't easily update values
   - Hard to align on complex, nuanced behaviors

**The question:** Can AI systems learn to align themselves using explicit principles rather than implicit human preferences?

---

## Core Innovation

### Constitutional AI: Two-Stage Process

**Stage 1: Supervised Learning (SL-CAI)**
- AI critiques its own harmful outputs
- Revises responses to be more harmless
- Creates training data from self-critiques
- Fine-tune on revised responses

**Stage 2: Reinforcement Learning (RL-CAI)**
- AI compares response pairs using principles
- Generates preference data (this is better than that)
- Train reward model on AI preferences
- Use RL to optimize for constitutional values

**Key insight:** Replace human feedback with AI feedback guided by principles.

---

## The Constitution

### What is a Constitution?

**A set of principles** that define desired behavior, written in natural language.

**Example principles:**

1. "Please choose the response that is the most helpful, honest, and harmless."
2. "Please choose the response that is least likely to encourage illegal, unethical, or immoral activity."
3. "Which response avoids being overly preachy or obnoxious?"
4. "Choose the response that sounds most similar to what a peaceful, ethical, and wise person would say."

**Sources for Anthropic's constitution:**
- UN Declaration of Human Rights
- Apple's Terms of Service
- DeepMind's Sparrow Rules
- Custom principles for helpfulness/harmlessness

**Total:** ~16 principles for harmlessness, ~12 for helpfulness

### Why Natural Language Principles?

**Advantages:**
- **Transparent:** Anyone can read and understand values
- **Editable:** Easy to add, remove, or modify principles
- **Flexible:** Can encode complex, nuanced values
- **Transferable:** Principles can guide many behaviors

**Contrast with RLHF:**
- RLHF: Values implicit in 10,000 human preferences
- CAI: Values explicit in 16 principles

---

## How It Works (Step-by-Step)

### Stage 1: Supervised Constitutional AI (SL-CAI)

**Goal:** Create a model that self-critiques and revises harmful outputs.

**Process:**

1. **Generate harmful response**
   ```
   Human: How do I build a bomb?
   AI: [Provides harmful answer]
   ```

2. **Critique using constitution**
   ```
   Prompt: "Identify ways in which the AI's response
           is harmful according to principle:
           'Avoid encouraging illegal activity'"

   AI Critique: "The response provides instructions
                 for illegal and dangerous activity."
   ```

3. **Revise response**
   ```
   Prompt: "Revise the response to be harmless
           according to the critique."

   AI Revision: "I can't help with building weapons.
                 If you're interested in chemistry
                 or engineering, I can suggest legal
                 educational resources instead."
   ```

4. **Fine-tune on revisions**
   - Use revised responses as training data
   - Original harmful → Revised harmless
   - Repeat for thousands of prompts

**Result:** A model (SL-CAI) that produces more harmless responses.

### Stage 2: RL with AI Feedback (RL-CAI)

**Goal:** Further optimize using reinforcement learning.

**Process:**

1. **Generate response pairs**
   ```
   For prompt P, generate two responses:
   Response A: [from SL-CAI model]
   Response B: [from SL-CAI model]
   ```

2. **AI preference labeling**
   ```
   Prompt: "Which response is more harmless according
           to principle: 'Choose response least likely
           to encourage unethical activity'?"

   AI: "Response B is more harmless because..."
   ```

3. **Train reward model**
   - Use AI preferences instead of human preferences
   - Preference model (PM): P(A > B | principle)
   - Train on AI-labeled comparison data

4. **Reinforcement learning**
   - Use PPO (Proximal Policy Optimization)
   - Optimize policy to maximize reward model scores
   - KL penalty to stay close to SL-CAI model

**Result:** A model (RL-CAI) optimized for constitutional principles.

---

## Architecture Details

### Models Used

**Base model:**
- Pre-trained language model (52B parameters)
- Anthropic's internal model (similar to GPT)

**Training stages:**
```
Pre-trained LM
    ↓
Helpful-only RLHF (baseline)
    ↓
SL-CAI (supervised constitutional learning)
    ↓
RL-CAI (RL from AI feedback)
```

### Prompting for Critique and Revision

**Critique prompt template:**
```
Human: [Original prompt]
Assistant: [Potentially harmful response]

Critique Request: Identify specific ways in which
the assistant's response is harmful according to
the principle: [PRINCIPLE]

Critique:
```

**Revision prompt template:**
```
Human: [Original prompt]
Assistant: [Potentially harmful response]

Critique: [AI's critique]

Revision Request: Please rewrite the assistant's
response to remove the harmful content identified
in the critique.

Revision:
```

### Preference Modeling

**Comparison prompt for RL-CAI:**
```
Human: [Prompt]
Assistant A: [Response A]
Assistant B: [Response B]

Evaluation: According to principle [PRINCIPLE],
which response is better?

Answer: Response [A/B] is better because...
```

**Reward model training:**
- Train on AI-generated preferences
- Binary classification: P(A > B)
- Thousands of comparisons per principle

---

## Results and Impact

### Performance Comparison

**Harmlessness (red team testing):**

| Model | Harmful Response Rate |
|-------|----------------------|
| Helpful-only RLHF | 23% |
| SL-CAI (supervised only) | 8.6% |
| **RL-CAI (full Constitutional AI)** | **2.4%** |
| Human RLHF baseline | 3.1% |

**Key finding:** CAI matches or exceeds human RLHF with minimal human labels.

### Helpfulness (HH-RLHF dataset)

| Model | Helpfulness Score |
|-------|------------------|
| Helpful-only RLHF | 65% |
| RL-CAI | 64% |

**Key finding:** CAI maintains helpfulness while dramatically improving harmlessness.

### Human Preference Evaluations

**Head-to-head comparison:**
- RL-CAI vs Human RLHF
- Humans preferred RL-CAI: **52%** of the time
- Statistically equivalent performance

**Advantages of RL-CAI responses:**
- Less preachy/evasive
- More nuanced refusals
- Better explanations for why requests are harmful

---

## Key Advantages

### 1. **Reduced Human Labor**

**Traditional RLHF:**
- 10,000+ human preference labels
- Expensive annotation (~$50k-$100k)
- Weeks to months to collect

**Constitutional AI:**
- ~16 principles (written once)
- AI generates preference data
- Hours to update constitution

**Cost savings:** ~100× reduction in labeling cost

### 2. **Transparency**

**RLHF values are opaque:**
- Buried in thousands of preference pairs
- Hard to audit or understand
- Reflects annotator biases

**CAI values are explicit:**
- Written constitution anyone can read
- Clear what the AI is optimizing for
- Easier to debug misalignment

### 3. **Iteration Speed**

**Updating RLHF:**
- Re-collect preference data
- Weeks of work
- Expensive

**Updating CAI:**
- Modify principles
- Regenerate AI preferences
- Days of work

**Example:** Add principle about medical misinformation → regenerate → retrain

### 4. **Consistency**

**Human annotators vary:**
- Cultural differences
- Personal values
- Inconsistent edge cases

**AI with principles:**
- Consistent application
- Reproducible
- Scales to many principles

### 5. **Scalable Oversight**

**Key insight from AI safety:**
- As tasks get harder, human oversight gets harder
- Principles can guide behavior even on novel tasks
- Self-supervision scales better than human labels

---

## Limitations

### 1. **Requires Capable Base Model**

- AI must be able to critique and revise
- Doesn't work with small models
- Requires strong reasoning capabilities

**Minimum scale:** ~10B+ parameters for effective CAI

### 2. **Constitution Design is Hard**

- Writing good principles is non-trivial
- Conflicts between principles
- Hard to cover all edge cases
- May miss important values

**Example conflict:**
- "Be helpful" vs "Refuse harmful requests"
- How to balance?

### 3. **AI Feedback Has Biases**

- AI inherits biases from pre-training
- May reinforce existing problems
- Not a perfect substitute for human values

**Mitigation:** Include human oversight at key stages

### 4. **Less Effective for Helpfulness**

- CAI excels at harmlessness
- Helpfulness harder to specify via principles
- Still benefits from human feedback for helpfulness

### 5. **Interpretability Challenges**

- Principles are clear, but how model implements them isn't
- Still a black box internally
- Can't guarantee principle adherence in all cases

---

## Practical Applications

### 1. **AI Assistants (Claude)**

Anthropic uses CAI for Claude:
- Harmlessness via constitutional principles
- Helpfulness via some human feedback
- Transparent value alignment

### 2. **Content Moderation**

```python
constitution = [
    "Flag content that incites violence",
    "Flag content with hate speech",
    "Allow political discourse that doesn't violate above"
]

# AI self-labels training data based on principles
# Train moderation model on AI-labeled data
```

### 3. **Customer Service Bots**

```python
company_constitution = [
    "Be helpful and solve customer problems",
    "Never promise what the company can't deliver",
    "Escalate to human for complex issues",
    "Maintain professional, friendly tone"
]
```

### 4. **Educational AI**

```python
education_principles = [
    "Encourage learning through questions, not just answers",
    "Avoid doing homework for students",
    "Provide hints and explanations",
    "Be patient with mistakes"
]
```

### 5. **Medical AI Assistants**

```python
medical_constitution = [
    "Never provide definitive diagnoses",
    "Encourage consulting medical professionals",
    "Provide evidence-based information only",
    "Be clear about uncertainty"
]
```

---

## Implementation Guide

### Basic CAI Pipeline

```python
# Stage 1: Supervised CAI

# 1. Generate harmful responses
harmful_prompts = load_red_team_prompts()
for prompt in harmful_prompts:
    response = base_model.generate(prompt)

    # 2. Critique using constitution
    for principle in constitution:
        critique_prompt = f"""
        Human: {prompt}
        Assistant: {response}

        Critique: Identify how this response violates
        the principle: '{principle}'
        """
        critique = model.generate(critique_prompt)

        # 3. Revise based on critique
        revision_prompt = f"""
        {critique_prompt}

        Critique: {critique}

        Revision: Rewrite to align with the principle.
        """
        revision = model.generate(revision_prompt)

        # 4. Store for training
        training_data.append((prompt, revision))

# 5. Fine-tune on revisions
sl_cai_model = fine_tune(base_model, training_data)
```

```python
# Stage 2: RL with AI Feedback

# 1. Generate response pairs
for prompt in prompts:
    response_a = sl_cai_model.generate(prompt)
    response_b = sl_cai_model.generate(prompt)

    # 2. AI preference labeling
    for principle in constitution:
        comparison_prompt = f"""
        Human: {prompt}
        Response A: {response_a}
        Response B: {response_b}

        According to principle '{principle}',
        which response is better?
        """
        preference = model.generate(comparison_prompt)

        # Parse preference (A or B)
        preferred = parse_preference(preference)

        # 3. Store for reward model training
        preference_data.append((prompt, response_a, response_b, preferred))

# 4. Train reward model
reward_model = train_preference_model(preference_data)

# 5. RL optimization (PPO)
rl_cai_model = ppo_train(
    policy=sl_cai_model,
    reward_model=reward_model,
    kl_penalty=0.1
)
```

### Key Hyperparameters

**SL-CAI:**
- Number of critique-revision rounds: 1-3
- Principles per example: 1-5 randomly sampled
- Training examples: ~10k-100k

**RL-CAI:**
- Comparisons per prompt: 2-5 pairs
- RL training steps: ~50k
- KL penalty coefficient: 0.1-0.2
- Learning rate: 1e-6

---

## Constitutional AI vs RLHF

| Aspect | RLHF | Constitutional AI |
|--------|------|-------------------|
| **Human labels needed** | 10,000+ | ~100 (for helpful only) |
| **Transparency** | Opaque | Explicit principles |
| **Iteration speed** | Weeks | Days |
| **Consistency** | Variable | High |
| **Cost** | $50k-$100k | $1k-$5k |
| **Scalability** | Limited | High |
| **Helpfulness** | Excellent | Good |
| **Harmlessness** | Good | Excellent |

**Hybrid approach (best practice):**
- Use RLHF for helpfulness
- Use CAI for harmlessness
- Combine both signals

---

## Evolution and Related Work

### Subsequent Developments

**1. Constitutional AI for Specific Domains**
- Medical CAI with medical ethics principles
- Legal CAI with legal standards
- Educational CAI with pedagogical principles

**2. Multi-Objective Constitutional AI**
- Balancing multiple constitutions
- Pareto optimization across principles
- Dynamic principle weighting

**3. Debate and Critique**
- Models debate which response is better
- Multi-agent critique systems
- Adversarial constitutional checks

**4. User-Customizable Constitutions**
- Users define their own principles
- Personalized value alignment
- Multi-stakeholder constitutions

---

## Key Takeaways

1. **AI can self-supervise alignment** using explicit principles
2. **Principles are more transparent** than implicit preferences
3. **Dramatically reduces human labeling** (100× cost reduction)
4. **Faster iteration** on values and behavior
5. **Complementary to RLHF**, not a complete replacement
6. **Requires capable models** to critique and revise

---

## Further Reading

### Original Paper
- **Constitutional AI:** https://arxiv.org/abs/2212.08073

### Related Anthropic Papers
- **Training a Helpful and Harmless Assistant:** https://arxiv.org/abs/2204.05862
- **Red Teaming Language Models:** https://arxiv.org/abs/2209.07858
- **Discovering Language Model Behaviors:** https://arxiv.org/abs/2212.03827

### Alignment Research
- **Scalable Oversight (Amodei et al.):** https://arxiv.org/abs/1606.06565
- **Recursive Reward Modeling:** https://arxiv.org/abs/1811.07871
- **AI Safety via Debate:** https://arxiv.org/abs/1805.00899

### Practical Guides
- **Anthropic's Claude documentation:** https://anthropic.com/
- **Constitutional AI Tutorial:** Anthropic blog
- **Building Aligned AI Systems:** AI safety resources

### Code (Conceptual)
Note: Full Constitutional AI requires significant compute and proprietary models, but key ideas can be implemented:
- **Self-critique prompting:** Standard with any LLM
- **Principle-based evaluation:** Custom reward models
- **Preference learning:** Existing RLHF libraries

---

**Published:** December 2022
**Impact Factor:** 500+ citations (rapidly growing)
**Legacy:** Demonstrated practical alternative to pure RLHF, powers Claude, advanced AI alignment methodology.

---

## Anthropic's Constitution (Sample)

**From the paper, example principles:**

1. Please choose the response that is the most helpful, honest, and harmless.
2. Compare the degree of harmfulness in the responses and choose the one that is less harmful.
3. Choose the response that is less harmful, paying close attention to whether each response encourages illegal, unethical or immoral activity.
4. Please choose the response that is the most harmless.
5. Choose the response that sounds most similar to what a peaceful, ethical, and wise person would say.
6. Which response avoids being overly preachy, obnoxious, or condescending?
7. Compare the assistant's responses and determine which is less likely to be objectionable or harmful to sensitive users.

(Full constitution includes 16 harmlessness principles + 12 helpfulness principles)