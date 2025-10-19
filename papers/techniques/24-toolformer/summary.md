# Toolformer: Language Models Can Teach Themselves to Use Tools

**Authors:** Timo Schick, Jane Dwivedi-Yu, et al. (Meta AI Research)
**Published:** February 2023
**Paper:** [arxiv.org/abs/2302.04761](https://arxiv.org/abs/2302.04761)

---

## Why This Matters

Toolformer showed that **LLMs can learn to use tools without human annotations**:

- 🛠️ **Self-taught tool use** - No manual labeling needed
- 🎯 **Zero-shot API calling** - Learns when and how to call tools
- 📊 **Improves on benchmarks** - Better math, QA, temporal reasoning
- 🤖 **Inspired ChatGPT Plugins** - Framework for tool-augmented LLMs
- 💡 **Key insight:** Models can generate their own training data

**Real-world impact:**
- ChatGPT function calling (similar approach)
- Claude's tool use capabilities
- Every AI assistant with tools (calculators, search, etc.)
- Self-supervised learning for capabilities

**The insight:** Don't manually annotate tool use. Let the LLM **teach itself** by generating examples and filtering for what helps.

---

## The Problem

### LLMs Have Built-in Limitations

**1. Outdated Knowledge**
```
User: What's the current population of Tokyo?
LLM: [Knowledge cutoff is 2021, gives old answer]
```

**2. Can't Do Math**
```
User: What's 847 × 923?
LLM: Approximately 781,000 [Wrong! Correct: 781,781]
```

**3. No Temporal Reasoning**
```
User: What day of the week was January 1, 2015?
LLM: [Guesses, often wrong]
```

### Previous Solutions Had Problems

**ReAct-style prompting:**
- ✅ Works, but needs manual prompts
- ❌ Requires careful prompt engineering
- ❌ Not always reliable

**Fine-tuning with tool annotations:**
- ✅ Can work well
- ❌ Need expensive human annotations
- ❌ Doesn't scale to many tools

**The question:** Can LLMs **learn to use tools automatically** without human labeling?

---

## Core Innovation

### Self-Supervised Tool Learning

**Toolformer's approach:**
```
1. Start with base LLM (GPT-J 6.7B)
2. Let model generate potential API calls
3. Execute APIs and see if they help
4. Keep only helpful examples
5. Fine-tune on filtered examples
6. Result: Model that uses tools when beneficial
```

**Key insight:** The model itself can determine when tools are useful!

---

## How It Works

### Step 1: Annotate Dataset with Potential API Calls

**Start with text:**
```
"The population of Tokyo is around 14 million."
```

**Model generates potential tool calls:**
```
"The population of Tokyo is [QA(What is the population of Tokyo?)] around 14 million."

"The population of [QA(What is the population of Tokyo?)] Tokyo is around 14 million."
```

**Insert at multiple positions, generate candidates**

### Step 2: Execute API Calls

**Run each API and get results:**
```
QA(What is the population of Tokyo?) → "13.96 million as of 2021"
```

**Create augmented text:**
```
"The population of Tokyo is [QA(What is the population of Tokyo?) → 13.96 million] around 14 million."
```

### Step 3: Filter Helpful Examples

**Key idea:** Keep API call only if it reduces loss

**Compare:**
```
Loss without API: L("around 14 million" | "The population of Tokyo is")
Loss with API:    L("around 14 million" | "The population of Tokyo is [QA(...) → 13.96 million]")

If loss_with_api < loss_without_api:
    Keep this API call (it helps!)
else:
    Discard (not useful)
```

**This is self-supervised:** Model decides what's helpful!

### Step 4: Fine-tune on Filtered Data

**Train on examples like:**
```
Input: The population of Tokyo is [QA(What is the population of Tokyo?) → 13.96 million] around 14 million.

Input: The result of 847 × 923 is [Calculator(847 * 923) → 781,781].

Input: Today is [Calendar() → Thursday, March 23, 2023] and the meeting is tomorrow.
```

**Model learns:**
- When to call tools
- How to format calls
- How to use results

---

## Available Tools

### 1. Question Answering

**API:** QA(question)
**Use:** Retrieve factual information

**Example:**
```
"Roger Federer was born in [QA(When was Roger Federer born?) → August 8, 1981]"
```

### 2. Calculator

**API:** Calculator(expression)
**Use:** Arithmetic and math

**Example:**
```
"The total is [Calculator(123 + 456 + 789) → 1368]"
```

### 3. Wikipedia Search

**API:** WikiSearch(query)
**Use:** Look up entities

**Example:**
```
"The capital of [WikiSearch(France) → France is a country...Paris is the capital] France is Paris"
```

### 4. Machine Translation

**API:** MT(text, source_lang, target_lang)
**Use:** Translate text

**Example:**
```
"'Hello' in French is [MT(Hello, English, French) → Bonjour]"
```

### 5. Calendar

**API:** Calendar()
**Use:** Current date/time

**Example:**
```
"Today is [Calendar() → Thursday, March 23, 2023]"
```

---

## Training Process

### Detailed Algorithm

**Input:** Text corpus C, set of tools T, base LM M

**Step 1: Generate API call candidates**
```python
for each position i in text:
    for each tool t in T:
        # Ask model to generate API call
        api_call = M.generate(f"<API>{t}(...)")

        # Create candidate annotation
        candidates.append((i, api_call))
```

**Step 2: Execute and filter**
```python
for (position, api_call) in candidates:
    # Execute API
    result = execute(api_call)

    # Create augmented text
    text_with_api = insert_at_position(
        text,
        position,
        f"[{api_call} → {result}]"
    )

    # Compute losses
    loss_original = M.loss(text)
    loss_with_api = M.loss(text_with_api)

    # Keep if helpful
    if loss_with_api < loss_original - threshold:
        training_data.append(text_with_api)
```

**Step 3: Fine-tune**
```python
M_finetuned = finetune(M, training_data)
```

**Result:** Model that calls tools when they help!

---

## Results

### Mathematical Reasoning (ASDiv)

| Model | Accuracy |
|-------|----------|
| GPT-J 6.7B (baseline) | 35.2% |
| GPT-J + Calculator prompting | 42.1% |
| **Toolformer 6.7B** | **51.3%** |
| GPT-3 175B (baseline) | 75.4% |

**Toolformer 6.7B beats GPT-J with 30× fewer parameters!**

### Question Answering (WebQA)

| Model | Exact Match |
|-------|-------------|
| GPT-J 6.7B | 8.6% |
| **Toolformer 6.7B** | **12.4%** |

**44% relative improvement** by using QA tool

### Temporal Reasoning (TEMPQUESTIONS)

| Model | Accuracy |
|-------|----------|
| GPT-J 6.7B | 28.1% |
| **Toolformer 6.7B** | **44.6%** |

**59% relative improvement** with Calendar tool

### Language Understanding (LAMA)

**Remains competitive** - Tool use doesn't hurt general performance

---

## Example Outputs

### Using Calculator

**Prompt:**
```
The company's revenue was $4.2M in Q1 and $5.8M in Q2.
```

**Toolformer output:**
```
The company's revenue was $4.2M in Q1 and $5.8M in Q2. The total revenue for H1 was [Calculator(4.2 + 5.8) → 10.0] $10.0M.
```

### Using QA

**Prompt:**
```
The movie was directed by
```

**Toolformer output:**
```
The movie was directed by [QA(Who directed Inception?) → Christopher Nolan] Christopher Nolan.
```

### Using Wikipedia

**Prompt:**
```
Albert Einstein was born in
```

**Toolformer output:**
```
Albert Einstein was born in [WikiSearch(Albert Einstein) → ...born March 14, 1879 in Ulm, Germany...] Ulm, Germany.
```

### Knowing When NOT to Use Tools

**Prompt:**
```
The sky is
```

**Toolformer output:**
```
The sky is blue.
```

**No tool call!** Model learned this is common knowledge.

---

## Implementation

### Training Toolformer

```python
import torch
from transformers import GPTJForCausalLM, AutoTokenizer

class ToolformerTrainer:
    def __init__(self, model_name="EleutherAI/gpt-j-6B"):
        self.model = GPTJForCausalLM.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.tools = {
            "Calculator": self.calculator,
            "QA": self.qa_system,
            "Calendar": self.calendar,
        }

    def annotate_dataset(self, texts):
        """Generate API call candidates for dataset."""
        annotated_data = []

        for text in texts:
            # Generate candidates at each position
            candidates = self.generate_candidates(text)

            # Execute and filter
            for candidate in candidates:
                if self.is_helpful(text, candidate):
                    annotated_data.append(candidate)

        return annotated_data

    def generate_candidates(self, text):
        """Generate potential API calls."""
        candidates = []
        words = text.split()

        for i in range(len(words)):
            for tool_name in self.tools:
                # Prompt model to generate API call
                prompt = f"{' '.join(words[:i])} [<API>{tool_name}("

                api_call = self.model.generate(
                    prompt,
                    max_new_tokens=20,
                    stop_token=")"
                )

                # Extract API call
                if self.is_valid_call(api_call):
                    candidates.append({
                        "position": i,
                        "api_call": api_call,
                        "text": text
                    })

        return candidates

    def is_helpful(self, text, candidate):
        """Check if API call reduces loss."""
        # Get API result
        result = self.execute_api(candidate["api_call"])

        # Create augmented text
        augmented = self.insert_api_result(
            text,
            candidate["position"],
            candidate["api_call"],
            result
        )

        # Compute losses
        loss_original = self.compute_loss(text)
        loss_augmented = self.compute_loss(augmented)

        # Keep if helpful
        threshold = 0.01
        return loss_augmented < (loss_original - threshold)

    def execute_api(self, api_call):
        """Execute API call and return result."""
        # Parse API call
        tool_name, args = self.parse_api_call(api_call)

        # Execute
        if tool_name in self.tools:
            return self.tools[tool_name](args)
        else:
            return None

    def calculator(self, expression):
        """Calculator tool."""
        try:
            return str(eval(expression))
        except:
            return "Error"

    def qa_system(self, question):
        """QA tool (placeholder)."""
        # Call actual QA system
        return call_qa_api(question)

    def calendar(self, _):
        """Calendar tool."""
        from datetime import datetime
        return datetime.now().strftime("%A, %B %d, %Y")

    def finetune(self, annotated_data):
        """Fine-tune model on annotated data."""
        # Standard supervised fine-tuning
        for epoch in range(3):
            for batch in annotated_data:
                loss = self.model(batch).loss
                loss.backward()
                optimizer.step()

        return self.model


# Usage
trainer = ToolformerTrainer()

# Annotate dataset
texts = load_corpus()
annotated_data = trainer.annotate_dataset(texts)

# Fine-tune
toolformer_model = trainer.finetune(annotated_data)

# Use the trained model
output = toolformer_model.generate("What is 847 × 923?")
# Output: "What is 847 × 923? [Calculator(847 * 923) → 781781] 781,781"
```

### Using a Pre-trained Toolformer

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load Toolformer (if available)
model = AutoModelForCausalLM.from_pretrained("toolformer-6.7b")
tokenizer = AutoTokenizer.from_pretrained("toolformer-6.7b")

# Define tool execution
def execute_tools(text):
    """Execute any API calls in generated text."""
    import re

    # Find API calls
    api_pattern = r'\[(.*?)\((.*?)\) → (.*?)\]'
    matches = re.finditer(api_pattern, text)

    for match in matches:
        tool, args, placeholder = match.groups()

        # Execute tool
        if tool == "Calculator":
            result = eval(args)
            text = text.replace(match.group(0), str(result))

    return text

# Generate with tools
prompt = "The sum of 123 and 456 is"
output = model.generate(prompt)
# Output: "The sum of 123 and 456 is [Calculator(123 + 456) → 579]"

final_output = execute_tools(output)
# Final: "The sum of 123 and 456 is 579"
```

---

## Comparison with Other Approaches

### Toolformer vs ReAct

| Aspect | ReAct | Toolformer |
|--------|-------|-----------|
| **Training** | Prompting (zero-shot) | Fine-tuning |
| **Data needed** | Few-shot examples | Self-generated |
| **Reliability** | Depends on prompts | More consistent |
| **Flexibility** | Easy to add tools | Need retraining |
| **Reasoning** | Explicit (visible) | Implicit (learned) |

### Toolformer vs ChatGPT Functions

| Aspect | ChatGPT Functions | Toolformer |
|--------|------------------|-----------|
| **Approach** | Prompted + RLHF | Self-supervised FT |
| **Data** | Human feedback | Auto-generated |
| **Format** | JSON structured | Natural language |
| **Models** | GPT-3.5/4 (175B+) | Works on 6.7B |

### Toolformer vs Manual Annotation

| Aspect | Manual Annotation | Toolformer |
|--------|------------------|-----------|
| **Cost** | High ($$$) | Low (compute only) |
| **Quality** | High quality | Good quality |
| **Scalability** | Limited | Scales easily |
| **Coverage** | May miss edge cases | Comprehensive |

---

## Limitations

### 1. **Tool Execution Overhead**
- Must execute APIs during generation
- Slower inference
- **Mitigation:** Cache common queries, async execution

### 2. **Tool Availability**
- Needs working APIs at inference time
- APIs can fail or change
- **Mitigation:** Fallback to no-tool generation

### 3. **Training Cost**
- Need to execute APIs during training (expensive)
- Filter many candidates
- **Mitigation:** Start with small corpus, scale up

### 4. **Limited Tool Set**
- Only learns tools seen during training
- Can't use new tools zero-shot
- **Mitigation:** Retrain with new tools

### 5. **No Complex Tool Composition**
- Typically single tool calls
- Not chained reasoning like ReAct
- **Mitigation:** Combine with ReAct-style prompting

---

## Modern Developments

### 1. **GPT-4 Function Calling (2023)**

**Similar principles, better implementation:**
```python
# OpenAI's approach (likely inspired by Toolformer)
response = openai.ChatCompletion.create(
    model="gpt-4",
    messages=[{"role": "user", "content": "What's 847 × 923?"}],
    functions=[{
        "name": "calculator",
        "description": "Perform calculations",
        "parameters": {
            "type": "object",
            "properties": {
                "expression": {"type": "string"}
            }
        }
    }]
)

# Model decides to call calculator
if response.choices[0].message.function_call:
    # Execute and return result
    ...
```

### 2. **Gorilla (2023)**

**Extension:** Toolformer for 1,600+ ML APIs
- Fine-tuned LLaMA to use ML model APIs
- Self-supervised like Toolformer
- Better API calling than GPT-4

### 3. **ToolLLM (2023)**

**Scaling up:**
- 16,000+ real-world APIs
- Multi-step tool use
- Better at complex workflows

### 4. **Self-Taught Optimizer**

**Generalization:**
- Same principle: let model generate training data
- Apply to other capabilities (reasoning, coding, etc.)
- Self-improvement loop

---

## Practical Applications

### 1. Scientific Computing

**Example:** LLM + computational tools
```
"The eigenvalues of the matrix are [NumPy(np.linalg.eigvals([[1,2],[3,4]])) → [5.37, -0.37]]"
```

### 2. Data Analysis

**Example:** LLM + pandas
```
"The correlation between sales and temperature is [Pandas(df['sales'].corr(df['temp'])) → 0.73]"
```

### 3. Information Retrieval

**Example:** LLM + search API
```
"According to recent news, [Search(latest GDP growth) → 2.3% in Q4 2023], the economy grew by 2.3%"
```

### 4. Personalized Assistants

**Example:** LLM + user data
```
"Your next meeting is [Calendar(get_next_event) → 3pm with Alice]"
```

---

## Key Innovations

### 1. Self-Supervised Tool Learning
**No human annotations needed** - Model generates its own training data

### 2. Loss-Based Filtering
**Automatic quality control** - Keep only helpful tool calls

### 3. Lightweight Fine-Tuning
**Works on smaller models** - 6.7B can learn tool use

### 4. Natural Integration
**Tools in natural language** - Not separate from text generation

### 5. Scalable
**Easy to add new tools** - Just define API and retrain

---

## Implementation Tips

### 1. Start with High-Precision Tools
```python
# Good first tools:
- Calculator (exact results)
- Calendar (reliable)
- Wikipedia (factual)

# Avoid initially:
- Unreliable APIs
- Ambiguous tools
- Tools with side effects
```

### 2. Use Conservative Thresholds
```python
# Don't keep marginal improvements
threshold = 0.05  # Only keep if clearly helpful

if loss_with_api < (loss_without - threshold):
    keep_example()
```

### 3. Validate Generated API Calls
```python
def is_valid_call(api_call):
    """Check if API call is well-formed."""
    try:
        # Parse and validate
        tool, args = parse(api_call)
        return tool in valid_tools and validate_args(args)
    except:
        return False
```

### 4. Handle API Failures Gracefully
```python
def execute_api_safe(api_call):
    """Execute API with fallback."""
    try:
        return execute(api_call)
    except:
        return "[API Error]"  # Model can learn to avoid
```

---

## Key Takeaways

1. **Self-supervision works** - LLMs can generate their own tool-use training data
2. **Loss-based filtering** - Automatic quality control
3. **Smaller models can use tools** - Not just for huge models
4. **Inspired modern systems** - ChatGPT functions, Claude tools
5. **Scalable approach** - Easy to add new tools

**Bottom line:** Toolformer proved that LLMs can **teach themselves** to use tools without expensive human annotation, paving the way for modern function-calling AI systems.

---

## Further Reading

### Original Paper
- **Toolformer:** https://arxiv.org/abs/2302.04761

### Related Work
- **ReAct (prompting approach):** https://arxiv.org/abs/2210.03629
- **Gorilla (API specialist):** https://arxiv.org/abs/2305.15334
- **ToolLLM (16k tools):** https://arxiv.org/abs/2307.16789

### Extensions
- **ToolkenGPT:** Token-efficient tool use
- **ART:** Automatic reasoning + tool use
- **HuggingGPT:** Using HuggingFace models as tools

### Implementations
- **Original Code:** https://github.com/lucidrains/toolformer-pytorch
- **Community Implementations:** Various GitHub repos
- **OpenAI Functions:** https://platform.openai.com/docs/guides/function-calling

---

**Published:** February 2023
**Impact:** 🔥🔥🔥🔥 **HIGH** - Self-supervised tool learning
**Citations:** 500+
**Adoption:** Inspired ChatGPT Functions, Claude Tools
**Current Relevance:** Core idea used in all modern tool-use systems
**Legacy:** Proved self-supervised learning works for tool use

**Modern Status (2024/2025):** The principles of Toolformer (self-supervised tool learning, loss-based filtering) are likely used in training modern AI assistants, though with more sophisticated implementations in GPT-4 and Claude.
