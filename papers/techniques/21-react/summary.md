# ReAct: Synergizing Reasoning and Acting in Language Models

**Authors:** Shunyu Yao, Jeffrey Zhao, Dian Yu, et al. (Google Research, Princeton)
**Published:** October 2022 (ICLR 2023)
**Paper:** [arxiv.org/abs/2210.03629](https://arxiv.org/abs/2210.03629)

---

## Why This Matters

ReAct is the **foundation for modern AI agents**. This paper enabled:

- 🤖 **ChatGPT Plugins & GPTs** - Framework for tool-using agents
- 🔧 **LangChain Agents** - Standard agent architecture
- 🚀 **AutoGPT, BabyAGI** - Autonomous agent systems
- 💡 **Simple but powerful** - Just prompt LLMs to think + act in turns
- 🎯 **Real-world utility** - Web search, API calls, database queries

**Real-world impact:**
- ChatGPT with browsing: Uses ReAct pattern
- LangChain: Built on ReAct principles
- Every major agent framework: Implements ReAct loop
- Enterprise AI: Powers workflow automation

**The insight:** LLMs can both **reason** (think) and **act** (use tools) by interleaving them in a simple loop.

---

## The Problem

### LLMs Had Two Limitations

**1. Hallucination & Lack of Knowledge**
```
User: What's the current weather in Tokyo?
LLM: [Makes up an answer, no way to check real data]
```

**2. Can't Take Actions**
```
User: Send an email to John about the meeting.
LLM: [Can only generate text, can't actually send email]
```

### Previous Approaches

**Chain-of-Thought (reasoning only):**
- Great for math, logic
- But can't access external info
- Can't take actions

**Action-only agents (no reasoning):**
- Can use tools
- But no step-by-step thinking
- Often fails on complex tasks

**The question:** Can we combine reasoning AND acting?

---

## Core Innovation

### The ReAct Loop

**Simple idea:** Alternate between thinking and acting.

```
Thought: I need to find current weather data for Tokyo
Action: search[current weather Tokyo]
Observation: Temperature is 18°C, partly cloudy...

Thought: Now I have the real data, I can answer
Action: finish[The current weather in Tokyo is 18°C and partly cloudy]
```

### Why This Works

**Synergy between reasoning and acting:**
1. **Reasoning guides actions** - Decide what tool to use and why
2. **Actions inform reasoning** - Real data updates the plan
3. **Dynamic replanning** - Can adjust based on observations

### Format

**Each step has 3 parts:**
```
Thought: [Internal reasoning about what to do next]
Action: [Call a tool or finish]
Observation: [Result from the environment/tool]
```

**Loop continues until:**
- `Action: finish[answer]` is called
- Maximum steps reached
- Error occurs

---

## How It Works

### 1. Prompt Structure

**System prompt defines available actions:**
```
You can use these actions:
- search[query]: Search Wikipedia
- lookup[term]: Find term on current page
- finish[answer]: Return final answer

Use this format:
Thought: [your reasoning]
Action: [action to take]
Observation: [will be provided]
```

### 2. Example: Question Answering

**Question:** "What is the elevation range for the area that the eastern sector of the Colorado orogeny extends into?"

**ReAct trajectory:**
```
Thought 1: I need to search Colorado orogeny, find the area that the eastern
sector extends into, then find the elevation range of that area.
Action 1: search[Colorado orogeny]
Observation 1: The Colorado orogeny was an episode of mountain building in
Colorado and surrounding areas. [...]

Thought 2: It does not mention the eastern sector. I need to look up eastern
sector.
Action 2: lookup[eastern sector]
Observation 2: The eastern sector extends into the High Plains.

Thought 3: The eastern sector extends into the High Plains. I need to search
High Plains and find its elevation range.
Action 3: search[High Plains]
Observation 3: The High Plains are a subregion of the Great Plains, elevation
ranging from 1,800 to 7,000 ft (550 to 2,130 m).

Thought 4: High Plains rise in elevation from 1,800 to 7,000 ft, so the answer
is 1,800 to 7,000 ft.
Action 4: finish[1,800 to 7,000 ft]
```

### 3. Available Actions

**Typical action space:**
- **search[query]**: Search knowledge base
- **lookup[term]**: Ctrl+F on current page
- **finish[answer]**: Return answer
- **calculate[expression]**: Use calculator
- **api_call[params]**: Call external API
- **write[content]**: Write to file/database

**Modern extensions:**
- **python[code]**: Execute Python
- **bash[command]**: Run shell commands
- **browse[url]**: Navigate web pages
- **ask_user[question]**: Get user input

---

## Results

### HotpotQA (Multi-hop QA)

| Method | Success Rate | Hallucination Rate |
|--------|--------------|-------------------|
| Chain-of-Thought only | 29.4% | 14.5% |
| Act only | 26.3% | 10.2% |
| **ReAct** | **37.1%** | **5.8%** |

**ReAct wins:** Better performance, fewer hallucinations!

### FEVER (Fact Verification)

| Method | Accuracy |
|--------|----------|
| Standard prompting | 55.8% |
| Chain-of-Thought | 57.2% |
| **ReAct** | **65.1%** |

**8-point improvement** over CoT by grounding in Wikipedia.

### ALFWorld (Interactive Tasks)

**Task:** "Put a clean plate on the table"

| Method | Success Rate |
|--------|--------------|
| Imitation Learning | 38% |
| Reinforcement Learning | 45% |
| **ReAct (few-shot)** | **71%** |

**Key finding:** Few-shot ReAct beats trained RL agents!

---

## Real-World Applications

### 1. ChatGPT Plugins (2023)

**How ChatGPT uses tools:**
```
User: What's the weather in Paris and convert it to Fahrenheit?

Thought: I need to get current weather for Paris
Action: weather_api.get_weather(location="Paris")
Observation: {"temp_c": 18, "condition": "Cloudy"}

Thought: Now convert 18°C to Fahrenheit using formula
Action: python_interpreter.run("(18 * 9/5) + 32")
Observation: 64.4

Thought: I have both pieces of information
Action: finish["The weather in Paris is 18°C (64.4°F) and cloudy"]
```

### 2. LangChain Agents

**Standard LangChain pattern:**
```python
from langchain.agents import initialize_agent, Tool
from langchain.llms import OpenAI

# Define tools
tools = [
    Tool(
        name="Search",
        func=search_tool,
        description="Search the web for current information"
    ),
    Tool(
        name="Calculator",
        func=calculator_tool,
        description="Calculate mathematical expressions"
    )
]

# Initialize ReAct agent
agent = initialize_agent(
    tools=tools,
    llm=OpenAI(temperature=0),
    agent="zero-shot-react-description",  # ReAct!
    verbose=True
)

# Use it
agent.run("What is the GDP of France in 2023 divided by its population?")
```

**LangChain output:**
```
Thought: I need to find France's GDP and population for 2023
Action: Search
Action Input: "France GDP 2023"
Observation: France's GDP in 2023 is approximately $2.96 trillion

Thought: Now I need the population
Action: Search
Action Input: "France population 2023"
Observation: France's population is approximately 67.8 million

Thought: Now I can calculate GDP per capita
Action: Calculator
Action Input: 2.96e12 / 67.8e6
Observation: 43,658.88

Thought: I have the answer
Final Answer: France's GDP per capita in 2023 is approximately $43,659
```

### 3. AutoGPT & Autonomous Agents

**AutoGPT architecture:**
1. Get objective from user
2. **Think:** Plan next action (ReAct reasoning)
3. **Act:** Execute command (file operations, web search, etc.)
4. **Observe:** Evaluate results
5. Loop until objective achieved

**Example task:** "Research competitors and write report"
```
Thought: I need to identify the company first
Action: read_file["company_info.txt"]
Observation: [company details]

Thought: Now search for competitors
Action: google_search["top competitors of [company]"]
Observation: [competitor list]

Thought: Research each competitor
Action: browse_website["competitor1.com"]
Observation: [website content]

... (continues for each competitor)

Thought: I have enough information
Action: write_file["report.md", content]
Observation: File written successfully

Thought: Task complete
Action: finish
```

---

## Implementation

### Basic ReAct Agent (Python)

```python
import openai

def react_agent(question, tools, max_steps=10):
    """
    Simple ReAct agent implementation.

    Args:
        question: User question
        tools: Dict of available tools (name -> function)
        max_steps: Maximum reasoning steps
    """
    # Build prompt with available actions
    tool_descriptions = "\n".join([
        f"- {name}: {func.__doc__}"
        for name, func in tools.items()
    ])

    prompt = f"""Answer the question using this format:
Thought: [your reasoning]
Action: [action_name(args)]
Observation: [will be provided]

Available actions:
{tool_descriptions}
- finish(answer): Return final answer

Question: {question}
"""

    trajectory = prompt

    for step in range(max_steps):
        # Get next thought + action from LLM
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[{"role": "user", "content": trajectory}],
            temperature=0
        )

        llm_output = response.choices[0].message.content
        trajectory += llm_output + "\n"

        # Parse action
        if "finish(" in llm_output:
            # Extract answer and return
            answer = extract_finish_answer(llm_output)
            return answer

        action_name, action_args = parse_action(llm_output)

        # Execute action
        if action_name in tools:
            observation = tools[action_name](action_args)
        else:
            observation = f"Error: Unknown action {action_name}"

        # Add observation to trajectory
        trajectory += f"Observation: {observation}\n"

    return "Max steps reached without answer"

# Example usage
tools = {
    "search": lambda q: wikipedia.search(q),
    "calculate": lambda expr: eval(expr),
}

answer = react_agent(
    "What is the capital of the country with the largest population?",
    tools
)
print(answer)  # "Beijing, China"
```

### LangChain Implementation

```python
from langchain.agents import AgentType, initialize_agent
from langchain.llms import OpenAI
from langchain.tools import Tool
import requests

# Define custom tools
def search_tool(query: str) -> str:
    """Search the web for current information."""
    # Your search implementation
    return search_result

def calculator_tool(expression: str) -> str:
    """Calculate mathematical expressions."""
    try:
        return str(eval(expression))
    except:
        return "Invalid expression"

def weather_tool(city: str) -> str:
    """Get current weather for a city."""
    # Call weather API
    response = requests.get(f"https://api.weather.com/{city}")
    return response.json()["description"]

# Create tools list
tools = [
    Tool(
        name="Search",
        func=search_tool,
        description="Search the web for current information"
    ),
    Tool(
        name="Calculator",
        func=calculator_tool,
        description="Calculate mathematical expressions"
    ),
    Tool(
        name="Weather",
        func=weather_tool,
        description="Get current weather for a city"
    )
]

# Initialize ReAct agent
llm = OpenAI(temperature=0)
agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True,  # Print reasoning steps
    max_iterations=10,
    early_stopping_method="generate"
)

# Use the agent
result = agent.run(
    "What's the weather in the city with the tallest building in the world?"
)
```

---

## Comparison with Alternatives

### ReAct vs Chain-of-Thought

| Aspect | Chain-of-Thought | ReAct |
|--------|------------------|-------|
| **Reasoning** | Yes (internal only) | Yes (guides actions) |
| **Tool use** | No | Yes |
| **Grounding** | Model knowledge only | External data |
| **Hallucination** | Higher | Lower |
| **Use case** | Math, logic puzzles | Real-world tasks |

### ReAct vs Tool-Use Without Reasoning

| Aspect | Direct Tool Use | ReAct |
|--------|----------------|-------|
| **Explainability** | Low (black box) | High (see reasoning) |
| **Debugging** | Hard | Easy (inspect thoughts) |
| **Complex tasks** | Often fails | Better planning |
| **Efficiency** | Sometimes wasteful | More targeted |

---

## Advanced Patterns

### 1. ReAct with Self-Correction

**Add reflection step:**
```
Thought: I should search for the capital
Action: search[capital of largest country]
Observation: Russia is the largest country. Capital is Moscow.

Reflection: Wait, the question asked for largest by population, not area!
Thought: I need to search for most populous country
Action: search[most populous country]
Observation: China has the largest population.

Thought: Now find China's capital
Action: search[capital of China]
Observation: Beijing is the capital of China.
Action: finish[Beijing]
```

### 2. Hierarchical ReAct

**Break complex tasks into subtasks:**
```
High-level agent:
  Thought: Need to book flight, hotel, and rental car
  Action: delegate_to_flight_agent[book flight to Paris]

  Flight agent (sub-agent):
    Thought: Search for flights
    Action: search_flights[origin=NYC, dest=Paris, date=2024-06-01]
    Observation: [flight options]
    Thought: Book cheapest option
    Action: book_flight[flight_id=123]
    Observation: Booked successfully
    Action: finish[Flight booked: AF456]

  Observation: Flight booked: AF456
  Thought: Now handle hotel
  Action: delegate_to_hotel_agent[book hotel in Paris]
  ...
```

### 3. Multi-Agent Collaboration

**Agents communicate via ReAct:**
```
Research Agent:
  Thought: I need competitor analysis
  Action: send_message_to[analyst_agent, "Analyze competitor X"]
  Observation: [waits for response]

Analyst Agent (receives message):
  Thought: I need to research competitor X
  Action: search[competitor X financial data]
  Observation: [data]
  Thought: Compile analysis
  Action: send_message_to[research_agent, analysis]

Research Agent:
  Observation: [receives analysis]
  Thought: I have the data now
  Action: continue_with[analysis]
```

---

## Limitations

### 1. **Token Cost**
- Each step uses tokens for full trajectory
- Long tasks can be expensive
- **Mitigation:** Summarize past steps, use cheaper models

### 2. **Error Propagation**
- Wrong action → wrong observation → wrong reasoning
- Can spiral into failure
- **Mitigation:** Add reflection/self-correction steps

### 3. **Max Steps Limit**
- Complex tasks may exceed step limit
- **Mitigation:** Hierarchical agents, better planning

### 4. **Tool Reliability**
- Broken APIs or tools cause failures
- **Mitigation:** Error handling, fallback tools

### 5. **Prompt Engineering Required**
- Need good descriptions of tools
- Action parsing can be fragile
- **Mitigation:** Structured outputs (JSON), few-shot examples

---

## Modern Developments

### 1. **Function Calling APIs (2023+)**

**OpenAI/Anthropic now have native function calling:**
```python
# Modern approach (more reliable than parsing)
response = openai.ChatCompletion.create(
    model="gpt-4",
    messages=[{"role": "user", "content": "What's the weather in Paris?"}],
    functions=[{
        "name": "get_weather",
        "description": "Get current weather",
        "parameters": {
            "type": "object",
            "properties": {
                "location": {"type": "string"}
            }
        }
    }],
    function_call="auto"
)

# LLM returns structured function call
function_call = response.choices[0].message.function_call
# {"name": "get_weather", "arguments": '{"location": "Paris"}'}
```

**Advantage:** No parsing needed, more reliable

### 2. **ReWOO (Reasoning WithOut Observation, 2023)**

**Optimization:** Plan all actions upfront
```
Instead of:
  Thought → Action → Observation → Thought → Action → ...

Do:
  Plan: [Action1, Action2, Action3] → Execute all → Reason over all results
```

**Benefit:** Fewer LLM calls, can parallelize actions

### 3. **Reflexion (2023)**

**Add explicit self-reflection:**
```
Attempt 1: [fails]
Reflection: What went wrong? How to improve?
Attempt 2: [uses reflection to do better]
```

---

## Impact on the Field

### Spawned an Ecosystem

**Agent Frameworks:**
- LangChain (most popular)
- AutoGPT
- BabyAGI
- MetaGPT
- CrewAI

**OpenAI Features:**
- ChatGPT Plugins (deprecated)
- GPTs with Actions
- Assistants API

**Enterprise Adoption:**
- Customer service agents
- Research assistants
- Data analysis agents
- Workflow automation

### Research Extensions

**Papers building on ReAct:**
- **Reflexion** - Self-reflection for agents
- **ReWOO** - Planning-based optimization
- **Tree of Thoughts** - Search over reasoning paths
- **Toolformer** - Training LLMs to use tools
- **Voyager** - Agents in Minecraft

---

## Key Takeaways

1. **Simple but powerful** - Just alternate thinking and acting
2. **Grounding reduces hallucination** - Real data beats guessing
3. **Synergy matters** - Reasoning + acting > either alone
4. **Foundation for agents** - Every major framework uses this
5. **Still actively evolving** - Function calling, better planning, multi-agent

**Bottom line:** ReAct made LLMs practical for real-world tasks by giving them tools and a framework to use them intelligently.

---

## Further Reading

### Original Paper
- **ReAct:** https://arxiv.org/abs/2210.03629

### Related Work
- **Chain-of-Thought:** https://arxiv.org/abs/2201.11903
- **Reflexion:** https://arxiv.org/abs/2303.11366
- **ReWOO:** https://arxiv.org/abs/2305.18323
- **Toolformer:** https://arxiv.org/abs/2302.04761

### Implementations
- **LangChain:** https://python.langchain.com/docs/modules/agents/
- **AutoGPT:** https://github.com/Significant-Gravitas/AutoGPT
- **LlamaIndex Agents:** https://docs.llamaindex.ai/en/stable/module_guides/deploying/agents/

### Tutorials
- **LangChain ReAct guide:** Official documentation
- **Building Agents from Scratch:** DeepLearning.AI course
- **ReAct prompting guide:** Anthropic docs

---

**Published:** October 2022
**Impact:** 🔥🔥🔥🔥🔥 **CRITICAL** - Foundation for all AI agents
**Citations:** 800+
**Adoption:** Universal in production agent systems
**Current Relevance:** Core pattern for ChatGPT, LangChain, enterprise agents
**Legacy:** Enabled the shift from passive LLMs to active AI agents

**Modern Status (2024/2025):** ReAct is the standard agent architecture. Enhanced by function calling APIs and better planning, but core loop remains the same.
