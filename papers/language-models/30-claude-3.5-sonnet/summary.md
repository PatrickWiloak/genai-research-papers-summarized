# Claude 3.5 Sonnet: Computer Use and Enhanced Capabilities

**Authors:** Anthropic
**Published:** October 22, 2024
**Announcement:** [anthropic.com/news/3-5-models-and-computer-use](https://www.anthropic.com/news/3-5-models-and-computer-use)

---

## Why This Matters

Claude 3.5 Sonnet introduced **computer use** - a paradigm shift:

- 🖱️ **First AI to control computers** - Mouse, keyboard, screen
- 💻 **Best coding model** - 49% SWE-Bench (SOTA at release)
- 🎯 **Agentic tool use** - 69.2% on TAU-bench
- 🚀 **Same price as before** - No cost increase for improvements
- 🔧 **Production ready** - Available via API

**Real-world impact:**
- Enabled AI agents that actually DO things
- Revolutionary for automation
- Best model for coding tasks
- Foundation for agentic applications

**The insight:** **AI should control computers like humans do** - not just talk about actions, actually perform them.

---

## Computer Use Feature

### How It Works

**AI can:**
```
1. View screen (screenshot)
2. Move mouse cursor
3. Click buttons
4. Type text
5. Reason about what to do next
```

**Example flow:**
```
Task: "Book a flight to Paris"

Claude:
1. Takes screenshot
2. Sees browser
3. Moves cursor to search bar
4. Types "flights to Paris"
5. Clicks search
6. Analyzes results
7. Clicks best option
8. Fills booking form
... continues until task complete
```

### Technical Implementation

```python
from anthropic import Anthropic

client = Anthropic(api_key="your-api-key")

# Define computer use tool
tools = [{
    "type": "computer_20241022",
    "name": "computer",
    "display_width_px": 1920,
    "display_height_px": 1080,
    "display_number": 1
}]

# Agent loop
messages = [{"role": "user", "content": "Go to github.com and create a new repository"}]

while True:
    response = client.messages.create(
        model="claude-3-5-sonnet-20241022",
        max_tokens=4096,
        tools=tools,
        messages=messages
    )
    
    if response.stop_reason == "end_turn":
        break
    
    # Claude wants to use computer
    tool_use = response.content[0]
    if tool_use.type == "tool_use":
        action = tool_use.input  # e.g., {"type": "mouse_move", "x": 100, "y": 200}
        
        # Execute action (your implementation)
        result = execute_computer_action(action)
        
        # Return result to Claude
        messages.append({
            "role": "assistant",
            "content": response.content
        })
        messages.append({
            "role": "user",
            "content": [{
                "type": "tool_result",
                "tool_use_id": tool_use.id,
                "content": result  # Screenshot, success/failure, etc.
            }]
        })
```

### Available Actions

**Mouse:**
- `mouse_move(x, y)`
- `left_click`
- `right_click`
- `double_click`
- `middle_click`

**Keyboard:**
- `type(text)`
- `key(key_name)`  # Enter, Tab, etc.

**Screenshot:**
- `screenshot()` - Returns current screen

---

## Performance Improvements

### Coding (SWE-Bench Verified)

**October 2024 update:**

| Model | Pass@1 |
|-------|--------|
| Claude 3.5 Sonnet (June) | 33.4% |
| **Claude 3.5 Sonnet (Oct)** | **49.0%** |
| GPT-4o | ~35% |
| DeepSeek-R1 (Jan 2025) | 66.8% |

**46% improvement over previous version!**

### Agentic Tool Use (TAU-bench)

**Retail domain:**
- June version: 62.6%
- **October version: 69.2%**

**Airline domain (harder):**
- June version: 36.0%
- **October version: 46.0%**

**28% improvement in difficult domain!**

### Graduate-level Reasoning (GPQA Diamond)

- Claude 3.5 Sonnet: **65.0%**
- GPT-4o: 49.9%
- Leading on expert-level science questions

---

## Real-World Applications

### 1. Automated Testing

```python
# Claude tests your web app
task = """
Test the login flow:
1. Go to https://myapp.com
2. Click 'Sign Up'
3. Fill registration form
4. Verify email is sent
5. Complete registration
6. Test login works
Report any bugs found
"""

# Claude executes entire flow, reports issues
```

### 2. Data Entry Automation

```python
# Process invoices
task = """
1. Open invoice PDFs from folder
2. Extract data (vendor, amount, date, items)
3. Enter into accounting system
4. Verify totals match
5. Mark as processed
"""

# Claude handles it all
```

### 3. Research Tasks

```python
task = """
Research competitor pricing:
1. Visit competitor websites
2. Find pricing pages
3. Extract prices for each plan
4. Create comparison spreadsheet
5. Identify our competitive advantages
"""
```

### 4. Software Installation

```python
task = """
Install and configure development environment:
1. Download VS Code
2. Install Python 3.11
3. Set up virtual environment
4. Install requirements.txt dependencies
5. Configure linters and formatters
6. Test with sample project
"""
```

---

## Limitations of Computer Use

### Current Issues

**1. Reliability:**
```
Success rate: ~50-60% on complex tasks
Sometimes gets confused
May click wrong elements
Can get stuck in loops
```

**2. Speed:**
```
Slower than human for simple tasks
Needs to think between actions
Screenshot processing takes time
```

**3. Error Recovery:**
```
Not great at recovering from mistakes
May need human intervention
Can't always figure out what went wrong
```

**4. Cost:**
```
Each action = API call
Screenshots = image tokens (expensive)
Long tasks = high cost
```

### Safety Restrictions

**Anthropic blocks:**
- Accessing user's private data
- Financial transactions without confirmation
- Posting to social media
- Deleting files without permission
- Installing software without approval

---

## Comparison with Alternatives

### Claude 3.5 vs GPT-4o

| Aspect | GPT-4o | Claude 3.5 Sonnet |
|--------|--------|-------------------|
| **Coding** | Good | **Better** (49% SWE-Bench) |
| **Computer use** | No | **Yes** |
| **Reasoning** | Good | **Better** (GPQA) |
| **Creative writing** | Excellent | Excellent |
| **Speed** | Fast | Fast |
| **Cost** | Lower | Moderate |

### Claude 3.5 vs DeepSeek-R1

| Aspect | Claude 3.5 | DeepSeek-R1 |
|--------|------------|-------------|
| **General tasks** | **Better** | Specialized for reasoning |
| **Computer use** | **Yes** | No |
| **Math** | Good | **Much better** |
| **Coding** | Excellent | **Better** (66.8% SWE-Bench) |
| **Cost** | Moderate | **Very low** |
| **Availability** | API only | **Open source** |

---

## Pricing

**Claude 3.5 Sonnet (October 2024):**
```
Input: $3 per million tokens
Output: $15 per million tokens

Computer use:
- Screenshot: ~1.6K tokens (depends on resolution)
- Typical task: 50-200 screenshots
- Cost: $0.25 - $2.00 per complex task
```

---

## Key Takeaways

1. **Computer use** - Revolutionary capability for automation
2. **Best coding model** - 49% SWE-Bench at release
3. **Agentic improvements** - 28% better on hard tool use
4. **Production ready** - Available via API
5. **Same price** - No increase despite improvements

**Bottom line:** Claude 3.5 Sonnet made AI agents that can actually control computers a reality, opening entirely new categories of automation.

---

## Further Reading

- **Announcement:** https://www.anthropic.com/news/3-5-models-and-computer-use
- **API Docs:** https://docs.anthropic.com/en/docs/build-with-claude/computer-use
- **Examples:** Anthropic cookbook

**Published:** October 22, 2024
**Impact:** 🔥🔥🔥🔥🔥 **CRITICAL** - Enabled computer-controlling AI
**Adoption:** Widespread in agentic applications
