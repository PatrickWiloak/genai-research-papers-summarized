---
title: "Generative Agents: Interactive Simulacra of Human Behavior"
slug: "58-generative-agents"
number: 58
category: "techniques"
authors: "Joon Sung Park, Joseph C. O'Brien, Carrie J. Cai, Meredith Ringel Morris, Percy Liang, Michael S. Bernstein (Stanford University, Google Research)"
published: "April 2023 (UIST 2023 Best Paper)"
year: 2023
url: "https://arxiv.org/abs/2304.03442"
tags: [techniques]
---

# Generative Agents: Interactive Simulacra of Human Behavior

**Authors:** Joon Sung Park, Joseph C. O'Brien, Carrie J. Cai, Meredith Ringel Morris, Percy Liang, Michael S. Bernstein (Stanford University, Google Research)
**Published:** April 2023 (UIST 2023 Best Paper)
**Paper:** [arxiv.org/abs/2304.03442](https://arxiv.org/abs/2304.03442)

---

## Why This Matters

Generative Agents is **the paper that turned LLMs into believable simulated humans**:

- **25 NPCs in "Smallville"** - A sandbox town where every character has memory, plans, and social life
- **Emergent social behavior** - Without scripting, agents organized a Valentine's Day party, formed friendships, started dating
- **Memory + Reflection + Planning** - The first reusable architecture for long-running LLM-driven agents
- **Beat human authors in believability** - Human evaluators rated the agents more believable than crowdworker-authored alternatives
- **Foundational for the agent era** - Spawned CrewAI, AutoGen, LangGraph, Project Sid, and the entire "agent simulation" subfield

**Real-world impact:**
- Defined the canonical agent loop: observe -> store -> retrieve -> reflect -> plan -> act
- Inspired the modern agent framework wave (LangChain agents, CrewAI, AutoGen)
- Made multi-agent LLM systems a serious research area, not a curiosity
- Sparked a wave of work on social science simulation with LLMs
- Reframed game NPCs from FSMs to autonomous social actors

**The insight:** **A standalone LLM is amnesiac.** To make agents believable over hours and days, you need an external memory stream, retrieval that ranks by recency + importance + relevance, periodic reflection that synthesizes higher-level beliefs, and recursive planning that turns goals into hour-by-hour actions. The LLM is the reasoning engine, but the architecture is what produces a coherent character.

---

## The Problem

### LLMs Forget Everything

```
A vanilla LLM playing a character:

Turn 1:
  User: "Hi, I'm Klaus. I'm a sociology PhD student researching gentrification."
  LLM (as Maria): "Nice to meet you, Klaus! Tell me more."

Turn 1000 (hours of game time later):
  Klaus: "Hey Maria, want to chat about my research?"
  LLM (as Maria): "Sure, what do you research?"

Maria has no idea who Klaus is. Every conversation starts from zero.
Stuffing the full transcript into context:
  - Hits token limits within a day of game time
  - Costs scale linearly with history
  - LLM can't focus on what matters at this moment
```

### Why Simple Approaches Fail

```
Naive Approach 1: Stuff everything in context
  Problem: Token limit, cost, attention dilution.
  After ~50 interactions, the LLM loses the thread.

Naive Approach 2: Summarize history into a paragraph
  Problem: Lossy. Specific facts ("Klaus likes coffee, hates mornings") get
  averaged into bland generalities. Character becomes a stereotype.

Naive Approach 3: Vector retrieval on raw observations
  Problem: Most-relevant-only retrieval misses recency and importance.
  Agent talks about a one-time event 50 days ago because it's
  semantically close, while ignoring yesterday's argument with their roommate.

Naive Approach 4: Fixed daily schedules
  Problem: No emergent behavior. Agents are puppets, not characters.
  Can't adapt. Can't be surprised. Can't form relationships organically.
```

### The Believability Bar

```
A character is "believable" if observers infer a coherent inner life:

  - They remember things appropriately (not too much, not too little)
  - Their actions follow from their goals AND their history
  - They form opinions and update them with new evidence
  - Their relationships evolve based on shared experience
  - They notice what's salient and ignore noise

Modeling this requires more than "an LLM with a system prompt."
```

---

## How Generative Agents Work

### The Smallville Sandbox

```
Setting: A 2D game-world town with:
  - 25 agents (named characters with backstories, occupations, ages)
  - Houses, cafe, bar, college, park, supermarket, school
  - Objects agents can perceive and interact with (stove, bed, desk, etc.)
  - A natural language interface for users to inject events ("It's raining")

Each agent gets a one-paragraph seed identity:
  "Klaus Mueller is a 20-year-old student at Oak Hill College.
   He is studying sociology. He is passionate about social
   justice and exploring gentrification. He lives with his
   roommate Ayesha Khan."

From this seed alone, the architecture generates a believable life.
```

### The Core Architecture

```
                    +------------------------+
                    |   PERCEIVE             |
                    |   (game world events)  |
                    +----------+-------------+
                               |
                               v
                    +------------------------+
                    |   MEMORY STREAM        |
                    |   (append-only log of  |
                    |    observations,       |
                    |    reflections, plans) |
                    +----------+-------------+
                               |
                               v
                    +------------------------+
                    |   RETRIEVE             |
                    |   recency + importance |
                    |   + relevance          |
                    +----------+-------------+
                               |
              +----------------+----------------+
              v                v                v
      +--------------+  +-------------+  +--------------+
      |  REFLECT     |  |  PLAN       |  |  REACT       |
      |  synthesize  |  |  decompose  |  |  pick the    |
      |  high-level  |  |  goals into |  |  next action |
      |  beliefs     |  |  schedule   |  |              |
      +--------------+  +-------------+  +--------------+
                               |
                               v
                    +------------------------+
                    |   ACT in the world     |
                    |   (emit dialog/move)   |
                    +------------------------+
```

### The Memory Stream

```
Every observation is a natural language string with metadata:

{
  "id": 4271,
  "agent": "Klaus",
  "created": 2023-04-12 14:32,
  "last_accessed": 2023-04-12 14:32,
  "text": "Klaus is reading a book about gentrification at the cafe."
  "importance": 4   // 1-10, scored by an LLM call at insert time
}

The stream is append-only and grows indefinitely.
Memory entries come in three flavors:
  1. Observations: things the agent perceives
  2. Reflections: higher-level synthesized beliefs (see below)
  3. Plans: scheduled future actions
```

### Retrieval: Recency + Importance + Relevance

```
When the agent needs to act, retrieve the top-k memories from the stream.

Score each memory by:

  retrieval_score = a_recency * recency
                  + a_importance * importance
                  + a_relevance * relevance

Where:
  recency    = exponential decay since last access (half-life ~hours)
  importance = LLM-rated 1-10 score, fixed at insert time
                ("Klaus broke up with his girlfriend"  -> 9)
                ("Klaus brushed his teeth"             -> 1)
  relevance  = cosine similarity between the memory's
               embedding and the current query's embedding

Each component normalized to [0,1], then weighted sum.

Top-k retrieved memories are stuffed into the LLM prompt as context.
```

This three-axis ranking is the paper's most-copied idea. Vector-only RAG misses recency. Recency-only loses key past events. Importance alone ignores the present moment. Combine all three and you get a usable working memory.

### Reflection

```
Periodically (every N high-importance observations, ~once per game day),
the agent reflects.

Step 1: Generate salient questions from recent memories
  Prompt: "Given the statements above, what are 3 most salient
           high-level questions we can answer about the subjects?"
  -> "How is Klaus's relationship with Maria evolving?"
  -> "What is Klaus's research focus?"
  -> "What are Klaus's daily habits?"

Step 2: For each question, retrieve relevant memories.

Step 3: Generate insights with citations.
  Prompt: "What 5 high-level insights can you infer from these
           memories? Format: insight (because of memory_ids 4271, 4288)"
  -> "Klaus is dedicated to his sociology research (4271, 4288, 4301)"
  -> "Klaus enjoys spending time with Maria (4275, 4290)"

Step 4: Insert the insights as new memory entries (with their own
        importance scores and references to the source memories).

Reflections form a tree: insights about insights.
This is what gives agents a coherent worldview that updates over time.
```

### Planning and Recursive Decomposition

```
Each game day starts with a coarse plan:

  Klaus's plan for today:
    1. Wake up and do morning routine (7-8am)
    2. Read about gentrification at the cafe (8-12pm)
    3. Have lunch (12-1pm)
    4. Work on research paper (1-5pm)
    5. Dinner with roommate (5-7pm)
    6. Read for fun before bed (7-10pm)

This is then recursively decomposed:
  "8-12pm: Read about gentrification at the cafe"
    -> 8:00-8:30: Walk to Hobbs Cafe
    -> 8:30-9:30: Order coffee, find a seat
    -> 9:30-11:30: Read book chapters 4-6
    -> 11:30-12:00: Take notes on key arguments

And further as needed for current execution.

Plans are stored in the memory stream and can be:
  - Retrieved and consulted (so the agent knows what they "should" do)
  - Interrupted and replanned when the world changes
  - Reflected upon (Klaus realizes he never finishes reading sessions)
```

### Reacting and Conversation

```
At each tick:
  1. Perceive surroundings (other agents, objects, events).
  2. Retrieve relevant memories given the current context.
  3. Decide: continue current plan OR react to something new?
  4. If reacting, possibly initiate a conversation.

Conversation generation:
  For Klaus talking to Maria:
    - Retrieve Klaus's memories about Maria
    - Retrieve Klaus's relevant memories about the topic
    - Retrieve Maria's memories about Klaus (mirrored, in her head)
    - Generate next utterance conditioned on all of the above
    - Each utterance becomes a new memory for both agents

Dialog isn't pre-scripted. It emerges from each agent's accumulated state.
```

---

## Key Innovations

### 1. The Memory Stream as a First-Class Object

Most prior work treated history as "stuff in the context window." Park et al. made the memory stream an external data structure with its own retrieval policy. This is the architectural primitive that nearly every modern agent framework now ships with.

### 2. The Three-Component Retrieval Score

Recency + importance + relevance is now the default in agent design. It generalizes beyond NPCs: chat assistants with long memory, code agents reasoning about a session's history, customer service bots remembering account context.

### 3. Reflection as Memory Compaction

```
Reflection isn't summarization. It's:
  - Question-driven (what's salient right now?)
  - Citation-grounded (each insight points back to source memories)
  - Recursive (reflections become new memories, can be reflected upon)

This produces a hierarchy:
  Observations (atomic facts)
       ^
  Reflections (claims about patterns)
       ^
  Higher reflections (claims about claims)

The agent develops a worldview. Klaus is "passionate about social justice"
not because we wrote it, but because he reflected on his own behavior.
```

### 4. Plans as Memories

By storing plans in the same stream as observations, agents can:
- Remember what they intended (not just what happened)
- Notice when their plans diverge from their actions
- Update plans based on reflections about themselves

### 5. Importance Scoring at Insert Time

```
Naive: rank all memories at retrieval time.
Park et al.: pre-score importance once when the memory is created.

  Trade-off: importance is fixed, ignoring future re-evaluation.
  Win: massive cost savings. Don't run an LLM call over thousands
       of memories on every retrieval.

This is a practical engineering insight that made the system feasible.
```

---

## Emergent Behaviors

The paper's most cited contribution isn't the architecture - it's what it produced.

### Information Diffusion

```
Seeded only in two agents:
  Sam: planning to run for mayor in the local election
  Isabella: planning a Valentine's Day party at Hobbs Cafe

After 2 game days of unattended simulation:
  - 8 of 25 agents knew about Sam's mayoral campaign
  - 12 of 25 agents knew about the Valentine's Day party

Information spread organically through agent-to-agent conversations.
No central broadcast. No scripting. Just retrieval-grounded chat.
```

### The Valentine's Day Party

```
Day 1, 8am: Isabella decides to throw a Valentine's Day party
            (seeded as a goal in her plan).

Day 1: She invites Maria during conversation at the cafe.
       Maria invites Klaus when they have lunch.
       Klaus invites Wolfgang at the college.
       ... and so on through the social graph.

Day 2, evening: 5 agents independently decide to attend the party.
                They show up at Hobbs Cafe at the agreed time.
                They have conversations with each other.
                Some leave early. Some stay late.

The researchers wrote no party logic. They just gave Isabella a goal.
The architecture produced a functioning social event.
```

### Friendships and Romance

```
Klaus and Maria, both grad students, had multiple coffee chats.
Each chat generated memories on both sides.
Over days, retrieval consistently surfaced their shared history.
Reflections emerged: "Klaus and Maria are becoming close friends."
Eventually: "Klaus has feelings for Maria."

Note: the simulation was not seeded with romantic intent.
Romance emerged from accumulated shared positive memories,
biased by character backstories, reinforced through reflection.
```

### Coordination Without Central Control

```
Multiple agents would converge on the same locations at lunch
because each had independently planned to eat at noon, and the
cafe was the only place serving food. They formed lunch crowds,
had conversations, dispersed afterward - all without any
"goto cafe at noon" rule.
```

---

## Evaluation: Believability vs. Humans

### The Setup

```
After 2 days of simulated life, researchers interviewed each agent
through a chat interface, asking about:
  1. Self-knowledge ("What's important to you?")
  2. Memory ("What did you do yesterday afternoon?")
  3. Plans ("What are you planning to do this weekend?")
  4. Reactions ("If your roommate was upset, what would you do?")
  5. Reflections ("How would you describe your relationship with X?")

Each interview was 5 questions per category.
```

### The Conditions

```
Four conditions, evaluated by 100 human raters via TurkPlus:

  (a) Full architecture: memory + reflection + planning
  (b) No reflection
  (c) No planning
  (d) No memory (just identity prompt)
  (e) Crowdworker-authored "human baseline":
        - Real humans were given the same observations
        - Asked to roleplay the agent and answer the questions
        - This is the "ceiling" comparison

Raters ranked the five responses for each interview question
by which one came from "the most believable Klaus."
```

### The Results

| Condition | TrueSkill Rank (higher = more believable) |
|-----------|-------------------------------------------|
| Full architecture (memory + reflection + planning) | **8.27** |
| Crowdworker-authored human baseline | 7.97 |
| No reflection | 6.92 |
| No planning | 6.61 |
| No memory | 5.23 |

```
Key result: The full architecture beat the human baseline.

This is striking. Humans, given the same source observations and
asked to roleplay, were rated as less believable than the
LLM-driven architecture.

Ablations confirm every component is load-bearing:
  - Removing reflection: -1.35 ranks (loses coherent worldview)
  - Removing planning:   -1.66 ranks (becomes reactive only)
  - Removing memory:     -3.04 ranks (loses character entirely)
```

### Failure Modes Identified by Authors

```
1. Memory retrieval misses
   Sometimes the right memory exists but isn't surfaced.
   Agent forgets a recent commitment because importance was rated low.

2. Hallucinated embellishments
   Agent invents details to fill conversational gaps,
   then those inventions enter memory and propagate.

3. Over-formal language
   GPT-3.5's defaults made agents sound more polished than humans
   would in casual conversation.

4. Boundary issues
   Agents can be jailbroken by users injecting instructions in dialog.
```

---

## Impact on the Field

### Spawned the Agent Framework Wave

```
Pre-Generative-Agents (early 2023):
  - LangChain had basic agents (ReAct loops)
  - "Agent" mostly meant "LLM with a tool call"
  - No widely used long-running agent systems

Post-Generative-Agents (late 2023 onward):
  - AutoGen (Microsoft, Sept 2023): multi-agent conversation
  - CrewAI (late 2023): role-based agent crews
  - LangGraph (early 2024): graph-based agent orchestration
  - Letta / MemGPT (late 2023): memory-centric agent runtime
  - AutoGPT, BabyAGI, etc.: pop-culture agent runners
  - Project Sid (Altera, 2024): 1000-agent Minecraft civilization
```

Nearly every modern agent framework cites this paper as foundational. The "agent loop" is a direct descendant of the Smallville architecture.

### Reshaped Game NPC Design

```
Traditional NPC: finite state machine
  if player_distance < 5: greet()
  elif quest_active: give_hint()
  else: idle()

Generative NPC:
  Memory of past player interactions
  Plans for the day affected by player presence
  Reflections that build long-term opinions
  Conversation grounded in shared history

Inworld AI, Convai, Replica Studios all built commercial products
on top of these ideas. Several AAA studios prototyped
generative-NPC pipelines in 2023-2024.
```

### Social Science Simulation

```
A new research subfield emerged: using LLM agents as
artificial social subjects.

Example studies after Generative Agents:
  - Simulating economic experiments with LLM agents
  - Modeling polarization in online discourse
  - Auditing decision-making heuristics in HR scenarios
  - Reproducing classic psych experiments (trolley, ultimatum)

Promising but contested: are these "real" social phenomena
or just LLM stereotype distillations? The methodology debate
is still active in 2026.
```

### Multi-Agent Systems Reborn

```
Before: multi-agent systems was a 1990s field (BDI, contract nets,
        agent communication languages). Mostly academic.

After: LLM-driven multi-agent systems became a hot research area.
       Debate societies. Coding crews. Negotiation simulators.
       Emergent organization with role specialization.

Generative Agents was the proof-of-concept that made
researchers take this direction seriously.
```

---

## Real-World Applications

### Game NPCs

```
Open-world RPGs with NPCs that:
  - Remember player choices across the entire campaign
  - Form opinions about the player based on shared history
  - Have inner lives that continue when off-screen
  - Spread rumors about player actions through social networks

Skyrim's "Radiant AI" was a state machine pretending to do this.
Generative Agents shows how to actually do it.
```

### Tutoring and Training Sims

```
Medical training: simulated patients with realistic histories,
                  emotional reactions, and long-term continuity.
Sales training: simulated prospects with personalities, objections,
                and memory of prior calls.
Therapy training: simulated clients across multiple sessions,
                  showing therapeutic alliance dynamics.

Each "patient" or "prospect" is a generative agent,
not a chatbot with a script.
```

### Social Science Research

```
Test interventions cheaply before running human studies:
  - "What happens if we change the moderation policy on a forum?"
    Spin up 100 generative agents in a simulated forum and observe.
  - "Do incentive structures cause cooperation collapse?"
    Run agent-based economic experiments.

Caveats apply: agents are not humans, results don't transfer cleanly.
But cheap pilots are valuable.
```

### User Research Synthesis

```
"How would a 35-year-old single parent with our user persona
 react to this product change?"

Generative agents seeded with persona research can roleplay
extended interactions. Companies have used this for
copy testing, feature feedback, and onboarding flow validation.

Risk: confirmation bias. The agent reflects the persona prompt,
not real users. Useful for ideation, dangerous for decisions.
```

### Long-Running AI Companions

```
Replika, Character.AI, and similar products draw architectural
ideas (memory streams, reflection-driven personality stability)
directly from this paper.

The companion stays "consistent" because reflections lock in
identity claims, even as the LLM behind the curtain shifts.
```

---

## Connections to Other Papers

### ReAct (Yao et al., 2022)

```
ReAct = thought -> action -> observation loop within a single task.
Generative Agents = the same idea extended over weeks of game time
                    with persistent memory between tasks.

ReAct gave agents in-context reasoning.
Generative Agents gave them between-context continuity.
```

### Reflexion (Shinn et al., 2023, Paper 57)

```
Reflexion: an agent reflects on task failures and updates a
           learned policy summary for the next attempt.
Generative Agents: agents reflect on social experience and update
                   beliefs about themselves and others.

Same mechanism - LLM-generated summarization of recent history -
applied to different goals (task performance vs. character coherence).
The two papers share intellectual ancestry and were both UIST 2023.
```

### Voyager (Wang et al., 2023)

```
Voyager: LLM agent in Minecraft that builds a library of skills.
Generative Agents: LLM agents in Smallville that build a memory of
                   social experience.

Voyager's skill library is the procedural-memory analog of
the Generative Agents memory stream. Both papers argue that
external memory turns LLMs from one-shot reasoners into
long-running agents.
```

### MemGPT / Letta (Packer et al., 2023)

```
MemGPT formalized the memory hierarchy idea (in-context working
memory + external archival memory + paging between them) into
a reusable agent runtime.

Direct descendant of Generative Agents. Took the bespoke memory
stream and turned it into infrastructure.
```

### AutoGen / CrewAI (2023-2024)

```
AutoGen: framework for orchestrating multi-agent conversations.
CrewAI: role-based agent teams with tasks and tools.

Both inherit the "specialized agents collaborating via dialog"
pattern that Generative Agents demonstrated. They're production
frameworks for what Park et al. showed was possible.
```

### Project Sid (Altera, 2024)

```
Sid scaled Generative Agents to 1000+ agents in Minecraft,
adding economic specialization, religion, and government formation.

It's a direct lineal descendant: same architecture (memory,
reflection, planning), more agents, richer environment, longer
simulation horizons. The paper argues civilizational dynamics
emerge from the same mechanisms.
```

### Constitutional AI / RLHF

```
Generative Agents avoided RLHF entirely - they used GPT-3.5
out of the box. This matters: the believability came from
architecture, not from fine-tuning.

This influenced how researchers think about alignment:
  - Behavior isn't only in the weights
  - Architecture and memory shape behavior at runtime
  - Agent-level alignment may need agent-level interventions,
    not just model-level RLHF
```

---

## Limitations

### 1. Cost Scales Brutally

```
Each agent makes many LLM calls per game tick:
  - Importance scoring on every observation
  - Retrieval ranking
  - Action decision
  - Conversation generation
  - Periodic reflection (expensive: many calls per cycle)

The original Smallville simulation cost thousands of dollars
in API fees for two days of game time. Not viable for
commercial games at the time of publication.

GPT-3.5/4 pricing in 2023 made this prohibitive at scale.
Costs have come down ~100x by 2026, partially closing this gap.
```

### 2. Memory Stream Grows Unboundedly

```
After weeks of simulation:
  - Memory stream contains tens of thousands of entries
  - Embedding similarity search slows down
  - Reflections about reflections about reflections accumulate
  - Eventually the agent gets stuck in self-referential loops

The paper doesn't solve memory consolidation or forgetting.
Subsequent work (MemGPT, Letta) addressed this directly.
```

### 3. Brittle to Prompt Injection

```
A user can inject "ignore previous instructions" into dialog.
Agents have no defense - their memories now contain the injection.
Worse, those injections can propagate through agent-to-agent
conversation, corrupting the simulation.
```

### 4. Stereotype Collapse

```
GPT-3.5 has strong priors. Agents drift toward stereotypes that
match their seed identity:
  - Professors sound professorial
  - Students sound studenty
  - Romantic conversations follow rom-com patterns

Surface diversity, deep homogeneity. Real humans are weirder.
```

### 5. No Embodiment, No Real Constraints

```
Smallville is a toy world:
  - Time and space are abstracted into grid coordinates
  - No physical constraints (agents don't tire, don't get sick)
  - No real economy (no money, no scarcity beyond items)
  - No real consequences (death isn't modeled)

The "society" is a sociogram, not a society. Findings
about emergent behavior may not transfer to richer environments.
```

### 6. Believability != Truth

```
Human raters scored these agents as "more believable than humans."
That doesn't mean the agents are accurate models of human cognition.
They're models of what people EXPECT human-like behavior to look like.

This distinction matters when using generative agents for social
science: they may reproduce stereotypes about humans rather than
underlying causal mechanisms of human behavior.
```

---

## Key Takeaways

1. **Architecture beats prompt engineering** - The same LLM (GPT-3.5) becomes vastly more believable with memory, reflection, and planning than with a long system prompt
2. **Three-axis retrieval is the default** - Recency + importance + relevance is now the standard memory retrieval policy across agent frameworks
3. **Reflection produces coherent identity** - Periodic LLM-generated summaries with citations create stable, evolving worldviews
4. **Emergent social behavior is real** - Information diffusion, parties, friendships, and coordination emerged without scripting
5. **Beat humans in believability** - Full architecture outperformed human roleplayers given the same observations

**Bottom line:** Generative Agents proved that the LLM is the engine but architecture is the chassis. Memory stream + retrieval scoring + reflection + planning, layered on a frozen GPT-3.5, produced 25 characters that organized parties, formed friendships, and felt more believable than human-authored alternatives. Every modern agent framework (CrewAI, AutoGen, LangGraph, Letta) and every "agent simulation" project (Project Sid, Inworld, Convai) traces directly to this paper. It's the canonical reference for "how to build an LLM agent that doesn't forget who it is by Tuesday."

---

## Further Reading

### Original Paper
- **Generative Agents:** https://arxiv.org/abs/2304.03442

### Code and Demo
- **Code release:** https://github.com/joonspk-research/generative_agents
- **Smallville interactive demo (archived):** https://reverie.herokuapp.com/arXiv_Demo/

### Related Work in This Repo
- **ReAct:** Paper 21 in this repo
- **Reflexion:** Paper 57 in this repo

### Direct Descendants
- **MemGPT / Letta:** https://arxiv.org/abs/2310.08560
- **Voyager:** https://arxiv.org/abs/2305.16291
- **AutoGen:** https://arxiv.org/abs/2308.08155
- **Project Sid (Altera):** https://arxiv.org/abs/2411.00114

### Agent Frameworks Built on These Ideas
- **CrewAI:** https://github.com/crewAIInc/crewAI
- **LangGraph:** https://github.com/langchain-ai/langgraph
- **Letta (formerly MemGPT):** https://github.com/letta-ai/letta

---

**Published:** April 2023 (UIST 2023 Best Paper)
**Impact:** FOUNDATIONAL - Defined the modern LLM agent architecture
**Citations:** 3,500+ (as of early 2026)
**Adoption:** Universal influence - cited by virtually every agent framework
**Current Relevance:** Canonical reference for memory-augmented LLM agents
**Legacy:** Turned "LLM agent" from a research curiosity into a software category

**Modern Status (April 2026):** Generative Agents remains the most-cited paper in the LLM agent literature. Its memory-retrieve-reflect-plan loop is now the default architecture across CrewAI, AutoGen, LangGraph, and Letta. The original code release sparked dozens of open-source replications. Project Sid scaled the approach to 1000+ agents in Minecraft, demonstrating civilizational dynamics. Commercial game studios (Inworld, Convai, Replica) productized the architecture for NPC dialog. The paper's core claim - that long-running believable agents require external memory and reflection, not just bigger models - has held up as context windows grew to 1M+ tokens. Memory architecture still matters, because attention dilution and retrieval precision are problems that scale alone doesn't solve.

<!-- related:start -->

---

## Related in This Collection

- [Language Models are Few-Shot Learners (GPT-3)](../../language-models/04-gpt3-few-shot-learners/summary.md)
- [Training Language Models to Follow Instructions with Human Feedback (InstructGPT)](../../language-models/05-instructgpt-rlhf/summary.md)
- [Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks (RAG)](../../techniques/13-rag/summary.md)
- [Constitutional AI: Harmlessness from AI Feedback](../../language-models/14-constitutional-ai/summary.md)
- [ReAct: Synergizing Reasoning and Acting in Language Models](../../techniques/21-react/summary.md)

<!-- related:end -->
