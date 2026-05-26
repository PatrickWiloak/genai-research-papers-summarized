# Voyager: An Open-Ended Embodied Agent with Large Language Models

**Authors:** Guanzhi Wang, Yuqi Xie, Yunfan Jiang, Ajay Mandlekar, Chaowei Xiao, Yuke Zhu, Linxi Fan, Anima Anandkumar (NVIDIA, Caltech, UT Austin, Stanford, ASU)
**Published:** May 2023 (TMLR 2024)
**Paper:** [arxiv.org/abs/2305.16291](https://arxiv.org/abs/2305.16291)

---

## Why This Paper Matters

Voyager was the first compelling demonstration that a GPT-4-powered agent could pursue *open-ended*, lifelong learning in an interactive environment. Dropped into Minecraft with no task-specific training, no demonstrations, and no human-authored curriculum, Voyager autonomously decided what to learn next, wrote code for new skills, debugged that code by reading the in-game environment's error messages, and stored successful skills in a growing library for later reuse. Over the course of an episode it acquired diamond tools roughly twice as fast as the strongest prior baselines and explored a Minecraft world more than three times more broadly.

The paper was a landmark for **LLM agents** in three senses. First, it showed that an LLM could plan over a multi-hour horizon by treating skill acquisition itself as the planning unit. Second, it introduced the now-common pattern of "skill library + curriculum + iterative prompting with error feedback." Third, it demonstrated that natural-language-driven agents can keep getting better simply by interacting with their world — without ever changing the underlying model weights. Voyager is one of the most-cited reference points whenever people talk about autonomous, self-improving AI agents.

---

## The Problem

Reinforcement learning agents in Minecraft (such as DreamerV3 or VPT) had achieved impressive feats by training on enormous amounts of gameplay data, but they suffered from three persistent limitations:

1. **Sample inefficiency.** Learning a single skill like "collect a diamond" required millions of environment steps and substantial compute.
2. **Catastrophic forgetting.** When a new skill was trained, old skills often degraded.
3. **No transparent reasoning.** It was hard to inspect what the agent "wanted" to do, which made debugging and steering nearly impossible.

Meanwhile, GPT-4 had clearly demonstrated strong common-sense reasoning and the ability to write Minecraft-flavored code (the modding community had been documenting this for months). But nobody had successfully closed the loop between an LLM's textual planning and the *long-horizon, sensorimotor* demands of an open-world game.

The Voyager team's question: can we use GPT-4 not as a policy in the RL sense, but as the *brain* of an agent that writes its own programs, learns its own curriculum, and accumulates skills indefinitely?

---

## The Core Innovation

Voyager has three tightly-coupled components that together produce open-ended learning:

1. **An Automatic Curriculum** — a GPT-4 prompt that proposes the next task the agent should attempt, conditioned on what it already knows, its current inventory, the time of day, and recently completed goals. The curriculum is biased toward novelty: "discover as many diverse things as possible."

2. **A Skill Library** — a growing collection of executable JavaScript programs (in the Mineflayer API). Each skill is indexed by a natural-language description embedding. When the agent faces a new task, it retrieves the top-k most relevant skills as in-context examples.

3. **An Iterative Prompting Loop with Self-Verification** — GPT-4 writes code for the proposed task. The code is executed in the Minecraft environment. Any runtime errors, environment feedback, or self-verification failures are fed back to GPT-4, which revises the code. The loop continues until the task succeeds.

```
   +-----------------+
   |   GPT-4 brain   |
   +-----------------+
       |        ^
       v        |
   Curriculum: "Mine a diamond"
       |
       v
   Skill Library <----+ retrieve relevant prior skills
       |              |
       v              |
   Write code ---> Execute in Minecraft
                     |
                     v
                 Env feedback / error / self-check
                     |
                     +-> if failed: revise code (loop)
                     +-> if succeeded: store skill, ask curriculum for next task
```

The key conceptual move is treating **skills (programs), not actions (keystrokes)**, as the unit of learning. The agent never "decides which key to press" — it writes a function called `mineDiamondWithIronPickaxe()` once, debugs it, then calls it forever afterward.

---

## How It Works

### The Automatic Curriculum

Periodically, Voyager asks GPT-4 something like:

> Given my current state (inventory, biome, equipped items, completed tasks), what is the next interesting and feasible task I should attempt to make progress in Minecraft?

GPT-4's broad world knowledge about Minecraft (gleaned from the millions of tutorials, wiki pages, and forum posts in its training data) makes it surprisingly good at proposing tasks at the right difficulty level: not so easy that nothing new is learned, not so hard that progress stalls.

### The Skill Library

Each successful skill is stored as a function plus a natural-language description, embedded into a vector store:

```javascript
// Description: Craft a wooden pickaxe using 3 planks and 2 sticks.
async function craftWoodenPickaxe(bot) {
    const planks = bot.inventory.findInventoryItem("oak_planks");
    // ... etc.
    await bot.craft(recipe, 1, craftingTable);
}
```

When a new task arrives, Voyager retrieves the 5 most similar prior skills and includes them in the GPT-4 prompt as both examples and reusable building blocks. New skills compose old ones, so the library produces compound capabilities the model never explicitly programmed.

### Iterative Prompting

Voyager writes code, runs it, and consumes three kinds of feedback:

- **Execution errors** (e.g., `Error: bot does not have a pickaxe`)
- **Environment state** (e.g., `inventory: { iron_ingot: 0 }` after a supposed mining attempt)
- **Self-verification** — a separate GPT-4 call that checks whether the agent has actually completed the task, returning either `success` or a textual explanation of what went wrong

These signals are concatenated and sent back to GPT-4, which writes a corrected version. The loop runs up to ~4 iterations per task. This is essentially Self-Refine specialized to embodied code-writing.

---

## Key Results

Voyager was compared to AutoGPT, ReAct, and Reflexion baselines, all running on GPT-4, in the same Minecraft environment.

| Metric | Voyager | Best Baseline (Reflexion / AutoGPT) |
|--------|---------|--------------------------------------|
| Unique items obtained | 3.3x more | baseline |
| Distance explored | 2.3x farther | baseline |
| Tech-tree milestones (wood -> stone -> iron -> diamond) | Reached diamond | Stalled around stone/iron |
| Time to wooden tools | Fast | Comparable |
| Time to diamond tools | ~2x faster than best baseline | Slow |

Other findings:

- **The skill library was critical.** Ablations removing it caused the agent to keep "forgetting" how to perform common subtasks; tech-tree progress collapsed.
- **GPT-3.5 fell off a cliff.** Voyager with GPT-3.5 could not pass even early tech-tree milestones, illustrating that the approach depends on a capable underlying LLM.
- **Skills transferred between worlds.** Skill libraries trained in one Minecraft world bootstrapped progress in fresh worlds, demonstrating genuine knowledge accumulation.
- **No human authoring of curriculum or rewards.** The entire learning trajectory emerged from the LLM-driven loop.

---

## Impact and Legacy

Voyager became one of the most influential reference architectures for autonomous LLM agents. Its specific contributions have propagated widely:

- The **skill library** pattern reappears in many later agent systems, where executable programs are accumulated and retrieved by semantic similarity (and in tool-using agents like ToolLLM and Gorilla).
- The **automatic curriculum** idea — letting the LLM choose its own next task — anticipated agentic systems like AutoGPT, BabyAGI, and the open-ended evaluation protocols used to study large agents.
- The **iterative-prompting-with-environment-feedback** loop is now standard in coding agents (Cursor, Aider, SWE-agent) and in robotic manipulation systems that translate LLM plans into low-level control.
- Voyager kicked off serious research interest in **lifelong / continual LLM agents**, including follow-ups in scientific discovery (AlphaEvolve, FunSearch), in robotics (Code-as-Policies, RT-X), and in gaming benchmarks (MineDojo, SmartPlay).

Perhaps most importantly, Voyager demonstrated that the unit of "learning" for an LLM agent does not need to be model weights. By accumulating *artifacts* (skills, plans, memories) outside the model, an agent can keep improving with a frozen LLM brain — a property that turns out to be central to how production agents are built today.

---

## Connections to Other Papers

- **ReAct (#21)** — provided the basic reason-then-act loop. Voyager extends it with a code-writing action space and a persistent skill library.
- **Self-Refine (#84)** — Voyager's iterative prompting with environment errors is essentially Self-Refine specialized to embodied code generation.
- **Toolformer (#24)** — both papers concern LLMs that use external capabilities, but Toolformer learns to *call* fixed tools, while Voyager *writes new ones* and stores them.
- **Generative Agents (#58)** — companion paper to Voyager in spirit: simulated humans using LLMs as their cognitive substrate, with memory and reflection. Voyager focuses on skill accumulation; Generative Agents focuses on social memory.
- **Tree of Thoughts (#25)** — alternate approach to LLM problem solving via search over thoughts; Voyager instead grows a library of skills.
- **Chain-of-Thought (#9) / Self-Consistency (#85)** — Voyager relies on CoT-style reasoning inside each skill-writing prompt.
- **MCP / Model Context Protocol (#59)** — Voyager's skill library foreshadows the modern view of "tools and capabilities as first-class objects an agent retrieves and composes."
- **AlphaEvolve (#62) and AlphaGeometry (#61)** — both build self-improving LLM systems that accumulate solutions and reuse them, in the same lineage Voyager pioneered.
- **AlphaZero (#89)** — both Voyager and AlphaZero achieve open-ended skill mastery without human demonstrations, but via very different mechanisms (LLM curriculum vs. self-play RL).

---

## Key Takeaways

1. **Skills, not actions, are the right learning unit for LLM agents.** Voyager writes JavaScript programs that compose into more powerful programs. This sidesteps the sample-inefficiency of low-level RL.
2. **Three pillars produce open-ended learning.** Automatic curriculum (decide what to learn), skill library (remember what you've learned), iterative prompting with environment feedback (actually learn it).
3. **GPT-4 is the engine, but artifacts are the memory.** Voyager never updates model weights. All long-term learning lives in the external skill library and is retrieved by semantic search.
4. **The approach depends critically on a capable base model.** GPT-3.5 cannot make Voyager work; GPT-4 can. The architecture scales with the underlying LLM.
5. **Voyager is the prototype for modern self-improving agents.** Its patterns appear, often without attribution, in AutoGPT, SWE-agent, AlphaEvolve, Code-as-Policies, and most production coding assistants.
