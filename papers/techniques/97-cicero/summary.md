# Human-Level Play in the Game of Diplomacy by Combining Language Models with Strategic Reasoning (CICERO)

**Authors:** Meta Fundamental AI Research Diplomacy Team (FAIR), including Anton Bakhtin, Noam Brown, David Wu, Adam Lerer, Hengyuan Hu, et al.
**Published:** November 2022, **Science** vol. 378, issue 6624 ([DOI: 10.1126/science.ade9097](https://doi.org/10.1126/science.ade9097))

---

## Why This Paper Matters

CICERO was the first AI system to reach **human-level performance in a game that requires natural-language negotiation with humans**. Meta's FAIR team built an agent that placed in the **top 10% of human players** across 40 online games of Diplomacy on webDiplomacy.net — a game where you literally cannot win without persuading other players to ally with you, coordinate attacks, and (often) eventually betray each other.

This mattered because Diplomacy is the canonical test bed for AI that must combine three things that no previous system had ever combined at scale: **strategic planning**, **theory of mind about other agents**, and **free-form language**. Chess, Go, and poker are all silent. Diplomacy is conversation. CICERO showed that a large language model could be **grounded in a strategic planning loop** so that its dialogue was not just plausible-sounding but **consistent with actual plans the agent intended to execute**. That architectural pattern — LLM for language, search/planning for intent, the two tightly coupled — became a template for the entire "language + planning" research direction.

---

## The Problem

Diplomacy is a seven-player board game set in pre-WWI Europe. Each turn:

1. Players spend ~5 minutes **negotiating privately** with each other via free-form text.
2. Then everyone submits orders **simultaneously**.
3. Orders resolve deterministically by the game's rules.
4. Whoever first controls 18 of the 34 supply centers wins.

Crucially, **alliances are not enforceable**. You can promise anything in dialogue, then betray it when orders resolve. Success requires building real trust with humans — which means saying things that are coordinated, credible, and consistent with what you actually do.

This destroys the standard AI playbooks:

**1. Pure self-play RL (AlphaZero-style)** doesn't work — there is no agreed dialogue protocol to self-play with, and trained-from-scratch language is gibberish to humans.

**2. Pure LLMs (GPT-3-era)** can produce plausible dialogue but have no concept of board state, no plan, and no incentive to keep promises. They drift, contradict themselves, and lose immediately.

**3. Specialized planners** can play "no-press" Diplomacy (no dialogue) well, but cannot communicate, so they cannot exploit (or survive) the social dynamics of the full game.

The problem CICERO solved: **how do you make an LLM that talks like a human, but whose words are grounded in — and predictive of — actual strategic decisions?**

---

## The Core Innovation

CICERO is a tightly coupled system of two subsystems:

```
Strategic Reasoning Module        Language Module
  - models other players          - generates dialogue
  - plans joint actions           - fine-tuned LLM
  - outputs intended moves        - conditioned on intents
            ^                              |
            |______________________________v
                (intents flow both ways)
```

The key idea: **plans are computed first; dialogue is generated to be consistent with those plans.** And then, the planner anticipates how its dialogue will change other players' behavior, and replans accordingly.

Concretely:

1. The **strategic reasoning module** predicts what each player intends to do this turn, using an equilibrium-search algorithm called **piKL** (policy-iteration KL-regularized) that mixes planning with imitation of human play. This produces **plausible joint intents** — not optimal play in a vacuum, but moves humans would actually consider.
2. The **language model** (a fine-tuned BART-style 2.7B-parameter model) generates dialogue **conditioned on those intents** — so what CICERO says reflects what it actually plans to do.
3. CICERO uses the same LM to **predict other players' intents** from incoming messages, feeding those predictions back into planning.
4. Outgoing messages are **filtered** to remove ones that are nonsensical, leak too much information, or contradict planned moves.

The result is dialogue that is grounded, consistent, and strategically useful — and a planner that takes the persuasive effect of its own messages into account.

---

## How It Works

### piKL: planning grounded in human play

A vanilla optimal-play search in Diplomacy yields strange, hyperaggressive strategies humans would never accept. piKL is a regularized equilibrium search that finds policies that are **both strong and similar to what humans actually do**. Mathematically each player optimizes:

```
maximize:  E[ utility(joint_action) ]
penalty:   - lambda * KL( policy || human_imitation_policy )
```

The KL penalty pulls play toward what a human imitation model predicts. This gives CICERO a planner whose intents are recognizable to human partners — which is essential because its dialogue has to make sense given those intents.

### Intent-conditioned dialogue generation

The language model is fine-tuned on ~40K human Diplomacy games (~13 million messages) with each message **labeled by the actual intents** the speaker held that turn — labels inferred automatically from subsequent moves. At generation time:

```
input  = (board state, dialogue history, MY intended moves, PARTNER's predicted intents)
output = a natural-language message consistent with all of the above
```

Because the LM was trained on (intent, message) pairs, conditioning on intents at inference yields dialogue that genuinely reflects what CICERO plans to do.

### Message filtering

Before sending, candidate messages are passed through filters:

- A **classifier** that flags messages contradicting the agent's intent.
- A **nonsense filter** for ungrammatical or off-topic outputs.
- A **value filter** that estimates whether sending each message helps CICERO's expected position.

Only messages that pass all filters are sent. Many candidates per turn are generated and the best surviving one is used.

### Iterated planning

When CICERO receives a message, it updates its predictions of the sender's intents (using the LM as an intent classifier), then re-runs piKL with the new beliefs. Outgoing dialogue is then regenerated to match the updated plan. Plans and messages co-evolve over the negotiation window.

---

## Key Results

### Performance against humans

Across 40 anonymous games on webDiplomacy.net's blitz format (5-minute negotiation rounds, 82 unique human opponents):

- **Top 10% of all players** with more than one game played.
- **Average score 2x** that of human players in the same games.
- Never reported as a bot — humans believed they were playing a person.

### Specific competence checks

- CICERO's tactical play was strong but not superhuman in a pure board sense; the dialogue was what won games.
- It successfully proposed alliances, coordinated joint attacks, and on multiple occasions negotiated escapes from losing positions through clever message construction.
- It rarely "lied" — its messages were grounded in intent — but it did sometimes change plans, which is fair in Diplomacy.

### Ablations

The authors compared CICERO against ablations that removed each component:

- **No dialogue** (intent-only orders): much weaker — couldn't coordinate.
- **Dialogue not conditioned on intent**: dialogue drifted, partners distrusted CICERO, alliances collapsed.
- **No piKL regularization** (pure self-play planning): aggressive play that humans refused to ally with.

All three pieces — language, planning, and grounding — were necessary.

---

## Impact and Legacy

CICERO was a landmark for several reasons that resonate well beyond Diplomacy:

**1. First credible LLM-grounded-in-planning system.** It demonstrated that an LLM could be wired into a planner so that its outputs were strategically coherent over many turns, not just locally plausible. Every modern "LLM as the policy of an agent with a planner" architecture is downstream of this idea.

**2. Negotiation as a benchmark.** It opened multi-agent negotiation as a serious AI evaluation domain. Follow-on work on LLM agents negotiating contracts, auctions, debate, and cooperative tasks all cite CICERO.

**3. Hybrid neuro-symbolic AI.** CICERO is one of the cleanest examples of combining a **learned LLM** with **classical search/planning**. The same architectural family — search guiding/grounding an LLM — now powers AlphaGeometry, AlphaProof, AlphaEvolve, and a large class of reasoning agents.

**4. A path between deception and helpfulness.** CICERO explicitly chose intent-grounded dialogue rather than free-form persuasion. That choice — language that is allowed to be selective but not strictly false — is exactly the kind of design constraint that later alignment work formalized for honest AI systems.

The release also generated significant public discussion about AI in social/strategic settings, and helped frame the question that defines current frontier work: **what does it mean for an AI system to negotiate, persuade, or commit in good faith?**

---

## Connections to Other Papers

- **AlphaZero (#89):** The previous high-water mark for AI in strategic games. AlphaZero is pure planning + self-play in deterministic perfect-information games. CICERO extends the paradigm to a game requiring imperfect-information negotiation in natural language.
- **AlphaGeometry (#61) and AlphaEvolve (#62):** Same architectural pattern — LLM proposes, classical search / verifier disciplines. CICERO is an early member of this lineage.
- **Generative Agents (#58):** Sister line of work on LLM-driven multi-agent social behavior. Generative Agents simulate believable social worlds; CICERO competes in a real one.
- **InstructGPT (#5) and Constitutional AI (#14):** Concerned with making LLM outputs aligned with human intent. CICERO's intent-conditioning is a domain-specific cousin — generating dialogue conditioned on the agent's own intent so that words and actions match.
- **Voyager (#86):** Another LLM-as-agent system in a complex environment (Minecraft). CICERO is the multi-agent, negotiation-driven analog.
- **DPO (#19) / RLHF:** CICERO predates these as the standard alignment toolkit but exemplifies the same broader goal — making generative models behave consistently with intended properties.
- **GPT-3 (#4):** CICERO uses a smaller fine-tuned LM, but the underlying capability (fluent context-aware text generation) is what made the architecture viable at all.

---

## Key Takeaways

1. **Language + planning can be tightly coupled.** CICERO's core architectural move — generate dialogue conditioned on planned intents, and update plans based on inferred intents from incoming dialogue — is now a template for grounded LLM agents.
2. **Human-grounded search beats pure optimization.** piKL's KL regularization toward human play made CICERO's strategies recognizable enough that humans would actually ally with it.
3. **Intent grounding without strict honesty.** CICERO's messages reflect its current plans but it is free to change those plans — a realistic and ethically defensible model for negotiation.
4. **Top-10% human-level in a language-rich game.** The first AI system to negotiate with humans in free-form text and play a complex multi-agent strategic game at competitive human level.
5. **A blueprint for hybrid systems.** CICERO is one of the earliest and clearest demonstrations that **LLM + classical reasoning / search** beats either alone — a thesis that continues to define frontier AI architectures today.
