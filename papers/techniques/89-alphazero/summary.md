# Mastering Chess and Shogi by Self-Play with a General Reinforcement Learning Algorithm

**Authors:** David Silver, Thomas Hubert, Julian Schrittwieser, Ioannis Antonoglou, Matthew Lai, Arthur Guez, Marc Lanctot, Laurent Sifre, Dharshan Kumaran, Thore Graepel, Timothy Lillicrap, Karen Simonyan, Demis Hassabis (DeepMind)
**Published:** December 2017 (later in Science, 2018)
**Paper:** [arxiv.org/abs/1712.01815](https://arxiv.org/abs/1712.01815)

---

## Why This Paper Matters

AlphaZero is one of the most important reinforcement learning papers ever written. Starting from nothing but the rules of the game — no human games, no opening books, no endgame tables, no positional heuristics — a single algorithm achieved superhuman performance in Chess, Shogi, and Go within 24 hours of self-play training. The same algorithm. The same architecture. Three games with utterly different strategic textures.

The result was philosophically and practically stunning. Philosophically, it demonstrated that mastery in domains humans had spent centuries refining (chess theory dates to the 15th century) could be re-derived from scratch by a learning algorithm, often arriving at strategies that startled grandmasters with their novelty. Practically, it cemented the recipe of **deep neural networks + Monte Carlo Tree Search + self-play** as a general blueprint for sequential decision-making.

For modern generative AI, AlphaZero is the conceptual ancestor of every "model-generates-its-own-training-data-and-improves" loop now used in RLHF, RLVR, DeepSeek-R1, and o1-style reasoning systems. The notion that an AI can improve indefinitely by playing against itself, with no external supervision beyond a verifier (here, the game's outcome), is the seed of the entire test-time-compute and self-improving-AI research agenda.

---

## The Problem

By 2017, DeepMind had already produced AlphaGo (2015) and AlphaGo Zero (2017). AlphaGo had beaten Lee Sedol in 2016 in a watershed moment for AI. AlphaGo Zero had then improved on AlphaGo by training entirely from self-play, with no human game data. Both were specialized to Go.

The chess community had a separate champion: Stockfish, a hand-engineered system descending from decades of work on alpha-beta search, opening books, and human-tuned evaluation functions. Stockfish was the strongest chess engine in the world. Shogi (Japanese chess) had its own champion, Elmo, similarly hand-engineered.

The questions AlphaZero asked:

1. Could the AlphaGo Zero recipe — self-play, no human data, a single neural net guiding MCTS — generalize beyond Go?
2. Could it compete with decades of hand-engineering in chess and shogi without using any domain-specific knowledge beyond the rules of the game?
3. How quickly could it learn?

---

## The Core Innovation

AlphaZero is conceptually simple. A single neural network takes a board state and outputs two things:

- A **policy** — a probability distribution over legal moves.
- A **value** — a scalar estimate of which side will win.

The network is used inside a **Monte Carlo Tree Search (MCTS)** procedure that explores possible move sequences. When deciding a move, the agent runs many MCTS simulations from the current position, biased by the network's policy prior and constrained by the network's value estimate. The most-visited move is then played.

The agent learns by **self-play**:

1. Play a game against yourself using MCTS guided by the current network.
2. Store the (state, MCTS-derived policy, eventual game outcome) triplets from the game.
3. Train the network to predict the MCTS policy and the game outcome.
4. Repeat.

The genius is the feedback loop: MCTS produces *stronger* policy targets than the raw network (because search refines the network's intuition), and training the network on these stronger targets makes both the network and subsequent MCTS searches stronger.

```
   Network -> guides MCTS -> better moves
                                |
                                v
                          self-play games
                                |
                                v
                training data (policy + value targets)
                                |
                                v
                       updated network -----+
                                ^           |
                                +-----------+
```

### What AlphaZero Removed vs. AlphaGo

AlphaZero is *simpler* than AlphaGo, deliberately. It removes:

- Human game data (AlphaGo bootstrapped from expert games).
- Game-specific input features (it sees only the raw board).
- Game-specific MCTS adjustments (it uses one general MCTS variant).
- Hand-tuned evaluation features.
- Domain-specific symmetries except those trivially shared by the rules.

What remains is a clean recipe: rules of the game + neural network + MCTS + self-play.

---

## How It Works

### The Network

A residual convolutional neural network (ResNet) takes the current board state (and a few previous boards, for history) as input. Two heads:

- **Policy head:** softmax over moves.
- **Value head:** scalar in [-1, 1] estimating the win/loss expectation.

### MCTS

For each move decision, the agent runs ~800 simulations. Each simulation:

1. Selects child nodes in the search tree using PUCT — a formula that balances exploration (try new moves) and exploitation (move toward high-value branches), biased by the network's policy prior.
2. When it reaches a leaf node, asks the network for a policy + value estimate.
3. Backs up the value estimate up the tree, updating visit counts and statistics.

After 800 simulations, the agent picks a move proportional to visit counts (with some exploration noise during training).

### The Training Loop

Many self-play games are played in parallel on TPUs. Each completed game contributes training examples of the form `(state, MCTS_policy, final_outcome)`. The network is trained by gradient descent on a loss that combines:

- Cross-entropy between predicted policy and MCTS-derived policy.
- Mean-squared error between predicted value and actual game outcome.
- L2 weight decay.

There is no replay buffer that lasts forever; training is on recent games. Network checkpoints are periodically evaluated by playing against the current champion checkpoint; only stronger checkpoints are promoted.

### Compute

The headline result used 5,000 TPUs for self-play data generation and 64 TPUs for training, for several hours. By modern standards (2024+) this is a small amount of compute, but the algorithmic elegance is the lasting contribution.

---

## Key Results

After 24 hours of self-play training, AlphaZero:

- **Chess:** Defeated Stockfish 8 in a 100-game match (28 wins, 72 draws, 0 losses), playing with a time control of 1 minute per move.
- **Shogi:** Defeated Elmo in a 100-game match (90 wins, 8 losses, 2 draws).
- **Go:** Defeated AlphaGo Zero in a 100-game match (60 wins, 40 losses), despite AlphaGo Zero having been the strongest Go-playing agent in the world.

Qualitative observations were equally important. AlphaZero's chess play was widely described as "creative," "intuitive," "human-like" by grandmasters. It favored unusual sacrifices, voluntarily ceded material for long-term positional advantage, and reinvented (or in some cases discarded) centuries of human opening theory. Stockfish, by contrast, exhibited the brute-force, materialistic style typical of hand-engineered engines.

Compute efficiency was striking too: AlphaZero examined ~60,000 positions per second in chess, vs. Stockfish's ~60 million. The MCTS-guided neural net was three orders of magnitude more selective than alpha-beta search, and still played better.

---

## Impact and Legacy

AlphaZero's influence has been enormous:

- **MuZero (2019)** — DeepMind's follow-up that removed even the need to know the rules of the game. MuZero learns a *model* of the dynamics from interaction, then plans inside that learned model. It mastered Atari, chess, shogi, and Go from the same algorithm.
- **Stockfish itself** — adopted a neural network evaluation function (NNUE) inspired by the AlphaZero results, and is now stronger than ever.
- **Modern chess engines** — Leela Chess Zero is an open-source AlphaZero clone that competes with Stockfish at the top of computer chess.
- **AlphaTensor, AlphaDev, AlphaEvolve** — DeepMind's progression of "AlphaZero applied to algorithmic discovery" projects, where the "game" is "find a faster matrix multiplication algorithm" or "find a better sorting routine."
- **RLHF / RLVR / o1 / DeepSeek-R1** — The conceptual lineage is direct: take a model, let it generate many candidate outputs, use a verifier to score them, train the model on its better outputs. AlphaZero pioneered this loop in games; modern reasoning models port it to language with verifiers like math correctness or unit tests.
- **rStar-Math and process reward models** — explicitly use MCTS-style search guided by LLM policy and value heads, an AlphaZero pattern transplanted into mathematical reasoning.

Perhaps more than any other paper of its era, AlphaZero shifted expectations about what was achievable. It made "tabula rasa learning to superhuman performance in 24 hours" a credible benchmark, not a fantasy.

---

## Connections to Other Papers

- **AlphaFold 2 (#87) and AlphaFold 3 (#88)** — sibling DeepMind systems. Different domain, same underlying philosophy: deep networks with strong domain-appropriate inductive biases and learned policies.
- **AlphaGeometry (#61) and AlphaEvolve (#62)** — direct intellectual descendants. Each uses a generative model in concert with a verifier or search procedure, with self-improvement loops borrowed from AlphaZero.
- **RLHF / InstructGPT (#5)** — RLHF is "AlphaZero with humans as the value function and language as the game." Both train a policy network with feedback from a value signal generated by a separate process.
- **DPO (#19)** — direct preference optimization, an alternative to PPO-based RLHF. Conceptually still in the AlphaZero family of "learn from pairwise comparisons of self-generated outputs."
- **DeepSeek-R1 (#26) and o1 (#31)** — apply the AlphaZero self-improvement loop with RLVR (RL from verifiable rewards) to language reasoning. The game is "produce a correct answer"; the verifier is a math checker or unit test.
- **RLVR (#39)** — formalizes the verifier-driven RL training loop that AlphaZero implicitly pioneered.
- **Self-Consistency (#85) and Tree of Thoughts (#25)** — inference-time analogues of AlphaZero's MCTS, with the LLM in the role of both policy and value network.
- **rStar-Math (#35)** — explicit MCTS over LLM-generated math reasoning, the most direct AlphaZero homage in the modern LLM literature.
- **Process Reward Models (#51)** — provide the step-level value signal that AlphaZero's value head provides in board games.

---

## Key Takeaways

1. **One algorithm, many games.** A single recipe (deep network + MCTS + self-play, no human data) mastered Chess, Shogi, and Go to superhuman level. The generality was as important as the strength.
2. **Self-play closes the data loop.** AlphaZero's training data is generated by AlphaZero itself, with the game's win/loss outcome as the only external signal. This is the foundational pattern for modern self-improving AI systems.
3. **MCTS amplifies neural intuition.** The network provides priors and value estimates; MCTS turns those priors into actually-stronger move choices, which then become training targets. The feedback loop produces compounding improvement.
4. **Removing human knowledge can help, not hurt.** AlphaZero's discoveries occasionally contradicted centuries of accumulated chess theory, with the algorithm proving right and the humans wrong. Strong learning + strong search can sometimes find better solutions than hand-engineering.
5. **The conceptual ancestor of modern reasoning RL.** RLHF, RLVR, o1, DeepSeek-R1, AlphaEvolve, AlphaGeometry, rStar-Math, process reward models — all descend, directly or indirectly, from AlphaZero's self-play-plus-verifier blueprint. If you want to understand why the field believes AI systems can keep getting smarter by training on their own outputs, this is the paper to read.
