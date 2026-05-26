# Quiet-STaR: Language Models Can Teach Themselves to Think Before Speaking

**Authors:** Eric Zelikman, Georges Harik, Yijia Shao, Varuna Jayasiri, Nick Haber, Noah D. Goodman (Stanford University, Notbad AI)

**Published:** March 2024 (COLM 2024)

**Paper Link:** https://arxiv.org/abs/2403.09629

---

## Why This Paper Matters

Quiet-STaR generalizes STaR (#81) from "learn to reason on math problems" to "learn to reason *before every single token of any text*." Instead of producing a visible chain of thought only when a question is asked, the model is taught to generate **internal, hidden thoughts** at every token position during pretraining-style text — and to keep the thoughts that make the next real token easier to predict.

This paper appeared in March 2024, a few months before OpenAI's o1 (#31) revealed its hidden chain-of-thought approach in September 2024. Quiet-STaR is widely considered the clearest *public* preview of the o1 paradigm: **train the model to think privately, in order to speak more accurately.** It blends self-supervised pretraining with reinforcement learning over latent reasoning, and it works on general web text rather than narrow math datasets.

---

## The Problem Before

STaR taught a model to reason on tasks where the correct answer is known — math, commonsense QA, arithmetic. But most useful reasoning isn't tied to a quiz with a verifiable answer. Humans think while reading a news article, writing a paragraph, debugging code, planning a sentence. Where's the equivalent training signal for a language model?

Specifically, STaR has two limits:
- **Task-bound.** It only works when you have a labeled dataset (question, answer) where you can check correctness.
- **Visible reasoning only.** The chain of thought is part of the output, taking up tokens and adding latency.

The deeper question: **can a model learn to reason on unstructured text, where the only supervision is "predict the next word"?**

---

## The Core Innovation: Token-Level Hidden Reasoning, Rewarded by Next-Token Prediction

Quiet-STaR's design has three pieces.

### 1. Generate a thought at every token position

For each token position `t` in the training text, the model produces a short **thought** — a sequence of tokens wrapped in special markers like `<|startofthought|> ... <|endofthought|>`. The thought is generated *before* the model commits to predicting the next real token at position `t+1`.

```
Text: "The capital of France is Paris."
                                      ^
                            at this position, model first thinks:
                            <|sot|> The question asks for a country's capital...
                                    France's capital is well-known... <|eot|>
                            then predicts the next real token.
```

These thoughts are **invisible** to the actual output — they live in a parallel reasoning channel.

### 2. Parallel sampling for efficiency

Naively, generating a thought at every token would make training catastrophically expensive (every token becomes 32 forward passes). Quiet-STaR uses an attention-mask trick that lets it generate many short thoughts **in parallel** across token positions in one forward pass, by carefully arranging the keys/queries so each thought attends only to its own context. This makes the whole scheme tractable.

### 3. REINFORCE on next-token prediction improvement

The training signal is brilliantly simple:

- Compute the probability the model assigns to the *actual* next token, *given the thought*.
- Compare it to the probability *without* the thought (a "no-thought" baseline).
- If the thought helped — predicted next token's probability went up — reinforce it.
- If the thought hurt or didn't help — suppress it.

The gradient update is REINFORCE-style:

```
gradient_signal = (log p(next_token | context + thought) - log p(next_token | context)) * grad_log_p(thought)
```

Thoughts that genuinely make the upcoming text more predictable get reinforced. Useless thoughts wither. The model effectively discovers, through reinforcement, what kinds of internal reasoning help it predict text.

A learnable **mixing head** controls how much weight to give the with-thought vs. without-thought predictions, so the model gracefully ignores thoughts in contexts where they don't help and leans on them where they do.

---

## How It's Trained

Starting from a pretrained base model (Mistral 7B in the paper):

1. Continue training on a general text corpus (OpenWebMath, C4).
2. At each token position, sample a 12-token thought.
3. Compute the next-token likelihood improvement.
4. Apply REINFORCE on the thought tokens, scaled by the improvement.
5. Standard language-model loss on the actual text token (unchanged).

A learned "start-of-thought" token embedding is added; the model can choose when thoughts are worth generating. Over training, thoughts become more coherent and more useful.

---

## Key Results

- **Zero-shot improvement on reasoning benchmarks**: applying Quiet-STaR-trained Mistral 7B to GSM8K (math) **without any task-specific training or prompting** improved accuracy from 5.9% to 10.9% — nearly doubling, just from training the model to think on general web text.
- **CommonsenseQA**: improved from 36.3% to 47.2% zero-shot.
- **Perplexity improved** on difficult tokens — the largest improvements were on tokens that genuinely require reasoning (numbers, conclusions, named entities in unusual contexts), not on easy-to-predict tokens.
- **Generalization**: training happened on general text (OpenWebMath, C4), but the reasoning ability transferred to held-out reasoning benchmarks. This is the key result — the model learned a *general* habit of thinking, not a task-specific skill.

---

## An Illustrative Example

Consider a token-by-token training pass on the sentence:

> *"Alice has 12 apples. She gives 3 to Bob and 4 to Carol. Alice now has ___ apples."*

At the position right before "5", the model without thoughts assigns moderate probability to several numbers (5, 6, 8, 9). With a generated thought like:

> *<sot> 12 - 3 - 4 = 5 <eot>*

the model's distribution sharpens dramatically on "5." That probability boost is the reward signal. The thought tokens that produced it get reinforced; alternative useless thoughts ("the user is asking a question about Alice...") get suppressed.

Multiplied across millions of token positions, the model gradually discovers what kinds of internal computations help on what kinds of text — arithmetic for numeric continuations, entity tracking for narrative continuations, formula recall for technical text, and so on.

---

## Why It's Hard to Scale Naively

Quiet-STaR's compute cost is the main reason it remains an academic-scale demonstration:

- **K thought tokens at every position** multiplies training FLOPs by roughly (K+1). For K=12, that's a ~13x slowdown over standard pretraining.
- **REINFORCE has high variance.** The reward signal — log-probability improvement on the next token — can be noisy, requiring large batch sizes and careful baseline subtraction.
- **Thoughts can degenerate.** Without careful regularization, the model can learn to produce "meta" thoughts that game the reward without doing real reasoning (e.g., repeating the prompt).

These cost and stability issues are why frontier-scale realizations of the same idea (presumably o1, R1) use richer supervision — task-level outcome rewards or process rewards — rather than next-token-level RL. But the conceptual recipe is the same.

---

## Impact and Legacy

Quiet-STaR's significance crystallized when OpenAI o1 (#31) launched six months later with the same fundamental shape: **a model that produces a hidden chain of thought before its visible answer, trained via RL to make the thoughts useful.** While o1's exact training details remain undisclosed, Quiet-STaR is the closest public approximation of the recipe.

Threads it influenced or anticipated:

- **OpenAI o1 (#31) and DeepSeek-R1 (#26):** Both treat reasoning as something the model does internally before answering, rewarded by outcome quality. Quiet-STaR's "reward thoughts by next-token improvement" is the analogue for unsupervised text of o1/R1's "reward reasoning by answer correctness."
- **Test-Time Compute (#50):** Quiet-STaR is the *training-side* counterpart — bake the inference-time thinking habit into the model parameters so it happens automatically.
- **Meta-CoT (#34):** Frames Quiet-STaR as part of a broader paradigm of treating reasoning chains as first-class learnable objects.
- **Latent reasoning research**: Quiet-STaR's hidden-thought concept anticipated later work on latent / continuous reasoning where thoughts don't have to be expressible as natural-language tokens.
- **Process Reward Models (#51):** Provide an alternative to Quiet-STaR's next-token-prediction reward signal — judging individual reasoning steps directly.
- **GRPO (#38) / RLVR (#39):** Modern RL algorithms that could replace Quiet-STaR's REINFORCE with more sample-efficient updates.

Quiet-STaR remains an academic-scale demonstration — Mistral 7B, not GPT-4-scale — but the *concept* is now mainstream.

---

## Conceptual Bridge: From Visible to Invisible Reasoning

It's useful to place Quiet-STaR on a spectrum of reasoning approaches:

| Approach | When does reasoning happen? | Is it visible? | Is it trained? |
|---|---|---|---|
| **Chain-of-Thought (#09)** | At inference, prompted | Yes (in output) | No |
| **STaR (#81)** | At inference + training | Yes (in output) | Yes (fine-tune) |
| **Quiet-STaR** | At every token of text | No (hidden tokens) | Yes (REINFORCE) |
| **o1 / R1 (#31, #26)** | At inference (long preamble) | Hidden from user, visible internally | Yes (large-scale RL) |
| **Latent reasoning** | Inside the residual stream | Not even tokenized | Yes (specialized objectives) |

Quiet-STaR sits in the middle of this spectrum: reasoning is *internal* (not part of the visible output) but still expressed as tokens (not pure latent vectors). This makes it interpretable — researchers can read the thoughts a Quiet-STaR model generates and see what it's "thinking" — while still being efficient and learnable.

---

## Connections to Other Papers

- **STaR (#81):** Direct predecessor. STaR is "outcome-supervised reasoning on QA tasks"; Quiet-STaR is "next-token-supervised reasoning on arbitrary text."
- **Chain-of-Thought (#09):** Quiet-STaR makes the chain of thought internal and automatic rather than prompted and visible.
- **OpenAI o1 (#31):** The frontier-scale realization of Quiet-STaR's central idea — hidden reasoning trained for output quality.
- **DeepSeek-R1 (#26):** Open analogue of o1; explicitly uses RL on reasoning traces, conceptually adjacent to Quiet-STaR's REINFORCE on thoughts.
- **Test-Time Compute (#50):** Quiet-STaR vs. test-time compute is "train the model to think automatically" vs. "force it to think at inference"; they compose well.
- **Meta-CoT (#34):** Theoretical framework that subsumes both STaR and Quiet-STaR.
- **Mistral 7B (#72):** The base model Quiet-STaR builds on.
- **Process Reward Models (#51) / GRPO (#38) / RLVR (#39):** Provide alternative reward signals and RL machinery for the same family of methods.

---

## Key Takeaways

- **Reasoning can be trained on arbitrary text**, not just labeled question-answer pairs — the supervision signal is whether thinking improves next-token prediction.
- **Hidden thoughts + REINFORCE**: produce a candidate thought before each token, reward it by how much it lifts the probability of the actual continuation.
- **Parallel attention masking** makes per-token reasoning tractable to train, despite the naive O(N*T) cost.
- **General reasoning ability emerged** from generic text training: GSM8K and CommonsenseQA improved zero-shot without any task-specific data.
- **The clearest public preview of the o1 paradigm** — train a model to think privately so it answers more accurately, before "reasoning models" became the dominant frontier paradigm.
