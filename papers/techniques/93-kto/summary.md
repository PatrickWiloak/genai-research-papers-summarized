# KTO: Model Alignment as Prospect Theoretic Optimization

**Authors:** Kawin Ethayarajh, Winnie Xu, Niklas Muennighoff, Dan Jurafsky, Douwe Kiela (Stanford / Contextual AI)
**Published:** February 2024
**Paper:** [arxiv.org/abs/2402.01306](https://arxiv.org/abs/2402.01306)

---

## Why This Paper Matters

KTO — Kahneman-Tversky Optimization — quietly solved one of the most annoying practical bottlenecks in alignment: needing paired preference data. DPO and RLHF both require pairs of responses (A is better than B) for every prompt, which is expensive to collect and even harder to source from real production traffic. KTO showed you can align a model using only **per-response thumbs-up / thumbs-down labels**, the kind of feedback that almost every chat product already collects.

The result is an alignment loss derived from **prospect theory** — Kahneman and Tversky's Nobel-winning model of how humans actually weigh gains and losses. Empirically, KTO matches or exceeds DPO across model sizes from 1B to 30B parameters, while being trainable on data shapes that DPO simply cannot use. That made it the go-to method when teams want to align models on operational feedback streams rather than expensively curated preference datasets.

---

## The Problem

Preference learning, as standardized by InstructGPT and DPO, looks like this:

```
For each prompt x, collect:
    y_chosen  (the response humans preferred)
    y_rejected (the response humans dispreferred)
```

This data shape has real costs:

**1. Pairs are expensive.** Annotators have to read two responses and decide which is better — twice the work of a binary judgment.

**2. Real product feedback isn't paired.** When a user clicks thumbs-up on a ChatGPT message, there isn't a matching rejected response. The data you actually have is **single responses with a quality signal**, not pairs.

**3. Pairwise data is brittle.** If both responses are bad (or both good), the signal is noisy. Annotators disagree more on hard pairs.

**4. Class imbalance is hidden.** RLHF and DPO assume roughly balanced preference data. Real feedback streams are heavily skewed — many more thumbs-up than thumbs-down, or vice versa.

The question: can we align models using the **binary, unpaired** feedback that products naturally produce?

---

## The Core Innovation

KTO replaces DPO's pairwise log-likelihood loss with a loss derived from prospect theory. Two ideas come together:

### Idea 1: Prospect theory's value function

Kahneman and Tversky observed that humans:
- Feel losses more strongly than gains (**loss aversion**)
- Have **diminishing sensitivity** — the difference between $100 and $200 feels bigger than $1,100 vs $1,200
- Evaluate outcomes relative to a **reference point**, not in absolute terms

Their value function looks like:

```
v(x) = (x - reference)^alpha    if x >= reference   (gain)
v(x) = -lambda * (reference - x)^alpha   if x < reference  (loss)
```

with `lambda > 1` (losses hurt more than equivalent gains).

### Idea 2: Map this onto the DPO reward

DPO already gives a closed-form implicit reward:

```
r(x, y) = beta * log( pi(y|x) / pi_ref(y|x) )
```

KTO uses this same reward, but plugs it through a prospect-theoretic value function with the **expected reward over the data** as the reference point. For a desirable example you maximize the gain; for an undesirable example you minimize the loss — and you don't need a paired counterpart.

---

## How It Works

### The KTO loss

For each example you have a prompt `x`, a response `y`, and a binary label "desirable" (thumbs-up) or "undesirable" (thumbs-down). Define:

```python
r(x, y) = beta * (log pi(y|x) - log pi_ref(y|x))
z_ref   = E[ KL(pi || pi_ref) ]   # estimated over the batch
```

Then:

```python
if y is desirable:
    loss = 1 - sigmoid( beta * (r(x, y) - z_ref) )
else:
    loss = 1 - sigmoid( beta * (z_ref - r(x, y)) )
```

In plain terms:
- **Desirable response:** push its reward above the reference KL — but with diminishing returns once it's well above.
- **Undesirable response:** push its reward below the reference — and the loss curve makes this more painful than the gain from a desirable example (loss aversion).

### Asymmetric weighting for imbalanced data

KTO introduces two hyperparameters, `lambda_D` and `lambda_U`, that weight desirable vs. undesirable examples. If you have 10x more thumbs-up than thumbs-down, you increase `lambda_U` so each thumbs-down still matters. This is impossible to do cleanly with DPO since preferences are paired by construction.

A good rule of thumb from the paper:

```
lambda_D * n_D  ~=  lambda_U * n_U   (within roughly 4:3 ratio)
```

### The training recipe

```
1. Start from an SFT model pi_ref.
2. Collect any binary-labeled data: (prompt, response, +/-)
3. Train with the KTO loss above, beta ~= 0.1
4. No reward model. No paired data. No PPO.
```

That's the whole method.

---

## Key Results

### Matches or beats DPO at every scale

The paper aligned Pythia (1B-12B) and Llama (7B/13B/30B) models with both methods on the same prompts:

| Model | DPO win rate | KTO win rate |
|-------|--------------|--------------|
| Pythia-1.4B | baseline | better |
| Pythia-6.9B | baseline | comparable |
| Pythia-12B | baseline | better |
| Llama-7B | baseline | comparable |
| Llama-13B | baseline | better |
| Llama-30B | baseline | better |

KTO is competitive everywhere and pulls ahead at larger model sizes — even though it uses *strictly weaker* supervision (binary labels vs. preference pairs).

### Works even when you discard half the data

The authors took preference pairs and threw away the "chosen" half — keeping only "rejected" examples — and KTO still trained a strong model. DPO can't do this at all.

### Robust to extreme class imbalance

They tested ratios up to 10:1 desirable:undesirable. KTO held up; DPO requires balanced pairs.

### Production-friendly data

KTO is the first major alignment method designed for the data shape that real chat products generate: a stream of conversations, some marked thumbs-up, some thumbs-down, most unmarked. You can throw the labeled subset directly at KTO.

---

## Impact and Legacy

KTO was rapidly adopted by the open-source alignment community. The Hugging Face `trl` library added a `KTOTrainer` within months, alongside `DPOTrainer`. It became one of the standard members of the post-DPO family of alignment losses:

- **DPO** (paired preferences)
- **IPO** (identity preference optimization — addresses DPO overfitting)
- **KTO** (binary feedback)
- **ORPO** (reference-free; combines SFT and preference)
- **SimPO** (length-normalized reference-free)

KTO's particular niche: alignment from **production telemetry**. If your product collects thumbs-up/thumbs-down (or any binary quality signal — completion vs. abandonment, copy vs. discard, retry vs. accept), KTO turns it directly into model improvement without an annotation pipeline.

Beyond the loss itself, the paper introduced an important framing: **alignment methods make implicit assumptions about how humans evaluate outcomes**. DPO assumes the Bradley-Terry preference model. KTO assumes prospect theory. Different assumptions yield different losses, and matching the assumption to the actual feedback process matters. This sparked a broader research thread on "human-aware loss functions" (HALOs).

---

## Connections to Other Papers

- **InstructGPT / RLHF (#5):** The original three-stage alignment pipeline that KTO simplifies.
- **DPO (#19):** Direct predecessor. KTO inherits the implicit-reward trick but trades pairwise data for binary data.
- **Constitutional AI (#14):** Generates AI feedback that can be either pairwise (for DPO) or binary (for KTO).
- **GRPO (#38):** A different post-RLHF method (used in DeepSeek-R1) — also reward-model-free, but designed for verifiable reasoning rather than human preferences.
- **RLVR (#39):** Like KTO, lets you do RL-style alignment without a learned reward model — but uses verifiable rewards (math, code) instead of human binary feedback.
- **Llama Guard (#92):** Can serve as the binary judge that produces KTO training labels at scale.
- **Generative Agents (#58):** Systems collecting interaction logs that could be turned into KTO training data.

---

## Key Takeaways

1. **Binary feedback is enough.** You don't need preference pairs to align a model — thumbs-up/thumbs-down works, and that's the data products actually have.
2. **Prospect theory as inductive bias.** Penalizing bad responses more than rewarding good ones (loss aversion) maps cleanly onto how humans judge AI output.
3. **Matches DPO with weaker supervision.** Same or better quality, with cheaper and more available data.
4. **Handles class imbalance directly.** The `lambda_D` / `lambda_U` knobs let you train on lopsided real-world feedback streams.
5. **Unlocks alignment from telemetry.** Any product with a quality signal — chat thumbs, code accepts, search clicks — can now feed alignment directly with KTO.
