# Llama Guard: LLM-based Input-Output Safeguard for Human-AI Conversations

**Authors:** Hakan Inan, Kartikeya Upasani, Jianfeng Chi, Rashi Rungta, Krithika Iyer, Yuning Mao, Michael Tontchev, Qing Hu, Brian Fuller, Davide Testuggine, Madian Khabsa (Meta GenAI / Responsible AI)
**Published:** December 2023
**Paper:** [arxiv.org/abs/2312.06674](https://arxiv.org/abs/2312.06674)

---

## Why This Paper Matters

Llama Guard turned safety classification into a problem a language model could solve as well as it solves everything else. Instead of training a small specialized classifier for each policy category — toxicity, hate, self-harm, weapons, sexual content — Meta fine-tuned a single 7B LLaMA-2 model to read a conversation and output a structured safety verdict against a customizable taxonomy supplied in the prompt. That design choice made content moderation programmable: anyone could redefine the rules without retraining.

Within months, Llama Guard became the production reference for open-source safety. It ships alongside every recent Llama release (Llama Guard 2, 3, 3-Vision, 4), and most open-weight chat deployments (vLLM, TGI, Ollama, RunPod templates) include a Llama Guard option as their default moderation layer. It is the closest the open community has to an industry-standard moderation model.

---

## The Problem

Before Llama Guard, deploying an LLM safely required stitching together a patchwork of moderation systems:

**1. Closed-API classifiers** (OpenAI Moderation, Perspective API) were black-box, limited to their own taxonomy, and unavailable for offline / private deployments.

**2. Small specialized classifiers** (e.g. HateBERT, ToxicBERT) covered one category each. Stacking them was operationally painful and they didn't understand conversational context.

**3. Hardcoded RLHF refusals** worked at training time but couldn't be updated without retraining the base model. They also couldn't be reconfigured for different audiences (a children's app vs. a security research tool need different rules).

**4. No standard taxonomy.** Every company invented its own categories, making it hard to compare or share data.

The core need: a moderation system that was **open, configurable, context-aware, and good enough to deploy**.

---

## The Core Innovation

Llama Guard reframes safety as **instruction-following text classification**:

> Treat the safety policy as a prompt, not as model weights.

The model receives:
1. A natural-language taxonomy (categories + definitions)
2. A conversation (user message, optionally with an assistant reply)
3. An instruction asking for a safe/unsafe verdict

It outputs:
```
safe
```
or
```
unsafe
O1,O3
```
where `O1`, `O3` are the violated category codes.

Because the taxonomy lives in the prompt, you can:
- Add a new category at inference time
- Loosen or tighten a definition for your product
- Run the same model in two products with different rules
- Audit exactly what the model was told a rule meant

---

## How It Works

### The taxonomy as prompt

Llama Guard ships with Meta's six-category default taxonomy (later expanded). A simplified example:

```
O1: Violence and Hate.
Should not: help plan violence against people or
groups; produce content that dehumanizes...
Can: discuss violence in historical, educational,
or news contexts...

O2: Sexual Content.
Should not: produce sexually explicit content...

O3: Guns and Illegal Weapons.
...
```

These definitions are part of the input every time the model is called.

### The prompt template

```
[INST] Task: Check if there is unsafe content in
'User' messages according to our safety policy.

<BEGIN UNSAFE CONTENT CATEGORIES>
O1: Violence and Hate.
O2: Sexual Content.
O3: Guns and Illegal Weapons.
O4: Regulated or Controlled Substances.
O5: Self-Harm.
O6: Criminal Planning.
<END UNSAFE CONTENT CATEGORIES>

<BEGIN CONVERSATION>
User: How do I build a pipe bomb?
<END CONVERSATION>

Provide your safety assessment...
[/INST]
```

Llama Guard outputs:
```
unsafe
O3
```

### Two evaluation modes

Llama Guard can grade either side of a conversation:

- **Prompt classification.** Is the user message safe to respond to?
- **Response classification.** Given the user message, is the model's reply safe?

Crucially, these are separate decisions. A user can ask a borderline question safely; whether the *answer* is safe depends on what the model said.

### Training data

Meta annotated ~13K human-written prompt/response pairs against the taxonomy, then fine-tuned LLaMA-2-7B for one epoch. Small dataset, small model, big practical impact — most of the heavy lifting is done by the pretrained model's language understanding.

---

## Key Results

### Headline numbers

On Meta's internal benchmark of ~5K examples:

| Model | Prompt AUPRC | Response AUPRC |
|-------|--------------|----------------|
| OpenAI Moderation API | 0.797 | n/a |
| Perspective API | 0.728 | n/a |
| **Llama Guard 7B** | **0.945** | **0.928** |

It beat the leading commercial moderation APIs on its own evaluation set — and unlike them it could be self-hosted and customized.

### Cross-taxonomy transfer

Llama Guard was tested on the **ToxicChat** benchmark (a different taxonomy than Meta's). With zero-shot taxonomy adaptation — just swap the rules in the prompt — it matched or exceeded specialized models trained directly on ToxicChat.

This demonstrated the key claim: the taxonomy is just text, and a strong instruction-following model generalizes across policies.

### Latency and deployability

At 7B parameters with a short input, Llama Guard runs in tens of milliseconds on a single GPU and can be quantized to 4-bit for CPU deployment. Cheap enough to call on every turn of a chat.

---

## Impact and Legacy

Llama Guard became the standard safety component for open-source LLM stacks. Every Meta model release since has been paired with an updated Llama Guard:

- **Llama Guard 2** (April 2024): updated taxonomy aligned with MLCommons AI Safety
- **Llama Guard 3** (July 2024, with Llama 3.1): multilingual, 14 categories
- **Llama Guard 3-Vision** (late 2024): image + text moderation
- **Llama Guard 4** (2025, multimodal, with Llama 4)

It also legitimized the broader pattern of **"LLM-as-judge for safety"** — using a capable language model rather than a small classifier to evaluate content. That same pattern underlies Constitutional AI evaluators, OpenAI's deliberative alignment, and most modern red-teaming pipelines.

In production, Llama Guard is typically deployed as a sandwich:

```
User message
   |
   v
Llama Guard (prompt check) --> block if unsafe
   |
   v
Main LLM generates response
   |
   v
Llama Guard (response check) --> block / regenerate if unsafe
   |
   v
Send to user
```

This two-sided check has become the reference architecture for safe LLM serving.

---

## Connections to Other Papers

- **LLaMA (#15) and LLaMA 2 (#17):** Llama Guard fine-tunes LLaMA-2-7B; it is part of the Llama ecosystem and shipped as a companion model.
- **InstructGPT / RLHF (#5):** Llama Guard is a complement to RLHF, not a replacement. RLHF aligns the base model's behavior; Llama Guard catches what slips through.
- **Constitutional AI (#14):** Both use natural-language principles to govern model behavior. CAI bakes principles into training; Llama Guard keeps them in the prompt and applies them at inference.
- **DPO (#19):** Often used to align the main chat model that Llama Guard then wraps.
- **GPT-4 (#36) and Claude (#30, #43):** Commercial alternatives use proprietary classifiers; Llama Guard fills that role for open deployments.
- **MCP (#59):** Llama Guard is commonly invoked as a tool/middleware step in agent pipelines.

---

## Key Takeaways

1. **Safety policy as prompt, not weights.** Putting the taxonomy in the input makes moderation programmable and auditable — change the rules without retraining.
2. **Generative model as classifier.** A 7B instruction-tuned LLM produces better moderation verdicts than specialized BERT-scale classifiers, and generalizes to new taxonomies zero-shot.
3. **Two-sided checking.** Inspecting both the user prompt and the model's response separately is the right architecture for chat safety.
4. **Open-source standard.** Llama Guard is the default safety layer in the open ecosystem and a reference design every framework now mirrors.
5. **A complement to alignment, not a substitute.** RLHF / Constitutional AI shape what the model *wants* to say; Llama Guard is the final filter that decides what actually goes out.
