---
title: "FLAN: Finetuned Language Models Are Zero-Shot Learners (Instruction Tuning)"
slug: "80-flan"
number: 80
category: "techniques"
authors: "Jason Wei, Maarten Bosma, Vincent Y. Zhao, Kelvin Guu, Adams Wei Yu, Brian Lester, Nan Du, Andrew M. Dai, Quoc V. Le (Google Research); scaling follow-up by Hyung Won Chung, Le Hou, Shayne Longpre et al."
published: "September 2021 (ICLR 2022); Scaling Instruction-Finetuned Language Models, October 2022"
year: 2021
url: "https://arxiv.org/abs/2109.01652"
tags: ["instruction-tuning", "pretraining", "scaling"]
---

# FLAN: Finetuned Language Models Are Zero-Shot Learners (Instruction Tuning)

**Authors:** Jason Wei, Maarten Bosma, Vincent Y. Zhao, Kelvin Guu, Adams Wei Yu, Brian Lester, Nan Du, Andrew M. Dai, Quoc V. Le (Google Research); scaling follow-up by Hyung Won Chung, Le Hou, Shayne Longpre et al.
**Published:** September 2021 (ICLR 2022); Scaling Instruction-Finetuned Language Models, October 2022
**Papers:** [FLAN arxiv.org/abs/2109.01652](https://arxiv.org/abs/2109.01652) | [Flan-T5/Flan-PaLM arxiv.org/abs/2210.11416](https://arxiv.org/abs/2210.11416)

---

## Why This Matters

FLAN introduced **instruction tuning**: the step that turns a text-completion engine into something that follows directions. It is the "IT" in every instruction-tuned model, the middle stage of the pretrain / instruction-tune / align pipeline, and the reason you can ask a model to do a task it has never seen and get a sensible attempt.

- **Zero-shot beat [GPT-3](../../language-models/04-gpt3-few-shot-learners/summary.md) 175B on 20 of 25 benchmarks** using a 137B model - not by being bigger, but by being instruction-tuned.
- **Established generalization across task types**, not just across examples. Train on translation and sentiment; improve at natural language inference.
- **Flan-T5 became the workhorse** open model of 2022-2023, and the Flan collection is still mixed into modern post-training data.
- **It is the cheap half of alignment.** Instruction tuning is supervised fine-tuning on a few thousand to a few hundred thousand examples - orders of magnitude cheaper than pretraining and simpler than RLHF.

**The insight:** GPT-3 showed that a big enough model can do new tasks from a few in-context examples. But zero-shot performance - just describing the task in words - was much weaker, because a raw pretrained model has no reason to interpret "Translate this to French:" as a command rather than as text to continue. Fine-tune it on many tasks *phrased as instructions*, and it learns the meta-skill of treating a natural-language description as something to obey.

---

## The Problem: Base Models Do Not Follow Instructions

A pretrained language model completes text. Given "Translate to French: Hello", a reasonable completion is another example line, or a discussion of translation, or the translation - the model has no notion that one of these is what you wanted.

GPT-3's few-shot prompting worked around this by *showing* the model the pattern. That is effective and awkward: it eats context, requires curating examples, and is brittle to example order and format. Zero-shot prompting is what users actually want, and GPT-3's zero-shot numbers were far below its few-shot numbers.

The question FLAN asked: can a model be taught, once, that natural-language task descriptions are commands to execute - in a way that transfers to task types it never saw during that training?

---

## The Core Innovation

Take many existing NLP datasets, rewrite each as natural-language instructions with multiple templates, fine-tune on a mixture, then evaluate on **held-out task clusters**.

```
  62 datasets grouped into 12 TASK CLUSTERS:
    NLI, commonsense, sentiment, paraphrase, closed-book QA,
    reading comprehension, coreference, translation,
    struct-to-text, summarization, misc, reading comp w/ commonsense

  Each dataset gets ~10 hand-written instruction TEMPLATES,
  including some deliberately "turned around" to add diversity:

    Original (sentiment classification):
      text: "The movie was a slog."   label: negative

    Template 1: "Is the following review positive or negative?
                 The movie was a slog."
    Template 2: "The movie was a slog.
                 Did the reviewer like the movie?"
    Template 3: "Write a negative movie review."     <- inverted task

  TRAINING: all clusters EXCEPT the evaluation cluster
  EVALUATION: the held-out cluster, ZERO-SHOT

  -> measures generalization to unseen TASK TYPES,
     not just unseen examples.
```

Holding out entire clusters is the methodological core. It is what makes the result about generalization rather than about memorizing task formats.

---

## Key Components Explained

### 1. Instruction Templates
**What it does:** Teaches the mapping from natural-language description to behavior.
**How it works:** Ten diverse phrasings per dataset. Diversity of phrasing is what prevents the model from latching onto one surface form. Later analysis (the Flan Collection paper) confirmed template diversity is one of the biggest levers on final quality.

### 2. Held-Out Cluster Evaluation
**What it does:** Makes the generalization claim credible.
**How it works:** When evaluating on natural language inference, every NLI dataset is removed from training. The model has never been instruction-tuned on anything resembling the eval task. Improvements therefore reflect a transferable skill.

### 3. Scale Dependence
**What it does:** Explains why instruction tuning was not discovered earlier.
**How it works:** In the paper's ablations, instruction tuning *hurt* models below roughly 8B parameters on held-out tasks and helped substantially above that. Small models appear to use their limited capacity to fit the training tasks, at the expense of general ability. This is an [emergent](../81-emergent-abilities/summary.md) behavior, and it is why the technique needed the large-model era to appear.

### 4. What Flan-T5 Added (2022 follow-up)
**What it does:** Scaled the recipe and added reasoning.
**How it works:** The follow-up scaled to **1,836 tasks**, added chain-of-thought examples to the mixture, and added few-shot as well as zero-shot templates. Findings:
- **Task count keeps helping**, with diminishing but positive returns into the thousands.
- **Including [chain-of-thought](../09-chain-of-thought/summary.md) data is essential** for preserving reasoning ability - instruction tuning *without* CoT data actively damages the model's step-by-step reasoning.
- **Mixing zero-shot and few-shot templates** improves both, rather than trading off.
- Flan-PaLM 540B set a then-state-of-the-art MMLU score, and **Flan-T5-XL/XXL outperformed much larger non-instruction-tuned models** on many tasks at a fraction of the size.

---

## Key Results

- FLAN 137B zero-shot beat GPT-3 175B zero-shot on **20 of 25** evaluated datasets, and beat GPT-3 *few-shot* on several including ANLI, RTE, BoolQ, and StoryCloze.
- Performance improved monotonically with the number of task clusters used in training.
- Instruction tuning helped large models and hurt small ones - a clean scale-dependent effect.
- Flan-T5-XXL (11B) matched or beat far larger models on many benchmarks; Flan-PaLM 540B improved substantially over PaLM on MMLU and on chain-of-thought reasoning benchmarks.
- The Flan Collection ablations later identified template diversity, task balancing, and CoT inclusion as the three main quality drivers.

---

## Why This Was Revolutionary

- **Created the middle stage of the modern pipeline.** Pretrain, instruction-tune, then align. Every assistant model does this.
- **Showed cross-task generalization is real and trainable**, converting "follow instructions" from an emergent curiosity into an engineering target.
- **Delivered capability without scale.** An 11B Flan-T5 beating a 175B base model made a durable point about post-training's leverage, and made strong models accessible to teams without frontier compute.
- **Published everything.** Datasets, templates, and checkpoints. The Flan collection is one of the most reused public assets in NLP.

---

## Real-World Impact

- **Flan-T5** became the default open encoder-decoder model for classification, extraction, and structured tasks - small, permissively licensed, and strong on exactly the tasks enterprises have.
- **Every instruction-tuned model uses this stage.** [InstructGPT](../../language-models/05-instructgpt-rlhf/summary.md)'s supervised fine-tuning step, [LLaMA 2](../../language-models/17-llama2/summary.md)-Chat's SFT, and Qwen and Mistral instruct variants are all instruction tuning; RLHF or [DPO](../../language-models/19-dpo/summary.md) comes after.
- **The Flan mixture is still used** as a component of post-training data blends years later, particularly for preserving broad task ability while training on narrower data.
- **The "add CoT data or lose reasoning" finding** is standard practice: reasoning traces are now a required ingredient in any SFT mix.
- **[Self-Instruct](../79-self-instruct/summary.md) is the synthetic answer to FLAN's human-curated question** - same target, different data source.

---

## Key Takeaways for Practitioners

1. **Instruction tuning is the highest-leverage cheap step.** A few thousand good examples change model behavior more than any amount of prompt engineering.
2. **Vary your phrasings.** Ten templates per task type is a good target; single-template SFT produces models that break when users phrase things differently.
3. **Always include reasoning traces.** SFT on answer-only data measurably degrades chain-of-thought ability. This is the most commonly repeated FLAN mistake.
4. **Mix zero-shot and few-shot formats.** You get both, cheaply.
5. **Do not instruction-tune a very small model on many tasks** and expect generalization - below a few billion parameters, expect the model to trade general ability for training-task fit.
6. **Flan-T5 is still a strong, cheap baseline** for classification and extraction where a frontier API is overkill.

---

## Limitations & Future Directions

- **Academic task distribution.** FLAN's tasks come from NLP benchmarks, not from what users actually ask. This is precisely the gap InstructGPT's human-written prompts and Self-Instruct's generated ones addressed.
- **No preference signal.** Instruction tuning teaches format and task-following, not helpfulness, harmlessness, or tone. RLHF/DPO exist because SFT alone does not produce a good assistant.
- **Template sensitivity remains** - reduced, not eliminated.
- **Human curation does not scale**, which is why synthetic instruction generation displaced it for volume, with curated data reserved for quality-critical slices.
- **Alignment tax.** Heavy instruction tuning can reduce diversity and creativity relative to the base model, a trade-off that is still being negotiated in every post-training recipe.

---

## Further Reading

- **FLAN:** [arxiv.org/abs/2109.01652](https://arxiv.org/abs/2109.01652)
- **Scaling Instruction-Finetuned Language Models (Flan-T5/Flan-PaLM):** [arxiv.org/abs/2210.11416](https://arxiv.org/abs/2210.11416)
- **The Flan Collection (design ablations):** [arxiv.org/abs/2301.13688](https://arxiv.org/abs/2301.13688)
- **T0 (concurrent work on prompted multitask training):** [arxiv.org/abs/2110.08207](https://arxiv.org/abs/2110.08207)

## Citation

```bibtex
@inproceedings{wei2022finetuned,
  title={Finetuned Language Models Are Zero-Shot Learners},
  author={Wei, Jason and Bosma, Maarten and Zhao, Vincent Y. and Guu, Kelvin and Yu, Adams Wei and Lester, Brian and Du, Nan and Dai, Andrew M. and Le, Quoc V.},
  booktitle={International Conference on Learning Representations},
  year={2022}
}
```

<!-- related:start -->

---

## Related in This Collection

- [Language Models are Few-Shot Learners (GPT-3)](../../language-models/04-gpt3-few-shot-learners/summary.md)
- [Training Language Models to Follow Instructions with Human Feedback (InstructGPT)](../../language-models/05-instructgpt-rlhf/summary.md)
- [Chain-of-Thought Prompting Elicits Reasoning in Large Language Models](../../techniques/09-chain-of-thought/summary.md)
- [LLaMA 2: Open Foundation and Fine-Tuned Chat Models](../../language-models/17-llama2/summary.md)
- [Direct Preference Optimization (DPO): Your Language Model is Secretly a Reward Model](../../language-models/19-dpo/summary.md)
- [Qwen3: Technical Report](../../language-models/28-qwen3/summary.md)
- [Self-Instruct: Aligning Language Models with Self-Generated Instructions](../../techniques/79-self-instruct/summary.md)
- [PaLM: Scaling Language Modeling with Pathways](../../language-models/94-palm/summary.md)

<!-- related:end -->
