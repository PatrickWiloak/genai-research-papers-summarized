---
title: "Textbooks Are All You Need (phi-1)"
slug: "135-phi-1-textbooks"
number: 135
category: "language-models"
authors: "Suriya Gunasekar, Yi Zhang, Jyoti Aneja, Caio César Teodoro Mendes, Allie Del Giorno, Sivakanth Gopi, Mojan Javaheripi, Piero Kauffmann, Gustavo de Rosa, Olli Saarikivi, Adil Salim, Shital Shah, Harkirat Singh Behl, Xin Wang, Sébastien Bubeck, Ronen Eldan, Adam Tauman Kalai, Yin Tat Lee, Yuanzhi Li (Microsoft Research)"
published: "June 2023 (arXiv preprint)"
year: 2023
url: "https://arxiv.org/abs/2306.11644"
tags: ["language-model", "synthetic-data", "code"]
---

# Textbooks Are All You Need (phi-1)

**Authors:** Suriya Gunasekar, Yi Zhang, Jyoti Aneja, Caio César Teodoro Mendes, Allie Del Giorno, Sivakanth Gopi, Mojan Javaheripi, Piero Kauffmann, Gustavo de Rosa, Olli Saarikivi, Adil Salim, Shital Shah, Harkirat Singh Behl, Xin Wang, Sébastien Bubeck, Ronen Eldan, Adam Tauman Kalai, Yin Tat Lee, Yuanzhi Li (Microsoft Research)
**Published:** June 2023 (arXiv preprint)
**Paper:** [arxiv.org/abs/2306.11644](https://arxiv.org/abs/2306.11644)

---

## Why This Matters

phi-1 is **the paper that made "data quality beats data quantity" a serious research program** for language models. A 1.3 billion parameter model, trained for about four days on eight A100 GPUs, scored **50.6 percent pass@1 on HumanEval**, ahead of open code models more than ten times its size. It did this not with a new architecture but with a new kind of training data: web code filtered for "educational value" plus synthetic textbooks and exercises written by GPT-3.5.

- **Small model, big result.** 1.3B parameters and roughly 7B unique training tokens, in a field where strong code models had 15B parameters and trained on hundreds of billions of tokens.
- **Synthetic data as a first-class ingredient.** Before phi-1, synthetic data was mostly used for fine-tuning. Here a large share of the training signal was written by another model.
- **It started a product line.** phi-1 led to phi-1.5, Phi-2, Phi-3 and Phi-4, Microsoft's family of "small language models", and the textbook-quality idea spread across the field.
- **It also started an argument.** Critics asked whether a model trained on GPT-written exercises that look like benchmark problems was learning to code or learning the test. That debate about contamination is as much a part of the paper's legacy as the result itself.

**The insight:** [scaling laws](../../techniques/12-scaling-laws/summary.md) and [Chinchilla](../../techniques/18-chinchilla/summary.md) treated data as a quantity to match to parameters. phi-1 argued that the quality of each token matters enough to move the curve itself: clean, instructive, self-contained examples teach far more per token than the average snippet scraped from GitHub.

---

## The Problem: Most Web Code Is a Bad Teacher

Code models of the time (StarCoder, CodeGen, [Codex](../56-codex/summary.md)) trained on huge dumps of public repositories. The authors argue that most of that code is poor teaching material:

- **Not self-contained.** Files depend on other modules or on configuration the model never sees.
- **Boilerplate-heavy.** Configuration files, generated code and repeated patterns carry little algorithmic content.
- **Poorly explained.** Real code rarely says what it is trying to do or why.
- **Skewed in difficulty.** The interesting logic is buried among trivial glue code.

A person learning to program from a textbook gets clear explanations, graded examples and exercises. A model trained on raw GitHub gets the opposite. The bet was that giving a small model the textbook experience would make up for its size.

---

## The Core Innovation

Build the training set the way you would build a course: filter real material for teaching value, write new explanatory material, then practise on exercises.

```
                 phi-1 training data (Python only)

  PRETRAINING: "CodeTextbook"  (~7B tokens)
  +---------------------------------------------------------+
  | Filtered web code  ~6B tokens                           |
  |   The Stack + StackOverflow, kept only where a          |
  |   classifier predicts high "educational value"          |
  |                                                         |
  | Synthetic textbooks  <1B tokens                         |
  |   GPT-3.5-written explanations interleaved with code    |
  +---------------------------------------------------------+
            | ~8 passes, ~50B tokens seen, 4 days on 8 A100s
            v
        phi-1-base  (29% HumanEval)

  FINE-TUNING: "CodeExercises"  (~180M tokens)
  +---------------------------------------------------------+
  | GPT-3.5-written Python exercises: a docstring           |
  | describing a function, plus its solution                |
  +---------------------------------------------------------+
            | ~7 more hours on 8 A100s
            v
        phi-1       (50.6% HumanEval, 55.5% MBPP)
```

The model itself is ordinary: a 24-layer decoder-only transformer with hidden size 2048, 32 attention heads, rotary position embeddings ([RoPE](../../techniques/54-rope-rotary-position-embedding/summary.md)), [FlashAttention](../../techniques/16-flash-attention/summary.md) and a 2048-token context. Everything interesting is in the data.

---

## Key Components Explained

### 1. The Educational-Value Filter
**What it does:** Picks out the most instructive slice of real web code.
**How it works:** GPT-4 rated about 100,000 samples from The Stack and StackOverflow for their educational value to a student learning basic coding concepts. Those labels trained a cheap random forest classifier on embeddings from a pretrained CodeGen-350M model, which was then run over the full corpus. Label a small sample with an expensive model, then scale the judgement with a cheap classifier: [FineWeb-Edu](../../techniques/133-fineweb/summary.md) later used the same pattern. The paper reports that filtering alone lifted a 350M-parameter model from 12.19 to 17.68 percent on HumanEval.

### 2. Synthetic Textbooks
**What it does:** Supplies the clear explanations that are rare on the open web.
**How it works:** GPT-3.5 wrote textbook-style passages mixing prose and code. The hard part, the authors stress, is **diversity**: ask a model for "a textbook section on Python" a million times and you get near-duplicates. They added randomness by constraining topics and target audiences in the prompts. The exact prompts and data were not released, which limits reproducibility.

### 3. CodeExercises Fine-Tuning
**What it does:** Turns a model that knows about code into one that writes functions from a specification.
**How it works:** About 180M tokens of GPT-3.5-written exercises, each a function signature with a docstring, plus a solution. Diversity again came from constraining function names. This stage took HumanEval from 29 to 50.6 percent.

### 4. Emergent Abilities After Fine-Tuning
**What it does:** Suggests the exercises unlocked general skill rather than memorised answers.
**How it works:** The exercises use only basic Python, yet after fine-tuning phi-1 got noticeably better at libraries such as PyGame, Tkinter and PyTorch, which the exercises never mention. The authors read this as fine-tuning reorganising knowledge learned in pretraining. It is suggestive, but the evidence is qualitative examples rather than a controlled measurement.

---

## Key Results

| Model | Parameters | HumanEval pass@1 | MBPP pass@1 |
|---|---|---|---|
| phi-1-base (before exercises) | 1.3B | 29% | - |
| phi-1-small | 350M | 45% | - |
| **phi-1** | **1.3B** | **50.6%** | **55.5%** |
| StarCoder (as reported in the paper) | 15.5B | 33.6% | 52.7% |

- **pass@1** means the model's single first attempt passes the hidden unit tests.
- **A 350M model at 45 percent** is arguably the more striking number.
- **Unconventional problems.** To head off the contamination question, a separate team wrote 50 new problems designed to be unlike training data or common exercise sets, graded by GPT-4 on a 0-10 scale. phi-1 scored 52 percent there, and the ranking of models matched HumanEval.

---

## The Contamination Critique

The obvious worry: CodeExercises are GPT-3.5-written Python functions with docstrings, and so is HumanEval. If some exercises are close paraphrases of HumanEval problems, the 50.6 percent partly measures memorisation.

### What the paper did about it

The authors gave the concern a whole section:

1. **N-gram overlap.** Only 4 HumanEval problems shared a 13-gram with any exercise, and inspection judged all 4 to be false positives.
2. **Embedding and syntax-tree similarity.** They measured how close each exercise was to each HumanEval problem using CodeGen embeddings and abstract syntax tree (AST) edit distance, then **deleted** similar exercises at stricter and stricter thresholds, removing between 42.5K and 354K of the 879.5K exercises.
3. **Retrained on the pruned data.** At the most aggressive threshold phi-1 still scored **45.1 percent**, down from 50.6 but above StarCoder-Prompted's 41.5 percent in the same comparison.

```
  Pruning test (AST threshold tau)
  tau = 0.8, 354K of 879.5K exercises removed
      -> retrained phi-1: 45.1% HumanEval (vs 50.6% unpruned)
```

So the paper's own analysis suggests some of the gain came from exercises close to the benchmark, but most of it survived.

### What critics said

- **Satire.** In September 2023 Rylan Schaeffer posted "[Pretraining on the Test Set Is All You Need](https://arxiv.org/abs/2309.08632)", a deliberately fake paper introducing "phi-CTNL" (pronounced "fictional"), a 1M-parameter model that scores perfectly on every benchmark by training on them. It was aimed at the trend of small models with suspiciously high benchmark scores, the phi series included.
- **Similarity is not a clean line.** Near-duplicates written by a model that has itself seen HumanEval-style problems are hard to catch with n-grams or embeddings. Decontamination can reduce the risk; it cannot prove the model never saw the test.
- **Evidence from later benchmarks.** When Scale AI built [GSM1k](https://arxiv.org/abs/2405.00332) in 2024, fresh grade-school maths problems written to match GSM8K, it named **the Phi and Mistral families** as showing "systematic tendencies to perform stronger on GSM8k compared to GSM1k for almost every release and scale of models". Phi-2 dropped about 6 points. That concerns later phi models on a different benchmark, not phi-1 on HumanEval, but it fed the general suspicion that the phi recipe optimises for benchmark-shaped data.
- **Microsoft's reply.** The Phi-2 announcement acknowledged that "many public benchmarks might leak into the training data" and pointed to phi-1's decontamination study as its evidence.

A fair reading as of 2026: phi-1's gains are real and mostly survive the paper's own tests. But "textbook-style synthetic data" and "benchmark-style data" overlap enough that phi-family scores deserve a discount, and tests written after training are the honest check.

---

## The phi Lineage

| Model | Date | Size | What changed |
|---|---|---|---|
| phi-1 | June 2023 | 1.3B | Python only; ~7B tokens of filtered and synthetic "textbook" data |
| [phi-1.5](https://arxiv.org/abs/2309.05463) ("Textbooks Are All You Need II") | Sept 2023 | 1.3B | Extended to common-sense reasoning and general knowledge; phi-1's data plus ~20B tokens of new synthetic text |
| [Phi-2](https://www.microsoft.com/en-us/research/blog/phi-2-the-surprising-power-of-small-language-models/) | Dec 2023 | 2.7B | Blog release; 1.4T tokens (multiple passes) of synthetic and web data, 14 days on 96 A100s |
| [Phi-3](https://arxiv.org/abs/2404.14219) | Apr 2024 | 3.8B mini, 7B small, 14B medium | Phi-3-mini trained on 3.3T tokens, reported 69% MMLU, pitched as running on a phone; Phi-3.5 later added MoE and vision variants |
| [Phi-4](https://arxiv.org/abs/2412.08905) | Dec 2024 | 14B | Synthetic data "throughout the training process"; reported surpassing its teacher GPT-4 on STEM-focused benchmarks |

The arc is telling. phi-1 claimed that tiny models plus great data are enough. By Phi-3 and Phi-4 the models trained on trillions of tokens and grew to 14B parameters. The lasting lesson was less "small is enough" than "**curated and synthetic data change what a given model size can do**", which is now standard practice.

---

## Why This Was Revolutionary

- **Moved attention from model size to data design.** It showed that a large efficiency gap could come from the data alone.
- **Made synthetic pretraining data respectable.** Using a stronger model to write a curriculum for a weaker one is now routine, and closely related to [knowledge distillation](../../techniques/134-knowledge-distillation/summary.md) and [Self-Instruct](../../techniques/79-self-instruct/summary.md).
- **Popularised LLM-labelled quality filters.** "Have a big model rate a sample, train a small classifier, filter the corpus" became a common pretraining step.
- **Made small models commercially interesting.** On-device and low-cost deployment became a product category partly because the phi line kept showing that small models could be useful.
- **Forced a better contamination conversation.** The pruning experiment is a good model for testing your own results, and the criticism that followed sharpened how the field reads small-model benchmark claims.

---

## Real-World Impact

- **The phi family** shipped through Azure and Hugging Face and became one of the most-used small open model lines.
- **Data-quality pipelines.** Educational-value classifiers ([FineWeb-Edu](../../techniques/133-fineweb/summary.md)) and synthetic textbook corpora follow the same recipe.
- **Synthetic data at scale** in frontier training, from reasoning traces to code, builds on the premise this paper made credible.
- **The contamination debate** it helped start contributed to fresh held-out benchmarks such as GSM1k and to more scepticism about headline numbers from small models.

---

## Key Takeaways for Practitioners

1. **Filter before you scale.** A cheap classifier trained on LLM-rated samples can remove most low-value data and is worth more than extra tokens.
2. **Synthetic data needs forced diversity.** Constrain topics, audiences or names in the prompts, or you get a million near-duplicates.
3. **Decontaminate the way this paper did, then go further.** Run n-gram, embedding and structural similarity checks, retrain on the pruned data, and report the drop.
4. **Test on something written after your training data.** A fresh held-out set is the only convincing answer to "did it see the test?"
5. **Be sceptical of small-model benchmark scores** until a real task confirms them. By its authors' own account, phi-1 is strong on HumanEval-shaped problems and brittle on prompts that are long, unusual or ungrammatical.

---

## Limitations & Future Directions

- **Python only**, and mostly short self-contained functions. The paper notes phi-1 lacks the breadth of larger multi-language code models.
- **Brittle to prompt variation.** Performance drops with longer prompts, grammatical errors or unusual phrasing.
- **Not reproducible.** The synthetic data and prompts were not released, so outsiders could not verify the pipeline or check contamination themselves.
- **Needs a stronger teacher.** The data came from GPT-3.5 and the filter labels from GPT-4. The method passes capability down from teacher to student; it does not show how to exceed the teacher (Phi-4 later claimed to beat its teacher on some STEM benchmarks).
- **Contamination can be reduced, not ruled out**, which is why later held-out benchmarks matter.
- **Model collapse risk.** Training repeatedly on model-generated text can erode the rare cases at the edges of the data; see [model collapse](../../techniques/140-model-collapse/summary.md).

---

## Further Reading

- **Original Paper:** [arxiv.org/abs/2306.11644](https://arxiv.org/abs/2306.11644)
- **phi-1.5 (Textbooks Are All You Need II):** [arxiv.org/abs/2309.05463](https://arxiv.org/abs/2309.05463)
- **Phi-2 announcement:** [microsoft.com/en-us/research/blog/phi-2-the-surprising-power-of-small-language-models](https://www.microsoft.com/en-us/research/blog/phi-2-the-surprising-power-of-small-language-models/)
- **Phi-3 Technical Report:** [arxiv.org/abs/2404.14219](https://arxiv.org/abs/2404.14219)
- **Phi-4 Technical Report:** [arxiv.org/abs/2412.08905](https://arxiv.org/abs/2412.08905)
- **Pretraining on the Test Set Is All You Need (satire):** [arxiv.org/abs/2309.08632](https://arxiv.org/abs/2309.08632)
- **GSM1k (benchmark overfitting study):** [arxiv.org/abs/2405.00332](https://arxiv.org/abs/2405.00332)
- **In this collection:** [Codex](../56-codex/summary.md), [Chinchilla](../../techniques/18-chinchilla/summary.md), [The FineWeb Datasets](../../techniques/133-fineweb/summary.md), [Knowledge Distillation](../../techniques/134-knowledge-distillation/summary.md), [MMLU](../../techniques/137-mmlu/summary.md)

## Citation

```bibtex
@article{gunasekar2023textbooks,
  title={Textbooks Are All You Need},
  author={Gunasekar, Suriya and Zhang, Yi and Aneja, Jyoti and Mendes, Caio C{\'e}sar Teodoro and Del Giorno, Allie and Gopi, Sivakanth and Javaheripi, Mojan and Kauffmann, Piero and de Rosa, Gustavo and Saarikivi, Olli and Salim, Adil and Shah, Shital and Behl, Harkirat Singh and Wang, Xin and Bubeck, S{\'e}bastien and Eldan, Ronen and Kalai, Adam Tauman and Lee, Yin Tat and Li, Yuanzhi},
  journal={arXiv preprint arXiv:2306.11644},
  year={2023}
}
```

<!-- related:start -->

---

## Related in This Collection

- [Language Models are Few-Shot Learners (GPT-3)](../../language-models/04-gpt3-few-shot-learners/summary.md)
- [Scaling Laws for Neural Language Models](../../techniques/12-scaling-laws/summary.md)
- [FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness](../../techniques/16-flash-attention/summary.md)
- [Training Compute-Optimal Large Language Models (Chinchilla)](../../techniques/18-chinchilla/summary.md)
- [GPT-4 Technical Report](../../language-models/36-gpt4/summary.md)
- [Mixtral of Experts (and the Mixture-of-Experts Architecture)](../../architectures/37-mixture-of-experts/summary.md)
- [RoFormer: Enhanced Transformer with Rotary Position Embedding (RoPE)](../../techniques/54-rope-rotary-position-embedding/summary.md)
- [Codex: Evaluating Large Language Models Trained on Code](../../language-models/56-codex/summary.md)

<!-- related:end -->
