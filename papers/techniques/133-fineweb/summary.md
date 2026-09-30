---
title: "The FineWeb Datasets: Decanting the Web for the Finest Text Data at Scale (FineWeb)"
slug: "133-fineweb"
number: 133
category: "techniques"
authors: "Guilherme Penedo, Hynek Kydlíček, Loubna Ben allal, Anton Lozhkov, Margaret Mitchell, Colin Raffel, Leandro Von Werra, Thomas Wolf (Hugging Face)"
published: "June 2024 (NeurIPS 2024 Datasets and Benchmarks Track)"
year: 2024
url: "https://arxiv.org/abs/2406.17557"
tags: ["pretraining", "datasets", "data-curation"]
---

# The FineWeb Datasets: Decanting the Web for the Finest Text Data at Scale (FineWeb)

**Authors:** Guilherme Penedo, Hynek Kydlíček, Loubna Ben allal, Anton Lozhkov, Margaret Mitchell, Colin Raffel, Leandro Von Werra, Thomas Wolf (Hugging Face)
**Published:** June 2024 (NeurIPS 2024 Datasets and Benchmarks Track)
**Paper:** [arxiv.org/abs/2406.17557](https://arxiv.org/abs/2406.17557)

---

## Why This Paper Matters

Ask what separates a strong language model from a weak one of the same size and the honest answer, by 2024, was increasingly **the data**. Yet the pretraining datasets behind leading open-weight models such as Llama 3 and Mixtral were not released, and little was published about how they were built. Everyone knew data quality mattered; few could study it.

FineWeb opened that box. Hugging Face released a **15-trillion-token** English web dataset built from **96 Common Crawl snapshots**, documented and ablated every processing decision, and showed that models trained on it beat models trained on the other open web datasets of the time.

- **A public, frontier-scale pretraining corpus**, large enough to train serious models, under an open licence.
- **Every choice ablated.** Text extraction, filtering thresholds, deduplication strategy - each was tested by training small models and comparing benchmark scores.
- **FineWeb-Edu**, a 1.3-trillion-token subset filtered for educational value, gave large jumps on knowledge and reasoning benchmarks.
- **Counter-intuitive findings** - most famously that deduplicating more aggressively made the data *worse*.

**The insight:** treat dataset construction as an experimental science. Train small proxy models on each variant of the data and let downstream benchmark scores decide, rather than relying on intuitions about what "clean" text looks like.

---

## Background: The Open Web Dataset Lineage

- **C4** (2019, [T5](../../language-models/65-t5/summary.md)) - a cleaned Common Crawl snapshot with simple heuristic filters.
- **The Pile** (2020, EleutherAI) - about 825 GiB drawn from 22 sources, mixing web text with books, code, papers and more.
- **RefinedWeb** (2023, TII, behind the Falcon models) - argued web data alone, well filtered, could match curated mixtures; about 5T tokens extracted, 600B released.
- **FineWeb** (2024) - scaled that approach to 15T tokens with published ablations.
- **DCLM** (2024) - a benchmark for data curation with a 3.8T-token baseline dataset built with a fastText quality classifier; its 7B model reached 64% on [MMLU](../137-mmlu/summary.md).

---

## The Core Innovation: An Ablated Pipeline

### How decisions were tested
For each candidate processing step, the team trained small models (on the order of 1-2B parameters) on the resulting data and evaluated them on a suite of benchmarks chosen to give a clear signal early in training. A step stayed if it improved the scores.

### The pipeline

```
Raw Common Crawl WARC files (96 snapshots)
   |
   v  trafilatura text extraction from raw HTML
   |  (better than using Common Crawl's own WET text)
   v
Base filtering: URL blocklist, English language ID, quality heuristics
   |                                      ~36T tokens
   v
MinHash deduplication, done PER SNAPSHOT  ~20T tokens
   |
   v
Selected C4 filters + 3 custom heuristic filters
   |   (the custom filters removed about 22% of tokens)
   v
FineWeb: 15T tokens
```

### Finding 1: Extract text yourself
Common Crawl provides pre-extracted text files (WET), but running a proper extractor (trafilatura) on the raw HTML produced better training data, despite the extra cost.

### Finding 2: Global deduplication hurt
The intuitive move is to deduplicate across all 96 snapshots at once. The team found that this produced **worse** models than deduplicating each snapshot separately. Their analysis: aggressive global deduplication disproportionately removed higher-quality documents that legitimately recur, and left behind a residue of lower-quality text that happened to be unique.

### Finding 3: Filters should be chosen by measurement
Rather than adopting every heuristic from previous datasets, the team kept only the C4 filters that helped and designed three new ones by comparing statistics of high- and low-performing data slices, such as the fraction of lines ending in punctuation and the amount of duplicated lines.

### FineWeb-Edu: let a model judge quality
The team asked **Llama-3-70B-Instruct** to rate a sample of web pages for educational value on a 0-5 scale, trained a small classifier on those ratings, and kept pages scoring 3 or above. The result was **FineWeb-Edu, 1.3T tokens**.

---

## Key Results

- **FineWeb beat other open web datasets** (including C4, RefinedWeb, Dolma and The Pile) in the paper's ablation-scale comparisons.
- **FineWeb-Edu improved knowledge and reasoning benchmarks sharply**: in the authors' 1.8B-parameter comparison, MMLU rose from about 33% to 37% and ARC from about 46% to 57%, relative to FineWeb.
- **Released openly**, together with the processing library (datatrove) and the ablation models.

---

## Why This Was Revolutionary

- **Data curation became reproducible research.** Before FineWeb, open work on pretraining data at 10T+ scale was rare; afterwards, it had a public baseline and method.
- **Model-rated quality filtering went mainstream.** Using a strong LLM to label a sample, then distilling that judgement into a cheap classifier, is now a standard recipe.
- **It punctured some intuitions.** "More deduplication is always better" did not survive measurement.

---

## Real-World Impact

- **Small open models** such as Hugging Face's SmolLM series were trained heavily on FineWeb-Edu.
- **Multilingual follow-ups** extended the pipeline beyond English (FineWeb 2).
- **DCLM and later curation work** built on the same principle of benchmarking data choices with proxy models.
- **The same "quality over quantity" argument** underlies [Textbooks Are All You Need](../../language-models/135-phi-1-textbooks/summary.md), which pushed filtering and synthetic data further.

---

## Key Takeaways for Practitioners

1. **Data choices are testable.** Train small models on each variant and compare; do not trust intuition alone.
2. **Deduplicate thoughtfully.** Some repetition signals quality. Measure the effect rather than maximising removal.
3. **Educational filters help knowledge benchmarks** - but may shift what the model is good at. A web corpus filtered for textbook-like content will under-represent casual or creative text.
4. **Benchmarks steer curation.** Anything chosen to raise MMLU scores risks overfitting to MMLU-like content; see [contamination and saturation](../../../explainers/benchmarks/contamination-and-saturation.md).

---

## Limitations & Future Directions

- **English only** in the original release.
- **Proxy-model scale.** Decisions validated on 1-2B-parameter models may not all transfer to much larger ones.
- **Benchmark-driven.** Filters that raise the chosen benchmarks are not guaranteed to improve everything users care about.
- **Web-only.** Code, books, papers and conversational data are separate problems; modern mixtures combine many sources.
- **Legal and consent questions** about training on crawled web data are unresolved, and the dataset inherits them.

---

## Further Reading

- **Original Paper:** [arxiv.org/abs/2406.17557](https://arxiv.org/abs/2406.17557)
- **Dataset:** [huggingface.co/datasets/HuggingFaceFW/fineweb](https://huggingface.co/datasets/HuggingFaceFW/fineweb) and [fineweb-edu](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu)
- **DCLM:** Li et al. 2024, [arxiv.org/abs/2406.11794](https://arxiv.org/abs/2406.11794)
- **In this collection:** [T5 (C4)](../../language-models/65-t5/summary.md), [LLaMA](../../language-models/15-llama/summary.md), [phi-1](../../language-models/135-phi-1-textbooks/summary.md), [Chinchilla](../18-chinchilla/summary.md)
- **Explainer:** [Synthetic data and model collapse](../../../explainers/open-questions/synthetic-data-and-model-collapse.md)

## Citation

```bibtex
@inproceedings{penedo2024fineweb,
  title={The FineWeb Datasets: Decanting the Web for the Finest Text Data at Scale},
  author={Penedo, Guilherme and Kydl{\'\i}{\v{c}}ek, Hynek and Ben allal, Loubna and Lozhkov, Anton and Mitchell, Margaret and Raffel, Colin and Von Werra, Leandro and Wolf, Thomas},
  booktitle={Advances in Neural Information Processing Systems Datasets and Benchmarks Track},
  year={2024}
}
```

<!-- related:start -->

---

## Related in This Collection

- [Training Compute-Optimal Large Language Models (Chinchilla)](../../techniques/18-chinchilla/summary.md)
- [LLaMA 3.3: Matching 405B Performance with 70B Parameters](../../language-models/33-llama3.3/summary.md)
- [Mixtral of Experts (and the Mixture-of-Experts Architecture)](../../architectures/37-mixture-of-experts/summary.md)
- [Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer (T5)](../../language-models/65-t5/summary.md)
- [Textbooks Are All You Need (phi-1)](../../language-models/135-phi-1-textbooks/summary.md)
- [Measuring Massive Multitask Language Understanding (MMLU)](../../techniques/137-mmlu/summary.md)
- [AI Models Collapse When Trained on Recursively Generated Data (Model Collapse)](../../techniques/140-model-collapse/summary.md)

<!-- related:end -->
