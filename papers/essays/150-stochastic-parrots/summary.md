---
title: "On the Dangers of Stochastic Parrots: Can Language Models Be Too Big?"
slug: "150-stochastic-parrots"
number: 150
category: "essays"
authors: "Emily M. Bender (University of Washington), Timnit Gebru (Black in AI), Angelina McMillan-Major (University of Washington), Shmargaret Shmitchell (listed as \"The Aether\"; a pseudonym, see Context)"
published: "March 2021 (FAccT '21, ACM Conference on Fairness, Accountability, and Transparency, held virtually 3-10 March 2021)"
year: 2021
url: "https://dl.acm.org/doi/10.1145/3442188.3445922"
tags: ["essay", "safety", "scaling"]
---

# On the Dangers of Stochastic Parrots: Can Language Models Be Too Big?

**Authors:** Emily M. Bender (University of Washington), Timnit Gebru (Black in AI), Angelina McMillan-Major (University of Washington), Shmargaret Shmitchell (listed as "The Aether"; a pseudonym, see Context)
**Published:** March 2021 (FAccT '21, ACM Conference on Fairness, Accountability, and Transparency, held virtually 3-10 March 2021)
**Essay:** [dl.acm.org/doi/10.1145/3442188.3445922](https://dl.acm.org/doi/10.1145/3442188.3445922)

---

## Why This Paper Matters

In 2018-2021 the dominant move in NLP was to make language models bigger: BERT, GPT-2, GPT-3 (175B parameters) and, just before the paper went out, Switch-C at 1.57 trillion parameters. The paper's table of recent models shows the climb in both parameters and training text. Bender, Gebru, McMillan-Major and Shmitchell stepped back and asked a question the leaderboards did not: "How big is too big?"

It is a position paper, not an experiment. It gathers risks that were then discussed separately (energy, data bias, hype, misuse) and ties them to the single trend of ever-larger models trained on ever-larger scraped web text. It also coined "stochastic parrot", which escaped the paper and became one of the most contested phrases in AI.

- **It asked who pays,** not only what scores.
- **It argued data should be curated and documented,** not just scaled.
- **It gave the "form versus meaning" critique a name** that still frames debates about whether language models understand anything.

**The insight:** fluent output is not evidence of understanding, and humans are primed to read meaning into fluent text, so a model that is good at imitating form can mislead both users and researchers.

---

## The Problem It Addressed

The field measured progress by benchmark leaderboards, mostly for English, and treated model size as a free lever. The authors saw costs left off the ledger: electricity and money, the makeup of the web data being swallowed, and downstream harms when synthetic text is released into the world. Their recommendations (stated in the abstract) are to weigh environmental and financial costs first, invest in curating and carefully documenting datasets rather than ingesting everything, run pre-development exercises on how a planned approach fits stakeholder values, and encourage research directions beyond ever-larger models.

---

## The Argument

### 1. Environmental and financial cost (section 3)
The paper leans on earlier work by Strubell et al. It notes that the average human is responsible for an estimated 5 tonnes of CO2e per year, while training a Transformer (big) model with neural architecture search was estimated to emit 284 tonnes, and that training one BERT base model on GPUs was estimated to need as much energy as a trans-American flight. For machine translation, they cite an estimate of $150,000 extra compute for a 0.1 BLEU gain. They also cite a 300,000x rise in compute for the largest models over six years.

The distinctive step is about fairness: the costs and benefits land on different people. Those most exposed to climate harm (the paper names the Maldives and flood-hit Sudan) are unlikely to benefit from large English models, so the authors say the first consideration at this scale should be environmental cost. They add that for deployed models, inference cost may eventually outweigh training cost.

### 2. Unmanageable training data (section 4)
Size does not equal diversity. The authors argue web text over-represents younger users and people from wealthy countries, and each filter on top (Reddit-link selection for GPT-2, moderation practices, blocklists of "bad words") shifts it further. Examples they give: 67% of US Reddit users were men and 64% were aged 18-29 in a 2016 Pew survey, and only 8.8-15% of Wikipedians were women or girls in surveys they cite. Blocklist filtering can also remove text written by and for marginalised groups. They also note static data can lock in outdated views and that poorly covered social movements are missing. The remedy: budget for curation and documentation at the start, and only build datasets as large as can be documented, to avoid "documentation debt".

### 3. Meaning versus form (section 6.1)
Drawing on Bender and Koller, the paper says language models are trained on form alone, with no access to communicative intent or meaning. Human language use is jointly constructed between people who model each other's mind; text from a language model "is not grounded in communicative intent, any model of the world, or any model of the reader's state of mind." Coherence, they say, is "in the eye of the beholder." A language model is, in their words, "a system for haphazardly stitching together sequences of linguistic forms it has observed in its vast training data, according to probabilistic information about how they combine, but without any reference to meaning: a stochastic parrot."

### 4. Risks and harms (section 6.2)
```
Absorbed bias        -> stereotypes amplified, fed into the next training set
Overt abuse          -> proliferated, plus classifier and search harms
Bad actors           -> cheap persuasive text, e.g. extremist recruitment
Fluent mistakes      -> machine translation that looks adequate and is not
Memorisation         -> extractable personal data, worse in larger models
```
Cited cases include a GPT-3 study on generating extremist personas, a Palestinian man arrested after machine translation rendered "good morning" as "hurt them", and work showing personal information can be extracted from a model.

---

## Key Results

There are no new benchmark results. The paper's contributions are an overview table of model sizes (for example BERT at 3.4E+08 parameters and 16GB, GPT-3 at 1.75E+11 and 570GB, Switch-C at 1.57E+12 and 745GB), the synthesis above, and a set of recommendations (section 7): consider cost and data up front, document datasets and intended use, run pre-mortems, and favour efforts that help understand how models reach results.

---

## Why This Was Revolutionary

- **It made cost and data a first-class research topic** for language models.
- **It popularised dataset documentation** (datasheets, data statements) as an expectation for large corpora.
- **It crystallised the form-versus-meaning critique** as a phrase anyone can repeat.
- **It showed a field-wide trend as a choice,** rather than inevitable.

---

## Context and Controversy

The paper became widely known partly through events around it. By press and community accounts, four of the six authors of the pre-publication draft were Google employees (the published paper lists four authors) (Gebru co-led Google's Ethical AI team with Margaret Mitchell). Google asked that the paper be withdrawn from FAccT or that Google authors' names be removed. In early December 2020 Gebru's employment ended: Google said it accepted her resignation, while Gebru and many colleagues described it as a firing. The dispute drew protests and public letters. Mitchell is credited under the pseudonym "Shmargaret Shmitchell". This summary does not rule on the employment dispute; see the sources in Further Reading. Note the details above come from secondary reporting, not the paper.

---

## Real-World Impact

The paper's themes shaped later work on model cards, data documentation, energy reporting and AI risk governance. "Stochastic parrot" became shorthand in public arguments about whether chatbots "understand". Sam Altman tweeted "i am a stochastic parrot, and so r u" shortly after ChatGPT launched, and the American Dialect Society nominated the term for 2023 AI-related word of the year. Related summaries: [GPT-3](../../language-models/04-gpt3-few-shot-learners/summary.md), [Attention Is All You Need](../../architectures/01-attention-is-all-you-need/summary.md), [Scaling Laws](../../techniques/12-scaling-laws/summary.md), and the opposing outlook in [The Bitter Lesson](../111-bitter-lesson/summary.md) and [The Scaling Hypothesis](../112-scaling-hypothesis/summary.md).

---

## Key Takeaways for Practitioners

1. **Budget for documentation before collecting data.** Data you cannot document is a liability.
2. **Report compute and energy** alongside accuracy.
3. **Ask who benefits and who bears the cost** of a large model before building it.
4. **Do not treat fluency as comprehension.** Test for the specific capability you need.
5. **Treat any user-facing generator as a source of plausible falsehood and bias** and design review accordingly.

---

## Limitations & Future Directions

What later work did and did not bear out, in a fair reading:

- **Scale kept winning on capability.** The paper did not predict that larger models and more data would keep unlocking abilities (see [Scaling Laws](../../techniques/12-scaling-laws/summary.md)). Many researchers say this weakens the implied case that size is a dead end.
- **Cost concerns proved real but are contested in detail.** Later work (for instance Patterson et al., arXiv:2104.10350) refined earlier training-emission estimates and argued model, datacenter and hardware choices change footprints by large factors, so single headline numbers like the 284 tonne figure are best treated as a snapshot of 2019-era methods.
- **Data and bias concerns persisted.** Documentation and curation became more common, but the largest training sets remain only partly documented.
- **The "no understanding" claim remains disputed.** Critics say behaviour on reasoning and world-model tests goes beyond parroting; defenders say benchmark success still does not show grounded meaning. This summary takes no side; the question is open.
- **Hallucination is the visible form of the form-versus-meaning worry.** See [Why Language Models Hallucinate](../../techniques/149-why-language-models-hallucinate/summary.md) for a statistical account of why fluent models state falsehoods.

---

## Further Reading

- **Essay:** https://dl.acm.org/doi/10.1145/3442188.3445922
- **Bender and Koller (2020), "Climbing towards NLU":** the form-and-meaning argument the paper builds on
- **Strubell et al. (2019), "Energy and Policy Considerations for Deep Learning in NLP":** the source of the cost figures
- **Patterson et al. (2021), carbon emissions and large neural network training:** https://arxiv.org/abs/2104.10350
- **Background on the term and the publication dispute:** https://en.wikipedia.org/wiki/Stochastic_parrot

---

## Citation

```bibtex
@inproceedings{bender2021parrots,
  title={On the Dangers of Stochastic Parrots: Can Language Models Be Too Big?},
  author={Bender, Emily M. and Gebru, Timnit and McMillan-Major, Angelina and Shmitchell, Shmargaret},
  booktitle={Proceedings of the 2021 ACM Conference on Fairness, Accountability, and Transparency (FAccT '21)},
  year={2021},
  doi={10.1145/3442188.3445922}
}
```

<!-- related:start -->

---

## Related in This Collection

- [Attention Is All You Need](../../architectures/01-attention-is-all-you-need/summary.md)
- [BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](../../language-models/03-bert/summary.md)
- [Language Models are Few-Shot Learners (GPT-3)](../../language-models/04-gpt3-few-shot-learners/summary.md)
- [Language Models are Unsupervised Multitask Learners (GPT-2)](../../language-models/64-gpt2/summary.md)
- [The Bitter Lesson](../../essays/111-bitter-lesson/summary.md)
- [The Scaling Hypothesis](../../essays/112-scaling-hypothesis/summary.md)
- [Why Language Models Hallucinate](../../techniques/149-why-language-models-hallucinate/summary.md)

<!-- related:end -->
