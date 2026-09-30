---
title: "Neural Machine Translation of Rare Words with Subword Units (BPE)"
slug: "136-bpe-subword-units"
number: 136
category: "techniques"
authors: "Rico Sennrich, Barry Haddow, Alexandra Birch (University of Edinburgh)"
published: "August 2015 (arXiv); August 2016 (ACL 2016)"
year: 2015
url: "https://arxiv.org/abs/1508.07909"
tags: ["tokenization", "language-model", "embeddings"]
---

# Neural Machine Translation of Rare Words with Subword Units (BPE)

**Authors:** Rico Sennrich, Barry Haddow, Alexandra Birch (University of Edinburgh)
**Published:** August 2015 (arXiv); August 2016 (ACL 2016)
**Paper:** [arxiv.org/abs/1508.07909](https://arxiv.org/abs/1508.07909)

---

## Why This Matters

Every large language model you have used reads text as **tokens**, and for almost all of them those tokens come from **byte pair encoding (BPE)**, the method this paper brought into neural NLP. GPT-2 through GPT-4o, Llama, Mistral, Qwen and DeepSeek all tokenize with a BPE descendant. When an API bills you per token, when a model miscounts the letters in "strawberry", or when a context window is quoted as 128K tokens, you are looking at the consequences of this paper.

- **It solved the open-vocabulary problem.** Before BPE, neural translation models had a fixed list of about 30,000 to 80,000 words, and anything outside it became an "unknown" token.
- **It is almost embarrassingly simple.** Start with characters; repeatedly merge the most frequent adjacent pair; stop after a set number of merges. The paper's reference implementation is a short Python snippet.
- **It gave a single dial for vocabulary size.** The number of merges sets the trade-off between short sequences (big vocabulary) and small embedding tables (small vocabulary).
- **It outlived its task.** The paper is about German and Russian translation. Its lasting impact is on the pretraining of every modern LLM.

**The insight:** many words are translatable from their parts. Names can be copied character by character, compounds such as German "Abwasserbehandlungsanlage" (sewage treatment plant) translate piece by piece, and cognates share sub-word patterns. So represent rare words as sequences of **subword units**, and learn those units from data with a compression algorithm.

---

## The Problem: Fixed Vocabularies Break on Rare Words

A 2015 neural machine translation system, typically an encoder-decoder with [attention](../../architectures/66-bahdanau-attention/summary.md) (see also [Seq2Seq](../../architectures/55-seq2seq/summary.md)), had a softmax over a fixed target vocabulary. Every word needed its own embedding row and output row, so the vocabulary could not grow without limit.

```
Fixed vocabulary, 50K most frequent words

  Input:  "The Abwasserbehandlungsanlage in Tschernobyl ..."
  Seen:   "The <UNK> in <UNK> ..."

  The model cannot translate what it cannot see.
```

The standard fix was a **back-off dictionary**: align each unknown source word to a target position and look it up in a bilingual dictionary, or copy it through. That fails for exactly the words that matter: names that need transliteration (English to Russian changes the alphabet), productive compounds that never appear in the dictionary, and inflected forms of known words. Character-level models avoided unknowns but made sequences several times longer and were slow and hard to train at the time.

---

## The Core Innovation

Adapt Philip Gage's 1994 **byte pair encoding** compression algorithm to words. BPE as compression "iteratively replaces the most frequent pair of bytes in a sequence with a single, unused byte." For segmentation, the paper merges characters into increasingly long symbols instead.

```
Learning BPE (toy example from the paper)

Word counts:  low: 5   lower: 2   newest: 6   widest: 3
Start as characters, with an end-of-word marker (shown here as </w>):

  l o w </w>          l o w e r </w>
  n e w e s t </w>    w i d e s t </w>

Count all adjacent pairs, weighted by word frequency, and merge the top one:

  merge 1:  e s   -> es        (9 occurrences)
  merge 2:  es t  -> est
  merge 3:  est </w> -> est</w>
  merge 4:  l o   -> lo
  merge 5:  lo w  -> low
  ...

Applying the merges to an unseen word, "lowest":
  l o w e s t </w>  ->  low est</w>
```

Two properties make this work:

1. **Frequent words become single tokens; rare words split into pieces.** "the" is one unit; a rare surname becomes several.
2. **Nothing is ever unknown.** In the worst case a word falls back to single characters, which are always in the vocabulary.

The end-of-word marker keeps "est" at the end of a word distinct from "est" inside one, and lets the original spacing be restored exactly after translation.

---

## Key Components Explained

### 1. Merge Count as the Vocabulary Dial
**What it does:** Sets the final vocabulary size.
**How it works:** Final size = number of characters + number of merges. The paper used tens of thousands of merges (for example about 59,500 for separate vocabularies and 89,500 for a joint one). Fewer merges give longer sequences and a smaller softmax; more merges give shorter sequences and bigger embedding tables. Modern LLMs use about 32K (Llama 1 and 2) to about 200K (OpenAI's o200k encoding) entries.

### 2. Joint BPE
**What it does:** Makes the source and target languages segment the same strings the same way.
**How it works:** Learn one set of merges on the concatenated source and target text. A name like "Obama" then splits identically in English and German, so the model can learn to copy it piece by piece. For language pairs with different scripts (English and Russian) the authors transliterated the Russian side while learning merges.

### 3. Deterministic Segmentation at Test Time
**What it does:** Tokenizes new text consistently.
**How it works:** Apply the learned merges in the order they were learned. No dictionary lookup and no model inference is needed; it is fast and fully reproducible.

---

## Key Results

On the WMT 2015 English-to-German (4.2M sentence pairs) and English-to-Russian (2.6M) translation tasks:

- **+1.1 BLEU (English-German) and +1.3 BLEU (English-Russian)** over a strong word-level baseline with a back-off dictionary. BLEU measures n-gram overlap with reference translations; a gain of about 1 was a solid result on these tasks.
- **Large gains on rare words.** The authors measured unigram F1 on rare and unseen words specifically, where subword systems clearly beat the back-off approach, especially for names and compounds.
- **Smaller vocabularies, no unknowns.** The subword systems translated with fixed vocabularies and no special unknown-word handling at all.
- The work was built into Edinburgh's WMT 2016 translation systems, and the open-source `subword-nmt` tool spread quickly.

---

## Descendants: From BPE to Modern Tokenizers

| Tokenizer | Where it appears | What it changed |
|---|---|---|
| **Original BPE** (this paper, 2016) | Neural machine translation | Merges over characters within pre-split words |
| **Byte-level BPE** ([GPT-2](../../language-models/64-gpt2/summary.md), 2019) | GPT-2, GPT-3, RoBERTa and many others | Starts from the 256 raw byte values instead of Unicode characters, so any text in any language or encoding is representable. GPT-2's vocabulary was 50,257 tokens. It also stopped merges from crossing character categories (letters, digits, punctuation) to avoid tokens like "dog." and "dog!" |
| **SentencePiece** ([Kudo and Richardson, EMNLP 2018](https://arxiv.org/abs/1808.06226)) | T5, [Llama 1 and 2](../../language-models/15-llama/summary.md), Gemma, many multilingual models | A library that trains directly on raw text with no language-specific pre-splitting, treating the space as an ordinary symbol. It supports both BPE and the **unigram language model** method (Kudo, 2018), which prunes a large candidate vocabulary by likelihood instead of merging upward |
| **tiktoken** (OpenAI, 2022 onward) | GPT-3.5/GPT-4 (`cl100k_base`), GPT-4o (`o200k_base`), Llama 3 | A fast byte-level BPE implementation ("between 3-6x faster than a comparable open source tokeniser" per its README), with larger vocabularies and regex pre-splitting rules. Llama 3 moved to a tiktoken-based 128K vocabulary, which Meta said yields "up to 15% fewer tokens compared to Llama 2" |

A close relative, **WordPiece** (used by [BERT](../../language-models/03-bert/summary.md)), also merges upward but picks merges by likelihood gain rather than raw frequency. Across all of these the core idea is unchanged from 2016: a learned, data-driven vocabulary of variable-length pieces with a guaranteed fallback to characters or bytes.

---

## Why This Was Revolutionary

- **Ended the unknown-word problem** in neural NLP with a few dozen lines of code.
- **Decoupled vocabulary from language.** The same algorithm works for any script and for code, which later made multilingual and code pretraining straightforward.
- **Made the vocabulary a tunable hyperparameter** rather than a list of words chosen by frequency cut-off.
- **Set the interface between text and model** that every transformer since has used, from the [original Transformer](../../architectures/01-attention-is-all-you-need/summary.md) (which used byte-pair and word-piece vocabularies) to today's frontier models.

---

## Real-World Impact

- **Every major LLM tokenizer** is BPE or a close relative.
- **Token counts drive cost and limits.** API pricing, context windows and rate limits are all denominated in BPE tokens, so tokenizer efficiency directly affects what users pay. Languages that tokenize poorly (many non-Latin scripts under English-heavy vocabularies) pay more per word.
- **Known quirks come from BPE.** Models struggle to count letters or reverse strings because they never see individual characters of common words. Oddly split numbers hurt arithmetic, which is why some tokenizers split digits individually. "Glitch tokens", vocabulary entries that were frequent in tokenizer training text but rare in model training text, produced bizarre behaviour in early GPT models.
- **Research on going beyond it** (byte-level models without a tokenizer, dynamic patching) is still active as of 2026, but BPE remains the default.

---

## Key Takeaways for Practitioners

1. **Tokenizer choice is a model decision you cannot undo cheaply.** Changing it later means retraining or at least re-learning embeddings.
2. **Measure tokens per word on your own data**, especially non-English text and code. A 20 percent difference in token count is a 20 percent difference in cost and effective context.
3. **Bigger vocabularies shorten sequences** but enlarge the embedding and output layers. For large models the trade usually favours bigger vocabularies, which is why sizes grew from about 32K to 128K-200K.
4. **Expect character-level blind spots.** Spelling, counting letters and exact string manipulation are hard for BPE models; use tools or code for these.
5. **Byte-level fallback matters.** It guarantees robustness to emoji, rare scripts and binary junk in production inputs.

---

## Limitations & Future Directions

- **Greedy and frequency-based.** Merges ignore linguistic structure, so morphemes can be split oddly (compare "unhappiness" across tokenizers).
- **Deterministic segmentation** means a word always splits the same way, which can make models brittle. BPE-dropout and the subword regularization of the unigram method add randomness during training to counter this.
- **Uneven cost across languages**, as noted above.
- **Trained separately from the model.** The tokenizer is fixed before pretraining and never updated by gradient descent, a design many researchers consider a wart. Tokenizer-free and learned-patching approaches aim to remove it.

---

## Further Reading

- **Original Paper:** [arxiv.org/abs/1508.07909](https://arxiv.org/abs/1508.07909) (ACL Anthology: [P16-1162](https://aclanthology.org/P16-1162/))
- **subword-nmt reference implementation:** [github.com/rsennrich/subword-nmt](https://github.com/rsennrich/subword-nmt)
- **SentencePiece:** [arxiv.org/abs/1808.06226](https://arxiv.org/abs/1808.06226)
- **Subword Regularization (unigram LM tokenizer):** [arxiv.org/abs/1804.10959](https://arxiv.org/abs/1804.10959)
- **tiktoken:** [github.com/openai/tiktoken](https://github.com/openai/tiktoken)
- **In this collection:** [GPT-2](../../language-models/64-gpt2/summary.md), [BERT](../../language-models/03-bert/summary.md), [Seq2Seq](../../architectures/55-seq2seq/summary.md), [Bahdanau Attention](../../architectures/66-bahdanau-attention/summary.md), [word2vec](../53-word2vec/summary.md)

## Citation

```bibtex
@inproceedings{sennrich-etal-2016-neural,
  title={Neural Machine Translation of Rare Words with Subword Units},
  author={Sennrich, Rico and Haddow, Barry and Birch, Alexandra},
  booktitle={Proceedings of the 54th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)},
  pages={1715--1725},
  year={2016},
  address={Berlin, Germany},
  publisher={Association for Computational Linguistics},
  doi={10.18653/v1/P16-1162}
}
```

<!-- related:start -->

---

## Related in This Collection

- [BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](../../language-models/03-bert/summary.md)
- [Language Models are Few-Shot Learners (GPT-3)](../../language-models/04-gpt3-few-shot-learners/summary.md)
- [LLaMA 2: Open Foundation and Fine-Tuned Chat Models](../../language-models/17-llama2/summary.md)
- [Qwen3: Technical Report](../../language-models/28-qwen3/summary.md)
- [LLaMA 3.3: Matching 405B Performance with 70B Parameters](../../language-models/33-llama3.3/summary.md)
- [GPT-4 Technical Report](../../language-models/36-gpt4/summary.md)
- [GPT-4o: The First Omni Model](../../language-models/40-gpt4o/summary.md)
- [Efficient Estimation of Word Representations in Vector Space (Word2Vec)](../../techniques/53-word2vec/summary.md)

<!-- related:end -->
