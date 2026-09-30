# Tokenization

**In one line:** Language models read and write in tokens (chunks of text, usually pieces of words), not words or letters, and that one design choice explains a surprising share of their costs, limits and odd mistakes.
**Last reviewed:** 2026-09-30

---

## The short version

- A **token** is the unit a language model actually sees. Before any text reaches the model, a **tokenizer** chops it into pieces from a fixed vocabulary and swaps each piece for an integer ID. The model predicts the next ID, and the tokenizer turns IDs back into text.
- Most modern tokenizers use **subword units**: common words get one token, rare words are split into several familiar pieces. The standard algorithm, **byte pair encoding (BPE)**, came to language models through a 2016 machine translation paper.
- Everything is priced and limited in tokens: API bills, context windows, output caps, speed. A rough English rule of thumb is about 4 bytes of text per token, but the ratio depends heavily on the tokenizer and the language.
- Tokenization is **unfair across languages**. The same sentence can take many times more tokens in one language than another, which means higher cost, slower responses and less room in the context window for speakers of those languages.
- Some famous model failures (counting letters in a word, some arithmetic slips) are partly tokenization artefacts: the model never saw the letters, only the chunks.
- **Byte-level** and **tokenizer-free** approaches try to remove the tokenizer altogether. They are promising in research but, as of September 2026, subword tokenizers remain the default in production models.

## Why not just use words or letters?

There are three obvious ways to turn text into numbers, and each fails in a different way.

| Unit | Vocabulary size | Sequence length | Problem |
|---|---|---|---|
| Whole words | Enormous and open-ended | Short | Any word not in the list (a new name, a typo, "unfollowable") becomes an "unknown" token and its meaning is lost |
| Characters or bytes | Tiny (256 possible bytes) | Very long | Every sentence becomes hundreds of steps; attention cost grows with length, so training and inference get much more expensive |
| Subwords | Tens of thousands to a few hundred thousand | Moderate | The compromise almost everyone uses |

Subwords keep frequent words whole ("the", "model") and break rare ones into reusable parts ("un" + "follow" + "able"). Nothing is ever truly unknown, because in the worst case a word falls back to smaller pieces or raw bytes, and sequences stay short enough to be affordable.

## How byte pair encoding works

BPE started life as a data compression trick. Sennrich, Haddow and Birch adapted it in 2016 to give neural translation systems an open vocabulary: rare words, names and compounds could be translated as sequences of smaller units instead of being dropped (see the [BPE summary](../../papers/techniques/136-bpe-subword-units/summary.md)). The recipe is simple:

```
1. Start with every character (or byte) as its own symbol.
2. Count every adjacent pair of symbols in a large training corpus.
3. Merge the most frequent pair into a new symbol ("t"+"h" -> "th").
4. Repeat steps 2-3 until the vocabulary reaches a target size.

Result: an ordered list of merges. To tokenize new text, apply the
same merges in the same order.

"lowest"  ->  l o w e s t  ->  lo w e s t  ->  low e s t  ->  low est
```

Frequent strings end up as single tokens; rare ones stay split. The vocabulary size is a knob the model builder chooses. For example, OpenAI's `cl100k_base` encoding has roughly 100,000 entries and its successor `o200k_base` roughly 200,000, and Meta's Llama 3 family uses a vocabulary of 128,256 tokens. A bigger vocabulary compresses text into fewer tokens (cheaper, longer effective context) at the cost of a bigger embedding table and more rarely seen tokens.

Modern tokenizers such as OpenAI's `tiktoken` run BPE over **bytes** rather than characters. That guarantees two useful properties the `tiktoken` README spells out: tokenization is reversible and lossless, and it works on any text, including text in scripts the tokenizer never saw in training. Some other model families use related algorithms (WordPiece in [BERT](../../papers/language-models/03-bert/summary.md), SentencePiece in [T5](../../papers/language-models/65-t5/summary.md) and early Llama), but the idea is the same: learn a vocabulary of frequent pieces from data.

For how tokens then become vectors the model computes with, see the hands-on [LLM basics](https://github.com/PatrickWiloak/cloud-data-ai-security-zero-to-hero/blob/main/learn/concepts/llm-basics.md) and [embeddings](https://github.com/PatrickWiloak/cloud-data-ai-security-zero-to-hero/blob/main/learn/concepts/embeddings-and-vector-search.md) pages in the sibling repo.

## Why tokens are the unit of cost

Almost every practical number about a language model is measured in tokens:

- **Price.** API providers charge per million input tokens and per million output tokens.
- **Context window.** The limit on how much a model can read at once is a token count, not a word or page count (see [context windows](context-windows.md)).
- **Speed.** Models generate one token per step, so output speed is quoted in tokens per second, and memory per request grows with every token held in the [KV cache](kv-cache.md).

Because each tokenizer splits text differently, **token counts are not comparable across models**. Anthropic's model documentation is a clear example: as of September 2026 it says 1M tokens is roughly 555,000 English words on the tokenizer introduced with Claude Opus 4.7, while earlier Claude models fit about 750,000 words into the same 1M tokens. Same text, same "1M context", different amount of actual content. When comparing prices or context sizes between vendors, compare on your own text, not on the headline token figure.

## Quirks that come from tokenization

A model does not see letters; it sees token IDs. Several well-known weaknesses follow from that.

- **Spelling and letter counting.** If "strawberry" arrives as two or three chunks, the model has to have memorised which letters live inside each chunk to count the r's. It usually has, imperfectly.
- **Arithmetic.** Numbers are split into chunks that do not line up with place value, and the split can change with the number of digits, so "1234" and "12345" may be broken up quite differently. Some tokenizers split every number into single digits to reduce this; the original [LLaMA](../../papers/language-models/15-llama/summary.md) tokenizer did exactly that.
- **Whitespace and formatting.** A leading space is often part of the token (" hello" and "hello" are different IDs), and code indentation can consume many tokens.
- **Rare and odd tokens.** A vocabulary built on one corpus can contain strings that almost never appear in the model's training data, leaving their embeddings poorly trained, which has produced strange behaviour when users type them.
- **Robustness.** A typo can change a common one-token word into three rare pieces, which is part of why misspellings and adversarial character tricks can shift model behaviour.

These are shrinking as models get larger and see more data, and tool use (letting the model run code to count or calculate) sidesteps many of them. But they are a useful reminder that a model's view of text is not ours.

## Multilingual unfairness

Tokenizers are trained on corpora dominated by a few languages, mostly English. Text in those languages compresses well; text in less represented languages and scripts gets split into many more pieces.

Petrov, La Malfa, Torr and Bibi (2023) measured this directly: the same text translated into different languages can differ in tokenized length by **up to 15 times**, and even character-level and byte-level models showed gaps of **over 4 times** for some language pairs, because some scripts need more bytes per character in UTF-8. They point out three concrete harms:

```
Same meaning, more tokens  ->  higher per-token bill
                           ->  slower responses (more generation steps)
                           ->  less content fits in the context window
```

Larger vocabularies with more non-English coverage narrow the gap, and newer tokenizers have generally grown their vocabularies partly for this reason. But no widely used subword tokenizer treats all languages equally, and the effect compounds with the fact that the models themselves are usually weaker in less represented languages.

## Byte-level and tokenizer-free approaches

If the tokenizer causes all this trouble, why not feed the model raw bytes?

- **ByT5 (Xue et al., 2021)** trained a standard Transformer directly on UTF-8 bytes. It handled any language out of the box, was more robust to noise and did better on spelling-sensitive tasks, while remaining competitive with token models. The catch is length: byte sequences are several times longer than token sequences, and attention cost grows with length.
- **Byte Latent Transformer (Meta, December 2024)** tackles the length problem by grouping bytes into variable-size **patches**, using the predictability (entropy) of the next byte to decide where patches end. Predictable stretches become long patches, hard stretches short ones, so compute goes where it is needed. The paper reports matching token-based Llama 3 style models in a FLOP-controlled study up to 8 billion parameters, with better robustness on rare and noisy inputs.

```
Subword model:   "The quick brown fox"  ->  [The][ quick][ brown][ fox]   (fixed vocabulary)
Byte model:      T h e _ q u i c k ...  ->  19 steps, no vocabulary
Patch model:     [The quick ][b][rown fox]  (boundaries chosen by entropy, learned)
```

As of September 2026 these remain research directions rather than the production default; the frontier models listed in this repo's [model family explainers](../model-families/gpt.md) all still ship with subword tokenizers.

## What to watch

- **Whether byte or patch models reach frontier scale.** A frontier lab shipping a production model without a fixed tokenizer would remove most of the quirks above.
- **Vocabulary growth.** Vocabularies have grown from about 50,000 (GPT-2's was 50,257) to 100,000-200,000+. Watch whether this continues and whether it closes the multilingual gap.
- **Tokenizer changes inside a model family.** As Claude's 2026 tokenizer change shows, a new tokenizer can silently change your token counts, bills and effective context size. Re-measure when you upgrade.

## Read next

- [BPE: Neural Machine Translation of Rare Words with Subword Units](../../papers/techniques/136-bpe-subword-units/summary.md) - where subword tokenization for neural models comes from
- [GPT-2](../../papers/language-models/64-gpt2/summary.md) - popularised byte-level BPE for language models
- [BERT](../../papers/language-models/03-bert/summary.md) and [T5](../../papers/language-models/65-t5/summary.md) - WordPiece and SentencePiece in practice
- [Word2Vec](../../papers/techniques/53-word2vec/summary.md) - what happens to tokens after tokenization: embeddings
- [Context windows](context-windows.md), [KV cache](kv-cache.md), [Sampling and decoding](sampling-and-decoding.md)
- Hands-on: [LLM basics](https://github.com/PatrickWiloak/cloud-data-ai-security-zero-to-hero/blob/main/learn/concepts/llm-basics.md), [Prompt caching](https://github.com/PatrickWiloak/cloud-data-ai-security-zero-to-hero/blob/main/learn/concepts/prompt-caching.md)

## Sources

- Sennrich, Haddow, Birch, "Neural Machine Translation of Rare Words with Subword Units", ACL 2016. https://arxiv.org/abs/1508.07909
- Petrov, La Malfa, Torr, Bibi, "Language Model Tokenizers Introduce Unfairness Between Languages", 2023. https://arxiv.org/abs/2305.15425
- Xue et al., "ByT5: Towards a token-free future with pre-trained byte-to-byte models", TACL 2022. https://arxiv.org/abs/2105.13626
- Pagnoni et al., "Byte Latent Transformer: Patches Scale Better Than Tokens", December 2024. https://arxiv.org/abs/2412.09871
- OpenAI `tiktoken` README (about 4 bytes per token; reversible, works on arbitrary text) and encoding definitions (`cl100k_base`, `o200k_base`). https://github.com/openai/tiktoken
- Touvron et al., "LLaMA: Open and Efficient Foundation Language Models", 2023 (BPE via SentencePiece; numbers split into individual digits). https://arxiv.org/abs/2302.13971
- GPT-2 configuration (`vocab_size: 50257`). https://huggingface.co/openai-community/gpt2
- Llama 3.3 70B configuration (`vocab_size: 128256`). https://huggingface.co/meta-llama/Llama-3.3-70B-Instruct
- Anthropic, Models overview (tokens-to-words ratio on the Claude Opus 4.7 tokenizer vs earlier models), accessed 2026-09-30. https://platform.claude.com/docs/en/about-claude/models/overview
