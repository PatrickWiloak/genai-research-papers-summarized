---
title: "Dense Passage Retrieval, ColBERT, and Sentence-BERT: The Retrieval Half of RAG"
slug: "87-dense-retrieval"
number: 87
category: "techniques"
authors: "Vladimir Karpukhin, Barlas Oguz, Sewon Min, Patrick Lewis, Ledell Wu, Sergey Edunov, Danqi Chen, Wen-tau Yih (Facebook AI, University of Washington, Princeton) - DPR; Omar Khattab, Matei Zaharia (Stanford) - ColBERT; Nils Reimers, Iryna Gurevych (TU Darmstadt) - Sentence-BERT"
published: "April 2020 (DPR, EMNLP 2020); April 2020 (ColBERT, SIGIR 2020); August 2019 (Sentence-BERT, EMNLP 2019)"
year: 2020
url: "https://arxiv.org/abs/2004.04906"
tags: ["retrieval", "embeddings", "search"]
---

# Dense Passage Retrieval, ColBERT, and Sentence-BERT: The Retrieval Half of RAG

**Authors:** Vladimir Karpukhin, Barlas Oguz, Sewon Min, Patrick Lewis, Ledell Wu, Sergey Edunov, Danqi Chen, Wen-tau Yih (Facebook AI, University of Washington, Princeton) - DPR; Omar Khattab, Matei Zaharia (Stanford) - ColBERT; Nils Reimers, Iryna Gurevych (TU Darmstadt) - Sentence-BERT
**Published:** April 2020 (DPR, EMNLP 2020); April 2020 (ColBERT, SIGIR 2020); August 2019 (Sentence-BERT, EMNLP 2019)
**Papers:** [DPR arxiv.org/abs/2004.04906](https://arxiv.org/abs/2004.04906) | [ColBERT arxiv.org/abs/2004.12832](https://arxiv.org/abs/2004.12832) | [Sentence-BERT arxiv.org/abs/1908.10084](https://arxiv.org/abs/1908.10084)

---

## Why This Matters

Everyone building with LLMs uses [RAG](../13-rag/summary.md), and RAG is two systems: a retriever and a generator. The generator gets all the attention. **The retriever is where RAG systems actually fail**, and these three papers are its foundation.

- **DPR** proved that learned dense embeddings beat keyword search (BM25) on open-domain question answering, by a large margin.
- **Sentence-BERT** made it computationally possible to embed sentences for similarity search, turning a 65-hour comparison task into about 5 seconds.
- **ColBERT** introduced late interaction, recovering much of the accuracy of expensive cross-encoders at near-dense-retrieval speed - the basis of modern reranking.
- **Every vector database and embedding API** you use - Pinecone, Weaviate, pgvector, OpenAI embeddings, Cohere Embed - implements the architecture these papers established.

**The insight:** keyword search matches strings. Questions and answers frequently share no strings. "Who wrote the Declaration of Independence" and a passage saying "Thomas Jefferson drafted the document in 1776" have almost no lexical overlap. Encode both into a shared vector space where meaning, not spelling, determines proximity, and the match becomes trivial.

---

## The Problem: Lexical Search Cannot Match Meaning

For decades, retrieval meant **BM25** - a well-tuned bag-of-words scoring function. It is fast, needs no training, generalizes to any corpus, and is still a strong baseline in 2026. Its ceiling is the vocabulary mismatch problem:

```
Query:   "How much does it cost to fix a broken windshield?"
Passage: "Auto glass replacement typically runs $200-$400."

Shared content words: essentially none.
BM25 score: near zero.
Semantic relevance: exact.
```

The obvious fix - embed both with BERT and compare - had a fatal cost problem. **BERT is a cross-encoder**: to score a (query, passage) pair it must process them together in one forward pass. Scoring one query against 21 million Wikipedia passages means 21 million BERT passes. Sentence-BERT measured this: finding the most similar pair in a collection of 10,000 sentences took about 65 hours with BERT.

---

## The Core Innovation

### Sentence-BERT: encode once, compare cheaply

Use a **bi-encoder**: run each text through BERT separately, pool to a single fixed vector, and compare with cosine similarity. Fine-tune with a siamese/triplet objective on natural language inference and similarity data so the resulting vectors are actually meaningful under cosine distance (raw BERT [CLS] vectors are notoriously not).

```
CROSS-ENCODER (accurate, unusable at scale)
   [query] [SEP] [passage]  ->  BERT  ->  relevance score
   Must run once PER PAIR. O(N) BERT passes per query.

BI-ENCODER (fast, slightly less accurate)
   [query]    ->  BERT  ->  q vector  \
                                        cosine similarity
   [passage]  ->  BERT  ->  p vector  /
   Passages embedded ONCE, offline, and indexed.
   Query time = 1 BERT pass + a nearest-neighbor lookup.
```

The 65-hour comparison task drops to roughly 5 seconds. This architectural split - expensive offline indexing, cheap online lookup - is what makes vector search possible at all.

### DPR: train the bi-encoder for retrieval specifically

Sentence-BERT made bi-encoders practical; DPR made them accurate for question answering. Two separate BERT encoders (one for questions, one for passages) trained contrastively:

```
Training objective: for each question, pull its correct passage
close and push wrong passages away.

The crucial detail - IN-BATCH NEGATIVES:
  In a batch of B question/passage pairs, each question uses the
  other B-1 passages as negatives. One forward pass yields B x B
  comparisons instead of B. Bigger batches mean more and harder
  negatives, and directly better retrieval.

Plus HARD NEGATIVES: passages that BM25 ranks highly but that
do not contain the answer. These teach the fine distinctions
that random negatives never will.
```

DPR reported top-20 retrieval accuracy of about 78.4 percent on Natural Questions versus about 59.1 percent for BM25 - a very large gap on the same corpus, and the result that convinced the field dense retrieval was ready.

### ColBERT: late interaction, the middle ground

Bi-encoders compress a whole passage into one vector, which loses detail. Cross-encoders keep all detail but cannot scale. ColBERT keeps **one vector per token** and defers the interaction:

```
MaxSim scoring:

  For each QUERY token, find its best-matching PASSAGE token
  (maximum cosine similarity), then sum those maxima.

  score(q, p) = sum over query tokens i of
                  max over passage tokens j of  ( q_i . p_j )

  Passage token vectors are precomputed offline.
  Only the cheap MaxSim operation happens at query time.
```

This preserves term-level matching signal - which specific words matched what - while keeping the expensive encoding offline. ColBERTv2 added residual compression to make the index size practical, since one vector per token is otherwise enormous.

---

## Key Components Explained

### 1. In-Batch Negatives
**What it does:** Makes contrastive training efficient enough to work.
**How it works:** Reusing every other passage in the batch as a negative turns B examples into B-squared comparisons. This is why embedding models are trained with very large batches, and why batch size is a first-order hyperparameter for retrieval quality rather than a memory detail.

### 2. Hard Negative Mining
**What it does:** Teaches the distinctions that matter.
**How it works:** Random negatives are trivially separable - a question about windshields versus a passage about the French Revolution. Hard negatives (retrieved by BM25 or by a previous version of the model) look relevant but are not, and are where the useful gradient lives. Iterative hard-negative mining is standard in every modern embedding model recipe.

### 3. The Retrieve-Then-Rerank Pipeline
**What it does:** Combines all three architectures into the standard production design.
**How it works:**
```
  Stage 1 RETRIEVE:  bi-encoder + vector index (+ often BM25)
                     millions of docs -> top 100        [fast]
  Stage 2 RERANK:    cross-encoder or ColBERT
                     top 100 -> top 10                  [accurate]
  Stage 3 GENERATE:  LLM reads the top 10
```
The reranker is the highest-return, most-skipped component of production RAG. It costs one small model call over 100 candidates and routinely produces double-digit improvements in answer quality.

### 4. Hybrid Search
**What it does:** Covers dense retrieval's blind spots.
**How it works:** Dense retrievers fail on exact identifiers - product codes, error numbers, rare proper nouns, API names - precisely where BM25 excels. Running both and fusing the rankings (typically with Reciprocal Rank Fusion) beats either alone on nearly every real corpus. **If your RAG system is dense-only, this is usually the cheapest available improvement.**

### 5. Approximate Nearest Neighbor Search
**What it does:** Makes million-scale vector search fast.
**How it works:** Exact nearest-neighbor search over 10 million vectors is too slow. **HNSW** (hierarchical navigable small world graphs) and IVF-PQ (inverted file with product quantization) trade a small amount of recall for orders of magnitude in speed. FAISS, which DPR used, is the reference implementation, and every vector database is built on one of these algorithms.

---

## Key Results

- **DPR:** top-20 accuracy about 78.4 percent versus BM25's about 59.1 percent on Natural Questions, with gains of roughly 9 to 19 points across five open-domain QA datasets; new state of the art on several end-to-end QA benchmarks when paired with a reader.
- **Sentence-BERT:** reduced the 10,000-sentence pairwise comparison task from about 65 hours to about 5 seconds while improving on prior sentence-embedding methods for semantic similarity.
- **ColBERT:** recovered most of a cross-encoder's effectiveness at orders-of-magnitude lower query latency; ColBERTv2 made the index size practical and set strong results on BEIR.
- **BEIR** (the zero-shot retrieval benchmark, 2021) delivered the important caveat: **BM25 remains highly competitive out of domain**, and dense retrievers trained on one domain often fail to transfer.

---

## Why This Was Revolutionary

- **Made semantic search work at web scale**, which is the precondition for RAG existing as a technique.
- **Established the bi-encoder / cross-encoder / late-interaction taxonomy** that organizes all of retrieval today.
- **Created the vector database industry.** Pinecone, Weaviate, Qdrant, Milvus, Chroma, and pgvector all serve the workload DPR defined.
- **In-batch negatives became the standard contrastive recipe**, reused far beyond retrieval - [CLIP](../../multimodal/08-clip/summary.md) trains the same way, with images and captions in place of questions and passages.

---

## Real-World Impact

- **Every RAG system** built on LangChain, LlamaIndex, or a homegrown stack uses a bi-encoder for stage one, whether or not the builder knows the lineage.
- **Embedding APIs** (OpenAI, Cohere, Voyage) and open models (E5, BGE, GTE, Nomic, Jina) are DPR's architecture with better training data, larger batches, and better hard negatives. The MTEB leaderboard tracks them.
- **Rerankers** (Cohere Rerank, BGE-reranker, ColBERT-based services) are the cross-encoder and late-interaction stage, now a standard product category.
- **Beyond text.** Multimodal retrieval, code search, and recommendation systems all use the same dual-encoder-plus-ANN pattern.
- **[GraphRAG](../60-graph-rag/summary.md)** and agentic retrieval build on top of this stage rather than replacing it - the embedding lookup is still there underneath.

---

## Key Takeaways for Practitioners

1. **Add a reranker before you do anything else.** Retrieve 50-100 with the bi-encoder, rerank to 5-10 with a cross-encoder. It is the single highest-return change in most RAG systems.
2. **Use hybrid search.** BM25 plus dense with rank fusion. Dense retrieval alone will miss exact identifiers, error codes, and rare names, and those failures are the ones users notice.
3. **Chunking dominates quality.** A single vector per chunk means chunk boundaries decide what can be retrieved. Test chunk sizes empirically; use overlap; keep semantic units intact.
4. **Check domain transfer before trusting benchmarks.** BEIR's lesson is that an embedding model excellent on web QA may underperform BM25 on your legal, medical, or internal-jargon corpus. Evaluate on your data.
5. **Batch size matters if you train your own.** In-batch negatives mean small batches produce weak embedding models.
6. **Build a retrieval eval set separate from your answer eval set.** Most RAG failures are retrieval failures, and you cannot see them by looking only at final answers.

---

## Limitations & Future Directions

- **Out-of-domain generalization** is dense retrieval's persistent weakness, and the main reason BM25 has not been retired.
- **One vector per chunk is lossy.** A long passage covering several topics compresses to a single point. Late interaction and multi-vector approaches address this at the cost of index size.
- **Fixed-size embeddings** cannot represent everything a document says; Matryoshka embeddings (variable truncatable dimensions) are a partial answer.
- **Index maintenance.** Updating documents means re-embedding; changing embedding models means rebuilding the entire index, which is a real operational cost.
- **Long-context models compete with retrieval** for short corpora - if the whole corpus fits in a 1M-token context, retrieval may be unnecessary. For large or frequently changing corpora, retrieval remains far cheaper and more current.
- **Learned sparse retrieval** (SPLADE and relatives) produces sparse vectors with term-level interpretability and strong out-of-domain behavior, and is an underused middle path.

---

## Further Reading

- **DPR:** [arxiv.org/abs/2004.04906](https://arxiv.org/abs/2004.04906)
- **ColBERT:** [arxiv.org/abs/2004.12832](https://arxiv.org/abs/2004.12832) | **ColBERTv2:** [arxiv.org/abs/2112.01488](https://arxiv.org/abs/2112.01488)
- **Sentence-BERT:** [arxiv.org/abs/1908.10084](https://arxiv.org/abs/1908.10084)
- **BEIR (zero-shot retrieval benchmark):** [arxiv.org/abs/2104.08663](https://arxiv.org/abs/2104.08663)
- **In this collection:** [RAG](../13-rag/summary.md), [GraphRAG](../60-graph-rag/summary.md), [BERT](../../language-models/03-bert/summary.md), [CLIP](../../multimodal/08-clip/summary.md)

## Citation

```bibtex
@inproceedings{karpukhin2020dense,
  title={Dense Passage Retrieval for Open-Domain Question Answering},
  author={Karpukhin, Vladimir and O{\u{g}}uz, Barlas and Min, Sewon and Lewis, Patrick and Wu, Ledell and Edunov, Sergey and Chen, Danqi and Yih, Wen-tau},
  booktitle={Proceedings of EMNLP},
  year={2020}
}

@inproceedings{khattab2020colbert,
  title={ColBERT: Efficient and Effective Passage Search via Contextualized Late Interaction over BERT},
  author={Khattab, Omar and Zaharia, Matei},
  booktitle={Proceedings of SIGIR},
  year={2020}
}
```

<!-- related:start -->

---

## Related in This Collection

- [BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](../../language-models/03-bert/summary.md)
- [Learning Transferable Visual Models From Natural Language Supervision (CLIP)](../../multimodal/08-clip/summary.md)
- [Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks (RAG)](../../techniques/13-rag/summary.md)
- [GraphRAG: From Local to Global - A Graph RAG Approach to Query-Focused Summarization](../../techniques/60-graph-rag/summary.md)

<!-- related:end -->
