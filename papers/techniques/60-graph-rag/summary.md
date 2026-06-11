---
title: "GraphRAG: From Local to Global - A Graph RAG Approach to Query-Focused Summarization"
slug: "60-graph-rag"
number: 60
category: "techniques"
authors: "Darren Edge, Ha Trinh, Newman Cheng, Joshua Bradley, Alex Chao, Apurva Mody, Steven Truitt, Jonathan Larson (Microsoft Research)"
published: "April 2024"
year: 2024
url: "https://arxiv.org/abs/2404.16130"
tags: ["retrieval"]
---

# GraphRAG: From Local to Global - A Graph RAG Approach to Query-Focused Summarization

**Authors:** Darren Edge, Ha Trinh, Newman Cheng, Joshua Bradley, Alex Chao, Apurva Mody, Steven Truitt, Jonathan Larson (Microsoft Research)
**Published:** April 2024
**Paper:** [arxiv.org/abs/2404.16130](https://arxiv.org/abs/2404.16130)

---

## Why This Matters

GraphRAG is **the technique that lets RAG answer questions about an entire corpus, not just chunks of it**:

- **Global queries unlocked** - Answers "what are the main themes?" instead of just "what does doc X say?"
- **LLM-extracted knowledge graph** - Entities, relationships, and claims pulled from text into a queryable graph
- **Hierarchical communities** - Leiden algorithm clusters the graph into nested topical communities
- **Pre-computed summaries** - Each community gets an LLM-written summary at index time
- **Map-reduce answering** - Query fans out across community summaries, then aggregates into a final answer
- **Microsoft open-source release** - The reference implementation (`microsoft/graphrag`) became the de facto standard

**Real-world impact:**
- Microsoft shipped GraphRAG into Copilot pipelines and Azure AI Search
- Kicked off a wave of "GraphRAG variants" (LightRAG, HippoRAG, nano-graphrag, KAG)
- Used by enterprises to query private corpora (legal, medical, financial) holistically
- Became the standard answer to "vector RAG can't see the forest for the trees"
- Influenced how every serious RAG system handles long-document and multi-document QA

**The insight:** **Vector RAG retrieves the most similar chunks, but global questions ("what are the dominant narratives in this dataset?") require synthesis across the whole corpus.** Build an LLM-generated knowledge graph at index time, cluster it into communities, summarize each community, then answer queries by aggregating community-level summaries.

---

## The Problem

### Vanilla RAG Hits a Wall on Global Questions

The standard RAG pipeline (paper #13) does well on local questions but fails on holistic ones.

```
User: "What does the policy doc say about remote work?"
  Vanilla RAG: Embed query -> top-k similar chunks -> LLM answers. WORKS.

User: "What are the five biggest themes across all 10,000 of our incident reports?"
  Vanilla RAG: Embed query -> top-k similar chunks -> LLM answers.
    Top-k chunks are about ONE theme (the most semantically similar one).
    The LLM never sees the other nine themes.
    Answer is incomplete, biased toward whichever theme has the most chunks.
  FAILS.
```

### Why Embedding Retrieval Cannot Cover the Corpus

```
Vector RAG assumes: "the answer lives in a small handful of chunks"

But for queries like:
  - "Summarize the main findings of this 500-page report"
  - "What changed across all our quarterly earnings calls in 2023?"
  - "What are the recurring failure modes in this set of bug reports?"

...the answer is DISTRIBUTED across the entire corpus.

Top-k retrieval cannot retrieve "everything." Even k=100 misses 99%
of a 10,000-chunk corpus, and an LLM cannot fit 10,000 chunks in context.
```

### The Naive Alternatives Are Worse

```
Option A: Stuff everything into context (long-context LLM)
  - Costs scale linearly with corpus size
  - Lost-in-the-middle effects on 200K+ contexts
  - Doesn't scale past one book of content

Option B: Hierarchical summarization (RAPTOR-style)
  - Recursive map-reduce summarization works, but is corpus-shaped, not query-shaped
  - Still retrieves leaves, so global queries hit the same problem

Option C: Just map-reduce over all chunks for every query
  - Re-summarizing 10,000 chunks per query is prohibitive
  - Every user question pays the full corpus cost
```

### What's Actually Needed

```
We need an index that:
  1. Captures structure (which entities relate, which topics co-occur)
  2. Summarizes at multiple zoom levels (chunk -> community -> corpus)
  3. Lets queries efficiently route to the right zoom level
  4. Pays the expensive summarization cost ONCE at index time
```

---

## How GraphRAG Works

### Architecture Overview

```
                        INDEX TIME (one-time, expensive)
   +---------------------------------------------------------------+
   |                                                               |
   |  Source     Chunk      LLM extracts        Graph              |
   |  Docs   ->  text   ->  entities &      ->  nodes/edges        |
   |             splits     relationships        (with claims)     |
   |                                                               |
   |                            |                                  |
   |                            v                                  |
   |                                                               |
   |                    Leiden clustering                          |
   |                    (hierarchical communities)                 |
   |                                                               |
   |                            |                                  |
   |                            v                                  |
   |                                                               |
   |                    LLM summarizes each                        |
   |                    community at every level                   |
   |                    (root -> leaf hierarchy)                   |
   |                                                               |
   +---------------------------------------------------------------+

                        QUERY TIME (cheap, parallel)
   +---------------------------------------------------------------+
   |                                                               |
   |   User query                                                  |
   |       |                                                       |
   |       v                                                       |
   |   Pick community level (resolution)                           |
   |       |                                                       |
   |       v                                                       |
   |   MAP: each community summary -> partial answer + score       |
   |       |                                                       |
   |       v                                                       |
   |   REDUCE: combine top-scored partial answers -> final answer  |
   |                                                               |
   +---------------------------------------------------------------+
```

### Step 1: Source Documents to Text Chunks

```
Standard chunking, but the chunk size is a real lever:

  Smaller chunks (300 tokens):
    + LLM extraction is more thorough (fewer entities missed)
    + Higher recall on entity references
    - More LLM calls, higher index cost

  Larger chunks (1200 tokens):
    + Fewer LLM calls
    + More context for relationship extraction
    - Misses some entities the LLM "skips over"

Paper finds: 600-token chunks with overlap is a reasonable default.
```

### Step 2: LLM-Driven Entity and Relationship Extraction

```
For each chunk, prompt the LLM to extract:

  Entities:
    (name, type, description)
    e.g. ("OpenAI", "ORGANIZATION", "AI research company...")

  Relationships:
    (source, target, description, strength)
    e.g. ("OpenAI", "Microsoft", "received $13B investment", 9)

  Claims (optional):
    (subject, object, type, status, period, description)
    e.g. ("OpenAI", "Microsoft", "INVESTMENT", "TRUE", "2023", "...")

Multi-pass "gleaning":
  After first extraction, ask LLM "did you miss any?"
  Repeat 1-2 times to catch entities buried in subordinate clauses.
  Empirically: 1 gleaning pass adds ~10-15% recall, 2nd pass diminishing returns.
```

### Step 3: Build the Entity Graph and Resolve Duplicates

```
Across all chunks:
  - Same entity may appear with slight name variations
    ("OpenAI", "Open AI", "OpenAI Inc.")
  - LLM-generated descriptions need to be merged
  - Edge weights = number of chunks where the relationship appears

Element summarization:
  For each entity (and each edge) with multiple descriptions,
  ask LLM to write ONE consolidated description.

Result: a single weighted graph where nodes = entities,
edges = relationships, with rich text descriptions on both.
```

### Step 4: Hierarchical Community Detection (Leiden Algorithm)

This is the key structural step.

```
Leiden algorithm (Traag et al. 2019) is the modern successor to Louvain:
  - Greedy modularity optimization
  - Guarantees well-connected communities
  - Hierarchical: produces communities at multiple resolutions

GraphRAG runs Leiden hierarchically:

  Level 0 (root):    [whole graph]
                          |
  Level 1:        [C1]   [C2]   [C3]
                   |      |      |
  Level 2:    [C1a][C1b] [C2a] [C3a][C3b][C3c]
                                          |
  Level 3:                            [C3c-i][C3c-ii]

At each level, every node belongs to exactly one community.
Each community at level N is a subset of some community at level N-1.
```

```
Why Leiden specifically:
  - Modularity-based (clusters dense subgraphs)
  - Guarantees connectedness (Louvain can produce disconnected communities)
  - Fast on million-node graphs
  - Hierarchical resolution comes for free
```

### Step 5: Community Summaries (The Index)

```
For each community at every level:

  Inputs:
    - All entity descriptions in the community
    - All relationship descriptions inside (and crossing into) the community
    - Optionally, all claims about those entities

  Prompt the LLM:
    "Given these entities and relationships, write a report-style
     summary of this community: what it's about, key entities,
     key findings, and how it relates to neighbors."

  Output:
    A structured report (title + summary + findings) per community.

Storage:
  Community summaries are the actual queryable index.
  They're the unit of retrieval at query time.
```

```
Cost shape:
  Index time = O(chunks * extraction_cost + communities * summary_cost)
    For a 1M-token corpus, expect ~$5-50 in LLM costs (GPT-4-class models).
  Query time = O(communities_at_level * map_cost + 1 * reduce_cost)
    Typically 10-200 LLM calls per query, runnable in parallel.
```

### Step 6: Query-Time Map-Reduce

```
User asks: "What are the dominant themes in this dataset?"

Step 6a: Pick a community level.
  Level 0: too coarse (one answer about the whole graph)
  Level 1-2: typical sweet spot for global queries
  Higher levels: more detail, more LLM calls

Step 6b: MAP (parallel)
  For each community summary at the chosen level:
    Prompt: "Given this community summary, answer the user's query.
             Also rate how helpful this community is for the answer (0-100)."
    Output: (partial_answer, helpfulness_score)

Step 6c: REDUCE
  Filter out partial answers with score = 0.
  Sort by helpfulness, take top-N until token budget is hit.
  Prompt: "Combine these partial answers into a single coherent response."
  Output: final answer.
```

### Visual: The Two RAG Pipelines

```
VANILLA RAG
   query -> embed -> top-k chunks -> LLM -> answer
   Good at: factoid lookup, "what does the doc say about X"
   Bad at:  global synthesis, "what are the main themes"

GRAPH RAG (Global)
   query -> community level -> MAP over all summaries at that level
                                        |
                                        v
                            (partial_answer, score) per community
                                        |
                                        v
                            REDUCE top-scored partials -> answer
   Good at: holistic queries, multi-doc synthesis, "themes/trends"
   Cost:    higher per query (many LLM calls), but parallelizable

GRAPH RAG (Local)
   query -> identify relevant entities -> expand to neighborhood
        -> gather attached chunks + relationships + claims
        -> LLM -> answer
   Good at: entity-centric questions, "tell me about X and its connections"
```

---

## Key Innovations

### 1. LLM as the Knowledge Graph Builder

```
Pre-LLM era: knowledge graphs required NER models, relation extractors,
ontology design, schema engineering, and human curation. Months of work.

GraphRAG: a single prompt to a general LLM extracts a usable graph
from arbitrary text. No schema, no labeled training data.

Trade-off: the graph is "fuzzy" (entity descriptions are free text,
not typed properties), but it's good enough for retrieval.
```

### 2. Communities as the Retrieval Unit

```
Vector RAG retrieves chunks. GraphRAG retrieves community summaries.

A community summary captures:
  - What entities cluster together
  - How they relate
  - The "story" of that subgraph

This is exactly the right granularity for global queries:
not too small (chunks miss the forest), not too big (the corpus
is too much for one prompt).
```

### 3. Hierarchical Resolution as a Cost/Quality Knob

```
Same query, different community levels:

  Level 1 (10 communities):  ~10 map calls, broad strokes answer
  Level 2 (50 communities):  ~50 map calls, more nuance
  Level 3 (200 communities): ~200 map calls, fine-grained but expensive

The user (or system) picks the resolution per query.
Vanilla RAG has no equivalent dial.
```

### 4. Map-Reduce With Helpfulness Scoring

```
Naive map-reduce over a corpus is wasteful: most chunks aren't relevant.

GraphRAG's twist: every map step ALSO returns a 0-100 helpfulness score.
The reduce step ignores zeros and prioritizes high-scoring partials.

This makes the reduce step linear in actual signal, not corpus size.
```

### 5. Pre-computed Summaries Amortize Cost

```
The expensive work happens ONCE at index time:
  - Entity extraction (per chunk)
  - Element summarization (per node/edge)
  - Community summarization (per community at every level)

Query time only does:
  - Map across community summaries (cheap, parallel)
  - One reduce step
```

---

## Performance and Evaluation

### Datasets Used

```
Two real-world corpora:
  - Podcast transcripts (Behind the Tech with Kevin Scott): ~1M tokens
  - News articles dataset: ~1.7M tokens

Both too large for context-stuffing, both with rich global structure.
```

### Evaluation Method

```
"Sensemaking" questions generated by an LLM persona pipeline:
  - 125 questions per dataset
  - Designed to require global synthesis (not factoid lookup)
  - Examples: "What are the recurring criticisms of...?",
              "How has the narrative around X evolved...?"

Head-to-head LLM-as-judge comparisons across these axes:
  - Comprehensiveness (covers all relevant points)
  - Diversity (multiple perspectives)
  - Empowerment (helps the reader form their own view)
  - Directness (specific to the question, no waffle)
```

### Headline Result

| Method | Comprehensiveness | Diversity | Directness |
|--------|-------------------|-----------|------------|
| Naive RAG (top-k) | Baseline | Baseline | Wins |
| Text summarization (TS) | Mixed | Mixed | Loses |
| **GraphRAG (C2 community level)** | **Wins ~72-83%** | **Wins ~72-82%** | Loses |
| GraphRAG (root level) | Wins comprehensiveness | Wins diversity | Loses |

```
Read this as: GraphRAG produces more thorough, multi-faceted answers
on global questions, at the cost of being less terse than naive RAG.

For factoid questions, naive RAG is still the right tool.
GraphRAG wins specifically on the queries it was designed for.
```

### Token Cost vs. Quality

```
Index cost (one-time):
  ~600K-1M tokens of LLM calls per 1M tokens of corpus
  Dominated by entity extraction; community summarization is smaller

Query cost:
  Naive RAG:    ~5-10K tokens per query
  GraphRAG L2:  ~30-100K tokens per query (parallel map calls)
  GraphRAG L1:  ~10-30K tokens per query

Conclusion: GraphRAG is 3-10x more expensive per query than naive RAG,
but produces qualitatively different answers on global questions.
```

---

## Real-World Applications

### Where GraphRAG Wins

```
1. Enterprise knowledge bases
   "What policies have changed across our employee handbook revisions?"

2. Legal discovery
   "What are the patterns of communication between these custodians?"

3. Scientific literature review
   "What competing hypotheses exist on topic X across these 500 papers?"

4. Investigative journalism / OSINT
   "Who connects to whom across these leaked documents?"

5. Customer support analytics
   "What are the recurring failure modes in support tickets this quarter?"

6. Compliance / audit
   "What risk themes appear across all incident reports in 2024?"
```

### Where GraphRAG Loses

```
1. Factoid lookup ("What's the address in section 3?") - vanilla RAG wins
2. Real-time / fresh data - the index is expensive to rebuild
3. Small corpora - just stuff it into context, don't bother
4. Highly structured data (SQL tables, knowledge graphs) - use those directly
5. Cost-sensitive consumer apps - the per-query cost adds up
```

### Microsoft's Production Use

```
Microsoft uses GraphRAG (and variants) inside:
  - Azure AI Search "agentic retrieval" pipelines
  - Microsoft Research's internal sensemaking tools
  - Customer-facing offerings that need long-document QA

The open-source repo (microsoft/graphrag) is the reference implementation.
LazyGraphRAG (a 2024 follow-up from the same team) defers community
summarization to query time to cut index cost dramatically.
```

---

## Comparison to Vector RAG

```
+-----------------------+--------------------+--------------------+
| Dimension             | Vector RAG         | GraphRAG           |
+-----------------------+--------------------+--------------------+
| Index unit            | Chunk + embedding  | Entity graph +     |
|                       |                    | community summary  |
| Index cost            | Low (embed only)   | High (LLM-heavy)   |
| Query cost            | Low (1 LLM call)   | Med-High (N calls) |
| Best for              | Local/factoid      | Global/holistic    |
| Worst for             | "Main themes" Qs   | "Quote section X"  |
| Freshness             | Easy (re-embed)    | Hard (re-extract)  |
| Explainability        | Cited chunks       | Cited communities  |
| Hallucination control | Source attribution | Source attribution |
| Multi-hop reasoning   | Weak               | Strong             |
+-----------------------+--------------------+--------------------+
```

```
The right architecture is usually BOTH:
  - Route factoid questions to vector RAG (cheap, fast)
  - Route holistic questions to GraphRAG (thorough)
  - A small classifier or LLM-as-router decides per query
```

---

## The Open-Source Library

Microsoft released `microsoft/graphrag` on GitHub with:

```python
# Index a corpus
graphrag index --root ./my_project

# Run a global query (map-reduce over community summaries)
graphrag query --root ./my_project \
    --method global \
    --query "What are the main themes in this corpus?"

# Run a local query (entity-centric expansion)
graphrag query --root ./my_project \
    --method local \
    --query "Tell me about Entity X and its connections"
```

```
Storage backends:
  - Local files (parquet)
  - Azure Blob Storage
  - CosmosDB

LLM backends:
  - Azure OpenAI
  - OpenAI
  - Any OpenAI-compatible endpoint (vLLM, Together, etc.)

The library has been forked dozens of times. Notable derivatives:
  - nano-graphrag: minimal Python implementation, ~1k LOC
  - LightRAG: cheaper variant with single-level communities
  - HippoRAG: graph + personalized PageRank for retrieval
  - KAG: knowledge-graph-augmented RAG for domain reasoning
```

---

## Limitations

### 1. Index Cost Is Real

```
Indexing a 10M-token corpus with GPT-4 can run $500-5000 in LLM costs.
For corpora that change weekly, this is a serious operational expense.
LazyGraphRAG and incremental indexing variants exist but are still maturing.
```

### 2. Extraction Quality Bounds Everything

```
The entire downstream pipeline rests on entity/relationship extraction.
Domain-specific corpora (legal, medical) need prompt-tuning to extract
the right entity types, otherwise the graph is shallow.
```

### 3. Community Summaries Can Drift

```
LLM-written summaries can:
  - Miss long-tail entities
  - Hallucinate relationships under pressure
  - Lose grounding when communities are too large

Mitigations: cite back to source chunks, restrict to extracted facts.
```

### 4. Not Truly "Global"

```
Even map-reduce over communities is a sample of the corpus, not
the full thing. Pathological queries that require integrating
100% of the data still struggle.
```

### 5. Evaluation Is Fuzzy

```
LLM-as-judge on "comprehensiveness" is itself imperfect.
Real human evaluation on global queries is expensive and rare.
Leaderboard numbers should be read as directional, not precise.
```

### 6. Updates Are Expensive

```
Adding 1% new documents shouldn't require a 100% reindex,
but maintaining graph + community structure under streaming updates
is an active research problem.
```

---

## Connections to Other Papers

### Builds on / extends

- **RAG (paper #13, Lewis et al. 2020)** - GraphRAG is RAG with a graph index instead of (or alongside) a vector index. Same retrieve-then-read shape, different retrieval substrate.
- **HyDE (paper #44 area)** - Both improve retrieval at the cost of more LLM calls. HyDE on the query side, GraphRAG on the index side.
- **Embedding models (paper #14 area)** - GraphRAG still uses embeddings inside (entity dedup, similarity for local search). It's a complement, not a replacement.
- **RAPTOR (Sarthi et al. 2024)** - Hierarchical summarization via clustering. RAPTOR clusters embeddings; GraphRAG clusters an entity graph. RAPTOR is purer summarization, GraphRAG adds structure.

### Influenced

- **LazyGraphRAG (Microsoft, 2024)** - Skips community summarization at index time, does query-conditioned summarization on demand. Cuts index cost ~700x.
- **LightRAG (HKU, 2024)** - Single-level retrieval with graph-aware deduplication. Cheaper than GraphRAG, often comparable quality.
- **HippoRAG (OSU, 2024)** - Adds Personalized PageRank over the entity graph for memory-style retrieval.
- **KAG (Ant Group, 2024)** - Combines GraphRAG-style indexing with mutual indexing of structured KGs and unstructured text.

### Adjacent infrastructure

- **PagedAttention / vLLM (paper #52)** - GraphRAG's parallel map calls benefit massively from high-throughput LLM serving.
- **Long-context models (Gemini 1.5, Claude 3, GPT-4-128K)** - The "competitor" approach. GraphRAG and long context are complementary: structured retrieval + big context window.

### Theoretical foundations

- **Leiden algorithm (Traag et al. 2019)** - The community detection method GraphRAG uses. Modularity-optimized and connectedness-guaranteed.
- **Graph summarization literature** - Decades of work on summarizing large graphs; GraphRAG is the LLM-era reincarnation.

---

## Key Takeaways

1. **Vanilla RAG cannot answer global questions** because top-k retrieval cannot cover the whole corpus, and a corpus rarely fits in context.
2. **Use the LLM to build a knowledge graph at index time.** Entity and relationship extraction from a single prompt is good enough to power retrieval, no schema engineering needed.
3. **Cluster the graph hierarchically (Leiden) to get communities at multiple resolutions.** This gives you a tunable zoom level for queries.
4. **Pre-compute community summaries.** They become the queryable unit, replacing chunks for global queries.
5. **Answer queries with map-reduce + helpfulness scoring.** Map across community summaries in parallel, score, reduce the top-scored partials.
6. **GraphRAG is not a replacement for vector RAG.** It's the global-question complement. Production systems usually run both and route by query type.
7. **The index is expensive; the queries are tractable.** Amortize the LLM cost at index time, keep query latency reasonable.

**Bottom line:** GraphRAG was the first widely adopted technique to push RAG past chunk-level retrieval into corpus-level synthesis. By using an LLM to build a knowledge graph, clustering it into hierarchical communities, and pre-computing community summaries, it makes "what are the main themes?" answerable at scale. The Microsoft open-source release made the approach accessible and triggered a wave of follow-up systems. For any production RAG system that needs to handle holistic queries, some form of graph-based or hierarchical retrieval is now table stakes.

---

## Further Reading

### Original Paper
- **GraphRAG:** https://arxiv.org/abs/2404.16130

### Code
- **microsoft/graphrag (reference impl):** https://github.com/microsoft/graphrag
- **nano-graphrag (minimal Python port):** https://github.com/gusye1234/nano-graphrag

### Microsoft Blog Posts
- **GraphRAG announcement:** https://www.microsoft.com/en-us/research/blog/graphrag-unlocking-llm-discovery-on-narrative-private-data/
- **LazyGraphRAG follow-up:** https://www.microsoft.com/en-us/research/blog/lazygraphrag-setting-a-new-standard-for-quality-and-cost/

### Related Papers
- **Original RAG:** Paper #13 in this repo (Lewis et al. 2020)
- **HyDE:** Hypothetical Document Embeddings for query expansion
- **RAPTOR:** https://arxiv.org/abs/2401.18059 (hierarchical summarization for retrieval)
- **HippoRAG:** https://arxiv.org/abs/2405.14831 (PageRank over LLM-extracted KG)
- **LightRAG:** https://arxiv.org/abs/2410.05779 (cheaper graph-aware RAG)

### Algorithm References
- **Leiden algorithm:** Traag, Waltman, van Eck (2019) - https://www.nature.com/articles/s41598-019-41695-z

---

**Published:** April 2024
**Impact:** 🔥🔥🔥🔥 **HIGH** - Defined the "graph-based RAG" category and shipped a usable reference implementation
**Citations:** 800+ (as of early 2026)
**Adoption:** Standard option in enterprise RAG stacks; spawned a family of derivative systems
**Current Relevance:** Live area of research and production deployment; routinely combined with long-context models and vector retrieval
**Legacy:** Established that LLMs can build their own retrieval indexes (graphs and summaries), shifting RAG from a pure embedding problem to a structured-knowledge problem

**Modern Status (early 2026):** GraphRAG is now one of three canonical RAG patterns alongside vector RAG and long-context "stuffing." Production systems typically route queries between them. The Microsoft repo continues active development; LazyGraphRAG is the recommended variant for cost-sensitive deployments. The broader lesson - that the LLM itself is the most flexible index-builder - has propagated into agentic retrieval, memory systems, and structured-output pipelines across the field.

<!-- related:start -->

---

## Related in This Collection

- [Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks (RAG)](../../techniques/13-rag/summary.md)
- [GPT-4 Technical Report](../../language-models/36-gpt4/summary.md)
- [PagedAttention: Efficient LLM Serving with vLLM](../../techniques/52-pagedattention-vllm/summary.md)

<!-- related:end -->
