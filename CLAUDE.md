# GenAI Research Papers Summarized

Curated collection of 63 foundational generative AI papers with comprehensive summaries.

## Structure
- `papers/` - Individual paper summaries (one folder per paper, `summary.md` inside)
- `papers/_TEMPLATE.md` - Template for new summaries
- `INDEX.md` - Generated category-grouped index of every paper
- `papers.json` / `papers.csv` - Generated machine-readable manifest
- `scripts/build_manifest.py` - Regenerates frontmatter, manifest, INDEX.md, and mkdocs nav (stdlib only, idempotent)
- `mkdocs.yml` + `requirements.txt` - MkDocs Material site (auto-deploys via `.github/workflows/pages.yml`)
- `CONTRIBUTING.md` - How to add a paper + house style
- `docs/ROADMAP.md` - Learning path for newcomers
- `docs/READING_GUIDE.md` - Historical vs modern relevance
- `docs/QUICK_REFERENCE.md` - Fast lookup
- `docs/COMPARISONS.md` - Decision guides
- `docs/GLOSSARY.md` - Term definitions

## Maintenance
- After adding or editing any `papers/**/summary.md`, run `python3 scripts/build_manifest.py` and commit the regenerated `papers.json`, `papers.csv`, `INDEX.md`, `mkdocs.yml`, and the frontmatter changes.
- Do not hand-edit YAML frontmatter or `INDEX.md` - they are generated.

## Usage
Educational resource - no code, just documentation. Start with ROADMAP.md for learning path.

## graphify

This project has a knowledge graph at graphify-out/ with god nodes, community structure, and cross-file relationships.

Rules:
- For codebase questions, first run `graphify query "<question>"` when graphify-out/graph.json exists. Use `graphify path "<A>" "<B>"` for relationships and `graphify explain "<concept>"` for focused concepts. These return a scoped subgraph, usually much smaller than GRAPH_REPORT.md or raw grep output.
- If graphify-out/wiki/index.md exists, use it for broad navigation instead of raw source browsing.
- Read graphify-out/GRAPH_REPORT.md only for broad architecture review or when query/path/explain do not surface enough context.
- After modifying code, run `graphify update .` to keep the graph current (AST-only, no API cost).
