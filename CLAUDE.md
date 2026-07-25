# GenAI Research Papers Summarized

Curated collection of 68 foundational generative AI papers with comprehensive summaries.

## Structure
- `papers/` - Individual paper summaries (one folder per paper, `summary.md` inside)
- `papers/_TEMPLATE.md` - Template for new summaries
- `INDEX.md` - Generated category-grouped index of every paper
- `papers.json` / `papers.csv` - Generated machine-readable manifest
- `scripts/build_manifest.py` - Regenerates frontmatter, manifest, INDEX.md, mkdocs nav, and the `site-build/` tree (stdlib only, idempotent)
- `scripts/add_cross_links.py` - Regenerates the "Related in This Collection" footer on each summary (stdlib only, idempotent)
- `scripts/check_links.py` - Validates relative Markdown links (used by CI)
- `mkdocs.yml` + `requirements.txt` - MkDocs Material site (auto-deploys via `.github/workflows/pages.yml`)
- `.github/workflows/ci.yml` - Link check + generated-content freshness gate
- `CONTRIBUTING.md` - How to add a paper + house style
- `docs/ROADMAP.md` - Learning path for newcomers
- `docs/READING_GUIDE.md` - Historical vs modern relevance
- `docs/QUICK_REFERENCE.md` - Fast lookup
- `docs/COMPARISONS.md` - Decision guides
- `docs/GLOSSARY.md` - Term definitions

## Maintenance
- After adding or editing any `papers/**/summary.md`, run the regeneration pipeline and commit the result:
  ```
  python3 scripts/build_manifest.py     # frontmatter, manifest, INDEX.md, mkdocs nav
  python3 scripts/add_cross_links.py     # "Related in This Collection" footers
  python3 scripts/build_manifest.py     # refresh after footers
  ```
- CI (`.github/workflows/ci.yml`) fails if these generated outputs are stale or if any relative link is broken, so run them before pushing.
- Do not hand-edit YAML frontmatter, `INDEX.md`, or the `<!-- related:* -->` footers - they are generated.
- When adding a new paper, give it the next number (currently up to 68), add its aliases to the `ALIASES` map in `scripts/add_cross_links.py` (so other papers can link to it), and add its topic tags to the `TOPICS` map in `scripts/build_manifest.py` (so it appears in `TAGS.md` and gets `tags:` frontmatter).

## Usage
Educational resource - no code, just documentation. Start with ROADMAP.md for learning path.
