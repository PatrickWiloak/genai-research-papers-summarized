# GenAI Research Papers Summarized

## Overview
Curated collection of 107 foundational generative AI papers with comprehensive summaries.
Docs-only educational resource (no application code) - markdown summaries plus a stdlib
Python regeneration pipeline that builds the index, manifests, and an MkDocs Material site.

## Structure
- `papers/` - Paper summaries grouped into category subfolders (`architectures/`, `image-generation/`, `language-models/`, `multimodal/`, `techniques/`); each summary is a `summary.md`
- `papers/_TEMPLATE.md` - Template for new summaries
- `INDEX.md` - Generated category-grouped index of every paper
- `papers.json` / `papers.csv` - Generated machine-readable manifest
- `scripts/build_manifest.py` - Regenerates frontmatter, manifest, INDEX.md, TAGS.md, `mkdocs.generated.yml`, and the `site-build/` tree (stdlib only, idempotent)
- `scripts/add_cross_links.py` - Regenerates the "Related in This Collection" footer on each summary (stdlib only, idempotent)
- `scripts/check_links.py` - Validates relative Markdown links (used by CI)
- `mkdocs.yml` - Hand-maintained MkDocs Material config (theme, palette, extensions). Carries **no** `nav`; the generator appends one into the git-ignored `mkdocs.generated.yml`, which is what the site builds from
- `.github/site/extra.css` - Site-only stylesheet: near-black + red palette, cards, landing-page styles. Staged to `site-build/assets/site/`
- `.github/site/home.md` - Site-only landing page. Written over the staged `README.md` so it becomes the site root; `{{token}}` counts are filled by `render_home()` so they cannot drift
- `requirements.txt` - Pinned docs toolchain (mkdocs-material, minify, pymdown-extensions)
- `.github/workflows/ci.yml` - Link check + generated-content freshness gate
- `.github/workflows/pages.yml` - Strict site build (blocking gate on every PR) + GitHub Pages deploy from `main`
- `CONTRIBUTING.md` - How to add a paper + house style
- `docs/ROADMAP.md` - Learning path for newcomers
- `docs/READING_GUIDE.md` - Historical vs modern relevance
- `docs/QUICK_REFERENCE.md` - Fast lookup
- `docs/COMPARISONS.md` - Decision guides
- `docs/GLOSSARY.md` - Term definitions
- `docs/GAPS.md` - Coverage map + queued papers (update when adding or spotting a gap)
- `assets/brand/` - Banner images used by the README promo block (mirrored into `site-build/` by the build script)

## Purpose / Usage
- Educational resource - no code, just documentation. Start with `docs/ROADMAP.md` for the learning path.
- `INDEX.md` is the category-grouped entry point in the repo; on the site, `.github/site/home.md` is the landing page and `README.md` stays the GitHub front page.
- The MkDocs Material site auto-deploys via `.github/workflows/pages.yml` to <https://patrickwiloak.github.io/genai-research-papers-summarized/>. It shares its near-black look with the sibling `cloud-data-ai-security-zero-to-hero` site; the red accent is this repo's own.

## House style / conventions
- After adding or editing any `papers/**/summary.md`, run the regeneration pipeline and commit the result:
  ```
  python3 scripts/build_manifest.py     # frontmatter, manifest, INDEX.md, TAGS.md, nav, site-build/
  python3 scripts/add_cross_links.py     # "Related in This Collection" footers
  python3 scripts/build_manifest.py     # refresh after footers
  ```
- CI (`.github/workflows/ci.yml`) fails if these generated outputs are stale or if any relative link is broken, so run them before pushing.
- Do not hand-edit YAML frontmatter, `INDEX.md`, `TAGS.md`, `mkdocs.generated.yml`, `site-build/`, the `<!-- related:* -->` footers, or README.md's `<!-- byyear:* -->` block - they are generated. Edit `mkdocs.yml` for theme/config and `write_mkdocs()` for nav.
- To preview or check the site locally:
  ```
  python3 -m venv .venv-docs && .venv-docs/bin/pip install -r requirements.txt
  python3 scripts/build_manifest.py
  .venv-docs/bin/mkdocs serve -f mkdocs.generated.yml    # or: mkdocs build -f mkdocs.generated.yml --strict
  ```
  `--strict` is what CI runs: it fails on a broken link or a link to an anchor that does not exist, so run it before pushing site changes.
- When adding a new paper, give it the next number (currently up to 107), add its aliases to the `ALIASES` map in `scripts/add_cross_links.py` (so other papers can link to it), and add its topic tags to the `TOPICS` map in `scripts/build_manifest.py` (so it appears in `TAGS.md` and gets `tags:` frontmatter).
