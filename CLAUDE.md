# GenAI Research Papers Summarized

## Overview
"Everything AI, Summarized": a curated collection of foundational AI papers and landmark
essays with comprehensive summaries (count: see `papers.json`), plus unnumbered explainers
on model families, benchmarks, compute, policy, ecosystem and open questions. Broadened from
papers-only on 2026-09-30; the repo name and Pages URL were deliberately kept (a rename
would break the Pages URL with no redirect). Docs-only educational resource (no application
code) - markdown plus a stdlib Python regeneration pipeline that builds the index,
manifests, and an MkDocs Material site.

## Structure
- `papers/` - Paper summaries grouped into category subfolders (`architectures/`, `language-models/`, `image-generation/`, `multimodal/`, `robotics/`, `techniques/`, `essays/`); each summary is a `summary.md`
- `explainers/<section>/<slug>.md` - Unnumbered explainer pages (sections in `EXPLAINER_SECTIONS` in `build_manifest.py`). Each needs `# Title`, `**In one line:**` and `**Last reviewed:** YYYY-MM-DD`; `check_counts.py` enforces them and the build warns after 180 days. Template: `explainers/_TEMPLATE.md`
- `EXPLAINERS.md` - Generated hub listing every explainer
- `papers/_TEMPLATE.md` - Template for new summaries
- `INDEX.md` - Generated category-grouped index of every paper
- `papers.json` / `papers.csv` - Generated machine-readable manifest
- `scripts/build_manifest.py` - Regenerates frontmatter, manifest, INDEX.md, TAGS.md, `mkdocs.generated.yml`, and the `site-build/` tree (stdlib only, idempotent)
- `scripts/add_cross_links.py` - Regenerates the "Related in This Collection" footer on each summary (stdlib only, idempotent)
- `scripts/check_links.py` - Validates relative Markdown links (used by CI)
- `scripts/check_counts.py` - Fails CI when a hand-typed count drifts from `papers.json`: BROWSE.md's per-category and badge tallies, the glossary term count in `docs/GLOSSARY.md` and README, and `docs/GAPS.md`'s coverage map (which must account for every paper)
- `scripts/measure_sources.py` - **The only networked script, and not part of the pipeline.** Fetches each paper's source PDF, counts words with `pdftotext`, and caches the result in `source_lengths.json`. Run by hand when papers are added
- `source_lengths.json` - Committed cache of per-paper source word counts. `build_manifest.py` reads it offline to compute the landing page's compression figures, so the normal build stays stdlib-only and network-free
- `scripts/site_hooks.py` - MkDocs hook (`hooks:` in mkdocs.yml): per-page meta description, a short search `<title>` ("<Short name> Paper Summary (<year>)"), schema.org JSON-LD (BreadcrumbList, WebSite, Article with `isBasedOn` the original paper) and noindex on the templates. Odd short names go in its `SEO_SHORT` map
- `.github/site-overrides/main.html` - Material `custom_dir` override: renders the hook's `<title>`, Open Graph/Twitter tags (image `assets/brand/social-preview-1280x640.png`), JSON-LD and the Search Console tag (`extra.google_site_verification`). After a mkdocs-material bump, confirm the `extrahead` block still exists - missing tags are not a build error
- `scripts/notify_indexnow.py` - Run by the Pages deploy job: sends changed page URLs to IndexNow (Bing, and through it DuckDuckGo/ChatGPT search). Key in `.github/site/indexnow-key.txt`, served at the site root by `build_manifest.py`. `--all` resubmits the whole sitemap
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
- CI (`.github/workflows/ci.yml`) fails if these generated outputs are stale, if any relative link is
  broken, or if a hand-maintained count has drifted, so run these before pushing:
  ```
  python3 scripts/check_links.py
  python3 scripts/check_counts.py
  ```
- Do not hand-edit YAML frontmatter, `INDEX.md`, `TAGS.md`, `mkdocs.generated.yml`, `site-build/`, the `<!-- related:* -->` footers, or README.md's `<!-- byyear:* -->` block - they are generated. Edit `mkdocs.yml` for theme/config and `write_mkdocs()` for nav.
- `BROWSE.md` **is** hand-maintained, and carries one card per paper plus per-category and badge
  tallies. Adding a paper means adding its card and updating those tallies; `check_counts.py` fails
  CI if you forget. `docs/GAPS.md`'s coverage map must likewise name every paper.
- Hand-written guides (`docs/ROADMAP.md`, `docs/READING_GUIDE.md`, `docs/QUICK_REFERENCE.md`,
  `docs/COMPARISONS.md`, `docs/GLOSSARY.md`) curate rather than enumerate - except
  `QUICK_REFERENCE.md`, which carries a row per paper. Do not hand-copy the by-year or by-topic
  groupings into them; link to the generated `README.md` block, `INDEX.md` and `TAGS.md` instead.
- To preview or check the site locally:
  ```
  python3 -m venv .venv-docs && .venv-docs/bin/pip install -r requirements.txt
  python3 scripts/build_manifest.py
  .venv-docs/bin/mkdocs serve -f mkdocs.generated.yml    # or: mkdocs build -f mkdocs.generated.yml --strict
  ```
  `--strict` is what CI runs: it fails on a broken link or a link to an anchor that does not exist, so run it before pushing site changes. Note that on the site `README.md` is replaced by `.github/site/home.md`, so a `README.md#anchor` link from a doc will fail the strict build even though it resolves on GitHub.
- After adding a paper, refresh the source measurement so the landing page's "words in" and
  "Nx shorter" figures stay correct (needs network + `pdftotext`; skips anything already cached):
  ```
  python3 scripts/measure_sources.py && python3 scripts/build_manifest.py
  ```
  Sources with no retrievable PDF (journal paywalls, blog-post papers) are recorded as `null`
  and excluded, so the published total is a floor rather than an estimate. Never hand-edit the
  numbers - the landing page says how they were measured and links to the script.
- When adding a new paper, give it the next number (see the highest in `papers.json`), add its aliases to the `ALIASES` map in `scripts/add_cross_links.py` (so other papers can link to it), and add its topic tags to the `TOPICS` map in `scripts/build_manifest.py` (so it appears in `TAGS.md` and gets `tags:` frontmatter).
