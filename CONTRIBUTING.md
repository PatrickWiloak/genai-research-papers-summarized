# Contributing

Thanks for helping make these summaries better. This repo is documentation
only - no code to build, just clear writing about influential papers.

## Adding a new paper summary

1. **Pick a category** under `papers/`: `architectures`, `language-models`,
   `image-generation`, `multimodal`, or `techniques`.
2. **Create a folder** named `NN-slug` where `NN` is the next free two-digit
   number (numbers are stable IDs, not a strict chronology - just don't reuse
   one) and `slug` is a short kebab-case name. Example:
   `papers/techniques/64-my-paper/`.
3. **Copy the template:** start from [`papers/_TEMPLATE.md`](./papers/_TEMPLATE.md)
   and save it as `summary.md` inside your new folder.
4. **Keep the header intact.** The first lines must be the `# Title` and a
   metadata block with `**Authors:**`, `**Published:**`, and a link line
   (`**Paper Link:**`, `**Paper:**`, `**System Card:**`, etc.). The build
   script parses these.
5. **Do not hand-write YAML frontmatter** - it is generated (see below).
6. **Run the build script** and commit what it changes:

   ```bash
   python3 scripts/build_manifest.py
   ```

## House style

- Write for a motivated beginner. Explain jargon the first time it appears.
- Lead with *why the paper matters* before the mechanics.
- Use concrete analogies and small diagrams or formulas in fenced code blocks.
- Be accurate with numbers, dates, and author lists. Cite the real paper.
- Cross-link sibling summaries in this repo where one paper builds on another.
- No em dashes. Use regular hyphens.

## What the build script does

`scripts/build_manifest.py` is the single source of truth for metadata. It is
idempotent - safe to run any time. On each run it:

- (re)writes YAML frontmatter (including topic `tags:`) on every `summary.md`,
- regenerates `papers.json` and `papers.csv` (machine-readable manifests),
- regenerates `INDEX.md` (category browse index) and `TAGS.md` (topic browse index),
- writes `mkdocs.generated.yml` - the hand-maintained `mkdocs.yml` plus the
  generated site navigation, and
- assembles the git-ignored `site-build/` tree the site is built from,
  including the site-only landing page and stylesheet from `.github/site/`.

`mkdocs.yml` itself is hand-maintained and deliberately carries no `nav` key.
Edit it for theme, palette, or markdown extensions; edit `write_mkdocs()` in
`build_manifest.py` for the navigation. Both `mkdocs.generated.yml` and
`site-build/` are git-ignored - never commit or hand-edit them.

A companion script, `scripts/add_cross_links.py`, regenerates the "Related in
This Collection" footers. When you add a paper, also add an entry to the
`TOPICS` map in `build_manifest.py` and the `ALIASES` map in
`add_cross_links.py`. Both scripts use only the Python standard library.

## Previewing the site locally

```bash
python3 -m venv .venv-docs
.venv-docs/bin/pip install -r requirements.txt
python3 scripts/build_manifest.py
.venv-docs/bin/mkdocs serve -f mkdocs.generated.yml
```

Then open http://127.0.0.1:8000.

Before pushing anything that touches the site, run the build the way CI does:

```bash
python3 scripts/build_manifest.py
.venv-docs/bin/mkdocs build -f mkdocs.generated.yml --strict
```

`--strict` turns every MkDocs warning into an error, so a broken relative link
or a link to an anchor that does not exist fails the build rather than shipping
a broken page. The same build runs as a blocking check on every pull request;
pushing to `main` also deploys it to GitHub Pages via
`.github/workflows/pages.yml`.
