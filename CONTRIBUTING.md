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

- (re)writes YAML frontmatter on every `summary.md` from the header block,
- regenerates `papers.json` and `papers.csv` (machine-readable manifests),
- regenerates `INDEX.md` (the browse index), and
- regenerates `mkdocs.yml` (the site navigation).

It uses only the Python standard library, so no install is needed.

## Previewing the site locally

```bash
pip install -r requirements.txt
mkdocs serve
```

Then open http://127.0.0.1:8000. Pushing to `main` deploys to GitHub Pages
automatically via `.github/workflows/pages.yml`.
