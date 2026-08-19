# TODO

Working task list for **genai-research-papers-summarized**. Read this at the start of a work session and keep it current as work completes - check items off with a date, add follow-ups as they surface. Stale TODOs are worse than none. Security debt (if any) is tracked separately in `SECURITY-DEBT.md`.

---

## Open

### Content
- [ ] 🟠 Work through the high-priority queue in [`docs/GAPS.md`](./docs/GAPS.md) - induction heads,
      adversarial attacks/jailbreaks, weak-to-strong generalization, long-context extension
      (YaRN/position interpolation), sparse attention, data curation at scale, MMLU/HELM/contamination.
- [ ] 🟠 Backfill `BROWSE.md` cards for the papers that are in `INDEX.md` but have no grid card yet.
      BROWSE currently covers 54 of 107 and says so explicitly; the gap is papers 53-68 and 88-107.
- [ ] 🟠 Refresh `docs/QUICK_REFERENCE.md`, `docs/COMPARISONS.md`, `docs/READING_GUIDE.md` and
      `docs/ROADMAP.md` to include papers 69-107. The 88-107 batch (MAE, VQ-VAE/VQ-GAN, Imagen,
      DreamBooth, GPT-1, PaLM, Mistral 7B, Llama Guard, STaR, Quiet-STaR, Self-Refine, Voyager,
      AlphaFold 3, AlphaZero, KTO, Genie, DreamerV3, ESM, CICERO) is in `INDEX.md`, `TAGS.md` and the
      site nav but is not yet woven into the guides.
- [ ] 🟡 The 88-107 summaries were salvaged from a stale branch and carry a hand-written
      "Connections to Other Papers" section with `(#NN)` references. The numbers were remapped to the
      current scheme, but the prose has not been re-read against main's versions of those papers -
      spot-check a few for claims that no longer hold.
- [ ] 🟡 `39-rlvr` has no parseable source URL - the build script warns on every run. Add a link line
      to its header block.

### Tooling
- [ ] 🟡 Add a CI check that the remaining hand-maintained counts in `README.md` and `BROWSE.md`
      match `papers.json`, the way the zero-to-hero repo checks its README counts. The paper count,
      Quick Stats and the By Year block are now generated or verified, but BROWSE's "54 of 107" and
      its total row are still typed by hand.

---

## Done

- [x] ~~Ship the site rebuild and the 88-107 salvage to `main` (`a559e1e`). CI and the Docs site workflow both green; verified live at <https://patrickwiloak.github.io/genai-research-papers-summarized/> - amber palette, new landing page, 107 papers, and the salvaged summaries all serving 200~~ ✅ done 2026-08-19
- [x] ~~Delete `claude/expand-paper-collection-Rif7n` after confirming all 20 salvaged summaries are on `origin/main`; recovery SHA `592be71` if ever needed~~ ✅ done 2026-08-19
- [x] ~~Delete the `claude/content-gaps-ads-ylbmrr` remote branch - fully merged into `main` (zero commits ahead, zero diff); recovery SHA `aa97fa8` if ever needed~~ ✅ done 2026-08-19
- [x] ~~Set the repo homepage to the Pages URL so GitHub links the site from the repo header~~ ✅ done 2026-08-19
- [x] ~~Confirm GitHub Pages is already wired to GitHub Actions and deploying (it is, since June 2026) - no manual setup needed~~ ✅ done 2026-08-19
- [x] ~~Rebuild the docs site to match the zero-to-hero repo: hand-maintained `mkdocs.yml` with the nav generated into a git-ignored `mkdocs.generated.yml`, a site-only landing page and stylesheet under `.github/site/`, near-black palette with an amber accent, Geist fonts, `toc.integrate`, pinned toolchain, and a `--strict` build as a blocking gate on every PR~~ ✅ done 2026-08-18
- [x] ~~Salvage the 20 genuinely-new summaries from the stale `claude/expand-paper-collection-Rif7n` branch as papers 88-107, dropping its 5 duplicate papers (Word2Vec, DDPM, Mixtral, AlphaFold 2, GPT-2) and its stale README/guide rewrites~~ ✅ done 2026-08-18
- [x] ~~Generate README's By Year block from `papers.json` instead of maintaining it by hand while claiming it was generated~~ ✅ done 2026-08-18
- [x] ~~Replace README's exhaustive per-paper directory tree with a category-level tree, removing a guaranteed drift source~~ ✅ done 2026-08-18
- [x] ~~Add Nobler Works + gitGood.dev promo block to the README, matching the zero-to-hero repo~~ ✅ done 2026-08-18
- [x] ~~Gap analysis pass: 19 summaries added (69-87) covering the modern diffusion pipeline, ResNet/U-Net/GQA, training systems, instruction tuning, retrieval, evaluation, interpretability and safety~~ ✅ done 2026-08-18
- [x] ~~Publish `docs/GAPS.md` so the collection's boundaries and queue are explicit~~ ✅ done 2026-08-18
