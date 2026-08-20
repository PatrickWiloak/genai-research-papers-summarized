# TODO

Working task list for **genai-research-papers-summarized**. Read this at the start of a work session and keep it current as work completes - check items off with a date, add follow-ups as they surface. Stale TODOs are worse than none. Security debt (if any) is tracked separately in `SECURITY-DEBT.md`.

---

## Open

### Content
- [ ] 🟠 Work through the high-priority queue in [`docs/GAPS.md`](./docs/GAPS.md) - induction heads,
      adversarial attacks/jailbreaks, weak-to-strong generalization, long-context extension
      (YaRN/position interpolation), sparse attention, data curation at scale, MMLU/HELM/contamination.
- [ ] 🟡 The 88-107 summaries were salvaged from a stale branch and carry a hand-written
      "Connections to Other Papers" section with `(#NN)` references. The numbers were remapped to the
      current scheme, but the prose has not been re-read against main's versions of those papers -
      spot-check a few for claims that no longer hold.
- [ ] 🟡 `docs/READING_GUIDE.md`, `docs/ROADMAP.md` and `docs/COMPARISONS.md` now reference the
      whole collection but still curate rather than enumerate. That is deliberate, but it means a
      new paper does not automatically appear in them - check whether it belongs on a learning path
      or changes a comparison when adding one.
- [ ] 🟡 `docs/GLOSSARY.md` defines 117 terms and is now count-checked, but it was written for the
      original 15-paper collection. Sweep it for terms introduced by papers 53-107 that have no
      entry yet (MoE routing, GRPO, RLVR, flow matching, latent action model, paged attention,
      speculative decoding, process reward model, sparse autoencoder).

### Tooling
- [ ] 🟡 15 of the 107 papers have no retrievable PDF, so they are excluded from the "1.4M words in"
      figure on the landing page (which is therefore a floor). Nature paywalls: `68-alphafold`,
      `61-alphageometry`, `101-alphafold3`. DOI redirects: `106-esm`, `107-cicero`. Published as web
      pages: the Anthropic, OpenAI, Meta, Google and transformer-circuits entries. `39-rlvr` now links
      to DeepSeek-R1 but is deliberately excluded via `SHARED_SOURCE` in `measure_sources.py`, since
      that PDF is already counted under `26-deepseek-r1`. If any gain an open PDF, re-run
      `scripts/measure_sources.py`.

---

## Done

- [x] ~~Documentation sweep: `BROWSE.md` completed to all 107 cards (was 54) and reorganised by
      category with corrected per-category and badge tallies; `docs/QUICK_REFERENCE.md` rebuilt with
      a row per paper (was 24); `docs/READING_GUIDE.md` rewritten for the current collection (was 15
      papers); `docs/COMPARISONS.md` gained reasoning, diffusion-sampling, control, tokenizer,
      serving, retrieval, agent, evaluation and beyond-language sections; `docs/ROADMAP.md` gained a
      Reasoning & Agents path and a production sprint~~ ✅ done 2026-08-20
- [x] ~~Add `scripts/check_counts.py` and wire it into CI - verifies BROWSE's per-category and badge
      tallies, the glossary term count in `GLOSSARY.md` and `README.md`, and that `docs/GAPS.md`'s
      coverage map accounts for every paper. Verified it fails on injected drift~~ ✅ done 2026-08-20
- [x] ~~Fix false counts: README footer claimed 460,000+ words (actual 219,000+ including guides),
      README and `GLOSSARY.md` claimed 250+/150+ glossary terms (actual 117), BROWSE's Quick Stats
      per-category rows were stale on four of five categories, and `docs/GAPS.md` still said 87
      papers and listed Imagen and AlphaZero as gaps after both were added~~ ✅ done 2026-08-20
- [x] ~~Give `39-rlvr` a source link (DeepSeek-R1) so the build no longer warns, and exclude it from
      the source-word measurement via `SHARED_SOURCE` so that PDF is not counted twice~~ ✅ done 2026-08-20

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
