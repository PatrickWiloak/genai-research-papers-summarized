# TODO

Working task list for **genai-research-papers-summarized**. Read this at the start of a work session and keep it current as work completes - check items off with a date, add follow-ups as they surface. Stale TODOs are worse than none. Security debt (if any) is tracked separately in `SECURITY-DEBT.md`.

---

## Open

### 🟠 Discoverability (added 2026-08-31)

- [ ] **Patrick: upload the social preview image.** The August render lived in a `/tmp` scratchpad
      and is gone; a new "Everything AI, summarized" card (144 papers, 31 explainers) is committed at
      [`assets/brand/social-preview-1280x640.png`](./assets/brand/social-preview-1280x640.png)
      (source: `assets/brand/social-preview.html`, re-render with headless Chromium
      `--window-size=1280,640 --screenshot`). Upload via **Settings → General → Social preview**
      (not exposed by the GitHub API). Re-render when the counts move a lot.
- [x] ~~Update the GitHub repo description and topics for the broader scope~~ ✅ done 2026-09-30:
      description now "Everything AI, summarized: 144 papers and landmark essays plus 31 dated
      explainers..."; topics `ai-policy`, `robotics`, `explainers` added. **Patrick:** the old
      description was "For folks who don't have time to read 220,000+ words of research 📚" - restore
      your voice with `gh repo edit --description` if you prefer it.
- [x] ~~Set GitHub topics~~ ✅ done 2026-08-31 (15 topics: llm, generative-ai, research-papers, rag, ...)
- [x] ~~Commit and push the rewritten `LICENSE` + new `NOTICE`~~ ✅ done - on `main` since
      2026-08-31 (commit "Replace the hand-written CC BY summary with the canonical legal text").

### Content
- [ ] 🟡 Work through the (new) high-priority queue in [`docs/GAPS.md`](./docs/GAPS.md): SWE-agent,
      AlphaCode, Toy Models of Superposition, Alignment Faking, Constitutional Classifiers, Ring
      Attention, Medusa/EAGLE, HELM/BIG-Bench. Code generation is the one area still marked Thin.
- [ ] 🟡 Explainer queue in `docs/GAPS.md`: xAI Grok / Phi / Kimi families, MoE routing in practice,
      multimodal tokenization, AI for code, jailbreaks and prompt injection, AI and jobs, copyright.
- [ ] **2027-03-29: re-review every explainer** - that is when all 31 cross the 180-day line and
      `build_manifest.py` starts printing `WARN explainers not reviewed in 180 days`. Model families,
      benchmarks, compute prices and policy go stale first; bump each page's `Last reviewed` as you go.
- [ ] **2026-12-31: re-check the three fastest-moving explainers** - `explainers/model-families/gpt.md`,
      `claude.md`, `gemini.md` - and the dated scores in `explainers/benchmarks/*.md` and
      `papers/techniques/139-osworld/summary.md` ("What Happened Next"). These were current on
      2026-09-30 and will be wrong within a quarter.
- [x] ~~Spot-check the salvaged 88-107 summaries~~ ✅ done 2026-09-30 - a read-only verifier checked
      all 20 against their papers; about 60 corrections applied (one wrong cross-reference, several
      "connections" that misdescribed the cited paper, and numeric tables in Mistral 7B, Llama Guard,
      Self-Refine, KTO, Voyager, PaLM, MAE and STaR rebuilt from the papers). Unverified and left as
      is: CICERO's headline numbers (Science paywall), ESM-2's contact-precision table.
- [ ] 🟡 `docs/READING_GUIDE.md`, `docs/ROADMAP.md` and `docs/COMPARISONS.md` now reference the
      whole collection but still curate rather than enumerate. That is deliberate, but it means a
      new paper does not automatically appear in them - check whether it belongs on a learning path
      or changes a comparison when adding one.


### Tooling
- [ ] 🟡 24 of the 144 papers have no retrievable PDF, so they are excluded from the "1.8M words in"
      figure on the landing page (which is therefore a floor). New since 2026-09-30: the nine essays
      (web pages), 126-induction-heads (transformer-circuits), 140-model-collapse (Nature). Nature paywalls: `68-alphafold`,
      `61-alphageometry`, `101-alphafold3`. DOI redirects: `106-esm`, `107-cicero`. Published as web
      pages: the Anthropic, OpenAI, Meta, Google and transformer-circuits entries. `39-rlvr` now links
      to DeepSeek-R1 but is deliberately excluded via `SHARED_SOURCE` in `measure_sources.py`, since
      that PDF is already counted under `26-deepseek-r1`. If any gain an open PDF, re-run
      `scripts/measure_sources.py`.

---

## Done

- [x] ~~**Broaden to "Everything AI"** (2026-09-30): site renamed "Everything AI, Summarized" (repo name
      and Pages URL deliberately unchanged - a rename breaks the Pages URL with no redirect). New
      `explainers/` content type (31 pages, 8 sections, generated `EXPLAINERS.md` hub, `Last reviewed`
      dates enforced by `check_counts.py`, 180-day staleness warning in the build). New paper
      categories `robotics` and `essays`. Papers 108-144 added (9 essays, 3 robotics, 3D, audio
      generation, safety/interpretability, long context, data, optimisers, evaluation). GAPS queue
      closed and rewritten; glossary 117 -> 144 terms; ROADMAP Path 6 (no-maths Big Picture);
      COMPARISONS and READING_GUIDE updated. All explainers and essays fact-checked against primary
      sources by a separate verification pass before publishing~~ ✅ done 2026-09-30
- [x] ~~Fix gitGood promo copy: trial is 7 days (was "10 days free") and price $8/$64 (was $5/$40),
      here and in the zero-to-hero sibling~~ ✅ done 2026-09-30
- [x] ~~Fix four factual errors found while fact-checking: 26-deepseek-r1 distilled-model table
      (wrong scores and base models throughout), 54-rope (Llama 3 does not use YaRN), 43-claude4
      (May not June 2025), 47-gemini3 (November not December 2025)~~ ✅ done 2026-09-30
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
