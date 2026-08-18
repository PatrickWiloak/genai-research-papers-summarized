# TODO

Working task list for **genai-research-papers-summarized**. Read this at the start of a work session and keep it current as work completes - check items off with a date, add follow-ups as they surface. Stale TODOs are worse than none. Security debt (if any) is tracked separately in `SECURITY-DEBT.md`.

---

## Open

### Content
- [ ] Work through the high-priority queue in [`docs/GAPS.md`](./docs/GAPS.md) - induction heads,
      adversarial attacks/jailbreaks, weak-to-strong generalization, long-context extension
      (YaRN/position interpolation), sparse attention, data curation at scale, MMLU/HELM/contamination.
- [ ] Backfill `BROWSE.md` cards for papers 53-68, which are in `INDEX.md` but have no grid card yet
      (BROWSE currently covers 54 of 87 and says so explicitly).
- [ ] Refresh `docs/QUICK_REFERENCE.md`, `docs/COMPARISONS.md`, `docs/READING_GUIDE.md` and
      `docs/ROADMAP.md` to include papers 69-87.
- [ ] `39-rlvr` has no parseable source URL - the build script warns on every run. Add a link line
      to its header block.

### Tooling
- [ ] Consider a CI check that the hand-maintained counts in `README.md` and `BROWSE.md` match
      `papers.json`, the way the zero-to-hero repo checks its README counts. These drifted badly
      before the August 2026 pass (BROWSE claimed 35 papers when there were 68).

---

## Done

- [x] ~~Add Nobler Works + gitGood.dev promo block to the README, matching the zero-to-hero repo~~ ✅ done 2026-08-18
- [x] ~~Gap analysis pass: 19 summaries added (69-87) covering the modern diffusion pipeline, ResNet/U-Net/GQA, training systems, instruction tuning, retrieval, evaluation, interpretability and safety~~ ✅ done 2026-08-18
- [x] ~~Publish `docs/GAPS.md` so the collection's boundaries and queue are explicit~~ ✅ done 2026-08-18
