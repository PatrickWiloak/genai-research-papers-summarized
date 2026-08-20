#!/usr/bin/env python3
"""
check_counts.py - fail if a hand-maintained count in the docs has drifted from
`papers.json`.

Most of the repo's listings are generated, but a few documents state counts in
prose or in a table that a human typed. Those are the ones that silently go
stale when papers are added. This script re-derives every such number from the
manifest and reports any that no longer match.

Checked:
  - README.md          - the headline paper count and the glossary-term count
  - BROWSE.md          - the intro count, each category heading's "N papers.",
                         the Quick Stats table, and the badge tallies
  - docs/GAPS.md       - the headline paper count, and that its coverage map
                         accounts for every paper in the manifest
  - docs/GLOSSARY.md   - the term count, against the actual number of entries

Run it directly, or let CI run it:

    python3 scripts/check_counts.py
"""

from __future__ import annotations

import collections
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

CATEGORY_LABELS = {
    "architectures": "Foundational Architectures",
    "language-models": "Language Models",
    "image-generation": "Image & Video Generation",
    "multimodal": "Multimodal",
    "techniques": "Techniques & Methods",
}

BADGE_WORDS = ("CRITICAL", "HIGH", "HISTORICAL", "THEORY")


def fail(errors: list[str], msg: str) -> None:
    errors.append(msg)


def check_browse(papers: list[dict], errors: list[str]) -> None:
    path = ROOT / "BROWSE.md"
    text = path.read_text(encoding="utf-8")
    total = len(papers)
    cat_counts = collections.Counter(p["category"] for p in papers)

    if f"every one of the {total} summaries" not in text:
        fail(errors, f"BROWSE.md intro does not state the current total ({total})")

    # Every paper needs a card.
    carded = set(re.findall(r"papers/[a-z-]+/([0-9]+-[a-z0-9.-]+)/summary\.md", text))
    missing = sorted({p["slug"] for p in papers} - carded)
    if missing:
        fail(errors, f"BROWSE.md has no card for: {', '.join(missing)}")
    extra = sorted(carded - {p["slug"] for p in papers})
    if extra:
        fail(errors, f"BROWSE.md links slugs that are not in papers.json: {', '.join(extra)}")

    # Per-section "N papers." claims.
    for cat, label in CATEGORY_LABELS.items():
        expected = cat_counts[cat]
        section = re.search(
            rf"^## .*{re.escape(label)}\n\n(.*?)$", text, re.MULTILINE)
        if not section:
            fail(errors, f"BROWSE.md has no section heading for {label}")
            continue
        claim = re.search(r"\*\*(\d+) papers\.\*\*", section.group(1))
        if not claim:
            fail(errors, f"BROWSE.md {label} section states no paper count")
        elif int(claim.group(1)) != expected:
            fail(errors, f"BROWSE.md {label} says {claim.group(1)} papers, manifest has {expected}")

    # Quick Stats table.
    for cat, label in CATEGORY_LABELS.items():
        row = re.search(rf"\|\s*\*\*{re.escape(label)}\*\*\s*\|\s*(\d+)\s*\|", text)
        if not row:
            fail(errors, f"BROWSE.md Quick Stats has no row for {label}")
        elif int(row.group(1)) != cat_counts[cat]:
            fail(errors, f"BROWSE.md Quick Stats {label} says {row.group(1)}, manifest has {cat_counts[cat]}")
    total_row = re.search(r"\|\s*\*\*Total\*\*\s*\|\s*\*\*(\d+)\*\*\s*\|", text)
    if not total_row:
        fail(errors, "BROWSE.md Quick Stats has no Total row")
    elif int(total_row.group(1)) != total:
        fail(errors, f"BROWSE.md Quick Stats Total says {total_row.group(1)}, manifest has {total}")

    # Badge tallies must add up to the number of cards.
    claimed = {}
    for word in BADGE_WORDS:
        m = re.search(rf"\*\*{word}\*\*\s*\((\d+) paper", text)
        if m:
            claimed[word] = int(m.group(1))
    actual = collections.Counter(
        m.group(1) for m in re.finditer(r"^- \S+ \*\*(%s)\*\*" % "|".join(BADGE_WORDS),
                                        text, re.MULTILINE))
    # The tally lines themselves are not cards, so discount them.
    for word, n in claimed.items():
        actual[word] -= 1
    for word, n in claimed.items():
        if actual[word] != n:
            fail(errors, f"BROWSE.md badge tally says {n} {word}, page has {actual[word]} cards")
    if sum(claimed.values()) != total:
        fail(errors, f"BROWSE.md badge tallies sum to {sum(claimed.values())}, expected {total}")


def check_glossary(errors: list[str]) -> None:
    """The glossary states how many terms it defines; count them and compare."""
    path = ROOT / "docs" / "GLOSSARY.md"
    if not path.exists():
        return
    text = path.read_text(encoding="utf-8")
    actual = len(re.findall(r"^### ", text, re.MULTILINE))
    for label, pattern in (
        ("intro", r"the (\d+) technical terms"),
        ("footer", r"\*\*Terms covered:\*\* (\d+)"),
    ):
        m = re.search(pattern, text)
        if not m:
            fail(errors, f"docs/GLOSSARY.md {label} no longer states a term count")
        elif int(m.group(1)) != actual:
            fail(errors, f"docs/GLOSSARY.md {label} says {m.group(1)} terms, file defines {actual}")

    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    m = re.search(r"\|\s*\*\*Terms Explained\*\*\s*\|\s*(\d+)\s*\|", readme)
    if not m:
        fail(errors, "README.md Quick Stats has no Terms Explained row")
    elif int(m.group(1)) != actual:
        fail(errors, f"README.md says {m.group(1)} glossary terms, GLOSSARY.md defines {actual}")


def check_gaps_coverage(papers: list[dict], errors: list[str]) -> None:
    """GAPS.md claims to map the whole collection; check that it really does."""
    path = ROOT / "docs" / "GAPS.md"
    if not path.exists():
        return
    text = path.read_text(encoding="utf-8")
    table = re.search(r"## Coverage map\n\n(.*?)\n\n---", text, re.S)
    if not table:
        fail(errors, "docs/GAPS.md has no coverage map table")
        return

    listed: set[int] = set()
    for row in table.group(1).splitlines():
        cells = row.split("|")
        if len(cells) < 3:
            continue
        for part in cells[2].split(","):
            part = part.strip()
            if re.fullmatch(r"\d+", part):
                listed.add(int(part))
            elif re.fullmatch(r"\d+-\d+", part):
                lo, hi = (int(x) for x in part.split("-"))
                listed.update(range(lo, hi + 1))

    numbers = {p["number"] for p in papers}
    missing = sorted(numbers - listed)
    if missing:
        fail(errors, "docs/GAPS.md coverage map does not account for paper(s): "
                     + ", ".join(str(n) for n in missing))
    unknown = sorted(listed - numbers)
    if unknown:
        fail(errors, "docs/GAPS.md coverage map lists non-existent paper(s): "
                     + ", ".join(str(n) for n in unknown))

    # A paper listed as queued must not already be in the collection.
    review = re.search(r"\*\*Papers at review time:\*\*\s*(\d+)", text)
    if not review:
        fail(errors, "docs/GAPS.md does not state the paper count at review time")
    elif int(review.group(1)) != len(papers):
        fail(errors, f"docs/GAPS.md says {review.group(1)} papers at review time, "
                     f"manifest has {len(papers)}")


def check_simple_total(rel: str, papers: list[dict], errors: list[str]) -> None:
    """Every '<n> papers'-style number in the doc must be the real total."""
    path = ROOT / rel
    if not path.exists():
        return
    total = len(papers)
    text = path.read_text(encoding="utf-8")
    for m in re.finditer(r"\b(\d{2,4})\s+(?:foundational\s+)?(?:papers|summaries)\b", text):
        n = int(m.group(1))
        # Only three-digit counts in this range are plausibly the collection size.
        if 90 <= n <= 999 and n != total:
            line = text[:m.start()].count("\n") + 1
            fail(errors, f"{rel}:{line} says '{m.group(0)}', manifest has {total}")


def main() -> int:
    manifest = json.loads((ROOT / "papers.json").read_text(encoding="utf-8"))
    papers = manifest["papers"] if isinstance(manifest, dict) else manifest

    errors: list[str] = []
    check_browse(papers, errors)
    check_glossary(errors)
    check_gaps_coverage(papers, errors)
    check_simple_total("README.md", papers, errors)
    check_simple_total("docs/GAPS.md", papers, errors)
    check_simple_total("CLAUDE.md", papers, errors)

    if errors:
        print("Count drift detected:\n", file=sys.stderr)
        for e in errors:
            print(f"  - {e}", file=sys.stderr)
        print(
            "\nRe-derive the numbers from papers.json and update the documents above.",
            file=sys.stderr)
        return 1

    print(f"All hand-maintained counts match papers.json ({len(papers)} papers).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
