#!/usr/bin/env python3
"""
measure_sources.py - measure how many words of source material the summaries
stand in for.

This is the one script in the repo that touches the network, and it is NOT part
of the regeneration pipeline. It fetches each paper's source PDF, counts the
words in the extracted text, and writes the result to `source_lengths.json`.
`build_manifest.py` then reads that cache offline, so the normal build stays
stdlib-only and network-free.

Run it only when papers are added, or to refresh a stale measurement:

    python3 scripts/measure_sources.py            # fetch anything not cached
    python3 scripts/measure_sources.py --refresh  # re-fetch everything

Method, so the published figure can be checked:

  - Only sources that resolve to a real PDF are counted. Paywalled journal
    pages, HTML blog posts and system-card landing pages are skipped rather
    than guessed at, so the total is a floor, not an estimate.
  - The count is every word `pdftotext` extracts, references and appendices
    included. That is deliberately the whole document: it is what sits between
    a reader and the end of the paper. No attempt is made to strip reference
    lists, because doing it reliably across 100+ differently-typeset PDFs is
    not possible and a half-applied rule would be worse than a stated one.
  - Papers whose fetch fails are recorded with `"words": null` so the next run
    retries them and the coverage count stays honest.

Requires `pdftotext` (poppler-utils) on PATH.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CACHE = ROOT / "source_lengths.json"

UA = "genai-research-papers-summarized/1.0 (+https://github.com/PatrickWiloak/genai-research-papers-summarized)"
DELAY_SECONDS = 3.0  # be a polite citizen of arxiv.org


def pdf_url(url: str) -> str | None:
    """Map a paper's source URL to a direct PDF, or None if there isn't one."""
    if not url:
        return None
    m = re.search(r"arxiv\.org/abs/([^\s?#]+)", url)
    if m:
        return f"https://arxiv.org/pdf/{m.group(1).rstrip('.')}"
    if url.lower().endswith(".pdf"):
        return url
    return None


def fetch_words(url: str) -> int | None:
    """Download a PDF and return its extracted word count, or None on failure."""
    try:
        req = urllib.request.Request(url, headers={"User-Agent": UA})
        with urllib.request.urlopen(req, timeout=90) as resp:
            data = resp.read()
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        print(f"    fetch failed: {exc}")
        return None

    if not data.startswith(b"%PDF"):
        print("    not a PDF (probably an HTML landing or paywall page)")
        return None

    with tempfile.NamedTemporaryFile(suffix=".pdf") as fh:
        fh.write(data)
        fh.flush()
        try:
            out = subprocess.run(
                ["pdftotext", fh.name, "-"],
                capture_output=True, timeout=120, check=True,
            )
        except (subprocess.SubprocessError, FileNotFoundError) as exc:
            print(f"    pdftotext failed: {exc}")
            return None

    return len(out.stdout.split())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--refresh", action="store_true",
                    help="re-fetch every paper, not just uncached/failed ones")
    args = ap.parse_args()

    if not shutil.which("pdftotext"):
        print("error: pdftotext not found. Install poppler-utils.", file=sys.stderr)
        return 1

    manifest = json.loads((ROOT / "papers.json").read_text(encoding="utf-8"))
    papers = manifest["papers"] if isinstance(manifest, dict) else manifest

    cache: dict[str, dict] = {}
    if CACHE.exists():
        cache = json.loads(CACHE.read_text(encoding="utf-8")).get("sources", {})

    todo = []
    for p in papers:
        slug = p["slug"]
        cached = cache.get(slug)
        if not args.refresh and cached and cached.get("words") is not None:
            continue
        todo.append(p)

    print(f"{len(papers)} papers, {len(todo)} to measure")

    for i, p in enumerate(todo, 1):
        slug = p["slug"]
        target = pdf_url(p.get("url") or "")
        print(f"[{i}/{len(todo)}] {slug}")
        if not target:
            print("    no direct PDF for this source - skipping")
            cache[slug] = {"url": p.get("url") or "", "pdf": None, "words": None}
            continue
        words = fetch_words(target)
        if words:
            print(f"    {words:,} words")
        cache[slug] = {"url": p.get("url") or "", "pdf": target, "words": words}
        time.sleep(DELAY_SECONDS)

    measured = {k: v for k, v in cache.items() if v.get("words")}
    total = sum(v["words"] for v in measured.values())
    payload = {
        "_comment": (
            "Generated by scripts/measure_sources.py - do not edit by hand. "
            "Word counts are every word pdftotext extracts from each paper's "
            "PDF, references and appendices included. Sources with no "
            "retrievable PDF are recorded with null and excluded from the "
            "total, so measured_words is a floor."
        ),
        "measured_papers": len(measured),
        "total_papers": len(papers),
        "measured_words": total,
        "sources": dict(sorted(cache.items(), key=lambda kv: kv[0])),
    }
    CACHE.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"\nmeasured {len(measured)}/{len(papers)} papers, {total:,} source words")
    print(f"wrote {CACHE.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
