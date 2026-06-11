#!/usr/bin/env python3
"""
build_manifest.py - single source of truth for repo metadata.

Walks papers/<category>/<slug>/summary.md, parses each summary's header,
and (re)generates:

  - YAML frontmatter on every summary.md (idempotent - safe to re-run)
  - papers.json    machine-readable manifest
  - papers.csv     spreadsheet-friendly manifest
  - INDEX.md       human browse index, grouped by category
  - mkdocs.yml     MkDocs Material site config with auto-generated nav

Run from anywhere:  python3 scripts/build_manifest.py
No third-party dependencies (standard library only).
"""

from __future__ import annotations

import csv
import json
import re
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PAPERS_DIR = ROOT / "papers"

# Display order + pretty names for categories.
CATEGORY_ORDER = [
    "architectures",
    "language-models",
    "image-generation",
    "multimodal",
    "techniques",
]
CATEGORY_TITLES = {
    "architectures": "Architectures",
    "language-models": "Language Models",
    "image-generation": "Image & Video Generation",
    "multimodal": "Multimodal",
    "techniques": "Techniques & Methods",
}

# Curated topic tags per slug (a controlled vocabulary, kebab-case). These
# power the `tags:` frontmatter and the generated TAGS.md tag-filtered index.
# A paper with no entry falls back to its category as a single tag.
TOPICS: dict[str, list[str]] = {
    "01-attention-is-all-you-need": ["transformers", "attention", "architecture"],
    "02-generative-adversarial-networks": ["image-generation", "gan"],
    "03-bert": ["language-model", "pretraining"],
    "04-gpt3-few-shot-learners": ["language-model", "scaling", "pretraining"],
    "05-instructgpt-rlhf": ["alignment", "rlhf", "instruction-tuning"],
    "06-diffusion-models": ["image-generation", "diffusion"],
    "07-stable-diffusion": ["image-generation", "diffusion", "efficiency"],
    "08-clip": ["multimodal", "vision"],
    "09-chain-of-thought": ["reasoning", "chain-of-thought"],
    "10-lora": ["efficiency", "fine-tuning"],
    "11-vision-transformer": ["vision", "transformers", "architecture"],
    "12-scaling-laws": ["scaling"],
    "13-rag": ["retrieval"],
    "14-constitutional-ai": ["alignment", "safety"],
    "15-llama": ["language-model", "pretraining"],
    "16-flash-attention": ["efficiency", "attention", "inference-optimization"],
    "17-llama2": ["language-model", "alignment", "rlhf"],
    "18-chinchilla": ["scaling"],
    "19-dpo": ["alignment", "preference-optimization"],
    "20-mamba": ["architecture", "state-space", "efficiency", "long-context"],
    "21-react": ["agents", "tool-use", "reasoning"],
    "22-qlora": ["efficiency", "fine-tuning", "quantization"],
    "23-gpt4v": ["multimodal", "vision"],
    "24-toolformer": ["agents", "tool-use"],
    "25-tree-of-thoughts": ["reasoning"],
    "26-deepseek-r1": ["reasoning", "reinforcement-learning"],
    "27-deepseek-v3": ["language-model", "moe", "efficiency"],
    "28-qwen3": ["language-model", "reasoning"],
    "29-gemini-2.5": ["multimodal", "long-context"],
    "30-claude-3.5-sonnet": ["language-model", "agents"],
    "31-openai-o1": ["reasoning", "test-time-compute"],
    "32-sam2": ["vision"],
    "33-llama3.3": ["language-model", "efficiency"],
    "34-meta-cot": ["reasoning", "chain-of-thought"],
    "35-rstar-math": ["reasoning"],
    "36-gpt4": ["language-model", "multimodal"],
    "37-mixture-of-experts": ["moe", "architecture", "efficiency"],
    "38-grpo": ["reinforcement-learning", "alignment"],
    "39-rlvr": ["reinforcement-learning", "reasoning"],
    "40-gpt4o": ["multimodal", "audio", "vision"],
    "41-llama4": ["language-model", "moe", "multimodal"],
    "42-gpt5": ["language-model", "reasoning"],
    "43-claude4": ["language-model", "agents"],
    "44-sora-dit": ["video-generation", "diffusion"],
    "45-speculative-decoding": ["efficiency", "inference-optimization"],
    "46-llava": ["multimodal", "vision", "instruction-tuning"],
    "47-gemini3": ["multimodal"],
    "48-dalle3": ["image-generation", "diffusion"],
    "49-whisper": ["audio"],
    "50-test-time-compute": ["reasoning", "test-time-compute", "scaling"],
    "51-process-reward-models": ["reasoning", "alignment"],
    "52-pagedattention-vllm": ["efficiency", "inference-optimization"],
    "53-word2vec": ["embeddings"],
    "54-rope-rotary-position-embedding": ["position-encoding", "attention"],
    "55-seq2seq": ["architecture"],
    "56-codex": ["code", "language-model"],
    "57-vae": ["image-generation", "vae"],
    "58-generative-agents": ["agents"],
    "59-model-context-protocol": ["agents", "tool-use"],
    "60-graph-rag": ["retrieval"],
    "61-alphageometry": ["reasoning", "science"],
    "62-alphaevolve": ["agents", "science", "code"],
    "63-ppo": ["reinforcement-learning", "alignment"],
    "64-gpt2": ["language-model", "scaling", "pretraining"],
    "65-t5": ["language-model", "architecture", "pretraining"],
    "66-bahdanau-attention": ["attention", "architecture"],
    "67-switch-transformer": ["moe", "architecture", "scaling"],
    "68-alphafold": ["science", "attention"],
}

FRONTMATTER_RE = re.compile(r"^---\s*\n.*?\n---\s*\n", re.DOTALL)
TITLE_RE = re.compile(r"^#\s+(.+?)\s*$", re.MULTILINE)
AUTHORS_RE = re.compile(
    r"^\*\*(?:Authors?|Organization|Author/Org)[^:]*:\*\*\s*(.+?)\s*$", re.MULTILINE
)
PUBLISHED_RE = re.compile(r"^\*\*Published:\*\*\s*(.+?)\s*$", re.MULTILINE)
YEAR_RE = re.compile(r"\b(19|20)\d{2}\b")
MD_LINK_RE = re.compile(r"\[([^\]]+)\]\((https?://[^)\s]+)\)")
BARE_URL_RE = re.compile(r"https?://[^\s)>\]]+")


def strip_frontmatter(text: str) -> str:
    """Remove a leading YAML frontmatter block if present."""
    return FRONTMATTER_RE.sub("", text, count=1)


def clean_markup(value: str) -> str:
    """Turn '[text](url)' into 'text' and drop stray bold markers."""
    value = MD_LINK_RE.sub(r"\1", value)
    value = value.replace("**", "").strip()
    return value


def header_block(body: str) -> str:
    """The metadata block: everything before the first '---' rule or '## ' heading."""
    end = len(body)
    rule = re.search(r"^---\s*$", body, re.MULTILINE)
    if rule:
        end = min(end, rule.start())
    heading = re.search(r"^##\s", body, re.MULTILINE)
    if heading:
        end = min(end, heading.start())
    return body[:end]


def pick_url(block: str) -> str:
    """Prefer an arXiv link, else the first link in the header block."""
    urls = [u for _, u in MD_LINK_RE.findall(block)]
    urls += BARE_URL_RE.findall(block)
    # de-dupe, keep order
    seen, ordered = set(), []
    for u in urls:
        if u not in seen:
            seen.add(u)
            ordered.append(u)
    for u in ordered:
        if "arxiv.org" in u:
            return u
    return ordered[0] if ordered else ""


def parse_summary(path: Path) -> dict:
    slug = path.parent.name
    category = path.parent.parent.name
    raw = path.read_text(encoding="utf-8")
    body = strip_frontmatter(raw)

    title_m = TITLE_RE.search(body)
    title = title_m.group(1).strip() if title_m else slug

    block = header_block(body)
    authors_m = AUTHORS_RE.search(block)
    authors = clean_markup(authors_m.group(1)) if authors_m else ""

    published_m = PUBLISHED_RE.search(block)
    published = clean_markup(published_m.group(1)) if published_m else ""

    year_m = YEAR_RE.search(published) or YEAR_RE.search(block)
    year = int(year_m.group(0)) if year_m else None

    num_m = re.match(r"(\d+)", slug)
    number = int(num_m.group(1)) if num_m else None

    return {
        "number": number,
        "slug": slug,
        "category": category,
        "title": title,
        "authors": authors,
        "published": published,
        "year": year,
        "url": pick_url(block),
        "topics": TOPICS.get(slug, [category]),
        "path": str(path.relative_to(ROOT)).replace("\\", "/"),
        "_file": path,
        "_body": body,
    }


def build_frontmatter(p: dict) -> str:
    def s(v):  # JSON string is valid YAML and safely quotes colons/quotes
        return json.dumps(v, ensure_ascii=False)

    lines = ["---"]
    lines.append(f"title: {s(p['title'])}")
    lines.append(f"slug: {s(p['slug'])}")
    if p["number"] is not None:
        lines.append(f"number: {p['number']}")
    lines.append(f"category: {s(p['category'])}")
    if p["authors"]:
        lines.append(f"authors: {s(p['authors'])}")
    if p["published"]:
        lines.append(f"published: {s(p['published'])}")
    if p["year"] is not None:
        lines.append(f"year: {p['year']}")
    if p["url"]:
        lines.append(f"url: {s(p['url'])}")
    tags = ", ".join(s(t) for t in p["topics"])
    lines.append(f"tags: [{tags}]")
    lines.append("---")
    return "\n".join(lines) + "\n\n"


def write_frontmatter(papers: list[dict]) -> int:
    changed = 0
    for p in papers:
        body = p["_body"].lstrip("\n")
        new_text = build_frontmatter(p) + body
        if new_text != p["_file"].read_text(encoding="utf-8"):
            p["_file"].write_text(new_text, encoding="utf-8")
            changed += 1
    return changed


def public_record(p: dict) -> dict:
    rec = {k: p[k] for k in
           ("number", "title", "slug", "category", "authors", "published", "year", "url", "path")}
    rec["topics"] = p["topics"]
    return rec


def write_json(papers: list[dict]) -> None:
    data = {
        "count": len(papers),
        "categories": CATEGORY_ORDER,
        "papers": [public_record(p) for p in papers],
    }
    (ROOT / "papers.json").write_text(
        json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )


def write_csv(papers: list[dict]) -> None:
    cols = ["number", "category", "title", "authors", "year", "published", "topics", "url", "path"]
    with (ROOT / "papers.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for p in papers:
            row = public_record(p)
            row["topics"] = ";".join(row["topics"])
            w.writerow(row)


def write_index(papers: list[dict]) -> None:
    lines = [
        "# Paper Index",
        "",
        f"All **{len(papers)}** summaries at a glance, grouped by category. "
        "Generated by `scripts/build_manifest.py` - do not edit by hand.",
        "",
    ]
    for cat in CATEGORY_ORDER:
        group = [p for p in papers if p["category"] == cat]
        if not group:
            continue
        lines.append(f"## {CATEGORY_TITLES.get(cat, cat)}")
        lines.append("")
        lines.append("| # | Paper | Year | Source |")
        lines.append("|---|-------|------|--------|")
        for p in group:
            num = f"{p['number']:02d}" if p["number"] is not None else ""
            link = f"[{p['title']}]({p['path']})"
            year = p["year"] or ""
            src = f"[link]({p['url']})" if p["url"] else ""
            lines.append(f"| {num} | {link} | {year} | {src} |")
        lines.append("")
    (ROOT / "INDEX.md").write_text("\n".join(lines), encoding="utf-8")


def write_tags(papers: list[dict]) -> None:
    """Generate TAGS.md: a tag-filtered index grouping papers by topic."""
    tag_to_papers: dict[str, list[dict]] = {}
    for p in papers:
        for t in p["topics"]:
            tag_to_papers.setdefault(t, []).append(p)

    total_tags = len(tag_to_papers)
    lines = [
        "# Browse by Topic",
        "",
        f"All **{len(papers)}** papers grouped by **{total_tags}** topic tags. "
        "A paper appears under each of its tags. Generated by "
        "`scripts/build_manifest.py` - do not edit by hand.",
        "",
        "**Jump to:** " + " · ".join(
            f"[{t}](#{t})" for t in sorted(tag_to_papers)),
        "",
    ]
    for tag in sorted(tag_to_papers):
        group = sorted(tag_to_papers[tag], key=lambda q: q["number"] or 0)
        lines.append(f"## {tag}")
        lines.append("")
        for p in group:
            num = f"{p['number']:02d} " if p["number"] is not None else ""
            year = f" ({p['year']})" if p["year"] else ""
            lines.append(f"- {num}[{p['title']}]({p['path']}){year}")
        lines.append("")
    (ROOT / "TAGS.md").write_text("\n".join(lines), encoding="utf-8")


MKDOCS_HEADER = """\
# Generated by scripts/build_manifest.py - do not edit by hand.
# The site is built from the curated `site-build/` directory, which this
# script assembles by copying the README, guides, index, and every paper
# summary out of the repo root (MkDocs requires a single docs_dir tree).
# `site-build/` is git-ignored; run `python3 scripts/build_manifest.py`
# to (re)create it, then `mkdocs serve`.
site_name: Foundational GenAI Papers, Summarized
site_description: Plain-language summaries of the papers that shaped modern generative AI.
docs_dir: site-build
site_dir: site
use_directory_urls: true

theme:
  name: material
  features:
    - navigation.instant
    - navigation.tracking
    - navigation.top
    - navigation.indexes
    - search.suggest
    - search.highlight
    - content.code.copy
    - toc.follow
  palette:
    - media: "(prefers-color-scheme: light)"
      scheme: default
      primary: indigo
      accent: indigo
      toggle:
        icon: material/weather-night
        name: Switch to dark mode
    - media: "(prefers-color-scheme: dark)"
      scheme: slate
      primary: indigo
      accent: indigo
      toggle:
        icon: material/weather-sunny
        name: Switch to light mode

markdown_extensions:
  - admonition
  - attr_list
  - md_in_html
  - tables
  - toc:
      permalink: true
  - pymdownx.highlight
  - pymdownx.superfences
  - pymdownx.emoji

plugins:
  - search
"""


def build_site_tree(papers: list[dict]) -> None:
    """Assemble the curated docs_dir the site is built from.

    MkDocs needs every page under one docs_dir, but our content lives at the
    repo root (README, BROWSE, INDEX, CONTRIBUTING), in docs/, and in
    papers/. Copy exactly those into site-build/ mirroring the same layout so
    the generated nav paths resolve unchanged. site-build/ is git-ignored.
    """
    site = ROOT / "site-build"
    shutil.rmtree(site, ignore_errors=True)
    site.mkdir()

    # Markdown pages plus the data/license files the README and pages link to.
    for name in ("README.md", "BROWSE.md", "INDEX.md", "TAGS.md", "CONTRIBUTING.md",
                 "papers.json", "papers.csv", "LICENSE"):
        src = ROOT / name
        if src.exists():
            shutil.copy2(src, site / name)

    docs_src = ROOT / "docs"
    if docs_src.exists():
        shutil.copytree(docs_src, site / "docs")

    template = PAPERS_DIR / "_TEMPLATE.md"
    if template.exists():
        (site / "papers").mkdir(parents=True, exist_ok=True)
        shutil.copy2(template, site / "papers" / "_TEMPLATE.md")

    for p in papers:
        dest = site / p["path"]
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / p["path"], dest)


def write_mkdocs(papers: list[dict]) -> None:
    nav = []
    nav.append("nav:")
    nav.append("  - Home: README.md")
    nav.append("  - Browse: BROWSE.md")
    nav.append("  - Index: INDEX.md")
    nav.append("  - By Topic: TAGS.md")
    nav.append("  - Guides:")
    nav.append("      - Learning Roadmap: docs/ROADMAP.md")
    nav.append("      - Reading Guide: docs/READING_GUIDE.md")
    nav.append("      - Quick Reference: docs/QUICK_REFERENCE.md")
    nav.append("      - Comparisons: docs/COMPARISONS.md")
    nav.append("      - Glossary: docs/GLOSSARY.md")
    nav.append("  - Papers:")
    for cat in CATEGORY_ORDER:
        group = [p for p in papers if p["category"] == cat]
        if not group:
            continue
        nav.append(f"      - {CATEGORY_TITLES.get(cat, cat)}:")
        for p in group:
            label = p["title"].replace('"', "'")
            nav.append(f'          - "{label}": {p["path"]}')
    nav.append("  - Contributing: CONTRIBUTING.md")
    (ROOT / "mkdocs.yml").write_text(
        MKDOCS_HEADER + "\n" + "\n".join(nav) + "\n", encoding="utf-8"
    )


def main() -> None:
    summaries = sorted(PAPERS_DIR.glob("*/*/summary.md"))
    papers = [parse_summary(p) for p in summaries]
    papers.sort(key=lambda p: (p["number"] is None, p["number"] or 0))

    changed = write_frontmatter(papers)
    write_json(papers)
    write_csv(papers)
    write_index(papers)
    write_tags(papers)
    write_mkdocs(papers)
    build_site_tree(papers)

    print(f"Parsed {len(papers)} summaries.")
    print(f"Frontmatter written/updated on {changed} file(s).")
    missing_url = [p["slug"] for p in papers if not p["url"]]
    missing_year = [p["slug"] for p in papers if p["year"] is None]
    if missing_url:
        print(f"WARN no source URL parsed: {', '.join(missing_url)}")
    if missing_year:
        print(f"WARN no year parsed: {', '.join(missing_year)}")
    print("Wrote papers.json, papers.csv, INDEX.md, TAGS.md, mkdocs.yml, site-build/")


if __name__ == "__main__":
    main()
