#!/usr/bin/env python3
"""
build_manifest.py - single source of truth for repo metadata.

Walks papers/<category>/<slug>/summary.md, parses each summary's header,
and (re)generates:

  - YAML frontmatter on every summary.md (idempotent - safe to re-run)
  - papers.json    machine-readable manifest
  - papers.csv     spreadsheet-friendly manifest
  - INDEX.md       human browse index, grouped by category
  - mkdocs.generated.yml  hand-maintained mkdocs.yml + auto-generated nav
  - site-build/    the staged docs_dir the site is built from

Run from anywhere:  python3 scripts/build_manifest.py
No third-party dependencies (standard library only).
"""

from __future__ import annotations

import csv
import datetime as dt
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
    "robotics",
    "techniques",
    "essays",
]
CATEGORY_TITLES = {
    "architectures": "Architectures",
    "language-models": "Language Models",
    "image-generation": "Image, Video & 3D Generation",
    "multimodal": "Multimodal & Audio",
    "robotics": "Robotics & Embodied AI",
    "techniques": "Techniques & Methods",
    "essays": "Essays & Landmark Posts",
}

# Explainers: pages that are not summaries of one paper - model family
# timelines, benchmark guides, history, policy, economics, open questions.
# They live in explainers/<section>/<slug>.md, are unnumbered, and carry a
# "Last reviewed" date because most of them describe a moving target.
EXPLAINERS_DIR = ROOT / "explainers"
EXPLAINER_SECTIONS = [
    ("history", "History"),
    ("model-families", "Model Families"),
    ("benchmarks", "Benchmarks"),
    ("concepts", "Concepts"),
    ("compute", "Compute & Economics"),
    ("policy", "Policy & Governance"),
    ("ecosystem", "Ecosystem"),
    ("open-questions", "Open Questions"),
]
# An explainer older than this is flagged on every build. Model families and
# policy pages go stale in months, so the build says so rather than letting a
# reader find out.
STALE_AFTER_DAYS = 180

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
    "69-classifier-free-guidance": ["image-generation", "diffusion", "guidance"],
    "70-ddim": ["image-generation", "diffusion", "sampling", "inference-optimization"],
    "71-controlnet": ["image-generation", "diffusion", "controllable-generation", "fine-tuning"],
    "72-flow-matching-sd3": ["image-generation", "diffusion", "flow-matching", "architecture"],
    "73-resnet": ["architecture", "vision", "computer-vision"],
    "74-unet": ["architecture", "vision", "computer-vision", "diffusion"],
    "75-grouped-query-attention": ["attention", "architecture", "efficiency", "inference-optimization"],
    "76-zero-megatron": ["distributed-training", "systems", "scaling", "efficiency"],
    "77-self-consistency": ["reasoning", "chain-of-thought", "test-time-compute", "prompting"],
    "78-reflexion": ["agents", "reasoning", "tool-use", "prompting"],
    "79-self-instruct": ["instruction-tuning", "synthetic-data", "alignment"],
    "80-flan": ["instruction-tuning", "pretraining", "scaling"],
    "81-emergent-abilities": ["scaling", "evaluation", "reasoning"],
    "82-sparse-autoencoders": ["interpretability", "safety"],
    "83-sleeper-agents": ["safety", "alignment", "interpretability"],
    "84-swe-bench": ["evaluation", "benchmarks", "agents", "code"],
    "85-llm-as-judge": ["evaluation", "benchmarks", "alignment"],
    "86-gptq-awq-quantization": ["quantization", "efficiency", "inference-optimization"],
    "87-dense-retrieval": ["retrieval", "embeddings", "search"],
    "88-mae": ["vision", "pretraining", "self-supervised", "architecture"],
    "89-vq-vae": ["image-generation", "vae", "discrete-representation"],
    "90-vq-gan": ["image-generation", "gan", "transformers", "discrete-representation"],
    "91-imagen": ["image-generation", "diffusion", "text-to-image"],
    "92-dreambooth": ["image-generation", "diffusion", "fine-tuning", "controllable-generation"],
    "93-gpt1": ["language-model", "pretraining", "transfer-learning"],
    "94-palm": ["language-model", "scaling", "pretraining"],
    "95-mistral-7b": ["language-model", "efficiency", "attention"],
    "96-llama-guard": ["safety", "alignment", "evaluation"],
    "97-star": ["reasoning", "self-improvement", "synthetic-data"],
    "98-quiet-star": ["reasoning", "self-improvement", "pretraining"],
    "99-self-refine": ["reasoning", "prompting", "self-improvement"],
    "100-voyager": ["agents", "tool-use", "self-improvement", "code"],
    "101-alphafold3": ["science", "diffusion"],
    "102-alphazero": ["reinforcement-learning", "search", "self-play"],
    "103-kto": ["alignment", "preference-optimization"],
    "104-genie": ["video-generation", "world-models", "self-supervised"],
    "105-dreamerv3": ["reinforcement-learning", "world-models"],
    "106-esm": ["science", "language-model", "embeddings"],
    "107-cicero": ["agents", "reinforcement-learning", "reasoning"],
    "108-computing-machinery-and-intelligence": ["essay", "history", "evaluation"],
    "109-unreasonable-effectiveness-of-rnns": ["essay", "language-model", "history"],
    "110-software-2": ["essay", "code"],
    "111-bitter-lesson": ["essay", "scaling"],
    "112-scaling-hypothesis": ["essay", "scaling"],
    "113-situational-awareness": ["essay", "scaling", "policy"],
    "114-machines-of-loving-grace": ["essay", "science", "policy"],
    "115-building-effective-agents": ["essay", "agents", "tool-use"],
    "116-era-of-experience": ["essay", "reinforcement-learning", "agents"],
    "117-nerf": ["3d", "image-generation", "computer-vision"],
    "118-3d-gaussian-splatting": ["3d", "computer-vision", "efficiency"],
    "119-dalle2-unclip": ["image-generation", "diffusion", "text-to-image"],
    "120-consistency-models": ["image-generation", "diffusion", "inference-optimization"],
    "121-audiolm": ["audio", "language-model", "discrete-representation"],
    "122-vall-e": ["audio", "language-model", "safety"],
    "123-rt2": ["robotics", "multimodal", "transfer-learning"],
    "124-pi0": ["robotics", "flow-matching", "multimodal"],
    "125-open-x-embodiment": ["robotics", "scaling", "benchmarks"],
    "126-induction-heads": ["interpretability", "attention", "transformers"],
    "127-gcg-adversarial-attacks": ["safety", "alignment", "evaluation"],
    "128-weak-to-strong": ["alignment", "safety", "scaling"],
    "129-red-teaming-lms": ["safety", "evaluation", "alignment"],
    "130-yarn-context-extension": ["long-context", "position-encoding", "fine-tuning"],
    "131-longformer": ["attention", "long-context", "efficiency"],
    "132-ruler": ["long-context", "evaluation", "benchmarks"],
    "133-fineweb": ["pretraining", "datasets", "data-curation"],
    "134-knowledge-distillation": ["efficiency", "distillation", "transfer-learning"],
    "135-phi-1-textbooks": ["language-model", "synthetic-data", "code"],
    "136-bpe-subword-units": ["tokenization", "language-model", "embeddings"],
    "137-mmlu": ["evaluation", "benchmarks"],
    "138-arc-agi": ["evaluation", "benchmarks", "reasoning"],
    "139-osworld": ["evaluation", "benchmarks", "agents"],
    "140-model-collapse": ["synthetic-data", "pretraining", "evaluation"],
    "141-multi-head-latent-attention": ["attention", "architecture", "efficiency", "inference-optimization"],
    "142-adam": ["optimization", "training"],
    "143-mixed-precision-training": ["efficiency", "training", "systems"],
    "144-muon": ["optimization", "training", "efficiency"],
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


def build_site_tree(papers: list[dict], explainers: list[dict] | None = None) -> None:
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
    for name in ("README.md", "BROWSE.md", "INDEX.md", "TAGS.md", "EXPLAINERS.md", "CONTRIBUTING.md",
                 "papers.json", "papers.csv", "LICENSE"):
        src = ROOT / name
        if src.exists():
            shutil.copy2(src, site / name)

    docs_src = ROOT / "docs"
    if docs_src.exists():
        shutil.copytree(docs_src, site / "docs")

    # Brand images referenced by the README's header block. MkDocs only serves
    # files under docs_dir, so they have to be mirrored in at the same path.
    assets_src = ROOT / "assets"
    if assets_src.exists():
        shutil.copytree(assets_src, site / "assets")

    # Site chrome: the stylesheet the theme loads, and the site-only landing
    # page that replaces the staged README. README.md is a repo front page
    # (badges, structure, "star this repo"); the landing page is a website front
    # page. MkDocs maps README.md to the site root exactly as it does index.md,
    # so writing over the staged copy puts the landing page at "/". The repo's
    # own README.md is never touched.
    css_src = ROOT / ".github" / "site" / "extra.css"
    if css_src.exists():
        css_dest = site / "assets" / "site" / "extra.css"
        css_dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(css_src, css_dest)

    # IndexNow ownership key, served at the site root as <key>.txt so
    # scripts/notify_indexnow.py can name it as keyLocation.
    key_src = ROOT / ".github" / "site" / "indexnow-key.txt"
    if key_src.exists():
        key = key_src.read_text(encoding="utf-8").strip()
        (site / f"{key}.txt").write_text(key + "\n", encoding="utf-8")

    home_src = ROOT / ".github" / "site" / "home.md"
    if home_src.exists():
        (site / "README.md").write_text(render_home(papers, explainers or []), encoding="utf-8")

    template = PAPERS_DIR / "_TEMPLATE.md"
    if template.exists():
        (site / "papers").mkdir(parents=True, exist_ok=True)
        shutil.copy2(template, site / "papers" / "_TEMPLATE.md")
    explainer_template = EXPLAINERS_DIR / "_TEMPLATE.md"
    if explainer_template.exists():
        (site / "explainers").mkdir(parents=True, exist_ok=True)
        shutil.copy2(explainer_template, site / "explainers" / "_TEMPLATE.md")

    for item in list(papers) + list(explainers or []):
        dest = site / item["path"]
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / item["path"], dest)


BY_YEAR_START = "<!-- byyear:start -->"
BY_YEAR_END = "<!-- byyear:end -->"


def write_by_year(papers: list[dict]) -> bool:
    """Regenerate README.md's year-distribution block between its markers.

    The block used to be hand-maintained while claiming to be generated from
    papers.json, so it drifted every time papers were added. It is counts plus a
    link rather than counts plus a list of names: the names were the part that
    went stale, and INDEX.md already carries the clickable list.
    """
    readme = ROOT / "README.md"
    if not readme.exists():
        return False
    text = readme.read_text(encoding="utf-8")
    if BY_YEAR_START not in text or BY_YEAR_END not in text:
        return False

    counts: dict[int, int] = {}
    for p in papers:
        if p["year"]:
            counts[p["year"]] = counts.get(p["year"], 0) + 1

    lines = [BY_YEAR_START]
    for year in sorted(counts):
        n = counts[year]
        lines.append(f"- **{year}:** {n} paper{'s' if n != 1 else ''}")
    lines.append(BY_YEAR_END)

    start = text.index(BY_YEAR_START)
    end = text.index(BY_YEAR_END) + len(BY_YEAR_END)
    updated = text[:start] + "\n".join(lines) + text[end:]
    if updated != text:
        readme.write_text(updated, encoding="utf-8")
        return True
    return False


def render_home(papers: list[dict], explainers: list[dict] | None = None) -> str:
    """Fill the landing page's count tokens from the parsed paper list.

    Every number on .github/site/home.md is a {{token}} rather than a typed
    figure, so the landing page cannot drift away from the tree the way a
    hand-maintained count does. An unknown token is left as-is and reported,
    which shows up as literal braces on the page rather than a silent zero.
    """
    src = (ROOT / ".github" / "site" / "home.md").read_text(encoding="utf-8")

    years = [p["year"] for p in papers if p["year"]]
    topics = sorted({t for p in papers for t in p["topics"]})
    guides = sorted((ROOT / "docs").glob("*.md")) if (ROOT / "docs").exists() else []

    # The written material: summary prose plus the guides. Deliberately not the
    # generated list pages (INDEX, TAGS, BROWSE) - counting a generated index of
    # the papers as words about the papers inflates the figure for nothing.
    # README.md's Quick Stats table reports the same definition.
    summary_words = sum(len(p["_body"].split()) for p in papers)
    explainers = explainers or []
    words = summary_words + sum(
        len(g.read_text(encoding="utf-8").split()) for g in guides
    ) + sum(e["words"] for e in explainers)

    # How much source material the summaries stand in for. source_lengths.json
    # is written by scripts/measure_sources.py (the one networked script, run by
    # hand); if it is absent or empty the tokens fall back to "-" rather than
    # inventing a figure. The ratio is computed over MATCHED papers only - the
    # measured papers' source words against those same papers' summary words -
    # so it is not distorted by the papers whose PDFs could not be retrieved.
    src_total = matched_summary = measured = 0
    cache_path = ROOT / "source_lengths.json"
    if cache_path.exists():
        sources = json.loads(cache_path.read_text(encoding="utf-8")).get("sources", {})
        for p in papers:
            n = (sources.get(p["slug"]) or {}).get("words")
            if not n:
                continue
            src_total += n
            matched_summary += len(p["_body"].split())
            measured += 1
    ratio = src_total / matched_summary if matched_summary else 0

    # One chip per category, linking into that section of the index. The `n`
    # span is the count, styled as the accent in extra.css.
    chips = []
    for cat in CATEGORY_ORDER:
        group = [p for p in papers if p["category"] == cat]
        if not group:
            continue
        title = CATEGORY_TITLES.get(cat, cat)
        # Match the GitHub-compatible slugifier mkdocs.yml configures: drop
        # everything that is not alphanumeric, space, or hyphen, then turn
        # spaces into hyphens. "Image & Video Generation" therefore becomes
        # "image--video-generation" - the ampersand leaves a gap, it does not
        # collapse. Building with --strict is what catches this if it drifts.
        anchor = re.sub(r"[^a-z0-9 \-]", "", title.lower()).replace(" ", "-")
        chips.append(f'- [{title} <span class="n">{len(group)}</span>](INDEX.md#{anchor})')

    tokens = {
        "papers": f"{len(papers)}",
        "words": f"{words // 1000},000+" if words >= 1000 else str(words),
        "summary_words": f"{summary_words // 1000},000",
        "source_words": f"{src_total / 1_000_000:.1f}M" if src_total else "-",
        "compression": f"{ratio:.0f}" if ratio else "-",
        "measured_papers": f"{measured}" if measured else "-",
        "unmeasured": f"{len(papers) - measured}" if measured else "-",
        "years": f"{max(years) - min(years) + 1}" if years else "-",
        "topics": f"{len(topics)}",
        "guides": f"{len(guides)}",
        "explainers": f"{len(explainers)}",
        "category_chips": "\n".join(chips),
    }

    missing = []

    def sub(m):
        key = m.group(1)
        if key not in tokens:
            missing.append(key)
            return m.group(0)
        return tokens[key]

    out = re.sub(r"\{\{(\w+)\}\}", sub, src)
    if missing:
        print(f"WARN unknown home.md token(s): {', '.join(sorted(set(missing)))}")
    return out


ONE_LINE_RE = re.compile(r"^\*\*In one line:\*\*\s*(.+?)\s*$", re.MULTILINE)
REVIEWED_RE = re.compile(r"^\*\*Last reviewed:\*\*\s*(\d{4}-\d{2}-\d{2})\s*$", re.MULTILINE)


def parse_explainers() -> list[dict]:
    """Every explainers/<section>/<slug>.md, in section order then by title.

    Required header: a `# Title`, an `**In one line:**` pitch and a
    `**Last reviewed:** YYYY-MM-DD` line. check_counts.py fails CI when one is
    missing, so this parser can treat them as present and fall back quietly.
    """
    if not EXPLAINERS_DIR.exists():
        return []
    order = {key: i for i, (key, _) in enumerate(EXPLAINER_SECTIONS)}
    out = []
    for path in EXPLAINERS_DIR.glob("*/*.md"):
        section = path.parent.name
        if path.name.startswith("_") or section not in order:
            continue
        body = path.read_text(encoding="utf-8")
        title_m = TITLE_RE.search(body)
        line_m = ONE_LINE_RE.search(body)
        rev_m = REVIEWED_RE.search(body)
        out.append({
            "section": section,
            "slug": path.stem,
            "title": title_m.group(1).strip() if title_m else path.stem,
            "one_line": clean_markup(line_m.group(1)) if line_m else "",
            "reviewed": rev_m.group(1) if rev_m else "",
            "path": str(path.relative_to(ROOT)).replace("\\", "/"),
            "words": len(body.split()),
        })
    out.sort(key=lambda e: (order[e["section"]], e["title"].lower()))
    return out


def write_explainers_hub(explainers: list[dict]) -> None:
    """Generate EXPLAINERS.md: every explainer, grouped by section."""
    lines = [
        "# Explainers",
        "",
        f"**{len(explainers)}** pages on the parts of AI that are not a single paper: how the model "
        "families evolved, what the benchmarks actually measure, what training and serving cost, "
        "the rules being written, and the questions nobody has settled. Each one links down into "
        "the paper summaries for the detail. Generated by `scripts/build_manifest.py` - do not "
        "edit by hand.",
        "",
        "Most of these describe a moving target, so every page states when it was last reviewed.",
        "",
    ]
    for key, title in EXPLAINER_SECTIONS:
        group = [e for e in explainers if e["section"] == key]
        if not group:
            continue
        lines.append(f"## {title}")
        lines.append("")
        lines.append("| Page | In one line | Reviewed |")
        lines.append("|------|-------------|----------|")
        for e in group:
            pitch = e["one_line"].replace("|", "\\|")
            lines.append(f"| [{e['title']}]({e['path']}) | {pitch} | {e['reviewed']} |")
        lines.append("")
    (ROOT / "EXPLAINERS.md").write_text("\n".join(lines), encoding="utf-8")


def write_mkdocs(papers: list[dict], explainers: list[dict] | None = None) -> None:
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
    nav.append("      - Coverage & Gaps: docs/GAPS.md")
    if explainers:
        nav.append("  - Explainers:")
        nav.append("      - All Explainers: EXPLAINERS.md")
        for key, title in EXPLAINER_SECTIONS:
            group = [e for e in explainers if e["section"] == key]
            if not group:
                continue
            nav.append(f"      - {title}:")
            for e in group:
                label = e["title"].replace('"', "'")
                nav.append(f'          - "{label}": {e["path"]}')
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

    # mkdocs.yml is hand-maintained (theme, palette, extensions, extra) and
    # carries no nav. Append the generated nav to a copy of it and build from
    # that. mkdocs.generated.yml is git-ignored - it is derived, and committing
    # it would make every paper addition a diff in two files.
    base = (ROOT / "mkdocs.yml").read_text(encoding="utf-8").rstrip("\n")
    banner = (
        "\n\n# ---------------------------------------------------------------\n"
        "# Generated by scripts/build_manifest.py - do not edit this file.\n"
        "# Edit mkdocs.yml (theme/config) or write_mkdocs() (nav) instead.\n"
        "# ---------------------------------------------------------------\n"
    )
    (ROOT / "mkdocs.generated.yml").write_text(
        base + banner + "\n".join(nav) + "\n", encoding="utf-8"
    )


def main() -> None:
    summaries = sorted(PAPERS_DIR.glob("*/*/summary.md"))
    papers = [parse_summary(p) for p in summaries]
    papers.sort(key=lambda p: (p["number"] is None, p["number"] or 0))

    explainers = parse_explainers()

    changed = write_frontmatter(papers)
    write_json(papers)
    write_csv(papers)
    write_index(papers)
    write_tags(papers)
    write_explainers_hub(explainers)
    write_mkdocs(papers, explainers)
    write_by_year(papers)
    build_site_tree(papers, explainers)

    print(f"Parsed {len(papers)} summaries and {len(explainers)} explainers.")
    today = dt.date.today()
    stale = [e["path"] for e in explainers if e["reviewed"]
             and (today - dt.date.fromisoformat(e["reviewed"])).days > STALE_AFTER_DAYS]
    if stale:
        print(f"WARN explainers not reviewed in {STALE_AFTER_DAYS} days: {', '.join(stale)}")
    print(f"Frontmatter written/updated on {changed} file(s).")
    missing_url = [p["slug"] for p in papers if not p["url"]]
    missing_year = [p["slug"] for p in papers if p["year"] is None]
    if missing_url:
        print(f"WARN no source URL parsed: {', '.join(missing_url)}")
    if missing_year:
        print(f"WARN no year parsed: {', '.join(missing_year)}")
    print("Wrote papers.json, papers.csv, INDEX.md, TAGS.md, EXPLAINERS.md, mkdocs.generated.yml, site-build/")


if __name__ == "__main__":
    main()
