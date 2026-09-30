"""MkDocs hooks: per-page search metadata for the published site.

Loaded through `hooks:` in mkdocs.yml. Site-only: it sets keys on `page.meta`
at build time and never writes to the markdown. Ported from the sibling
cloud-data-ai-security-zero-to-hero site (2026-09-28) and adapted to papers and
explainers.

Why this exists. Until 2026-09-30 every page shipped the same
`<meta name="description">` - the site-wide `site_description` - and a
`<title>` of "<full paper title> - Everything AI, Summarized", which search
results cut off long before the part people search for. Google treats a
repeated description as boilerplate and writes its own snippet. So:

- `description`: an explainer's `**In one line:**` pitch; for a paper, a lead
  naming it as a plain-language summary plus the first prose paragraph of its
  "Why This Matters" section. A frontmatter `description:` always wins.
- `seo_title`: the `<title>` text, rendered by the `htmltitle` override in
  `.github/site-overrides/main.html`. Papers become "<Short name> Paper Summary
  (<year>)" when the full title is too long for a result line.
- `jsonld`: schema.org BreadcrumbList for every page; WebSite on the home
  page; for papers, an Article that names the original work in `isBasedOn`.
- `robots`: noindex on the two authoring templates.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from add_cross_links import ALIASES  # noqa: E402  curated short names per slug

SITE_SHORT = "Everything AI"
DESC_MAX = 160
DESC_MIN = 50
NOINDEX = ("papers/_TEMPLATE.md", "explainers/_TEMPLATE.md")

_EMOJI_RE = re.compile("[\U0001F000-\U0001FAFF☀-➿⬀-⯿️‍←-⇿⌀-⏿]")
_IMAGE_RE = re.compile(r"!\[[^\]]*\]\([^)]*\)")
_LINK_RE = re.compile(r"\[([^\]]*)\]\([^)]*\)")
_HTML_RE = re.compile(r"<[^>]+>")
_ATTR_LIST_RE = re.compile(r"\{:?[^}]*\}")
_FENCE_RE = re.compile(r"^\s{0,3}(```+|~~~+)")
_NON_PROSE_START = ("#", "|", "- ", "* ", "+ ", ">", "<", "!", "---", "***", "___", "{")
_LABEL_LINE_RE = re.compile(r"^\*\*[^*]{1,60}(:\*\*|\*\*:)")
_NUMBERED_RE = re.compile(r"^\d+[.)]\s")
_ONE_LINE_RE = re.compile(r"^\*\*In one line:\*\*\s*(.+?)\s*$", re.MULTILINE)
_SHORT_RE = re.compile(r"\(([^()]{2,60})\)\s*$")

# Search-title names where neither the heading's "(Short)" nor the curated
# aliases give a clean one. Keyed by slug.
SEO_SHORT = {
    "37-mixture-of-experts": "Mixtral and Mixture of Experts",
    "72-flow-matching-sd3": "Flow Matching and Stable Diffusion 3",
    "75-grouped-query-attention": "Grouped-Query Attention",
    "81-emergent-abilities": "Emergent Abilities of LLMs",
    "82-sparse-autoencoders": "Sparse Autoencoders",
    "108-computing-machinery-and-intelligence": "Turing's Computing Machinery and Intelligence",
    "109-unreasonable-effectiveness-of-rnns": "The Unreasonable Effectiveness of RNNs",
}


def _plain(text: str) -> str:
    text = _IMAGE_RE.sub("", text)
    text = _LINK_RE.sub(r"\1", text)
    text = _HTML_RE.sub("", text)
    text = _ATTR_LIST_RE.sub("", text)
    text = _EMOJI_RE.sub("", text)
    text = re.sub(r"(\*\*|__|\*|`|~~)", "", text)
    # Attribute-safe: the template renders these inside content="...".
    text = text.replace('"', "'").replace("<", "").replace(">", "")
    return re.sub(r"\s+", " ", text).strip()


def _clip(text: str, limit: int = DESC_MAX) -> str:
    if len(text) <= limit:
        return text
    cut = text[: limit - 3].rsplit(" ", 1)[0].rstrip(",;:-(")
    return cut + "..."


def _blocks(markdown: str) -> list[list[str]]:
    """Blank-line separated blocks, code fences skipped, headings on their own."""
    blocks, cur, fence = [], [], None
    for line in markdown.splitlines():
        m = _FENCE_RE.match(line)
        if fence:
            if m and m.group(1)[0] == fence[0] and len(m.group(1)) >= len(fence):
                fence = None
            continue
        if m:
            fence = m.group(1)
            if cur:
                blocks.append(cur)
                cur = []
            continue
        if line.lstrip().startswith("#"):
            if cur:
                blocks.append(cur)
                cur = []
            blocks.append([line.rstrip()])
        elif line.strip():
            cur.append(line.rstrip())
        elif cur:
            blocks.append(cur)
            cur = []
    if cur:
        blocks.append(cur)
    return blocks


def _is_prose(block: list[str]) -> bool:
    first = block[0].lstrip()
    if first.startswith(_NON_PROSE_START) or _NUMBERED_RE.match(first):
        return False
    if all(_LABEL_LINE_RE.match(l.strip()) for l in block):
        return False
    return bool(re.search(r"[A-Za-z]{3,}", _plain(" ".join(block))))


def description_from_body(markdown: str) -> str:
    """The first real prose paragraph, preferring the one under "Why This ... Matters"."""
    blocks = _blocks(markdown)
    start = 0
    for i, b in enumerate(blocks):
        if re.match(r"^##\s+Why\b", b[0].strip()):
            start = i + 1
            break
    for block in blocks[start:start + 40]:
        if _is_prose(block):
            lines = []
            for line in block:
                l = line.lstrip()
                if lines and (l.startswith(_NON_PROSE_START) or _NUMBERED_RE.match(l)):
                    break
                lines.append(line)
            text = _plain(" ".join(lines))
            if len(text) >= DESC_MIN:
                return text
    return ""


def _title_text(title: str | None) -> str:
    return re.sub(r"^[^\w(]+\s*", "", _plain(title or ""))


def _kind(src: str) -> str:
    if src.startswith("papers/essays/"):
        return "essay"
    if src.startswith("papers/") and src.endswith("/summary.md"):
        return "paper"
    if src.startswith("explainers/") and not src.endswith("_TEMPLATE.md"):
        return "explainer"
    return ""


def _short_name(h1: str) -> str:
    m = _SHORT_RE.search(h1)
    return m.group(1).strip() if m else ""


def on_page_markdown(markdown, page, config, files, **kwargs):
    meta = page.meta
    src = page.file.src_uri
    kind = _kind(src)
    h1m = re.search(r"^#\s+(.+?)\s*#*\s*$", markdown, re.MULTILINE)
    h1 = _title_text(h1m.group(1) if h1m else page.title)
    if not re.search(r"[A-Za-z]{3}", h1):
        h1 = _title_text(page.title)  # e.g. a template whose H1 is a placeholder
    if not re.search(r"[A-Za-z]{3}", h1):
        h1 = "Template"
    # "Adam: A Method... (Adam)" -> drop a trailing "(X)" that repeats the title.
    m = _SHORT_RE.search(h1)
    if m and m.group(1).lower() in h1[: m.start()].lower():
        h1 = h1[: m.start()].strip()
    short = _short_name(h1)
    slug = src.split("/")[-2] if src.count("/") >= 2 else ""
    if slug in SEO_SHORT:
        short = SEO_SHORT[slug]
    if not short and kind in ("paper", "essay"):
        head = h1.split(":")[0].strip()
        if ":" in h1 and len(head) <= 32:
            short = head
        elif ALIASES.get(slug):
            short = ALIASES[slug][0].rstrip(":. ")
    year = meta.get("year")

    if not meta.get("description") and not page.is_homepage:
        desc = ""
        if kind == "explainer":
            m = _ONE_LINE_RE.search(markdown)
            desc = _plain(m.group(1)) if m else ""
        if not desc:
            body = description_from_body(markdown)
            if kind in ("paper", "essay"):
                name = short or re.split(r"[:(]", h1)[0].strip()
                lead = f"{name} ({year}) explained in plain language" if year else f"{name} explained in plain language"
                desc = f"{lead}: {body}" if body else f"{lead}."
            else:
                desc = body
        if not desc and h1:
            desc = f"{h1} - from {config.site_name}."
        if desc:
            meta["description"] = _clip(desc)

    if src.startswith(NOINDEX):
        meta["robots"] = "noindex, follow"

    if page.is_homepage:
        meta["seo_title"] = f"{config.site_name}: AI Papers and Explainers in Plain Language"
    elif h1:
        if kind in ("paper", "essay") and short and (len(h1) > 50 or slug in SEO_SHORT):
            label = "Essay Summary" if kind == "essay" else "Paper Summary"
            seo = f"{short} {label}" + (f" ({year})" if year else "")
        elif kind in ("paper", "essay"):
            seo = f"{h1} - Summary"
        else:
            seo = h1
        meta["seo_title"] = f"{seo} | {SITE_SHORT}" if len(seo) <= 46 else seo
    return markdown


def _section_url(section) -> str | None:
    for child in getattr(section, "children", None) or []:
        if getattr(child, "is_page", False) and getattr(child, "is_index", False):
            return child.canonical_url
    return None


def on_page_context(context, page, config, nav, **kwargs):
    site = config.site_url
    crumbs = [{"name": "Home", "item": site}]
    for section in reversed(page.ancestors):
        url = _section_url(section)
        if url and url != site:
            crumbs.append({"name": _title_text(section.title), "item": url})
    if not page.is_homepage and page.canonical_url not in {c["item"] for c in crumbs}:
        crumbs.append({"name": _title_text(page.title), "item": page.canonical_url})

    author = {"@type": "Person", "name": config.site_author, "url": "https://patrickwiloak.com"}
    publisher = {"@type": "Organization", "name": "Nobler Works", "url": "https://noblerworks.com/"}
    graph = []
    if len(crumbs) > 1:
        graph.append({
            "@type": "BreadcrumbList",
            "itemListElement": [
                {"@type": "ListItem", "position": i + 1, **c} for i, c in enumerate(crumbs)
            ],
        })
    if page.is_homepage:
        graph.append({
            "@type": "WebSite",
            "name": config.site_name,
            "alternateName": SITE_SHORT,
            "url": site,
            "description": config.site_description.strip(),
            "inLanguage": "en",
            "author": author,
            "publisher": publisher,
        })
    kind = _kind(page.file.src_uri)
    if kind:
        meta = page.meta
        article = {
            "@type": "Article",
            "headline": _title_text(page.title)[:110],
            "description": meta.get("description", ""),
            "url": page.canonical_url,
            "inLanguage": "en",
            "author": author,
            "publisher": publisher,
            "isAccessibleForFree": True,
            "license": "https://creativecommons.org/licenses/by/4.0/",
        }
        if meta.get("tags"):
            article["keywords"] = ", ".join(meta["tags"])
        if kind in ("paper", "essay") and meta.get("url"):
            original = {
                "@type": "CreativeWork" if kind == "essay" else "ScholarlyArticle",
                "name": re.sub(r"\s*\([^()]*\)\s*$", "", _title_text(page.title)),
                "url": meta["url"],
            }
            if meta.get("authors"):
                original["author"] = meta["authors"][:300]
            if meta.get("year"):
                original["datePublished"] = str(meta["year"])
            article["isBasedOn"] = original
            article["about"] = original["name"]
        graph.append(article)
    if graph:
        doc = {"@context": "https://schema.org", "@graph": graph}
        # "</" inside a <script> would end it early.
        page.meta["jsonld"] = json.dumps(doc, ensure_ascii=False).replace("</", "<\\/")
    return context
