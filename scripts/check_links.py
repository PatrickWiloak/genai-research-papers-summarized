#!/usr/bin/env python3
"""
check_links.py - validate relative Markdown links across the repo.

Walks every tracked .md file, strips fenced/inline code (so pseudocode like
`foo[i](bar)` is not mistaken for a link), and checks that each relative link
target exists on disk. External (http/https/mailto) and pure-anchor (#...)
links are skipped. Exits non-zero if any relative link is broken.

No third-party dependencies. Run:  python3 scripts/check_links.py
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SKIP_DIRS = {"site-build", "site", "_site", "graphify-out", "node_modules",
             ".git", ".github", "venv", "__pycache__"}

FENCE_RE = re.compile(r"```.*?```", re.DOTALL)
INLINE_CODE_RE = re.compile(r"`[^`]*`")
LINK_RE = re.compile(r"\[[^\]]+\]\(([^)]+)\)")


def strip_code(text: str) -> str:
    text = FENCE_RE.sub("", text)
    text = INLINE_CODE_RE.sub("", text)
    return text


def is_external(target: str) -> bool:
    return target.startswith(("http://", "https://", "mailto:", "#", "tel:"))


def iter_markdown():
    for md in ROOT.rglob("*.md"):
        if any(part in SKIP_DIRS for part in md.relative_to(ROOT).parts):
            continue
        yield md


def main() -> int:
    broken: list[str] = []
    checked = 0
    for md in iter_markdown():
        text = strip_code(md.read_text(encoding="utf-8", errors="ignore"))
        for m in LINK_RE.finditer(text):
            target = m.group(1).strip()
            # links may be "path \"title\"" - drop any title part
            target = target.split()[0] if target else target
            if not target or is_external(target):
                continue
            path_part = target.split("#")[0]
            if not path_part:
                continue
            checked += 1
            resolved = (md.parent / path_part).resolve()
            if not resolved.exists():
                broken.append(f"{md.relative_to(ROOT)}  ->  {target}")

    if broken:
        print(f"Broken relative links ({len(broken)} of {checked} checked):")
        for b in sorted(broken):
            print(f"  {b}")
        return 1
    print(f"All {checked} relative links OK.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
