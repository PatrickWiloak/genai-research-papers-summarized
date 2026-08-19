<!--
  Site-only landing page. Written over the staged copy of README.md by
  scripts/build_manifest.py - MkDocs maps README.md to the site root exactly
  as it does index.md. The repo's own README.md is never touched and stays
  the GitHub front page.

  The two pages have different jobs. README.md is a repo front page: badges,
  repository structure, contributing, "star this repo". This is a website
  landing page: what the site is, who it is for, and the handful of links a
  first-time visitor actually needs.

  Never hard-code a count here. Every number is a double-braced token filled by
  render_home() in scripts/build_manifest.py from the same parsed papers list
  that builds papers.json and INDEX.md, so a number on this page cannot drift
  away from the tree. Adding a new number means adding a token, not typing a
  figure.

  No `hide: toc` here. With toc.integrate the page's headings fold into the left
  sidebar, and hiding them would make Home the one page whose sections do not
  appear where every other page's do.
-->
<!-- markdownlint-disable MD030 -->
<!-- Material's card grids use `-   ` (three spaces) so the card body lines up at
     a four-space indent and stays part of the list item. MD030 wants one. -->


<div class="home-hero" markdown>

# The papers that built generative AI, in plain language { .home-title }

<p class="home-intro">{{papers}} foundational papers - Transformers to diffusion, RLHF to reasoning models - each read in full and rewritten as something you can actually finish.</p>

<p class="home-sub">No maths degree assumed, no paywall, no signup. Press <kbd>/</kbd> to search all {{words}} words.</p>

[Start the roadmap](docs/ROADMAP.md){ .md-button .md-button--primary }
[Browse all {{papers}} papers](INDEX.md){ .md-button }
[Look up a term](docs/GLOSSARY.md){ .md-button }

<div class="stat-strip" markdown>

- **{{papers}}** papers
- **{{years}}** years covered
- **{{words}}** words
- **{{topics}}** topic tags
- **{{guides}}** guides
- **$0** to read

</div>

</div>

<p class="home-kicker">Four ways in</p>

## Pick your entry point

<div class="grid cards" markdown>

-   :material-map-marker-path:{ .lg .middle } **Follow a path**

    ---

    A staged reading order that assumes nothing, from Word2Vec through to today's reasoning models. Start here if the field is new to you.

    [:octicons-arrow-right-24: Learning roadmap](docs/ROADMAP.md)

-   :material-book-open-page-variant:{ .lg .middle } **Browse the library**

    ---

    All {{papers}} summaries grouped by category, or filtered down to the {{topics}} topic tags - retrieval, alignment, efficiency, agents, scaling.

    [:octicons-arrow-right-24: Paper index](INDEX.md)

-   :material-scale-balance:{ .lg .middle } **Compare approaches**

    ---

    LoRA or full fine-tuning, RAG or long context, DPO or PPO. Decision guides for the choices you actually face when building.

    [:octicons-arrow-right-24: Comparisons](docs/COMPARISONS.md)

-   :material-clock-fast:{ .lg .middle } **Get the gist fast**

    ---

    One-line takeaways for every paper, plus which ones still matter in production and which are history you can safely skim.

    [:octicons-arrow-right-24: Quick reference](docs/QUICK_REFERENCE.md)

</div>

<p class="home-kicker">The library</p>

## Papers by category

<div class="cat-grid" markdown>

{{category_chips}}

</div>

[Full index with titles, authors, and years](INDEX.md){ .md-button }

<p class="home-kicker">Quick links</p>

## Jump to what you need

<div class="home-links" markdown>

- [Learning Roadmap](docs/ROADMAP.md) - the staged order to read them in
- [Reading Guide](docs/READING_GUIDE.md) - what is still current, what is history
- [Quick Reference](docs/QUICK_REFERENCE.md) - every paper in one line
- [Comparisons](docs/COMPARISONS.md) - decision guides for real build choices
- [Glossary](docs/GLOSSARY.md) - the vocabulary, defined once
- [Coverage & Gaps](docs/GAPS.md) - what is not here yet, and why
- [Browse visually](BROWSE.md) - grid view with a three-line pitch each
- [By topic tag](TAGS.md) - {{topics}} tags across the collection
- [papers.json](papers.json) - the whole collection as machine-readable data
- [Contributing](CONTRIBUTING.md) - how to add a paper, and the house style

</div>

## What each summary gives you

Every paper here was read in full, then rewritten to a fixed shape so you can move between them without relearning the format each time:

- **Why this matters** - the one thing that changed, stated before any notation.
- **The problem** - what people were stuck on, and why the obvious fix did not work.
- **The core innovation** - the actual idea, with diagrams instead of proofs.
- **Key results** - the numbers the paper reported, not the numbers people remember.
- **Real-world impact** - what shipped because of it.
- **Limitations** - what it does not do, including where the field has since moved past it.
- **Related in this collection** - generated cross-links, so following a thread never dead-ends.

Summaries are not a substitute for the paper. Every one links to the original, and the good ones are worth your time once the summary has told you what to look for.

## Who made this

Built by **[Patrick Wiloak](https://patrickwiloak.com)** - ex-AWS Solutions Architect, 10 years in tech, 18x multi-cloud certified.
[YouTube](https://youtube.com/@patrickwiloak) · [LinkedIn](https://www.linkedin.com/in/patricklukewilson/) · [Source on GitHub](https://github.com/PatrickWiloak/genai-research-papers-summarized)

Learning the cloud, data, and security side too? **[Cloud, Data, AI and Security - Zero to Hero](https://patrickwiloak.github.io/cloud-data-ai-security-zero-to-hero/)** is the sibling site: concepts, hands-on builds, and the most comprehensive certification library on GitHub.

We build custom software and products at **[Nobler Works](https://noblerworks.com/)**. Open-source training like this is how we give back - we are nothing without the community that supports us. If you need software built, [get in touch](https://noblerworks.com/).

!!! tip "Want the reps as well as the reading?"

    [![gitGood.dev - practice questions, coding challenges, system design and cloud certification practice exams for engineers, data, security, DevOps and product technologists](assets/brand/gitgood-banner.png){ width="300" }](https://gitgood.dev)

    This site gives you the material. **[gitGood](https://gitgood.dev)** gives you the reps, and tells you whether you actually know it - including ML and AI practice paths built for exactly the material on this site.

    10 days free, then $5/month or $40/year. The free tier needs no card.

## Fine print

Summaries are original work, released under [CC BY 4.0](LICENSE) - free to use with attribution. The papers themselves belong to their authors and are linked, never reproduced.

This is an independent educational resource. It is not affiliated with, endorsed by, or sponsored by OpenAI, Google, Anthropic, Meta, or any other organisation whose work is summarised here. All trademarks belong to their respective owners.

Found something wrong or out of date? [Contributions welcome](CONTRIBUTING.md).
