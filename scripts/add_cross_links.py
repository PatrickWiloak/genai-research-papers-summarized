#!/usr/bin/env python3
"""
add_cross_links.py - append a "Related in This Collection" footer to each
paper summary, linking sibling papers that the text actually references.

Deterministic and idempotent: for every paper we know a set of aliases
(acronyms / short names). For each summary we scan its prose (code blocks
stripped, and the existing footer removed first) for the aliases of OTHER
papers, then write a fresh footer linking the matches. Re-running replaces
the footer in place between HTML-comment markers, so it never accumulates.

Run after scripts/build_manifest.py (needs papers.json). No dependencies.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

START = "<!-- related:start -->"
END = "<!-- related:end -->"
MAX_LINKS = 8

FENCE_RE = re.compile(r"```.*?```", re.DOTALL)
INLINE_CODE_RE = re.compile(r"`[^`]*`")
BLOCK_RE = re.compile(re.escape(START) + r".*?" + re.escape(END) + r"\s*", re.DOTALL)

# Curated aliases per slug. Each alias is matched case-sensitively on word
# boundaries, so distinctive names ("LoRA", "RAG") do not collide with prose.
# Multi-word aliases are matched as-is. Keep aliases specific to avoid noise.
ALIASES: dict[str, list[str]] = {
    "01-attention-is-all-you-need": ["Attention Is All You Need", "the Transformer", "Transformer architecture"],
    "02-generative-adversarial-networks": ["GAN", "GANs", "Generative Adversarial"],
    "03-bert": ["BERT"],
    "04-gpt3-few-shot-learners": ["GPT-3"],
    "05-instructgpt-rlhf": ["InstructGPT", "RLHF"],
    "06-diffusion-models": ["DDPM", "denoising diffusion"],
    "07-stable-diffusion": ["Stable Diffusion", "latent diffusion"],
    "08-clip": ["CLIP"],
    "09-chain-of-thought": ["Chain-of-Thought", "chain of thought", "CoT"],
    "10-lora": ["LoRA"],
    "11-vision-transformer": ["Vision Transformer", "ViT"],
    "12-scaling-laws": ["scaling laws"],
    "13-rag": ["RAG", "retrieval-augmented"],
    "14-constitutional-ai": ["Constitutional AI"],
    "15-llama": ["LLaMA 1", "LLaMA:"],
    "16-flash-attention": ["FlashAttention", "Flash Attention"],
    "17-llama2": ["LLaMA 2", "Llama 2"],
    "18-chinchilla": ["Chinchilla"],
    "19-dpo": ["DPO", "Direct Preference Optimization"],
    "20-mamba": ["Mamba", "state space model", "SSM"],
    "21-react": ["ReAct"],
    "22-qlora": ["QLoRA"],
    "23-gpt4v": ["GPT-4V"],
    "24-toolformer": ["Toolformer"],
    "25-tree-of-thoughts": ["Tree of Thoughts", "ToT"],
    "26-deepseek-r1": ["DeepSeek-R1"],
    "27-deepseek-v3": ["DeepSeek-V3"],
    "28-qwen3": ["Qwen3", "Qwen"],
    "29-gemini-2.5": ["Gemini 2.5"],
    "30-claude-3.5-sonnet": ["Claude 3.5"],
    "31-openai-o1": ["o1"],
    "32-sam2": ["SAM 2", "Segment Anything"],
    "33-llama3.3": ["Llama 3.3", "LLaMA 3.3", "Llama 3 ", "Llama 3.1"],
    "34-meta-cot": ["Meta Chain-of-Thought", "Meta-CoT"],
    "35-rstar-math": ["rStar-Math"],
    "36-gpt4": ["GPT-4 ", "GPT-4."],
    "37-mixture-of-experts": ["Mixture-of-Experts", "Mixture of Experts", "Mixtral", "MoE"],
    "38-grpo": ["GRPO"],
    "39-rlvr": ["RLVR", "verifiable rewards"],
    "40-gpt4o": ["GPT-4o"],
    "41-llama4": ["Llama 4", "LLaMA 4"],
    "42-gpt5": ["GPT-5"],
    "43-claude4": ["Claude 4"],
    "44-sora-dit": ["Sora", "Diffusion Transformer", "DiT"],
    "45-speculative-decoding": ["Speculative Decoding", "speculative decoding"],
    "46-llava": ["LLaVA"],
    "47-gemini3": ["Gemini 3"],
    "48-dalle3": ["DALL-E 3", "DALL-E"],
    "49-whisper": ["Whisper"],
    "50-test-time-compute": ["test-time compute", "Test-Time Compute"],
    "51-process-reward-models": ["Process Reward Model", "PRM", "Let's Verify Step by Step"],
    "52-pagedattention-vllm": ["PagedAttention", "vLLM"],
    "53-word2vec": ["Word2Vec", "word2vec", "Skip-gram", "CBOW"],
    "54-rope-rotary-position-embedding": ["RoPE", "Rotary Position", "RoFormer"],
    "55-seq2seq": ["Seq2Seq", "sequence to sequence", "sequence-to-sequence"],
    "56-codex": ["Codex"],
    "57-vae": ["VAE", "Variational Autoencoder", "variational autoencoder"],
    "58-generative-agents": ["Generative Agents"],
    "59-model-context-protocol": ["Model Context Protocol", "MCP"],
    "60-graph-rag": ["GraphRAG", "Graph RAG"],
    "61-alphageometry": ["AlphaGeometry"],
    "62-alphaevolve": ["AlphaEvolve"],
    "63-ppo": ["PPO", "Proximal Policy Optimization"],
    "64-gpt2": ["GPT-2"],
    "65-t5": ["T5", "text-to-text"],
    "66-bahdanau-attention": ["Bahdanau", "align and translate", "additive attention"],
    "67-switch-transformer": ["Switch Transformer"],
    "68-alphafold": ["AlphaFold", "Evoformer"],
}


def strip_for_scan(text: str) -> str:
    text = BLOCK_RE.sub("", text)
    text = FENCE_RE.sub("", text)
    text = INLINE_CODE_RE.sub("", text)
    return text


def compile_alias(alias: str) -> re.Pattern:
    # Left word boundary; the alias itself carries any needed trailing
    # punctuation (e.g. "GPT-4 ", "GPT-4.") to avoid matching "GPT-4o".
    return re.compile(r"(?<![\w-])" + re.escape(alias.rstrip()))


def main() -> int:
    data = json.load(open(ROOT / "papers.json", encoding="utf-8"))
    by_slug = {p["slug"]: p for p in data["papers"]}

    # Precompile alias patterns per known slug.
    patterns = {slug: [(a, compile_alias(a)) for a in al]
                for slug, al in ALIASES.items() if slug in by_slug}

    changed = 0
    for p in data["papers"]:
        path = ROOT / p["path"]
        raw = path.read_text(encoding="utf-8")
        scan = strip_for_scan(raw)

        hits = []
        for slug, pats in patterns.items():
            if slug == p["slug"]:
                continue
            if any(rx.search(scan) for _, rx in pats):
                hits.append(by_slug[slug])

        hits.sort(key=lambda q: q["number"] or 0)
        hits = hits[:MAX_LINKS]

        body = BLOCK_RE.sub("", raw).rstrip() + "\n"
        if hits:
            depth = p["path"].count("/")  # papers/<cat>/<slug>/summary.md -> 3
            prefix = "../" * (depth - 1)  # from this summary up to papers/
            lines = [START,
                     "",
                     "---",
                     "",
                     "## Related in This Collection",
                     ""]
            for q in hits:
                rel = q["path"][len("papers/"):]  # <cat>/<slug>/summary.md
                lines.append(f"- [{q['title']}]({prefix}{rel})")
            lines += ["", END, ""]
            new = body.rstrip() + "\n\n" + "\n".join(lines)
        else:
            new = body

        if new != raw:
            path.write_text(new, encoding="utf-8")
            changed += 1

    print(f"Cross-link footers updated on {changed} file(s).")
    unknown = [p["slug"] for p in data["papers"] if p["slug"] not in ALIASES]
    if unknown:
        print(f"WARN no alias entry for: {', '.join(unknown)}")
    return 0


if __name__ == "__main__":
    main()
