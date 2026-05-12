"""Fire a chat and an embedding request concurrently and print both results.

Usage:
    python ping-both.py                 # hits the public tunnel
    python ping-both.py --local         # hits the local proxies
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time

import aiohttp

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

API_KEY = os.environ.get("LLAMA_API_KEY", "rRZsSjRvaUuRMr5AeDA14rO9jaSlhSRhRtBI5ZlO")

REMOTE_CHAT = "https://ai.htk-hrt.cc"
REMOTE_EMBED = "https://embed.htk-hrt.cc"
LOCAL_CHAT = "http://127.0.0.1:8001"
LOCAL_EMBED = "http://127.0.0.1:8003"

CHAT_MODEL = "qwen3.6-35b-q3-32k"
EMBED_MODEL = "qwen3-embedding-4b-8k"


async def ask_chat(session: aiohttp.ClientSession, base: str) -> tuple[float, str]:
    started = time.monotonic()
    payload = {
        "model": CHAT_MODEL,
        "messages": [{"role": "user", "content": "hello"}],
        "stream": False,
        "max_tokens": 4096,
    }
    headers = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}
    async with session.post(f"{base}/v1/chat/completions", json=payload, headers=headers) as r:
        data = await r.json()
        elapsed = time.monotonic() - started
        if r.status != 200:
            return elapsed, f"HTTP {r.status}: {json.dumps(data)[:300]}"
        msg = data["choices"][0].get("message") or {}
        content = (msg.get("content") or "").strip()
        reasoning = (msg.get("reasoning_content") or "").strip()
        finish = data["choices"][0].get("finish_reason")
        usage = data.get("usage") or {}
        if not content:
            return elapsed, (
                f"<empty content> finish={finish} usage={usage} "
                f"reasoning_preview={reasoning[:120]!r}"
            )
        return elapsed, f"{content}  (finish={finish}, completion_tokens={usage.get('completion_tokens')})"


async def ask_embed(session: aiohttp.ClientSession, base: str) -> tuple[float, str]:
    started = time.monotonic()
    payload = {"model": EMBED_MODEL, "input": "hello"}
    headers = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}
    async with session.post(f"{base}/v1/embeddings", json=payload, headers=headers) as r:
        data = await r.json()
        elapsed = time.monotonic() - started
        if r.status != 200:
            return elapsed, f"HTTP {r.status}: {json.dumps(data)[:300]}"
        vec = data["data"][0]["embedding"]
        preview = ", ".join(f"{v:.4f}" for v in vec[:5])
        return elapsed, f"dim={len(vec)}  first 5=[{preview}, ...]"


async def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--local", action="store_true", help="hit local proxies instead of the tunnel")
    args = p.parse_args()

    chat_base = LOCAL_CHAT if args.local else REMOTE_CHAT
    embed_base = LOCAL_EMBED if args.local else REMOTE_EMBED

    print(f"chat  -> {chat_base}")
    print(f"embed -> {embed_base}")
    print("firing both concurrently...\n")

    timeout = aiohttp.ClientTimeout(total=600)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        wall_start = time.monotonic()
        (chat_t, chat_out), (embed_t, embed_out) = await asyncio.gather(
            ask_chat(session, chat_base),
            ask_embed(session, embed_base),
        )
        wall = time.monotonic() - wall_start

    print(f"[chat  {chat_t:6.2f}s] {chat_out}")
    print(f"[embed {embed_t:6.2f}s] {embed_out}")
    print(f"\nwall time: {wall:.2f}s (max of the two; both ran in parallel)")


if __name__ == "__main__":
    asyncio.run(main())
