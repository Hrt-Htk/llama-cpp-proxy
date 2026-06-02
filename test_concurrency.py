"""Concurrency reproduction harness for the chat proxy.

Tests whether OVERLAPPING requests for DIFFERENT models cause 5xx
(mid-stream model eviction) vs. SEQUENTIAL requests being clean.

Run against an isolated proxy, e.g.:
    .venv\\Scripts\\python.exe proxy.py --proxy-port 8011 --server-port 8012 --no-chat-log
    .venv\\Scripts\\python.exe test_concurrency.py
"""
import asyncio
import time
import aiohttp

BASE = "http://[::1]:8011"
KEY = "ZXY0UVZt8lbPVj3fSTC4gp0JatpRfOBQqGDAcvaVl3RjmWoq"
MODEL_A = "qwen3.6-27b-q4-32k"
MODEL_B = "qwen3.6-35b-q3-32k"
HEADERS = {"Authorization": f"Bearer {KEY}", "Content-Type": "application/json"}
PROMPT = "List the numbers from 1 to 200, one per line, nothing else."


async def chat(session: aiohttp.ClientSession, model: str, tag: str) -> dict:
    body = {
        "model": model,
        "messages": [{"role": "user", "content": PROMPT}],
        "max_tokens": 512,
        "stream": False,
        "temperature": 0,
    }
    t0 = time.monotonic()
    try:
        async with session.post(
            f"{BASE}/v1/chat/completions", headers=HEADERS, json=body,
            timeout=aiohttp.ClientTimeout(total=180),
        ) as r:
            text = await r.text()
            dt = time.monotonic() - t0
            return {"tag": tag, "model": model, "status": r.status,
                    "sec": round(dt, 1), "err": "" if r.status == 200 else text[:160]}
    except Exception as exc:
        dt = time.monotonic() - t0
        return {"tag": tag, "model": model, "status": "EXC",
                "sec": round(dt, 1), "err": f"{type(exc).__name__}: {exc}"}


def show(title, results):
    print(f"\n=== {title} ===")
    for r in results:
        line = f"  [{r['tag']}] {r['model']:<22} -> {r['status']}  ({r['sec']}s)"
        if r["err"]:
            line += f"\n        {r['err']}"
        print(line)


async def main():
    async with aiohttp.ClientSession() as s:
        # Warm up: load A once so the sequential timings aren't dominated by
        # the very first cold load.
        print("warming up (loading model A)...")
        await chat(s, MODEL_A, "warmup")

        # --- SEQUENTIAL control: switch models with no overlap ---
        seq = []
        seq.append(await chat(s, MODEL_B, "seq-1"))   # switch A->B
        seq.append(await chat(s, MODEL_A, "seq-2"))   # switch B->A
        seq.append(await chat(s, MODEL_B, "seq-3"))   # switch A->B
        show("SEQUENTIAL different-model (expect all 200)", seq)

        # --- CONCURRENT test: fire A and B at the same instant ---
        for i in range(2):
            con = await asyncio.gather(
                chat(s, MODEL_A, f"con{i}-A"),
                chat(s, MODEL_B, f"con{i}-B"),
            )
            show(f"CONCURRENT different-model round {i} (hypothesis: a 5xx)", con)

        # --- CONCURRENT same-model (should be clean: router queues) ---
        same = await asyncio.gather(
            chat(s, MODEL_A, "same-A1"),
            chat(s, MODEL_A, "same-A2"),
        )
        show("CONCURRENT same-model (expect all 200)", same)


if __name__ == "__main__":
    asyncio.run(main())
