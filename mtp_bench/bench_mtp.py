"""MTP spec-draft-n-max sweep across context depths.

For each n-max value, launch a standalone llama-server with the MTP GGUF,
then generate 256 tokens at increasing context depths (prompt-cache makes the
long prefix cheap to re-use within one server instance). Record generation
eval tok/s (from response timings) and draft acceptance rate (from the log).

Run AFTER the live chat router is stopped — needs exclusive VRAM.
Stdlib only (no requests dependency).
"""
from __future__ import annotations

import json
import random
import re
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SERVER = ROOT / "llama.cpp_latest" / "llama-server.exe"
MODEL = ROOT / "models" / "Qwen3.6-27B-UD-Q4_K_XL.mtp.gguf"
BENCH = Path(__file__).resolve().parent
PORT = 8090
BASE = f"http://127.0.0.1:{PORT}"

N_MAX_VALUES = [2, 3, 4, 6]
DEPTHS = [2000, 32000, 64000, 96000]   # actual prompt tokens in KV cache
CTX = 102400                            # allocation; covers 96k + 256 gen
GEN = 256
ACCEPT_RE = re.compile(r"draft acceptance rate = ([0-9.]+)")


def _post(path: str, payload: dict, timeout: float) -> dict:
    data = json.dumps(payload).encode()
    req = urllib.request.Request(
        BASE + path, data=data, headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode())


def make_base_text(target_chars: int = 520_000) -> str:
    """Semi-natural varied filler — enough to slice ~100k tokens from."""
    rng = random.Random(1234)
    subjects = ["the system", "a model", "the engineer", "our cluster", "the kernel",
                "this routine", "the scheduler", "a tensor", "the cache", "the pipeline",
                "the operator", "a request", "the daemon", "the buffer", "each shard"]
    verbs = ["allocates", "evaluates", "streams", "compresses", "rebalances", "validates",
             "schedules", "prefetches", "synchronizes", "throttles", "serializes", "rejects"]
    objs = ["the gradient buffers", "incoming token batches", "the speculative drafts",
            "a contiguous memory region", "the attention scores", "the routing table",
            "several pending futures", "the quantized weights", "a ring of descriptors",
            "the verification window"]
    tails = ["under heavy load", "before the next epoch", "when latency spikes",
             "to keep throughput high", "across the NUMA nodes", "without blocking callers",
             "as the context grows", "while preserving accuracy", "during a cold start",
             "once the quota resets"]
    out = []
    n = 0
    while n < target_chars:
        s = (f"{rng.choice(subjects).capitalize()} {rng.choice(verbs)} "
             f"{rng.choice(objs)} {rng.choice(tails)} (step {rng.randint(1, 99999)}). ")
        out.append(s)
        n += len(s)
    return "".join(out)


def wait_health(timeout: float = 240.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(f"{BASE}/health", timeout=5) as r:
                if r.status == 200:
                    return
        except (urllib.error.URLError, OSError):
            pass
        time.sleep(2)
    raise TimeoutError("server did not become healthy")


def tokenize(text: str) -> list[int]:
    return _post("/tokenize", {"content": text, "add_special": False}, timeout=120)["tokens"]


def launch(n_max: int, log_path: Path) -> subprocess.Popen:
    cmd = [
        str(SERVER),
        "--model", str(MODEL),
        "--no-mmproj",
        "--ctx-size", str(CTX),
        "--n-gpu-layers", "999",
        "--flash-attn", "on",
        "--cache-type-k", "q4_0",
        "--cache-type-v", "q4_0",
        "--no-mmap",
        "--parallel", "1",
        "--spec-type", "draft-mtp",
        "--spec-draft-n-max", str(n_max),
        "--spec-draft-p-min", "0.0",
        "--host", "127.0.0.1",
        "--port", str(PORT),
        "--log-file", str(log_path),
        "--log-timestamps",
    ]
    return subprocess.Popen(cmd, cwd=str(ROOT))


def latest_accept(log_path: Path, since: int) -> tuple[float | None, int]:
    if not log_path.exists():
        return None, since
    data = log_path.read_bytes()
    chunk = data[since:].decode("utf-8", errors="ignore")
    matches = ACCEPT_RE.findall(chunk)
    rate = float(matches[-1]) if matches else None
    return rate, len(data)


def main() -> None:
    base_text = make_base_text()
    results = []
    for n_max in N_MAX_VALUES:
        log_path = BENCH / f"server-nmax{n_max}.log"
        if log_path.exists():
            log_path.unlink()
        print(f"\n=== launching n-max={n_max} ===", flush=True)
        proc = launch(n_max, log_path)
        try:
            wait_health()
            tokens = tokenize(base_text)
            print(f"  base tokenized: {len(tokens)} tokens", flush=True)
            offset = log_path.stat().st_size if log_path.exists() else 0
            for depth in DEPTHS:
                if depth > len(tokens):
                    continue
                prompt = tokens[:depth]
                t0 = time.monotonic()
                resp = _post("/completion", {
                    "prompt": prompt,
                    "n_predict": GEN,
                    "cache_prompt": True,
                    "temperature": 0.6, "top_p": 0.95, "top_k": 20,
                }, timeout=1800)
                wall = time.monotonic() - t0
                tm = resp.get("timings", {})
                time.sleep(0.5)  # let log flush
                accept, offset = latest_accept(log_path, offset)
                row = {
                    "n_max": n_max,
                    "depth": depth,
                    "prompt_n": tm.get("prompt_n"),
                    "predicted_n": tm.get("predicted_n"),
                    "gen_tok_s": round(tm.get("predicted_per_second", 0), 2),
                    "prompt_tok_s": round(tm.get("prompt_per_second", 0), 2),
                    "accept_rate": accept,
                    "wall_s": round(wall, 1),
                }
                results.append(row)
                print(f"  depth={depth:>6} gen={row['gen_tok_s']:>7} tok/s "
                      f"accept={accept} (wall {row['wall_s']}s)", flush=True)
        except Exception as exc:  # noqa: BLE001
            print(f"  ERROR at n-max={n_max}: {exc}", flush=True)
        finally:
            proc.terminate()
            try:
                proc.wait(timeout=20)
            except subprocess.TimeoutExpired:
                proc.kill()
            time.sleep(8)  # let VRAM free before next launch

    (BENCH / "results.json").write_text(json.dumps(results, indent=2))

    print("\n\n=== RESULTS (gen tok/s | accept%) ===", flush=True)
    print("| n-max | " + " | ".join(f"{d//1000}k" for d in DEPTHS) + " |")
    print("|" + "---|" * (len(DEPTHS) + 1))
    for n_max in N_MAX_VALUES:
        cells = []
        for depth in DEPTHS:
            row = next((x for x in results if x["n_max"] == n_max and x["depth"] == depth), None)
            if row:
                acc = f"{row['accept_rate']*100:.0f}%" if row["accept_rate"] is not None else "?"
                cells.append(f"{row['gen_tok_s']:.1f} \\| {acc}")
            else:
                cells.append("-")
        print(f"| {n_max} | " + " | ".join(cells) + " |")
    print("\nSaved raw rows to mtp_bench/results.json", flush=True)


if __name__ == "__main__":
    sys.exit(main())
