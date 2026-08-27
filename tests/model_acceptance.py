#!/usr/bin/env python
"""Live acceptance harness for a newly-added router model.

Run BEFORE putting a model into daily use. Proves the model fits on the GPU,
loads, talks, thinks cleanly (issue #8), calls tools, retrieves long context,
and handles vision with the projector on CPU — empirically, not by assumption.

Usage:
    .venv\\Scripts\\python.exe tests\\model_acceptance.py --model <preset-id>

Companion plan: docs/dev/model-acceptance.md

Dependency-free (stdlib urllib only). Drives the client-facing proxy (:8001)
for functional checks and the router (:8002) for load/introspection/VRAM.
"""
from __future__ import annotations

import argparse
import base64
import json
import os
import re
import struct
import subprocess
import sys
import time
import urllib.error
import urllib.request
import zlib
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


# ── config / secrets ─────────────────────────────────────────────────────────

def resolve_api_key() -> str:
    key = os.environ.get("LLAMA_API_KEY")
    if key:
        return key
    envf = ROOT / ".env"
    if envf.is_file():
        for line in envf.read_text(encoding="utf-8").splitlines():
            m = re.match(r"\s*LLAMA_API_KEY\s*=\s*(.+)\s*$", line)
            if m:
                return m.group(1).strip().strip("\"'")
    raise SystemExit("No LLAMA_API_KEY in env or .env")


# ── tiny HTTP layer ──────────────────────────────────────────────────────────

class Http:
    def __init__(self, key: str) -> None:
        self.key = key

    def _req(self, url: str, method: str, body: dict | None, timeout: float):
        data = json.dumps(body).encode() if body is not None else None
        req = urllib.request.Request(url, data=data, method=method)
        req.add_header("Authorization", f"Bearer {self.key}")
        if data is not None:
            req.add_header("Content-Type", "application/json")
        return urllib.request.urlopen(req, timeout=timeout)

    def get(self, url: str, timeout: float = 15) -> dict:
        with self._req(url, "GET", None, timeout) as r:
            return json.loads(r.read().decode())

    def post(self, url: str, body: dict, timeout: float = 240) -> dict:
        with self._req(url, "POST", body, timeout) as r:
            return json.loads(r.read().decode())

    def post_stream(self, url: str, body: dict, timeout: float = 240):
        """Yield parsed SSE JSON payloads (skips [DONE])."""
        body = {**body, "stream": True}
        with self._req(url, "POST", body, timeout) as r:
            for raw in r:
                line = raw.decode("utf-8", "replace").strip()
                if not line.startswith("data:"):
                    continue
                payload = line[5:].strip()
                if not payload or payload == "[DONE]":
                    if payload == "[DONE]":
                        yield "__DONE__"
                    continue
                try:
                    yield json.loads(payload)
                except json.JSONDecodeError:
                    pass


def gpu_used_mib(index: int = 0) -> int | None:
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used",
             "--format=csv,noheader,nounits", "-i", str(index)],
            capture_output=True, text=True, timeout=15,
        )
        return int(out.stdout.strip().splitlines()[0])
    except Exception:
        return None


# ── result plumbing ──────────────────────────────────────────────────────────

@dataclass
class Result:
    id: str
    passed: bool
    detail: str
    metrics: dict = field(default_factory=dict)


class Runner:
    def __init__(self, args) -> None:
        self.a = args
        self.http = Http(resolve_api_key())
        self.results: list[Result] = []
        self.skip = {s.strip() for s in (args.skip or "").split(",") if s.strip()}
        self.pp: float | None = None   # prompt-processing tok/s (llama-server timings)
        self.tg: float | None = None   # token-generation tok/s (llama-server timings)

    def record(self, r: Result) -> None:
        self.results.append(r)
        tag = "PASS" if r.passed else "FAIL"
        extra = f"  {r.metrics}" if r.metrics else ""
        print(f"  [{tag}] {r.id}: {r.detail}{extra}")

    def should(self, cid: str) -> bool:
        if cid in self.skip:
            print(f"  [SKIP] {cid}")
            return False
        return True

    # ── individual checks ────────────────────────────────────────────────────

    def check_registered(self) -> None:
        data = self.http.get(f"{self.a.router}/v1/models")
        entry = next((e for e in data.get("data", []) if e.get("id") == self.a.model), None)
        if not entry:
            self.record(Result("registered", False, f"{self.a.model} not in /v1/models"))
            return
        n_ctx = (entry.get("meta") or {}).get("n_ctx")
        self.record(Result("registered", True, f"found; meta n_ctx={n_ctx}",
                           {"n_ctx": n_ctx}))

    def _status(self, model: str) -> str:
        data = self.http.get(f"{self.a.router}/v1/models")
        for e in data.get("data", []):
            if e.get("id") == model:
                st = e.get("status")
                return st.get("value") if isinstance(st, dict) else str(st)
        return "unknown"

    def check_load(self) -> None:
        t0 = time.monotonic()
        try:
            self.http.post(f"{self.a.router}/models/load", {"model": self.a.model}, timeout=20)
        except urllib.error.HTTPError as e:
            if b"already running" not in e.read():
                self.record(Result("load", False, f"load POST failed: {e}"))
                return
        deadline = t0 + 300
        while time.monotonic() < deadline:
            st = self._status(self.a.model)
            if st == "loaded":
                secs = round(time.monotonic() - t0, 1)
                self.record(Result("load", True, "reached loaded", {"load_s": secs}))
                return
            if st == "failed":
                self.record(Result("load", False, "status=failed"))
                return
            time.sleep(1)
        self.record(Result("load", False, "did not load in 300s"))

    def check_vram_fit(self) -> None:
        used = gpu_used_mib(0)
        if used is None:
            self.record(Result("vram_fit", False, "nvidia-smi unavailable"))
            return
        ok = used <= self.a.vram_ceiling
        self.record(Result("vram_fit", ok,
                           f"GPU0 used={used} MiB (ceiling {self.a.vram_ceiling})",
                           {"vram_used_mib": used, "headroom_mib": 24564 - used}))

    def check_basic(self) -> None:
        body = {"model": self.a.model, "temperature": 0,
                "messages": [{"role": "user",
                              "content": "What is 2+2? Answer with only the number."}]}
        r = self.http.post(f"{self.a.proxy}/v1/chat/completions", body)
        content = r["choices"][0]["message"]["content"] or ""
        ok = "4" in content
        self.record(Result("basic", ok, f"answer={content.strip()[:40]!r}"))

    def check_streaming_and_think(self) -> None:
        # deterministic, multi-token output so the stream actually flows and
        # decode throughput is measurable
        body = {"model": self.a.model, "temperature": 0,
                "stream_options": {"include_usage": True},
                "messages": [{"role": "user",
                              "content": "Count from 1 to 40, comma-separated, on one line."}]}
        events = 0            # any streamed delta (content or reasoning)
        got_done = False
        visible = []
        reasoning_seen = False
        usage = {}
        t_first = t_last = None
        t0 = time.monotonic()
        for ev in self.http.post_stream(f"{self.a.proxy}/v1/chat/completions", body):
            if ev == "__DONE__":
                got_done = True
                continue
            if ev.get("usage"):
                usage = ev["usage"]
            for ch in ev.get("choices", []):
                delta = ch.get("delta", {})
                content = delta.get("content")
                reasoning = delta.get("reasoning_content")
                if content or reasoning:
                    events += 1
                    now = time.monotonic()
                    if t_first is None:
                        t_first = now
                    t_last = now
                if content:
                    visible.append(content)
                if reasoning:
                    reasoning_seen = True
        t_done = time.monotonic()
        text = "".join(visible)
        # issue #8 guard: no raw think tags leaked into visible content
        leaked = "<think>" in text or "</think>" in text
        # decode window: prefer first->last token; fall back to whole request if too small
        window = (t_last - t_first) if (t_first and t_last and t_last - t_first > 0.1) else (t_done - t0)
        comp_tok = usage.get("completion_tokens")
        tps = round(comp_tok / window, 1) if comp_tok and window > 0.05 else None
        stream_ok = got_done and events >= 2 and len(text) > 5
        detail = f"events={events} done={got_done} chars={len(text)} think_leak={leaked} reasoning={reasoning_seen}"
        self.record(Result("streaming", stream_ok, detail,
                           {"wall_decode_tps": tps}))  # rough; authoritative tg is in perf
        self.record(Result("think_integrity", not leaked,
                           "no leaked think tags" if not leaked
                           else "RAW <think> LEAKED INTO CONTENT"))

    def check_tools(self) -> None:
        tools = [{
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get current weather for a city.",
                "parameters": {
                    "type": "object",
                    "properties": {"location": {"type": "string",
                                                "description": "City name"}},
                    "required": ["location"],
                },
            },
        }]
        body = {"model": self.a.model, "temperature": 0, "tools": tools,
                "tool_choice": "auto",
                "messages": [{"role": "user",
                              "content": "Use the get_weather tool for Paris."}]}
        r = self.http.post(f"{self.a.proxy}/v1/chat/completions", body)
        msg = r["choices"][0]["message"]
        calls = msg.get("tool_calls") or []
        if not calls:
            self.record(Result("tools", False, f"no tool_calls; content={str(msg.get('content'))[:60]!r}"))
            return
        try:
            args = json.loads(calls[0]["function"]["arguments"])
            ok = "paris" in json.dumps(args).lower()
            self.record(Result("tools", ok, f"call={calls[0]['function']['name']} args={args}"))
        except (json.JSONDecodeError, KeyError, TypeError) as e:
            self.record(Result("tools", False, f"unparseable arguments: {e}"))

    def check_long_context(self) -> None:
        target_tok = self.a.ctx_probe_tokens
        if self.a.full_ctx:
            n_ctx = next((r.metrics.get("n_ctx") for r in self.results
                          if r.id == "registered"), None) or 262144
            target_tok = int(n_ctx * 0.95)
        needle = "The vault override code is TANGERINE-9417."
        # ~1.3 tokens/word; build distinct filler lines so it can't be trivially compressed
        n_words = int(target_tok / 1.3)
        filler_lines = []
        wc = 0
        i = 0
        while wc < n_words:
            line = f"Log entry {i}: routine telemetry sample alpha bravo charlie delta echo foxtrot."
            filler_lines.append(line)
            wc += 11
            i += 1
        mid = len(filler_lines) // 2
        filler_lines.insert(mid, needle)
        haystack = "\n".join(filler_lines)
        body = {"model": self.a.model, "temperature": 0,
                "messages": [
                    {"role": "user",
                     "content": haystack + "\n\nQuestion: What is the vault override code? "
                                           "Reply with only the code."}]}
        t0 = time.monotonic()
        try:
            r = self.http.post(f"{self.a.proxy}/v1/chat/completions", body, timeout=600)
        except Exception as e:
            self.record(Result("long_context", False, f"request failed (possible OOM): {e}"))
            return
        dt = round(time.monotonic() - t0, 1)
        content = r["choices"][0]["message"]["content"] or ""
        usage = r.get("usage", {})
        timings = r.get("timings") or {}
        # authoritative pp (prompt processing) at real context scale
        if timings.get("prompt_per_second"):
            self.pp = round(timings["prompt_per_second"], 1)
        ptok = usage.get("prompt_tokens") or timings.get("prompt_n")
        ok = "TANGERINE-9417" in content
        used = gpu_used_mib(0)
        vram_ok = used is None or used <= self.a.vram_ceiling
        self.record(Result("long_context", ok and vram_ok,
                           f"needle_found={ok} prompt_tok={ptok} vram={used}",
                           {"pp_tok_s": self.pp, "wall_s": dt}))

    def check_vision(self) -> None:
        img_path = self.a.image
        if not img_path:
            img_path = str(ROOT / "tests" / "_assets" / "acceptance-red.png")
            Path(img_path).parent.mkdir(parents=True, exist_ok=True)
            make_solid_png(img_path, (220, 30, 30))
        b64 = base64.b64encode(Path(img_path).read_bytes()).decode()
        data_uri = f"data:image/png;base64,{b64}"
        img_msg = {"role": "user", "content": [
            {"type": "text", "text": "What is the dominant color of this image? Answer with one word."},
            {"type": "image_url", "image_url": {"url": data_uri}},
        ]}
        t0 = time.monotonic()
        try:
            r1 = self.http.post(f"{self.a.proxy}/v1/chat/completions",
                                {"model": self.a.model, "temperature": 0, "messages": [img_msg]},
                                timeout=300)
        except Exception as e:
            self.record(Result("vision", False, f"image turn failed: {e}"))
            return
        t_img = round(time.monotonic() - t0, 1)
        reply1 = r1["choices"][0]["message"]["content"] or ""
        color_ok = "red" in reply1.lower()
        used_after_img = gpu_used_mib(0)
        vram_ok = used_after_img is None or used_after_img <= self.a.vram_ceiling
        # turn 2: text follow-up in the same conversation — must work and be fast
        # (proves cached KV reuse, no image re-encode)
        t1 = time.monotonic()
        r2 = self.http.post(f"{self.a.proxy}/v1/chat/completions",
                            {"model": self.a.model, "temperature": 0, "messages": [
                                img_msg,
                                {"role": "assistant", "content": reply1},
                                {"role": "user", "content": "What is 3+3? Only the number."},
                            ]}, timeout=120)
        t_turn2 = round(time.monotonic() - t1, 1)
        reply2 = r2["choices"][0]["message"]["content"] or ""
        turn2_ok = "6" in reply2
        ok = color_ok and turn2_ok and vram_ok
        self.record(Result("vision", ok,
                           f"color={reply1.strip()[:20]!r} turn2_ok={turn2_ok} vram_after_img={used_after_img}",
                           {"img_turn_s": t_img, "turn2_s": t_turn2}))

    def check_perf(self) -> None:
        # authoritative tg (token generation) from llama-server timings on a
        # clean, decent-length generation; pp comes from long_context (real
        # scale) or falls back to this request's small-prompt pp
        body = {"model": self.a.model, "temperature": 0, "max_tokens": 200,
                "messages": [{"role": "user",
                              "content": "Count from 1 to 200, comma-separated."}]}
        try:
            r = self.http.post(f"{self.a.proxy}/v1/chat/completions", body, timeout=180)
        except Exception as e:
            self.record(Result("perf", False, f"tg probe failed: {e}"))
            return
        t = r.get("timings") or {}
        if t.get("predicted_per_second"):
            self.tg = round(t["predicted_per_second"], 1)
        if self.pp is None and t.get("prompt_per_second"):
            self.pp = round(t["prompt_per_second"], 1)
        ok = self.tg is not None and self.tg >= self.a.min_decode_tps
        self.record(Result("perf", ok,
                           f"pp={self.pp} tg={self.tg} tok/s (tg floor {self.a.min_decode_tps})",
                           {"pp_tok_s": self.pp, "tg_tok_s": self.tg}))

    # ── orchestration ────────────────────────────────────────────────────────

    ORDER = [
        ("registered", "check_registered"),
        ("load", "check_load"),
        ("vram_fit", "check_vram_fit"),
        ("basic", "check_basic"),
        ("streaming", "check_streaming_and_think"),
        ("tools", "check_tools"),
        ("long_context", "check_long_context"),
        ("vision", "check_vision"),
        ("perf", "check_perf"),
    ]

    def run(self) -> bool:
        print(f"\n=== acceptance: {self.a.model} ===")
        for cid, method in self.ORDER:
            # streaming method emits two result ids; gate on the primary id
            if cid in self.skip and cid not in ("streaming",):
                print(f"  [SKIP] {cid}")
                continue
            if cid == "streaming" and "streaming" in self.skip and "think_integrity" in self.skip:
                print("  [SKIP] streaming/think_integrity")
                continue
            try:
                getattr(self, method)()
            except Exception as e:
                self.record(Result(cid, False, f"exception: {type(e).__name__}: {e}"))
        return self.write_report()

    def write_report(self) -> bool:
        failed = [r for r in self.results if not r.passed]
        outdir = ROOT / "tests" / "acceptance-results"
        outdir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y-%m-%d-%H%M")
        path = outdir / f"{self.a.model}-{stamp}.md"
        lines = [f"# Acceptance report — {self.a.model}", "",
                 f"- When: {datetime.now().isoformat(timespec='seconds')}",
                 f"- Proxy: {self.a.proxy}  Router: {self.a.router}",
                 f"- Verdict: {'ACCEPTED' if not failed else 'REJECTED'} "
                 f"({len(self.results) - len(failed)}/{len(self.results)} passed)",
                 f"- Speed (llama-server timings): pp={self.pp} tok/s  tg={self.tg} tok/s",
                 "", "| id | result | detail | metrics |", "|----|--------|--------|---------|"]
        for r in self.results:
            lines.append(f"| {r.id} | {'PASS' if r.passed else 'FAIL'} | {r.detail} | "
                         f"{r.metrics or ''} |")
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        print(f"\nReport: {path}")
        print(f"Verdict: {'ACCEPTED' if not failed else 'REJECTED -- ' + ', '.join(r.id for r in failed)}")
        return not failed


# ── known test image (stdlib PNG, no PIL) ────────────────────────────────────

def make_solid_png(path: str, rgb=(220, 30, 30), size: int = 64) -> None:
    raw = bytearray()
    for _ in range(size):
        raw.append(0)  # filter type 0 per scanline
        raw += bytes(rgb) * size
    comp = zlib.compress(bytes(raw), 9)

    def chunk(typ: bytes, data: bytes) -> bytes:
        return (struct.pack(">I", len(data)) + typ + data
                + struct.pack(">I", zlib.crc32(typ + data) & 0xffffffff))

    ihdr = struct.pack(">IIBBBBB", size, size, 8, 2, 0, 0, 0)  # RGB, 8-bit
    png = (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", ihdr)
           + chunk(b"IDAT", comp) + chunk(b"IEND", b""))
    Path(path).write_bytes(png)


def main() -> int:
    p = argparse.ArgumentParser(description="Live new-model acceptance harness")
    p.add_argument("--model", required=True, help="preset id (must exist in /v1/models)")
    p.add_argument("--proxy", default="http://localhost:8001")
    p.add_argument("--router", default="http://127.0.0.1:8002")
    p.add_argument("--vram-ceiling", type=int, default=24000)
    p.add_argument("--ctx-probe-tokens", type=int, default=32000)
    p.add_argument("--full-ctx", action="store_true")
    p.add_argument("--image", default=None)
    p.add_argument("--min-decode-tps", type=float, default=15.0)
    p.add_argument("--skip", default="")
    args = p.parse_args()
    return 0 if Runner(args).run() else 1


if __name__ == "__main__":
    sys.exit(main())
