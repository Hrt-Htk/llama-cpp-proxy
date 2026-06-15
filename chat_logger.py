"""Chat logging and SSE chunk parsing.

Extracted from proxy.py — code copied verbatim.
"""
from __future__ import annotations

import asyncio
import json

from pathlib import Path

from log_paths import (
    DATE_FMT,
    current_week_dir,
    fmt_ts_full,
    fmt_ts_short,
    local_now,
)


RAW_BODY_CAP = 1024 * 1024


class ChatLogger:
    """Rotating chat logger — one file per day, bucketed by ISO week folder.

    Reopens when the local date rolls over (which also moves into a new week
    folder when needed). Uses local Europe/Zurich timestamps with offset.
    """

    def __init__(self, log_dir: Path) -> None:
        self.log_dir = log_dir
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self._date: str | None = None
        self._fh = None
        self._raw_fh = None
        self._lock = asyncio.Lock()
        self._open_for_today()

    def _open_for_today(self) -> None:
        date = local_now().strftime(DATE_FMT)
        if self._date == date and self._fh is not None:
            return
        if self._fh is not None:
            self._fh.close()
        if self._raw_fh is not None:
            self._raw_fh.close()
        week_dir = current_week_dir(self.log_dir)
        self.log_file = week_dir / f"chat-{date}.log"
        self.raw_file = week_dir / f"chat-{date}.raw.jsonl"
        self._fh = open(self.log_file, "a", encoding="utf-8")
        self._raw_fh = open(self.raw_file, "a", encoding="utf-8")
        self._date = date

    async def log_request(self, method: str, path: str, body: bytes | None, req_id: str) -> None:
        async with self._lock:
            self._open_for_today()
            ts = fmt_ts_full()
            self._fh.write(f"=== [{ts}] [req={req_id}] {method} {path} ===\n")
            if body and path.rstrip("/") == "/v1/chat/completions":
                self._write_latest_user_turn(body)
            self._fh.flush()
            self._write_raw(ts, method, path, body, req_id)

    def _write_raw(self, ts: str, method: str, path: str, body: bytes | None, req_id: str) -> None:
        record: dict[str, object] = {"ts": ts, "req_id": req_id, "method": method, "path": path}
        if body is None:
            record["body"] = None
        elif len(body) > RAW_BODY_CAP:
            record["body"] = None
            record["body_truncated"] = body[:RAW_BODY_CAP].decode("utf-8", errors="replace")
            record["original_size"] = len(body)
        else:
            try:
                record["body"] = json.loads(body)
            except (ValueError, TypeError):
                record["body_raw"] = body.decode("utf-8", errors="replace")
        self._raw_fh.write(json.dumps(record, ensure_ascii=False) + "\n")
        self._raw_fh.flush()

    def _write_latest_user_turn(self, body: bytes) -> None:
        try:
            payload = json.loads(body)
        except (ValueError, TypeError):
            return
        messages = payload.get("messages") or []
        if not messages:
            return
        last = messages[-1]
        if last.get("role") != "user":
            return
        text = _stringify_message_content(last.get("content"))
        if text:
            self._fh.write(f"  [user] {text}\n")

    async def log_response(self, data: str, is_done: bool) -> None:
        async with self._lock:
            self._open_for_today()
            ts = fmt_ts_short()
            if is_done:
                self._fh.write(f"  [{ts}] [DONE]\n")
            else:
                self._fh.write(f"  [{ts}] {data}\n")
            self._fh.flush()

    def close(self) -> None:
        if self._fh is not None:
            self._fh.close()
            self._fh = None
        if self._raw_fh is not None:
            self._raw_fh.close()
            self._raw_fh = None


def _stringify_message_content(content: object) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content.replace("\n", " ").strip()
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if not isinstance(item, dict):
                continue
            ctype = item.get("type")
            if ctype == "text" and item.get("text"):
                parts.append(str(item["text"]).replace("\n", " ").strip())
            elif ctype in ("image_url", "image"):
                parts.append("[image]")
            elif ctype == "input_audio":
                parts.append("[audio]")
        return " ".join(p for p in parts if p)
    return str(content)


class SSEChunkLogger:
    # Markers that can start a tool-call XML block
    _START_MARKERS = ("<tool_call>", "<function=")
    # End markers keyed by kind
    _END_MARKERS = {"tool_call": "</tool_call>", "function": "</function>"}

    def __init__(self, wrapped, chat_logger: ChatLogger) -> None:
        self._wrapped = wrapped
        self._chat_logger = chat_logger
        self._buffer = b""
        # --- existing logging state ---
        self._current_kind: str | None = None
        self._current_text = ""
        self._tool_calls: dict[int, dict[str, str]] = {}
        # --- rescue state machine ---
        self._rescue_capturing: bool = False
        self._rescue_buf: str = ""
        self._rescue_kind: str | None = None  # "tool_call" or "function"
        self._reasoning_holdback: str = ""
        self._rescued_any: bool = False
        self._rescue_index: int = 0
        # cache upstream chunk id for synthesised events
        self._last_chunk_id: str = "rescued"

    async def _flush_text(self) -> None:
        if self._current_kind and self._current_text:
            await self._chat_logger.log_response(
                f"[{self._current_kind}] {self._current_text.strip()}", False
            )
        self._current_kind = None
        self._current_text = ""

    async def _flush_tool_calls(self) -> None:
        if not self._tool_calls:
            return
        for idx in sorted(self._tool_calls):
            tc = self._tool_calls[idx]
            name = tc.get("name") or "?"
            args = tc.get("arguments") or ""
            await self._chat_logger.log_response(f"[tool_call] {name}({args})", False)
        self._tool_calls = {}

    async def _flush_all(self) -> None:
        await self._flush_text()
        await self._flush_tool_calls()

    # ---- rescue helpers ----

    @staticmethod
    def _split_safe_prefix(text: str, markers: tuple[str, ...]) -> tuple[str, str]:
        """Return (emit, holdback) where holdback is the longest suffix of *text*
        that is a proper prefix of any *marker*."""
        for length in range(len(text), 0, -1):
            suffix = text[len(text) - length:]
            for marker in markers:
                if len(suffix) < len(marker) and marker.startswith(suffix):
                    return text[: len(text) - length], suffix
        return text, ""

    @staticmethod
    def _parse_tool_call_xml(block: str) -> dict | None:
        """Parse a tool-call XML block. Returns {name, arguments} or None."""
        # Find <function=NAME>
        import re as _re
        m = _re.search(r"<function=([^>\s]+)", block)
        if not m:
            return None
        name: str = m.group(1)
        # Find all <parameter=KEY>VALUE</parameter>
        args: dict[str, object] = {}
        for pm in _re.finditer(r"<parameter=(\S+)>(.*?)</parameter>", block, _re.DOTALL):
            key = pm.group(1)
            value = pm.group(2).strip()
            # Coerce: try JSON parse
            try:
                value = json.loads(value)
            except (json.JSONDecodeError, ValueError):
                pass
            args[key] = value
        return {"name": name, "arguments": args}

    def _build_synthesized_event(self, parsed: dict) -> bytes:
        """Build a synthesised tool_calls SSE event from a parsed tool call."""
        import secrets as _secrets
        event: dict = {
            "id": self._last_chunk_id,
            "object": "chat.completion.chunk",
            "choices": [
                {
                    "index": 0,
                    "delta": {
                        "tool_calls": [
                            {
                                "index": self._rescue_index,
                                "id": f"call_{_secrets.token_hex(4)}",
                                "type": "function",
                                "function": {
                                    "name": parsed["name"],
                                    "arguments": json.dumps(parsed["arguments"]),
                                },
                            }
                        ]
                    },
                    "finish_reason": None,
                }
            ],
        }
        self._rescue_index += 1
        body = json.dumps(event, ensure_ascii=False)
        return f"data: {body}\r\n\r\n".encode()

    # ---- main read loop ----

    async def readany(self) -> bytes:
        # Loop so we only ever return b"" at true EOF — the consumer treats an
        # empty return as end-of-stream. A single upstream chunk may not complete
        # an SSE event, in which case _readany_once returns None and we read more.
        while True:
            out = await self._readany_once()
            if out is not None:
                return out

    async def _readany_once(self) -> bytes | None:
        data = await self._wrapped.content.readany()
        if not data:
            await self._flush_all()
            if self._buffer:
                leftover = self._buffer
                self._buffer = b""
                return leftover
            return b""
        self._buffer += data
        outbound: list[bytes] = []
        while True:
            crlf_idx = self._buffer.find(b"\r\n\r\n")
            lf_idx = self._buffer.find(b"\n\n")
            if crlf_idx == -1 and lf_idx == -1:
                break
            if crlf_idx != -1 and (lf_idx == -1 or crlf_idx <= lf_idx):
                idx, sep_len = crlf_idx, 4
            else:
                idx, sep_len = lf_idx, 2
            raw_event = self._buffer[: idx + sep_len]
            self._buffer = self._buffer[idx + sep_len:]
            event_text = raw_event.decode("utf-8", errors="replace").strip()
            if not event_text:
                outbound.append(raw_event)
                continue
            # Extract payload line
            payload = ""
            for line in event_text.splitlines():
                if line.startswith("data:"):
                    payload = line[5:].strip()
            # Non-data lines, comments, [DONE] → pass through
            if not payload:
                outbound.append(raw_event)
                continue
            if payload == "[DONE]":
                await self._flush_all()
                await self._chat_logger.log_response("[DONE]", True)
                outbound.append(raw_event)
                continue
            try:
                obj = json.loads(payload)
            except json.JSONDecodeError:
                outbound.append(raw_event)
                continue
            # --- (a) existing logging on original delta ---
            choices = obj.get("choices") or []
            if choices:
                delta = choices[0].get("delta") or {}
                reasoning = delta.get("reasoning_content")
                content = delta.get("content")
                tool_calls = delta.get("tool_calls")
                if reasoning:
                    if self._current_kind != "thinking":
                        await self._flush_all()
                        self._current_kind = "thinking"
                    self._current_text += reasoning
                if content:
                    if self._current_kind != "content":
                        await self._flush_all()
                        self._current_kind = "content"
                    self._current_text += content
                if tool_calls:
                    await self._flush_text()
                    for tc in tool_calls:
                        i = tc.get("index", 0)
                        slot = self._tool_calls.setdefault(i, {"name": "", "arguments": ""})
                        fn = tc.get("function") or {}
                        if fn.get("name"):
                            slot["name"] = fn["name"]
                        if fn.get("arguments"):
                            slot["arguments"] += fn["arguments"]
                # Track chunk id for synthesised events
                cid = obj.get("id")
                if cid:
                    self._last_chunk_id = cid
            # --- (b) build outbound bytes with rescue transform ---
            outbound_event = self._transform_event(obj)
            outbound.append(outbound_event)
        return b"".join(outbound) if outbound else None

    def _transform_event(self, obj: dict) -> bytes:
        """Transform a single parsed event dict into outbound SSE bytes,
        applying the rescue state machine."""
        choices = obj.get("choices") or []
        if not choices:
            # No choices — pass through
            body = json.dumps(obj, ensure_ascii=False)
            return f"data: {body}\r\n\r\n".encode()

        delta = choices[0].get("delta") or {}
        reasoning = delta.get("reasoning_content")

        # If no reasoning_content, just rewrite finish_reason if needed
        if not reasoning:
            if self._rescued_any and choices[0].get("finish_reason") == "stop":
                choices[0]["finish_reason"] = "tool_calls"
            body = json.dumps(obj, ensure_ascii=False)
            return f"data: {body}\r\n\r\n".encode()

        # Run rescue state machine on reasoning_content
        work = self._reasoning_holdback + reasoning
        self._reasoning_holdback = ""
        prose_parts: list[str] = []
        synthesized: list[bytes] = []

        while work:
            if not self._rescue_capturing:
                # Look for earliest start marker
                earliest_pos = len(work)
                earliest_marker: str | None = None
                for marker in self._START_MARKERS:
                    pos = work.find(marker)
                    if pos != -1 and pos < earliest_pos:
                        earliest_pos = pos
                        earliest_marker = marker

                if earliest_marker is None:
                    # No marker found — apply split_safe_prefix
                    emit, holdback = self._split_safe_prefix(work, self._START_MARKERS)
                    prose_parts.append(emit)
                    self._reasoning_holdback = holdback
                    work = ""
                else:
                    # A complete start marker is present, so everything before it
                    # is safe prose — no partial-marker holdback needed here.
                    prose_parts.append(work[:earliest_pos])
                    # Start capturing
                    self._rescue_capturing = True
                    self._rescue_kind = (
                        "tool_call" if earliest_marker == "<tool_call>" else "function"
                    )
                    self._rescue_buf = earliest_marker
                    work = work[earliest_pos + len(earliest_marker):]
            else:
                # Capturing — look for end marker
                end_marker = self._END_MARKERS[self._rescue_kind]
                end_pos = work.find(end_marker)
                if end_pos != -1:
                    self._rescue_buf += work[: end_pos + len(end_marker)]
                    # Parse the block
                    parsed = self._parse_tool_call_xml(self._rescue_buf)
                    if parsed:
                        synthesized.append(self._build_synthesized_event(parsed))
                        self._rescued_any = True
                    self._rescue_capturing = False
                    self._rescue_buf = ""
                    self._rescue_kind = None
                    work = work[end_pos + len(end_marker):]
                else:
                    # End marker not found — keep all of work in buffer
                    self._rescue_buf += work
                    work = ""

        # Build the outbound event
        forwarded_reasoning = "".join(prose_parts)
        if forwarded_reasoning:
            delta["reasoning_content"] = forwarded_reasoning
        else:
            delta.pop("reasoning_content", None)
            # If delta is now empty and has no other keys, we still emit the event
            # (the caller handles skipping if needed)

        # Rewrite finish_reason if rescued
        if self._rescued_any and choices[0].get("finish_reason") == "stop":
            choices[0]["finish_reason"] = "tool_calls"

        body = json.dumps(obj, ensure_ascii=False)
        result = f"data: {body}\r\n\r\n".encode()
        # Append any synthesised events after the reasoning event
        for syn in synthesized:
            result += syn
        return result

    def __getattr__(self, name: str) -> object:
        return getattr(self._wrapped, name)
