"""Shared log-path & timestamp utilities.

All log files land under ``logs/<YYYY-WNN>/`` where the folder name is the
ISO year + ISO week (e.g. ``2026-W20``) of the current Europe/Zurich
date. When a brand-new week folder is created, week folders older than the
two most recent ones are removed wholesale — so on-disk retention is always
"current week + previous week", no more.

Timestamps everywhere are local (Europe/Zurich) with a ``+0200``/``+0100``
offset suffix so cross-stream alignment is unambiguous across DST.
"""
from __future__ import annotations

import logging
import shutil
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

LOCAL_TZ = ZoneInfo("Europe/Zurich")
WEEK_DIR_FMT = "%G-W%V"                  # ISO year + ISO week, e.g. 2026-W20
TS_FULL_FMT = "%Y-%m-%d %H:%M:%S.{ms}%z"  # with millis, used for proxy line prefix
TS_SHORT_FMT = "%H:%M:%S.{ms}%z"          # hh:mm:ss.mmm+offset for SSE chunks
DATE_FMT = "%Y-%m-%d"


def local_now() -> datetime:
    return datetime.now(LOCAL_TZ)


def fmt_ts_full(dt: datetime | None = None) -> str:
    dt = dt or local_now()
    return dt.strftime(TS_FULL_FMT.format(ms=f"{dt.microsecond // 1000:03d}"))


def fmt_ts_short(dt: datetime | None = None) -> str:
    dt = dt or local_now()
    return dt.strftime(TS_SHORT_FMT.format(ms=f"{dt.microsecond // 1000:03d}"))


def current_week_dir(root: Path) -> Path:
    """Return ``logs/<YYYY-WNN>/`` for now, creating it on first use of a new week.

    On folder creation the helper also prunes any week folder older than the
    two most recent so disk usage stays bounded.
    """
    root.mkdir(parents=True, exist_ok=True)
    week = local_now().strftime(WEEK_DIR_FMT)
    target = root / week
    if not target.exists():
        target.mkdir(parents=True, exist_ok=True)
        _prune_old_weeks(root, keep_latest=2)
    return target


def _prune_old_weeks(root: Path, *, keep_latest: int) -> None:
    weeks = sorted(
        (p for p in root.iterdir() if p.is_dir() and _is_week_dir(p.name)),
        key=lambda p: p.name,
    )
    for old in weeks[:-keep_latest]:
        try:
            shutil.rmtree(old)
            logging.info("Pruned old log week folder: %s", old.name)
        except OSError as e:
            logging.warning("Could not prune %s: %s", old, e)


def _is_week_dir(name: str) -> bool:
    # 2026-W20 — 4-digit year, "-W", 2-digit week
    if len(name) != 8 or name[4:6] != "-W":
        return False
    return name[:4].isdigit() and name[6:].isdigit()


class LocalTzFormatter(logging.Formatter):
    """Logging formatter that prints local Europe/Zurich timestamps with offset."""

    def formatTime(self, record, datefmt=None):
        dt = datetime.fromtimestamp(record.created, LOCAL_TZ)
        if datefmt:
            return dt.strftime(datefmt)
        return f"{dt.strftime('%Y-%m-%d %H:%M:%S')}.{int(record.msecs):03d}{dt.strftime('%z')}"
