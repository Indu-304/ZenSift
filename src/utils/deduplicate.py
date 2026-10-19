"""SQLite ledger of already-sent article URLs."""

from __future__ import annotations

import os
import sqlite3
from collections.abc import Iterable
from contextlib import closing
from datetime import UTC, datetime
from pathlib import Path

DEFAULT_DB_PATH = "data/zensift.db"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS sent_articles (
    url     TEXT PRIMARY KEY,
    sent_at TEXT NOT NULL
)
"""


def db_path_from_env() -> Path:
    return Path(os.getenv("ZENSIFT_DB_PATH", DEFAULT_DB_PATH))


def _connect(db_path: str | Path) -> sqlite3.Connection:
    path = Path(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.execute(_SCHEMA)
    return conn


def get_sent_urls(urls: Iterable[str], db_path: str | Path) -> set[str]:
    """Return the subset of `urls` already recorded as sent."""
    unique = list(dict.fromkeys(urls))
    if not unique:
        return set()
    # Only "?" placeholders are interpolated; values stay parameterized.
    placeholders = ",".join("?" * len(unique))
    with closing(_connect(db_path)) as conn:
        rows = conn.execute(
            f"SELECT url FROM sent_articles WHERE url IN ({placeholders})", unique
        ).fetchall()
    return {row[0] for row in rows}


def mark_sent(urls: Iterable[str], db_path: str | Path) -> None:
    sent_at = datetime.now(UTC).isoformat()
    rows = [(url, sent_at) for url in dict.fromkeys(urls)]
    if not rows:
        return
    with closing(_connect(db_path)) as conn, conn:
        conn.executemany(
            "INSERT OR IGNORE INTO sent_articles (url, sent_at) VALUES (?, ?)", rows
        )
