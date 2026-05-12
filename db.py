"""db.py — SQLite persistence for streams and history."""

import json
import sqlite3
import threading
from contextlib import contextmanager
from datetime import datetime, timezone, timedelta
from pathlib import Path

DB_PATH  = Path(__file__).parent / "data.db"
_wr_lock = threading.Lock()


@contextmanager
def _db(write: bool = False):
    conn = sqlite3.connect(str(DB_PATH), timeout=30, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    try:
        if write:
            with _wr_lock:
                yield conn
                conn.commit()
        else:
            yield conn
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def init_db():
    conn = sqlite3.connect(str(DB_PATH), timeout=30)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS streams (
            id         TEXT PRIMARY KEY,
            name       TEXT NOT NULL,
            url        TEXT NOT NULL,
            zone       TEXT,
            active     INTEGER NOT NULL DEFAULT 1,
            created_at TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS history (
            id         INTEGER PRIMARY KEY AUTOINCREMENT,
            stream_id  TEXT NOT NULL,
            ts         TEXT NOT NULL,
            total      INTEGER NOT NULL DEFAULT 0,
            car        INTEGER NOT NULL DEFAULT 0,
            motorcycle INTEGER NOT NULL DEFAULT 0,
            bus        INTEGER NOT NULL DEFAULT 0,
            truck      INTEGER NOT NULL DEFAULT 0
        );
        CREATE INDEX IF NOT EXISTS idx_hist ON history (stream_id, ts);
    """)
    conn.close()


def _to_stream(row) -> dict:
    d = dict(row)
    d["detection_zone"] = json.loads(d.pop("zone")) if d.get("zone") else None
    return d


# ── Stream CRUD ───────────────────────────────────────────────────────────────

def list_streams() -> list[dict]:
    with _db() as conn:
        rows = conn.execute("SELECT * FROM streams ORDER BY created_at").fetchall()
    return [_to_stream(r) for r in rows]


def get_stream(sid: str) -> dict | None:
    with _db() as conn:
        row = conn.execute("SELECT * FROM streams WHERE id=?", (sid,)).fetchone()
    return _to_stream(row) if row else None


def create_stream(sid: str, name: str, url: str) -> dict:
    now = datetime.now(timezone.utc).isoformat()
    with _db(write=True) as conn:
        conn.execute(
            "INSERT INTO streams (id, name, url, zone, active, created_at) VALUES (?,?,?,NULL,1,?)",
            (sid, name, url, now),
        )
    return get_stream(sid)


def update_stream(sid: str, **kwargs) -> dict | None:
    cols: dict = {}
    for k in ("name", "url", "active"):
        if k in kwargs:
            cols[k] = kwargs[k]
    if "detection_zone" in kwargs:
        z = kwargs["detection_zone"]
        cols["zone"] = json.dumps(z) if z is not None else None
    if not cols:
        return get_stream(sid)
    sets = ", ".join(f"{k}=?" for k in cols)
    with _db(write=True) as conn:
        conn.execute(f"UPDATE streams SET {sets} WHERE id=?", (*cols.values(), sid))
    return get_stream(sid)


def delete_stream(sid: str):
    with _db(write=True) as conn:
        conn.execute("DELETE FROM streams WHERE id=?", (sid,))
        conn.execute("DELETE FROM history WHERE stream_id=?", (sid,))


# ── History ───────────────────────────────────────────────────────────────────

def insert_sample(sid: str, ts: datetime, total: int, counts: dict):
    with _db(write=True) as conn:
        conn.execute(
            "INSERT INTO history (stream_id,ts,total,car,motorcycle,bus,truck)"
            " VALUES (?,?,?,?,?,?,?)",
            (
                sid, ts.isoformat(), total,
                counts.get("car", 0), counts.get("motorcycle", 0),
                counts.get("bus", 0), counts.get("truck", 0),
            ),
        )


def load_history(sid: str, hours: int = 25) -> list[dict]:
    cutoff = (datetime.now(timezone.utc) - timedelta(hours=hours)).isoformat()
    with _db() as conn:
        rows = conn.execute(
            "SELECT ts,total,car,motorcycle,bus,truck FROM history"
            " WHERE stream_id=? AND ts>=? ORDER BY ts",
            (sid, cutoff),
        ).fetchall()
    return [dict(r) for r in rows]


def prune_history(hours: int = 26):
    cutoff = (datetime.now(timezone.utc) - timedelta(hours=hours)).isoformat()
    with _db(write=True) as conn:
        conn.execute("DELETE FROM history WHERE ts<?", (cutoff,))
