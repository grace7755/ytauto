"""Tiny sqlite JSON cache shared across projects (search, metadata, gemini answers)."""
from __future__ import annotations

import hashlib
import json
import sqlite3
import threading
from pathlib import Path


class Cache:
    def __init__(self, directory: Path):
        directory.mkdir(parents=True, exist_ok=True)
        self.dir = directory
        self._db = sqlite3.connect(directory / "cache.sqlite", check_same_thread=False)
        self._db.execute("create table if not exists kv (k text primary key, v text)")
        self._lock = threading.Lock()

    @staticmethod
    def key(*parts) -> str:
        return hashlib.sha1(json.dumps(parts, sort_keys=True, default=str).encode()).hexdigest()

    def get(self, ns: str, *parts):
        with self._lock:
            row = self._db.execute("select v from kv where k=?", (f"{ns}:{self.key(*parts)}",)).fetchone()
        return json.loads(row[0]) if row else None

    def set(self, ns: str, value, *parts):
        with self._lock:
            self._db.execute("insert or replace into kv values (?,?)",
                             (f"{ns}:{self.key(*parts)}", json.dumps(value, default=str)))
            self._db.commit()
