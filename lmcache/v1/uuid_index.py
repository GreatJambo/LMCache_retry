# SPDX-License-Identifier: Apache-2.0
# Standard
from __future__ import annotations
from typing import Any, Dict, List, Optional, Tuple
import json
import os
import sqlite3
import threading
import time


class UUIDIndex:
    """SQLite-backed index to map a UUID to ordered KV chunk hashes and offsets.

    Schema:
      - uuid_mapping(uuid TEXT PRIMARY KEY,
                     model_name TEXT,
                     fmt TEXT,
                     world_size INTEGER,
                     worker_id INTEGER,
                     total_tokens INTEGER,
                     created_at REAL,
                     request_configs TEXT)
      - uuid_chunks(uuid TEXT,
                    ordinal INTEGER,
                    chunk_hash INTEGER,
                    offset INTEGER,
                    PRIMARY KEY(uuid, ordinal))
    """

    def __init__(self, db_path: str):
        self._db_path = db_path
        os.makedirs(os.path.dirname(self._db_path), exist_ok=True)
        # Allow access across threads from the same process
        self._conn = sqlite3.connect(self._db_path, check_same_thread=False)
        # Use a conservative default journal mode for compatibility.
        # Override via env: LMCACHE_UUID_JOURNAL_MODE=DELETE|WAL|TRUNCATE|MEMORY|OFF
        journal_mode = os.getenv("LMCACHE_UUID_JOURNAL_MODE", "DELETE")
        try:
            self._conn.execute(f"PRAGMA journal_mode={journal_mode};")
        except Exception:
            # Fallback silently
            pass
        self._conn.execute("PRAGMA synchronous=NORMAL;")
        self._lock = threading.Lock()
        self._init_schema()
        # Ensure a clean, fully materialized db file
        try:
            self._conn.execute("VACUUM;")
        except Exception:
            pass

    def _init_schema(self) -> None:
        with self._conn:
            self._conn.execute(
                """
                CREATE TABLE IF NOT EXISTS uuid_mapping (
                    uuid TEXT PRIMARY KEY,
                    model_name TEXT NOT NULL,
                    fmt TEXT NOT NULL,
                    world_size INTEGER NOT NULL,
                    worker_id INTEGER NOT NULL,
                    total_tokens INTEGER NOT NULL,
                    created_at REAL NOT NULL,
                    request_configs TEXT
                )
                """
            )
            self._conn.execute(
                """
                CREATE TABLE IF NOT EXISTS uuid_chunks (
                    uuid TEXT NOT NULL,
                    ordinal INTEGER NOT NULL,
                    chunk_hash INTEGER NOT NULL,
                    offset INTEGER NOT NULL,
                    PRIMARY KEY(uuid, ordinal)
                )
                """
            )

    def upsert(
        self,
        uuid: str,
        *,
        model_name: str,
        fmt: str,
        world_size: int,
        worker_id: int,
        total_tokens: int,
        request_configs: Optional[Dict[str, Any]],
        chunk_hashes: List[int],
        offsets: List[int],
    ) -> None:
        if len(chunk_hashes) != len(offsets):
            raise ValueError("chunk_hashes and offsets must have the same length")

        created_at = time.time()
        req_cfg_json = json.dumps(request_configs) if request_configs else None
        with self._lock:
            with self._conn:
                self._conn.execute(
                    """
                    INSERT INTO uuid_mapping(
                        uuid, model_name, fmt, world_size, worker_id,
                        total_tokens, created_at, request_configs
                    ) VALUES(?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(uuid) DO UPDATE SET
                        model_name=excluded.model_name,
                        fmt=excluded.fmt,
                        world_size=excluded.world_size,
                        worker_id=excluded.worker_id,
                        total_tokens=excluded.total_tokens,
                        created_at=excluded.created_at,
                        request_configs=excluded.request_configs
                    """,
                    (
                        uuid,
                        model_name,
                        fmt,
                        world_size,
                        worker_id,
                        total_tokens,
                        created_at,
                        req_cfg_json,
                    ),
                )

                self._conn.execute("DELETE FROM uuid_chunks WHERE uuid=?", (uuid,))
                self._conn.executemany(
                    (
                        "INSERT INTO uuid_chunks(uuid, ordinal, chunk_hash, offset) "
                        "VALUES(?, ?, ?, ?)"
                    ),
                    [
                        (uuid, ordinal, int(chunk_hash), int(offset))
                        for ordinal, (chunk_hash, offset) in enumerate(
                            zip(chunk_hashes, offsets, strict=False), start=0
                        )
                    ],
                )

    def resolve(
        self, uuid: str
    ) -> Optional[Tuple[Dict[str, Any], List[int], List[int]]]:
        with self._lock:
            cur = self._conn.cursor()
            cur.execute(
                (
                    "SELECT model_name, fmt, world_size, worker_id, total_tokens, "
                    "request_configs FROM uuid_mapping WHERE uuid=?"
                ),
                (uuid,),
            )
            row = cur.fetchone()
            if row is None:
                return None
            model_name, fmt, world_size, worker_id, total_tokens, request_configs = row
            req_cfg = json.loads(request_configs) if request_configs else None

            cur.execute(
                (
                    "SELECT chunk_hash, offset FROM uuid_chunks WHERE uuid=? "
                    "ORDER BY ordinal ASC"
                ),
                (uuid,),
            )
            pairs = cur.fetchall()
            chunk_hashes = [int(p[0]) for p in pairs]
            offsets = [int(p[1]) for p in pairs]

            meta = {
                "model_name": model_name,
                "fmt": fmt,
                "world_size": int(world_size),
                "worker_id": int(worker_id),
                "total_tokens": int(total_tokens),
                "request_configs": req_cfg,
            }
            return meta, chunk_hashes, offsets

    def delete(self, uuid: str) -> None:
        with self._lock:
            with self._conn:
                self._conn.execute("DELETE FROM uuid_chunks WHERE uuid=?", (uuid,))
                self._conn.execute("DELETE FROM uuid_mapping WHERE uuid=?", (uuid,))

    def close(self) -> None:
        with self._lock:
            try:
                self._conn.close()
            except Exception:
                pass


