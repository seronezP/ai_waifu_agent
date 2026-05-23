from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path


@dataclass(frozen=True)
class MemoryItem:
    id: int
    kind: str
    content: str
    created_at: str
    score: float = 0.0


class MemoryStore:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self.path)

    def _init_db(self) -> None:
        with self._connect() as db:
            db.execute(
                """
                CREATE TABLE IF NOT EXISTS memories (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    kind TEXT NOT NULL,
                    content TEXT NOT NULL,
                    created_at TEXT NOT NULL
                )
                """
            )
            db.execute(
                """
                CREATE VIRTUAL TABLE IF NOT EXISTS memories_fts
                USING fts5(content, kind, content='memories', content_rowid='id')
                """
            )
            db.execute(
                """
                CREATE TRIGGER IF NOT EXISTS memories_ai AFTER INSERT ON memories BEGIN
                    INSERT INTO memories_fts(rowid, content, kind)
                    VALUES (new.id, new.content, new.kind);
                END
                """
            )
            db.execute(
                """
                CREATE TRIGGER IF NOT EXISTS memories_ad AFTER DELETE ON memories BEGIN
                    INSERT INTO memories_fts(memories_fts, rowid, content, kind)
                    VALUES ('delete', old.id, old.content, old.kind);
                END
                """
            )

    def add(self, content: str, kind: str = "fact") -> MemoryItem:
        clean = " ".join(content.strip().split())
        if not clean:
            raise ValueError("memory content is empty")
        now = datetime.now(timezone.utc).isoformat(timespec="seconds")
        with self._connect() as db:
            cursor = db.execute(
                "INSERT INTO memories(kind, content, created_at) VALUES (?, ?, ?)",
                (kind, clean, now),
            )
            item_id = int(cursor.lastrowid)
        return MemoryItem(id=item_id, kind=kind, content=clean, created_at=now)

    def recent(self, limit: int = 10) -> list[MemoryItem]:
        with self._connect() as db:
            rows = db.execute(
                """
                SELECT id, kind, content, created_at
                FROM memories
                ORDER BY id DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
        return [MemoryItem(*row) for row in rows]

    def search(self, query: str, limit: int = 6) -> list[MemoryItem]:
        clean = " ".join(query.strip().split())
        if not clean:
            return self.recent(limit)
        with self._connect() as db:
            try:
                rows = db.execute(
                    """
                    SELECT m.id, m.kind, m.content, m.created_at, bm25(memories_fts) AS score
                    FROM memories_fts
                    JOIN memories m ON m.id = memories_fts.rowid
                    WHERE memories_fts MATCH ?
                    ORDER BY score
                    LIMIT ?
                    """,
                    (clean, limit),
                ).fetchall()
            except sqlite3.OperationalError:
                rows = db.execute(
                    """
                    SELECT id, kind, content, created_at, 0.0
                    FROM memories
                    WHERE lower(content) LIKE lower(?)
                    ORDER BY id DESC
                    LIMIT ?
                    """,
                    (f"%{clean}%", limit),
                ).fetchall()
        return [MemoryItem(*row) for row in rows]

    def format_context(self, query: str, limit: int = 6) -> str:
        items = self.search(query, limit)
        if not items:
            return "Пока нет сохраненной долговременной памяти."
        return "\n".join(f"- [{item.kind}] {item.content}" for item in items)
