import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Iterator


class Database:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.init()

    @contextmanager
    def connect(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(self.path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    def init(self) -> None:
        with self.connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS generations (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    filename TEXT NOT NULL,
                    image_path TEXT NOT NULL,
                    product_name TEXT NOT NULL,
                    category TEXT NOT NULL,
                    audience TEXT NOT NULL,
                    price TEXT NOT NULL,
                    origin_price TEXT NOT NULL,
                    analysis_json TEXT NOT NULL,
                    html TEXT NOT NULL,
                    created_at TEXT NOT NULL
                )
                """
            )

    def create_generation(self, payload: dict[str, Any]) -> dict[str, Any]:
        now = datetime.utcnow().isoformat(timespec="seconds") + "Z"
        with self.connect() as conn:
            cursor = conn.execute(
                """
                INSERT INTO generations (
                    filename, image_path, product_name, category, audience,
                    price, origin_price, analysis_json, html, created_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    payload["filename"],
                    payload["image_path"],
                    payload["product_name"],
                    payload["category"],
                    payload["audience"],
                    payload["price"],
                    payload["origin_price"],
                    json.dumps(payload["analysis"], ensure_ascii=False),
                    payload["html"],
                    now,
                ),
            )
            row_id = cursor.lastrowid
        return self.get_generation(row_id)

    def list_generations(self) -> list[dict[str, Any]]:
        with self.connect() as conn:
            rows = conn.execute(
                "SELECT * FROM generations ORDER BY id DESC LIMIT 100"
            ).fetchall()
        return [self._row_to_dict(row) for row in rows]

    def get_generation(self, generation_id: int) -> dict[str, Any]:
        with self.connect() as conn:
            row = conn.execute(
                "SELECT * FROM generations WHERE id = ?", (generation_id,)
            ).fetchone()
        if row is None:
            raise KeyError(f"Generation {generation_id} not found")
        return self._row_to_dict(row)

    def _row_to_dict(self, row: sqlite3.Row) -> dict[str, Any]:
        data = dict(row)
        data["analysis"] = json.loads(data.pop("analysis_json"))
        return data

