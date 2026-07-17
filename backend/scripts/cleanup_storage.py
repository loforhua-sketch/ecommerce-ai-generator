"""Audit and optionally clean generated storage.

Dry-run is the default. Destructive actions require both --apply and --yes.
The database is backed up before any database rows are deleted.
"""

from __future__ import annotations

import argparse
import shutil
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

from backend.app.config import get_settings


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit generated records and uploaded files safely.")
    parser.add_argument("--max-record-mb", type=int, default=20, help="Large-record threshold (default: 20 MB).")
    parser.add_argument("--delete-orphans", action="store_true", help="Delete upload files not referenced by the database.")
    parser.add_argument("--delete-large-records", action="store_true", help="Delete records exceeding the threshold.")
    parser.add_argument("--vacuum", action="store_true", help="Run SQLite VACUUM after selected cleanup operations.")
    parser.add_argument("--apply", action="store_true", help="Apply selected deletion operations; otherwise dry-run.")
    parser.add_argument("--yes", action="store_true", help="Confirm destructive operations when used with --apply.")
    return parser.parse_args()


def database_backup(database_file: Path) -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    backup = database_file.with_name(f"{database_file.name}.backup-{stamp}")
    shutil.copy2(database_file, backup)
    return backup


def main() -> int:
    args = parse_args()
    settings = get_settings()
    database_file = settings.database_file
    upload_path = settings.upload_path

    if not database_file.is_file():
        print(f"Database not found: {database_file}")
        return 1

    threshold = args.max_record_mb * 1024 * 1024
    with sqlite3.connect(database_file) as conn:
        rows = conn.execute(
            "SELECT id, image_path, length(analysis_json) + length(html) AS payload_size "
            "FROM generations ORDER BY id"
        ).fetchall()

        referenced = {Path(str(row[1])).name for row in rows if row[1]}
        large_rows = [(int(row[0]), int(row[2] or 0)) for row in rows if int(row[2] or 0) >= threshold]
        upload_files = [path for path in upload_path.iterdir() if path.is_file() and path.name != ".gitkeep"] if upload_path.exists() else []
        orphan_files = [path for path in upload_files if path.name not in referenced]

        print(f"Database: {database_file}")
        print(f"Uploads: {upload_path}")
        print(f"Records: {len(rows)}")
        print(f"Referenced uploads: {len(referenced)}")
        print(f"Orphan uploads: {len(orphan_files)}")
        for path in orphan_files:
            print(f"  ORPHAN {path.name} ({path.stat().st_size} bytes)")
        print(f"Records >= {args.max_record_mb} MB: {len(large_rows)}")
        for row_id, size in large_rows:
            print(f"  LARGE id={row_id} payload_bytes={size}")

        destructive = args.delete_orphans or args.delete_large_records or args.vacuum
        if not args.apply:
            print("Dry-run only. No files or records were deleted.")
            return 0
        if destructive and not args.yes:
            print("Refusing destructive operation: add --yes together with --apply.")
            return 2
        if not destructive:
            print("No deletion option selected; nothing to apply.")
            return 0

        if args.delete_large_records and large_rows:
            backup = database_backup(database_file)
            print(f"Database backup created: {backup}")
            conn.executemany("DELETE FROM generations WHERE id = ?", [(row_id,) for row_id, _ in large_rows])
            conn.commit()
            print(f"Deleted large records: {len(large_rows)}")

        if args.delete_orphans:
            for path in orphan_files:
                path.unlink()
            print(f"Deleted orphan uploads: {len(orphan_files)}")

        if args.vacuum:
            conn.execute("VACUUM")
            print("SQLite VACUUM completed.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
