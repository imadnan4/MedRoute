"""Apply the checked-in SQL migrations using the direct Neon connection."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import psycopg

from config import settings


def main() -> None:
    database_url = settings.database_url_unpooled or settings.database_url
    if not database_url:
        raise SystemExit("Set MEDROUTE_DATABASE_URL_UNPOOLED (preferred) or MEDROUTE_DATABASE_URL")
    migration_dir = ROOT / "migrations"
    with psycopg.connect(database_url) as connection:
        for migration in sorted(migration_dir.glob("*.sql")):
            connection.execute(migration.read_text(encoding="utf-8"))
            print(f"Applied {migration.name}")


if __name__ == "__main__":
    main()
