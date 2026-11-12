import os
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

BASE_DIR = Path(__file__).resolve().parent.parent
DB_PATH = BASE_DIR / "database" / "ab_testing.db"
SCHEMA_PATH = BASE_DIR / "database" / "schema.sql"
POSTGRES_SCHEMA = """
CREATE TABLE IF NOT EXISTS experiments (
    id BIGSERIAL PRIMARY KEY,
    name TEXT NOT NULL,
    dataset_name TEXT,
    status TEXT,
    notes TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS analysis_results (
    id BIGSERIAL PRIMARY KEY,
    experiment_id BIGINT REFERENCES experiments(id),
    metric_name TEXT,
    metric_value DOUBLE PRECISION,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS experiment_runs (
    id BIGSERIAL PRIMARY KEY,
    experiment_id BIGINT REFERENCES experiments(id),
    summary_text TEXT,
    recommendation TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS alert_statuses (
    alert_id TEXT PRIMARY KEY,
    status TEXT NOT NULL CHECK (status IN ('New', 'Investigating', 'Resolved')),
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
"""


def _postgres_url() -> str | None:
    return os.environ.get("DATABASE_URL") or None


def _connect():
    database_url = _postgres_url()
    if database_url:
        try:
            import psycopg
        except ImportError as exc:
            raise RuntimeError(
                "PostgreSQL is configured with DATABASE_URL, but the driver is missing. "
                'Install it with: python -m pip install "psycopg[binary]"'
            ) from exc
        return psycopg.connect(database_url)

    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    return sqlite3.connect(DB_PATH)


def _sql(statement: str) -> str:
    return statement.replace("?", "%s") if _postgres_url() else statement


@contextmanager
def _connection() -> Iterator:
    conn = _connect()
    try:
        with conn:
            if isinstance(conn, sqlite3.Connection):
                conn.execute("PRAGMA foreign_keys = ON")
            yield conn
    finally:
        conn.close()


def init_db() -> str:
    database_url = _postgres_url()
    if database_url:
        with _connection() as conn:
            for statement in POSTGRES_SCHEMA.split(";"):
                if statement.strip():
                    conn.execute(statement)
        return "PostgreSQL"

    with _connection() as conn:
        conn.executescript(SCHEMA_PATH.read_text(encoding="utf-8"))
    return str(DB_PATH)


def insert_experiment(name: str, dataset_name: str | None = None, status: str = "draft", notes: str = "") -> int:
    init_db()
    statement = (
        "INSERT INTO experiments (name, dataset_name, status, notes) VALUES (?, ?, ?, ?)"
    )
    if _postgres_url():
        statement += " RETURNING id"
        with _connection() as conn:
            row = conn.execute(_sql(statement), (name, dataset_name, status, notes)).fetchone()
            return int(row[0])

    with _connection() as conn:
        cursor = conn.execute(statement, (name, dataset_name, status, notes))
        return int(cursor.lastrowid)


def save_analysis_result(experiment_id: int, metric_name: str, metric_value: float) -> None:
    with _connection() as conn:
        conn.execute(
            _sql("INSERT INTO analysis_results (experiment_id, metric_name, metric_value) VALUES (?, ?, ?)"),
            (experiment_id, metric_name, metric_value),
        )


def save_run_summary(experiment_id: int, summary_text: str, recommendation: str) -> None:
    with _connection() as conn:
        conn.execute(
            _sql("INSERT INTO experiment_runs (experiment_id, summary_text, recommendation) VALUES (?, ?, ?)"),
            (experiment_id, summary_text, recommendation),
        )


def list_experiments() -> list[dict]:
    init_db()
    with _connection() as conn:
        rows = conn.execute(
            "SELECT id, name, dataset_name, status, notes, created_at FROM experiments ORDER BY id DESC"
        ).fetchall()
    return [
        {
            "id": row[0],
            "name": row[1],
            "dataset_name": row[2],
            "status": row[3],
            "notes": row[4],
            "created_at": row[5],
        }
        for row in rows
    ]


def list_alert_statuses() -> dict[str, str]:
    init_db()
    with _connection() as conn:
        rows = conn.execute("SELECT alert_id, status FROM alert_statuses").fetchall()
    return {str(alert_id): str(status) for alert_id, status in rows}


def update_alert_status(alert_id: str, status: str) -> None:
    if status not in {"New", "Investigating", "Resolved"}:
        raise ValueError("Alert status must be New, Investigating, or Resolved.")
    init_db()
    statement = (
        "INSERT INTO alert_statuses (alert_id, status) VALUES (?, ?) "
        "ON CONFLICT(alert_id) DO UPDATE SET status = excluded.status, "
        "updated_at = CURRENT_TIMESTAMP"
    )
    with _connection() as conn:
        conn.execute(_sql(statement), (alert_id, status))
