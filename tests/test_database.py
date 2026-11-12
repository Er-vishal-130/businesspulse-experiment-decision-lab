import sqlite3
import sys
from types import ModuleType

from database import database


def test_sqlite_persistence_round_trip(monkeypatch, tmp_path):
    db_path = tmp_path / "test.db"
    monkeypatch.setattr(database, "DB_PATH", db_path)
    monkeypatch.delenv("DATABASE_URL", raising=False)

    database.init_db()
    experiment_id = database.insert_experiment("Checkout test", "test.csv", "completed", "test run")
    database.save_analysis_result(experiment_id, "p_value", 0.03)
    database.save_run_summary(experiment_id, "Summary", "Ship treatment")

    experiments = database.list_experiments()
    assert experiments[0]["id"] == experiment_id
    assert experiments[0]["name"] == "Checkout test"
    with sqlite3.connect(db_path) as conn:
        metric = conn.execute(
            "SELECT metric_value FROM analysis_results WHERE experiment_id = ?",
            (experiment_id,),
        ).fetchone()
        recommendation = conn.execute(
            "SELECT recommendation FROM experiment_runs WHERE experiment_id = ?",
            (experiment_id,),
        ).fetchone()
    assert metric == (0.03,)
    assert recommendation == ("Ship treatment",)


def test_alert_status_persistence_round_trip(monkeypatch, tmp_path):
    db_path = tmp_path / "alerts.db"
    monkeypatch.setattr(database, "DB_PATH", db_path)
    monkeypatch.delenv("DATABASE_URL", raising=False)

    database.update_alert_status("alert-123", "Investigating")
    assert database.list_alert_statuses() == {"alert-123": "Investigating"}

    database.update_alert_status("alert-123", "Resolved")
    assert database.list_alert_statuses() == {"alert-123": "Resolved"}


def test_alert_status_rejects_unknown_values(monkeypatch, tmp_path):
    monkeypatch.setattr(database, "DB_PATH", tmp_path / "alerts.db")
    monkeypatch.delenv("DATABASE_URL", raising=False)

    try:
        database.update_alert_status("alert-123", "Closed")
    except ValueError as exc:
        assert "Alert status" in str(exc)
    else:
        raise AssertionError("An unknown alert status must be rejected.")


def test_postgresql_uses_parameter_placeholders_and_returning(monkeypatch):
    executed = []

    class FakeConnection:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc_value, traceback):
            return False

        def execute(self, statement, params=None):
            executed.append((statement, params))
            return self

        def fetchone(self):
            return (17,)

        def close(self):
            pass

    fake_driver = ModuleType("psycopg")
    fake_driver.connect = lambda database_url: FakeConnection()
    monkeypatch.setitem(sys.modules, "psycopg", fake_driver)
    monkeypatch.setenv("DATABASE_URL", "postgresql://user:password@localhost/test")

    assert database.init_db() == "PostgreSQL"
    experiment_id = database.insert_experiment("Checkout test", "test.csv")
    database.update_alert_status("alert-17", "Investigating")

    assert experiment_id == 17
    insert_statement = next(statement for statement, _ in executed if "INSERT INTO experiments" in statement)
    assert "VALUES (%s, %s, %s, %s) RETURNING id" in insert_statement
    alert_statement = next(
        statement for statement, _ in executed if "INSERT INTO alert_statuses" in statement
    )
    assert "VALUES (%s, %s) ON CONFLICT(alert_id)" in alert_statement
