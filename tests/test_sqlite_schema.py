import sqlite3

from agent_app.sqlite_schema import ensure_sqlite_schema


def test_ensure_sqlite_schema_bootstraps_tables_and_migrates_retrieval_events(tmp_path):
    db_path = tmp_path / "agent.db"
    conn = sqlite3.connect(db_path)
    conn.execute(
        """
        CREATE TABLE retrieval_events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id TEXT NOT NULL,
            thread_id TEXT NOT NULL,
            user_text TEXT NOT NULL,
            memory_count INTEGER NOT NULL,
            memory_preview TEXT NOT NULL,
            created_at TEXT NOT NULL
        )
        """
    )
    conn.commit()
    conn.close()

    ensure_sqlite_schema(db_path)

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    columns = {row["name"] for row in conn.execute("PRAGMA table_info(retrieval_events)").fetchall()}
    profile = conn.execute("SELECT * FROM agent_profile WHERE id = 1").fetchone()
    conn.close()

    assert "memory_ids" in columns
    assert profile is not None
    assert "像真人一样" in profile["identity"]
