"""Unit tests for MemoryDatabase; no patching or live PostgreSQL required."""
# TODO remove score from the tests!

import unittest
from contextlib import contextmanager

import pandas as pd

from memory_database import MemoryDatabase, MEMORY_COLUMNS


class FakeCursor:
    def __init__(self, connection):
        self.connection = connection

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def executemany(self, sql, params):
        self.connection.executemany_calls.append((sql, params))


class FakeConnection:
    def __init__(self, rows=None):
        self.rows = rows or []
        self.execute_calls = []
        self.executemany_calls = []
        self.commits = 0

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def execute(self, sql, params=None):
        self.execute_calls.append((sql, params))
        return self

    def executemany(self, sql, params):
        self.executemany_calls.append((sql, params))

    def fetchall(self):
        return self.rows

    def cursor(self):
        return FakeCursor(self)

    def commit(self):
        self.commits += 1


class ConnectionFactory:
    def __init__(self, rows=None):
        self.connections = []
        self.rows = rows or []

    def __call__(self, dsn):
        connection = FakeConnection(self.rows)
        self.connections.append((dsn, connection))
        return connection


class TestMemoryDatabase(unittest.TestCase):
    """All public API tests for MemoryDatabase live in one test class."""

    def make_db(self, rows=None, initialize=False):
        factory = ConnectionFactory(rows)
        db = MemoryDatabase(
            pg_dsn="postgresql://test",
            connection=factory,
            getenv=lambda key: None,
            initialize=initialize,
        )
        return db, factory

    def test_requires_dsn(self):
        with self.assertRaises(ValueError):
            MemoryDatabase(connection=ConnectionFactory(), getenv=lambda key: None,
                           initialize=False)

    def test_initialization_executes_schema_and_commits(self):
        db, factory = self.make_db(initialize=True)
        _, conn = factory.connections[0]
        self.assertIn("CREATE TABLE IF NOT EXISTS memories", conn.execute_calls[0][0])
        self.assertEqual(conn.commits, 1)

    def test_upsert_requires_dataframe(self):
        db, _ = self.make_db()
        with self.assertRaises(TypeError):
            db.upsert_memories([{"memory_id": "id"}])

    def test_upsert_checks_required_columns(self):
        db, _ = self.make_db()
        with self.assertRaisesRegex(ValueError, "Missing required columns"):
            db.upsert_memories(pd.DataFrame([{"memory_id": "id"}]))

    def test_upsert_empty_dataframe_returns_zero_without_connecting(self):
        db, factory = self.make_db()
        frame = pd.DataFrame(columns=["memory_id", "agent_id", "content", "t_created"])
        self.assertEqual(db.upsert_memories(frame), 0)
        self.assertEqual(factory.connections, [])

    def test_upsert_batches_rows_and_commits(self):
        db, factory = self.make_db()
        frame = pd.DataFrame([
            {"memory_id": "00000000-0000-0000-0000-000000000001",
             "agent_id": "agent-a", "content": "one", "t_created": 1.0, "score": 0.7},
            {"memory_id": "00000000-0000-0000-0000-000000000002",
             "agent_id": "agent-a", "content": "two", "t_created": 2.0, "score": None},
        ])
        self.assertEqual(db.upsert_memories(frame), 2)
        _, conn = factory.connections[0]
        self.assertEqual(len(conn.executemany_calls), 1)
        self.assertEqual(len(conn.executemany_calls[0][1]), 2)
        self.assertIn("ON CONFLICT (memory_id) DO UPDATE", conn.executemany_calls[0][0])
        self.assertEqual(conn.commits, 1)

    def test_upsert_rejects_null_ids(self):
        db, _ = self.make_db()
        frame = pd.DataFrame([{
            "memory_id": None, "agent_id": "agent-a", "content": "x", "t_created": 1
        }])
        with self.assertRaisesRegex(ValueError, "must not be null"):
            db.upsert_memories(frame)

    def test_upsert_rejects_non_dictionary_metadata(self):
        db, _ = self.make_db()
        frame = pd.DataFrame([{
            "memory_id": "00000000-0000-0000-0000-000000000001",
            "agent_id": "agent-a", "content": "x", "t_created": 1, "metadata": "bad"
        }])
        with self.assertRaisesRegex(TypeError, "metadata values"):
            db.upsert_memories(frame)

    def test_recall_by_id_empty_ids_returns_empty_dataframe_without_connecting(self):
        db, factory = self.make_db()
        result = db.recall_by_id([], "agent-a")
        self.assertTrue(result.empty)
        self.assertEqual(list(result.columns), MEMORY_COLUMNS)
        self.assertEqual(factory.connections, [])

    def test_recall_by_id_filters_agent_and_returns_dataframe(self):
        row = ("00000000-0000-0000-0000-000000000001", "agent-a", "hello", "user",
               1.0, None, 1.0, 0, {}, None)
        db, factory = self.make_db(rows=[row])
        result = db.recall_by_id(["00000000-0000-0000-0000-000000000001"], "agent-a")
        self.assertEqual(list(result.columns), MEMORY_COLUMNS)
        self.assertEqual(result.iloc[0]["agent_id"], "agent-a")
        _, conn = factory.connections[0]
        sql, params = conn.execute_calls[0]
        self.assertIn("agent_id = %s", sql)
        self.assertEqual(params[1], "agent-a")

    def test_recall_by_score_filters_agent_and_ignores_null_scores(self):
        row = ("00000000-0000-0000-0000-000000000001", "agent-a", "scored", "user",
               1.0, None, 1.0, 0, {}, 0.9)
        db, factory = self.make_db(rows=[row])
        result = db.recall_by_score("agent-a", 5)
        self.assertEqual(list(result.columns), MEMORY_COLUMNS)
        _, conn = factory.connections[0]
        sql, params = conn.execute_calls[0]
        self.assertIn("agent_id = %s AND score IS NOT NULL", sql)
        self.assertIn("ORDER BY score DESC", sql)
        self.assertIn("LIMIT %s", sql)
        self.assertEqual(params, ("agent-a", 5))

    def test_recall_by_score_zero_returns_empty_without_connecting(self):
        db, factory = self.make_db()
        result = db.recall_by_score("agent-a", 0)
        self.assertTrue(result.empty)
        self.assertEqual(factory.connections, [])

    def test_recall_by_score_rejects_negative_k(self):
        db, _ = self.make_db()
        with self.assertRaises(ValueError):
            db.recall_by_score("agent-a", -1)

    def test_recall_by_score_rejects_non_integer_k(self):
        db, _ = self.make_db()
        with self.assertRaises(TypeError):
            db.recall_by_score("agent-a", 2.5)


if __name__ == "__main__":
    unittest.main()
