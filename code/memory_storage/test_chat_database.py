import unittest
from uuid import uuid4

import pandas as pd

from chat_database import ChatDatabase, CREATE_SQL


class FakeConnection:
    """Minimal connection/context-manager fake for unit tests."""

    def __init__(self):
        self.executed = []
        self.commit_count = 0
        self.rows_to_return = []

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        return False

    def execute(self, sql, params=None):
        self.executed.append((sql, params))
        return self

    def commit(self):
        self.commit_count += 1

    def fetchall(self):
        return self.rows_to_return


class FakeConnectionFactory:
    """Callable replacement for psycopg.connect."""

    def __init__(self):
        self.connections = []
        self.last_dsn = None

    def __call__(self, dsn):
        self.last_dsn = dsn
        connection = FakeConnection()
        self.connections.append(connection)
        return connection


class ChatDatabaseTests(unittest.TestCase):

    def setUp(self):
        self.connection = FakeConnectionFactory()

    def test_init_uses_explicit_dsn(self):
        db = ChatDatabase(
            pg_dn="postgresql://test",
            connection=self.connection,
        )

        self.assertEqual(db.pg_dsn, "postgresql://test")

    def test_init_uses_getenv_when_dsn_not_provided(self):
        def fake_getenv(name):
            self.assertEqual(name, "PG_DSN")
            return "postgresql://from-environment"

        db = ChatDatabase(
            connection=self.connection,
            getenv=fake_getenv,
        )

        self.assertEqual(db.pg_dsn, "postgresql://from-environment")

    def test_init_raises_when_dsn_is_missing(self):
        def fake_getenv(name):
            return None

        with self.assertRaises(Exception) as context:
            ChatDatabase(
                connection=self.connection,
                getenv=fake_getenv,
            )

        self.assertEqual(
            str(context.exception),
            "Environment variable PG_DSN must be set",
        )

    def test_create_executes_create_sql_and_commits(self):
        db = ChatDatabase(
            pg_dn="postgresql://test",
            connection=self.connection,
        )

        db.create()

        self.assertEqual(self.connection.last_dsn, "postgresql://test")
        conn = self.connection.connections[0]

        self.assertEqual(len(conn.executed), 1)
        self.assertEqual(conn.executed[0][0], CREATE_SQL)
        self.assertIsNone(conn.executed[0][1])
        self.assertEqual(conn.commit_count, 1)

    def test_insert_memory_converts_uuid_to_string(self):
        db = ChatDatabase(
            pg_dn="postgresql://test",
            connection=self.connection,
        )

        memory_id = uuid4()

        result = db.insert_memory(
            memory_id=memory_id,
            content="John ate a worm at the park",
            t_created="2026-10-07T12:00:00Z",
            t_last_recalled=None,
            connection=self.connection,
        )

        self.assertEqual(result, str(memory_id))

        conn = self.connection.connections[0]
        self.assertEqual(len(conn.executed), 1)

        sql, params = conn.executed[0]
        self.assertIn("INSERT INTO memories", sql)
        self.assertEqual(params[0], str(memory_id))
        self.assertEqual(params[1], "John ate a worm at the park")
        self.assertEqual(params[6], {})

        self.assertEqual(conn.commit_count, 1)

    def test_insert_memory_preserves_metadata(self):
        db = ChatDatabase(
            pg_dn="postgresql://test",
            connection=self.connection,
        )

        metadata = {"source": "test", "importance": 3}

        db.insert_memory(
            memory_id=uuid4(),
            content="A baby is crying",
            t_created="2026-10-07T12:00:00Z",
            t_last_recalled="2026-10-07T12:05:00Z",
            consolidation=2.5,
            recall_count=4,
            metadata=metadata,
            connection=self.connection,
        )

        conn = self.connection.connections[0]
        _, params = conn.executed[0]

        self.assertEqual(params[4], 2.5)
        self.assertEqual(params[5], 4)
        self.assertEqual(params[6], metadata)

    def test_retrieve_memories_converts_ids_to_strings(self):
        db = ChatDatabase(
            pg_dn="postgresql://test",
            connection=self.connection,
        )

        memory_ids = [uuid4(), uuid4()]

        conn = self.connection()
        conn.rows_to_return = [
            (
                "first memory",
                "2026-10-07T10:00:00Z",
                None,
                1.0,
                0,
                {"source": "test"},
            )
        ]

        # The factory created a new connection above. Replace the connection
        # that the method will receive with the prepared one.
        self.connection.connections.pop()
        self.connection.connections.append(conn)

        result = db.retrieve_memories(
            memory_ids,
            connection=self.connection,
        )

        self.assertIsInstance(result, pd.DataFrame)

        executed_sql, params = conn.executed[0]
        self.assertIn("WHERE memory_id = ANY(%s::uuid[])", executed_sql)
        self.assertEqual(params, ([str(x) for x in memory_ids],))

    def test_retrieve_memories_returns_expected_dataframe(self):
        db = ChatDatabase(
            pg_dn="postgresql://test",
            connection=self.connection,
        )

        memory_id = uuid4()

        conn = self.connection()
        conn.rows_to_return = [
            (
                "A baby is crying",
                "2026-10-07T10:00:00Z",
                None,
                1.5,
                2,
                {"source": "test"},
            )
        ]

        self.connection.connections.pop()
        self.connection.connections.append(conn)

        result = db.retrieve_memories(
            [memory_id],
            connection=self.connection,
        )

        expected_columns = [
            "memory_id",
            "content",
            "t_created",
            "t_last_recalled",
            "consolidation",
            "recall_count",
            "metadata",
        ]

        self.assertEqual(list(result.columns), expected_columns)
        self.assertEqual(len(result), 1)
        self.assertEqual(result.iloc[0]["content"], "A baby is crying")
        self.assertEqual(result.iloc[0]["consolidation"], 1.5)
        self.assertEqual(result.iloc[0]["recall_count"], 2)
        self.assertEqual(result.iloc[0]["metadata"], {"source": "test"})


if __name__ == "__main__":
    unittest.main()
