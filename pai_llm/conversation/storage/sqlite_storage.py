import os
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Optional

from loguru import logger

from pai_llm.conversation.exceptions import DatabaseOperationError
from pai_llm.conversation.storage.sql_storage import (
    SQLSchema,
    SQLStorage,
    SQLiteError,
    SQLConnection,
    SQLLiteRow,
)


class SQLiteSchema(SQLSchema):
    def create_conversation_table(self) -> str:
        return """
            CREATE TABLE IF NOT EXISTS conversation (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                user_id TEXT NOT NULL,
                metadata TEXT,
                created_at TIMESTAMP NOT NULL,
                updated_at TIMESTAMP NOT NULL
            )
        """

    def create_message_table(self) -> str:
        return """
            CREATE TABLE IF NOT EXISTS message (
                id TEXT PRIMARY KEY,
                conversation_id TEXT NOT NULL,
                user_id TEXT NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                metadata TEXT,
                token_count INTEGER,
                message_index INTEGER,
                created_at TIMESTAMP NOT NULL,
                FOREIGN KEY (conversation_id) REFERENCES CONVERSATION(id) ON DELETE CASCADE
            )
        """

    def insert_conversation(self) -> str:
        return """
            INSERT OR REPLACE INTO conversation (id, name, user_id, metadata, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?)
        """

    def get_conversation(self) -> str:
        return """
            SELECT * FROM conversation WHERE id = ?
        """

    def list_conversations(self) -> str:
        return """
            SELECT * FROM conversation WHERE user_id = ? ORDER BY updated_at DESC LIMIT ? OFFSET ?
        """

    def rename_conversation(self) -> str:
        return """
            UPDATE conversation SET name = ? WHERE id = ?
        """

    def delete_conversation(self) -> str:
        return """
           DELETE FROM conversation WHERE id = ?
        """

    def insert_message(self) -> str:
        return """
            INSERT INTO message (id, conversation_id, user_id, role, content, metadata, token_count, message_index, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """

    def get_message(self) -> str:
        return """
            SELECT * FROM message WHERE id = ?
        """

    def get_messages(self) -> str:
        return """
            SELECT * FROM message WHERE conversation_id = ? ORDER BY message_index ASC
        """

    def list_messages(self) -> str:
        return """
            SELECT * FROM message WHERE conversation_id = ? ORDER BY message_index ASC LIMIT ?
        """

    def delete_messages(self) -> str:
        return """
            DELETE FROM message WHERE conversation_id = ?
        """


class SQLiteStorage(SQLStorage):

    def __init__(self, db_file_path: str | Path, max_messages: Optional[int] = 100):
        assert db_file_path, "Database file path cannot be empty"
        assert max_messages is None or max_messages > 0, "Max messages must be greater than 0"
        super().__init__(SQLiteSchema(), max_messages)

        self.db_file_path = db_file_path or ":memory:"
        if self.db_file_path != ":memory:":
            os.makedirs(os.path.dirname(os.path.abspath(self.db_file_path)), exist_ok=True)
        logger.debug(f"Initializing SQLiteStorage with db_file_path: {db_file_path}")

        try:
            with self._open_connection() as conn:
                self._init_tables(conn)
        except SQLiteError as e:
            logger.error(f"SQLiteStorage initialization failed: {e}")
            raise DatabaseOperationError(f"SQLiteStorage initialization failed: {e}") from e

    @contextmanager
    def _open_connection(self):
        conn: SQLConnection | None = None
        try:
            conn = sqlite3.connect(self.db_file_path)
            conn.row_factory = SQLLiteRow
            conn.execute("PRAGMA foreign_keys = ON")
            yield conn
        except Exception as e:
            logger.error(f"Failed to connect to SQLite database: {e}")
            if conn:
                conn.rollback()
            raise DatabaseOperationError(f"Failed to connect to SQLite database: {e}") from e
        finally:
            if conn:
                conn.close()

    def _get_param_placeholder(self) -> str:
        return "?"

    def _start_transaction(self, conn: SQLConnection):
        conn.execute("BEGIN TRANSACTION")
