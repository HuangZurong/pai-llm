import json
import sqlite3
from abc import ABC, abstractmethod
from contextlib import contextmanager
from datetime import datetime
from typing import Any, Dict, Optional, List, Tuple

import mysql
from commons_lang import object_utils
from loguru import logger
from mysql.connector.abstracts import MySQLConnectionAbstract
from mysql.connector.pooling import PooledMySQLConnection

from pai_llm.conversation.exceptions import DatabaseOperationError
from pai_llm.conversation.models import Message, Conversation
from pai_llm.conversation.storage.base import BaseStorage
from pai_llm.exceptions import DataConvertError

SQLiteConnection = sqlite3.Connection
MySQLConnection = PooledMySQLConnection | MySQLConnectionAbstract
SQLConnection = SQLiteConnection | MySQLConnection

SQLiteError = sqlite3.Error
MySQLError = mysql.connector.Error
SQLError = SQLiteError | MySQLError

SQLiteIntegrityError = sqlite3.IntegrityError
MySQLIntegrityError = mysql.connector.errors.IntegrityError
SQIntegrityError = SQLiteIntegrityError | MySQLIntegrityError

SQLLiteRow = sqlite3.Row
MySQLRow = Dict
SQLRow = SQLLiteRow | MySQLRow


class SQLSchema(ABC):

    @abstractmethod
    def create_conversation_table(self) -> str:
        pass

    @abstractmethod
    def create_message_table(self) -> str:
        pass

    @abstractmethod
    def insert_conversation(self) -> str:
        pass

    @abstractmethod
    def get_conversation(self) -> str:
        pass

    @abstractmethod
    def list_conversations(self) -> str:
        pass

    @abstractmethod
    def rename_conversation(self) -> str:
        pass

    @abstractmethod
    def delete_conversation(self) -> str:
        pass

    @abstractmethod
    def insert_message(self) -> str:
        pass

    @abstractmethod
    def get_message(self) -> str:
        pass

    @abstractmethod
    def get_messages(self) -> str:
        pass

    @abstractmethod
    def list_messages(self) -> str:
        pass

    @abstractmethod
    def delete_messages(self) -> str:
        pass


class SQLStorage(BaseStorage):

    def __init__(
            self,
            sql_scheme: SQLSchema,
            max_messages: Optional[int] = 100
    ):
        super().__init__(max_messages)
        self.sql_scheme = sql_scheme

    @abstractmethod
    @contextmanager
    def _open_connection(self):
        pass

    @abstractmethod
    def _get_param_placeholder(self) -> str:
        pass

    @abstractmethod
    def _start_transaction(self, conn: SQLConnection):
        pass

    def _init_tables(self, conn: SQLConnection):
        try:
            conn.execute(self.sql_scheme.create_conversation_table())
            conn.execute(self.sql_scheme.create_message_table())
            conn.commit()
            logger.debug(f"SQStorage tables initialization success")
        except SQLError as e:
            logger.error(f"SQStorage tables initialization failed: {e}")
            raise DatabaseOperationError(f"SQStorage tables initialization failed: {e}") from e

    @staticmethod
    def _serialize_metadata(metadata: Dict[str, Any]) -> str:
        try:
            return json.dumps(metadata)
        except Exception as e:
            logger.error(f"Failed to serialize metadata: {e}")
            raise DataConvertError(f"Failed to serialize metadata: {e}") from e

    @staticmethod
    def _deserialize_metadata(metadata_str: str) -> Dict[str, Any]:
        try:
            return json.loads(metadata_str)
        except Exception as e:
            logger.error(f"Failed to deserialize metadata: {e}")
            raise DataConvertError(f"Failed to deserialize metadata: {e}") from e

    @staticmethod
    def _convert_datetime(dt: str | datetime) -> datetime:
        try:
            if isinstance(dt, str):
                return datetime.strptime(dt, "%Y-%m-%d %H:%M:%S")
            else:
                return dt
        except Exception as e:
            logger.error(f"Failed to convert datetime: {e}")
            raise DataConvertError(f"Failed to convert datetime: {e}") from e

    def _conversation_to_row(self, conversation: Conversation) -> Tuple:
        assert conversation is not None, "Conversation cannot be None"
        assert conversation.id is not None, "Conversation id cannot be None"

        try:
            return (
                conversation.id,
                conversation.name,
                conversation.user_id,
                self._serialize_metadata(conversation.metadata),
                conversation.created_at.strftime("%Y-%m-%d %H:%M:%S"),
                conversation.updated_at.strftime("%Y-%m-%d %H:%M:%S")
            )
        except Exception as e:
            logger.error(f"Failed to convert conversation to row: {e}")
            raise DataConvertError(f"Failed to convert conversation to row: {e}") from e

    def _row_to_conversation(self, row: SQLRow, messages: List[Message]) -> Conversation:
        try:
            return Conversation(
                id=row["id"],
                name=row["name"],
                user_id=row["user_id"],
                messages=messages,
                metadata=self._deserialize_metadata(row["metadata"]),
                created_at=self._convert_datetime(row["created_at"]),
                updated_at=self._convert_datetime(row["updated_at"]),
            )
        except Exception as e:
            logger.error(f"Failed to convert row to conversation: {e}")
            raise DataConvertError(f"Failed to convert row to conversation: {e}") from e

    def _message_to_row(self, message: Message, conversation_id: str, user_id: str, message_index: int) -> Tuple:
        assert message is not None, "Message cannot be None"
        assert conversation_id is not None, "Conversation id cannot be None"
        assert user_id is not None, "User id cannot be None"
        assert message_index >= 0, "MessageIndex must be greater than or equal to 0"

        try:
            return (
                message.id,
                conversation_id,
                user_id,
                message.role,
                message.content,
                self._serialize_metadata(message.metadata),
                message.token_count,
                message_index,
                message.created_at.strftime("%Y-%m-%d %H:%M:%S")
            )
        except Exception as e:
            logger.error(f"Failed to convert message to row: {e}")
            raise DataConvertError(f"Failed to convert message to row: {e}") from e

    def _row_to_message(self, row: SQLRow) -> Message:
        assert row is not None, "Row cannot be None"
        try:
            return Message(
                id=row["id"],
                role=row["role"],
                content=row["content"],
                metadata=self._deserialize_metadata(row["metadata"]),
                token_count=row["token_count"],
                created_at=self._convert_datetime(row["created_at"]),
            )
        except (KeyError, ValueError) as e:
            logger.error(f"Failed to convert row to message: {e}")
            raise DataConvertError(f"Failed to convert row to message: {e}") from e

    def save_conversation(self, conversation: Conversation):
        assert conversation is not None, "Conversation cannot be None"
        assert conversation.id is not None, "Conversation id cannot be None"

        try:
            with self._open_connection() as conn:
                self._start_transaction(conn)

                conversation_row = self._conversation_to_row(conversation)
                conn.execute(self.sql_scheme.insert_conversation(), conversation_row)
                conn.execute(self.sql_scheme.delete_messages(), (conversation.id,))

                messages = conversation.messages
                if object_utils.is_not_empty(messages) and (len(messages) > self.max_messages):
                    messages = messages[-self.max_messages:]

                for i, message in enumerate(messages):
                    message_row = self._message_to_row(message, conversation.id, conversation.user_id, i)
                    conn.execute(self.sql_scheme.insert_message(), message_row)

                conn.commit()
                logger.debug(f"SQLStorage saved conversation {conversation.id}")
        except SQIntegrityError as e:
            logger.error(f"Database integrity error while saving conversation {conversation.id}: {e}")
            raise DatabaseOperationError(
                f"Database integrity error while saving conversation {conversation.id}: {e}"
            ) from e
        except SQLiteError as e:
            logger.error(f"Database error while saving conversation {conversation.id}: {e}")
            raise DatabaseOperationError(
                f"Database error while saving conversation {conversation.id}: {e}"
            ) from e
        except ValueError as e:
            logger.error(f"Value error while saving conversation {conversation.id}: {e}")
            raise DatabaseOperationError(
                f"Value error while saving conversation {conversation.id}: {e}"
            ) from e

    def get_conversation(
            self,
            conversation_id: str,
            messages_limit: Optional[int] = None
    ) -> Optional[Conversation]:
        assert conversation_id is not None, "Conversation id cannot be None"
        assert messages_limit is None or messages_limit > 0, "Messages limit must be greater than 0"

        try:
            with self._open_connection() as conn:
                conversation_row = conn.execute(self.sql_scheme.get_conversation(), (conversation_id,)).fetchone()
                if not conversation_row:
                    logger.debug(f"SQLStorage got conversation {conversation_id} not found")
                    return None

                if messages_limit is not None:
                    message_rows = conn.execute(
                        self.sql_scheme.list_messages(),
                        (conversation_id, messages_limit)
                    ).fetchall()
                else:
                    message_rows = conn.execute(
                        self.sql_scheme.get_messages(),
                        (conversation_id,)
                    ).fetchall()

                messages = [self._row_to_message(row) for row in message_rows]
                return self._row_to_conversation(conversation_row, messages)
        except SQLiteError as e:
            logger.error(f"Database error while getting conversation {conversation_id}: {e}")
            raise DatabaseOperationError(
                f"Database error while getting conversation {conversation_id}: {e}"
            ) from e

    def list_conversations(self, user_id: str, page_no: int = 1, page_size: int = 10) -> List[Conversation]:
        assert page_no >= 1, "Page number must be greater than or equal to 1"
        assert page_size >= 1, "Page size must be greater than or equal to 1"

        conversations = []
        try:
            offset = (page_no - 1) * page_size
            limit = page_size
            with self._open_connection() as conn:
                rows = conn.execute(
                    self.sql_scheme.list_conversations(),
                    (user_id, limit, offset)
                ).fetchall()
                conversations = [self._row_to_conversation(row, []) for row in rows]
        except SQLiteError as e:
            logger.error(f"Database error while listing conversations: {e}")
            raise DatabaseOperationError(f"Database error while listing conversations: {e}") from e
        finally:
            return conversations

    def rename_conversation(self, conversation_id: str, new_name: str) -> bool:
        assert conversation_id is not None, "Conversation id cannot be None"
        assert new_name is not None, "New name cannot be None"

        ret = True
        try:
            with self._open_connection() as conn:
                conn.execute(self.sql_scheme.rename_conversation(), (new_name, conversation_id))
                conn.commit()
                logger.debug(f"SQLStorage renamed conversation {conversation_id} to {new_name}")
        except SQLiteError as e:
            logger.error(f"Database error while renaming conversation {conversation_id} to {new_name}: {e}")
            ret = False
            raise DatabaseOperationError(f"Database error while listing conversations: {e}") from e
        finally:
            return ret

    def delete_conversation(self, conversation_id: str) -> bool:
        assert conversation_id is not None, "Conversation id cannot be None"

        ret = True
        try:
            with self._open_connection() as conn:
                conn.execute(self.sql_scheme.delete_conversation(), (conversation_id,))
                conn.commit()
                logger.debug(f"SQLStorage deleted conversation {conversation_id}")
        except SQLiteError as e:
            logger.error(f"Database error while deleting conversation {conversation_id}: {e}")
            ret = False
            raise DatabaseOperationError(f"Database error while listing conversations: {e}") from e
        finally:
            return ret

    def search_conversations(self, user_id: str, query: Dict[str, any]) -> List[Conversation]:
        # TODO 支持语义搜索
        assert query is not None, "Query cannot be None"

        conversations = []
        conditions = []
        params = []
        try:
            conditions.append(f"user_id = {self._get_param_placeholder()}")
            params.append(user_id)

            if "name" in query:
                if not isinstance(query["name"], str):
                    raise ValueError("Name search criteria must be a string")
                conditions.append(f"name LIKE {self._get_param_placeholder()}")
                params.append(f"%{query['name']}%")

            if "metadata" in query:
                if not isinstance(query["metadata"], dict):
                    raise ValueError("Metadata search criteria must be a dictionary")
                for key, value in query["metadata"].items():
                    conditions.append(f"json_extract(metadata, '$.{key}') = {self._get_param_placeholder()}")
                    params.append(value)

            if "content" in query:
                if not isinstance(query["content"], str):
                    raise ValueError("Content search criteria must be a string")
                conditions.append(
                    f"""
                    id IN (
                        SELECT DISTINCT conversation_id
                        from message
                        WHERE content LIKE {self._get_param_placeholder()} and user_id = {self._get_param_placeholder()}
                    )
                    """
                )
                params.append(f"%{query['content']}%")
                params.append(user_id)
            if not conditions:
                return []

            sql = f"""
            SELECT * FROM conversation
            WHERE {" AND ".join(conditions)}
            ORDER BY updated_at DESC
            """

            with self._open_connection() as conn:
                rows = conn.execute(sql, params).fetchall()
                for row in rows:
                    conversations.append(self._row_to_conversation(row, []))
        except SQLiteError as e:
            logger.error(f"Database error while searching conversations: {e}")
            raise DatabaseOperationError(f"Database error while listing conversations: {e}") from e
        finally:
            return conversations

    def get_message(self, message_id: str) -> Message | None:
        assert message_id is not None, "Message id cannot be None"
        try:
            with self._open_connection() as conn:
                message_row = conn.execute(self.sql_scheme.get_message(), (message_id,)).fetchone()
                if message_row is None:
                    logger.debug(f"SQLStorage message {message_id} not found")
                    return None
                return self._row_to_message(message_row)
        except SQLiteError as e:
            logger.error(f"Database error while getting message {message_id}: {e}")
            raise DatabaseOperationError(f"Database error while getting message {message_id}: {e}") from e
