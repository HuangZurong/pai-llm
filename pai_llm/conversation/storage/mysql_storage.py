from contextlib import contextmanager
from typing import Optional

from commons_lang import object_utils
from loguru import logger
from mysql.connector import pooling, errorcode
from mysql.connector.cursor import MySQLCursor

from pai_llm.conversation.exceptions import DatabaseOperationError
from pai_llm.conversation.storage.sql_storage import (
    SQLConnection,
    SQLSchema,
    SQLStorage,
    MySQLError,
)


class MySQLConfig:
    def __init__(
            self,
            host: str = "localhost",
            port: int = 3306,
            user: str = "root",
            password: str = "",
            database: str = None,
            charset: str = "utf8mb4",
            autocommit: bool = False,
            connect_timeout: int = 10,
            pool_size: int = 5
    ):
        """
        MySQL connection parameters configuration class

        Parameters:
            host: Host address
            port: Port number
            user: Username
            password: Password
            database: Database name
            charset: Character set
            autocommit: Whether to auto-commit transactions
            connect_timeout: Connection timeout in seconds
            pool_size: Connection pool size
        """
        self.host = host
        self.port = port
        self.user = user
        self.password = password
        self.database = database
        self.charset = charset
        self.autocommit = autocommit
        self.connect_timeout = connect_timeout
        self.pool_size = pool_size

    def get_connection_params(self):
        """Returns connection parameters as a dictionary"""
        params = {
            "host": self.host,
            "port": self.port,
            "user": self.user,
            "password": self.password,
            "charset": self.charset,
            "connect_timeout": self.connect_timeout
        }

        # Only add database when it's not None
        if self.database:
            params["database"] = self.database

        return params

    def get_pool_params(self):
        """Returns connection pool parameters as a dictionary"""
        params = self.get_connection_params()
        params["pool_size"] = self.pool_size
        return params


class MySQLSchema(SQLSchema):

    def create_conversation_table(self) -> str:
        return """
            CREATE TABLE IF NOT EXISTS conversation (
                id VARCHAR(255) PRIMARY KEY,
                name VARCHAR(255) NOT NULL,
                user_id VARCHAR(255) NOT NULL,
                metadata TEXT,
                created_at TIMESTAMP NOT NULL,
                updated_at TIMESTAMP NOT NULL
            );
        """

    def create_message_table(self) -> str:
        return """
            CREATE TABLE IF NOT EXISTS message (
                id VARCHAR(255) PRIMARY KEY,
                conversation_id VARCHAR(255) NOT NULL,
                user_id VARCHAR(255) NOT NULL,
                role VARCHAR(50) NOT NULL,
                content TEXT NOT NULL,
                metadata TEXT,
                token_count INT,
                message_index INT,
                created_at TIMESTAMP NOT NULL,
                FOREIGN KEY (conversation_id) REFERENCES conversation(id) ON DELETE CASCADE
            );
        """

    def insert_conversation(self) -> str:
        return """
            INSERT INTO conversation (id, name, user_id, metadata, created_at, updated_at)
            VALUES (%s, %s, %s, %s, %s, %s)
            ON DUPLICATE KEY 
            UPDATE 
                name = VALUES(name),
                user_id = VALUES(user_id),
                metadata = VALUES(metadata),
                created_at = VALUES(created_at),
                updated_at = VALUES(updated_at);
        """

    def get_conversation(self) -> str:
        return """
            SELECT * FROM conversation WHERE id = %s
        """

    def list_conversations(self) -> str:
        return """
            SELECT * FROM conversation WHERE user_id = %s ORDER BY updated_at DESC LIMIT %s OFFSET %s
        """

    def rename_conversation(self) -> str:
        return """
            UPDATE conversation SET name = %s WHERE id = %s
        """

    def delete_conversation(self) -> str:
        return """
           DELETE FROM conversation WHERE id = %s
       """

    def insert_message(self) -> str:
        return """
            INSERT INTO message (id, conversation_id, user_id, role, content, metadata, token_count, message_index, created_at)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
        """

    def get_message(self) -> str:
        return """
            SELECT * FROM message WHERE id = %s
        """

    def get_messages(self) -> str:
        return """
            SELECT * FROM message WHERE conversation_id = %s ORDER BY message_index ASC
        """

    def list_messages(self) -> str:
        return """
            SELECT * FROM message WHERE conversation_id = %s ORDER BY message_index ASC LIMIT %s
        """

    def delete_messages(self) -> str:
        return """
            DELETE FROM message WHERE conversation_id = %s
        """


class MySQLStorage(SQLStorage):

    def __init__(
            self,
            mysql_config: MySQLConfig,
            init_tables: bool = True,
            max_messages: Optional[int] = 100
    ):
        assert mysql_config is not None, "MySQLConfig cannot be None"
        assert max_messages is None or max_messages > 0, "Max messages must be greater than 0"

        super().__init__(MySQLSchema(), max_messages)
        try:
            self.pool = pooling.MySQLConnectionPool(
                pool_name="pai_llm_pool",
                **mysql_config.get_pool_params()
            )
        except MySQLError as e:
            errno = object_utils.get(e, "errno")
            if errno == errorcode.ER_ACCESS_DENIED_ERROR:
                raise DatabaseOperationError("MySQL connection failed: Invalid username or password")
            elif errno == errorcode.ER_BAD_DB_ERROR:
                raise DatabaseOperationError("MySQL connection failed: Database does not exist")
            else:
                raise DatabaseOperationError(f"MySQL connection failed: {e}") from e
        except Exception as e:
            logger.error(f"Failed to create MySQL connection pool: {e}")
            raise DatabaseOperationError(f"Failed to create MySQL connection pool: {e}") from e

        if init_tables:
            try:
                with self._open_connection() as conn:
                    self._init_tables(conn)
            except MySQLError as e:
                logger.error(f"MySQLStorage initialization failed: {e}")
                raise DatabaseOperationError(f"MySQLStorage initialization failed: {e}") from e

    @staticmethod
    def _dict_factory(cursor, row):
        d = {}
        for idx, col in enumerate(cursor.description):
            d[col[0]] = row[idx]
        return d

    @contextmanager
    def _open_connection(self):
        conn: SQLConnection | None = None
        cursor: MySQLCursor | None = None

        def execute(*args, **kwargs) -> MySQLCursor:
            nonlocal cursor
            cursor.execute(*args, **kwargs)
            return cursor

        try:
            conn = self.pool.get_connection()
            cursor = conn.cursor(dictionary=True)
            cursor.execute("SET FOREIGN_KEY_CHECKS = 1;")
            conn.execute = execute
            yield conn
        except Exception as e:
            logger.error(f"Failed to get MySQL connection: {e}")
            if conn:
                conn.rollback()
            raise DatabaseOperationError(f"Failed to get MySQL connection: {e}") from e
        finally:
            if cursor:
                cursor.close()
            if conn:
                conn.close()

    def _get_param_placeholder(self) -> str:
        return "%s"

    def _start_transaction(self, conn: SQLConnection):
        conn.execute("START TRANSACTION")
