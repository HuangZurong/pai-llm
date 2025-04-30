from pai_llm.conversation.storage.mysql_storage import MySQLStorage, MySQLConfig
from test.unit.conversation.storage.test_sql_storage import TestSQStorage

mysql_config = MySQLConfig(
    host="localhost",
    port=3306,
    user="root",
    password="e9Epsdji1192@",
    database="pai_llm",
    charset="utf8mb4",
    autocommit=False,
    connect_timeout=10,
    pool_size=5
)

storage = MySQLStorage(mysql_config, init_tables=True)


class TestMySQLStorage(TestSQStorage):

    def get_storage(self):
        return storage

    def test_save_conversation(self):
        super().test_save_conversation()

    def test_get_conversation(self):
        super().test_get_conversation()

    def test_list_conversations(self):
        super().test_list_conversations()

    def test_rename_conversation(self):
        super().test_rename_conversation()

    def test_delete_conversation(self):
        super().test_delete_conversation()

    def test_search_conversations(self):
        super().test_search_conversations()

    def test_get_message(self):
        super().test_get_message()
