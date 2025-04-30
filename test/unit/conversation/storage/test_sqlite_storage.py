from pai_llm.conversation.storage.sqlite_storage import SQLiteStorage
from test import test_support
from test.unit.conversation.storage.test_sql_storage import TestSQStorage

db_file_path = test_support.root_path().joinpath("temp", "test.db")
storage = SQLiteStorage(db_file_path=db_file_path, max_messages=10)


class TestSQLiteStorage(TestSQStorage):

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
