import uuid
from abc import abstractmethod
from datetime import datetime

from pai_llm.conversation.models import Conversation, Message
from pai_llm.types.chat.completion_role import UserRole, AssistantRole
from test.test_support import BaseTest

user_id = "test_user_id"
conversation_id = "test_conversation_id"
conversation_name = "test_conversation_name"
conversation_new_name = "test_conversation_new_name"

message_id = "test_message_id"


class TestSQStorage(BaseTest):

    @abstractmethod
    def get_storage(self):
        pass

    def test_save_conversation(self):
        messages = [
            Message(
                id=message_id,
                role=UserRole,
                content="Hello!",
                metadata={"type": "input"}
            ),
            Message(
                id=str(uuid.uuid4()),
                role=AssistantRole,
                content="Hi there!",
                metadata={"type": "output"}
            )
        ]
        conversation = Conversation(
            id=conversation_id,
            name=conversation_name,
            user_id=user_id,
            messages=messages,
            metadata={"test_key": "test_value"},
            created_at=datetime.now(),
            updated_at=datetime.now()
        )

        self.get_storage().save_conversation(conversation)
        ret = self.get_storage().get_conversation(conversation.id, 10)
        self.assertEqual(ret.id, conversation.id)
        self.assertEqual(ret.name, conversation.name)
        self.assertEqual(ret.user_id, conversation.user_id)
        self.assertEqual(ret.metadata, conversation.metadata)

    def test_get_conversation(self):
        conversation = self.get_storage().get_conversation(conversation_id)
        self.assertIsNotNone(conversation)
        self.assertEqual(conversation.id, conversation_id)
        self.assertEqual(conversation.name, conversation_name)
        self.assertEqual(len(conversation.messages), 2)

    def test_list_conversations(self):
        conversations1 = self.get_storage().list_conversations(user_id, 1, 10)
        conversations2 = self.get_storage().list_conversations("test_user_id_not_exists", 1, 10)
        assert len(conversations1) > 0
        assert len(conversations2) == 0

    def test_rename_conversation(self):
        success = self.get_storage().rename_conversation(conversation_id, conversation_new_name)
        self.assertTrue(success)
        conversation = self.get_storage().get_conversation(conversation_id)
        self.assertEqual(conversation.name, conversation_new_name)

    def test_delete_conversation(self):
        cid = str(uuid.uuid4())
        conversation = Conversation(
            id=cid,
            name=conversation_name,
            user_id=user_id,
            metadata={"test_key": "test_value"},
            created_at=datetime.now(),
            updated_at=datetime.now()
        )
        self.get_storage().save_conversation(conversation)
        conversation = self.get_storage().get_conversation(cid)
        assert conversation is not None
        success = self.get_storage().delete_conversation(cid)
        self.assertTrue(success)
        conversation = self.get_storage().get_conversation(cid)
        self.assertIsNone(conversation)

    def test_search_conversations(self):
        conversations1 = self.get_storage().search_conversations(user_id, {"name": conversation_name})
        conversations2 = self.get_storage().search_conversations(user_id, {"name": conversation_new_name})
        conversations3 = self.get_storage().search_conversations(user_id, {"metadata": {"test_key": "test_value"}})
        conversations4 = self.get_storage().search_conversations(user_id, {"content": "Hello!"})
        assert len(conversations1) == 0
        assert len(conversations2) > 0
        assert len(conversations3) > 0
        assert len(conversations4) > 0

    def test_get_message(self):
        message = self.get_storage().get_message(message_id)
        self.assertIsNotNone(message)
        self.assertEqual(message.id, message_id)
        self.assertEqual(message.content, "Hello!")
        self.assertEqual(message.role, UserRole)
        self.assertEqual(message.metadata, {"type": "input"})
