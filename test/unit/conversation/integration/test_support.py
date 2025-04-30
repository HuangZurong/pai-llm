import typing

from pai_llm.conversation.algorithm.base import BaseAlgorithm
from pai_llm.conversation.models import Conversation, Message
from pai_llm.conversation.storage.base import BaseStorage


class MockStorage(BaseStorage):

    def __init__(self):
        super().__init__()
        self.conversations: typing.Dict[str, Conversation] = {}

    def save_conversation(self, conversation: Conversation):
        self.conversations[conversation.id] = conversation

    def get_conversation(
            self,
            conversation_id: str,
            messages_limit: typing.Optional[int] = None
    ) -> typing.Optional[Conversation]:
        conversation = self.conversations.get(conversation_id)
        if not conversation:
            return None

        if messages_limit is not None and len(conversation.messages) > messages_limit:
            conversation_copy = Conversation(
                conversation.id,
                conversation.name,
                conversation.user_id,
                metadata=conversation.metadata,
                messages=conversation.messages[:messages_limit],
                created_at=conversation.created_at,
                updated_at=conversation.updated_at
            )
            return conversation_copy

        return conversation

    def list_conversations(
            self,
            user_id: str,
            page_no: int = 1,
            page_size: int = 10
    ) -> typing.List[Conversation]:
        offset = (page_no - 1) * page_size
        limit = page_size
        return list(self.conversations.values())[offset:offset + limit]

    def rename_conversation(self, conversation_id: str, new_name: str) -> bool:
        find = next(
            (
                conversation
                for conversation in self.conversations.values()
                if conversation.id == conversation_id
            ),
            None
        )
        if not find:
            return False
        find.name = new_name
        return True

    def delete_conversation(self, conversation_id: str) -> bool:
        if conversation_id in self.conversations:
            del self.conversations[conversation_id]
            return True
        return False

    def search_conversations(self, user_id: str, query: typing.Dict[str, any]) -> typing.List[Conversation]:
        return list(self.conversations.values())

    def get_message(self, message_id: str) -> Message | None:
        for conversation in self.conversations.values():
            for message in conversation.messages:
                if message.id == message_id:
                    return message
        return None


class MockAlgorithm(BaseAlgorithm):

    def add_message(self, conversation: Conversation, new_message: Message) -> None:
        conversation.add_message(new_message)

    def get_message_window(self, messages: typing.List[Message]) -> typing.List[Message]:
        return messages
