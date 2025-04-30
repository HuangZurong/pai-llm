from abc import ABC, abstractmethod
from typing import List, Dict, Optional

from pai_llm.conversation.models import (
    Conversation,
    Message
)


class BaseStorage(ABC):

    def __init__(self, max_messages: Optional[int] = 100):
        self.max_messages = max_messages

    @abstractmethod
    def save_conversation(self, conversation: Conversation):
        pass

    @abstractmethod
    def get_conversation(
            self,
            conversation_id: str,
            messages_limit: Optional[int] = None
    ) -> Optional[Conversation]:
        pass

    @abstractmethod
    def list_conversations(
            self,
            user_id: str,
            page_no: int = 1,
            page_size: int = 10
    ) -> List[Conversation]:
        pass

    @abstractmethod
    def rename_conversation(self, conversation_id: str, new_name: str) -> bool:
        pass

    @abstractmethod
    def delete_conversation(self, conversation_id: str) -> bool:
        pass

    @abstractmethod
    def search_conversations(
            self,
            user_id: str,
            query: Dict[str, any]
    ) -> List[Conversation]:
        pass

    @abstractmethod
    def get_message(self, message_id: str) -> Message | None:
        pass
