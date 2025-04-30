from typing import Dict, Any, Optional, List, get_args

from loguru import logger

from pai_llm.conversation.algorithm.base import BaseAlgorithm
from pai_llm.conversation.exceptions import StorageError, ConversationNotFoundError, MessageNotFoundError
from pai_llm.conversation.models import Conversation, Message, ToolCallContent
from pai_llm.conversation.storage.base import BaseStorage
from pai_llm.exceptions import ValidationError
from pai_llm.types.chat.completion_role import CompletionRole

DEFAULT_USER_ID = "default_user"


class HistoryManager:

    def __init__(self, storage: BaseStorage, algorithm: Optional[BaseAlgorithm] = None):
        assert storage is not None, "Storage is required"
        self.storage = storage
        self.algorithm = algorithm
        logger.debug(f"HistoryManager initialized with storage: {storage}, algorithm: {algorithm}")

    def create_conversation(
            self,
            user_id: Optional[str] = DEFAULT_USER_ID,
            metadata: Optional[Dict[str, Any]] = None
    ) -> Conversation:
        try:
            if user_id is None:
                raise ValidationError("User ID is required")
            conversation = Conversation(
                user_id=user_id,
                metadata=metadata if metadata else {},
            )
            self.storage.save_conversation(conversation)
            logger.debug(f"Conversation created for user: {user_id}, with id: {conversation.id}")
            return self.storage.get_conversation(conversation.id)
        except Exception as e:
            logger.error(f"Failed to create conversation: {e}")
            raise StorageError(f"Failed to create conversation: {e}") from e

    def list_conversations(self, user_id: str, page_no: int = 1, page_size: int = 10) -> List[Conversation]:
        if user_id is None:
            raise ValidationError("User ID is required")
        if page_no < 1 or page_size < 1:
            raise ValidationError("Page number and page size must be greater than 0")
        try:
            return self.storage.list_conversations(user_id, page_no, page_size)
        except Exception as e:
            logger.error(f"Failed to list conversations: {e}")
            raise StorageError(f"Failed to list conversations: {e}") from e

    def get_conversation(self, conversation_id: str) -> Conversation:
        if conversation_id is None:
            raise ValidationError("Conversation ID is required")
        try:
            conversation = self.storage.get_conversation(conversation_id)
            if conversation is None:
                logger.error(f"Conversation with id {conversation_id} not found")
                raise ConversationNotFoundError(f"Conversation with id {conversation_id} not found")

            if self.algorithm is not None:
                conversation.messages = self.algorithm.get_message_window(conversation.messages)
            logger.debug(f"Retrieved Conversion {conversation} with {len(conversation.messages)} messages")
            return conversation
        except ConversationNotFoundError:
            raise
        except Exception as e:
            logger.error(f"Failed to get conversation: {e}")
            raise StorageError(f"Failed to get conversation: {e}") from e

    def rename_conversation(self, conversation_id: str, new_name: str) -> bool:
        try:
            return self.storage.rename_conversation(conversation_id, new_name)
        except Exception as e:
            logger.error(f"Failed to rename conversation: {e}")
            raise StorageError(f"Failed to rename conversation: {e}") from e

    def delete_conversation(self, conversation_id: str) -> bool:
        try:
            return self.storage.delete_conversation(conversation_id)
        except Exception as e:
            logger.error(f"Failed to delete conversation: {e}")
            raise StorageError(f"Failed to delete conversation: {e}") from e

    def search_conversations(
            self,
            user_id: str,
            query: Dict[str, Any]
    ) -> List[Conversation]:
        if user_id is None:
            raise ValidationError("User ID is required")
        try:
            return self.storage.search_conversations(user_id, query)
        except Exception as e:
            logger.error(f"Failed to search conversations: {e}")
            raise StorageError(f"Failed to search conversations: {e}") from e

    def add_message(
            self,
            conversation_id: str,
            content: str,
            role: CompletionRole,
            metadata: Dict[str, Any] = None,
            tool_calls: List[ToolCallContent] = None,
            tool_call_id: str = None,
            tool_call_result: str = None
    ) -> Message:
        if conversation_id is None:
            raise ValidationError("Conversation ID is required")
        if content is not None and not isinstance(content, str):
            raise ValidationError("Content must be a string or None")
        allow_roles = get_args(CompletionRole)
        if role not in allow_roles:
            raise ValidationError(f"Invalid role: {role}, must be one of {allow_roles}")

        try:
            conversation = self.storage.get_conversation(conversation_id)
            if conversation is None:
                raise ConversationNotFoundError(f"Conversation with id {conversation_id} not found")

            message = Message(
                content=content,
                role=role,
                metadata=metadata,
                tool_calls=tool_calls,
                tool_call_id=tool_call_id,
                tool_call_result=tool_call_result
            )
            conversation.messages.append(message)
            conversation.updated_at = message.created_at
            self.storage.save_conversation(conversation)
            logger.debug(f"Message added to conversation with id: {conversation_id}")
            return message
        except ConversationNotFoundError:
            raise
        except ValidationError:
            raise
        except Exception as e:
            logger.error(f"Failed to add message: {e}")
            raise StorageError(f"Failed to add message: {e}") from e

    def get_messages(self, conversation_id: str) -> List[Message]:
        if conversation_id is None:
            raise ValidationError("Conversation ID is required")
        try:
            conversation = self.storage.get_conversation(conversation_id)
            if conversation is None:
                raise ConversationNotFoundError(f"Conversation with id {conversation_id} not found")
            return conversation.messages
        except ConversationNotFoundError:
            raise
        except Exception as e:
            logger.error(f"Failed to get messages: {e}")
            raise StorageError(f"Failed to get messages: {e}") from e

    def get_message(self, message_id: str) -> Message:
        if message_id is None:
            raise ValidationError("Message ID is required")
        try:
            message = self.storage.get_message(message_id)
            if message is not None:
                return message
            raise MessageNotFoundError(f"Message with id {message_id} not found")
        except ConversationNotFoundError:
            raise
        except Exception as e:
            logger.error(f"Failed to get message: {e}")
            raise StorageError(f"Failed to get message: {e}") from e
