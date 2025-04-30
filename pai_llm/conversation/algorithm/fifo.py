import typing

from pai_llm.conversation.algorithm.base import BaseAlgorithm
from pai_llm.conversation.models import Conversation, Message


class FIFOAlgorithm(BaseAlgorithm):

    def __init__(self, max_messages: typing.Optional[int] = 100):
        assert max_messages is not None and max_messages > 0, "max_messages must be greater than 0"
        super().__init__(max_messages)

    def add_message(self, conversation: Conversation, new_message: Message) -> None:
        conversation.add_message(new_message)
        conversation.messages = self.get_message_window(conversation.messages)

    def get_message_window(self, messages: typing.List[Message]) -> typing.List[Message]:
        if len(messages) > self.max_messages:
            return messages[-self.max_messages:]
        return messages
