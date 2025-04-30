import typing
from abc import ABC, abstractmethod

from pai_llm.conversation.models import Conversation, Message


class BaseAlgorithm(ABC):

    def __init__(self, max_messages: typing.Optional[int] = 100):
        self.max_messages = max_messages

    @abstractmethod
    def add_message(self, conversation: Conversation, new_message: Message) -> None:
        pass

    @abstractmethod
    def get_message_window(self, messages: typing.List[Message]) -> typing.List[Message]:
        pass
