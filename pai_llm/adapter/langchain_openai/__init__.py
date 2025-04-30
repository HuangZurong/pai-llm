from langchain_openai import ChatOpenAI as RawChatOpenAI
from langchain_openai.chat_models.base import BaseChatOpenAI


class ChatOpenAI(RawChatOpenAI):
    def get_num_tokens_from_messages(self, *args, **kwargs) -> int:
        model, encoding = self._get_encoding_model()
        if model.startswith("cl100k_base"):
            return super(BaseChatOpenAI, self).get_num_tokens_from_messages(*args, **kwargs)
        else:
            return super().get_num_tokens_from_messages(*args, **kwargs)


__all__ = [
    "ChatOpenAI"
]
