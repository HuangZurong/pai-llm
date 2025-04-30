from typing import Optional, TypeVar, Callable, cast

from pai_llm.adapter.langchain_openai import ChatOpenAI
from pai_llm.conversation.algorithm.base import BaseAlgorithm
from pai_llm.conversation.history import HistoryManager
from pai_llm.conversation.integration.openai import with_history as openai_with_history
from pai_llm.conversation.storage.base import BaseStorage

T = TypeVar("T", bound=ChatOpenAI)


def with_history(
        storage: Optional[BaseStorage] = None,
        algorithm: Optional[BaseAlgorithm] = None,
        history_manager: Optional[HistoryManager] = None
) -> Callable[[T], T]:
    def decorator(llm: T) -> T:
        if llm.client:
            llm.root_client = openai_with_history(
                storage=cast(BaseStorage, storage),
                algorithm=algorithm,
                history_manager=history_manager
            )(llm.root_client)
            llm.client = llm.root_client.chat.completions
        if llm.async_client:
            llm.root_async_client = openai_with_history(
                storage=cast(BaseStorage, storage),
                algorithm=algorithm,
                history_manager=history_manager
            )(llm.root_async_client)
            llm.async_client = llm.root_async_client.chat.completions
        return llm

    return decorator
