import os

import pytest
from openai.types import CompletionUsage
from openai.types.chat import ChatCompletion, ChatCompletionMessage
from openai.types.chat.chat_completion import Choice
from pydantic import SecretStr

from pai_llm.adapter.langchain_openai import ChatOpenAI
from pai_llm.conversation.algorithm.base import BaseAlgorithm
from pai_llm.conversation.algorithm.fifo import FIFOAlgorithm
from pai_llm.conversation.history import HistoryManager
from pai_llm.conversation.integration.langchain import with_history
from pai_llm.conversation.storage.base import BaseStorage
from pai_llm.conversation.storage.sqlite_storage import SQLiteStorage
from test.test_support import root_path
from test.unit.conversation.integration.test_support import MockStorage, MockAlgorithm


@pytest.fixture
def mock_storage() -> BaseStorage:
    return MockStorage()


@pytest.fixture
def storage() -> BaseStorage:
    return SQLiteStorage(root_path().joinpath("temp", "test.db"))


@pytest.fixture
def mock_algorithm() -> BaseAlgorithm:
    return MockAlgorithm()


@pytest.fixture
def algorithm() -> BaseAlgorithm:
    return FIFOAlgorithm()


@pytest.fixture
def mock_history_manager(mock_storage: BaseStorage, mock_algorithm: BaseAlgorithm) -> HistoryManager:
    return HistoryManager(storage=mock_storage, algorithm=mock_algorithm)


@pytest.fixture
def history_manager(
        storage: BaseStorage,
        algorithm: BaseAlgorithm
) -> HistoryManager:
    return HistoryManager(storage=storage, algorithm=algorithm)


@pytest.fixture
def mock_chat_completion() -> ChatCompletion:
    return ChatCompletion(
        id="chatcmpl-123",
        model="gpt-3.5-turbo",
        choices=[
            Choice(
                index=0,
                message=ChatCompletionMessage(
                    role="assistant",
                    content="Hello, how can I assist you today?"
                ),
                finish_reason="stop"
            )
        ],
        created=1234567890,
        object="chat.completion",
        usage=CompletionUsage(
            prompt_tokens=100,
            completion_tokens=100,
            total_tokens=200
        )
    )


def sync_client_wrapper(
        history_manager: HistoryManager,
        chat_completion: ChatCompletion
):
    conversation = history_manager.create_conversation(user_id="test_user_id")

    # Create a real llm
    real_llm = ChatOpenAI(
        model="qwen-max",
        api_key=SecretStr(os.environ["QWEN_API_KEY"]),
        base_url=os.environ["QWEN_BASE_URL"],
        temperature=0.7,
        max_tokens=102
    )

    # Wrap llm
    wrapper_llm = with_history(history_manager=history_manager)(real_llm)
    response = wrapper_llm.invoke("你是谁", conversation_id=conversation.id)

    saved_conversation = history_manager.get_conversation(conversation.id)
    assert saved_conversation is not None
    assert response is not None


def test_sync_client_wrapper_with_mock_storage(
        mock_history_manager: HistoryManager,
        mock_chat_completion: ChatCompletion
):
    sync_client_wrapper(mock_history_manager, mock_chat_completion)


def test_sync_client_wrapper_with_sqlite_storage(
        history_manager: HistoryManager,
        mock_chat_completion: ChatCompletion
):
    sync_client_wrapper(history_manager, mock_chat_completion)
