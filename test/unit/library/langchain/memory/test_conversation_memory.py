import os

from langchain.chains import ConversationChain
from langchain.memory import (
    ConversationStringBufferMemory,
    ConversationBufferWindowMemory,
    ConversationTokenBufferMemory,
    ConversationSummaryBufferMemory
)
from pydantic import SecretStr

from pai_llm.adapter.langchain_openai import ChatOpenAI

llm = ChatOpenAI(
    model="qwen-max",
    api_key=SecretStr(os.environ["QWEN_API_KEY"]),
    base_url=os.environ["QWEN_BASE_URL"],
    temperature=0.7,
)


def test_conversation_string_buffer_memory():
    memory = ConversationStringBufferMemory()
    conversation = ConversationChain(llm=llm, memory=memory, verbose=True)
    response1 = conversation.predict(input="我是hzr，你是谁？")
    response2 = conversation.predict(input="今天杭州天气")
    response3 = conversation.predict(input="请说出我的名字")
    assert response1 is not None
    assert response2 is not None
    assert response3 is not None


def test_conversation_buffer_window_memory():
    memory = ConversationBufferWindowMemory(k=1)
    conversation = ConversationChain(llm=llm, memory=memory, verbose=True)
    response1 = conversation.predict(input="我是hzr，你是谁？")
    response2 = conversation.predict(input="今天杭州天气")
    response3 = conversation.predict(input="请说出我的名字")
    assert response1 is not None
    assert response2 is not None
    assert response3 is not None


def test_conversation_token_buffer_memory():
    memory = ConversationTokenBufferMemory(llm=llm, max_token_limit=100)
    conversation = ConversationChain(llm=llm, memory=memory, verbose=True)
    response1 = conversation.predict(input="我是hzr，你是谁？")
    response2 = conversation.predict(input="今天杭州天气")
    response3 = conversation.predict(input="请Speak English")
    response4 = conversation.predict(input="请说出我的名字")
    assert response1 is not None
    assert response2 is not None
    assert response3 is not None
    assert response4 is not None


def test_conversation_summary_buffer_memory():
    memory = ConversationSummaryBufferMemory(llm=llm, max_token_limit=100)
    conversation = ConversationChain(llm=llm, memory=memory, verbose=True)
    response1 = conversation.predict(input="我是hzr，你是谁？")
    response2 = conversation.predict(input="今天杭州天气")
    response3 = conversation.predict(input="请Speak English")
    response4 = conversation.predict(input="请说出我的名字")
    assert response1 is not None
    assert response2 is not None
    assert response3 is not None
    assert response4 is not None
