import os
from typing import Dict, List

from langchain import LLMChain
from langchain_core.messages import SystemMessage, AIMessage, HumanMessage
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from loguru import logger
from pydantic import SecretStr

llm = ChatOpenAI(
    model="qwen-max",
    api_key=SecretStr(os.environ["QWEN_API_KEY"]),
    base_url=os.environ["QWEN_BASE_URL"],
    temperature=0.7,
    max_tokens=102
)


def convert_chat_history_to_prompt_sting(history: List[Dict]) -> str:
    return "\n".join([f"{item['role']}: {item['content']}" for item in history])


def convert_chat_history_to_chat_prompt(history: List[Dict]):
    prompt = SystemMessage(content="You are a useful assistant")
    for item in history:
        if item["role"] == "user":
            prompt += HumanMessage(content=item["content"])
        elif item["role"] == "bor":
            prompt += AIMessage(content=item["content"])
    prompt += "{input}"
    return prompt


def get_history():
    return {
        "query": "My name is xxx",
        "chat_history": [
            {"role": "user", "content": "Hi, how are you?"},
            {"role": "bot", "content": "I am fine. Thank you. And you?"},
        ]
    }


def test_single_round_prompt():
    template = \
        """
You are a useful assistant.

chat_history: 
{chat_history}

user: 
{input}
        """

    prompt = PromptTemplate(
        template=template,
        input_variables=["chat_history", "input"]
    )

    llm_chain = LLMChain(
        llm=llm,
        prompt=prompt,
        verbose=True
    )

    history = get_history()
    query = history["query"]
    chat_history = convert_chat_history_to_prompt_sting(history["chat_history"])

    response = llm_chain.invoke({
        "input": query,
        "chat_history": chat_history
    })
    logger.info(response)


def test_multi_round_prompt():
    history = get_history()
    query = history["query"]
    chat_prompt = convert_chat_history_to_chat_prompt(history["chat_history"])

    llm_chain = LLMChain(
        llm=llm,
        prompt=chat_prompt,
        verbose=True
    )
    response = llm_chain.invoke(query)
    logger.info(chat_prompt)
    logger.info(response)
