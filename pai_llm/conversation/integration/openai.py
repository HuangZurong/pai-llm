from functools import wraps
from typing import TypeVar, Sequence, Union, Any, cast, get_args, List, Optional, Dict, Callable

from commons_lang import object_utils
from commons_lang.text import string_utils
from loguru import logger
from openai import OpenAI, AsyncOpenAI
from openai.types.chat import (
    ChatCompletion,
    ChatCompletionMessage,
    ChatCompletionMessageParam,
    ChatCompletionSystemMessageParam,
    ChatCompletionAssistantMessageParam,
    ChatCompletionUserMessageParam,
    ChatCompletionDeveloperMessageParam,
    ChatCompletionToolMessageParam,
    ChatCompletionFunctionMessageParam,
    ChatCompletionContentPartTextParam,
    ChatCompletionContentPartImageParam
)

from pai_llm.conversation.algorithm.base import BaseAlgorithm
from pai_llm.conversation.exceptions import UnsupportedContentTypeError
from pai_llm.conversation.history import HistoryManager
from pai_llm.conversation.models import (
    Message,
    MessageContent,
    TextContent,
    ImageContent,
    ToolCallContent
)
from pai_llm.conversation.storage.base import BaseStorage
from pai_llm.exceptions import ConfigurationError, ValidationError
from pai_llm.types.chat.completion_role import (
    CompletionRole,
    SystemRole,
    AssistantRole,
    UserRole,
    DeveloperRole,
    ToolRole,
    FunctionRole
)

T = TypeVar("T", bound=OpenAI | AsyncOpenAI)


def _convert_to_message(
        openai_message: Union[Dict[str, Any], ChatCompletionMessage]
) -> Message:
    try:
        if isinstance(openai_message, Dict):
            role = object_utils.get(openai_message, "role")
            allow_roles = get_args(CompletionRole)
            if role not in allow_roles:
                raise ValidationError(f"Invalid role: {role}, must be one of {allow_roles}")

            content = object_utils.get(openai_message, "content")

            # Handle tool calls
            tool_calls = None
            openai_tool_calls = object_utils.get(openai_message, "tool_calls")
            if object_utils.is_not_none(openai_tool_calls):
                tool_calls = [
                    ToolCallContent(
                        id=object_utils.get(openai_tool_call, "id"),
                        type=object_utils.get(openai_tool_call, "type"),
                        function=object_utils.get(openai_tool_call, "function"),
                    ) for openai_tool_call in openai_tool_calls
                ]

            # Handle tool calls result
            tool_call_id = object_utils.get(openai_message, "tool_call_id")
            tool_call_result: str | None = None
            if ToolRole == role and string_utils.is_not_blank(tool_call_id):
                tool_call_result = object_utils.get(openai_message, "content")

            # Handle multimodal content
            if isinstance(content, List):
                processed_contents: List[MessageContent] = []
                for item in content:
                    t = object_utils.get(item, "type")
                    if "text" == t:
                        text = object_utils.get(item, "text")
                        processed_contents.append(TextContent(text))
                    elif "image_url" == t:
                        image_url = object_utils.get(item, "image_url", {})
                        url = object_utils.get(image_url, "url")
                        detail = object_utils.get(image_url, "detail")
                        processed_contents.append(
                            ImageContent(
                                url=url,
                                detail=detail
                            )
                        )
                    else:
                        logger.warning(f"Unsupported content type: {t}")
                content = processed_contents

            return Message(
                role=role,
                content=content,
                tool_calls=tool_calls,
                tool_call_id=tool_call_id,
                tool_call_result=tool_call_result
            )
        else:
            role = object_utils.get(openai_message, "role")
            allow_roles = get_args(CompletionRole)
            if role not in allow_roles:
                raise ValidationError(f"Invalid role: {role}, must be one of {allow_roles}")

            content = object_utils.get(openai_message, "content")

            # Handle tool calls
            tool_calls = None
            openai_tool_calls = object_utils.get(openai_message, "tool_calls")
            if object_utils.is_not_none(openai_tool_calls):
                tool_calls = [
                    ToolCallContent(
                        id=object_utils.get(openai_tool_call, "id"),
                        type=object_utils.get(openai_tool_call, "type"),
                        function=object_utils.get(openai_tool_call, "function"),
                    ) for openai_tool_call in openai_tool_calls
                ]

            # Handle tool calls result
            tool_call_id = object_utils.get(openai_message, "tool_call_id")
            tool_call_result: str | None = None
            if ToolRole == role and string_utils.is_not_blank(tool_call_id):
                tool_call_result = object_utils.get(openai_message, "content")

            return Message(
                role=cast(CompletionRole, role),
                content=content,
                tool_calls=tool_calls,
                tool_call_id=tool_call_id,
                tool_call_result=tool_call_result
            )
    except Exception as e:
        if isinstance(e, ValidationError):
            raise
        logger.error(f"Failed to convert OpenAI message to message: {e}")
        raise ValidationError(f"Failed to convert OpenAI message to message: {e}") from e


def _convert_to_openai_messages(
        messages: Sequence[Message]
) -> List[ChatCompletionMessageParam]:
    openai_message: List[ChatCompletionMessageParam] = []

    for message in messages:
        content = _convert_content_to_openai_format(content=message.content)
        role = message.role

        if SystemRole == role:
            openai_message.append(
                ChatCompletionSystemMessageParam(
                    role="system",
                    content=cast(str, content)
                )
            )
        elif AssistantRole == role:
            message_param: Dict[str, Any] = {"role": AssistantRole}
            if object_utils.is_not_none(content):
                message_param["content"] = cast(str, content)
            tool_calls = object_utils.get(message, "tool_calls")
            if object_utils.is_not_none(tool_calls):
                message_param["tool_calls"] = [
                    {
                        "id": tool_call.id,
                        "type": tool_call.type,
                        "function": tool_call.function
                    } for tool_call in tool_calls
                ]
            openai_message.append(
                cast(ChatCompletionAssistantMessageParam, message_param)
            )
        elif UserRole == role:
            if isinstance(content, str):
                openai_message.append(
                    ChatCompletionUserMessageParam(
                        role="user",
                        content=cast(str, content)
                    )
                )
            else:
                content_parts = cast(
                    List[
                        Union[
                            ChatCompletionContentPartTextParam,
                            ChatCompletionContentPartImageParam
                        ]
                    ],
                    content
                )
                openai_message.append(
                    ChatCompletionUserMessageParam(
                        role="user",
                        content=content_parts
                    )
                )
        elif DeveloperRole == role:
            openai_message.append(
                ChatCompletionDeveloperMessageParam(
                    role="developer",
                    content=cast(str, content)
                )
            )
        elif ToolRole == role:
            openai_message.append(
                ChatCompletionToolMessageParam(
                    role="tool",
                    content=cast(str, content),
                    tool_call_id=message.tool_call_id or ""
                )
            )
        elif FunctionRole == role:
            openai_message.append(
                ChatCompletionFunctionMessageParam(
                    role="function",
                    content=cast(str, content)
                    # TODO 考虑如何填充
                    # name=message.tool_call_id or ""
                )
            )

    return openai_message


def _convert_content_to_openai_format(
        content: Union[str, List[MessageContent], None]
) -> Union[str, List[Dict[str, Any]], None]:
    if content is None:
        return None
    if isinstance(content, str):
        return content

    openai_format: List[Dict[str, Any]] = []
    for item in content:
        if isinstance(item, TextContent):
            openai_format.append({"type": "text", "text": item.text})
        elif isinstance(item, ImageContent):
            image_url: Dict[str, Any] = {"url": item.url}
            openai_format.append(
                {
                    "type": "image_url",
                    "image_url": image_url
                }
            )
        else:
            raise UnsupportedContentTypeError(f"Unsupported content type: {type(item)}")
    return openai_format


def with_history(
        storage: Optional[BaseStorage] = None,
        algorithm: Optional[BaseAlgorithm] = None,
        history_manager: Optional[HistoryManager] = None
) -> Callable[[T], T]:
    if not storage and not history_manager:
        raise ConfigurationError("Either storage or history_manager must be provided.")

    manager = history_manager or HistoryManager(
        storage=cast(BaseStorage, storage),
        algorithm=algorithm
    )

    def decorator(client: T) -> T:
        nonlocal manager
        original_create = client.chat.completions.create

        def _prepare_messages(
                conversation_id: str,
                openai_messages: Sequence[
                    Union[
                        Dict[str, Any],
                        ChatCompletionMessage
                    ]
                ]
        ) -> List[ChatCompletionMessageParam]:

            conversation = manager.get_conversation(conversation_id)
            converted_messages = [_convert_to_message(message) for message in openai_messages]

            if conversation is None:
                return _convert_to_openai_messages(converted_messages)

            system_message = next(
                (message for message in converted_messages if message.role == SystemRole),
                None
            )

            tool_responses = [
                message for message in converted_messages
                if message.role == ToolRole
            ]

            assistant_tools_pairs = []
            if tool_responses:
                for tool_response in tool_responses:
                    tool_call_id = tool_response.tool_call_id
                    if not tool_call_id:
                        continue
                    assistant_with_tool_call = None
                    for message in conversation.messages:
                        if AssistantRole == message.role and message.tool_call_id:
                            for tc in message.tool_calls:
                                if tc.id == tool_call_id:
                                    assistant_with_tool_call = message
                                    break
                            if assistant_with_tool_call:
                                break

                    if assistant_with_tool_call:
                        assistant_tools_pairs.append(
                            (assistant_with_tool_call, tool_response)
                        )
            else:
                for message in conversation.messages:
                    if system_message and SystemRole == message.role:
                        continue
                    converted_messages.append(message)

            # Prepare conversation messages
            conversation_messages: List[Message] = []

            # Add system message to the beginning of the conversation messages
            if system_message:
                conversation_messages.insert(0, system_message)

            for message in converted_messages:
                # Skip system message if it was handled
                if (
                        system_message
                        and SystemRole == message.role
                        and message.content == system_message.content
                ):
                    continue

                # Skip tool responses that we're handling
                if ToolRole == message.role and message.tool_call_id:
                    if any(message.tool_call_id == pair[1].tool_call_id for pair in assistant_tools_pairs):
                        continue

                conversation_messages.append(message)

            for assistant_message, tool_message in assistant_tools_pairs:
                conversation_messages.append(assistant_message)
                conversation_messages.append(tool_message)

            return _convert_to_openai_messages(conversation_messages)

        def _prepare_content_for_storage(content: Union[str, List[MessageContent], None]) -> str:
            if content is None:
                return ""

            if isinstance(content, str):
                return content

            text_paris = []
            for item in content:
                if isinstance(item, TextContent):
                    text_paris.append(item.text)
                elif isinstance(item, ImageContent):
                    text_paris.append(f"[Image: {item.url}]")
                elif isinstance(item, ToolCallContent):
                    text_paris.append(f"[ToolCall: {item.type} - {item.id}]")
                else:
                    logger.warning(f"Unsupported content type: {type(item)}")

            return " ".join(text_paris) if text_paris else ""

        def chat_completions_create(
                conversation_id: str,
                openai_messages: Sequence[
                    Union[
                        Dict[str, Any],
                        ChatCompletionMessage
                    ]
                ],
                response: Any,
        ) -> ChatCompletion:
            if not isinstance(response, ChatCompletion):
                raise TypeError(f"Expected ChatCompletion, got {type(response)}")

            for openai_message in openai_messages:
                converted_message = _convert_to_message(openai_message)
                content_for_storage = _prepare_content_for_storage(converted_message.content)

                manager.add_message(
                    conversation_id=conversation_id,
                    content=content_for_storage,
                    role=converted_message.role,
                    metadata={"type": "input"},
                    tool_calls=converted_message.tool_calls,
                    tool_call_id=converted_message.tool_call_id,
                    tool_call_result=converted_message.tool_call_result
                )

            for choice in response.choices:
                if isinstance(choice.message, ChatCompletionMessage):
                    converted_message = _convert_to_message(choice.message)
                    content_for_storage = _prepare_content_for_storage(converted_message.content)
                    manager.add_message(
                        conversation_id=conversation_id,
                        content=content_for_storage,
                        role=converted_message.role,
                        metadata={
                            "type": "output",
                            "model": response.model,
                            "usage": response.usage.model_dump() if response.usage else None,
                            "finish_reason": choice.finish_reason
                        },
                        tool_calls=converted_message.tool_calls,
                        tool_call_id=converted_message.tool_call_id,
                        tool_call_result=converted_message.tool_call_result
                    )

            return response

        @wraps(original_create)
        def sync_chat_completions_create(
                *args: Any,
                conversation_id: Optional[str],
                **kwargs: Any
        ) -> ChatCompletion:
            if not conversation_id:
                conversation = manager.create_conversation()
                conversation_id = conversation.id

            openai_messages = kwargs.get("messages", [])
            prepared_messages = _prepare_messages(conversation_id, openai_messages)
            kwargs["messages"] = prepared_messages

            response = original_create(*args, **kwargs)
            return chat_completions_create(conversation_id, prepared_messages, response)

        @wraps(original_create)
        async def async_chat_completions_create(
                *args: Any,
                conversation_id: Optional[str],
                **kwargs: Any
        ) -> ChatCompletion:
            if not conversation_id:
                conversation = manager.create_conversation()
                conversation_id = conversation.id

            openai_messages = kwargs.get("messages", [])
            prepared_messages = _prepare_messages(conversation_id, openai_messages)
            kwargs["messages"] = prepared_messages

            response = await original_create(*args, **kwargs)
            return chat_completions_create(conversation_id, prepared_messages, response)

        if isinstance(client, OpenAI):
            client.chat.completions.create = sync_chat_completions_create
        elif isinstance(client, AsyncOpenAI):
            client.chat.completions.create = async_chat_completions_create
        else:
            raise TypeError(f"Expected OpenAI or AsyncOpenAI, got {type(client)}")

        return client

    return decorator
