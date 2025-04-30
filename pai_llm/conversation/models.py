import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional

from pai_llm.types.chat.completion_role import CompletionRole


@dataclass
class MessageContent:
    pass


## Basic MessageContent ##
@dataclass
class TextContent(MessageContent):
    text: str


@dataclass
class ImageContent(MessageContent):
    url: str
    format: str = field(default_factory=lambda: "png")
    detail: Optional[str] = None


@dataclass
class AudioContent(MessageContent):
    url: str
    detail: Optional[str] = None


@dataclass
class VideoContent(MessageContent):
    url: str
    detail: Optional[str] = None


@dataclass
class DocumentContent(MessageContent):
    url: str
    format: str = field(default_factory=lambda: "pdf")
    detail: Optional[str] = None


@dataclass
class ToolCallContent(MessageContent):
    id: str
    type: str
    function: Dict[str, any] = None


## Structure MessageContent ##
@dataclass
class JsonContent(MessageContent):
    data: Dict[str, any] = field(default_factory=dict),
    schema: Optional[str] = None


@dataclass
class XmlContent(MessageContent):
    pass


@dataclass
class MarkdownContent(MessageContent):
    pass


@dataclass
class HtmlContent(MessageContent):
    pass


@dataclass
class CsvContent(MessageContent):
    pass


## Fixed and Interactive MessageContent ##
@dataclass
class MultimodalContent(MessageContent):
    pass


@dataclass
class InteractiveContent(MessageContent):
    pass


@dataclass
class StreamContent(MessageContent):
    pass


@dataclass
class ChartContent(MessageContent):
    pass


@dataclass
class MapContent(MessageContent):
    pass


## Specified Application MessageContent ##
@dataclass
class CodeContent(MessageContent):
    pass


@dataclass
class TranslationContent(MessageContent):
    pass


@dataclass
class EmbeddingContent(MessageContent):
    pass


@dataclass
class Message:
    role: CompletionRole
    content: str | MessageContent
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    tool_calls: Optional[List[ToolCallContent]] = None
    tool_call_id: Optional[str] = None
    tool_call_result: Optional[str] = None
    metadata: Dict[str, any] = field(default_factory=dict)
    token_count: Optional[int] = None
    created_at: datetime = field(default_factory=lambda: datetime.now())


@dataclass
class Conversation:
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = field(default_factory=lambda: "new conversation")
    user_id: str = field(default_factory=str)
    metadata: Dict[str, any] = field(default_factory=dict)
    messages: List[Message] = field(default_factory=list)
    created_at: datetime = field(default_factory=lambda: datetime.now())
    updated_at: datetime = field(default_factory=lambda: datetime.now())

    def add_message(self, message: Message):
        self.messages.append(message)
        self.updated_at = datetime.now()
