from enum import Enum

from pai_llm.types.media.base import MediaType


class DocumentType(Enum):
    PDF: MediaType = MediaType("pdf", "application/pdf")
    DOCX: MediaType = MediaType("docx", "application/vnd.openxmlformats-officedocument.wordprocessingml.document")
    PPTX: MediaType = MediaType("pptx", "application/vnd.openxmlformats-officedocument.presentationml.presentation")
