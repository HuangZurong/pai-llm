from pai_llm.exceptions import PaiLLMError


class ConversationNotFoundError(PaiLLMError):
    pass


class MessageNotFoundError(PaiLLMError):
    pass


class SQLiteStorageError(PaiLLMError):
    pass


class DatabaseOperationError(SQLiteStorageError):
    pass


class StorageError(PaiLLMError):
    pass


class UnsupportedContentTypeError(PaiLLMError):
    pass
