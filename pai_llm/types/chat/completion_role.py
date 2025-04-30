from typing_extensions import Literal, TypeAlias

SystemRole = "system"
AssistantRole = "assistant"
UserRole = "user"
DeveloperRole = "developer"
ToolRole = "tool"
FunctionRole = "function"

CompletionRole: TypeAlias = Literal["system", "user", "assistant", "developer", "tool", "function"]
