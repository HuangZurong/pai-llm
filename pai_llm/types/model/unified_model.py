from threading import Lock
from typing import ClassVar, Union, cast, Dict


class UnifiedModel(str):
    _cache: ClassVar[Dict[str, "UnifiedModel"]] = {}
    _lock: ClassVar[Lock] = Lock()

    def __new__(cls, value: Union["UnifiedModel", str]) -> "UnifiedModel":
        with cls._lock:
            if value not in cls._cache:
                instance = super().__new__(cls, value)
                cls._cache[value] = cast(UnifiedModel, instance)
            else:
                instance = cls._cache[value]
            return instance
