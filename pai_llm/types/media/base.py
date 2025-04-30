from dataclasses import dataclass


@dataclass
class MediaType:
    name: str
    mime: str


class Media:
    type: MediaType
    content: bytes | str
