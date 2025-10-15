from pydantic import BaseModel
from typing import Dict

class Page(BaseModel):
    page_num: int
    text: str


class Chunk(BaseModel):
    id: str
    content: str
    metadata: Dict[str, str]


class DocumentProcessorConfig(BaseModel):
    chunk_size: int = 1500
    chunk_overlap: int = 150
    title: str = ""
    source: str = ""