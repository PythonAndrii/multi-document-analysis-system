import json
from pathlib import Path
from typing import List
from src.utils.dto.data_processing import Chunk
from src.utils.loggers import logger


def write_chunks_json(path: str | Path, chunks: List[Chunk]) -> None:  # noqa: D401
    """Serialize *chunks* (list[Chunk]) to pretty JSON."""
    serialisable = [
        {"content": c.content, "metadata": c.metadata} for c in chunks
    ]
    try:
        with Path(path).open("w", encoding="utf-8") as fp:
            json.dump(serialisable, fp, indent=2, ensure_ascii=False)
    except Exception as exc:  # pragma: no cover
        logger.exception("Failed to write %s: %s", path, exc)