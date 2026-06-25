"""智能分块：按段落聚合到接近目标长度的 chunk，相邻 chunk 间保留重叠上下文。

对应设计文档 3.1 节 "智能分块（按章节/段落，保留上下文）"。
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from ..config import get_config
from .parser import ParsedPaper

_HEADING_RE = re.compile(
    r"^\s*(\d+\.?\s+)?(abstract|introduction|related work|background|method|"
    r"methodology|experiment|experiments|result|results|discussion|conclusion|"
    r"references|acknowledg(e)?ments?)\b",
    re.IGNORECASE,
)


@dataclass
class TextChunk:
    """一个分块的内容及其元数据，写入 SQLite 的 ``chunks`` 表前的中间表示。"""

    content: str
    section: str | None
    chunk_index: int
    page_number: int | None = None


class Chunker:
    """把 :class:`ParsedPaper` 切分为若干 :class:`TextChunk`。

    策略：
    1. 先按段落（双换行）拆分每一页文本；
    2. 用简单的启发式规则识别章节标题，作为后续 chunk 的 ``section`` 标签；
    3. 贪心地把连续段落拼接到接近 ``max_chars``，避免把语义连续的内容切碎；
    4. 每个新 chunk 开头携带上一个 chunk 末尾 ``overlap_chars`` 字符，保留上下文。
    """

    def __init__(self, max_chars: int | None = None, overlap_chars: int | None = None) -> None:
        chunking_config = get_config().chunking
        max_chars = max_chars if max_chars is not None else chunking_config.max_chars
        overlap_chars = overlap_chars if overlap_chars is not None else chunking_config.overlap_chars
        if overlap_chars >= max_chars:
            raise ValueError("overlap_chars 必须小于 max_chars")
        self.max_chars = max_chars
        self.overlap_chars = overlap_chars

    def chunk(self, paper: ParsedPaper) -> list[TextChunk]:
        chunks: list[TextChunk] = []
        buffer = ""
        buffer_page: int | None = None
        current_section: str | None = None
        chunk_index = 0

        def flush() -> None:
            nonlocal buffer, buffer_page, chunk_index
            text = buffer.strip()
            if text:
                chunks.append(
                    TextChunk(
                        content=text,
                        section=current_section,
                        chunk_index=chunk_index,
                        page_number=buffer_page,
                    )
                )
                chunk_index += 1
            buffer = ""
            buffer_page = None

        for section in paper.sections:
            for paragraph in self._split_paragraphs(section.text):
                heading = self._detect_heading(paragraph)
                if heading is not None:
                    current_section = heading

                if buffer and len(buffer) + len(paragraph) + 1 > self.max_chars:
                    overlap = buffer[-self.overlap_chars :]
                    flush()
                    buffer = overlap
                    buffer_page = section.page_number

                if not buffer:
                    buffer_page = section.page_number
                buffer = f"{buffer}\n{paragraph}".strip() if buffer else paragraph

        flush()
        return chunks

    @staticmethod
    def _split_paragraphs(text: str) -> list[str]:
        return [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]

    @staticmethod
    def _detect_heading(paragraph: str) -> str | None:
        first_line = paragraph.splitlines()[0].strip()
        if len(first_line) <= 60 and _HEADING_RE.match(first_line):
            return first_line
        return None
