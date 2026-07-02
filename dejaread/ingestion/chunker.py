"""
分块：按段落聚合到接近目标长度的 chunk，相邻 chunk 间保留重叠上下文。
"""

from __future__ import annotations

import re

from pydantic import BaseModel

from ..config import get_config
from ..utils.utils import setup_logger
from .parser import ParsedPaper

logger = setup_logger(log_dir="logs/log_ingestion_chunker", logger_name="ingestion_chunker")

# 匹配编号式章节标题，如 "2. Architecture"、"3.1 Experiment"、"A.2 Acknowledgment"
_NUMBERED_HEADING_RE = re.compile(
    r"^\s*(?:[A-Z]\.?\d*|\d+)(?:\.\d+)*\.?\s+\S+",
)

# 匹配 Markdown 风格标题 (marker / PaddleOCR parser 输出)
_MD_HEADING_RE = re.compile(r"^\s*#{1,6}\s+\S+")

# PDF 断字：行尾连字符 + 换行 + 下一行以小写字母开头，表示单词被断开
_DEHYPHEN_RE = re.compile(r"(\w)-\s*\n\s*([a-z])")


class TextChunk(BaseModel):
    """
    一个分块的内容及其元数据，写入 SQLite 的 chunks 表前的中间表示。
    """

    content: str
    section: str | None
    chunk_index: int
    page_number: int | None = None


class Chunker:
    """
    把 :class: ParsedPaper 切分为若干 :class: TextChunk。

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
            cleaned_text = self._dehyphenate(section.text)
            for paragraph in self._split_paragraphs(cleaned_text):
                heading = self._detect_heading(paragraph)
                if heading is not None:
                    current_section = heading

                if buffer and len(buffer) + len(paragraph) + 1 > self.max_chars:
                    overlap = self._word_safe_overlap(buffer)
                    flush()
                    buffer = overlap
                    buffer_page = section.page_number

                if not buffer:
                    buffer_page = section.page_number
                buffer = f"{buffer}\n{paragraph}".strip() if buffer else paragraph

        flush()
        logger.info("chunk 完成：sections=%d chunks=%d", len(paper.sections), len(chunks))
        return chunks

    @staticmethod
    def _split_paragraphs(text: str) -> list[str]:
        return [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]

    @staticmethod
    def _dehyphenate(text: str) -> str:
        """
        修复 PDF 提取文本中的断字：行尾连字符 + 换行 + 小写字母开头 → 拼合单词。
        """
        return _DEHYPHEN_RE.sub(r"\1\2", text)

    def _word_safe_overlap(self, text: str) -> str:
        """
        取 text 末尾约 overlap_chars 字符，但在单词边界处截断，避免切断单词。
        """
        if len(text) <= self.overlap_chars:
            return text
        overlap = text[-self.overlap_chars :]

        # 确保不在单词中间截断：如果 overlap 以空格开头，则直接返回；否则找到第一个空格，截断到空格后
        space_idx = overlap.find(" ")
        if space_idx > 0:
            overlap = overlap[space_idx + 1 :]
        return overlap

    @staticmethod
    def _detect_heading(paragraph: str) -> str | None:
        """
        识别章节标题。支持编号式标题和 Markdown 标题。
        """
        first_line = paragraph.splitlines()[0].strip()
        if len(first_line) > 80:
            return None
        # 编号式标题: "2. Architecture", "3.1 Experiment Results", "A.1 Appendix"
        if _NUMBERED_HEADING_RE.match(first_line):
            return first_line
        # Markdown 标题: "## Architecture"
        if _MD_HEADING_RE.match(first_line):
            return first_line.lstrip("# ").strip()
        return None
