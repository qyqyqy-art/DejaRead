"""3.4 笔记模块的数据结构。"""

from __future__ import annotations

from pydantic import BaseModel


class ParsedSection(BaseModel):
    """笔记 markdown 按 `## ` 二级标题切分出的一段（不含标题行本身）。"""

    heading: str
    content: str
    section_index: int
