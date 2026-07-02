"""笔记 markdown 分段（3.4.1）。

只识别 `## ` 二级标题作为分段边界——文件开头到第一个 `## ` 之前的内容（通常是
`# {论文标题}` 这一行）不生成 section，不入库、不参与检索；`### ` 及更深层级的标题
内容并入其父 `## ` 段一起检索。
"""

from __future__ import annotations

import re

from ..utils.utils import setup_logger
from .schemas import ParsedSection

_HEADING_RE = re.compile(r"^## (.+)$")

logger = setup_logger(log_dir="logs/log_notes_parser", logger_name="notes_parser")


def split_sections(markdown: str) -> list[ParsedSection]:
    """按 `## ` 标题切分笔记全文，标题重复出现时各自成段（不去重合并）。"""
    lines = markdown.splitlines()
    sections: list[ParsedSection] = []
    current_heading: str | None = None
    current_lines: list[str] = []
    section_index = 0

    def flush() -> None:
        nonlocal section_index
        if current_heading is not None:
            sections.append(
                ParsedSection(
                    heading=current_heading,
                    content="\n".join(current_lines).strip(),
                    section_index=section_index,
                )
            )
            section_index += 1

    for line in lines:
        match = _HEADING_RE.match(line)
        if match:
            flush()
            current_heading = match.group(1).strip()
            current_lines = []
        elif current_heading is not None:
            current_lines.append(line)
    flush()
    logger.debug("split_sections 完成：sections=%d", len(sections))
    return sections
