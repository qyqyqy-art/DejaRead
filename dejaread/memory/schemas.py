"""记忆模块的数据结构：论文记忆（按 paper_id）与用户画像（全局唯一）。"""

from __future__ import annotations

from pydantic import BaseModel


class ParsedPaperMemory(BaseModel):
    """论文记忆文件解析结果。`title_line` 是文件首行（如 "# DAPO - 记忆"），
    合并更新时原样保留，不经过 LLM 重写。
    """

    title_line: str = ""
    summary: str = ""
    key_concepts: str = ""
    open_questions: str = ""


class ParsedUserMemory(BaseModel):
    """用户画像文件解析结果。"""

    response_preference: str = ""
    research_interests: str = ""
    reading_habits: str = ""
    background: str = ""
