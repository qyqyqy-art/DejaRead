"""问答模块的数据结构。"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class ChatTurn(BaseModel):
    """一轮问答历史。"""

    question: str
    answer: str


class QARequest(BaseModel):
    """一次问答请求：当前论文 + 问题 + 历史对话。"""

    paper_id: str
    question: str
    history: list[ChatTurn] = Field(default_factory=list)


class Citation(BaseModel):
    """一条回答依据的来源标注。"""

    source_type: Literal["chunk", "note", "concept", "memory"]
    paper_id: str
    snippet: str


class QAResult(BaseModel):
    """一次问答的完整结果：回答 + 引用来源。"""

    answer: str
    citations: list[Citation] = Field(default_factory=list)
    memory_snippets: list[str] = Field(default_factory=list)
