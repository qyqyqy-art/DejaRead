"""最小化问答模块（3.5 预览）的数据结构。"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal


@dataclass
class ChatTurn:
    """一轮问答历史。"""

    question: str
    answer: str


@dataclass
class QARequest:
    """一次问答请求：当前论文 + 问题 + 历史对话。"""

    paper_id: str
    question: str
    history: list[ChatTurn] = field(default_factory=list)


@dataclass
class Citation:
    """一条回答依据的来源标注。"""

    source_type: Literal["chunk", "note"]
    paper_id: str
    snippet: str


@dataclass
class QAResult:
    """一次问答的完整结果：回答 + 引用来源。"""

    answer: str
    citations: list[Citation] = field(default_factory=list)
