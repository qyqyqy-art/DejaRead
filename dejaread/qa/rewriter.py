"""
多轮对话问题改写与关键词提取

一次 LLM 调用同时完成两件事：
- 指代消解 / 问题补全（依赖历史对话）
- 抽取用于 grep 增强的精确技术关键词

history 为空时跳过 LLM，直接返回原始问题，避免不必要的 API 开销。
"""

from __future__ import annotations

import json
import re

from pydantic import BaseModel, Field

from ..llm import LLMClient
from .schemas import ChatTurn
from ..utils.utils import parse_json

_REWRITE_SYSTEM_PROMPT = """你是一个学术论文问答助手的预处理模块。给定用户的历史问答和一个新问题，完成两件事：
1. 改写新问题：补全省略的主语/宾语，消解指代（"它"、"这个方法"等），使问题脱离历史对话也能独立理解。
2. 提取关键词：从改写后的问题中识别 3-6 个技术概念，优先保留完整专有名词，不要拆分。
   论文原文通常是英文，因此每个技术概念都必须同时给出中文说法和对应的英文术语这两个版本，
   即使问题里只用了其中一种语言表达，也要把另一种语言的对应术语翻译补全后一并加入 keywords 列表
   （例如问题里只提到"注意力机制"，keywords 中也要包含 "attention mechanism"）。
   如果某个概念本身就是专有名词/缩写（如 "ResNet-50"），中英文视为同一个词，只需保留一次。

输出严格遵守以下 JSON 格式，不要输出任何其他内容：
{{"rewritten_query": "...", "keywords": ["注意力机制", "attention mechanism", "ResNet-50", ...]}}
"""


class RewriteResult(BaseModel):
    rewritten_query: str
    keywords: list[str] = Field(default_factory=list)


class QueryRewriter:
    """根据历史对话改写当前问题并提取检索关键词。"""

    def __init__(self, llm_client: LLMClient, max_history_turns: int = 5) -> None:
        self.llm_client = llm_client
        self.max_history_turns = max_history_turns

    def rewrite(self, question: str, history: list[ChatTurn]) -> RewriteResult:
        if not history:
            user = f"问题：{question}"
        else:
            recent = history[-self.max_history_turns :]
            history_text = "\n".join(f"Q: {t.question}" for t in recent)
            user = f"历史问答：\n{history_text}\n\n新问题：{question}"

        raw = self.llm_client.chat(_REWRITE_SYSTEM_PROMPT, user)
        return self._parse(raw, question)

    def _parse(self, raw: str, fallback_question: str) -> RewriteResult:
        try:
            # 兼容模型在 JSON 前后输出少量多余文本的情况
            data = parse_json(raw)
            return RewriteResult(
                rewritten_query=data.get("rewritten_query") or fallback_question,
                keywords=data.get("keywords") or [],
            )
        except (json.JSONDecodeError, KeyError):
            pass
        return RewriteResult(rewritten_query=fallback_question)
