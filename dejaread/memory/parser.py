"""记忆文件（论文记忆 / 用户画像）的固定标题解析与渲染。

标题切分逻辑直接复用 :mod:`dejaread.notes.parser`——两边都是"按 `## ` 二级标题
切分整篇 Markdown"，语义完全一致，不重复实现。
"""

from __future__ import annotations

from ..notes.parser import split_sections
from .schemas import ParsedPaperMemory, ParsedUserMemory


def parse_paper_memory(content: str) -> ParsedPaperMemory:
    """解析论文记忆全文。缺失的标题段对应字段留空字符串。"""
    lines = content.splitlines()
    title_line = lines[0] if lines and lines[0].startswith("# ") else ""
    sections = {s.heading: s.content for s in split_sections(content)}
    return ParsedPaperMemory(
        title_line=title_line,
        summary=sections.get("摘要", ""),
        key_concepts=sections.get("讨论过的概念", ""),
        open_questions=sections.get("待解决问题", ""),
    )


def render_paper_memory(memory: ParsedPaperMemory) -> str:
    """把 :class:`ParsedPaperMemory` 渲染回固定标题的 Markdown 全文。"""
    return (
        f"{memory.title_line}\n\n"
        f"## 摘要\n{memory.summary}\n\n"
        f"## 讨论过的概念\n{memory.key_concepts}\n\n"
        f"## 待解决问题\n{memory.open_questions}\n"
    )


def parse_user_memory(content: str) -> ParsedUserMemory:
    """解析用户画像全文。缺失的标题段对应字段留空字符串。"""
    sections = {s.heading: s.content for s in split_sections(content)}
    return ParsedUserMemory(
        response_preference=sections.get("回答偏好", ""),
        research_interests=sections.get("研究兴趣", ""),
        reading_habits=sections.get("阅读习惯", ""),
        background=sections.get("知识背景", ""),
    )


def render_user_memory(memory: ParsedUserMemory) -> str:
    """把 :class:`ParsedUserMemory` 渲染回固定标题的 Markdown 全文。"""
    return (
        "# 用户画像\n\n"
        f"## 回答偏好\n{memory.response_preference}\n\n"
        f"## 研究兴趣\n{memory.research_interests}\n\n"
        f"## 阅读习惯\n{memory.reading_habits}\n\n"
        f"## 知识背景\n{memory.background}\n"
    )
