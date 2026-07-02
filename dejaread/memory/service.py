"""记忆模块（用户画像 + 论文记忆）：纯文件存储 + LLM 合并生成。

两类记忆都不需要相似度检索（论文记忆按 paper_id 精确查找，用户画像全局唯一），
所以不进 DB、不进向量库/关键词索引，直接读写 Markdown 文件。每次更新都是
"旧文件全文 + 新信息 → LLM 合并重写"，不是清空重建，用户手动编辑的内容会作为
LLM 输入的一部分保留下去。
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable

from sqlalchemy.orm import Session

from ..config import get_config
from ..db import Concept, Paper, get_session
from ..llm import LLMClient
from ..qa.schemas import ChatTurn
from ..utils.utils import setup_logger
from .parser import (
    parse_paper_memory,
    parse_user_memory,
    render_paper_memory,
    render_user_memory,
)
from .schemas import ParsedPaperMemory

logger = setup_logger(log_dir="logs/log_memory", logger_name="memory_service")

_PAPER_MEMORY_SYSTEM_PROMPT = (
    "你在维护一份关于某篇论文的持续演进的阅读记忆。下面会给你旧记忆的三段内容、"
    "该论文目前标注过的概念列表、以及本次新增的问答。请合并生成新版本的三段内容：\n"
    "- 摘要：累积式改写，融合旧摘要和新内容成一段连贯的中文摘要，不要简单拼接或重复罗列。\n"
    "- 讨论过的概念：去重后的概念名简要罗列。\n"
    "- 待解决问题：增补新出现的疑问，去掉已经在本次问答中解决的旧疑问。\n"
    "旧记忆中如果有用户手动写的内容，必须保留，不能凭空丢弃。\n"
    "严格按照下面的格式输出，不要有其他内容：\n\n"
    "## 摘要\n...\n\n## 讨论过的概念\n...\n\n## 待解决问题\n..."
)

_USER_MEMORY_SYSTEM_PROMPT = (
    "你在维护一份跨论文的用户阅读画像。下面会给你旧画像的四段内容、以及全部论文记忆的全文。"
    "请从中提炼稳定的用户偏好，合并生成新版本的四段内容：\n"
    "- 回答偏好：语言、详略程度等。\n"
    "- 研究兴趣：长期关注的领域/方向。\n"
    "- 阅读习惯：阅读顺序等行为模式。\n"
    "- 知识背景：已经掌握的知识概览。\n"
    "旧画像中用户手动写的内容必须保留，不能凭空丢弃。\n"
    "严格按照下面的格式输出，不要有其他内容：\n\n"
    "## 回答偏好\n...\n\n## 研究兴趣\n...\n\n## 阅读习惯\n...\n\n## 知识背景\n..."
)


class MemoryService:
    """论文记忆与用户画像的文件读写 + LLM 合并生成。"""

    def __init__(
        self,
        llm_client: LLMClient,
        memory_dir: str | Path | None = None,
        session_factory: Callable[[], Session] = get_session,
    ) -> None:
        config = get_config().memory
        self.llm_client = llm_client
        self.memory_dir = Path(memory_dir if memory_dir is not None else config.memory_dir)
        self._session_factory = session_factory

    def _paper_memory_path(self, paper_id: str) -> Path:
        return self.memory_dir / f"paper_{paper_id}.md"

    def _user_memory_path(self) -> Path:
        return self.memory_dir / "user_memory.md"

    def read_paper_memory(self, paper_id: str) -> str:
        """读取论文记忆全文；文件不存在返回空字符串。"""
        path = self._paper_memory_path(paper_id)
        return path.read_text(encoding="utf-8") if path.exists() else ""

    def read_user_memory(self) -> str:
        """读取用户画像全文；文件不存在返回空字符串。"""
        path = self._user_memory_path()
        return path.read_text(encoding="utf-8") if path.exists() else ""

    def update_paper_memory(self, paper_id: str, chat_history: list[ChatTurn]) -> str:
        """合并旧论文记忆 + 该论文的概念列表 + 本次新增问答，生成新论文记忆并写文件。

        写入成功后自动联动 :meth:`update_user_memory`；联动失败只记日志，
        不影响本次论文记忆已经写入成功的返回值。
        """
        logger.info("update_paper_memory 开始：paper_id=%s history_len=%d", paper_id, len(chat_history))
        old = parse_paper_memory(self.read_paper_memory(paper_id))

        session = self._session_factory()
        try:
            paper = session.get(Paper, paper_id)
            if paper is None:
                raise ValueError(f"未找到论文: {paper_id}")
            title = paper.title
            concepts = session.query(Concept).filter_by(paper_id=paper_id).all()
            concepts_text = (
                "\n".join(
                    f"- {c.name}（讨论 {c.discussion_count} 次）：{c.definition or ''}"
                    for c in concepts
                )
                or "（暂无标注的概念）"
            )
        finally:
            session.close()

        title_line = old.title_line or f"# {title} - 记忆"
        history_text = (
            "\n".join(f"Q: {t.question}\nA: {t.answer}" for t in chat_history)
            or "（本次没有新增问答）"
        )
        user_prompt = (
            f"旧记忆：\n摘要：{old.summary or '（无）'}\n"
            f"讨论过的概念：{old.key_concepts or '（无）'}\n"
            f"待解决问题：{old.open_questions or '（无）'}\n\n"
            f"该论文标注过的概念：\n{concepts_text}\n\n"
            f"本次新增问答：\n{history_text}"
        )

        try:
            response = self.llm_client.chat(_PAPER_MEMORY_SYSTEM_PROMPT, user_prompt)
        except Exception:
            logger.exception("update_paper_memory LLM 调用失败：paper_id=%s", paper_id)
            raise

        merged = parse_paper_memory(response)
        new_memory = ParsedPaperMemory(
            title_line=title_line,
            summary=merged.summary,
            key_concepts=merged.key_concepts,
            open_questions=merged.open_questions,
        )
        new_content = render_paper_memory(new_memory)

        self.memory_dir.mkdir(parents=True, exist_ok=True)
        self._paper_memory_path(paper_id).write_text(new_content, encoding="utf-8")
        logger.info("update_paper_memory 完成：paper_id=%s content_len=%d", paper_id, len(new_content))

        try:
            self.update_user_memory()
        except Exception:
            logger.exception(
                "update_paper_memory 联动 update_user_memory 失败，论文记忆已写入成功：paper_id=%s",
                paper_id,
            )

        return new_content

    def update_user_memory(self) -> str:
        """合并旧用户画像 + 全部论文记忆全文，生成新用户画像并写文件。"""
        logger.info("update_user_memory 开始")
        old = parse_user_memory(self.read_user_memory())

        paper_texts: list[str] = []
        if self.memory_dir.exists():
            for path in sorted(self.memory_dir.glob("paper_*.md")):
                paper_texts.append(f"### {path.name}\n{path.read_text(encoding='utf-8')}")
        papers_block = "\n\n".join(paper_texts) or "（暂无论文记忆）"

        user_prompt = (
            f"旧画像：\n回答偏好：{old.response_preference or '（无）'}\n"
            f"研究兴趣：{old.research_interests or '（无）'}\n"
            f"阅读习惯：{old.reading_habits or '（无）'}\n"
            f"知识背景：{old.background or '（无）'}\n\n"
            f"全部论文记忆：\n{papers_block}"
        )

        response = self.llm_client.chat(_USER_MEMORY_SYSTEM_PROMPT, user_prompt)
        merged = parse_user_memory(response)
        new_content = render_user_memory(merged)

        self.memory_dir.mkdir(parents=True, exist_ok=True)
        self._user_memory_path().write_text(new_content, encoding="utf-8")
        logger.info("update_user_memory 完成：content_len=%d", len(new_content))
        return new_content
