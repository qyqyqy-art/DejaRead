"""最小化问答服务（3.5 预览）：当前论文 chunk 检索 + 全库笔记检索 + LLM 生成。

完整的 3.5 QA Agent（意图识别 / self-reflection / memory 召回 / 多轮 query 重写）留给
后续单独实现，这里只覆盖"检索 + 生成"这一条直通路径，同时为笔记模块的对话摘要导入
功能（见 :mod:`dejaread.notes`）提供"对话"载体。
"""

from __future__ import annotations

from typing import Callable

from sqlalchemy.orm import Session

from ..config import get_config
from ..db import Chunk, NoteSection, get_session
from ..embedding import Embedder, VectorStore
from ..keyword import KeywordStore
from ..llm import LLMClient
from ..retrieval import HybridRetriever
from .schemas import ChatTurn, Citation, QARequest, QAResult


class QAService:
    """检索（当前论文 chunk + 全库笔记）→ 拼 prompt → LLM 生成回答。"""

    def __init__(
        self,
        embedder: Embedder,
        vector_store: VectorStore,
        keyword_store: KeywordStore,
        llm_client: LLMClient,
        chunk_collection: str | None = None,
        note_collection: str | None = None,
        session_factory: Callable[[], Session] = get_session,
    ) -> None:
        config = get_config()
        self.llm_client = llm_client
        self._session_factory = session_factory
        self.chunk_collection = (
            chunk_collection if chunk_collection is not None else config.vector_store.chunk_collection
        )
        self.note_collection = (
            note_collection if note_collection is not None else config.notes.section_collection
        )
        self.chunk_retriever = HybridRetriever(
            vector_store=vector_store,
            keyword_store=keyword_store,
            embedder=embedder,
            collection=self.chunk_collection,
            top_k=config.qa.chunk_top_k,
        )
        self.note_retriever = HybridRetriever(
            vector_store=vector_store,
            keyword_store=keyword_store,
            embedder=embedder,
            collection=self.note_collection,
            top_k=config.qa.note_top_k,
        )
        self.overfetch_factor = config.qa.overfetch_factor

    def ask(self, request: QARequest) -> QAResult:
        chunk_matches = self.chunk_retriever.search(
            request.question,
            metadata_filter=lambda m: m.get("paper_id") == request.paper_id,
            overfetch_factor=self.overfetch_factor,
        )
        note_matches = self.note_retriever.search(request.question)

        session = self._session_factory()
        try:
            chunks = self._load_chunks(session, [m.id for m in chunk_matches])
            sections = self._load_sections(session, [m.id for m in note_matches])
        finally:
            session.close()

        context_blocks: list[str] = []
        citations: list[Citation] = []
        for chunk in chunks:
            context_blocks.append(f"[论文原文] {chunk.content}")
            citations.append(
                Citation(source_type="chunk", paper_id=chunk.paper_id, snippet=chunk.content[:200])
            )
        for section in sections:
            context_blocks.append(f"[笔记 - 论文{section.paper_id}] {section.heading}: {section.content}")
            citations.append(
                Citation(source_type="note", paper_id=section.paper_id, snippet=section.content[:200])
            )

        system = (
            "你是一个论文阅读助手，基于下面提供的论文原文片段和笔记摘录回答用户问题。"
            "回答时区分两种来源；如果提供的内容不足以回答，直接说明信息不足，不要编造。"
        )
        user_parts: list[str] = []
        history_text = "\n".join(f"Q: {t.question}\nA: {t.answer}" for t in request.history)
        if history_text:
            user_parts.append(f"历史对话：\n{history_text}")
        if context_blocks:
            user_parts.append("检索到的内容：\n" + "\n\n".join(context_blocks))
        else:
            user_parts.append("检索到的内容：（未检索到相关内容）")
        user_parts.append(f"当前问题：{request.question}")
        user = "\n\n".join(user_parts)

        answer = self.llm_client.chat(system, user).strip()
        return QAResult(answer=answer, citations=citations)

    def summarize_conversation(self, history: list[ChatTurn]) -> str:
        """把一段问答历史总结成 3-6 句中文摘要，供导入笔记的 "## 对话摘要" 段使用。"""
        history_text = "\n".join(f"Q: {t.question}\nA: {t.answer}" for t in history)
        system = (
            "你是一个论文阅读助手，请将下面这段问答对话总结成 3-6 句中文摘要，"
            "提炼讨论到的概念和结论，不要逐句复述原文。"
        )
        return self.llm_client.chat(system, history_text).strip()

    def _load_chunks(self, session: Session, ids: list[str]) -> list[Chunk]:
        if not ids:
            return []
        rows = {c.id: c for c in session.query(Chunk).filter(Chunk.id.in_(ids)).all()}
        return [rows[id_] for id_ in ids if id_ in rows]

    def _load_sections(self, session: Session, ids: list[str]) -> list[NoteSection]:
        if not ids:
            return []
        rows = {s.id: s for s in session.query(NoteSection).filter(NoteSection.id.in_(ids)).all()}
        return [rows[id_] for id_ in ids if id_ in rows]
