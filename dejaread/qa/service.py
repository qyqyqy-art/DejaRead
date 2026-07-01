"""问答服务（3.5 完整版）：query 改写 → 多源并行检索 → token budget 截断 → LLM 生成。

检索来源：
  - 当前论文 chunk（向量 + BM25 + grep 关键词增强）
  - 全库笔记 section（向量 + BM25）
  - 全库概念图谱（向量 + BM25）

paper_memory_provider 为占位接口，待记忆模块完成后注入真实实现。
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Callable

from sqlalchemy import or_, select
from sqlalchemy.orm import Session

from ..config import get_config
from ..db import Chunk, Concept, NoteSection, get_session
from ..embedding import Embedder, VectorStore
from ..keyword import KeywordStore
from ..llm import LLMClient
from ..retrieval import HybridRetriever
from .rewriter import QueryRewriter
from .schemas import ChatTurn, Citation, QARequest, QAResult

_CHARS_PER_TOKEN = 1.5  # 中文估算：1 token ≈ 1.5 字符


class QAService:
    """检索（当前论文 chunk + 全库笔记 + 全库概念）→ 拼 prompt → LLM 生成回答。"""

    def __init__(
        self,
        embedder: Embedder,
        vector_store: VectorStore,
        keyword_store: KeywordStore,
        llm_client: LLMClient,
        chunk_collection: str | None = None,
        note_collection: str | None = None,
        concept_collection: str | None = None,
        paper_memory_provider: Callable[[str], str] | None = None,
        session_factory: Callable[[], Session] = get_session,
    ) -> None:
        config = get_config()
        self.llm_client = llm_client
        self._session_factory = session_factory
        self.paper_memory_provider = paper_memory_provider

        self.chunk_collection = chunk_collection or config.vector_store.chunk_collection
        self.note_collection = note_collection or config.notes.section_collection
        self.concept_collection = concept_collection or config.vector_store.concept_collection

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
        self.concept_retriever = HybridRetriever(
            vector_store=vector_store,
            keyword_store=keyword_store,
            embedder=embedder,
            collection=self.concept_collection,
            top_k=config.qa.concept_top_k,
        )
        self.rewriter = QueryRewriter(
            llm_client=llm_client,
            max_history_turns=config.qa.rewrite_history_turns,
        )
        self.context_token_budget = config.qa.context_token_budget

    def ask(self, request: QARequest) -> QAResult:
        rewrite = self.rewriter.rewrite(request.question, request.history)
        query = rewrite.rewritten_query
        keywords = rewrite.keywords

        # 三路并行检索
        with ThreadPoolExecutor(max_workers=3) as pool:
            future_chunks = pool.submit(
                self.chunk_retriever.search,
                query,
                metadata_filter={"paper_id": request.paper_id},
            )
            future_notes = pool.submit(self.note_retriever.search, query)
            future_concepts = pool.submit(self.concept_retriever.search, query)
            chunk_matches = future_chunks.result()
            note_matches = future_notes.result()
            concept_matches = future_concepts.result()

        session = self._session_factory()
        try:
            # grep 增强：把关键词精确匹配的 chunk 追加进结果（去重）
            grep_ids = self._grep_chunk_ids(session, request.paper_id, keywords)
            existing_chunk_ids = {m.id for m in chunk_matches}
            extra_ids = [id_ for id_ in grep_ids if id_ not in existing_chunk_ids]

            chunks = self._load_chunks(session, [m.id for m in chunk_matches] + extra_ids)
            sections = self._load_sections(session, [m.id for m in note_matches])
            concepts = self._load_concepts(session, [m.id for m in concept_matches])
        finally:
            session.close()

        context_blocks, citations = self._build_context(chunks, sections, concepts)

        paper_memory = ""
        if self.paper_memory_provider is not None:
            paper_memory = self.paper_memory_provider(request.paper_id)

        system = self._build_system_prompt(paper_memory)

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

    # ------------------------------------------------------------------
    # 内部辅助方法
    # ------------------------------------------------------------------

    def _build_system_prompt(self, paper_memory: str) -> str:
        base = (
            "你是一个论文阅读助手，基于下面提供的论文原文片段、笔记摘录和概念图谱信息回答用户问题。"
            "回答时区分来源（论文原文 / 笔记 / 概念图谱）；如果提供的内容不足以回答，直接说明信息不足，不要编造。"
        )
        if paper_memory:
            return f"{base}\n\n关于当前论文，你已积累的背景信息：\n{paper_memory}"
        return base

    def _build_context(
        self,
        chunks: list[Chunk],
        sections: list[NoteSection],
        concepts: list[Concept],
    ) -> tuple[list[str], list[Citation]]:
        """按 token budget 截断并组装 context，优先级：chunk > note > concept。"""
        budget_chars = int(self.context_token_budget * _CHARS_PER_TOKEN)
        used = 0
        blocks: list[str] = []
        citations: list[Citation] = []

        for chunk in chunks:
            block = f"[论文原文] {chunk.content}"
            if used + len(block) > budget_chars:
                break
            blocks.append(block)
            citations.append(Citation(source_type="chunk", paper_id=chunk.paper_id, snippet=chunk.content[:200]))
            used += len(block)

        for section in sections:
            block = f"[笔记 - 论文{section.paper_id}] {section.heading}: {section.content}"
            if used + len(block) > budget_chars:
                break
            blocks.append(block)
            citations.append(Citation(source_type="note", paper_id=section.paper_id, snippet=section.content[:200]))
            used += len(block)

        for concept in concepts:
            definition = concept.definition or ""
            block = f"[概念图谱] {concept.name}（来自论文 {concept.paper_id}）：{definition}"
            if used + len(block) > budget_chars:
                break
            blocks.append(block)
            citations.append(Citation(source_type="concept", paper_id=concept.paper_id, snippet=definition[:200]))
            used += len(block)

        return blocks, citations

    def _grep_chunk_ids(self, session: Session, paper_id: str, keywords: list[str]) -> list[str]:
        if not keywords:
            return []
        conditions = [Chunk.content.contains(kw) for kw in keywords]
        stmt = (
            select(Chunk.id)
            .where(Chunk.paper_id == paper_id)
            .where(or_(*conditions))
            .limit(self.chunk_retriever.top_k)
        )
        return [row[0] for row in session.execute(stmt)]

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

    def _load_concepts(self, session: Session, ids: list[str]) -> list[Concept]:
        if not ids:
            return []
        rows = {c.id: c for c in session.query(Concept).filter(Concept.id.in_(ids)).all()}
        return [rows[id_] for id_ in ids if id_ in rows]
