"""用户驱动的概念图谱标注服务（3.2 节核心流程）。

实现 3.2.1 选词标注流程：

    用户选中词/短语
      → 定位所在 chunk，提取上下文
      → LLM 生成语境化解释
      → 向量相似度匹配已有概念图谱
          ├→ 有匹配 → LLM 精排 + 生成跨论文关联描述 → 创建关联边
          └→ 无匹配 → 仅创建孤立概念节点
      → 写入概念图谱，返回结果
"""

from __future__ import annotations

from datetime import datetime
from typing import Callable

from sqlalchemy import select
from sqlalchemy.orm import Session

from ..config import get_config
from ..db import Chunk, Concept, ConceptLink, get_session
from ..embedding import Embedder, InMemoryVectorStore, VectorStore
from ..keyword import KeywordStore, SQLiteFTSStore
from .linking import LinkDiscovery
from .llm import ConceptLLM, MockConceptLLM
from .schemas import AnnotationRequest, AnnotationResult, LinkResult


class ConceptAnnotationService:
    """选词标注 → 概念解释 → 跨论文关联发现 → 写入图谱，一站式服务。"""

    def __init__(
        self,
        embedder: Embedder,
        vector_store: VectorStore | None = None,
        keyword_store: KeywordStore | None = None,
        llm_client: ConceptLLM | None = None,
        link_discovery: LinkDiscovery | None = None,
        session_factory: Callable[[], Session] = get_session,
        context_window_chars: int | None = None,
    ) -> None:
        self.embedder = embedder
        self.vector_store = vector_store or InMemoryVectorStore()
        self.keyword_store = keyword_store or SQLiteFTSStore()
        self.llm_client = llm_client or MockConceptLLM()
        self.link_discovery = link_discovery or LinkDiscovery(self.vector_store)
        self._session_factory = session_factory
        self.context_window_chars = (
            context_window_chars
            if context_window_chars is not None
            else get_config().annotation.context_window_chars
        )
        self._concept_collection = get_config().vector_store.concept_collection

    def annotate(self, request: AnnotationRequest) -> AnnotationResult:
        """处理一次选词标注请求，返回概念解释 + 跨论文关联（如有）。"""
        session = self._session_factory()
        try:
            chunk = self._locate_chunk(session, request)
            context_snippet = self._extract_context(chunk.content, request.selected_text)
            paper_title = chunk.paper.title

            definition = self.llm_client.generate_definition(
                concept_text=request.selected_text,
                context_snippet=context_snippet,
                paper_title=paper_title,
            )

            concept = Concept(
                name=request.selected_text,
                paper_id=request.paper_id,
                selected_text=request.selected_text,
                page_number=request.page_number,
                source_chunk_id=chunk.id,
                context_snippet=context_snippet,
                definition=definition,
                last_discussed=datetime.utcnow(),
            )
            session.add(concept)
            session.flush()  # 拿到 concept.id，供向量库 metadata 与关联边使用

            concept_text = f"{concept.name}: {definition}"
            # document 侧向量用于写入向量库（供之后被检索到）；query 侧向量用于本次
            # 主动检索已有概念——非对称 embedding 模型（如 Qwen3-Embedding）对两者的
            # 编码方式不同（见 dejaread.embedding.RemoteEmbedder.embed_query）。
            document_embedding = self.embedder.embed_one(concept_text)
            query_embedding = self.embedder.embed_query(concept_text)
            links = self._discover_links(session, concept, query_embedding)

            session.commit()
            session.refresh(concept)
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

        # 写向量库/关键词库放在 ORM 事务提交并关闭 session 之后：SQLiteFTSStore 用独立的
        # sqlite 连接写同一个数据库文件，若 ORM 事务仍未提交（持有写锁），会触发
        # "database is locked"。同时也保证标注完成后才让新概念可被检索到，避免它把自己
        # 检索回来（_discover_links 在它之前执行）。
        concept_metadata = [{"paper_id": concept.paper_id, "name": concept.name}]
        self.vector_store.upsert(
            collection=self._concept_collection,
            ids=[concept.id],
            embeddings=[document_embedding],
            metadatas=concept_metadata,
        )
        self.keyword_store.upsert(
            collection=self._concept_collection,
            ids=[concept.id],
            texts=[concept_text],
            metadatas=concept_metadata,
        )

        return AnnotationResult(
            concept_id=concept.id,
            name=concept.name,
            definition=concept.definition or "",
            context_snippet=concept.context_snippet or "",
            links=links,
        )

    def _locate_chunk(self, session: Session, request: AnnotationRequest) -> Chunk:
        """定位选中词所在的 chunk（3.2.1："后端定位选中词所在的 chunk，提取上下文"）。"""
        if request.chunk_id is not None:
            chunk = session.get(Chunk, request.chunk_id)
            if chunk is None:
                raise ValueError(f"未找到 chunk: {request.chunk_id}")
            return chunk

        stmt = (
            select(Chunk)
            .where(Chunk.paper_id == request.paper_id)
            .where(Chunk.content.contains(request.selected_text))
            .order_by(Chunk.chunk_index)
        )
        chunk = session.execute(stmt).scalars().first()
        if chunk is not None:
            return chunk

        # 兜底：选中文本未能在任何 chunk 中精确匹配时，退回该论文的第一个 chunk，
        # 保证标注流程不中断（仍能生成解释，只是上下文不够精确）。
        fallback_stmt = (
            select(Chunk)
            .where(Chunk.paper_id == request.paper_id)
            .order_by(Chunk.chunk_index)
        )
        chunk = session.execute(fallback_stmt).scalars().first()
        if chunk is None:
            raise ValueError(f"论文 {request.paper_id} 没有任何已入库的 chunk")
        return chunk

    def _extract_context(self, chunk_content: str, selected_text: str) -> str:
        """从 chunk 内容中截取选中词周围 ``context_window_chars`` 字符作为上下文片段。"""
        index = chunk_content.find(selected_text)
        if index == -1:
            return chunk_content[: self.context_window_chars * 2]
        start = max(0, index - self.context_window_chars)
        end = min(len(chunk_content), index + len(selected_text) + self.context_window_chars)
        return chunk_content[start:end]

    def _discover_links(
        self, session: Session, concept: Concept, concept_embedding: list[float]
    ) -> list[LinkResult]:
        """跨论文关联发现（3.2.3）：向量粗筛 + LLM 精排，写入 concept_links 表。"""
        candidates = self.link_discovery.find_candidates(
            concept_embedding,
            exclude_concept_id=concept.id,
            exclude_paper_id=concept.paper_id,
        )

        results: list[LinkResult] = []
        for candidate in candidates:
            related = session.get(Concept, candidate.concept_id)
            if related is None:
                continue

            classification = self.llm_client.classify_link(
                new_concept=concept.name,
                new_definition=concept.definition or "",
                candidate_concept=related.name,
                candidate_definition=related.definition or "",
            )
            if classification is None:
                continue

            link_type, description, confidence = classification
            link = ConceptLink(
                source_id=concept.id,
                target_id=related.id,
                link_type=link_type,
                description=description,
                confidence=confidence,
            )
            session.add(link)
            session.flush()

            results.append(
                LinkResult(
                    link_id=link.id,
                    related_concept_id=related.id,
                    related_concept_name=related.name,
                    related_paper_id=related.paper_id,
                    link_type=link_type,
                    description=description,
                    confidence=confidence,
                )
            )
        return results
