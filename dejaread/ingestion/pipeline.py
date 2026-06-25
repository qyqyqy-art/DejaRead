"""论文入库管线（3.1 Ingestion Pipeline）。

串联 PDF 解析 → 智能分块 → embedding → 向量库 → 元数据写入 SQLite 的完整流程。
入库阶段不做概念抽取（设计文档 3.1 节的明确决策）——概念图谱由用户阅读时的选词
标注渐进式构建，见 :mod:`dejaread.concepts`。
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable

from sqlalchemy.orm import Session

from ..config import get_config
from ..db import Chunk, Paper, get_session
from ..embedding import Embedder, InMemoryVectorStore, VectorStore
from ..keyword import KeywordStore, SQLiteFTSStore
from .chunker import Chunker
from .parser import ParsedPaper, PDFParser, PyPDFParser


class IngestionPipeline:
    """协调解析器 / 分块器 / embedder / 向量库 / 关键词库 / SQLite 完成一篇论文的入库。"""

    def __init__(
        self,
        embedder: Embedder,
        parser: PDFParser | None = None,
        chunker: Chunker | None = None,
        vector_store: VectorStore | None = None,
        keyword_store: KeywordStore | None = None,
        session_factory: Callable[[], Session] = get_session,
        chunk_collection: str | None = None,
    ) -> None:
        self.parser = parser or PyPDFParser()
        self.chunker = chunker or Chunker()
        self.embedder = embedder
        self.vector_store = vector_store or InMemoryVectorStore()
        self.keyword_store = keyword_store or SQLiteFTSStore()
        self._session_factory = session_factory
        self.chunk_collection = (
            chunk_collection if chunk_collection is not None else get_config().vector_store.chunk_collection
        )

    def ingest(
        self,
        pdf_path: str | Path,
        *,
        title: str | None = None,
        authors: str | None = None,
        venue: str | None = None,
        year: int | None = None,
    ) -> Paper:
        """入库一篇 PDF 论文，返回写入数据库后的 :class:`Paper` 记录。"""
        parsed = self.parser.parse(pdf_path)
        text_chunks = self.chunker.chunk(parsed)

        session = self._session_factory()
        try:
            paper = self._save_paper(session, parsed, pdf_path, title, authors, venue, year)
            chunks = self._save_chunks(session, paper, text_chunks)
            session.commit()
            session.refresh(paper)
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

        self._index_chunks(paper.id, chunks)
        return paper

    def _save_paper(
        self,
        session: Session,
        parsed: ParsedPaper,
        pdf_path: str | Path,
        title: str | None,
        authors: str | None,
        venue: str | None,
        year: int | None,
    ) -> Paper:
        paper = Paper(
            title=title or parsed.title,
            authors=authors or parsed.authors,
            venue=venue or parsed.venue,
            year=year if year is not None else parsed.year,
            pdf_path=str(pdf_path),
        )
        session.add(paper)
        session.flush()  # 拿到自动生成的 paper.id
        return paper

    def _save_chunks(
        self, session: Session, paper: Paper, text_chunks
    ) -> list[Chunk]:
        chunks = [
            Chunk(
                paper_id=paper.id,
                content=tc.content,
                section=tc.section,
                chunk_index=tc.chunk_index,
            )
            for tc in text_chunks
        ]
        session.add_all(chunks)
        session.flush()
        return chunks

    def _index_chunks(self, paper_id: str, chunks: list[Chunk]) -> None:
        if not chunks:
            return
        embeddings = self.embedder.embed([c.content for c in chunks])
        metadatas = [
            {"paper_id": paper_id, "section": c.section, "chunk_index": c.chunk_index}
            for c in chunks
        ]
        self.vector_store.upsert(
            collection=self.chunk_collection,
            ids=[c.id for c in chunks],
            embeddings=embeddings,
            metadatas=metadatas,
        )
        self.keyword_store.upsert(
            collection=self.chunk_collection,
            ids=[c.id for c in chunks],
            texts=[c.content for c in chunks],
            metadatas=metadatas,
        )
