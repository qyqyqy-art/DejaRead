"""笔记系统（3.4）：文件存储 + DB/索引同步。

核心策略：notes/{paper_id}.md 是唯一真实来源；每次 ``save`` 都全量重建——重新解析
全文、清空旧的 NoteSection 及其向量/关键词索引、写入新内容。不做增量 diff，逻辑简单，
不会出现"文件和 DB 不一致"的中间态。
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable

from sqlalchemy.orm import Session

from ..config import get_config
from ..db import Note, NoteSection, Paper, get_session
from ..embedding import Embedder, VectorStore
from ..keyword import KeywordStore
from .parser import split_sections


class NotesService:
    """笔记文件的读写 + Section 级向量/关键词索引同步。"""

    def __init__(
        self,
        embedder: Embedder,
        vector_store: VectorStore,
        keyword_store: KeywordStore,
        notes_dir: str | Path | None = None,
        section_collection: str | None = None,
        session_factory: Callable[[], Session] = get_session,
    ) -> None:
        notes_config = get_config().notes
        self.embedder = embedder
        self.vector_store = vector_store
        self.keyword_store = keyword_store
        self.notes_dir = Path(notes_dir if notes_dir is not None else notes_config.notes_dir)
        self.section_collection = (
            section_collection if section_collection is not None else notes_config.section_collection
        )
        self._session_factory = session_factory

    def _file_path(self, paper_id: str) -> Path:
        return self.notes_dir / f"{paper_id}.md"

    def read_or_create(self, paper_id: str) -> str:
        """读取笔记全文；笔记文件不存在时按模板 ``# {论文标题}`` 创建并返回。"""
        path = self._file_path(paper_id)
        if path.exists():
            return path.read_text(encoding="utf-8")

        session = self._session_factory()
        try:
            paper = session.get(Paper, paper_id)
            if paper is None:
                raise ValueError(f"未找到论文: {paper_id}")
            title = paper.title
        finally:
            session.close()

        content = f"# {title}\n"
        self.notes_dir.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        return content

    def save(self, paper_id: str, raw_markdown: str) -> None:
        """全量重建：写文件 → 清除旧 section 索引 → 重新解析 → 写入新索引。"""
        self.notes_dir.mkdir(parents=True, exist_ok=True)
        path = self._file_path(paper_id)
        path.write_text(raw_markdown, encoding="utf-8")

        session = self._session_factory()
        try:
            old_section_ids = [
                row.id for row in session.query(NoteSection).filter_by(paper_id=paper_id).all()
            ]

            note = session.get(Note, paper_id)
            if note is None:
                note = Note(paper_id=paper_id, file_path=str(path))
                session.add(note)
            else:
                note.file_path = str(path)

            session.query(NoteSection).filter_by(paper_id=paper_id).delete()

            parsed_sections = split_sections(raw_markdown)
            new_sections = [
                NoteSection(
                    paper_id=paper_id,
                    heading=ps.heading,
                    content=ps.content,
                    section_index=ps.section_index,
                )
                for ps in parsed_sections
            ]
            session.add_all(new_sections)
            session.flush()
            new_section_data = [
                {"id": s.id, "heading": s.heading, "content": s.content} for s in new_sections
            ]
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

        # 向量/关键词库写操作放在 ORM 事务提交并关闭 session 之后：两者是独立的 sqlite
        # 连接，若 ORM 事务仍持有写锁会触发 "database is locked"（与 annotation_service
        # 的写入顺序一致）。
        self.vector_store.delete(self.section_collection, old_section_ids)
        self.keyword_store.delete(self.section_collection, old_section_ids)

        if new_section_data:
            texts = [f"{s['heading']}\n{s['content']}" for s in new_section_data]
            embeddings = self.embedder.embed(texts)
            ids = [s["id"] for s in new_section_data]
            metadatas = [
                {"paper_id": paper_id, "heading": s["heading"], "note_section_id": s["id"]}
                for s in new_section_data
            ]
            self.vector_store.upsert(self.section_collection, ids, embeddings, metadatas)
            self.keyword_store.upsert(self.section_collection, ids, texts, metadatas)
