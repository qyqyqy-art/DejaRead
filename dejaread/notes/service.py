"""笔记系统（3.4）：文件存储 + DB/索引同步。

核心策略：notes/{paper_id}.md 是唯一真实来源；每次 ``save`` 都全量重建——重新解析
全文、清空旧的 NoteSection 及其向量/关键词索引、写入新内容。不做增量 diff，逻辑简单，
不会出现"文件和 DB 不一致"的中间态。
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

from sqlalchemy.orm import Session

from ..config import get_config
from ..db import Note, NoteSection, Paper, get_session
from ..embedding import Embedder, VectorStore
from ..keyword import KeywordStore
from ..utils.utils import setup_logger
from .parser import split_sections

logger = setup_logger(log_dir="logs/log_notes", logger_name="notes_service")


def _ensure_heading_and_append(markdown: str, heading: str, block: str) -> str:
    """在第一个 `## {heading}` 段末尾追加 block；该标题不存在则在文件末尾新建。"""
    heading_line = f"## {heading}"
    lines = markdown.splitlines()

    for i, line in enumerate(lines):
        if line.strip() == heading_line:
            insert_at = len(lines)
            for j in range(i + 1, len(lines)):
                if lines[j].startswith("## "):
                    insert_at = j
                    break
            new_lines = lines[:insert_at] + ["", block.rstrip("\n"), ""] + lines[insert_at:]
            return "\n".join(new_lines).rstrip("\n") + "\n"

    new_lines = lines + ["", heading_line, "", block.rstrip("\n"), ""]
    return "\n".join(new_lines).rstrip("\n") + "\n"


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
        logger.info("笔记文件不存在，已按模板创建：paper_id=%s path=%s", paper_id, path)
        return content

    def save(self, paper_id: str, raw_markdown: str) -> None:
        """全量重建：写文件 → 清除旧 section 索引 → 重新解析 → 写入新索引。"""
        logger.info("save 开始：paper_id=%s content_len=%d", paper_id, len(raw_markdown))
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
            logger.exception("save 事务失败，已回滚：paper_id=%s", paper_id)
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
            self.vector_store.upsert(self.section_collection, ids, embeddings, metadatas, documents=texts)
            self.keyword_store.upsert(self.section_collection, ids, texts, metadatas)

        logger.info("save 完成：paper_id=%s old_sections=%d new_sections=%d", paper_id, len(old_section_ids), len(new_section_data))

    def append_concept(
        self,
        paper_id: str,
        concept_name: str,
        definition: str,
        context_snippet: str,
        links_text: str = "",
    ) -> str:
        """选词标注后一键插入笔记的 "## 标注概念" 段。返回保存后的最新全文。"""
        block = (
            f"### {concept_name}\n\n"
            f"**语境化解释**：{definition}\n\n"
            f"**上下文片段**：{context_snippet}"
        )
        if links_text:
            block += f"\n\n{links_text}"

        current = self.read_or_create(paper_id)
        updated = _ensure_heading_and_append(current, "标注概念", block)
        self.save(paper_id, updated)
        return updated

    def append_conversation_summary(self, paper_id: str, summary_text: str) -> str:
        """将对话摘要插入笔记的 "## 对话摘要" 段。返回保存后的最新全文。"""
        timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
        block = f"**{timestamp}**\n\n{summary_text}"

        current = self.read_or_create(paper_id)
        updated = _ensure_heading_and_append(current, "对话摘要", block)
        self.save(paper_id, updated)
        return updated
