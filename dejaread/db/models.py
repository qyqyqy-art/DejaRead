"""ORM 模型定义，对应设计文档第 5 节数据库 Schema 中与 3.1 / 3.2 模块相关的表。"""

from __future__ import annotations

import enum
import uuid
from datetime import datetime
from typing import Optional

from sqlalchemy import DateTime, Enum, Float, ForeignKey, Integer, String, Text, func
from sqlalchemy.orm import Mapped, mapped_column, relationship

from .base import Base


def _new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:12]}"


class UserFamiliarity(str, enum.Enum):
    """用户对概念的熟悉程度。"""

    low = "low"
    medium = "medium"
    high = "high"


class LinkType(str, enum.Enum):
    """跨论文概念关联类型，定义见设计文档 3.2.3。"""

    same_concept = "same_concept"
    evolution = "evolution"
    contrast = "contrast"
    dependency = "dependency"
    generalization = "generalization"


class Paper(Base):
    """论文元数据表（3.1 入库管线产物）。"""

    __tablename__ = "papers"

    id: Mapped[str] = mapped_column(String, primary_key=True, default=lambda: _new_id("paper"))
    title: Mapped[str] = mapped_column(String, nullable=False)
    authors: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    venue: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    year: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    pdf_path: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    added_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())

    chunks: Mapped[list["Chunk"]] = relationship(
        back_populates="paper", cascade="all, delete-orphan", order_by="Chunk.chunk_index"
    )
    concepts: Mapped[list["Concept"]] = relationship(
        back_populates="paper", cascade="all, delete-orphan"
    )

    def __repr__(self) -> str:  # pragma: no cover - 仅用于调试输出
        return f"<Paper id={self.id!r} title={self.title!r}>"


class Chunk(Base):
    """论文分块表（3.1 智能分块产物，内容向量化后写入 ChromaDB，本表存元数据）。"""

    __tablename__ = "chunks"

    id: Mapped[str] = mapped_column(String, primary_key=True, default=lambda: _new_id("chunk"))
    paper_id: Mapped[str] = mapped_column(ForeignKey("papers.id"), nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    section: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    chunk_index: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)

    paper: Mapped["Paper"] = relationship(back_populates="chunks")
    concepts: Mapped[list["Concept"]] = relationship(back_populates="source_chunk")

    def __repr__(self) -> str:  # pragma: no cover
        return f"<Chunk id={self.id!r} paper_id={self.paper_id!r} index={self.chunk_index}>"


class Concept(Base):
    """概念节点表（3.2 用户驱动概念图谱，由选词标注渐进式构建）。"""

    __tablename__ = "concepts"

    id: Mapped[str] = mapped_column(String, primary_key=True, default=lambda: _new_id("concept"))
    name: Mapped[str] = mapped_column(String, nullable=False)
    paper_id: Mapped[str] = mapped_column(ForeignKey("papers.id"), nullable=False)
    selected_text: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    page_number: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    source_chunk_id: Mapped[Optional[str]] = mapped_column(
        ForeignKey("chunks.id"), nullable=True
    )
    context_snippet: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    definition: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    user_familiarity: Mapped[UserFamiliarity] = mapped_column(
        Enum(UserFamiliarity), default=UserFamiliarity.medium
    )
    user_perspective: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    related_confusions: Mapped[Optional[str]] = mapped_column(Text, nullable=True)  # JSON array
    discussion_count: Mapped[int] = mapped_column(Integer, default=1)
    last_discussed: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())

    paper: Mapped["Paper"] = relationship(back_populates="concepts")
    source_chunk: Mapped[Optional["Chunk"]] = relationship(back_populates="concepts")

    outgoing_links: Mapped[list["ConceptLink"]] = relationship(
        back_populates="source",
        foreign_keys="ConceptLink.source_id",
        cascade="all, delete-orphan",
    )
    incoming_links: Mapped[list["ConceptLink"]] = relationship(
        back_populates="target",
        foreign_keys="ConceptLink.target_id",
        cascade="all, delete-orphan",
    )

    def __repr__(self) -> str:  # pragma: no cover
        return f"<Concept id={self.id!r} name={self.name!r} paper_id={self.paper_id!r}>"


class ConceptLink(Base):
    """概念关联边表（跨论文关联发现产物，见 3.2.3）。"""

    __tablename__ = "concept_links"

    id: Mapped[str] = mapped_column(String, primary_key=True, default=lambda: _new_id("link"))
    source_id: Mapped[str] = mapped_column(ForeignKey("concepts.id"), nullable=False)
    target_id: Mapped[str] = mapped_column(ForeignKey("concepts.id"), nullable=False)
    link_type: Mapped[LinkType] = mapped_column(Enum(LinkType), nullable=False)
    description: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    confidence: Mapped[float] = mapped_column(Float, default=0.0)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())

    source: Mapped["Concept"] = relationship(
        back_populates="outgoing_links", foreign_keys=[source_id]
    )
    target: Mapped["Concept"] = relationship(
        back_populates="incoming_links", foreign_keys=[target_id]
    )

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"<ConceptLink {self.source_id!r} -[{self.link_type}]-> "
            f"{self.target_id!r}>"
        )
