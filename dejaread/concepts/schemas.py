"""3.2 概念图谱模块的请求/响应数据结构。"""

from __future__ import annotations

from pydantic import BaseModel, Field

from ..db.models import LinkType


class AnnotationRequest(BaseModel):
    """用户一次选词标注的输入（对应 3.2.1 流程的第一步：前端发送选中文本 + 论文 ID + 页码）。"""

    paper_id: str
    selected_text: str
    page_number: int | None = None
    chunk_id: str | None = None


class LinkResult(BaseModel):
    """一条跨论文关联边的结果，供返回给前端展示。"""

    link_id: str
    related_concept_id: str
    related_concept_name: str
    related_paper_id: str
    link_type: LinkType
    description: str
    confidence: float


class AnnotationResult(BaseModel):
    """一次选词标注的完整结果：概念解释 + 跨论文关联（如有）。"""

    concept_id: str
    name: str
    definition: str
    context_snippet: str
    links: list[LinkResult] = Field(default_factory=list)
