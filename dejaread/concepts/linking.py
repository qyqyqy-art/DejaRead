"""跨论文关联发现（3.2.3）：在已有概念库中做混合检索，找出值得让 LLM 精排的候选。"""

from __future__ import annotations

from pydantic import BaseModel

from ..config import get_config
from ..embedding import VectorStore
from ..keyword import KeywordStore
from ..retrieval.hybrid import fuse_rrf


class LinkCandidate(BaseModel):
    """一个候选关联概念（向量+关键词粗筛通过，尚未经过 LLM 精排）。"""

    concept_id: str
    paper_id: str
    score: float  # RRF 融合分数


class LinkDiscovery:
    """新概念 embedding/文本 → 向量+关键词混合检索 → 过滤，产出候选列表。

    向量侧先按 ``similarity_threshold`` 过滤掉相似度太低的噪声，再与关键词侧
    （BM25 精确匹配，本身就是较强的相关信号，不额外设阈值）按 RRF 融合排序。
    只在其中一侧命中也会被保留——例如概念名完全相同但 embedding 距离稍远，或
    语义相近但用词不同——交由后续 LLM 精排决定是否真的构成关联。

    LLM 精排（判断是否真的存在关联、分类关联类型）由调用方（见
    :mod:`dejaread.concepts.annotation_service`）结合 :class:`~dejaread.concepts.llm.ConceptLLM`
    完成，这里只负责"粗筛"。
    """

    def __init__(
        self,
        vector_store: VectorStore,
        keyword_store: KeywordStore | None = None,
        similarity_threshold: float | None = None,
        top_k: int | None = None,
        collection: str | None = None,
        rrf_k: int | None = None,
    ) -> None:
        linking_config = get_config().linking
        self.vector_store = vector_store
        self.keyword_store = keyword_store
        self.similarity_threshold = (
            similarity_threshold if similarity_threshold is not None else linking_config.similarity_threshold
        )
        self.top_k = top_k if top_k is not None else linking_config.top_k
        self.rrf_k = rrf_k if rrf_k is not None else linking_config.rrf_k
        self.collection = collection if collection is not None else get_config().vector_store.concept_collection

    def find_candidates(
        self,
        new_embedding: list[float],
        query_text: str = "",
        *,
        exclude_concept_id: str | None = None,
        exclude_paper_id: str | None = None,
    ) -> list[LinkCandidate]:
        """检索与新概念相关的已有概念，排除自身所在论文（关联发现的目标是"跨论文"）。"""
        vector_matches = self.vector_store.query(
            collection=self.collection, embedding=new_embedding, top_k=self.top_k
        )
        vector_matches = [m for m in vector_matches if m.score >= self.similarity_threshold]

        keyword_matches = []
        if self.keyword_store is not None and query_text:
            keyword_matches = self.keyword_store.query(
                collection=self.collection, query_text=query_text, top_k=self.top_k
            )

        candidates: list[LinkCandidate] = []
        for match in fuse_rrf(vector_matches, keyword_matches, self.rrf_k):
            if match.id == exclude_concept_id:
                continue
            paper_id = match.metadata.get("paper_id")
            if exclude_paper_id is not None and paper_id == exclude_paper_id:
                continue
            candidates.append(
                LinkCandidate(concept_id=match.id, paper_id=paper_id, score=match.fused_score)
            )
        return candidates
