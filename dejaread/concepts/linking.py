"""跨论文关联发现（3.2.3）：在已有概念库中做向量检索，找出值得让 LLM 精排的候选。"""

from __future__ import annotations

from dataclasses import dataclass

from ..config import get_config
from ..embedding import VectorStore


@dataclass
class LinkCandidate:
    """一个候选关联概念（向量相似度初筛通过，尚未经过 LLM 精排）。"""

    concept_id: str
    paper_id: str
    score: float


class LinkDiscovery:
    """新概念 embedding → 向量 Top-K 检索 → 相似度阈值过滤，产出候选列表。

    LLM 精排（判断是否真的存在关联、分类关联类型）由调用方（见
    :mod:`dejaread.concepts.annotation_service`）结合 :class:`~dejaread.concepts.llm.ConceptLLM`
    完成，这里只负责"粗筛"。
    """

    def __init__(
        self,
        vector_store: VectorStore,
        similarity_threshold: float | None = None,
        top_k: int | None = None,
        collection: str | None = None,
    ) -> None:
        linking_config = get_config().linking
        self.vector_store = vector_store
        self.similarity_threshold = (
            similarity_threshold if similarity_threshold is not None else linking_config.similarity_threshold
        )
        self.top_k = top_k if top_k is not None else linking_config.top_k
        self.collection = collection if collection is not None else get_config().vector_store.concept_collection

    def find_candidates(
        self,
        new_embedding: list[float],
        *,
        exclude_concept_id: str | None = None,
        exclude_paper_id: str | None = None,
    ) -> list[LinkCandidate]:
        """检索与新概念相似的已有概念，排除自身所在论文（关联发现的目标是"跨论文"）。"""
        matches = self.vector_store.query(
            collection=self.collection, embedding=new_embedding, top_k=self.top_k
        )
        candidates: list[LinkCandidate] = []
        for match in matches:
            if match.score < self.similarity_threshold:
                continue
            if match.id == exclude_concept_id:
                continue
            paper_id = match.metadata.get("paper_id")
            if exclude_paper_id is not None and paper_id == exclude_paper_id:
                continue
            candidates.append(
                LinkCandidate(concept_id=match.id, paper_id=paper_id, score=match.score)
            )
        return candidates
