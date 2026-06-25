"""向量检索 + 关键词检索的混合查询层（RRF 融合）。

向量 cosine 分数和 BM25 分数量纲不同，无法直接加权相加，因此用 Reciprocal Rank
Fusion（RRF）——只依赖每路结果内部的排名，是异构检索融合的标准做法。当前没有调用方
（QA Agent 尚未实现，见设计文档 3.5 节），作为独立可测试组件先行落地，供未来复用。
"""

from __future__ import annotations

from dataclasses import dataclass

from ..embedding import Embedder, VectorStore
from ..keyword import KeywordStore


@dataclass
class HybridMatch:
    """一次混合检索命中的结果。"""

    id: str
    fused_score: float
    metadata: dict
    vector_rank: int | None = None
    keyword_rank: int | None = None


class HybridRetriever:
    """对同一个 collection 同时做向量检索和关键词检索，按 RRF 融合排序后返回。"""

    def __init__(
        self,
        vector_store: VectorStore,
        keyword_store: KeywordStore,
        embedder: Embedder,
        collection: str,
        top_k: int = 10,
        rrf_k: int = 60,
    ) -> None:
        self.vector_store = vector_store
        self.keyword_store = keyword_store
        self.embedder = embedder
        self.collection = collection
        self.top_k = top_k
        self.rrf_k = rrf_k

    def search(self, query_text: str) -> list[HybridMatch]:
        query_embedding = self.embedder.embed_query(query_text)
        vector_hits = self.vector_store.query(self.collection, query_embedding, top_k=self.top_k)
        keyword_hits = self.keyword_store.query(self.collection, query_text, top_k=self.top_k)
        return self._fuse_rrf(vector_hits, keyword_hits)

    def _fuse_rrf(self, vector_hits, keyword_hits) -> list[HybridMatch]:
        matches: dict[str, HybridMatch] = {}

        for rank, hit in enumerate(vector_hits, start=1):
            match = matches.setdefault(
                hit.id, HybridMatch(id=hit.id, fused_score=0.0, metadata=hit.metadata)
            )
            match.vector_rank = rank
            match.fused_score += 1 / (self.rrf_k + rank)

        for rank, hit in enumerate(keyword_hits, start=1):
            match = matches.setdefault(
                hit.id, HybridMatch(id=hit.id, fused_score=0.0, metadata=hit.metadata)
            )
            match.keyword_rank = rank
            match.fused_score += 1 / (self.rrf_k + rank)

        return sorted(matches.values(), key=lambda m: m.fused_score, reverse=True)
