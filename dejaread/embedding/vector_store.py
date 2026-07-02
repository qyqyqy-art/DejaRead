"""通用向量库接口与实现。

技术选型表中向量数据库选用 ChromaDB（本地模式）。这里把它抽象成 :class:`VectorStore`
接口，供项目内所有需要向量检索的模块共享：3.1 入库管线用它存 chunk 向量，
3.2 概念图谱（见 ``dejaread.concepts``）用它存概念向量做跨论文相似度检索，未来
3.3 记忆系统 / 3.4 笔记模块的语义检索同样可以复用。

同样提供一个纯内存实现 :class:`InMemoryVectorStore`，用于开发/测试，避免引入
chromadb 这一较重的依赖。两个实现都按 ``collection`` 名称隔离不同模块的数据。
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod

from pydantic import BaseModel

from ..config import get_config
from ..utils.utils import setup_logger

logger = setup_logger(log_dir="logs/log_vector_store", logger_name="embedding_vector_store")


class VectorMatch(BaseModel):
    """一次相似度检索命中的结果。"""

    id: str
    score: float  # 余弦相似度，越大越相似
    metadata: dict


class VectorStore(ABC):
    """向量库接口：按集合（collection）存储 id -> (embedding, metadata)。"""

    @abstractmethod
    def upsert(
        self,
        collection: str,
        ids: list[str],
        embeddings: list[list[float]],
        metadatas: list[dict] | None = None,
        documents: list[str] | None = None,
    ) -> None:
        """插入或覆盖一批向量。"""

    @abstractmethod
    def query(
        self,
        collection: str,
        embedding: list[float],
        top_k: int = 5,
        metadata_filter: dict | None = None,
    ) -> list[VectorMatch]:
        """在指定集合中做向量 Top-K 检索，按相似度降序返回。

        metadata_filter 为简单等值过滤字典（如 {"paper_id": "xxx"}），
        在检索阶段就完成过滤，返回结果已是目标子集内的 top-k。
        """

    @abstractmethod
    def delete(self, collection: str, ids: list[str]) -> None:
        """删除指定集合中给定 id 的向量。"""


class InMemoryVectorStore(VectorStore):
    """纯内存向量库，线性扫描做余弦相似度检索。仅用于开发/测试规模的数据量。"""

    def __init__(self) -> None:
        self._collections: dict[str, dict[str, tuple[list[float], dict]]] = {}

    def upsert(
        self,
        collection: str,
        ids: list[str],
        embeddings: list[list[float]],
        metadatas: list[dict] | None = None,
        documents: list[str] | None = None,
    ) -> None:
        store = self._collections.setdefault(collection, {})
        metadatas = metadatas or [{} for _ in ids]
        for id_, embedding, metadata in zip(ids, embeddings, metadatas):
            store[id_] = (embedding, metadata)

    def query(
        self,
        collection: str,
        embedding: list[float],
        top_k: int = 5,
        metadata_filter: dict | None = None,
    ) -> list[VectorMatch]:
        store = self._collections.get(collection, {})
        scored = [
            VectorMatch(id=id_, score=self._cosine(embedding, vec), metadata=meta)
            for id_, (vec, meta) in store.items()
            if not metadata_filter or all(meta.get(k) == v for k, v in metadata_filter.items())
        ]
        scored.sort(key=lambda m: m.score, reverse=True)
        return scored[:top_k]

    def delete(self, collection: str, ids: list[str]) -> None:
        store = self._collections.get(collection)
        if not store:
            return
        for id_ in ids:
            store.pop(id_, None)

    @staticmethod
    def _cosine(a: list[float], b: list[float]) -> float:
        dot = sum(x * y for x, y in zip(a, b))
        norm_a = math.sqrt(sum(x * x for x in a)) or 1.0
        norm_b = math.sqrt(sum(y * y for y in b)) or 1.0
        return dot / (norm_a * norm_b)


class ChromaVectorStore(VectorStore):
    """基于 ChromaDB 的向量库实现（本地持久化模式）。"""

    def __init__(self, persist_directory: str | None = None) -> None:
        try:
            import chromadb
        except ImportError as exc:  # pragma: no cover - 依赖缺失时的提示
            raise ImportError(
                "ChromaVectorStore 需要安装 chromadb：pip install chromadb"
            ) from exc

        if persist_directory is None:
            persist_directory = get_config().vector_store.persist_directory
        self._client = chromadb.PersistentClient(path=persist_directory)

    def _get_collection(self, name: str):
        return self._client.get_or_create_collection(name=name)

    def upsert(
        self,
        collection: str,
        ids: list[str],
        embeddings: list[list[float]],
        metadatas: list[dict] | None = None,
        documents: list[str] | None = None,
    ) -> None:
        col = self._get_collection(collection)
        col.upsert(ids=ids, embeddings=embeddings, metadatas=metadatas, documents=documents)
        logger.info("upsert 完成：collection=%s ids=%d", collection, len(ids))

    def query(
        self,
        collection: str,
        embedding: list[float],
        top_k: int = 5,
        metadata_filter: dict | None = None,
    ) -> list[VectorMatch]:
        col = self._get_collection(collection)
        if col.count() == 0:
            return []
        where = self._build_where(metadata_filter)
        result = col.query(
            query_embeddings=[embedding],
            n_results=min(top_k, col.count()),
            where=where,
        )
        ids = result["ids"][0]
        distances = result["distances"][0]
        metadatas = result["metadatas"][0]
        # Chroma 默认距离度量是 L2，这里转换成与余弦相似度同向（越大越相似）的分数。
        matches = [
            VectorMatch(id=id_, score=1.0 / (1.0 + dist), metadata=meta or {})
            for id_, dist, meta in zip(ids, distances, metadatas)
        ]
        logger.info("query 完成：collection=%s matches=%d", collection, len(matches))
        return matches

    @staticmethod
    def _build_where(metadata_filter: dict | None) -> dict | None:
        if not metadata_filter:
            return None
        items = list(metadata_filter.items())
        if len(items) == 1:
            k, v = items[0]
            return {k: {"$eq": v}}
        return {"$and": [{k: {"$eq": v}} for k, v in items]}

    def delete(self, collection: str, ids: list[str]) -> None:
        if not ids:
            return
        self._get_collection(collection).delete(ids=ids)
        logger.info("delete 完成：collection=%s ids=%d", collection, len(ids))
