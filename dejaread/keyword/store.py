"""关键词索引接口与 SQLite FTS5 实现。

镜像 :mod:`dejaread.embedding.vector_store` 的 ``VectorStore``/``VectorMatch`` 风格：
按 ``collection`` 名称隔离数据，``query`` 只返回 ``id + score + metadata``，原文由调用方
按 id 去主表取。唯一实现 ``SQLiteFTSStore`` 直接用标准库 ``sqlite3`` 连接同一个数据库文件，
不经过 SQLAlchemy ORM——FTS5 的 ``CREATE VIRTUAL TABLE`` 和 ``bm25()`` 是 SQLite 专有能力，
和向量库完全独立于 SQLAlchemy 的解耦思路一致。
"""

from __future__ import annotations

import json
import sqlite3
from abc import ABC, abstractmethod
from dataclasses import dataclass

from ..config import get_config
from .tokenizer import segment_for_index, segment_for_query


@dataclass
class KeywordMatch:
    """一次关键词检索命中的结果。"""

    id: str
    score: float  # 取 -bm25()，越大越相关，与 VectorMatch.score 语义一致
    metadata: dict


class KeywordStore(ABC):
    """关键词库接口：按集合（collection）存储 id -> (分词文本, metadata)。"""

    @abstractmethod
    def upsert(
        self,
        collection: str,
        ids: list[str],
        texts: list[str],
        metadatas: list[dict] | None = None,
    ) -> None:
        """插入或覆盖一批文本的关键词索引。"""

    @abstractmethod
    def query(self, collection: str, query_text: str, top_k: int = 5) -> list[KeywordMatch]:
        """在指定集合中做关键词检索，按 BM25 相关度降序返回。"""

    @abstractmethod
    def delete(self, collection: str, ids: list[str]) -> None:
        """删除指定集合中给定 id 的关键词索引项。"""


class SQLiteFTSStore(KeywordStore):
    """基于 SQLite FTS5 的关键词库实现，每个 collection 对应一张 ``fts_<collection>`` 虚拟表。"""

    def __init__(self, db_path: str | None = None) -> None:
        if db_path is None:
            db_path = self._default_db_path()
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._known_collections: set[str] = set()

    @staticmethod
    def _default_db_path() -> str:
        config = get_config()
        if config.keyword_store.db_path is not None:
            return config.keyword_store.db_path
        url = config.database.url
        prefix = "sqlite:///"
        if not url.startswith(prefix):
            raise ValueError(f"SQLiteFTSStore 只支持 sqlite 数据库，当前 database.url={url!r}")
        return url[len(prefix) :]

    def _ensure_collection(self, collection: str) -> str:
        table = f"fts_{collection}"
        if table not in self._known_collections:
            self._conn.execute(
                f"CREATE VIRTUAL TABLE IF NOT EXISTS {table} USING fts5("
                "content_tokens, id UNINDEXED, metadata_json UNINDEXED)"
            )
            self._conn.commit()
            self._known_collections.add(table)
        return table

    def upsert(
        self,
        collection: str,
        ids: list[str],
        texts: list[str],
        metadatas: list[dict] | None = None,
    ) -> None:
        table = self._ensure_collection(collection)
        metadatas = metadatas or [{} for _ in ids]
        for id_, text, metadata in zip(ids, texts, metadatas):
            self._conn.execute(f"DELETE FROM {table} WHERE id = ?", (id_,))
            self._conn.execute(
                f"INSERT INTO {table}(content_tokens, id, metadata_json) VALUES (?, ?, ?)",
                (segment_for_index(text), id_, json.dumps(metadata)),
            )
        self._conn.commit()

    def query(self, collection: str, query_text: str, top_k: int = 5) -> list[KeywordMatch]:
        table = self._ensure_collection(collection)
        match_query = segment_for_query(query_text)
        if not match_query:
            return []
        rows = self._conn.execute(
            f"SELECT id, metadata_json, bm25({table}) AS rank FROM {table} "
            f"WHERE {table} MATCH ? ORDER BY rank LIMIT ?",
            (match_query, top_k),
        ).fetchall()
        return [
            KeywordMatch(id=id_, score=-rank, metadata=json.loads(metadata_json))
            for id_, metadata_json, rank in rows
        ]

    def delete(self, collection: str, ids: list[str]) -> None:
        if not ids:
            return
        table = self._ensure_collection(collection)
        placeholders = ",".join("?" for _ in ids)
        self._conn.execute(f"DELETE FROM {table} WHERE id IN ({placeholders})", ids)
        self._conn.commit()
