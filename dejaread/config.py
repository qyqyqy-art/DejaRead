"""集中式配置加载模块。

各功能模块（数据库连接、分块、embedding、向量库、跨论文关联、LLM、概念标注）的可变
参数统一在 ``config/config.yaml`` 中维护，业务代码通过 :func:`get_config` 取得全局
单例 :class:`AppConfig`，不直接读文件。

配置文件路径优先级：环境变量 ``DEJAREAD_CONFIG`` > 默认路径
``<项目根目录>/config/config.yaml``。文件中缺失的字段使用各 dataclass 字段的默认值，
因此配置文件本身可以只写需要覆盖的部分。
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field, fields
from pathlib import Path

import yaml

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_DEFAULT_CONFIG_PATH = _PROJECT_ROOT / "config" / "config.yaml"


@dataclass
class DatabaseConfig:
    url: str = "sqlite:///dejaread.db"
    echo: bool = False


@dataclass
class ChunkingConfig:
    max_chars: int = 1200
    overlap_chars: int = 150


@dataclass
class EmbeddingConfig:
    base_url: str = "http://localhost:8000/v1"
    model: str = "Qwen3-Embedding-0.6B"
    api_key: str | None = None
    timeout: float = 10.0
    dimensions: int | None = None
    batch_size: int = 64
    use_query_instruct: bool = True
    query_task: str = "Given a user query, retrieve relevant documents related to the query"


@dataclass
class VectorStoreConfig:
    backend: str = "memory"
    persist_directory: str = "./chroma_data"
    chunk_collection: str = "paper_chunks"
    concept_collection: str = "concepts"


@dataclass
class LinkingConfig:
    similarity_threshold: float = 0.6
    top_k: int = 5
    rrf_k: int = 60


@dataclass
class LLMConfig:
    model: str = "qwen3:8b"
    base_url: str | None = None
    api_key: str = "ollama"


@dataclass
class AnnotationConfig:
    context_window_chars: int = 200


@dataclass
class KeywordStoreConfig:
    db_path: str | None = None  # None 时复用 database.url 解析出的 sqlite 文件路径


@dataclass
class AppConfig:
    database: DatabaseConfig = field(default_factory=DatabaseConfig)
    chunking: ChunkingConfig = field(default_factory=ChunkingConfig)
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    vector_store: VectorStoreConfig = field(default_factory=VectorStoreConfig)
    linking: LinkingConfig = field(default_factory=LinkingConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    annotation: AnnotationConfig = field(default_factory=AnnotationConfig)
    keyword_store: KeywordStoreConfig = field(default_factory=KeywordStoreConfig)


def _build_section(section_cls: type, data: dict | None) -> object:
    if not data:
        return section_cls()
    valid_keys = {f.name for f in fields(section_cls)}
    unknown = set(data) - valid_keys
    if unknown:
        raise ValueError(f"{section_cls.__name__} 不支持的配置项: {sorted(unknown)}")
    return section_cls(**data)


def _build_app_config(raw: dict) -> AppConfig:
    return AppConfig(
        database=_build_section(DatabaseConfig, raw.get("database")),
        chunking=_build_section(ChunkingConfig, raw.get("chunking")),
        embedding=_build_section(EmbeddingConfig, raw.get("embedding")),
        vector_store=_build_section(VectorStoreConfig, raw.get("vector_store")),
        linking=_build_section(LinkingConfig, raw.get("linking")),
        llm=_build_section(LLMConfig, raw.get("llm")),
        annotation=_build_section(AnnotationConfig, raw.get("annotation")),
        keyword_store=_build_section(KeywordStoreConfig, raw.get("keyword_store")),
    )


def load_config(path: str | Path | None = None) -> AppConfig:
    """从 YAML 文件加载配置，返回一份新的 :class:`AppConfig`（不影响全局单例）。

    文件不存在时返回全部使用默认值的 :class:`AppConfig`，方便测试/开发环境零配置运行。
    """
    config_path = Path(path) if path is not None else _DEFAULT_CONFIG_PATH
    if not config_path.exists():
        return AppConfig()

    with config_path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    return _build_app_config(raw)


_config: AppConfig | None = None


def get_config() -> AppConfig:
    """获取全局配置单例，首次调用时按 ``DEJAREAD_CONFIG`` 环境变量或默认路径加载。"""
    global _config
    if _config is None:
        _config = load_config(os.environ.get("DEJAREAD_CONFIG"))
    return _config


def set_config(config: AppConfig) -> None:
    """覆盖全局配置单例（主要用于测试）。"""
    global _config
    _config = config


def reset_config() -> None:
    """重置全局配置单例，下次 :func:`get_config` 调用会重新从文件加载。"""
    global _config
    _config = None
