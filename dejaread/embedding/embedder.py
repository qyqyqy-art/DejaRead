"""通用 Embedding 接口与实现。

项目内所有需要做向量化的模块——3.1 论文入库管线的 chunk embedding、3.2 概念图谱的
概念 embedding、未来 3.3 记忆系统 / 3.4 笔记模块——共享同一套 :class:`Embedder` 接口。
唯一实现 :class:`RemoteEmbedder` 调用独立部署的 embedding 服务（如 vLLM 跑
Qwen3-Embedding / bge-m3），业务进程不加载任何模型权重。
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from ..config import get_config


class Embedder(ABC):
    """文本向量化接口。"""

    @property
    @abstractmethod
    def dimension(self) -> int:
        """向量维度。"""

    @abstractmethod
    def embed(self, texts: list[str]) -> list[list[float]]:
        """批量将文本编码为向量。"""

    def embed_one(self, text: str) -> list[float]:
        return self.embed([text])[0]

    def embed_query(self, text: str) -> list[float]:
        """编码检索时的 query 文本。

        部分模型（如 Qwen3 Embedding 系列）建议给 query 加 instruct 前缀以提升检索
        效果，而 document 侧不需要——默认实现两者一致，只有需要区分的 embedder
        （如 :class:`RemoteEmbedder`）才重写它。
        """
        return self.embed_one(text)


class RemoteEmbedder(Embedder):
    """通过 HTTP API 调用远程 embedding 服务（vLLM OpenAI-compatible ``/embeddings``）。

    ``base_url`` 需包含版本路径，例如 ``http://host:8000/v1``，请求会发往
    ``{base_url}/embeddings``，payload/response 格式对齐 OpenAI embeddings 接口：

        POST {base_url}/embeddings
        payload: {"model": "...", "input": [...], ...}
        resp:    {"data": [{"embedding": [...]}, ...], ...}

    项目唯一的 embedding 实现：把 embedding 模型单独部署成服务（如 vLLM 跑
    Qwen3-Embedding），业务进程不需要加载模型权重，多个服务可共享同一个 embedding
    服务实例。
    """

    def __init__(
        self,
        base_url: str | None = None,
        model: str | None = None,
        api_key: str | None = None,
        timeout: float | None = None,
        dimensions: int | None = None,
        batch_size: int | None = None,
        use_query_instruct: bool | None = None,
        query_task: str | None = None,
    ) -> None:
        try:
            import requests
        except ImportError as exc:  # pragma: no cover - 依赖缺失时的提示
            raise ImportError(
                "RemoteEmbedder 需要安装 requests：pip install requests"
            ) from exc

        embedding_config = get_config().embedding
        self._requests = requests
        self.base_url = (base_url if base_url is not None else embedding_config.base_url).rstrip("/")
        self.model = model if model is not None else embedding_config.model
        self.api_key = api_key if api_key is not None else embedding_config.api_key
        self.timeout = timeout if timeout is not None else embedding_config.timeout
        self.dimensions = dimensions if dimensions is not None else embedding_config.dimensions
        batch_size = batch_size if batch_size is not None else embedding_config.batch_size
        self.batch_size = max(1, int(batch_size))
        self.use_query_instruct = (
            use_query_instruct if use_query_instruct is not None else embedding_config.use_query_instruct
        )
        self.query_task = query_task if query_task is not None else embedding_config.query_task

        self._dimension = dimensions

    @property
    def dimension(self) -> int:
        if self._dimension is None:
            # 维度未显式配置时，发一次探测请求并缓存结果。
            self._dimension = len(self.embed_one("probe"))
        return self._dimension

    def embed(self, texts: list[str]) -> list[list[float]]:
        """编码 document 侧文本（建库用），按 ``batch_size`` 分批请求，不加 instruct 前缀。"""
        if not texts:
            return []
        vectors: list[list[float]] = []
        for i in range(0, len(texts), self.batch_size):
            vectors.extend(self._post_embeddings(texts[i : i + self.batch_size]))
        return vectors

    def embed_query(self, text: str) -> list[float]:
        """编码检索 query 文本，按需加 instruct 前缀（Qwen3 Embedding 推荐用法）。"""
        formatted = self._format_query(text)
        return self._post_embeddings([formatted])[0]

    def _format_query(self, query: str) -> str:
        if not self.use_query_instruct:
            return query
        return f"Instruct: {self.query_task}\nQuery:{query}"

    def _headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    def _post_embeddings(self, inputs: list[str]) -> list[list[float]]:
        url = f"{self.base_url}/embeddings"
        payload: dict = {"model": self.model, "input": inputs}
        if self.dimensions is not None:
            payload["dimensions"] = int(self.dimensions)

        response = self._requests.post(url, headers=self._headers(), json=payload, timeout=self.timeout)
        if response.status_code >= 400:
            raise RuntimeError(f"Embedding API failed: {response.status_code} {response.text}")

        data = response.json()
        if "data" not in data:
            raise RuntimeError(f"Unexpected embedding response: {data}")
        return [item["embedding"] for item in data["data"]]
