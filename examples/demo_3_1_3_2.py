"""演示 3.1（论文入库管线）与 3.2（用户驱动概念图谱）的端到端流程。

embedding 用 :class:`~dejaread.embedding.RemoteEmbedder`，对接一个 OpenAI 兼容的
embedding 服务（如 vLLM 部署的 Qwen3-Embedding）。运行前需要先启动该服务，并通过
环境变量配置地址：

    export EMBEDDING_API_BASE=http://localhost:8000/v1
    export EMBEDDING_MODEL=Qwen3-Embedding-0.6B
    python examples/demo_3_1_3_2.py

向量库 / LLM 仍用本地实现（InMemoryVectorStore + MockConceptLLM），生产环境可替换为：
    ChromaVectorStore / PromptedConceptLLM(OpenAICompatibleLLMClient(...))
"""

from __future__ import annotations

import os

from dejaread.concepts import AnnotationRequest, ConceptAnnotationService
from dejaread.concepts.linking import LinkDiscovery
from dejaread.db import Paper, init_db
from dejaread.embedding import Embedder, InMemoryVectorStore, RemoteEmbedder
from dejaread.ingestion import Chunker, IngestionPipeline
from dejaread.ingestion.parser import ParsedPaper, ParsedSection, PDFParser


class DemoParser(PDFParser):
    """演示用：跳过真实 PDF 文件 IO，直接返回构造好的解析结果。"""

    def __init__(self, parsed: ParsedPaper) -> None:
        self._parsed = parsed

    def parse(self, pdf_path):  # noqa: D401 - 接口实现
        return self._parsed


def ingest_demo_paper(
    embedder: Embedder, vector_store: InMemoryVectorStore, title: str, page_number: int, text: str
) -> Paper:
    pipeline = IngestionPipeline(
        embedder=embedder,
        parser=DemoParser(ParsedPaper(title=title, sections=[ParsedSection(page_number=page_number, text=text)])),
        chunker=Chunker(max_chars=300),
        vector_store=vector_store,
    )
    return pipeline.ingest(f"papers/{title}.pdf")


def main() -> None:
    init_db("sqlite:///:memory:")

    embedder = RemoteEmbedder(
        base_url=os.environ.get("EMBEDDING_API_BASE", "http://localhost:8000/v1"),
        model=os.environ.get("EMBEDDING_MODEL", "Qwen3-Embedding-0.6B"),
    )
    vector_store = InMemoryVectorStore()

    # ---- 3.1 论文入库管线：解析 → 分块 → embedding → 向量库 + SQLite ----
    grpo = ingest_demo_paper(
        embedder,
        vector_store,
        title="GRPO: Group Relative Policy Optimization",
        page_number=3,
        text=(
            "We use PPO as the base RL algorithm. The advantage is estimated at "
            "the group level rather than per-token."
        ),
    )
    dapo = ingest_demo_paper(
        embedder,
        vector_store,
        title="DAPO: an open-source RL system",
        page_number=4,
        text=(
            "We propose clip-higher, which removes the lower bound of the "
            "clipping range. PPO is still used as the optimizer."
        ),
    )
    print(f"入库完成：{grpo.title!r} ({grpo.id})\n         {dapo.title!r} ({dapo.id})")

    # ---- 3.2 用户驱动概念图谱：选词标注 → 解释 → 跨论文关联发现 ----
    service = ConceptAnnotationService(
        embedder=embedder,
        vector_store=vector_store,
        link_discovery=LinkDiscovery(vector_store, similarity_threshold=0.3),
    )

    result_a = service.annotate(AnnotationRequest(paper_id=grpo.id, selected_text="PPO", page_number=3))
    print(f"\n[标注 1：在 GRPO 中标注 'PPO'] {result_a.definition}")
    print(f"  关联: {result_a.links}  (图谱为空，应无关联)")

    result_b = service.annotate(AnnotationRequest(paper_id=dapo.id, selected_text="PPO", page_number=4))
    print(f"\n[标注 2：在 DAPO 中标注 'PPO'] {result_b.definition}")
    for link in result_b.links:
        print(f"  关联 -> {link.related_concept_name} ({link.link_type.value}): {link.description}")


if __name__ == "__main__":
    main()
