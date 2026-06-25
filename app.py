"""DejaRead Gradio 前端（设计文档 3.2.1 "初版" 方案）。

覆盖 3.1（论文入库管线）与 3.2（用户驱动概念图谱）的核心交互：

    Tab 1 论文入库：上传 PDF → 解析 → 分块 → embedding → 写入向量库 + SQLite
    Tab 2 选词标注：手动输入选中的词/短语 + 指定论文 + 页码 → 生成解释 → 跨论文关联发现

所有可变参数（数据库连接、分块、embedding 服务、向量库、LLM 等）从
``config/config.yaml`` 读取，见 :mod:`dejaread.config`。

运行：

    python app.py
"""

from __future__ import annotations

import gradio as gr

from dejaread.concepts import (
    AnnotationRequest,
    ConceptAnnotationService,
    ConceptLLM,
    MockConceptLLM,
    PromptedConceptLLM,
)
from dejaread.concepts.linking import LinkDiscovery
from dejaread.config import get_config
from dejaread.db import Chunk, Paper, init_db, session_scope
from dejaread.embedding import (
    ChromaVectorStore,
    Embedder,
    InMemoryVectorStore,
    RemoteEmbedder,
    VectorStore,
)
from dejaread.ingestion import IngestionPipeline
from dejaread.keyword import SQLiteFTSStore
from dejaread.llm import OpenAICompatibleLLMClient


def _build_vector_store() -> VectorStore:
    backend = get_config().vector_store.backend
    if backend == "chroma":
        try:
            return ChromaVectorStore()
        except ImportError as exc:
            print(f"[警告] 无法使用 ChromaDB（{exc}），回退到内存向量库。")
    return InMemoryVectorStore()


def _build_concept_llm() -> ConceptLLM:
    try:
        return PromptedConceptLLM(OpenAICompatibleLLMClient())
    except ImportError as exc:
        print(f"[警告] 无法使用真实 LLM 客户端（{exc}），回退到 Mock 实现。")
        return MockConceptLLM()


embedder: Embedder = RemoteEmbedder()
vector_store: VectorStore = _build_vector_store()
keyword_store = SQLiteFTSStore()
concept_llm: ConceptLLM = _build_concept_llm()
link_discovery = LinkDiscovery(vector_store, keyword_store)

ingestion_pipeline = IngestionPipeline(embedder=embedder, vector_store=vector_store)
annotation_service = ConceptAnnotationService(
    embedder=embedder,
    vector_store=vector_store,
    keyword_store=keyword_store,
    llm_client=concept_llm,
    link_discovery=link_discovery,
)


def _list_papers() -> list[tuple[str, str]]:
    with session_scope() as session:
        papers = session.query(Paper).order_by(Paper.added_at.desc()).all()
        return [(f"{p.title} ({p.id})", p.id) for p in papers]


def _count_chunks(paper_id: str) -> int:
    with session_scope() as session:
        return session.query(Chunk).filter_by(paper_id=paper_id).count()


def refresh_paper_choices() -> gr.Dropdown:
    return gr.Dropdown(choices=_list_papers())


def ingest_paper(
    pdf_file: str | None,
    title: str,
    authors: str,
    venue: str,
    year: float | None,
) -> tuple[str, gr.Dropdown]:
    if not pdf_file:
        return "请先上传 PDF 文件。", refresh_paper_choices()

    try:
        paper = ingestion_pipeline.ingest(
            pdf_file,
            title=title.strip() or None,
            authors=authors.strip() or None,
            venue=venue.strip() or None,
            year=int(year) if year else None,
        )
    except Exception as exc:  # noqa: BLE001 - 直接把错误展示给用户
        return f"入库失败：{exc}", refresh_paper_choices()

    chunk_count = _count_chunks(paper.id)
    message = (
        f"入库成功！\n\n"
        f"- 论文 ID：`{paper.id}`\n"
        f"- 标题：{paper.title}\n"
        f"- 作者：{paper.authors or '（未提供）'}\n"
        f"- 分块数：{chunk_count}\n"
    )
    return message, refresh_paper_choices()


def annotate(paper_id: str | None, selected_text: str, page_number: float | None) -> tuple[str, str]:
    if not paper_id:
        return "请先选择一篇论文。", ""
    if not selected_text or not selected_text.strip():
        return "请输入选中的词/短语。", ""

    request = AnnotationRequest(
        paper_id=paper_id,
        selected_text=selected_text.strip(),
        page_number=int(page_number) if page_number else None,
    )
    try:
        result = annotation_service.annotate(request)
    except Exception as exc:  # noqa: BLE001 - 直接把错误展示给用户
        return f"标注失败：{exc}", ""

    definition_md = (
        f"### {result.name}\n\n"
        f"**语境化解释**：{result.definition}\n\n"
        f"**上下文片段**：{result.context_snippet}"
    )

    if result.links:
        links_md = "\n".join(
            f"- [{link.link_type.value}] 与「{link.related_concept_name}」"
            f"（论文 `{link.related_paper_id}`）：{link.description}（置信度 {link.confidence:.2f}）"
            for link in result.links
        )
    else:
        links_md = "（暂未发现跨论文关联）"

    return definition_md, links_md


with gr.Blocks(title="DejaRead") as demo:
    gr.Markdown("# DejaRead — 论文入库 & 选词标注")

    with gr.Tab("3.1 论文入库"):
        with gr.Row():
            with gr.Column():
                pdf_input = gr.File(label="上传 PDF", file_types=[".pdf"], type="filepath")
                title_input = gr.Textbox(label="标题（留空则从 PDF 提取）")
                authors_input = gr.Textbox(label="作者（可选）")
                venue_input = gr.Textbox(label="会议/期刊（可选）")
                year_input = gr.Number(label="年份（可选）", precision=0)
                ingest_button = gr.Button("上传并入库", variant="primary")
            with gr.Column():
                ingest_output = gr.Markdown(label="入库结果")

    with gr.Tab("3.2 选词标注"):
        with gr.Row():
            with gr.Column():
                paper_dropdown = gr.Dropdown(label="选择论文", choices=_list_papers())
                refresh_button = gr.Button("刷新论文列表")
                selected_text_input = gr.Textbox(label="选中的词/短语，例如 PPO、clip-higher")
                page_number_input = gr.Number(label="页码（可选）", precision=0)
                annotate_button = gr.Button("标注", variant="primary")
            with gr.Column():
                definition_output = gr.Markdown(label="概念解释")
                links_output = gr.Markdown(label="跨论文关联")

    ingest_button.click(
        ingest_paper,
        inputs=[pdf_input, title_input, authors_input, venue_input, year_input],
        outputs=[ingest_output, paper_dropdown],
    )
    refresh_button.click(refresh_paper_choices, outputs=[paper_dropdown])
    annotate_button.click(
        annotate,
        inputs=[paper_dropdown, selected_text_input, page_number_input],
        outputs=[definition_output, links_output],
    )


if __name__ == "__main__":
    init_db()
    demo.launch()
