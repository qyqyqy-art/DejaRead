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
from dejaread.llm import LLMClient, MockLLMClient, OpenAICompatibleLLMClient
from dejaread.notes.service import NotesService
from dejaread.qa.schemas import ChatTurn, QARequest
from dejaread.qa.service import QAService


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


def _build_llm_client() -> LLMClient:
    try:
        return OpenAICompatibleLLMClient()
    except ImportError as exc:
        print(f"[警告] 无法使用真实 LLM 客户端（{exc}），回退到 Mock 实现。")
        return MockLLMClient()


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

notes_service = NotesService(
    embedder=embedder,
    vector_store=vector_store,
    keyword_store=keyword_store,
)

qa_service = QAService(
    embedder=embedder,
    vector_store=vector_store,
    keyword_store=keyword_store,
    llm_client=_build_llm_client(),
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


def annotate(
    paper_id: str | None, selected_text: str, page_number: float | None
) -> tuple[str, str, dict | None]:
    if not paper_id:
        return "请先选择一篇论文。", "", None
    if not selected_text or not selected_text.strip():
        return "请输入选中的词/短语。", "", None

    request = AnnotationRequest(
        paper_id=paper_id,
        selected_text=selected_text.strip(),
        page_number=int(page_number) if page_number else None,
    )
    try:
        result = annotation_service.annotate(request)
    except Exception as exc:  # noqa: BLE001 - 直接把错误展示给用户
        return f"标注失败：{exc}", "", None

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
        links_text_for_note = links_md
    else:
        links_md = "（暂未发现跨论文关联）"
        links_text_for_note = ""

    annotation_state = {
        "paper_id": paper_id,
        "name": result.name,
        "definition": result.definition,
        "context_snippet": result.context_snippet,
        "links_text": links_text_for_note,
    }
    return definition_md, links_md, annotation_state


def add_annotation_to_note(annotation_state: dict | None) -> str:
    if not annotation_state:
        return "请先完成一次标注。"
    notes_service.append_concept(
        paper_id=annotation_state["paper_id"],
        concept_name=annotation_state["name"],
        definition=annotation_state["definition"],
        context_snippet=annotation_state["context_snippet"],
        links_text=annotation_state["links_text"],
    )
    return f"已添加到论文 `{annotation_state['paper_id']}` 的笔记。"


def load_note(paper_id: str | None) -> str:
    if not paper_id:
        return "请先选择一篇论文。"
    return notes_service.read_or_create(paper_id)


def save_note(paper_id: str | None, content: str) -> str:
    if not paper_id:
        return "请先选择一篇论文。"
    notes_service.save(paper_id, content)
    return "保存成功。"


def ask_question(
    paper_id: str | None, question: str, history: list[tuple[str, str]] | None
) -> tuple[list[tuple[str, str]], list[tuple[str, str]], str]:
    history = history or []
    if not paper_id or not question or not question.strip():
        return history, history, question

    chat_history = [ChatTurn(question=q, answer=a) for q, a in history]
    result = qa_service.ask(
        QARequest(paper_id=paper_id, question=question.strip(), history=chat_history)
    )
    new_history = history + [(question.strip(), result.answer)]
    return new_history, new_history, ""


def import_conversation(paper_id: str | None, history: list[tuple[str, str]] | None) -> str:
    if not paper_id:
        return "请先选择一篇论文。"
    if not history:
        return "没有可导入的对话内容。"

    chat_history = [ChatTurn(question=q, answer=a) for q, a in history]
    summary = qa_service.summarize_conversation(chat_history)
    notes_service.append_conversation_summary(paper_id, summary)
    return f"已将对话摘要写入论文 `{paper_id}` 的笔记。"


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
                annotation_state = gr.State(None)
                add_to_note_button = gr.Button("添加到笔记")
                add_to_note_status = gr.Markdown()

    with gr.Tab("3.4 笔记"):
        with gr.Row():
            with gr.Column():
                notes_paper_dropdown = gr.Dropdown(label="选择论文", choices=_list_papers())
                notes_refresh_button = gr.Button("刷新论文列表")
                load_note_button = gr.Button("加载笔记")
                save_note_button = gr.Button("保存笔记", variant="primary")
                note_status = gr.Markdown()
            with gr.Column():
                note_textbox = gr.Textbox(label="笔记内容（Markdown）", lines=20)

    with gr.Tab("3.5 问答（预览）"):
        with gr.Row():
            with gr.Column():
                qa_paper_dropdown = gr.Dropdown(label="当前论文", choices=_list_papers())
                qa_refresh_button = gr.Button("刷新论文列表")
                qa_history_state = gr.State([])
                qa_chatbot = gr.Chatbot(label="问答")
                qa_question_input = gr.Textbox(label="提问")
                qa_ask_button = gr.Button("提问", variant="primary")
                qa_import_button = gr.Button("导入到笔记")
                qa_import_status = gr.Markdown()

    ingest_button.click(
        ingest_paper,
        inputs=[pdf_input, title_input, authors_input, venue_input, year_input],
        outputs=[ingest_output, paper_dropdown],
    )
    refresh_button.click(refresh_paper_choices, outputs=[paper_dropdown])
    annotate_button.click(
        annotate,
        inputs=[paper_dropdown, selected_text_input, page_number_input],
        outputs=[definition_output, links_output, annotation_state],
    )
    add_to_note_button.click(
        add_annotation_to_note, inputs=[annotation_state], outputs=[add_to_note_status]
    )

    notes_refresh_button.click(refresh_paper_choices, outputs=[notes_paper_dropdown])
    load_note_button.click(load_note, inputs=[notes_paper_dropdown], outputs=[note_textbox])
    save_note_button.click(
        save_note, inputs=[notes_paper_dropdown, note_textbox], outputs=[note_status]
    )

    qa_refresh_button.click(refresh_paper_choices, outputs=[qa_paper_dropdown])
    qa_ask_button.click(
        ask_question,
        inputs=[qa_paper_dropdown, qa_question_input, qa_history_state],
        outputs=[qa_chatbot, qa_history_state, qa_question_input],
    )
    qa_import_button.click(
        import_conversation,
        inputs=[qa_paper_dropdown, qa_history_state],
        outputs=[qa_import_status],
    )


if __name__ == "__main__":
    init_db()
    demo.launch()
