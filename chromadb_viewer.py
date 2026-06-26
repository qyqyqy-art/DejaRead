"""ChromaDB 数据可视化工具，基于 Gradio 构建。

用法：
    python chromadb_viewer.py
"""

from __future__ import annotations

import json

import chromadb
import gradio as gr

from dejaread.config import get_config

client = chromadb.PersistentClient(path=get_config().vector_store.persist_directory)

PAGE_SIZE = 20


def list_collections() -> list[str]:
    return [col.name for col in client.list_collections()]


def view_collection(collection_name: str | None, page: int = 0) -> str:
    if not collection_name:
        return "请选择一个 collection。"

    col = client.get_collection(collection_name)
    total = col.count()
    if total == 0:
        return f"**{collection_name}** — 空集合（0 条）"

    # 一次性获取全部数据，在内存中分页（避免 ChromaDB offset bug）
    all_data = col.get(include=["metadatas", "documents", "embeddings"])
    start = page * PAGE_SIZE
    end = min(start + PAGE_SIZE, total)
    if start >= total:
        return f"**{collection_name}** — 页码超出范围"

    lines = [f"## {collection_name}（共 {total} 条，第 {start + 1}–{end} 条）\n"]

    for i in range(start, end):
        id_ = all_data["ids"][i]
        meta = all_data["metadatas"][i] if all_data["metadatas"] else {}
        doc = (all_data["documents"][i] or "") if all_data["documents"] else ""
        emb = all_data["embeddings"][i] if all_data["embeddings"] is not None else None

        lines.append(f"### [{i}] `{id_}`")
        lines.append(f"**metadata:** `{json.dumps(meta, ensure_ascii=False)}`\n")
        if doc:
            preview = doc[:300] + ("..." if len(doc) > 300 else "")
            lines.append(f"**document:**\n```\n{preview}\n```\n")
        if emb is not None:
            lines.append(f"**embedding:** `[{emb[0]:.4f}, {emb[1]:.4f}, ... ({len(emb)}d)]`\n")
        lines.append("---")

    return "\n".join(lines)


def on_view(collection_name: str | None, page: float) -> str:
    return view_collection(collection_name, page=int(page))


def on_prev(collection_name: str | None, page: float) -> tuple[str, int]:
    new_page = max(0, int(page) - 1)
    return view_collection(collection_name, page=new_page), new_page


def on_next(collection_name: str | None, page: float) -> tuple[str, int]:
    if collection_name:
        col = client.get_collection(collection_name)
        max_page = max(0, (col.count() - 1) // PAGE_SIZE)
        new_page = min(int(page) + 1, max_page)
    else:
        new_page = 0
    return view_collection(collection_name, page=new_page), new_page


def get_stats() -> str:
    lines = ["## ChromaDB 概览\n", "| Collection | 条数 |", "|---|---|"]
    for col in client.list_collections():
        lines.append(f"| {col.name} | {col.count()} |")
    return "\n".join(lines)


with gr.Blocks(title="ChromaDB Viewer") as app:
    gr.Markdown("# 🔍 ChromaDB Viewer")

    with gr.Tab("概览"):
        stats_output = gr.Markdown()
        refresh_stats_btn = gr.Button("刷新")
        refresh_stats_btn.click(get_stats, outputs=[stats_output])
        app.load(get_stats, outputs=[stats_output])

    with gr.Tab("浏览数据"):
        with gr.Row():
            col_dropdown = gr.Dropdown(label="Collection", choices=list_collections())
            refresh_col_btn = gr.Button("刷新列表")
            page_num = gr.Number(label="页码", value=0, precision=0, minimum=0)
        with gr.Row():
            prev_btn = gr.Button("⬅ 上一页")
            next_btn = gr.Button("下一页 ➡")

        data_output = gr.Markdown()

        col_dropdown.change(on_view, inputs=[col_dropdown, page_num], outputs=[data_output])
        refresh_col_btn.click(
            lambda: gr.Dropdown(choices=list_collections()), outputs=[col_dropdown]
        )
        prev_btn.click(
            on_prev, inputs=[col_dropdown, page_num], outputs=[data_output, page_num]
        )
        next_btn.click(
            on_next, inputs=[col_dropdown, page_num], outputs=[data_output, page_num]
        )


if __name__ == "__main__":
    app.launch(server_name="0.0.0.0", server_port=8060)
