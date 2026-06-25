"""概念图谱模块对公共 LLM 客户端（:mod:`dejaread.llm`）的领域封装。

封装 3.2 节中两处用到 LLM 的地方：

1. 概念语境化解释生成（3.2.1）
2. 跨论文关联精排 + 关联描述生成（3.2.3）

具体的模型连接方式（Ollama / DeepSeek API）由公共模块 :class:`dejaread.llm.LLMClient`
负责，这里只关心 prompt 怎么写、返回结果怎么解析。
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod

from ..db.models import LinkType
from ..llm import LLMClient


class ConceptLLM(ABC):
    """概念解释 / 关联判别所需的领域接口。"""

    @abstractmethod
    def generate_definition(
        self, *, concept_text: str, context_snippet: str, paper_title: str
    ) -> str:
        """根据论文语境生成该概念的解释（对应 3.2.1：LLM 生成该概念在此论文语境下的解释）。"""

    @abstractmethod
    def classify_link(
        self,
        *,
        new_concept: str,
        new_definition: str,
        candidate_concept: str,
        candidate_definition: str,
    ) -> tuple[LinkType, str, float] | None:
        """判断两个概念是否存在语义关联，并给出关联类型、描述、置信度。

        对应 3.2.3："LLM 精排：判断是否真的存在语义关联，并分类关联类型" +
        "生成关联描述"。若 LLM 判断两者不存在有意义的关联，返回 ``None``。
        """


class PromptedConceptLLM(ConceptLLM):
    """基于公共 :class:`~dejaread.llm.LLMClient` 构造 prompt 完成概念任务（生产用）。

    可注入任意实现了 ``chat(system, user) -> str`` 的客户端，例如
    :class:`dejaread.llm.OpenAICompatibleLLMClient`（指向本地 Ollama 或 DeepSeek API）。
    """

    def __init__(self, client: LLMClient) -> None:
        self._client = client

    def generate_definition(
        self, *, concept_text: str, context_snippet: str, paper_title: str
    ) -> str:
        system = (
            "你是一个论文阅读助手，需要结合论文上下文，为用户标注的概念给出简洁、"
            "准确的语境化解释（2-4 句中文，技术术语可保留英文）。"
        )
        user = (
            f"论文标题：{paper_title}\n"
            f"用户选中的概念：{concept_text}\n"
            f"上下文片段：{context_snippet}\n\n"
            "请解释该概念在此论文语境下的含义。"
        )
        return self._client.chat(system, user).strip()

    def classify_link(
        self,
        *,
        new_concept: str,
        new_definition: str,
        candidate_concept: str,
        candidate_definition: str,
    ) -> tuple[LinkType, str, float] | None:
        system = (
            "你是一个判断论文概念之间语义关联的助手。给定两个概念及其语境化定义，"
            "判断它们是否存在有意义的关联。关联类型只能是以下之一："
            "same_concept / evolution / contrast / dependency / generalization。"
            "如果不存在有意义的关联，输出 null。"
            '严格输出 JSON：{"link_type": "...", "description": "...", "confidence": 0.0}'
            "或者 null，不要输出其他内容。"
        )
        user = (
            f"概念 A：{new_concept}\n定义 A：{new_definition}\n\n"
            f"概念 B：{candidate_concept}\n定义 B：{candidate_definition}"
        )
        raw = self._client.chat(system, user).strip()
        if raw.lower() == "null":
            return None
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            return None
        if data is None:
            return None
        try:
            link_type = LinkType(data["link_type"])
        except (KeyError, ValueError):
            return None
        return link_type, data.get("description", ""), float(data.get("confidence", 0.0))


class MockConceptLLM(ConceptLLM):
    """规则模板实现，不依赖任何 LLMClient，用于开发与单元测试。"""

    def generate_definition(
        self, *, concept_text: str, context_snippet: str, paper_title: str
    ) -> str:
        snippet = context_snippet.strip().replace("\n", " ")
        return f"在论文《{paper_title}》中，「{concept_text}」的含义基于上下文：{snippet}"

    def classify_link(
        self,
        *,
        new_concept: str,
        new_definition: str,
        candidate_concept: str,
        candidate_definition: str,
    ) -> tuple[LinkType, str, float] | None:
        if new_concept.strip().lower() == candidate_concept.strip().lower():
            return (
                LinkType.same_concept,
                f"「{new_concept}」与「{candidate_concept}」是同一概念在不同论文中的表述。",
                0.9,
            )
        return (
            LinkType.contrast,
            f"「{new_concept}」与「{candidate_concept}」在主题上相关，可能采用了不同方案。",
            0.5,
        )
