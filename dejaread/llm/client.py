"""通用 LLM 客户端：项目内所有需要调用 LLM 的模块（概念图谱、记忆系统、笔记、
问答 Agent）共享同一套连接/调用逻辑，只在各自模块里编写 prompt 和解析返回结果。

技术选型表中本地用 Qwen3-8B（通过 Ollama），复杂任务用 DeepSeek API——两者都暴露
OpenAI 兼容接口，因此一个 :class:`OpenAICompatibleLLMClient` 即可覆盖，只需切换
``base_url`` / ``model``。
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from ..config import get_config


class LLMClient(ABC):
    """最小化的 LLM 调用接口：一轮 system + user 输入，返回文本输出。

    上层模块（如 :mod:`dejaread.concepts.llm`）基于这一接口构造领域相关的 prompt，
    不需要关心具体连的是 Ollama 本地模型还是 DeepSeek API。
    """

    @abstractmethod
    def chat(self, system: str, user: str) -> str:
        """发起一轮对话，返回模型的文本回复。"""


class MockLLMClient(LLMClient):
    """不依赖任何外部服务的假客户端，用于开发与单元测试。

    仅原样回显输入，不做任何"理解"——需要确定性业务逻辑的场景（如概念解释、
    关联分类）应使用各领域模块提供的基于规则的 Mock 实现（例如
    :class:`dejaread.concepts.llm.MockConceptLLM`），而不是依赖这里的回显内容。
    """

    def chat(self, system: str, user: str) -> str:
        return f"[mock] {user.strip()}"


class OpenAICompatibleLLMClient(LLMClient):
    """通过 OpenAI 兼容接口调用 LLM（适配 Ollama 本地部署 / DeepSeek API）。"""

    def __init__(
        self,
        model: str | None = None,
        base_url: str | None = None,
        api_key: str | None = None,
        temperature: float | None = None,
    ) -> None:
        try:
            from openai import OpenAI
        except ImportError as exc:  # pragma: no cover - 依赖缺失时的提示
            raise ImportError(
                "OpenAICompatibleLLMClient 需要安装 openai：pip install openai"
            ) from exc

        llm_config = get_config().llm
        base_url = base_url if base_url is not None else llm_config.base_url
        api_key = api_key if api_key is not None else llm_config.api_key
        self._client = OpenAI(base_url=base_url, api_key=api_key)
        self._model = model if model is not None else llm_config.model
        self._temperature = 0.6

    def chat(self, system: str, user: str) -> str:
        response = self._client.chat.completions.create(
            model=self._model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature=self._temperature,
        )
        return response.choices[0].message.content or ""
