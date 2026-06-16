"""LLM 客户端（阿里百炼 qwen，OpenAI 兼容接口）。

默认 qwen-plus，支持非流式 chat 与流式 chat_stream（P4 的 SSE 用）。
key / base_url / 模型 取自 config。
"""

from __future__ import annotations

from collections.abc import Iterator

from openai import OpenAI

from src.config import settings


class LLMClient:
    """百炼 Chat 客户端。"""

    def __init__(self, model: str | None = None):
        self.model = model or settings.llm_model
        self.client = OpenAI(
            api_key=settings.dashscope_api_key, base_url=settings.llm_base_url
        )

    def chat(self, messages: list[dict], temperature: float | None = None) -> str:
        """非流式：返回完整答案文本。"""
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=settings.llm_temperature if temperature is None else temperature,
        )
        return resp.choices[0].message.content or ""

    def chat_stream(
        self, messages: list[dict], temperature: float | None = None
    ) -> Iterator[str]:
        """流式：逐段 yield 文本增量。"""
        stream = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=settings.llm_temperature if temperature is None else temperature,
            stream=True,
        )
        for chunk in stream:
            delta = chunk.choices[0].delta.content if chunk.choices else None
            if delta:
                yield delta
