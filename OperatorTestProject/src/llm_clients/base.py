"""LLM provider 基础抽象"""

import logging
from abc import ABC, abstractmethod
from typing import Any

from ..config_loader import LLMParams, LLMProviderConfig

logger = logging.getLogger(__name__)


def looks_like_token_limit_error(error_text: str) -> bool:
    """根据错误文本粗略识别 token / 上下文超限错误"""
    normalized = error_text.lower()
    return any(
        marker in normalized
        for marker in (
            "token limit",
            "context length",
            "maximum context",
            "prompt is too long",
            "max tokens",
            "status code: 400",
            "status 400",
        )
    )


def extract_text_from_content(content: Any) -> str:
    """兼容多种响应结构，尽量提取文本内容"""
    if isinstance(content, str):
        return content.strip()

    if isinstance(content, list):
        parts = [extract_text_from_content(item) for item in content]
        return "\n".join(part for part in parts if part).strip()

    if isinstance(content, dict):
        text_value = content.get("text")
        if isinstance(text_value, str):
            return text_value.strip()

        if "content" in content:
            return extract_text_from_content(content["content"])

    if content is None:
        return ""

    return str(content).strip()


class BaseLLMClient(ABC):
    """LLM provider 抽象客户端"""

    def __init__(self, config: LLMProviderConfig, app_root):
        self.config = config
        self.app_root = app_root

    @property
    def name(self) -> str:
        return self.config.name

    @property
    def transport_type(self) -> str:
        return self.config.type

    @property
    def provider(self) -> str:
        return self.config.provider

    def describe(self) -> str:
        """返回便于日志展示的 provider 描述"""
        return f"{self.name} ({self.transport_type}/{self.provider})"

    @abstractmethod
    async def invoke(self, prompt: str, params: LLMParams) -> str:
        """发送 LLM 请求并返回文本响应"""
