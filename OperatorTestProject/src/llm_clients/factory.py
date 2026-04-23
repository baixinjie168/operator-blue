"""LLM provider 工厂"""

from pathlib import Path

from ..config_loader import LLMProviderConfig
from ..exceptions import LLMInvocationError
from .base import BaseLLMClient
from .cli_clients import ClaudeCLIClient, CodexCLIClient
from .interface_clients import AnthropicInterfaceClient, OpenAIInterfaceClient


def create_llm_client(config: LLMProviderConfig, app_root: Path) -> BaseLLMClient:
    """根据配置创建具体 provider 客户端"""
    client_mapping = {
        ("interface", "openai"): OpenAIInterfaceClient,
        ("interface", "anthropic"): AnthropicInterfaceClient,
        ("cli", "codex"): CodexCLIClient,
        ("cli", "claude"): ClaudeCLIClient,
    }

    client_class = client_mapping.get((config.type, config.provider))
    if client_class is None:
        supported = ", ".join(f"{transport}/{provider}" for transport, provider in sorted(client_mapping))
        raise LLMInvocationError(
            f"不支持的LLM provider组合: {config.type}/{config.provider}，支持: {supported}"
        )

    return client_class(config, app_root)
