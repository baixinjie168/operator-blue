"""接口型 LLM provider 客户端"""

import asyncio
import json
from typing import Any, Dict
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from ..config_loader import LLMParams
from ..exceptions import LLMInvocationError
from .base import BaseLLMClient, extract_text_from_content, looks_like_token_limit_error


class BaseHTTPClient(BaseLLMClient):
    """HTTP 接口型 LLM 客户端基类"""

    default_api_path = "/"

    async def invoke(self, prompt: str, params: LLMParams) -> str:
        """异步调用 HTTP 接口"""
        return await asyncio.to_thread(self._invoke_sync, prompt, params)

    def _build_url(self) -> str:
        """拼接请求 URL"""
        base_url = (self.config.base_url or "").rstrip("/")
        api_path = (self.config.api_path or self.default_api_path).strip()

        if not api_path:
            return base_url

        normalized_api_path = api_path.rstrip("/")
        if base_url.endswith(normalized_api_path):
            return base_url

        return f"{base_url}/{api_path.lstrip('/')}"

    def _build_headers(self) -> Dict[str, str]:
        """构建请求头"""
        headers = {
            "Content-Type": "application/json",
        }
        headers.update({key: str(value) for key, value in self.config.headers.items()})
        return headers

    def _build_request_body(self, prompt: str, params: LLMParams) -> Dict[str, Any]:
        raise NotImplementedError

    def _parse_response(self, response_data: Dict[str, Any]) -> str:
        raise NotImplementedError

    def _invoke_sync(self, prompt: str, params: LLMParams) -> str:
        """同步调用 HTTP 接口"""
        request_url = self._build_url()
        request_headers = self._build_headers()
        request_body = self._build_request_body(prompt, params)
        request_data = json.dumps(request_body).encode("utf-8")
        request = Request(
            url=request_url,
            data=request_data,
            headers=request_headers,
            method="POST",
        )

        try:
            with urlopen(request, timeout=params.timeout) as response:
                response_text = response.read().decode("utf-8", errors="ignore")
        except HTTPError as error:
            error_body = error.read().decode("utf-8", errors="ignore")
            error_message = (
                f"[{self.describe()}] 接口调用失败: HTTP {error.code} {error.reason}"
                f"\n{error_body}"
            ).strip()
            raise LLMInvocationError(
                error_message,
                error,
                looks_like_token_limit_error(error_message),
            )
        except URLError as error:
            error_message = f"[{self.describe()}] 接口调用失败: {error}"
            raise LLMInvocationError(
                error_message,
                error,
                looks_like_token_limit_error(error_message),
            )
        except TimeoutError as error:
            error_message = f"[{self.describe()}] 接口调用超时（>{params.timeout}秒）"
            raise LLMInvocationError(error_message, error)
        except Exception as error:
            error_message = f"[{self.describe()}] 接口调用失败: {error}"
            raise LLMInvocationError(
                error_message,
                error,
                looks_like_token_limit_error(error_message),
            )

        try:
            response_data = json.loads(response_text)
        except json.JSONDecodeError as error:
            raise LLMInvocationError(
                f"[{self.describe()}] 接口返回的不是合法 JSON: {error}\n原始响应:\n{response_text}",
                error,
            )

        try:
            parsed_text = self._parse_response(response_data)
        except Exception as error:
            raise LLMInvocationError(
                f"[{self.describe()}] 解析接口响应失败: {error}\n原始响应:\n{response_text}",
                error,
            )

        if not parsed_text:
            raise LLMInvocationError(f"[{self.describe()}] 接口响应为空")

        return parsed_text


class OpenAIInterfaceClient(BaseHTTPClient):
    """OpenAI 风格接口客户端"""

    default_api_path = "/chat/completions"

    def _build_headers(self) -> Dict[str, str]:
        headers = super()._build_headers()
        headers["Authorization"] = f"Bearer {self.config.api_key}"
        return headers

    def _build_request_body(self, prompt: str, params: LLMParams) -> Dict[str, Any]:
        # 构建消息列表
        messages = []

        # 检查是否有系统提示
        system_prompt = self.config.options.get("system")
        if system_prompt:
            messages.append({
                "role": "system",
                "content": system_prompt
            })

        # 添加用户消息
        messages.append({
            "role": "user",
            "content": prompt,
        })

        # 构建基础请求体
        request_body = {
            "model": self.config.model,
            "messages": messages,
            "temperature": params.temperature,
            "max_tokens": params.max_tokens,
        }

        # 添加其他选项（排除 system，因为它已经被处理了）
        other_options = {k: v for k, v in self.config.options.items() if k != "system"}
        request_body.update(other_options)

        return request_body

    def _parse_response(self, response_data: Dict[str, Any]) -> str:
        choices = response_data.get("choices") or []
        if not choices:
            raise ValueError("未找到 choices 字段")

        first_choice = choices[0]
        message = first_choice.get("message") or {}
        content = message.get("content")
        parsed_text = extract_text_from_content(content)

        if not parsed_text:
            parsed_text = extract_text_from_content(first_choice.get("text"))

        if not parsed_text:
            raise ValueError("未找到 message.content 或 text 字段")

        return parsed_text


class AnthropicInterfaceClient(BaseHTTPClient):
    """Anthropic 风格接口客户端"""

    default_api_path = "/messages"

    def _build_headers(self) -> Dict[str, str]:
        headers = super()._build_headers()
        headers["x-api-key"] = self.config.api_key or ""
        headers["anthropic-version"] = str(
            self.config.options.get("anthropic_version", "2023-06-01")
        )
        return headers

    def _build_request_body(self, prompt: str, params: LLMParams) -> Dict[str, Any]:
        request_options = dict(self.config.options)
        request_options.pop("anthropic_version", None)
        content_as_text = bool(request_options.pop("content_as_text", False))
        system_prompt = request_options.pop("system", None)

        content_value = prompt if content_as_text else [
            {
                "type": "text",
                "text": prompt,
            }
        ]

        request_body = {
            "model": self.config.model,
            "max_tokens": params.max_tokens,
            "temperature": params.temperature,
            "messages": [
                {
                    "role": "user",
                    "content": content_value,
                }
            ],
        }

        if system_prompt is not None:
            request_body["system"] = system_prompt

        request_body.update(request_options)
        return request_body

    def _parse_response(self, response_data: Dict[str, Any]) -> str:
        content = response_data.get("content")
        parsed_text = extract_text_from_content(content)

        if not parsed_text and "completion" in response_data:
            parsed_text = extract_text_from_content(response_data["completion"])

        if not parsed_text:
            raise ValueError("未找到 content 或 completion 字段")

        return parsed_text
