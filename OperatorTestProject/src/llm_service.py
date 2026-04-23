"""
LLM服务层模块
负责多 provider 调用封装和执行槽位管理
"""
import asyncio
import logging
import os
import random
import threading
from typing import List
from urllib.parse import urlparse

from .config_loader import ConfigLoader, LLMProviderConfig
from .exceptions import LLMInvocationError
from .llm_clients import BaseLLMClient, create_llm_client
from .path_utils import get_app_root

logger = logging.getLogger(__name__)


class LLMService:
    """LLM服务类"""

    def __init__(self, config_loader: ConfigLoader):
        """
        初始化LLM服务

        Args:
            config_loader: 配置加载器实例
        """
        self.config_loader = config_loader
        self._llm_pool: List[LLMProviderConfig] = []
        self._client_pool: List[BaseLLMClient] = []
        self._current_index = 0
        self._pool_lock = threading.Lock()
        self._app_root = get_app_root()

        # 设置环境变量，避免接口型 provider 被代理拦截
        self._setup_no_proxy()

        # 初始化 LLM 执行槽位
        self._init_pool()

    def _setup_no_proxy(self) -> None:
        """设置 NO_PROXY 环境变量，包含所有接口型 provider 的 host"""
        existing_hosts = [
            host.strip()
            for host in os.environ.get("NO_PROXY", "").split(",")
            if host.strip()
        ]
        no_proxy_hosts = list(existing_hosts)

        for provider in self.config_loader.get_llm_providers():
            if not provider.base_url:
                continue

            parsed = urlparse(provider.base_url)
            host = parsed.netloc
            if host and host not in no_proxy_hosts:
                no_proxy_hosts.append(host)

        if no_proxy_hosts:
            no_proxy_value = ",".join(no_proxy_hosts)
            os.environ["NO_PROXY"] = no_proxy_value
            os.environ["no_proxy"] = no_proxy_value

    def _init_pool(self) -> None:
        """初始化 LLM 执行槽位"""
        providers = self.config_loader.get_llm_providers()
        logger.debug("初始化LLM执行池，provider数量: %s", len(providers))

        for provider in providers:
            client = create_llm_client(provider, self._app_root)
            self._llm_pool.append(provider)
            self._client_pool.append(client)
            logger.debug("添加LLM provider槽位: %s", client.describe())

    def _next_index(self) -> int:
        """轮询获取下一个 provider 槽位索引"""
        if not self._client_pool:
            raise LLMInvocationError("LLM连接池为空")

        with self._pool_lock:
            index = self._current_index
            self._current_index = (self._current_index + 1) % len(self._client_pool)
            return index

    def get_pool_size(self) -> int:
        """
        获取连接池大小

        Returns:
            int: 连接池大小
        """
        return len(self._llm_pool)

    def get_llm(self) -> LLMProviderConfig:
        """
        轮询获取LLM provider 配置

        Returns:
            LLMProviderConfig: LLM provider 配置
        """
        return self._llm_pool[self._next_index()]

    def get_llm_by_index(self, index: int) -> LLMProviderConfig:
        """
        根据索引获取LLM provider 配置

        Args:
            index: LLM provider 索引

        Returns:
            LLMProviderConfig: LLM provider 配置
        """
        if not self._llm_pool:
            raise LLMInvocationError("LLM连接池为空")

        if index < 0 or index >= len(self._llm_pool):
            raise LLMInvocationError(f"LLM索引超出范围: {index}")

        return self._llm_pool[index]

    def allocate_modules_to_interfaces(self, modules: List[str]) -> dict:
        """
        将模块平均分配到LLM provider槽位

        Args:
            modules: 模块列表

        Returns:
            dict: 槽位索引到模块列表的映射 {interface_index: [module1, module2, ...]}
        """
        num_interfaces = len(self._llm_pool)
        num_modules = len(modules)

        if num_interfaces == 0:
            raise LLMInvocationError("LLM连接池为空")

        base_count = num_modules // num_interfaces
        remainder = num_modules % num_interfaces

        allocation = {}
        module_index = 0

        for interface_index in range(num_interfaces):
            count = base_count + (1 if interface_index < remainder else 0)
            allocated_modules = modules[module_index:module_index + count]
            allocation[interface_index] = allocated_modules
            module_index += count

        logger.info(f"模块分配到provider槽位: {num_modules}个模块, {num_interfaces}个槽位")
        for interface_index, allocated_modules in allocation.items():
            logger.info(f"  槽位 {interface_index}: {allocated_modules}")

        return allocation

    async def invoke(self, prompt: str) -> str:
        """
        调用 provider 生成响应

        Args:
            prompt: 输入提示词

        Returns:
            str: LLM响应内容

        Raises:
            LLMInvocationError: LLM调用失败
        """
        index = self._next_index()
        client = self._client_pool[index]
        params = self.config_loader.get_llm_params()
        logger.debug("使用 provider 槽位 %s: %s", index, client.describe())
        return await client.invoke(prompt, params)

    async def invoke_with_retry(
        self,
        prompt: str,
        max_retries: int = 5,
        context: str = "",
    ) -> str:
        """
        带重试的LLM调用

        Args:
            prompt: 输入提示词
            max_retries: 最大重试次数(默认5次)
            context: 调用上下文信息(如: "[basic_info] 提取操作")

        Returns:
            str: LLM响应内容
        """
        last_error = None
        context_prefix = f"{context} " if context else ""

        for attempt in range(max_retries):
            try:
                logger.debug(f"{context_prefix}LLM调用尝试 {attempt + 1}/{max_retries}")
                result = await self.invoke(prompt)
                logger.debug(f"{context_prefix}LLM调用成功 (尝试 {attempt + 1}/{max_retries})")
                return result
            except LLMInvocationError as e:
                last_error = e

                if e.is_token_limit_error:
                    logger.error(f"{context_prefix}检测到token超限错误(400), 终止当前流程")
                    raise e

                if attempt < max_retries - 1:
                    wait_time = random.uniform(1, 5)
                    logger.warning(
                        f"{context_prefix}LLM调用失败, 等待 {wait_time:.2f}秒后重试 "
                        f"(尝试 {attempt + 1}/{max_retries})"
                    )
                    await asyncio.sleep(wait_time)

        logger.error(f"{context_prefix}LLM调用失败, 已达到最大重试次数 {max_retries}")
        raise last_error
