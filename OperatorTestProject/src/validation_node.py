# -*- coding: utf-8 -*-
"""
校验节点模块
负责校验提取的数据是否符合规则
只负责校验，不负责重新提取
"""
import json
import logging
import traceback
from pathlib import Path
from typing import Dict, Any, Optional, List

from pydantic import ValidationError

from .module_graph_state import ModuleProcessingState, ValidationResult
from .config_loader import ConfigLoader
from .llm_service import LLMService
from .prompt_builder import PromptBuilder
from .rule_loader import RuleLoader
from .result_saver import ResultSaver
from .exceptions import ModuleProcessingError
from .check_error import CheckError

# 获取logger
logger = logging.getLogger(__name__)


class ValidationNode:
    """校验节点（只负责校验，不负责重新提取）"""

    def __init__(
            self,
            config_loader: ConfigLoader,
            llm_service: LLMService,
            rules_dir: Path
    ):
        """
        初始化校验节点

        Args:
            config_loader: 配置加载器
            llm_service: LLM服务
            rules_dir: 规则目录
        """
        self.config_loader = config_loader
        self.llm_service = llm_service
        self.rule_loader = RuleLoader(rules_dir)

    async def _call_llm(
            self,
            prompt: str,
            module: str
    ) -> str:
        """调用LLM进行校验"""
        logger.info(f"[{module}] 调用LLM进行业务校验...")
        response = await self.llm_service.invoke_with_retry(
            prompt, context=f"[{module}] 业务校验操作"
        )
        logger.debug(f"[{module}] 校验响应长度: {len(response)} 字符")
        return response

    def _check_extracted_data(
            self,
            state: ModuleProcessingState
    ) -> tuple[bool, Optional[str]]:
        """
        步骤1: 检查是否有提取数据

        Args:
            state: 当前状态

        Returns:
            tuple[bool, Optional[str]]: (是否有数据, 错误信息)
        """
        module = state["module_name"]
        extracted_data = state.get("extracted_data")

        if not extracted_data:
            logger.error(f"[{module}] 没有提取数据可供校验")
            return False, "没有提取数据可供校验"

        return True, None

    def _load_validation_resources(
            self,
            state: ModuleProcessingState
    ) -> tuple[str, str]:
        """
        步骤2: 加载校验资源（校验规则和已生成的JSON数据）

        Args:
            state: 当前状态

        Returns:
            tuple[str, str]: (校验规则内容, 已生成的JSON内容)
        """
        module = state["module_name"]
        version_dir = state.get("version_dir")

        # 加载校验规则
        check_rule_content = self.rule_loader.load_rule_file(module, is_check=True)

        # 读取已生成的规则文件内容
        rule_json_content = ResultSaver.read_rule_json(
            Path(version_dir),
            module
        )

        return check_rule_content, rule_json_content

    def _build_validation_prompt(
            self,
            state: ModuleProcessingState,
            check_rule_content: str,
            rule_json_content: str
    ) -> str:
        """
        步骤3: 构建校验提示词

        Args:
            state: 当前状态
            check_rule_content: 校验规则内容
            rule_json_content: 已生成的JSON内容

        Returns:
            str: 校验提示词
        """
        module = state["module_name"]
        operator_doc = state["operator_doc"]

        prompt = PromptBuilder.build_validation_prompt(
            operator_doc,
            check_rule_content,
            module,
            rule_json_content
        )

        return prompt

    def _parse_check_errors(
            self,
            response: str,
            module: str
    ) -> tuple[Optional[List[CheckError]], Optional[str]]:
        """
        解析LLM响应为CheckError数组

        Args:
            response: LLM响应内容
            module: 模块名称

        Returns:
            tuple[Optional[List[CheckError]], Optional[str]]: (解析成功的错误列表, 解析失败的错误信息)
        """
        try:
            # 尝试解析JSON
            data = json.loads(response)

            # 验证是否为列表
            if not isinstance(data, list):
                return None, f"响应不是JSON数组格式，实际类型: {type(data).__name__}"

            # 如果是空数组，表示校验通过
            if len(data) == 0:
                return [], None

            # 尝试将每个元素解析为CheckError对象
            errors = [CheckError(**item) for item in data]
            return errors, None

        except json.JSONDecodeError as e:
            error_msg = f"JSON解析失败: {str(e)}\n堆栈信息:\n{traceback.format_exc()}"
            logger.error(f"[{module}] {error_msg}")
            return None, error_msg
        except ValidationError as e:
            error_msg = f"数据验证失败: {str(e)}\n堆栈信息:\n{traceback.format_exc()}"
            logger.error(f"[{module}] {error_msg}")
            return None, error_msg
        except Exception as e:
            error_msg = f"解析响应失败: {str(e)}\n堆栈信息:\n{traceback.format_exc()}"
            logger.error(f"[{module}] {error_msg}")
            return None, error_msg

    def _build_retry_prompt(
            self,
            original_prompt: str,
            parse_error: str,
            response: str
    ) -> str:
        """
        构建重试提示词，包含解析错误信息

        Args:
            original_prompt: 原始提示词
            parse_error: 解析错误信息
            response: 上次的LLM响应

        Returns:
            str: 重试提示词
        """
        retry_prompt = f"""
{original_prompt}

--------------------------------------------------
【上次输出解析失败】
解析错误信息:
{parse_error}
--------------------------------------------------
【上次输出内容】:
{response}

请修正输出格式，确保输出合法的JSON数组格式。
"""
        return retry_prompt

    async def _perform_validation(
            self,
            prompt: str,
            module: str,
            max_retries: int = 5
    ) -> tuple[bool, Optional[str]]:
        """
        步骤4: 执行校验（调用LLM并解析结果）

        Args:
            prompt: 校验提示词
            module: 模块名称
            max_retries: 最大重试次数（默认5次）

        Returns:
            tuple[bool, Optional[str]]: (是否通过, 错误信息)
        """
        current_prompt = prompt
        last_response = None

        for attempt in range(max_retries):
            logger.info(f"[{module}] 第 {attempt + 1}/{max_retries} 次校验尝试")

            # 调用LLM进行校验
            response = await self._call_llm(current_prompt, module)
            last_response = response

            # 尝试解析为CheckError数组
            errors, parse_error = self._parse_check_errors(response, module)

            if parse_error is None:
                # 解析成功
                if errors is None or len(errors) == 0:
                    # 空数组，校验通过
                    logger.info(f"[{module}] 校验结果: 通过")
                    return True, None
                else:
                    # 有错误，校验失败
                    logger.info(f"[{module}] 校验结果: 失败，发现 {len(errors)} 个错误")
                    # 将CheckError列表转换为JSON字符串作为错误信息
                    error_msg = json.dumps([error.model_dump() for error in errors], ensure_ascii=False, indent=2)
                    return False, error_msg
            else:
                # 解析失败
                if attempt < max_retries - 1:
                    # 还可以重试
                    logger.warning(f"[{module}] 第 {attempt + 1} 次尝试解析失败，准备重试...")
                    current_prompt = self._build_retry_prompt(prompt, parse_error, response)
                else:
                    # 已达到最大重试次数
                    logger.error(f"[{module}] 已达到最大重试次数 {max_retries} 次，解析仍然失败")
                    # 返回原始响应作为错误信息
                    return False, response

        # 理论上不会执行到这里
        return False, last_response

    def _handle_validation_success(
            self,
            state: ModuleProcessingState
    ) -> ModuleProcessingState:
        """
        步骤5a: 处理校验成功的情况

        Args:
            state: 当前状态

        Returns:
            ModuleProcessingState: 更新后的状态
        """
        module = state["module_name"]

        state["validation_result"] = ValidationResult(
            success=True,
            error=None,
            error_path=None
        )
        state["validation_error"] = None
        logger.info(f"[{module}] 业务校验通过")

        return state

    def _handle_validation_failure(
            self,
            state: ModuleProcessingState,
            error_msg: str
    ) -> ModuleProcessingState:
        """
        步骤5b: 处理校验失败的情况

        Args:
            state: 当前状态
            error_msg: 错误信息

        Returns:
            ModuleProcessingState: 更新后的状态
        """
        module = state["module_name"]
        version_dir = state.get("version_dir")
        previous_error_path = state.get("previous_error_path")
        outer_iteration = state["outer_iteration"]
        max_outer_iterations = state["max_outer_iterations"]
        version_dir_path = Path(version_dir)

        # 保存错误信息
        error_json_path = ResultSaver.save_error(
            version_dir_path,
            error_msg,
            module,
            Path(previous_error_path) if previous_error_path else None
        )
        error_markdown_path = ResultSaver.get_error_markdown_path(version_dir_path)
        error_markdown_content = ResultSaver.read_text_if_exists(error_markdown_path)

        state["validation_result"] = ValidationResult(
            success=False,
            error=error_msg,
            error_path=str(error_markdown_path)
        )
        state["previous_error_path"] = str(error_markdown_path)
        state["validation_error"] = error_markdown_content or error_msg

        logger.warning(f"[{module}] 业务校验失败! 错误信息已保存: {error_json_path}")
        logger.info(f"[{module}] 业务校验失败，错误信息已记录，将由 extract node 重新提取")

        # 更新迭代次数
        if outer_iteration < max_outer_iterations - 1:
            # 大循环可以重试
            new_outer_iteration = outer_iteration + 1
            state["outer_iteration"] = new_outer_iteration
            state["inner_iteration"] = 0  # 重置小循环

            logger.info(
                f"[{module}] 业务校验失败，"
                f"大循环第 {outer_iteration + 1} 次迭代，"
                f"准备第 {new_outer_iteration + 1} 次大循环迭代"
            )
        else:
            # 大循环达到最大迭代次数
            error_json_content = ResultSaver.read_text_if_exists(error_json_path)
            error_markdown_content = ResultSaver.read_text_if_exists(error_markdown_path)
            if error_json_content or error_markdown_content:
                state["error"] = (
                    f"[{module}] 大循环达到最大迭代次数 {max_outer_iterations}，"
                    f"v{outer_iteration} 校验后错误文件仍非空，处理失败"
                )
            else:
                state["error"] = (
                    f"[{module}] 大循环达到最大迭代次数 {max_outer_iterations}，"
                    f"处理失败"
                )
            logger.error(state["error"])

        return state

    def _handle_validation_error(
            self,
            state: ModuleProcessingState,
            error: Exception
    ) -> ModuleProcessingState:
        """
        步骤6: 处理校验过程中的异常

        Args:
            state: 当前状态
            error: 异常对象

        Returns:
            ModuleProcessingState: 更新后的状态
        """
        module = state["module_name"]

        logger.error(f"[{module}] 业务校验失败: {error}")

        state["validation_result"] = ValidationResult(
            success=False,
            error=str(error),
            error_path=None
        )
        state["validation_error"] = str(error)
        state["error"] = f"业务校验失败: {str(error)}"

        return state

    async def validate(
            self,
            state: ModuleProcessingState
    ) -> ModuleProcessingState:
        """
        执行业务校验操作（只校验，不重新提取）

        Args:
            state: 当前状态

        Returns:
            ModuleProcessingState: 更新后的状态

        说明:
            - 调用LLM进行业务校验
            - 如果校验通过，设置 validation_result.success=True
            - 如果校验失败，设置 validation_result.success=False，并记录错误信息
            - 错误信息将传递给 extract node，由 extract node 负责重新提取
        """
        module = state["module_name"]
        outer_iteration = state["outer_iteration"]

        # 步骤1: 检查是否有提取数据
        has_data, error = self._check_extracted_data(state)
        if not has_data:
            state["validation_result"] = ValidationResult(
                success=False,
                error=error,
                error_path=None
            )
            state["validation_error"] = error
            state["error"] = error
            return state

        logger.info(f"[{module}] 开始业务校验")

        try:
            # 步骤2: 加载校验资源
            check_rule_content, rule_json_content = self._load_validation_resources(state)

            # 步骤3: 构建校验提示词
            prompt = self._build_validation_prompt(state, check_rule_content, rule_json_content)

            # 步骤4: 执行校验
            is_valid, error_msg = await self._perform_validation(prompt, module)

            # 步骤5: 处理校验结果
            if is_valid:
                # 步骤5a: 处理校验成功
                state = self._handle_validation_success(state)
            else:
                # 步骤5b: 处理校验失败
                state = self._handle_validation_failure(state, error_msg)

            return state

        except Exception as e:
            # 步骤6: 处理异常
            state = self._handle_validation_error(state, e)
            return state
