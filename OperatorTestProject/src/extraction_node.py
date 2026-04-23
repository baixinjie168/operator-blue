# -*- coding: utf-8 -*-
"""
提取节点模块
负责从算子文档中提取结构化数据
"""
import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional

from .module_graph_state import ModuleProcessingState, ExtractionResult
from .config_loader import ConfigLoader
from .llm_service import LLMService
from .prompt_builder import PromptBuilder
from .rule_loader import RuleLoader
from .result_saver import ResultSaver
from .path_utils import resolve_path
from .exceptions import ModuleProcessingError

# 获取logger
logger = logging.getLogger(__name__)

# 允许提取为空的模块
OPTIONAL_MODULES = ["other_parameters"]


class ExtractionNode:
    """提取节点"""

    def __init__(
            self,
            config_loader: ConfigLoader,
            llm_service: LLMService,
            rules_dir: Path
    ):
        """
        初始化提取节点

        Args:
            config_loader: 配置加载器
            llm_service: LLM服务
            rules_dir: 规则目录
        """
        self.config_loader = config_loader
        self.llm_service = llm_service
        self.rule_loader = RuleLoader(rules_dir)
        self.paths_config = config_loader.get_paths_config()
        self.workspace_dir = resolve_path(self.paths_config.workspace_dir)

    def _create_version_dir(
            self,
            operator_name: str,
            module: str,
            version: int
    ) -> Path:
        """创建版本目录"""
        version_dir = self.workspace_dir / operator_name / module / f"v{version}"
        version_dir.mkdir(parents=True, exist_ok=True)
        return version_dir

    async def _call_llm(
            self,
            prompt: str,
            module: str,
            retry_count: int = 0
    ) -> str:
        """调用LLM进行提取"""
        context = f"[{module}] 提取操作" if retry_count == 0 else f"[{module}] 提取操作(重试{retry_count + 1})"
        response = await self.llm_service.invoke_with_retry(prompt, context=context)
        logger.debug(f"[{module}] LLM响应长度: {len(response)} 字符")
        return response

    def _parse_json_response(
            self,
            response: str,
            module: str
    ) -> Dict[str, Any]:
        """解析LLM响应中的JSON数据"""
        # 尝试提取JSON部分（支持对象和数组）
        # 先尝试找数组
        json_start_array = response.find('[')
        json_start_object = response.find('{')

        # 确定JSON的起始位置和类型
        if json_start_array != -1 and (json_start_object == -1 or json_start_array < json_start_object):
            # JSON是数组
            json_start = json_start_array
            json_end = response.rfind(']') + 1
        elif json_start_object != -1:
            # JSON是对象
            json_start = json_start_object
            json_end = response.rfind('}') + 1
        else:
            # 未找到JSON数据
            json_start = -1
            json_end = -1

        # 检查是否找到JSON数据
        if json_start == -1 or json_end <= json_start:
            # 未找到JSON数据
            if module in OPTIONAL_MODULES:
                logger.info(f"[{module}] 可选模块未提取到JSON数据,返回空字典")
                return {}
            else:
                raise ModuleProcessingError(module, "LLM响应中未找到有效的JSON数据")

        # 解析JSON响应
        json_str = response[json_start:json_end]
        try:
            data = json.loads(json_str)
            logger.debug(f"[{module}] JSON解析成功,数据大小: {len(json_str)} 字符")

            # 👇 如果data是数组，转换为字典（根据模块类型）
            if isinstance(data, list):
                data = self._convert_array_to_dict(data, module)

            return data
        except json.JSONDecodeError as e:
            raise ModuleProcessingError(module, f"JSON解析失败: {str(e)}")

    def _convert_array_to_dict(
            self,
            data: list,
            module: str
    ) -> Dict[str, Any]:
        """
        将数组转换为字典格式

        Args:
            data: JSON数组
            module: 模块名称

        Returns:
            Dict[str, Any]: 转换后的字典

        说明:
            根据模块类型，将数组包装成合适的字典格式
        """
        # 根据模块类型确定包装方式
        if module == "param_constraints":
            # param_constraints 模块期望格式: {"param_constraints": [...]}
            return {"param_constraints": data}
        elif module == "functions":
            # functions 模块期望格式: {"functions": [...]}
            return {"functions": data}
        elif module == "dtype_map":
            # dtype_map 模块期望格式: {"dtype_map": [...]}
            return {"dtype_map": data}
        elif module == "inter_parameter_constraints":
            # inter_parameter_constraints 模块期望格式: {"inter_parameter_constraints": [...]}
            return {"inter_parameter_constraints": data}
        elif module == "platform_specifics":
            # platform_specifics 模块期望格式: {"platform_specifics": [...]}
            return {"platform_specifics": data}
        else:
            # 其他模块，直接返回第一个元素（如果是单元素数组）
            # 或者包装成通用格式
            if len(data) == 1 and isinstance(data[0], dict):
                logger.info(f"[{module}] 数组包含单个对象，直接返回该对象")
                return data[0]
            else:
                # 包装成 {module: data} 格式
                logger.info(f"[{module}] 将数组包装为字典格式: {{{module}: [...]}}")
                return {module: data}

    def _build_extraction_prompt(
            self,
            module: str,
            operator_doc: str,
            rule_content: str,
            json_validation_error: Optional[str] = None,
            validation_error: Optional[str] = None
    ) -> str:
        """
        构建提取提示词

        Args:
            module: 模块名称
            operator_doc: 算子文档
            rule_content: 提取规则内容
            json_validation_error: JSON校验错误信息
            validation_error: 业务校验错误信息

        Returns:
            str: 完整的提示词
        """
        # 直接调用PromptBuilder.build_extraction_prompt
        # 它会内部调用build_error_hint来构建错误提示
        return PromptBuilder.build_extraction_prompt(
            operator_doc,
            rule_content,
            module,
            json_validation_error,
            validation_error
        )

    async def extract(
            self,
            state: ModuleProcessingState
    ) -> ModuleProcessingState:
        """
        执行提取操作

        Args:
            state: 当前状态

        Returns:
            ModuleProcessingState: 更新后的状态
        """
        module = state["module_name"]
        operator_name = state["operator_name"]
        operator_doc = state["operator_doc"]
        outer_iteration = state["outer_iteration"]  # 大循环迭代次数
        inner_iteration = state["inner_iteration"]  # 小循环迭代次数

        # 获取错误信息（JSON校验错误 或 业务校验错误）
        json_validation_error = state.get("json_validation_error")
        validation_error = state.get("validation_error")

        logger.info(
            f"[{module}] 开始提取 "
            f"(大循环第{outer_iteration + 1}次，小循环第{inner_iteration + 1}次)"
        )

        try:
            # 1. 创建版本目录
            version_dir = self._create_version_dir(operator_name, module, outer_iteration)
            state["version_dir"] = str(version_dir)
            logger.debug(
                f"[{module}] 创建版本目录: {version_dir} "
                f"(大循环第{outer_iteration + 1}次，小循环第{inner_iteration + 1}次)"
            )

            # 2. 加载提取规则（基础规则）
            rule_content = self.rule_loader.load_rule_file(
                module,
                is_check=False,
                previous_error_path=None
            )

            # 3. 构建提示词
            prompt = self._build_extraction_prompt(
                module,
                operator_doc,
                rule_content,
                json_validation_error,
                validation_error
            )

            # 4. 调用LLM进行提取（最多重试3次）
            extracted_data = await self._extract_with_retry(
                module,
                prompt
            )

            # 5. 保存提取结果
            result_path = ResultSaver.save_result(version_dir, module, extracted_data)
            logger.info(f"[{module}] 提取完成,结果已保存: {result_path}")

            # 6. 更新状态
            state["extracted_data"] = extracted_data
            state["extraction_result"] = ExtractionResult(
                success=True,
                data=extracted_data,
                error=None,
                version=outer_iteration,
                result_path=str(result_path)
            )

            # 清除JSON校验错误信息（已使用）
            # 保留validation_error，让validation_node自己管理
            # 这样在json_validate失败后重试时，仍能看到之前的业务校验错误
            state["json_validation_error"] = None

            return state

        except Exception as e:
            logger.error(f"[{module}] 提取失败: {e}")

            # 更新状态
            state["extraction_result"] = ExtractionResult(
                success=False,
                data=None,
                error=str(e),
                version=outer_iteration,
                result_path=None
            )
            state["error"] = f"提取失败: {str(e)}"

            return state

    async def _extract_with_retry(
            self,
            module: str,
            initial_prompt: str
    ) -> Dict[str, Any]:
        """
        执行提取操作（LLM调用失败由llm_service处理重试）

        Args:
            module: 模块名称
            initial_prompt: 初始提示词

        Returns:
            Dict[str, Any]: 提取的数据

        Raises:
            ModuleProcessingError: 提取失败

        说明:
            - LLM调用失败（如500错误）由 llm_service.invoke_with_retry 处理重试（最多5次）
            - 如果LLM调用重试5次后仍失败，会抛出异常，本方法不再重试
            - JSON解析失败不重试，直接抛出异常
        """
        try:
            # 调用LLM进行提取（LLM调用失败会自动重试5次）
            response = await self._call_llm(initial_prompt, module, 0)

            # 解析JSON响应
            extracted_data = self._parse_json_response(response, module)
            return extracted_data

        except Exception as e:
            # LLM调用失败或JSON解析失败
            if module in OPTIONAL_MODULES:
                logger.warning(f"[{module}] 可选模块提取失败，返回空数据: {e}")
                return {}
            else:
                raise ModuleProcessingError(
                    module,
                    f"提取失败: {e}"
                )
