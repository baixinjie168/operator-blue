# -*- coding: utf-8 -*-
"""
模块处理图
使用LangGraph编排提取-JSON校验-业务校验-迭代流程
"""
import logging
from pathlib import Path
from typing import Literal

from langgraph.graph import StateGraph, END

from .module_graph_state import (
    ModuleProcessingState,
    ModuleProcessingStatus
)
from .extraction_node import ExtractionNode
from .json_validation_node import JsonValidationNode
from .validation_node import ValidationNode
from .config_loader import ConfigLoader
from .llm_service import LLMService
from .exceptions import MaxIterationExceededError, ModuleProcessingError
from .path_utils import resolve_path

# 获取logger
logger = logging.getLogger(__name__)


class ModuleProcessingGraph:
    """单个模块的处理图"""

    def __init__(
            self,
            config_loader: ConfigLoader,
            llm_service: LLMService,
            rules_dir: Path
    ):
        """
        初始化模块处理图

        Args:
            config_loader: 配置加载器
            llm_service: LLM服务
            rules_dir: 规则目录
        """
        self.config_loader = config_loader
        self.llm_service = llm_service
        self.rules_dir = rules_dir

        # 初始化节点
        self.extraction_node = ExtractionNode(config_loader, llm_service, rules_dir)
        self.json_validation_node = JsonValidationNode(config_loader, llm_service, rules_dir)
        self.validation_node = ValidationNode(config_loader, llm_service, rules_dir)

        # 获取迭代配置
        self.iteration_config = config_loader.get_iteration_config()

        # 构建图
        self.graph = self._build_graph()

    def _initialize_state(
            self,
            module_name: str,
            operator_name: str,
            operator_doc: str
    ) -> ModuleProcessingState:
        """初始化状态"""
        return ModuleProcessingState(
            module_name=module_name,
            operator_name=operator_name,
            operator_doc=operator_doc,
            status=ModuleProcessingStatus.PENDING.value,
            # 双层迭代控制
            outer_iteration=0,  # 大循环从0开始
            max_outer_iterations=self.iteration_config.max_iterations,  # 大循环最大次数
            inner_iteration=0,  # 小循环从0开始
            max_inner_iterations=self.iteration_config.max_iterations,  # 小循环最大次数
            extracted_data=None,
            extraction_result=None,
            version_dir=None,
            previous_error_path=None,
            json_validation_success=None,
            json_validation_error=None,
            validation_result=None,
            validation_error=None,
            final_result_path=None,
            final_error_path=None,
            total_iterations=0,
            error=None
        )

    async def _extract_node(self, state: ModuleProcessingState) -> ModuleProcessingState:
        """提取节点"""
        state["status"] = ModuleProcessingStatus.EXTRACTING.value
        return await self.extraction_node.extract(state)

    async def _json_validate_node(self, state: ModuleProcessingState) -> ModuleProcessingState:
        """JSON结构校验节点"""
        state["status"] = ModuleProcessingStatus.JSON_VALIDATING.value
        return await self.json_validation_node.validate(state)

    async def _validate_node(self, state: ModuleProcessingState) -> ModuleProcessingState:
        """业务校验节点"""
        state["status"] = ModuleProcessingStatus.VALIDATING.value
        return await self.validation_node.validate(state)

    def _route_after_extraction(
            self,
            state: ModuleProcessingState
    ) -> Literal["json_validate", "fail", "end"]:
        """
        提取后的路由决策

        Returns:
            "json_validate": 提取成功，进入JSON结构校验
            "fail": 提取失败，结束
            "end": 可选模块提取为空，结束
        """
        extraction_result = state.get("extraction_result")

        if not extraction_result:
            return "fail"

        if not extraction_result.get("success"):
            # 检查是否是可选模块且提取为空
            module = state["module_name"]
            if module in ["other_parameters"] and not extraction_result.get("data"):
                logger.info(f"[{module}] 可选模块提取为空，结束处理")
                return "end"
            return "fail"

        return "json_validate"

    def _route_after_json_validation(
            self,
            state: ModuleProcessingState
    ) -> Literal["validate", "extract", "fail"]:
        """
        JSON结构校验后的路由决策（小循环控制）
        
        小循环：json_validate ↔ extract
        每次大循环内，小循环最多迭代 max_inner_iterations 次
        
        Returns:
            "validate": JSON结构校验通过，进入业务校验
            "extract": JSON结构校验失败，小循环内重试
            "fail": 小循环达到最大迭代次数，失败结束
            
        注意：迭代次数的更新已经在 json_validate 节点中完成
        """
        json_validation_success = state.get("json_validation_success")
        module = state["module_name"]

        if state.get("error"):
            logger.error(f"[{module}] JSON结构校验流程终止: {state['error']}")
            return "fail"
        
        if json_validation_success:
            # JSON结构校验通过，进入业务校验
            logger.info(f"[{module}] JSON结构校验通过，进入业务校验")
            # 重置小循环计数器，为下一次大循环做准备
            state["inner_iteration"] = 0
            return "validate"
        
        # JSON结构校验失败，继续走下一次提取；达到最大次数时由 json_validate 节点设置 error
        return "extract"

    def _route_after_validation(
            self,
            state: ModuleProcessingState
    ) -> Literal["extract", "success", "fail"]:
        """
        业务校验后的路由决策（大循环控制）
        
        大循环：validate ↔ extract
        大循环最多迭代 max_outer_iterations 次
        
        Returns:
            "extract": 校验失败，大循环重试
            "success": 校验通过，成功结束
            "fail": 大循环达到最大迭代次数，失败结束
            
        注意：迭代次数的更新已经在 validate 节点中完成
        """
        validation_result = state.get("validation_result")
        module = state["module_name"]

        if state.get("error"):
            logger.error(f"[{module}] 业务校验流程终止: {state['error']}")
            return "fail"

        if not validation_result:
            return "fail"

        if validation_result.get("success"):
            # 校验通过
            logger.info(f"[{module}] 校验通过，处理成功")
            return "success"

        # 校验失败，继续走下一次提取；达到最大次数时由 validate 节点设置 error
        return "extract"

    def _success_node(self, state: ModuleProcessingState) -> ModuleProcessingState:
        """成功节点"""
        state["status"] = ModuleProcessingStatus.SUCCESS.value
        extraction_result = state.get("extraction_result")
        if extraction_result:
            state["final_result_path"] = extraction_result.get("result_path")
        # 计算总迭代次数
        state["total_iterations"] = (
            state["outer_iteration"] * state["max_inner_iterations"] + 
            state["inner_iteration"] + 1
        )
        logger.info(
            f"[{state['module_name']}] 处理成功，"
            f"大循环迭代 {state['outer_iteration'] + 1} 次，"
            f"小循环迭代 {state['inner_iteration'] + 1} 次，"
            f"总迭代 {state['total_iterations']} 次"
        )
        return state

    def _fail_node(self, state: ModuleProcessingState) -> ModuleProcessingState:
        """失败节点"""
        state["status"] = ModuleProcessingStatus.FAILED.value
        validation_result = state.get("validation_result")
        if validation_result:
            state["final_error_path"] = validation_result.get("error_path")
        state["total_iterations"] = (
            state["outer_iteration"] * state["max_inner_iterations"] +
            state["inner_iteration"] + 1
        )
        logger.error(f"[{state['module_name']}] 处理失败")
        return state

    def _build_graph(self) -> StateGraph:
        """
        构建处理图
        
        图结构:
            START → extract → json_validate → validate → success
                       ↑              ↓            ↓
                       └──────────────┴────────────┴→ fail
        
        流程说明:
            1. extract: 提取数据
            2. json_validate: 校验JSON结构
            3. validate: 业务校验
            4. success/fail: 结束节点
        """
        # 创建状态图
        workflow = StateGraph(ModuleProcessingState)

        # 添加节点
        workflow.add_node("extract", self._extract_node)
        workflow.add_node("json_validate", self._json_validate_node)
        workflow.add_node("validate", self._validate_node)
        workflow.add_node("success", self._success_node)
        workflow.add_node("fail", self._fail_node)

        # 设置入口点
        workflow.set_entry_point("extract")

        # 添加条件边（提取后）
        workflow.add_conditional_edges(
            "extract",
            self._route_after_extraction,
            {
                "json_validate": "json_validate",
                "fail": "fail",
                "end": "success"  # 可选模块提取为空视为成功
            }
        )

        # 添加条件边（JSON结构校验后）
        workflow.add_conditional_edges(
            "json_validate",
            self._route_after_json_validation,
            {
                "validate": "validate",  # 进入业务校验
                "extract": "extract",    # 重新提取
                "fail": "fail"
            }
        )

        # 添加条件边（业务校验后）
        workflow.add_conditional_edges(
            "validate",
            self._route_after_validation,
            {
                "extract": "extract",  # 重新提取
                "success": "success",
                "fail": "fail"
            }
        )

        # 添加结束边
        workflow.add_edge("success", END)
        workflow.add_edge("fail", END)

        # 编译图
        return workflow.compile()

    async def process(
            self,
            module_name: str,
            operator_name: str,
            operator_doc: str
    ) -> ModuleProcessingState:
        """
        处理单个模块

        Args:
            module_name: 模块名称
            operator_name: 算子名称
            operator_doc: 原始文档

        Returns:
            ModuleProcessingState: 最终状态
        """
        # 初始化状态
        initial_state = self._initialize_state(
            module_name,
            operator_name,
            operator_doc
        )

        logger.info(f"[{module_name}] 开始处理模块")

        # 执行图
        final_state = await self.graph.ainvoke(initial_state)

        return final_state
