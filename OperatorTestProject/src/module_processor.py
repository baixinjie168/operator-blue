"""
模块处理器模块（重构版）
作为ModuleProcessingGraph的适配器，保持向后兼容
"""
import logging
from pathlib import Path
from typing import Dict, Any

from .config_loader import ConfigLoader
from .llm_service import LLMService
from .skill_executor import SkillExecutor
from .models import ModuleResult
from .module_processing_graph import ModuleProcessingGraph
from .module_graph_state import ModuleProcessingStatus
from .path_utils import resolve_path
from .exceptions import ModuleProcessingError

# 获取logger
logger = logging.getLogger(__name__)

# 7个核心模块列表(按处理顺序)
MODULES = [
    "basic_info",
    "parameter_constraints",
    "inter_parameter_constraints",
    "platform_specifics",
    "other_parameters",
    "functions",
    "dtype_map",
]


class ModuleProcessor:
    """
    模块处理器（适配器模式）

    将新的ModuleProcessingGraph包装为旧的接口，
    保持向后兼容性
    """

    def __init__(
            self,
            config_loader: ConfigLoader,
            llm_service: LLMService,
            skill_executor: SkillExecutor
    ):
        """
        初始化模块处理器

        Args:
            config_loader: 配置加载器
            llm_service: LLM服务
            skill_executor: Skill执行器
        """
        self.config_loader = config_loader
        self.llm_service = llm_service
        self.skill_executor = skill_executor

        self.paths_config = config_loader.get_paths_config()
        self.workspace_dir = resolve_path(self.paths_config.workspace_dir)
        self.rules_dir = resolve_path(self.paths_config.rules_dir)

        # 初始化模块处理图
        self.module_graph = ModuleProcessingGraph(
            config_loader,
            llm_service,
            self.rules_dir
        )

    async def process(
            self,
            operator_name: str,
            operator_doc: str,
            module: str
    ) -> ModuleResult:
        """
        处理单个模块

        Args:
            operator_name: 算子名称
            operator_doc: 原始算子文档
            module: 模块名称

        Returns:
            ModuleResult: 处理结果
        """
        logger.info(f"[{module}] 开始处理模块")

        try:
            # 使用新的图处理
            final_state = await self.module_graph.process(
                module,
                operator_name,
                operator_doc
            )

            # 转换为旧的ModuleResult格式
            status = final_state["status"]

            if status == ModuleProcessingStatus.SUCCESS.value:
                # 成功
                return ModuleResult(
                    module=module,
                    status="success",
                    version=final_state["outer_iteration"],
                    result_path=final_state.get("final_result_path"),
                    error_path=None,
                    iterations=final_state["total_iterations"],
                    error_message=None
                )
            else:
                # 失败
                error_msg = final_state.get("error", "处理失败")
                return ModuleResult(
                    module=module,
                    status="failed",
                    version=final_state["outer_iteration"],
                    result_path=None,
                    error_path=final_state.get("final_error_path"),
                    iterations=final_state["total_iterations"],
                    error_message=error_msg
                )

        except Exception as e:
            logger.error(f"[{module}] 处理失败: {e}")
            return ModuleResult(
                module=module,
                status="failed",
                version=0,
                result_path=None,
                error_path=None,
                iterations=0,
                error_message=str(e)
            )
