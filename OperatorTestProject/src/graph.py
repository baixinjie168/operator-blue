"""
LangGraph工作流模块
负责整体工作流编排和并发处理
"""
import asyncio
import logging
from pathlib import Path

from langgraph.graph import StateGraph, END

from .config_loader import ConfigLoader
from .llm_service import LLMService
from .models import GraphState, ModuleResult
from .module_processor import ModuleProcessor, MODULES
from .skill_executor import SkillExecutor

# 获取logger
logger = logging.getLogger(__name__)


class OperatorProcessingGraph:
    """算子处理工作流"""
    
    def __init__(
        self,
        config_loader: ConfigLoader,
        llm_service: LLMService,
        module_processor: ModuleProcessor,
        skill_executor: SkillExecutor
    ):
        """
        初始化工作流
        
        Args:
            config_loader: 配置加载器
            llm_service: LLM服务
            module_processor: 模块处理器
            skill_executor: Skill执行器
        """
        self.config_loader = config_loader
        self.llm_service = llm_service
        self.module_processor = module_processor
        self.skill_executor = skill_executor
        
        # 获取线程池大小
        pool_config = config_loader.get_thread_pool_config()
        if pool_config.size == "auto":
            self.pool_size = llm_service.get_pool_size()
        else:
            self.pool_size = int(pool_config.size)

        # 获取模块执行模式
        module_execution_config = config_loader.get_module_execution_config()
        self.execution_mode = module_execution_config.mode
        
        # 构建工作流图
        self.graph = self._build_graph()
    
    def _initialize_node(self, state: GraphState) -> GraphState:
        """
        初始化节点
        
        Args:
            state: 当前状态
            
        Returns:
            GraphState: 更新后的状态
        """
        state["module_results"] = {}
        state["merged_result_path"] = None
        state["test_cases_path"] = None
        state["current_iteration"] = 0
        state["error"] = None
        return state
    
    async def _process_module_async(
        self,
        operator_name: str,
        operator_doc: str,
        module: str
    ) -> ModuleResult:
        """
        异步处理单个模块
        
        Args:
            operator_name: 算子名称
            operator_doc: 原始文档
            module: 模块名称
            
        Returns:
            ModuleResult: 处理结果
        """
        return await self.module_processor.process(operator_name, operator_doc, module)
    
    def _extract_modules_node(self, state: GraphState) -> GraphState:
        """
        提取模块节点(根据配置选择顺序或并发处理)

        Args:
            state: 当前状态

        Returns:
            GraphState: 更新后的状态
        """
        operator_name = state["operator_name"]
        operator_doc = state["operator_doc"]

        # 根据执行模式选择处理方式
        if self.execution_mode == "sequential":
            return self._extract_modules_sequential(state, operator_name, operator_doc)
        else:
            return self._extract_modules_concurrent(state, operator_name, operator_doc)

    def _extract_modules_sequential(self, state: GraphState, operator_name: str, operator_doc: str) -> GraphState:
        """
        顺序提取模块(一个模块完成后继续处理下一个模块)

        Args:
            state: 当前状态
            operator_name: 算子名称
            operator_doc: 原始文档

        Returns:
            GraphState: 更新后的状态
        """
        logger.info(f"开始顺序处理 {len(MODULES)} 个模块: {', '.join(MODULES)}")

        # 创建事件循环
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        try:
            success_count = 0
            failed_count = 0
            failed_modules = []

            # 顺序处理每个模块
            for module in MODULES:
                logger.info(f"开始处理模块: {module}")

                try:
                    # 处理单个模块
                    result = loop.run_until_complete(
                        self._process_module_async(operator_name, operator_doc, module)
                    )

                    state["module_results"][module] = result

                    # 检查模块处理结果
                    if result["status"] == "success":
                        success_count += 1
                        logger.info(f"模块 {module} 处理成功")
                    else:
                        failed_count += 1
                        failed_modules.append(module)
                        logger.error(f"模块 {module} 处理失败: {result.get('error_message', '未知错误')}")

                except Exception as e:
                    failed_count += 1
                    failed_modules.append(module)
                    logger.error(f"模块 {module} 处理失败: {e}")
                    # 记录失败结果
                    state["module_results"][module] = {
                        "module": module,
                        "status": "failed",
                        "error_message": str(e)
                    }

            # 所有模块处理完成
            if failed_count > 0:
                state["error"] = f"{failed_count} 个模块处理失败: {', '.join(failed_modules)}"
                logger.warning(f"部分模块处理失败! 成功: {success_count}, 失败: {failed_count}")
                logger.warning(f"失败模块: {', '.join(failed_modules)}")
            else:
                logger.info(f"所有模块处理完成! 成功: {success_count}, 失败: {failed_count}")

        finally:
            loop.close()

        return state

    def _extract_modules_concurrent(self, state: GraphState, operator_name: str, operator_doc: str) -> GraphState:
        """
        并发提取模块(所有模块同时处理)

        Args:
            state: 当前状态
            operator_name: 算子名称
            operator_doc: 原始文档

        Returns:
            GraphState: 更新后的状态
        """
        logger.info(f"开始并发处理 {len(MODULES)} 个模块: {', '.join(MODULES)}")

        # 创建事件循环
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        try:
            # 创建所有模块的处理任务
            tasks = [
                self._process_module_async(operator_name, operator_doc, module)
                for module in MODULES
            ]

            # 并发执行所有任务
            results = loop.run_until_complete(asyncio.gather(*tasks, return_exceptions=True))

            success_count = 0
            failed_count = 0
            failed_modules = []

            # 处理结果
            for module, result in zip(MODULES, results):
                if isinstance(result, Exception):
                    failed_count += 1
                    failed_modules.append(module)
                    logger.error(f"模块 {module} 处理失败: {result}")
                    state["module_results"][module] = ModuleResult(
                        module=module,
                        status="failed",
                        version=0,
                        result_path=None,
                        error_path=None,
                        iterations=0,
                        error_message=str(result)
                    )
                else:
                    state["module_results"][module] = result
                    if result["status"] == "success":
                        success_count += 1
                        logger.info(f"模块 {module} 处理成功")
                    else:
                        failed_count += 1
                        failed_modules.append(module)
                        logger.error(f"模块 {module} 处理失败: {result.get('error_message', '未知错误')}")

            # 如果有模块失败，记录错误但不终止
            if failed_count > 0:
                state["error"] = f"{failed_count} 个模块处理失败: {', '.join(failed_modules)}"
                logger.warning(f"部分模块处理失败! 成功: {success_count}, 失败: {failed_count}")
                logger.warning(f"失败模块: {', '.join(failed_modules)}")
            else:
                logger.info(f"所有模块处理完成! 成功: {success_count}, 失败: {failed_count}")

        finally:
            loop.close()

        return state
    
    async def _merge_results_node_async(self, state: GraphState) -> GraphState:
        """
        异步合并结果节点
        
        Args:
            state: 当前状态
            
        Returns:
            GraphState: 更新后的状态
        """
        # 如果有错误,跳过合并
        if state.get("error"):
            logger.warning(f"检测到错误,跳过合并: {state['error']}")
            return state
        
        operator_name = state["operator_name"]
        
        logger.info("开始合并各模块结果...")
        
        # 收集各模块的JSON文件路径
        module_json_files = {}
        for module, result in state["module_results"].items():
            if result["result_path"]:
                module_json_files[module] = Path(result["result_path"])
        
        # 检查是否有成功的模块
        if not module_json_files:
            logger.error("没有成功的模块结果可供合并!")
            state["error"] = "没有成功的模块结果可供合并"
            return state
        
        logger.info(f"收集到 {len(module_json_files)} 个模块的JSON文件")
        
        # 执行合并
        merged_data = await self.skill_executor.merge_operator_rules(
            operator_name,
            module_json_files
        )
        
        # 保存合并结果
        merged_path = self.skill_executor.save_merged_result(operator_name, merged_data)
        state["merged_result_path"] = str(merged_path)
        
        logger.info(f"合并完成! 结果已保存: {merged_path}")
        
        return state
    
    def _merge_results_node(self, state: GraphState) -> GraphState:
        """
        合并结果节点
        
        Args:
            state: 当前状态
            
        Returns:
            GraphState: 更新后的状态
        """
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        try:
            return loop.run_until_complete(self._merge_results_node_async(state))
        finally:
            loop.close()
    
    async def _generate_test_cases_node_async(self, state: GraphState) -> GraphState:
        """
        异步生成测试用例节点
        
        Args:
            state: 当前状态
            
        Returns:
            GraphState: 更新后的状态
        """
        # 如果有错误,跳过测试用例生成
        if state.get("error"):
            logger.warning(f"检测到错误,跳过测试用例生成: {state['error']}")
            return state
        
        operator_name = state["operator_name"]
        merged_result_path = Path(state["merged_result_path"])
        
        logger.info("开始生成测试用例...")
        
        # 获取测试用例数量配置
        test_case_config = self.config_loader.get_test_case_config()
        logger.info(f"测试用例数量配置: {test_case_config.count}")
        
        # 生成测试用例（已自动保存）
        test_cases_path = await self.skill_executor.generate_test_cases(
            merged_result_path,
            test_case_config.count
        )
        state["test_cases_path"] = str(test_cases_path)

        logger.info(f"测试用例生成完成! 已保存: {test_cases_path}")
        
        return state
    
    def _generate_test_cases_node(self, state: GraphState) -> GraphState:
        """
        生成测试用例节点
        
        Args:
            state: 当前状态
            
        Returns:
            GraphState: 更新后的状态
        """
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        try:
            return loop.run_until_complete(self._generate_test_cases_node_async(state))
        finally:
            loop.close()
    
    def _build_graph(self) -> StateGraph:
        """
        构建工作流图
        
        Returns:
            StateGraph: 工作流图
        """
        # 创建状态图
        workflow = StateGraph(GraphState)
        
        # 添加节点
        workflow.add_node("initialize", self._initialize_node)
        workflow.add_node("extract_modules", self._extract_modules_node)
        workflow.add_node("merge_results", self._merge_results_node)
        workflow.add_node("generate_test_cases", self._generate_test_cases_node)
        
        # 设置入口点
        workflow.set_entry_point("initialize")
        
        # 添加边
        workflow.add_edge("initialize", "extract_modules")
        workflow.add_edge("extract_modules", "merge_results")
        workflow.add_edge("merge_results", "generate_test_cases")
        workflow.add_edge("generate_test_cases", END)
        
        # 编译工作流
        return workflow.compile()
    
    async def run(self, operator_name: str, operator_doc: str) -> GraphState:
        """
        运行工作流
        
        Args:
            operator_name: 算子名称
            operator_doc: 原始算子文档
            
        Returns:
            GraphState: 最终状态
        """
        # 初始化状态
        initial_state: GraphState = {
            "operator_name": operator_name,
            "operator_doc": operator_doc,
            "module_results": {},
            "merged_result_path": None,
            "test_cases_path": None,
            "current_iteration": 0,
            "error": None
        }
        
        # 执行工作流
        final_state = await self.graph.ainvoke(initial_state)
        
        return final_state
