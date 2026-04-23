# -*- coding: utf-8 -*-
"""
JSON结构校验节点
负责校验提取的JSON数据结构是否符合模型定义
只负责校验，不负责修正
"""
import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional, Set

from langchain_core.tools import tool

from .module_graph_state import ModuleProcessingState
from .config_loader import ConfigLoader
from .llm_service import LLMService
from .json_to_model_loader import load_json_to_model

# 获取logger
logger = logging.getLogger(__name__)


# ==================== 定义工具函数 ====================

@tool
def validate_json_structure_tool(json_str: str, module_name: str) -> str:
    """
    校验JSON字符串结构是否符合模型定义
    
    Args:
        json_str: JSON字符串
        module_name: 模块名称（如 basic_info, functions 等）
        
    Returns:
        str: 校验结果
            - "VALID" 表示校验通过
            - 错误信息表示校验失败
    """
    try:
        # 👇 计算需要忽略的字段
        fields_to_ignore = _calculate_fields_to_ignore(json_str, module_name)
        
        # 使用 fields_to_ignore 参数进行校验
        model, error = load_json_to_model(
            json_str, 
            module_name,
            fields_to_ignore=fields_to_ignore
        )
        
        if model is not None:
            logger.info(f"[{module_name}] JSON结构校验通过")
            return "VALID"
        else:
            logger.warning(f"[{module_name}] JSON结构校验失败: {error}")
            return f"INVALID: {error}"
    except Exception as e:
        logger.error(f"[{module_name}] JSON结构校验异常: {e}")
        return f"ERROR: {str(e)}"


@tool
def get_validation_details_tool(json_str: str, module_name: str) -> str:
    """
    获取JSON结构校验的详细信息
    
    Args:
        json_str: JSON字符串
        module_name: 模块名称
        
    Returns:
        str: 详细的校验信息（包括字段、类型等）
    """
    try:
        # 👇 计算需要忽略的字段
        fields_to_ignore = _calculate_fields_to_ignore(json_str, module_name)
        
        # 使用 fields_to_ignore 参数进行校验
        model, error = load_json_to_model(
            json_str, 
            module_name,
            fields_to_ignore=fields_to_ignore
        )
        
        if model is not None:
            # 返回模型信息
            return f"校验通过。模型类型: {type(model).__name__}"
        else:
            # 返回详细错误
            return f"校验失败。详细信息:\n{error}"
    except Exception as e:
        return f"校验异常: {str(e)}"


def _calculate_fields_to_ignore(json_str: str, module_name: str) -> Optional[Set[str]]:
    """
    计算需要忽略的字段
    
    逻辑：排除当前model的所有其他属性（即模型中有但JSON中没有的字段）
    
    Args:
        json_str: JSON字符串
        module_name: 模块名称
        
    Returns:
        Optional[Set[str]]: 需要忽略的字段集合
    """
    try:
        from .json_to_model_loader import JsonToModelLoader
        import json
        
        # 1. 解析JSON，获取JSON中实际存在的字段
        json_data = json.loads(json_str)
        json_fields = set(json_data.keys())
        
        # 2. 获取模型类
        model_class = JsonToModelLoader.get_model_class(module_name)
        
        # 3. 获取模型中定义的所有字段
        model_fields = set(model_class.model_fields.keys())
        
        # 4. 计算模型中有但JSON中没有的字段
        fields_to_ignore = model_fields - json_fields
        
        if fields_to_ignore:
            logger.debug(
                f"[{module_name}] JSON中缺少的字段将被忽略: {fields_to_ignore}"
            )
        
        return fields_to_ignore if fields_to_ignore else None
        
    except Exception as e:
        logger.warning(f"[{module_name}] 计算忽略字段失败: {e}")
        return None


class JsonValidationNode:
    """JSON结构校验节点（只负责校验，不负责修正）"""
    
    def __init__(
            self,
            config_loader: ConfigLoader,
            llm_service: LLMService,
            rules_dir: Path
    ):
        """
        初始化JSON结构校验节点
        
        Args:
            config_loader: 配置加载器
            llm_service: LLM服务（不使用，但保持接口一致）
            rules_dir: 规则目录（不使用，但保持接口一致）
        """
        self.config_loader = config_loader
        self.llm_service = llm_service
        self.rules_dir = rules_dir
    
    async def validate(
            self,
            state: ModuleProcessingState
    ) -> ModuleProcessingState:
        """
        执行JSON结构校验（只校验，不修正）
        
        Args:
            state: 当前状态
            
        Returns:
            ModuleProcessingState: 更新后的状态
            
        说明:
            - 如果校验通过，设置 json_validation_success=True
            - 如果校验失败，设置 json_validation_success=False，并记录错误信息
            - 错误信息将传递给 extract node，由 extract node 负责重新提取
            - 👇 如果校验失败，在这里更新迭代次数（而不是在路由函数中）
        """
        module = state["module_name"]
        extracted_data = state.get("extracted_data")
        inner_iteration = state["inner_iteration"]
        max_inner_iterations = state["max_inner_iterations"]
        
        logger.info(f"[{module}] 开始JSON结构校验")
        
        # 检查是否有提取数据
        if not extracted_data:
            logger.warning(f"[{module}] 没有提取数据，跳过JSON结构校验")
            state["json_validation_success"] = True
            state["json_validation_error"] = None
            return state
        
        try:
            # 将数据转换为JSON字符串
            json_str = json.dumps(extracted_data, ensure_ascii=False)
            
            # 👇 使用 @tool 装饰的工具函数进行校验
            validation_result = validate_json_structure_tool.invoke({
                "json_str": json_str,
                "module_name": module
            })
            
            if validation_result == "VALID":
                # ✅ 校验通过
                logger.info(f"[{module}] JSON结构校验通过")
                state["json_validation_success"] = True
                state["json_validation_error"] = None
                return state
            
            # ❌ 校验失败，获取详细错误信息
            error = validation_result.replace("INVALID: ", "").replace("ERROR: ", "")
            logger.warning(f"[{module}] JSON结构校验失败: {error}")
            
            # 获取详细错误信息
            details = get_validation_details_tool.invoke({
                "json_str": json_str,
                "module_name": module
            })
            logger.debug(f"[{module}] 详细错误: {details}")
            
            # 👇 只记录错误信息，不调用LLM修正
            # 错误信息将传递给 extract node，由 extract node 负责重新提取
            state["json_validation_success"] = False
            state["json_validation_error"] = f"{error}\n\n详细错误:\n{details}"
            
            logger.info(f"[{module}] JSON结构校验失败，错误信息已记录，将由 extract node 重新提取")
            
            # 👇 在这里更新迭代次数（而不是在路由函数中）
            if inner_iteration < max_inner_iterations - 1:
                # 小循环可以重试
                new_inner_iteration = inner_iteration + 1
                state["inner_iteration"] = new_inner_iteration
                
                logger.info(
                    f"[{module}] JSON结构校验失败，"
                    f"小循环第 {inner_iteration + 1} 次迭代，"
                    f"准备第 {new_inner_iteration + 1} 次小循环迭代"
                )
            else:
                # 小循环达到最大迭代次数
                state["error"] = (
                    f"[{module}] 小循环达到最大迭代次数 {max_inner_iterations}，"
                    f"JSON结构校验失败"
                )
                logger.error(state["error"])
            
            return state
                
        except Exception as e:
            logger.error(f"[{module}] JSON结构校验异常: {e}")
            state["json_validation_success"] = False
            state["json_validation_error"] = str(e)
            return state
