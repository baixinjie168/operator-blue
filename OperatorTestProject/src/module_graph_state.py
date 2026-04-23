"""
模块处理状态定义
用于LangGraph工作流的状态管理
"""
from enum import Enum
from typing import Dict, Any, Optional, List
from typing_extensions import TypedDict
from pathlib import Path


class ModuleProcessingStatus(str, Enum):
    """模块处理状态枚举"""
    PENDING = "pending"              # 待处理
    EXTRACTING = "extracting"        # 提取中
    JSON_VALIDATING = "json_validating"  # JSON结构校验中
    VALIDATING = "validating"        # 业务校验中
    SUCCESS = "success"              # 成功
    FAILED = "failed"                # 失败
    RETRY = "retry"                  # 重试


class ExtractionResult(TypedDict):
    """提取结果"""
    success: bool                    # 是否成功
    data: Optional[Dict[str, Any]]   # 提取的数据
    error: Optional[str]             # 错误信息
    version: int                     # 版本号
    result_path: Optional[str]       # 结果文件路径


class ValidationResult(TypedDict):
    """校验结果"""
    success: bool                    # 是否通过
    error: Optional[str]             # 错误信息
    error_path: Optional[str]        # 错误文件路径


class ModuleProcessingState(TypedDict):
    """单个模块处理状态"""
    # 基础信息
    module_name: str                         # 模块名称
    operator_name: str                       # 算子名称
    operator_doc: str                        # 原始文档

    # 状态跟踪
    status: str                              # 当前状态
    
    # 双层迭代控制
    outer_iteration: int                     # 大循环迭代次数（validate ↔ extract）
    max_outer_iterations: int                # 大循环最大迭代次数
    inner_iteration: int                     # 小循环迭代次数（json_validate ↔ extract）
    max_inner_iterations: int                # 小循环最大迭代次数

    # 提取相关
    extracted_data: Optional[Dict[str, Any]] # 提取的数据
    extraction_result: Optional[ExtractionResult]  # 提取结果
    version_dir: Optional[str]               # 版本目录路径
    previous_error_path: Optional[str]       # 上一版本的错误文件路径

    # JSON结构校验相关
    json_validation_success: Optional[bool]  # JSON结构校验是否通过
    json_validation_error: Optional[str]     # JSON结构校验错误信息

    # 业务校验相关
    validation_result: Optional[ValidationResult]   # 校验结果
    validation_error: Optional[str]          # 业务校验错误信息（用于传递给extract node）

    # 最终结果
    final_result_path: Optional[str]         # 最终结果文件路径
    final_error_path: Optional[str]          # 最终错误文件路径
    total_iterations: int                    # 总迭代次数

    # 错误信息
    error: Optional[str]                     # 全局错误信息


class ModuleGraphState(TypedDict):
    """模块处理工作流状态（顶层）"""
    # 算子信息
    operator_name: str                       # 算子名称
    operator_doc: str                        # 原始文档

    # 当前处理的模块
    current_module: str                      # 当前模块名
    current_module_state: ModuleProcessingState  # 当前模块状态

    # 所有模块列表
    modules_to_process: List[str]            # 待处理的模块列表
    processed_modules: List[str]             # 已处理的模块列表

    # 模块结果
    module_results: Dict[str, Dict[str, Any]]  # 各模块处理结果

    # 全局状态
    global_error: Optional[str]              # 全局错误信息
