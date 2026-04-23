"""
程序实体定义模块
定义程序运行时需要的实体类
"""
from enum import Enum
from typing import Dict, Any, Optional
from typing_extensions import TypedDict


class ModuleStatus(Enum):
    """模块处理状态"""
    PENDING = "pending"          # 待处理
    PROCESSING = "processing"    # 处理中
    SUCCESS = "success"          # 成功
    FAILED = "failed"            # 失败


class ModuleResult(TypedDict):
    """模块处理结果"""
    module: str                  # 模块名称
    status: str                  # 处理状态
    version: int                 # 版本号
    result_path: Optional[str]   # 结果文件路径
    error_path: Optional[str]    # 错误文件路径
    iterations: int              # 迭代次数
    error_message: Optional[str] # 错误信息


class GraphState(TypedDict):
    """工作流状态"""
    operator_name: str                          # 算子名称
    operator_doc: str                           # 原始文档内容
    module_results: Dict[str, ModuleResult]     # 各模块处理结果
    merged_result_path: Optional[str]           # 合并结果路径
    test_cases_path: Optional[str]              # 测试用例路径
    current_iteration: int                      # 当前迭代次数
    error: Optional[str]                        # 全局错误信息
