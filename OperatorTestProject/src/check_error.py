"""
错误检查数据模型定义
"""

from pydantic import BaseModel, Field


class CheckError(BaseModel):
    """错误检查模型"""
    error_path: str = Field(..., description="错误位置的JSON路径")
    error_message: str = Field(..., description="错误信息")
    fix_suggestion: str = Field(..., description="修复建议")
    is_fixed: str = Field(..., description="是否已修复")

    model_config = {"extra": "forbid"}