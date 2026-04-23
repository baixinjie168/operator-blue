"""
算子规则 JSON 的 Pydantic 数据模型定义
用于严格校验 JSON 数据结构和类型
"""

from enum import Enum
from typing import List, Union, Optional

from pydantic import BaseModel, Field


# ==================== 枚举定义 ====================

class InterConstraintsRuleType(str, Enum):
    """参数间约束类型枚举"""
    SHAPE_BROADCAST = "shape_broadcast"
    SHAPE_CHOICE = "shape_choice"
    SHAPE_EQUALITY = "shape_equality"
    SHAPE_DEPENDENCY = "shape_dependency"
    SHAPE_VALUE_DEPENDENCY = "shape_value_dependency"
    PRESENCE_DEPENDENCY = "presence_dependency"
    TYPE_DEPENDENCY = "type_dependency"
    TYPE_EQUALITY = "type_equality"
    VALUE_DEPENDENCY = "value_dependency"
    FORMAT_EQUALITY = "format_equality"


class ParameterRole(str, Enum):
    """参数角色枚举"""
    INPUT = "input"
    OUTPUT = "output"


class ShapeStructure(str, Enum):
    """Shape约束结构类型枚举"""
    DIMS = "dims"
    AXIS_VALUE = "axis_value"


class ApiFlow(str, Enum):
    """API流程类型枚举"""
    ONE_STEP = "one-step"
    TWO_STEP = "two-step"


class ParameterType(str, Enum):
    """参数类型枚举"""
    ACL_TENSOR = "aclTensor"
    ACL_SCALAR = "aclScalar"
    ACL_INT_ARRAY = "aclIntArray"
    ACL_FLOAT_ARRAY = "aclFloatArray"
    ACL_BOOL_ARRAY = "aclBoolArray"
    ACL_TENSOR_LIST = "aclTensorList"
    ACL_SCALAR_LIST = "aclScalarList"
    ACL_OP_EXECUTOR = "aclOpExecutor"
    ACL_RT_STREAM = "aclrtStream"
    # 标量类型
    DOUBLE = "double"
    BOOL = "bool"
    UINT64_T = "uint64_t"
    INT64_T = "int64_t"
    INT32_T = "int32_t"
    FLOAT = "float"
    FLOAT16 = "float16"
    VOID = "void"
    SIZE_T = "size_t"


# ==================== 顶层模型 ====================

class OperatorRule(BaseModel):
    """算子规则顶层模型"""
    operation_name: str = Field(..., description="算子名称")
    description: str = Field(..., description="算子描述")
    api_flow: ApiFlow = Field(..., description="API流程类型: one-step 或 two-step")
    functions: List["OperatorFunction"] = Field(..., description="函数列表")
    parameter_constraints: List["ParamConstraints"] = Field(..., description="参数约束列表")
    other_parameters: List["OtherParameters"] = Field(default_factory=list,
                                                      description="其他未在'函数原型'中定义的参数")
    dtype_map: List["DtypeMapItem"] = Field(default_factory=list, description="数据类型映射表")
    inter_parameter_constraints: List["InterParamConstraint"] = Field(default_factory=list, description="参数间约束")
    platform_specifics: List["PlatformSpecific"] = Field(default_factory=list, description="平台特定规则")

    model_config = {"extra": "forbid"}


# ==================== Functions 相关模型 ====================

class Parameter(BaseModel):
    """函数参数模型"""
    name: str = Field(..., description="参数名称")
    role: ParameterRole = Field(..., description="参数角色")
    type: str = Field(..., description="参数类型")
    is_optional: bool = Field(..., description="是否可选")
    description: str = Field(..., description="参数描述")
    format: Optional[Union[List[str], str]] = Field(default=None, description="Tensor格式")

    model_config = {"extra": "forbid"}


class OperatorFunction(BaseModel):
    """算子函数模型"""
    api_name: str = Field(..., description="API名称")
    description: str = Field(..., description="函数描述")
    parameters: List[Parameter] = Field(..., description="参数列表")

    model_config = {"extra": "forbid"}


# ==================== Parameter Constraints 相关模型 ====================

class ShapeRule(BaseModel):
    """Shape约束规则模型"""
    structure: ShapeStructure = Field(..., description="结构类型: dims 或 axis_value")
    rule: str = Field(..., description="Python表达式规则")
    dim_num: List[List[int]] | List[int] = Field(default_factory=list,
                                                 description="dim的取值范围，如[[2,4],6]表示shape的dim取值为6或者2<=dim<=4")
    dim_valid_value: List[List[int]] | List[int] = Field(default_factory=list, description="shape中每一维合法取值的范围")
    dim_invalid_value: List[List[int]] | List[int] = Field(default_factory=list, description="shape中每一维非法取值范围")

    model_config = {"extra": "forbid"}


class ParamShape(BaseModel):
    """参数Shape约束模型"""
    platform: str = Field(..., description="平台名称或列表")
    constraint: List[ShapeRule] = Field(..., description="约束规则列表")

    model_config = {"extra": "forbid"}


class ParamDataType(BaseModel):
    """参数数据类型约束模型"""
    platform: str = Field(..., description="平台名称或列表")
    types: List[str] = Field(..., description="支持的类型列表")

    model_config = {"extra": "forbid"}


class ParamMemory(BaseModel):
    """参数内存约束模型"""
    platform: str = Field(..., description="平台名称或列表")
    discontinuous: bool = Field(..., description="是否支持非连续Tensor")

    model_config = {"extra": "forbid"}


class ParamValue(BaseModel):
    """参数值约束模型"""
    platform: str = Field(..., description="平台名称或列表")
    value: List[List[float]] | List[float] | List[List[int]] | List[int] | List[str] | List[bool] = Field(
        default_factory=list, description="允许或禁止的值")

    model_config = {"extra": "forbid"}


class SingleParamConstraints(BaseModel):
    """单个参数的约束模型"""
    shape: List[ParamShape] = Field(..., description="Shape约束列表")
    data_types: List[ParamDataType] = Field(..., description="数据类型约束列表")
    memory: Optional[List[ParamMemory]] = Field(..., description="内存约束列表")
    allowed_values: Optional[List[ParamValue]] = Field(..., description="允许的值列表")
    not_allowed_values: Optional[List[ParamValue]] = Field(..., description="禁止的值列表")

    model_config = {"extra": "forbid"}


class ParamConstraints(BaseModel):
    """参数约束模型"""
    name: str = Field(..., description="参数名称")
    constraints: SingleParamConstraints = Field(..., description="参数约束详情")

    model_config = {"extra": "forbid"}


# ================== other parameters相关模型 ====================


class OtherParameterConstraint(BaseModel):
    platform: str = Field(..., description="平台名称")
    value: List[List[float]] | List[float] | List[List[int]] | List[int] | List[str] | List[bool] = Field(
        default_factory=list, description="参数的允许或禁止的值")
    rule: str = Field(default='', description="未在'函数原型'中定义的参数需要满足的约束条件")

    model_config = {"extra": "forbid"}


class OtherParameters(BaseModel):
    name: str = Field(..., description="参数名称")
    type: str = Field(..., description="参数类型")
    description: str = Field(default='', description="参数描述")
    constraints: List[OtherParameterConstraint] = Field(..., description="参数相关约束")

    model_config = {"extra": "forbid"}


# ==================== dtype_map 相关模型 ====================

class DtypeMapItem(BaseModel):
    """数据类型映射表项模型"""
    platform: str = Field(..., description="平台名称")
    columns: List[str] = Field(..., description="列名列表")
    rows: List[List[str]] = Field(..., description="数据类型组合行列表")

    model_config = {"extra": "forbid"}


# ==================== inter_parameter_constraints 相关模型 ====================

class InterParamConstraint(BaseModel):
    """参数间约束模型"""
    type: InterConstraintsRuleType = Field(..., description="约束类型")
    params: List[str] = Field(..., description="涉及的参数列表")
    expr: str = Field(..., description="约束表达式")
    description: str = Field(default="", description="约束描述")
    source: Optional[str] = Field(default=None, description="源参数")
    target: Optional[List[str]] = Field(default=None, description="目标参数列表")

    model_config = {"extra": "forbid"}


# ==================== platform_specifics 相关模型 ====================

class PlatformSpecific(BaseModel):
    """平台特定规则模型"""
    platform: str = Field(..., description="平台名称")
    description: str = Field(..., description="平台描述")
    constraint_detail: str = Field(..., description="约束详情表达式")

    model_config = {"extra": "forbid"}


# ==================== 更新向前引用 ====================

OperatorRule.model_rebuild()
