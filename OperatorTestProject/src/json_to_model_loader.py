"""
JSON到模型加载器
根据模块名从common_model_definition.py中找到对应的实体类，
并将JSON字符串加载为该实体对象
"""
import json
import logging
from typing import Any, Type, Union, Tuple, Optional, Set
from pydantic import BaseModel, ValidationError, create_model

from .common_model_definition import (
    OperatorRule,
    OperatorFunction,
    Parameter,
    ParamConstraints,
    SingleParamConstraints,
    ShapeRule,
    ParamShape,
    ParamDataType,
    ParamMemory,
    ParamValue,
    OtherParameters,
    OtherParameterConstraint,
    DtypeMapItem,
    InterParamConstraint,
    PlatformSpecific
)

# 获取logger
logger = logging.getLogger(__name__)

# 模块名到模型类的映射
MODULE_MODEL_MAP = {
    "basic_info": OperatorRule,
    "parameter_constraints": ParamConstraints,
    "inter_parameter_constraints": InterParamConstraint,
    "platform_specifics": PlatformSpecific,
    "functions": OperatorFunction,
    "other_parameters": OtherParameters,
    "dtype_map": DtypeMapItem,

}


class JsonToModelLoader:
    """JSON到模型加载器"""

    @staticmethod
    def get_model_class(module_name: str) -> Type[BaseModel]:
        """
        根据模块名获取对应的模型类

        Args:
            module_name: 模块名称

        Returns:
            Type[BaseModel]: 对应的Pydantic模型类

        Raises:
            ValueError: 模块名不存在
        """
        if module_name not in MODULE_MODEL_MAP:
            available_modules = list(MODULE_MODEL_MAP.keys())
            raise ValueError(
                f"未找到模块 '{module_name}' 对应的模型类。\n"
                f"可用的模块名: {available_modules}"
            )

        return MODULE_MODEL_MAP[module_name]

    @staticmethod
    def create_partial_model(
            model_class: Type[BaseModel],
            fields_to_validate: Optional[Set[str]] = None,
            fields_to_ignore: Optional[Set[str]] = None
    ) -> Type[BaseModel]:
        """
        创建部分字段校验的模型

        Args:
            model_class: 原始模型类
            fields_to_validate: 只校验这些字段（白名单）
            fields_to_ignore: 忽略这些字段的校验（黑名单）

        Returns:
            Type[BaseModel]: 新的模型类
        """
        # 获取原始模型的所有字段
        original_fields = model_class.model_fields

        # 确定要包含的字段
        if fields_to_validate is not None:
            # 白名单模式：只包含指定字段
            target_fields = fields_to_validate
        elif fields_to_ignore is not None:
            # 黑名单模式：排除指定字段
            target_fields = set(original_fields.keys()) - fields_to_ignore
        else:
            # 默认：包含所有字段
            target_fields = set(original_fields.keys())

        # 构建新模型的字段定义
        new_fields = {}
        for field_name in target_fields:
            if field_name in original_fields:
                field_info = original_fields[field_name]
                # 将必填字段改为可选
                new_fields[field_name] = (
                    field_info.annotation,
                    field_info
                )

        # 动态创建新模型
        new_model_name = f"Partial{model_class.__name__}"
        new_model = create_model(
            new_model_name,
            __base__=BaseModel,
            **new_fields
        )

        return new_model

    @staticmethod
    def load_json_to_model(
            json_str: str,
            module_name: str,
            fields_to_validate: Optional[Set[str]] = None,
            fields_to_ignore: Optional[Set[str]] = None
    ) -> Tuple[Union[BaseModel, None], Union[str, None]]:
        """
        将JSON字符串加载为指定模块的模型对象

        Args:
            json_str: JSON字符串
            module_name: 模块名称
            fields_to_validate: 只校验这些字段（白名单），优先级高于fields_to_ignore
            fields_to_ignore: 忽略这些字段的校验（黑名单）

        Returns:
            Tuple[Union[BaseModel, None], Union[str, None]]:
                - 成功时: (模型对象, None)
                - 失败时: (None, 错误信息)

        Examples:
            # 只校验指定字段
            >>> model, error = load_json_to_model(json_str, "basic_info",
            ...     fields_to_validate={"operation_name", "description", "api_flow"})

            # 忽略某些字段
            >>> model, error = load_json_to_model(json_str, "basic_info",
            ...     fields_to_ignore={"functions", "parameter_constraints"})
        """
        try:
            # 1. 解析JSON字符串
            try:
                json_data = json.loads(json_str)
            except json.JSONDecodeError as e:
                error_msg = f"JSON解析失败: {str(e)}\n位置: 第{e.lineno}行, 第{e.colno}列"
                logger.error(error_msg)
                return None, error_msg

            # 2. 获取模型类
            try:
                model_class = JsonToModelLoader.get_model_class(module_name)
            except ValueError as e:
                error_msg = str(e)
                logger.error(error_msg)
                return None, error_msg

            # 3. 如果指定了部分校验，创建部分模型
            if fields_to_validate is not None or fields_to_ignore is not None:
                model_class = JsonToModelLoader.create_partial_model(
                    model_class,
                    fields_to_validate=fields_to_validate,
                    fields_to_ignore=fields_to_ignore
                )
                logger.info(f"使用部分校验模型: {model_class.__name__}")

            # 4. 加载为模型对象
            try:
                model_instance = model_class(**json_data)
                logger.info(f"成功将JSON加载为 {module_name} 模型对象")
                return model_instance, None
            except ValidationError as e:
                # Pydantic校验错误
                error_details = []
                for error in e.errors():
                    loc = " -> ".join(str(x) for x in error['loc'])
                    msg = error['msg']
                    error_type = error['type']
                    error_details.append(f"  - 字段: {loc}\n    类型: {error_type}\n    信息: {msg}")

                error_msg = (
                        f"模型校验失败 ({module_name}):\n"
                        + "\n".join(error_details)
                )
                logger.error(error_msg)
                return None, error_msg
            except Exception as e:
                error_msg = f"模型实例化失败: {str(e)}"
                logger.error(error_msg)
                return None, error_msg

        except Exception as e:
            error_msg = f"未知错误: {str(e)}"
            logger.error(error_msg, exc_info=True)
            return None, error_msg

    @staticmethod
    def load_json_to_model_with_retry(
            json_str: str,
            module_name: str,
            auto_fix: bool = True
    ) -> Tuple[Union[BaseModel, None], Union[str, None]]:
        """
        将JSON字符串加载为模型对象，支持自动修复常见问题

        Args:
            json_str: JSON字符串
            module_name: 模块名称
            auto_fix: 是否自动修复常见问题

        Returns:
            Tuple[Union[BaseModel, None], Union[str, None]]:
                - 成功时: (模型对象, None)
                - 失败时: (None, 错误信息)
        """
        # 第一次尝试
        model, error = JsonToModelLoader.load_json_to_model(json_str, module_name)
        if model is not None:
            return model, None

        if not auto_fix:
            return None, error

        # 尝试自动修复
        logger.info("尝试自动修复JSON...")

        try:
            json_data = json.loads(json_str)

            # 修复1: 移除额外字段（如果模型配置了extra="forbid"）
            model_class = JsonToModelLoader.get_model_class(module_name)
            if hasattr(model_class, 'model_fields'):
                valid_fields = set(model_class.model_fields.keys())
                fixed_data = {k: v for k, v in json_data.items() if k in valid_fields}

                if fixed_data != json_data:
                    removed_fields = set(json_data.keys()) - valid_fields
                    logger.info(f"移除额外字段: {removed_fields}")

                    # 使用修复后的数据重试
                    fixed_json_str = json.dumps(fixed_data, ensure_ascii=False)
                    model, error = JsonToModelLoader.load_json_to_model(fixed_json_str, module_name)
                    if model is not None:
                        return model, None

            # 修复2: 尝试处理None值和空值
            # (这里可以添加更多修复逻辑)

        except Exception as e:
            logger.error(f"自动修复失败: {str(e)}")

        return None, error


def load_json_to_model(
        json_str: str,
        module_name: str,
        fields_to_validate: Optional[Set[str]] = None,
        fields_to_ignore: Optional[Set[str]] = None
) -> Tuple[Union[BaseModel, None], Union[str, None]]:
    """
    工具函数：将JSON字符串加载为指定模块的模型对象

    Args:
        json_str: JSON字符串
        module_name: 模块名称
        fields_to_validate: 只校验这些字段（白名单），优先级高于fields_to_ignore
        fields_to_ignore: 忽略这些字段的校验（黑名单）

    Returns:
        Tuple[Union[BaseModel, None], Union[str, None]]:
            - 成功时: (模型对象, None)
            - 失败时: (None, 错误信息)

    Examples:
        # 完整校验
        >>> model, error = load_json_to_model(json_str, "basic_info")

        # 只校验指定字段（白名单）
        >>> model, error = load_json_to_model(json_str, "basic_info",
        ...     fields_to_validate={"operation_name", "description", "api_flow"})

        # 忽略某些字段（黑名单）
        >>> model, error = load_json_to_model(json_str, "basic_info",
        ...     fields_to_ignore={"functions", "parameter_constraints"})
    """
    return JsonToModelLoader.load_json_to_model(
        json_str,
        module_name,
        fields_to_validate=fields_to_validate,
        fields_to_ignore=fields_to_ignore
    )


def get_available_modules() -> list:
    """
    获取所有可用的模块名列表

    Returns:
        list: 模块名列表
    """
    # 去重（移除别名）
    unique_modules = set()
    seen_classes = set()

    for module_name, model_class in MODULE_MODEL_MAP.items():
        if model_class not in seen_classes:
            unique_modules.add(module_name)
            seen_classes.add(model_class)

    return sorted(list(unique_modules))


def validate_json_schema(json_str: str, module_name: str) -> Tuple[bool, Union[str, None]]:
    """
    验证JSON字符串是否符合指定模块的schema

    Args:
        json_str: JSON字符串
        module_name: 模块名称

    Returns:
        Tuple[bool, Union[str, None]]:
            - (True, None) 如果验证通过
            - (False, 错误信息) 如果验证失败
    """
    model, error = load_json_to_model(json_str, module_name)
    return (model is not None, error)


if __name__ == '__main__':
    dtype_map = """
{
  "dtype_map": [
    {
      "platform": "Atlas A3 训练系列产品 / Atlas A3 推理系列产品",
      "columns": [
        "self",
        "out"
      ],
      "rows": [
        [
          "BFLOAT16",
          "BFLOAT16"
        ],
        [
          "FLOAT16",
          "FLOAT16"
        ],
        [
          "FLOAT32",
          "FLOAT32"
        ]
      ]
    },
    {
      "platform": "Atlas A2 训练系列产品 / Atlas A2 推理系列产品",
      "columns": [
        "self",
        "out"
      ],
      "rows": [
        [
          "BFLOAT16",
          "BFLOAT16"
        ],
        [
          "FLOAT16",
          "FLOAT16"
        ],
        [
          "FLOAT32",
          "FLOAT32"
        ]
      ]
    },
    {
      "platform": "Atlas 推理系列产品",
      "columns": [
        "self",
        "out"
      ],
      "rows": [
        [
          "FLOAT16",
          "FLOAT16"
        ],
        [
          "FLOAT32",
          "FLOAT32"
        ]
      ]
    }
  ]
}
    """

    print("\n" + "=" * 60)
    print("测试3: 忽略某些字段（黑名单模式）")
    print("=" * 60)
    model, error = load_json_to_model(
        dtype_map,
        "basic_info",
        fields_to_ignore={"functions", "parameter_constraints", "other_parameters", "operation_name", "description",
                          "api_flow"
                          "dtype_map", "inter_parameter_constraints", "platform_specifics"}
    )
    if model:
        print(f"成功!")
        print(f"  operation_name: {model.operation_name}")
        print(f"  api_flow: {model.api_flow}")
    else:
        print(f"失败: {error}")
