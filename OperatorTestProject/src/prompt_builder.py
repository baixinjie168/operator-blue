"""
提示词构建模块
负责构建各种LLM提示词
"""
import json
import logging
from typing import Dict, Any, Optional

# 获取logger
logger = logging.getLogger(__name__)


class PromptBuilder:
    """提示词构建器"""

    @staticmethod
    def build_extraction_prompt(
            operator_doc: str,
            rule_content: str,
            module: str = "",
            json_validation_error: Optional[str] = None,
            validation_error: Optional[str] = None
    ) -> str:
        """
        构建提取提示词（使用基础规则 + 扩展规则）

        Args:
            operator_doc: 原始算子文档
            rule_content: 规则内容（基础规则 + 扩展规则）
            module: 模块名称（用于构建错误提示）
            json_validation_error: JSON校验错误信息（可选）
            validation_error: 业务校验错误信息（可选）

        Returns:
            str: 提取提示词
        """

        # 构建错误提示
        json_error_title = ",【JSON解析错误】" if json_validation_error else ""
        json_error_info = PromptBuilder.build_json_error_info(module=module,
                                                              json_validation_error=json_validation_error)

        content_error_title = ",【上一版 error.md】" if validation_error else ""
        business_error_info = PromptBuilder.build_business_error_info(module=module,
                                                                      validation_error=validation_error)

        # 构建基础提示词
        prompt = f"""
你是一个"CANN算子文档结构化解析专家"。

你的任务是：
根据【算子文档】,【提取规则】{json_error_title} {content_error_title}提取结构化信息，并输出标准JSON。
----------------------------------------------------------------------------------------------------
【算子文档】
``````markdown
{operator_doc}
``````

----------------------------------------------------------------------------------------------------
【提取规则】
``````markdown
{rule_content}
``````
{json_error_info}

{business_error_info}

请以JSON格式输出提取结果,不要包含任何其他说明文字。
"""
        return prompt

    @staticmethod
    def build_json_error_info(module: str, json_validation_error: Optional[str]) -> str:
        json_error_info = ""

        # JSON校验错误
        if json_validation_error:
            logger.info(f"[{module}] 检测到JSON校验错误，将错误信息添加到提示词")
            json_error_info += f"""
----------------------------------------------------------------------------------------------------
【JSON解析错误】
``````markdown
{json_validation_error}
``````
"""
        return json_error_info

    @staticmethod
    def build_business_error_info(
            module: str,
            validation_error: Optional[str]
    ) -> str:
        """
        构建错误提示信息

        Args:
            module: 模块名称
            json_validation_error: JSON校验错误信息
            validation_error: 业务校验错误信息

        Returns:
            str: 错误提示信息
        """
        business_error_info = ""

        # 业务校验错误
        if validation_error:
            logger.info(f"[{module}] 检测到业务校验错误，将错误信息添加到提示词")
            business_error_info += f"""
----------------------------------------------------------------------------------------------------
【上一版 error.md】
``````markdown
    {validation_error}
``````
    """

        return business_error_info

    @staticmethod
    def build_validation_prompt(
            operator_doc: str,
            check_rule_content: str,
            module: str,
            rule_json_content: str = ""
    ) -> str:
        """
        构建校验提示词

        Args:
            extracted_data: 提取的数据
            check_rule_content: 校验规则内容
            module: 模块名称
            rule_json_content: 已生成的规则文件内容（用于定位错误行号）

        Returns:
            str: 校验提示词
        """
        prompt = f"""
你是一个“CANN算子结构化数据校验专家”。

你的任务是：
根据【原始算子文档】、【校验规则】和【结构化JSON数据】，找出所有不符合规则或与文档不一致的地方，并输出错误报告。
说明：【结构化JSON数据】是从【原始算子文档】中提取得到的
--------------------------------------------------
【原始算子文档】
``````markdown
{operator_doc}
``````
--------------------------------------------------
【结构化JSON数据】
``````markdown
```json
{rule_json_content}
```
``````
--------------------------------------------------
【校验规则】
``````markdown
{check_rule_content}
``````
--------------------------------------------------
请验证数据是否符合规则要求，并输出CheckError对象的数组。

CheckError类定义如下:
```python
class CheckError(BaseModel):
    error_path: str = Field(..., description="错误位置的JSON路径")
    error_message: str = Field(..., description="错误信息")
    fix_suggestion: str = Field(..., description="修复建议")
    is_fixed: str = Field(..., description="是否已修复")
```

输出规则:
1. 如果校验通过，没有错误，输出空数组: []
2. 如果校验失败，有错误，输出包含CheckError的数组

字段说明:
- error_path: 错误在JSON中的路径，例如: "function[0].type" 或 "input[2].shape[0]"
- error_message: 校验的错误信息描述
- fix_suggestion: 针对该错误的修复建议
- is_fixed: 这个错误是否已修复(初始值为"否")，已修复用 "是"表示，未修复用"否"表示

注意：必须输出合法的JSON数组格式，不要包含任何额外的说明文字。
"""
        return prompt
