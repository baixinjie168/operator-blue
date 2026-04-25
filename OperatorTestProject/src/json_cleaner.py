"""JSON 字符串清理工具"""

import json
import re
import logging

logger = logging.getLogger(__name__)


def clean_json_string(json_str: str) -> str:
    """
    清理 JSON 字符串中的常见问题

    Args:
        json_str: 可能包含问题的 JSON 字符串

    Returns:
        清理后的 JSON 字符串
    """
    if not json_str or not isinstance(json_str, str):
        return json_str

    cleaned = json_str

    # 1. 修复无效的转义字符
    cleaned = fix_invalid_escapes(cleaned)

    # 2. 修复常见的 JSON 格式问题
    cleaned = fix_common_json_issues(cleaned)

    # 3. 移除多余的逗号
    cleaned = remove_trailing_commas(cleaned)

    return cleaned


def fix_invalid_escapes(json_str: str) -> str:
    """
    修复 JSON 字符串中的无效转义字符

    常见问题：
    - 单独的 \ 应该是 \\
    - 字符串外的 \ 应该移除
    """
    # 使用正则表达式找到所有字符串内容
    string_pattern = r'"(?:[^"\\]|\\.)*"'

    def fix_string_escapes(match):
        """修复字符串中的转义字符"""
        string_content = match.group(0)

        # 检查字符串内部是否有无效转义
        # 移除不在有效转义序列前的单独反斜杠
        valid_escapes = {'\\', '"', 'n', 't', 'r', 'b', 'f', '/', 'u'}

        result = []
        i = 0
        while i < len(string_content):
            if string_content[i] == '\\':
                if i + 1 < len(string_content):
                    next_char = string_content[i + 1]
                    if next_char in valid_escapes:
                        result.append('\\' + next_char)
                        i += 2
                    else:
                        # 无效转义，移除反斜杠
                        result.append(next_char)
                        i += 2
                else:
                    # 字符串末尾的反斜杠，移除
                    i += 1
            else:
                result.append(string_content[i])
                i += 1

        return '"' + ''.join(result[1:-1]) + '"'  # 重新构建字符串

    try:
        # 只处理 JSON 字符串内部的内容
        cleaned = re.sub(string_pattern, fix_string_escapes, json_str)
        return cleaned
    except Exception as e:
        logger.debug(f"修复转义字符时出错: {e}")
        return json_str


def fix_common_json_issues(json_str: str) -> str:
    """修复常见的 JSON 格式问题"""
    cleaned = json_str.strip()

    # 移除 BOM 标记
    if cleaned.startswith('﻿'):
        cleaned = cleaned[1:]

    # 修复未引用的键名
    # 例如: {name: "value"} -> {"name": "value"}
    cleaned = re.sub(r'([{,]\s*)([a-zA-Z_][a-zA-Z0-9_]*)(\s*:)', r'\1"\2"\3', cleaned)

    # 修复单引号字符串
    # 例如: {'key': 'value'} -> {"key": "value"}
    cleaned = cleaned.replace("'", '"')

    return cleaned


def remove_trailing_commas(json_str: str) -> str:
    """移除 JSON 中的尾随逗号"""
    # 移除对象中的尾随逗号: {, "key": "value",} -> {"key": "value"}
    cleaned = re.sub(r',(\s*[}\]])', r'\1', json_str)
    return cleaned


def parse_json_safely(json_str: str, module_name: str = "") -> tuple:
    """
    安全地解析 JSON，尝试多种修复策略

    Args:
        json_str: 要解析的 JSON 字符串
        module_name: 模块名称（用于日志）

    Returns:
        (解析成功, 数据或错误信息)
    """
    if not json_str or not isinstance(json_str, str):
        return False, "空或无效的 JSON 字符串"

    # 尝试直接解析
    try:
        data = json.loads(json_str)
        return True, data
    except json.JSONDecodeError as e:
        logger.debug(f"[{module_name}] 直接 JSON 解析失败: {e}")

    # 尝试清理后解析
    try:
        cleaned_json = clean_json_string(json_str)
        data = json.loads(cleaned_json)
        logger.info(f"[{module_name}] JSON 清理后解析成功")
        return True, data
    except json.JSONDecodeError as e:
        logger.debug(f"[{module_name}] 清理后 JSON 解析仍然失败: {e}")

    # 尝试提取 JSON 代码块（如果存在）
    try:
        extracted_json = extract_json_from_code_blocks(json_str)
        data = json.loads(extracted_json)
        logger.info(f"[{module_name}] 从代码块提取 JSON 并解析成功")
        return True, data
    except json.JSONDecodeError as e:
        logger.debug(f"[{module_name}] 代码块 JSON 解析失败: {e}")

    # 最后尝试：使用更激进的清理
    try:
        aggressively_cleaned = aggressive_json_clean(json_str)
        data = json.loads(aggressively_cleaned)
        logger.info(f"[{module_name}] 激进清理后 JSON 解析成功")
        return True, data
    except json.JSONDecodeError as e:
        error_msg = f"JSON解析失败: {str(e)}\n位置: 第{e.lineno}行, 第{e.colno}列"
        logger.error(f"[{module_name}] {error_msg}")
        return False, error_msg


def extract_json_from_code_blocks(text: str) -> str:
    """从文本中提取 JSON 代码块"""
    import re

    # 匹配 ```json...``` 或 ```...``` 代码块
    json_block_pattern = r'```(?:json)?\s*\n?(.*?)\n?```'
    matches = re.findall(json_block_pattern, text, re.DOTALL | re.IGNORECASE)

    if matches:
        return matches[0].strip()

    return text


def aggressive_json_clean(json_str: str) -> str:
    """激进的 JSON 清理策略"""
    cleaned = json_str.strip()

    # 移除所有注释
    cleaned = re.sub(r'//.*?\n', '\n', cleaned)
    cleaned = re.sub(r'/\*.*?\*/', '', cleaned, flags=re.DOTALL)

    # 移除控制字符（除了换行、制表符等）
    cleaned = re.sub(r'[\x00-\x08\x0b-\x0c\x0e-\x1f\x7f-\x9f]', '', cleaned)

    # 尝试修复 Unicode 转义序列
    cleaned = re.sub(r'\\u([0-9a-fA-F]{4})', lambda m: chr(int(m.group(1), 16)), cleaned)

    return cleaned


def validate_and_clean_json(json_str: str, module_name: str = "") -> tuple:
    """
    验证并清理 JSON 字符串

    Args:
        json_str: 要验证和清理的 JSON 字符串
        module_name: 模块名称

    Returns:
        (is_valid, cleaned_json_or_error)
    """
    success, result = parse_json_safely(json_str, module_name)

    if success:
        # 重新序列化以确保格式正确
        try:
            cleaned = json.dumps(result, ensure_ascii=False, indent=2)
            return True, cleaned
        except Exception as e:
            logger.error(f"[{module_name}] 重新序列化 JSON 失败: {e}")
            return False, str(e)
    else:
        return False, result
