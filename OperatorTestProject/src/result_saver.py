"""
结果保存模块
负责保存提取结果和错误信息
"""
import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional

from .check_error import CheckError

# 获取logger
logger = logging.getLogger(__name__)


class ResultSaver:
    """结果保存器"""

    @staticmethod
    def get_error_markdown_path(version_dir: Path) -> Path:
        """获取 error.md 路径"""
        return version_dir / "error.md"

    @staticmethod
    def get_error_json_path(version_dir: Path) -> Path:
        """获取 error.json 路径"""
        return version_dir / "error.json"

    @staticmethod
    def read_text_if_exists(file_path: Optional[Path]) -> str:
        """读取文件内容，不存在时返回空字符串"""
        if not file_path or not file_path.exists():
            return ""

        return file_path.read_text(encoding='utf-8').strip()

    @staticmethod
    def _normalize_error_payload(error_msg: str) -> Any:
        """
        将错误信息标准化为可写入 JSON 文件的对象

        Args:
            error_msg: 原始错误信息

        Returns:
            Any: 可直接 json.dump 的对象
        """
        try:
            return json.loads(error_msg)
        except (TypeError, json.JSONDecodeError):
            return {"error": error_msg}

    @staticmethod
    def _normalize_markdown_error_entries(error_payload: Any) -> list[Dict[str, str]]:
        """
        将错误负载标准化为可写入 Markdown 的错误条目列表

        Args:
            error_payload: 已标准化的错误负载

        Returns:
            list[Dict[str, str]]: Markdown 错误条目列表
        """
        default_fix = "请根据错误信息修复后重新执行校验。"
        default_is_fixed = "否"

        def build_entry(item: Any) -> Dict[str, str]:
            if isinstance(item, CheckError):
                return {
                    "error_path": item.error_path,
                    "error_message": item.error_message,
                    "fix_suggestion": item.fix_suggestion,
                    "is_fixed": item.is_fixed,
                }

            if isinstance(item, dict):
                try:
                    check_error = CheckError(
                        error_path=str(item.get("error_path") or "未知路径"),
                        error_message=str(
                            item.get("error_message")
                            or item.get("error")
                            or json.dumps(item, ensure_ascii=False)
                        ),
                        fix_suggestion=str(item.get("fix_suggestion") or default_fix),
                        is_fixed=str(item.get("is_fixed") or default_is_fixed),
                    )
                    return {
                        "error_path": check_error.error_path,
                        "error_message": check_error.error_message,
                        "fix_suggestion": check_error.fix_suggestion,
                        "is_fixed": check_error.is_fixed,
                    }
                except Exception:
                    pass

                return {
                    "error_path": str(item.get("error_path") or "未知路径"),
                    "error_message": str(
                        item.get("error_message")
                        or item.get("error")
                        or json.dumps(item, ensure_ascii=False)
                    ),
                    "fix_suggestion": str(item.get("fix_suggestion") or default_fix),
                    "is_fixed": str(item.get("is_fixed") or default_is_fixed),
                }

            return {
                "error_path": "未知路径",
                "error_message": str(item),
                "fix_suggestion": default_fix,
                "is_fixed": default_is_fixed,
            }

        if isinstance(error_payload, list):
            if not error_payload:
                return [{
                    "error_path": "未知路径",
                    "error_message": "未提供具体错误信息。",
                    "fix_suggestion": default_fix,
                    "is_fixed": default_is_fixed,
                }]
            return [build_entry(item) for item in error_payload]

        return [build_entry(error_payload)]

    @staticmethod
    def _build_error_markdown(error_payload: Any) -> str:
        """
        构建错误 Markdown 内容

        Args:
            error_payload: 已标准化的错误负载

        Returns:
            str: Markdown 文本内容
        """
        entries = ResultSaver._normalize_markdown_error_entries(error_payload)
        lines = []

        for index, entry in enumerate(entries, start=1):
            lines.extend([
                f"错误{index}：",
                f"错误路径:  {entry['error_path']}",
                f"错误信息:  {entry['error_message']}",
                f"修复建议:  {entry['fix_suggestion']}",
                f"是否已修复:  {entry['is_fixed']}",
            ])
            if index < len(entries):
                lines.append("------------------------------")

        return "\n".join(lines) + "\n"

    @staticmethod
    def _parse_error_markdown(markdown_content: str) -> list[Dict[str, str]]:
        """
        解析 error.md 内容为标准错误条目列表

        Args:
            markdown_content: error.md 文本内容

        Returns:
            list[Dict[str, str]]: 错误条目列表
        """
        if not markdown_content.strip():
            return []

        entries: list[Dict[str, str]] = []
        current: Dict[str, str] = {}

        field_mapping = {
            "错误路径": "error_path",
            "错误信息": "error_message",
            "修复建议": "fix_suggestion",
            "是否已修复": "is_fixed",
        }

        for raw_line in markdown_content.splitlines():
            line = raw_line.strip()
            if not line:
                continue

            if line.startswith("错误") and line.endswith("："):
                if current:
                    entries.append({
                        "error_path": current.get("error_path", "未知路径"),
                        "error_message": current.get("error_message", "未提供具体错误信息。"),
                        "fix_suggestion": current.get("fix_suggestion", "请根据错误信息修复后重新执行校验。"),
                        "is_fixed": current.get("is_fixed", "否"),
                    })
                    current = {}
                continue

            if line == "------------------------------":
                continue

            for field_label, field_name in field_mapping.items():
                prefix = f"{field_label}:"
                if line.startswith(prefix):
                    current[field_name] = line[len(prefix):].strip()
                    break

        if current:
            entries.append({
                "error_path": current.get("error_path", "未知路径"),
                "error_message": current.get("error_message", "未提供具体错误信息。"),
                "fix_suggestion": current.get("fix_suggestion", "请根据错误信息修复后重新执行校验。"),
                "is_fixed": current.get("is_fixed", "否"),
            })

        return entries

    @staticmethod
    def _load_previous_error_entries(previous_error_path: Optional[Path]) -> list[Dict[str, str]]:
        """读取上一版本 error.md 中的错误条目"""
        markdown_content = ResultSaver.read_text_if_exists(previous_error_path)
        if not markdown_content:
            return []

        return ResultSaver._parse_error_markdown(markdown_content)

    @staticmethod
    def _pop_matching_entry(
            previous_entry: Dict[str, str],
            current_entries: list[Dict[str, str]]
    ) -> Optional[Dict[str, str]]:
        """
        从当前错误列表中找到与历史错误最匹配的一条并移除

        优先按 error_path + error_message 精确匹配，失败后退化为仅按 error_path 匹配。
        """
        previous_path = previous_entry.get("error_path", "").strip()
        previous_message = previous_entry.get("error_message", "").strip()

        for index, current_entry in enumerate(current_entries):
            if (
                    current_entry.get("error_path", "").strip() == previous_path
                    and current_entry.get("error_message", "").strip() == previous_message
            ):
                return current_entries.pop(index)

        for index, current_entry in enumerate(current_entries):
            if current_entry.get("error_path", "").strip() == previous_path:
                return current_entries.pop(index)

        return None

    @staticmethod
    def _merge_error_entries(
            previous_entries: list[Dict[str, str]],
            current_payload: Any
    ) -> list[Dict[str, str]]:
        """
        合并上一版本错误和本次错误

        - 当前仍存在的错误保留为“否”
        - 上一版本存在但本次已消失的错误标记为“是”
        - 新出现的错误追加为“否”
        """
        current_entries = ResultSaver._normalize_markdown_error_entries(current_payload)
        remaining_current_entries = [dict(entry) for entry in current_entries]
        merged_entries: list[Dict[str, str]] = []

        for previous_entry in previous_entries:
            matched_entry = ResultSaver._pop_matching_entry(previous_entry, remaining_current_entries)
            if matched_entry:
                matched_entry["is_fixed"] = "否"
                merged_entries.append(matched_entry)
            else:
                resolved_entry = dict(previous_entry)
                resolved_entry["is_fixed"] = "是"
                merged_entries.append(resolved_entry)

        for current_entry in remaining_current_entries:
            current_entry["is_fixed"] = "否"
            merged_entries.append(current_entry)

        return merged_entries

    @staticmethod
    def _save_error_markdown(
            version_dir: Path,
            error_payload: Any,
            module: str
    ) -> Path:
        """
        保存错误 Markdown 文件

        Args:
            version_dir: 版本目录
            error_payload: 已标准化的错误负载
            module: 模块名称

        Returns:
            Path: 保存的 Markdown 文件路径
        """
        markdown_file = version_dir / "error.md"
        markdown_content = ResultSaver._build_error_markdown(error_payload)

        with open(markdown_file, 'w', encoding='utf-8') as f:
            f.write(markdown_content)

        logger.debug(f"[{module}] 保存错误信息到 Markdown 文件: {markdown_file}")
        return markdown_file

    @staticmethod
    def save_result(
            version_dir: Path,
            module: str,
            data: Dict[str, Any]
    ) -> Path:
        """
        保存提取结果

        Args:
            version_dir: 版本目录
            module: 模块名称
            data: 提取的数据

        Returns:
            Path: 保存的文件路径
        """
        result_file = version_dir / f"{module}.json"
        with open(result_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        logger.debug(f"[{module}] 保存提取结果到文件: {result_file}")
        return result_file

    @staticmethod
    def save_error(
            version_dir: Path,
            error_msg: str,
            module: str,
            previous_error_path: Optional[Path] = None
    ) -> Path:
        """
        保存错误信息到 error.json

        Args:
            version_dir: 版本目录
            error_msg: 错误信息（LLM响应）
            module: 模块名称
            previous_error_path: 上一版本的error.json文件路径（未使用）

        Returns:
            Path: 保存的文件路径
        """
        error_file = ResultSaver.get_error_json_path(version_dir)
        markdown_file = ResultSaver.get_error_markdown_path(version_dir)
        error_payload = ResultSaver._normalize_error_payload(error_msg)
        previous_entries = ResultSaver._load_previous_error_entries(previous_error_path)
        merged_entries = ResultSaver._merge_error_entries(previous_entries, error_payload)

        with open(error_file, 'w', encoding='utf-8') as f:
            json.dump(merged_entries, f, ensure_ascii=False, indent=2)

        markdown_content = ResultSaver._build_error_markdown(merged_entries)
        with open(markdown_file, 'w', encoding='utf-8') as f:
            f.write(markdown_content)
        
        logger.debug(f"[{module}] 保存错误信息到文件: {error_file}")
        return error_file

    @staticmethod
    def read_rule_json(
            version_dir: Path,
            module: str
    ) -> str:
        """
        读取已生成的规则文件内容

        Args:
            version_dir: 版本目录
            module: 模块名称

        Returns:
            str: 规则文件内容
        """
        rule_json_path = version_dir / f"{module}.json"
        if rule_json_path.exists():
            with open(rule_json_path, 'r', encoding='utf-8') as f:
                return f.read()
        return ""
