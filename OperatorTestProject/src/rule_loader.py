"""
规则加载模块
负责加载规则文件
"""
import logging
from pathlib import Path
from typing import Optional

from .exceptions import RuleFileNotFoundError

# 获取logger
logger = logging.getLogger(__name__)


class RuleLoader:
    """规则加载器"""

    def __init__(self, rules_dir: Path):
        """
        初始化规则加载器

        Args:
            rules_dir: 规则文件目录
        """
        self.rules_dir = rules_dir

    def load_rule_file(
            self,
            module: str,
            is_check: bool = False,
            previous_error_path: Optional[Path] = None
    ) -> str:
        """
        加载规则文件（仅基础规则，不从 error.json 提取修复建议）

        Args:
            module: 模块名称
            is_check: 是否为校验规则
            previous_error_path: 上一版本的 error.json 文件路径（未使用）

        Returns:
            str: 规则文件内容
        """
        suffix = "_check_guide.md" if is_check else "_extracting_guide.md"
        rule_file = self.rules_dir / f"{module}{suffix}"

        if not rule_file.exists():
            raise RuleFileNotFoundError(str(rule_file))

        # 👇 只读取基础规则，不从 error.json 提取修复建议
        with open(rule_file, 'r', encoding='utf-8') as f:
            base_rule_content = f.read()
        
        logger.debug(f"[{module}] 已加载基础规则")
        return base_rule_content
