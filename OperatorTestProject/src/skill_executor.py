"""
Skill执行器模块
负责调用各种Skill和规则备份功能
"""
import json
import logging
import shutil
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List

from .config_loader import ConfigLoader
from .exceptions import SkillExecutionError
from .path_utils import resolve_path

# 获取logger
logger = logging.getLogger(__name__)


class SkillExecutor:
    """Skill执行器"""

    @staticmethod
    def _derive_operator_name_from_rule_path(operator_rule_path: Path) -> str:
        """从规则文件路径推导原始算子名"""
        operator_name = operator_rule_path.stem.replace("_rule", "")
        extracted_constraints_suffix = "_extracted_constraints"
        if operator_name.endswith(extracted_constraints_suffix):
            operator_name = operator_name[:-len(extracted_constraints_suffix)]
        return operator_name

    def __init__(self, config_loader: ConfigLoader):
        """
        初始化Skill执行器
        
        Args:
            config_loader: 配置加载器实例
        """
        self.config_loader = config_loader
        self.paths_config = config_loader.get_paths_config()
        self.skill_dir = resolve_path(self.paths_config.skill_dir)
        self.backup_dir = resolve_path(self.paths_config.backup_dir)

    def _backup_rule_file(self, rule_file_path: Path) -> Path:
        """
        备份规则文件
        
        Args:
            rule_file_path: 规则文件路径
            
        Returns:
            Path: 备份文件路径
        """
        # 创建备份目录(按日期)
        date_str = datetime.now().strftime("%Y%m%d")
        backup_subdir = self.backup_dir / date_str
        backup_subdir.mkdir(parents=True, exist_ok=True)

        # 生成备份文件名(带时间戳)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        rule_name = rule_file_path.stem
        backup_file_name = f"{rule_name}_{timestamp}.md"
        backup_file_path = backup_subdir / backup_file_name

        # 复制文件
        shutil.copy2(rule_file_path, backup_file_path)

        return backup_file_path

    async def merge_operator_rules(
            self,
            operator_name: str,
            module_json_files: Dict[str, Path]
    ) -> Dict[str, Any]:
        """
        执行规则合并skill
        
        Args:
            operator_name: 算子名称
            module_json_files: 各模块JSON文件路径字典
            
        Returns:
            Dict[str, Any]: 合并后的规则数据
        """
        # 读取所有模块的JSON文件
        merged_data = {}
        for module, json_path in module_json_files.items():
            if not json_path.exists():
                raise SkillExecutionError(
                    "merge-operator-rules",
                    f"模块JSON文件不存在: {json_path}"
                )

            logger.debug(f"读取模块JSON文件: {json_path}")
            with open(json_path, 'r', encoding='utf-8') as f:
                module_data = json.load(f)
                merged_data.update(module_data)

        return merged_data

    async def generate_test_cases(
            self,
            operator_rule_path: Path,
            count: int
    ) -> Path:
        """
        执行测试用例生成

        Args:
            operator_rule_path: 算子规则文件路径
            count: 生成的测试用例数量

        Returns:
            Path: 测试用例文件保存路径
        """
        # 检查算子规则文件是否存在
        if not operator_rule_path.exists():
            raise SkillExecutionError(
                "test-case-generator",
                f"算子规则文件不存在: {operator_rule_path}"
            )

        logger.debug(f"开始生成测试用例,规则文件: {operator_rule_path}")
        logger.debug(f"测试用例数量: {count}")

        try:
            # 导入测试用例生成器
            from .test_case_generator import generate_test_cases_from_file

            # 构建输出路径
            operator_name = self._derive_operator_name_from_rule_path(operator_rule_path)
            workspace_dir = resolve_path(self.paths_config.workspace_dir)
            output_path = workspace_dir / operator_name / f"{operator_name}_cases.json"

            # 调用生成函数
            test_cases = generate_test_cases_from_file(
                config_path=str(operator_rule_path),
                output_path=str(output_path),
                count=count
            )

            logger.debug(f"成功生成 {len(test_cases)} 个测试用例")
            logger.debug(f"测试用例已保存到: {output_path}")

            return output_path

        except FileNotFoundError as e:
            logger.error(f"规则文件不存在: {e}")
            raise SkillExecutionError("test-case-generator", str(e))
        except ValueError as e:
            logger.error(f"规则文件内容无效: {e}")
            raise SkillExecutionError("test-case-generator", str(e))
        except Exception as e:
            logger.error(f"测试用例生成失败: {e}")
            raise SkillExecutionError("test-case-generator", f"生成失败: {e}")

    def save_merged_result(
            self,
            operator_name: str,
            merged_data: Dict[str, Any]
    ) -> Path:
        """
        保存合并结果（规则文件）

        Args:
            operator_name: 算子名称
            merged_data: 合并后的数据

        Returns:
            Path: 保存的文件路径
        """
        workspace_dir = resolve_path(self.paths_config.workspace_dir)
        output_file = workspace_dir / operator_name / f"{operator_name}_extracted_constraints.json"
        output_file.parent.mkdir(parents=True, exist_ok=True)

        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(merged_data, f, ensure_ascii=False, indent=2)

        logger.debug(f"保存合并结果到文件: {output_file}")
        return output_file

    def save_test_cases(
            self,
            operator_name: str,
            test_cases: List[Dict[str, Any]]
    ) -> Path:
        """
        保存测试用例
        
        Args:
            operator_name: 算子名称
            test_cases: 测试用例列表
            
        Returns:
            Path: 保存的文件路径
        """
        workspace_dir = resolve_path(self.paths_config.workspace_dir)
        output_file = workspace_dir / operator_name / f"{operator_name}_cases.json"
        output_file.parent.mkdir(parents=True, exist_ok=True)

        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(test_cases, f, ensure_ascii=False, indent=2)

        logger.debug(f"保存测试用例到文件: {output_file}")
        return output_file
