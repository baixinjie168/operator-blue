"""
配置加载器模块
负责加载和验证系统配置文件
"""
from typing import Any, Dict, List, Optional

import yaml
from pydantic import BaseModel, Field, field_validator, model_validator

from .path_utils import resolve_path


class LLMProviderConfig(BaseModel):
    """LLM provider 配置"""

    name: str
    type: str = Field(default="cli", description="provider接入方式: cli 或 interface")
    provider: str = Field(default="codex", description="provider名称: codex/openai/anthropic/claude")
    base_url: Optional[str] = None
    api_key: str = ""
    model: str = ""
    command: Optional[str] = None
    args: List[str] = Field(default_factory=list)
    env: Dict[str, str] = Field(default_factory=dict)
    headers: Dict[str, str] = Field(default_factory=dict)
    api_path: Optional[str] = None
    options: Dict[str, Any] = Field(default_factory=dict)

    @field_validator("type")
    @classmethod
    def validate_type(cls, value: str) -> str:
        """标准化 provider 接入方式"""
        normalized = value.strip().lower()
        if normalized not in {"cli", "interface"}:
            raise ValueError('type must be either "cli" or "interface"')
        return normalized

    @field_validator("provider")
    @classmethod
    def normalize_provider(cls, value: str) -> str:
        """标准化 provider 名称"""
        normalized = value.strip().lower()
        if not normalized:
            raise ValueError("provider must not be empty")
        return normalized

    @field_validator("base_url", "command", "api_path")
    @classmethod
    def normalize_optional_text(cls, value: Optional[str]) -> Optional[str]:
        """清理可选文本配置"""
        if value is None:
            return None
        normalized = value.strip()
        return normalized or None

    @model_validator(mode="after")
    def apply_defaults(self) -> "LLMProviderConfig":
        """补齐 provider 默认值"""
        if self.type == "cli" and not self.command:
            self.command = self.provider

        if self.type == "interface" and not self.base_url:
            raise ValueError("interface 类型的 provider 必须配置 base_url")

        return self


class LLMInterfaceConfig(LLMProviderConfig):
    """兼容旧命名的 LLM provider 配置"""


class LLMParams(BaseModel):
    """LLM参数配置"""
    temperature: float = Field(default=0.7, ge=0, le=1)
    max_tokens: int = Field(default=1024, gt=0)
    timeout: int = Field(default=600, gt=0, description="LLM请求超时时间(秒)")


class IterationConfig(BaseModel):
    """迭代配置"""
    max_iterations: int = Field(default=5, gt=0)


class TestCaseConfig(BaseModel):
    """测试用例生成配置"""
    count: int = Field(default=10, gt=0)


class ThreadPoolConfig(BaseModel):
    """线程池配置"""
    size: str = Field(default="auto")


class ModuleExecutionConfig(BaseModel):
    """模块执行配置"""
    mode: str = Field(default="sequential", description="执行模式: sequential(顺序) 或 concurrent(并发)")

    @field_validator('mode')
    @classmethod
    def validate_mode(cls, v):
        """验证执行模式"""
        if v not in ['sequential', 'concurrent']:
            raise ValueError('mode must be either "sequential" or "concurrent"')
        return v


class LoggingConfig(BaseModel):
    """日志配置"""
    level: str = Field(default="INFO")
    format: str = Field(default="%(asctime)s - %(name)s - %(levelname)s - %(message)s")


class PathsConfig(BaseModel):
    """路径配置"""
    rules_dir: str
    workspace_dir: str
    skill_dir: str
    backup_dir: str


class Config(BaseModel):
    """顶层配置模型"""
    llm_interfaces: List[LLMProviderConfig]
    llm_params: LLMParams
    iteration: IterationConfig
    test_case_generator: TestCaseConfig
    thread_pool: ThreadPoolConfig
    module_execution: ModuleExecutionConfig
    logging: LoggingConfig
    paths: PathsConfig

    @model_validator(mode="after")
    def validate_llm_interfaces(self) -> "Config":
        """校验至少存在一个 provider"""
        if len(self.llm_interfaces) == 0:
            raise ValueError("至少需要配置一个LLM provider")
        return self


class ConfigLoader:
    """配置加载器"""

    def __init__(self, config_path: str = "config/config.yaml"):
        """
        初始化配置加载器

        Args:
            config_path: 配置文件路径（相对于应用程序根目录）
        """
        # 将相对路径转换为绝对路径
        self.config_path = resolve_path(config_path)
        self._config: Optional[Config] = None
    
    def load_config(self) -> Config:
        """
        加载配置文件
        
        Returns:
            Config: 配置对象
            
        Raises:
            FileNotFoundError: 配置文件不存在
            ValueError: 配置文件格式错误
        """
        if not self.config_path.exists():
            raise FileNotFoundError(f"配置文件不存在: {self.config_path}")
        
        with open(self.config_path, 'r', encoding='utf-8') as f:
            config_dict = yaml.safe_load(f)
        
        try:
            self._config = Config(**config_dict)
            return self._config
        except Exception as e:
            raise ValueError(f"配置文件格式错误: {e}")
    
    def get_config(self) -> Config:
        """
        获取配置对象
        
        Returns:
            Config: 配置对象
        """
        if self._config is None:
            self.load_config()
        return self._config
    
    def get_llm_interfaces(self) -> List[LLMProviderConfig]:
        """获取LLM provider 配置列表（兼容旧命名）"""
        return self.get_config().llm_interfaces

    def get_llm_providers(self) -> List[LLMProviderConfig]:
        """获取LLM provider 配置列表"""
        return self.get_config().llm_interfaces
    
    def get_llm_params(self) -> LLMParams:
        """获取LLM参数配置"""
        return self.get_config().llm_params
    
    def get_iteration_config(self) -> IterationConfig:
        """获取迭代配置"""
        return self.get_config().iteration
    
    def get_test_case_config(self) -> TestCaseConfig:
        """获取测试用例生成配置"""
        return self.get_config().test_case_generator
    
    def get_thread_pool_config(self) -> ThreadPoolConfig:
        """获取线程池配置"""
        return self.get_config().thread_pool

    def get_module_execution_config(self) -> ModuleExecutionConfig:
        """获取模块执行配置"""
        return self.get_config().module_execution
    
    def get_logging_config(self) -> LoggingConfig:
        """获取日志配置"""
        return self.get_config().logging
    
    def get_paths_config(self) -> PathsConfig:
        """获取路径配置"""
        return self.get_config().paths

    def validate_config(self) -> bool:
        """
        验证配置有效性

        Returns:
            bool: 配置是否有效
        """
        try:
            config = self.get_config()

            # 验证LLM接口数量
            if len(config.llm_interfaces) == 0:
                raise ValueError("至少需要配置一个LLM provider")

            # 验证路径存在性（将相对路径转换为绝对路径）
            paths = config.paths
            for path_name, path_value in [
                ("rules_dir", paths.rules_dir),
                ("workspace_dir", paths.workspace_dir),
                ("skill_dir", paths.skill_dir),
            ]:
                # 将配置中的相对路径转换为绝对路径
                abs_path = resolve_path(path_value)
                if not abs_path.exists():
                    print(f"警告: {path_name} 路径不存在: {abs_path}")

            return True
        except Exception as e:
            print(f"配置验证失败: {e}")
            return False

    def get_absolute_paths_config(self) -> dict:
        """
        获取转换为绝对路径的路径配置

        Returns:
            dict: 包含绝对路径的字典
        """
        paths = self.get_paths_config()
        return {
            "rules_dir": str(resolve_path(paths.rules_dir)),
            "workspace_dir": str(resolve_path(paths.workspace_dir)),
            "skill_dir": str(resolve_path(paths.skill_dir)),
            "backup_dir": str(resolve_path(paths.backup_dir)),
        }
