"""
自定义异常模块
定义系统中使用的所有自定义异常类
"""


class OperatorProcessingError(Exception):
    """算子处理基础异常"""
    pass


class MaxIterationExceededError(OperatorProcessingError):
    """超过最大迭代次数异常"""
    def __init__(self, module: str, max_iterations: int):
        self.module = module
        self.max_iterations = max_iterations
        super().__init__(f"模块 {module} 超过最大迭代次数 {max_iterations}")


class RuleFileNotFoundError(OperatorProcessingError):
    """规则文件未找到异常"""
    def __init__(self, rule_file: str):
        self.rule_file = rule_file
        super().__init__(f"规则文件未找到: {rule_file}")


class ValidationError(OperatorProcessingError):
    """校验失败异常"""
    def __init__(self, module: str, error_details: str):
        self.module = module
        self.error_details = error_details
        super().__init__(f"模块 {module} 校验失败: {error_details}")


class LLMInvocationError(OperatorProcessingError):
    """LLM调用异常"""
    def __init__(self, message: str, original_error: Exception = None, is_token_limit_error: bool = False):
        self.original_error = original_error
        self.is_token_limit_error = is_token_limit_error
        super().__init__(message)


class ConfigError(OperatorProcessingError):
    """配置错误异常"""
    pass


class ModuleProcessingError(OperatorProcessingError):
    """模块处理异常"""
    def __init__(self, module: str, message: str):
        self.module = module
        super().__init__(f"模块 {module} 处理失败: {message}")


class SkillExecutionError(OperatorProcessingError):
    """Skill执行异常"""
    def __init__(self, skill_name: str, message: str):
        self.skill_name = skill_name
        super().__init__(f"Skill {skill_name} 执行失败: {message}")
