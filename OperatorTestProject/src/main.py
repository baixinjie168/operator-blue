"""
主程序模块
负责程序入口和整体流程控制
"""
import asyncio
import logging
import os
import sys
from pathlib import Path

from .config_loader import ConfigLoader
from .llm_service import LLMService
from .module_processor import ModuleProcessor
from .skill_executor import SkillExecutor
from .graph import OperatorProcessingGraph
from .exceptions import OperatorProcessingError
from .path_utils import resolve_path, ensure_dir

PROCESS_LOG_FILE_ENV = "OPERATOR_PROCESS_LOG_FILE"
DISABLE_STDOUT_LOGGING_ENV = "OPERATOR_DISABLE_STDOUT_LOGGING"


def configure_console_encoding() -> None:
    """尽量将控制台输出编码切换为 UTF-8，避免中文帮助信息输出失败"""
    for stream_name in ("stdout", "stderr"):
        stream = getattr(sys, stream_name, None)
        if stream is None or not hasattr(stream, "reconfigure"):
            continue
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except ValueError:
            continue


def _resolve_log_path() -> Path:
    """解析当前进程应使用的日志文件路径"""
    configured_log_path = os.environ.get(PROCESS_LOG_FILE_ENV)
    if configured_log_path:
        return resolve_path(configured_log_path)
    return resolve_path("logs/operator_processor.log")


def setup_logging(config_loader: ConfigLoader) -> None:
    """配置日志"""
    logging_config = config_loader.get_logging_config()

    # 自定义日志格式
    custom_format = "%(asctime)s - %(message)s"

    # 确保日志目录存在
    log_path = _resolve_log_path()
    ensure_dir(log_path.parent)

    handlers: list[logging.Handler] = [
        logging.FileHandler(str(log_path), encoding='utf-8')
    ]
    if os.environ.get(DISABLE_STDOUT_LOGGING_ENV) != "1":
        handlers.append(logging.StreamHandler(sys.stdout))

    logging.basicConfig(
        level=getattr(logging, logging_config.level),
        format=custom_format,
        handlers=handlers,
        force=True,
    )

    # 配置第三方库的日志级别
    # httpx的HTTP请求日志设置为WARNING,避免大量INFO日志
    logging.getLogger("httpx").setLevel(logging.WARNING)


async def main(operator_doc_path: str) -> None:
    """
    主函数

    Args:
        operator_doc_path: 算子文档路径
    """
    logger = logging.getLogger(__name__)

    try:
        # 加载配置并初始化
        config_loader = load_and_validate_config()

        # 初始化服务
        llm_service, skill_executor, module_processor = init_services(config_loader)

        # 创建工作流
        graph = create_workflow(config_loader, llm_service, module_processor, skill_executor)

        # 读取算子文档
        operator_name, operator_doc = load_operator_document(operator_doc_path)

        # 执行工作流
        final_state = await execute_workflow(graph, operator_name, operator_doc)

        # 处理结果
        process_results(final_state)

    except OperatorProcessingError as e:
        logger.error(f"算子处理错误: {e}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"未知错误: {e}", exc_info=True)
        sys.exit(1)


def load_and_validate_config() -> ConfigLoader:
    """加载并验证配置"""
    logger = logging.getLogger(__name__)
    logger.info("加载配置文件...")
    config_loader = ConfigLoader()
    config_loader.validate_config()
    setup_logging(config_loader)
    return config_loader


def init_services(config_loader: ConfigLoader):
    """
    初始化所有服务

    Returns:
        (llm_service, skill_executor, module_processor)
    """
    logger = logging.getLogger(__name__)

    logger.info("初始化LLM服务...")
    llm_service = LLMService(config_loader)

    logger.debug("初始化Skill执行器...")
    skill_executor = SkillExecutor(config_loader)

    logger.debug("初始化模块处理器...")
    module_processor = ModuleProcessor(
        config_loader,
        llm_service,
        skill_executor
    )

    return llm_service, skill_executor, module_processor


def create_workflow(config_loader, llm_service, module_processor, skill_executor):
    """创建算子处理工作流"""
    logger = logging.getLogger(__name__)
    logger.info("创建工作流...")
    graph = OperatorProcessingGraph(
        config_loader,
        llm_service,
        module_processor,
        skill_executor
    )
    return graph


def load_operator_document(operator_doc_path: str):
    """
    加载算子文档

    Returns:
        (operator_name, operator_doc)
    """
    logger = logging.getLogger(__name__)
    logger.info(f"读取算子文档: {operator_doc_path}")

    doc_path = Path(operator_doc_path)
    if not doc_path.exists():
        raise FileNotFoundError(f"算子文档不存在: {operator_doc_path}")

    with open(doc_path, 'r', encoding='utf-8') as f:
        operator_doc = f.read()

    operator_name = doc_path.stem
    return operator_name, operator_doc


async def execute_workflow(graph, operator_name: str, operator_doc: str):
    """执行工作流并返回最终状态"""
    logger = logging.getLogger(__name__)
    logger.debug(f"开始处理算子: {operator_name}")
    final_state = await graph.run(operator_name, operator_doc)
    return final_state


def process_results(final_state: dict) -> None:
    """处理并输出执行结果"""
    logger = logging.getLogger(__name__)

    if final_state.get("error"):
        logger.error(f"处理失败: {final_state['error']}")
        sys.exit(1)

    logger.info("处理完成!")
    logger.info(f"合并结果: {final_state.get('merged_result_path')}")
    logger.info(f"测试用例: {final_state.get('test_cases_path')}")


def run() -> None:
    """
    程序入口
    """
    configure_console_encoding()
    if len(sys.argv) < 2:
        print("用法: python -m src.main <算子文档路径>")
        print("示例: python -m src.main docs/aclnnMoeDistributeDispatchV2.md")
        sys.exit(1)

    operator_doc_path = sys.argv[1]
    # operator_doc_path = "D:\\tools\\cases-generate\\operators\\aclnnAdaptiveAvgPool2dBackward.md"
    asyncio.run(main(operator_doc_path))


if __name__ == "__main__":
    run()
