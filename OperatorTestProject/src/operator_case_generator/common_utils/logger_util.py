"""
日志类实现
支持功能：
1. 从主函数入口传递参数创建日志文件名
2. 每行日志包含代码文件名、类名、函数名、行数
3. 日志轮转（按时间/按大小），参数可配置
4. 线程安全
"""

import logging
import os
import sys
from datetime import datetime
from logging.handlers import TimedRotatingFileHandler, RotatingFileHandler
from typing import Optional
import threading


class ContextFilter(logging.Filter):
    """
    自定义过滤器，用于添加代码位置信息（文件名、类名、函数名、行号）
    """
    def filter(self, record):
        # 获取调用者的帧信息
        frame = sys._getframe(8)  # 向上追溯调用栈
        
        # 获取文件名
        record.code_filename = os.path.basename(frame.f_code.co_filename)
        
        # 获取行号
        record.code_lineno = frame.f_lineno
        
        # 获取函数名
        record.code_funcname = frame.f_code.co_name
        
        # 尝试获取类名（如果是在类方法中调用）
        try:
            # 尝试从局部变量中获取self
            if 'self' in frame.f_locals:
                record.code_classname = frame.f_locals['self'].__class__.__name__
            elif 'cls' in frame.f_locals:
                record.code_classname = frame.f_locals['cls'].__name__
            else:
                record.code_classname = 'N/A'
        except Exception:
            record.code_classname = 'N/A'
        
        return True


class ThreadSafeLogger:
    """
    线程安全的日志类
    
    功能：
    1. 支持从主函数入口传递参数创建日志文件名
    2. 支持打印的每一行日志包含打印所在位置的代码文件名、类名、函数名，以及所在行数
    3. 支持日志轮转，支持按不同的方式轮转，且参数可配置
    4. 支持线程安全功能
    
    注意：推荐使用 init_logger() 和 get_logger() 函数来管理全局日志实例
    """
    
    def __init__(
        self,
        log_name: Optional[str] = None,
        log_dir: str = "./logs",
        log_level: int = logging.DEBUG,
        rotation_type: str = "time",  # "time" 或 "size"
        # 按时间轮转参数
        rotation_interval: int = 1,  # 轮转间隔
        rotation_when: str = "midnight",  # 轮转时间单位：S-秒, M-分, H-小时, D-天, midnight-午夜, W0-W6-星期几
        rotation_backup_count: int = 7,  # 保留数量
        # 按大小轮转参数
        max_bytes: int = 10 * 1024 * 1024,  # 文件大小（字节），默认10MB
        size_backup_count: int = 5,  # 保留备份数量
        # 控制台输出
        console_output: bool = True
    ):
        """
        初始化日志类
        
        Args:
            log_name: 日志文件名（不含扩展名），如果为None则使用时间戳
            log_dir: 日志目录
            log_level: 日志级别
            rotation_type: 轮转类型，"time"按时间轮转，"size"按大小轮转
            rotation_interval: 按时间轮转的间隔
            rotation_when: 按时间轮转的时间单位
            rotation_backup_count: 按时间轮转的保留数量
            max_bytes: 按大小轮转的最大文件大小（字节）
            size_backup_count: 按大小轮转的保留备份数量
            console_output: 是否输出到控制台
        """
        self.log_dir = log_dir
        self.rotation_type = rotation_type
        self.console_output = console_output
        pwd = os.getcwd()
        
        # 创建日志目录
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        
        # 生成日志文件名
        if log_name is None:
            log_name = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        self.log_file = os.path.join(log_dir, f"{log_name}.log")
        
        # 创建logger
        self.logger = logging.getLogger(log_name)
        self.logger.setLevel(log_level)
        
        # 清除已有的handlers
        self.logger.handlers.clear()
        
        # 设置日志格式
        self.formatter = logging.Formatter(
            fmt='[%(asctime)s] [%(levelname)s] [%(threadName)s] [%(code_filename)s:%(code_lineno)d] %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        # 添加上下文过滤器
        self.context_filter = ContextFilter()
        self.logger.addFilter(self.context_filter)
        
        # 创建文件处理器（根据轮转类型）
        if rotation_type == "time":
            self.file_handler = TimedRotatingFileHandler(
                filename=self.log_file,
                when=rotation_when,
                interval=rotation_interval,
                backupCount=rotation_backup_count,
                encoding='utf-8'
            )
        elif rotation_type == "size":
            self.file_handler = RotatingFileHandler(
                filename=self.log_file,
                maxBytes=max_bytes,
                backupCount=size_backup_count,
                encoding='utf-8'
            )
        else:
            raise ValueError(f"Unsupported rotation type: {rotation_type}，please use 'time' or 'size'")
        
        self.file_handler.setLevel(log_level)
        self.file_handler.setFormatter(self.formatter)
        self.logger.addHandler(self.file_handler)
        
        # 添加控制台处理器
        if console_output:
            self.console_handler = logging.StreamHandler()
            self.console_handler.setLevel(log_level)
            self.console_handler.setFormatter(self.formatter)
            self.logger.addHandler(self.console_handler)
        
        # 线程锁，确保线程安全
        self._write_lock = threading.Lock()
    
    def _log(self, level, msg, *args, **kwargs):
        """内部日志方法，添加线程锁确保线程安全"""
        with self._write_lock:
            self.logger.log(level, msg, *args, **kwargs)
    
    def debug(self, msg, *args, **kwargs):
        """调试级别日志"""
        self._log(logging.DEBUG, msg, *args, **kwargs)
    
    def info(self, msg, *args, **kwargs):
        """信息级别日志"""
        self._log(logging.INFO, msg, *args, **kwargs)
    
    def warning(self, msg, *args, **kwargs):
        """警告级别日志"""
        self._log(logging.WARNING, msg, *args, **kwargs)
    
    def error(self, msg, *args, **kwargs):
        """错误级别日志"""
        self._log(logging.ERROR, msg, *args, **kwargs)
    
    def critical(self, msg, *args, **kwargs):
        """严重错误级别日志"""
        self._log(logging.CRITICAL, msg, *args, **kwargs)
    
    def exception(self, msg, *args, **kwargs):
        """异常日志，自动包含异常堆栈信息"""
        with self._write_lock:
            self.logger.exception(msg, *args, **kwargs)
    
    def set_level(self, level: int):
        """
        设置日志级别
        
        Args:
            level: 日志级别 (logging.DEBUG, INFO, WARNING, ERROR, CRITICAL)
        """
        self.logger.setLevel(level)
        self.file_handler.setLevel(level)
        if self.console_output:
            self.console_handler.setLevel(level)


# 便捷函数，用于快速创建全局logger实例
_global_logger: Optional[ThreadSafeLogger] = None


def init_logger(
    log_name: Optional[str] = None,
    log_dir: str = "./logs",
    log_level: int = logging.DEBUG,
    rotation_type: str = "time",
    rotation_interval: int = 1,
    rotation_when: str = "midnight",
    rotation_backup_count: int = 7,
    max_bytes: int = 10 * 1024 * 1024,
    size_backup_count: int = 5,
    console_output: bool = True
) -> ThreadSafeLogger:
    """
    初始化全局日志实例
    
    Args:
        log_name: 日志文件名（不含扩展名），如果为None则使用时间戳
        log_dir: 日志目录
        log_level: 日志级别
        rotation_type: 轮转类型，"time"按时间轮转，"size"按大小轮转
        rotation_interval: 按时间轮转的间隔
        rotation_when: 按时间轮转的时间单位
        rotation_backup_count: 按时间轮转的保留数量
        max_bytes: 按大小轮转的最大文件大小（字节）
        size_backup_count: 按大小轮转的保留备份数量
        console_output: 是否输出到控制台
    
    Returns:
        ThreadSafeLogger实例
    """
    global _global_logger
    _global_logger = ThreadSafeLogger(
        log_name=log_name,
        log_dir=log_dir,
        log_level=log_level,
        rotation_type=rotation_type,
        rotation_interval=rotation_interval,
        rotation_when=rotation_when,
        rotation_backup_count=rotation_backup_count,
        max_bytes=max_bytes,
        size_backup_count=size_backup_count,
        console_output=console_output
    )
    return _global_logger


def get_logger() -> ThreadSafeLogger:
    """
    获取全局日志实例
    
    Returns:
        ThreadSafeLogger实例
    
    Raises:
        RuntimeError: 如果未初始化全局日志实例
    """
    global _global_logger
    if _global_logger is None:
        raise RuntimeError("Log file don't init, please call init_logger() function to init")
    return _global_logger


class LazyLogger:
    """
    延迟获取日志实例的包装类
    
    用于在模块顶层安全地创建 logger 对象，实际调用时会延迟到 init_logger() 执行后。
    
    使用示例:
        # sub_module.py
        from logger import LazyLogger
        
        logger = LazyLogger()  # 在顶层安全使用，不会报错
        
        def my_function():
            logger.info("这条日志会正常输出")  # 实际调用时才获取真正的 logger
    """
    
    def __init__(self):
        """初始化延迟日志对象"""
        self._real_logger = None
    
    def _get_real_logger(self) -> ThreadSafeLogger:
        """获取真正的日志实例（延迟获取）"""
        if self._real_logger is None:
            self._real_logger = get_logger()
        return self._real_logger
    
    def debug(self, msg, *args, **kwargs):
        self._get_real_logger().debug(msg, *args, **kwargs)
    
    def info(self, msg, *args, **kwargs):
        self._get_real_logger().info(msg, *args, **kwargs)
    
    def warning(self, msg, *args, **kwargs):
        self._get_real_logger().warning(msg, *args, **kwargs)
    
    def error(self, msg, *args, **kwargs):
        self._get_real_logger().error(msg, *args, **kwargs)
    
    def critical(self, msg, *args, **kwargs):
        self._get_real_logger().critical(msg, *args, **kwargs)
    
    def exception(self, msg, *args, **kwargs):
        self._get_real_logger().exception(msg, *args, **kwargs)


if __name__ == "__main__":
    # 示例用法
    
    # 方式1：直接创建实例
    print("=== 方式1：直接创建实例 ===")
    logger1 = ThreadSafeLogger(
        log_name="test_app",
        log_dir="./logs",
        log_level=logging.DEBUG,
        rotation_type="time",
        rotation_when="midnight",
        rotation_backup_count=7,
        console_output=True
    )
    
    logger1.info("这是一条信息日志")
    logger1.debug("这是一条调试日志")
    logger1.warning("这是一条警告日志")
    logger1.error("这是一条错误日志")
    
    # 方式2：使用全局初始化函数（推荐在主函数入口使用）
    print("\n=== 方式2：使用全局初始化函数 ===")
    init_logger(
        log_name="my_application",
        log_dir="./logs",
        rotation_type="size",
        max_bytes=1 * 1024 * 1024,  # 1MB
        size_backup_count=3,
        console_output=True
    )
    
    logger2 = get_logger()
    logger2.info("应用程序启动")
    logger2.debug("调试信息")
    
    # 测试类中的日志
    class TestClass:
        def test_method(self):
            logger2.info("这是类方法中的日志")
    
    obj = TestClass()
    obj.test_method()
    
    # 测试线程安全
    print("\n=== 测试线程安全 ===")
    
    def worker(thread_id):
        logger2.info(f"线程 {thread_id} 开始工作")
        for i in range(3):
            logger2.debug(f"线程 {thread_id} - 消息 {i}")
        logger2.info(f"线程 {thread_id} 完成工作")
    
    threads = []
    for i in range(5):
        t = threading.Thread(target=worker, args=(i,))
        threads.append(t)
        t.start()
    
    for t in threads:
        t.join()
    
    logger2.info("所有线程完成")
