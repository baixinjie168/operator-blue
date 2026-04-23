"""
路径工具模块
负责获取程序运行时的根目录和解析相对路径
"""
import sys
import os
from pathlib import Path


def get_app_root() -> Path:
    """
    获取应用程序根目录

    在打包后的exe环境中，根目录是exe所在目录
    在开发环境中，根目录是项目根目录

    Returns:
        Path: 应用程序根目录
    """
    if getattr(sys, 'frozen', False):
        # 打包后的exe环境
        # sys.executable 是exe文件的完整路径
        return Path(sys.executable).parent
    else:
        # 开发环境
        # 假设此文件在 src/ 目录下，项目根目录是其父目录
        return Path(__file__).parent.parent


def resolve_path(relative_path: str) -> Path:
    """
    将相对路径解析为基于应用程序根目录的绝对路径

    Args:
        relative_path: 相对路径字符串（如 "config/config.yaml"）

    Returns:
        Path: 绝对路径
    """
    root = get_app_root()

    # 处理以 ./ 开头的路径
    if relative_path.startswith('./'):
        relative_path = relative_path[2:]

    return root / relative_path


def ensure_dir(path: Path) -> None:
    """
    确保目录存在，如果不存在则创建

    Args:
        path: 目录路径
    """
    path.mkdir(parents=True, exist_ok=True)