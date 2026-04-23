"""
打包入口文件
用于 PyInstaller 打包，避免相对导入问题
"""
import sys
import asyncio

# 确保src模块可以被导入
sys.path.insert(0, '')

from src.main import main, setup_logging
from src.config_loader import ConfigLoader
from src.path_utils import resolve_path, ensure_dir


def run():
    """
    程序入口
    """
    if len(sys.argv) < 2:
        print("用法: operator-cases-tool.exe <算子文档路径>")
        print("示例: operator-cases-tool.exe ./aclnnAddLayerNorm.md")
        sys.exit(1)
    
    operator_doc_path = sys.argv[1]
    asyncio.run(main(operator_doc_path))


if __name__ == "__main__":
    run()
