"""
批量处理入口文件
默认遍历指定目录下的所有算子文档，并调用 src.main 逐个处理
"""
import argparse
import asyncio
import shutil
import sys
import time
from pathlib import Path

from src.main import main as process_operator

PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_OPERATORS_DIR = Path(r"C:\Users\ttelab\code\operators")
PROCESSED_DIR = PROJECT_ROOT / "processed"
FAILED_DOC_DIR = PROJECT_ROOT / "failed_doc"


def collect_operator_docs(operators_dir: Path) -> list[Path]:
    """收集目录下所有 Markdown 算子文档"""
    return sorted(path for path in operators_dir.rglob("*.md") if path.is_file())


def _build_processed_destination(
        doc_path: Path,
        operators_dir: Path,
        processed_dir: Path
) -> Path:
    """构建处理成功文档的移动目标路径，保留相对目录结构并避免重名覆盖"""
    try:
        relative_path = doc_path.relative_to(operators_dir)
        destination = processed_dir / relative_path
    except ValueError:
        destination = processed_dir / doc_path.name

    if not destination.exists():
        return destination

    counter = 1
    while True:
        candidate = destination.with_name(f"{destination.stem}_{counter}{destination.suffix}")
        if not candidate.exists():
            return candidate
        counter += 1


def move_processed_doc(
        doc_path: Path,
        operators_dir: Path,
        processed_dir: Path
) -> Path:
    """将处理成功的文档移动到项目 processed 目录"""
    processed_dir.mkdir(parents=True, exist_ok=True)
    destination = _build_processed_destination(doc_path, operators_dir, processed_dir)
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(doc_path), str(destination))
    return destination


def move_failed_doc(
        doc_path: Path,
        operators_dir: Path,
        failed_doc_dir: Path
) -> Path:
    """将处理失败的文档移动到项目 failed_doc 目录"""
    failed_doc_dir.mkdir(parents=True, exist_ok=True)
    destination = _build_processed_destination(doc_path, operators_dir, failed_doc_dir)
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(doc_path), str(destination))
    return destination


async def run_batch(operators_dir: Path, stop_on_error: bool) -> int:
    """批量执行算子处理流程"""
    doc_paths = collect_operator_docs(operators_dir)

    if not doc_paths:
        print(f"未找到可处理的文档: {operators_dir}")
        return 1

    print(f"找到 {len(doc_paths)} 个算子文档，开始批量处理。")
    print(f"处理成功的文档将移动到: {PROCESSED_DIR}")
    print(f"处理失败的文档将移动到: {FAILED_DOC_DIR}")

    success_count = 0
    failed_docs: list[tuple[Path, str]] = []
    started_at = time.perf_counter()

    for index, doc_path in enumerate(doc_paths, start=1):
        print(f"[{index}/{len(doc_paths)}] 开始处理: {doc_path}")

        try:
            await process_operator(str(doc_path))
            moved_path = move_processed_doc(doc_path, operators_dir, PROCESSED_DIR)
            success_count += 1
            print(f"[{index}/{len(doc_paths)}] 处理成功: {doc_path.name} -> {moved_path}")
        except SystemExit as exc:
            exit_code = exc.code if isinstance(exc.code, int) else 1
            error_message = f"SystemExit({exit_code})"
            moved_path = move_failed_doc(doc_path, operators_dir, FAILED_DOC_DIR)
            failed_docs.append((moved_path, error_message))
            print(f"[{index}/{len(doc_paths)}] 处理失败: {doc_path.name} - {error_message} -> {moved_path}")

            if stop_on_error:
                break
        except Exception as exc:
            error_message = f"{type(exc).__name__}: {exc}"
            moved_path = move_failed_doc(doc_path, operators_dir, FAILED_DOC_DIR)
            failed_docs.append((moved_path, error_message))
            print(f"[{index}/{len(doc_paths)}] 处理失败: {doc_path.name} - {error_message} -> {moved_path}")

            if stop_on_error:
                break

    elapsed = time.perf_counter() - started_at
    print("")
    print("批量处理完成。")
    print(f"总数: {len(doc_paths)}")
    print(f"成功: {success_count}")
    print(f"失败: {len(failed_docs)}")
    print(f"耗时: {elapsed:.2f} 秒")

    if failed_docs:
        print("")
        print("失败文档列表:")
        for doc_path, error_message in failed_docs:
            print(f"- {doc_path}: {error_message}")

        return 1

    return 0


def parse_args() -> argparse.Namespace:
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description=f"批量调用 src.main 处理算子文档，固定目录: {DEFAULT_OPERATORS_DIR}"
    )
    parser.add_argument(
        "--stop-on-error",
        action="store_true",
        help="遇到第一个失败文档时立即停止",
    )
    return parser.parse_args()


def run() -> None:
    """程序入口"""
    args = parse_args()
    operators_dir = DEFAULT_OPERATORS_DIR

    if not operators_dir.exists():
        print(f"目录不存在: {operators_dir}")
        sys.exit(1)

    if not operators_dir.is_dir():
        print(f"路径不是目录: {operators_dir}")
        sys.exit(1)

    exit_code = asyncio.run(run_batch(operators_dir, args.stop_on_error))
    sys.exit(exit_code)


if __name__ == "__main__":
    run()
