from __future__ import annotations

import sys
from pathlib import Path

from .errors import LexError, ParseError, RuntimeError, SylphError
from .interpreter import Interpreter
from .lexer import Lexer
from .parser import Parser


def run_source(source: str, filename: str = "unknown") -> None:
    lexer = Lexer(source)
    tokens = lexer.scan_tokens()

    parser = Parser(tokens, filename)
    program = parser.parse()

    interpreter = Interpreter()
    interpreter.interpret(program.statements)


def run_file_with_mmap(path: Path) -> None:
    """使用内存映射文件处理大文件"""
    import mmap
    with open(path, 'r+b') as f:
        # 创建内存映射
        with mmap.mmap(f.fileno(), length=0, access=mmap.ACCESS_READ) as mm:
            # 读取整个文件内容
            source = mm.read().decode('utf-8')
            run_source(source, str(path))


def run_file(path: Path) -> None:
    """运行单个文件"""
    # 检查文件大小，如果文件较大，使用内存映射
    file_size = path.stat().st_size
    if file_size > 1024 * 1024:  # 大于 1MB 的文件使用内存映射
        run_file_with_mmap(path)
    else:
        source = path.read_text(encoding="utf-8")
        run_source(source, str(path))


def run_files_in_parallel(paths: List[Path]) -> None:
    """并行处理多个文件"""
    import concurrent.futures
    # 使用线程池并行处理文件
    with concurrent.futures.ThreadPoolExecutor() as executor:
        # 提交所有文件处理任务
        future_to_path = {executor.submit(run_file, path): path for path in paths}
        # 收集结果和错误
        for future in concurrent.futures.as_completed(future_to_path):
            path = future_to_path[future]
            try:
                future.result()
            except Exception as exc:
                sys.stderr.write(f"Error processing file '{path}': {exc}\n")


def main(argv: list[str] | None = None) -> int:
    if argv is None:
        argv = sys.argv[1:]

    if not argv:
        sys.stderr.write("Usage: python -m sylph <script.syl> [script2.syl ...] or python -m sylph --file <script.syl>\n")
        return 64

    # Handle --file option
    if argv[0] == "--file":
        if len(argv) < 2:
            sys.stderr.write("Usage: python -m sylph --file <script.syl>\n")
            return 64
        paths = [Path(argv[1])]
    else:
        # 处理多个文件
        paths = [Path(arg) for arg in argv]

    try:
        if len(paths) > 1:
            # 并行处理多个文件
            run_files_in_parallel(paths)
        else:
            # 处理单个文件
            run_file(paths[0])
    except OSError as exc:
        sys.stderr.write(f"Error: cannot read file: {exc}\n")
        return 66
    except (LexError, ParseError, RuntimeError, SylphError) as exc:
        # All language-level errors are already formatted
        sys.stderr.write(str(exc) + "\n")
        return 65

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

