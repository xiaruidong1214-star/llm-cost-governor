"""最小化的行覆盖统计（不依赖 pytest-cov）。

用法::

    python -m tools.coverage_probe

它用 ``sys.settrace`` 统计 ``costgovernor`` 包内被执行过的行，跑完整个测试套件后
打印每个模块的行覆盖率。这是一个**近似**指标（只看行是否执行过，不看分支），
但足以回答「哪些模块根本没被测试碰到」这个问题。
"""

from __future__ import annotations

import os
import sys

PACKAGE = "costgovernor"


def main() -> int:
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    package_dir = os.path.normcase(os.path.join(root, PACKAGE))

    executed: dict[str, set[int]] = {}

    def tracer(frame, event, arg):  # type: ignore[no-untyped-def]
        if event == "call":
            filename = os.path.normcase(frame.f_code.co_filename)
            if filename.startswith(package_dir):
                executed.setdefault(filename, set()).add(frame.f_lineno)
                return tracer
            return None
        if event == "line":
            executed.setdefault(os.path.normcase(frame.f_code.co_filename), set()).add(frame.f_lineno)
        return tracer

    def run() -> int:
        import pytest

        return pytest.main(["-q", "-o", "addopts=", "-p", "no:cacheprovider", "tests"])

    sys.settrace(tracer)
    try:
        run()
    finally:
        sys.settrace(None)

    print("\n=== costgovernor 行覆盖（近似，只看行是否执行）===")
    total_statements = 0
    total_covered = 0
    for name in sorted(os.listdir(package_dir)):
        if not name.endswith(".py"):
            continue
        path = os.path.join(package_dir, name)
        with open(path, encoding="utf-8") as handle:
            source = handle.read()
        statements = _statement_lines(source, path)
        covered = executed.get(os.path.normcase(os.path.abspath(path)), set())
        hit = len(statements & covered)
        total_statements += len(statements)
        total_covered += hit
        percent = (hit / len(statements) * 100) if statements else 100.0
        print(f"  {name:<16} {hit:>4}/{len(statements):<4} {percent:6.1f}%")
    overall = (total_covered / total_statements * 100) if total_statements else 100.0
    print(f"  {'合计':<14} {total_covered:>4}/{total_statements:<4} {overall:6.1f}%")
    return 0


def _statement_lines(source: str, filename: str) -> set[int]:
    """把源码编译成字节码，收集所有带行号的指令 → 近似「可执行行」。"""
    import dis

    code = compile(source, filename, "exec")
    lines: set[int] = set()

    def walk(obj) -> None:
        for instruction in dis.get_instructions(obj):
            if instruction.starts_line:
                lines.add(instruction.starts_line)
        for constant in obj.co_consts:
            if hasattr(constant, "co_code"):
                walk(constant)

    walk(code)
    return lines


if __name__ == "__main__":
    raise SystemExit(main())
