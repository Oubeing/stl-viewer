"""包入口。

用法::

    python -m stl_viewer                     # 打开图形界面
    python -m stl_viewer model.stl           # 打开模型
    python -m stl_viewer --cli info model.stl
    python -m stl_viewer --cli render model.stl -o out.png --colored
    python -m stl_viewer --mcp --root .      # 启动 MCP 服务（零依赖 stdio 版）
    python -m stl_viewer --mcp-sdk --root .  # 启动 MCP 服务（官方 SDK 版）
"""

from __future__ import annotations

import sys


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] in ("--cli", "-c"):
        from .cli import main as cli_main

        return cli_main(argv[1:])
    if argv and argv[0] in ("--mcp", "--mcp-sdk"):
        target = ".mcp_stdio" if argv[0] == "--mcp" else ".mcp_fastmcp"
        from importlib import import_module

        return import_module(target, __package__).main(argv[1:])
    try:
        from .viewer import main as gui_main
    except ImportError as exc:
        print("未检测到 PyQt5，无法启动图形界面。", file=sys.stderr)
        print("请执行:  pip install PyQt5", file=sys.stderr)
        print(f"（导入错误: {exc}）", file=sys.stderr)
        print("也可以使用命令行模式: python -m stl_viewer --cli --help",
              file=sys.stderr)
        return 2
    return gui_main(argv)


if __name__ == "__main__":
    sys.exit(main())
