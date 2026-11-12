"""官方 MCP SDK 适配（独立进程模式的可选传输）。

与 ``mcp_stdio.py`` 等价，但由 SDK 负责 JSON-RPC 收发、握手、能力协商、错误码，
并从函数类型注解 + docstring 生成 ``inputSchema``；资源与提示词用装饰器注册。
所有工具函数只把参数转发给 ``MCPService.call_tool()``，业务层不感知传输。

SDK 版本兼容：

* ``mcp`` 1.x：``from mcp.server.fastmcp import FastMCP``
* ``mcp`` 2.x：``FastMCP`` 改名 ``MCPServer``，位于 ``mcp.server.mcpserver``

两种都做了兼容；手写 stdio 版与 GUI 内嵌 HTTP 版不依赖 SDK。

启动：

    pip install "mcp>=1.2"
    python -m stl_viewer.mcp_fastmcp --root E:/zone6/stl-viewer-py-main
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Any

from .mcp_service import MCPService

try:
    from mcp import types as mcp_types
    try:                                             # mcp >= 2
        from mcp.server.mcpserver import MCPServer as _ServerClass
    except ImportError:                              # mcp 1.x
        from mcp.server.fastmcp import FastMCP as _ServerClass
    _SDK_ERROR: str | None = None
except ImportError as exc:                           # 未安装 SDK 也能导入模块
    mcp_types = None                                 # type: ignore[assignment]
    _ServerClass = None                              # type: ignore[assignment]
    _SDK_ERROR = str(exc)

SERVICE: MCPService | None = None


def _run(tool: str, **kwargs) -> list:
    """统一入口：过滤 None → 调用业务层 → 转成 SDK 的 content 对象。

    形参叫 ``tool`` 而非 ``name``：部分工具本身有 ``name`` 参数，用
    ``**kwargs`` 转发时会与首个形参冲突（``got multiple values for argument
    'name'``）。
    """
    args = {k: v for k, v in kwargs.items() if v is not None}
    assert SERVICE is not None
    result = SERVICE.call_tool(tool, args)
    parts: list = []
    for item in result.to_content():
        if item["type"] == "text":
            parts.append(mcp_types.TextContent(type="text", text=item["text"]))
        else:
            parts.append(mcp_types.ImageContent(
                type="image", data=item["data"], mimeType=item["mimeType"]))
    return parts


def build_server():
    """把所有工具 / 资源 / 提示词注册到 SDK 服务器上。"""
    if _ServerClass is None:
        raise ImportError(_SDK_ERROR or "MCP SDK 不可用")
    server = _ServerClass("stl-viewer")

    # ---------------- Tools ----------------
    @server.tool()
    def server_info() -> list:
        """查看服务器能力、工作区根目录、测量口径与限制。"""
        return _run("server_info")

    @server.tool()
    def list_models(directory: str | None = None,
                    recursive: bool = True) -> list:
        """列出工作区内可载入的 STL / PLY 模型文件。"""
        return _run("list_models", directory=directory, recursive=recursive)

    @server.tool()
    def load_model(path: str, name: str | None = None,
                   weld: bool = True, repair: bool = False) -> list:
        """载入一个 STL / PLY 模型到当前场景，并自动适配相机。"""
        return _run("load_model", path=path, name=name, weld=weld, repair=repair)

    @server.tool()
    def list_objects() -> list:
        """列出当前场景中的模型对象。"""
        return _run("list_objects")

    @server.tool()
    def inspect_model(name: str | None = None, index: int | None = None,
                      deep: bool = True) -> list:
        """深度统计模型：面数、体积、表面积、是否封闭、边界边、连通分量等。"""
        return _run("inspect_model", name=name, index=index, deep=deep)

    @server.tool()
    def remove_object(name: str | None = None,
                      index: int | None = None) -> list:
        """从场景移除一个对象。"""
        return _run("remove_object", name=name, index=index)

    @server.tool()
    def clear_scene() -> list:
        """清空场景中的全部对象。"""
        return _run("clear_scene")

    @server.tool()
    def set_view(view: str | None = None, yaw: float | None = None,
                 pitch: float | None = None, roll: float | None = None,
                 orthographic: bool | None = None,
                 fit: bool = False) -> list:
        """设置相机视角（标准视图或 yaw/pitch/roll）。"""
        return _run("set_view", view=view, yaw=yaw, pitch=pitch, roll=roll,
                    orthographic=orthographic, fit=fit)

    @server.tool()
    def measure(kind: str, points: list,
                plane: list | None = None, mode: str = "view",
                units: str = "mm") -> list:
        """执行一次测量。长度默认取测量平面内的投影长度（CAD 口径），
        需要真实空间距离请把 mode 设为 3d。"""
        return _run("measure", kind=kind, points=points, plane=plane,
                    mode=mode, units=units)

    @server.tool()
    def list_measurements(units: str = "mm") -> list:
        """列出本次会话已完成的全部测量结果。"""
        return _run("list_measurements", units=units)

    @server.tool()
    def clear_measurements() -> list:
        """清空测量结果。"""
        return _run("clear_measurements")

    @server.tool()
    def render_picture(path: str | None = None, width: int = 640,
                       height: int = 480, view: str | None = None,
                       supersample: int = 1, color_mode: str | None = None,
                       wire: bool | None = None,
                       transparent_background: bool | None = None,
                       include_image: bool | None = None) -> list:
        """渲染当前场景。不给 path 时把 PNG 作为图片内容直接返回给 AI。"""
        return _run("render_picture", path=path, width=width, height=height,
                    view=view, supersample=supersample, color_mode=color_mode,
                    wire=wire, transparent_background=transparent_background,
                    include_image=include_image)

    @server.tool()
    def section_radius(z: float, name: str | None = None) -> list:
        """求回转体在指定 Z 高度处的最大回转半径。"""
        return _run("section_radius", z=z, name=name)

    @server.tool()
    def export_stl(path: str, name: str | None = None, ascii: bool = False,
                   weld: bool = False, repair: bool = False) -> list:
        """把场景中的模型写出为 STL（二进制或 ASCII）。"""
        return _run("export_stl", path=path, name=name, ascii=ascii,
                    weld=weld, repair=repair)

    @server.tool()
    def split_components(name: str | None = None,
                         out_prefix: str | None = None,
                         min_triangles: int = 0) -> list:
        """把模型按连通分量拆成多个 STL 文件。"""
        return _run("split_components", name=name, out_prefix=out_prefix,
                    min_triangles=min_triangles)

    # ---------------- Resources ----------------
    def _read(uri: str) -> str:
        assert SERVICE is not None
        return SERVICE.read_resource(uri)[1]

    @server.resource("stl://scene/objects", mime_type="application/json")
    def scene_objects() -> str:
        """当前场景对象清单（JSON）。"""
        return _read("stl://scene/objects")

    @server.resource("stl://scene/measurements", mime_type="application/json")
    def scene_measurements() -> str:
        """本次会话测量结果（JSON）。"""
        return _read("stl://scene/measurements")

    @server.resource("stl://guide/measurement", mime_type="text/markdown")
    def measurement_guide() -> str:
        """测量口径说明。"""
        return _read("stl://guide/measurement")

    @server.resource("stl://guide/tools", mime_type="text/markdown")
    def tools_guide() -> str:
        """工具清单与调用顺序。"""
        return _read("stl://guide/tools")

    # ---------------- Prompts ----------------
    @server.prompt("inspect_model")
    def prompt_inspect_model(path: str) -> str:
        """引导 AI 完整检查一个模型：载入 → 统计 → 渲染 → 结论。"""
        assert SERVICE is not None
        return SERVICE.get_prompt("inspect_model", {"path": path})["messages"][0][
            "content"]["text"]

    @server.prompt("measure_plan")
    def prompt_measure_plan(goal: str) -> str:
        """引导 AI 按 CAD 口径规划一次测量。"""
        assert SERVICE is not None
        return SERVICE.get_prompt("measure_plan", {"goal": goal})["messages"][0][
            "content"]["text"]

    return server


def main(argv=None) -> int:
    global SERVICE
    parser = argparse.ArgumentParser(
        prog="stl_viewer.mcp_fastmcp",
        description="STL Viewer 的 MCP 服务器（官方 SDK 版）")
    parser.add_argument("--root", default=None,
                        help="允许 AI 访问的沙箱根目录，默认当前工作目录")
    args = parser.parse_args(argv)
    if _SDK_ERROR is not None:
        print("未安装官方 MCP SDK，请执行：pip install \"mcp>=1.2\"",
              file=sys.stderr)
        print(f"（导入错误: {_SDK_ERROR}）", file=sys.stderr)
        print("也可以改用零依赖版：python -m stl_viewer.mcp_stdio --root <目录>",
              file=sys.stderr)
        return 2
    try:
        SERVICE = MCPService(args.root or os.getcwd())
        server = build_server()
    except Exception as exc:
        print(f"启动失败: {exc}", file=sys.stderr)
        return 2
    server.run(transport="stdio")
    return 0


if __name__ == "__main__":
    sys.exit(main())
