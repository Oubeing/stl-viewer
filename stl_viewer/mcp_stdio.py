"""独立进程模式的 stdio 传输层（零第三方依赖）。

GUI 内嵌服务走 ``mcp_http.py`` + ``mcp_runtime.py``；本文件用于 AI 客户端自行
拉起的无界面服务，以及命令行自检。协议规则：

* **一条消息 = 一行 JSON**（JSON-RPC 2.0），客户端写 stdin，服务端写 stdout；
* stdout 只允许出现协议消息，日志一律写 stderr（写错地方会污染协议流）；
* 带 ``id`` 的是请求，必须回一条带同 ``id`` 的响应；
  不带 ``id`` 的是通知（notification），**不能回复**；
* 生命周期：``initialize`` → 客户端发 ``notifications/initialized`` → 之后
  才能 ``tools/list`` / ``tools/call`` / ``resources/read`` / ``prompts/get``。

只用标准库实现，业务层 ``mcp_service.py`` 与传输解耦；官方 SDK 版见
``mcp_fastmcp.py``。两种传输都通过 ``MCPStdioServer.handle()`` 复用同一套分发。

启动：

    python -m stl_viewer.mcp_stdio --root E:/zone6/stl-viewer-py-main
    # 或者：python -m stl_viewer --mcp --root <工作区>
"""

from __future__ import annotations

import argparse
import json
import sys
import traceback
from typing import Any

from .mcp_service import (PROTOCOL_VERSION, SERVER_NAME, SERVER_VERSION,
                          MCPService, ToolError)

# JSON-RPC 2.0 标准错误码
PARSE_ERROR = -32700
INVALID_REQUEST = -32600
METHOD_NOT_FOUND = -32601
INVALID_PARAMS = -32602
INTERNAL_ERROR = -32603

INSTRUCTIONS = (
    "STL Viewer MCP：先 list_models 找模型，再 load_model 载入；"
    "长度测量默认取测量平面内的投影长度（与 CATIA/SolidWorks 视图口径一致），"
    "需要真实空间距离时把 measure 的 mode 设为 3d；"
    "读 stl://guide/measurement 了解完整口径。"
)

#: 服务器支持的协议版本；客户端带来其中之一就回显，否则回默认版本让客户端降级
SUPPORTED_PROTOCOL_VERSIONS = (
    "2024-11-05", "2025-03-26", PROTOCOL_VERSION, "2025-11-25",
)


#: 本服务器认识的方法；其余请求返回 JSON-RPC -32601
_KNOWN_METHODS = {
    "initialize", "ping", "logging/setLevel",
    "tools/list", "tools/call",
    "resources/list", "resources/templates/list", "resources/read",
    "prompts/list", "prompts/get",
    "notifications/initialized", "notifications/cancelled",
    "notifications/progress", "notifications/roots/list_changed",
}


class MCPStdioServer:
    """把 :class:`MCPService` 挂到 stdin/stdout 上。

    类里只做「协议编解码 + 方法分发」，不写任何 STL 业务逻辑 —— 换传输
    （stdio / HTTP / SDK）时业务层无需改动。
    """

    def __init__(self, service: MCPService, log=sys.stderr):
        self.service = service
        self.log = log
        self.initialized = False
        self.client_version: str | None = None

    # ---------------- 响应构造 ----------------
    @staticmethod
    def _ok(msg_id: Any, result: dict) -> dict:
        return {"jsonrpc": "2.0", "id": msg_id, "result": result}

    @staticmethod
    def _error(msg_id: Any, code: int, message: str) -> dict:
        return {"jsonrpc": "2.0", "id": msg_id,
                "error": {"code": code, "message": message}}

    def _note(self, text: str) -> None:
        """日志只进 stderr。"""
        try:
            print(f"[{SERVER_NAME}] {text}", file=self.log, flush=True)
        except Exception:
            pass

    # ---------------- 单条消息 ----------------
    def handle(self, message: Any) -> dict | None:
        """处理一条 JSON-RPC 消息；通知返回 None（不发响应）。"""
        if not isinstance(message, dict) or message.get("jsonrpc") != "2.0":
            return self._error(message.get("id") if isinstance(message, dict)
                               else None, INVALID_REQUEST, "不是合法的 JSON-RPC 2.0 请求")
        method = message.get("method")
        params = message.get("params") or {}
        msg_id = message.get("id")
        notification = "id" not in message

        if not isinstance(params, dict):
            return (None if notification else
                    self._error(msg_id, INVALID_PARAMS, "params 必须是对象"))
        if not notification and method not in _KNOWN_METHODS:
            return self._error(msg_id, METHOD_NOT_FOUND, f"未知方法: {method}")

        # 通知：只记状态，绝不回复
        if method == "notifications/initialized":
            self.initialized = True
            return None
        if method and method.startswith("notifications/"):
            return None

        try:
            result = self._dispatch(method, params)
        except ToolError as exc:
            # 工具层面的参数/业务错误：用 isError 结果返回，让 AI 能读并重试
            result = {"content": [{"type": "text", "text": f"错误: {exc}"}],
                      "isError": True}
        except (KeyError, TypeError, ValueError) as exc:
            return (None if notification else
                    self._error(msg_id, INVALID_PARAMS, str(exc)))
        except Exception as exc:                       # pragma: no cover - 兜底
            self._note(traceback.format_exc())
            return (None if notification else
                    self._error(msg_id, INTERNAL_ERROR,
                                f"{type(exc).__name__}: {exc}"))
        return None if notification else self._ok(msg_id, result)

    def _dispatch(self, method: str, params: dict) -> dict:
        svc = self.service

        if method == "initialize":
            # 协议版本协商：客户端版本可用则回显，否则用服务器支持的版本
            asked = params.get("protocolVersion")
            self.client_version = asked if isinstance(asked, str) else PROTOCOL_VERSION
            version = (asked if asked in SUPPORTED_PROTOCOL_VERSIONS
                       else PROTOCOL_VERSION)
            return {
                "protocolVersion": version,
                "capabilities": {
                    "tools": {"listChanged": False},
                    "resources": {"subscribe": False, "listChanged": False},
                    "prompts": {"listChanged": False},
                    "logging": {},
                },
                "serverInfo": {"name": SERVER_NAME, "version": SERVER_VERSION},
                "instructions": INSTRUCTIONS,
            }

        if method == "ping":
            return {}

        if method == "logging/setLevel":
            return {}

        if method == "tools/list":
            return {"tools": svc.list_tools()}

        if method == "tools/call":
            name = params.get("name")
            if not isinstance(name, str):
                raise ToolError("tools/call 缺少字符串参数 name")
            arguments = params.get("arguments") or {}
            if not isinstance(arguments, dict):
                raise ToolError("tools/call 的 arguments 必须是对象")
            result = svc.call_tool(name, arguments)
            return {"content": result.to_content(), "isError": False}

        if method == "resources/list":
            return {"resources": svc.list_resources()}

        if method == "resources/templates/list":
            return {"resourceTemplates": []}

        if method == "resources/read":
            uri = params.get("uri")
            if not isinstance(uri, str):
                raise ToolError("resources/read 缺少 uri")
            mime, text = svc.read_resource(uri)
            return {"contents": [{"uri": uri, "mimeType": mime, "text": text}]}

        if method == "prompts/list":
            return {"prompts": svc.list_prompts()}

        if method == "prompts/get":
            name = params.get("name")
            if not isinstance(name, str):
                raise ToolError("prompts/get 缺少 name")
            return svc.get_prompt(name, params.get("arguments") or {})

        # handle() 已按 _KNOWN_METHODS 拦截；这里只是防御性兜底。
        raise ToolError(f"未知方法: {method}")

    # ---------------- 主循环 ----------------
    def serve(self, stdin=None, stdout=None) -> int:
        """逐行读取请求、逐行写回响应，直到 EOF。"""
        stdin = stdin or sys.stdin
        stdout = stdout or sys.stdout
        for line in stdin:
            line = line.strip()
            if not line:
                continue
            try:
                message = json.loads(line)
            except json.JSONDecodeError as exc:
                response = self._error(None, PARSE_ERROR, f"JSON 解析失败: {exc}")
            else:
                response = self.handle(message)
            if response is None:
                continue                              # 通知 / 无响应错误
            stdout.write(json.dumps(response, ensure_ascii=False,
                                    separators=(",", ":")) + "\n")
            stdout.flush()
        return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        prog="stl_viewer.mcp_stdio",
        description="STL Viewer 的 MCP stdio 服务器（零第三方依赖）")
    parser.add_argument("--root", default=None,
                        help="允许 AI 访问的沙箱根目录，默认当前工作目录")
    args = parser.parse_args(argv)

    # Windows 控制台默认编码未必是 UTF-8；MCP stdio 规范要求 UTF-8。
    for stream, kw in ((sys.stdin, {"encoding": "utf-8"}),
                       (sys.stdout, {"encoding": "utf-8", "newline": "\n"})):
        if hasattr(stream, "reconfigure"):
            try:
                stream.reconfigure(**kw)
            except Exception:
                pass

    try:
        service = MCPService(args.root)
    except Exception as exc:
        print(f"启动失败: {exc}", file=sys.stderr, flush=True)
        return 2
    server = MCPStdioServer(service)
    # 启动日志用 ASCII：MCP 客户端会捕获 stderr，中文在部分 Windows 控制台
    # 上会变成乱码（不影响协议，但排查时很难读）。
    server._note(f"stdio ready, root={service.root}, "
                 f"tools={len(service.list_tools())}")
    try:
        return server.serve()
    except (KeyboardInterrupt, BrokenPipeError):
        return 0


if __name__ == "__main__":
    sys.exit(main())
