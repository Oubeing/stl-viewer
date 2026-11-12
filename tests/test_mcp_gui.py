"""内嵌 MCP 服务回归（GUI 设置 / 权限 / HTTP 协议 / 官方客户端 / 实时会话）。

    QT_QPA_PLATFORM=offscreen python tests/test_mcp_gui.py

分五组验证：

1. 配置层：MCPSettings、INI 持久化、令牌、客户端 mcp.json 合并写入；
2. 服务层策略：工具白名单、只读模式、写权限、像素上限、默认值、实时会话绑定；
3. HTTP 协议：鉴权、会话、通知、错误码、SSE、绝对 URL、健康检查、启停；
4. 官方 MCP SDK 的 streamable_http 客户端端到端；
5. Qt 界面：设置页、菜单/工具条/状态栏、Apply 生效、AI 调用改动当前窗口场景。
"""

from __future__ import annotations

import asyncio
import http.client
import inspect
import json
import os
import shutil
import socket
import sys
import tempfile
import threading
import time
import urllib.request

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from stl_viewer.mcp_config import (DEFAULT_PORT, MCPSettings,          # noqa: E402
                                   client_config_preview, default_client_config_path,
                                   generate_token, port_available,
                                   read_client_config, remove_client_config,
                                   write_client_config)
from stl_viewer.mcp_http import MCPHTTPServer                          # noqa: E402
from stl_viewer.mcp_service import MCPService, TOOLS, ToolError        # noqa: E402
from stl_viewer.mcp_stdio import MCPStdioServer                        # noqa: E402
from stl_viewer.mesh import Mesh                                       # noqa: E402
from stl_viewer.settings import AppSettings                            # noqa: E402

PASSED, FAILED, SKIPPED = [], [], []


def check(name, condition, detail=""):
    (PASSED if condition else FAILED).append(name)
    print(f"  [{'PASS' if condition else 'FAIL'}] {name} {detail}")


def skip(name, reason):
    SKIPPED.append(name)
    print(f"  [SKIP] {name} {reason}")


def free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def http_request(port: int, method: str, path: str, body=None,
                 headers=None, timeout=6.0):
    """发一条原始 HTTP 请求；path 可传绝对 URL 以模拟代理式请求目标。"""
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=timeout)
    hdrs = {"Content-Type": "application/json",
            "Accept": "application/json, text/event-stream",
            "Authorization": "Bearer test-token"}
    hdrs.update(headers or {})
    payload = body
    if isinstance(body, (dict, list)):
        payload = json.dumps(body).encode("utf-8")
    elif isinstance(body, str):
        payload = body.encode("utf-8")
    try:
        conn.request(method, path, body=payload, headers=hdrs)
        resp = conn.getresponse()
        data = resp.read()
        return resp.status, {k.lower(): v for k, v in resp.getheaders()}, data
    finally:
        conn.close()


def rpc(_id, method, params=None):
    msg = {"jsonrpc": "2.0", "id": _id, "method": method}
    if params is not None:
        msg["params"] = params
    return msg


def parse(data: bytes):
    try:
        return json.loads(data.decode("utf-8"))
    except (ValueError, UnicodeDecodeError):
        return {}


# ==========================================================================
# 1. 配置层
# ==========================================================================
def test_config() -> str:
    print("\n=== 配置层 ===")
    tmp = tempfile.mkdtemp(prefix="stl_mcp_gui_")
    cfg = MCPSettings()
    check("默认关闭服务", cfg.enabled is False)
    check("默认端口与回环地址", cfg.port == DEFAULT_PORT and
          cfg.endpoint == f"http://127.0.0.1:{DEFAULT_PORT}/mcp")
    check("默认要求令牌、禁止写文件",
          cfg.require_token is True and cfg.allow_file_write is False)
    check("默认工具全部开放", len(cfg.enabled_tool_names(
        [t["name"] for t in TOOLS])) == len(TOOLS))

    cfg.host = "0.0.0.0"
    check("回环地址被强制保护", cfg.bind_host == "127.0.0.1" and
          "127.0.0.1" in cfg.endpoint)
    cfg.host = "127.0.0.1"
    token = cfg.ensure_token()
    check("令牌生成且幂等", len(token) >= 32 and cfg.ensure_token() == token)
    check("默认客户端配置路径", default_client_config_path().endswith(
        os.path.join(".workbuddy", "mcp.json")))

    bad = MCPSettings(port=80, workspace_root=tmp)
    try:
        bad.validate()
        check("非法端口被拒绝", False)
    except ValueError:
        check("非法端口被拒绝", True)
    bad = MCPSettings(workspace_root=os.path.join(tmp, "missing"))
    try:
        bad.validate()
        check("不存在的沙箱目录被拒绝", False)
    except ValueError:
        check("不存在的沙箱目录被拒绝", True)
    check("工具白名单过滤", MCPSettings(enabled_tools=["server_info", "bogus"])
          .enabled_tool_names(["server_info", "load_model"]) == ("server_info",))

    s = AppSettings()
    s.mcp.enabled = True
    s.mcp.port = 18901
    s.mcp.enabled_tools = ["server_info", "list_models"]
    s.mcp.token = "persisted-token"
    s.mcp.workspace_root = tmp
    ini = os.path.join(tmp, "settings.ini")
    s.save(ini)
    loaded = AppSettings().load(ini)
    check("INI 往返：布尔/整数/列表/字符串",
          loaded.mcp.enabled and loaded.mcp.port == 18901 and
          loaded.mcp.enabled_tools == ["server_info", "list_models"] and
          loaded.mcp.token == "persisted-token" and
          loaded.mcp.workspace_root == tmp)
    loaded.reset_to_factory()
    check("恢复出厂设置重置 MCP", loaded.mcp.enabled is False and
          loaded.mcp.port == DEFAULT_PORT and loaded.mcp.token == "")

    cfgp = os.path.join(tmp, "client", "mcp.json")
    os.makedirs(os.path.dirname(cfgp), exist_ok=True)
    with open(cfgp, "w", encoding="utf-8") as fh:
        json.dump({"mcpServers": {"other": {"command": "x"}}, "extra": 7}, fh)
    live = MCPSettings(port=18902, token="abc", workspace_root=tmp,
                       client_config_path=cfgp)
    preview = client_config_preview(live)
    check("配置预览是 HTTP + Bearer 形式",
          '"type": "http"' in preview and "18902/mcp" in preview and
          "Bearer abc" in preview)
    write_client_config(cfgp, live)
    data = read_client_config(cfgp)
    check("写入配置保留其它服务与顶层字段",
          "other" in data["mcpServers"] and data.get("extra") == 7 and
          data["mcpServers"]["stl-viewer"]["url"].endswith("18902/mcp"))
    check("写入前生成 .bak 备份", os.path.isfile(cfgp + ".bak"))
    check("移除配置只删本服务", remove_client_config(cfgp) and
          "stl-viewer" not in read_client_config(cfgp)["mcpServers"] and
          "other" in read_client_config(cfgp)["mcpServers"])
    with open(cfgp, "w", encoding="utf-8") as fh:
        fh.write("{broken json")
    try:
        read_client_config(cfgp)
        check("损坏的客户端配置被拒绝", False)
    except ValueError:
        check("损坏的客户端配置被拒绝", True)

    port = free_port()
    check("空闲端口可绑定", port_available("127.0.0.1", port))
    with socket.socket() as holder:
        holder.bind(("127.0.0.1", port))
        holder.listen(1)
        check("已占用端口被识别", not port_available("127.0.0.1", port))
    return tmp


# ==========================================================================
# 2. 服务层策略
# ==========================================================================
def test_service_policy(tmp: str) -> None:
    print("\n=== 服务层策略 ===")
    plain = MCPService(ROOT)
    check("独立服务默认开放 15 工具", len(plain.list_tools()) == 15 and
          plain.allow_file_write and plain.allow_scene_changes)
    info = plain.call_tool("server_info", {}).data
    check("server_info 报告独立会话与策略",
          info["live_session"] is False and
          info["max_image_pixels"] == plain.max_image_pixels)

    cfg = MCPSettings(workspace_root=tmp, enabled_tools=["server_info",
                                                          "list_objects"])
    svc = MCPService(settings=cfg)
    check("白名单同步到 tools/list", len(svc.list_tools()) == 2)
    try:
        svc.call_tool("load_model", {"path": "x.stl"})
        check("被禁用工具被拒绝", False)
    except ToolError:
        check("被禁用工具被拒绝", True)

    cfg2 = MCPSettings(workspace_root=tmp, allow_scene_changes=False)
    svc2 = MCPService(settings=cfg2)
    check("只读模式允许查询", "workspace_root" in
          svc2.call_tool("server_info", {}).data)
    for name, args in (("load_model", {"path": "x.stl"}),
                       ("set_view", {"view": "top"}),
                       ("clear_scene", {})):
        try:
            svc2.call_tool(name, args)
            check(f"只读模式拒绝 {name}", False)
        except ToolError:
            check(f"只读模式拒绝 {name}", True)

    cfg3 = MCPSettings(workspace_root=tmp, allow_file_write=False)
    svc3 = MCPService(settings=cfg3)
    for name, args in (("export_stl", {"path": "o.stl"}),
                       ("split_components", {}),
                       ("render_picture", {"path": "o.png", "width": 64,
                                           "height": 64})):
        try:
            svc3.call_tool(name, args)
            check(f"禁止写文件：{name}", False)
        except ToolError:
            check(f"禁止写文件：{name}", True)
    check("禁止写文件仍允许返回图片",
          svc3.call_tool("render_picture", {"width": 64, "height": 64}).image)

    cfg4 = MCPSettings(workspace_root=tmp, max_image_pixels=100_000,
                       default_units="inch", default_measure_mode="3d",
                       render_width=320, render_height=240)
    svc4 = MCPService(settings=cfg4)
    d = svc4.call_tool("server_info", {}).data["defaults"]
    check("默认单位/口径/尺寸生效",
          d["units"] == "inch" and d["measure_mode"] == "3d" and
          d["render_size"] == [320, 240])
    check("默认渲染尺寸生效", "320x240" in
          svc4.call_tool("render_picture", {}).summary)
    try:
        svc4.call_tool("render_picture", {"width": 800, "height": 600})
        check("自定义像素上限生效", False)
    except ToolError:
        check("自定义像素上限生效", True)

    # 实时会话绑定：服务直接用外部 Scene，改动与回调都可见
    from stl_viewer.scene import Scene
    shared = Scene()
    changes = []
    cfg5 = MCPSettings(workspace_root=ROOT)
    bound = MCPService(ROOT, settings=cfg5, scene=shared,
                       on_change=lambda name, args: changes.append(name))
    shutil.copy(os.path.join(ROOT, "samples", "drill_blank.stl"),
                os.path.join(tmp, "drill_blank.stl"))
    cfg5.workspace_root = tmp
    bound.root = tmp
    bound.call_tool("load_model", {"path": "drill_blank.stl"})
    check("绑定场景：载入模型进入外部 Scene",
          len(shared.objects) == 1 and shared.objects[0].triangle_count > 0)
    check("变更回调被调用", changes == ["load_model"])
    check("server_info 报告实时会话",
          bound.call_tool("server_info", {}).data["live_session"] is True)


# ==========================================================================
# 3 & 4. HTTP 协议 + 官方 SDK 客户端
# ==========================================================================
def test_http_protocol(tmp: str) -> None:
    print("\n=== HTTP 协议 ===")
    port = free_port()
    cfg = MCPSettings(enabled=True, port=port, token="test-token",
                      require_token=True, workspace_root=tmp,
                      allow_file_write=True)
    svc = MCPService(settings=cfg)
    logs = []
    http = MCPHTTPServer(cfg, MCPStdioServer(svc).handle, on_log=logs.append)
    check("服务启动", http.start() is None)
    try:
        status, _, body = http_request(port, "GET", "/health",
                                       headers={"Authorization": ""})
        check("health 端点返回运行信息",
              status == 200 and parse(body)["status"] == "running")
        status, _, _ = http_request(port, "POST", "/mcp", rpc(1, "ping"),
                                    headers={"Authorization": "Bearer wrong"})
        check("错误令牌返回 401", status == 401)
        status, _, body = http_request(port, "POST", "/mcp", rpc(2, "ping"),
                                       headers={"Authorization": "",
                                                "X-API-Key": "test-token"})
        check("X-API-Key 也可鉴权", status == 200 and parse(body)["id"] == 2)
        status, _, _ = http_request(port, "POST", "/mcp", rpc(3, "ping"),
                                    headers={"Origin": "http://evil.example"})
        check("非法 Origin 被拒绝", status == 403)
        status, _, _ = http_request(port, "POST", "/mcp", rpc(4, "ping"),
                                    headers={"Host": "evil.example"})
        check("非法 Host 被拒绝", status == 403)

        status, hdrs, body = http_request(port, "POST", "/mcp", rpc(5, "initialize", {
            "protocolVersion": "2025-06-18", "capabilities": {},
            "clientInfo": {"name": "probe", "version": "1"}}))
        init = parse(body)
        check("initialize 握手与能力", status == 200 and
              init["result"]["serverInfo"]["name"] == "stl-viewer" and
              "tools" in init["result"]["capabilities"])
        check("initialize 返回会话 ID", bool(hdrs.get("mcp-session-id")))
        session = hdrs.get("mcp-session-id", "")
        status, _, _ = http_request(port, "POST", "/mcp", rpc(6, "ping"),
                                    headers={"Mcp-Session-Id": "bogus"})
        check("失效会话返回 404", status == 404)
        status, _, body = http_request(port, "POST", "/mcp", rpc(7, "ping"),
                                       headers={"Mcp-Session-Id": session})
        check("有效会话可继续请求", status == 200 and parse(body)["id"] == 7)

        status, _, body = http_request(port, "POST", "/mcp",
                                       {"jsonrpc": "2.0",
                                        "method": "notifications/initialized"})
        check("通知返回 202 且无 body", status == 202 and not body)
        status, _, body = http_request(port, "POST", "/mcp", rpc(8, "tools/list"))
        check("tools/list 返回 15 工具", status == 200 and
              len(parse(body)["result"]["tools"]) == 15)
        status, _, body = http_request(
            port, "POST", "/mcp",
            rpc(9, "tools/call", {"name": "server_info", "arguments": {}}))
        check("tools/call 正常执行", status == 200 and
              parse(body)["result"]["isError"] is False and
              "workspace_root" in parse(body)["result"]["content"][0]["text"])
        status, _, body = http_request(
            port, "POST", "/mcp",
            rpc(10, "tools/call", {"name": "load_model",
                                   "arguments": {"path": "missing.stl"}}))
        check("业务错误用 isError 返回",
              parse(body)["result"]["isError"] is True and
              "文件不存在" in parse(body)["result"]["content"][0]["text"])
        status, _, body = http_request(port, "POST", "/mcp", rpc(11, "nope"))
        check("未知方法返回 -32601", parse(body).get("error", {}).get("code") == -32601)
        status, _, body = http_request(port, "POST", "/mcp", body="not json")
        check("解析错误返回 400 / -32700", status == 400 and
              parse(body).get("error", {}).get("code") == -32700)
        status, _, body = http_request(port, "POST", "/mcp", rpc(12, "ping"),
                                       headers={"Accept": "text/event-stream"})
        check("只接受 SSE 时返回事件流", status == 200 and
              b"event: message" in body and b"data:" in body)
        absolute = f"http://127.0.0.1:{port}/mcp"
        status, _, body = http_request(port, "POST", absolute, rpc(13, "ping"))
        check("兼容代理式绝对 URL 请求目标",
              status == 200 and parse(body).get("id") == 13)
        status, _, _ = http_request(port, "DELETE", "/mcp",
                                    headers={"Mcp-Session-Id": session})
        check("DELETE 结束会话返回 204", status == 204)
        check("请求计数与日志", http.request_count >= 12 and
              len(logs) >= 12 and any("POST /mcp" in x or "→" in x for x in logs))
        check("status() 报告运行态", http.status()["status"] == "running" and
              http.status()["live_session"] is True)

        test_sdk_client(port)
    finally:
        http.stop()
        check("服务停止后释放端口",
              not http.running and port_available("127.0.0.1", port))


def test_sdk_client(port: int) -> None:
    print("\n=== 官方 SDK HTTP 客户端 ===")
    try:
        import httpx2
        from mcp import ClientSession
        from mcp.client.streamable_http import streamable_http_client
    except ImportError as exc:
        skip("SDK streamable_http 客户端", f"未安装 SDK: {exc}")
        return
    url = f"http://127.0.0.1:{port}/mcp"
    params = inspect.signature(streamable_http_client).parameters
    results = {}

    async def run():
        client = None
        if "headers" in params:
            cm = streamable_http_client(
                url, headers={"Authorization": "Bearer test-token"})
        else:
            client = httpx2.AsyncClient(
                headers={"Authorization": "Bearer test-token"}, timeout=10.0)
            cm = streamable_http_client(url, http_client=client)
        async with cm as (reader, writer):
            async with ClientSession(reader, writer) as session:
                init = await session.initialize()
                info = getattr(init, "server_info", None) or \
                    getattr(init, "serverInfo", None)
                results["name"] = info.name
                results["tools"] = len((await session.list_tools()).tools)
                results["resources"] = len((await session.list_resources()).resources)
                results["prompts"] = len((await session.list_prompts()).prompts)
                res = await session.call_tool("server_info", {})
                results["call"] = res.content[0].text
                pic = await session.call_tool("render_picture",
                                              {"width": 80, "height": 60})
                results["image"] = any(p.type == "image" for p in pic.content)
                await session.send_ping()
        if client is not None:
            await client.aclose()

    try:
        asyncio.run(run())
    except Exception as exc:                                  # noqa: BLE001
        check("SDK 客户端端到端", False, f"{type(exc).__name__}: {exc}")
        return
    check("SDK 握手", results.get("name") == "stl-viewer")
    check("SDK tools/list", results.get("tools") == 15)
    check("SDK resources/prompts",
          results.get("resources") == 4 and results.get("prompts") == 2)
    check("SDK tools/call", "workspace_root" in results.get("call", ""))
    check("SDK 返回图片内容", results.get("image") is True)


# ==========================================================================
# 5. Qt 界面与实时会话
# ==========================================================================
def test_gui(tmp: str) -> None:
    print("\n=== Qt 界面与实时会话 ===")
    from PyQt5 import QtCore, QtWidgets

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(["test"])
    from stl_viewer.mcp_dialog import MCPSettingsDialog
    from stl_viewer.viewer import MainWindow, ViewerState

    ini = os.path.join(tmp, "gui.ini")
    cfgp = os.path.join(tmp, "gui_mcp.json")
    state = ViewerState(ini)
    port = free_port()
    cfg = state.settings.mcp
    cfg.enabled = False
    cfg.port = port
    cfg.token = "gui-token"
    cfg.require_token = True
    cfg.workspace_root = ROOT
    cfg.client_config_path = cfgp
    cfg.enabled_tools = ["server_info", "list_objects", "load_model"]
    window = MainWindow(state)
    window.show()
    app.processEvents()

    check("菜单含设置与启停动作",
          "mcp_settings" in window.acts and "mcp_toggle" in window.acts)
    check("工具条与状态栏有 MCP 指示",
          window.mcp_button is not None and window.mcp_status is not None and
          window.mcp_button.text() == window.state.tr.tr("mcp_off"))
    check("默认不自动启动服务", not window.mcp.running)

    dialog = MCPSettingsDialog(window, window.mcp)
    check("设置页四个分类", dialog.nav.count() == 4 and
          dialog.stack.count() == 4)
    check("工具白名单列出 15 项", dialog.tool_list.count() == 15)
    checked = sum(dialog.tool_list.item(i).checkState() == QtCore.Qt.Checked
                  for i in range(dialog.tool_list.count()))
    check("白名单回填正确", checked == 3, f"{checked} 项勾选")
    check("参数回填正确", dialog.port.value() == port and
          dialog.root_edit.text() == ROOT and
          dialog.token_edit.text() == "gui-token")
    check("配置预览含令牌与端口",
          f":{port}/mcp" in dialog.preview.toPlainText() and
          "Bearer gui-token" in dialog.preview.toPlainText())

    dialog.enable_box.setChecked(True)
    check("Apply 成功", dialog._apply())
    check("Apply 后服务运行", window.mcp.running and
          window.state.settings.mcp.enabled)
    check("状态栏/工具条同步为运行",
          window.mcp_button.text() == window.state.tr.tr("mcp_on") and
          window.acts["mcp_toggle"].isChecked())
    reloaded = AppSettings().load(ini)
    check("启用状态写入 INI",
          reloaded.mcp.enabled and reloaded.mcp.port == port and
          reloaded.mcp.token == cfg.token)
    check("服务按白名单暴露 3 个工具",
          len(window.mcp._service.allowed_tools) == 3)

    QtWidgets.QMessageBox.information = staticmethod(lambda *a, **k: None)
    dialog._write_config()
    entry = read_client_config(cfgp)["mcpServers"]["stl-viewer"]
    check("一键写入客户端配置",
          entry["type"] == "http" and entry["url"].endswith(f":{port}/mcp") and
          entry["headers"]["Authorization"] == "Bearer gui-token")

    def post(message, timeout=20.0):
        """HTTP 请求放后台线程，主线程继续跑 Qt 事件循环。"""
        out = {}
        payload = json.dumps(message).encode("utf-8")

        def worker():
            req = urllib.request.Request(
                cfg.endpoint, data=payload,
                headers={"Content-Type": "application/json",
                         "Accept": "application/json, text/event-stream",
                         "Authorization": "Bearer gui-token"})
            try:
                with urllib.request.urlopen(req, timeout=timeout) as resp:
                    out["data"] = json.loads(resp.read().decode("utf-8"))
            except Exception as exc:                      # noqa: BLE001
                out["error"] = str(exc)

        thread = threading.Thread(target=worker, daemon=True)
        thread.start()
        deadline = time.monotonic() + timeout
        while thread.is_alive() and time.monotonic() < deadline:
            app.processEvents()
            time.sleep(0.01)
        thread.join(timeout=1)
        return out

    reply = post(rpc(1, "tools/call", {"name": "load_model",
                                       "arguments": {"path": "samples/drill_blank.stl"}}))
    ok = "data" in reply and not reply["data"]["result"]["isError"]
    check("AI 调用 load_model 成功", ok,
          str(reply.get("error", ""))[:80])
    check("AI 载入的模型出现在当前窗口",
          len(state.scene.objects) == 1 and window.panel.list.count() == 1)
    app.processEvents()
    check("窗口状态栏统计已刷新",
          "1 obj" in window.status_label.text())

    reply = post(rpc(2, "tools/call", {"name": "list_objects", "arguments": {}}))
    check("AI 读到当前窗口的对象清单",
          "data" in reply and "drill_blank" in
          reply["data"]["result"]["content"][0]["text"])

    reply = post(rpc(3, "tools/list"))
    check("HTTP 层只暴露白名单工具",
          "data" in reply and len(reply["data"]["result"]["tools"]) == 3)
    reply = post(rpc(4, "tools/call", {"name": "render_picture",
                                       "arguments": {"width": 80, "height": 60}}))
    check("白名单外的工具被拒绝",
          "data" in reply and reply["data"]["result"]["isError"] is True)

    # 放开全部工具后，重工具（后台快照路径）也要能跑通并返回图片
    cfg.enabled_tools = []
    window.mcp.restart()
    app.processEvents()
    reply = post(rpc(5, "tools/call", {"name": "render_picture",
                                       "arguments": {"width": 80, "height": 60}}))
    parts = reply.get("data", {}).get("result", {}).get("content", [])
    check("耗时工具用场景快照在后台执行",
          any(p.get("type") == "image" for p in parts),
          str(reply.get("error", ""))[:80])

    # 官方 SDK 客户端连到「正在运行的 GUI」，验证整条真实链路
    sdk_box = {}
    t = threading.Thread(target=_sdk_against_gui,
                         args=(port, "gui-token", sdk_box), daemon=True)
    t.start()
    deadline = time.monotonic() + 30
    while t.is_alive() and time.monotonic() < deadline:
        app.processEvents()
        time.sleep(0.01)
    t.join(timeout=1)
    check("官方 SDK 客户端连上 GUI 内嵌服务", sdk_box.get("ok") is True,
          str(sdk_box.get("error", ""))[:100])
    check("SDK 客户端的调用改动当前窗口场景",
          len(state.scene.objects) == 2, f"{len(state.scene.objects)} 个对象")

    toggle = window.acts["mcp_toggle"]
    toggle.trigger()
    app.processEvents()
    check("菜单开关停用服务", not window.mcp.running and
          not window.state.settings.mcp.enabled and
          window.mcp_button.text() == window.state.tr.tr("mcp_off"))
    toggle.trigger()
    app.processEvents()
    check("菜单开关重新启动服务", window.mcp.running and
          window.state.settings.mcp.enabled)

    window.set_language("CH")
    app.processEvents()
    check("中文界面下 MCP 菜单已翻译",
          "MCP" in window.acts["mcp_settings"].text() and
          "设置" in window.acts["mcp_settings"].text())

    dialog.close()
    window.close()
    app.processEvents()
    check("关闭窗口自动停止服务", not window.mcp.running)


def _sdk_against_gui(port: int, token: str, box: dict) -> None:
    """在后台线程用官方客户端连 GUI 服务，并载入第二个模型。"""
    try:
        import httpx2
        from mcp import ClientSession
        from mcp.client.streamable_http import streamable_http_client
        url = f"http://127.0.0.1:{port}/mcp"
        params = inspect.signature(streamable_http_client).parameters

        async def run():
            client = None
            if "headers" in params:
                cm = streamable_http_client(
                    url, headers={"Authorization": f"Bearer {token}"})
            else:
                client = httpx2.AsyncClient(
                    headers={"Authorization": f"Bearer {token}"}, timeout=10.0)
                cm = streamable_http_client(url, http_client=client)
            async with cm as (reader, writer):
                async with ClientSession(reader, writer) as session:
                    await session.initialize()
                    res = await session.call_tool(
                        "load_model", {"path": "samples/drill_blank_ascii.stl",
                                       "name": "from_ai"})
                    failed = getattr(res, "is_error", None)
                    if failed is None:
                        failed = getattr(res, "isError", False)
                    box["loaded"] = not failed
            if client is not None:
                await client.aclose()

        asyncio.run(run())
        box["ok"] = True
    except Exception as exc:                                  # noqa: BLE001
        box["ok"] = False
        box["error"] = f"{type(exc).__name__}: {exc}"


def main() -> int:
    tmp = test_config()
    try:
        test_service_policy(tmp)
        test_http_protocol(tmp)
        test_gui(tmp)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print(f"\n总计: {len(PASSED)} 通过, {len(FAILED)} 失败, "
          f"{len(SKIPPED)} 跳过")
    if FAILED:
        print("失败项: " + ", ".join(FAILED))
        return 1
    print("内嵌 MCP 服务回归通过。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
