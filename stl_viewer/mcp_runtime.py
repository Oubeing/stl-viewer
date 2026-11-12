"""MCP 服务的 GUI 运行时：启停、跨线程调度、实时会话同步。

HTTP 服务跑在后台线程，而场景对象属于 GUI 线程。所有会读写场景的工具调用
都通过 ``_CallBridge`` 排队到 GUI 线程执行，避免渲染与 AI 改动互相踩数据；
计算密集的工具（渲染 / 统计 / 导出）先在 GUI 线程做一份场景快照，再回到
后台线程执行，因此 AI 出图时窗口仍然可以拖动、切视角、点菜单。
"""

from __future__ import annotations

import json
import threading
import time
import urllib.request
from collections import deque
from typing import Callable

from PyQt5 import QtCore

from .mcp_config import HEAVY_TOOLS, port_available
from .mcp_http import MCPHTTPServer
from .mcp_service import MCPService, ToolError
from .mcp_stdio import MCPStdioServer

GUI_CALL_TIMEOUT = 60.0


class _CallBridge(QtCore.QObject):
    """把可调用对象投递到对象所属线程（GUI 线程）并等待结果。"""

    requested = QtCore.pyqtSignal(object)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.requested.connect(self._execute, QtCore.Qt.AutoConnection)

    @QtCore.pyqtSlot(object)
    def _execute(self, job) -> None:
        fn, box = job
        try:
            box["result"] = fn()
        except BaseException as exc:                  # noqa: BLE001
            box["error"] = exc
        finally:
            box["event"].set()


class MCPController(QtCore.QObject):
    """管理内嵌 MCP 服务的完整生命周期。"""

    statusChanged = QtCore.pyqtSignal(bool, str)     # running, endpoint
    logLine = QtCore.pyqtSignal(str)
    requestCount = QtCore.pyqtSignal(int)
    sessionChanged = QtCore.pyqtSignal(str)          # 触发界面刷新的工具名

    def __init__(self, state, parent=None):
        super().__init__(parent)
        self.state = state
        self._bridge = _CallBridge(self)
        self._service: MCPService | None = None
        self._handler: MCPStdioServer | None = None
        self._http: MCPHTTPServer | None = None
        self._logs: deque = deque(maxlen=300)

    # ---------------- 配置 ----------------
    @property
    def cfg(self):
        """始终从 AppSettings 取最新对象（恢复出厂设置会替换它）。"""
        return self.state.settings.mcp

    @property
    def running(self) -> bool:
        return bool(self._http and self._http.running)

    # ---------------- 生命周期 ----------------
    def start(self) -> tuple[bool, str]:
        """启动服务。返回 (是否成功, 地址或错误说明)。"""
        self.stop()
        cfg = self.cfg
        if not cfg.enabled:
            return False, "服务未启用"
        try:
            cfg.validate()
        except (ValueError, OSError) as exc:
            return False, str(exc)
        if cfg.require_token:
            cfg.ensure_token()
        if not port_available(cfg.bind_host, int(cfg.port)):
            return False, f"端口 {cfg.port} 已被占用，请在设置页更换端口"

        try:
            service = MCPService(
                cfg.resolved_root(), settings=cfg,
                scene=self.state.scene, camera=self.state.camera,
                measures=self.state.measures, options=self.state.options,
                units=self.state.units, on_change=self._service_changed)
        except Exception as exc:                     # noqa: BLE001
            return False, f"服务初始化失败: {exc}"

        http = MCPHTTPServer(cfg, self.dispatch, on_log=self._log,
                             on_request=self._on_request)
        error = http.start()
        if error:
            return False, error
        self._service, self._http = service, http
        self._handler = MCPStdioServer(service)
        self._log(f"service ready: {cfg.endpoint} "
                  f"(live session, {len(service.allowed_tools)} tools)")
        self.statusChanged.emit(True, cfg.endpoint)
        return True, cfg.endpoint

    def stop(self) -> None:
        http = self._http
        self._http = self._handler = self._service = None
        if http is not None:
            http.stop()
        self.statusChanged.emit(False, "")

    def restart(self) -> tuple[bool, str]:
        if not self.cfg.enabled:
            self.stop()
            return False, "服务未启用"
        return self.start()

    def set_enabled(self, enabled: bool) -> tuple[bool, str]:
        """界面开关：立即启停，并把状态写回设置对象（由调用方保存）。"""
        self.cfg.enabled = bool(enabled)
        if enabled:
            return self.start()
        self.stop()
        return False, "服务已停用"

    def reload(self) -> tuple[bool, str]:
        """载入 / 恢复默认设置后重新应用（端口、令牌、权限都可能变了）。"""
        return self.restart()

    def update_policy(self) -> None:
        """只刷新策略与默认值，不中断正在运行的连接。"""
        if self._service is not None:
            self._service.configure(self.cfg)

    # ---------------- 状态 ----------------
    def status(self) -> dict:
        if self._http is not None:
            return self._http.status()
        return {"status": "stopped", "server": "stl-viewer",
                "endpoint": self.cfg.endpoint, "live_session": False,
                "requests": 0, "sessions": 0, "last_request": None,
                "error": None}

    def logs(self) -> list[str]:
        return list(self._logs)

    def clear_logs(self) -> None:
        self._logs.clear()

    def probe(self, timeout: float = 1.5) -> tuple[bool, str]:
        """测试本机 health 端点是否可达（设置页「测试连接」）。"""
        # 界面里可能刚改了端口但还没 Apply，探测必须打向真正在跑的那个服务
        endpoint = self.status()["endpoint"]
        url = endpoint.replace("/mcp", "/health") if self.running \
            else self.cfg.health_url
        try:
            with urllib.request.urlopen(url, timeout=timeout) as r:
                data = json.loads(r.read().decode("utf-8"))
            return True, f"{data.get('status')} · {data.get('requests', 0)} 次请求"
        except Exception as exc:                     # noqa: BLE001
            return False, str(exc)

    # ---------------- 回调 ----------------
    def _log(self, text: str) -> None:
        stamp = time.strftime("%H:%M:%S")
        line = f"[{stamp}] {text}"
        self._logs.append(line)
        self.logLine.emit(line)

    def _on_request(self) -> None:
        if self._http is not None:
            self.requestCount.emit(self._http.request_count)

    def _service_changed(self, name: str, args: dict) -> None:
        self.sessionChanged.emit(name)

    # ---------------- 跨线程调度 ----------------
    def call_on_gui(self, fn: Callable, timeout: float = GUI_CALL_TIMEOUT):
        """在 GUI 线程执行 fn 并等待结果；已在 GUI 线程时直接执行。"""
        if QtCore.QThread.currentThread() == self.thread():
            return fn()
        box = {"event": threading.Event()}
        self._bridge.requested.emit((fn, box))
        if not box["event"].wait(timeout):
            raise ToolError(f"界面线程响应超时（{timeout:.0f} 秒）")
        if "error" in box:
            raise box["error"]
        return box["result"]

    def _make_snapshot(self) -> MCPService:
        """在 GUI 线程生成只读会话快照，供后台线程做重活。"""
        cfg = self.cfg
        snap = MCPService(cfg.resolved_root(), settings=cfg)
        snap.scene.objects = list(self.state.scene.objects)
        active = self.state.scene.active_index
        if 0 <= active < len(snap.scene.objects):
            snap.scene.set_active(active)
        snap.scene.background = self.state.scene.background
        snap.scene.background_top = self.state.scene.background_top
        snap.scene.gradient_background = self.state.scene.gradient_background
        snap.camera = self.state.camera.copy()
        snap.measures.items = list(self.state.measures.items)
        snap.options = self.state.options
        snap.units = self.state.units
        return snap

    def dispatch(self, message: dict) -> dict | None:
        """HTTP 线程上的 JSON-RPC 分发入口。"""
        method = message.get("method") if isinstance(message, dict) else None
        params = message.get("params") or {} if isinstance(message, dict) else {}
        try:
            heavy = (method == "tools/call" and
                     isinstance(params, dict) and
                     params.get("name") in HEAVY_TOOLS)
            if heavy:
                # 快照在 GUI 线程生成，实际计算留在当前后台线程。
                snapshot = self.call_on_gui(self._make_snapshot)
                return MCPStdioServer(snapshot).handle(message)
            if self._handler is None:
                return {"jsonrpc": "2.0", "id": message.get("id"),
                        "error": {"code": -32603, "message": "服务尚未启动"}}
            return self.call_on_gui(lambda: self._handler.handle(message))
        except ToolError as exc:
            self._log(f"dispatch rejected: {exc}")
            return {"content": [{"type": "text", "text": f"错误: {exc}"}],
                    "isError": True}
