"""内嵌 MCP 服务（Streamable HTTP 传输）。

它随 STL Viewer 主窗口一起启停，AI 客户端通过
``http://127.0.0.1:<port>/mcp`` 连接，直接操作**当前窗口**里的场景：
载入模型、切视角、测量、渲染图片，窗口会同步刷新。

协议要点（MCP 2025-06-18 Streamable HTTP）：

* ``POST /mcp``：一条 JSON-RPC 消息，响应为 ``application/json``（或客户端
  只接受 SSE 时的 ``text/event-stream``）；通知类消息返回 ``202`` 且无 body；
* ``GET /mcp``：服务器主动推送用的 SSE 长连接，本服务不发服务端请求，
  因此只保持心跳；
* ``DELETE /mcp``：结束会话；
* ``initialize`` 响应带 ``Mcp-Session-Id``，后续请求可带回。

安全：只绑定回环地址、校验 Host/Origin（防 DNS rebinding）、Bearer 令牌
（``Authorization`` 或 ``X-API-Key``）、请求体大小上限。日志不写 stdout。
"""

from __future__ import annotations

import json
import secrets
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Callable
from urllib.parse import urlsplit

MCP_PATH = "/mcp"
HEALTH_PATH = "/health"
SESSION_HEADER = "Mcp-Session-Id"
MAX_BODY_BYTES = 8 * 1024 * 1024
LOCAL_HOSTS = ("127.0.0.1", "localhost", "::1")

# JSON-RPC 2.0 错误码
PARSE_ERROR = -32700
INVALID_REQUEST = -32600
INTERNAL_ERROR = -32603


def _json(data: Any) -> bytes:
    return json.dumps(data, ensure_ascii=False, separators=(",", ":")).encode("utf-8")


class _Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    server_version = "STLViewerMCP/1.0"

    # BaseHTTPRequestHandler 默认把每个请求打到 stderr；GUI 里改走设置页日志。
    def log_message(self, fmt: str, *args) -> None:
        pass

    @property
    def ctl(self) -> "MCPHTTPServer":
        return self.server.controller                     # type: ignore[attr-defined]

    # ---------------- 响应工具 ----------------
    def _send(self, status: int, body: bytes = b"", ctype: str | None = None,
              extra: dict | None = None) -> None:
        self.send_response(status)
        if ctype:
            self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        for key, value in (extra or {}).items():
            self.send_header(key, value)
        self.end_headers()
        if body:
            self.wfile.write(body)

    def _json_response(self, status: int, payload: Any,
                       extra: dict | None = None) -> None:
        self._send(status, _json(payload), "application/json; charset=utf-8", extra)

    @staticmethod
    def _endpoint(raw_path: str) -> str:
        """兼容代理式绝对 URL（官方 SDK 2.x 会发完整 URL 作为请求目标）。"""
        if raw_path.startswith(("http://", "https://")):
            return urlsplit(raw_path).path
        return raw_path.split("?", 1)[0]

    # ---------------- 安全检查 ----------------
    def _local_request(self) -> bool:
        host = self.headers.get("Host", "").split(":")[0].strip().lower()
        if host and host.strip("[]") not in LOCAL_HOSTS:
            return False
        origin = self.headers.get("Origin")
        if origin:
            return any(origin.startswith(f"http://{h}") or
                       origin.startswith(f"https://{h}") for h in LOCAL_HOSTS[:2])
        return True

    def _authorized(self) -> bool:
        cfg = self.ctl.settings
        if not cfg.require_token:
            return True
        if not cfg.token:
            return False
        auth = self.headers.get("Authorization", "")
        api_key = self.headers.get("X-API-Key", "")
        return (secrets.compare_digest(auth, f"Bearer {cfg.token}") or
                secrets.compare_digest(api_key, cfg.token))

    def _guard(self) -> bool:
        """返回 False 表示已写好拒绝响应，调用方直接 return。"""
        if not self._local_request():
            self._send(403, _json({"error": "只允许本机访问"}),
                       "application/json; charset=utf-8")
            return False
        if not self._authorized():
            self._send(401, _json({"error": "访问令牌无效"}),
                       "application/json; charset=utf-8",
                       {"WWW-Authenticate": 'Bearer realm="stl-viewer"'})
            return False
        return True

    # ---------------- POST ----------------
    def do_POST(self) -> None:                                  # noqa: N802
        path = self._endpoint(self.path)
        # 先把请求体读完，哪怕这个路径 / 鉴权会被拒绝：keep-alive 连接上
        # 残留的 body 会被当成下一条请求行，导致协议流彻底错乱。
        try:
            length = int(self.headers.get("Content-Length") or 0)
        except ValueError:
            length = 0
        if length > MAX_BODY_BYTES:
            self.close_connection = True
            self._send(413, b"request too large", "text/plain; charset=utf-8")
            return
        raw = self.rfile.read(length) if length else b""
        if path != MCP_PATH:
            self._json_response(404, {"jsonrpc": "2.0", "id": None,
                                     "error": {"code": INVALID_REQUEST,
                                               "message": "未知端点"}})
            return
        if not self._guard():
            self.ctl.record("POST /mcp", 401, "鉴权失败")
            return
        try:
            message = json.loads(raw.decode("utf-8"))
        except (ValueError, UnicodeDecodeError) as exc:
            self.ctl.record("POST /mcp", 400, "JSON 解析失败")
            self._json_response(400, {"jsonrpc": "2.0", "id": None,
                                     "error": {"code": PARSE_ERROR,
                                               "message": f"JSON 解析失败: {exc}"}})
            return

        messages = message if isinstance(message, list) else [message]
        if not isinstance(message, (dict, list)):
            self._json_response(400, {"jsonrpc": "2.0", "id": None,
                                     "error": {"code": INVALID_REQUEST,
                                               "message": "消息必须是对象或数组"}})
            return

        # 会话校验：带 session id 但已失效 → 404（MCP 规范）
        session = self.headers.get(SESSION_HEADER)
        if session and not self.ctl.has_session(session):
            self.ctl.record("POST /mcp", 404, "会话已失效")
            self._json_response(404, {"jsonrpc": "2.0", "id": None,
                                     "error": {"code": INVALID_REQUEST,
                                               "message": "会话已终止"}})
            return

        responses = []
        for item in messages:
            try:
                reply = self.ctl.dispatch(item)
            except Exception as exc:                       # noqa: BLE001
                self.ctl.log(f"dispatch error: {type(exc).__name__}: {exc}")
                msg_id = item.get("id") if isinstance(item, dict) else None
                reply = {"jsonrpc": "2.0", "id": msg_id,
                         "error": {"code": INTERNAL_ERROR,
                                   "message": f"{type(exc).__name__}: {exc}"}}
            if reply is not None:
                responses.append(reply)

        # initialize 时新建会话并回传 header
        extra = {}
        if isinstance(message, dict) and message.get("method") == "initialize":
            extra[SESSION_HEADER] = self.ctl.new_session()

        accept = (self.headers.get("Accept") or "").lower()
        method = message.get("method") if isinstance(message, dict) else "batch"
        if not responses:
            self.ctl.record(method, 202, "通知")
            self._send(202, b"", None, extra)
            return
        payload = responses if isinstance(message, list) else responses[0]
        if "application/json" in accept or not accept:
            self.ctl.record(method, 200, "")
            self._json_response(200, payload, extra)
            return
        # 客户端只接受 SSE：把响应包成一条 event
        chunks = []
        for reply in responses:
            chunks.append(b"event: message\ndata: " + _json(reply) + b"\n\n")
        self.ctl.record(method, 200, "SSE")
        self._send(200, b"".join(chunks), "text/event-stream; charset=utf-8",
                   {**extra, "Cache-Control": "no-cache"})

    # ---------------- GET ----------------
    def do_GET(self) -> None:                                   # noqa: N802
        path = self._endpoint(self.path)
        if path == HEALTH_PATH:
            self._json_response(200, self.ctl.status())
            return
        if path != MCP_PATH:
            self._send(404, b"not found", "text/plain; charset=utf-8")
            return
        if not self._guard():
            self.ctl.record("GET /mcp", 401, "鉴权失败")
            return
        # 服务器不主动发请求，但官方客户端会在 initialize 之后打开 GET 流；
        # 返回 405 会触发客户端反复重连，这里保持一条只发心跳的 SSE。
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream; charset=utf-8")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "keep-alive")
        self.end_headers()
        try:
            self.wfile.write(b": connected\n\n")
            self.wfile.flush()
            ticks = 0
            while self.ctl.running:
                time.sleep(0.5)
                ticks += 1
                if ticks % 30 == 0:
                    self.wfile.write(b": keepalive\n\n")
                    self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError, OSError):
            pass
        finally:
            self.close_connection = True

    # ---------------- DELETE ----------------
    def do_DELETE(self) -> None:                                # noqa: N802
        if self._endpoint(self.path) != MCP_PATH:
            self._send(404, b"not found", "text/plain; charset=utf-8")
            return
        if not self._guard():
            return
        session = self.headers.get(SESSION_HEADER)
        if session:
            self.ctl.drop_session(session)
        self._send(204)


class _ThreadingServer(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True


class MCPHTTPServer:
    """HTTP 服务的生命周期与状态；``dispatch`` 由 Qt 控制器提供。"""

    def __init__(self, settings, dispatch: Callable[[dict], dict | None],
                 on_log: Callable[[str], None] | None = None,
                 on_request: Callable[[], None] | None = None):
        self.settings = settings
        self.dispatch = dispatch
        self.on_log = on_log
        self.on_request = on_request
        self.sessions: set[str] = set()
        self._lock = threading.Lock()
        self.request_count = 0
        self.last_request: str | None = None
        self.last_activity: float | None = None
        self.running = False
        self.error: str | None = None
        self._httpd: _ThreadingServer | None = None
        self._thread: threading.Thread | None = None

    # ---------------- 生命周期 ----------------
    def start(self) -> str | None:
        """启动服务；成功返回 None，失败返回可读的错误文本。"""
        if self.running:
            return None
        host, port = self.settings.bind_host, int(self.settings.port)
        try:
            httpd = _ThreadingServer((host, port), _Handler)
        except OSError as exc:
            self.error = f"端口 {port} 无法绑定：{exc}"
            return self.error
        httpd.controller = self                              # type: ignore[attr-defined]
        self._httpd = httpd
        self.running = True
        self.error = None
        self._thread = threading.Thread(
            target=httpd.serve_forever, kwargs={"poll_interval": 0.2},
            name="stl-viewer-mcp-http", daemon=True)
        self._thread.start()
        self.log(f"listening on http://{host}:{port}/mcp")
        return None

    def stop(self) -> None:
        self.running = False
        httpd, thread = self._httpd, self._thread
        self._httpd = self._thread = None
        if httpd is not None:
            try:
                httpd.shutdown()
            except Exception:                                # noqa: BLE001
                pass
            try:
                httpd.server_close()
            except Exception:                                # noqa: BLE001
                pass
        if thread is not None and thread.is_alive():
            thread.join(timeout=2.0)
        with self._lock:
            self.sessions.clear()
        self.log("service stopped")

    # ---------------- 会话与状态 ----------------
    def new_session(self) -> str:
        sid = secrets.token_hex(16)
        with self._lock:
            self.sessions.add(sid)
        return sid

    def has_session(self, sid: str) -> bool:
        with self._lock:
            return sid in self.sessions

    def drop_session(self, sid: str) -> None:
        with self._lock:
            self.sessions.discard(sid)

    def status(self) -> dict:
        return {
            "status": "running" if self.running else "stopped",
            "server": "stl-viewer",
            "endpoint": self.settings.endpoint,
            "live_session": True,
            "requests": self.request_count,
            "sessions": len(self.sessions),
            "last_request": self.last_request,
            "error": self.error,
        }

    # ---------------- 日志 ----------------
    def log(self, text: str) -> None:
        if self.on_log is not None:
            try:
                self.on_log(text)
            except Exception:                                # noqa: BLE001
                pass

    def record(self, method: Any, code: int, detail: str) -> None:
        self.request_count += 1
        self.last_request = f"{time.strftime('%H:%M:%S')} {method} → {code}"
        self.last_activity = time.time()
        if self.settings.log_requests and self.on_log is not None:
            text = f"[{self.last_request}] {detail}".strip()
            try:
                self.on_log(text)
            except Exception:                                # noqa: BLE001
                pass
        if self.on_request is not None:
            try:
                self.on_request()
            except Exception:                                # noqa: BLE001
                pass
