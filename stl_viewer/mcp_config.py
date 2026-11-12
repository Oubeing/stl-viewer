"""MCP 服务配置与 AI 客户端注册。

图形界面中的「Extras → MCP Service Settings」直接读写 :class:`MCPSettings`，
由 ``AppSettings`` 持久化到 INI 的 ``[MCP]`` 段。服务只监听回环地址，
用 Bearer 令牌鉴权；``build_client_entry()`` 生成的条目格式与 WorkBuddy /
Claude Desktop 等客户端的 ``mcp.json`` 一致（``type: http`` + ``url`` +
``Authorization`` 请求头）。
"""

from __future__ import annotations

import json
import os
import secrets
import shutil
import tempfile
from dataclasses import asdict, dataclass, field, fields
from typing import Any

SERVER_KEY = "stl-viewer"
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8765
DEFAULT_MAX_IMAGE_PIXELS = 2_000_000
MAX_TOOL_IMAGE_PIXELS = 20_000_000

#: 会向磁盘写文件的工具（render_picture 仅在显式指定 path 时算写操作）
FILE_WRITE_TOOLS = ("export_stl", "split_components")
#: 会改变当前 GUI 会话的工具（只读模式下拒绝）
SCENE_CHANGE_TOOLS = (
    "load_model", "remove_object", "clear_scene", "set_view",
    "measure", "clear_measurements",
)
#: 计算密集、放到后台线程用场景快照执行的工具
HEAVY_TOOLS = ("inspect_model", "render_picture", "export_stl",
               "split_components", "section_radius")


def generate_token() -> str:
    """生成一个新的访问令牌（URL 安全，约 43 字符）。"""
    return secrets.token_urlsafe(32)


def default_client_config_path() -> str:
    """WorkBuddy 的 MCP 客户端配置文件；其它客户端可在界面里另行指定。"""
    return os.path.join(os.path.expanduser("~"), ".workbuddy", "mcp.json")


@dataclass
class MCPSettings:
    """内嵌 MCP 服务的全部可配置参数。"""

    enabled: bool = False                 # 服务总开关
    autostart: bool = True                # 随 GUI 启动自动拉起
    host: str = DEFAULT_HOST              # 仅允许回环地址
    port: int = DEFAULT_PORT
    token: str = ""                       # 空串表示尚未生成
    require_token: bool = True            # 是否要求 Bearer 令牌
    workspace_root: str = ""              # AI 可访问的沙箱根目录
    allow_scene_changes: bool = True      # 是否允许 AI 改动当前场景
    allow_file_write: bool = False        # 是否允许 AI 写文件
    max_image_pixels: int = DEFAULT_MAX_IMAGE_PIXELS
    default_units: str = "mm"
    default_measure_mode: str = "view"
    render_width: int = 640
    render_height: int = 480
    enabled_tools: list = field(default_factory=list)   # 空 = 开放全部工具
    client_config_path: str = ""
    log_requests: bool = True

    # ---------------- 派生值 ----------------
    @property
    def bind_host(self) -> str:
        """只允许回环地址，避免把会话控制权暴露到局域网。"""
        return self.host if self.host in ("127.0.0.1", "localhost") \
            else DEFAULT_HOST

    @property
    def endpoint(self) -> str:
        return f"http://{self.bind_host}:{int(self.port)}/mcp"

    @property
    def health_url(self) -> str:
        return f"http://{self.bind_host}:{int(self.port)}/health"

    def resolved_root(self) -> str:
        raw = self.workspace_root or os.getcwd()
        return os.path.realpath(os.path.expanduser(os.path.expandvars(raw)))

    def resolved_client_config_path(self) -> str:
        raw = self.client_config_path or default_client_config_path()
        return os.path.abspath(os.path.expanduser(os.path.expandvars(raw)))

    def ensure_token(self) -> str:
        """尚无令牌时生成一个；已有令牌原样返回。"""
        if not self.token:
            self.token = generate_token()
        return self.token

    def enabled_tool_names(self, all_names) -> tuple:
        """工具白名单；空列表表示开放全部，未知名称自动忽略。"""
        if not self.enabled_tools:
            return tuple(all_names)
        allowed = set(str(x) for x in self.enabled_tools)
        return tuple(n for n in all_names if n in allowed)

    # ---------------- 校验 ----------------
    def validate(self) -> None:
        if not 1024 <= int(self.port) <= 65535:
            raise ValueError("端口必须为 1024–65535")
        if not os.path.isdir(self.resolved_root()):
            raise ValueError(f"工作区根目录不存在: {self.resolved_root()}")
        if not 100_000 <= int(self.max_image_pixels) <= MAX_TOOL_IMAGE_PIXELS:
            raise ValueError("图片像素上限必须在 100000–20000000 之间")
        if not (64 <= int(self.render_width) <= 2048 and
                64 <= int(self.render_height) <= 2048):
            raise ValueError("默认渲染尺寸必须在 64–2048 之间")
        if self.default_units not in ("mm", "inch"):
            raise ValueError("单位只能是 mm 或 inch")
        if self.default_measure_mode not in ("view", "3d"):
            raise ValueError("测量口径只能是 view 或 3d")

    # ---------------- 序列化 ----------------
    def state(self) -> dict:
        return asdict(self)

    def set_state(self, state: dict) -> None:
        """从 INI 的 [MCP] 段恢复；类型按 dataclass 字段自动转换。"""
        for f in fields(self):
            if f.name not in state:
                continue
            current = getattr(self, f.name)
            raw = state[f.name]
            if isinstance(current, bool):
                value = _to_bool(raw, current)
            elif isinstance(current, int):
                value = int(_to_float(raw, current))
            elif isinstance(current, list):
                if isinstance(raw, (list, tuple)):
                    items = raw
                else:
                    items = str(raw).split(",")
                value = [str(x).strip() for x in items if str(x).strip()]
            else:
                value = str(raw).strip()
            setattr(self, f.name, value)


# --------------------------------------------------------------------------
# AI 客户端配置（mcp.json）
# --------------------------------------------------------------------------
def build_client_entry(settings: MCPSettings) -> dict:
    """生成一条可直接使用的 HTTP MCP 服务配置。"""
    entry: dict[str, Any] = {"type": "http", "url": settings.endpoint}
    if settings.require_token:
        entry["headers"] = {"Authorization": f"Bearer {settings.ensure_token()}"}
    return entry


def client_config_preview(settings: MCPSettings, name: str = SERVER_KEY) -> str:
    """供界面预览 / 复制的单条配置片段。"""
    return json.dumps({"mcpServers": {name: build_client_entry(settings)}},
                      ensure_ascii=False, indent=2)


def read_client_config(path: str) -> dict:
    if not os.path.isfile(path):
        return {"mcpServers": {}}
    try:
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"客户端配置文件无法解析: {path}\n{exc}") from exc
    if not isinstance(data, dict):
        raise ValueError(f"客户端配置文件根节点不是对象: {path}")
    if "mcpServers" not in data:
        data["mcpServers"] = {}
    if not isinstance(data["mcpServers"], dict):
        raise ValueError(f"mcpServers 不是对象: {path}")
    return data


def write_client_config(path: str, settings: MCPSettings,
                        name: str = SERVER_KEY) -> str:
    """合并写入一条服务配置，保留文件中已有的其它服务与顶层字段。

    写入前生成 ``.bak`` 备份；用同目录临时文件 + ``os.replace`` 原子替换，
    避免客户端在写文件的瞬间读到半截 JSON。
    """
    path = os.path.abspath(path)
    data = read_client_config(path)
    data["mcpServers"][name] = build_client_entry(settings)
    parent = os.path.dirname(path)
    os.makedirs(parent, exist_ok=True)
    if os.path.isfile(path):
        try:
            shutil.copy2(path, path + ".bak")
        except OSError:                          # 备份失败不阻断写入
            pass
    fd, tmp = tempfile.mkstemp(prefix=".mcp-", suffix=".json", dir=parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(data, fh, ensure_ascii=False, indent=2)
            fh.write("\n")
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)
    return path


def remove_client_config(path: str, name: str = SERVER_KEY) -> bool:
    """从客户端配置中移除本服务条目；返回是否真的删掉了。"""
    if not os.path.isfile(path):
        return False
    data = read_client_config(path)
    if name not in data["mcpServers"]:
        return False
    del data["mcpServers"][name]
    parent = os.path.dirname(path) or "."
    parent = os.path.abspath(parent)
    shutil.copy2(path, path + ".bak")
    fd, tmp = tempfile.mkstemp(prefix=".mcp-", suffix=".json", dir=parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(data, fh, ensure_ascii=False, indent=2)
            fh.write("\n")
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)
    return True


def port_available(host: str, port: int) -> bool:
    """探测端口能否绑定（启动前给出明确的错误提示）。"""
    import socket

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            sock.bind((host, int(port)))
            return True
        except OSError:
            return False


# --------------------------------------------------------------------------
def _to_float(value: Any, default: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _to_bool(value: Any, default: bool) -> bool:
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in ("1", "true", "yes", "on"):
        return True
    if text in ("0", "false", "no", "off"):
        return False
    return default
