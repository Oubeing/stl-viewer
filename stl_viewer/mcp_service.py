"""MCP 业务层：工具 / 资源 / 提示词，以及它们的实现。

这一层不依赖任何传输协议或 MCP SDK，只提供：

* ``TOOLS`` / ``RESOURCES`` / ``PROMPTS`` —— 符合 MCP 规范的普通 dict；
* ``MCPService.call_tool()`` —— 返回 ``ToolResult`` 的普通 Python 方法；
* ``ToolResult.to_content()`` —— 生成 MCP ``content`` 数组。

GUI 内嵌的 HTTP 服务、独立进程的 stdio 服务、官方 SDK 服务与命令行共用这一份
业务逻辑。服务可以绑定正在运行的 GUI 会话（``scene`` / ``camera`` /
``measures``），此时 AI 的 ``load_model``、``set_view``、``measure`` 会直接
作用于用户当前看到的场景；也可以独立运行，持有自己的会话。

``settings``（MCPSettings）决定服务策略：工具白名单、只读模式、写文件开关、
图片像素上限、默认单位 / 测量口径 / 渲染尺寸。未传 settings 时（命令行与
单元测试）保持完全开放，行为与独立服务一致。
"""

from __future__ import annotations

import base64
import dataclasses
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from . import stl_io
from .camera import STANDARD_VIEWS, Camera
from .lighting import LightSettings
from .mcp_config import (DEFAULT_MAX_IMAGE_PIXELS, FILE_WRITE_TOOLS,
                         SCENE_CHANGE_TOOLS)
from .measure import MEASURE_DEFS, MEASURE_MODES, MeasureManager
from .mesh import Mesh
from .png import encode_png
from .renderer import RenderOptions, SoftwareRenderer
from .scene import Scene
from .settings import UnitSystem

SERVER_NAME = "stl-viewer"
SERVER_VERSION = "1.0.0"
#: MCP 协议版本。initialize 时若客户端给的是旧版本，回显客户端版本即可继续。
PROTOCOL_VERSION = "2025-06-18"
#: 独立服务（无 GUI 设置）的默认像素上限；GUI 服务以设置页里的值为准。
MAX_IMAGE_PIXELS = DEFAULT_MAX_IMAGE_PIXELS
MODEL_SUFFIXES = (".stl", ".ply")


class ToolError(Exception):
    """工具参数错误 / 业务失败。

    协议层会把它翻译成 ``isError: true`` 的工具结果（而不是 JSON-RPC error），
    这样 AI 能看到错误原因并自行修正参数重试 —— 这是 MCP 的约定。
    """


def _plain(value: Any) -> Any:
    """把 numpy / dataclass / Path 变成纯 JSON 类型（供 json.dumps 使用）。"""
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return {f.name: _plain(getattr(value, f.name))
                for f in dataclasses.fields(value)}
    if isinstance(value, dict):
        return {str(k): _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_plain(v) for v in value]
    return str(value)


def dumps(data: Any) -> str:
    return json.dumps(_plain(data), ensure_ascii=False, indent=2)


def _p(kind: str, description: str, **extra) -> dict:
    """JSON Schema 属性的小助手。"""
    return {"type": kind, "description": description, **extra}


def _tool(name: str, description: str, properties: dict,
          required: tuple = ()) -> dict:
    """构造一个 MCP 工具描述符（name + description + inputSchema）。"""
    return {
        "name": name,
        "description": description,
        "inputSchema": {
            "type": "object",
            "properties": properties,
            "required": list(required),
            "additionalProperties": False,
        },
    }


_POINT_SCHEMA = _p("array", "三维坐标 [x, y, z]，单位为 mm",
                   items={"type": "number"}, minItems=3, maxItems=3)
_UNIT_SCHEMA = _p("string", "显示单位", enum=["mm", "inch"])
_VIEW_SCHEMA = _p("string", "标准视角名称", enum=list(STANDARD_VIEWS.keys()))

#: 第 1 类 MCP 原语：Tools（AI 可以「执行」的函数）
TOOLS = [
    _tool("server_info", "查看服务器能力、工作区根目录、测量口径与限制。",
          {}),
    _tool("list_models", "列出工作区内可载入的 STL / PLY 模型文件。",
          {"directory": _p("string", "相对工作区的子目录，默认根目录"),
           "recursive": _p("boolean", "是否递归查找子目录，默认 true")}),
    _tool("load_model", "载入一个 STL / PLY 模型到当前场景，并自动适配相机。",
          {"path": _p("string", "模型文件，绝对路径或相对工作区根目录"),
           "name": _p("string", "场景内对象名，默认取文件名"),
           "weld": _p("boolean", "载入时焊接重合顶点，默认 true"),
           "repair": _p("boolean", "载入后修复网格，默认 false")},
          ("path",)),
    _tool("list_objects", "列出当前场景中的模型对象。", {}),
    _tool("inspect_model", "深度统计某个模型：面数、体积、表面积、是否封闭、边界边、连通分量等。",
          {"name": _p("string", "对象名，缺省取激活对象"),
           "index": _p("integer", "对象索引（0 起），与 name 二选一"),
           "deep": _p("boolean", "是否做拓扑/体积计算，默认 true")}),
    _tool("remove_object", "从场景移除一个对象。",
          {"name": _p("string", "对象名"),
           "index": _p("integer", "对象索引（0 起）")}),
    _tool("clear_scene", "清空场景中的全部对象。", {}),
    _tool("set_view", "设置相机视角（标准视图或 yaw/pitch/roll），必要时重新适配包围盒。",
          {"view": _VIEW_SCHEMA,
           "yaw": _p("number", "方位角（度）"),
           "pitch": _p("number", "俯仰角（度）"),
           "roll": _p("number", "滚转角（度）"),
           "orthographic": _p("boolean", "是否正交投影"),
           "fit": _p("boolean", "是否重新适配窗口，默认 false")}),
    _tool("measure", "执行一次测量。长度默认取测量平面内的投影长度（CAD 口径），"
                     "需要真实空间距离请把 mode 设为 3d。",
          {"kind": _p("string", "测量类型", enum=list(MEASURE_DEFS.keys())),
           "points": _p("array", "坐标点列表，点数由 kind 决定",
                        items=_POINT_SCHEMA, minItems=1),
           "plane": _p("array", "测量平面法向，默认 [0,0,1]（XY 平面）",
                       items={"type": "number"}, minItems=3, maxItems=3),
           "mode": _p("string", "view=投影长度（默认）；3d=真实三维距离",
                      enum=list(MEASURE_MODES)),
           "units": _UNIT_SCHEMA},
          ("kind", "points")),
    _tool("list_measurements", "列出本次会话已完成的全部测量结果。",
          {"units": _UNIT_SCHEMA}),
    _tool("clear_measurements", "清空测量结果。", {}),
    _tool("render_picture", "用软件光栅化渲染当前场景。不给 path 时把 PNG 作为图片内容"
                            "直接返回给 AI；给 path 时同时保存到工作区内的文件。",
          {"path": _p("string", "输出 PNG 路径（须在工作区内）"),
           "width": _p("integer", "图片宽度，默认 640"),
           "height": _p("integer", "图片高度，默认 480"),
           "view": _VIEW_SCHEMA,
           "supersample": _p("integer", "超采样倍数 1~2，默认 1"),
           "color_mode": _p("string", "着色模式",
                            enum=["auto", "uniform", "orientation",
                                  "component", "curvature"]),
           "wire": _p("boolean", "线框模式"),
           "transparent_background": _p("boolean", "透明背景"),
           "include_image": _p("boolean", "是否把 PNG 一并返回给 AI；"
                                          "默认仅在未指定 path 时返回")}),
    _tool("section_radius", "求回转体在指定 Z 高度处的最大回转半径（RZ 测量）。",
          {"z": _p("number", "截面高度 Z（mm）"),
           "name": _p("string", "对象名，缺省取激活对象")},
          ("z",)),
    _tool("export_stl", "把场景中的模型写出为 STL（二进制或 ASCII）。",
          {"path": _p("string", "输出路径（须在工作区内）"),
           "name": _p("string", "只导出指定对象；缺省导出全部可见对象（会合并）"),
           "ascii": _p("boolean", "写 ASCII STL，默认二进制"),
           "weld": _p("boolean", "写出前焊接重合顶点"),
           "repair": _p("boolean", "写出前修复网格")},
          ("path",)),
    _tool("split_components", "把模型按连通分量拆成多个 STL 文件。",
          {"name": _p("string", "对象名，缺省取激活对象"),
           "out_prefix": _p("string", "输出文件名前缀，默认 <工作区>/mcp_output/<对象名>_part"),
           "min_triangles": _p("integer", "跳过面数少于该值的分量，默认 0")}),
]

TOOL_NAMES = tuple(t["name"] for t in TOOLS)

#: 第 2 类 MCP 原语：Resources（AI 可以「读取」的上下文）
RESOURCES = [
    {"uri": "stl://scene/objects", "name": "场景对象清单",
     "description": "当前已载入模型的名称、面数、包围盒、可见性（JSON）",
     "mimeType": "application/json"},
    {"uri": "stl://scene/measurements", "name": "测量结果",
     "description": "本次会话的全部测量读数（JSON）",
     "mimeType": "application/json"},
    {"uri": "stl://guide/measurement", "name": "测量口径说明",
     "description": "长度为什么是投影长度、何时用 3d 模式、三点圆为何是平面圆",
     "mimeType": "text/markdown"},
    {"uri": "stl://guide/tools", "name": "工具清单",
     "description": "服务器提供的工具、参数与调用顺序建议（由 TOOLS 动态生成）",
     "mimeType": "text/markdown"},
]

#: 第 3 类 MCP 原语：Prompts（预置的任务模板，客户端可主动选用）
PROMPTS = [
    {"name": "inspect_model",
     "description": "引导 AI 完整地检查一个模型：载入 → 统计 → 渲染 → 结论",
     "arguments": [{"name": "path", "description": "模型文件（相对工作区）",
                    "required": True}]},
    {"name": "measure_plan",
     "description": "引导 AI 按 CAD 口径规划一次测量",
     "arguments": [{"name": "goal", "description": "想量什么，例如「TOP 视图的台阶长度」",
                    "required": True}]},
]

MEASUREMENT_GUIDE = """# STL Viewer 测量口径（给 AI 的说明）

- **长度默认是投影长度**：一次测量的点都落在同一测量平面上，读数等于
  垂直于该平面方向上的投影长度，不把沿视线的深度差算进去。这与在 CATIA /
  SolidWorks 的 TOP 视图里量一条边完全一致。
- **真实三维距离**：把 `measure` 的 `mode` 设为 `3d`；结果里始终同时给出
  `projected_length_mm`、`spatial_length_mm`、`depth_delta_mm` 三项。
- **圆是平面圆**：`circle2p` / `circle3p` 都在同一测量平面内求解，画出来是
  正圆，半径是投影半径。`plane` 默认 `[0,0,1]`（XY 平面）。
- **测量平面由 `plane` 指定**：GUI 里由当前视图自动决定；无界面服务里显式
  传法向即可。想量 XZ 平面的尺寸就传 `[0,1,0]`。
- **单位**：模型坐标一律 mm；`units=inch` 只影响显示文本，不改变几何数值。
- **RZ 测量**：`rzpoint` 给半径 / 角度 / Z；`section_radius` 给某高度截面
  的最大回转半径，适合砂轮、刀具这类回转体。
"""


@dataclass
class ToolResult:
    """工具执行结果，与传输方式无关。"""

    summary: str
    data: Any = None
    image: bytes | None = None
    image_mime: str = "image/png"

    def to_content(self) -> list[dict]:
        """转成 MCP ``CallToolResult.content``：文本 +（可选）图片。"""
        text = self.summary
        if self.data is not None:
            text += "\n\n" + dumps(self.data)
        parts: list[dict] = [{"type": "text", "text": text}]
        if self.image:
            parts.append({
                "type": "image",
                "data": base64.b64encode(self.image).decode("ascii"),
                "mimeType": self.image_mime,
            })
        return parts


class MCPService:
    """一次 MCP 会话的状态与全部业务逻辑。

    ``root`` 是**沙箱根目录**：AI 传入的所有读/写路径都必须落在其中。
    传入 ``scene`` / ``camera`` / ``measures`` 即绑定正在运行的 GUI 会话，
    AI 的改动会实时反映到用户看到的窗口（``on_change`` 负责通知刷新）。
    """

    def __init__(self, root: str | None = None, *, settings=None,
                 scene=None, camera=None, measures=None, options=None,
                 units=None, on_change=None):
        root_value = root or (settings.resolved_root() if settings else None)
        self.root = self._norm(root_value or os.getcwd())
        if not os.path.isdir(self.root):
            raise ValueError(f"工作区根目录不存在: {self.root}")
        self.live_session = scene is not None
        self.scene = scene if scene is not None else Scene()
        self.camera = camera if camera is not None else Camera(640, 480)
        self.measures = measures if measures is not None else MeasureManager()
        self.options = options if options is not None else RenderOptions()
        self.units = units if units is not None else UnitSystem("mm")
        self.on_change = on_change
        self.configure(settings)

    # ---------------- 策略 ----------------
    def configure(self, settings=None) -> None:
        """按 GUI 设置刷新策略；命令行 / 独立服务传 None 表示完全开放。"""
        self.settings = settings
        if settings is None:
            self.max_image_pixels = MAX_IMAGE_PIXELS
            self.allow_file_write = True
            self.allow_scene_changes = True
            self.allowed_tools = TOOL_NAMES
            self.default_units = "mm"
            self.default_measure_mode = "view"
            self.render_width, self.render_height = 640, 480
            return
        self.max_image_pixels = int(settings.max_image_pixels)
        self.allow_file_write = bool(settings.allow_file_write)
        self.allow_scene_changes = bool(settings.allow_scene_changes)
        self.allowed_tools = tuple(settings.enabled_tool_names(TOOL_NAMES))
        self.default_units = settings.default_units
        self.default_measure_mode = settings.default_measure_mode
        self.render_width = int(settings.render_width)
        self.render_height = int(settings.render_height)

    # ---------------- 路径沙箱 ----------------
    @staticmethod
    def _norm(path: str) -> str:
        return os.path.realpath(os.path.expanduser(os.path.expandvars(str(path))))

    def _path(self, value: str, *, must_exist: bool = False,
              write: bool = False) -> str:
        """解析路径并确认没有逃出工作区；写路径顺便创建父目录。"""
        raw = str(value)
        candidate = raw if os.path.isabs(raw) else os.path.join(self.root, raw)
        path = self._norm(candidate)
        root_key = os.path.normcase(self.root)
        try:
            inside = (os.path.commonpath([root_key, os.path.normcase(path)])
                      == root_key)
        except ValueError:                       # 跨盘符
            inside = False
        if not inside:
            raise ToolError(f"路径越出工作区 {self.root}: {raw}")
        if must_exist and not os.path.exists(path):
            raise ToolError(f"文件不存在: {raw}")
        if write:
            os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        return path

    def _display(self, path: str) -> str:
        try:
            return os.path.relpath(path, self.root).replace("\\", "/")
        except ValueError:
            return path

    # ---------------- 公共分发入口 ----------------
    def call_tool(self, name: str, arguments: dict | None = None) -> ToolResult:
        args = dict(arguments or {})
        if name not in TOOL_NAMES:
            raise ToolError(f"未知工具 {name!r}；可用工具: {', '.join(TOOL_NAMES)}")
        if name not in self.allowed_tools:
            raise ToolError(f"工具 {name!r} 已在 MCP 服务设置中被禁用")
        if not self.allow_scene_changes and name in SCENE_CHANGE_TOOLS:
            raise ToolError("服务处于只读模式：不允许改动当前场景")
        if not self.allow_file_write and (
                name in FILE_WRITE_TOOLS or
                (name == "render_picture" and args.get("path"))):
            raise ToolError("服务设置禁止写文件：请在 MCP 服务设置中开启文件写入")
        handler = getattr(self, "_tool_" + name)
        result = handler(args)
        if self.on_change is not None:
            try:
                self.on_change(name, args)
            except Exception:                        # 刷新通知不能影响工具结果
                pass
        return result

    def list_tools(self) -> list[dict]:
        """只暴露策略允许的工具，保证 tools/list 与 tools/call 一致。"""
        return [t for t in TOOLS if t["name"] in self.allowed_tools]

    def list_resources(self) -> list[dict]:
        return RESOURCES

    def list_prompts(self) -> list[dict]:
        return PROMPTS

    def read_resource(self, uri: str) -> tuple[str, str]:
        """返回 (mimeType, text)。资源内容可以动态生成，不必是静态文件。"""
        if uri == "stl://scene/objects":
            return "application/json", dumps(self._objects_data())
        if uri == "stl://scene/measurements":
            return "application/json", dumps(self.measures.export(self.units))
        if uri == "stl://guide/measurement":
            return "text/markdown", MEASUREMENT_GUIDE
        if uri == "stl://guide/tools":
            lines = ["# STL Viewer MCP 工具", ""]
            for t in self.list_tools():
                required = t["inputSchema"]["required"]
                args = ", ".join(
                    f"{k}{'' if k in required else '?'}"
                    for k in t["inputSchema"]["properties"])
                lines.append(f"- **{t['name']}**({args}) — {t['description']}")
            lines += ["", "建议顺序：`list_models` → `load_model` → "
                      "`inspect_model` / `set_view` → `render_picture` / `measure`。"]
            return "text/markdown", "\n".join(lines) + "\n"
        raise ToolError(f"未知资源: {uri}")

    def get_prompt(self, name: str, arguments: dict | None = None) -> dict:
        args = dict(arguments or {})
        if name == "inspect_model":
            path = args.get("path", "<模型路径>")
            text = (f"请用 STL Viewer MCP 检查模型 `{path}`：先用 list_models 确认文件，"
                    "再 load_model 载入，inspect_model 读出面数/体积/封闭性/法向，"
                    "之后 set_view 到 top 与 iso 各 render_picture 一张，"
                    "最后用中文总结外形尺寸、是否为封闭实体、有无需要修复的问题。")
        elif name == "measure_plan":
            goal = args.get("goal", "量一段长度")
            text = (f"目标：{goal}。请先读资源 stl://guide/measurement 了解口径，"
                    "再决定测量平面 plane（默认 [0,0,1]）与 kind；"
                    "用 measure 取点后同时报告投影长度与 3D 距离，"
                    "并说明两者差异来自哪一段深度差。")
        else:
            raise ToolError(f"未知提示词: {name}")
        return {"description": next(p["description"] for p in PROMPTS
                                    if p["name"] == name),
                "messages": [{"role": "user",
                              "content": {"type": "text", "text": text}}]}

    # ---------------- 公共小工具 ----------------
    def _objects_data(self) -> list[dict]:
        out = []
        for i, obj in enumerate(self.scene.objects):
            lo, hi = obj.mesh.bounds()
            out.append({"index": i, "name": obj.name, "visible": obj.visible,
                        "active": obj.active,
                        "triangles": obj.triangle_count,
                        "vertices": obj.mesh.vertex_count,
                        "bounds_min": np.asarray(lo).tolist(),
                        "bounds_max": np.asarray(hi).tolist(),
                        "source": self._display(obj.source_path)
                        if obj.source_path else ""})
        return out

    def _object(self, args: dict):
        """按 name / index / 默认激活对象 取一个对象。"""
        objs = self.scene.objects
        if not objs:
            raise ToolError("场景为空，请先 load_model")
        if args.get("name") is not None:
            obj = self.scene.find(str(args["name"]))
            if obj is None:
                raise ToolError(f"未找到对象 {args['name']!r}")
            return obj
        if args.get("index") is not None:
            i = int(args["index"])
            if not 0 <= i < len(objs):
                raise ToolError(f"对象索引越界: {i}（共 {len(objs)} 个）")
            return objs[i]
        obj = self.scene.active_object()
        return obj or objs[-1]

    @staticmethod
    def _vec3(value, label: str = "坐标") -> np.ndarray:
        try:
            v = np.asarray(value, dtype=np.float64).reshape(-1)
        except (TypeError, ValueError) as exc:
            raise ToolError(f"{label} 必须是三个数字: {value!r}") from exc
        if v.size != 3 or not np.isfinite(v).all():
            raise ToolError(f"{label} 必须是有限的三元组: {value!r}")
        return v

    # ---------------- 各个工具的实现 ----------------
    def _tool_server_info(self, a: dict) -> ToolResult:
        data = {
            "server": SERVER_NAME,
            "version": SERVER_VERSION,
            "protocol": PROTOCOL_VERSION,
            "workspace_root": self.root,
            "tools": len(TOOLS),
            "resources": len(RESOURCES),
            "prompts": len(PROMPTS),
            "standard_views": list(STANDARD_VIEWS.keys()),
            "measure_kinds": list(MEASURE_DEFS.keys()),
            "measure_modes": list(MEASURE_MODES),
            "max_image_pixels": self.max_image_pixels,
            "live_session": self.live_session,
            "allow_scene_changes": self.allow_scene_changes,
            "allow_file_write": self.allow_file_write,
            "enabled_tools": list(self.allowed_tools),
            "defaults": {"units": self.default_units,
                         "measure_mode": self.default_measure_mode,
                         "render_size": [self.render_width, self.render_height]},
            "endpoint": self.settings.endpoint if self.settings else None,
            "policy": ("长度默认取测量平面内的投影长度（CAD 口径）；"
                       "所有路径必须位于 workspace_root 内；模型坐标单位为 mm"),
        }
        return ToolResult("STL Viewer MCP 服务已就绪（无 GUI 依赖）", data)

    def _tool_list_models(self, a: dict) -> ToolResult:
        directory = self._path(a.get("directory") or self.root, must_exist=True)
        if not os.path.isdir(directory):
            raise ToolError(f"不是目录: {a.get('directory')}")
        walker = Path(directory).rglob("*") if a.get("recursive", True) else \
            Path(directory).glob("*")
        files = []
        for p in sorted(walker):
            if p.is_file() and p.suffix.lower() in MODEL_SUFFIXES:
                files.append({"path": self._display(str(p)),
                              "bytes": p.stat().st_size})
                if len(files) >= 200:
                    break
        return ToolResult(f"找到 {len(files)} 个模型文件",
                          {"directory": self._display(directory), "files": files})

    def _tool_load_model(self, a: dict) -> ToolResult:
        path = self._path(a["path"], must_exist=True)
        mesh = Mesh.load(path, weld=bool(a.get("weld", True)))
        repaired = None
        if a.get("repair"):
            repaired = _plain(mesh.repair())
        name = a.get("name") or Path(path).stem
        obj = self.scene.add_mesh(mesh, name=str(name), source_path=path)
        self.camera.fit(*self.scene.bounds())
        return ToolResult(f"已载入 {obj.name}（{obj.triangle_count} 面）",
                          {"object": self._objects_data()[-1],
                           "repair": repaired,
                           "scene": self.scene.statistics()})

    def _tool_list_objects(self, a: dict) -> ToolResult:
        items = self._objects_data()
        return ToolResult(f"场景共 {len(items)} 个对象",
                          {"objects": items, "scene": self.scene.statistics()})

    def _tool_inspect_model(self, a: dict) -> ToolResult:
        obj = self._object(a)
        st = obj.mesh.statistics(deep=bool(a.get("deep", True)))
        lo, hi = obj.mesh.bounds()
        report = None
        if obj.source_path and os.path.exists(obj.source_path):
            try:
                report = _plain(stl_io.file_size_report(obj.source_path))
            except Exception as exc:                    # 文件报告失败不影响统计
                report = {"error": str(exc)}
        data = {"name": obj.name, "statistics": _plain(st),
                "bounds": {"min": np.asarray(lo).tolist(),
                           "max": np.asarray(hi).tolist(),
                           "size": obj.mesh.size().tolist()},
                "file": report}
        return ToolResult(
            f"{obj.name}: {st.triangles} 面 / {st.vertices} 顶点，"
            f"封闭={st.closed}，体积={st.volume:.3f} mm³", data)

    def _tool_remove_object(self, a: dict) -> ToolResult:
        obj = self._object(a)
        index = self.scene.objects.index(obj)
        name = obj.name
        self.scene.remove_object(index)
        return ToolResult(f"已移除 {name}", {"objects": self._objects_data()})

    def _tool_clear_scene(self, a: dict) -> ToolResult:
        count = len(self.scene.objects)
        self.scene.clear()
        self.measures.clear()
        return ToolResult(f"已清空 {count} 个对象与全部测量",
                          {"objects": []})

    def _tool_set_view(self, a: dict) -> ToolResult:
        cam = self.camera
        view = a.get("view")
        if view is not None:
            if view not in STANDARD_VIEWS:
                raise ToolError(f"未知视角 {view!r}；可用: "
                                f"{', '.join(STANDARD_VIEWS)}")
            cam.set_standard_view(view)
        for key in ("yaw", "pitch", "roll"):
            if a.get(key) is not None:
                setattr(cam, key, float(a[key]))
        if a.get("orthographic") is not None:
            cam.set_orthographic(bool(a["orthographic"]))
        if a.get("fit"):
            cam.fit(*self.scene.bounds())
        return ToolResult(f"相机已设置：yaw={cam.yaw:.1f}° "
                          f"pitch={cam.pitch:.1f}° roll={cam.roll:.1f}°",
                          {"camera": _plain(cam.state())})

    def _tool_measure(self, a: dict) -> ToolResult:
        kind = a.get("kind")
        if kind not in MEASURE_DEFS:
            raise ToolError(f"未知测量类型 {kind!r}；可用: "
                            f"{', '.join(MEASURE_DEFS)}")
        needed = MEASURE_DEFS[kind][0]
        raw_points = a.get("points")
        if not isinstance(raw_points, list) or len(raw_points) != needed:
            raise ToolError(f"{kind} 需要 {needed} 个点，收到 "
                            f"{len(raw_points) if isinstance(raw_points, list) else '?'} 个")
        points = [self._vec3(p, f"第 {i + 1} 个点")
                  for i, p in enumerate(raw_points)]
        units = UnitSystem(a.get("units", self.default_units))
        mode = a.get("mode", self.default_measure_mode)
        if mode not in MEASURE_MODES:
            raise ToolError(f"mode 必须是 {MEASURE_MODES}")
        plane = None
        if a.get("plane") is not None:
            plane = self._vec3(a["plane"], "plane")
        elif MEASURE_DEFS[kind][2]:
            plane = np.array([0.0, 0.0, 1.0])
        mgr = MeasureManager()
        mgr.set_mode(mode)
        mgr.start(kind)
        for p in points[:-1]:
            mgr.add_point(p)
        m = mgr.add_point(points[-1], units=units, plane=plane)
        if m is None:
            raise ToolError(f"测量未完成（{mgr.last_reject or '点数不足'}）；"
                            "检查点是否重合或三点共线")
        m.refresh_label(m.value_text(units))
        self.measures.items.append(m)
        result = {"index": len(self.measures.items), "kind": kind,
                  "text": m.value_text(units), **_plain(m.results())}
        return ToolResult(result["text"], result)

    def _tool_list_measurements(self, a: dict) -> ToolResult:
        units = UnitSystem(a.get("units", self.default_units))
        items = _plain(self.measures.export(units))
        return ToolResult(f"共 {len(items)} 条测量结果", {"items": items})

    def _tool_clear_measurements(self, a: dict) -> ToolResult:
        n = self.measures.delete_all()
        return ToolResult(f"已清空 {n} 条测量", {"removed": n})

    def _tool_render_picture(self, a: dict) -> ToolResult:
        width = int(a.get("width", self.render_width))
        height = int(a.get("height", self.render_height))
        ss = max(1, min(int(a.get("supersample", 1)), 2))
        width, height = max(16, min(width, 2048)), max(16, min(height, 2048))
        if width * height * ss * ss > self.max_image_pixels:
            raise ToolError(f"图片过大（{width}x{height}@{ss}x 超过 "
                            f"{self.max_image_pixels} 像素上限），请调小尺寸")
        cam = self.camera.copy()
        view = a.get("view")
        if view is not None:
            if view not in STANDARD_VIEWS:
                raise ToolError(f"未知视角 {view!r}")
            cam.set_standard_view(view)
        cam.set_viewport(width, height)
        cam.fit(*self.scene.bounds())
        overrides = {"supersample": ss}
        if a.get("color_mode"):
            overrides["color_mode"] = str(a["color_mode"])
        if a.get("wire") is not None:
            overrides["wire"] = bool(a["wire"])
            overrides["solid"] = not bool(a["wire"])
        if a.get("transparent_background") is not None:
            overrides["transparent_background"] = bool(a["transparent_background"])
        opts = dataclasses.replace(self.options, **overrides)
        renderer = SoftwareRenderer()
        image = renderer.render(self.scene, cam, size=(width, height),
                                options=opts, light=LightSettings(),
                                measurements=self.measures.items)
        png_bytes = encode_png(image)
        saved = None
        if a.get("path"):
            out = self._path(a["path"], write=True)
            with open(out, "wb") as fh:
                fh.write(png_bytes)
            saved = self._display(out)
        include = bool(a.get("include_image", not a.get("path")))
        stats = _plain(renderer.last_stats)
        return ToolResult(
            f"渲染完成 {width}x{height}，{len(png_bytes)} 字节"
            + (f"，已保存 {saved}" if saved else ""),
            {"size": [width, height], "png_bytes": len(png_bytes),
             "path": saved, "stats": stats},
            image=png_bytes if include else None)

    def _tool_section_radius(self, a: dict) -> ToolResult:
        obj = self._object(a)
        z = float(a["z"])
        segments = obj.mesh.section_polygons((0.0, 0.0, z), (0.0, 0.0, 1.0))
        rmax = 0.0
        count = 0
        for p0, p1 in segments:
            count += 1
            for p in (p0, p1):
                rmax = max(rmax, float(np.hypot(p[0], p[1])))
        return ToolResult(f"Z={z:.4f} 处最大回转半径 R={rmax:.4f} mm",
                          {"z": z, "radius_mm": rmax, "diameter_mm": rmax * 2,
                           "segments": count, "object": obj.name})

    def _tool_export_stl(self, a: dict) -> ToolResult:
        path = self._path(a["path"], write=True)
        if a.get("name"):
            objs = [self._object({"name": a["name"]})]
        else:
            objs = self.scene.visible_objects() or list(self.scene.objects)
        if not objs:
            raise ToolError("场景为空，没有可导出的模型")
        verts, tris, offset = [], [], 0
        for obj in objs:
            verts.append(np.asarray(obj.mesh.vertices, dtype=np.float64))
            tris.append(np.asarray(obj.mesh.triangles, dtype=np.int64) + offset)
            offset += obj.mesh.vertex_count
        vertices = np.vstack(verts)
        triangles = np.vstack(tris)
        if a.get("weld") or a.get("repair"):
            mesh = Mesh.from_arrays(vertices, triangles)
            if a.get("repair"):
                mesh.repair()
            elif a.get("weld"):
                mesh.weld(1e-7)
            vertices, triangles = mesh.vertices, mesh.triangles
        if a.get("ascii"):
            stl_io.write_ascii_stl(path, vertices, triangles, "stl-viewer-mcp")
        else:
            stl_io.write_binary_stl(path, vertices, triangles)
        return ToolResult(
            f"已导出 {len(triangles)} 面到 {self._display(path)}",
            {"path": self._display(path), "bytes": os.path.getsize(path),
             "vertices": int(vertices.shape[0]),
             "triangles": int(triangles.shape[0]),
             "objects": [o.name for o in objs], "format":
             "ascii" if a.get("ascii") else "binary"})

    def _tool_split_components(self, a: dict) -> ToolResult:
        obj = self._object(a)
        parts = obj.mesh.split_components()
        prefix = a.get("out_prefix") or os.path.join(
            "mcp_output", f"{obj.name}_part")
        prefix_path = self._path(prefix, write=True)
        minimum = int(a.get("min_triangles", 0))
        written = []
        for i, part in enumerate(parts, 1):
            if part.triangle_count < minimum:
                continue
            out = f"{prefix_path}_{i:02d}.stl"
            stl_io.write_binary_stl(out, part.vertices, part.triangles)
            written.append({"path": self._display(out),
                            "triangles": part.triangle_count,
                            "volume_mm3": abs(part.volume())})
        return ToolResult(f"{obj.name} 共 {len(parts)} 个连通分量，写出 {len(written)} 个",
                          {"components": len(parts), "files": written})
