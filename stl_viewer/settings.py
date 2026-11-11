"""应用设置与单位 —— 设置与单位持久化
（Load default settings / Save default settings /
Reset selected default settings / File units: mm, inch）。

使用 INI 文件持久化。
"""

from __future__ import annotations

import configparser
import os
from dataclasses import asdict, dataclass, field, fields
from typing import Any

from .camera import Camera
from .lighting import LightSettings
from .scene import Scene

APP_NAME = "STL Viewer"
SETTINGS_FILENAME = "stl_viewer.ini"

MM_PER_INCH = 25.4


# --------------------------------------------------------------------------
# 单位
# --------------------------------------------------------------------------
class UnitSystem:
    """测量单位换算与格式化（File units: mm / inch）。"""

    MM = "mm"
    INCH = "inch"

    def __init__(self, unit: str = MM):
        self.unit = unit if unit in (self.MM, self.INCH) else self.MM

    @property
    def label(self) -> str:
        return "mm" if self.unit == self.MM else "in"

    @property
    def factor(self) -> float:
        """mm -> 当前单位 的换算系数。"""
        return 1.0 if self.unit == self.MM else 1.0 / MM_PER_INCH

    def to_display(self, value_mm: float) -> float:
        return float(value_mm) * self.factor

    def from_display(self, value: float) -> float:
        return float(value) / self.factor

    def format(self, value_mm: float, digits: int | None = None) -> str:
        if digits is None:
            digits = 3 if self.unit == self.MM else 4
        return f"{self.to_display(value_mm):.{digits}f} {self.label}"

    def format_length(self, value_mm: float, digits: int | None = None) -> str:
        if digits is None:
            digits = 3 if self.unit == self.MM else 4
        return f"{self.to_display(value_mm):.{digits}f}"

    def format_angle(self, degrees: float, digits: int = 3) -> str:
        return f"{degrees:.{digits}f}\u00b0"

    def format_point(self, p, digits: int | None = None) -> str:
        if digits is None:
            digits = 3 if self.unit == self.MM else 4
        return "(" + ", ".join(f"{self.to_display(v):.{digits}f}" for v in p) + ")"

    def state(self) -> dict:
        return {"unit": self.unit}

    def set_state(self, state: dict) -> None:
        if state and state.get("unit") in (self.MM, self.INCH):
            self.unit = state["unit"]


# --------------------------------------------------------------------------
# 视图选项
# --------------------------------------------------------------------------
@dataclass
class ViewOptions:
    """RenderOptions 的可持久化部分（View options / Mesh 菜单）。"""

    wire: bool = False
    solid: bool = True
    color_mode: str = "auto"          # auto | uniform | component | curvature
    transparent: bool = False
    show_axis: bool = True
    axis_labels: bool = True
    show_grid: bool = False
    crosshair: bool = False
    point_size: int = 1
    line_width: int = 1
    # 默认剔除背面：封闭 STL 实体上，背面三角形会在共享深度处和正面
    # Z-fighting，放大看是一条条细缝/斑点；剔除后表面干净且快约 1/3。
    # 需要观察开放曲面背面时，用 Mesh > Back and Front 打开。
    back_and_front: bool = False
    show_edges: bool = False
    edge_color: tuple = (0.08, 0.09, 0.11)
    # 默认正交投影：检测/测量工具下严格等比例——近大远小的透视畸变消失，
    # 正视的圆保持正圆、缩放保持等比例，且测量读数直接等于真实尺寸。
    # 需要透视（近大远小、立体感）时，用 View > Orthogonal 切换。
    orthographic: bool = True
    fixed_on_screen: bool = False
    fullscreen: bool = False
    background: tuple = (0.16, 0.17, 0.20)
    background_top: tuple = (0.30, 0.33, 0.38)
    gradient_background: bool = False
    mesh_color: tuple = (0.85, 0.85, 0.88)

    def state(self) -> dict:
        return asdict(self)

    def set_state(self, state: dict) -> None:
        for f in fields(self):
            if f.name not in state:
                continue
            value = state[f.name]
            current = getattr(self, f.name)
            if isinstance(current, bool):
                value = _to_bool(value, current)
            elif isinstance(current, int):
                value = int(_to_float(value, current))
            elif isinstance(current, float):
                value = _to_float(value, current)
            elif isinstance(current, tuple):
                if isinstance(value, (list, tuple)):
                    value = tuple(float(v) for v in value)
                else:
                    value = tuple(float(v) for v in _parse_list(value, current))
            elif isinstance(value, str):
                value = value.strip()
            setattr(self, f.name, value)


# --------------------------------------------------------------------------
# 应用设置
# --------------------------------------------------------------------------
@dataclass
class AppSettings:
    """全局设置容器。"""

    language: str = "EN"
    units: str = "mm"
    render_system: str = "OpenGL"        # OpenGL | Direct3D | Software
    hardware_acceleration: bool = True   # miEnableHwAccel
    antialias: bool = True
    double_resolution_pictures: bool = False
    picture_transparent_background: bool = False
    interpolate_normals: bool = False    # 逐顶点法向插值；STL 是面片格式，默认按面着色
    auto_weld: bool = True               # 载入时自动焊接重合顶点
    mesh_color: tuple = (0.85, 0.85, 0.88)
    wire_color: tuple = (0.10, 0.11, 0.13)
    default_zoom_factor: float = 1.25
    max_fps: int = 60
    # 后台异步渲染后拖拽不再阻塞 UI，默认给到接近全分辨率；只有实测掉帧
    # 才自适应下调。默认 0.5 会让拖拽全程糊成低清图（显示失真）。
    interactive_quality: float = 0.85    # 拖拽时的渲染分辨率比例上限
    # ---- 交互灵敏度 ----
    # 0.35 度/像素时拖 300px 就转 105°；0.20 仍偏快，一次随手拖动容易过冲。
    # 现在配合画布的单事件限幅 + 非线性映射，默认 0.12 度/像素，小幅拖动更细腻。
    rotate_speed: float = 0.12           # 轨道旋转：度 / 像素
    pan_speed: float = 1.0               # 平移：1.0 表示模型严格跟随光标
    zoom_speed: float = 0.07             # 滚轮缩放：每档的指数系数
    auto_fit_views: bool = True          # 切换标准视图时自动 Zoom Fit
    use_opengl: bool = True              # 硬件视口（不可用时自动回退软件渲染）
    view: ViewOptions = field(default_factory=ViewOptions)
    camera: dict = field(default_factory=dict)
    light: dict = field(default_factory=dict)
    default_directory: str = ""
    recent_files: list = field(default_factory=list)

    # ---------------- 路径 ----------------
    @staticmethod
    def default_path() -> str:
        base = os.environ.get("APPDATA") or os.path.expanduser("~")
        return os.path.join(base, APP_NAME, SETTINGS_FILENAME)

    # ---------------- 读写 ----------------
    def save(self, path: str | None = None) -> str:
        path = path or self.default_path()
        os.makedirs(os.path.dirname(path), exist_ok=True)
        cp = configparser.ConfigParser()
        cp["General"] = {
            "Language": self.language,
            "Units": self.units,
            "RenderSystem": self.render_system,
            "HardwareAcceleration": str(self.hardware_acceleration),
            "Antialias": str(self.antialias),
            "DoubleResolution": str(self.double_resolution_pictures),
            "TransparentPictures": str(self.picture_transparent_background),
            "InterpolateNormals": str(self.interpolate_normals),
            "AutoWeld": str(self.auto_weld),
            "DefaultZoomFactor": str(self.default_zoom_factor),
            "MaxFPS": str(self.max_fps),
            "InteractiveQuality": str(self.interactive_quality),
            "RotateSpeed": str(self.rotate_speed),
            "PanSpeed": str(self.pan_speed),
            "ZoomSpeed": str(self.zoom_speed),
            "AutoFitViews": str(self.auto_fit_views),
            "UseOpenGL": str(self.use_opengl),
            "DefaultDirectory": self.default_directory,
        }
        cp["View"] = {k: _to_str(v) for k, v in self.view.state().items()}
        cp["Camera"] = {k: _to_str(v) for k, v in self.camera.items()}
        cp["Light"] = {k: _to_str(v) for k, v in self.light.items()}
        cp["Colors"] = {
            "MeshColor": _to_str(self.mesh_color),
            "WireColor": _to_str(self.wire_color),
        }
        cp["Recent"] = {f"File{i}": p for i, p in enumerate(self.recent_files[:10])}
        with open(path, "w", encoding="utf-8") as fh:
            cp.write(fh)
        return path

    def load(self, path: str | None = None) -> "AppSettings":
        path = path or self.default_path()
        if not os.path.exists(path):
            return self
        cp = configparser.ConfigParser()
        try:
            cp.read(path, encoding="utf-8")
        except Exception:
            return self
        g = cp["General"] if "General" in cp else {}
        self.language = g.get("Language", self.language)
        self.units = g.get("Units", self.units)
        self.render_system = g.get("RenderSystem", self.render_system)
        self.hardware_acceleration = _to_bool(g.get("HardwareAcceleration"), self.hardware_acceleration)
        self.antialias = _to_bool(g.get("Antialias"), self.antialias)
        self.double_resolution_pictures = _to_bool(
            g.get("DoubleResolution"), self.double_resolution_pictures)
        self.picture_transparent_background = _to_bool(
            g.get("TransparentPictures"), self.picture_transparent_background)
        self.interpolate_normals = _to_bool(
            g.get("InterpolateNormals"), self.interpolate_normals)
        self.auto_weld = _to_bool(g.get("AutoWeld"), self.auto_weld)
        self.default_zoom_factor = _to_float(g.get("DefaultZoomFactor"), self.default_zoom_factor)
        self.max_fps = int(_to_float(g.get("MaxFPS"), self.max_fps))
        self.interactive_quality = _to_float(g.get("InteractiveQuality"), self.interactive_quality)
        self.rotate_speed = _to_float(g.get("RotateSpeed"), self.rotate_speed)
        self.pan_speed = _to_float(g.get("PanSpeed"), self.pan_speed)
        self.zoom_speed = _to_float(g.get("ZoomSpeed"), self.zoom_speed)
        self.auto_fit_views = _to_bool(g.get("AutoFitViews"), self.auto_fit_views)
        self.use_opengl = _to_bool(g.get("UseOpenGL"), self.use_opengl)
        self.default_directory = g.get("DefaultDirectory", self.default_directory)

        if "View" in cp:
            self.view.set_state(dict(cp["View"]))
        if "Camera" in cp:
            self.camera = {k: _parse_value(v) for k, v in cp["Camera"].items()}
        if "Light" in cp:
            self.light = {k: _parse_value(v) for k, v in cp["Light"].items()}
        if "Colors" in cp:
            self.mesh_color = tuple(_parse_list(cp["Colors"].get("MeshColor"), self.mesh_color))
            self.wire_color = tuple(_parse_list(cp["Colors"].get("WireColor"), self.wire_color))
        if "Recent" in cp:
            self.recent_files = [v for k, v in cp["Recent"].items() if v]
        return self

    def reset_to_factory(self) -> None:
        """Reset selected default settings：恢复出厂默认。"""
        fresh = AppSettings()
        for f in fields(self):
            setattr(self, f.name, getattr(fresh, f.name))

    # ---------------- 应用到运行时对象 ----------------
    def apply_to(self, camera: Camera, scene: Scene, light: LightSettings) -> None:
        camera.set_state(self.camera)
        camera.set_orthographic(self.view.orthographic)
        light.set_state(self.light)
        scene.background = self.view.background
        scene.background_top = self.view.background_top
        scene.gradient_background = self.view.gradient_background

    def capture_from(self, camera: Camera, scene: Scene, light: LightSettings) -> None:
        self.camera = camera.state()
        self.light = light.state()
        self.view.set_state({
            "orthographic": camera.orthographic,
            "background": scene.background,
            "background_top": scene.background_top,
            "gradient_background": scene.gradient_background,
        })

    def add_recent(self, path: str) -> None:
        if path in self.recent_files:
            self.recent_files.remove(path)
        self.recent_files.insert(0, path)
        del self.recent_files[10:]


def load_settings(path: str | None = None) -> AppSettings:
    return AppSettings().load(path)


# --------------------------------------------------------------------------
# INI 值解析工具
# --------------------------------------------------------------------------
def _to_str(value: Any) -> str:
    if isinstance(value, (tuple, list)):
        return ",".join(str(v) for v in value)
    if isinstance(value, bool):
        return "1" if value else "0"
    return str(value)


def _parse_list(value: str | None, default):
    if value is None:
        return default
    parts = [p.strip() for p in value.split(",") if p.strip()]
    try:
        return [float(p) for p in parts]
    except ValueError:
        return default


def _parse_value(value: str):
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value
    if isinstance(value, (list, tuple)):
        return list(value)
    text = str(value).strip()
    if "," in text:
        return _parse_list(text, [])
    low = text.lower()
    if low in ("true", "yes", "on"):
        return True
    if low in ("false", "no", "off"):
        return False
    try:
        return int(text)
    except ValueError:
        pass
    try:
        return float(text)
    except ValueError:
        return text


def _to_bool(value, default: bool) -> bool:
    if isinstance(value, bool):
        return value
    parsed = _parse_value(value)
    if isinstance(parsed, bool):
        return parsed
    if isinstance(parsed, (int, float)):
        return bool(parsed)
    if isinstance(parsed, str):
        return parsed.lower() in ("true", "yes", "on", "1")
    return default


def _to_float(value, default: float) -> float:
    if isinstance(value, bool):
        return default
    if isinstance(value, (int, float)):
        return float(value)
    parsed = _parse_value(value)
    if isinstance(parsed, (int, float)) and not isinstance(parsed, bool):
        return float(parsed)
    return default
