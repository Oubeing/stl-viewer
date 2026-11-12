"""PyQt5 图形界面 —— STL 查看器的窗口、菜单与交互。

界面结构：

    ┌ 菜单栏 File | View | Mesh | Light | Measurement | Extras | Help
    ├ 视图工具条：缩放 / 复位 / 8 个标准视图 / 自动适配 / 投影 / 视图选项
    ├ 主工具条：文件、显示模式、着色与光源下拉、测量、旋转动画、截图
    ├ 左侧停靠面板：Objects（对象列表 + 显示属性）与 Measurements
    └ 状态栏：单位、对象/面数、渲染耗时、帧率

渲染：优先 OpenGL 硬件视口（QOpenGLWidget + VBO），84 万面模型也能实时旋转；
无 GL 上下文时自动回退到「后台线程软件光栅化 + 快照合成」。两种方式都不在
paintEvent 里同步跑光栅化，鼠标事件永远不会被一帧渲染堵住。

交互（CAD 风格导航，详见 navigation.py 与 README）：
    左键拖拽    旋转（Ctrl 为定轴旋转；Alt 为缩放；Shift 为平移）
    中键拖拽    平移；中键+左/右键 = 旋转；Ctrl/Shift+中键 = 缩放
    右键拖拽    平移；右键单击弹出上下文菜单
    滚轮        以光标为锚点缩放（zoom to cursor）
    拖拽中按 Shift   精细模式（旋转/缩放/平移降到 1/4）
    左键单击    测量模式下取点（投影到测量视图平面，读数即屏幕长度）
    方向键      旋转   Shift+方向键 平移   Ctrl+方向键 绕 Z / 俯仰
    Ctrl+Z      绕 Z 轴快捷开关
    Ctrl+Shift+F/X/Y/Z/V   旋转轴：自由 / X / Y / Z / 视线（菜单与工具条下拉同款）
    +/-         缩放      双击中键 / 双击左键 / F  适应窗口
    1~8 标准视图      R 复位中心      空格 播放/暂停     Esc 取消测量

导出：Save Picture 直接抓取当前画布帧（不做重渲染），后台只负责 PNG 编码；
透明背景 / 多分辨率这类确实需要新图的入口才重新光栅化一次。抓帧返回的是
**拥有自己内存的 numpy 副本**，绝不是 QImage 像素缓冲的视图 —— 后者在局部
QImage 析构后会让后台编码读到野指针，真实 GL 上表现为保存图片直接崩溃。

旋转：定轴旋转（View > 旋转轴）与旋转动画（Extras > 启用旋转 / 空格）共用
同一根轴，菜单、工具条、快捷键、空格键的状态由 _set_spin_enabled() 与
_sync_rotation_ui() 统一维护。
"""

from __future__ import annotations

import copy
import json
import math
import os
import sys
import time
from collections import deque

import numpy as np
from PyQt5 import QtCore, QtGui, QtWidgets

from . import stl_io
from . import mathutil as mu
from .camera import STANDARD_VIEWS, Camera
from .glviewport import GLViewport, ViewportOverlay
from .i18n import Translator
from .lighting import LIGHT_PRESETS, LightSettings
from .macro import MacroRecorder, SpinConfig, SpinEngine, SPIN_KINDS
from .measure import MEASURE_DEFS, MeasureManager, snap_pick
from .navigation import (AXIS_LABELS, BTN_LEFT, BTN_MIDDLE, BTN_RIGHT, NONE,
                         PAN, ROTATE, ROTATE_AXES, ROTATE_Z, ZOOM, NAV_STYLES,
                         NavOptions, NavigationController, resolve_action)
from .mesh import Mesh
from .renderer import RenderOptions, SoftwareRenderer, axis_screen_geometry
from .scene import Object3D, Scene
from .screenpick import ScreenPicker
from .settings import AppSettings, UnitSystem, ViewOptions

#: View > 旋转轴：(i18n 键, 轴标识, 快捷键)
ROTATE_AXIS_MENU = (
    ("rotate_axis_free", "free", "Ctrl+Shift+F"),
    ("rotate_axis_x", "x", "Ctrl+Shift+X"),
    ("rotate_axis_y", "y", "Ctrl+Shift+Y"),
    ("rotate_axis_z", "z", "Ctrl+Shift+Z"),
    ("rotate_axis_view", "view", "Ctrl+Shift+V"),
)

#: 定轴旋转与旋转动画必须共用同一根轴：以前「视图 > 旋转轴」只影响鼠标
#: 拖拽，「启用旋转」动画始终绕 Z 转，用户选了 X/Y 再启用动画会觉得按钮没用。
AXIS_TO_SPIN = {"free": "horizontal", "x": "rx", "y": "ry",
                "z": "rz", "view": "roll"}
SPIN_TO_AXIS = {"horizontal": "free", "vertical": "free", "rx": "x",
                "ry": "y", "rz": "z", "roll": "view", "random": "free"}

PICTURE_FILTER = "PNG 图片 (*.png);;JPEG 图片 (*.jpg);;BMP 图片 (*.bmp);;所有文件 (*)"
MODEL_FILTER = ("模型文件 (*.stl *.stpl *.ply *.stpack *.exe);;"
                "STL (*.stl);;PLY (*.ply);;自包含查看器 (*.exe *.stpack);;"
                "所有文件 (*)")

# 用颜色表达信息的着色模式（"彩色"开关即以是否落在这些模式内为准）
COLORED_MODES = ("auto", "component", "curvature", "orientation")
COLOR_MODE_CHOICES = ("auto", "uniform", "component", "curvature", "orientation")


def is_colored_mode(mode: str) -> bool:
    return mode in COLORED_MODES


def _write_image_file(path: str, image: np.ndarray) -> None:
    """把渲染结果写到磁盘；PNG 走自带的 zlib 编码，其它格式交给 Qt。

    这个函数只依赖 numpy / QImage，不碰任何 QWidget，可以安全地在后台
    线程里执行（保存图片不再阻塞界面）。
    """
    # 后台线程里编码时，来源 QImage / 共享缓冲可能已经被释放。凡是「不拥有
    # 自己内存」的视图一律先复制一份，杜绝跨线程读野指针导致进程崩溃。
    if not image.flags.owndata:
        image = np.array(image, dtype=np.uint8, copy=True, order="C")
    if path.lower().endswith(".png"):
        from .png import write_png

        write_png(path, image)
        return
    buffer = np.ascontiguousarray(image)
    hh, ww, ch = buffer.shape
    fmt = QtGui.QImage.Format_RGBA8888 if ch == 4 else QtGui.QImage.Format_RGB888
    qimg = QtGui.QImage(buffer.data, ww, hh, buffer.strides[0], fmt)
    if not qimg.save(path):
        raise RuntimeError(f"无法写入图片：{path}")


def _gl_platform_available() -> bool:
    """离屏 / 无窗口系统平台没有 GL 上下文，避免创建 QOpenGLWidget 报错。"""
    app = QtWidgets.QApplication.instance()
    if app is None:
        return False
    try:
        platform = QtGui.QGuiApplication.platformName().lower()
    except Exception:
        platform = ""
    return platform not in ("offscreen", "minimal")


# ==========================================================================
# 运行时状态
# ==========================================================================
class ViewerState:
    """把场景、相机、渲染器、设置等聚合成一个对象，供窗口与画布共享。"""

    def __init__(self, settings_path: str | None = None):
        self.settings_path = settings_path
        self.settings = AppSettings().load(settings_path)
        self.tr = Translator(self.settings.language)
        self.units = UnitSystem(self.settings.units)

        self.scene = Scene()
        self.camera = Camera(1024, 768)
        self.light = LightSettings()
        self.renderer = SoftwareRenderer()
        self.measures = MeasureManager()
        self.measures.set_mode(self.settings.measure_mode)
        self.spin = SpinEngine(SpinConfig())
        self.macro = MacroRecorder()
        self.options = RenderOptions.from_view_options(self.settings.view)

        # 真实画布像素尺寸；fit / 屏幕射线都用它，避免用构造时的默认值
        self.viewport: tuple = (1024, 768)
        self.current_path: str | None = None
        self.clipboard_object = None
        #: 定轴旋转：free | x | y | z | view
        self.rotate_axis = "free"
        self.last_render_seconds = 0.0
        self.fps = 0.0
        self._frame_times: deque = deque(maxlen=240)

        self.settings.apply_to(self.camera, self.scene, self.light)
        self.options = self.render_options()

    @property
    def rotate_around_z(self) -> bool:
        """兼容旧调用：绕 Z 开关等价于「定轴旋转选到 Z」。"""
        return self.rotate_axis == "z"

    @rotate_around_z.setter
    def rotate_around_z(self, value: bool) -> None:
        self.rotate_axis = "z" if value else "free"

    # ---------------- 选项 ----------------
    def render_options(self, interactive: bool = False) -> RenderOptions:
        """把 ViewOptions + 运行期开关合成为渲染参数。"""
        opts = RenderOptions.from_view_options(self.settings.view)
        opts.transparent = bool(self.settings.view.transparent)
        opts.point_size = int(self.settings.view.point_size)
        opts.line_width = int(self.settings.view.line_width)
        opts.show_edges = bool(self.settings.view.show_edges)
        opts.gouraud = bool(self.settings.interpolate_normals)
        if interactive:
            opts.supersample = 1
        elif self.settings.antialias:
            opts.supersample = 2
        return opts

    def sync_view_options(self) -> None:
        """把运行期状态写回 ViewOptions（保存设置时调用）。"""
        v = self.settings.view
        v.wire = self.options.wire
        v.solid = self.options.solid
        v.color_mode = self.options.color_mode
        v.transparent = self.options.transparent
        v.show_axis = self.options.show_axis
        v.axis_labels = self.options.axis_labels
        v.show_grid = self.options.show_grid
        v.crosshair = self.options.crosshair
        v.point_size = self.options.point_size
        v.line_width = self.options.line_width
        v.show_edges = self.options.show_edges
        v.background = self.options.background
        v.background_top = self.options.background_top
        v.gradient_background = self.options.gradient_background
        v.orthographic = self.camera.orthographic
        self.settings.units = self.units.unit
        self.settings.language = self.tr.language

    def save_settings(self) -> str:
        self.sync_view_options()
        self.settings.capture_from(self.camera, self.scene, self.light)
        return self.settings.save(self.settings_path)

    def load_settings(self) -> str:
        self.settings.load(self.settings_path)
        self.tr.set_language(self.settings.language)
        self.units.unit = self.settings.units
        self.settings.apply_to(self.camera, self.scene, self.light)
        self.options = self.render_options()
        self.measures.set_color(self.settings.view.mesh_color)
        self.measures.set_mode(self.settings.measure_mode)
        return self.settings_path or self.settings.default_path()

    # ---------------- 场景操作 ----------------
    def load_file(self, path: str) -> Object3D:
        mesh = Mesh.load(path, weld=bool(self.settings.auto_weld))
        name = os.path.splitext(os.path.basename(path))[0]
        obj = self.scene.add_mesh(mesh, name=name, source_path=path)
        self.current_path = path
        self.settings.add_recent(path)
        self.measures.clear()
        return obj

    def fit_view(self) -> None:
        """Zoom Fit：先把相机视口同步为真实画布尺寸，再按当前朝向精确适配。

        旧代码直接用构造时的 1024×768 视口算宽高比，最大化窗口（画布可能
        1500×820）后 fit 出来的距离偏大，模型自然显得小。
        """
        if not self.scene.objects:
            return
        w, h = self.viewport
        if w > 0 and h > 0:
            self.camera.set_viewport(w, h)
        lo, hi = self.scene.visible_bounds()
        # 用真实顶点（超过 20 万时等间隔抽样）求投影范围，比包围盒角点贴合
        chunks = []
        for obj in self.scene.visible_objects():
            v = obj.mesh.vertices
            if len(v) > 200_000:
                v = v[::int(math.ceil(len(v) / 200_000))]
            chunks.append(v)
        points = np.concatenate(chunks) if chunks else None
        self.camera.fit(lo, hi, points=points)

    def standard_view(self, name: str, auto_fit: bool | None = None) -> None:
        """切换标准视角；默认同时重新 Zoom Fit，让模型按新视角铺满视口。"""
        self.camera.set_standard_view(name)
        do_fit = self.settings.auto_fit_views if auto_fit is None else auto_fit
        if do_fit:
            self.fit_view()

    def statistics(self) -> dict:
        return self.scene.statistics()


# ==========================================================================
# 后台渲染任务
# ==========================================================================
class _RenderBridge(QtCore.QObject):
    """把 worker 线程的渲染结果送回 GUI 线程（自动 queued connection）。"""

    frameReady = QtCore.pyqtSignal(object, float, int)   # image, seconds, token


class _RenderTask(QtCore.QRunnable):
    """一次离屏渲染。

    只持有状态快照：相机副本、场景对象的浅拷贝、渲染选项与光照副本。
    真实的软件光栅化几乎全程在 numpy 里执行（会释放 GIL），放到单线程
    工作池里跑，GUI 线程只负责合成已完成的位图 —— 鼠标事件不再排队等
    一帧几百毫秒的全质量渲染，拖拽因此不会「卡住后突然跳一大截」。
    """

    def __init__(self, renderer, scene, camera, size, options, light,
                 measurements, bridge, token):
        super().__init__()
        self.renderer = renderer
        self.scene = scene
        self.camera = camera
        self.size = size
        self.options = options
        self.light = light
        self.measurements = measurements
        self.bridge = bridge
        self.token = token

    def run(self) -> None:
        try:
            image = self.renderer.render(
                self.scene, self.camera, size=self.size,
                options=self.options, light=self.light,
                measurements=self.measurements)
            seconds = float(self.renderer.last_stats.get("seconds", 0.0))
        except Exception:
            image, seconds = None, 0.0
        self.bridge.frameReady.emit(image, seconds, self.token)


class _PickBridge(QtCore.QObject):
    """把后台建好的测量拾取索引送回 GUI 线程。"""

    ready = QtCore.pyqtSignal(object, int)     # ScreenPicker | None, token


class _PickBuildTask(QtCore.QRunnable):
    """后台构建 ScreenPicker 的屏幕投影索引。

    84 万面模型建一次索引要几百毫秒；放在 GUI 线程里会让「点测量菜单 /
    转完视角停一下」明显顿一下。这里用场景快照在后台建，建好再整体替换
    —— 构建期间点击走视图平面（投影长度本来就是准的），永远不会等它。
    """

    def __init__(self, scene, camera, token: int, bridge: _PickBridge):
        super().__init__()
        self.scene = scene
        self.camera = camera
        self.token = token
        self.bridge = bridge

    def run(self) -> None:
        picker = None
        try:
            candidate = ScreenPicker()
            if candidate.ensure(self.scene, self.camera):
                picker = candidate
        except Exception:
            picker = None
        self.bridge.ready.emit(picker, self.token)


class _WorkBridge(QtCore.QObject):
    """后台任务完成通知（tag, result, error）—— 跨线程自动排队。"""

    finished = QtCore.pyqtSignal(str, object, object)


class _WorkTask(QtCore.QRunnable):
    """在后台线程执行一个无副作用（或只操作自己快照）的调用。

    打开大模型、导出 STL、统计、图片编码、网格修复都不该占着 GUI 线程；
    统一走这里，菜单 / 快捷键 / 工具栏点击后立刻返回，界面继续响应。
    """

    def __init__(self, tag: str, fn, bridge: _WorkBridge):
        super().__init__()
        self.tag = tag
        self.fn = fn
        self.bridge = bridge

    def run(self) -> None:
        try:
            result = self.fn()
            error = None
        except Exception as exc:          # noqa: BLE001 - 统一回报给 GUI
            result, error = None, exc
        self.bridge.finished.emit(self.tag, result, error)


# ==========================================================================
# 渲染画布
# ==========================================================================
class Canvas(QtWidgets.QWidget):
    """3D 视口：后台软件光栅化 / GPU 硬件视口 + CAD 风格鼠标导航。"""

    statusMessage = QtCore.pyqtSignal(str)
    sceneChanged = QtCore.pyqtSignal()

    # 后台渲染不阻塞事件循环，画质优先：只有实测明显掉帧才降分辨率，
    # 避免拖拽过程中画面一直糊成低清图（这也是「显示失真」的来源之一）。
    TARGET_FRAME_SECONDS = 1.0 / 30.0
    MIN_QUALITY = 0.35             # 交互分辨率下限
    MIN_RENDER_EDGE = 160          # 交互位图最小边长
    REFINE_DELAY_MS = 90           # 停止操作多久后开始精修全质量帧
    KEY_ROTATE_STEP = 5.0          # 方向键每次旋转的角度（度）
    # 兼容旧调用方保留的类常量；实际限幅与灵敏度已移到 navigation.py，
    # 那里放宽到 200/300 px 且改为严格线性，避免快速拖动被「吃掉」距离。
    MAX_MOUSE_STEP = 60
    MAX_PAN_STEP = 120
    ZOOM_DRAG_PER_PX = 0.020
    FINE_MODIFIER_FACTOR = 0.25
    # ---- 交互流畅度 ----
    STATUS_THROTTLE_MS = 70        # 状态栏文本合并刷新，鼠标事件里不做格式化
    RESIZE_DEBOUNCE_MS = 50        # 拖窗口边框时不逐像素重排渲染
    NAV_IDLE_MS = 160              # 最后一次导航操作后多久恢复全质量帧
    NAV_QUALITY = 0.60             # 交互开始时的预测性画质上限
    INERTIA_TICK_MS = 16           # 惯性动画步进（≈60Hz）
    # 测量取点的点击容差（屏幕像素，欧氏距离）：手抖 / 触控板 / 高 DPI 下按住
    # 鼠标总会移动几个像素，容差太小会把「想点一下」误判成「拖动旋转」而取不到点。
    # 只要在这个范围内，按压→释放就一律算一次取点（不会误旋转）。
    MEASURE_CLICK_TOL = 12
    # 完成一次测量后，保持同一测量工具继续激活（连续测量），
    # 这样「测完一个接着测下一个」不必每次回菜单点工具。
    MEASURE_CONTINUOUS = True
    # 单次渲染的像素预算：交互帧按窗口面积封顶，避免 4K 屏一帧几百毫秒；
    # 精修帧按 w*h*ss^2 限制超采样，让「静止后立刻又拖动」最多等一个短任务。
    MAX_INTERACTIVE_PIXELS = 800_000
    MAX_RENDER_PIXELS = 2_200_000

    def __init__(self, state: ViewerState, parent=None):
        super().__init__(parent)
        self.state = state
        self.setMinimumSize(320, 240)
        self.setFocusPolicy(QtCore.Qt.StrongFocus)
        self.setMouseTracking(True)
        self.setAcceptDrops(True)
        self.setAutoFillBackground(False)

        self._image: QtGui.QImage | None = None
        self._buffer: np.ndarray | None = None
        self._dirty = True
        self._interactive = False
        self._animating = False
        self._drag_button = None
        self._drag_mode = None
        self._press_pos = QtCore.QPoint()
        self._last_pos = QtCore.QPoint()
        self._hover_measure = None
        self._hover_point = None       # 悬停点（落在本次测量的视图平面上）
        self._hover_snapped = False
        self._measure_nav = False      # 本次按压是否已转为视图导航（拖动旋转）

        # 测量取点用的屏幕投影索引（纯 CPU，绝不调用 GL）：把当前视图下的
        # 模型投影到屏幕建二维格子，点击只在该格内做精确求交。
        self._screen_picker = ScreenPicker()
        # 索引用后台线程构建：菜单/快捷键进入测量、视图静止后都只排队，
        # 绝不在 GUI 线程里跑全量投影（大模型会顿几百毫秒）。
        self._pick_pool = QtCore.QThreadPool(self)
        self._pick_pool.setMaxThreadCount(1)
        self._pick_bridge = _PickBridge(self)
        self._pick_bridge.ready.connect(self._on_pick_ready,
                                        QtCore.Qt.QueuedConnection)
        self._pick_building = False
        self._pick_token = 0

        # 视图停下来后（250ms 没有新的重绘请求）提前把索引建好：
        # 用户旋转完视图再点下去时就不用等那一次重建。
        self._pick_prepare_timer = QtCore.QTimer(self)
        self._pick_prepare_timer.setSingleShot(True)
        self._pick_prepare_timer.timeout.connect(self._prepare_measure_pick)

        # 悬停取点用「停手再算」：鼠标一动就做表面拾取会拖慢大模型，
        # 预览直接按本次测量的视图平面算（与最终读数完全同口径）。
        self._hover_pick_timer = QtCore.QTimer(self)
        self._hover_pick_timer.setSingleShot(True)
        self._hover_pick_timer.timeout.connect(self._hover_pick)

        # ESC 取消测量：除了 keyPressEvent，这里再挂一个快捷键，
        # 保证焦点落在子控件（GL/叠加层/面板）时也能取消。
        esc = QtWidgets.QShortcut(QtGui.QKeySequence(QtCore.Qt.Key_Escape), self)
        esc.setContext(QtCore.Qt.WidgetWithChildrenShortcut)
        esc.activated.connect(self._cancel_measure)
        self._esc_shortcut = esc

        # 拖拽时按实测耗时自适应分辨率，保证大模型也能流畅旋转
        self._quality_ceiling = float(
            np.clip(state.settings.interactive_quality, 0.15, 1.0))
        self._adaptive_scale = self._quality_ceiling
        # 渐进式渲染：先立刻出一张低分辨率快帧，空闲后再精修成整幅质量。
        # 静止帧 ss=2 要几百毫秒，若同步画完会阻塞事件循环 —— 拖拽期间鼠标
        # 事件只能排队，松手后模型「跳」一大截，观感就是又卡又不好控制。
        self._quick = True
        self._refine_timer = QtCore.QTimer(self)
        self._refine_timer.setSingleShot(True)
        self._refine_timer.timeout.connect(self._refine)

        # 单线程渲染池：同一时刻只跑一帧，忙时只记「有待渲染」，不堆积任务。
        # 这样鼠标移动事件永远能立刻被处理，相机始终取最新状态出帧。
        self._pool = QtCore.QThreadPool(self)
        self._pool.setMaxThreadCount(1)
        self._bridge = _RenderBridge(self)
        self._bridge.frameReady.connect(self._on_frame,
                                        QtCore.Qt.QueuedConnection)
        self._worker_renderer = SoftwareRenderer()
        self._render_busy = False
        self._pending_render = False
        self._token = 0
        self._task = None
        self._task_scale = 1.0
        self._task_reduced = True
        self._last_frame_token = -1

        # ---- 硬件视口 ----
        # 84 万面模型用纯软件光栅化单帧 >1s，靠后台线程只能做到「不阻塞」，
        # 做不到「不卡」。有 OpenGL 时把实时显示交给 GPU，软件渲染器只用于
        # 导出与回退。离屏测试平台没有 GL 上下文，直接走原来的软件路径。
        self._gl_active = False
        self._gl_probed = False
        self._gl = None
        self._overlay = None
        if _gl_platform_available() and state.settings.use_opengl:
            try:
                self._gl = GLViewport(state, self, self._gl_frame_done)
                self._gl.setGeometry(0, 0, self.width(), self.height())
                self._gl.hide()
                self._overlay = ViewportOverlay(self._draw_overlay, self)
                self._overlay.setGeometry(0, 0, self.width(), self.height())
                self._overlay.hide()
            except Exception:
                self._gl = None
                self._overlay = None

        # ---- CAD 风格导航 ----
        settings = state.settings
        nav_options = NavOptions(
            style=settings.nav_style,
            rotate_mode=settings.nav_rotate_mode,
            zoom_anchor=settings.nav_zoom_anchor,
            pivot=settings.nav_pivot,
            inertia=settings.nav_inertia,
        )
        self._nav = NavigationController(
            state.camera, nav_options,
            {"rotate": settings.rotate_speed, "pan": settings.pan_speed,
             "zoom": settings.zoom_speed})
        self._nav_active = False
        self._saved_scale = self._adaptive_scale

        # 交互事件只做「相机数学 + 一次 update」；状态栏、精修、resize、
        # 惯性都交给定时器合并，鼠标事件循环因此不会被任何一次昂贵操作堵住。
        self._status_timer = QtCore.QTimer(self)
        self._status_timer.setSingleShot(True)
        self._status_timer.timeout.connect(self._publish_status)
        self._nav_idle_timer = QtCore.QTimer(self)
        self._nav_idle_timer.setSingleShot(True)
        self._nav_idle_timer.timeout.connect(self._finish_nav)
        self._resize_timer = QtCore.QTimer(self)
        self._resize_timer.setSingleShot(True)
        self._resize_timer.timeout.connect(self._resize_render)
        self._spin_timer = QtCore.QTimer(self)
        self._spin_timer.timeout.connect(self._spin_step)
        self._last_spin = time.perf_counter()
        self.setCursor(QtCore.Qt.OpenHandCursor)

    # ---------------- 硬件视口接入 ----------------
    def showEvent(self, event) -> None:
        super().showEvent(event)
        if self._gl is not None and not self._gl_probed:
            # 等到窗口真正显示、布局稳定后再探测 GL 上下文
            QtCore.QTimer.singleShot(0, self._probe_gl)

    def _probe_gl(self) -> None:
        if self._gl_probed or self._gl is None:
            return
        self._gl_probed = True
        app = QtWidgets.QApplication.instance()
        try:
            self._gl.setGeometry(0, 0, self.width(), self.height())
            self._gl.show()
            self._gl.update()
            if app is not None:
                app.processEvents()
            ok = bool(self._gl.ready and self._gl.isValid()
                      and self._gl.context() is not None)
        except Exception:
            ok = False
        if not ok:
            self._gl.hide()
            self._gl_active = False
            self._overlay.hide()
            self._request_render()
            return
        self._gl_active = True
        self._overlay.setGeometry(0, 0, self.width(), self.height())
        self._overlay.show()
        self._overlay.raise_()
        self._gl.update()
        # 软件渲染队列里可能还有任务，收掉即可，避免白占 CPU
        self._pool.clear()
        self._dirty = False
        self._publish_status()

    def _gl_frame_done(self, seconds: float) -> None:
        """GPU 帧计时写回状态栏（与软件帧共用同一套统计）。"""
        state = self.state
        state.last_render_seconds = seconds
        now = time.perf_counter()
        state._frame_times.append(now)
        while len(state._frame_times) > 1 and now - state._frame_times[0] > 1.0:
            state._frame_times.popleft()
        span = state._frame_times[-1] - state._frame_times[0]
        state.fps = (len(state._frame_times) - 1) / span if span > 0 else 0.0

    def _publish_status(self) -> None:
        try:
            self.statusMessage.emit(self._status_text())
        except Exception:
            pass

    # ---------------- 交互调度（流畅度） ----------------
    def _queue_status(self) -> None:
        """合并状态栏刷新：拖动时每 ~70ms 只格式化一次文本。"""
        if not self._status_timer.isActive():
            self._status_timer.start(self.STATUS_THROTTLE_MS)

    def _nav_begin(self) -> None:
        """进入导航窗口：预测性降画质，避免第一帧就掉帧。"""
        if not self._nav_active:
            self._saved_scale = self._adaptive_scale
            self._adaptive_scale = min(self._adaptive_scale, self.NAV_QUALITY)
        self._nav_active = True
        self._nav_note()

    def _nav_note(self) -> None:
        """记录一次导航活动，推迟「恢复全质量帧」的时机。"""
        self._nav_idle_timer.start(self.NAV_IDLE_MS)

    def _finish_nav(self) -> None:
        """导航停手：退出交互画质并请求一张全质量精修帧。"""
        if self._drag_button is not None or self._spin_timer.isActive():
            self._nav_note()
            return
        if not self._nav_active:
            return
        self._nav_active = False
        self._adaptive_scale = max(self._adaptive_scale,
                                   min(self._quality_ceiling, self._saved_scale))
        self._quick = True
        self.mark_dirty()
        self._queue_status()

    def _spin_step(self) -> None:
        """惯性旋转的一帧；速度衰减到阈值以下自动停止。"""
        now = time.perf_counter()
        dt = now - self._last_spin
        self._last_spin = now
        if not self._nav.advance(dt):
            self._spin_timer.stop()
            self._nav_note()
        self.mark_dirty()
        self._queue_status()

    def _resize_render(self) -> None:
        self.mark_dirty()

    def set_nav_options(self, options: NavOptions, speeds: dict | None = None) -> None:
        """由「视图选项」对话框更新导航方案与灵敏度。"""
        self._nav.configure(options, speeds)
        self._queue_status()

    def _sync_nav_speeds(self) -> None:
        """每次开始交互时同步一次灵敏度（设置在会话之间可随时修改）。"""
        s = self.state.settings
        self._nav.configure(speeds={"rotate": s.rotate_speed,
                                    "pan": s.pan_speed,
                                    "zoom": s.zoom_speed})

    def _button_bits(self, event) -> tuple:
        """Qt 按键枚举 -> navigation.py 的抽象位标记。"""
        def bits(buttons):
            value = 0
            if buttons & QtCore.Qt.LeftButton:
                value |= BTN_LEFT
            if buttons & QtCore.Qt.MiddleButton:
                value |= BTN_MIDDLE
            if buttons & QtCore.Qt.RightButton:
                value |= BTN_RIGHT
            return value
        return bits(event.button()), bits(event.buttons())

    @staticmethod
    def _mod_names(event) -> tuple:
        mods = event.modifiers()
        names = []
        if mods & QtCore.Qt.ShiftModifier:
            names.append("shift")
        if mods & QtCore.Qt.ControlModifier:
            names.append("ctrl")
        if mods & QtCore.Qt.AltModifier:
            names.append("alt")
        return tuple(names)

    # ---------------- 渲染 ----------------
    def mark_dirty(self) -> None:
        self._dirty = True
        if self.state.measures.pending is not None:
            # 测量中：每次重绘都重置这个「视图已静止」计时器，静止 250ms 后
            # 重建屏幕投影索引（拖动过程中不会触发，所以旋转不卡）。
            self._pick_prepare_timer.start(250)
        if self._gl is not None and not self._gl_probed:
            # 尚未探测 GL：先别启动昂贵的软件帧，等 probe 决定走哪条路径
            self.update()
            return
        if self._gl_active:
            # GPU 路径：一次 update 即可，不必再排软件帧 / 精修计时器
            self._gl.update()
            if self._overlay is not None:
                self._overlay.update()
            self.update()
            return
        self.update()
        # 滚轮 / 惯性 / 连续缩放都算「交互中」：此时绝不能插一张全质量精修帧，
        # 否则连续滚动会在每一档之后各卡一次。
        if not (self._interactive or self._animating or self._nav_active):
            self._quick = True
            self._refine_timer.start(self.REFINE_DELAY_MS)

    def _refine(self) -> None:
        """空闲一段时间后把快帧替换成全质量帧。"""
        # 手动触发精修（离屏测试 / 外部调用）时，若导航已停手则先退出
        # 交互画质窗口；真实运行里精修计时器不会在 _nav_active 时启动。
        if (self._nav_active and self._drag_button is None
                and not self._spin_timer.isActive()):
            self._finish_nav()
        if (self._interactive or self._animating or self._nav_active
                or not self._quick):
            return
        self._quick = False
        self._dirty = True
        self.update()

    def resizeEvent(self, event) -> None:
        self.state.viewport = (self.width(), self.height())
        self.state.camera.set_viewport(self.width(), self.height())
        if self._gl is not None:
            self._gl.setGeometry(0, 0, self.width(), self.height())
        if self._overlay is not None:
            self._overlay.setGeometry(0, 0, self.width(), self.height())
        # 拖窗口边框会触发几十次 resize；合并成一次渲染请求，避免逐像素排队。
        self._resize_timer.start(self.RESIZE_DEBOUNCE_MS)
        super().resizeEvent(event)

    def paintEvent(self, event) -> None:
        """只合成已完成的位图；绝不在绘制事件里同步跑光栅化。"""
        if self._gl is not None and not self._gl_probed:
            # 探测结果还没出来，先不排软件帧；等 0ms 定时器决定后端
            return
        if self._gl_active:
            # 硬件视口是覆盖整个画布的子控件，父控件的绘制被它遮住。
            # 叠加层单独刷新，避免再启动昂贵的软件渲染任务。
            if self._overlay is not None:
                self._overlay.update()
            return
        painter = QtGui.QPainter(self)
        bg = self.state.options.background
        painter.fillRect(self.rect(), QtGui.QColor.fromRgbF(*bg))
        if self._dirty or self._image is None:
            self._request_render()
        if self._image is not None:
            # 只有确实降采样放大时才平滑，全分辨率帧保持 1:1，避免整体糊化
            painter.setRenderHint(QtGui.QPainter.SmoothPixmapTransform,
                                  self._image.width() < self.width())
            painter.drawImage(QtCore.QRectF(self.rect()), self._image)
        self._draw_overlay(painter)

    def _scaled_size(self, scale: float) -> tuple:
        """按窗口比例缩放渲染尺寸。

        位图会被直接拉伸铺满整个视口，因此 w/h 必须与窗口比例严格一致。
        早先 w、h 各自 ``int(W*scale)`` 再各自 ``max(..., 32)``，比例会漂移
        （窄窗口下实测偏差可达 40%），放大后就表现为模型被拉扁或拉长。
        """
        W = max(self.width(), 1)
        H = max(self.height(), 1)
        scale = float(np.clip(scale, 0.05, 1.0))
        h = max(int(round(H * scale)), 1)
        h = min(max(h, self.MIN_RENDER_EDGE), H)
        w = max(int(round(W * h / H)), 1)
        return min(w, W), h

    # ---------------- 帧调度 ----------------
    def _frame_params(self) -> tuple:
        """本帧的 (位图尺寸, 渲染选项, 分辨率比例, 是否快帧)。

        像素预算保证任何一帧都是「短任务」：后台线程不会长时间占着渲染池，
        用户随时开始拖动都能很快拿到新的交互帧。导出的高清大图走
        MainWindow._render_picture，不受这里的预算约束。
        """
        reduced = (self._interactive or self._animating or self._quick
                   or self._nav_active)
        scale = 1.0
        if reduced:
            scale = float(np.clip(self._adaptive_scale, self.MIN_QUALITY, 1.0))
            area = max(self.width(), 1) * max(self.height(), 1)
            scale = min(scale, max(self.MIN_QUALITY,
                                   math.sqrt(self.MAX_INTERACTIVE_PIXELS / area)))
        size = self._scaled_size(scale)
        opts = self.state.render_options(interactive=reduced)
        if opts.supersample > 1:
            budget = math.sqrt(self.MAX_RENDER_PIXELS /
                               max(size[0] * size[1], 1))
            opts.supersample = max(1, min(int(opts.supersample), int(budget)))
        return size, opts, scale, reduced

    def _render(self) -> None:
        """同步渲染一帧（供离屏测试与可视化脚本直接调用）。

        正常交互走 _request_render → 后台 worker → _on_frame，paintEvent
        绝不在这里跑光栅化，否则软件渲染会阻塞 Qt 事件循环。
        """
        state = self.state
        size, opts, scale, reduced = self._frame_params()
        # 供 pick / ray 使用的视口始终按窗口尺寸；渲染内部临时切换到位图尺寸
        state.camera.set_viewport(self.width(), self.height())
        start = time.perf_counter()
        image = state.renderer.render(
            state.scene, state.camera, size=size, options=opts,
            light=state.light, measurements=state.measures.items)
        self._publish(image, time.perf_counter() - start, scale, reduced)

    def _request_render(self) -> None:
        """提交一帧后台渲染；忙时只记「有待渲染」，不排队堆积任务。"""
        if self._render_busy:
            self._pending_render = True
            return
        state = self.state
        size, opts, scale, reduced = self._frame_params()
        # 状态快照：worker 不接触运行中的相机 / 选项 / 对象缓存
        snap_scene = Scene()
        snap_scene.objects = [copy.copy(o) for o in state.scene.objects]
        snap_scene.background = state.scene.background
        snap_scene.background_top = state.scene.background_top
        snap_scene.gradient_background = state.scene.gradient_background
        snap_cam = state.camera.copy()
        snap_cam.set_viewport(self.width(), self.height())
        self._token += 1
        self._task_scale = scale
        self._task_reduced = reduced
        task = _RenderTask(self._worker_renderer, snap_scene, snap_cam,
                           size, copy.copy(opts), copy.copy(state.light),
                           list(state.measures.items), self._bridge, self._token)
        self._task = task
        self._render_busy = True
        self._pending_render = False
        self._dirty = False
        self._pool.start(task)

    def _on_frame(self, image, seconds: float, token: int) -> None:
        """worker 完成：发布位图；若期间状态又变了，立刻接上下一帧。"""
        self._render_busy = False
        self._task = None
        # token 过期说明期间已提交过更新的任务，那一帧会自己发布，无需补帧。
        need_more = self._dirty or self._pending_render
        if token == self._token and image is not None:
            self._last_frame_token = token
            self._publish(image, seconds, self._task_scale,
                          self._task_reduced)
        self.update()
        if need_more:
            self._request_render()

    def _publish(self, image, elapsed: float, scale: float,
                 reduced: bool = True) -> None:
        """位图 → QImage，并更新耗时 / 帧率 / 自适应画质。"""
        state = self.state
        self._buffer = np.ascontiguousarray(image)
        hh, ww, ch = self._buffer.shape
        fmt = QtGui.QImage.Format_RGBA8888 if ch == 4 else QtGui.QImage.Format_RGB888
        self._image = QtGui.QImage(self._buffer.data, ww, hh,
                                   self._buffer.strides[0], fmt)
        self._dirty = False
        state.last_render_seconds = elapsed
        now = time.perf_counter()
        state._frame_times.append(now)
        while len(state._frame_times) > 1 and now - state._frame_times[0] > 1.0:
            state._frame_times.popleft()
        span = state._frame_times[-1] - state._frame_times[0]
        state.fps = (len(state._frame_times) - 1) / span if span > 0 else 0.0
        if reduced:
            self._adapt_quality(scale, elapsed)

    def _frame_from_qimage(self, qimg, W: int, H: int,
                           overlay: bool) -> np.ndarray:
        """QImage → 一份 **自己拥有内存** 的 (H,W,3) uint8 数组。

        这里必须显式 ``copy=True``：``np.frombuffer(qimg.bits())`` 只是借了
        QImage 的像素内存，``np.ascontiguousarray`` 对本来就连续的切片不会
        复制。若把它交给后台线程编码，局部 QImage 一释放就是野指针，
        真实 GL 下会直接 access violation（离屏软件路径常常侥幸不崩）。
        """
        if qimg is None or qimg.isNull() or qimg.width() < 1 or qimg.height() < 1:
            qimg = QtGui.QImage(W, H, QtGui.QImage.Format_RGB888)
            bg = np.clip(np.asarray(self.state.options.background,
                                    dtype=np.float64) * 255.0, 0, 255)
            qimg.fill(QtGui.QColor(int(bg[0]), int(bg[1]), int(bg[2])))
            overlay = True
        if qimg.width() != W or qimg.height() != H:
            qimg = qimg.scaled(W, H, QtCore.Qt.IgnoreAspectRatio,
                               QtCore.Qt.SmoothTransformation)
        if overlay:
            painter = QtGui.QPainter(qimg)
            try:
                self._draw_overlay(painter)
            finally:
                painter.end()
        rgb = qimg.convertToFormat(QtGui.QImage.Format_RGB888).copy()
        stride = int(rgb.bytesPerLine())
        ptr = rgb.bits()
        ptr.setsize(int(rgb.byteCount()))
        flat = np.frombuffer(ptr, dtype=np.uint8).reshape(rgb.height(), stride)
        frame = np.array(flat[:, :W * 3].reshape(H, W, 3),
                         dtype=np.uint8, copy=True, order="C")
        return frame

    def snapshot(self) -> np.ndarray:
        """抓取当前画布像素 (H,W,3) uint8 —— 所见即所得，绝不重新渲染。

        「保存图片」要的就是屏幕上已经画好的那一帧。以前会另起一次
        2 倍超采样全场景重渲染，等于把最贵的一帧再算一遍；这里直接从
        Qt 已合成的画布取图（软件路径 grab 整个 widget，GL 路径读帧缓冲
        再补画 2D 叠加层），耗时只有一次内存拷贝。
        """
        W, H = max(self.width(), 1), max(self.height(), 1)
        if self._gl_active and self._gl is not None:
            try:
                qimg = self._gl.grabFramebuffer()
            except Exception:
                qimg = None
            if qimg is not None and not qimg.isNull():
                # GL 帧缓冲里没有 Qt 叠加层（引导线 / 测量文字 / 提示框）。
                return self._frame_from_qimage(qimg, W, H, True)
            # 读帧缓冲失败时绝不能退化成 self.grab()：把含有可见
            # QOpenGLWidget 子控件的 widget 抓成 QPixmap 会让部分驱动在
            # 非绘制阶段再做一次 GL 绘制，是硬崩溃。改用上一次软件帧兜底。
            buf = getattr(self, "_buffer", None)
            if isinstance(buf, np.ndarray) and buf.ndim == 3 and buf.shape[2] >= 3:
                base = np.ascontiguousarray(buf[:, :, :3]).copy()
                qimg = QtGui.QImage(base.data, base.shape[1], base.shape[0],
                                    base.strides[0], QtGui.QImage.Format_RGB888)
                return self._frame_from_qimage(qimg.copy(), W, H, True)
            return self._frame_from_qimage(None, W, H, True)
        return self._frame_from_qimage(self.grab().toImage(), W, H, False)

    def shutdown(self) -> None:
        """窗口关闭时收掉后台任务，避免任务回调打到已销毁的画布。"""
        self._refine_timer.stop()
        for name in ("_status_timer", "_nav_idle_timer", "_resize_timer",
                     "_spin_timer", "_hover_pick_timer"):
            timer = getattr(self, name, None)
            if timer is not None:
                timer.stop()
        if hasattr(self, "_pick_prepare_timer"):
            self._pick_prepare_timer.stop()
        if self._gl is not None:
            try:
                self._gl.invalidate()
            except Exception:
                pass
        self._pool.clear()
        # 不能直接 pool.waitForDone()：它是阻塞的 C++ 调用且不释放 GIL，
        # 渲染线程若正在 Python 侧（numpy 之外的少量语句）就会等 GIL 等到死，
        # 表现为「关窗口后进程不退」。这里用 sleep 让出 GIL 轮询，2 秒封顶。
        deadline = time.monotonic() + 2.0
        while time.monotonic() < deadline:
            try:
                if self._pool.activeThreadCount() <= 0:
                    break
            except Exception:
                break
            time.sleep(0.005)

    def _adapt_quality(self, scale: float, seconds: float) -> None:
        """按上一帧耗时把交互分辨率往目标帧率附近调。

        留 1.5x / 0.5x 的迟滞带，避免在临界点附近来回抖动导致忽清忽糊。
        """
        target = self.TARGET_FRAME_SECONDS
        if seconds > target * 1.5:
            self._adaptive_scale = max(self.MIN_QUALITY, scale * 0.80)
        elif seconds < target * 0.5:
            self._adaptive_scale = min(self._quality_ceiling, scale * 1.12)

    def set_interactive_quality(self, value: float) -> None:
        """由设置对话框调整交互分辨率上限。"""
        self._quality_ceiling = float(np.clip(value, 0.2, 1.0))
        self._adaptive_scale = min(self._adaptive_scale, self._quality_ceiling)

    def set_animating(self, value: bool) -> None:
        """旋转动画 / 宏播放期间同样降分辨率渲染，保证流畅。"""
        value = bool(value)
        if value == self._animating:
            return
        self._animating = value
        self.mark_dirty()               # 开/关都重绘（关闭时恢复全质量）

    def _draw_overlay(self, painter: QtGui.QPainter) -> None:
        """在视口上叠加测量提示与旋转状态（文字由 Qt 绘制）。"""
        state = self.state
        tr = state.tr
        if self._gl_active:
            # 软件渲染器负责画这些引导元素；GL 路径改由 2D 叠加层投影绘制
            self._draw_guides(painter)
        painter.setRenderHint(QtGui.QPainter.Antialiasing, True)
        pending = state.measures.pending
        if pending is not None:
            need = pending.needed
            have = len(pending.points)
            text = self._measure_hint_text(pending)
            font = painter.font()
            font.setPointSize(max(font.pointSize(), 10))
            font.setBold(True)
            painter.setFont(font)
            metrics = QtGui.QFontMetrics(font)
            rect = metrics.boundingRect(text).adjusted(-10, -6, 10, 6)
            rect.moveTopLeft(QtCore.QPoint(12, 12))
            painter.fillRect(rect, QtGui.QColor(0, 0, 0, 150))
            painter.setPen(QtGui.QColor(255, 220, 90))
            painter.drawText(rect, QtCore.Qt.AlignCenter, text)

        # 已点下的临时点：立刻画成十字标记，用户能确认「这一点已经记下了」
        # （以前 pending 只存在数据里、画面上什么都没有，很容易以为是没点上）。
        # 实心点 = 该像素下有模型表面点（附带 3D 读数），空心点 = 空白处。
        if pending is not None and pending.points:
            mcol = state.measures.color
            pen = QtGui.QPen(QtGui.QColor(int(mcol[0] * 255),
                                         int(mcol[1] * 255),
                                         int(mcol[2] * 255)), 1.6)
            painter.setPen(pen)
            for i, p in enumerate(pending.points):
                s, _, v = state.camera.project(
                    np.asarray([p], dtype=np.float64))
                if not bool(v[0]):
                    continue
                px, py = float(s[0, 0]), float(s[0, 1])
                painter.setBrush(QtCore.Qt.NoBrush)
                painter.drawLine(QtCore.QPointF(px - 6, py),
                                 QtCore.QPointF(px + 6, py))
                painter.drawLine(QtCore.QPointF(px, py - 6),
                                 QtCore.QPointF(px, py + 6))
                snapped = bool(pending.snapped[i]) \
                    if i < len(pending.snapped) else False
                painter.setBrush(
                    QtGui.QColor(int(mcol[0] * 255), int(mcol[1] * 255),
                                 int(mcol[2] * 255)) if snapped
                    else QtCore.Qt.NoBrush)
                painter.drawEllipse(QtCore.QPointF(px, py), 2.2, 2.2)
                painter.setBrush(QtCore.Qt.NoBrush)

        # 测量悬停预览（橡皮筋 + 实时段长）：让用户在按下鼠标之前就看清
        # 「下一个点」会落在屏幕视图的何处、当前这一段多长。任意位置可拾取。
        # 预览点与最终取点都落在同一个视图平面上，因此按下鼠标之前看到的
        # 长度就是最终结果（默认口径为投影长度）。
        if pending is not None and self._hover_measure is not None:
            cam = state.camera
            hx, hy = self._hover_measure.x(), self._hover_measure.y()
            hp = self._hover_point if self._hover_point is not None \
                else cam.ray_to_plane(hx, hy)
            mcol = state.measures.color
            pen = QtGui.QPen(QtGui.QColor(int(mcol[0] * 255),
                                         int(mcol[1] * 255),
                                         int(mcol[2] * 255)), 1.0)
            painter.setPen(pen)
            painter.drawLine(QtCore.QPointF(hx - 7, hy), QtCore.QPointF(hx + 7, hy))
            painter.drawLine(QtCore.QPointF(hx, hy - 7), QtCore.QPointF(hx, hy + 7))
            if self._hover_snapped:
                painter.setBrush(QtGui.QColor(int(mcol[0] * 255),
                                              int(mcol[1] * 255),
                                              int(mcol[2] * 255)))
                painter.drawEllipse(QtCore.QPointF(hx, hy), 2.0, 2.0)
                painter.setBrush(QtCore.Qt.NoBrush)
            if len(pending.points) >= 1:
                last = pending.points[-1]
                s0, _, v0 = cam.project(np.asarray([last], dtype=np.float64))
                s1, _, v1 = cam.project(np.asarray([hp], dtype=np.float64))
                if bool(v0[0]) and bool(v1[0]):
                    dash = QtGui.QPen(QtGui.QColor(int(mcol[0] * 255),
                                                  int(mcol[1] * 255),
                                                  int(mcol[2] * 255)), 1.0,
                                      QtCore.Qt.DashLine)
                    painter.setPen(dash)
                    painter.drawLine(QtCore.QPointF(float(s0[0, 0]), float(s0[0, 1])),
                                     QtCore.QPointF(float(s1[0, 0]), float(s1[0, 1])))
                    seg = float(np.linalg.norm(hp - last))
                    if seg > 0 and pending.kind != "circle2p":
                        # 两点测圆时这条线的长度就是半径，由下方 ⌀/R 标注给出，
                        # 不再重复显示段长。
                        label = (f"{state.units.format_length(seg)} "
                                 f"{state.units.label}")
                        painter.setPen(QtGui.QColor(255, 220, 90))
                        painter.drawText(
                            QtCore.QPointF(float(s1[0, 0]) + 8,
                                          float(s1[0, 1]) - 8), label)

                    # 两点测圆：第一点是圆心，点下后圆即随光标实时缩放，
                    # 再点一下才确定半径（所见即所得）。
                    if pending.kind == "circle2p":
                        center = pending.points[0]
                        normal = pending.plane_normal if pending.plane_normal \
                            is not None else cam.forward()
                        circ2 = mu.circle_from_two_points(center, hp, normal)
                        if circ2 is not None:
                            c, radius, cn = circ2
                            circ = mu.circle_points(c, radius, cn, 96)
                            circ = np.vstack([circ, circ[:1]])
                            sc, _, vc = cam.project(circ)
                            painter.setPen(QtGui.QPen(
                                QtGui.QColor(int(mcol[0] * 255),
                                             int(mcol[1] * 255),
                                             int(mcol[2] * 255)), 1.4))
                            for k in range(len(circ) - 1):
                                if bool(vc[k]) and bool(vc[k + 1]):
                                    painter.drawLine(
                                        QtCore.QPointF(float(sc[k, 0]),
                                                       float(sc[k, 1])),
                                        QtCore.QPointF(float(sc[k + 1, 0]),
                                                       float(sc[k + 1, 1])))
                            painter.setPen(QtGui.QColor(255, 220, 90))
                            painter.drawText(
                                QtCore.QPointF(float(s1[0, 0]) + 10,
                                              float(s1[0, 1]) - 12),
                                f"\u2300{state.units.format_length(radius * 2.0)}"
                                f" R={state.units.format_length(radius)}"
                                f" {state.units.label}")
        if state.spin.config.enabled:
            painter.setPen(QtGui.QColor(120, 220, 160))
            font = painter.font()
            font.setPointSize(max(font.pointSize(), 9))
            painter.setFont(font)
            painter.drawText(12, self.height() - 12,
                             f"\u25b6 spin: {state.spin.config.kind} "
                             f"({state.spin.config.speed})")

        # 视图支点标记：旋转默认绕画布中心（可设为绕按下点），
        # 缩放默认以光标为锚点；这里把当前支点显式画出来。
        # 开启完整十字准线时由渲染器画，避免重复叠一层。
        if not state.options.crosshair:
            cx, cy = self.width() / 2.0, self.height() / 2.0
            painter.setPen(QtGui.QPen(QtGui.QColor(168, 182, 205, 120), 1.0))
            painter.setBrush(QtCore.Qt.NoBrush)
            for a, b in (((cx - 11, cy), (cx - 5, cy)),
                         ((cx + 5, cy), (cx + 11, cy)),
                         ((cx, cy - 11), (cx, cy - 5)),
                         ((cx, cy + 5), (cx, cy + 11))):
                painter.drawLine(QtCore.QPointF(*a), QtCore.QPointF(*b))
            painter.drawEllipse(QtCore.QPointF(cx, cy), 2.5, 2.5)

        # 空场景（启动默认状态）：只给一句提示，不生成任何占位 objects。
        if not state.scene.objects:
            text = ("拖放 STL / PLY 模型到此处，或用「文件 > 打开」载入"
                    if tr.language == "CH" else
                    "Drop an STL / PLY file here, or use File > Open")
            font = painter.font()
            font.setPointSize(max(font.pointSize(), 12))
            painter.setFont(font)
            rect = QtGui.QFontMetrics(font).boundingRect(text).adjusted(
                -22, -14, 22, 14)
            rect.moveCenter(QtCore.QPoint(self.width() // 2,
                                          self.height() // 2 + 48))
            painter.fillRect(rect, QtGui.QColor(20, 24, 30, 170))
            painter.setPen(QtGui.QColor(205, 214, 228))
            painter.drawText(rect, QtCore.Qt.AlignCenter, text)

    def _draw_guides(self, painter: QtGui.QPainter) -> None:
        """GL 路径的坐标轴 / 网格 / 十字准线 / 测量标注（2D 投影叠加）。"""
        state = self.state
        opts = state.options
        cam = state.camera
        cam.set_viewport(self.width(), self.height())
        W, H = self.width(), self.height()
        has_objects = bool(state.scene.objects)
        painter.setRenderHint(QtGui.QPainter.Antialiasing, True)

        def qcolor(rgb, alpha=255):
            c = np.clip(np.asarray(rgb, dtype=np.float64) * 255.0, 0, 255)
            return QtGui.QColor(int(c[0]), int(c[1]), int(c[2]), alpha)

        if has_objects and opts.show_grid:
            lo, hi = state.scene.bounds()
            scale = max(float(np.linalg.norm(hi - lo)), 1e-9)
            step = scale / 10.0
            extent = step * 10.0
            lines = []
            for t in range(-10, 11):
                v = t * step
                lines.append(((v, -extent, 0.0), (v, extent, 0.0)))
                lines.append(((-extent, v, 0.0), (extent, v, 0.0)))
            pts = np.asarray(lines, dtype=np.float64).reshape(-1, 3)
            s, _, valid = cam.project(pts)
            base = np.asarray(opts.background, dtype=np.float64)
            pen = QtGui.QPen(qcolor(base + 0.09, 130), 1.0)
            painter.setPen(pen)
            for i in range(0, len(valid), 2):
                if valid[i] and valid[i + 1]:
                    p1, p2 = s[i], s[i + 1]
                    if np.isfinite(p1).all() and np.isfinite(p2).all():
                        painter.drawLine(QtCore.QPointF(*p1), QtCore.QPointF(*p2))

        if has_objects and opts.show_axis:
            lo, hi = state.scene.bounds()
            scale = max(float(np.linalg.norm(hi - lo)), 1e-9)
            length = scale * opts.axis_scale
            colors = ((0.86, 0.24, 0.22), (0.34, 0.72, 0.32),
                      (0.26, 0.52, 0.90))
            names = ("X", "Y", "Z")
            metrics = QtGui.QFontMetrics(painter.font())
            tw = (metrics.horizontalAdvance("X") if hasattr(metrics, "horizontalAdvance")
                  else metrics.width("X"))
            th = metrics.height()
            for i, p0, p1 in axis_screen_geometry(
                    cam, W, H, length, 1, opts.axis_scale):
                painter.setPen(QtGui.QPen(qcolor(colors[i]), 2.0))
                painter.drawLine(QtCore.QPointF(*p0), QtCore.QPointF(*p1))
                d = p1 - p0
                norm = float(np.hypot(*d))
                if norm > 12:
                    d = d / norm
                    n = np.array([-d[1], d[0]])
                    for side in (1.0, -1.0):
                        a = p1 - d * 9 + n * 5 * side
                        painter.drawLine(QtCore.QPointF(*a), QtCore.QPointF(*p1))
                if opts.axis_labels:
                    tx = float(np.clip(p1[0] + 8, 0, max(W - tw - 2, 0)))
                    ty = float(np.clip(p1[1] - 4, th, max(H - 2, 0)))
                    painter.drawText(QtCore.QPointF(tx, ty), names[i])

        if opts.crosshair:
            cx, cy = W / 2.0, H / 2.0
            gap = 4.0
            base = np.asarray(opts.background, dtype=np.float64)
            painter.setPen(QtGui.QPen(qcolor(1.0 - base * 0.55), 1.0))
            for a, b in (((0.0, cy), (cx - gap, cy)),
                         ((cx + gap, cy), (float(W - 1), cy)),
                         ((cx, 0.0), (cx, cy - gap)),
                         ((cx, cy + gap), (cx, float(H - 1)))):
                painter.drawLine(QtCore.QPointF(*a), QtCore.QPointF(*b))

        for item in state.measures.items:
            col = state.measures.color if hasattr(state.measures, "color") \
                else (1.0, 0.85, 0.25)
            pen = QtGui.QPen(qcolor(col), 1.5)
            painter.setPen(pen)
            for poly in item.polylines():
                if len(poly) < 2:
                    continue
                s, _, valid = cam.project(np.asarray(poly, dtype=np.float64))
                for i in range(len(poly) - 1):
                    if valid[i] and valid[i + 1]:
                        painter.drawLine(QtCore.QPointF(*s[i]),
                                         QtCore.QPointF(*s[i + 1]))
            painter.setPen(QtGui.QPen(qcolor(col), 2.0))
            for p in item.markers():
                s, _, valid = cam.project(np.asarray([p], dtype=np.float64))
                if valid[0]:
                    painter.drawLine(QtCore.QPointF(s[0, 0] - 4, s[0, 1]),
                                     QtCore.QPointF(s[0, 0] + 4, s[0, 1]))
                    painter.drawLine(QtCore.QPointF(s[0, 0], s[0, 1] - 4),
                                     QtCore.QPointF(s[0, 0], s[0, 1] + 4))
            text = item.label()
            if text:
                anchor = item.label_anchor()
                if item.kind in ("circle2p", "circle3p"):
                    cc = item.circle
                    if cc is not None:
                        # 圆的标注放到圆周右侧，避免压在圆心十字上
                        anchor = cc[0] + cam.right() * float(cc[1])
                s, _, valid = cam.project(
                    np.asarray([anchor], dtype=np.float64))
                if valid[0]:
                    fm = QtGui.QFontMetrics(painter.font())
                    tw = (fm.horizontalAdvance(text) if hasattr(fm, "horizontalAdvance")
                          else fm.width(text))
                    tx = float(np.clip(s[0, 0] + 8, 0, max(W - tw - 2, 0)))
                    ty = float(np.clip(s[0, 1] - 8, fm.height(), max(H - 2, 0)))
                    painter.drawText(QtCore.QPointF(tx, ty), text)

    # ---------------- 鼠标 ----------------
    def _update_cursor(self) -> None:
        mode = self._drag_mode
        if mode in ("rotate", "rotate_z"):
            cursor = QtCore.Qt.ClosedHandCursor
        elif mode in ("pan", "zoom"):
            cursor = QtCore.Qt.SizeAllCursor
        elif mode == "measure" or self.state.measures.pending is not None:
            cursor = QtCore.Qt.CrossCursor
        else:
            cursor = QtCore.Qt.OpenHandCursor
        # GL 视口与叠加层覆盖在 Canvas 之上，光标由最上层控件显示，
        # 必须显式把光标透传给它们，否则看不到十字光标 / 手形。
        self.setCursor(cursor)
        if self._gl is not None:
            self._gl.setCursor(cursor)
        if self._overlay is not None:
            self._overlay.setCursor(cursor)

    def mousePressEvent(self, event) -> None:
        self.setFocus()
        self._sync_nav_speeds()
        self._press_pos = event.pos()
        self._last_pos = event.pos()
        self._drag_button = event.button()
        self._measure_nav = False
        state = self.state
        button, buttons = self._button_bits(event)
        mods = self._mod_names(event)
        action = resolve_action(button, buttons, mods,
                                self._nav.options.style)
        if action == ROTATE and state.settings.view.fixed_on_screen:
            action = ROTATE_Z
        # 定轴旋转：选中 X / Y / Z / 视线后，普通左键拖拽也按该轴旋转；
        # Ctrl+左键始终走定轴，未选轴时保持旧的「绕 Z」习惯。
        fixed_axis = state.rotate_axis
        if fixed_axis not in ROTATE_AXES:
            fixed_axis = "free"
        self._nav.set_axis("z" if fixed_axis == "free" else fixed_axis)
        if action == ROTATE and fixed_axis != "free":
            action = ROTATE_Z
        # 测量取点优先级：左键的普通旋转在测量进行中先当作「可能的点击」，
        # 只有明显拖动后才在 mouseMoveEvent 里转成视图导航。
        if (event.button() == QtCore.Qt.LeftButton and action == ROTATE
                and state.measures.pending is not None):
            action = "measure"
        self._drag_mode = action if action != NONE else None
        self._interactive = self._drag_mode is not None
        if self._interactive:
            self._quick = True
            self._refine_timer.stop()
        self._nav_begin()
        if action != NONE and action != "measure":
            self._nav.begin(action, event.pos().x(), event.pos().y())
        self._update_cursor()

    @staticmethod
    def _soft_step(value: float, cap: float) -> float:
        """兼容旧调用：位移只做线性限幅（见 navigation.limit_step）。

        旧版对小幅位移叠加 0.55~1.0 的增益，慢速微调被系统性压缩，
        手感发黏；现在严格与鼠标位移成正比，与 CAD 软件一致。
        """
        from .navigation import limit_step
        return limit_step(value, cap)

    def mouseMoveEvent(self, event) -> None:
        pos = event.pos()
        state = self.state
        # 测量模式下的悬停预览：即便没有按下鼠标也实时更新「橡皮筋」，让用户
        # 清楚下一个点会落在屏幕视图的什么位置（任意位置都能拾取）。
        if state.measures.pending is not None and self._drag_button is None:
            self._hover_measure = pos
            # 先用视图平面快速算一个预览点（不拾取，移动中零开销），
            # 预览半径与最终半径共用同一个视图平面与圆心，二者严格一致。
            cam = state.camera
            ref = state.measures.pending.points[0] \
                if state.measures.pending.points else cam.target
            self._hover_point = cam.ray_to_plane(pos.x(), pos.y(),
                                                 plane_point=ref)
            self._hover_snapped = False
            self._hover_pick_timer.start(90)
            if self._gl_active and self._overlay is not None:
                self._overlay.update()
            self.update()
            return
        if self._drag_button is None:
            return
        self._last_pos = pos
        state = self.state
        total = float(np.hypot(pos.x() - self._press_pos.x(),
                               pos.y() - self._press_pos.y()))
        if self._drag_mode == "measure":
            if total <= self.MEASURE_CLICK_TOL:
                return
            # 明显拖动才转成视图导航；从按下点开始累积，拖动不丢距离。
            self._drag_mode = ROTATE
            self._measure_nav = True
            self._hover_measure = None
            self._hover_point = None
            self._hover_pick_timer.stop()
            self._nav.begin(ROTATE, self._press_pos.x(), self._press_pos.y())
            self._update_cursor()
        if self._nav.drag(pos.x(), pos.y(), self._mod_names(event)):
            self._nav_note()
            self.mark_dirty()
            self._queue_status()

    def mouseReleaseEvent(self, event) -> None:
        state = self.state
        moved = (event.pos() - self._press_pos).manhattanLength()
        took_point = False
        self._hover_pick_timer.stop()
        if event.button() == QtCore.Qt.LeftButton:
            if state.measures.pending is not None:
                # 测量取点：只要这次按压没有真的变成视图导航（拖动旋转），
                # 就一律算一次取点 —— 不再依赖精确的像素容差，手抖也不会漏点。
                if not self._measure_nav and moved <= self.MEASURE_CLICK_TOL * 3:
                    self._add_measure_point(event.pos())
                    took_point = True
            elif moved <= 4:
                self._pick_info(event.pos())
        elif event.button() == QtCore.Qt.RightButton and moved <= 4:
            self._context_menu(event.globalPos())
        self._drag_button = None
        self._drag_mode = None
        self._interactive = False
        self._measure_nav = False
        self._update_cursor()
        if self._nav.end():
            self._last_spin = time.perf_counter()
            self._spin_timer.start(self.INERTIA_TICK_MS)
        self._nav_note()
        self.mark_dirty()        # 先出快帧，停手后由 _finish_nav 请求精修
        # 取点后由 _add_measure_point 自己给「下一点」提示，别被就绪文本冲掉
        if not took_point or state.measures.pending is None:
            self._queue_status()

    def wheelEvent(self, event) -> None:
        # 支持鼠标滚轮（angleDelta）与触控板（pixelDelta）两种滚动源
        pd = event.pixelDelta()
        if pd is not None and pd.y() != 0:
            steps = pd.y() / 60.0
        else:
            steps = event.angleDelta().y() / 120.0
        if steps:
            # CAD 式 zoom-to-cursor：缩放前后光标下的世界点保持不动；
            # 在「视图选项」里可切回经典的中心锚点。Shift 为精细缩放。
            self._sync_nav_speeds()
            pos = event.pos()
            if self._nav.wheel(pos.x(), pos.y(), steps,
                               self._mod_names(event)):
                self._nav_begin()
                self._nav_note()
                self.mark_dirty()
                self._queue_status()
        event.accept()

    def mouseDoubleClickEvent(self, event) -> None:
        if event.button() == QtCore.Qt.MiddleButton:
            # CAD 习惯：双击中键 = Zoom Fit
            self.state.fit_view()
            self.mark_dirty()
            event.accept()
            return
        if event.button() == QtCore.Qt.LeftButton:
            if self.state.measures.pending is not None:
                # 测量中连续快速点击会被 Qt 合成为「双击」：此时绝不能 fit 视图
                # （否则画面突然跳变，用户以为点坏了）。取点已由上一次 release 完成。
                event.accept()
                return
            self.state.fit_view()
            self.mark_dirty()

    def keyPressEvent(self, event) -> None:
        key = event.key()
        mods = event.modifiers()
        cam = self.state.camera
        handled = True
        if key == QtCore.Qt.Key_F or key == QtCore.Qt.Key_Home:
            self.state.fit_view()
        elif key == QtCore.Qt.Key_R:
            cam.reset_center()
        elif key == QtCore.Qt.Key_Space:
            # 空格与「启用旋转」按钮必须走同一个入口，否则按钮的勾选状态
            # 会和实际动画状态脱节，用户再点一次按钮看起来「没反应」。
            owner = self.window()
            if hasattr(owner, "_set_spin_enabled"):
                owner._set_spin_enabled(not self.state.spin.config.enabled)
            else:
                cfg = self.state.spin.config
                cfg.enabled = not cfg.enabled
                self.state.spin.reset()
                self.mark_dirty()
        elif QtCore.Qt.Key_1 <= key <= QtCore.Qt.Key_8:
            names = ["iso", "front", "back", "left", "right", "top", "bottom",
                     "iso_back"]
            self.state.standard_view(names[key - QtCore.Qt.Key_1])
        elif key == QtCore.Qt.Key_Escape:
            # ESC 取消正在进行的测量（点亮十字光标状态下点 ESC 即退出选点）
            if self.state.measures.pending is not None:
                self._cancel_measure()
            else:
                handled = False
        elif key in (QtCore.Qt.Key_Plus, QtCore.Qt.Key_Equal):
            cam.dolly_about_center(1.0, self.state.settings.zoom_speed)
        elif key in (QtCore.Qt.Key_Minus, QtCore.Qt.Key_Underscore):
            cam.dolly_about_center(-1.0, self.state.settings.zoom_speed)
        elif key in (QtCore.Qt.Key_Left, QtCore.Qt.Key_Right,
                     QtCore.Qt.Key_Up, QtCore.Qt.Key_Down):
            # 键盘用固定角度步进（与鼠标灵敏度设置无关），按一下走一格，可预期
            step = self.KEY_ROTATE_STEP
            move = 18.0
            left = key == QtCore.Qt.Key_Left
            right = key == QtCore.Qt.Key_Right
            up = key == QtCore.Qt.Key_Up
            down = key == QtCore.Qt.Key_Down
            if mods & QtCore.Qt.ShiftModifier:      # 平移（严格跟随步进）
                cam.pan_about_center(move if right else (-move if left else 0.0),
                                     -move if up else (move if down else 0.0),
                                     self.state.settings.pan_speed)
            elif mods & QtCore.Qt.ControlModifier:  # 绕 Z / 俯仰
                if left:
                    cam.set_rotation_angles(yaw=cam.yaw - step)
                elif right:
                    cam.set_rotation_angles(yaw=cam.yaw + step)
                elif up:
                    cam.set_rotation_angles(pitch=cam.pitch + step)
                else:
                    cam.set_rotation_angles(pitch=cam.pitch - step)
            else:                                    # 轨道旋转
                if left:
                    cam.set_rotation_angles(yaw=cam.yaw - step)
                elif right:
                    cam.set_rotation_angles(yaw=cam.yaw + step)
                elif up:
                    cam.set_rotation_angles(pitch=cam.pitch + step)
                else:
                    cam.set_rotation_angles(pitch=cam.pitch - step)
        else:
            handled = False
        if not handled:
            super().keyPressEvent(event)
            return
        self.mark_dirty()
        self._queue_status()

    # ---------------- 拾取 / 测量 ----------------
    def _pick_info(self, pos: QtCore.QPoint) -> None:
        """单击（非测量）时在状态栏显示光标处的信息。

        大模型绝不走 CPU 全量射线相交（84 万面约 170 ms，连点会明显顿住）：
        优先用 GPU 拾取，其次只对小模型做 CPU 拾取，最后退化为视图平面坐标。
        """
        state = self.state
        units = state.units
        x, y = pos.x(), pos.y()
        if self._gl_active and self._gl is not None and self._gl.ready:
            hit = self._gl.pick(x, y, state.scene, state.camera, state.options)
            if hit is None:
                self.statusMessage.emit(state.tr.tr("ready"))
                return
            self.statusMessage.emit(
                f"{hit['object'].name}  {units.format_point(hit['point'])} "
                f"{units.label}")
            return
        # 软件路径用已经建好的屏幕投影索引（格子查询 + 局部求交，微秒级）。
        # 以前这里对 ≤25 万面做全量射线求交，单击会顿几十毫秒；现在索引
        # 没准备好就直接给视图平面坐标，并在后台补建索引供下一次使用。
        hit = self._screen_picker.pick(state.scene, state.camera, x, y,
                                       pad=1.5, allow_build=False)
        if hit is not None:
            self.statusMessage.emit(
                f"{units.format_point(hit)} {units.label}")
            return
        if state.scene.objects:
            self._prepare_measure_pick()
        point = state.camera.ray_to_plane(x, y)
        self.statusMessage.emit(
            f"{units.format_point(point)} {units.label}")

    def _vertex_snap(self, obj, camera, x: float, y: float,
                     world_point, tol_px: float = 12.0) -> tuple:
        """（可选）把世界点吸附到屏幕上最近的顶点 —— 供需要「吸附」的场景调用。

        测量主路径不用它：测量定义在**当前屏幕视图平面**上，任意位置都能取点，
        与模型顶点无关（大模型上逐点吸附还会拖慢点击）。
        """
        import numpy as np
        from . import mathutil as mu
        mesh = obj.mesh
        matrix = getattr(obj, "matrix", None)
        world_verts = mu.transform_points(matrix, mesh.vertices) if matrix \
            else np.asarray(mesh.vertices, dtype=np.float64)
        tol = float(tol_px)
        radius_world = tol * camera.world_per_pixel() * 3.0
        diff = np.linalg.norm(world_verts - np.asarray(world_point), axis=1)
        near = np.nonzero(diff < max(radius_world, 1e-9))[0]
        if len(near) == 0:
            return world_point, False
        scr, _, _ = camera.project(world_verts[near])
        d = np.linalg.norm(scr - np.array([x, y], dtype=np.float64), axis=1)
        k = int(np.argmin(d))
        if d[k] <= tol:
            return world_verts[near[k]].copy(), True
        return world_point, False

    def _measure_hint_text(self, m) -> str:
        """取点提示：两点测圆单独说「点圆心 / 定半径」，其余说第几点。"""
        tr = self.state.tr
        have = len(m.points)
        if m.kind == "circle2p":
            return tr.tr("measure_circle_center" if have == 0
                         else "measure_circle_radius")
        return tr.tr("measure_hint", have + 1, m.needed)

    # ---------------- 测量取点：把模型投影到屏幕 ----------------
    # 取点绝不再调用 GL 拾取（离屏 FBO + glReadPixels 对驱动状态敏感，装上后
    # 会出现「第一个点就点不了」）。改为把当前视图下的模型顶点投影到屏幕、
    # 建二维格子索引（ScreenPicker），点击时只在命中格子里做精确求交：
    # 纯 numpy，微秒级，永不阻塞；拿到的仍是真实表面点，尺寸与视角无关。
    def _surface_hit(self, x: float, y: float, neighbors: bool = True,
                     allow_build: bool = True):
        """光标处的模型表面点（世界坐标）；未命中或不可用时返回 None。

        ``neighbors`` 只影响命中容差（边缘补找），已并入格子索引的包围盒
        外扩里，保留参数是为了调用方兼容。
        ``allow_build=False`` 时若索引已失效就直接返回 None（悬停预览用，
        避免用户旋转视图时鼠标一停就重建索引）。
        """
        state = self.state
        pad = 2.0 if neighbors else 1.0
        return self._screen_picker.pick(state.scene, state.camera, x, y,
                                        pad=pad, allow_build=allow_build)

    def _prepare_measure_pick(self) -> None:
        """后台准备屏幕投影索引（菜单点击 / 视图静止时调用，立即返回）。

        索引与当前视图一致时直接复用；正在构建时不重复排队。构建期间点击
        使用视图平面点，投影长度仍然正确，所以用户不会感到任何等待。
        """
        state = self.state
        if not state.scene.objects or self._pick_building:
            return
        if self._screen_picker.is_current(state.scene, state.camera):
            return
        snap_scene = Scene()
        snap_scene.objects = [copy.copy(o) for o in state.scene.objects]
        snap_scene.background = state.scene.background
        snap_scene.background_top = state.scene.background_top
        snap_scene.gradient_background = state.scene.gradient_background
        snap_cam = state.camera.copy()
        snap_cam.set_viewport(self.width(), self.height())
        self._pick_token += 1
        self._pick_building = True
        self._pick_pool.start(_PickBuildTask(snap_scene, snap_cam,
                                             self._pick_token, self._pick_bridge))

    def _on_pick_ready(self, picker, token: int) -> None:
        """后台索引就绪：只在与当前视图完全一致时安装。"""
        self._pick_building = False
        if picker is None or token != self._pick_token:
            return
        try:
            if picker.is_current(self.state.scene, self.state.camera):
                self._screen_picker = picker
        except Exception:
            pass

    def _measure_point_at(self, x: float, y: float, pending,
                          pick: bool = True, neighbors: bool = True) -> tuple:
        """把一次点击变成测量点。

        返回 ``(view_point, surface_point, plane_point, plane_normal)``：

        * ``view_point`` 是光标射线与**本次测量的视图平面**的交点；一次测量
          内所有点都在这个平面上，所以两点距离就是屏幕上的投影长度。
          在 TOP 视图量一条边时，不会因为一端落在凸台顶面、另一端落在
          底面而把深度差算进长度 —— 这正是 CAD「在视图里量」的口径。
        * ``surface_point`` 是同一像素下的真实模型表面点（没有命中为
          None），只用于可选的 3D 读数/3D 模式，不再影响默认长度。
        * 第一点决定测量平面的深度（取表面点，空白处取支点），随后固定
          不变；中途旋转视图也不会悄悄改变基准。

        取点全程**不在 GUI 线程重建拾取索引**：索引还没准备好时直接用
        视图平面点，点击依然立刻响应，后台会补建索引供下一次使用。
        """
        cam = self.state.camera
        hit = (self._surface_hit(x, y, neighbors=neighbors, allow_build=False)
               if pick else None)
        normal = pending.plane_normal
        anchor = pending.plane_point
        if normal is None:
            normal = cam.forward()
        if anchor is None:
            anchor = hit if hit is not None else cam.target
        # 视图在两次点击之间转过较大角度时，旧平面可能与视线接近平行；
        # 此时退回当前视线方向，避免射线-平面求交退化。
        _, direction = cam.ray(x, y)
        n = np.asarray(normal, dtype=np.float64)
        if abs(float(np.dot(direction, n))) < 1e-6:
            normal = cam.forward()
            anchor = cam.target if pending.plane_point is None \
                else pending.plane_point
        view_point = cam.ray_to_plane(x, y, plane_point=anchor,
                                      plane_normal=normal)
        return view_point, hit, np.asarray(anchor, dtype=np.float64), \
            np.asarray(normal, dtype=np.float64)

    def _hover_pick(self) -> None:
        """鼠标停住后刷新一次预览点（移动中不做，保证拖动不卡）。"""
        state = self.state
        pending = state.measures.pending
        if pending is None or self._drag_button is not None \
                or self._hover_measure is None:
            return
        pos = self._hover_measure
        # 悬停预览复用已有索引；索引失效（用户刚转过视图）就先不重建，
        # 免得停一下鼠标卡一帧 —— 后台会补建，点击始终走视图平面。
        point, surface, _, _ = self._measure_point_at(
            pos.x(), pos.y(), pending, neighbors=False)
        self._hover_point = point
        self._hover_snapped = surface is not None
        if self._overlay is not None:
            self._overlay.update()
        else:
            self.update()

    def _add_measure_point(self, pos: QtCore.QPoint) -> None:
        """在当前屏幕视图平面内取一个测量点。

        画布上 **任意位置**（模型上、模型外、空白背景、空场景）都能取点：
        所有点都落在同一条测量视图平面上，读数就是屏幕投影长度；模型表面
        点另存一份，供 ``3D=`` 读数和可选的三维模式使用。取点只用已经
        建好的拾取索引，**绝不在 GUI 线程重建**，84 万面模型点击也不卡。
        """
        state = self.state
        pending = state.measures.pending
        if pending is None:
            return
        kind = pending.kind
        x, y = pos.x(), pos.y()
        point, surface, anchor, normal = self._measure_point_at(x, y, pending)
        done = state.measures.add_point(
            point, units=state.units, plane=normal, snapped=surface is not None,
            surface=surface, plane_point=anchor)
        if done is not None:
            self._hover_measure = None
            self._hover_point = None
            text = done.value_text(state.units)
            result = (f"{state.tr.tr('measure_result')}: {text}"
                      if text else state.tr.tr("measure_result"))
            delta = done.delta_text(state.units)
            if delta:
                result += f"   |   {delta}"
            tiny = self._measured_in_pixels(done)
            if tiny is not None:
                result += "   |   " + state.tr.tr("measure_zoom_hint")
            self.sceneChanged.emit()
            if self.MEASURE_CONTINUOUS:
                # 连续测量：同一工具保持激活，下一个点即开始新的一次测量
                nxt = state.measures.start(kind)
                result += "   |   " + self._measure_hint_text(nxt)
                self._update_cursor()
            self.statusMessage.emit(result)
        elif state.measures.last_reject:
            # 退化取点（两点重合 / 三点共线）：撤销该点、保持激活并提示
            self._hover_measure = None
            self._hover_point = None
            self.statusMessage.emit(state.tr.tr(state.measures.last_reject))
        else:
            src = state.tr.tr("measure_on_surface" if surface is not None
                              else "measure_on_plane")
            self.statusMessage.emit(f"{self._measure_hint_text(pending)}"
                                    f"   |   {src}")
        self.mark_dirty()

    def _measured_in_pixels(self, m) -> float | None:
        """测量结果在屏幕上的跨度（像素）；小于 ~15px 时提示放大更准。

        0.2mm 的直径在 828px 画布上只有 11 像素，点击精度本身就是 ±1px，
        这类特征必须先放大再量，否则数值必然有百分之几的偏差。
        """
        pts = [p for p in m.points[:2]]
        if len(pts) < 2:
            return None
        cam = self.state.camera
        scr, _, valid = cam.project(np.stack(pts))
        if not bool(np.all(valid)):
            return None
        px = float(np.linalg.norm(scr[1] - scr[0]))
        return px if px < 15.0 else None

    def _context_menu(self, global_pos) -> None:
        menu = QtWidgets.QMenu(self)
        state = self.state
        tr = state.tr
        menu.addAction(tr.tr("zoom_fit"), self._fit)
        menu.addAction(tr.tr("reset_center"), self._reset_center)
        menu.addSeparator()
        for name in ("iso", "front", "top", "right"):
            menu.addAction(tr.tr("view_" + name),
                           lambda n=name: self._standard_view(n))
        menu.addSeparator()
        wire = menu.addAction(tr.tr("wire"), self._toggle_wire)
        wire.setCheckable(True)
        wire.setChecked(state.options.wire)
        solid = menu.addAction(tr.tr("solid"), self._toggle_solid)
        solid.setCheckable(True)
        solid.setChecked(state.options.solid)
        colored = menu.addAction(tr.tr("colored"), self._toggle_colored)
        colored.setCheckable(True)
        colored.setChecked(is_colored_mode(state.options.color_mode))
        menu.addSeparator()
        pending = state.measures.pending
        if pending is not None:
            menu.addAction(f"{tr.tr('measure_result')}: "
                           f"{pending.kind} ({len(pending.points)}/{pending.needed})",
                           lambda: None).setEnabled(False)
            menu.addAction(tr.tr("cancel"), self._cancel_measure)
        else:
            for kind in MEASURE_DEFS:
                menu.addAction(tr.tr(MEASURE_DEFS[kind][1]),
                               lambda k=kind: self.start_measure(k))
        menu.exec_(global_pos)

    def _fit(self) -> None:
        self.state.fit_view()
        self.mark_dirty()

    def _reset_center(self) -> None:
        self.state.camera.reset_center()
        self.mark_dirty()

    def _standard_view(self, name: str) -> None:
        self.state.standard_view(name)
        self.mark_dirty()

    def _toggle_wire(self) -> None:
        self.state.options.wire = not self.state.options.wire
        self.state.options.solid = not self.state.options.wire
        self.mark_dirty()
        self.sceneChanged.emit()

    def _toggle_solid(self) -> None:
        self.state.options.solid = not self.state.options.solid
        if not self.state.options.solid:
            self.state.options.wire = True
        self.mark_dirty()
        self.sceneChanged.emit()

    def _toggle_colored(self) -> None:
        opts = self.state.options
        opts.color_mode = "uniform" if is_colored_mode(opts.color_mode) else "auto"
        for obj in self.state.scene.objects:
            obj.color_mode = opts.color_mode
            obj.auto_color = opts.color_mode != "uniform"
            obj.invalidate_colors()
        self.mark_dirty()
        self.sceneChanged.emit()

    def start_measure(self, kind: str) -> None:
        m = self.state.measures.start(kind)
        self._hover_measure = None
        self._hover_point = None
        self._hover_snapped = False
        self._hover_pick_timer.stop()
        self._prepare_measure_pick()    # 先把屏幕投影索引建好，首点零延迟
        self.statusMessage.emit(self._measure_hint_text(m))
        self._update_cursor()           # 立即切换十字光标（点工具即进入选点）
        self.setFocus()                 # 确保 ESC 能被画布收到
        self.mark_dirty()

    def _cancel_measure(self) -> None:
        was_pending = self.state.measures.pending is not None
        self.state.measures.cancel()
        self._hover_measure = None
        self._hover_point = None
        self._hover_snapped = False
        self._measure_nav = False
        self._hover_pick_timer.stop()
        if self._drag_mode == "measure":       # 拖拽中按 ESC：别让状态卡在 measure
            self._drag_mode = None
            self._interactive = False
        self._update_cursor()
        self.mark_dirty()
        if was_pending:
            self.statusMessage.emit(self.state.tr.tr("measure_cancel"))

    # ---------------- 拖放 ----------------
    def dragEnterEvent(self, event) -> None:
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dropEvent(self, event) -> None:
        for url in event.mimeData().urls():
            path = url.toLocalFile()
            if path and os.path.isfile(path):
                window = self.window()
                if hasattr(window, "open_path"):
                    window.open_path(path)
        event.acceptProposedAction()

    # ---------------- 状态文本 ----------------
    def _status_text(self) -> str:
        state = self.state
        stats = state.statistics()
        mode = ""
        if self._drag_mode and self._drag_mode != "measure":
            mode = {"rotate": "旋转", "rotate_z": "定轴旋转",
                    "pan": "平移", "zoom": "缩放"}.get(self._drag_mode, "")
            if self._drag_mode == "rotate_z":
                mode += f"({AXIS_LABELS.get(self._nav.axis, 'Z')})"
            mode = f"{mode} | " if mode else ""
        nav = NavigationController.summary(self._nav.options)
        pivot = "支点:光标" if self._nav.options.pivot == "cursor" else "支点:画布中心"
        return (f"{mode}{stats['objects']} obj | {stats['triangles']} tri | "
                f"{state.last_render_seconds * 1000:.0f} ms | "
                f"{state.fps:.1f} FPS | {pivot} | {nav} | {state.units.label}")


# ==========================================================================
# 侧边面板
# ==========================================================================
class ObjectPanel(QtWidgets.QWidget):
    """对象列表（对象面板与对象右键菜单）。"""

    objectSelected = QtCore.pyqtSignal(int)
    objectToggleVisible = QtCore.pyqtSignal(int, bool)
    requestAction = QtCore.pyqtSignal(str, int)      # action, index
    measureSelected = QtCore.pyqtSignal(int)

    def __init__(self, state: ViewerState, parent=None):
        super().__init__(parent)
        self.state = state
        self._updating = False

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(4)

        self.title = QtWidgets.QLabel()
        self.title.setStyleSheet("font-weight: bold;")
        layout.addWidget(self.title)

        self.list = QtWidgets.QListWidget()
        self.list.setSelectionMode(QtWidgets.QAbstractItemView.SingleSelection)
        self.list.itemSelectionChanged.connect(self._on_selection)
        self.list.itemChanged.connect(self._on_item_changed)
        self.list.setContextMenuPolicy(QtCore.Qt.CustomContextMenu)
        self.list.customContextMenuRequested.connect(self._context_menu)
        layout.addWidget(self.list, 3)

        self.measure_title = QtWidgets.QLabel()
        self.measure_title.setStyleSheet("font-weight: bold;")
        layout.addWidget(self.measure_title)

        self.measure_list = QtWidgets.QListWidget()
        self.measure_list.setSelectionMode(
            QtWidgets.QAbstractItemView.SingleSelection)
        layout.addWidget(self.measure_list, 2)

        self.stats_label = QtWidgets.QLabel()
        self.stats_label.setWordWrap(True)
        self.stats_label.setStyleSheet("color: #888;")
        layout.addWidget(self.stats_label)

    # ---------------- 刷新 ----------------
    def refresh(self) -> None:
        tr = self.state.tr
        self.title.setText(tr.tr("objects"))
        self.measure_title.setText(tr.tr("measurement"))
        self._updating = True
        self.list.clear()
        for i, obj in enumerate(self.state.scene.objects):
            item = QtWidgets.QListWidgetItem(
                f"{obj.name}  [{obj.mesh.triangle_count}]")
            item.setFlags(item.flags() | QtCore.Qt.ItemIsUserCheckable)
            item.setCheckState(QtCore.Qt.Checked if obj.visible
                               else QtCore.Qt.Unchecked)
            if obj.transparent:
                item.setForeground(QtGui.QBrush(QtGui.QColor(150, 200, 255)))
            if obj.wire:
                item.setText(item.text() + "  (wire)")
            if obj.locked:
                item.setText(item.text() + "  [L]")
            self.list.addItem(item)
            if obj.active:
                self.list.setCurrentRow(i)
        self._updating = False

        self.measure_list.clear()
        for item in self.state.measures.items:
            text = f"{item.kind}: {item.value_text(self.state.units)}"
            self.measure_list.addItem(text)

        stats = self.state.statistics()
        self.stats_label.setText(
            f"{tr.tr('stat_objects')}: {stats['objects']} / {stats['visible']}\n"
            f"{tr.tr('stat_triangles')}: {stats['triangles']}\n"
            f"{tr.tr('stat_vertices')}: {stats['vertices']}")

    # ---------------- 事件 ----------------
    def _on_selection(self) -> None:
        if self._updating:
            return
        row = self.list.currentRow()
        if row >= 0:
            self.objectSelected.emit(row)

    def _on_item_changed(self, item) -> None:
        if self._updating:
            return
        row = self.list.row(item)
        self.objectToggleVisible.emit(row, item.checkState() == QtCore.Qt.Checked)

    def _context_menu(self, pos) -> None:
        row = self.list.currentRow()
        if row < 0:
            return
        tr = self.state.tr
        obj = self.state.scene.objects[row]
        menu = QtWidgets.QMenu(self)
        for key, action in (("visible", "visible"), ("wire", "wire"),
                            ("transparent", "transparent"),
                            ("always_on_top", "always_on_top"),
                            ("locked", "locked")):
            act = menu.addAction(tr.tr(key), lambda a=action: self.requestAction.emit(a, row))
            act.setCheckable(True)
            act.setChecked(bool(getattr(obj, action)))
        menu.addSeparator()
        menu.addAction(tr.tr("active"), lambda: self.requestAction.emit("active", row))
        menu.addSeparator()
        menu.addAction(tr.tr("object_properties"),
                       lambda: self.requestAction.emit("properties", row))
        menu.addAction(tr.tr("statistics"),
                       lambda: self.requestAction.emit("statistics", row))
        menu.addAction(tr.tr("copy_object"),
                       lambda: self.requestAction.emit("copy", row))
        menu.addAction(tr.tr("save_object") + " (STL)",
                       lambda: self.requestAction.emit("save_stl", row))
        menu.addSeparator()
        menu.addAction(tr.tr("repair_surface"),
                       lambda: self.requestAction.emit("repair", row))
        menu.addAction(tr.tr("generate_components"),
                       lambda: self.requestAction.emit("components", row))
        menu.addAction(tr.tr("remove_tri_intersected"),
                       lambda: self.requestAction.emit("remove_tri", row))
        menu.addSeparator()
        menu.addAction(tr.tr("delete_object"),
                       lambda: self.requestAction.emit("delete", row))
        menu.exec_(self.list.mapToGlobal(pos))


# ==========================================================================
# 对话框
# ==========================================================================
class LightDialog(QtWidgets.QDialog):
    """Light Settings：环境光/漫反射/高光、光源方向、颜色。"""

    def __init__(self, state: ViewerState, parent=None):
        super().__init__(parent)
        self.state = state
        tr = state.tr
        self.setWindowTitle(tr.tr("light_settings"))
        self.setMinimumWidth(420)
        self.light = state.light

        form = QtWidgets.QFormLayout(self)
        self.enabled = QtWidgets.QCheckBox(tr.tr("lights_enabled"))
        self.enabled.setChecked(self.light.enabled)
        form.addRow(self.enabled)

        self.ambient = self._slider(self.light.ambient)
        self.diffuse = self._slider(self.light.diffuse)
        self.specular = self._slider(self.light.specular)
        form.addRow(tr.tr("ambient"), self.ambient)
        form.addRow(tr.tr("diffuse"), self.diffuse)
        form.addRow(tr.tr("specular"), self.specular)

        self.shininess = QtWidgets.QSpinBox()
        self.shininess.setRange(1, 256)
        self.shininess.setValue(int(self.light.shininess))
        form.addRow(tr.tr("shininess"), self.shininess)

        self.primary = QtWidgets.QComboBox()
        self.secondary = QtWidgets.QComboBox()
        for name in LIGHT_PRESETS:
            self.primary.addItem(name, name)
            self.secondary.addItem(name, name)
        self.primary.setCurrentText(self.light.primary)
        self.secondary.setCurrentText(self.light.secondary)
        form.addRow(tr.tr("light_options") + " 1", self.primary)
        form.addRow(tr.tr("light_options") + " 2", self.secondary)

        self.light_color_btn = QtWidgets.QPushButton()
        self.shine_color_btn = QtWidgets.QPushButton()
        self._paint_button(self.light_color_btn, self.light.light_color)
        self._paint_button(self.shine_color_btn, self.light.shine_color)
        self.light_color_btn.clicked.connect(
            lambda: self._pick_color("light_color", self.light_color_btn))
        self.shine_color_btn.clicked.connect(
            lambda: self._pick_color("shine_color", self.shine_color_btn))
        form.addRow(tr.tr("light_color"), self.light_color_btn)
        form.addRow(tr.tr("shine_color"), self.shine_color_btn)

        self.two_sided = QtWidgets.QCheckBox(tr.tr("back_and_front"))
        self.two_sided.setChecked(self.light.two_sided)
        form.addRow(self.two_sided)

        buttons = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel
            | QtWidgets.QDialogButtonBox.Reset)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        buttons.button(QtWidgets.QDialogButtonBox.Reset).setText(tr.tr("reset_light_settings"))
        buttons.button(QtWidgets.QDialogButtonBox.Reset).clicked.connect(self._reset)
        form.addRow(buttons)

        for widget in (self.ambient, self.diffuse, self.specular):
            widget.valueChanged.connect(self._apply)
        self.shininess.valueChanged.connect(self._apply)
        self.primary.currentIndexChanged.connect(self._apply)
        self.secondary.currentIndexChanged.connect(self._apply)
        self.enabled.toggled.connect(self._apply)
        self.two_sided.toggled.connect(self._apply)

    def _slider(self, value: float) -> QtWidgets.QSlider:
        slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        slider.setRange(0, 100)
        slider.setValue(int(round(value * 100)))
        return slider

    def _paint_button(self, button, color) -> None:
        c = QtGui.QColor.fromRgbF(*color)
        button.setStyleSheet(f"background-color: {c.name()}; min-width: 80px;")
        button.setText(c.name())

    def _pick_color(self, attr: str, button) -> None:
        current = QtGui.QColor.fromRgbF(*getattr(self.light, attr))
        color = QtWidgets.QColorDialog.getColor(current, self)
        if color.isValid():
            setattr(self.light, attr, (color.redF(), color.greenF(), color.blueF()))
            self._paint_button(button, getattr(self.light, attr))
            self._apply()

    def _apply(self) -> None:
        self.light.enabled = self.enabled.isChecked()
        self.light.ambient = self.ambient.value() / 100.0
        self.light.diffuse = self.diffuse.value() / 100.0
        self.light.specular = self.specular.value() / 100.0
        self.light.shininess = float(self.shininess.value())
        self.light.set_primary(self.primary.currentData())
        self.light.set_secondary(self.secondary.currentData())
        self.light.two_sided = self.two_sided.isChecked()
        parent = self.parent()
        for name in ("_mark_dirty", "mark_dirty"):
            hook = getattr(parent, name, None)
            if callable(hook):
                hook()
                break

    def _reset(self) -> None:
        self.light.reset()
        self.ambient.setValue(int(self.light.ambient * 100))
        self.diffuse.setValue(int(self.light.diffuse * 100))
        self.specular.setValue(int(self.light.specular * 100))
        self.shininess.setValue(int(self.light.shininess))
        self.enabled.setChecked(self.light.enabled)
        self.primary.setCurrentText(self.light.primary)
        self.secondary.setCurrentText(self.light.secondary)
        self._apply()


class RotationDialog(QtWidgets.QDialog):
    """Rotation Angles：直接输入偏航/俯仰/滚转。"""

    def __init__(self, camera: Camera, tr: Translator, parent=None):
        super().__init__(parent)
        self.setWindowTitle(tr.tr("rotation_angles_title"))
        self.camera = camera
        form = QtWidgets.QFormLayout(self)
        self.yaw = QtWidgets.QDoubleSpinBox()
        self.pitch = QtWidgets.QDoubleSpinBox()
        self.roll = QtWidgets.QDoubleSpinBox()
        for box, value, lo, hi in ((self.yaw, camera.yaw, -360, 360),
                                   (self.pitch, camera.pitch, -89.9, 89.9),
                                   (self.roll, camera.roll, -180, 180)):
            box.setRange(lo, hi)
            box.setDecimals(2)
            box.setValue(value)
            box.setSingleStep(5.0)
        form.addRow(tr.tr("yaw"), self.yaw)
        form.addRow(tr.tr("pitch"), self.pitch)
        form.addRow(tr.tr("roll"), self.roll)
        buttons = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        form.addRow(buttons)

    def values(self) -> tuple:
        return self.yaw.value(), self.pitch.value(), self.roll.value()


class StatisticsDialog(QtWidgets.QDialog):
    """Statistics / Show STL File Size。"""

    def __init__(self, text: str, title: str, parent=None):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.resize(520, 460)
        layout = QtWidgets.QVBoxLayout(self)
        view = QtWidgets.QPlainTextEdit(text)
        view.setReadOnly(True)
        view.setFont(QtGui.QFont("Consolas", 10))
        layout.addWidget(view)
        buttons = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Close)
        buttons.rejected.connect(self.reject)
        buttons.accepted.connect(self.accept)
        layout.addWidget(buttons)


class PointSizeDialog(QtWidgets.QDialog):
    """Mesh > Point Size / Line Width。"""

    def __init__(self, state: ViewerState, parent=None):
        super().__init__(parent)
        tr = state.tr
        self.setWindowTitle(f"{tr.tr('point_size')} / {tr.tr('line_width')}")
        form = QtWidgets.QFormLayout(self)
        self.point = QtWidgets.QSpinBox()
        self.point.setRange(1, 20)
        self.point.setValue(int(state.options.point_size))
        self.line = QtWidgets.QSpinBox()
        self.line.setRange(1, 20)
        self.line.setValue(int(state.options.line_width))
        form.addRow(tr.tr("point_size"), self.point)
        form.addRow(tr.tr("line_width"), self.line)
        buttons = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        form.addRow(buttons)


class ObjectPropertiesDialog(QtWidgets.QDialog):
    """Object Editor / 对象属性。"""

    def __init__(self, obj, tr: Translator, parent=None):
        super().__init__(parent)
        self.setWindowTitle(tr.tr("object_properties"))
        self.obj = obj
        form = QtWidgets.QFormLayout(self)
        self.name = QtWidgets.QLineEdit(obj.name)
        form.addRow(tr.tr("object"), self.name)

        self.color_btn = QtWidgets.QPushButton()
        self._paint(obj.color)
        self.color_btn.clicked.connect(self._pick_color)
        form.addRow(tr.tr("select_mesh_color"), self.color_btn)

        self.alpha = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.alpha.setRange(0, 100)
        self.alpha.setValue(int(obj.alpha * 100))
        form.addRow(tr.tr("transparency"), self.alpha)

        self.mode = QtWidgets.QComboBox()
        for key in COLOR_MODE_CHOICES:
            label = tr.tr("orientation_colors" if key == "orientation" else key)
            self.mode.addItem(label, key)
        idx = self.mode.findData(obj.color_mode)
        if idx >= 0:
            self.mode.setCurrentIndex(idx)
        form.addRow(tr.tr("colored"), self.mode)

        self.checks = {}
        for key in ("visible", "wire", "transparent", "always_on_top", "locked"):
            box = QtWidgets.QCheckBox(tr.tr(key))
            box.setChecked(bool(getattr(obj, key)))
            self.checks[key] = box
            form.addRow(box)

        buttons = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        form.addRow(buttons)

    def _paint(self, color) -> None:
        self._color = tuple(color)
        c = QtGui.QColor.fromRgbF(*self._color)
        self.color_btn.setStyleSheet(f"background-color: {c.name()};")
        self.color_btn.setText(c.name())

    def _pick_color(self) -> None:
        c = QtWidgets.QColorDialog.getColor(
            QtGui.QColor.fromRgbF(*self._color), self)
        if c.isValid():
            self._paint((c.redF(), c.greenF(), c.blueF()))

    def apply(self) -> None:
        obj = self.obj
        obj.name = self.name.text() or obj.name
        obj.color = self._color
        obj.alpha = self.alpha.value() / 100.0
        obj.color_mode = self.mode.currentData()
        obj.auto_color = obj.color_mode != "uniform"
        for key, box in self.checks.items():
            setattr(obj, key, box.isChecked())
        obj.invalidate_colors()


class ColorDialog(QtWidgets.QDialog):
    """Background Colors / Select Mesh Color / Measure Color。"""

    def __init__(self, title: str, color, parent=None, with_gradient: bool = False,
                 top_color=None):
        super().__init__(parent)
        self.setWindowTitle(title)
        form = QtWidgets.QFormLayout(self)
        self.color = tuple(color)
        self.top = tuple(top_color or color)
        self.btn = QtWidgets.QPushButton()
        self._paint(self.btn, self.color)
        self.btn.clicked.connect(lambda: self._pick("color", self.btn))
        form.addRow(title, self.btn)
        self.gradient = None
        self.btn_top = None
        if with_gradient:
            self.gradient = QtWidgets.QCheckBox("Gradient Background")
            form.addRow(self.gradient)
            self.btn_top = QtWidgets.QPushButton()
            self._paint(self.btn_top, self.top)
            self.btn_top.clicked.connect(lambda: self._pick("top", self.btn_top))
            form.addRow("Top", self.btn_top)
        buttons = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        form.addRow(buttons)

    def _paint(self, button, color) -> None:
        c = QtGui.QColor.fromRgbF(*color)
        button.setStyleSheet(f"background-color: {c.name()};")
        button.setText(c.name())

    def _pick(self, which: str, button) -> None:
        current = QtGui.QColor.fromRgbF(*(self.color if which == "color" else self.top))
        c = QtWidgets.QColorDialog.getColor(current, self)
        if not c.isValid():
            return
        value = (c.redF(), c.greenF(), c.blueF())
        if which == "color":
            self.color = value
        else:
            self.top = value
        self._paint(button, value)


# ==========================================================================
# 主窗口
# ==========================================================================
class MainWindow(QtWidgets.QMainWindow):
    """STL Viewer 主窗口。"""

    def __init__(self, state: ViewerState | None = None):
        super().__init__()
        self.state = state or ViewerState()
        self.setWindowTitle(self.state.tr.tr("app_title"))
        self.resize(1280, 860)
        self.setAcceptDrops(True)

        # 后台任务池：菜单 / 快捷键触发的重活（载入、导出、统计、截图编码）
        # 全部丢到这里，GUI 线程只做「发任务」和「收结果 + 刷新控件」。
        self._work_pool = QtCore.QThreadPool(self)
        self._work_pool.setMaxThreadCount(2)
        self._work_bridge = _WorkBridge(self)
        self._work_bridge.finished.connect(self._on_work_finished,
                                           QtCore.Qt.QueuedConnection)
        self._work_jobs: dict = {}          # tag -> (callback, actions)
        self._work_tasks: dict = {}         # tag -> QRunnable（防被 GC）
        self._work_busy_tag: str | None = None

        self.canvas = Canvas(self.state, self)
        self.setCentralWidget(self.canvas)
        self.canvas.statusMessage.connect(self._show_status)
        self.canvas.sceneChanged.connect(self._on_scene_changed)

        self.panel = ObjectPanel(self.state, self)
        dock = QtWidgets.QDockWidget(self.state.tr.tr("objects"), self)
        dock.setWidget(self.panel)
        dock.setAllowedAreas(QtCore.Qt.LeftDockWidgetArea |
                             QtCore.Qt.RightDockWidgetArea)
        self.addDockWidget(QtCore.Qt.LeftDockWidgetArea, dock)
        self.object_dock = dock
        self.panel.objectSelected.connect(self._select_object)
        self.panel.objectToggleVisible.connect(self._toggle_visible)
        self.panel.requestAction.connect(self._object_action)

        # 内嵌 MCP 服务：菜单 / 工具条 / 状态栏都要反映它的实时运行状态
        from .mcp_runtime import MCPController
        self.mcp = MCPController(self.state, self)
        self.mcp.statusChanged.connect(self._on_mcp_status)
        self.mcp.sessionChanged.connect(self._on_mcp_session_changed)

        self._build_toolbar()
        self._build_menus()
        self._build_statusbar()
        self._create_timers()
        self.retranslate()
        self.panel.refresh()

        # 启动保持空场景：不自动生成任何占位 objects，Objects 面板必须为空。
        # 模型只通过 File > Open / 拖放 / 命令行参数进入场景；若设置里恢复
        # 出了对象，则把它们居中到画布中心。
        self._fit_pending = False
        if self.state.scene.objects:
            self.state.fit_view()
            self.schedule_fit()
        # 等事件循环起来再启动服务，避免窗口还没显示就被后台线程回调
        if self.state.settings.mcp.enabled and self.state.settings.mcp.autostart:
            QtCore.QTimer.singleShot(0, self._autostart_mcp)

    def schedule_fit(self) -> None:
        """窗口尚未完成布局时先标记，等尺寸稳定后再按真实画布重新 fit。"""
        self._fit_pending = True
        QtCore.QTimer.singleShot(150, self._deferred_fit)

    def _deferred_fit(self) -> None:
        if not self._fit_pending or not self.state.scene.objects:
            self._fit_pending = False
            return
        self._fit_pending = False
        self.state.viewport = (self.canvas.width(), self.canvas.height())
        self.state.fit_view()
        self.canvas.mark_dirty()

    # ---------------- 工具条 ----------------
    def _build_toolbar(self) -> None:
        """两行工具条：第一行全是视图操作（含标准视图），第二行是文件/显示/测量。

        原实现把 Standard Views 折叠在 View 菜单的子菜单里，切视角要点两次；
        放大/缩小、旋转角度、全屏、视图选项、光源预设、着色模式同样埋在菜单
        深处。这些都提到工具条上，「举一反三」把常用项一次性摊开。
        """
        view = QtWidgets.QToolBar("view")
        view.setObjectName("view_toolbar")
        view.setMovable(False)
        view.setToolButtonStyle(QtCore.Qt.ToolButtonTextOnly)
        self.addToolBar(view)
        self.view_toolbar = view
        self.addToolBarBreak()

        bar = QtWidgets.QToolBar("main")
        bar.setObjectName("main_toolbar")
        bar.setMovable(False)
        self.addToolBar(bar)
        self.toolbar = bar
        self._toolbar_actions = []

    # ---------------- 菜单 ----------------
    def _act(self, parent, key: str, slot, shortcut=None, checkable=False,
             checked=False, extra_key: str | None = None) -> QtWidgets.QAction:
        text = self.state.tr.tr(extra_key or key)
        action = QtWidgets.QAction(text, self)
        if shortcut:
            action.setShortcut(shortcut)
        action.setCheckable(checkable)
        action.setChecked(checked)
        if slot is not None:
            action.triggered.connect(slot)
        parent.addAction(action)
        self._toolbar_actions.append((action, extra_key or key))
        return action

    def _build_menus(self) -> None:
        state = self.state
        mbar = self.menuBar()
        self.menus = {}
        self.acts = {}

        # -------- File --------
        m = mbar.addMenu("")
        self.menus["file"] = m
        new_menu = m.addMenu("")
        self.menus["new"] = new_menu
        self.acts["new_box"] = self._act(new_menu, "new_box",
                                         lambda: self._new_primitive("box"))
        self.acts["new_cylinder"] = self._act(new_menu, "new_cylinder",
                                              lambda: self._new_primitive("cylinder"))
        self.acts["new_step_cylinder"] = self._act(
            new_menu, "new_step_cylinder", lambda: self._new_primitive("step_cylinder"))
        self.acts["new_tube"] = self._act(new_menu, "new_tube",
                                          lambda: self._new_primitive("tube"))
        self.acts["new_sphere"] = self._act(new_menu, "new_sphere",
                                            lambda: self._new_primitive("sphere"))
        m.addSeparator()
        self.acts["open"] = self._act(m, "open", self.open_dialog, "Ctrl+O")
        self.acts["recent"] = m.addMenu("")
        self.menus["recent"] = self.acts["recent"]
        self.acts["save_object"] = self._act(m, "save_object", self.save_active_stl,
                                             "Ctrl+S")
        self.acts["export_stl"] = self._act(m, "export_stl", self.save_active_stl)
        m.addSeparator()
        self.acts["save_picture"] = self._act(m, "save_picture",
                                              self.save_picture, "Ctrl+Shift+S")
        self.acts["save_multi_res_picture"] = self._act(
            m, "save_multi_res_picture", self.save_multi_res_picture)
        self.acts["save_picture_transp"] = self._act(
            m, "save_picture_transp",
            lambda: self.save_picture(transparent=True))
        self.acts["copy_picture_clipboard"] = self._act(
            m, "copy_picture_clipboard", self.copy_picture_to_clipboard, "Ctrl+C")
        m.addSeparator()
        self.acts["copy_object"] = self._act(m, "copy_object", self.copy_object,
                                             "Ctrl+Shift+C")
        self.acts["paste_object"] = self._act(m, "paste_object", self.paste_object,
                                              "Ctrl+V")
        self.acts["delete_object"] = self._act(m, "delete_object", self.delete_object,
                                               "Delete")
        self.acts["delete_all_pictures"] = self._act(m, "delete_all_pictures",
                                                     self.delete_all_objects)
        m.addSeparator()
        units_menu = m.addMenu("")
        self.menus["units"] = units_menu
        self._units_group = QtWidgets.QActionGroup(self)
        for key, unit in (("unit_mm", "mm"), ("unit_inch", "inch")):
            act = QtWidgets.QAction(self.state.tr.tr(key), self)
            act.setCheckable(True)
            act.setChecked(self.state.settings.units == unit)
            act.triggered.connect(lambda checked, u=unit: self.set_units(u))
            self._units_group.addAction(act)
            units_menu.addAction(act)
            self.acts["unit_" + unit] = act
        self.acts["show_stl_filesize"] = self._act(
            m, "show_stl_filesize", self.show_file_size)
        self.acts["reload_object_settings"] = self._act(
            m, "reload_object_settings", self.reload_object_settings)
        self.acts["save_object_settings"] = self._act(
            m, "save_object_settings", self.save_object_settings)
        m.addSeparator()
        self.acts["close_window"] = self._act(m, "close_window", self.close, "Ctrl+Q")

        # -------- View --------
        m = mbar.addMenu("")
        self.menus["view"] = m
        self.acts["zoom_in"] = self._act(m, "zoom_in",
                                         lambda: self._zoom(0.8), "Ctrl++")
        self.acts["zoom_out"] = self._act(m, "zoom_out",
                                          lambda: self._zoom(1.25), "Ctrl+-")
        self.acts["zoom_fit"] = self._act(m, "zoom_fit", self._fit, "F")
        self.acts["rotate_around_z"] = self._act(
            m, "rotate_around_z", self._toggle_rotate_z, "Ctrl+Z", checkable=True)
        axis_menu = m.addMenu("")
        self.menus["rotate_axis"] = axis_menu
        self._rotate_axis_group = QtWidgets.QActionGroup(self)
        self._rotate_axis_group.setExclusive(True)
        for key, ax, shortcut in ROTATE_AXIS_MENU:
            act = self._act(
                axis_menu, key,
                lambda checked=False, a=ax: self._set_rotation_axis(a),
                shortcut, checkable=True, checked=(ax == "free"))
            self._rotate_axis_group.addAction(act)
            self.acts[key] = act
        self.acts["rotation_angles"] = self._act(m, "rotation_angles",
                                                 self.show_rotation_dialog)
        m.addSeparator()
        views_menu = m.addMenu("")
        self.menus["standard_views"] = views_menu
        for name in ("iso", "top", "bottom", "front", "back", "left", "right",
                     "iso_back"):
            self._act(views_menu, "view_" + name,
                      lambda checked=False, n=name: self._standard_view(n))
        m.addSeparator()
        self.acts["orthogonal"] = self._act(m, "orthogonal", self._toggle_ortho,
                                            "Ctrl+P", checkable=True,
                                            checked=self.state.camera.orthographic)
        self.acts["fixed_on_screen"] = self._act(m, "fixed_on_screen",
                                                 self._toggle_fixed, checkable=True)
        self.acts["full_screen"] = self._act(m, "full_screen", self._toggle_fullscreen,
                                             "F11", checkable=True)
        self.acts["crosshair"] = self._act(m, "crosshair", self._toggle_crosshair,
                                           checkable=True)
        self.acts["reset_center"] = self._act(m, "reset_center", self._reset_center,
                                              "R")
        self.acts["change_center"] = self._act(m, "change_center",
                                               self._change_center)
        self.acts["view_options"] = self._act(m, "view_options", self._view_options)

        # -------- Mesh --------
        m = mbar.addMenu("")
        self.menus["mesh"] = m
        self.acts["wire"] = self._act(m, "wire", self._set_wire, "W", checkable=True)
        self.acts["solid"] = self._act(m, "solid", self._set_solid, checkable=True,
                                       checked=True)
        self.acts["colored"] = self._act(m, "colored", self._set_colored, "C",
                                         checkable=True)
        self.acts["auto_colors"] = self._act(m, "auto_colors",
                                             lambda: self._set_color_mode("auto"))
        self.acts["uni_color"] = self._act(m, "uni_color",
                                           lambda: self._set_color_mode("uniform"))
        self.acts["component_colors"] = self._act(
            m, "component_colors", lambda: self._set_color_mode("component"))
        self.acts["curvature_colors"] = self._act(
            m, "curvature_colors", lambda: self._set_color_mode("curvature"))
        self.acts["orientation_colors"] = self._act(
            m, "orientation_colors", lambda: self._set_color_mode("orientation"))
        self.acts["transparent"] = self._act(m, "transparent", self._toggle_transparent,
                                             checkable=True)
        self.acts["back_and_front"] = self._act(
            m, "back_and_front", self._toggle_two_sided, checkable=True,
            checked=bool(state.settings.view.back_and_front))
        self.acts["show_edges"] = self._act(m, "show_edges", self._toggle_edges,
                                            checkable=True)
        m.addSeparator()
        self.acts["select_mesh_color"] = self._act(m, "select_mesh_color",
                                                   self._select_object_color)
        self.acts["point_size"] = self._act(m, "point_size", self._point_size_dialog)
        self.acts["object_editor"] = self._act(m, "object_editor",
                                               self._object_properties)

        # -------- Light --------
        m = mbar.addMenu("")
        self.menus["light"] = m
        self.acts["light_settings"] = self._act(m, "light_settings",
                                                self.show_light_dialog)
        self.acts["light_down"] = self._act(
            m, "light_down", lambda: self._set_light("down"))
        self.acts["light_down_left"] = self._act(
            m, "light_down_left", lambda: self._set_light("down_left"))
        self.acts["light_left"] = self._act(
            m, "light_left", lambda: self._set_light("left"))
        self.acts["light_up_left"] = self._act(
            m, "light_up_left", lambda: self._set_light("up_left"))
        self.acts["reset_light_settings"] = self._act(
            m, "reset_light_settings", self._reset_light)
        m.addSeparator()
        self.acts["background_colors"] = self._act(m, "background_colors",
                                                   self._background_colors)

        # -------- Measurement --------
        m = mbar.addMenu("")
        self.menus["measurement"] = m
        for kind in ("line2d", "line3d", "point3d", "circle2p", "circle3p",
                     "angle2d", "rzpoint"):
            key = MEASURE_DEFS[kind][1]
            self.acts["measure_" + kind] = self._act(
                m, key, lambda checked=False, k=kind: self.canvas.start_measure(k))
        m.addSeparator()
        self.acts["measure_true3d"] = self._act(
            m, "measure_true3d", self._toggle_measure_3d, checkable=True,
            checked=self.state.settings.measure_mode == "3d")
        m.addSeparator()
        self.acts["delete_first_measure"] = self._act(m, "delete_first_measure",
                                                      self._delete_first_measure)
        self.acts["delete_last_measure"] = self._act(m, "delete_last_measure",
                                                     self._delete_last_measure)
        self.acts["delete_all_measures"] = self._act(m, "delete_all_measures",
                                                     self._delete_all_measures)
        m.addSeparator()
        self.acts["measure_color"] = self._act(m, "measure_color",
                                                self._measure_color)
        units_menu2 = m.addMenu("")
        self.menus["measure_units"] = units_menu2
        for key, unit in (("unit_mm", "mm"), ("unit_inch", "inch")):
            act = QtWidgets.QAction(self.state.tr.tr(key), self)
            act.setCheckable(True)
            act.setChecked(self.state.settings.units == unit)
            act.triggered.connect(lambda checked=False, u=unit: self.set_units(u))
            units_menu2.addAction(act)
            self.acts["measure_unit_" + unit] = act

        # -------- Extras --------
        m = mbar.addMenu("")
        self.menus["extras"] = m
        lang_menu = m.addMenu("")
        self.menus["language"] = lang_menu
        self._lang_group = QtWidgets.QActionGroup(self)
        for code, name, native in Translator.language_names():
            act = QtWidgets.QAction(f"{name} ({native})", self)
            act.setCheckable(True)
            act.setChecked(code == self.state.tr.language)
            act.triggered.connect(lambda checked, c=code: self.set_language(c))
            self._lang_group.addAction(act)
            lang_menu.addAction(act)

        render_menu = m.addMenu("")
        self.menus["render_system"] = render_menu
        self.acts["enable_hw_accel"] = self._act(
            m, "enable_hw_accel", self._toggle_hw_accel, checkable=True,
            checked=self.state.settings.hardware_acceleration)
        for value in ("OpenGL", "Direct3D", "Software"):
            key = "render_" + value.lower()
            act = QtWidgets.QAction(value, self)
            act.setCheckable(True)
            act.setChecked(self.state.settings.render_system == value)
            act.triggered.connect(lambda checked=False, v=value:
                                  self._set_render_system(v))
            render_menu.addAction(act)
            self.acts[key] = act
        m.addSeparator()
        spin_menu = m.addMenu("")
        self.menus["spin"] = spin_menu
        self.acts["spin_enable"] = self._act(
            spin_menu, "spin_enable", self._toggle_spin, checkable=True)
        type_menu = spin_menu.addMenu("")
        self.menus["spin_type"] = type_menu
        self._spin_type_group = QtWidgets.QActionGroup(self)
        for key, kind in (("spin_horizontal", "horizontal"),
                          ("spin_vertical", "vertical"), ("spin_rx", "rx"),
                          ("spin_ry", "ry"), ("spin_rz", "rz"),
                          ("spin_roll", "roll"), ("spin_random", "random")):
            act = QtWidgets.QAction(self.state.tr.tr(key), self)
            act.setCheckable(True)
            act.setChecked(self.state.spin.config.kind == kind)
            act.triggered.connect(lambda checked, k=kind: self._set_spin_type(k))
            self._spin_type_group.addAction(act)
            type_menu.addAction(act)
            self.acts[key] = act
            self._toolbar_actions.append((act, key))
        dir_menu = spin_menu.addMenu("")
        self.menus["spin_direction"] = dir_menu
        self._spin_dir_group = QtWidgets.QActionGroup(self)
        for key, value in (("spin_positive", 1), ("spin_negative", -1)):
            act = QtWidgets.QAction(self.state.tr.tr(key), self)
            act.setCheckable(True)
            act.setChecked(self.state.spin.config.direction == value)
            act.triggered.connect(lambda checked, v=value: self._set_spin_direction(v))
            self._spin_dir_group.addAction(act)
            dir_menu.addAction(act)
        speed_menu = spin_menu.addMenu("")
        self.menus["spin_speed"] = speed_menu
        self._spin_speed_group = QtWidgets.QActionGroup(self)
        for key, value in (("speed_low", "low"), ("speed_medium", "medium"),
                           ("speed_high", "high")):
            act = QtWidgets.QAction(self.state.tr.tr(key), self)
            act.setCheckable(True)
            act.setChecked(self.state.spin.config.speed == value)
            act.triggered.connect(lambda checked, v=value: self._set_spin_speed(v))
            self._spin_speed_group.addAction(act)
            speed_menu.addAction(act)

        macro_menu = m.addMenu("")
        self.menus["macro"] = macro_menu
        self.acts["macro_start_recording"] = self._act(macro_menu, "macro_start_recording",
                                                       self._toggle_macro_record)
        self.acts["macro_execute"] = self._act(macro_menu, "macro_execute",
                                               self._play_macro)
        self.acts["macro_open"] = self._act(macro_menu, "macro_open", self._open_macro)
        self.acts["macro_save"] = self._act(macro_menu, "macro_save", self._save_macro)

        stl_menu = m.addMenu("")
        self.menus["stl_tools"] = stl_menu
        self.acts["repair_surface"] = self._act(stl_menu, "repair_surface",
                                                lambda: self._object_action("repair", -1))
        self.acts["generate_components"] = self._act(
            stl_menu, "generate_components",
            lambda: self._object_action("components", -1))
        self.acts["remove_tri_intersected"] = self._act(
            stl_menu, "remove_tri_intersected",
            lambda: self._object_action("remove_tri", -1))

        m.addSeparator()
        self.acts["mcp_settings"] = self._act(
            m, "mcp_settings", self.show_mcp_settings)
        self.acts["mcp_toggle"] = self._act(
            m, "mcp_toggle", self.toggle_mcp, checkable=True,
            checked=bool(state.settings.mcp.enabled))
        m.addSeparator()
        self.acts["statistics"] = self._act(m, "statistics", self.show_statistics)
        self.acts["object_editor2"] = self._act(m, "object_editor",
                                                self._object_properties)
        self.acts["load_default_settings"] = self._act(
            m, "load_default_settings", self.load_default_settings)
        self.acts["save_default_settings"] = self._act(
            m, "save_default_settings", self.save_default_settings)
        self.acts["reset_selected_defaults"] = self._act(
            m, "reset_selected_defaults", self.reset_default_settings)

        # -------- Help --------
        m = mbar.addMenu("")
        self.menus["help"] = m
        self.acts["about"] = self._act(m, "about", self.show_about)

        # 工具条（菜单动作创建完之后再排布）
        self._populate_toolbars()

    def _populate_toolbars(self) -> None:
        """把常用菜单项摊到工具条上，标准视图直接一排按钮可点。"""
        tr = self.state.tr
        view = self.view_toolbar
        bar = self.toolbar

        # ---- 第一行：缩放 / 复位 / 标准视图 / 投影 / 视图选项 ----
        for key in ("zoom_in", "zoom_out", "zoom_fit", "reset_center",
                    "rotation_angles"):
            if self.acts.get(key) is not None:
                view.addAction(self.acts[key])
        view.addSeparator()
        names = ("iso", "front", "back", "left", "right", "top", "bottom",
                 "iso_back")
        for i, name in enumerate(names):
            key = "view_" + name
            act = QtWidgets.QAction(tr.tr(key), self)
            act.setToolTip(f"{tr.tr(key)}  ({i + 1})")
            act.triggered.connect(
                lambda checked=False, n=name: self._standard_view(n))
            view.addAction(act)
            self.acts[key] = act
            self._toolbar_actions.append((act, key))
        view.addSeparator()
        fit_act = QtWidgets.QAction(tr.tr("auto_fit_views"), self)
        fit_act.setCheckable(True)
        fit_act.setChecked(bool(self.state.settings.auto_fit_views))
        fit_act.setToolTip(tr.tr("auto_fit_views"))
        fit_act.triggered.connect(self._toggle_auto_fit)
        view.addAction(fit_act)
        self.acts["auto_fit_views"] = fit_act
        self._toolbar_actions.append((fit_act, "auto_fit_views"))
        view.addSeparator()
        # 旋转轴下拉：X / Y / Z / 视线 / 自由，五个入口一次摊开
        axis_btn = QtWidgets.QToolButton(view)
        axis_btn.setText(tr.tr("rotate_axis"))
        axis_btn.setPopupMode(QtWidgets.QToolButton.InstantPopup)
        axis_menu = QtWidgets.QMenu(axis_btn)
        for key, _, _ in ROTATE_AXIS_MENU:
            act = self.acts.get(key)
            if act is not None:
                axis_menu.addAction(act)
        axis_btn.setMenu(axis_menu)
        view.addWidget(axis_btn)
        self._rotate_axis_button = (axis_btn, "rotate_axis")
        view.addSeparator()
        for key in ("orthogonal", "full_screen", "view_options"):
            if self.acts.get(key) is not None:
                view.addAction(self.acts[key])

        # ---- 第二行：文件 / 显示 / 测量 / 动画 / 截图 ----
        for key in ("open", "save_picture", "wire", "solid", "colored",
                    "transparent", "show_edges", "crosshair"):
            if self.acts.get(key) is not None:
                bar.addAction(self.acts[key])
        # 着色模式下拉（原本在 Mesh 菜单里）
        color_btn = QtWidgets.QToolButton(bar)
        color_btn.setText(tr.tr("color_modes"))
        color_btn.setPopupMode(QtWidgets.QToolButton.InstantPopup)
        color_menu = QtWidgets.QMenu(color_btn)
        for key in ("auto_colors", "uni_color", "component_colors",
                    "curvature_colors", "orientation_colors"):
            if self.acts.get(key) is not None:
                color_menu.addAction(self.acts[key])
        color_btn.setMenu(color_menu)
        bar.addWidget(color_btn)
        self._color_mode_button = (color_btn, "color_modes")
        # 光源预设下拉（原本在 Light 菜单里）
        light_btn = QtWidgets.QToolButton(bar)
        light_btn.setText(tr.tr("light_presets"))
        light_btn.setPopupMode(QtWidgets.QToolButton.InstantPopup)
        light_menu = QtWidgets.QMenu(light_btn)
        for key in ("light_down", "light_down_left", "light_left",
                    "light_up_left", "reset_light_settings",
                    "light_settings"):
            if self.acts.get(key) is not None:
                light_menu.addAction(self.acts[key])
        light_btn.setMenu(light_menu)
        bar.addWidget(light_btn)
        self._light_preset_button = (light_btn, "light_presets")
        bar.addSeparator()
        for kind in ("line3d", "point3d", "circle3p", "angle2d", "rzpoint"):
            act = self.acts.get("measure_" + kind)
            if act:
                bar.addAction(act)
        bar.addSeparator()
        for key in ("spin_enable", "statistics"):
            act = self.acts.get(key)
            if act:
                bar.addAction(act)
        bar.addSeparator()
        # MCP 状态按钮：文字由 _update_mcp_ui() 动态维护，不参与 retranslate
        self.mcp_button = QtWidgets.QToolButton(bar)
        self.mcp_button.setToolButtonStyle(QtCore.Qt.ToolButtonTextOnly)
        self.mcp_button.clicked.connect(self.show_mcp_settings)
        bar.addWidget(self.mcp_button)

    def _toggle_auto_fit(self, checked: bool) -> None:
        """切换标准视图时是否自动 Zoom Fit（工具条上的开关）。"""
        self.state.settings.auto_fit_views = bool(checked)
        self._show_status("auto fit: " + ("on" if checked else "off"))

    def _build_statusbar(self) -> None:
        bar = self.statusBar()
        self.status_label = QtWidgets.QLabel("")
        self.units_label = QtWidgets.QLabel("")
        self.render_label = QtWidgets.QLabel("")
        for widget in (self.status_label, self.units_label, self.render_label):
            widget.setMinimumWidth(140)
        self.mcp_status = QtWidgets.QToolButton()
        self.mcp_status.setAutoRaise(True)
        self.mcp_status.clicked.connect(self.show_mcp_settings)
        bar.addPermanentWidget(self.mcp_status)
        bar.addPermanentWidget(self.units_label)
        bar.addPermanentWidget(self.status_label)
        bar.addPermanentWidget(self.render_label)
        bar.showMessage(self.state.tr.tr("ready"))

    def _create_timers(self) -> None:
        self.timer = QtCore.QTimer(self)
        self.timer.setInterval(16)
        self.timer.timeout.connect(self._on_tick)
        self.timer.start()

    # ---------------- 翻译 ----------------
    def retranslate(self) -> None:
        tr = self.state.tr
        self.setWindowTitle(tr.tr("app_title"))
        for key, menu in self.menus.items():
            if key == "new":
                menu.setTitle(tr.tr("new_object"))
            elif key == "recent":
                menu.setTitle(tr.tr("recent_files"))
            elif key == "units":
                menu.setTitle(tr.tr("file_units"))
            elif key == "standard_views":
                menu.setTitle(tr.tr("standard_views"))
            elif key == "measure_units":
                menu.setTitle(tr.tr("measurement_units"))
            elif key == "language":
                menu.setTitle(tr.tr("language"))
            elif key == "render_system":
                menu.setTitle(tr.tr("render_system"))
            elif key == "spin":
                menu.setTitle(tr.tr("spin"))
            elif key == "spin_type":
                menu.setTitle(tr.tr("spin_type"))
            elif key == "spin_direction":
                menu.setTitle(tr.tr("spin_direction"))
            elif key == "spin_speed":
                menu.setTitle(tr.tr("spin_speed"))
            elif key == "macro":
                menu.setTitle(tr.tr("macro"))
            elif key == "stl_tools":
                menu.setTitle(tr.tr("stl_tools"))
            else:
                menu.setTitle(tr.tr(key))
        for action, key in self._toolbar_actions:
            if action is None:
                continue
            action.setText(tr.tr(key))
            action.setToolTip(tr.tr(key))
        for pair in (getattr(self, "_color_mode_button", None),
                     getattr(self, "_light_preset_button", None)):
            if pair is not None:
                pair[0].setText(tr.tr(pair[1]))
        mode_act = self.acts.get("measure_true3d")
        if mode_act is not None:
            mode_act.setChecked(self.state.settings.measure_mode == "3d")
        # 旋转轴按钮的文字带当前轴（翻译后重新生成），三处勾选一并同步
        self._sync_rotation_ui()
        self.object_dock.setWindowTitle(tr.tr("objects"))
        self.panel.refresh()
        self._update_statusbar()
        self._update_mcp_ui()

    def _update_statusbar(self) -> None:
        state = self.state
        stats = state.statistics()
        self.status_label.setText(f"{stats['objects']} obj / {stats['triangles']} tri")
        self.units_label.setText(f"{state.tr.tr('units')}: {state.units.label}")
        self.render_label.setText(
            f"{state.last_render_seconds * 1000:.0f} ms | {state.fps:.1f} FPS")

    def _show_status(self, text: str) -> None:
        self.statusBar().showMessage(text, 4000)
        self._update_statusbar()

    # ---------------- 内嵌 MCP 服务 ----------------
    def _update_mcp_ui(self) -> None:
        tr = self.state.tr
        running = self.mcp.running
        text = tr.tr("mcp_on") if running else tr.tr("mcp_off")
        self.mcp_button.setText(text)
        self.mcp_button.setToolTip(tr.tr("mcp_settings"))
        self.mcp_status.setText(("● " if running else "○ ") + text)
        self.mcp_status.setToolTip(tr.tr("mcp_settings"))
        toggle = self.acts.get("mcp_toggle")
        if toggle is not None:
            toggle.setChecked(bool(self.state.settings.mcp.enabled))

    def _on_mcp_status(self, running: bool, endpoint: str) -> None:
        self._update_mcp_ui()
        if running:
            self._show_status(self.state.tr.tr("mcp_started", endpoint))

    def _on_mcp_session_changed(self, name: str) -> None:
        """AI 通过 MCP 改动了当前会话：同步画布、对象面板与状态栏。"""
        if name in ("load_model", "remove_object", "clear_scene"):
            self.state.fit_view()
        self.canvas.mark_dirty()
        self.panel.refresh()
        self._on_scene_changed()

    def show_mcp_settings(self) -> None:
        """打开 MCP 服务设置（服务 / 安全 / 工作区 / AI 客户端）。"""
        from .mcp_dialog import MCPSettingsDialog

        dialog = MCPSettingsDialog(self, self.mcp)
        dialog.exec_()
        self._update_mcp_ui()

    def toggle_mcp(self, checked: bool) -> None:
        """菜单开关：立即启停服务并保存设置。"""
        ok, message = self.mcp.set_enabled(bool(checked))
        try:
            self.state.save_settings()
        except OSError:
            pass
        self._update_mcp_ui()
        if checked and not ok:
            QtWidgets.QMessageBox.warning(
                self, self.state.tr.tr("mcp_settings"), message)

    def _autostart_mcp(self) -> None:
        if (self.state.settings.mcp.enabled and
                self.state.settings.mcp.autostart and not self.mcp.running):
            ok, message = self.mcp.start()
            if not ok:
                self._show_status(f"MCP: {message}")

    # ---------------- 后台任务 ----------------
    #: 小于这个面数的网格操作直接同步完成（几毫秒，保持调用语义直观）；
    #: 更大的才进后台，避免几十万面时菜单点一下顿住。
    SYNC_MESH_LIMIT = 50_000

    def _run_mesh_task(self, tag: str, work, done, triangles: int,
                       busy: str | None = None, actions=()) -> None:
        if triangles <= self.SYNC_MESH_LIMIT:
            try:
                done(work())
            except Exception as exc:      # noqa: BLE001
                QtWidgets.QMessageBox.critical(
                    self, self.state.tr.tr("error"), f"{exc}")
            return
        self._run_async(tag, work, done, busy=busy, actions=actions)

    def _run_async(self, tag: str, fn, on_done=None, busy: str | None = None,
                   actions=()) -> bool:
        """提交一个后台任务，立即返回；完成后在 GUI 线程调用 ``on_done``。

        同一个 tag 同时只允许一个任务（避免连点排队）。``actions`` 里的
        动作在任务期间临时置灰，其余界面（画布、菜单、快捷键）照常响应。
        """
        if tag in self._work_jobs:
            self._show_status(self.state.tr.tr("busy_wait"))
            return False
        self._work_jobs[tag] = (on_done, tuple(actions))
        self._work_busy_tag = tag
        for key in actions:
            act = self.acts.get(key)
            if act is not None:
                act.setEnabled(False)
        if busy:
            self._show_status(busy)
        task = _WorkTask(tag, fn, self._work_bridge)
        self._work_tasks[tag] = task
        self._work_pool.start(task)
        return True

    def _on_work_finished(self, tag: str, result, error) -> None:
        self._work_tasks.pop(tag, None)
        job = self._work_jobs.pop(tag, None)
        callback, actions = job if job else (None, ())
        for key in actions:
            act = self.acts.get(key)
            if act is not None:
                act.setEnabled(True)
        if self._work_busy_tag == tag:
            self._work_busy_tag = None
        if error is not None:
            QtWidgets.QMessageBox.critical(
                self, self.state.tr.tr("error"), f"{error}")
            return
        if callback is not None:
            try:
                callback(result)
            except Exception as exc:      # noqa: BLE001
                QtWidgets.QMessageBox.critical(
                    self, self.state.tr.tr("error"), f"{exc}")

    def keyPressEvent(self, event) -> None:
        # ESC 取消测量：焦点在主窗口（工具条/菜单刚点过）时也能生效
        if event.key() == QtCore.Qt.Key_Escape and \
                self.state.measures.pending is not None:
            self.canvas._cancel_measure()
            event.accept()
            return
        super().keyPressEvent(event)

    def _on_scene_changed(self) -> None:
        """画布内部改变了显示状态时，同步菜单勾选与侧栏面板。"""
        opts = self.state.options
        for key, value in (("wire", opts.wire), ("solid", opts.solid),
                           ("colored", is_colored_mode(opts.color_mode)),
                           ("transparent", opts.transparent),
                           ("back_and_front", opts.back_and_front),
                           ("show_edges", opts.show_edges),
                           ("crosshair", opts.crosshair),
                           ("orthogonal", self.state.camera.orthographic)):
            act = self.acts.get(key)
            if act is not None:
                act.setChecked(bool(value))
        self.panel.refresh()
        self._update_statusbar()

    def _mark_dirty(self) -> None:
        self.canvas.mark_dirty()

    # ---------------- 周期任务 ----------------
    def _on_tick(self) -> None:
        state = self.state
        changed = False
        if state.spin.config.enabled:
            if state.spin.tick(state.camera):
                changed = True
        if state.macro.recording:
            state.macro.capture(state.camera)
        if state.macro.playback_active:
            if state.macro.playback_state(state.camera):
                changed = True
            else:
                self.statusBar().showMessage("macro playback finished", 3000)
        # 动画（旋转/宏播放）期间用交互质量渲染，避免逐帧全质量卡顿
        self.canvas.set_animating(
            bool(state.spin.config.enabled or state.macro.playback_active))
        if changed:
            self.canvas.mark_dirty()
            self._update_statusbar()

    # ---------------- 文件 ----------------
    def open_dialog(self) -> None:
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, self.state.tr.tr("open"), self.state.settings.default_directory,
            MODEL_FILTER)
        if path:
            self.open_path(path)

    #: 小于这个体积的模型直接同步载入（几十毫秒，测试与日常小件都不受影响）
    SYNC_LOAD_LIMIT = 4 * 1024 * 1024

    def open_path(self, path: str) -> None:
        tr = self.state.tr
        if "open" in self._work_jobs:
            self._show_status(tr.tr("busy_wait"))
            return
        try:
            size = os.path.getsize(path)
        except OSError as exc:
            QtWidgets.QMessageBox.critical(
                self, tr.tr("error"), tr.tr("file_open_error", path) + f"\n\n{exc}")
            return
        weld = bool(self.state.settings.auto_weld)
        if size <= self.SYNC_LOAD_LIMIT:
            try:
                mesh = Mesh.load(path, weld=weld)
            except Exception as exc:
                QtWidgets.QMessageBox.critical(
                    self, tr.tr("error"),
                    tr.tr("file_open_error", path) + f"\n\n{exc}")
                return
            self._finish_open(path, mesh)
            return
        # 大模型：解析 + 焊接顶点可能要几秒，放到后台，界面继续响应。
        def work():
            return Mesh.load(path, weld=weld)

        self._run_async("open", work,
                        lambda mesh: self._finish_open(path, mesh),
                        busy=tr.tr("loading_model"),
                        actions=("open",))

    def _finish_open(self, path: str, mesh) -> None:
        """载入完成后的 GUI 收尾（同步/后台两条路径共用）。"""
        name = os.path.splitext(os.path.basename(path))[0]
        obj = self.state.scene.add_mesh(mesh, name=name, source_path=path)
        self.state.current_path = path
        self.state.settings.add_recent(path)
        self.state.settings.default_directory = os.path.dirname(path)
        self.state.measures.clear()
        self.state.fit_view()
        self.schedule_fit()          # 布局稳定后按真实画布尺寸再适配一次
        self.panel.refresh()
        self.canvas.mark_dirty()
        self._recent_menu()
        # 深度统计（连通分量 / 边使用计数）在 84 万面模型上要十几秒，
        # 打开文件时只为显示一行提示，直接用面数即可。
        try:
            size_txt = stl_io.human_size(os.path.getsize(path))
        except OSError:
            size_txt = ""
        self._show_status(
            f"{obj.name}: {obj.mesh.triangle_count} tri  {size_txt}")

    def _recent_menu(self) -> None:
        menu = self.acts["recent"]
        menu.clear()
        for path in self.state.settings.recent_files:
            menu.addAction(os.path.basename(path),
                           lambda checked=False, p=path: self.open_path(p))

    def save_active_stl(self) -> None:
        obj = self.state.scene.active_object() or (
            self.state.scene.objects[0] if self.state.scene.objects else None)
        if obj is None:
            return
        if "save_stl" in self._work_jobs:
            self._show_status(self.state.tr.tr("busy_wait"))
            return
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, self.state.tr.tr("export_stl"),
            os.path.join(self.state.settings.default_directory, obj.name + ".stl"),
            "STL (*.stl)")
        if not path:
            return
        # 只传数组引用：网格改动（修复/拆分）会整体替换数组，后台写盘期间
        # 拿到的仍是一致的旧快照；这样不必在点击时复制几十 MB 顶点。
        vertices, triangles, name = obj.mesh.vertices, obj.mesh.triangles, obj.name
        ascii_mode = path.lower().endswith(".txt")

        def work():
            if ascii_mode:
                stl_io.write_ascii_stl(path, vertices, triangles, name)
            else:
                stl_io.write_binary_stl(path, vertices, triangles)
            return path

        self._run_async(
            "save_stl", work,
            lambda saved: self._show_status(
                f"{self.state.tr.tr('save')}: {saved}"),
            busy=self.state.tr.tr("saving_stl"),
            actions=("save_object", "export_stl"))

    def _render_picture(self, transparent: bool = False,
                        supersample: int | None = None,
                        double: bool = False) -> np.ndarray:
        """同步渲染一张高分辨率图片（离屏测试 / 脚本 / 透明与多分辨率导出）。

        普通「保存图片」不再走这里：它直接抓取画布当前帧（见
        :meth:`Canvas.snapshot`），无需重新光栅化整个场景。
        """
        state = self.state
        opts = state.render_options()
        if transparent:
            opts.transparent_background = True
        if supersample:
            opts.supersample = max(int(supersample), 1)
        w, h = self.canvas.width(), self.canvas.height()
        if double:
            w, h = w * 2, h * 2
        return state.renderer.render(state.scene, state.camera, size=(w, h),
                                     options=opts, light=state.light,
                                     measurements=state.measures.items)

    # ---------------- 图片导出（后台） ----------------
    #: 单张导出的渲染像素上限；超过就自动降超采样，避免 4K/多分辨率导出
    #: 一次吃掉几十秒 CPU（渲染再快，用户也不想点一下等半天）。
    MAX_PICTURE_PIXELS = 12_000_000
    MAX_MULTI_RES_PIXELS = 16_000_000

    def _bounded_supersample(self, width: int, height: int,
                             supersample: int, limit: int) -> int:
        ss = max(int(supersample), 1)
        while ss > 1 and width * height * ss * ss > limit:
            ss -= 1
        return max(ss, 1)

    def _snapshot_job(self, path: str):
        """普通保存：只把画布当前帧编码写盘，没有任何渲染工作。"""
        frame = self.canvas.snapshot()
        size = (int(frame.shape[1]), int(frame.shape[0]))

        def work():
            _write_image_file(path, frame)
            return size, 1

        return work

    def _picture_job(self, path: str, width: int, height: int,
                     transparent: bool = False, supersample: int = 2,
                     limit: int = MAX_PICTURE_PIXELS):
        """构造一个可在后台线程执行的「渲染 + 写文件」任务。

        只捕获渲染需要的快照，不引用任何 QWidget；返回 (work, 尺寸信息)。
        """
        state = self.state
        size = (max(int(width), 1), max(int(height), 1))
        opts = state.render_options()
        opts.transparent_background = bool(transparent)
        opts.supersample = self._bounded_supersample(
            size[0], size[1], supersample, limit)
        scene = Scene()
        scene.objects = [copy.copy(o) for o in state.scene.objects]
        scene.background = state.scene.background
        scene.background_top = state.scene.background_top
        scene.gradient_background = state.scene.gradient_background
        camera = state.camera.copy()
        camera.set_viewport(*size)
        light = copy.deepcopy(state.light)
        measurements = list(state.measures.items)

        def work():
            renderer = SoftwareRenderer()
            image = renderer.render(scene, camera, size=size, options=opts,
                                    light=light, measurements=measurements)
            _write_image_file(path, image)
            return size, opts.supersample

        return work, size

    def _export_picture_async(self, title: str, default_name: str,
                              transparent: bool = False,
                              supersample: int = 2, scale: int = 1,
                              limit: int | None = None) -> None:
        """先问路径，再把「当前画布」编码写盘 —— 点击后立即有文件对话框。

        * 普通保存 / 不透明 / 1 倍尺寸：直接抓帧，无渲染；
        * 透明背景与多分辨率是真正需要新图的场景，才重新渲染一次。
        """
        if "picture" in self._work_jobs:
            self._show_status(self.state.tr.tr("busy_wait"))
            return
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, title,
            os.path.join(self.state.settings.default_directory, default_name),
            PICTURE_FILTER)
        if not path:
            return
        tr = self.state.tr
        snapshot_mode = not transparent and max(int(scale), 1) == 1
        if snapshot_mode:
            work = self._snapshot_job(path)
        else:
            width = self.canvas.width() * max(int(scale), 1)
            height = self.canvas.height() * max(int(scale), 1)
            cap = (self.MAX_MULTI_RES_PIXELS if scale > 1
                   else self.MAX_PICTURE_PIXELS)
            if limit is not None:
                cap = limit
            work, _size = self._picture_job(path, width, height, transparent,
                                            supersample, cap)
        tag = "picture"
        self._run_async(
            tag, work,
            lambda info: self._picture_saved(path, info),
            busy=tr.tr("saving_picture" if snapshot_mode
                       else "rendering_picture"),
            actions=("save_picture", "save_picture_transp",
                     "save_multi_res_picture", "copy_picture_clipboard"))

    def _picture_saved(self, path: str, info) -> None:
        tr = self.state.tr
        try:
            size_txt = f"{int(info[0][0])}\u00d7{int(info[0][1])}"
            note = f"  ({size_txt})"
        except Exception:
            note = ""
        self._show_status(f"{tr.tr('picture_saved')}: {path}{note}")

    def save_picture(self, transparent: bool = False) -> None:
        """普通保存直接抓当前画布帧；透明背景需要一张新图，只渲染 1 倍。"""
        tr = self.state.tr
        self._export_picture_async(
            tr.tr("save_picture_transp" if transparent else "save_picture"),
            "picture_transparent.png" if transparent else "picture.png",
            transparent=transparent,
            supersample=1 if transparent else 2)

    def save_multi_res_picture(self) -> None:
        # 2 倍分辨率、1 倍超采样：像素量是普通导出的 4 倍，但不会像
        # 「2 倍尺寸 × 3 倍超采样」那样到 36 倍，几秒内即可完成。
        self._export_picture_async(
            self.state.tr.tr("save_multi_res_picture"),
            "picture_2x.png", supersample=1, scale=2)

    def copy_picture_to_clipboard(self) -> None:
        """复制当前画布帧到剪贴板。

        抓帧 + 一次内存拷贝都在毫秒级，直接在 GUI 线程完成即可（QClipboard
        本来就要求主线程），不再排后台渲染任务。
        """
        state = self.state
        buffer = self.canvas.snapshot()
        hh, ww, ch = buffer.shape
        fmt = (QtGui.QImage.Format_RGBA8888 if ch == 4
               else QtGui.QImage.Format_RGB888)
        qimg = QtGui.QImage(buffer.data, ww, hh,
                            buffer.strides[0], fmt).copy()
        QtWidgets.QApplication.clipboard().setImage(qimg)
        self._show_status(state.tr.tr("copy_picture_done"))

    def show_file_size(self) -> None:
        path = self.state.current_path
        if not path:
            self._show_status(self.state.tr.tr("no_objects"))
            return

        def show(report):
            lines = [f"{k}: {v}" for k, v in report.items()]
            StatisticsDialog("\n".join(lines),
                             self.state.tr.tr("show_stl_filesize"),
                             self).exec_()

        # file_size_report 会完整解析一遍文件，大模型上要几百毫秒；
        # 放到后台，菜单点击立即返回。
        self._run_async("file_size", lambda: stl_io.file_size_report(path), show,
                        busy=self.state.tr.tr("busy_working"))

    def save_object_settings(self) -> None:
        obj = self.state.scene.active_object()
        if obj is None:
            return
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, self.state.tr.tr("save_object_settings"),
            os.path.join(self.state.settings.default_directory,
                         obj.name + ".obj.json"), "JSON (*.json)")
        if not path:
            return
        payload = {"object": obj.to_state(), "camera": self.state.camera.state(),
                   "light": self.state.light.state()}
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=1)
        self._show_status(f"{self.state.tr.tr('save_object_settings')}: {path}")

    def reload_object_settings(self) -> None:
        obj = self.state.scene.active_object()
        if obj is None:
            return
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, self.state.tr.tr("reload_object_settings"),
            self.state.settings.default_directory, "JSON (*.json)")
        if not path:
            return
        with open(path, "r", encoding="utf-8") as fh:
            payload = json.load(fh)
        obj.set_state(payload.get("object", {}))
        self.state.camera.set_state(payload.get("camera", {}))
        self.state.light.set_state(payload.get("light", {}))
        self.panel.refresh()
        self.canvas.mark_dirty()

    # ---------------- 视图 ----------------
    def _zoom(self, factor: float) -> None:
        """factor < 1 放大，> 1 缩小；锚点为画布中心。

        dolly 的倍率是 ``exp(-rate * steps)``，因此要用当前灵敏度换算档数，
        否则用户调了「缩放速度」按钮却不起作用。
        """
        rate = max(float(self.state.settings.zoom_speed), 1e-4)
        self.state.camera.dolly_about_center(
            -math.log(max(factor, 1e-6)) / rate, rate)
        self.canvas.mark_dirty()

    def _fit(self) -> None:
        self.state.fit_view()
        self.canvas.mark_dirty()

    def _reset_center(self) -> None:
        self.state.camera.reset_center()
        self.canvas.mark_dirty()

    def _change_center(self) -> None:
        obj = self.state.scene.visible_objects()
        if obj:
            self.state.camera.change_center(obj[0].center())
            self.canvas.mark_dirty()

    def _standard_view(self, name: str) -> None:
        self.state.standard_view(name)
        self.canvas.mark_dirty()

    def _sync_rotation_ui(self) -> None:
        """把「旋转轴 / 旋转方式 / 启用旋转」三处勾选与运行状态对齐。

        菜单、工具条、快捷键、空格键共用一个状态，任何一个入口改动后都
        由这里统一刷新，避免出现「按钮打着勾但没转」或「按钮没反应」。
        """
        state = self.state
        axis = state.rotate_axis
        for key, ax, _ in ROTATE_AXIS_MENU:
            act = self.acts.get(key)
            if act is not None:
                act.setChecked(ax == axis)
        z_act = self.acts.get("rotate_around_z")
        if z_act is not None:
            z_act.setChecked(axis == "z")
        for kind in SPIN_KINDS:
            act = self.acts.get("spin_" + kind)
            if act is not None:
                act.setChecked(state.spin.config.kind == kind)
        spin_act = self.acts.get("spin_enable")
        if spin_act is not None and spin_act.isChecked() != state.spin.config.enabled:
            blocked = spin_act.blockSignals(True)   # 只同步显示，别再触发一次
            spin_act.setChecked(state.spin.config.enabled)
            spin_act.blockSignals(blocked)
        pair = getattr(self, "_rotate_axis_button", None)
        if pair is not None:
            tr = state.tr
            pair[0].setText(f"{tr.tr('rotate_axis')}: "
                            f"{AXIS_LABELS.get(axis, axis)}")

    def _set_spin_enabled(self, enabled: bool) -> None:
        """启用 / 停止旋转动画（菜单、工具条、空格键的唯一入口）。"""
        state = self.state
        enabled = bool(enabled)
        state.spin.config.enabled = enabled
        state.spin.reset()
        # 动画用的轴必须与当前定轴旋转一致，否则选了 X 轴却仍绕 Z 转。
        kind = AXIS_TO_SPIN.get(state.rotate_axis)
        if kind is not None and state.spin.config.kind != kind:
            state.spin.config.kind = kind
            state.spin.reset()
        self._sync_rotation_ui()
        self.canvas.set_animating(enabled or state.macro.playback_active)
        self.canvas.mark_dirty()
        tr = state.tr
        note = (f"  [{AXIS_LABELS.get(state.rotate_axis, state.rotate_axis)}]"
                if enabled else "")
        self._show_status(
            f"{tr.tr('spin_started' if enabled else 'spin_stopped')}{note}")

    def _set_rotation_axis(self, axis: str) -> None:
        """选择定轴旋转的轴：free / x / y / z / view。

        菜单、快捷键、工具条下拉共用这一入口；鼠标拖拽与「启用旋转」动画
        共用同一根轴，选中即生效，动画进行中也会立刻换轴。
        """
        axis = axis if axis in ROTATE_AXES else "free"
        state = self.state
        state.rotate_axis = axis
        self.canvas._nav.set_axis("z" if axis == "free" else axis)
        kind = AXIS_TO_SPIN.get(axis, "horizontal")
        if state.spin.config.kind != kind:
            state.spin.config.kind = kind
            state.spin.reset()
        self._sync_rotation_ui()
        self.canvas.mark_dirty()
        tr = state.tr
        self._show_status(f"{tr.tr('rotate_axis')}: "
                          f"{AXIS_LABELS.get(axis, axis)}"
                          + (f"  ·  {tr.tr('spin')}" if state.spin.config.enabled
                             else ""))

    def _toggle_rotate_z(self, checked: bool) -> None:
        """View > Rotate around Z（Ctrl+Z）：绕 Z 的快捷开关。"""
        self._set_rotation_axis("z" if checked else "free")

    def _toggle_ortho(self, checked: bool) -> None:
        self.state.camera.set_orthographic(checked)
        self.state.settings.view.orthographic = checked
        self.canvas.mark_dirty()

    def _toggle_fixed(self, checked: bool) -> None:
        self.state.settings.view.fixed_on_screen = checked

    def _toggle_fullscreen(self, checked: bool) -> None:
        if checked:
            self.showFullScreen()
        else:
            self.showNormal()

    def _toggle_crosshair(self, checked: bool) -> None:
        self.state.options.crosshair = checked
        self.state.settings.view.crosshair = checked
        self.canvas.mark_dirty()

    def _slider_row(self, dialog, value: float, lo: int, hi: int,
                    suffix: str = "%"):
        """带数值回显的百分比滑条，返回 (slider, 可直接加入表单的容器)。"""
        slider = QtWidgets.QSlider(QtCore.Qt.Horizontal, dialog)
        slider.setRange(lo, hi)
        slider.setSingleStep(5)
        slider.setPageStep(10)
        slider.setValue(int(round(value)))
        label = QtWidgets.QLabel(f"{slider.value()}{suffix}")
        label.setMinimumWidth(52)
        slider.valueChanged.connect(lambda v: label.setText(f"{v}{suffix}"))
        row = QtWidgets.QHBoxLayout()
        row.setContentsMargins(0, 0, 0, 0)
        row.addWidget(slider, 1)
        row.addWidget(label)
        wrap = QtWidgets.QWidget(dialog)
        wrap.setLayout(row)
        return slider, wrap

    def _view_options(self) -> None:
        tr = self.state.tr
        settings = self.state.settings
        items = ["show_axis", "axis_labels", "show_grid", "crosshair",
                 "show_edges", "back_and_front", "gradient_background"]
        dialog = QtWidgets.QDialog(self)
        dialog.setWindowTitle(tr.tr("view_options"))
        form = QtWidgets.QFormLayout(dialog)
        boxes = {}
        for key in items:
            box = QtWidgets.QCheckBox(tr.tr(key))
            box.setChecked(bool(getattr(self.state.options, key, False)))
            boxes[key] = box
            form.addRow(box)

        # ---- 鼠标灵敏度：直接对应"拖一下转多少" ----
        form.addRow(QtWidgets.QLabel(f"— {tr.tr('interaction')} —"))
        rotate = self._slider_row(dialog, settings.rotate_speed * 100, 5, 60)
        pan = self._slider_row(dialog, settings.pan_speed * 100, 20, 300)
        zoom = self._slider_row(dialog, settings.zoom_speed * 100, 2, 40)
        form.addRow(tr.tr("rotate_speed"), rotate[1])
        form.addRow(tr.tr("pan_speed"), pan[1])
        form.addRow(tr.tr("zoom_speed"), zoom[1])

        # ---- CAD 风格视图导航 ----
        form.addRow(QtWidgets.QLabel("— 视图导航（对标 CATIA / SolidWorks） —"))

        def combo(pairs, current):
            box = QtWidgets.QComboBox(dialog)
            for value, label in pairs:
                box.addItem(label, value)
            idx = box.findData(current)
            box.setCurrentIndex(idx if idx >= 0 else 0)
            return box

        nav_style = combo(
            [("classic", "经典（原行为）"), ("cad", "CAD（默认）"),
             ("catia", "CATIA"), ("solidworks", "SolidWorks")],
            settings.nav_style)
        rotate_mode = combo(
            [("turntable", "转台（Z 轴朝上）"),
             ("arcball", "自由轨迹球（可越过极点 / 滚转）")],
            settings.nav_rotate_mode)
        zoom_anchor = combo(
            [("cursor", "光标（zoom to cursor）"), ("center", "画布中心")],
            settings.nav_zoom_anchor)
        pivot_mode = combo(
            [("center", "画布中心"), ("cursor", "按下点（Center on point）")],
            settings.nav_pivot)
        nav_inertia = QtWidgets.QCheckBox("释放后惯性旋转", dialog)
        nav_inertia.setChecked(bool(settings.nav_inertia))
        form.addRow("导航方案", nav_style)
        form.addRow("旋转方式", rotate_mode)
        form.addRow("缩放锚点", zoom_anchor)
        form.addRow("旋转支点", pivot_mode)
        form.addRow(nav_inertia)
        # CATIA 的自由旋转习惯：切到该方案时默认启用轨迹球（仍可手动改回）
        def _style_changed(idx):
            if nav_style.itemData(idx) == "catia" and rotate_mode.currentData() == "turntable":
                rotate_mode.setCurrentIndex(rotate_mode.findData("arcball"))
        nav_style.currentIndexChanged.connect(_style_changed)

        # ---- 渲染性能与网格处理 ----
        form.addRow(QtWidgets.QLabel(f"— {tr.tr('render_perf')} —"))
        quality = self._slider_row(dialog, settings.interactive_quality * 100, 15, 100)
        form.addRow(tr.tr("interactive_quality"), quality[1])

        antialias = QtWidgets.QCheckBox(tr.tr("antialias"), dialog)
        antialias.setChecked(bool(settings.antialias))
        form.addRow(antialias)

        auto_weld = QtWidgets.QCheckBox(tr.tr("auto_weld"), dialog)
        auto_weld.setChecked(bool(settings.auto_weld))
        form.addRow(auto_weld)

        buttons = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel)
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        form.addRow(buttons)

        if dialog.exec_() == QtWidgets.QDialog.Accepted:
            for key, box in boxes.items():
                setattr(self.state.options, key, box.isChecked())
            value = max(quality[0].value(), 1) / 100.0
            settings.interactive_quality = value
            settings.antialias = antialias.isChecked()
            settings.auto_weld = auto_weld.isChecked()
            settings.rotate_speed = max(rotate[0].value(), 1) / 100.0
            settings.pan_speed = max(pan[0].value(), 1) / 100.0
            settings.zoom_speed = max(zoom[0].value(), 1) / 100.0
            nav_options = NavOptions(
                style=nav_style.currentData(),
                rotate_mode=rotate_mode.currentData(),
                zoom_anchor=zoom_anchor.currentData(),
                pivot=pivot_mode.currentData(),
                inertia=nav_inertia.isChecked(),
            ).normalized()
            settings.nav_style = nav_options.style
            settings.nav_rotate_mode = nav_options.rotate_mode
            settings.nav_zoom_anchor = nav_options.zoom_anchor
            settings.nav_pivot = nav_options.pivot
            settings.nav_inertia = nav_options.inertia
            self.canvas.set_nav_options(
                nav_options,
                {"rotate": settings.rotate_speed, "pan": settings.pan_speed,
                 "zoom": settings.zoom_speed})
            self.canvas.set_interactive_quality(value)
            self.state.settings.antialias = antialias.isChecked()
            self.canvas.mark_dirty()

    def show_rotation_dialog(self) -> None:
        dialog = RotationDialog(self.state.camera, self.state.tr, self)
        if dialog.exec_() == QtWidgets.QDialog.Accepted:
            yaw, pitch, roll = dialog.values()
            self.state.camera.set_rotation_angles(yaw, pitch, roll)
            self.canvas.mark_dirty()

    # ---------------- 显示模式 ----------------
    def _set_wire(self, checked: bool) -> None:
        self.state.options.wire = checked
        if checked:
            self.state.options.solid = False
            self.acts["solid"].setChecked(False)
        self.state.settings.view.wire = checked
        self.canvas.mark_dirty()

    def _set_solid(self, checked: bool) -> None:
        self.state.options.solid = checked
        if checked:
            self.state.options.wire = False
            self.acts["wire"].setChecked(False)
        self.state.settings.view.solid = checked
        self.canvas.mark_dirty()

    def _set_colored(self, checked: bool) -> None:
        """彩色 / 单色。两者都是"整件同色"，区别是按面还是逐顶点平滑着色。"""
        self._set_color_mode("auto" if checked else "uniform")

    def _set_color_mode(self, mode: str) -> None:
        self.state.options.color_mode = mode
        self.state.settings.view.color_mode = mode
        for obj in self.state.scene.objects:
            obj.color_mode = mode
            obj.auto_color = mode != "uniform"
            obj.invalidate_colors()
        # 传进来的可能不是 "auto"/"uniform"（分量、曲率、法向），此时开关按
        # "当前是否使用逐面取色" 置灰，避免勾选状态与实际显示不符。
        self.acts["colored"].setChecked(is_colored_mode(mode))
        self.canvas.mark_dirty()

    def _toggle_transparent(self, checked: bool) -> None:
        self.state.options.transparent = checked
        self.state.settings.view.transparent = checked
        self.canvas.mark_dirty()

    def _toggle_two_sided(self, checked: bool) -> None:
        self.state.light.two_sided = checked
        self.state.options.back_and_front = checked
        self.state.settings.view.back_and_front = checked
        self.canvas.mark_dirty()

    def _toggle_edges(self, checked: bool) -> None:
        self.state.options.show_edges = checked
        self.state.settings.view.show_edges = checked
        self.canvas.mark_dirty()

    def _select_object_color(self) -> None:
        obj = self.state.scene.active_object()
        color = obj.color if obj else self.state.options.background
        dialog = ColorDialog(self.state.tr.tr("select_mesh_color"), color, self)
        if dialog.exec_() == QtWidgets.QDialog.Accepted:
            targets = [obj] if obj else self.state.scene.objects
            for target in targets:
                target.color = dialog.color
                target.invalidate_colors()
            self.canvas.mark_dirty()

    def _point_size_dialog(self) -> None:
        dialog = PointSizeDialog(self.state, self)
        if dialog.exec_() == QtWidgets.QDialog.Accepted:
            self.state.options.point_size = dialog.point.value()
            self.state.options.line_width = dialog.line.value()
            self.state.settings.view.point_size = dialog.point.value()
            self.state.settings.view.line_width = dialog.line.value()
            self.canvas.mark_dirty()

    def _object_properties(self) -> None:
        obj = self.state.scene.active_object()
        if obj is None:
            return
        dialog = ObjectPropertiesDialog(obj, self.state.tr, self)
        if dialog.exec_() == QtWidgets.QDialog.Accepted:
            dialog.apply()
            self.panel.refresh()
            self.canvas.mark_dirty()

    # ---------------- 光照 ----------------
    def show_light_dialog(self) -> None:
        dialog = LightDialog(self.state, self)
        dialog.exec_()
        self.canvas.mark_dirty()

    def _set_light(self, direction: str) -> None:
        self.state.light.set_primary(direction)
        self.canvas.mark_dirty()

    def _reset_light(self) -> None:
        self.state.light.reset()
        self.canvas.mark_dirty()

    def _background_colors(self) -> None:
        scene = self.state.scene
        dialog = ColorDialog(self.state.tr.tr("background_colors"), scene.background,
                             self, with_gradient=True, top_color=scene.background_top)
        if dialog.gradient is not None:
            dialog.gradient.setChecked(scene.gradient_background)
        if dialog.exec_() == QtWidgets.QDialog.Accepted:
            scene.background = dialog.color
            scene.background_top = dialog.top
            if dialog.gradient is not None:
                scene.gradient_background = dialog.gradient.isChecked()
            self.state.options.background = scene.background
            self.state.options.background_top = scene.background_top
            self.state.options.gradient_background = scene.gradient_background
            self.canvas.mark_dirty()

    # ---------------- 测量 ----------------
    def _toggle_measure_3d(self, checked: bool) -> None:
        """切换长度口径：视图投影（默认）/ 真实三维距离。

        默认口径与 CAD 的「在视图里量」一致：TOP 视图量到的就是投影长度，
        不受取点深度影响；需要在轴测视图里量真实空间距离时再打开。
        """
        tr = self.state.tr
        mode = "3d" if checked else "view"
        self.state.settings.measure_mode = mode
        self.state.measures.set_mode(mode)
        self._show_status(tr.tr("measure_mode_changed",
                                tr.tr("measure_mode_3d" if checked
                                      else "measure_mode_view")))

    def _delete_first_measure(self) -> None:
        self.state.measures.delete_first()
        self.panel.refresh()
        self.canvas.mark_dirty()

    def _delete_last_measure(self) -> None:
        self.state.measures.delete_last()
        self.panel.refresh()
        self.canvas.mark_dirty()

    def _delete_all_measures(self) -> None:
        self.state.measures.delete_all()
        self.panel.refresh()
        self.canvas.mark_dirty()

    def _measure_color(self) -> None:
        dialog = ColorDialog(self.state.tr.tr("measure_color"),
                             self.state.measures.color, self)
        if dialog.exec_() == QtWidgets.QDialog.Accepted:
            self.state.measures.set_color(dialog.color)
            self.canvas.mark_dirty()

    # ---------------- 单位 ----------------
    def set_units(self, unit: str) -> None:
        self.state.units.unit = unit
        self.state.settings.units = unit
        self.state.measures.refresh_labels(self.state.units)
        for key in ("unit_mm", "unit_inch", "measure_unit_mm", "measure_unit_inch"):
            act = self.acts.get(key)
            if act is not None:
                act.setChecked(key.endswith(unit))
        self.panel.refresh()
        self._update_statusbar()
        self.canvas.mark_dirty()

    # ---------------- 语言 ----------------
    def set_language(self, code: str) -> None:
        self.state.tr.set_language(code)
        self.state.settings.language = code
        self.retranslate()
        self._show_status(f"language: {code}")

    # ---------------- 渲染系统 / 加速 ----------------
    def _set_render_system(self, value: str) -> None:
        self.state.settings.render_system = value
        for label in ("OpenGL", "Direct3D", "Software"):
            key = "render_" + label.lower()
            if key in self.acts:
                self.acts[key].setChecked(label == value)
        self._show_status(f"render system: {value}")

    def _toggle_hw_accel(self, checked: bool) -> None:
        self.state.settings.hardware_acceleration = checked
        self._show_status("hardware acceleration: " + ("on" if checked else "off"))

    # ---------------- 旋转动画 ----------------
    def _toggle_spin(self, checked: bool) -> None:
        self._set_spin_enabled(checked)

    def _set_spin_type(self, kind: str) -> None:
        """旋转方式与定轴旋转双向联动：选 RX 等于把拖拽轴也切成 X。"""
        if kind not in SPIN_KINDS:
            return
        state = self.state
        state.spin.config.kind = kind
        state.spin.reset()
        axis = SPIN_TO_AXIS.get(kind, "free")
        state.rotate_axis = axis
        self.canvas._nav.set_axis("z" if axis == "free" else axis)
        self._sync_rotation_ui()
        self.canvas.mark_dirty()
        tr = state.tr
        self._show_status(f"{tr.tr('spin_axis_hint')}: "
                          f"{AXIS_LABELS.get(axis, axis)}")

    def _set_spin_direction(self, value: int) -> None:
        self.state.spin.config.direction = value

    def _set_spin_speed(self, value: str) -> None:
        self.state.spin.config.speed = value

    # ---------------- 宏 ----------------
    def _toggle_macro_record(self) -> None:
        rec = self.state.macro
        if not rec.recording:
            rec.start()
            self.acts["macro_start_recording"].setText(
                self.state.tr.tr("macro_stop_recording"))
            self.statusBar().showMessage("recording...", 3000)
        else:
            n = rec.stop()
            self.acts["macro_start_recording"].setText(
                self.state.tr.tr("macro_start_recording"))
            self.statusBar().showMessage(f"macro frames: {n}", 4000)

    def _play_macro(self) -> None:
        rec = self.state.macro
        if not rec.macro.frames:
            self.statusBar().showMessage("empty macro", 3000)
            return
        rec.start_playback()

    def _open_macro(self) -> None:
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, self.state.tr.tr("macro_open"),
            self.state.settings.default_directory, "JSON (*.json)")
        if not path:
            return
        try:
            n = self.state.macro.load(path)
            self.statusBar().showMessage(f"macro loaded: {n} frames", 4000)
        except Exception as exc:
            QtWidgets.QMessageBox.critical(self, self.state.tr.tr("error"), str(exc))

    def _save_macro(self) -> None:
        if not self.state.macro.macro.frames:
            self.statusBar().showMessage("empty macro", 3000)
            return
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, self.state.tr.tr("macro_save"),
            os.path.join(self.state.settings.default_directory, "macro.json"),
            "JSON (*.json)")
        if path:
            self.state.macro.save(path)
            self.statusBar().showMessage(f"saved: {path}", 4000)

    # ---------------- 新建图元 ----------------
    def _new_primitive(self, kind: str) -> None:
        scene = self.state.scene
        if kind == "box":
            scene.add_box(size=(30, 30, 30), name="Box")
        elif kind == "cylinder":
            scene.add_cylinder(radius=10, height=40, name="Cylinder")
        elif kind == "step_cylinder":
            scene.add_step_cylinder(name="StepCylinder")
        elif kind == "tube":
            scene.add_tube(name="Tube")
        elif kind == "sphere":
            scene.add_sphere(name="Sphere")
        self.panel.refresh()
        self.canvas.mark_dirty()

    # ---------------- 对象操作 ----------------
    def _select_object(self, index: int) -> None:
        self.state.scene.set_active(index)
        self.panel.refresh()
        self.canvas.mark_dirty()

    def _toggle_visible(self, index: int, visible: bool) -> None:
        if 0 <= index < len(self.state.scene.objects):
            self.state.scene.objects[index].visible = visible
            self.canvas.mark_dirty()

    def _resolve_object(self, index: int):
        scene = self.state.scene
        if index is None or index < 0:
            return scene.active_object()
        if 0 <= index < len(scene.objects):
            return scene.objects[index]
        return None

    def _object_action(self, action: str, index: int = -1) -> None:
        state = self.state
        obj = self._resolve_object(index)
        if obj is None:
            self._show_status(state.tr.tr("no_objects"))
            return
        if action == "visible":
            obj.visible = not obj.visible
        elif action == "wire":
            obj.wire = not obj.wire
        elif action == "transparent":
            obj.transparent = not obj.transparent
        elif action == "always_on_top":
            obj.always_on_top = not obj.always_on_top
        elif action == "locked":
            obj.locked = not obj.locked
        elif action == "active":
            idx = state.scene.objects.index(obj)
            state.scene.set_active(idx)
        elif action == "properties":
            idx = state.scene.objects.index(obj)
            state.scene.set_active(idx)
            self._object_properties()
            return
        elif action == "statistics":
            self._show_mesh_statistics(obj)
            return
        elif action == "copy":
            # 84 万面深拷贝要几百毫秒；大模型放到后台，小模型直接完成。
            def copied(copy_obj):
                state.clipboard_object = copy_obj
                self._show_status(state.tr.tr("copy_object"))

            self._run_mesh_task(
                "copy_object", lambda: copy.deepcopy(obj), copied,
                obj.mesh.triangle_count,
                busy=state.tr.tr("processing_mesh"),
                actions=("copy_object", "paste_object"))
            return
        elif action == "save_stl":
            idx = state.scene.objects.index(obj)
            state.scene.set_active(idx)
            self.save_active_stl()
            return
        elif action == "repair":
            def repaired(info):
                obj.invalidate_colors()
                self.panel.refresh()
                self.canvas.mark_dirty()
                QtWidgets.QMessageBox.information(
                    self, state.tr.tr("repair_surface"),
                    f"weld_merged={info['weld_merged']}\n"
                    f"degenerate_removed={info['degenerate_removed']}\n"
                    f"triangles_flipped={info['triangles_flipped']}\n"
                    f"loose_removed={info['loose_removed']}\n"
                    f"before={info['before']} -> after={info['after']}")

            self._run_mesh_task("repair", obj.mesh.repair, repaired,
                                obj.mesh.triangle_count,
                                busy=state.tr.tr("processing_mesh"),
                                actions=("statistics",))
            return
        elif action == "components":
            def split_done(parts):
                for part in parts:
                    state.scene.add_mesh(part, name=part.name, activate=False)
                self.panel.refresh()
                self.canvas.mark_dirty()
                self._show_status(
                    f"{state.tr.tr('generate_components')}: {len(parts)}")

            self._run_mesh_task("components", obj.mesh.split_components,
                                split_done, obj.mesh.triangle_count,
                                busy=state.tr.tr("processing_mesh"))
            return
        elif action == "remove_tri":
            others = [o for o in state.scene.objects if o is not obj]
            if not others:
                self._show_status(state.tr.tr("no_objects"))
                return
            names = [o.name for o in others]
            name, ok = QtWidgets.QInputDialog.getItem(
                self, state.tr.tr("remove_tri_intersected"), state.tr.tr("objects"),
                names, 0, False)
            if not ok:
                return
            other = others[names.index(name)]

            def removed_done(removed):
                obj.invalidate_colors()
                self.panel.refresh()
                self.canvas.mark_dirty()
                self._show_status(f"removed triangles: {removed}")

            self._run_mesh_task(
                "remove_tri",
                lambda: obj.mesh.remove_triangles_intersecting(other.mesh),
                removed_done, obj.mesh.triangle_count,
                busy=state.tr.tr("processing_mesh"))
            return
        elif action == "delete":
            idx = state.scene.objects.index(obj)
            state.scene.remove_object(idx)
            self.panel.refresh()
            self.canvas.mark_dirty()
            return
        self.panel.refresh()
        self.canvas.mark_dirty()

    def copy_object(self) -> None:
        obj = self.state.scene.active_object()
        if obj is None:
            return

        def done(copied):
            self.state.clipboard_object = copied
            self._show_status(self.state.tr.tr("copy_object"))

        self._run_mesh_task("copy_object", lambda: copy.deepcopy(obj), done,
                            obj.mesh.triangle_count,
                            busy=self.state.tr.tr("processing_mesh"),
                            actions=("copy_object", "paste_object"))

    def paste_object(self) -> None:
        src = self.state.clipboard_object
        if src is None:
            self._show_status(self.state.tr.tr("no_objects"))
            return

        def done(mesh):
            obj = Object3D(mesh=mesh, name=src.name + "_copy"
                           if hasattr(src, "name") else "copy")
            self.state.scene.objects.append(obj)
            self.state.scene.set_active(len(self.state.scene.objects) - 1)
            self.panel.refresh()
            self.canvas.mark_dirty()

        self._run_mesh_task("paste_object", src.mesh.copy, done,
                            src.mesh.triangle_count,
                            busy=self.state.tr.tr("processing_mesh"),
                            actions=("copy_object", "paste_object"))

    def delete_object(self) -> None:
        idx = self.state.scene.active_index
        if idx >= 0:
            self.state.scene.remove_object(idx)
            self.panel.refresh()
            self.canvas.mark_dirty()

    def delete_all_objects(self) -> None:
        self.state.scene.clear()
        self.state.measures.clear()
        self.panel.refresh()
        self.canvas.mark_dirty()

    def _show_mesh_statistics(self, obj) -> None:
        # 深度统计对超大网格是 O(M log M) 且要十几秒；此时只给轻量统计，
        # 并且整个统计放到后台，点一下菜单不会把界面冻住。
        mesh = obj.mesh
        source = obj.source_path
        deep = mesh.triangle_count <= 200_000

        def work():
            text = mesh.statistics(deep=deep).as_text()
            if not deep:
                text += "\n\n(大模型：已跳过连通分量/边界边等深度统计，避免界面卡顿)"
            if source and os.path.exists(source):
                report = stl_io.file_size_report(source)
                text += f"\n\n文件: {source}\n大小: {report['human']}"
            return text

        self._run_async(
            "statistics", work,
            lambda text: StatisticsDialog(
                text, self.state.tr.tr("statistics_title"), self).exec_(),
            busy=self.state.tr.tr("computing_statistics"),
            actions=("statistics",))

    def show_statistics(self) -> None:
        obj = self.state.scene.active_object()
        if obj is not None:
            self._show_mesh_statistics(obj)
        else:
            stats = self.state.statistics()
            text = "\n".join(f"{k}: {v}" for k, v in stats.items())
            StatisticsDialog(text, self.state.tr.tr("statistics_title"), self).exec_()

    # ---------------- 设置 ----------------
    def load_default_settings(self) -> None:
        path = self.state.load_settings()
        self.canvas.set_interactive_quality(
            self.state.settings.interactive_quality)
        self.panel.refresh()
        self.canvas.mark_dirty()
        self.retranslate()
        self.mcp.reload()
        self._update_mcp_ui()
        self._show_status(self.state.tr.tr("settings_loaded", path))

    def save_default_settings(self) -> None:
        path = self.state.save_settings()
        self._show_status(self.state.tr.tr("settings_saved", path))

    def reset_default_settings(self) -> None:
        self.state.settings.reset_to_factory()
        self.state.load_settings()
        self.canvas.set_interactive_quality(
            self.state.settings.interactive_quality)
        self.panel.refresh()
        self.canvas.mark_dirty()
        self.retranslate()
        self.mcp.reload()
        self._update_mcp_ui()

    def show_about(self) -> None:
        tr = self.state.tr
        from . import __version__

        QtWidgets.QMessageBox.about(
            self, tr.tr("about"),
            tr.tr("about_text") + f"\n\n{tr.tr('version')}: {__version__}\n"
            f"Python: {sys.version.split()[0]}")

    # ---------------- 关闭 ----------------
    def closeEvent(self, event) -> None:
        # 先停掉内嵌 MCP 服务：窗口关了就不该继续接受 AI 的会话操作。
        try:
            self.mcp.stop()
        except Exception:                                    # noqa: BLE001
            pass
        # 再把后台任务收干净：正在写盘的图片 / STL 不能因为关窗口被截断。
        deadline = time.monotonic() + 3.0
        while time.monotonic() < deadline:
            try:
                if self._work_pool.activeThreadCount() <= 0:
                    break
            except Exception:
                break
            time.sleep(0.02)
        self._work_pool.clear()
        try:
            self._work_bridge.finished.disconnect(self._on_work_finished)
        except Exception:
            pass
        self.canvas.shutdown()
        try:
            self.state.save_settings()
        except Exception:
            pass
        super().closeEvent(event)


# ==========================================================================
def main(argv=None) -> int:
    argv = list(sys.argv if argv is None else argv)
    QtWidgets.QApplication.setAttribute(QtCore.Qt.AA_EnableHighDpiScaling, True)
    QtWidgets.QApplication.setAttribute(QtCore.Qt.AA_UseHighDpiPixmaps, True)
    app = QtWidgets.QApplication(argv)
    app.setApplicationName("STL Viewer")
    app.setOrganizationName("STL Viewer")

    settings_path = None
    files = []
    args = argv[1:]
    i = 0
    while i < len(args):
        if args[i] in ("--settings", "-s") and i + 1 < len(args):
            settings_path = args[i + 1]
            i += 2
            continue
        files.append(args[i])
        i += 1

    state = ViewerState(settings_path)
    window = MainWindow(state)
    window.showMaximized()          # 启动即最大化
    for path in files:
        if os.path.isfile(path):
            window.open_path(path)
    return app.exec_()


if __name__ == "__main__":
    sys.exit(main())
