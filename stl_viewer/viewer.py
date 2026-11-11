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

交互（全部以画布中心为支点）：
    左键拖拽    旋转（按住 Ctrl 为绕 Z 轴旋转；开启 Rotate around Z 后左键即绕 Z）
    中键拖拽    平移（Shift + 左键 亦可）
    右键拖拽    平移；右键单击弹出上下文菜单
    Alt+左键    缩放（滚轮 / 触控板亦可）
    拖拽中按 Shift   精细模式（旋转/缩放降到 1/4，平移也放慢）
    左键单击    测量模式下取点（自动吸附最近顶点）
    方向键      旋转   Shift+方向键 平移   Ctrl+方向键 绕 Z / 俯仰
    +/-         缩放
    F / Home    适应窗口      R  复位中心      1~8 标准视图    空格 播放/暂停旋转
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
from .macro import MacroRecorder, SpinConfig, SpinEngine
from .measure import MEASURE_DEFS, MeasureManager, snap_pick
from .mesh import Mesh
from .renderer import RenderOptions, SoftwareRenderer
from .scene import Object3D, Scene
from .screenpick import ScreenPicker
from .settings import AppSettings, UnitSystem, ViewOptions

PICTURE_FILTER = "PNG 图片 (*.png);;JPEG 图片 (*.jpg);;BMP 图片 (*.bmp);;所有文件 (*)"
MODEL_FILTER = ("模型文件 (*.stl *.stpl *.ply *.stpack *.exe);;"
                "STL (*.stl);;PLY (*.ply);;自包含查看器 (*.exe *.stpack);;"
                "所有文件 (*)")

# 用颜色表达信息的着色模式（"彩色"开关即以是否落在这些模式内为准）
COLORED_MODES = ("auto", "component", "curvature", "orientation")
COLOR_MODE_CHOICES = ("auto", "uniform", "component", "curvature", "orientation")


def is_colored_mode(mode: str) -> bool:
    return mode in COLORED_MODES


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
        self.spin = SpinEngine(SpinConfig())
        self.macro = MacroRecorder()
        self.options = RenderOptions.from_view_options(self.settings.view)

        # 真实画布像素尺寸；fit / 屏幕射线都用它，避免用构造时的默认值
        self.viewport: tuple = (1024, 768)
        self.current_path: str | None = None
        self.clipboard_object = None
        self.rotate_around_z = False
        self.last_render_seconds = 0.0
        self.fps = 0.0
        self._frame_times: deque = deque(maxlen=240)

        self.settings.apply_to(self.camera, self.scene, self.light)
        self.options = self.render_options()

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


# ==========================================================================
# 渲染画布
# ==========================================================================
class Canvas(QtWidgets.QWidget):
    """3D 视口：后台软件光栅化 + 以画布中心为支点的鼠标交互。"""

    statusMessage = QtCore.pyqtSignal(str)
    sceneChanged = QtCore.pyqtSignal()

    # 后台渲染不阻塞事件循环，画质优先：只有实测明显掉帧才降分辨率，
    # 避免拖拽过程中画面一直糊成低清图（这也是「显示失真」的来源之一）。
    TARGET_FRAME_SECONDS = 1.0 / 30.0
    MIN_QUALITY = 0.35             # 交互分辨率下限
    MIN_RENDER_EDGE = 160          # 交互位图最小边长
    REFINE_DELAY_MS = 90           # 停止操作多久后开始精修全质量帧
    KEY_ROTATE_STEP = 5.0          # 方向键每次旋转的角度（度）
    # 单次鼠标事件的最大位移：高刷新率鼠标 / 触控板偶尔给出上百像素的
    # 跳变，不夹住就会「一下转过头 / 一下缩到极限」。
    MAX_MOUSE_STEP = 60            # 旋转/缩放的每事件像素上限
    MAX_PAN_STEP = 120             # 平移略放宽，保持跟手
    ZOOM_DRAG_PER_PX = 0.020       # Alt+左键每像素折算的滚轮档数
    FINE_MODIFIER_FACTOR = 0.25    # 拖拽中按住 Shift = 精细模式
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
        self._hover_point = None       # 悬停点（带真实深度），None 表示按视图平面算
        self._hover_snapped = False
        self._measure_nav = False      # 本次按压是否已转为视图导航（拖动旋转）

        # 测量取点用的屏幕投影索引（纯 CPU，绝不调用 GL）：把当前视图下的
        # 模型投影到屏幕建二维格子，点击只在该格内做精确求交。
        self._screen_picker = ScreenPicker()

        # 视图停下来后（250ms 没有新的重绘请求）提前把索引建好：
        # 用户旋转完视图再点下去时就不用等那一次重建。
        self._pick_prepare_timer = QtCore.QTimer(self)
        self._pick_prepare_timer.setSingleShot(True)
        self._pick_prepare_timer.timeout.connect(self._prepare_measure_pick)

        # 悬停取点用「停手再算」：鼠标一动就做表面拾取会拖慢大模型，
        # 先按视图平面画预览，用户停住 90ms 后再补一次真实深度。
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
        if not (self._interactive or self._animating):
            self._quick = True
            self._refine_timer.start(self.REFINE_DELAY_MS)

    def _refine(self) -> None:
        """空闲一段时间后把快帧替换成全质量帧。"""
        if self._interactive or self._animating or not self._quick:
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
        self.mark_dirty()
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
        reduced = self._interactive or self._animating or self._quick
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

    def shutdown(self) -> None:
        """窗口关闭时收掉后台任务，避免任务回调打到已销毁的画布。"""
        self._refine_timer.stop()
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
        # 实心点 = 取自模型表面（有真实深度），空心点 = 取自视图平面。
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
        # 预览点默认按视图平面算，停手 90ms 后换成模型真实深度（_hover_pick），
        # 因此按下鼠标之前看到的长度就是最终结果。
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
                        circ2 = mu.circle_from_two_points(
                            center, hp, cam.forward())
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

        # 视图原点标记：旋转与缩放都围绕画布中心，这里把支点显式画出来。
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
            tips = np.array([[0.0, 0.0, 0.0],
                             [length, 0.0, 0.0],
                             [0.0, length, 0.0],
                             [0.0, 0.0, length]], dtype=np.float64)
            s, _, valid = cam.project(tips)
            colors = ((0.86, 0.24, 0.22), (0.34, 0.72, 0.32),
                      (0.26, 0.52, 0.90))
            names = ("X", "Y", "Z")
            if valid.all() and np.isfinite(s).all():
                o = QtCore.QPointF(*s[0])
                for i in range(3):
                    tip = QtCore.QPointF(*s[i + 1])
                    painter.setPen(QtGui.QPen(qcolor(colors[i]), 2.0))
                    painter.drawLine(o, tip)
                    if opts.axis_labels:
                        painter.drawText(tip + QtCore.QPointF(7, -4), names[i])

        if opts.crosshair:
            cx, cy = W / 2.0, H / 2.0
            arm = max(min(W, H) * 0.03, 10.0)
            gap = arm * 0.28
            base = np.asarray(opts.background, dtype=np.float64)
            painter.setPen(QtGui.QPen(qcolor(1.0 - base * 0.55), 1.0))
            for a, b in (((cx - arm, cy), (cx - gap, cy)),
                         ((cx + gap, cy), (cx + arm, cy)),
                         ((cx, cy - arm), (cx, cy - gap)),
                         ((cx, cy + gap), (cx, cy + arm))):
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
                    painter.drawText(QtCore.QPointF(s[0, 0] + 8, s[0, 1] - 8),
                                     text)

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
        self._press_pos = event.pos()
        self._last_pos = event.pos()
        self._drag_button = event.button()
        self._measure_nav = False
        state = self.state
        mods = event.modifiers()
        if event.button() == QtCore.Qt.LeftButton:
            if mods & QtCore.Qt.ShiftModifier:
                self._drag_mode = "pan"
            elif mods & QtCore.Qt.AltModifier:
                self._drag_mode = "zoom"
            elif (mods & QtCore.Qt.ControlModifier) or state.rotate_around_z:
                self._drag_mode = "rotate_z"
            elif state.measures.pending is not None:
                self._drag_mode = "measure"
            else:
                self._drag_mode = "rotate"
        elif event.button() == QtCore.Qt.MiddleButton:
            self._drag_mode = "pan"
        elif event.button() == QtCore.Qt.RightButton:
            self._drag_mode = "pan"          # 右键拖动平移；单击仍弹上下文菜单
        else:
            self._drag_mode = None
        self._interactive = self._drag_mode is not None
        if self._interactive:
            # 拖拽期间固定用快帧，别让精修计时器插进来抢时间
            self._quick = True
            self._refine_timer.stop()
        self._update_cursor()

    @staticmethod
    def _soft_step(value: float, cap: float) -> float:
        """鼠标位移限幅 + 非线性映射。

        单个事件可能给出上百像素（高刷新率鼠标、触控板、系统延迟后的补发），
        不夹住就会一步转/缩过头。小幅位移增益更低，方便对准细节；
        大幅位移正常增益但仍受 cap 限制，整体手感「灵活而不过冲」。
        """
        v = float(np.clip(value, -cap, cap))
        gain = 0.55 + 0.45 * min(1.0, abs(v) / 20.0)
        return v * gain

    def mouseMoveEvent(self, event) -> None:
        pos = event.pos()
        state = self.state
        # 测量模式下的悬停预览：即便没有按下鼠标也实时更新「橡皮筋」，让用户
        # 清楚下一个点会落在屏幕视图的什么位置（任意位置都能拾取）。
        if state.measures.pending is not None and self._drag_button is None:
            self._hover_measure = pos
            # 先用视图平面快速算一个预览点（不拾取，移动中零开销），
            # 停手 90ms 后再用模型真实深度修正 —— 预览半径与最终数值一致。
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
        dx = pos.x() - self._last_pos.x()
        dy = pos.y() - self._last_pos.y()
        self._last_pos = pos
        total = float(np.hypot(pos.x() - self._press_pos.x(),
                               pos.y() - self._press_pos.y()))
        state = self.state
        settings = state.settings
        mode = self._drag_mode
        if mode == "measure" and total > self.MEASURE_CLICK_TOL:
            mode = self._drag_mode = "rotate"   # 只有明显拖动才转为旋转
            self._measure_nav = True            # 本次按压不再算「取点」
            self._hover_measure = None
            self._hover_point = None
            self._hover_pick_timer.stop()
            self._update_cursor()
        # 拖拽过程中按住 Shift 进入精细模式（不影响「按下时 Shift=平移」）
        fine = bool(event.modifiers() & QtCore.Qt.ShiftModifier)
        multiplier = self.FINE_MODIFIER_FACTOR if fine else 1.0

        # 所有视图操作统一以画布中心为原点（支点 = camera.pivot）：
        # 旋转绕支点、缩放以支点为锚点、平移严格跟随光标且平移后支点仍
        # 落在画布中心。这样不会出现「绕世界原点转 / 缩放时模型跑掉」。
        cam = state.camera
        if mode == "pan":
            # 平移严格 1:1 跟手，只做跳变限幅，不加非线性增益
            px = float(np.clip(dx, -self.MAX_PAN_STEP, self.MAX_PAN_STEP))
            py = float(np.clip(dy, -self.MAX_PAN_STEP, self.MAX_PAN_STEP))
            cam.pan_about_center(px, py, settings.pan_speed * multiplier)
        else:
            sx = self._soft_step(dx, self.MAX_MOUSE_STEP) * multiplier
            sy = self._soft_step(dy, self.MAX_MOUSE_STEP) * multiplier
            if mode == "rotate":
                if settings.view.fixed_on_screen:
                    cam.rotate_around_z(sx, settings.rotate_speed)
                else:
                    cam.orbit_about_center(sx, -sy, settings.rotate_speed)
            elif mode == "rotate_z":
                cam.rotate_around_z(sx, settings.rotate_speed)
            elif mode == "zoom":
                # Alt+左键上下拖动缩放，锚点固定为画布中心。
                # 旧系数 0.12 档/像素：拖 100px 就是 12 档 ≈ 3 倍缩放，
                # 一拖就冲到极限；现在 100px 约 2 档（≈15%）。
                cam.dolly_about_center(-sy * self.ZOOM_DRAG_PER_PX,
                                       settings.zoom_speed)
            else:
                return
        self.mark_dirty()
        self.statusMessage.emit(self._status_text())

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
        self.mark_dirty()        # 先出快帧，随后自动精修
        # 取点后由 _add_measure_point 自己给「下一点」提示，别被就绪文本冲掉
        if not took_point or state.measures.pending is None:
            self.statusMessage.emit(self._status_text())

    def wheelEvent(self, event) -> None:
        # 支持鼠标滚轮（angleDelta）与触控板（pixelDelta）两种滚动源
        pd = event.pixelDelta()
        if pd is not None and pd.y() != 0:
            steps = pd.y() / 60.0
        else:
            steps = event.angleDelta().y() / 120.0
        if steps:
            # 锚点固定为画布中心，缩放与光标位置无关；触控板一次可能给出
            # 很大的 pixelDelta，限幅避免一档就跳到极限。按住 Shift 为精细缩放。
            fine = self.FINE_MODIFIER_FACTOR if (
                event.modifiers() & QtCore.Qt.ShiftModifier) else 1.0
            self.state.camera.dolly_about_center(
                float(np.clip(steps, -2.0, 2.0)) * fine,
                self.state.settings.zoom_speed)
            self.mark_dirty()
            self.statusMessage.emit(self._status_text())
        event.accept()

    def mouseDoubleClickEvent(self, event) -> None:
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
            cfg = self.state.spin.config
            cfg.enabled = not cfg.enabled
            self.state.spin.reset()
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
        self.statusMessage.emit(self._status_text())

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
        total = sum(o.mesh.triangle_count for o in state.scene.objects)
        if total <= 250_000 and total > 0:
            hit = state.renderer.pick(state.scene, state.camera, x, y)
            if hit is not None:
                self.statusMessage.emit(
                    f"{hit['object'].name}  #{hit['triangle']}  "
                    f"{units.format_point(hit['point'])} {units.label}")
                return
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
        """开启测量工具时先把屏幕投影索引建好，第一次点击就没有额外延迟。"""
        try:
            self._screen_picker.ensure(self.state.scene, self.state.camera)
        except Exception:
            pass

    def _measure_point_at(self, x: float, y: float, pending,
                          pick: bool = True, neighbors: bool = True) -> tuple:
        """把一次点击变成三维测量点，返回 (point, on_surface)。

        关键：**深度取自模型表面，平面位置取自光标**。
        只把光标投影到「过支点、垂直视线」的平面，量到的是屏幕投影长度：
        转个视角同一段 11mm 会变成 9.7mm，镜头一近又会按前后表面的缩放差
        被放大 —— 这正是用户反馈的「尺寸不对」。
        拾取到表面后，把光标射线投到「过该表面点、垂直视线」的平面上：
        点仍然精确落在光标下（所见即所得、标记不跳），深度却是模型的真实
        深度，于是两点距离就是真实尺寸，不再随视角变化。

        光标落在空白处时，退回视图平面（过第一个已取点 / 支点），
        保证「画布任意位置都能取点」这一行为不丢失。
        """
        cam = self.state.camera
        normal = cam.forward()
        ref = pending.points[0] if pending.points else cam.target
        hit = (self._surface_hit(
            x, y, neighbors=neighbors,
            allow_build=getattr(self, "_hover_allow_build", True))
            if pick else None)
        if hit is not None:
            return cam.ray_to_plane(x, y, plane_point=hit,
                                    plane_normal=normal), True
        return cam.ray_to_plane(x, y, plane_point=ref,
                                plane_normal=normal), False

    def _hover_pick(self) -> None:
        """鼠标停住后补一次真实深度拾取（移动中不做，保证拖动不卡）。"""
        state = self.state
        pending = state.measures.pending
        if pending is None or self._drag_button is not None \
                or self._hover_measure is None:
            return
        pos = self._hover_measure
        # 悬停预览复用已有索引；索引失效（用户刚转过视图）就先不重建，
        # 免得停一下鼠标卡一帧 —— 点击时再建，点一定落得下去。
        self._hover_allow_build = False
        try:
            point, snapped = self._measure_point_at(pos.x(), pos.y(), pending,
                                                    neighbors=False)
        finally:
            self._hover_allow_build = True
        self._hover_point = point
        self._hover_snapped = snapped
        if self._overlay is not None:
            self._overlay.update()
        else:
            self.update()

    def _add_measure_point(self, pos: QtCore.QPoint) -> None:
        """在当前屏幕视图平面内取一个测量点。

        画布上 **任意位置**（模型上、模型外、空白背景、空场景）都能取点：
        落在模型上时取真实表面点（量出真实三维尺寸），落在空白处时投影到
        视图平面。两种方式都是纯 CPU 计算（屏幕投影索引 / 几何投影），
        不调用 GL，也不会因为 84 万面模型而卡住点击。
        """
        state = self.state
        pending = state.measures.pending
        if pending is None:
            return
        kind = pending.kind
        x, y = pos.x(), pos.y()
        point, snapped = self._measure_point_at(x, y, pending)
        done = state.measures.add_point(point, units=state.units,
                                        plane=state.camera.forward(),
                                        snapped=snapped)
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
            src = state.tr.tr("measure_on_surface" if snapped
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
            mode = {"rotate": "旋转", "rotate_z": "绕Z旋转",
                    "pan": "平移", "zoom": "缩放"}.get(self._drag_mode, "")
            mode = f"{mode} | " if mode else ""
        return (f"{mode}{stats['objects']} obj | {stats['triangles']} tri | "
                f"{state.last_render_seconds * 1000:.0f} ms | "
                f"{state.fps:.1f} FPS | 支点:画布中心 | {state.units.label}")


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
        self.acts["save_picture"] = self._act(
            m, "save_picture", self.save_picture)
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
                          ("spin_random", "random")):
            act = QtWidgets.QAction(self.state.tr.tr(key), self)
            act.setCheckable(True)
            act.setChecked(self.state.spin.config.kind == kind)
            act.triggered.connect(lambda checked, k=kind: self._set_spin_type(k))
            self._spin_type_group.addAction(act)
            type_menu.addAction(act)
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
        for key in ("rotate_around_z", "orthogonal", "full_screen",
                    "view_options"):
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
        self.object_dock.setWindowTitle(tr.tr("objects"))
        self.panel.refresh()
        self._update_statusbar()

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

    def open_path(self, path: str) -> None:
        tr = self.state.tr
        # 启动已是空场景，直接追加载入；载入后把模型包围盒中心对齐画布中心。
        try:
            obj = self.state.load_file(path)
        except Exception as exc:
            QtWidgets.QMessageBox.critical(
                self, tr.tr("error"), tr.tr("file_open_error", path) + f"\n\n{exc}")
            return
        self.state.settings.default_directory = os.path.dirname(path)
        self.state.fit_view()
        self.schedule_fit()          # 布局稳定后按真实画布尺寸再适配一次
        self.panel.refresh()
        self.canvas.mark_dirty()
        self._recent_menu()
        # 深度统计（连通分量 / 边使用计数）在 84 万面模型上要十几秒，
        # 打开文件时只为显示一行提示，直接用面数即可。
        self._show_status(
            f"{obj.name}: {obj.mesh.triangle_count} tri  "
            f"{stl_io.human_size(os.path.getsize(path))}")

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
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, self.state.tr.tr("export_stl"),
            os.path.join(self.state.settings.default_directory, obj.name + ".stl"),
            "STL (*.stl)")
        if not path:
            return
        if path.lower().endswith(".txt"):
            stl_io.write_ascii_stl(path, obj.mesh.vertices, obj.mesh.triangles, obj.name)
        else:
            stl_io.write_binary_stl(path, obj.mesh.vertices, obj.mesh.triangles)
        self._show_status(f"{self.state.tr.tr('save')}: {path}")

    def _render_picture(self, transparent: bool = False,
                        supersample: int | None = None,
                        double: bool = False) -> np.ndarray:
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

    def _save_picture(self, image: np.ndarray, title: str) -> None:
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, title, os.path.join(self.state.settings.default_directory,
                                      "picture.png"), PICTURE_FILTER)
        if not path:
            return
        if path.lower().endswith(".png"):
            from .png import write_png

            write_png(path, image)
        else:  # 其它格式交给 Qt 编码
            buffer = np.ascontiguousarray(image)
            hh, ww, ch = buffer.shape
            fmt = QtGui.QImage.Format_RGBA8888 if ch == 4 else QtGui.QImage.Format_RGB888
            qimg = QtGui.QImage(buffer.data, ww, hh, buffer.strides[0], fmt)
            qimg.save(path)
        self._show_status(f"{self.state.tr.tr('save_picture')}: {path}")

    def save_picture(self) -> None:
        self._save_picture(self._render_picture(), self.state.tr.tr("save_picture"))

    def save_multi_res_picture(self) -> None:
        self._save_picture(self._render_picture(supersample=3, double=True),
                           self.state.tr.tr("save_multi_res_picture"))

    def save_picture(self, transparent: bool = False) -> None:
        title = self.state.tr.tr("save_picture_transp" if transparent
                                 else "save_picture")
        image = self._render_picture(transparent=transparent, supersample=2)
        self._save_picture(image, title)

    def copy_picture_to_clipboard(self) -> None:
        image = self._render_picture(supersample=2)
        buffer = np.ascontiguousarray(image)
        hh, ww, ch = buffer.shape
        fmt = QtGui.QImage.Format_RGBA8888 if ch == 4 else QtGui.QImage.Format_RGB888
        qimg = QtGui.QImage(buffer.data, ww, hh, buffer.strides[0], fmt).copy()
        QtWidgets.QApplication.clipboard().setImage(qimg)
        self._show_status(self.state.tr.tr("copy_picture_clipboard"))

    def show_file_size(self) -> None:
        path = self.state.current_path
        if not path:
            self._show_status(self.state.tr.tr("no_objects"))
            return
        report = stl_io.file_size_report(path)
        lines = [f"{k}: {v}" for k, v in report.items()]
        StatisticsDialog("\n".join(lines), self.state.tr.tr("show_stl_filesize"),
                         self).exec_()

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

    def _toggle_rotate_z(self, checked: bool) -> None:
        """View > Rotate around Z：左键拖拽改为绕刀具回转轴旋转。"""
        self.state.rotate_around_z = checked
        self._show_status("rotate around Z: " + ("on" if checked else "off"))

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
        self.state.spin.config.enabled = checked
        self.state.spin.reset()
        self.canvas.mark_dirty()

    def _set_spin_type(self, kind: str) -> None:
        self.state.spin.config.kind = kind
        self.state.spin.reset()

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
            state.clipboard_object = copy.deepcopy(obj)
            self._show_status(state.tr.tr("copy_object"))
            return
        elif action == "save_stl":
            idx = state.scene.objects.index(obj)
            state.scene.set_active(idx)
            self.save_active_stl()
            return
        elif action == "repair":
            info = obj.mesh.repair()
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
            return
        elif action == "components":
            parts = obj.mesh.split_components()
            for part in parts:
                state.scene.add_mesh(part, name=part.name, activate=False)
            self.panel.refresh()
            self.canvas.mark_dirty()
            self._show_status(f"{state.tr.tr('generate_components')}: {len(parts)}")
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
            removed = obj.mesh.remove_triangles_intersecting(other.mesh)
            obj.invalidate_colors()
            self.panel.refresh()
            self.canvas.mark_dirty()
            self._show_status(f"removed triangles: {removed}")
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
        self.state.clipboard_object = copy.deepcopy(obj)
        self._show_status(self.state.tr.tr("copy_object"))

    def paste_object(self) -> None:
        src = self.state.clipboard_object
        if src is None:
            self._show_status(self.state.tr.tr("no_objects"))
            return
        obj = Object3D(mesh=src.mesh.copy(), name=src.name + "_copy"
                       if hasattr(src, "name") else "copy")
        self.state.scene.objects.append(obj)
        self.state.scene.set_active(len(self.state.scene.objects) - 1)
        self.panel.refresh()
        self.canvas.mark_dirty()

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
        # 避免点一下「Statistics」就把界面冻住。
        deep = obj.mesh.triangle_count <= 200_000
        text = obj.mesh.statistics(deep=deep).as_text()
        if not deep:
            text += "\n\n(大模型：已跳过连通分量/边界边等深度统计，避免界面卡顿)"
        if obj.source_path and os.path.exists(obj.source_path):
            report = stl_io.file_size_report(obj.source_path)
            text += f"\n\n文件: {obj.source_path}\n大小: {report['human']}"
        StatisticsDialog(text, self.state.tr.tr("statistics_title"), self).exec_()

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

    def show_about(self) -> None:
        tr = self.state.tr
        from . import __version__

        QtWidgets.QMessageBox.about(
            self, tr.tr("about"),
            tr.tr("about_text") + f"\n\n{tr.tr('version')}: {__version__}\n"
            f"Python: {sys.version.split()[0]}")

    # ---------------- 关闭 ----------------
    def closeEvent(self, event) -> None:
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
