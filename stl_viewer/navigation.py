"""CAD 风格视图导航 —— 纯计算层，零 Qt / 零 GUI 依赖。

把「哪个键做什么」与「相机怎么动」从 :mod:`viewer` 里拆出来，目的有两个：

* 对齐 CATIA / SolidWorks 等 CAD 软件的导航习惯 —— 中键组合键、
  光标锚点缩放（zoom to cursor）、抓取式平移（抓住光标下那个点）、
  可越过极点的自由轨迹球旋转、绕指定点旋转、释放后的惯性衰减。
* 所有交互数学都能在只装 numpy 的环境里做单元测试，不依赖桌面。

模块只接受抽象的按键位标记（``BTN_*``）与修饰键字符串，Qt 枚举到这些
标记的换算留在 viewer.py，因此这里可以被 CLI、脚本与测试直接复用。
"""

from __future__ import annotations

import math
from dataclasses import dataclass

# 按键位标记（与 Qt 的 Button 枚举无关）
BTN_LEFT = 1
BTN_MIDDLE = 2
BTN_RIGHT = 4

ROTATE = "rotate"
ROTATE_Z = "rotate_z"
PAN = "pan"
ZOOM = "zoom"
NONE = "none"

#: 定轴旋转可选轴：free = 常规转台/轨迹球；x/y/z = 世界坐标轴；view = 视线滚转
ROTATE_AXES = ("free", "x", "y", "z", "view")
AXIS_LABELS = {"free": "自由", "x": "X", "y": "Y", "z": "Z", "view": "视线"}

#: 可选导航方案。``cad`` 为默认值：融合两种 CAD 的常见习惯，同时保留
#: 本项目原有的左键旋转 / 右键平移 / Alt+左键缩放等快捷方式。
NAV_STYLES = ("classic", "cad", "catia", "solidworks")
ROTATE_MODES = ("turntable", "arcball")
ZOOM_ANCHORS = ("cursor", "center")
PIVOT_MODES = ("center", "cursor")

ACTION_LABELS = {
    ROTATE: "旋转", ROTATE_Z: "绕Z旋转", PAN: "平移", ZOOM: "缩放",
}

# 与旧版相比放宽很多：高刷新率鼠标 / 触控板一次补发上百像素是常态，
# 夹到 60px 会让快速拖动「丢距离」，正是 CAD 里最明显的迟滞感来源。
MAX_ROTATE_STEP = 200.0
MAX_PAN_STEP = 300.0
MAX_WHEEL_STEPS = 4.0
ZOOM_DRAG_PER_PX = 0.020     # Alt+左键每像素折算的滚轮档数
FINE_FACTOR = 0.25           # 拖拽中按住 Shift = 精细模式


@dataclass
class NavOptions:
    """可持久化的导航偏好（对应「视图选项」对话框）。"""

    style: str = "cad"
    rotate_mode: str = "turntable"   # turntable: Z 朝上的转台；arcball: 自由轨迹球
    zoom_anchor: str = "cursor"      # cursor: 缩放时保持光标下的点不动
    pivot: str = "center"            # center: 绕画布中心；cursor: 绕按下点
    inertia: bool = False            # 释放后短暂惯性旋转
    inertia_halflife: float = 0.12   # 惯性速度半衰期（秒）

    def normalized(self) -> "NavOptions":
        """把外部（INI / 测试）传入的取值收敛到合法集合。"""
        self.style = self.style if self.style in NAV_STYLES else "cad"
        self.rotate_mode = (self.rotate_mode if self.rotate_mode in ROTATE_MODES
                            else "turntable")
        self.zoom_anchor = (self.zoom_anchor if self.zoom_anchor in ZOOM_ANCHORS
                            else "cursor")
        self.pivot = self.pivot if self.pivot in PIVOT_MODES else "center"
        self.inertia = bool(self.inertia)
        self.inertia_halflife = float(min(max(self.inertia_halflife, 0.02), 1.0))
        return self


def resolve_action(button: int, buttons: int, mods, style: str = "cad") -> str:
    """把「当前按下的键 + 已按住的键 + 修饰键」映射为导航动作。

    ``button`` 是本次触发的键，``buttons`` 是当前所有按下键的位或；两者
    分开才能识别 CATIA 的「中键 + 左/右键 = 旋转」这类组合键。
    ``mods`` 是 ``{"shift", "ctrl", "alt"}`` 的任意可迭代集合。
    """
    mods = {str(m).lower() for m in (mods or ())}
    shift, ctrl, alt = "shift" in mods, "ctrl" in mods, "alt" in mods
    combo = bool(buttons & ~(button or 0))       # 已有其它鼠标键按住

    if button & BTN_LEFT:
        if alt:
            return ZOOM
        if ctrl:
            return ROTATE_Z
        if shift:
            return PAN
        # 后按下左键时，若中键已按住：CATIA 的中键+左键 = 旋转
        if style in ("catia", "cad") and (buttons & BTN_MIDDLE):
            return ROTATE
        return ROTATE

    if button & BTN_MIDDLE:
        if style == "solidworks":
            if ctrl:
                return PAN
            if shift:
                return ZOOM
            return ROTATE
        if style in ("catia", "cad"):
            if buttons & (BTN_LEFT | BTN_RIGHT):
                return ROTATE
            if ctrl or shift:
                return ZOOM
            return PAN
        return PAN                                # classic

    if button & BTN_RIGHT:
        if style in ("catia", "cad") and (buttons & BTN_MIDDLE):
            return ROTATE
        return PAN

    if combo:                                     # 其它组合键按平移兜底
        return PAN
    return NONE


def limit_step(value: float, cap: float) -> float:
    """只做跳变限幅，保持线性。

    旧版对小幅位移额外乘 0.55~1.0 的增益，慢速微调时的位移被系统性压缩，
    手感发「黏」。CAD 软件里旋转量与鼠标位移严格成正比，这里对齐这一点。
    """
    return float(max(-cap, min(cap, float(value))))


class NavigationController:
    """一次视图交互会话的状态机（按下 → 拖动 → 释放 / 惯性）。

    ``camera`` 是运行中的 :class:`~stl_viewer.camera.Camera` 实例，控制器
    直接就地修改它；``speeds`` 用 ``rotate`` / ``pan`` / ``zoom`` 三个键接受
    ``AppSettings`` 里的灵敏度，未提供的项走 Camera 的默认常量。
    """

    def __init__(self, camera, options: NavOptions | None = None,
                 speeds: dict | None = None):
        self.camera = camera
        self.options = (options or NavOptions()).normalized()
        self.speeds = dict(speeds or {})
        self.action = NONE
        self.active = False
        self.start = (0.0, 0.0)
        self.last = (0.0, 0.0)
        self._grab = None
        self._velocity = (0.0, 0.0)
        # Ctrl+左键 / 「固定轴旋转」使用的轴，默认 Z（保持旧习惯）
        self.axis = "z"

    # ---------------- 配置 ----------------
    def configure(self, options: NavOptions | None = None,
                  speeds: dict | None = None, axis: str | None = None) -> None:
        if options is not None:
            self.options = options.normalized()
        if speeds is not None:
            self.speeds.update(speeds)
        if axis is not None:
            self.set_axis(axis)

    def set_axis(self, axis: str) -> str:
        axis = str(axis).lower()
        self.axis = axis if axis in ROTATE_AXES else "free"
        return self.axis

    def _speed(self, key: str, default: float) -> float:
        value = self.speeds.get(key, default)
        try:
            return max(float(value), 1e-6)
        except (TypeError, ValueError):
            return default

    @property
    def label(self) -> str:
        return ACTION_LABELS.get(self.action, "")

    # ---------------- 会话 ----------------
    def begin(self, action: str, x: float, y: float) -> None:
        """开始一次拖拽：建立抓取平移参考系 / 绕点旋转支点。"""
        self.action = action if action != NONE else NONE
        self.active = self.action != NONE
        self.start = self.last = (float(x), float(y))
        self._grab = None
        self._velocity = (0.0, 0.0)
        if not self.active:
            return
        cam = self.camera
        if self.action == PAN:
            # 抓住光标下的世界点；后续每一步都把这个点贴回光标。
            self._grab = cam.pan_grab(x, y)
        elif self.action in (ROTATE, ROTATE_Z) and self.options.pivot == "cursor":
            # Center on point：把按下点移到画布中心再绕它旋转。
            cam.reframe(cam.ray_to_plane(x, y))

    def drag(self, x: float, y: float, mods=()) -> bool:
        """处理一次鼠标移动，返回相机是否发生变化。"""
        if not self.active:
            return False
        x, y = float(x), float(y)
        px, py = self.last
        dx, dy = x - px, y - py
        self.last = (x, y)
        if abs(dx) < 1e-9 and abs(dy) < 1e-9:
            return False
        mods = {str(m).lower() for m in (mods or ())}
        fine = FINE_FACTOR if "shift" in mods else 1.0
        cam = self.camera
        action = self.action

        if action == PAN:
            # 抓取式平移：与「每事件位移」无关，直接用光标的绝对位置，
            # 因此高刷新率下也不会累积误差，透视深度同样精确。
            # 倍率作用于「相对按下点的位移」而不是每步的增量，精细模式
            # 才能稳定停在跟手的位置，不会逐事件指数逼近。
            factor = self._speed("pan", cam.PAN_SPEED) * fine
            tx = self.start[0] + (x - self.start[0]) * factor
            ty = self.start[1] + (y - self.start[1]) * factor
            cam.pan_drag_to(tx, ty, self._grab)
            return True

        sx = limit_step(dx, MAX_ROTATE_STEP) * fine
        sy = limit_step(dy, MAX_ROTATE_STEP) * fine
        if action == ZOOM:
            cam.dolly_at(self.start[0], self.start[1],
                         -sy * ZOOM_DRAG_PER_PX,
                         self._speed("zoom", cam.DOLLY_RATE))
        elif action == ROTATE_Z:
            cam.rotate_about_axis(-sx * self._speed("rotate", cam.ROTATE_SPEED),
                                  self.axis)
        elif action == ROTATE:
            if self.options.rotate_mode == "arcball":
                cam.rotate_arcball(px, py, x, y,
                                   self._speed("rotate", cam.ROTATE_SPEED) * fine)
            else:
                cam.orbit_about_center(sx, -sy,
                                       self._speed("rotate", cam.ROTATE_SPEED))
            self._velocity = (sx, sy)
        else:
            return False
        return True

    def wheel(self, x: float, y: float, steps: float, mods=()) -> bool:
        """滚轮 / 触控板缩放；默认以光标为锚点。"""
        steps = limit_step(steps, MAX_WHEEL_STEPS)
        if abs(steps) < 1e-9:
            return False
        mods = {str(m).lower() for m in (mods or ())}
        if "shift" in mods:
            steps *= FINE_FACTOR
        cam = self.camera
        if self.options.zoom_anchor == "cursor":
            cam.dolly_at(x, y, steps, self._speed("zoom", cam.DOLLY_RATE))
        else:
            cam.dolly_about_center(steps, self._speed("zoom", cam.DOLLY_RATE))
        return True

    def end(self) -> bool:
        """结束拖拽，返回是否还有惯性需要继续驱动。"""
        active = self.active
        self.active = False
        keep_spin = (active and self.options.inertia
                     and self.action == ROTATE
                     and math.hypot(*self._velocity) > 0.5)
        if not keep_spin:
            self._velocity = (0.0, 0.0)
        self.action = NONE
        self._grab = None
        return keep_spin

    def advance(self, dt: float) -> bool:
        """惯性衰减一步；返回是否仍需要下一帧。

        ``dt`` 为距上一帧的秒数。半衰期衰减与帧率无关，60Hz / 144Hz 下的
        总滑行距离一致；速度低于 0.5 px/frame 即停止。
        """
        vx, vy = self._velocity
        if math.hypot(vx, vy) < 0.5:
            self._velocity = (0.0, 0.0)
            return False
        cam = self.camera
        if self.options.rotate_mode == "arcball":
            cx, cy = cam.width * 0.5, cam.height * 0.5
            cam.rotate_arcball(cx, cy, cx + vx, cy + vy,
                               self._speed("rotate", cam.ROTATE_SPEED))
        else:
            cam.orbit_about_center(vx, -vy,
                                   self._speed("rotate", cam.ROTATE_SPEED))
        decay = 0.5 ** (max(float(dt), 0.0) / self.options.inertia_halflife)
        self._velocity = (vx * decay, vy * decay)
        return math.hypot(*self._velocity) >= 0.5

    # ---------------- 供状态栏 / 文档使用 ----------------
    @staticmethod
    def summary(options: NavOptions) -> str:
        labels = {"classic": "经典", "cad": "CAD", "catia": "CATIA",
                  "solidworks": "SolidWorks"}
        return (f"{labels.get(options.style, options.style)} | "
                f"{'轨迹球' if options.rotate_mode == 'arcball' else '转台'} | "
                f"{'光标' if options.zoom_anchor == 'cursor' else '中心'}锚点")
