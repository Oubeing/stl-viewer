"""轨道相机 —— View 菜单（Zoom / 旋转轴 X·Y·Z·视线 / Rotation Angles /
Orthogonal / Reset center / Change center / Fixed on screen）。

世界坐标：Z 为刀具回转轴，Z 向上。
相机由 目标点 + 方位角 yaw + 俯仰角 pitch + 滚转 roll + 距离 distance 决定，
支持透视与正交两种投影。
"""

from __future__ import annotations

import math

import numpy as np

from . import mathutil as mu

WORLD_UP = np.array([0.0, 0.0, 1.0])

# 标准视角预设：(yaw_deg, pitch_deg)
STANDARD_VIEWS = {
    "iso": (45.0, 30.0),
    "iso_back": (-135.0, 30.0),
    "front": (-90.0, 0.0),
    "back": (90.0, 0.0),
    "left": (180.0, 0.0),
    "right": (0.0, 0.0),
    "top": (-90.0, 89.9),
    "bottom": (-90.0, -89.9),
}


class Camera:
    """轨道相机。"""

    def __init__(self, width: int = 1024, height: int = 768):
        self.width = int(width)
        self.height = int(height)
        self.target = np.zeros(3, dtype=np.float64)
        self.yaw = 45.0          # 绕 Z 轴方位角（度）
        self.pitch = 30.0        # 俯仰角（度）
        self.roll = 0.0          # 绕视线滚转（度）
        self.distance = 200.0
        self.fov = 30.0          # 竖直视场角（度）
        self.orthographic = False
        self.ortho_height = 1.0  # 正交视口世界高度
        self.zoom = 1.0          # 附加缩放系数（>1 放大）
        self.near = 0.1
        self.far = 10000.0
        self._model_radius = 100.0

    # ---------------- 尺寸 ----------------
    def set_viewport(self, width: int, height: int) -> None:
        self.width = max(int(width), 1)
        self.height = max(int(height), 1)

    @property
    def aspect(self) -> float:
        return self.width / max(self.height, 1)

    # ---------------- 朝向 ----------------
    def offset_vector(self) -> np.ndarray:
        """从目标点指向相机的单位向量。"""
        y = np.radians(self.yaw)
        p = np.radians(np.clip(self.pitch, -89.99, 89.99))
        return np.array([np.cos(p) * np.cos(y), np.cos(p) * np.sin(y), np.sin(p)])

    def eye(self) -> np.ndarray:
        return self.target + self.offset_vector() * self.distance

    def forward(self) -> np.ndarray:
        return -self.offset_vector()

    def up(self) -> np.ndarray:
        """视图上方向（与视线严格正交，和 look_at 的 u 完全一致）。

        此前直接返回 WORLD_UP：俯视时它带有沿视线的分量，于是 pan 的竖直
        位移、ray 的屏幕方向、pan 的世界/像素换算都偏离投影矩阵，表现为
        「平移不跟手 / 拾取点偏移 / 视图操作不像以画布中心为原点」。
        """
        f = self.forward()
        up = WORLD_UP
        if abs(float(np.dot(f, up))) > 0.9999:
            up = np.array([0.0, 1.0, 0.0])
        if abs(self.roll) > 1e-9:
            up = mu.rotation_matrix(f, np.radians(self.roll)) @ up
        return mu.normalize(up - f * float(np.dot(up, f)))

    def right(self) -> np.ndarray:
        return mu.normalize(np.cross(self.forward(), self.up()))

    def distance_range(self) -> tuple:
        """(near, far)，随模型尺寸自适应。"""
        r = max(self._model_radius, 1e-6)
        near = max(self.distance * 0.002, r * 1e-4, 1e-6)
        far = self.distance + r * 8.0 + 10.0
        if self.orthographic:
            near = max(self.distance - r * 4.0, 1e-6)
            far = self.distance + r * 4.0 + 10.0
        return near, far

    # ---------------- 矩阵 ----------------
    def view_matrix(self) -> np.ndarray:
        return mu.look_at(self.eye(), self.target, self.up())

    def projection_matrix(self) -> np.ndarray:
        near, far = self.distance_range()
        if self.orthographic:
            h = self.visible_height()
            return mu.orthographic(h * self.aspect, h, near, far)
        # zoom 必须真正进入透视投影：此前 visible_height()/world_per_pixel()
        # 除以了 zoom，而投影矩阵没有，导致缩放按钮不改变画面、平移不跟手、
        # ray()/拾取与实际画面不一致。
        return mu.perspective(
            2.0 * np.degrees(np.arctan(self.tan_half_fov())),
            self.aspect, near, far)

    def tan_half_fov(self) -> float:
        """竖直半视场角的正切（含 zoom），投影与屏幕射线共用同一口径。"""
        return float(np.tan(np.radians(self.fov) * 0.5) / max(self.zoom, 1e-6))

    def visible_height(self) -> float:
        """视口上下边缘在世界中的高度（正交模式下含缩放）。"""
        if self.orthographic:
            return max(self.ortho_height / max(self.zoom, 1e-6), 1e-9)
        return 2.0 * self.distance * np.tan(np.radians(self.fov) * 0.5) / max(self.zoom, 1e-6)

    def world_per_pixel(self) -> float:
        return self.visible_height() / max(self.height, 1)

    # ---------------- 投影 ----------------
    def project(self, points: np.ndarray) -> tuple:
        """世界点 -> 屏幕坐标。

        返回 (screen_xy (N,2) float, depth (N,), valid (N,) bool)。
        screen 原点在左上角，depth 为 NDC 深度（越小越近）。
        """
        pts = np.asarray(points, dtype=np.float64)
        if pts.size == 0:
            return np.zeros((0, 2)), np.zeros(0), np.zeros(0, dtype=bool)
        view = self.view_matrix()
        cam = mu.transform_points(view, pts)
        proj = self.projection_matrix()
        # 手工计算齐次裁剪坐标，保留 w（投影矩阵第 4 行不是单位行，
        # 不能用 transform_points 那种丢掉 w 的 3x3 变换）
        h = np.concatenate([cam, np.ones((cam.shape[0], 1))], axis=1)
        clip4 = h @ proj.T
        w = clip4[:, 3]
        w = np.where(np.abs(w) < 1e-12, 1e-12, w)
        ndc = clip4[:, :3] / w[:, None]
        screen = np.empty((pts.shape[0], 2))
        screen[:, 0] = (ndc[:, 0] * 0.5 + 0.5) * self.width
        screen[:, 1] = (0.5 - ndc[:, 1] * 0.5) * self.height
        valid = np.isfinite(ndc).all(axis=1) & (w > 0) if not self.orthographic \
            else np.isfinite(ndc).all(axis=1)
        return screen, ndc[:, 2], valid

    def project_camera_space(self, cam_points: np.ndarray) -> tuple:
        """相机空间点 -> 屏幕坐标（渲染器内部使用，避免重复变换）。"""
        cam = np.asarray(cam_points, dtype=np.float64)
        proj = self.projection_matrix()
        h = np.concatenate([cam, np.ones((cam.shape[0], 1))], axis=1)
        clip = h @ proj.T
        w = clip[:, 3]
        w = np.where(np.abs(w) < 1e-12, 1e-12, w)
        ndc = clip[:, :3] / w[:, None]
        screen = np.empty((cam.shape[0], 2))
        screen[:, 0] = (ndc[:, 0] * 0.5 + 0.5) * self.width
        screen[:, 1] = (0.5 - ndc[:, 1] * 0.5) * self.height
        return screen, ndc[:, 2], ndc[:, :3]

    def to_camera_space(self, points: np.ndarray) -> np.ndarray:
        return mu.transform_points(self.view_matrix(), np.asarray(points, dtype=np.float64))

    # ---------------- 以画布中心为支点的交互 ----------------
    # 轨道相机的 target 就是「视图支点」：look_at(eye, target, up) 保证它
    # 恒投影在画布正中心。旋转只改变 eye 方向、缩放只改变 distance /
    # ortho_height，两者都天然以画布中心为原点；平移把支点连同模型一起移动，
    # 平移结束后支点依旧落在画布中心。下面三个 *_about_center 包装方法保留给
    # 菜单 / 键盘 / CLI 等调用方；GUI 的拖拽交互改走 navigation.py（光标锚点
    # 缩放与抓取式平移），两者共用同一套投影口径，不会绕世界原点转。
    @property
    def pivot(self) -> np.ndarray:
        """视图支点（世界坐标），恒位于画布中心。"""
        return self.target

    def center_world_point(self) -> np.ndarray:
        """画布中心对应的世界点副本。"""
        return self.target.copy()

    # 灵敏度集中在这里的默认值上，GUI 会用「视图选项」里的设置覆盖它们。
    # 早期 orbit 固定 0.35 度/像素：拖 300px 就转 105°，一次随手拖动模型就
    # 跑掉了；再配合 GUI 的单事件限幅与小幅低增益映射，现在既灵活又不过冲。
    ROTATE_SPEED = 0.12          # 轨道旋转：度 / 像素
    PAN_SPEED = 1.0              # 平移：1.0 表示严格跟随光标
    DOLLY_RATE = 0.07            # 滚轮：每档 (120°) 的指数缩放系数

    def orbit(self, dx_pixels: float, dy_pixels: float,
              speed: float | None = None) -> None:
        """左键拖拽：绕目标点旋转。"""
        speed = self.ROTATE_SPEED if speed is None else float(speed)
        self.yaw -= dx_pixels * speed
        self.pitch = float(np.clip(self.pitch + dy_pixels * speed, -89.9, 89.9))

    def rotate_around_z(self, dx_pixels: float, speed: float | None = None) -> None:
        """Rotate around Z：只改变方位角（沿机床回转轴转）。"""
        speed = self.ROTATE_SPEED if speed is None else float(speed)
        self.rotate_about_axis(-dx_pixels * speed, "z")

    def rotate_about_axis(self, degrees: float, axis: str = "z") -> None:
        """绕指定世界坐标轴（X / Y / Z）或视线轴旋转视角。

        ``z`` 走 yaw 快路径（与旧行为完全一致）；``x`` / ``y`` / ``view``
        把相机朝向绕世界轴旋转后用 ``set_orientation`` 反解 yaw/pitch/roll，
        因此投影、拾取、射线等所有下游代码无需改动。鼠标拖拽、键盘步进、
        旋转动画共用这一入口，三个轴的旋转手感保持一致。
        """
        deg = float(degrees)
        if abs(deg) < 1e-12:
            return
        key = str(axis).lower()
        if key == "z":
            self.yaw += deg
            return
        if key == "x":
            direction = np.array([1.0, 0.0, 0.0])
        elif key == "y":
            direction = np.array([0.0, 1.0, 0.0])
        elif key == "view":
            direction = self.forward()
        else:
            return
        rot = mu.rotation_matrix(direction, np.radians(deg))
        self.set_orientation(rot @ self.offset_vector(), rot @ self.up())

    def set_rotation_angles(self, yaw=None, pitch=None, roll=None) -> None:
        if yaw is not None:
            self.yaw = float(yaw)
        if pitch is not None:
            self.pitch = float(np.clip(pitch, -89.9, 89.9))
        if roll is not None:
            self.roll = float(roll)

    def pan(self, dx_pixels: float, dy_pixels: float,
            factor: float | None = None) -> None:
        """中键拖拽：平移目标点。``factor=1.0`` 时模型严格跟随光标。"""
        factor = self.PAN_SPEED if factor is None else float(factor)
        scale = self.world_per_pixel() * factor
        self.target = self.target - self.right() * dx_pixels * scale + self.up() * dy_pixels * scale

    def dolly(self, steps: float, rate: float | None = None) -> None:
        """滚轮：缩放（改变距离或正交视口）。``steps`` 为滚轮档数。"""
        rate = self.DOLLY_RATE if rate is None else float(rate)
        factor = float(np.exp(-rate * steps))
        if self.orthographic:
            self.ortho_height = float(np.clip(self.ortho_height * factor, 1e-6, 1e9))
        else:
            self.distance = float(np.clip(self.distance * factor, 1e-6, 1e9))

    def orbit_about_center(self, dx_pixels: float, dy_pixels: float,
                           speed: float | None = None) -> None:
        """绕画布中心支点旋转；支点世界坐标在旋转前后保持不变。"""
        self.orbit(dx_pixels, dy_pixels, speed)

    def pan_about_center(self, dx_pixels: float, dy_pixels: float,
                         factor: float | None = None) -> None:
        """平移：模型严格跟随光标，平移后支点重新对齐画布中心。"""
        self.pan(dx_pixels, dy_pixels, factor)

    def dolly_about_center(self, steps: float, rate: float | None = None,
                           anchor: tuple | None = None) -> None:
        """缩放：默认以画布中心为锚点（支点不动）。

        ``anchor=(x, y)`` 时改为以该屏幕点为锚点，缩放后补偿 target，
        使锚点下的世界点保持不动（CAD 软件的 zoom-to-cursor）。
        """
        if anchor is not None:
            self.dolly_at(anchor[0], anchor[1], steps, rate)
        else:
            self.dolly(steps, rate)

    def zoom_by_factor(self, factor: float) -> None:
        self.zoom = float(np.clip(self.zoom * factor, 0.01, 100.0))

    # ---------------- 复位 / 适配 ----------------
    def fit(self, bounds_lo, bounds_hi, margin: float = 1.10,
            keep_orientation: bool = True, points: np.ndarray | None = None) -> None:
        """Zoom Fit：按「当前朝向的实际投影范围」把模型铺满画布。

        旧实现用包围盒外接球半径 ``||hi-lo||/2`` 估算距离，对细长件极不
        准确：HLFΦ0.20-6.5（0.2 × 0.2 × 11）的外接球半径 5.5，
        于是 11 长的模型只占竖直视场 ~70%，切到俯视更是缩成一个十几像素
        的小点 —— 用户看到的「zoom fit 没匹配视图大小，显示很小」。

        现在把包围盒 8 个角点变换到相机基（right / up / offset）下，
        对每个角点解出透视投影不等式，取最大者即恰好让所有角点进入视野
        的最小距离：

            ndc_y = y / (tan_half * (d - z))
            |ndc_y| <= 1/margin  ==>  d >= z + margin * |y| / tan_half
            ndc_x = x / (aspect * tan_half * (d - z))
                                 ==>  d >= z + margin * |x| / (aspect*tan_half)

        这样任意视角、任意画布宽高比下模型都占满约 1/margin 的视口。
        """
        lo = np.asarray(bounds_lo, dtype=np.float64).reshape(3)
        hi = np.asarray(bounds_hi, dtype=np.float64).reshape(3)
        lo, hi = np.minimum(lo, hi), np.maximum(lo, hi)
        center = (lo + hi) * 0.5
        size = hi - lo
        radius = float(np.linalg.norm(size) * 0.5)
        self.target = center
        self._model_radius = max(radius, 1e-6)
        if not keep_orientation:
            self.yaw, self.pitch, self.roll = 45.0, 30.0, 0.0
        self.zoom = 1.0

        # 优先用真实顶点（可抽样）求投影范围：包围盒 8 角点常落在空处，
        # 例如圆柱在等轴测下角点比实体外扩约 20%，fit 出来仍显得偏小。
        if points is not None and len(points) > 0:
            q = np.asarray(points, dtype=np.float64).reshape(-1, 3) - center
        else:
            ix, iy, iz = np.meshgrid((0, 1), (0, 1), (0, 1), indexing="ij")
            corners = np.stack([np.where(ix, hi[0], lo[0]),
                                np.where(iy, hi[1], lo[1]),
                                np.where(iz, hi[2], lo[2])], axis=-1).reshape(-1, 3)
            q = corners - center
        # 相机基：行分别为 right / up / offset(由目标指向相机)
        basis = np.stack([self.right(), self.up(), self.offset_vector()])
        cam = q @ basis.T                       # (8,3) = x, y, z
        z = cam[:, 2]
        tan_half = float(np.tan(np.radians(self.fov) * 0.5))
        aspect = max(self.aspect, 1e-6)
        need = np.maximum(np.abs(cam[:, 1]) / tan_half,
                          np.abs(cam[:, 0]) / (tan_half * aspect))
        margin = max(float(margin), 1.001)
        distance = float(np.max(z + margin * need))
        # 保证所有角点都在近裁剪面之前（留一点余量，避免透视除零）
        distance = max(distance, float(np.max(z)) + max(radius * 0.05, 1e-6),
                       max(radius * 0.02, 1e-6))
        self.distance = distance
        # 正交视口高度：由投影后的 x/y 范围直接给出，与透视结果同视角
        half_h = max(float(np.max(np.abs(cam[:, 1]))),
                     float(np.max(np.abs(cam[:, 0]))) / aspect)
        self.ortho_height = max(2.0 * margin * half_h, 1e-6)
        # 正交模式的 near/far 也要覆盖模型的前后深度
        if self.orthographic:
            self.distance = max(self.distance, radius * 2.0)
        near, far = self.distance_range()
        self.near, self.far = near, far

    def reset_center(self) -> None:
        """Reset center：把目标点移回世界原点。"""
        self.target = np.zeros(3, dtype=np.float64)

    def change_center(self, point) -> None:
        """Change center：把旋转中心移到指定点。"""
        self.target = np.asarray(point, dtype=np.float64)

    def set_standard_view(self, name: str) -> None:
        if name in STANDARD_VIEWS:
            self.yaw, self.pitch = STANDARD_VIEWS[name]
            self.roll = 0.0

    def set_orthographic(self, enabled: bool) -> None:
        """View > Orthogonal 切换；切换时保持视觉尺寸大致连续。"""
        if enabled == self.orthographic:
            return
        if enabled:
            self.ortho_height = self.visible_height()
        else:
            self.distance = max(self.visible_height() /
                                (2.0 * np.tan(np.radians(self.fov) * 0.5)), 1e-6)
        self.orthographic = bool(enabled)

    # ---------------- 拾取 ----------------
    def ray(self, x: float, y: float) -> tuple:
        """屏幕像素 -> 世界射线 (origin, direction)。"""
        ndc_x = (x / max(self.width, 1)) * 2.0 - 1.0
        ndc_y = 1.0 - (y / max(self.height, 1)) * 2.0
        if self.orthographic:
            h = self.visible_height() * 0.5
            w = h * self.aspect
            # 起点必须放在相机所在的近端平面，而不是过支点的平面：
            # 支点可能落在实体内部（例如俯视一个带凸台的零件），从支点
            # 向里发射线只能打到后半部分，拾取到的深度就是错的（顶面
            # 会被当成底面）。沿视线平移起点不改变射线所在的直线，
            # 平面投影 / 缩放锚点等行为完全不变。
            origin = self.eye() + self.right() * ndc_x * w + self.up() * ndc_y * h
            return origin, self.forward()
        tan_half = self.tan_half_fov()
        direction = mu.normalize(
            self.forward()
            + self.right() * ndc_x * tan_half * self.aspect
            + self.up() * ndc_y * tan_half
        )
        return self.eye(), direction

    def ray_to_plane(self, x: float, y: float, plane_point=None,
                     plane_normal=None) -> np.ndarray:
        """屏幕像素投射到平面；默认投射到过目标点且垂直于视线的平面。"""
        origin, direction = self.ray(x, y)
        if plane_point is None:
            plane_point = self.target
        if plane_normal is None:
            plane_normal = self.forward()
        hit = mu.ray_plane_intersection(origin, direction,
                                        np.asarray(plane_point, dtype=np.float64),
                                        mu.normalize(np.asarray(plane_normal, dtype=np.float64)))
        return hit if hit is not None else self.target.copy()

    # ---------------- CAD 风格导航用的低层运算 ----------------
    # 这一组方法是 navigation.py 的基础。它们只改 target / yaw / pitch /
    # roll / distance / ortho_height，不改变任何既有 API 的语义，因此
    # 投影、拾取、fit、动画与导出逻辑完全不受影响。
    def _base_up(self, forward: np.ndarray) -> np.ndarray:
        """与 :meth:`up` 完全一致的「无滚转」基准上方向。"""
        up = WORLD_UP if abs(float(np.dot(forward, WORLD_UP))) <= 0.9999 \
            else np.array([0.0, 1.0, 0.0])
        return mu.normalize(up - forward * float(np.dot(up, forward)))

    def set_orientation(self, back, up) -> None:
        """由一对正交的世界方向反解 yaw / pitch / roll。

        轨迹球旋转会越过极点并产生滚转，此时用「绕世界 Z 的方位角 + 俯仰角」
        已无法表达朝向；这里把任意朝向投影回既有的 yaw/pitch/roll 参数化，
        使 :meth:`up`、:meth:`right`、:meth:`view_matrix`、:meth:`ray` 等
        全部继续工作，无需改动渲染与拾取代码。
        """
        b = mu.normalize(np.asarray(back, dtype=np.float64))
        f = -b
        if float(np.linalg.norm(b)) < 1e-9:
            return
        base_up = self._base_up(f)
        base_right = mu.normalize(np.cross(f, base_up))
        u = np.asarray(up, dtype=np.float64)
        u = mu.normalize(u - f * float(np.dot(u, f)))
        if float(np.linalg.norm(u)) < 1e-9:
            u = base_up
        # up() = R(forward, roll) @ base_up；R 的一阶项为 forward × base_up
        # = base_right，所以 atan2 恰好还原 roll（符号与 up() 一致）。
        roll = np.degrees(np.arctan2(float(np.dot(u, base_right)),
                                    float(np.dot(u, base_up))))
        self.pitch = float(np.clip(np.degrees(np.arcsin(np.clip(b[2], -1.0, 1.0))),
                                   -89.99, 89.99))
        self.yaw = float(np.degrees(np.arctan2(b[1], b[0])))
        self.roll = float(roll)

    def _arcball_vector(self, x: float, y: float) -> np.ndarray:
        """屏幕点 -> 单位球面向量（相机空间：right / up / 指向观察者）。"""
        r = max(min(self.width, self.height), 1) * 0.5
        nx = (float(x) - self.width * 0.5) / r
        # y 与屏幕同向（向下为正）：与转台式旋转的「向右拖 yaw 减小、
        # 向下拖 pitch 减小」保持同一跟手方向。
        ny = (float(y) - self.height * 0.5) / r
        d2 = nx * nx + ny * ny
        if d2 <= 1.0:
            nz = math.sqrt(max(0.0, 1.0 - d2))
        else:
            d = math.sqrt(d2)
            nx, ny, nz = nx / d, ny / d, 0.0
        return np.array([nx, ny, nz], dtype=np.float64)

    def rotate_arcball(self, x0: float, y0: float, x1: float, y1: float,
                       speed: float | None = None) -> None:
        """自由轨迹球旋转（CATIA 式）：可越过极点、可产生滚转。

        旋转轴来自球面上起点→终点的叉积，因此越靠近画布边缘拖动，滚转越
        明显；绕极点时不会像转台式相机那样「卡住」。旋转量仍以
        ``度/像素`` 计，保持与设置里的旋转灵敏度严格成正比。
        """
        speed = self.ROTATE_SPEED if speed is None else float(speed)
        dx, dy = float(x1) - float(x0), float(y1) - float(y0)
        dist = math.hypot(dx, dy)
        if dist < 1e-9:
            return
        v0 = self._arcball_vector(x0, y0)
        v1 = self._arcball_vector(x1, y1)
        axis_c = np.cross(v0, v1)
        if float(np.linalg.norm(axis_c)) < 1e-9:
            return
        # 相机空间基的行向量：[right, up, back]
        basis = np.stack([self.right(), self.up(), self.offset_vector()])
        axis_w = mu.normalize(axis_c @ basis)
        # 与转台式旋转保持同一手感：向右拖时 yaw 减小、向下拖时 pitch 减小。
        angle = np.radians(-dist * speed)
        rot = mu.rotation_matrix(axis_w, float(angle))
        back = self.offset_vector()
        up = self.up()
        self.set_orientation(rot @ back, rot @ up)

    def dolly_at(self, x: float, y: float, steps: float,
                 rate: float | None = None, plane_point=None) -> None:
        """以指定屏幕点为锚点缩放：缩放前后该点下的世界点保持不动。

        旧 ``dolly_about_center(anchor=...)`` 在 dolly 之后重新取 ``target``
        作为参考平面，而 target 已被补偿改动，两次射线落在了不同的平面上，
        补偿量因此偏差。这里把平面点与法向固定下来再求交，透视与正交都精确。
        """
        p0 = (self.target.copy() if plane_point is None
              else np.asarray(plane_point, dtype=np.float64))
        n0 = self.forward().copy()
        before = mu.ray_plane_intersection(*self.ray(x, y), p0, n0)
        self.dolly(steps, rate)
        if before is None:
            return
        after = mu.ray_plane_intersection(*self.ray(x, y), p0, n0)
        if after is not None:
            self.target = self.target + (before - after)

    def pan_grab(self, x: float, y: float, plane_point=None) -> tuple:
        """开始抓取式平移，返回不随视图变化的参考系 (世界点, 平面点, 法向)。

        CAD 软件的平移手感来自「抓住光标下的那个点」：拖动过程中该点始终
        贴在光标上，透视下近处/远处的模型也不会漂。
        """
        p0 = (self.target.copy() if plane_point is None
              else np.asarray(plane_point, dtype=np.float64))
        n0 = self.forward().copy()
        hit = mu.ray_plane_intersection(*self.ray(x, y), p0, n0)
        world = p0.copy() if hit is None else np.asarray(hit, dtype=np.float64)
        return world, p0, n0

    def pan_drag_to(self, x: float, y: float, grab: tuple,
                    factor: float = 1.0) -> None:
        """把 :meth:`pan_grab` 抓住的点移动到当前光标位置。"""
        world, p0, n0 = grab
        hit = mu.ray_plane_intersection(*self.ray(x, y), p0, n0)
        if hit is None:
            return
        delta = (world - np.asarray(hit, dtype=np.float64)) * float(factor)
        self.target = self.target + delta

    def reframe(self, point) -> None:
        """把指定世界点移到画布中心，作为新的旋转支点（Center on point）。"""
        self.target = np.asarray(point, dtype=np.float64).reshape(3).copy()

    # ---------------- 序列化 ----------------
    def state(self) -> dict:
        return {
            "target": self.target.tolist(),
            "yaw": self.yaw,
            "pitch": self.pitch,
            "roll": self.roll,
            "distance": self.distance,
            "fov": self.fov,
            "orthographic": self.orthographic,
            "ortho_height": self.ortho_height,
            "zoom": self.zoom,
        }

    def set_state(self, state: dict) -> None:
        if not state:
            return
        if "target" in state:
            self.target = np.asarray(state["target"], dtype=np.float64)
        for key in ("yaw", "pitch", "roll", "distance", "fov", "ortho_height", "zoom"):
            if key in state:
                setattr(self, key, float(state[key]))
        if "orthographic" in state:
            self.orthographic = bool(state["orthographic"])

    def copy(self) -> "Camera":
        cam = Camera(self.width, self.height)
        cam.set_state(self.state())
        # _model_radius 不参与序列化，但决定 near/far；后台渲染快照必须带上，
        # 否则与主线程同步渲染的裁剪范围不同，画面会出现莫名裁切。
        cam._model_radius = self._model_radius
        return cam
