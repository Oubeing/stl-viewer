"""轨道相机 —— View 菜单（Zoom / Rotate around Z / Rotation Angles /
Orthogonal / Reset center / Change center / Fixed on screen）。

世界坐标：Z 为刀具回转轴，Z 向上。
相机由 目标点 + 方位角 yaw + 俯仰角 pitch + 滚转 roll + 距离 distance 决定，
支持透视与正交两种投影。
"""

from __future__ import annotations

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
    # 平移结束后支点依旧落在画布中心。GUI 统一调用下面三个 *_about_center
    # 包装方法，避免再出现「绕世界原点或某个模型点转」的偏移支点。
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
        self.yaw -= dx_pixels * speed

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
        使锚点下的世界点保持不动。查看器默认不用这个分支，因为视图操作
        统一要求以画布中心为原点。
        """
        before = self.ray_to_plane(*anchor) if anchor is not None else None
        self.dolly(steps, rate)
        if before is not None:
            after = self.ray_to_plane(*anchor)
            if after is not None:
                self.target = self.target + (before - after)

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
            origin = self.target + self.right() * ndc_x * w + self.up() * ndc_y * h
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
