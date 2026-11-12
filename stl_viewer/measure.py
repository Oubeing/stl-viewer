"""测量工具 —— Measurement 菜单：

    Measure 2D Line / Measure 3D Line / Measure 3D Point /
    Measure 2 Point Circle / Measure 3 Point Circle /
    Measure 2D Angle / Measure RZ Point /
    Delete first / Delete last / Delete All Measures / Measure color

长度口径与 CAD 的「在视图里量」一致：

* 一次测量里的所有点都落在**同一个视图平面**（过第一点、垂直于开始测量时
  的视线）上。因此两点距离就是**屏幕投影长度**，与取点落在模型的哪一层
  深度无关 —— 在 TOP 视图量一条边，不会因为一端在凸台顶面、另一端在底面
  而多算进一个深度差。
* 两点测圆 / 三点圆同样在这个视图平面内求解，圆的读数与画面上看到的严格
  一致（这也是用户确认「圆是准的」的原因）。
* 真实三维距离并没有丢：表面点单独保存，状态栏与结果里以 ``3D=`` 给出；
  需要以三维距离为主读数时，打开 Measurement > True 3D Distance（在轴测
  视图里取点更有意义）。

RZ 点是磨削行业的专用测量：把空间点换算为柱坐标 (R 半径, Z 轴向, φ 角)，
用于描述砂轮/刀具回转体上的位置。
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from . import mathutil as mu

# 测量类型定义： (需要点数, i18n 键, 是否需要参考平面)
MEASURE_DEFS = {
    "point3d": (1, "measure_3d_point", False),
    "line3d": (2, "measure_3d_line", False),
    "line2d": (2, "measure_2d_line", True),
    "circle2p": (2, "measure_2_point_circle", True),
    "circle3p": (3, "measure_3_point_circle", False),
    "angle2d": (3, "measure_2d_angle", True),
    "rzpoint": (1, "measure_rz_point", False),
}

MEASURE_ORDER = list(MEASURE_DEFS.keys())

#: 长度口径：view = 视图投影（默认，CAD 的「在视图里量」）；3d = 真实三维距离
MEASURE_MODES = ("view", "3d")

#: 这些测量天然描述「模型上的点」，读数优先取真实表面点。
#: 注意：三点圆**不在此列** —— 它是平面测量，三个点必须落在同一个视图
#: 平面上。若改用带深度差的表面点，拟合出的圆会躺在倾斜的空间平面上，
#: 投影到屏幕就成了椭圆（用户看到的「三点圆不圆」），半径也不再是投影值。
_SURFACE_FIRST_KINDS = ("point3d", "rzpoint")

#: 只在 3D 模式下改用真实表面点的测量；2D 直线始终是投影长度
_MODE_AWARE_KINDS = ("line3d", "angle2d")


@dataclass
class Measure:
    """一次测量。

    ``points`` 是**视图平面点**（一次测量内严格共面）；``surface_points``
    保存每个点对应的真实模型表面点（空白处取点为 None）。前者用于投影
    读数与绘制，后者只用于可选的 3D 读数与 3D 模式 —— 两者分开保存，
    才能同时做到「视图里量的是投影长度」和「3D 模式量的是真实距离」。
    """

    kind: str
    color: tuple = (1.0, 0.86, 0.32)
    points: list = field(default_factory=list)          # 视图平面点（共面）
    surface_points: list = field(default_factory=list)  # 真实表面点 / None
    snapped: list = field(default_factory=list)         # 每个点是否取自模型表面
    plane_normal: np.ndarray | None = None              # 测量平面的法向（视线）
    plane_point: np.ndarray | None = None               # 测量平面经过的点
    mode: str = "view"                                  # view | 3d
    visible: bool = True

    # ---------------- 状态 ----------------
    @property
    def needed(self) -> int:
        return MEASURE_DEFS.get(self.kind, (1, "", False))[0]

    @property
    def complete(self) -> bool:
        return len(self.points) >= self.needed

    def add_point(self, point, surface=None) -> bool:
        """追加一个点，返回测量是否已完成。"""
        if self.complete:
            return True
        self.points.append(np.asarray(point, dtype=np.float64))
        self.surface_points.append(
            None if surface is None else np.asarray(surface, dtype=np.float64))
        self.snapped.append(surface is not None)
        return self.complete

    def uses_surface(self) -> bool:
        """本次测量是否以真实表面点为主读数。"""
        if self.kind in _SURFACE_FIRST_KINDS:
            return True
        return self.mode == "3d" and self.kind in _MODE_AWARE_KINDS

    def effective_points(self) -> list:
        """读数与绘制使用的点。

        * 3D 点 / RZ 点：优先真实表面点（描述模型上的几何）；
        * 3D 模式下的 3D 直线 / 角度：优先真实表面点；
        * 两点圆 / 三点圆 / 2D 直线：始终使用视图平面点（平面测量），
          与 3D 模式开关无关；
        * 其余情况：使用视图平面点 —— 长度即屏幕投影长度。
        """
        out = []
        for i, p in enumerate(self.points):
            s = self.surface_points[i] if i < len(self.surface_points) else None
            if self.uses_surface() and s is not None:
                out.append(np.asarray(s, dtype=np.float64))
            else:
                out.append(np.asarray(p, dtype=np.float64))
        return out

    def invalid_reason(self) -> str | None:
        """完成后几何是否退化；返回 i18n 键，None 表示有效。

        屏幕视图测量时用户很可能在同一像素连点两下（以为没点上），
        两点重合会让圆半径 / 线段长度变成 0 —— 以前这种测量会被「完成」
        加入列表但画不出任何东西、也没有数值，看起来就像测量工具失灵。
        这里显式识别，交由调用方提示并继续等待有效点。
        """
        tol = 1e-9
        if self.kind in ("line3d", "line2d"):
            if not self.complete:
                return None
            return None if self.length > tol else "measure_point_too_close"
        if self.kind == "circle2p":
            if not self.complete:
                return None
            # 半径是第二点在视图平面内的投影距离（见 mathutil），
            # 退化时 circle 为 None，说明两点在该平面内几乎重合。
            return None if self.circle is not None else "measure_point_too_close"
        if self.kind == "circle3p":
            if not self.complete:
                return None
            return None if self.circle is not None else "measure_points_collinear"
        if self.kind == "angle2d":
            if not self.complete:
                return None
            pts = self.effective_points()
            v1 = pts[1] - pts[0]
            v2 = pts[2] - pts[0]
            if float(np.linalg.norm(v1)) <= tol or float(np.linalg.norm(v2)) <= tol:
                return "measure_point_too_close"
            return None
        return None

    # ---------------- 结果 ----------------
    @property
    def length(self) -> float:
        """主读数长度：视图投影（默认）或真实三维距离（3D 模式）。"""
        if self.kind in ("line3d", "line2d") and self.complete:
            pts = self.effective_points()
            return float(np.linalg.norm(pts[1] - pts[0]))
        return 0.0

    @property
    def projected_length(self) -> float:
        """视图平面内的投影长度（与屏幕所见一致）。"""
        if self.kind in ("line3d", "line2d") and self.complete:
            return float(np.linalg.norm(
                np.asarray(self.points[1]) - np.asarray(self.points[0])))
        return 0.0

    @property
    def spatial_length(self) -> float | None:
        """两个真实表面点之间的三维距离；空白处取点时为 None。"""
        if self.kind not in ("line3d", "line2d") or not self.complete:
            return None
        if len(self.surface_points) < 2:
            return None
        a, b = self.surface_points[0], self.surface_points[1]
        if a is None or b is None:
            return None
        return float(np.linalg.norm(np.asarray(b) - np.asarray(a)))

    @property
    def depth_delta(self) -> float:
        """两点沿视线方向的深度差（表面点）；无表面深度时为 0。"""
        if len(self.surface_points) < 2:
            return 0.0
        a, b = self.surface_points[0], self.surface_points[1]
        if a is None or b is None or self.plane_normal is None:
            return 0.0
        n = mu.normalize(np.asarray(self.plane_normal, dtype=np.float64))
        return float(np.dot(np.asarray(b) - np.asarray(a), n))

    @property
    def circle(self) -> tuple | None:
        """返回 (center, radius, normal)。

        ``circle2p`` 的第一点即圆心，第二点决定半径（两点测圆语义）：
        点下圆心后圆就随光标实时放大/缩小，再点一下确定半径。
        两点都在同一个视图平面内，半径就是屏幕上看到的半径。
        """
        if self.kind == "circle3p" and self.complete:
            # 三点圆 = 屏幕视图平面上的圆：三个点一律取视图平面点（共面），
            # 法向固定为测量平面法向。这样投影到画布上始终是正圆，半径就是
            # 屏幕上量到的半径，与 CATIA/SolidWorks 的平面测圆口径一致。
            pts = [np.asarray(p, dtype=np.float64) for p in self.points[:3]]
            solved = mu.circle_from_three_points(*pts)
            if solved is None:
                return None
            center, radius, fitted = solved
            normal = (mu.normalize(np.asarray(self.plane_normal,
                                              dtype=np.float64))
                      if self.plane_normal is not None else fitted)
            return center, radius, normal
        if self.kind == "circle2p" and self.complete:
            normal = self.plane_normal if self.plane_normal is not None \
                else np.array([0.0, 0.0, 1.0])
            return mu.circle_from_two_points(self.points[0], self.points[1], normal)
        return None

    @property
    def angle(self) -> float:
        if self.kind == "angle2d" and self.complete:
            pts = self.effective_points()
            v1 = pts[1] - pts[0]
            v2 = pts[2] - pts[0]
            return mu.angle_between(v1, v2)
        return 0.0

    def results(self) -> dict:
        """结构化结果（供 GUI/CLI 展示与导出）。"""
        out = {"kind": self.kind, "mode": self.mode}
        if self.kind == "point3d":
            pts = self.effective_points()
            out["point"] = pts[0].tolist() if pts else None
        elif self.kind in ("line3d", "line2d"):
            out["length_mm"] = self.length
            if self.complete:
                pts = self.effective_points()
                d = pts[1] - pts[0]
                out["delta"] = d.tolist()
                out["direction"] = mu.normalize(d).tolist()
                out["projected_length_mm"] = self.projected_length
                spatial = self.spatial_length
                if spatial is not None:
                    out["spatial_length_mm"] = spatial
                    out["depth_delta_mm"] = self.depth_delta
        elif self.kind == "rzpoint":
            pts = self.effective_points()
            if pts:
                r, phi, z = mu.cartesian_to_cylindrical(pts[0])
                out.update(radius=r, angle_deg=phi, z=z)
        elif self.kind in ("circle2p", "circle3p"):
            c = self.circle
            if c:
                out.update(center=c[0].tolist(), radius=float(c[1]),
                           normal=c[2].tolist(), diameter=float(c[1]) * 2.0)
        elif self.kind == "angle2d":
            out["angle_deg"] = self.angle
        return out

    def delta_text(self, units, digits: int = 3) -> str:
        """线段的三轴分量 ΔX/ΔY/ΔZ（供状态栏显示）。

        默认口径下这些分量是**视图投影**上的分量：TOP 视图量一条水平边，
        ΔZ 就是 0.000 —— 这正是用户期望的「不考虑往屏幕里外的深度」。
        3D 模式下则是真实三维分量。
        """
        if self.kind not in ("line2d", "line3d") or len(self.points) < 2:
            return ""
        pts = self.effective_points()
        d = pts[1] - pts[0]
        return (f"\u0394X={units.to_display(d[0]):+.{digits}f} "
                f"\u0394Y={units.to_display(d[1]):+.{digits}f} "
                f"\u0394Z={units.to_display(d[2]):+.{digits}f} {units.label}")

    def value_text(self, units) -> str:
        """带单位格式化的主结果文本。"""
        if self.kind == "point3d":
            pts = self.effective_points()
            if not pts:
                return ""
            return f"{units.format_point(pts[0])} {units.label}"
        if self.kind in ("line3d", "line2d"):
            text = f"L={units.format_length(self.length)} {units.label}"
            spatial = self.spatial_length
            if spatial is not None:
                # 深度差不可忽略时把真实三维距离一并给出：默认读数仍是
                # 屏幕投影长度，用户不用切换模式就能看到两者差多少。
                scale = max(abs(spatial), abs(self.length), 1e-9)
                if abs(spatial - self.length) > 5e-4 * scale:
                    text += (f"  3D={units.format_length(spatial)} "
                             f"{units.label}")
            return text
        if self.kind == "rzpoint":
            pts = self.effective_points()
            if not pts:
                return ""
            r, phi, z = mu.cartesian_to_cylindrical(pts[0])
            return (f"R={units.format_length(r)} Z={units.format_length(z)} "
                    f"\u03c6={phi:.2f}\u00b0")
        if self.kind in ("circle2p", "circle3p"):
            c = self.circle
            if not c:
                return ""
            return (f"\u2300={units.format_length(c[1] * 2.0)} R="
                    f"{units.format_length(c[1])} {units.label}")
        if self.kind == "angle2d":
            return f"{self.angle:.3f}\u00b0"
        return ""

    # ---------------- 绘制数据 ----------------
    def polylines(self) -> list:
        """返回需要绘制的折线（世界坐标点数组列表）。"""
        pts = self.effective_points()
        if self.kind in ("line3d", "line2d") and self.complete:
            return [np.stack([pts[0], pts[1]])]
        if self.kind == "angle2d" and self.complete:
            arc = self._angle_arc()
            out = [np.stack([pts[0], pts[1]]), np.stack([pts[0], pts[2]])]
            if arc is not None:
                out.append(arc)
            return out
        if self.kind in ("circle2p", "circle3p") and self.complete:
            c = self.circle
            if c is None:
                return []
            circumference = mu.circle_points(c[0], c[1], c[2], 96)
            return [np.vstack([circumference, circumference[:1]])]
        if self.kind == "rzpoint" and pts:
            p = pts[0]
            axis_pt = np.array([0.0, 0.0, p[2]])
            foot = np.array([p[0], p[1], 0.0])
            ground = np.array([0.0, 0.0, 0.0])
            return [np.stack([axis_pt, p]), np.stack([axis_pt, ground]),
                    np.stack([p, foot])]
        return []

    def markers(self) -> list:
        pts = self.effective_points()
        if not pts:
            return []
        if self.kind in ("line3d", "line2d") and self.complete:
            mid = (pts[0] + pts[1]) * 0.5
            return [pts[0], pts[1], mid]
        if self.kind in ("circle2p", "circle3p") and self.complete:
            c = self.circle
            out = list(pts)
            if c is not None and self.kind == "circle3p":
                out.append(c[0])       # 三点圆的圆心；两点圆的圆心就是第一点
            return out
        return list(pts)

    def label_anchor(self) -> np.ndarray:
        return self.midpoint()

    def midpoint(self) -> np.ndarray:
        pts = self.effective_points()
        if not pts:
            return np.zeros(3)
        if self.kind in ("line3d", "line2d") and self.complete:
            return (pts[0] + pts[1]) * 0.5
        if self.kind in ("circle2p", "circle3p") and self.complete:
            c = self.circle
            if c is not None:
                return c[0]
        return np.mean(np.stack(pts), axis=0)

    def label(self) -> str:
        """标注文本（不带单位换算，单位由调用方决定）。"""
        return self._label_cache or ""

    def refresh_label(self, text: str) -> None:
        self._label_cache = text

    _label_cache: str = ""

    def _angle_arc(self, count: int = 48) -> np.ndarray | None:
        """2D 角度的圆弧。"""
        pts = self.effective_points()
        if len(pts) < 3:
            return None
        vertex = pts[0]
        v1 = pts[1] - vertex
        v2 = pts[2] - vertex
        n1 = np.linalg.norm(v1)
        n2 = np.linalg.norm(v2)
        if n1 < 1e-12 or n2 < 1e-12:
            return None
        u = v1 / n1
        w = v2 / n2 - u * float(np.dot(v2 / n2, u))
        wn = np.linalg.norm(w)
        if wn < 1e-12:
            return None
        w = w / wn
        radius = min(n1, n2) * 0.35
        angles = np.linspace(0.0, np.arccos(np.clip(np.dot(u, v2 / n2), -1, 1)), count)
        return vertex + np.outer(np.cos(angles) * radius, u) + \
            np.outer(np.sin(angles) * radius, w)


class MeasureManager:
    """测量集合管理（新建/删除/清空/导出）。"""

    def __init__(self, color=(1.0, 0.86, 0.32)):
        self.items: list[Measure] = []
        self.color = tuple(color)
        self.mode = "view"
        self._pending: Measure | None = None
        # 最近一次取点被拒的原因（i18n 键）；供 GUI 提示，None 表示正常
        self.last_reject: str | None = None

    # ---------------- 状态 ----------------
    @property
    def pending(self) -> Measure | None:
        return self._pending

    def active_kind(self) -> str | None:
        return self._pending.kind if self._pending else None

    def set_mode(self, mode: str) -> None:
        self.mode = mode if mode in MEASURE_MODES else "view"

    def start(self, kind: str) -> Measure:
        """开启一个新的测量（对应点击菜单项）。"""
        self._pending = Measure(kind=kind, color=self.color, mode=self.mode)
        self.last_reject = None
        return self._pending

    def cancel(self) -> None:
        self._pending = None
        self.last_reject = None

    def add_point(self, point, units=None, plane=None, snapped: bool = False,
                  surface=None, plane_point=None) -> Measure | None:
        """给当前测量加点；完成时把测量加入列表并返回它。

        * ``point`` 是视图平面点（同一测量内共面）；
        * ``surface`` 是同一个像素下的真实模型表面点，用于 ``3D=`` 读数；
        * ``plane`` / ``plane_point`` 固定测量平面，第一点之后不再改变。

        完成后的几何若退化（两点重合 / 三点共线等），该点会被撤销、
        ``last_reject`` 置为提示键，pending 继续保持激活 —— 用户再点一次
        即可，不会出现「测量凭空消失」。
        """
        if self._pending is None:
            return None
        m = self._pending
        if m.plane_normal is None and plane is not None:
            m.plane_normal = mu.normalize(np.asarray(plane, dtype=np.float64))
        if m.plane_point is None and plane_point is not None:
            m.plane_point = np.asarray(plane_point, dtype=np.float64).copy()
        m.points.append(np.asarray(point, dtype=np.float64))
        m.snapped.append(bool(snapped))
        # 兼容旧调用方：只给 snapped=True、没有单独 surface 时，把 point
        # 当作表面点（旧语义里两者本就是同一个点）。
        if surface is None and snapped:
            surface = point
        m.surface_points.append(
            None if surface is None else np.asarray(surface, dtype=np.float64))
        if not m.complete:
            self.last_reject = None
            return None
        reason = m.invalid_reason()
        if reason is not None:
            m.points.pop()                 # 撤销刚点的这个点，继续等待有效点
            m.snapped.pop()
            m.surface_points.pop()
            self.last_reject = reason
            return None
        self._pending = None
        self.last_reject = None
        m.refresh_label(m.value_text(units) if units else "")
        self.items.append(m)
        return m

    # ---------------- 删除 ----------------
    def delete_first(self) -> bool:
        if self.items:
            self.items.pop(0)
            return True
        return False

    def delete_last(self) -> bool:
        if self.items:
            self.items.pop()
            return True
        return False

    def delete_all(self) -> int:
        n = len(self.items)
        self.items.clear()
        self._pending = None
        self.last_reject = None
        return n

    def set_color(self, color) -> None:
        self.color = tuple(color)
        for item in self.items:
            item.color = self.color
        if self._pending:
            self._pending.color = self.color

    def clear(self) -> None:
        self.items.clear()
        self._pending = None
        self.last_reject = None

    def refresh_labels(self, units) -> None:
        for item in self.items:
            item.refresh_label(item.value_text(units))

    def export(self, units) -> list:
        out = []
        for i, item in enumerate(self.items, 1):
            out.append({"index": i, "kind": item.kind, "text": item.value_text(units),
                        **item.results()})
        return out


# --------------------------------------------------------------------------
# 吸附拾取
# --------------------------------------------------------------------------
def snap_pick(scene, camera, renderer, x: float, y: float,
              snap_pixels: float = 12.0, snap_vertices: bool = True) -> dict | None:
    """屏幕拾取并吸附到最近顶点。

    返回 {point, object, triangle, snapped}。
    """
    hit = renderer.pick(scene, camera, x, y)
    if hit is None:
        return None
    point = hit["point"]
    result = {"point": point, "object": hit["object"], "triangle": hit["triangle"],
              "snapped": False, "distance": hit["distance"]}
    if not snap_vertices:
        return result

    obj = hit["object"]
    mesh = obj.mesh
    matrix = renderer._object_matrix(obj)
    world_verts = mu.transform_points(matrix, mesh.vertices)
    scr_click = np.array([x, y], dtype=np.float64)
    scale = max(1.0, getattr(camera, "width", 1000) / 1000.0)
    tol = snap_pixels * scale
    # 先用世界距离粗筛，避免对全部顶点做投影
    radius_world = tol * camera.world_per_pixel() * 3.0
    near = np.nonzero(np.linalg.norm(world_verts - point, axis=1) <
                      max(radius_world, 1e-9))[0]
    if len(near) == 0:
        return result
    scr, _, _ = camera.project(world_verts[near])
    d = np.linalg.norm(scr - scr_click, axis=1)
    k = int(np.argmin(d))
    if d[k] <= tol:
        result["point"] = world_verts[near[k]].copy()
        result["snapped"] = True
        result["vertex_index"] = int(near[k])
    return result
