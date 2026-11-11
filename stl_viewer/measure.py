"""测量工具 —— Measurement 菜单：

    Measure 2D Line / Measure 3D Line / Measure 3D Point /
    Measure 2 Point Circle / Measure 3 Point Circle /
    Measure 2D Angle / Measure RZ Point /
    Delete first / Delete last / Delete All Measures / Measure color

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


@dataclass
class Measure:
    """一次测量。"""

    kind: str
    color: tuple = (1.0, 0.86, 0.32)
    points: list = field(default_factory=list)          # 已点击的世界点
    snapped: list = field(default_factory=list)         # 每个点是否取自模型表面
    plane_normal: np.ndarray | None = None              # 2D 测量的参考平面法向
    plane_point: np.ndarray | None = None
    visible: bool = True

    # ---------------- 状态 ----------------
    @property
    def needed(self) -> int:
        return MEASURE_DEFS.get(self.kind, (1, "", False))[0]

    @property
    def complete(self) -> bool:
        return len(self.points) >= self.needed

    def add_point(self, point) -> bool:
        """追加一个点，返回测量是否已完成。"""
        if self.complete:
            return True
        self.points.append(np.asarray(point, dtype=np.float64))
        return self.complete

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
            v1 = self.points[1] - self.points[0]
            v2 = self.points[2] - self.points[0]
            if float(np.linalg.norm(v1)) <= tol or float(np.linalg.norm(v2)) <= tol:
                return "measure_point_too_close"
            return None
        return None

    # ---------------- 结果 ----------------
    @property
    def length(self) -> float:
        if self.kind in ("line3d", "line2d") and self.complete:
            return float(np.linalg.norm(self.points[1] - self.points[0]))
        return 0.0

    @property
    def circle(self) -> tuple | None:
        """返回 (center, radius, normal)。

        ``circle2p`` 的第一点即圆心，第二点决定半径（两点测圆语义）：
        点下圆心后圆就随光标实时放大/缩小，再点一下确定半径。
        """
        if self.kind == "circle3p" and self.complete:
            return mu.circle_from_three_points(*self.points[:3])
        if self.kind == "circle2p" and self.complete:
            normal = self.plane_normal if self.plane_normal is not None \
                else np.array([0.0, 0.0, 1.0])
            return mu.circle_from_two_points(self.points[0], self.points[1], normal)
        return None

    @property
    def angle(self) -> float:
        if self.kind == "angle2d" and self.complete:
            v1 = self.points[1] - self.points[0]
            v2 = self.points[2] - self.points[0]
            return mu.angle_between(v1, v2)
        return 0.0

    def results(self) -> dict:
        """结构化结果（供 GUI/CLI 展示与导出）。"""
        out = {"kind": self.kind}
        if self.kind == "point3d":
            out["point"] = self.points[0].tolist() if self.points else None
        elif self.kind in ("line3d", "line2d"):
            out["length_mm"] = self.length
            if self.complete:
                d = self.points[1] - self.points[0]
                out["delta"] = d.tolist()
                out["direction"] = mu.normalize(d).tolist()
        elif self.kind == "rzpoint":
            if self.points:
                r, phi, z = mu.cartesian_to_cylindrical(self.points[0])
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

        三维距离是两点间的真实长度：斜视角下如果点的是圆柱端面的轮廓极值，
        量到的会比名义长度长（那两个点确实相距更远）。把三轴分量一并给出，
        用户能立刻看到「Z 向确实是 40.000，多出来的是端面半径造成的」。
        """
        if self.kind not in ("line2d", "line3d") or len(self.points) < 2:
            return ""
        d = self.points[1] - self.points[0]
        return (f"\u0394X={units.to_display(d[0]):+.{digits}f} "
                f"\u0394Y={units.to_display(d[1]):+.{digits}f} "
                f"\u0394Z={units.to_display(d[2]):+.{digits}f} {units.label}")

    def value_text(self, units) -> str:
        """带单位格式化的主结果文本。"""
        if self.kind == "point3d":
            if not self.points:
                return ""
            return f"{units.format_point(self.points[0])} {units.label}"
        if self.kind in ("line3d", "line2d"):
            return f"L={units.format_length(self.length)} {units.label}"
        if self.kind == "rzpoint":
            if not self.points:
                return ""
            r, phi, z = mu.cartesian_to_cylindrical(self.points[0])
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
        pts = self.points
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
        if self.kind == "rzpoint" and self.points:
            p = pts[0]
            axis_pt = np.array([0.0, 0.0, p[2]])
            foot = np.array([p[0], p[1], 0.0])
            ground = np.array([0.0, 0.0, 0.0])
            return [np.stack([axis_pt, p]), np.stack([axis_pt, ground]),
                    np.stack([p, foot])]
        return []

    def markers(self) -> list:
        if not self.points:
            return []
        if self.kind in ("line3d", "line2d") and self.complete:
            mid = (self.points[0] + self.points[1]) * 0.5
            return [self.points[0], self.points[1], mid]
        if self.kind in ("circle2p", "circle3p") and self.complete:
            c = self.circle
            out = list(self.points)
            if c is not None and self.kind == "circle3p":
                out.append(c[0])       # 三点圆的圆心；两点圆的圆心就是第一点
            return out
        return list(self.points)

    def label_anchor(self) -> np.ndarray:
        return self.midpoint()

    def midpoint(self) -> np.ndarray:
        if not self.points:
            return np.zeros(3)
        if self.kind in ("line3d", "line2d") and self.complete:
            return (self.points[0] + self.points[1]) * 0.5
        if self.kind in ("circle2p", "circle3p") and self.complete:
            c = self.circle
            if c is not None:
                return c[0]
        return np.mean(np.stack(self.points), axis=0)

    def label(self) -> str:
        """标注文本（不带单位换算，单位由调用方决定）。"""
        return self._label_cache or ""

    def refresh_label(self, text: str) -> None:
        self._label_cache = text

    _label_cache: str = ""

    def _angle_arc(self, count: int = 48) -> np.ndarray | None:
        """2D 角度的圆弧。"""
        if len(self.points) < 3:
            return None
        vertex = self.points[0]
        v1 = self.points[1] - vertex
        v2 = self.points[2] - vertex
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
        self._pending: Measure | None = None
        # 最近一次取点被拒的原因（i18n 键）；供 GUI 提示，None 表示正常
        self.last_reject: str | None = None

    # ---------------- 状态 ----------------
    @property
    def pending(self) -> Measure | None:
        return self._pending

    def active_kind(self) -> str | None:
        return self._pending.kind if self._pending else None

    def start(self, kind: str) -> Measure:
        """开启一个新的测量（对应点击菜单项）。"""
        self._pending = Measure(kind=kind, color=self.color)
        self.last_reject = None
        return self._pending

    def cancel(self) -> None:
        self._pending = None
        self.last_reject = None

    def add_point(self, point, units=None, plane=None, snapped: bool = False) -> Measure | None:
        """给当前测量加点；完成时把测量加入列表并返回它。

        完成后的几何若退化（两点重合 / 三点共线等），该点会被撤销、
        ``last_reject`` 置为提示键，pending 继续保持激活 —— 用户再点一次
        即可，不会出现「测量凭空消失」。

        ``snapped`` 表示该点取自模型表面（有真实深度），否则取自视图平面；
        仅用于标注与提示，不影响距离计算（都是三维点之间的距离）。
        """
        if self._pending is None:
            return None
        m = self._pending
        if m.plane_normal is None and plane is not None:
            m.plane_normal = np.asarray(plane, dtype=np.float64)
        m.points.append(np.asarray(point, dtype=np.float64))
        m.snapped.append(bool(snapped))
        if not m.complete:
            self.last_reject = None
            return None
        reason = m.invalid_reason()
        if reason is not None:
            m.points.pop()                 # 撤销刚点的这个点，继续等待有效点
            m.snapped.pop()
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
