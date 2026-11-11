"""场景与对象模型 —— 对象面板与对象右键菜单
（miObjVisible / miObjWire / miObjTransparent / miObjActive /
miObjAlwaysOnTop / miNewObject / miCopyObject / miDeleteObject）。

一个场景可同时载入多个模型（例如毛坯 + 刀具 + 仿真切除结果），
每个对象拥有独立的显示属性。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

import numpy as np

from .mesh import Mesh

# 对象默认调色板（依次分配给新载入的对象）
OBJECT_PALETTE = [
    (0.82, 0.84, 0.88),
    (0.93, 0.75, 0.42),
    (0.55, 0.78, 0.90),
    (0.78, 0.62, 0.86),
    (0.62, 0.85, 0.66),
    (0.90, 0.58, 0.55),
    (0.70, 0.70, 0.72),
    (0.45, 0.68, 0.82),
]


@dataclass
class Object3D:
    """场景中的一个模型对象。"""

    mesh: Mesh
    name: str = ""
    visible: bool = True
    wire: bool = False            # miObjWire：仅线框
    transparent: bool = False     # miObjTransparent：半透明
    always_on_top: bool = False   # miObjAlwaysOnTop：始终显示在最前
    active: bool = False          # miObjActive：当前编辑对象
    locked: bool = False          # 锁定后不参与拾取/编辑
    color: tuple = (0.82, 0.84, 0.88)     # Uni Color 模式的基准色
    auto_color: bool = True               # Colored / Auto Colors
    color_mode: str = "auto"              # auto | uniform | component | curvature
    alpha: float = 0.45
    point_size: int = 1
    line_width: int = 1
    source_path: str = ""
    visible_override: bool = True
    _color_cache: np.ndarray | None = field(default=None, repr=False, compare=False)
    _flat_cache: tuple | None = field(default=None, repr=False, compare=False)

    def __post_init__(self) -> None:
        if not self.name:
            self.name = self.mesh.name or "object"

    # ---------------- 派生属性 ----------------
    @property
    def triangle_count(self) -> int:
        return self.mesh.triangle_count

    def bounds(self) -> tuple:
        return self.mesh.bounds()

    def center(self) -> np.ndarray:
        return self.mesh.center()

    def flat_color(self, global_mode: str | None = None) -> bool:
        """该着色模式是否为"整件同色"。

        整件同色时才可以安全地做逐顶点（Gouraud）平滑着色；按法向/分量/曲率
        取色时逐面颜色本身携带信息，插值反而会把它抹掉。
        """
        mode = global_mode or self.color_mode
        return (not self.auto_color) or mode in ("auto", "uniform")

    def face_colors(self, global_mode: str | None = None) -> np.ndarray:
        """返回逐面基础色 (M,3)，0..1。

        ``auto`` / ``uniform`` 表示"整件同色"，用来呈现真实零件外观（再由光照
        产生明暗）。注意：早期实现把 ``auto`` 接到了法向彩虹图上，导致默认实体
        视图看起来像一张法向贴图；法向取色现已独立为 ``orientation`` 模式。
        """
        mode = global_mode or self.color_mode
        if mode in ("auto", "uniform") or not self.auto_color:
            base = tuple(float(c) for c in self.color)
            cached = self._flat_cache
            if (cached is not None and cached[0] == base
                    and len(cached[1]) == self.triangle_count):
                return cached[1]
            arr = np.tile(np.asarray(base, dtype=np.float64),
                          (self.triangle_count, 1))
            self._flat_cache = (base, arr)
            return arr
        if self._color_cache is not None and len(self._color_cache) == self.triangle_count:
            return self._color_cache
        if mode == "component":
            colors = self.mesh.color_by_component()
        elif mode == "curvature":
            colors = self.mesh.color_by_curvature()
        else:                                   # orientation：法向彩虹图
            colors = self.mesh.color_by_orientation()
        # 与对象基准色混合，使不同对象仍可区分
        base = np.asarray(self.color, dtype=np.float64)
        colors = 0.14 * base[None, :] + 0.86 * colors
        self._color_cache = np.clip(colors, 0.0, 1.0)
        return self._color_cache

    def invalidate_colors(self) -> None:
        self._color_cache = None
        self._flat_cache = None

    # ---------------- 属性开关 ----------------
    def toggle_visible(self) -> bool:
        self.visible = not self.visible
        return self.visible

    def to_state(self) -> dict:
        return {
            "name": self.name,
            "visible": self.visible,
            "wire": self.wire,
            "transparent": self.transparent,
            "always_on_top": self.always_on_top,
            "active": self.active,
            "locked": self.locked,
            "color": list(self.color),
            "auto_color": self.auto_color,
            "color_mode": self.color_mode,
            "alpha": self.alpha,
            "point_size": self.point_size,
            "line_width": self.line_width,
            "source_path": self.source_path,
        }

    def set_state(self, state: dict) -> None:
        for key, value in state.items():
            if key == "color":
                self.color = tuple(float(c) for c in value)
            elif hasattr(self, key) and key not in ("mesh", "name"):
                setattr(self, key, value)
        self.invalidate_colors()


class Scene:
    """对象集合与场景级设置。"""

    def __init__(self):
        self.objects: list[Object3D] = []
        self.background: tuple = (0.16, 0.17, 0.20)   # Background Colors
        self.background_top: tuple = (0.28, 0.31, 0.36)  # 渐变背景上端
        self.gradient_background: bool = False
        self._counter = 0
        self._active = -1

    # ---------------- 增删 ----------------
    def add_mesh(self, mesh: Mesh, name: str | None = None, activate: bool = True,
                 source_path: str = "") -> Object3D:
        obj = Object3D(
            mesh=mesh,
            name=name or mesh.name or f"Object{len(self.objects) + 1}",
            color=OBJECT_PALETTE[self._counter % len(OBJECT_PALETTE)],
            source_path=source_path or mesh.source,
        )
        self._counter += 1
        self.objects.append(obj)
        if activate:
            self.set_active(len(self.objects) - 1)
        return obj

    def remove_object(self, index: int) -> Object3D | None:
        if 0 <= index < len(self.objects):
            obj = self.objects.pop(index)
            if self._active >= len(self.objects):
                self._active = len(self.objects) - 1
            self._sync_active_flags()
            return obj
        return None

    def clear(self) -> None:
        self.objects.clear()
        self._active = -1

    def duplicate_object(self, index: int) -> Object3D | None:
        """对应 miCopyObject / miRendSys... 的对象复制。"""
        if not (0 <= index < len(self.objects)):
            return None
        src = self.objects[index]
        obj = Object3D(mesh=src.mesh.copy(), name=f"{src.name}_copy")
        obj.set_state(src.to_state())
        obj.name = f"{src.name}_copy"
        self.objects.append(obj)
        self.set_active(len(self.objects) - 1)
        return obj

    # ---------------- 激活 / 查找 ----------------
    @property
    def active_index(self) -> int:
        """当前激活对象索引，无有效对象时为 -1。"""
        if 0 <= self._active < len(self.objects):
            return self._active
        return -1

    @active_index.setter
    def active_index(self, value: int) -> None:
        self.set_active(int(value))

    def _sync_active_flags(self) -> None:
        current = self.active_index
        for i, obj in enumerate(self.objects):
            obj.active = (i == current)

    def set_active(self, index: int) -> None:
        self._active = index if 0 <= index < len(self.objects) else -1
        self._sync_active_flags()

    def active_object(self) -> Object3D | None:
        i = self.active_index
        return self.objects[i] if i >= 0 else None

    def find(self, name: str) -> Object3D | None:
        for obj in self.objects:
            if obj.name == name:
                return obj
        return None

    def visible_objects(self) -> list:
        return [o for o in self.objects if o.visible]

    # ---------------- 包围盒 ----------------
    def bounds(self) -> tuple:
        """全部对象（含不可见）的总包围盒。"""
        los, his = [], []
        for obj in self.objects:
            if obj.mesh.vertex_count == 0:
                continue
            lo, hi = obj.mesh.bounds()
            los.append(lo)
            his.append(hi)
        if not los:
            return np.zeros(3), np.zeros(3)
        return np.min(np.stack(los), axis=0), np.max(np.stack(his), axis=0)

    def visible_bounds(self) -> tuple:
        los, his = [], []
        for obj in self.objects:
            if not obj.visible or obj.mesh.vertex_count == 0:
                continue
            lo, hi = obj.mesh.bounds()
            los.append(lo)
            his.append(hi)
        if not los:
            return np.zeros(3), np.zeros(3)
        return np.min(np.stack(los), axis=0), np.max(np.stack(his), axis=0)

    def has_geometry(self) -> bool:
        return any(o.mesh.triangle_count > 0 for o in self.objects)

    # ---------------- 统计 ----------------
    def statistics(self) -> dict:
        return {
            "objects": len(self.objects),
            "visible": len(self.visible_objects()),
            "triangles": sum(o.triangle_count for o in self.objects),
            "vertices": sum(o.mesh.vertex_count for o in self.objects),
            "bytes": sum(o.mesh.nbytes for o in self.objects),
        }

    # ---------------- 程序化图元 ----------------
    def add_box(self, size=(20.0, 20.0, 20.0), center=(0, 0, 0), name="Box") -> Object3D:
        return self.add_mesh(_make_box(size, center), name=name)

    def add_cylinder(self, radius: float = 10.0, height: float = 40.0,
                     segments: int = 64, center=(0, 0, 0), name="Cylinder") -> Object3D:
        return self.add_mesh(_make_cylinder(radius, height, segments, center), name=name)

    def add_step_cylinder(self, radii=(6.0, 10.0), heights=(20.0, 20.0),
                          center=(0, 0, 0), name="StepCylinder") -> Object3D:
        """阶梯圆柱（对应 miNewStepCylinder，刀具毛坯常用形状）。"""
        meshes = []
        z = 0.0
        verts_all, tris_all = [], []
        base = 0
        for r, h in zip(radii, heights):
            m = _make_cylinder(r, h, 64, (0, 0, z + h * 0.5))
            verts_all.append(m.vertices)
            tris_all.append(m.triangles + base)
            base += m.vertex_count
            z += h
        mesh = Mesh(np.concatenate(verts_all), np.concatenate(tris_all), name=name)
        mesh.translate(np.asarray(center, dtype=np.float64))
        return self.add_mesh(mesh, name=name)

    def add_sphere(self, radius: float = 10.0, center=(0, 0, 0),
                   segments: int = 32, name="Sphere") -> Object3D:
        return self.add_mesh(_make_sphere(radius, segments, center), name=name)

    def add_tube(self, outer: float = 10.0, inner: float = 6.0, height: float = 30.0,
                 center=(0, 0, 0), name="Tube") -> Object3D:
        outer_m = _make_cylinder(outer, height, 64, (0, 0, 0))
        # 内孔：翻转法向并缩放
        inner_m = _make_cylinder(inner, height * 1.2, 64, (0, 0, 0))
        inner_m.flip_normals()
        verts = np.concatenate([outer_m.vertices, inner_m.vertices])
        tris = np.concatenate([outer_m.triangles, inner_m.triangles + outer_m.vertex_count])
        mesh = Mesh(verts, tris, name=name)
        mesh.translate(np.asarray(center, dtype=np.float64))
        return self.add_mesh(mesh, name=name)


# --------------------------------------------------------------------------
# 程序化图元构造
# --------------------------------------------------------------------------
def _make_box(size, center) -> Mesh:
    sx, sy, sz = (float(s) * 0.5 for s in size)
    cx, cy, cz = center
    v = np.array([
        [cx - sx, cy - sy, cz - sz], [cx + sx, cy - sy, cz - sz],
        [cx + sx, cy + sy, cz - sz], [cx - sx, cy + sy, cz - sz],
        [cx - sx, cy - sy, cz + sz], [cx + sx, cy - sy, cz + sz],
        [cx + sx, cy + sy, cz + sz], [cx - sx, cy + sy, cz + sz],
    ], dtype=np.float64)
    f = np.array([
        [0, 3, 2], [0, 2, 1],   # 底
        [4, 5, 6], [4, 6, 7],   # 顶
        [0, 1, 5], [0, 5, 4],   # -Y
        [1, 2, 6], [1, 6, 5],   # +X
        [2, 3, 7], [2, 7, 6],   # +Y
        [3, 0, 4], [3, 4, 7],   # -X
    ], dtype=np.int64)
    return Mesh(v, f, name="Box")


def _make_cylinder(radius: float, height: float, segments: int, center) -> Mesh:
    segments = max(int(segments), 3)
    ang = np.linspace(0.0, 2.0 * np.pi, segments, endpoint=False)
    cx, cy = np.cos(ang), np.sin(ang)
    hz = height * 0.5
    cz = center[2]
    ring_b = np.stack([cx * radius + center[0], cy * radius + center[1],
                       np.full_like(cx, cz - hz)], axis=1)
    ring_t = np.stack([cx * radius + center[0], cy * radius + center[1],
                       np.full_like(cx, cz + hz)], axis=1)
    v = np.concatenate([ring_b, ring_t, [[center[0], center[1], cz - hz]],
                        [[center[0], center[1], cz + hz]]])
    i0 = segments
    cb, ct = 2 * segments, 2 * segments + 1
    faces = []
    for i in range(segments):
        j = (i + 1) % segments
        # 侧面（逆时针绕向 -> 法线朝外）
        faces.append([i, j, i0 + j])
        faces.append([i, i0 + j, i0 + i])
        # 底面（法线朝 -Z）与顶面（法线朝 +Z）
        faces.append([cb, j, i])
        faces.append([ct, i0 + i, i0 + j])
    return Mesh(v, np.array(faces, dtype=np.int64), name="Cylinder")


def _make_sphere(radius: float, segments: int, center) -> Mesh:
    segments = max(int(segments), 4)
    rings = max(segments // 2, 2)
    thetas = np.linspace(0.0, np.pi, rings + 1)
    phis = np.linspace(0.0, 2.0 * np.pi, segments, endpoint=False)
    verts = []
    for t in thetas:
        for p in phis:
            verts.append([np.sin(t) * np.cos(p), np.sin(t) * np.sin(p), np.cos(t)])
    v = np.array(verts, dtype=np.float64) * radius + np.asarray(center, dtype=np.float64)
    faces = []
    for r in range(rings):
        for c in range(segments):
            a = r * segments + c
            b = r * segments + (c + 1) % segments
            d = (r + 1) * segments + c
            e = (r + 1) * segments + (c + 1) % segments
            faces.append([a, d, e])
            faces.append([a, e, b])
    return Mesh(v, np.array(faces, dtype=np.int64), name="Sphere")
