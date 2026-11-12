"""三角网格模型 —— T3dStl / TTriangle / TTrianglePlus / TBBTree 风格的实现。

包含：包围盒、法线、焊接、连通分量、自动着色、修复、
相交三角形剔除（Extras > STL > RemoveTriIntersectedChildren）、
剖切与统计信息。
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from . import mathutil as mu
from .stl_io import RawMesh, read_mesh


@dataclass
class MeshStatistics:
    vertices: int = 0
    triangles: int = 0
    loose_vertices: int = 0
    degenerate: int = 0
    components: int = 0
    boundary_edges: int = 0
    non_manifold_edges: int = 0
    closed: bool = False
    inverted: bool = False
    area: float = 0.0
    volume: float = 0.0
    size: tuple = (0.0, 0.0, 0.0)
    center: tuple = (0.0, 0.0, 0.0)

    def as_text(self) -> str:
        lines = [
            f"顶点数      : {self.vertices}",
            f"三角面数    : {self.triangles}",
            f"游离顶点    : {self.loose_vertices}",
            f"退化三角形  : {self.degenerate}",
            f"连通分量    : {self.components}",
            f"边界边      : {self.boundary_edges}",
            f"非流形边    : {self.non_manifold_edges}",
            f"封闭实体    : {'是' if self.closed else '否'}",
            f"法向取向    : {'朝内(需翻转)' if self.inverted else '朝外(正常)'}",
            f"表面积      : {self.area:.4f}",
            f"体积        : {self.volume:.4f}",
            f"外形尺寸    : {self.size[0]:.4f} x {self.size[1]:.4f} x {self.size[2]:.4f}",
            f"几何中心    : ({self.center[0]:.4f}, {self.center[1]:.4f}, {self.center[2]:.4f})",
        ]
        return "\n".join(lines)


class Mesh:
    """可渲染的三角网格。

    顶点数组与面索引数组始终为 numpy 数组：
        vertices  : (N, 3) float64
        triangles : (M, 3) int64
    """

    def __init__(self, vertices, triangles, name: str = "model", source: str = ""):
        self.vertices = np.ascontiguousarray(np.asarray(vertices, dtype=np.float64).reshape(-1, 3))
        tri = np.asarray(triangles, dtype=np.int64)
        if tri.size:
            tri = tri.reshape(-1, 3)
        else:
            tri = np.zeros((0, 3), dtype=np.int64)
        self.triangles = np.ascontiguousarray(tri)
        self.name = name
        self.source = source
        self._normals: np.ndarray | None = None
        self._vertex_normals: np.ndarray | None = None
        self._components: np.ndarray | None = None
        self._bounds: tuple | None = None

    # ---------------- 构造 ----------------
    @classmethod
    def from_raw(cls, raw: RawMesh) -> "Mesh":
        return cls(raw.vertices, raw.triangles, name=raw.name, source=raw.source)

    def weld_auto(self, relative: float = 1e-6) -> int:
        """按模型自身尺寸选取容差并焊接。

        STL/PLY 里同一个几何点常常按三角形重复存储（三角汤），不焊接时每个
        三角形都是独立连通分量：体积算不出、边界边满天飞、法向插值无从谈起。
        容差取包围盒对角线的一个极小比例，避免误合并相邻但不同的顶点。
        """
        lo, hi = self.bounds()
        diag = float(np.linalg.norm(np.asarray(hi, dtype=np.float64)
                                    - np.asarray(lo, dtype=np.float64)))
        return self.weld(max(diag * relative, 1e-9))

    @classmethod
    def load(cls, path: str, weld: bool = True) -> "Mesh":
        mesh = cls.from_raw(read_mesh(path))
        if weld:
            mesh.weld_auto()
        return mesh

    @classmethod
    def from_arrays(cls, vertices, triangles) -> "Mesh":
        return cls(vertices, triangles)

    def copy(self) -> "Mesh":
        return Mesh(self.vertices.copy(), self.triangles.copy(), self.name, self.source)

    def invalidate(self) -> None:
        """数据变更后清空派生缓存。"""
        self._normals = None
        self._vertex_normals = None
        self._components = None
        self._bounds = None

    # ---------------- 基本属性 ----------------
    @property
    def triangle_count(self) -> int:
        return int(len(self.triangles))

    @property
    def vertex_count(self) -> int:
        return int(len(self.vertices))

    @property
    def nbytes(self) -> int:
        return int(self.vertices.nbytes + self.triangles.nbytes)

    def bounds(self) -> tuple:
        """返回 (min_xyz, max_xyz)。空网格返回全零。"""
        if self._bounds is None:
            if self.vertex_count == 0:
                self._bounds = (np.zeros(3), np.zeros(3))
            else:
                self._bounds = (self.vertices.min(axis=0), self.vertices.max(axis=0))
        return self._bounds

    def center(self) -> np.ndarray:
        lo, hi = self.bounds()
        return (lo + hi) * 0.5

    def size(self) -> np.ndarray:
        lo, hi = self.bounds()
        return hi - lo

    def radius(self) -> float:
        """以几何中心为球心的外接球半径。"""
        if self.vertex_count == 0:
            return 1.0
        return float(np.linalg.norm(self.vertices - self.center(), axis=1).max())

    def diagonal(self) -> float:
        return float(np.linalg.norm(self.size())) or 1.0

    # ---------------- 法线 ----------------
    def face_normals(self) -> np.ndarray:
        if self._normals is None:
            self._normals = mu.triangle_normals(self.vertices, self.triangles)
        return self._normals

    def vertex_normals(self) -> np.ndarray:
        if self._vertex_normals is None:
            self._vertex_normals = mu.vertex_normals(
                self.vertices, self.triangles, self.face_normals()
            )
        return self._vertex_normals

    def triangle_areas(self) -> np.ndarray:
        return mu.triangle_areas(self.vertices, self.triangles)

    def area(self) -> float:
        return float(self.triangle_areas().sum())

    def volume(self) -> float:
        """有向体积；负数说明整体法向朝内。"""
        if self.triangle_count == 0:
            return 0.0
        return float(mu.signed_volume(self.vertices, self.triangles))

    # ---------------- 拓扑整理 ----------------
    def weld(self, tolerance: float = 1e-7) -> int:
        """按容差合并重合顶点，返回合并掉的顶点数。

        STL 二进制为每个三角形存 3 个顶点，焊接后可显著压缩数据，
        并使边去重（线框）与连通分量分析成为可能。
        """
        if self.vertex_count == 0:
            return 0
        key = np.round(self.vertices / max(tolerance, 1e-12)).astype(np.int64)
        _, first_index, inverse = np.unique(key, axis=0, return_index=True,
                                            return_inverse=True)
        inverse = np.asarray(inverse).reshape(-1)
        merged = self.vertex_count - len(first_index)
        if merged == 0 and np.array_equal(inverse, np.arange(self.vertex_count)):
            return 0
        new_vertices = self.vertices[first_index]
        new_triangles = inverse[self.triangles]
        # 丢弃退化三角形（焊接后出现重复索引）
        keep = ((new_triangles[:, 0] != new_triangles[:, 1]) &
                (new_triangles[:, 1] != new_triangles[:, 2]) &
                (new_triangles[:, 0] != new_triangles[:, 2]))
        self.vertices = np.ascontiguousarray(new_vertices)
        self.triangles = np.ascontiguousarray(new_triangles[keep])
        self.invalidate()
        return int(merged)

    def remove_degenerate(self, tol_ratio: float = 1e-9) -> int:
        """移除零面积三角形，返回移除数量。"""
        if self.triangle_count == 0:
            return 0
        scale = max(self.diagonal(), 1e-12)
        areas = self.triangle_areas()
        keep = areas > (tol_ratio * scale) ** 2
        removed = int((~keep).sum())
        if removed:
            self.triangles = np.ascontiguousarray(self.triangles[keep])
            self.invalidate()
        return removed

    def remove_unreferenced_vertices(self) -> int:
        """删除未被任何三角形引用的孤立顶点。"""
        if self.vertex_count == 0:
            return 0
        used = np.unique(self.triangles) if self.triangle_count else np.array([], dtype=np.int64)
        if len(used) == self.vertex_count:
            return 0
        remap = np.full(self.vertex_count, -1, dtype=np.int64)
        remap[used] = np.arange(len(used))
        self.vertices = np.ascontiguousarray(self.vertices[used])
        self.triangles = np.ascontiguousarray(remap[self.triangles])
        self.invalidate()
        return int(self.vertex_count - len(used))

    def unique_edges(self) -> np.ndarray:
        """去重后的边（Kx2）。"""
        if self.triangle_count == 0:
            return np.zeros((0, 2), dtype=np.int64)
        tri = self.triangles
        edges = np.concatenate([tri[:, [0, 1]], tri[:, [1, 2]], tri[:, [2, 0]]], axis=0)
        return np.unique(np.sort(edges, axis=1), axis=0)

    def edge_use_counts(self) -> tuple:
        """返回 (unique_edges, counts)，用于边界/非流形统计。"""
        if self.triangle_count == 0:
            return np.zeros((0, 2), dtype=np.int64), np.zeros(0, dtype=np.int64)
        tri = self.triangles
        edges = np.concatenate([tri[:, [0, 1]], tri[:, [1, 2]], tri[:, [2, 0]]], axis=0)
        edges = np.sort(edges, axis=1)
        uniq, counts = np.unique(edges, axis=0, return_counts=True)
        return uniq, counts

    def components(self) -> np.ndarray:
        """按共享顶点划分连通分量，返回每个三角形所属分量编号。"""
        if self._components is None:
            self._components = _triangle_components(self.vertex_count, self.triangles)
        return self._components

    def component_count(self) -> int:
        comps = self.components()
        return int(comps.max() + 1) if len(comps) else 0

    def split_components(self) -> list:
        """拆分为独立网格列表（Extras > STL > Generate Components）。"""
        comps = self.components()
        out = []
        for cid in range(self.component_count()):
            mask = comps == cid
            sub_tris = self.triangles[mask]
            used = np.unique(sub_tris)
            remap = np.full(self.vertex_count, -1, dtype=np.int64)
            remap[used] = np.arange(len(used))
            name = f"{self.name}_part{cid + 1:02d}"
            out.append(Mesh(self.vertices[used], remap[sub_tris], name=name, source=self.source))
        return out

    # ---------------- 修复 ----------------
    def repair(self, tolerance: float = 1e-7, fix_orientation: bool = True,
               drop_loose: bool = True) -> dict:
        """对应 Extras > STL > RepairSurface。

        执行：焊接顶点 -> 去退化面 -> 统一法向 -> 去孤立顶点。
        返回修复前后的对比信息。
        """
        before = (self.vertex_count, self.triangle_count)
        merged = self.weld(tolerance)
        removed = self.remove_degenerate()
        flipped = 0
        if fix_orientation:
            flipped = self.orient_outward()
        loose = self.remove_unreferenced_vertices() if drop_loose else 0
        return {
            "weld_merged": merged,
            "degenerate_removed": removed,
            "triangles_flipped": flipped,
            "loose_removed": loose,
            "before": before,
            "after": (self.vertex_count, self.triangle_count),
        }

    def orient_outward(self) -> int:
        """使封闭网格法向朝外（体积为正）。返回翻面数量。"""
        if self.triangle_count == 0:
            return 0
        if self.volume() < 0:
            self.flip_normals()
            return self.triangle_count
        return 0

    def flip_normals(self) -> None:
        """翻转全部三角形绕向。"""
        self.triangles = np.ascontiguousarray(self.triangles[:, ::-1])
        self.invalidate()

    # ---------------- 变换 ----------------
    def transform(self, matrix: np.ndarray) -> None:
        self.vertices = np.ascontiguousarray(mu.transform_points(matrix, self.vertices))
        self.invalidate()

    def translate(self, offset) -> None:
        self.vertices = np.ascontiguousarray(self.vertices + np.asarray(offset, dtype=np.float64))
        self.invalidate()

    def scale(self, factor, center=None) -> None:
        factor = np.asarray(factor, dtype=np.float64)
        if factor.ndim == 0:
            factor = np.repeat(factor, 3)
        c = self.center() if center is None else np.asarray(center, dtype=np.float64)
        self.vertices = np.ascontiguousarray(c + (self.vertices - c) * factor)
        self.invalidate()

    def rotate(self, axis, angle_rad: float, center=None) -> None:
        c = self.center() if center is None else np.asarray(center, dtype=np.float64)
        r = mu.rotation_matrix(axis, angle_rad)
        m = mu.affine_from_3x3(r, c - r @ c)
        self.transform(m)

    # ---------------- 剖切与相交 ----------------
    def section_polygons(self, plane_point, plane_normal, eps: float = 1e-9) -> list:
        """平面剖切，返回若干条交线（对应 Enable section 的剖切断面）。"""
        if self.triangle_count == 0:
            return []
        n = mu.normalize(np.asarray(plane_normal, dtype=np.float64))
        p0 = np.asarray(plane_point, dtype=np.float64)
        verts = self.vertices
        dist = (verts - p0) @ n
        d = dist[self.triangles]                     # (M,3)
        pos = d > eps
        neg = d < -eps
        candidates = np.nonzero((pos.any(axis=1) & neg.any(axis=1)) |
                                (~pos & ~neg).any(axis=1))[0]
        segments = []
        for i in candidates:
            tri = self.triangles[i]
            di = d[i]
            pts = []
            for a in range(3):
                b = (a + 1) % 3
                da, db = di[a], di[b]
                if abs(da) <= eps:
                    pts.append(verts[tri[a]])
                if (da < -eps and db > eps) or (da > eps and db < -eps):
                    t = da / (da - db)
                    pts.append(verts[tri[a]] + (verts[tri[b]] - verts[tri[a]]) * t)
            if len(pts) < 2:
                continue
            uniq = []
            for p in pts:
                if not any(np.linalg.norm(p - q) < 1e-9 for q in uniq):
                    uniq.append(p)
            if len(uniq) >= 2:
                segments.append((uniq[0], uniq[1]))
        return segments

    def remove_triangles_intersecting(self, other: "Mesh", expand: float = 0.0) -> int:
        """删除与 other 网格相交的三角形（RemoveTriIntersectedChildren）。

        用于布尔仿真结果：去掉被"子对象"（刀具/毛坯）穿过的表面片，
        使可视效果看起来像真实切除。返回删除的三角形数量。
        """
        if self.triangle_count == 0 or other.triangle_count == 0:
            return 0
        a = _TriangleIndex(self.vertices, self.triangles)
        b = _TriangleIndex(other.vertices, other.triangles)
        cell = max(min(a.cell, b.cell), 1e-12)
        b.build_grid(cell)
        candidates = b.query_mask(a.aabb_lo, a.aabb_hi, expand)
        mask = np.zeros(self.triangle_count, dtype=bool)
        a_ids, b_ids = candidates
        for i, j in zip(a_ids, b_ids):
            if mask[i]:
                continue
            if _tri_tri_intersect(self.vertices[self.triangles[i]],
                                  other.vertices[other.triangles[j]]):
                mask[i] = True
        removed = int(mask.sum())
        if removed:
            self.triangles = np.ascontiguousarray(self.triangles[~mask])
            self.invalidate()
        return removed

    # ---------------- 自动着色 ----------------
    def color_by_orientation(self, palette: np.ndarray | None = None) -> np.ndarray:
        """按面法线方向着色（Mesh > Colored / Auto Colors）。

        采用行业常用的"法线贴图色"：RGB = 0.5 + 0.5 * 法线分量，
        再按 Z 分量附加冷暖偏移，使回转体呈现出层次分明的彩色面。
        """
        if palette is not None:
            return palette
        normals = self.face_normals()
        if len(normals) == 0:
            return np.zeros((0, 3), dtype=np.float64)
        rgb = 0.5 + 0.5 * normals
        # 冷暖偏移：竖直方向（+Z）偏暖，向下偏冷
        warm = (normals[:, 2] * 0.5 + 0.5)[:, None]
        rgb = rgb * (0.92 + 0.16 * np.array([1.0, 0.88, 0.72])[None, :] * warm)
        return np.clip(rgb, 0.0, 1.0)

    def color_by_component(self, palette: np.ndarray | None = None) -> np.ndarray:
        """按连通分量着色。"""
        comps = self.components()
        if len(comps) == 0:
            return np.zeros((0, 3), dtype=np.float64)
        if palette is None:
            n = max(int(comps.max()) + 1, 1)
            hue = (np.arange(n) / n * 0.83 + 0.02) % 1.0
            palette = _hsv_to_rgb(hue, 0.68, 0.95)
        return palette[comps % len(palette)]

    def color_by_curvature(self) -> np.ndarray:
        """按曲率（相邻面法线夹角）着色，突出圆角与棱边。"""
        if self.triangle_count == 0:
            return np.zeros((0, 3), dtype=np.float64)
        tri = self.triangles
        edges = np.sort(np.concatenate(
            [tri[:, [0, 1]], tri[:, [1, 2]], tri[:, [2, 0]]], axis=0), axis=1)
        uniq, inverse = np.unique(edges, axis=0, return_inverse=True)
        inverse = np.asarray(inverse).reshape(-1)
        normals = self.face_normals()
        flat = np.concatenate([normals, normals, normals], axis=0)
        accum = np.zeros((len(uniq), 3))
        counts = np.zeros(len(uniq))
        np.add.at(accum, inverse, flat)
        np.add.at(counts, inverse, 1.0)
        counts[counts == 0] = 1.0
        avg = mu.normalize(accum / counts[:, None])
        ang = np.degrees(np.arccos(np.clip(
            np.einsum("ij,ij->i", avg[inverse], flat), -1.0, 1.0)))
        ang = ang.reshape(len(tri), 3).mean(axis=1)      # 每面的平均折角
        t = np.clip(ang / 60.0, 0.0, 1.0)
        rgb = np.empty((len(t), 3))
        rgb[:, 0] = t
        rgb[:, 1] = 1.0 - np.abs(t - 0.5) * 2.0
        rgb[:, 2] = 1.0 - t
        return rgb

    # ---------------- 统计 ----------------
    def statistics(self, deep: bool = True) -> MeshStatistics:
        st = MeshStatistics()
        st.vertices = self.vertex_count
        st.triangles = self.triangle_count
        lo, hi = self.bounds()
        st.size = tuple(float(v) for v in self.size())
        st.center = tuple(float(v) for v in self.center())
        if self.triangle_count == 0:
            st.loose_vertices = self.vertex_count
            return st
        if deep:
            used = np.unique(self.triangles)
            st.loose_vertices = int(self.vertex_count - len(used))
            st.degenerate = int((self.triangle_areas() <= 1e-18).sum())
            _, counts = self.edge_use_counts()
            st.boundary_edges = int((counts == 1).sum())
            st.non_manifold_edges = int((counts > 2).sum())
            st.components = self.component_count()
            st.closed = st.boundary_edges == 0 and st.non_manifold_edges == 0
            st.area = self.area()
            vol = self.volume()
            st.volume = abs(vol)
            st.inverted = vol < 0
        return st

    # ---------------- 采样 / 拾取 ----------------
    def closest_vertex(self, point) -> tuple:
        """返回 (顶点索引, 距离)。"""
        if self.vertex_count == 0:
            return -1, float("inf")
        d = np.linalg.norm(self.vertices - np.asarray(point, dtype=np.float64), axis=1)
        i = int(d.argmin())
        return i, float(d[i])

    def closest_point_on_surface(self, point) -> tuple:
        """返回表面上最近点 (点, 三角形索引, 距离)。"""
        point = np.asarray(point, dtype=np.float64)
        if self.triangle_count == 0:
            return point, -1, float("inf")
        v0 = self.vertices[self.triangles[:, 0]]
        v1 = self.vertices[self.triangles[:, 1]]
        v2 = self.vertices[self.triangles[:, 2]]
        # 质心距离粗筛，再对最近的若干面做精确投影
        cent = (v0 + v1 + v2) / 3.0
        dc = np.linalg.norm(cent - point, axis=1)
        cand = np.argsort(dc)[: max(16, int(len(dc) * 0.02))]
        best = (None, -1, float("inf"))
        for i in cand:
            p, d = _point_triangle(point, v0[i], v1[i], v2[i])
            if d < best[2]:
                best = (p, int(i), float(d))
        return best

    def __repr__(self) -> str:  # pragma: no cover
        return f"<Mesh {self.name!r} V={self.vertex_count} F={self.triangle_count}>"


# --------------------------------------------------------------------------
# 内部工具
# --------------------------------------------------------------------------
def _hsv_to_rgb(h, s, v) -> np.ndarray:
    h = np.asarray(h, dtype=np.float64) % 1.0
    s = np.asarray(s, dtype=np.float64)
    v = np.asarray(v, dtype=np.float64)
    i = np.floor(h * 6.0)
    f = h * 6.0 - i
    p = v * (1.0 - s)
    q = v * (1.0 - f * s)
    t = v * (1.0 - (1.0 - f) * s)
    i = (i % 6).astype(np.int64)
    r = np.select([i == 0, i == 1, i == 2, i == 3, i == 4, i == 5], [v, q, p, p, t, v])
    g = np.select([i == 0, i == 1, i == 2, i == 3, i == 4, i == 5], [t, v, v, q, p, p])
    b = np.select([i == 0, i == 1, i == 2, i == 3, i == 4, i == 5], [p, p, t, v, v, q])
    return np.stack([r, g, b], axis=-1)


def _triangle_components(vertex_count: int, triangles: np.ndarray) -> np.ndarray:
    """并查集求三角形连通分量。"""
    if len(triangles) == 0:
        return np.zeros(0, dtype=np.int64)
    parent = np.arange(vertex_count, dtype=np.int64)

    def find(x: int) -> int:
        root = x
        while parent[root] != root:
            root = parent[root]
        while parent[x] != root:
            parent[x], x = root, parent[x]
        return root

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for t in triangles:
        union(int(t[0]), int(t[1]))
        union(int(t[1]), int(t[2]))
    roots = np.array([find(int(v)) for v in range(vertex_count)], dtype=np.int64)
    _, comp_of_vertex = np.unique(roots, return_inverse=True)
    return comp_of_vertex[triangles[:, 0]].astype(np.int64)


def _point_triangle(p, a, b, c) -> tuple:
    """点到三角形最近点与距离。"""
    ab = b - a
    ac = c - a
    ap = p - a
    d1 = float(np.dot(ab, ap))
    d2 = float(np.dot(ac, ap))
    if d1 <= 0 and d2 <= 0:
        q = a
    else:
        bp = p - b
        d3 = float(np.dot(ab, bp))
        d4 = float(np.dot(ac, bp))
        if d3 >= 0 and d4 <= d3:
            q = b
        else:
            vc = d1 * d4 - d3 * d2
            cp = p - c
            d5 = float(np.dot(ab, cp))
            d6 = float(np.dot(ac, cp))
            if vc <= 0 and d1 >= 0 and d3 <= 0:
                t = d1 / (d1 - d3) if d1 != d3 else 0.0
                q = a + ab * t
            else:
                vb = d5 * d2 - d1 * d6
                if vb <= 0 and d2 >= 0 and d6 <= 0:
                    t = d2 / (d2 - d6) if d2 != d6 else 0.0
                    q = a + ac * t
                else:
                    va = d3 * d6 - d5 * d4
                    if va <= 0 and (d4 - d3) >= 0 and (d5 - d6) >= 0:
                        t = (d4 - d3) / ((d4 - d3) + (d5 - d6))
                        q = b + (c - b) * t
                    else:
                        denom = va + vb + vc
                        if abs(denom) < 1e-18:
                            q = (a + b + c) / 3.0
                        else:
                            v = vb / denom
                            w = vc / denom
                            q = a + ab * v + ac * w
    return q, float(np.linalg.norm(p - q))


class _TriangleIndex:
    """均匀网格空间索引，用于三角形相交的候选筛选。"""

    def __init__(self, vertices: np.ndarray, triangles: np.ndarray):
        self.vertices = vertices
        self.triangles = triangles
        v = vertices[triangles]
        self.aabb_lo = v.min(axis=1)
        self.aabb_hi = v.max(axis=1)
        span = np.linalg.norm(self.aabb_hi - self.aabb_lo, axis=1)
        self.cell = float(np.median(span)) if len(span) else 1.0
        self.grid: dict = {}

    def build_grid(self, cell: float) -> None:
        self.cell = max(cell, 1e-12)
        lo = np.floor(self.aabb_lo / self.cell).astype(np.int64)
        hi = np.floor(self.aabb_hi / self.cell).astype(np.int64)
        for i in range(len(self.triangles)):
            for x in range(lo[i, 0], hi[i, 0] + 1):
                for y in range(lo[i, 1], hi[i, 1] + 1):
                    for z in range(lo[i, 2], hi[i, 2] + 1):
                        self.grid.setdefault((x, y, z), []).append(i)

    def query_mask(self, lo: np.ndarray, hi: np.ndarray, expand: float = 0.0):
        """对每个查询 AABB 找出网格中可能相交的三角形对。"""
        a_ids, b_ids = [], []
        cell = self.cell
        lo_c = np.floor((lo - expand) / cell).astype(np.int64)
        hi_c = np.floor((hi + expand) / cell).astype(np.int64)
        for i in range(len(lo)):
            found = set()
            for x in range(lo_c[i, 0], hi_c[i, 0] + 1):
                for y in range(lo_c[i, 1], hi_c[i, 1] + 1):
                    for z in range(lo_c[i, 2], hi_c[i, 2] + 1):
                        found.update(self.grid.get((x, y, z), ()))
            for j in found:
                a_ids.append(i)
                b_ids.append(j)
        return np.array(a_ids, dtype=np.int64), np.array(b_ids, dtype=np.int64)


def _tri_tri_intersect(t1: np.ndarray, t2: np.ndarray) -> bool:
    """三角形-三角形相交判定（分离轴 + 边对面测试）。"""
    for tri_a, tri_b in ((t1, t2), (t2, t1)):
        n = np.cross(tri_a[1] - tri_a[0], tri_a[2] - tri_a[0])
        nn = float(np.dot(n, n))
        if nn < 1e-20:
            continue
        d = (tri_b - tri_a[0]) @ n
        if np.all(d > 1e-12) or np.all(d < -1e-12):
            return False
    # 边与三角形相交测试
    for tri_a, tri_b in ((t1, t2), (t2, t1)):
        for k in range(3):
            a = tri_b[k]
            b = tri_b[(k + 1) % 3]
            direction = b - a
            length = float(np.linalg.norm(direction))
            if length < 1e-14:
                continue
            hit, _, _ = mu.ray_triangle_intersection(
                a, direction / length, tri_a[0], tri_a[1], tri_a[2])
            if hit:
                return True
    return False
