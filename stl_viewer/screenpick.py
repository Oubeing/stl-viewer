"""屏幕投影拾取 —— 把模型投影到屏幕，用于测量取点。

测量取点要的是「光标下方那个模型表面点」。之前用 GL 深度拾取（离屏 FBO
画两遍 + glReadPixels）来拿它：读回像素对驱动/上下文状态很敏感，一旦不合
预期，点击就整个卡住 —— 用户实测「装上它以后第一个点就点不了」。

这里换成本文件的做法，**完全不碰 GL**：

1. 取点前，把当前视图下所有可见模型的顶点投影到屏幕（纯 numpy 矩阵乘）；
2. 按 16 像素一格，把每个三角形的屏幕包围盒登记进二维格子索引；
3. 点击时只取该像素所在格子里的那几个三角形，做精确的射线-三角形求交，
   取最近命中点。

于是每次点击都是「查一次格子 + 算十几个三角形」，微秒级；索引在视图不变时
复用，只有相机或模型变了才重建一次。得到的点是**真实表面点**，所以两点
距离就是真实三维尺寸，与视角、缩放无关。
"""

from __future__ import annotations

import numpy as np

from . import mathutil as mu

_IDENTITY = np.eye(4, dtype=np.float64)


class ScreenPicker:
    """屏幕投影 + 二维格子索引的表面取点器（纯 CPU）。"""

    #: 格子边长（逻辑像素）
    CELL = 16.0
    #: 单个三角形最多登记多少个格子；超过就放进「大三角形」列表按需全测
    MAX_CELLS_PER_TRIANGLE = 96

    def __init__(self, cell: float = CELL):
        self.cell = float(cell)
        self.last_build_ms = 0.0
        self.triangle_count = 0
        self._key = None
        self._verts = None          # (V,3) 世界坐标
        self._screen = None         # (V,2) 屏幕坐标
        self._tris = None           # (T,3) 顶点索引
        self._cell_start = None     # (gw*gh+1,) 每格的三角形区间起点
        self._cell_tris = None      # (M,) 按格子排好序的三角形索引
        self._big = None            # (B,) 屏幕包围盒过大、每次都要测的三角形
        self._gw = 1
        self._gh = 1
        self._width = 0
        self._height = 0

    # ---------------- 索引 ----------------
    def invalidate(self) -> None:
        self._key = None
        self._verts = None
        self._screen = None
        self._tris = None
        self._cell_start = None
        self._cell_tris = None
        self._big = None

    @property
    def ready(self) -> bool:
        return self._tris is not None and self._key is not None

    def _key_for(self, scene, camera):
        """视图 + 场景指纹：任何一个变了就要重建索引。"""
        vm = camera.view_matrix()
        pm = camera.projection_matrix()
        objs = []
        for o in scene.objects:
            if not o.visible or o.mesh.triangle_count == 0:
                continue
            mat = getattr(o, "matrix", None)
            m = _IDENTITY if mat is None else np.asarray(mat, dtype=np.float64)
            objs.append((id(o.mesh.vertices),
                         int(o.mesh.triangle_count),
                         round(float(np.abs(m).sum()), 9)))
        return (int(camera.width), int(camera.height),
                tuple(np.round(vm, 9).ravel()),
                tuple(np.round(pm, 9).ravel()),
                tuple(objs))

    def ensure(self, scene, camera) -> bool:
        """视图/场景没变就复用索引，变了就重建。返回索引是否可用。"""
        key = self._key_for(scene, camera)
        if key == self._key and self._tris is not None:
            return True
        return self._build(scene, camera, key)

    def _build(self, scene, camera, key) -> bool:
        import time

        t0 = time.perf_counter()
        verts_parts, tris_parts = [], []
        offset = 0
        for o in scene.objects:
            if not o.visible or o.mesh.triangle_count == 0:
                continue
            mesh = o.mesh
            mat = getattr(o, "matrix", None)
            m = _IDENTITY if mat is None else np.asarray(mat, dtype=np.float64)
            wv = mesh.vertices if np.allclose(m, _IDENTITY) \
                else mu.transform_points(m, mesh.vertices)
            verts_parts.append(wv)
            tris_parts.append(mesh.triangles + offset)
            offset += len(wv)
        if not verts_parts:
            self.invalidate()
            self.last_build_ms = (time.perf_counter() - t0) * 1000.0
            return False

        verts = verts_parts[0] if len(verts_parts) == 1 \
            else np.vstack(verts_parts)
        tris = tris_parts[0] if len(tris_parts) == 1 \
            else np.vstack(tris_parts)
        tris = np.asarray(tris, dtype=np.int32)

        screen, _, valid = camera.project(verts)
        # 逐三角形算屏幕包围盒：分块处理，避免 80 万面模型一次吃掉上百 MB
        width = max(int(camera.width), 1)
        height = max(int(camera.height), 1)
        cell = self.cell
        gw = max(1, int(np.ceil(width / cell)) + 1)
        gh = max(1, int(np.ceil(height / cell)) + 1)

        chunk = 150_000
        cell_parts = []
        tri_parts = []
        big_parts = []
        for start in range(0, len(tris), chunk):
            block = tris[start:start + chunk]
            sub = screen[block]                      # (k,3,2)
            ok = valid[block].all(axis=1) & np.isfinite(sub).all(axis=(1, 2))
            if not ok.any():
                continue
            x0 = sub[:, :, 0].min(axis=1)
            x1 = sub[:, :, 0].max(axis=1)
            y0 = sub[:, :, 1].min(axis=1)
            y1 = sub[:, :, 1].max(axis=1)
            ok &= (x1 >= -1.0) & (x0 <= width + 1.0) \
                & (y1 >= -1.0) & (y0 <= height + 1.0)
            if not ok.any():
                continue
            cx0 = np.clip((x0 / cell).astype(np.int32), 0, gw - 1)
            cx1 = np.clip((x1 / cell).astype(np.int32), 0, gw - 1)
            cy0 = np.clip((y0 / cell).astype(np.int32), 0, gh - 1)
            cy1 = np.clip((y1 / cell).astype(np.int32), 0, gh - 1)
            nx = cx1 - cx0 + 1
            ny = cy1 - cy0 + 1
            cnt = nx * ny
            small = ok & (cnt <= self.MAX_CELLS_PER_TRIANGLE)
            if small.any():
                idx = np.nonzero(small)[0]
                counts = cnt[idx]
                total = int(counts.sum())
                rep = np.repeat(idx, counts)
                starts = np.repeat(np.cumsum(counts) - counts, counts)
                local = np.arange(total, dtype=np.int64) - starts
                nx_rep = np.repeat(nx[idx], counts)
                dx = local % nx_rep
                dy = local // nx_rep
                cells = ((cy0[rep] + dy).astype(np.int64) * gw
                         + (cx0[rep] + dx))
                tri_parts.append((start + rep).astype(np.int32))
                cell_parts.append(cells.astype(np.int64))
            heavy = ok & ~small
            if heavy.any():
                big_parts.append((start + np.nonzero(heavy)[0]).astype(np.int32))

        if cell_parts:
            cells = np.concatenate(cell_parts)
            owners = np.concatenate(tri_parts)
            order = np.argsort(cells, kind="stable")
            cells = cells[order]
            owners = owners[order]
            counts = np.bincount(cells, minlength=gw * gh + 1)
            start_index = np.zeros(gw * gh + 1, dtype=np.int64)
            np.cumsum(counts[:gw * gh], out=start_index[1:])
            cell_tris = owners
        else:
            start_index = np.zeros(gw * gh + 1, dtype=np.int64)
            cell_tris = np.zeros(0, dtype=np.int32)

        self._verts = verts
        self._screen = screen
        self._tris = tris
        self._cell_start = start_index
        self._cell_tris = cell_tris
        self._big = np.concatenate(big_parts) if big_parts \
            else np.zeros(0, dtype=np.int32)
        self._gw, self._gh = gw, gh
        self._width, self._height = width, height
        self._key = key
        self.triangle_count = len(tris)
        self.last_build_ms = (time.perf_counter() - t0) * 1000.0
        return True

    # ---------------- 取点 ----------------
    def is_current(self, scene, camera) -> bool:
        """索引是否与当前视图 / 场景一致（不重建、不阻塞）。"""
        if not self.ready:
            return False
        try:
            return self._key_for(scene, camera) == self._key
        except Exception:
            return False

    def pick(self, scene, camera, x: float, y: float, pad: float = 1.5,
             allow_build: bool = True) -> np.ndarray | None:
        """(x, y) 像素下方的模型表面点（世界坐标）；没打中返回 None。

        ``allow_build=False`` 时**绝不重建索引**：只查询与当前视图匹配的
        现成索引，过期就直接返回 None。交互路径（点击 / 悬停）一律用它，
        这样大模型转完视角马上点击也不会在 GUI 线程里等一次全量重建；
        重建交给后台线程提前做好。
        """
        try:
            if allow_build:
                if not self.ensure(scene, camera):
                    return None
            elif not self.is_current(scene, camera):
                return None
            return self._query(camera, x, y, pad)
        except Exception:
            # 任何异常都当作「没打中」，让调用方退回视图平面，
            # 绝不能因为取点失败让点击没有反应。
            return None

    def _candidates(self, x, y, pad):
        """该像素所在格子内、屏幕包围盒（外扩 ``pad``）覆盖它的三角形。"""
        cell = self.cell
        cx = int(min(max(np.floor(x / cell), 0), self._gw - 1))
        cy = int(min(max(np.floor(y / cell), 0), self._gh - 1))
        idx = cy * self._gw + cx
        s = int(self._cell_start[idx])
        e = int(self._cell_start[idx + 1])
        cand = self._cell_tris[s:e]
        if self._big is not None and len(self._big):
            cand = np.concatenate([cand, self._big]) if len(cand) \
                else self._big
        if len(cand) == 0:
            return None
        sp = self._screen[self._tris[cand]]
        inside = ((sp[:, :, 0].min(axis=1) - pad <= x)
                  & (sp[:, :, 0].max(axis=1) + pad >= x)
                  & (sp[:, :, 1].min(axis=1) - pad <= y)
                  & (sp[:, :, 1].max(axis=1) + pad >= y))
        if not inside.any():
            return None
        return cand[inside]

    def _intersect(self, camera, origin, direction, cand):
        """候选三角形上的精确射线求交，返回最近命中的世界点。"""
        best_t = None
        best_p = None
        step = 20_000                 # 分块：极端密集处也不会一次吃掉大内存
        for start in range(0, len(cand), step):
            block = cand[start:start + step]
            p = self._verts[self._tris[block]]          # (k,3,3)
            v0 = p[:, 0]
            e1 = p[:, 1] - v0
            e2 = p[:, 2] - v0
            pvec = np.cross(np.broadcast_to(direction, e2.shape), e2)
            det = np.einsum("ij,ij->i", e1, pvec)
            ok = np.abs(det) > 1e-14
            if not ok.any():
                continue
            inv = np.zeros_like(det)
            inv[ok] = 1.0 / det[ok]
            tvec = origin - v0
            u = np.einsum("ij,ij->i", tvec, pvec) * inv
            ok &= (u >= -1e-7) & (u <= 1.0 + 1e-7)
            qvec = np.cross(tvec, e1)
            v = np.einsum("ij,ij->i", np.broadcast_to(direction, e1.shape),
                          qvec) * inv
            ok &= (v >= -1e-7) & (u + v <= 1.0 + 1e-7)
            t = np.einsum("ij,ij->i", e2, qvec) * inv
            ok &= t > 1e-9
            if not ok.any():
                continue
            hit = np.nonzero(ok)[0]
            k = int(hit[np.argmin(t[hit])])
            if best_t is None or float(t[k]) < best_t:
                best_t = float(t[k])
                best_p = origin + direction * best_t
        return best_p

    def _snap_edge(self, camera, x, y, cand, tol_px):
        """候选三角形里离光标最近的「边上一点」（屏幕距离 ≤ tol_px）。

        光栅化时轮廓最外一圈像素可能刚好压在几何边缘的 0.2~0.5 像素之外：
        精确求交会说「没打中」，而画面上那一点明明还是模型。这时与其退回
        视图平面（会丢掉真实深度），不如吸附到最近的可见边缘上 —— 误差
        不到一个像素，但深度是模型的真实深度。
        """
        if cand is None or len(cand) == 0:
            return None
        tri = self._tris[cand]
        sp = self._screen[tri]                       # (k,3,2)
        pts3 = self._verts[tri]                      # (k,3,3)
        click = np.array([x, y], dtype=np.float64)
        best = None
        edges = ((0, 1), (1, 2), (2, 0))
        for i, j in edges:
            a = sp[:, i]
            b = sp[:, j]
            ab = b - a
            ap = click - a
            denom = np.einsum("ij,ij->i", ab, ab)
            denom = np.where(denom < 1e-12, 1e-12, denom)
            t = np.clip(np.einsum("ij,ij->i", ap, ab) / denom, 0.0, 1.0)
            q = a + ab * t[:, None]
            d = np.linalg.norm(click - q, axis=1)
            k = int(np.argmin(d))
            if d[k] <= tol_px and (best is None or d[k] < best[0]):
                w = np.zeros(3)
                w[i] = 1.0 - t[k]
                w[j] = t[k]
                best = (float(d[k]), pts3[k, 0] * w[0] + pts3[k, 1] * w[1]
                        + pts3[k, 2] * w[2])
        return None if best is None else best[1]

    def _query(self, camera, x, y, pad):
        if self._tris is None or len(self._tris) == 0:
            return None
        if not np.isfinite(x) or not np.isfinite(y):
            return None
        origin, direction = camera.ray(x, y)
        # 先按「贴边」容差找；只有确实没打中才放宽到 gl 反锯齿带来的
        # 一圈像素级边缘（这样密集模型上的候选三角形数量小一个量级）。
        for tol in (0.75, max(float(pad), 2.0)):
            cand = self._candidates(x, y, tol)
            if cand is None:
                continue
            hit = self._intersect(camera, origin, direction, cand)
            if hit is not None:
                return hit
        # 仍没打中：可能是点压在轮廓最外一圈像素上（几何边缘在光标外
        # 半个像素以内）。吸附到最近的可见边缘，保住真实深度。
        tol = max(float(pad), 3.0)
        cand = self._candidates(x, y, tol)
        return self._snap_edge(camera, x, y, cand, tol)
