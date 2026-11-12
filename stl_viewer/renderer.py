"""软件光栅化渲染器。

除硬件渲染（OpenGL）之外，此处用 numpy 实现等价能力的纯软件渲染器：

* 三角形批量光栅化 + Z 缓冲（遮挡关系正确）
* 线框（边去重 + 批量采样画线，与实体共用 Z 缓冲）
* 光照（环境光 / 漫反射 / 高光，支持双面受光 "Back and Front"）
* 透明混合（Transparent，按深度由远及近混合）
* 坐标轴（箭头与 X/Y/Z 文字始终留在画布内）、地面网格、贯穿画布的十字
  准线、测量标注（内置 5x7 点阵字体）
* 背景纯色 / 渐变 / 全透明（透明背景导出）
* 超采样抗锯齿（Save Multi Res Picture）
"""

from __future__ import annotations

import time
from dataclasses import dataclass, fields

import numpy as np

from . import font5x7
from . import mathutil as mu
from .camera import Camera
from .lighting import LightSettings
from .scene import Scene
from .settings import ViewOptions

AXIS_COLORS = (
    (0.86, 0.24, 0.22),   # X 红
    (0.34, 0.72, 0.32),   # Y 绿
    (0.26, 0.52, 0.90),   # Z 蓝
)
AXIS_NAMES = ("X", "Y", "Z")

PIXEL_LIMIT = 2_000_000            # 单批次最大像素数（控制峰值内存）
PIXEL_LIMIT_GOURAUD = 300_000
# 单批次最大三角形数。每批约 40 次 numpy 调用，批数过多时调用开销会超过
# 真正的光栅化计算（十万面模型实测：56 批 / 152ms，其中 74ms 是分批开销）。
MAX_TRIS_PER_BATCH = 16384
EDGE_DEPTH_BIAS = 3e-4             # 边线相对面片的深度偏移


@dataclass
class RenderOptions:
    """一次渲染的完整参数（与 View options / Mesh 菜单一一对应）。"""

    wire: bool = False
    solid: bool = True
    color_mode: str = "auto"
    transparent: bool = False
    show_axis: bool = True
    axis_labels: bool = True
    axis_scale: float = 0.6
    show_grid: bool = False
    grid_scale: float = 1.0
    crosshair: bool = False
    point_size: int = 1
    line_width: int = 1
    back_and_front: bool = False       # False = 剔除背面（封闭实体更干净、更快）
    show_edges: bool = False
    edge_color: tuple = (0.08, 0.09, 0.11)
    background: tuple = (0.16, 0.17, 0.20)
    background_top: tuple = (0.30, 0.33, 0.38)
    gradient_background: bool = False
    transparent_background: bool = False
    supersample: int = 1
    gouraud: bool = False
    cull_backfaces: bool = True
    shade: bool = True
    draw_measurements: bool = True

    @classmethod
    def from_view_options(cls, view: ViewOptions, **overrides) -> "RenderOptions":
        known = {f.name for f in fields(cls)}
        data = {k: v for k, v in view.state().items() if k in known}
        data.update({k: v for k, v in overrides.items() if k in known})
        return cls(**data)


class SoftwareRenderer:
    """纯 numpy 实现的 3D 渲染器。"""

    def __init__(self, options: RenderOptions | None = None):
        self.options = options or RenderOptions()
        self.width = 0
        self.height = 0
        self._color: np.ndarray | None = None
        self._depth: np.ndarray | None = None
        self._alpha: np.ndarray | None = None
        self._buf_shape: tuple | None = None
        self._ramp: np.ndarray | None = None
        self._stats: dict = {}
        self.last_stats: dict = {}

    # ==================================================================
    # 对外接口
    # ==================================================================
    def render(self, scene: Scene, camera: Camera, size: tuple | None = None,
               options: RenderOptions | None = None,
               light: LightSettings | None = None,
               measurements: list | None = None) -> np.ndarray:
        """渲染并返回 uint8 图像 (H,W,3)；透明背景时为 (H,W,4)。"""
        opts = options or self.options
        light = light or LightSettings()
        w, h = (camera.width, camera.height) if size is None else size
        w, h = max(int(w), 1), max(int(h), 1)
        ss = max(int(opts.supersample), 1)

        start = time.perf_counter()
        old_w, old_h = camera.width, camera.height
        try:
            # 视口必须与本次渲染使用的像素尺寸一致，否则投影与位图不匹配，
            # 会出现模型偏移 / 裁切（拖拽降分辨率渲染时尤其明显）。
            camera.set_viewport(w * ss, h * ss)
            image = self._render_core(scene, camera, light, opts, measurements or [],
                                      w * ss, h * ss)
        finally:
            camera.set_viewport(old_w, old_h)
        if ss > 1:
            image = _downsample(image, w, h)
        self.last_stats = {"size": (w, h), "supersample": ss,
                           "seconds": time.perf_counter() - start, **self._stats}
        return image

    def render_to_file(self, path: str, scene: Scene, camera: Camera,
                       options: RenderOptions | None = None,
                       light: LightSettings | None = None,
                       measurements: list | None = None,
                       double_resolution: bool = False) -> dict:
        """对应 保存图片 / 保存多分辨率图片 / 透明背景导出。"""
        from .png import write_png

        opts = options or self.options
        w, h = camera.width, camera.height
        if double_resolution:
            w, h = w * 2, h * 2
        image = self.render(scene, camera, size=(w, h), options=opts, light=light,
                            measurements=measurements)
        write_png(path, image)
        return {"path": path, "size": (w, h), **self.last_stats}

    # ==================================================================
    # 主流程
    # ==================================================================
    def _ensure_buffers(self, width: int, height: int) -> None:
        """按尺寸缓存 color/depth/alpha 缓冲，避免每帧重新分配整幅图像。

        ss=2 时一次渲染要动 300 万像素，反复 malloc/free 数十 MB 是主要开销之一。
        """
        shape = (height, width)
        if self._buf_shape != shape:
            self._color = np.empty((height, width, 3), dtype=np.float32)
            self._depth = np.empty(shape, dtype=np.float32)
            self._alpha = np.empty(shape, dtype=np.float32)
            self._buf_shape = shape
            self._ramp = None
        self._depth.fill(np.inf)
        self._alpha.fill(0.0)

    def _render_core(self, scene: Scene, camera: Camera, light: LightSettings,
                     opts: RenderOptions, measurements: list,
                     width: int, height: int) -> np.ndarray:
        self.width, self.height = width, height
        self._ensure_buffers(width, height)
        self._stats = {"objects": 0, "triangles": 0, "edges": 0, "batches": 0}

        self._fill_background(opts)

        objects = [o for o in scene.objects if o.visible and o.mesh.triangle_count > 0]
        opaque = [o for o in objects
                  if not (o.transparent or opts.transparent) and not o.always_on_top]
        on_top = [o for o in objects
                  if not (o.transparent or opts.transparent) and o.always_on_top]
        transparent = [o for o in objects if (o.transparent or opts.transparent)]

        # 空场景不画网格/坐标轴：没有模型时 _scene_scale 会退化成 100，
        # 画出的一对长轴会占满画布，看起来像凭空多出来的 objects。
        if opts.show_grid and objects:
            self._draw_grid(scene, camera, opts)

        for obj in opaque + on_top:
            self._draw_object(obj, camera, light, opts, scene, transparent_pass=False)

        if transparent:
            # 画家算法：远的对象先画
            transparent.sort(key=lambda o: _object_depth(o, camera, self) )
            for obj in transparent:
                self._draw_object(obj, camera, light, opts, scene, transparent_pass=True)

        if opts.show_axis and objects:
            self._draw_axes(scene, camera, opts)

        if measurements and opts.draw_measurements:
            self._draw_measurements(measurements, camera)

        if opts.crosshair:
            self._draw_crosshair(opts)

        # 原地裁剪/量化，避免再复制一份全分辨率的中间图像
        np.clip(self._color, 0.0, 1.0, out=self._color)
        self._color *= 255.0
        self._color += 0.5
        np.clip(self._color, 0.0, 255.0, out=self._color)
        out = self._color.astype(np.uint8)
        if opts.transparent_background:
            alpha = np.clip(self._alpha, 0.0, 1.0)[:, :, None]
            out = np.concatenate([out, (alpha * 255.0 + 0.5).astype(np.uint8)], axis=2)
        return out

    # ==================================================================
    # 背景
    # ==================================================================
    def _fill_background(self, opts: RenderOptions) -> None:
        if opts.transparent_background:
            self._color[...] = 0.0
            return
        bottom = np.asarray(opts.background, dtype=np.float32)
        self._color[...] = bottom
        if opts.gradient_background:
            top = np.asarray(opts.background_top, dtype=np.float32)
            if self._ramp is None or self._ramp.shape[0] != self.height:
                self._ramp = np.linspace(0.0, 1.0, self.height,
                                         dtype=np.float32)[:, None, None]
            ramp = self._ramp
            self._color *= ramp
            self._color += top * (1.0 - ramp)

    # ==================================================================
    # 单个对象
    # ==================================================================
    def _object_matrix(self, obj) -> np.ndarray:
        matrix = getattr(obj, "matrix", None)
        return np.asarray(matrix, dtype=np.float64) if matrix is not None else np.eye(4)

    def _draw_object(self, obj, camera: Camera, light: LightSettings,
                     opts: RenderOptions, scene: Scene, transparent_pass: bool) -> None:
        mesh = obj.mesh
        if mesh.triangle_count == 0:
            return
        model = self._object_matrix(obj)
        mvp_cam = camera.view_matrix() @ model
        verts_cam = mu.transform_points(mvp_cam, mesh.vertices)
        normal_matrix = np.linalg.inv(mvp_cam[:3, :3]).T

        near = camera.distance_range()[0]
        triangles = mesh.triangles
        src_face = np.arange(mesh.triangle_count, dtype=np.int64)
        if (verts_cam[:, 2] > -near).any():
            triangles, verts_cam, src_face = _clip_triangles_near(verts_cam, triangles, near)
        if len(triangles) == 0:
            return

        self._stats["objects"] += 1
        self._stats["triangles"] += len(triangles)

        screen, ndc_z, _ = camera.project_camera_space(verts_cam)
        alpha = float(obj.alpha) if transparent_pass else 1.0
        wire_only = (obj.wire or (opts.wire and not opts.solid)) and not opts.show_edges

        if wire_only:
            self._draw_mesh_wire(mesh, triangles, screen, ndc_z, obj, opts, alpha)
            return

        self._draw_mesh_solid(mesh, triangles, src_face, screen, ndc_z, verts_cam,
                              normal_matrix, obj, light, opts, alpha, transparent_pass)

    # ------------------------------------------------------------------
    # 实体填充
    # ------------------------------------------------------------------
    def _draw_mesh_solid(self, mesh, tris_idx, src_face, screen, ndc_z, verts_cam,
                         normal_matrix, obj, light: LightSettings, opts: RenderOptions,
                         alpha: float, transparent_pass: bool) -> None:
        tris_screen = screen[tris_idx]
        tris_depth = ndc_z[tris_idx]

        v0 = verts_cam[tris_idx[:, 0]]
        v1 = verts_cam[tris_idx[:, 1]]
        v2 = verts_cam[tris_idx[:, 2]]
        face_n = mu.normalize(np.cross(v1 - v0, v2 - v0), axis=1)
        front = face_n[:, 2] > 0.0

        cull = (opts.cull_backfaces and not opts.back_and_front and
                not transparent_pass)
        if cull:
            if not front.any():
                return
            tris_screen = tris_screen[front]
            tris_depth = tris_depth[front]
            face_n = face_n[front]
            tris_idx = tris_idx[front]
            src_face = src_face[front]
            if len(tris_screen) == 0:
                return

        # ---- 基础色 ----
        base = obj.face_colors(opts.color_mode)
        if len(base) == mesh.triangle_count:
            colors = np.asarray(base[src_face], dtype=np.float32)
        else:
            colors = np.tile(np.asarray(obj.color, dtype=np.float32),
                             (len(tris_screen), 1))

        # ---- 光照 ----
        if opts.shade:
            intensity = light.shade(face_n).astype(np.float32)
        else:
            intensity = np.ones((len(face_n), 3), dtype=np.float32)
        rgb = np.clip(colors * intensity, 0.0, 1.0)

        # ---- 顶点插值着色（仅"整件同色"模式才有意义）----
        vertex_colors = None
        if opts.gouraud and obj.flat_color(opts.color_mode):
            # 顶点法线插值着色：法线按逆转置矩阵变换到相机空间
            vn_cam = mu.normalize(mesh.vertex_normals() @ np.asarray(normal_matrix).T)
            if opts.shade:
                vshade = light.shade(vn_cam).astype(np.float32)
            else:
                vshade = np.ones((len(vn_cam), 3), dtype=np.float32)
            vcols = np.asarray(obj.color, dtype=np.float32)[None, :] * vshade
            vertex_colors = np.ascontiguousarray(vcols, dtype=np.float32)[tris_idx]

        # ---- 透明对象：按三角形深度排序（远 -> 近）----
        if transparent_pass:
            cent_z = verts_cam[tris_idx].mean(axis=1)[:, 2]
            order = np.argsort(cent_z)          # z 越小越远
            tris_screen = tris_screen[order]
            tris_depth = tris_depth[order]
            rgb = rgb[order]
            if vertex_colors is not None:
                vertex_colors = vertex_colors[order]

        self._rasterize(tris_screen, tris_depth, rgb, alpha=alpha,
                        write_depth=not transparent_pass, blend=transparent_pass,
                        vertex_colors=vertex_colors)

        if opts.show_edges:
            edges = np.concatenate([tris_idx[:, [0, 1]], tris_idx[:, [1, 2]],
                                    tris_idx[:, [2, 0]]], axis=0)
            if len(edges) > 300000:
                step = int(np.ceil(len(edges) / 300000))
                edges = edges[::step]
            self._stats["edges"] += len(edges)
            self._draw_edges(screen, ndc_z, edges,
                             np.asarray(opts.edge_color, dtype=np.float32),
                             max(int(opts.line_width), 1), 1.0,
                             depth_bias=EDGE_DEPTH_BIAS)

    # ------------------------------------------------------------------
    # 线框
    # ------------------------------------------------------------------
    def _draw_mesh_wire(self, mesh, tris_idx, screen, ndc_z, obj, opts, alpha) -> None:
        """纯线框显示：边去重后批量绘制。"""
        if len(tris_idx) * 3 > 300000:
            edges = np.concatenate([tris_idx[:, [0, 1]], tris_idx[:, [1, 2]],
                                    tris_idx[:, [2, 0]]], axis=0)
            step = int(np.ceil(len(edges) / 300000))
            edges = edges[::step]
        else:
            local = np.unique(tris_idx.ravel())
            remap = np.full(len(screen), -1, dtype=np.int64)
            remap[local] = np.arange(len(local))
            sub = remap[tris_idx]
            edges_local = np.unique(np.sort(np.concatenate(
                [sub[:, [0, 1]], sub[:, [1, 2]], sub[:, [2, 0]]], axis=0), axis=1), axis=0)
            edges = local[edges_local]
        color = np.clip(np.asarray(obj.color, dtype=np.float32) * 1.05, 0.0, 1.0)
        self._stats["edges"] += len(edges)
        self._draw_edges(screen, ndc_z, edges, color, max(int(opts.line_width), 1), alpha)
        if opts.point_size > 1:
            self._draw_points(screen, ndc_z, np.unique(edges.ravel()), color,
                              opts.point_size)

    # ==================================================================
    # 光栅化核心
    # ==================================================================
    def _rasterize(self, tris_screen, tris_depth, colors, alpha: float = 1.0,
                   write_depth: bool = True, blend: bool = False,
                   vertex_colors: np.ndarray | None = None) -> None:
        W, H = self.width, self.height
        if len(tris_screen) == 0:
            return
        s = np.asarray(tris_screen, dtype=np.float32)
        z = np.asarray(tris_depth, dtype=np.float32)
        col = np.asarray(colors, dtype=np.float32)

        area = ((s[:, 1, 0] - s[:, 0, 0]) * (s[:, 2, 1] - s[:, 0, 1]) -
                (s[:, 1, 1] - s[:, 0, 1]) * (s[:, 2, 0] - s[:, 0, 0]))
        good = np.abs(area) > 1e-9
        if not good.any():
            return
        s, z, col, area = s[good], z[good], col[good], area[good]
        if vertex_colors is not None:
            vertex_colors = vertex_colors[good]

        xmin = np.maximum(np.floor(s[:, :, 0].min(axis=1)), 0).astype(np.int64)
        xmax = np.minimum(np.ceil(s[:, :, 0].max(axis=1)), W - 1).astype(np.int64)
        ymin = np.maximum(np.floor(s[:, :, 1].min(axis=1)), 0).astype(np.int64)
        ymax = np.minimum(np.ceil(s[:, :, 1].max(axis=1)), H - 1).astype(np.int64)
        bw = xmax - xmin + 1
        bh = ymax - ymin + 1
        valid = (bw > 0) & (bh > 0)
        if not valid.any():
            return
        s, z, col, area = s[valid], z[valid], col[valid], area[valid]
        xmin, xmax, ymin, ymax = xmin[valid], xmax[valid], ymin[valid], ymax[valid]
        bw, bh = bw[valid], bh[valid]
        if vertex_colors is not None:
            vertex_colors = vertex_colors[valid]

        limit = PIXEL_LIMIT_GOURAUD if vertex_colors is not None else PIXEL_LIMIT
        limit = max(limit, W * H)
        # 三角形按 NDC 深度从远到近排序（NDC 中 z 越大越远）
        depth_key = z.mean(axis=1)

        if blend or not write_depth:
            # 透明混合依赖全局"远 -> 近"的顺序，只能按深度分批
            chunks = _depth_ordered_chunks(bw, bh, depth_key, limit)
        else:
            # 不透明分批按 bbox 尺寸分桶减少空洞；批内仍必须按"远 -> 近"
            # 排列。同一批里多个三角形常覆盖同一个像素，而 numpy 的
            # color_flat[idx] = rgb 对重复 idx 只保留最后一次写入 —— 全局
            # Z 缓冲只能挡住「之前批次」的像素，管不了同一批内的覆盖。
            chunks = _size_bucketed_chunks(bw, bh, limit, depth_key)

        vcol = vertex_colors
        for mw, mh, sub in chunks:
            self._rasterize_batch(sub, s, z, col, area, xmin, xmax, ymin, ymax,
                                  mw, mh, alpha, write_depth, blend,
                                  None if vcol is None else vcol[sub])

    def _rasterize_batch(self, sub, s, z, col, area, xmin, xmax, ymin, ymax,
                         mw, mh, alpha, write_depth, blend, vcol) -> None:
        """光栅化一批三角形（边函数与深度均采用线性增量形式）。

        约定：``sub`` 已按"远 -> 近"排序，因此同一像素被多个三角形覆盖时，
        后写入的（更近的）三角形自然获胜，无需对像素排序。
        """
        W = self.width
        B = len(sub)
        self._stats["batches"] += 1

        xs = xmin[sub][:, None].astype(np.int32) + \
            np.arange(mw, dtype=np.int32)[None, :]        # (B,mw)
        ys = ymin[sub][:, None].astype(np.int32) + \
            np.arange(mh, dtype=np.int32)[None, :]        # (B,mh)

        ax = s[sub, 0, 0][:, None, None]
        ay = s[sub, 0, 1][:, None, None]
        bx = s[sub, 1, 0][:, None, None]
        by = s[sub, 1, 1][:, None, None]
        cx = s[sub, 2, 0][:, None, None]
        cy = s[sub, 2, 1][:, None, None]

        Yc = ys[:, :, None].astype(np.float32)                     # (B,mh,1)
        X0 = xmin[sub].astype(np.float32)[:, None, None]           # (B,1,1)
        sign = np.where(area[sub] >= 0, 1.0, -1.0).astype(np.float32)[:, None, None]
        # c0/c1/c2（以及 d0/d1/d2）已经乘过 sign，因此重心坐标是
        # w_i = c_i / |area|，而不是 c_i / area。用带符号的 area 作分母时，
        # 反向绕序三角形（约占一半）的深度与顶点色会被整体取反：远处的面
        # 反而通过深度测试盖住近处的面，模型看起来像被穿透/半透明。
        inv_area = (1.0 / np.abs(area[sub])).astype(np.float32)[:, None, None]

        # 三条边在每行起点 (X0, y) 处的值，已并入绕向符号
        c0 = ((cx - bx) * (Yc - by) - (cy - by) * (X0 - bx)) * sign
        c1 = ((ax - cx) * (Yc - cy) - (ay - cy) * (X0 - cx)) * sign
        c2 = ((bx - ax) * (Yc - ay) - (by - ay) * (X0 - ax)) * sign
        # 沿 +x 方向的增量
        d0 = -(cy - by) * sign
        d1 = -(ay - cy) * sign
        d2 = -(by - ay) * sign

        j = np.arange(mw, dtype=np.float32)[None, None, :]
        edge = c0 + d0 * j
        tmp = c1 + d1 * j
        np.minimum(edge, tmp, out=edge)
        tmp = c2 + d2 * j
        np.minimum(edge, tmp, out=edge)
        inside = edge >= 0
        inside &= (xs <= xmax[sub][:, None])[:, None, :]
        inside &= (ys <= ymax[sub][:, None])[:, :, None]
        del edge, tmp

        # 先压成一维再取非零下标：对 (B,mh,mw) 直接 nonzero 反而更慢
        sel = np.flatnonzero(inside.ravel())
        del inside
        if sel.size == 0:
            return
        # 由线性下标反解出 (批次, 行, 列)
        tri, rem = np.divmod(sel, mh * mw)
        ry, rx = np.divmod(rem, mw)

        # 命中像素的线性下标。row_base 只在 B 个元素中随机取值（缓存友好），
        # 比按 (B*mh) 规模去 ys.ravel() 里取快约 3 倍。
        row_base = ymin[sub].astype(np.int64) * W + xmin[sub].astype(np.int64)
        idx = row_base[tri] + ry * W + rx
        ri = tri * mh + ry                      # col_dep 的线性下标

        # 深度也随 x 线性变化（硬件 Z 缓冲即按屏幕空间线性插值）。
        # 只在实际命中的像素上求值，避免再造一份 (B,mh,mw) 的整幅临时数组。
        z0 = z[sub, 0][:, None, None]
        z1 = z[sub, 1][:, None, None]
        z2 = z[sub, 2][:, None, None]
        step_dep = ((z0 * d0 + z1 * d1 + z2 * d2) * inv_area).ravel()   # (B,)
        col_dep = ((z0 * c0 + z1 * c1 + z2 * c2) * inv_area).ravel()    # (B*mh,)
        dep = col_dep[ri] + step_dep[tri] * rx

        if vcol is None:
            rgb = col[sub][tri]
        else:
            i0 = (c0 * inv_area).ravel()        # (B*mh,)
            i1 = (c1 * inv_area).ravel()
            i2 = (c2 * inv_area).ravel()
            s0 = (d0 * inv_area).ravel()        # (B,)
            s1 = (d1 * inv_area).ravel()
            s2 = (d2 * inv_area).ravel()
            w0 = i0[ri] + s0[tri] * rx
            w1 = i1[ri] + s1[tri] * rx
            w2 = i2[ri] + s2[tri] * rx
            rgb = np.clip(w0[:, None] * vcol[:, 0][tri] + w1[:, None] * vcol[:, 1][tri] +
                          w2[:, None] * vcol[:, 2][tri], 0.0, 1.0)

        depth_flat = self._depth.reshape(-1)
        color_flat = self._color.reshape(-1, 3)
        alpha_flat = self._alpha.reshape(-1)

        current = depth_flat[idx]
        keep = (dep <= current + 1e-9) if (blend or not write_depth) else (dep < current)
        if not keep.all():
            idx = idx[keep]
            dep = dep[keep]
            rgb = rgb[keep]
            if idx.size == 0:
                return

        if blend:
            color_flat[idx] = rgb * alpha + color_flat[idx] * (1.0 - alpha)
            alpha_flat[idx] = np.maximum(alpha_flat[idx], alpha)
        else:
            color_flat[idx] = rgb
            alpha_flat[idx] = 1.0
        if write_depth:
            depth_flat[idx] = dep

    # ==================================================================
    # 线段
    # ==================================================================
    def _draw_edges(self, screen, ndc_z, edges: np.ndarray, color, width: int = 1,
                    alpha: float = 1.0, depth_bias: float = 0.0,
                    depth_test: bool = True) -> None:
        if len(edges) == 0:
            return
        p1 = screen[edges[:, 0]]
        p2 = screen[edges[:, 1]]
        z1 = ndc_z[edges[:, 0]]
        z2 = ndc_z[edges[:, 1]]
        key = z1 + z2 if depth_test else None
        self._draw_lines(p1, p2, z1, z2, color, width, alpha, depth_bias, depth_test,
                         depth_key=key)

    def _draw_lines(self, p1, p2, z1, z2, color, width: int = 1, alpha: float = 1.0,
                    depth_bias: float = 0.0, depth_test: bool = True,
                    depth_key: np.ndarray | None = None) -> None:
        W, H = self.width, self.height
        p1 = np.asarray(p1, dtype=np.float64).reshape(-1, 2)
        p2 = np.asarray(p2, dtype=np.float64).reshape(-1, 2)
        if p1.size == 0:
            return
        z1 = np.asarray(z1, dtype=np.float64).reshape(-1)
        z2 = np.asarray(z2, dtype=np.float64).reshape(-1)
        if depth_key is not None:
            depth_key = np.asarray(depth_key, dtype=np.float64).reshape(-1)
        dx = p2[:, 0] - p1[:, 0]
        dy = p2[:, 1] - p1[:, 1]
        n = np.ceil(np.maximum(np.abs(dx), np.abs(dy))).astype(np.int64) + 1

        on_screen = ~(((np.maximum(p1[:, 0], p2[:, 0]) < 0) |
                       (np.minimum(p1[:, 0], p2[:, 0]) > W - 1)) |
                      ((np.maximum(p1[:, 1], p2[:, 1]) < 0) |
                       (np.minimum(p1[:, 1], p2[:, 1]) > H - 1)))
        keep = on_screen & (n > 1)
        if not keep.any():
            return
        p1, p2, dx, dy, n = p1[keep], p2[keep], dx[keep], dy[keep], n[keep]
        z1, z2 = z1[keep], z2[keep]
        if depth_key is not None:
            depth_key = depth_key[keep]

        offsets = _width_offsets(int(width))
        color = np.asarray(color, dtype=np.float32)
        order = np.argsort(-n)
        cursor, total = 0, len(order)
        while cursor < total:
            end = min(cursor + 4096, total)
            sub = order[cursor:end]
            maxn = int(n[sub].max())
            while len(sub) > 1 and maxn * len(sub) > PIXEL_LIMIT:
                end = cursor + max(1, len(sub) // 2)
                sub = order[cursor:end]
                maxn = int(n[sub].max())
            if depth_key is not None and len(sub) > 1:
                sub = sub[np.argsort(-depth_key[sub])]      # 远 -> 近
            self._draw_lines_batch(p1[sub], p2[sub], dx[sub], dy[sub], n[sub],
                                   z1[sub], z2[sub], maxn, color, offsets, alpha,
                                   depth_bias, depth_test)
            cursor = end if end > cursor else cursor + 1

    def _draw_lines_batch(self, p1, p2, dx, dy, n, z1, z2, maxn, color, offsets,
                          alpha, depth_bias, depth_test) -> None:
        """批量画线。

        采样点按"线段"顺序排列；调用方已把线段按远到近排好，
        因此重复像素由后写入的较近线段覆盖，与 Z 缓冲配合即得正确遮挡。
        """
        W, H = self.width, self.height
        # 每条线按自己的采样点数归一化 t。以前整批共用 linspace(0,1,maxn)，
        # 短线的有效样本只走到 maxn 的前 n 个，端点实际停在 n/maxn 处 ——
        # 长短线混在一批时短的那条会被「画半截」（贯穿画布的十字准线
        # 下半段就曾只画到 209/240 行）。
        denom = np.maximum(n[:, None] - 1, 1).astype(np.float32)
        t = np.arange(maxn, dtype=np.float32)[None, :] / denom
        xs = p1[:, 0][:, None] + dx[:, None] * t
        ys = p1[:, 1][:, None] + dy[:, None] * t
        zs = z1[:, None] + (z2 - z1)[:, None] * t
        sample_ok = np.arange(maxn, dtype=np.int64)[None, :] < n[:, None]
        xi = np.rint(xs).astype(np.int64)
        yi = np.rint(ys).astype(np.int64)

        idx_parts, dep_parts = [], []
        for ox, oy in offsets:
            xo = xi + ox
            yo = yi + oy
            ok = sample_ok & (xo >= 0) & (xo < W) & (yo >= 0) & (yo < H)
            if not ok.any():
                continue
            idx_parts.append(yo[ok] * W + xo[ok])
            dep_parts.append(zs[ok])
        if not idx_parts:
            return
        idx = np.concatenate(idx_parts)
        dep = np.concatenate(dep_parts)
        if depth_bias:
            dep = dep - depth_bias

        depth_flat = self._depth.reshape(-1)
        color_flat = self._color.reshape(-1, 3)
        alpha_flat = self._alpha.reshape(-1)

        if depth_test:
            keep = dep <= depth_flat[idx]
            if not keep.all():
                idx = idx[keep]
                dep = dep[keep]
            if idx.size == 0:
                return
        rgb = np.broadcast_to(color[None, :], (len(idx), 3))
        if alpha >= 1.0:
            color_flat[idx] = rgb
            alpha_flat[idx] = 1.0
        else:
            color_flat[idx] = rgb * alpha + color_flat[idx] * (1.0 - alpha)
            alpha_flat[idx] = np.maximum(alpha_flat[idx], alpha)
        if depth_test:
            depth_flat[idx] = dep

    def _draw_points(self, screen, ndc_z, indices, color, size: int,
                     depth_test: bool = True, alpha: float = 1.0) -> None:
        if len(indices) == 0:
            return
        pts = np.asarray(screen, dtype=np.float64)[indices]
        z = np.asarray(ndc_z, dtype=np.float64)[indices]
        size = max(int(size), 1)
        half = (size - 1) * 0.5
        p1 = pts - half
        p2 = pts + half
        self._draw_lines(p1, p2, z, z, color, size, alpha, 0.0, depth_test)

    def _draw_marker(self, point_screen, point_z, color, size: int = 3,
                     alpha: float = 1.0) -> None:
        """十字标记（不做深度测试，保证测量点始终可见）。"""
        p = np.asarray(point_screen, dtype=np.float64).reshape(2)
        z = float(np.asarray(point_z).reshape(-1)[0])
        s = max(int(size), 1)
        a = np.array([[p[0] - s, p[1]], [p[0], p[1] - s]], dtype=np.float64)
        b = np.array([[p[0] + s, p[1]], [p[0], p[1] + s]], dtype=np.float64)
        self._draw_lines(a, b, np.array([z, z]), np.array([z, z]), color, 1, alpha,
                         0.0, False)

    # ==================================================================
    # 网格 / 坐标轴 / 十字准线
    # ==================================================================
    def _text_scale(self) -> int:
        return 1 if self.width < 900 else 2

    def _clamped_text_xy(self, x: int, y: int, text: str, ts: int) -> tuple:
        """把标注文字的左上角收进画布；任何视角下都不让文字被边缘切掉。"""
        tw, th = font5x7.text_size(text, ts)
        max_x = max(self.width - tw - 1, 0)
        max_y = max(self.height - th - 1, 0)
        return int(np.clip(int(x), 0, max_x)), int(np.clip(int(y), 0, max_y))


    def _draw_grid(self, scene: Scene, camera: Camera, opts: RenderOptions) -> None:
        scale = _scene_scale(scene)
        step = _nice_step(scale / 10.0) * max(opts.grid_scale, 1e-6)
        extent = step * 10.0
        p1, p2, is_axis = [], [], []
        for t in range(-10, 11):
            v = t * step
            p1.append((v, -extent, 0.0))
            p2.append((v, extent, 0.0))
            is_axis.append(t == 0)
            p1.append((-extent, v, 0.0))
            p2.append((extent, v, 0.0))
            is_axis.append(t == 0)
        view = camera.view_matrix()
        c1 = mu.transform_points(view, np.array(p1, dtype=np.float64))
        c2 = mu.transform_points(view, np.array(p2, dtype=np.float64))
        s1, z1, _ = camera.project_camera_space(c1)
        s2, z2, _ = camera.project_camera_space(c2)
        is_axis = np.array(is_axis)
        base = np.asarray(opts.background, dtype=np.float32)
        for flag, color in ((~is_axis, np.clip(base + 0.09, 0, 1)),
                            (is_axis, np.clip(base + 0.20, 0, 1))):
            if flag.any():
                self._draw_lines(s1[flag], s2[flag], z1[flag], z2[flag], color, 1)

    def _draw_axes(self, scene: Scene, camera: Camera, opts: RenderOptions) -> None:
        """坐标轴：原点在工件坐标系零点；箭头与 X/Y/Z 文字始终留在画布内。

        旧实现直接把世界端点投影后连线：一放大，箭头就跑到画布外，标签
        跟着消失；支点平移后甚至整条轴都不见了。这里把每段箭头裁进
        「预留给文字的安全矩形」，方向不变、长度自适应，任何缩放/旋转/
        平移下三个轴和字母都完整可见。
        """
        scale = _scene_scale(scene)
        length = scale * opts.axis_scale
        ts = self._text_scale()
        segments = axis_screen_geometry(camera, self.width, self.height,
                                        length, ts, opts.axis_scale)
        zero = np.zeros(1)
        for i, p0, p1 in segments:
            color = np.asarray(AXIS_COLORS[i], dtype=np.float32)
            self._draw_lines(p0.reshape(1, 2), p1.reshape(1, 2),
                             zero, zero, color, max(2, ts), 1.0, 0.0, False)
            d = p1 - p0
            norm = float(np.hypot(*d))
            if norm > 12:
                d = d / norm
                n = np.array([-d[1], d[0]])
                for side in (1.0, -1.0):
                    a = (p1 - d * 9 + n * 5 * side).reshape(1, 2)
                    self._draw_lines(a, p1.reshape(1, 2), zero, zero, color,
                                     max(2, ts), 1.0, 0.0, False)
            if opts.axis_labels:
                x, y = self._clamped_text_xy(
                    int(p1[0] + 7 * ts), int(p1[1] - 3 * ts),
                    AXIS_NAMES[i], ts)
                self._draw_text(x, y, AXIS_NAMES[i], color, ts)

    def _draw_crosshair(self, opts: RenderOptions) -> None:
        """十字准线：水平/垂直两条线贯穿整个画布（中心留一个很小的缺口）。"""
        cx, cy = self.width / 2.0, self.height / 2.0
        gap = 4.0
        base = np.asarray(opts.background, dtype=np.float32)
        color = np.clip(1.0 - base * 0.55, 0.2, 1.0)
        segs_a = [(0.0, cy), (cx + gap, cy), (cx, 0.0), (cx, cy + gap)]
        segs_b = [(cx - gap, cy), (float(self.width - 1), cy),
                  (cx, cy - gap), (cx, float(self.height - 1))]
        self._draw_lines(np.array(segs_a, dtype=np.float64),
                         np.array(segs_b, dtype=np.float64),
                         np.zeros(4), np.zeros(4), color, 1, 1.0, 0.0, False)

    # ==================================================================
    # 文字与测量标注
    # ==================================================================
    def _draw_text(self, x: int, y: int, text: str, color, scale: int = 1,
                   outline: bool = True) -> None:
        mask = font5x7.text_mask(text, scale)
        mh, mw = mask.shape
        if mw == 0 or mh == 0:
            return
        x0, y0 = int(x), int(y)
        if x0 >= self.width or y0 >= self.height or x0 + mw <= 0 or y0 + mh <= 0:
            return
        sx0, sy0 = max(x0, 0), max(y0, 0)
        sx1, sy1 = min(x0 + mw, self.width), min(y0 + mh, self.height)
        sub = mask[sy0 - y0: sy1 - y0, sx0 - x0: sx1 - x0]
        if not sub.any():
            return
        region = self._color[sy0:sy1, sx0:sx1]
        alpha_region = self._alpha[sy0:sy1, sx0:sx1]
        col = np.asarray(color, dtype=np.float32)
        if outline:
            ring = _dilate(sub) & ~sub
            region[ring] = np.clip(col * 0.12, 0.0, 1.0)
            alpha_region[ring] = 1.0
        region[sub] = col
        alpha_region[sub] = 1.0

    def _draw_measurements(self, measurements, camera: Camera) -> None:
        ts = self._text_scale()
        for item in measurements:
            try:
                polylines = item.polylines()
                markers = item.markers()
                color = np.asarray(getattr(item, "color", (1.0, 0.86, 0.32)),
                                   dtype=np.float32)
            except Exception:
                continue
            for poly in polylines:
                pts = np.asarray(poly, dtype=np.float64).reshape(-1, 3)
                if len(pts) < 2:
                    continue
                scr, z, _ = camera.project(pts)
                self._draw_lines(scr[:-1], scr[1:], z[:-1], z[1:], color,
                                 max(2, ts), 1.0, 0.0, False)
            for marker in markers:
                scr, z, _ = camera.project(np.asarray(marker, dtype=np.float64)[None, :])
                if np.isfinite(scr).all():
                    self._draw_marker(scr[0], z[0], color, 3 + ts)
            label = None
            try:
                label = item.label()
            except Exception:
                label = None
            if label:
                anchor = np.asarray(item.label_anchor(), dtype=np.float64)[None, :]
                scr, _, _ = camera.project(anchor)
                if np.isfinite(scr).all():
                    tx, ty = self._clamped_text_xy(
                        int(scr[0][0] + 10 * ts), int(scr[0][1] - 6 * ts),
                        label, ts)
                    self._draw_text(tx, ty, label, (1.0, 1.0, 1.0), ts)

    # ==================================================================
    # 拾取
    # ==================================================================
    def pick(self, scene: Scene, camera: Camera, x: float, y: float) -> dict | None:
        """屏幕拾取：返回最近命中的 {object, triangle, point, distance}。"""
        origin, direction = camera.ray(x, y)
        best = None
        for obj in scene.objects:
            mesh = obj.mesh
            if not obj.visible or obj.locked or mesh.triangle_count == 0:
                continue
            matrix = self._object_matrix(obj)
            inv = np.linalg.inv(matrix)
            o_local = mu.transform_points(inv, origin[None, :])[0]
            d_local = mu.normalize(inv[:3, :3] @ direction)
            v0 = mesh.vertices[mesh.triangles[:, 0]]
            e1 = mesh.vertices[mesh.triangles[:, 1]] - v0
            e2 = mesh.vertices[mesh.triangles[:, 2]] - v0
            pvec = np.cross(np.broadcast_to(d_local, e2.shape), e2)
            det = np.einsum("ij,ij->i", e1, pvec)
            ok = np.abs(det) > 1e-12
            if not ok.any():
                continue
            inv_det = np.zeros_like(det)
            inv_det[ok] = 1.0 / det[ok]
            tvec = o_local - v0
            u = np.einsum("ij,ij->i", tvec, pvec) * inv_det
            ok &= (u >= -1e-9) & (u <= 1 + 1e-9)
            qvec = np.cross(tvec, e1)
            v = np.einsum("ij,ij->i", np.broadcast_to(d_local, e1.shape), qvec) * inv_det
            ok &= (v >= -1e-9) & (u + v <= 1 + 1e-9)
            t = np.einsum("ij,ij->i", e2, qvec) * inv_det
            ok &= t > 1e-9
            if not ok.any():
                continue
            idx = np.nonzero(ok)[0]
            k = int(idx[np.argmin(t[idx])])
            world_p = mu.transform_points(matrix, (o_local + d_local * t[k])[None, :])[0]
            if best is None or t[k] < best["distance"]:
                best = {"object": obj, "triangle": k, "point": world_p,
                        "distance": float(t[k])}
        return best


# ======================================================================
# 辅助函数
# ======================================================================
def _object_depth(obj, camera: Camera, renderer: SoftwareRenderer) -> float:
    """相机空间 z（越小越远），用于透明对象的绘制顺序。"""
    matrix = renderer._object_matrix(obj)
    center = mu.transform_points(matrix, obj.center()[None, :])[0]
    return float(camera.to_camera_space(center[None, :])[0, 2])


def _scene_scale(scene: Scene) -> float:
    lo, hi = scene.bounds()
    span = float(np.linalg.norm(hi - lo))
    return span if span > 1e-9 else 100.0


def _clip_segment_to_rect(a, b, x0: float, y0: float,
                          x1: float, y1: float):
    """Liang-Barsky：把线段裁进安全矩形，返回两端点或 None。

    放大/平移/旋转之后，坐标轴箭头的世界端点常常跑到画布外；直接绘制时
    Qt 会把它裁掉，X/Y/Z 标签更是整个消失。这里把每个箭头按方向缩短到
    画布内，坐标系在任何变换下都保持完整可读。
    """
    a = np.asarray(a, dtype=np.float64).reshape(2)
    b = np.asarray(b, dtype=np.float64).reshape(2)
    d = b - a
    if x1 <= x0 or y1 <= y0 or not np.isfinite(d).all():
        return None
    t0, t1 = 0.0, 1.0
    for i, (lo, hi) in enumerate(((x0, x1), (y0, y1))):
        if abs(d[i]) < 1e-12:
            if a[i] < lo or a[i] > hi:
                return None
            continue
        ta, tb = (lo - a[i]) / d[i], (hi - a[i]) / d[i]
        t0 = max(t0, min(ta, tb))
        t1 = min(t1, max(ta, tb))
    if t1 <= t0:
        return None
    return a + t0 * d, a + t1 * d


def axis_screen_geometry(camera, width, height, world_length, ts=1,
                         axis_scale=0.6):
    """坐标轴在屏幕上的线段：``[(轴序号, 起点, 终点), ...]``。

    两种形态，都保证箭头与 X/Y/Z 文字完整落在画布内：

    * 工件零点仍在画面里：保留「从原点出发」的坐标轴，把每条轴按方向
      裁进安全矩形（放大/旋转后箭头会缩短，但方向和原点不变）；
    * 零点被平移出画面：切换成 CAD 常见的左下角小型坐标架，方向仍由
      当前相机给出，任何变换下都能看到 X/Y/Z 三轴和字母。

    软件渲染器与 GL 的 2D 叠加层共用这份几何，避免两条路径各写一遍。
    """
    tw, th = font5x7.text_size("X", ts)
    x0 = 12.0 * ts + 8.0
    x1 = width - (8 * ts + tw + 8.0)
    y0 = 4 * ts + th + 8.0
    y1 = height - (th + 12.0)
    radius = max(18.0, min(34.0 * ts * max(float(axis_scale) / 0.6, 0.5),
                           width * 0.13, height * 0.16))
    center = np.array([x0 + radius, y1 - radius], dtype=np.float64)
    right, up = camera.right(), camera.up()

    def corner_triad():
        out = []
        for i in range(3):
            a = np.zeros(3)
            a[i] = 1.0
            d = np.array([float(np.dot(a, right)), -float(np.dot(a, up))])
            n = float(np.linalg.norm(d))
            d = d / n if n > 1e-9 else np.array([1.0, 0.0])
            out.append((i, center.copy(), center + d * radius))
        return out

    if x1 <= x0 or y1 <= y0 or world_length <= 0:
        return corner_triad()
    pts = np.vstack([np.zeros((1, 3)), np.eye(3) * float(world_length)])
    scr, _, valid = camera.project(pts)
    if not (bool(valid[0]) and np.isfinite(scr).all()):
        return corner_triad()
    o = scr[0]
    if not (0.0 <= o[0] <= width and 0.0 <= o[1] <= height):
        return corner_triad()
    out = []
    for i in range(3):
        seg = _clip_segment_to_rect(o, scr[i + 1], x0, y0, x1, y1)
        if seg is not None:
            out.append((i, seg[0], seg[1]))
    return out if len(out) >= 2 else corner_triad()


def _width_offsets(width: int) -> list:
    width = max(int(width), 1)
    if width <= 1:
        return [(0, 0)]
    offsets = []
    half = width // 2
    for ox in range(-half, width - half):
        for oy in range(-half, width - half):
            offsets.append((ox, oy))
    return offsets


def _nice_step(value: float) -> float:
    """取 1/2/5 x 10^k 形式的整齐刻度。"""
    if value <= 0 or not np.isfinite(value):
        return 1.0
    exp = float(np.floor(np.log10(value)))
    base = value / (10.0 ** exp)
    for candidate in (1.0, 2.0, 5.0, 10.0):
        if base <= candidate + 1e-12:
            return candidate * (10.0 ** exp)
    return 10.0 ** (exp + 1)


def _depth_ordered_chunks(bw: np.ndarray, bh: np.ndarray,
                          depth_key: np.ndarray, limit: int) -> list:
    """按"远 -> 近"分批（透明混合必须保持这个顺序）。

    每批的矩形尺寸取批内 bbox 的最大值，并用 ``limit`` 限制批内总单元数。
    """
    out = []
    order = np.argsort(-(bw * bh))
    cursor, total = 0, len(order)
    while cursor < total:
        end = min(cursor + 384, total)
        sub = order[cursor:end]
        mw, mh = int(bw[sub].max()), int(bh[sub].max())
        while len(sub) > 1 and mw * mh * len(sub) > limit:
            end = cursor + max(1, len(sub) // 2)
            sub = order[cursor:end]
            mw, mh = int(bw[sub].max()), int(bh[sub].max())
        out.append((mw, mh, sub[np.argsort(-depth_key[sub])]))
        cursor = end if end > cursor else cursor + 1
    return out


def _size_bucketed_chunks(bw: np.ndarray, bh: np.ndarray, limit: int,
                          depth_key: np.ndarray) -> list:
    """按 bbox 边长的 2 次幂分桶分批，使 (B,mh,mw) 矩形贴近真实 bbox。

    原先按批内 max(bw)/max(bh) 取矩形，一个 31x170 的三角形会把同批所有
    三角形都撑成 31x170，实测空洞率达 8.0x。分桶后降到 2.0x，光栅化耗时随之
    下降数倍。三角形 bbox 中位数 (15x12) 与最大值 (31x170) 差一个量级，
    这正是"一刀切矩形"最吃亏的情形。

    返回 [(mw, mh, sub), ...]，每个 sub 内部按 NDC 深度「远 -> 近」排列：
    批内重叠像素靠后写入的三角形胜出，必须与全局 Z 缓冲保持同一顺序，
    否则远处三角形会盖住同批的近处三角形（模型出现穿透/花面）。
    """
    if len(bw) == 0:
        return []
    # 桶键 = (ceil(log2(bw)), ceil(log2(bh)))，边长向上取整到 2 的幂
    pw = np.ceil(np.log2(np.maximum(bw, 1))).astype(np.int32)
    ph = np.ceil(np.log2(np.maximum(bh, 1))).astype(np.int32)
    key = pw * 32 + ph                      # ph < 32（图像高度不可能到 2^31）
    order = np.argsort(key, kind="stable")
    sorted_key = key[order]
    cuts = np.nonzero(np.diff(sorted_key))[0] + 1
    starts = np.concatenate(([0], cuts))
    ends = np.concatenate((cuts, [len(sorted_key)]))

    out = []
    for st, en in zip(starts, ends):
        idx = order[st:en]
        k = int(sorted_key[st])
        mw, mh = 1 << (k // 32), 1 << (k % 32)
        cap = max(1, min(MAX_TRIS_PER_BATCH, limit // max(mw * mh, 1)))
        for i in range(0, len(idx), cap):
            part = idx[i:i + cap]
            if len(part) > 1:
                part = part[np.argsort(-depth_key[part], kind="stable")]
            out.append((mw, mh, part))
    return out


def _dilate(mask: np.ndarray, iterations: int = 1) -> np.ndarray:
    out = mask
    for _ in range(iterations):
        pad = np.pad(out, 1, mode="constant", constant_values=False)
        out = (pad[:-2, :-2] | pad[:-2, 1:-1] | pad[:-2, 2:] |
               pad[1:-1, :-2] | pad[1:-1, 1:-1] | pad[1:-1, 2:] |
               pad[2:, :-2] | pad[2:, 1:-1] | pad[2:, 2:])
    return out


def _downsample(image: np.ndarray, width: int, height: int) -> np.ndarray:
    """盒式降采样（超采样抗锯齿）。

    直接对 uint8 视图分两步求和，避免先 astype 出一份全尺寸 float32 副本
    （ss=2 时那是 30 MB 级别的无谓搬运）。
    """
    h, w = image.shape[:2]
    fy, fx = h // height, w // width
    if fy < 1 or fx < 1:
        return image[:height, :width]
    cropped = image[:height * fy, :width * fx]
    if cropped.ndim == 3:
        ch = cropped.shape[2]
        block = cropped.reshape(height, fy, width, fx, ch)
        total = block.sum(axis=1, dtype=np.float32).sum(axis=2, dtype=np.float32)
    else:
        block = cropped.reshape(height, fy, width, fx)
        total = block.sum(axis=1, dtype=np.float32).sum(axis=2, dtype=np.float32)
    avg = total / float(fx * fy)
    return np.clip(avg + 0.5, 0, 255).astype(np.uint8)


def _clip_triangles_near(verts_cam: np.ndarray, triangles: np.ndarray, near: float):
    """把跨近平面的三角形裁剪到近平面之内（相机朝 -Z 观察）。

    返回 (新三角形索引, 扩充后的相机空间顶点, 每个新三角形对应的源面索引)。
    """
    z = verts_cam[:, 2]
    limit = -near
    new_tris, src, extra = [], [], []
    base = len(verts_cam)
    for fi in range(len(triangles)):
        tri = triangles[fi]
        dz = z[tri]
        if (dz <= limit).all():
            new_tris.append(tri)
            src.append(fi)
            continue
        if (dz > limit).all():
            continue
        poly = _clip_polygon_z(verts_cam[tri], limit)
        if len(poly) < 3:
            continue
        start = base + len(extra)
        extra.extend(poly.tolist())
        for k in range(1, len(poly) - 1):
            new_tris.append(np.array([start, start + k, start + k + 1], dtype=np.int64))
            src.append(fi)
    if not new_tris:
        return (np.zeros((0, 3), dtype=np.int64), verts_cam,
                np.zeros(0, dtype=np.int64))
    verts = (np.concatenate([verts_cam, np.array(extra, dtype=np.float64)])
             if extra else verts_cam)
    return (np.array(new_tris, dtype=np.int64), verts, np.array(src, dtype=np.int64))


def _clip_polygon_z(points: np.ndarray, z_limit: float) -> np.ndarray:
    """把多边形裁剪到 z <= z_limit 的半空间（Sutherland–Hodgman）。"""
    out = []
    n = len(points)
    for i in range(n):
        a = points[i]
        b = points[(i + 1) % n]
        a_in = a[2] <= z_limit
        b_in = b[2] <= z_limit
        if a_in:
            out.append(a)
        if a_in != b_in:
            t = (z_limit - a[2]) / (b[2] - a[2])
            out.append(a + (b - a) * t)
    return np.array(out, dtype=np.float64) if out else np.zeros((0, 3))
