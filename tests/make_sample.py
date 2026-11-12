"""生成示例模型并渲染预览图（无需 GUI）。

用法：
    python tests/make_sample.py [输出目录]
默认输出到 <项目根>/samples。

生成内容：
    samples/drill_blank.stl          二进制 STL（阶梯毛坯 + 螺旋槽）
    samples/drill_blank_ascii.stl    ASCII STL
    samples/preview_solid.png        实体 + 测量标注 + 坐标轴 + 网格
    samples/preview_wire.png         线框
    samples/preview_color.png        按连通分量着色 + 棱边
"""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from stl_viewer import png, stl_io                        # noqa: E402
from stl_viewer.camera import Camera                      # noqa: E402
from stl_viewer.lighting import LightSettings             # noqa: E402
from stl_viewer.measure import MeasureManager             # noqa: E402
from stl_viewer.mesh import Mesh                          # noqa: E402
from stl_viewer.renderer import RenderOptions, SoftwareRenderer  # noqa: E402
from stl_viewer.scene import Scene                        # noqa: E402
from stl_viewer.settings import UnitSystem                # noqa: E402


def make_drill_like_mesh(radius: float = 6.0, length: float = 60.0,
                         segments: int = 96) -> Mesh:
    """构造一个近似钻头毛坯：阶梯圆柱 + 两条螺旋槽。"""
    parts = []
    offset = 0
    # 柄部（较粗）+ 刃部（略细）
    for r, h, z0 in ((radius * 1.15, length * 0.35, 0.0),
                     (radius, length * 0.65, length * 0.35)):
        ring = _cylinder_arrays(r, h, segments, z0)
        verts, tris = ring
        parts.append((verts, tris + offset))
        offset += len(verts)

    # 两条螺旋槽：用细长四边形条带贴附在刃部表面
    for phase in (0.0, np.pi):
        v, t = _helix_flute(radius * 1.002, length * 0.65, length * 0.35,
                            turns=1.6, width_deg=48.0, phase=phase, steps=160)
        parts.append((v, t + offset))
        offset += len(v)

    verts = np.concatenate([p[0] for p in parts])
    tris = np.concatenate([p[1] for p in parts])
    mesh = Mesh(verts, tris, name="drill_blank")
    mesh.translate((0.0, 0.0, -length * 0.5))
    # 让 Z 轴与刀具回转轴一致，原点位于圆柱中部
    mesh.weld(1e-7)
    mesh.orient_outward()
    return mesh


def _cylinder_arrays(radius: float, height: float, segments: int, z0: float):
    ang = np.linspace(0.0, 2.0 * np.pi, segments, endpoint=False)
    cx, cy = np.cos(ang), np.sin(ang)
    ring_b = np.stack([cx * radius, cy * radius, np.full_like(cx, z0)], axis=1)
    ring_t = np.stack([cx * radius, cy * radius, np.full_like(cx, z0 + height)], axis=1)
    verts = np.concatenate([ring_b, ring_t,
                            [[0.0, 0.0, z0]], [[0.0, 0.0, z0 + height]]])
    i0 = segments
    cb, ct = 2 * segments, 2 * segments + 1
    tris = []
    for i in range(segments):
        j = (i + 1) % segments
        tris.append([i, j, i0 + j])
        tris.append([i, i0 + j, i0 + i])
        tris.append([cb, j, i])
        tris.append([ct, i0 + i, i0 + j])
    return verts, np.array(tris, dtype=np.int64)


def _helix_flute(radius: float, height: float, z0: float, turns: float,
                 width_deg: float, phase: float, steps: int):
    """螺旋槽条带（四边形网格）。"""
    t = np.linspace(0.0, 1.0, steps)
    z = z0 + height * t
    theta = phase + 2.0 * np.pi * turns * t
    half = np.radians(width_deg) * 0.5
    verts = []
    for k in range(steps):
        for d in (-half, half):
            a = theta[k] + d
            verts.append([radius * np.cos(a), radius * np.sin(a), z[k]])
    tris = []
    for k in range(steps - 1):
        b = k * 2
        tris.append([b, b + 1, b + 3])
        tris.append([b, b + 3, b + 2])
    return np.array(verts), np.array(tris, dtype=np.int64)


def main() -> int:
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    out_dir = sys.argv[1] if len(sys.argv) > 1 else os.path.join(root, "samples")
    os.makedirs(out_dir, exist_ok=True)

    mesh = make_drill_like_mesh()
    st = mesh.statistics()
    print(f"模型: {mesh.name}  V={mesh.vertex_count} F={mesh.triangle_count}")
    print(st.as_text())

    stl_path = os.path.join(out_dir, "drill_blank.stl")
    stl_io.write_binary_stl(stl_path, mesh.vertices, mesh.triangles)
    stl_io.write_ascii_stl(os.path.join(out_dir, "drill_blank_ascii.stl"),
                           mesh.vertices, mesh.triangles, mesh.name)
    print("已写出:", stl_path)

    # ---- 场景 ----
    scene = Scene()
    obj = scene.add_mesh(mesh, name="drill_blank")
    obj.color = (0.80, 0.83, 0.88)
    scene.add_box(size=(46, 46, 4), center=(0, 0, -33), name="fixture")
    cam = Camera(960, 680)
    cam.fit(*scene.bounds())
    cam.yaw, cam.pitch = 38.0, 24.0
    light = LightSettings()
    renderer = SoftwareRenderer()
    units = UnitSystem("mm")

    # ---- 测量标注 ----
    mgr = MeasureManager()
    mgr.start("circle3p")
    for p in ((6.9, 0, 20), (0, 6.9, 20), (-6.9, 0, 20)):
        mgr.add_point(p)
    mgr.start("line3d")
    mgr.add_point((0, 0, -18))
    mgr.add_point((0, 0, 18))
    mgr.start("rzpoint")
    mgr.add_point((6.9, 0, 0))
    mgr.refresh_labels(units)

    solid = renderer.render(
        scene, cam, size=(960, 680), light=light,
        measurements=mgr.items,
        options=RenderOptions(show_axis=True, show_grid=True, supersample=2))
    png.write_png(os.path.join(out_dir, "preview_solid.png"), solid)
    print("preview_solid.png", renderer.last_stats)

    wire = renderer.render(
        scene, cam, size=(960, 680), light=light,
        options=RenderOptions(wire=True, solid=False, show_axis=True,
                              background=(0.10, 0.11, 0.13), supersample=2))
    png.write_png(os.path.join(out_dir, "preview_wire.png"), wire)

    colored = renderer.render(
        scene, cam, size=(960, 680), light=light,
        options=RenderOptions(color_mode="component", show_edges=True,
                              show_axis=True, supersample=2))
    png.write_png(os.path.join(out_dir, "preview_color.png"), colored)
    print("预览图已写入", out_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
