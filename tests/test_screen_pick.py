"""屏幕投影拾取（ScreenPicker）的单元回归 —— 不需要 OpenGL。

这一层是测量取点的地基：把模型投影到屏幕、建二维格子索引，点击时只在
命中格子里做精确射线求交。它必须做到

    * 点在模型上 -> 拿到真实表面点（真实三维尺寸，不随视角变）
    * 点在轮廓最外一圈像素（几何边缘差半个像素）-> 吸附到边缘，不退回视图平面
    * 点在空白处 -> 返回 None，由上层退回视图平面（任意位置可测）
    * 视图/模型变了自动重建索引
    * 大模型（几十万面）也够快，且**完全不调用 GL**

运行：python tests/test_screen_pick.py
"""

from __future__ import annotations

import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from stl_viewer.camera import Camera
from stl_viewer.mesh import Mesh
from stl_viewer.scene import Scene
from stl_viewer.screenpick import ScreenPicker

PASSED, FAILED = [], []


def check(name, condition, detail=""):
    (PASSED if condition else FAILED).append(name)
    print(f"  [{'PASS' if condition else 'FAIL'}] {name} {detail}")


def make_box(sx=30.0, sy=20.0, sz=10.0) -> Mesh:
    """轴对齐长方体（12 个三角形），用于验证「大三角形」分支。"""
    x, y, z = sx / 2, sy / 2, sz / 2
    verts = np.array([[-x, -y, -z], [x, -y, -z], [x, y, -z], [-x, y, -z],
                      [-x, -y, z], [x, -y, z], [x, y, z], [-x, y, z]],
                     dtype=np.float64)
    faces = [(0, 1, 2), (0, 2, 3), (4, 6, 5), (4, 7, 6),
             (0, 4, 5), (0, 5, 1), (1, 5, 6), (1, 6, 2),
             (2, 6, 7), (2, 7, 3), (3, 7, 4), (3, 4, 0)]
    return Mesh(verts, np.array(faces, dtype=np.int32), name="box")


def make_cylinder(radius=10.0, height=40.0, segments=240) -> Mesh:
    """细密圆柱（≈ segments*4 个三角形），验证密集网格路径。"""
    t = np.linspace(0.0, 2.0 * np.pi, segments, endpoint=False)
    ring = np.stack([radius * np.cos(t), radius * np.sin(t)], axis=1)
    bottom = np.column_stack([ring, np.full(segments, -height / 2)])
    top = np.column_stack([ring, np.full(segments, height / 2)])
    verts = np.vstack([bottom, top, [[0, 0, -height / 2], [0, 0, height / 2]]])
    tris = []
    for i in range(segments):
        j = (i + 1) % segments
        tris += [[i, j, segments + j], [i, segments + j, segments + i]]
    cb, ct = 2 * segments, 2 * segments + 1
    for i in range(segments):
        j = (i + 1) % segments
        tris += [[cb, j, i], [ct, segments + i, segments + j]]
    return Mesh(verts, np.array(tris, dtype=np.int32), name="cyl")


def view_depth(camera, points) -> np.ndarray:
    """沿视线的深度（越小越近），用于判断点落在哪一面。"""
    pts = np.atleast_2d(np.asarray(points, dtype=np.float64))
    return (pts - camera.eye()) @ camera.forward()


def screen_bbox(camera, verts):
    scr, _, ok = camera.project(verts)
    ok = np.isfinite(scr).all(axis=1)
    return (scr[ok, 0].min(), scr[ok, 0].max(),
            scr[ok, 1].min(), scr[ok, 1].max())


def main() -> int:
    scene = Scene()
    box = make_box()
    scene.add_mesh(box, name="box")
    camera = Camera(900, 700)
    camera.set_standard_view("front")
    camera.fit(box.vertices.min(axis=0), box.vertices.max(axis=0),
               points=box.vertices)
    picker = ScreenPicker()

    print("\n=== 点在模型上：返回真实表面点 ===")
    xlo, xhi, ylo, yhi = screen_bbox(camera, box.vertices)
    xm, ym = (xlo + xhi) * 0.5, (ylo + yhi) * 0.5
    t0 = time.perf_counter()
    hit = picker.pick(scene, camera, xm, ym)
    cost = (time.perf_counter() - t0) * 1000.0
    check("正面中心取到表面点", hit is not None,
          f"{np.round(hit, 3) if hit is not None else 'None'}（{cost:.1f}ms 含建索引）")
    if hit is not None:
        back, _, ok = camera.project(hit[None, :])
        check("取到的点正落在光标像素上",
              bool(ok[0]) and abs(back[0, 0] - xm) < 0.75
              and abs(back[0, 1] - ym) < 0.75,
              f"投影 {np.round(back[0], 2)} vs 点击 {(round(xm, 2), round(ym, 2))}")
        depth = float(view_depth(camera, hit)[0])
        near = float(np.min(view_depth(camera, box.vertices)))
        check("深度 = 模型靠近相机的那一面（真实深度）",
              abs(depth - near) < 1e-6,
              f"depth={depth:.3f} near={near:.3f}")

    print("\n=== 点在空白处：返回 None（上层退回视图平面）===")
    check("模型外远处不命中", picker.pick(scene, camera, 20, 20) is None)
    check("画面另一角不命中",
          picker.pick(scene, camera, camera.width - 10, camera.height - 10) is None)

    print("\n=== 轮廓最外一圈像素：吸附到边缘而不是判空 ===")
    edge_hits = 0
    edge_pts = []
    for dx in (0.4, 0.8, 1.2):
        p = picker.pick(scene, camera, xlo - dx, ym, pad=2.0)
        if p is not None:
            edge_hits += 1
            edge_pts.append(p)
    check("轮廓外 0.4~1.2 像素仍取到点", edge_hits == 3, f"{edge_hits}/3")
    if edge_pts:
        p = edge_pts[0]
        back, _, ok = camera.project(p[None, :])
        check("吸附点贴近光标（≤4 像素）",
              bool(ok[0]) and abs(back[0, 0] - (xlo - 0.4)) <= 4.0,
              f"Δx={abs(back[0,0]-(xlo-0.4)):.2f}px")

    print("\n=== 视图变化后自动重建 ===")
    key_before = picker._key
    camera.dolly(6)
    hit2 = picker.pick(scene, camera, xm, ym)
    check("缩放后索引重建", picker._key != key_before)
    check("缩放后仍能取点", hit2 is not None)
    if hit is not None and hit2 is not None:
        check("缩放不改变表面点的深度位置（同一面）",
              abs(float(np.dot(hit2 - camera.eye(), camera.forward()))
                  - float(np.dot(hit - camera.eye(), camera.forward())))
              < 3.0,
              "同一前表面")

    print("\n=== 密集网格：精度与耗时 ===")
    scene2 = Scene()
    cyl = make_cylinder()
    scene2.add_mesh(cyl, name="cyl")
    cam2 = Camera(900, 700)
    cam2.set_standard_view("front")
    cam2.fit(cyl.vertices.min(axis=0), cyl.vertices.max(axis=0),
             points=cyl.vertices)
    picker2 = ScreenPicker()
    t0 = time.perf_counter()
    picker2.ensure(scene2, cam2)
    build_ms = (time.perf_counter() - t0) * 1000.0
    n_tri = cyl.triangle_count
    check("索引规模与三角形数一致",
          picker2.triangle_count == n_tri, f"{picker2.triangle_count} tris")
    check("建索引耗时合理（< 1s）", build_ms < 1000.0, f"{build_ms:.0f}ms")

    xlo2, xhi2, ylo2, yhi2 = screen_bbox(cam2, cyl.vertices)
    cx, cy = (xlo2 + xhi2) * 0.5, (ylo2 + yhi2) * 0.5
    times = []
    for _ in range(5):
        t0 = time.perf_counter()
        p = picker2.pick(scene2, cam2, cx, cy)
        times.append((time.perf_counter() - t0) * 1000.0)
    check("单次取点 < 20ms（不含建索引）",
          max(times) < 20.0, f"max={max(times):.1f}ms")
    if p is not None:
        near2 = float(np.min(view_depth(cam2, cyl.vertices)))
        check("圆柱正面取点深度 = 前表面",
              abs(float(view_depth(cam2, p)[0]) - near2) < 1e-6,
              f"depth={float(view_depth(cam2, p)[0]):.4f} near={near2:.4f}")

    print("\n=== 空场景 / 隐藏对象 ===")
    empty = Scene()
    p3 = picker2.pick(empty, cam2, cx, cy)
    check("空场景返回 None", p3 is None)
    scene2.objects[0].visible = False
    p4 = picker2.pick(scene2, cam2, cx, cy)
    check("对象隐藏后返回 None", p4 is None)

    total = len(PASSED) + len(FAILED)
    print(f"\n总计: {len(PASSED)} 通过, {len(FAILED)} 失败  (共 {total} 项)")
    if FAILED:
        print("失败项:", ", ".join(FAILED))
        return 1
    print("屏幕投影拾取回归通过。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
