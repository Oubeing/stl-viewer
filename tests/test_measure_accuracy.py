"""测量数值精度回归（真实 OpenGL，无 GL 时自动跳过）。

覆盖用户反馈「测量功能能用了，但尺寸值不对」：

    * 已知模型在标准视图下量出的长度 / 直径 / 圆直径必须与名义尺寸一致；
    * 平面前表面上的两点距离（最典型的「量个台阶宽度」）必须等于真实尺寸，
      并且**缩放前后读数一致** —— 旧实现把点投影到「过支点、垂直视线」的
      平面，模型前表面离镜头越近，读数被放大得越多（放大 2 倍可偏大 9%）；
    * 斜视角（iso）取点给出的是三维距离，ΔX/ΔY/ΔZ 里能对上名义长度；
    * 空白处点击仍然可用（投影到视图平面），「任意位置都能取点」不丢失。

运行：QT_QPA_PLATFORM=windows python tests/test_measure_accuracy.py
"""

from __future__ import annotations

import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from PyQt5 import QtCore, QtTest, QtWidgets

from stl_viewer.mesh import Mesh
from stl_viewer.viewer import MainWindow

PASSED, FAILED = [], []

RADIUS = 10.0        # 圆柱：名义直径 20
HEIGHT = 40.0        # 圆柱：名义长度 40
BLOCK = (60.0, 60.0, 10.0)   # 方块：宽 × 高 × 厚
SPAN = 15.0          # 方块前表面上要量的两点间距


def check(name, condition, detail=""):
    (PASSED if condition else FAILED).append(name)
    print(f"  [{'PASS' if condition else 'FAIL'}] {name} {detail}")


def make_cylinder(segments: int = 720) -> Mesh:
    """精确圆柱网格：R=10、H=40，轴线为世界 Z。"""
    t = np.linspace(0.0, 2.0 * np.pi, segments, endpoint=False)
    ring = np.stack([RADIUS * np.cos(t), RADIUS * np.sin(t)], axis=1)
    verts = np.vstack([
        np.column_stack([ring, np.full(segments, -HEIGHT * 0.5)]),
        np.column_stack([ring, np.full(segments, HEIGHT * 0.5)]),
        [[0.0, 0.0, -HEIGHT * 0.5], [0.0, 0.0, HEIGHT * 0.5]],
    ])
    tris = []
    for i in range(segments):
        j = (i + 1) % segments
        tris.append([i, j, segments + j])
        tris.append([i, segments + j, segments + i])
    bottom, top = 2 * segments, 2 * segments + 1
    for i in range(segments):
        j = (i + 1) % segments
        tris.append([bottom, j, i])
        tris.append([top, segments + i, segments + j])
    return Mesh(verts, np.asarray(tris, dtype=np.int64), name="cyl")


def make_block() -> Mesh:
    """长方体：前表面在 y = -厚度/2（正对前视相机）。"""
    w, h, d = BLOCK
    x0, x1 = -w * 0.5, w * 0.5
    y0, y1 = -d * 0.5, d * 0.5
    z0, z1 = -h * 0.5, h * 0.5
    verts = np.array([[x0, y0, z0], [x1, y0, z0], [x1, y1, z0], [x0, y1, z0],
                      [x0, y0, z1], [x1, y0, z1], [x1, y1, z1], [x0, y1, z1]],
                     dtype=np.float64)
    faces = [(0, 3, 2), (0, 2, 1),        # 前面 (y0)
             (4, 5, 6), (4, 6, 7),        # 后面
             (0, 1, 5), (0, 5, 4),        # 底面
             (3, 7, 6), (3, 6, 2),        # 顶面
             (0, 4, 7), (0, 7, 3),        # 左面
             (1, 2, 6), (1, 6, 5)]        # 右面
    return Mesh(verts, np.asarray(faces, dtype=np.int64), name="block")


def main() -> int:
    app = QtWidgets.QApplication(sys.argv[:1])
    win = MainWindow()
    win.resize(1100, 800)
    win.show()
    app.processEvents()
    canvas, state = win.canvas, win.state

    cyl = make_cylinder()
    obj_cyl = state.scene.add_mesh(cyl, name="cyl")
    obj_block = state.scene.add_mesh(make_block(), name="block")
    obj_block.visible = False
    state.fit_view()
    canvas.mark_dirty()
    for _ in range(10):
        app.processEvents()
        time.sleep(0.02)
    canvas._probe_gl()
    for _ in range(6):
        app.processEvents()
        time.sleep(0.02)

    if not (canvas._gl_active and canvas._gl is not None and canvas._gl.ready):
        print("  [SKIP] 当前环境没有可用的 OpenGL 上下文（请在有显卡的桌面运行）")
        win.close()
        return 0

    def click(x, y):
        pt = QtCore.QPoint(int(round(x)), int(round(y)))
        QtTest.QTest.mousePress(canvas, QtCore.Qt.LeftButton,
                                QtCore.Qt.NoModifier, pt)
        QtTest.QTest.mouseRelease(canvas, QtCore.Qt.LeftButton,
                                  QtCore.Qt.NoModifier, pt)
        app.processEvents()

    def span_x(y, verts, half=2.0):
        """该行附近的模型轮廓像素范围（直接用投影几何，和取点算法同一套定义）。

        不用 GL 拾取来探边界：光栅化会把最外一圈像素算成「在模型上」，
        比真实几何宽出 1~2 像素，测出来的直径就会系统性偏小。
        """
        scr, _, ok = state.camera.project(verts)
        ok = np.isfinite(scr).all(axis=1)
        sel = ok & (np.abs(scr[:, 1] - y) <= half)
        if not sel.any():
            return (float(scr[ok, 0].min()), float(scr[ok, 0].max()))
        return (float(scr[sel, 0].min()), float(scr[sel, 0].max()))

    def span_y(x, verts, half=2.0):
        scr, _, ok = state.camera.project(verts)
        ok = np.isfinite(scr).all(axis=1)
        sel = ok & (np.abs(scr[:, 0] - x) <= half)
        if not sel.any():
            return (float(scr[ok, 1].min()), float(scr[ok, 1].max()))
        return (float(scr[sel, 1].min()), float(scr[sel, 1].max()))

    def bbox(verts):
        scr, _, ok = state.camera.project(verts)
        ok = np.isfinite(scr).all(axis=1)
        return (scr[ok, 0].min(), scr[ok, 0].max(),
                scr[ok, 1].min(), scr[ok, 1].max())

    def measure(p1, p2):
        state.measures.clear()
        win.acts["measure_line3d"].trigger()
        app.processEvents()
        click(*p1)
        click(*p2)
        return state.measures.items[-1] if state.measures.items else None

    def reset(view="front", dolly=0):
        state.standard_view(view)
        state.fit_view()
        if dolly:
            state.camera.dolly(dolly)
        canvas.mark_dirty()
        app.processEvents()
        time.sleep(0.05)

    print(f"=== 圆柱：名义 \u03a6{2 * RADIUS:.1f} \u00d7 {HEIGHT:.1f}（前视）===")
    reset("front")
    xlo, xhi, ylo, yhi = bbox(cyl.vertices)
    xm, ym = (xlo + xhi) * 0.5, (ylo + yhi) * 0.5
    top, bottom = span_y(xm, cyl.vertices)
    left, right = span_x(ym, cyl.vertices)
    print(f"      可见轮廓 {right - left:.0f}px \u00d7 {bottom - top:.0f}px")

    m = measure((xm, top + 0.5), (xm, bottom - 0.5))
    check("前视长度 = 名义长度", m is not None
          and abs(m.length - HEIGHT) / HEIGHT < 0.005,
          f"L={m.length:.3f}" if m else "no measure")

    # 轮廓最外一列像素对应圆周上 cos 很小的角度，量到的直径天然略小于名义值
    # （点击精度 ±1 像素），这里按 3% 验收。
    m = measure((left + 0.5, ym), (right - 0.5, ym))
    check("前视直径 = 名义直径（光栅化极限内）", m is not None
          and abs(m.length - 2 * RADIUS) / (2 * RADIUS) < 0.03,
          f"\u2300={m.length:.3f}" if m else "no measure")

    state.measures.clear()
    win.acts["measure_circle2p"].trigger()
    app.processEvents()
    click(xm, ym)
    click(right - 0.5, ym)
    c = state.measures.items[-1].circle if state.measures.items else None
    check("前视两点圆直径 = 名义直径（光栅化极限内）",
          c is not None and abs(2 * c[1] - 2 * RADIUS) / (2 * RADIUS) < 0.03,
          f"\u2300={2 * c[1]:.3f}" if c else "no circle")

    print("\n=== 方块前表面：两点距离 = 真实尺寸，且不随缩放变化 ===")
    obj_cyl.visible = False
    obj_block.visible = True
    spans = []
    for step in (0, 6, 12):
        reset("front", step)
        pts = np.array([[-SPAN * 0.5, -BLOCK[2] * 0.5, 0.0],
                        [SPAN * 0.5, -BLOCK[2] * 0.5, 0.0]])
        scr, _, _ = state.camera.project(pts)
        m = measure((scr[0, 0], scr[0, 1]), (scr[1, 0], scr[1, 1]))
        spans.append(m.length if m else float("nan"))
        print(f"      dolly {step:2d}  距离 {state.camera.distance:7.1f}  "
              f"L={spans[-1]:.3f}")
    check("前表面两点距离 = 真实尺寸",
          abs(spans[0] - SPAN) / SPAN < 0.01, f"L={spans[0]:.3f}")
    spread = max(spans) - min(spans)
    check("放大前后读数一致（极差 < 1%）", spread / SPAN < 0.01,
          f"极差={spread:.4f}")

    print("\n=== 斜视角：三维距离 + 轴向分量 ===")
    obj_block.visible = False
    obj_cyl.visible = True
    reset("iso")
    xlo, xhi, ylo, yhi = bbox(cyl.vertices)
    xm = (xlo + xhi) * 0.5
    top, bottom = span_y(xm, cyl.vertices)
    m = measure((xm, top + 0.5), (xm, bottom - 0.5))
    check("iso 视角量出的仍是三维长度（≥ 名义长度）",
          m is not None and m.length >= HEIGHT * 0.995,
          f"L={m.length:.3f}" if m else "no measure")
    if m is not None:
        dz = float(m.points[1][2] - m.points[0][2])
        check("\u0394Z 与名义长度一致", abs(abs(dz) - HEIGHT) / HEIGHT < 0.01,
              f"\u0394Z={dz:+.3f}")
        check("状态栏给出三轴分量", "\u0394Z=" in m.delta_text(state.units),
              m.delta_text(state.units))

    print("\n=== 空白处仍可任意取点（视图平面）===")
    reset("front")
    state.measures.clear()
    win.acts["measure_line3d"].trigger()
    app.processEvents()
    click(20, 20)
    click(20, canvas.height() - 20)
    check("空白处两点完成测量", len(state.measures.items) == 1,
          state.measures.items[-1].value_text(state.units)
          if state.measures.items else "no measure")
    if state.measures.items:
        m = state.measures.items[-1]
        expect = (canvas.height() - 40) * state.camera.world_per_pixel()
        check("空白测量 = 像素距离 × 每像素世界长度",
              abs(m.length - expect) / max(expect, 1e-9) < 0.01,
              f"L={m.length:.3f} expect={expect:.3f}")

    win.close()
    print(f"\n总计: {len(PASSED)} 通过, {len(FAILED)} 失败")
    if FAILED:
        for name in FAILED:
            print(" -", name)
        return 1
    print("测量数值精度回归通过。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
