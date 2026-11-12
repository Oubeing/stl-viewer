"""测量口径回归：长度 = 视图投影，不受取点深度影响（无需 OpenGL）。

用户反馈「圆是准的，但长度差得太多」：旧实现把每个点放在各自模型表面的
深度上，两点三维距离会把「往屏幕里外」的深度差也算进去。TOP 视图量一条
台阶边时，一端在凸台顶面、另一端在底面，30 mm 的边会量成 30.6 mm。

现在同一次测量的所有点都落在同一个视图平面上：

    * ``length`` / ``projected_length`` = 屏幕投影长度（默认读数）；
    * ``spatial_length`` = 两个真实表面点的三维距离（以 ``3D=`` 附带显示）；
    * 打开 Measurement > True 3D Distance 后主读数改为三维距离。

运行：QT_QPA_PLATFORM=offscreen python tests/test_measure_projection.py
"""

from __future__ import annotations

import os
import sys

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from PyQt5 import QtCore, QtWidgets

from stl_viewer.mesh import Mesh
from stl_viewer.viewer import MainWindow

PASSED, FAILED = [], []


def check(name, condition, detail=""):
    (PASSED if condition else FAILED).append(name)
    print(f"  [{'PASS' if condition else 'FAIL'}] {name} {detail}")


def box(x0, x1, z0, z1, y0=-10.0, y1=10.0):
    verts = np.array([[x0, y0, z0], [x1, y0, z0], [x1, y1, z0], [x0, y1, z0],
                      [x0, y0, z1], [x1, y0, z1], [x1, y1, z1], [x0, y1, z1]],
                     dtype=np.float64)
    faces = [(0, 3, 2), (0, 2, 1), (4, 5, 6), (4, 6, 7),
             (0, 1, 5), (0, 5, 4), (3, 7, 6), (3, 6, 2),
             (0, 4, 7), (0, 7, 3), (1, 2, 6), (1, 6, 5)]
    return verts, np.asarray(faces, dtype=np.int64)


def stepped_mesh() -> Mesh:
    """台阶件：x∈[-30,-10] 高 10；x∈[-10,30] 高 4。

    TOP 视图下两块顶面都可见，深度差 6 mm —— 正好用来区分「投影长度」
    与「三维距离」。
    """
    v1, f1 = box(-30, -10, 0, 10)
    v2, f2 = box(-10, 30, 0, 4)
    return Mesh(np.vstack([v1, v2]), np.vstack([f1, f2 + 8]), name="step")


def main() -> int:
    app = QtWidgets.QApplication(sys.argv[:1])
    win = MainWindow()
    win.resize(900, 720)
    win.show()
    app.processEvents()
    state, canvas = win.state, win.canvas
    state.settings.measure_mode = "view"
    state.measures.set_mode("view")
    win.acts["measure_true3d"].setChecked(False)

    state.scene.add_mesh(stepped_mesh(), name="step")
    state.standard_view("top", auto_fit=True)
    state.camera.set_viewport(canvas.width(), canvas.height())
    canvas.mark_dirty()
    app.processEvents()

    # 测试里显式同步建好索引；真实界面由后台线程建，点击不会等待。
    canvas._screen_picker.ensure(state.scene, state.camera)

    def click_world(p):
        scr, _, ok = state.camera.project(np.asarray([p], dtype=np.float64))
        assert bool(ok[0])
        canvas._add_measure_point(
            QtCore.QPoint(int(round(scr[0, 0])), int(round(scr[0, 1]))))

    def measure(kind, p1, p2, mode="view"):
        state.measures.clear()
        state.measures.set_mode(mode)
        state.measures.start(kind)
        click_world(p1)
        click_world(p2)
        return state.measures.items[-1] if state.measures.items else None

    print("=== TOP 视图：两端深度不同，长度仍是投影长度 ===")
    p1, p2 = (-20.0, 0.0, 10.0), (10.0, 0.0, 4.0)
    m = measure("line3d", p1, p2)
    check("测量完成", m is not None)
    if m is not None:
        check("长度 = 投影长度 30（±1.5%）", abs(m.length - 30.0) / 30.0 < 0.015,
              f"L={m.length:.3f}")
        check("默认口径下 length 就是 projected_length",
              abs(m.projected_length - m.length) < 1e-9)
        check("深度差约 6 mm（表面点）", abs(abs(m.depth_delta) - 6.0) < 0.05,
              f"depth={m.depth_delta:+.3f}")
        expect3d = float(np.hypot(30.0, 6.0))
        check("3D 读数为三维距离", m.spatial_length is not None
              and abs(m.spatial_length - expect3d) / expect3d < 0.02,
              f"3D={m.spatial_length:.3f}" if m.spatial_length else "None")
        check("状态栏 ΔZ≈0（投影口径）",
              "ΔZ=+0.000" in m.delta_text(state.units)
              or "ΔZ=-0.000" in m.delta_text(state.units),
              m.delta_text(state.units))
        check("结果文本同时给出 3D=", "3D=" in m.value_text(state.units),
              m.value_text(state.units))
        check("导出结果带投影/三维两个字段",
              m.results().get("projected_length_mm") is not None
              and m.results().get("spatial_length_mm") is not None)

    print("\n=== 3D 模式：主读数改为真实三维距离 ===")
    m3 = measure("line3d", p1, p2, mode="3d")
    check("3D 模式测量完成", m3 is not None)
    if m3 is not None:
        expect3d = float(np.hypot(30.0, 6.0))
        check("3D 模式长度 = 三维距离", abs(m3.length - expect3d) / expect3d < 0.02,
              f"L={m3.length:.3f} expect={expect3d:.3f}")
        check("3D 模式 ΔZ≈-6", "ΔZ=-6." in m3.delta_text(state.units),
              m3.delta_text(state.units))

    print("\n=== 两点圆：半径仍是视图平面内半径 ===")
    state.measures.clear()
    state.measures.set_mode("view")
    state.measures.start("circle2p")
    click_world((-20.0, 0.0, 10.0))      # 圆心在凸台顶面
    click_world((10.0, 0.0, 4.0))        # 圆周点在低一级顶面
    c = state.measures.items[-1].circle if state.measures.items else None
    check("两点圆完成", c is not None)
    if c is not None:
        check("半径 = 投影半径 30（不受 6mm 深度差影响）",
              abs(c[1] - 30.0) / 30.0 < 0.015, f"R={c[1]:.3f}")
        check("圆心深度取真实表面（z≈10）", abs(c[0][2] - 10.0) < 0.05,
              f"center={np.round(c[0], 3)}")

    print("\n=== 3D 点：坐标取真实表面点 ===")
    state.measures.clear()
    state.measures.start("point3d")
    click_world((-20.0, 0.0, 10.0))
    pt = state.measures.items[-1]
    check("3D 点 z 坐标 = 10（不是视图平面深度）",
          len(pt.surface_points) == 1 and abs(pt.surface_points[0][2] - 10.0) < 0.05,
          f"p={np.round(pt.effective_points()[0], 3)}")

    print("\n=== 空白处：仍然任意位置可测 ===")
    state.measures.clear()
    state.measures.start("line3d")
    win.state.scene.clear()
    canvas.mark_dirty()
    app.processEvents()
    canvas._screen_picker.ensure(state.scene, state.camera)
    canvas._add_measure_point(QtCore.QPoint(30, 30))
    canvas._add_measure_point(QtCore.QPoint(30, 300))
    m = state.measures.items[-1] if state.measures.items else None
    check("空场景两点完成测量", m is not None)
    if m is not None:
        expect = 270.0 * state.camera.world_per_pixel()
        check("空白测量 = 像素距离 × 每像素世界长度",
              abs(m.length - expect) / max(expect, 1e-9) < 0.02,
              f"L={m.length:.3f} expect={expect:.3f}")
        check("空白处没有 3D 读数", m.spatial_length is None)

    win.close()
    print(f"\n总计: {len(PASSED)} 通过, {len(FAILED)} 失败")
    if FAILED:
        for name in FAILED:
            print(" -", name)
        return 1
    print("测量投影口径回归通过。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
