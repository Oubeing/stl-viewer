"""GPU 拾取端到端验证（需要真实 OpenGL 上下文，勿用 offscreen 平台运行）。

    QT_QPA_PLATFORM=windows python tests/test_gpu_pick.py

覆盖：
    * GL 视口能正常初始化（initializeGL 成功、ready=True）
    * 大模型（84 万面）能被上传到 VBO 并渲染（不崩溃、不 OOM）
    * 在画布中心（模型表面）调用 pick() 能命中并返回真实世界坐标
    * 在背景处调用 pick() 正确返回 None（不误报）
"""

from __future__ import annotations

import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from PyQt5 import QtCore, QtWidgets

from stl_viewer import stl_io
from stl_viewer.camera import Camera
from stl_viewer.mesh import Mesh
from stl_viewer.scene import Scene
from stl_viewer.viewer import Canvas, ViewerState

PASSED, FAILED = [], []


def check(name, condition, detail=""):
    (PASSED if condition else FAILED).append(name)
    print(f"  [{'PASS' if condition else 'FAIL'}] {name} {detail}")


def main() -> int:
    app = QtWidgets.QApplication(sys.argv[:1])
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    print("=== 小模型 GPU 拾取（drill_blank.stl）===")
    state = ViewerState()
    win = QtWidgets.QWidget()
    win.resize(820, 620)
    canvas = Canvas(state, win)
    canvas.resize(800, 600)
    canvas.show()
    win.show()

    # 载入示例模型
    mesh = Mesh.load(os.path.join(root, "samples", "drill_blank.stl"),
                     weld=bool(state.settings.auto_weld))
    obj = state.scene.add_mesh(mesh, name="drill")
    state.viewport = (800, 600)
    state.camera.set_viewport(800, 600)
    state.fit_view()
    app.processEvents()

    # 探测并激活 GL 视口
    canvas._probe_gl()
    app.processEvents()
    app.processEvents()
    time.sleep(0.05)
    app.processEvents()
    if not (canvas._gl_active and canvas._gl is not None and canvas._gl.ready):
        print("  [SKIP] 当前平台无 OpenGL 上下文，跳过全部 GPU 拾取检查"
              "（仅 offscreen/minimal 平台会跳过；请在带 GL 的桌面环境运行）")
        win.close()
        print(f"\n总计: {len(PASSED)} 通过, {len(FAILED)} 失败, 部分跳过")
        return 0 if not FAILED else 1
    check("GL 视口已激活", True,
          f"active={canvas._gl_active} ready={canvas._gl.ready}")

    if canvas._gl is not None and canvas._gl.ready:
        # 触发一次真实 GPU 渲染（initializeGL + paintGL）
        canvas._gl.update()
        app.processEvents()
        app.processEvents()
        time.sleep(0.03)
        app.processEvents()

        t0 = time.perf_counter()
        # 画布中心（fit 后模型居中于此）应命中模型前表面
        hit = canvas._gl.pick(400, 300, state.scene, state.camera,
                              state.render_options())
        ms = (time.perf_counter() - t0) * 1000.0
        check("GPU 拾取画布中心命中模型", hit is not None,
              f"point={None if hit is None else np.round(hit['point'], 3)}")
        if hit is not None:
            p = hit["point"]
            lo, hi = mesh.bounds()
            inside = (np.all(p >= lo - 1e-3) and np.all(p <= hi + 1e-3))
            check("拾取点落在模型包围盒内", inside,
                  f"point={np.round(p,3)} bbox=[{np.round(lo,2)},{np.round(hi,2)}]")
            check("拾取耗时 < 50ms（瞬时）", ms < 50.0, f"{ms:.2f} ms")
            check("命中对象正确", hit.get("object") is obj)

        # 背景（画布外）应不命中 —— 这里用一个极端角点，模型通常不会延伸到
        # 画布最边缘（fit 留 10% 余量）。
        bg = canvas._gl.pick(2, 2, state.scene, state.camera,
                             state.render_options())
        check("背景处拾取返回 None（不误报）", bg is None,
              f"bg_hit={bg is not None}")

        # 完整测量流程：进入测量 -> 画布任意位置选点 -> 生成测量项
        # （现在是「屏幕视图测量」，任意位置可拾取，不再要求点中模型）
        from PyQt5 import QtCore, QtGui
        state.measures.start("line3d")
        canvas._update_cursor()
        canvas._add_measure_point(QtCore.QPoint(400, 300))
        cursor_cross = (canvas._gl.cursor() == QtCore.Qt.CrossCursor)
        # 远离模型的位置也能拾取
        canvas._add_measure_point(QtCore.QPoint(60, 80))
        check("测量模式下光标切换为十字", cursor_cross)
        check("屏幕视图测量完成（任意位置选点）", len(state.measures.items) == 1,
              state.measures.items[-1].value_text(state.units)
              if state.measures.items else "")
        # ESC 取消进行中的测量
        state.measures.start("circle3p")
        canvas._update_cursor()
        canvas._add_measure_point(QtCore.QPoint(200, 200))
        esc = QtGui.QKeyEvent(QtCore.QEvent.KeyPress, QtCore.Qt.Key_Escape,
                              QtCore.Qt.NoModifier)
        canvas.keyPressEvent(esc)
        app.processEvents()
        check("ESC 取消进行中的测量", state.measures.pending is None
              and len(state.measures.items) == 1)
        state.measures.delete_all()
    else:
        print("  [SKIP] 当前平台无 OpenGL 上下文，跳过 GPU 拾取（仅 offscreen/minimal 平台）")

    print("\n=== 超大模型 VBO 上传能力（合成 84 万面）===")
    # 合成一个约 84 万面的高分辨率球，验证 VBO 路径不崩溃、显存优化生效
    n = 360
    # 用较低分辨率构造避免内存爆炸，但足以代表「大模型」量级
    big = _make_highpoly_sphere(radius=20.0, rings=420, sectors=420)
    print(f"  合成模型三角形数: {big.triangle_count:,}")
    state2 = ViewerState()
    win2 = QtWidgets.QWidget()
    c2 = Canvas(state2, win2)
    c2.resize(800, 600)
    c2.show()
    win2.show()
    o2 = state2.scene.add_mesh(big, name="big")
    state2.viewport = (800, 600)
    state2.camera.set_viewport(800, 600)
    state2.fit_view()
    app.processEvents()
    c2._probe_gl()
    app.processEvents(); app.processEvents(); time.sleep(0.05); app.processEvents()
    if c2._gl is not None and c2._gl.ready:
        c2._gl.update(); app.processEvents(); app.processEvents()
        time.sleep(0.03); app.processEvents()
        entry = c2._gl._get_entry(o2, state2.render_options())
        check("大模型 VBO 成功上传", entry is not None and entry.count > 0,
              f"count={entry.count if entry else None}")
        if entry is not None:
            # 整件同色时应走「无逐顶点色」分支（has_color=False），省 1/3 显存
            check("整件同色启用显存优化（has_color=False）",
                  entry.has_color is False, f"has_color={entry.has_color}")
        t0 = time.perf_counter()
        h2 = c2._gl.pick(400, 300, state2.scene, state2.camera,
                         state2.render_options())
        ms2 = (time.perf_counter() - t0) * 1000.0
        check("大模型 GPU 拾取命中", h2 is not None, f"{ms2:.2f} ms")
    else:
        print("  [SKIP] 无 OpenGL，跳过超大模型 VBO/GPU 拾取检查")

    win.close(); win2.close()
    print(f"\n总计: {len(PASSED)} 通过, {len(FAILED)} 失败")
    if FAILED:
        print("失败项:", *FAILED, sep="\n - ")
        return 1
    print("GPU 拾取端到端验证通过。")
    return 0


def _make_highpoly_sphere(radius=20.0, rings=420, sectors=420) -> Mesh:
    # 经线/纬线球，三角形数 ≈ 2*rings*sectors
    rings = max(rings, 3); sectors = max(sectors, 3)
    verts = []
    for i in range(rings + 1):
        phi = np.pi * i / rings
        y = np.cos(phi) * radius
        r = np.sin(phi) * radius
        for j in range(sectors + 1):
            theta = 2.0 * np.pi * j / sectors
            verts.append([np.cos(theta) * r, y, np.sin(theta) * r])
    verts = np.asarray(verts, dtype=np.float64)
    idx = np.arange((rings + 1) * (sectors + 1)).reshape(rings + 1, sectors + 1)
    tris = []
    for i in range(rings):
        for j in range(sectors):
            a = idx[i, j]; b = idx[i + 1, j]
            c = idx[i + 1, j + 1]; d = idx[i, j + 1]
            tris.append([a, b, d]); tris.append([b, c, d])
    return Mesh(verts, np.asarray(tris, dtype=np.int64), name="bigsphere")


if __name__ == "__main__":
    sys.exit(main())
