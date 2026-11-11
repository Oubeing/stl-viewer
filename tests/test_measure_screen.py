"""屏幕视图测量回归（无需 OpenGL，offscreen 即可运行）。

验证用户需求：
    * 画布上任意位置都能拾取：点在模型上取模型表面点（真实三维尺寸），
      点在空白处投影到视图平面（过第一个已取点 / 支点）。本文件不加载模型，
      因此走的是视图平面分支，用来锁住「任意位置可测」这条行为。
    * 即便场景为空（没有载入任何模型）也能正常选点测量。
    * 已知屏幕像素距离，按 world_per_pixel 换算得到世界长度（正交模式下精确）。
    * 按 ESC 可取消进行中的测量。
    * 两点直线 / 两点圆 / 三点圆 等各类型都能从任意位置完成。

运行：QT_QPA_PLATFORM=offscreen python tests/test_measure_screen.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from PyQt5 import QtCore, QtGui, QtWidgets

from stl_viewer.camera import Camera
from stl_viewer.mesh import Mesh
from stl_viewer.viewer import Canvas, ViewerState

PASSED, FAILED = [], []


def check(name, condition, detail=""):
    (PASSED if condition else FAILED).append(name)
    print(f"  [{'PASS' if condition else 'FAIL'}] {name} {detail}")


def _make_canvas(size=400):
    state = ViewerState()
    win = QtWidgets.QWidget()
    win.resize(size + 20, size + 20)
    canvas = Canvas(state, win)
    canvas.resize(size, size)
    canvas.show()
    win.show()
    # 正交相机：屏幕像素 -> 世界 有精确的线性映射（world_per_pixel）
    cam = state.camera
    cam.set_viewport(size, size)
    cam.orthographic = True
    cam.ortho_height = 200.0          # 1 像素 = 0.5 世界单位
    cam.zoom = 1.0
    cam.target = np.zeros(3)
    cam.yaw = 0.0
    cam.pitch = 0.0
    cam.roll = 0.0
    state.viewport = (size, size)
    return state, canvas, win


def main() -> int:
    app = QtWidgets.QApplication(sys.argv[:1])

    print("=== 屏幕视图测量：任意位置选点（正交，精确映射）===")
    state, canvas, win = _make_canvas(400)
    wpp = state.camera.world_per_pixel()
    check("world_per_pixel 已知", abs(wpp - 0.5) < 1e-6, f"wpp={wpp}")

    # 画布中心投影应等于视图支点（世界原点）
    center = state.camera.ray_to_plane(200, 200)
    check("画布中心投影到支点", np.linalg.norm(center) < 1e-9,
          f"center={np.round(center,4)}")

    # 两点直线：水平拉开 100 像素 -> 世界长度 = 100 * wpp = 50.0
    state.measures.start("line3d")
    canvas._update_cursor()
    canvas._add_measure_point(QtCore.QPoint(200, 200))
    canvas._add_measure_point(QtCore.QPoint(300, 200))
    check("屏幕视图两点直线完成", len(state.measures.items) == 1,
          state.measures.items[-1].value_text(state.units)
          if state.measures.items else "")
    if state.measures.items:
        L = state.measures.items[-1].length
        check("屏幕距离换算正确（100px≈50mm）", abs(L - 100.0 * wpp) < 1e-6,
              f"L={L:.6f}")
    state.measures.delete_all()

    print("\n=== 任意位置选点：远离模型 / 角落也能拾取 ===")
    # 在画布四个角与边缘中段选点（正交，wpp=0.5）：应都能生成有效的世界点
    for name, pt in (("左上角", QtCore.QPoint(5, 5)),
                      ("右下角", QtCore.QPoint(395, 395)),
                      ("上边中段", QtCore.QPoint(200, 8)),
                      ("左边缘", QtCore.QPoint(6, 250))):
        p = state.camera.ray_to_plane(pt.x(), pt.y())
        finite = np.all(np.isfinite(p))
        check(f"任意位置投影有效（{name}）", finite,
              f"p={np.round(p,2)}")
        # 该点应落在过支点的视图平面上（到支点的 forward 投影≈0）
        fwd = state.camera.forward()
        check(f"落在视图平面上（{name}）",
              abs(float(np.dot(p - state.camera.target, fwd))) < 1e-6)

    print("\n=== 空场景也能测量（不依赖模型拾取）===")
    state.measures.start("line3d")
    canvas._add_measure_point(QtCore.QPoint(120, 120))
    canvas._add_measure_point(QtCore.QPoint(320, 260))
    check("空场景仍可完成屏幕视图测量", len(state.measures.items) == 1,
          state.measures.items[-1].value_text(state.units)
          if state.measures.items else "")
    state.measures.delete_all()

    print("\n=== ESC 取消进行中的测量 ===")
    state.measures.start("circle3p")
    canvas._update_cursor()
    canvas._add_measure_point(QtCore.QPoint(150, 150))   # 已选 1/3
    check("ESC 前测量进行中", state.measures.pending is not None)
    esc = QtGui.QKeyEvent(QtCore.QEvent.KeyPress, QtCore.Qt.Key_Escape,
                          QtCore.Qt.NoModifier)
    canvas.keyPressEvent(esc)
    app.processEvents()
    check("ESC 后测量取消", state.measures.pending is None)
    check("ESC 未误删已完成的测量", len(state.measures.items) == 0)
    # 光标应在取消后复位为手形（非十字）
    check("ESC 后光标复位", canvas.cursor() != QtCore.Qt.CrossCursor)

    print("\n=== 两点圆 / 三点圆 从任意位置完成 ===")
    state.measures.start("circle2p")
    canvas._add_measure_point(QtCore.QPoint(100, 200))
    canvas._add_measure_point(QtCore.QPoint(300, 200))
    check("两点圆完成", len(state.measures.items) == 1,
          state.measures.items[-1].value_text(state.units)
          if state.measures.items else "")
    state.measures.delete_all()

    state.measures.start("circle3p")
    canvas._add_measure_point(QtCore.QPoint(200, 100))
    canvas._add_measure_point(QtCore.QPoint(300, 250))
    canvas._add_measure_point(QtCore.QPoint(100, 250))
    check("三点圆完成", len(state.measures.items) == 1,
          state.measures.items[-1].value_text(state.units)
          if state.measures.items else "")
    state.measures.delete_all()

    print("\n=== 真实鼠标点击路径（press+release 选点）===")
    state.measures.start("line3d")
    canvas._update_cursor()
    press = QtGui.QMouseEvent(QtCore.QEvent.MouseButtonPress,
                              QtCore.QPointF(200, 200), QtCore.Qt.LeftButton,
                              QtCore.Qt.LeftButton, QtCore.Qt.NoModifier)
    release = QtGui.QMouseEvent(QtCore.QEvent.MouseButtonRelease,
                               QtCore.QPointF(200, 200), QtCore.Qt.LeftButton,
                               QtCore.Qt.NoButton, QtCore.Qt.NoModifier)
    canvas.mousePressEvent(press)
    canvas.mouseReleaseEvent(release)
    app.processEvents()
    have_after_1 = (len(state.measures.pending.points)
                   if state.measures.pending else 0)
    check("真实点击添加第 1 个测量点", have_after_1 == 1, f"have={have_after_1}")
    press2 = QtGui.QMouseEvent(QtCore.QEvent.MouseButtonPress,
                               QtCore.QPointF(260, 200), QtCore.Qt.LeftButton,
                               QtCore.Qt.LeftButton, QtCore.Qt.NoModifier)
    release2 = QtGui.QMouseEvent(QtCore.QEvent.MouseButtonRelease,
                                QtCore.QPointF(260, 200), QtCore.Qt.LeftButton,
                                QtCore.Qt.NoButton, QtCore.Qt.NoModifier)
    canvas.mousePressEvent(press2)
    canvas.mouseReleaseEvent(release2)
    app.processEvents()
    check("真实点击完成两点直线", len(state.measures.items) == 1,
          state.measures.items[-1].value_text(state.units)
          if state.measures.items else "")
    state.measures.delete_all()

    print("\n=== 两点测圆：第一点即圆心，第二点定半径 ===")
    state.measures.start("circle2p")
    canvas._add_measure_point(QtCore.QPoint(200, 200))      # 圆心 = 画布中心
    canvas._add_measure_point(QtCore.QPoint(300, 200))      # 圆周点：右移 100px
    if state.measures.items:
        it = state.measures.items[-1]
        c = it.circle
        check("两点圆以第一个点为圆心",
              c is not None and np.linalg.norm(c[0] - state.camera.target) < 1e-9,
              f"center={np.round(c[0], 3) if c else None}")
        check("半径 = 圆心到第二点的距离（100px×wpp=50）",
              c is not None and abs(c[1] - 100.0 * wpp) < 1e-6,
              f"R={c[1]:.4f}" if c else "")
        check("直径 = 2×半径",
              c is not None and abs(c[1] * 2.0 - 200.0 * wpp) < 1e-6,
              f"D={c[1] * 2:.4f}" if c else "")
    state.measures.delete_all()

    print("\n=== 手抖点击不漏点（press -> 移动 6px -> release）===")
    state.measures.start("circle2p")
    canvas._update_cursor()

    def click_with_jitter(x, y, jitter):
        ev = QtGui.QMouseEvent(QtCore.QEvent.MouseButtonPress,
                               QtCore.QPointF(x, y), QtCore.Qt.LeftButton,
                               QtCore.Qt.LeftButton, QtCore.Qt.NoModifier)
        canvas.mousePressEvent(ev)
        if jitter:
            mv = QtGui.QMouseEvent(QtCore.QEvent.MouseMove,
                                   QtCore.QPointF(x + jitter, y + jitter),
                                   QtCore.Qt.NoButton, QtCore.Qt.LeftButton,
                                   QtCore.Qt.NoModifier)
            canvas.mouseMoveEvent(mv)
        rel = QtGui.QMouseEvent(QtCore.QEvent.MouseButtonRelease,
                                QtCore.QPointF(x + jitter, y + jitter),
                                QtCore.Qt.LeftButton, QtCore.Qt.NoButton,
                                QtCore.Qt.NoModifier)
        canvas.mouseReleaseEvent(rel)
        app.processEvents()

    click_with_jitter(150, 200, 6)          # 高 DPI / 触控板常见的手抖
    have = len(state.measures.pending.points) if state.measures.pending else -1
    check("手抖 6px 仍算一次取点", have == 1, f"pts={have}")
    click_with_jitter(250, 200, 5)
    check("手抖的第二次点击也能完成测量", len(state.measures.items) == 1,
          state.measures.items[-1].value_text(state.units)
          if state.measures.items else "")
    state.measures.delete_all()

    print("\n=== 两点重合：拒绝退化测量，工具保持激活 ===")
    state.measures.start("circle2p")
    canvas._add_measure_point(QtCore.QPoint(200, 200))
    canvas._add_measure_point(QtCore.QPoint(200, 200))     # 同一像素
    check("重合两点不产生空测量", len(state.measures.items) == 0)
    check("重合时测量仍在等待（不会消失）", state.measures.pending is not None)
    check("重合时撤销刚点的点并给出提示",
          state.measures.pending is not None
          and len(state.measures.pending.points) == 1
          and state.measures.last_reject == "measure_point_too_close",
          f"reject={state.measures.last_reject}")
    canvas._add_measure_point(QtCore.QPoint(320, 160))     # 换个位置立刻可完成
    check("换位置后立刻完成", len(state.measures.items) == 1,
          state.measures.items[-1].value_text(state.units)
          if state.measures.items else "")
    state.measures.delete_all()

    print("\n=== 连续测量：完成一次后工具仍激活 ===")
    state.measures.start("line3d")
    canvas._add_measure_point(QtCore.QPoint(100, 100))
    canvas._add_measure_point(QtCore.QPoint(200, 100))
    check("第一次测量完成", len(state.measures.items) == 1)
    check("完成后同一工具保持激活",
          state.measures.pending is not None
          and state.measures.pending.kind == "line3d")
    canvas._add_measure_point(QtCore.QPoint(120, 300))
    canvas._add_measure_point(QtCore.QPoint(220, 300))
    check("无需回菜单即可继续测第二次", len(state.measures.items) == 2)
    state.measures.delete_all()

    print("\n=== 测量中双击 / 拖动 不误触视图操作 ===")
    state.measures.start("circle2p")
    canvas._add_measure_point(QtCore.QPoint(180, 180))
    dist_before = state.camera.distance
    dbl = QtGui.QMouseEvent(QtCore.QEvent.MouseButtonDblClick,
                            QtCore.QPointF(180, 180), QtCore.Qt.LeftButton,
                            QtCore.Qt.LeftButton, QtCore.Qt.NoModifier)
    canvas.mouseDoubleClickEvent(dbl)
    app.processEvents()
    check("测量中双击不触发 zoom fit",
          abs(state.camera.distance - dist_before) < 1e-9,
          f"dist {dist_before:.4f} -> {state.camera.distance:.4f}")
    # 明显拖动：应转为旋转，且不误取点
    state.measures.cancel()
    state.measures.start("line3d")
    yaw0 = state.camera.yaw
    canvas.mousePressEvent(QtGui.QMouseEvent(
        QtCore.QEvent.MouseButtonPress, QtCore.QPointF(200, 200),
        QtCore.Qt.LeftButton, QtCore.Qt.LeftButton, QtCore.Qt.NoModifier))
    for i in range(1, 7):
        canvas.mouseMoveEvent(QtGui.QMouseEvent(
            QtCore.QEvent.MouseMove, QtCore.QPointF(200 + i * 12, 200),
            QtCore.Qt.NoButton, QtCore.Qt.LeftButton, QtCore.Qt.NoModifier))
    canvas.mouseReleaseEvent(QtGui.QMouseEvent(
        QtCore.QEvent.MouseButtonRelease, QtCore.QPointF(272, 200),
        QtCore.Qt.LeftButton, QtCore.Qt.NoButton, QtCore.Qt.NoModifier))
    app.processEvents()
    check("测量态下拖动旋转生效", abs(state.camera.yaw - yaw0) > 1.0,
          f"yaw {yaw0:.1f} -> {state.camera.yaw:.1f}")
    check("拖动旋转不误取点",
          state.measures.pending is None
          or len(state.measures.pending.points) == 0)
    state.measures.cancel()

    # 收尾：停掉「视图静止后预建索引」的计时器和后台渲染池，
    # 否则退出解释器时可能还在等一个正在跑的光栅化任务（表现为脚本不返回）。
    canvas._pick_prepare_timer.stop()
    canvas.shutdown()
    win.close()
    app.processEvents()
    print(f"\n总计: {len(PASSED)} 通过, {len(FAILED)} 失败")
    if FAILED:
        print("失败项:")
        for f in FAILED:
            print(" -", f)
        return 1
    print("屏幕视图测量回归通过。")
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)          # 见文件末尾说明：不销毁局部 Qt 对象，避免线程池互等


if __name__ == "__main__":
    # Qt 的线程池析构会在主线程里等后台光栅化线程，而等待期间不释放 GIL，
    # 无人取景时会互相等死（应用关窗口由 shutdown() 的轮询兜住，不受影响）。
    # 命令行跑测试时直接结束进程，保证有确定的退出码。
    _code = main()
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(_code)
