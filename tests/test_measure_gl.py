"""真实 OpenGL 环境下的测量点击回归（需有 GL 上下文，否则自动跳过）。

覆盖用户实际报障场景：
    * 点「两点测圆」后必须在画布上真正拾取到点（事件穿过 GL/叠加层到 Canvas）
    * 第一点为圆心，第二点决定半径；点下圆心后圆随光标实时变化
    * 手抖（按下后移动数像素）仍算一次取点
    * 两点重合被拒绝而不是产生空测量
    * 完成一次后同工具保持激活（连续测量）
    * ESC 取消

运行：QT_QPA_PLATFORM=windows python tests/test_measure_gl.py
"""

from __future__ import annotations

import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from PyQt5 import QtCore, QtGui, QtTest, QtWidgets

from stl_viewer.viewer import MainWindow

PASSED, FAILED, SKIPPED = [], [], []


def check(name, condition, detail=""):
    (PASSED if condition else FAILED).append(name)
    print(f"  [{'PASS' if condition else 'FAIL'}] {name} {detail}")


def main() -> int:
    app = QtWidgets.QApplication(sys.argv[:1])
    win = MainWindow()
    win.resize(1100, 800)
    win.show()
    app.processEvents()
    canvas, state = win.canvas, win.state

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    model = os.path.join(root, "samples", "drill_blank.stl")
    win.open_path(model)
    for _ in range(20):
        app.processEvents()
        time.sleep(0.02)
    state.standard_view("front")
    state.fit_view()
    canvas.mark_dirty()
    canvas._probe_gl()
    for _ in range(5):
        app.processEvents()
        time.sleep(0.02)

    if not (canvas._gl_active and canvas._gl is not None and canvas._gl.ready):
        print("  [SKIP] 当前环境没有可用的 OpenGL 上下文，跳过（请在有显卡的桌面运行）")
        win.close()
        return 0

    check("GL 视口已激活", canvas._gl.ready)

    # 用户报障：「第一个点就点不了」。根因是点击走 GL 深度拾取（离屏 FBO +
    # glReadPixels），在某些驱动/上下位状态下一按就卡住。现在取点全部走
    # 屏幕投影索引（纯 CPU），这里把 GL 拾取挂上计数器来钉住这条约束。
    gl_pick_calls = {"n": 0}
    _orig_gl_pick = canvas._gl.pick

    def _counting_pick(*args, **kwargs):
        gl_pick_calls["n"] += 1
        return _orig_gl_pick(*args, **kwargs)

    canvas._gl.pick = _counting_pick

    W, H = canvas.width(), canvas.height()
    cx, cy = W // 2, H // 2

    def click(x, y, jitter=0):
        QtTest.QTest.mousePress(canvas, QtCore.Qt.LeftButton, QtCore.Qt.NoModifier,
                                QtCore.QPoint(x, y))
        if jitter:
            app.sendEvent(canvas, QtGui.QMouseEvent(
                QtCore.QEvent.MouseMove, QtCore.QPointF(x + jitter, y + jitter),
                QtCore.Qt.NoButton, QtCore.Qt.LeftButton, QtCore.Qt.NoModifier))
        QtTest.QTest.mouseRelease(canvas, QtCore.Qt.LeftButton, QtCore.Qt.NoModifier,
                                  QtCore.QPoint(x + jitter, y + jitter))
        app.processEvents()

    print("=== 两点测圆：点击必须真的落在画布上（穿过 GL 视口）===")
    win.acts["measure_circle2p"].trigger()
    app.processEvents()
    check("进入测量态", state.measures.pending is not None)
    check("光标切到十字", canvas.cursor().shape() == QtCore.Qt.CrossCursor)

    click(cx, cy)                                   # 圆心
    have = len(state.measures.pending.points) if state.measures.pending else -1
    check("第一次点击成功取到圆心", have == 1, f"pts={have}")

    # 圆心已定：把光标放到不同半径处，圆应随之变化（预览数据）。
    # 取点会带上模型真实深度，因此用画布自己的取点函数算期望值，
    # 这里要验证的是「预览随光标移动、半径按光标位置确定」这一行为。
    cam = state.camera
    pending = state.measures.pending
    center = pending.points[0]
    # _measure_point_at 返回 (view_point, surface_point, plane_point, normal)；
    # 测量平面点与预览半径都用 view_point，和当前 CAD 投影口径一致。
    p_small = canvas._measure_point_at(cx + 60, cy, pending)[0]
    p_large = canvas._measure_point_at(cx + 140, cy, pending)[0]
    r_small = float(np.linalg.norm(p_small - center))
    r_large = float(np.linalg.norm(p_large - center))
    check("半径随光标位置变化", r_large > r_small > 0,
          f"r_small={r_small:.2f} r_large={r_large:.2f}")
    check("半径随光标距离单调增大", r_large > r_small * 1.5,
          f"ratio={r_large / max(r_small, 1e-9):.3f}")

    # 预览与最终结果必须一致：先在目标像素处算一次期望值，再点下去
    rim_pt = canvas._measure_point_at(cx + 145, cy + 5, pending)[0]
    click(cx + 140, cy, jitter=5)                   # 手抖着点圆周点
    check("手抖点击也能完成圆", len(state.measures.items) == 1,
          state.measures.items[-1].value_text(state.units)
          if state.measures.items else "")
    if state.measures.items:
        c = state.measures.items[-1].circle
        r_expect = float(np.linalg.norm(rim_pt - center))
        check("圆心 = 第一次点击的位置",
              c is not None and np.allclose(c[0], center, atol=1e-9))
        check("半径 = 圆心到实际落点的距离",
              c is not None and abs(c[1] - r_expect) < 1e-3,
              f"R={c[1]:.4f} expect={r_expect:.4f}")
    check("完成后工具仍激活（连续测量）", state.measures.pending is not None)

    print("\n=== 重合点应被拒绝 ===")
    n_items = len(state.measures.items)
    click(W // 3, H // 3)
    click(W // 3, H // 3)
    check("重合两点不产生空测量", len(state.measures.items) == n_items)
    check("测量继续等待有效点", state.measures.pending is not None)

    print("\n=== ESC 取消 ===")
    ev = QtGui.QKeyEvent(QtCore.QEvent.KeyPress, QtCore.Qt.Key_Escape,
                         QtCore.Qt.NoModifier)
    app.sendEvent(canvas, ev)
    app.processEvents()
    check("ESC 结束测量", state.measures.pending is None)
    check("ESC 后光标复位",
          canvas.cursor().shape() != QtCore.Qt.CrossCursor)

    print("\n=== 取点路径不依赖 GL 拾取（避免驱动层卡死点击）===")
    check("整个测量过程没有调用 GL 深度拾取",
          gl_pick_calls["n"] == 0, f"calls={gl_pick_calls['n']}")
    canvas._gl.pick = _orig_gl_pick

    win.close()
    print(f"\n总计: {len(PASSED)} 通过, {len(FAILED)} 失败")
    if FAILED:
        for f in FAILED:
            print(" -", f)
        return 1
    print("真实 GL 测量点击回归通过。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
