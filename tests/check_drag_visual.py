"""可视化验证：拖拽中（降分辨率）与静止（全分辨率）的画面应完全对齐。"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.pop("QT_QPA_PLATFORM", None)   # 用真实平台，保证字体正常

import numpy as np
from PyQt5 import QtCore, QtGui, QtWidgets

from stl_viewer import png
from stl_viewer.viewer import MainWindow, ViewerState


def main():
    app = QtWidgets.QApplication(sys.argv[:1])
    state = ViewerState(os.path.join("samples", "_gui_settings.ini"))
    win = MainWindow(state)
    win.resize(1200, 780)
    win.show()
    app.processEvents()

    win.open_path(os.path.join("samples", "drill_blank.stl"))
    app.processEvents()

    canvas = win.canvas
    print("canvas size:", canvas.width(), canvas.height())

    def ev(kind, pos, button, buttons, mods=QtCore.Qt.NoModifier):
        return QtGui.QMouseEvent(kind, QtCore.QPointF(*pos), button, buttons, mods)

    # 模拟左键拖拽（保持按下，捕获“拖拽中”这一帧）
    canvas.mousePressEvent(ev(QtCore.QEvent.MouseButtonPress, (520, 360),
                              QtCore.Qt.LeftButton, QtCore.Qt.LeftButton))
    canvas.mouseMoveEvent(ev(QtCore.QEvent.MouseMove, (640, 430),
                             QtCore.Qt.NoButton, QtCore.Qt.LeftButton))
    canvas._dirty = True
    canvas._render()
    inter = canvas._buffer.copy()
    print("drag  scale=%.2f  img=%s  %dms" % (
        canvas._adaptive_scale, inter.shape, state.last_render_seconds * 1000))

    # 松手 → 先出快帧，再精修为全分辨率（渐进式渲染）
    canvas.mouseReleaseEvent(ev(QtCore.QEvent.MouseButtonRelease, (640, 430),
                                QtCore.Qt.LeftButton, QtCore.Qt.NoButton))
    canvas._dirty = True
    canvas._render()
    quick = canvas._buffer.copy()
    print("松手快帧 img=%s  %dms" % (quick.shape, state.last_render_seconds * 1000))

    canvas._refine()                 # 等价于空闲计时器触发
    canvas._dirty = True
    canvas._render()
    full = canvas._buffer.copy()
    print("精修全质量 img=%s  %dms" % (full.shape, state.last_render_seconds * 1000))

    # 把拖拽帧放大到全分辨率，比较内容是否在同一位置
    def resize_nn(img, w, h):
        ys = (np.arange(h) * img.shape[0] / h).astype(int)
        xs = (np.arange(w) * img.shape[1] / w).astype(int)
        return img[ys][:, xs]

    up = resize_nn(inter, full.shape[1], full.shape[0])
    diff = np.abs(up.astype(int) - full.astype(int)).mean()
    print("拖拽帧与全分辨率帧平均像素差: %.2f / 255" % diff)
    # 比例检查：降分辨率位图必须与全分辨率同宽高比，否则放大就是拉伸
    print("宽高比 拖拽 %.4f / 全质量 %.4f" %
          (inter.shape[1] / inter.shape[0], full.shape[1] / full.shape[0]))

    # 并排输出对比图
    gap = np.full((full.shape[0], 12, 3), 40, np.uint8)
    combo = np.concatenate([full, gap, up], axis=1)
    png.write_png(os.path.join("samples", "_drag_compare.png"), combo)

    # 整窗截图（按拖拽后的姿态）
    canvas._interactive = False
    canvas._quick = False
    canvas._dirty = True
    canvas._render()
    app.processEvents()
    win.grab().save(os.path.join("samples", "preview_gui.png"))
    print("已保存 samples/_drag_compare.png 与 samples/preview_gui.png")

    win.close()
    return 0 if diff < 18 else 1


if __name__ == "__main__":
    sys.exit(main())
