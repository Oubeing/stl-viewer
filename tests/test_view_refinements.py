"""画布细节优化回归（无需 OpenGL，offscreen 即可运行）。

覆盖本轮五个问题：

1. 坐标系箭头与 X/Y/Z 文字在任何缩放/旋转/平移下都不越出画布；
2. Save Picture 直接抓当前画布帧，不再重新渲染整个场景；
3. 三点圆是**视图平面测量**：表面点深度不参与拟合，投影出来是正圆；
4. 旋转不只有 Z：X / Y / 视线轴在鼠标、菜单、快捷键、动画里都可用；
5. 十字准线水平/垂直贯穿整个画布。

运行：QT_QPA_PLATFORM=offscreen python tests/test_view_refinements.py
"""

from __future__ import annotations

import gc
import os
import sys
import tempfile
import time

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from PyQt5 import QtCore, QtGui, QtTest, QtWidgets

from stl_viewer.camera import Camera
from stl_viewer.lighting import LightSettings
from stl_viewer.macro import SPIN_KINDS, SpinConfig, SpinEngine
from stl_viewer.measure import Measure
from stl_viewer.navigation import (ROTATE_AXES, ROTATE_Z, NavigationController,
                                   NavOptions)
from stl_viewer.png import encode_png, read_png
from stl_viewer.renderer import (AXIS_COLORS, RenderOptions, SoftwareRenderer,
                                 _clip_segment_to_rect)
from stl_viewer.scene import Scene
from stl_viewer.viewer import (_write_image_file, MainWindow, ROTATE_AXIS_MENU)

PASSED, FAILED = [], []


def check(name, condition, detail=""):
    (PASSED if condition else FAILED).append(name)
    print(f"  [{'PASS' if condition else 'FAIL'}] {name} {detail}")


def pump_until(app, predicate, timeout=15.0) -> bool:
    end = time.time() + timeout
    while time.time() < end:
        app.processEvents()
        if predicate():
            return True
        time.sleep(0.01)
    return False


def wait_frame(app, win, timeout=8.0) -> None:
    end = time.time() + timeout
    while time.time() < end:
        app.processEvents()
        if win.canvas._image is not None and not win.canvas._render_busy:
            return
        time.sleep(0.01)


def make_scene() -> Scene:
    scene = Scene()
    scene.add_box(size=(30, 30, 30), center=(0, 0, 0), name="box")
    return scene


def render_axes(size=(360, 260)) -> tuple:
    """极端放大 + 支点偏移，坐标轴端点必然跑到画布外。"""
    scene = make_scene()
    cam = Camera(*size)
    cam.set_viewport(*size)
    cam.orthographic = True
    cam.ortho_height = 6.0             # 放大到原来的 5 倍以上
    cam.target = np.array([40.0, -30.0, 25.0])
    cam.yaw, cam.pitch, cam.roll = 37.0, 24.0, 15.0
    opts = RenderOptions(show_axis=True, axis_labels=True,
                         show_grid=False, supersample=1)
    image = SoftwareRenderer().render(scene, cam, size=size, options=opts,
                                      light=LightSettings())
    return image, opts


def test_axis_clipping() -> None:
    print("\n=== 坐标系与文字不越界 ===")
    seg = _clip_segment_to_rect((-900.0, -500.0), (1200.0, 900.0),
                                20.0, 20.0, 300.0, 200.0)
    check("线段被裁进安全矩形", seg is not None)
    if seg:
        a, b = seg
        check("裁剪后端点都在矩形内",
              all(20 - 1e-6 <= p[0] <= 300 + 1e-6 and
                  20 - 1e-6 <= p[1] <= 200 + 1e-6 for p in (a, b)),
              f"{np.round(a, 1)} -> {np.round(b, 1)}")

    r = SoftwareRenderer()
    r.width, r.height = 360, 260
    from stl_viewer import font5x7
    tw, th = font5x7.text_size("XYZ", 2)
    for x, y in ((-100, -100), (5000, 5000)):
        cx, cy = r._clamped_text_xy(x, y, "XYZ", 2)
        check(f"文字坐标 ({x},{y}) 被夹进画布",
              0 <= cx <= r.width - tw and 0 <= cy <= r.height - th,
              f"({cx},{cy})")

    image, _ = render_axes()
    hh, ww = image.shape[:2]
    # 蓝色 Z 轴颜色 (0.26, 0.52, 0.90)：R 低、G 中、B 高
    blue = (image[:, :, 0] < 110) & (image[:, :, 1] > 90) & \
           (image[:, :, 1] < 180) & (image[:, :, 2] > 190)
    check("放大后仍能看到 Z 轴与文字", int(blue.sum()) > 20,
          f"{int(blue.sum())} px")
    border = np.zeros_like(blue)
    border[:3, :] = border[-3:, :] = True
    border[:, :3] = border[:, -3:] = True
    check("坐标系像素没有贴到画布边缘", not bool((blue & border).any()),
          f"边缘命中 {int((blue & border).sum())} px")
    ys, xs = np.nonzero(blue)
    if len(xs):
        check("坐标系整体在画布安全区内",
              xs.min() >= 8 and xs.max() <= ww - 8 and
              ys.min() >= 8 and ys.max() <= hh - 8,
              f"x[{xs.min()},{xs.max()}] y[{ys.min()},{ys.max()}]")


def test_crosshair() -> None:
    print("\n=== 十字准线贯穿画布 ===")
    size = (320, 240)
    cam = Camera(*size)
    opts = RenderOptions(show_axis=False, show_grid=False,
                         crosshair=True, supersample=1)
    img = SoftwareRenderer().render(Scene(), cam, size=size, options=opts,
                                    light=LightSettings())
    W, H = size
    cx, cy = W // 2, H // 2
    corner = img[5, 5].astype(int)
    samples = [img[cy, 1], img[cy, W - 2], img[1, cx], img[H - 2, cx]]
    diffs = [float(np.abs(p.astype(int) - corner).sum()) for p in samples]
    check("水平/垂直准线都到达四条边", all(d > 20 for d in diffs),
          f"色差 {[round(d) for d in diffs]}")
    gap = img[cy, cx].astype(int)
    check("中心保留小缺口", float(np.abs(gap - corner).sum()) < 20,
          f"中心色差 {float(np.abs(gap - corner).sum()):.0f}")


def test_three_point_circle() -> None:
    print("\n=== 三点圆是平面测量（投影为正圆） ===")
    m = Measure(kind="circle3p", plane_normal=np.array([0.0, 0.0, 1.0]),
                plane_point=np.zeros(3), mode="view")
    # 三个视图平面点：半径 10 的圆
    for p in ((10, 0, 0), (0, 10, 0), (-10, 0, 0)):
        m.add_point(np.array(p, dtype=float))
    # 三个表面点带有完全不同的深度（正是旧实现把圆拟成倾斜空间圆的原因）
    m.surface_points = [np.array(v, dtype=float) for v in
                        ((10, 0, 5), (0, 10, -3), (-10, 0, 8))]
    c = m.circle
    check("三点圆求解成功", c is not None)
    if c is None:
        return
    check("半径取视图平面值 10", abs(c[1] - 10.0) < 1e-9, f"R={c[1]:.6f}")
    check("圆面法向就是测量平面法向",
          abs(abs(float(np.dot(c[2], np.array([0.0, 0.0, 1.0])))) - 1.0) < 1e-9,
          f"n={np.round(c[2], 3)}")
    pts = m.effective_points()
    check("表面点深度不参与拟合",
          all(abs(float(p[2])) < 1e-12 for p in pts))
    check("3D 模式下仍是平面测量",
          (setattr(m, "mode", "3d") or True) and
          abs(m.circle[1] - 10.0) < 1e-9)
    m.mode = "view"

    # 投影到 TOP 视图：外接矩形宽高相等 = 正圆，不是椭圆
    cam = Camera(400, 400)
    cam.set_viewport(400, 400)
    cam.orthographic = True
    cam.ortho_height = 80.0
    cam.yaw, cam.pitch, cam.roll = -90.0, 89.9, 0.0
    scr, _, valid = cam.project(m.polylines()[0])
    ok = bool(valid.all())
    w = float(scr[:, 0].max() - scr[:, 0].min()) if ok else 0.0
    h = float(scr[:, 1].max() - scr[:, 1].min()) if ok else 0.0
    check("投影外接矩形是正方形（正圆）", ok and abs(w - h) < 0.05,
          f"{w:.3f} x {h:.3f} px")


def test_rotation_axes() -> None:
    print("\n=== X / Y / Z / 视线 多轴旋转 ===")
    check("可选轴集合含五个入口", ROTATE_AXES == ("free", "x", "y", "z", "view"))

    base = Camera(); base.yaw, base.pitch = 45.0, 30.0
    z = Camera(); z.yaw, z.pitch = 45.0, 30.0
    z.rotate_about_axis(20, "z")
    check("绕 Z 只改方位角", abs(z.yaw - 65.0) < 1e-9 and abs(z.pitch - 30) < 1e-9,
          f"yaw={z.yaw:.1f} pitch={z.pitch:.1f}")
    for axis in ("x", "y"):
        c = Camera(); c.yaw, c.pitch = 45.0, 30.0
        before_back, before_up = c.offset_vector(), c.up()
        c.rotate_about_axis(20, axis)
        check(f"绕 {axis.upper()} 改变相机朝向",
              float(np.linalg.norm(c.offset_vector() - before_back)) > 0.1 and
              float(np.linalg.norm(c.up() - before_up)) > 0.1,
              f"yaw={c.yaw:.1f} pitch={c.pitch:.1f} roll={c.roll:.1f}")
    v = Camera(); v.yaw, v.pitch = 45.0, 30.0
    back = v.offset_vector()
    v.rotate_about_axis(20, "view")
    check("绕视线只滚转、视线不变",
          float(np.linalg.norm(v.offset_vector() - back)) < 1e-9 and
          abs(v.roll - 20.0) < 1e-6, f"roll={v.roll:.2f}")

    nav_cam = Camera(); nav_cam.yaw, nav_cam.pitch = 45.0, 30.0
    nav = NavigationController(nav_cam, NavOptions(), {})
    nav.set_axis("x")
    up0 = nav_cam.up()
    nav.begin(ROTATE_Z, 100.0, 100.0)
    nav.drag(180.0, 100.0)
    check("定轴拖拽走所选 X 轴", nav.axis == "x" and
          float(np.linalg.norm(nav_cam.up() - up0)) > 0.1)

    for kind in ("rx", "ry"):
        cam = Camera(); cam.yaw, cam.pitch = 45.0, 30.0
        back0 = cam.offset_vector()
        engine = SpinEngine(SpinConfig(enabled=True, kind=kind, speed="high"))
        engine._last -= 0.1
        engine.tick(cam)
        check(f"旋转动画 {kind} 真的绕世界轴转",
              float(np.linalg.norm(cam.offset_vector() - back0)) > 0.05)


def test_gui_axes_and_snapshot(app) -> None:
    print("\n=== 菜单/快捷键/工具条 + Save Picture 抓帧 ===")
    win = MainWindow()
    win.resize(900, 700)
    win.show()
    app.processEvents()
    win.state.settings.default_directory = tempfile.mkdtemp()

    win._set_rotation_axis("x")
    check("选择 X 轴后状态同步", win.state.rotate_axis == "x" and
          win.canvas._nav.axis == "x" and
          win.acts["rotate_axis_x"].isChecked() and
          not win.acts["rotate_around_z"].isChecked())
    win._set_rotation_axis("z")
    check("选择 Z 轴后旧开关同步勾选",
          win.state.rotate_axis == "z" and
          win.acts["rotate_around_z"].isChecked())
    win._toggle_rotate_z(False)
    check("Ctrl+Z 关闭后回到自由旋转", win.state.rotate_axis == "free")
    check("五个轴都有快捷键",
          all(win.acts[k].shortcut().toString() for k, _, _ in ROTATE_AXIS_MENU))
    check("工具条上有旋转轴下拉",
          getattr(win, "_rotate_axis_button", None) is not None)
    win.close()


def test_save_picture_snapshot(app) -> None:
    win = MainWindow()
    win.resize(900, 700)
    win.show()
    app.processEvents()
    win.state.settings.default_directory = tempfile.mkdtemp()
    win.open_path(os.path.join(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))), "samples", "drill_blank.stl"))
    wait_frame(app, win)
    app.processEvents()

    tmp = tempfile.mkdtemp()
    pic = os.path.join(tmp, "snapshot.png")
    original_dialog = QtWidgets.QFileDialog.getSaveFileName
    original_job = MainWindow._picture_job
    QtWidgets.QFileDialog.getSaveFileName = staticmethod(
        lambda *a, **k: (pic, "PNG (*.png)"))

    calls = []

    def forbidden(*args, **kwargs):
        calls.append(args)
        raise AssertionError("普通保存不应重新渲染")

    MainWindow._picture_job = forbidden
    expected = win.canvas.snapshot()
    t0 = time.perf_counter()
    win.save_picture()
    click_ms = (time.perf_counter() - t0) * 1000.0
    check("Save Picture 立即返回", click_ms < 50.0, f"{click_ms:.1f} ms")
    check("没有触发任何重新渲染", not calls, f"{len(calls)} 次")
    ok = pump_until(app, lambda: not win._work_jobs)
    check("后台编码完成", ok)
    check("PNG 已写出", os.path.exists(pic) and os.path.getsize(pic) > 1000)
    if os.path.exists(pic):
        saved = read_png(pic)
        check("保存内容就是当前画布帧",
              saved.shape == expected.shape and
              float(np.abs(saved.astype(int) - expected.astype(int)).mean()) < 1.0,
              f"{saved.shape} vs {expected.shape}")
    check("剪贴板复制也走抓帧",
          (win.copy_picture_to_clipboard() is None) and
          not QtWidgets.QApplication.clipboard().image().isNull())

    MainWindow._picture_job = original_job
    QtWidgets.QFileDialog.getSaveFileName = original_dialog
    win.close()


def test_snapshot_memory(app) -> None:
    """抓帧必须在 QImage 释放后仍然可读 —— 真实 GL 下曾因此硬崩溃。"""
    print("\n=== 抓帧内存所有权（Save Picture 崩溃回归） ===")
    win = MainWindow()
    win.resize(640, 480)
    win.show()
    app.processEvents()
    sample = os.path.join(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))), "samples", "drill_blank.stl")
    win.open_path(sample)
    wait_frame(app, win)

    frame = win.canvas.snapshot()
    check("抓帧数组自己拥有内存", bool(frame.flags.owndata) and
          bool(frame.flags.c_contiguous), f"shape={frame.shape}")
    before = frame.copy()
    # 大量分配/释放 QImage，模拟局部 QImage 析构后像素内存被复用的情形
    for _ in range(120):
        img = QtGui.QImage(96, 96, QtGui.QImage.Format_RGB888)
        img.fill(QtGui.QColor(9, 8, 7))
        del img
    gc.collect()
    data = encode_png(frame)
    check("QImage 释放后编码不崩溃且内容不变",
          len(data) > 1000 and np.array_equal(frame, before))

    view = frame[::2, ::2]
    check("构造出非拥有内存的视图用于回归", not view.flags.owndata)
    path = os.path.join(tempfile.mkdtemp(), "view.png")
    _write_image_file(path, view)
    check("非拥有内存的视图也会被安全复制后写盘",
          os.path.exists(path) and read_png(path).shape == view.shape)
    win.close()


def test_spin_axis_link(app) -> None:
    """旋转轴、旋转动画、按钮勾选三处状态必须一致。"""
    print("\n=== 启用旋转按钮 / 旋转轴联动 ===")
    check("旋转动画新增视线滚转", "roll" in SPIN_KINDS)
    cam = Camera()
    back0 = cam.offset_vector().copy()
    engine = SpinEngine(SpinConfig(enabled=True, kind="roll", speed="high"))
    engine._last -= 0.1
    engine.tick(cam)
    check("roll 只滚转、视线不变",
          float(np.linalg.norm(cam.offset_vector() - back0)) < 1e-9 and
          abs(cam.roll) > 1.0)

    win = MainWindow()
    win.resize(640, 480)
    win.show()
    app.processEvents()
    win._set_rotation_axis("x")
    check("选 X 轴后动画方式同步为 RX",
          win.state.spin.config.kind == "rx" and
          win.acts["spin_rx"].isChecked())
    check("工具条下拉直接显示当前轴",
          "X" in win._rotate_axis_button[0].text(),
          win._rotate_axis_button[0].text())

    act = win.acts["spin_enable"]
    act.trigger()
    check("启用旋转立即生效并勾选按钮",
          win.state.spin.config.enabled and act.isChecked())
    QtTest.QTest.keyClick(win.canvas, QtCore.Qt.Key_Space)
    app.processEvents()
    check("空格关闭后按钮同步取消勾选",
          not win.state.spin.config.enabled and not act.isChecked())
    act.trigger()
    check("再点按钮可重新启动",
          win.state.spin.config.enabled and act.isChecked())
    win._set_spin_type("ry")
    check("旋转方式 RY 反向联动到 Y 轴",
          win.state.rotate_axis == "y" and win.canvas._nav.axis == "y" and
          win.acts["rotate_axis_y"].isChecked())
    win._set_rotation_axis("view")
    check("视线轴映射为 roll 动画",
          win.state.spin.config.kind == "roll" and
          win.acts["spin_roll"].isChecked())
    win._toggle_spin(False)
    check("关闭旋转后按钮回到未勾选",
          not win.state.spin.config.enabled and not act.isChecked())
    win.close()


def main() -> int:
    app = QtWidgets.QApplication(sys.argv[:1])
    test_axis_clipping()
    test_crosshair()
    test_three_point_circle()
    test_rotation_axes()
    test_gui_axes_and_snapshot(app)
    test_save_picture_snapshot(app)
    test_snapshot_memory(app)
    test_spin_axis_link(app)
    print(f"\n总计: {len(PASSED)} 通过, {len(FAILED)} 失败")
    if FAILED:
        for name in FAILED:
            print(" -", name)
        return 1
    print("画布细节优化回归通过。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
