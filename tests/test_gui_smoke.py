"""图形界面冒烟测试（离屏运行，不需要真实显示器）。

    QT_QPA_PLATFORM=offscreen python tests/test_gui_smoke.py

覆盖：主窗口/菜单构建、对象面板、显示模式切换、单位与语言切换、
测量交互、旋转动画、图片渲染与导出、设置存取、STL 打开与导出。
"""

from __future__ import annotations

import os
import sys
import tempfile

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np                                    # noqa: E402
from PyQt5 import QtCore, QtWidgets                   # noqa: E402

from stl_viewer import png, stl_io                # noqa: E402
from stl_viewer.viewer import MainWindow, ViewerState  # noqa: E402

PASSED, FAILED = [], []


def check(name, condition, detail=""):
    (PASSED if condition else FAILED).append(name)
    print(f"  [{'PASS' if condition else 'FAIL'}] {name} {detail}")


def main() -> int:
    tmp = tempfile.mkdtemp(prefix="numstl_gui_")
    settings_path = os.path.join(tmp, "settings.ini")

    app = QtWidgets.QApplication(sys.argv[:1])
    state = ViewerState(settings_path)
    window = MainWindow(state)
    window.showMaximized()
    app.processEvents()

    print("=== 窗口与菜单 ===")
    check("主窗口创建", window.isVisible())
    check("启动窗口最大化", window.isMaximized())
    check("启动无任何 objects",
          not state.scene.objects and window.panel.list.count() == 0,
          f"{window.panel.list.count()} 项")
    titles = [m.title() for m in window.menuBar().findChildren(QtWidgets.QMenu)
              if m.parent() is window.menuBar()]
    check("顶级菜单", len([m for m in window.menuBar().actions()
                          if m.menu()]) >= 7, str(titles[:9]))
    check("工具条动作", len(window.toolbar.actions()) >= 10)
    check("独立视图工具条", window.view_toolbar is not None
          and len(window.view_toolbar.actions()) >= 15,
          f"{len(window.view_toolbar.actions())} 项")
    view_keys = ("view_iso", "view_iso_back", "view_front", "view_back",
                 "view_left", "view_right", "view_top", "view_bottom")
    check("标准视图全部直达工具栏",
          all(k in window.acts for k in view_keys))
    toolbar_acts = list(window.view_toolbar.actions())
    check("缩放/视图选项已提到工具栏",
          any(a is window.acts["zoom_in"] for a in toolbar_acts)
          and any(a is window.acts["zoom_out"] for a in toolbar_acts)
          and any(a is window.acts["view_options"] for a in toolbar_acts))
    check("自动适配开关存在且可选",
          "auto_fit_views" in window.acts
          and window.acts["auto_fit_views"].isCheckable())
    # 离屏平台没有 GL 上下文，必须自动回退到软件渲染，不能创建坏控件
    check("无 GL 时自动回退软件渲染",
          window.canvas._gl is None and not window.canvas._gl_active)

    # 启动本身不加载任何模型；后续交互测试需要的几何体在这里显式载入。
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    window.open_path(os.path.join(root, "samples", "drill_blank.stl"))
    app.processEvents()
    check("载入模型后对象面板有内容", window.panel.list.count() >= 1,
          f"{window.panel.list.count()} 项")

    print("\n=== 标准视图 / Zoom Fit / 灵敏度 ===")
    state.settings.auto_fit_views = True
    window.acts["auto_fit_views"].setChecked(True)
    window._standard_view("front")
    d_front = state.camera.distance
    window._standard_view("top")
    check("工具栏切换视角改变朝向", abs(state.camera.pitch) > 80,
          f"pitch={state.camera.pitch:.1f}")
    check("切换视角自动重新 Zoom Fit",
          abs(state.camera.distance - d_front) > 1e-6,
          f"front={d_front:.2f} top={state.camera.distance:.2f}")
    verts = state.scene.objects[0].mesh.vertices
    scr, _, valid = state.camera.project(verts[::max(1, len(verts) // 5000)])
    ok = valid & np.isfinite(scr).all(axis=1)
    check("fit 后模型完整位于画布内",
          ok.any() and (scr[ok, 0] >= -1).all()
          and (scr[ok, 0] <= window.canvas.width() + 1).all()
          and (scr[ok, 1] >= -1).all()
          and (scr[ok, 1] <= window.canvas.height() + 1).all())
    d_auto = state.camera.distance
    state.settings.auto_fit_views = False
    window._standard_view("front")
    check("关闭自动适配后保留当前缩放",
          abs(state.camera.distance - d_auto) < 1e-9)
    state.settings.auto_fit_views = True
    window._standard_view("iso")
    check("单次大位移被限幅",
          abs(window.canvas._soft_step(500, 60)) <= 60.0)
    # CAD 手感：位移严格线性，小幅微调不再被额外增益压扁
    check("小幅位移线性保持（CAD 手感）",
          abs(window.canvas._soft_step(4, 60) - 4.0) < 1e-9)
    speed = state.settings.zoom_speed
    state.settings.zoom_speed = 0.2
    # 用 visible_height（正交改 ortho_height、透视改 distance，二者都体现在
    # 可见世界高度上）测量缩放效果，使断言与投影类型无关。
    h0 = state.camera.visible_height()
    window._zoom(0.8)
    ratio = state.camera.visible_height() / h0
    window._zoom(1.25)
    state.settings.zoom_speed = speed
    check("缩放按钮按灵敏度精确生效", abs(ratio - 0.8) < 1e-6,
          f"ratio={ratio:.4f}")

    print("\n=== 渲染 ===")
    window.canvas.resize(640, 480)
    window.canvas._render()
    check("画布成像", window.canvas._image is not None
          and not window.canvas._image.isNull())
    check("渲染耗时记录", state.last_render_seconds >= 0.0,
          f"{state.last_render_seconds * 1000:.0f} ms")

    print("\n=== 后台异步渲染 ===")
    import time
    canvas0 = window.canvas
    canvas0._dirty = True
    t0 = time.perf_counter()
    canvas0._request_render()
    submit_ms = (time.perf_counter() - t0) * 1000.0
    token = canvas0._token
    check("渲染任务提交到后台", canvas0._render_busy)
    check("提交帧不阻塞 UI 线程", submit_ms < 20.0, f"{submit_ms:.1f} ms")
    app.processEvents()
    canvas0._pool.waitForDone(3000)
    app.processEvents()
    app.processEvents()
    # 全质量精修帧可能紧跟着又接单（token 递增），因此验证「已发布到该帧
    # 或更新的帧」，不要求渲染池此刻空闲。
    check("后台帧自动回传并发布",
          canvas0._last_frame_token >= token
          and canvas0._image is not None,
          f"submitted={token}, published={canvas0._last_frame_token}")

    print("\n=== 显示模式 ===")
    window._set_wire(True)
    check("线框模式", state.options.wire and not state.options.solid)
    window._set_solid(True)
    check("实体模式", state.options.solid and not state.options.wire)
    window._set_color_mode("auto")
    check("自动配色", state.options.color_mode == "auto"
          and state.scene.objects[0].auto_color)
    window._toggle_transparent(True)
    check("透明显示", state.options.transparent)
    window._toggle_transparent(False)
    window._toggle_edges(True)
    check("显示棱边", state.options.show_edges)
    window._toggle_crosshair(True)
    check("十字准线", state.options.crosshair)
    window._toggle_ortho(True)
    check("正交投影", state.camera.orthographic)
    window._toggle_ortho(False)
    window._set_light("down")
    check("光源切换", state.light.primary == "down")
    window._reset_light()
    check("光照复位", state.light.primary == "up_left")

    print("\n=== 视图操作 ===")
    yaw = state.camera.yaw
    window._standard_view("top")
    check("标准视图", abs(state.camera.pitch) > 80)
    window._standard_view("iso")
    window._fit()
    check("适应窗口", state.camera.distance > 0)
    window._reset_center()
    check("复位中心", np.allclose(state.camera.target, 0))
    window._zoom(0.8)
    check("缩放", state.camera.zoom != 1.0 or state.camera.distance > 0)

    print("\n=== 鼠标交互 ===")
    from PyQt5 import QtGui

    def drag(canvas, start, end, button=QtCore.Qt.LeftButton,
             mods=QtCore.Qt.NoModifier):
        canvas.mousePressEvent(QtGui.QMouseEvent(
            QtCore.QEvent.MouseButtonPress, QtCore.QPointF(*start),
            button, button, mods))
        canvas.mouseMoveEvent(QtGui.QMouseEvent(
            QtCore.QEvent.MouseMove, QtCore.QPointF(*end),
            QtCore.Qt.NoButton, button, mods))
        canvas.mouseReleaseEvent(QtGui.QMouseEvent(
            QtCore.QEvent.MouseButtonRelease, QtCore.QPointF(*end),
            button, QtCore.Qt.NoButton, mods))

    canvas = window.canvas
    canvas.resize(640, 480)
    yaw0, pitch0 = state.camera.yaw, state.camera.pitch
    pivot0 = state.camera.center_world_point()
    drag(canvas, (320, 240), (420, 300), QtCore.Qt.LeftButton)
    check("左键拖拽旋转", abs(state.camera.yaw - yaw0) > 1.0
          and abs(state.camera.pitch - pitch0) > 1.0,
          f"d_yaw={state.camera.yaw - yaw0:.1f} d_pitch={state.camera.pitch - pitch0:.1f}")
    check("旋转以画布中心为支点", np.allclose(state.camera.pivot, pivot0))

    tgt = state.camera.target.copy()
    drag(canvas, (320, 240), (400, 240), QtCore.Qt.RightButton)
    check("右键拖拽平移", not np.allclose(tgt, state.camera.target))
    check("右键拖动不弹菜单", canvas._drag_mode is None)
    scr, _, _ = state.camera.project(state.camera.target[None, :])
    check("平移后支点仍投影在画布中心",
          np.allclose(scr[0], (canvas.width() / 2, canvas.height() / 2),
                      atol=1e-6), str(scr[0]))

    d0 = state.camera.distance
    drag(canvas, (320, 240), (320, 180), QtCore.Qt.LeftButton,
         QtCore.Qt.AltModifier)
    check("Alt+左键缩放", abs(state.camera.distance - d0) > 1e-6)

    yaw0 = state.camera.yaw
    drag(canvas, (320, 240), (400, 240), QtCore.Qt.LeftButton,
         QtCore.Qt.ControlModifier)
    check("Ctrl+左键绕 Z", abs(state.camera.yaw - yaw0) > 1.0
          and abs(state.camera.pitch - pitch0) >= 0.0)

    # 拖拽时降分辨率（交互质量）；位图宽高比必须与窗口严格一致，
    # 否则放大铺满视口时模型会被拉扁/拉长。
    canvas._interactive = True
    canvas._quick = True
    canvas._dirty = True
    canvas._render()
    iw, ih = canvas._image.width(), canvas._image.height()
    check("交互渲染降采样", iw <= canvas.width() and iw >= canvas.MIN_RENDER_EDGE,
          f"{iw}x{ih}")
    check("交互位图保持窗口宽高比",
          abs(iw / ih - canvas.width() / canvas.height()) < 0.02,
          f"{iw / ih:.4f} vs {canvas.width() / canvas.height():.4f}")

    # 渐进式渲染：松手瞬间先出快帧（不阻塞事件循环），随后精修为全质量
    canvas._interactive = False
    canvas._quick = True
    canvas._dirty = True
    canvas._render()
    quick_w = canvas._image.width()
    check("松手后先出快帧", quick_w < canvas.width(),
          f"快帧 {quick_w} < 全质量 {canvas.width()}")
    canvas._refine()                 # 等价于空闲计时器触发
    canvas._dirty = True
    canvas._render()
    check("精修后恢复全分辨率", canvas._image.width() == canvas.width(),
          f"{canvas._image.width()} == {canvas.width()}")

    # 灵敏度可调：旋转量应严格与设置成正比
    state.settings.rotate_speed = 0.20
    y0 = state.camera.yaw
    drag(canvas, (300, 200), (400, 200), QtCore.Qt.LeftButton)
    slow = abs(state.camera.yaw - y0)
    state.settings.rotate_speed = 0.40
    y1 = state.camera.yaw
    drag(canvas, (300, 200), (400, 200), QtCore.Qt.LeftButton)
    fast = abs(state.camera.yaw - y1)
    check("旋转灵敏度与设置成正比", slow > 1.0 and abs(fast - 2.0 * slow) < 0.05 * slow,
          f"0.20 -> {slow:.2f}°, 0.40 -> {fast:.2f}°")
    state.settings.rotate_speed = 0.20

    # 缩放灵敏度同样可调
    state.settings.zoom_speed = 0.10
    d0 = state.camera.distance
    canvas.wheelEvent(QtGui.QWheelEvent(
        QtCore.QPointF(320, 240), QtCore.QPointF(320, 240),
        QtCore.QPoint(0, 0), QtCore.QPoint(0, 120),
        QtCore.Qt.NoButton, QtCore.Qt.NoModifier, QtCore.Qt.NoScrollPhase, False))
    one_notch = abs(np.log(d0 / state.camera.distance))
    check("滚轮缩放灵敏度可调", 0.05 < one_notch < 0.2, f"每档 {one_notch:.3f}")
    check("自适应质量在范围内", canvas.MIN_QUALITY <= canvas._adaptive_scale <= 1.0,
          f"scale={canvas._adaptive_scale:.2f}")

    # 动画期间也应降分辨率
    canvas.set_animating(True)
    check("动画降分辨率标志", canvas._animating)
    canvas.set_animating(False)
    check("动画结束恢复", not canvas._animating)

    print("\n=== CAD 风格导航 ===")
    from stl_viewer import mathutil as mu
    from stl_viewer.navigation import NavOptions

    cam = state.camera
    ax, ay = 150.0, 110.0
    p0, n0 = cam.target.copy(), cam.forward().copy()
    before = mu.ray_plane_intersection(*cam.ray(ax, ay), p0, n0)
    canvas.wheelEvent(QtGui.QWheelEvent(
        QtCore.QPointF(ax, ay), QtCore.QPointF(ax, ay),
        QtCore.QPoint(0, 0), QtCore.QPoint(0, 120),
        QtCore.Qt.NoButton, QtCore.Qt.NoModifier, QtCore.Qt.NoScrollPhase, False))
    after = mu.ray_plane_intersection(*cam.ray(ax, ay), p0, n0)
    drift = np.linalg.norm(before - after)
    check("滚轮以光标为锚点缩放（zoom to cursor）", drift < 1e-6,
          f"锚点漂移={drift:.2e}")
    check("滚轮进入交互画质窗口", canvas._nav_active
          and bool(canvas._frame_params()[3]))
    canvas._refine_timer.stop()
    canvas.mark_dirty()
    check("交互中不会插入全质量精修帧", not canvas._refine_timer.isActive())
    canvas._finish_nav()
    check("停手后退出交互窗口", not canvas._nav_active)
    canvas._refine_timer.stop()

    # 中键按住后再按左键 = CATIA 式旋转
    canvas.mousePressEvent(QtGui.QMouseEvent(
        QtCore.QEvent.MouseButtonPress, QtCore.QPointF(300, 200),
        QtCore.Qt.MiddleButton, QtCore.Qt.MiddleButton, QtCore.Qt.NoModifier))
    check("中键单独按下为平移", canvas._drag_mode == "pan")
    combo = QtCore.Qt.MiddleButton | QtCore.Qt.LeftButton
    canvas.mousePressEvent(QtGui.QMouseEvent(
        QtCore.QEvent.MouseButtonPress, QtCore.QPointF(300, 200),
        QtCore.Qt.LeftButton, combo, QtCore.Qt.NoModifier))
    check("中键+左键组合为旋转", canvas._drag_mode == "rotate")
    yaw0 = cam.yaw
    canvas.mouseMoveEvent(QtGui.QMouseEvent(
        QtCore.QEvent.MouseMove, QtCore.QPointF(360, 200),
        QtCore.Qt.NoButton, combo, QtCore.Qt.NoModifier))
    check("组合键拖动确实旋转", abs(cam.yaw - yaw0) > 1.0,
          f"d_yaw={cam.yaw - yaw0:.2f}")
    canvas.mouseReleaseEvent(QtGui.QMouseEvent(
        QtCore.QEvent.MouseButtonRelease, QtCore.QPointF(360, 200),
        QtCore.Qt.LeftButton, QtCore.Qt.MiddleButton, QtCore.Qt.NoModifier))
    canvas.mouseReleaseEvent(QtGui.QMouseEvent(
        QtCore.QEvent.MouseButtonRelease, QtCore.QPointF(360, 200),
        QtCore.Qt.MiddleButton, QtCore.Qt.NoButton, QtCore.Qt.NoModifier))

    # 轨迹球自由旋转 + 惯性
    canvas.set_nav_options(NavOptions(style="catia", rotate_mode="arcball",
                                      inertia=True))
    roll0 = cam.roll
    drag(canvas, (300, 200), (360, 230), QtCore.Qt.LeftButton)
    check("轨迹球旋转可产生滚转", abs(cam.roll - roll0) > 0.1,
          f"d_roll={cam.roll - roll0:.2f}")
    check("开启惯性后释放启动动画", canvas._spin_timer.isActive())
    canvas._spin_timer.stop()

    # 绕按下点旋转（Center on point）
    canvas.set_nav_options(NavOptions(pivot="cursor", inertia=False))
    expected = cam.ray_to_plane(200, 150)
    canvas.mousePressEvent(QtGui.QMouseEvent(
        QtCore.QEvent.MouseButtonPress, QtCore.QPointF(200, 150),
        QtCore.Qt.LeftButton, QtCore.Qt.LeftButton, QtCore.Qt.NoModifier))
    check("按下点成为新的旋转支点", np.allclose(cam.target, expected))
    canvas.mouseReleaseEvent(QtGui.QMouseEvent(
        QtCore.QEvent.MouseButtonRelease, QtCore.QPointF(200, 150),
        QtCore.Qt.LeftButton, QtCore.Qt.NoButton, QtCore.Qt.NoModifier))
    canvas.set_nav_options(NavOptions())
    canvas._queue_status()
    check("状态栏刷新合并到定时器", canvas._status_timer.isActive())

    print("\n=== 单位与语言 ===")
    window.set_units("inch")
    check("切换英寸", state.units.label == "in" and state.settings.units == "inch")
    window.set_units("mm")
    check("切回毫米", state.units.label == "mm")
    window.set_language("CH")
    check("切换中文", state.tr.tr("file") == "文件"
          and window.menuBar().actions()[0].text() == "文件")
    window.set_language("EN")
    check("切回英文", state.tr.tr("file") == "File")

    print("\n=== 测量 ===")
    window.canvas.start_measure("line3d")
    check("进入测量", state.measures.pending is not None)
    window.canvas._add_measure_point(QtCore.QPoint(300, 240))
    window.canvas._add_measure_point(QtCore.QPoint(380, 300))
    check("测量完成", len(state.measures.items) == 1,
          state.measures.items[-1].value_text(state.units) if state.measures.items else "")
    window.panel.refresh()
    check("测量面板刷新", window.panel.measure_list.count() == 1)
    window._delete_last_measure()
    check("删除测量", not state.measures.items)

    print("\n=== 旋转动画 ===")
    window._toggle_spin(True)
    check("启用旋转", state.spin.config.enabled)
    window._set_spin_type("rz")
    window._set_spin_speed("high")
    window._set_spin_direction(-1)
    check("旋转参数", state.spin.config.kind == "rz"
          and state.spin.config.speed == "high"
          and state.spin.config.direction == -1)
    state.spin._last -= 0.5
    before = state.camera.yaw
    state.spin.tick(state.camera)
    check("旋转推进", abs(state.camera.yaw - before) > 1.0)
    window._toggle_spin(False)

    print("\n=== 宏录制 ===")
    window._toggle_macro_record()
    for i in range(4):
        state.camera.yaw = i * 20.0
        state.macro.capture(state.camera, force=True)
    window._toggle_macro_record()
    check("宏帧数", len(state.macro.macro.frames) == 4)
    macro_path = os.path.join(tmp, "macro.json")
    state.macro.save(macro_path)
    check("宏保存", os.path.exists(macro_path))

    print("\n=== 对象操作 ===")
    n0 = len(state.scene.objects)
    window._new_primitive("box")
    check("新建长方体", len(state.scene.objects) == n0 + 1)
    window.copy_object()
    window.paste_object()
    check("复制/粘贴对象", len(state.scene.objects) == n0 + 2
          and state.clipboard_object is not None)
    window._object_action("transparent", -1)
    check("对象透明开关", state.scene.active_object().transparent)
    window._object_action("wire", -1)
    check("对象线框开关", state.scene.active_object().wire)
    window._object_action("components", -1)
    check("连通分量拆分", len(state.scene.objects) > n0 + 2)
    window.delete_object()
    check("删除对象", len(state.scene.objects) == n0 + 2)

    print("\n=== 图片导出 ===")
    img = window._render_picture(supersample=2)
    check("离屏渲染图片", img.shape[2] in (3, 4) and img.shape[0] > 100,
          str(img.shape))
    pic = os.path.join(tmp, "view.png")
    png.write_png(pic, img)
    reloaded = png.read_png(pic)
    check("PNG 往返一致", np.array_equal(reloaded[:, :, :3], img[:, :, :3]))
    transp = window._render_picture(transparent=True)
    check("透明背景图片", transp.shape[2] == 4 and transp[:, :, 3].max() == 255)
    double = window._render_picture(double=True)
    check("双倍分辨率", double.shape[0] == window.canvas.height() * 2
          or double.shape[0] > window.canvas.height())

    print("\n=== STL 导出与再打开 ===")
    stl_path = os.path.join(tmp, "export.stl")
    if state.scene.active_object() is None and state.scene.objects:
        state.scene.set_active(0)
    obj = state.scene.active_object()
    stl_io.write_binary_stl(stl_path, obj.mesh.vertices, obj.mesh.triangles)
    check("STL 导出", os.path.getsize(stl_path) > 84)
    expected_tris = obj.mesh.triangle_count
    window.delete_all_objects()
    check("清空场景", not state.scene.objects)
    window.open_path(stl_path)
    check("重新打开 STL", len(state.scene.objects) == 1
          and state.scene.objects[0].mesh.triangle_count == expected_tris)

    print("\n=== 设置存取 ===")
    window.save_default_settings()
    check("设置已写出", os.path.exists(settings_path)
          and os.path.getsize(settings_path) > 100)
    window.load_default_settings()
    check("设置已载入", True)
    window.canvas.set_interactive_quality(0.8)
    check("拖拽画质设置生效", abs(window.canvas._quality_ceiling - 0.8) < 1e-9
          and window.canvas._adaptive_scale <= 0.8,
          f"ceiling={window.canvas._quality_ceiling:.2f}")
    window.canvas.set_interactive_quality(0.5)

    print("\n=== 视图选项对话框（构造） ===")
    from stl_viewer.viewer import (ColorDialog, LightDialog,
                                       ObjectPropertiesDialog, PointSizeDialog,
                                       RotationDialog, StatisticsDialog)
    dialogs = [
        LightDialog(state, window),
        RotationDialog(state.camera, state.tr, window),
        PointSizeDialog(state, window),
        ObjectPropertiesDialog(state.scene.active_object(), state.tr, window),
        ColorDialog("bg", state.scene.background, window, with_gradient=True),
        StatisticsDialog("stats", "Statistics", window),
    ]
    check("对话框构造", all(d is not None for d in dialogs)
          and len(dialogs) == 6)
    for d in dialogs:
        d.close()

    window.close()
    app.processEvents()
    print(f"\n总计: {len(PASSED)} 通过, {len(FAILED)} 失败")
    if FAILED:
        print("失败项:", *FAILED, sep="\n - ")
        return 1
    print("图形界面冒烟测试通过。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
