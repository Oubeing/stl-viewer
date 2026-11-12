"""菜单 / 快捷键重活的后台化回归（无需 OpenGL）。

用户反馈「点 Save Picture 反应很慢」「任何操作不要卡顿」。这里量化验证：

    * 保存图片：点击后 **立刻返回**（先弹文件框），渲染与 PNG 编码在后台；
    * 复制到剪贴板：后台渲染，完成后主线程写剪贴板；
    * 导出 STL：点击立刻返回，写盘在后台；
    * 打开模型：超过阈值走后台，返回时场景尚未改变，完成后自动加入。

运行：QT_QPA_PLATFORM=offscreen python tests/test_async_ops.py
"""

from __future__ import annotations

import os
import sys
import tempfile
import time

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from PyQt5 import QtWidgets

from stl_viewer.viewer import MainWindow

PASSED, FAILED = [], []
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SAMPLE = os.path.join(ROOT, "samples", "drill_blank.stl")


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


def main() -> int:
    app = QtWidgets.QApplication(sys.argv[:1])
    win = MainWindow()
    win.resize(900, 700)
    win.show()
    app.processEvents()
    state = win.state
    state.settings.default_directory = tempfile.mkdtemp()
    win.open_path(SAMPLE)
    app.processEvents()
    tmp = tempfile.mkdtemp()

    # 文件对话框没法在无人值守环境里点；直接桩掉，返回目标路径。
    original_save = QtWidgets.QFileDialog.getSaveFileName
    original_open = QtWidgets.QFileDialog.getOpenFileName

    print("=== 保存图片：点击立即返回，后台渲染 + 编码 ===")
    pic = os.path.join(tmp, "view.png")
    QtWidgets.QFileDialog.getSaveFileName = staticmethod(
        lambda *a, **k: (pic, "PNG (*.png)"))
    t0 = time.perf_counter()
    win.save_picture()
    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    check("save_picture 点击后 50ms 内返回", elapsed_ms < 50.0,
          f"{elapsed_ms:.1f} ms")
    check("保存任务进入后台", "picture" in win._work_jobs, str(list(win._work_jobs)))
    ok = pump_until(app, lambda: not win._work_jobs)
    check("后台任务在 15s 内完成", ok)
    check("PNG 已写出且非空",
          os.path.exists(pic) and os.path.getsize(pic) > 1000,
          f"{os.path.getsize(pic) if os.path.exists(pic) else 0} bytes")

    print("\n=== 多分辨率 / 透明背景图片 ===")
    multi = os.path.join(tmp, "view_2x.png")
    QtWidgets.QFileDialog.getSaveFileName = staticmethod(
        lambda *a, **k: (multi, "PNG (*.png)"))
    win.save_multi_res_picture()
    check("多分辨率保存进入后台", "picture" in win._work_jobs)
    pump_until(app, lambda: not win._work_jobs)
    check("多分辨率 PNG 已写出", os.path.exists(multi)
          and os.path.getsize(multi) > 1000)

    transp = os.path.join(tmp, "view_t.png")
    QtWidgets.QFileDialog.getSaveFileName = staticmethod(
        lambda *a, **k: (transp, "PNG (*.png)"))
    win.save_picture(transparent=True)
    pump_until(app, lambda: not win._work_jobs)
    check("透明背景 PNG 已写出", os.path.exists(transp)
          and os.path.getsize(transp) > 1000)

    print("\n=== 复制到剪贴板 ===")
    t0 = time.perf_counter()
    win.copy_picture_to_clipboard()
    copy_ms = (time.perf_counter() - t0) * 1000.0
    check("复制图片点击后 50ms 内返回", copy_ms < 50.0, f"{copy_ms:.1f} ms")
    ok = pump_until(app, lambda: not win._work_jobs)
    image = QtWidgets.QApplication.clipboard().image()
    check("剪贴板已有图片", ok and not image.isNull(),
          f"{image.width()}x{image.height()}")

    print("\n=== 导出 STL ===")
    stl_out = os.path.join(tmp, "export.stl")
    QtWidgets.QFileDialog.getSaveFileName = staticmethod(
        lambda *a, **k: (stl_out, "STL (*.stl)"))
    t0 = time.perf_counter()
    win.save_active_stl()
    save_ms = (time.perf_counter() - t0) * 1000.0
    check("导出 STL 点击后 50ms 内返回", save_ms < 50.0, f"{save_ms:.1f} ms")
    pump_until(app, lambda: not win._work_jobs)
    check("STL 已写出", os.path.exists(stl_out) and os.path.getsize(stl_out) > 0)

    print("\n=== 打开大模型：后台载入 ===")
    win.SYNC_LOAD_LIMIT = 0          # 强制走后台路径（样例很小，只为验证流程）
    before = len(state.scene.objects)
    t0 = time.perf_counter()
    win.open_path(SAMPLE)
    open_ms = (time.perf_counter() - t0) * 1000.0
    check("open_path 点击后 50ms 内返回", open_ms < 50.0, f"{open_ms:.1f} ms")
    check("载入任务进入后台", "open" in win._work_jobs, str(list(win._work_jobs)))
    ok = pump_until(app, lambda: not win._work_jobs)
    check("模型已加入场景", ok and len(state.scene.objects) == before + 1,
          f"{len(state.scene.objects)} objects")

    QtWidgets.QFileDialog.getSaveFileName = original_save
    QtWidgets.QFileDialog.getOpenFileName = original_open
    win.close()
    print(f"\n总计: {len(PASSED)} 通过, {len(FAILED)} 失败")
    if FAILED:
        for name in FAILED:
            print(" -", name)
        return 1
    print("后台任务回归通过。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
