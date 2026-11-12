"""核心功能自测（无 GUI 依赖，可直接运行）。

覆盖：
    * STL（二进制/ASCII）读写往返、错误识别
    * PLY 读取、嵌入式模型提取
    * 网格统计 / 焊接 / 连通分量 / 法向修复 / 相交剔除
    * 相机投影与反投影一致性、拾取
    * 渲染管线（实体 / 线框 / 透明背景 / 超采样）与 PNG 编解码
    * 测量工具几何结果
    * 设置读写与多语言

运行：
    python tests/test_core.py            # 全部
    python tests/test_core.py -v         # 打印渲染耗时等细节
"""

from __future__ import annotations

import math
import os
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from stl_viewer import mathutil as mu          # noqa: E402
from stl_viewer import png, stl_io             # noqa: E402
from stl_viewer import cli as cli_mod          # noqa: E402
from stl_viewer.camera import Camera           # noqa: E402
from stl_viewer.i18n import Translator         # noqa: E402
from stl_viewer.lighting import LightSettings  # noqa: E402
from stl_viewer.macro import MacroRecorder, SpinConfig, SpinEngine  # noqa: E402
from stl_viewer.measure import MeasureManager, snap_pick  # noqa: E402
from stl_viewer.mesh import Mesh               # noqa: E402
from stl_viewer.renderer import RenderOptions, SoftwareRenderer  # noqa: E402
from stl_viewer.scene import Scene             # noqa: E402
from stl_viewer.settings import AppSettings, UnitSystem  # noqa: E402

PASSED = []
FAILED = []


def check(name: str, condition: bool, detail: str = "") -> None:
    if condition:
        PASSED.append(name)
        print(f"  [PASS] {name} {detail}")
    else:
        FAILED.append(name)
        print(f"  [FAIL] {name} {detail}")


def section(title: str) -> None:
    print(f"\n=== {title} ===")


# --------------------------------------------------------------------------
def test_math() -> None:
    section("数学与几何工具")
    c, r, n = mu.circle_from_three_points((1, 0, 0), (0, 1, 0), (-1, 0, 0))
    check("三点定圆", abs(r - 1.0) < 1e-9 and np.allclose(c, [0, 0, 0]),
          f"r={r:.6f}")
    # 第一点为圆心、第二点在圆周上（半径 = 两点距离）
    c2, r2, n2 = mu.circle_from_two_points((0, 0, 0), (10, 0, 0), (0, 0, 1))
    check("两点定圆（圆心+半径）", abs(r2 - 10.0) < 1e-9 and np.allclose(c2, [0, 0, 0]),
          f"r={r2:.6f}")
    check("夹角计算", abs(mu.angle_between((1, 0, 0), (0, 1, 0)) - 90.0) < 1e-9)
    rr, phi, zz = mu.cartesian_to_cylindrical((3, 4, 7))
    check("柱坐标换算", abs(rr - 5.0) < 1e-12 and abs(zz - 7.0) < 1e-12,
          f"R={rr}, phi={phi:.2f}, Z={zz}")
    back = mu.cylindrical_to_cartesian(rr, phi, zz)
    check("柱坐标反算", np.allclose(back, [3, 4, 7], atol=1e-12))


def test_stl_io(tmp: str) -> Mesh:
    section("STL / PLY / 嵌入式模型 IO")
    box = Mesh(*_box_arrays(), name="unitbox")
    mesh = box
    bin_path = os.path.join(tmp, "box_bin.stl")
    ascii_path = os.path.join(tmp, "box_ascii.stl")
    stl_io.write_binary_stl(bin_path, mesh.vertices, mesh.triangles)
    stl_io.write_ascii_stl(ascii_path, mesh.vertices, mesh.triangles, "unitbox")

    raw_bin = stl_io.read_stl(bin_path)
    check("二进制 STL 往返", raw_bin.vertex_count == 36 and raw_bin.triangle_count == 12,
          f"V={raw_bin.vertex_count} F={raw_bin.triangle_count}")
    check("二进制 STL 头部", "Exported by STL Viewer" in raw_bin.header, repr(raw_bin.header))
    raw_ascii = stl_io.read_stl(ascii_path)
    check("ASCII STL 往返", raw_ascii.triangle_count == 12 and raw_ascii.name == "unitbox")

    # 焊接后应回到 8 顶点、12 三角形
    welded = Mesh.from_raw(raw_bin)
    welded.weld(1e-6)
    st = welded.statistics()
    check("顶点焊接", welded.vertex_count == 8 and welded.triangle_count == 12,
          f"V={welded.vertex_count}")
    check("封闭实体判定", st.closed and st.boundary_edges == 0)
    check("体积计算", abs(st.volume - 8.0) < 1e-9, f"V={st.volume:.6f}")
    check("表面积计算", abs(st.area - 24.0) < 1e-9, f"A={st.area:.6f}")

    # ASCII STL 转换为紧凑 std::string
    txt = open(ascii_path, "r", encoding="ascii").read()
    check("ASCII STL 结构", txt.startswith("solid unitbox") and "endsolid" in txt)

    report = stl_io.file_size_report(bin_path)
    check("文件大小报告", report["triangles"] == 12 and report["bytes"] > 0,
          report["human"])

    # 文件头为 solid 的二进制文件不应被误判为 ASCII
    check("编码判定", stl_io._looks_like_binary_stl(
        b"solid fake" + b"\x00" * 71 + (1).to_bytes(4, "little") + b"\x00" * 50))

    # PLY ASCII
    ply_path = os.path.join(tmp, "box.ply")
    with open(ply_path, "w", encoding="ascii") as fh:
        fh.write("ply\nformat ascii 1.0\nelement vertex 3\n"
                 "property float x\nproperty float y\nproperty float z\n"
                 "element face 1\nproperty list uchar int vertex_indices\n"
                 "end_header\n0 0 0\n1 0 0\n0 1 0\n3 0 1 2\n")
    ply = stl_io.read_ply(ply_path)
    check("PLY 读取", ply.vertex_count == 3 and ply.triangle_count == 1)

    # 嵌入式（自包含导出件）：把 STL 追加到伪 exe 之后
    embedded_path = os.path.join(tmp, "viewer_embed.exe")
    with open(embedded_path, "wb") as fh:
        fh.write(b"MZ" + os.urandom(4096))
        fh.write(open(bin_path, "rb").read())
        fh.write(os.urandom(64))
    emb = stl_io.extract_embedded_mesh(embedded_path)
    check("自包含 EXE 提取", emb.triangle_count == 12, f"F={emb.triangle_count}")
    return Mesh.from_raw(raw_bin)


def test_mesh_ops() -> None:
    section("网格操作")
    scene = Scene()
    scene.add_cylinder(radius=10, height=40, segments=48)
    scene.add_cylinder(radius=10, height=40, segments=48)  # 与第一个重叠但独立
    obj = scene.objects[0]
    mesh = obj.mesh
    mesh.weld(1e-6)
    check("圆柱焊接", mesh.vertex_count == 98, f"V={mesh.vertex_count}")
    st = mesh.statistics()
    check("圆柱封闭", st.closed and not st.inverted, f"边界边={st.boundary_edges}")
    expect_vol = math.pi * 100 * 40
    check("圆柱体积近似", abs(st.volume - expect_vol) / expect_vol < 0.01,
          f"{st.volume:.1f} vs {expect_vol:.1f}")

    # 反向后应判定法向朝内并修复
    mesh.flip_normals()
    check("翻转法向检测", mesh.statistics().inverted)
    flipped = mesh.orient_outward()
    check("法向修复", flipped == mesh.triangle_count and not mesh.statistics().inverted)

    # 连通分量
    scene2 = Scene()
    a = scene2.add_cylinder(radius=5, height=20, segments=24).mesh
    b = scene2.add_cylinder(radius=5, height=20, segments=24).mesh
    b.translate((100, 0, 0))
    joined = Mesh(np.concatenate([a.vertices, b.vertices]),
                  np.concatenate([a.triangles, b.triangles + a.vertex_count]))
    check("连通分量个数", joined.component_count() == 2,
          f"components={joined.component_count()}")
    parts = joined.split_components()
    check("拆分为独立网格", len(parts) == 2 and sum(p.triangle_count for p in parts)
          == joined.triangle_count)

    # 相交剔除：刀具块穿过目标块的一个面
    target = Mesh(*_box_arrays())
    target.scale(10.0)
    cutter = Mesh(*_box_arrays())
    cutter.scale(2.0)
    cutter.translate((10.0, 0.0, 0.0))       # 与 +X 面相交
    before = target.triangle_count
    removed = target.remove_triangles_intersecting(cutter)
    check("相交三角形剔除", 0 < removed < before, f"移除 {removed}/{before} 面")

    # 自动着色
    colors = target.color_by_orientation()
    check("自动配色", colors.shape[0] == target.triangle_count
          and colors.min() >= 0 and colors.max() <= 1)
    check("分量配色", joined.color_by_component().shape[0] == joined.triangle_count)
    check("曲率配色", joined.color_by_curvature().shape[0] == joined.triangle_count)

    # 统计文本
    check("统计文本输出", "三角面数" in joined.statistics().as_text())


def test_camera() -> None:
    section("相机与投影")
    cam = Camera(800, 600)
    cam.fit((-10, -10, -10), (10, 10, 10))
    check("适配距离", cam.distance > 20.0, f"d={cam.distance:.2f}")
    pts = np.array([[0.0, 0, 0], [10, 0, 0], [0, 10, 0], [0, 0, 10]])
    scr, z, valid = cam.project(pts)
    check("投影有效", valid.all() and np.isfinite(scr).all())
    center_scr, _, _ = cam.project(np.zeros((1, 3)))
    check("目标点投影到屏幕中心",
          abs(center_scr[0, 0] - 400) < 1e-6 and abs(center_scr[0, 1] - 300) < 1e-6,
          f"{center_scr[0]}")

    # 反投影：屏幕中心射线应指向目标点
    origin, direction = cam.ray(400, 300)
    check("射线方向", np.allclose(mu.normalize(direction), mu.normalize(
        cam.target - cam.eye()), atol=1e-9))
    hit = cam.ray_to_plane(400, 300)
    check("屏幕中心射线交于目标点", np.allclose(hit, cam.target, atol=1e-6),
          f"{hit}")

    # 正交 / 透视切换保持尺寸连续
    h_before = cam.visible_height()
    cam.set_orthographic(True)
    h_after = cam.visible_height()
    check("正交切换连续", abs(h_before - h_after) / h_before < 0.01,
          f"{h_before:.3f} -> {h_after:.3f}")
    cam.set_orthographic(False)

    # 交互
    yaw0 = cam.yaw
    cam.orbit(50, 0)
    check("轨道旋转", abs(cam.yaw - yaw0) > 1.0, f"yaw {yaw0:.2f} -> {cam.yaw:.2f}")
    yaw1 = cam.yaw
    cam.set_rotation_angles(yaw=yaw1, pitch=0, roll=0)
    cam.rotate_around_z(30)
    check("绕 Z 旋转", abs(cam.yaw - yaw1) > 1.0 and cam.pitch == 0)
    tgt = cam.target.copy()
    cam.pan(40, 20)
    check("平移", not np.allclose(tgt, cam.target))
    cam.reset_center()
    check("复位中心", np.allclose(cam.target, 0))
    cam.set_standard_view("top")
    check("标准视图", cam.pitch > 89.0)
    d0 = cam.distance
    cam.dolly(3)
    check("缩放", cam.distance < d0)

    # 状态存取
    st = cam.state()
    cam2 = Camera(800, 600)
    cam2.set_state(st)
    check("相机状态往返", np.allclose(cam2.eye(), cam.eye(), atol=1e-9))


def test_render(verbose: bool = False) -> None:
    section("渲染管线")
    scene = Scene()
    scene.add_cylinder(radius=8, height=60, segments=64, name="blank")
    scene.add_box(size=(24, 24, 6), center=(0, 0, -33), name="plate")
    cam = Camera(640, 480)
    cam.fit(*scene.bounds())
    light = LightSettings()
    renderer = SoftwareRenderer()

    opts = RenderOptions(show_axis=True, show_grid=True)
    img = renderer.render(scene, cam, options=opts, light=light)
    check("实体渲染尺寸", img.shape == (480, 640, 3), str(img.shape))
    bg = np.array(opts.background) * 255.0
    ratio = (np.abs(img.astype(float) - bg).sum(axis=2) > 12).mean()
    check("实体渲染覆盖率", 0.02 < ratio < 0.95, f"{ratio:.3f}")

    # Z 缓冲：近处的圆柱必须遮挡下方的远处底板
    hit = renderer.pick(scene, cam, 320, 240)
    check("屏幕拾取命中", hit is not None and hit["object"].name == "blank",
          hit["object"].name if hit else "None")

    # 线框模式
    wire = renderer.render(scene, cam,
                           options=RenderOptions(wire=True, solid=False))
    ratio_wire = (np.abs(wire.astype(float) - bg).sum(axis=2) > 12).mean()
    check("线框渲染", 0.001 < ratio_wire < ratio, f"{ratio_wire:.3f} < {ratio:.3f}")

    # 着色模式语义：auto/uniform 是"整件同色"（明暗交给光照），只有
    # 法向/分量/曲率模式才用颜色本身表达信息。早期实现把 auto 接到了法向
    # 彩虹图上，默认实体视图因此看起来像一张花花的贴图（即"显示失真"）。
    obj0 = scene.objects[0]
    auto_cols = obj0.face_colors("auto")
    uni_cols = obj0.face_colors("uniform")
    ori_cols = obj0.face_colors("orientation")
    check("auto 模式为整件同色",
          len(auto_cols) == obj0.triangle_count and np.allclose(auto_cols, auto_cols[0]))
    check("auto 与 uniform 使用同一零件本色",
          np.allclose(auto_cols, uni_cols) and np.allclose(auto_cols[0], obj0.color, atol=1e-9))
    check("法向模式逐面颜色各异", not np.allclose(ori_cols, ori_cols[0]))
    check("整件同色才启用逐顶点平滑着色",
          obj0.flat_color("auto") and obj0.flat_color("uniform")
          and not obj0.flat_color("orientation"))

    colored = renderer.render(scene, cam, options=RenderOptions(color_mode="auto"))
    mono = renderer.render(scene, cam, options=RenderOptions(color_mode="uniform"))
    orient = renderer.render(scene, cam, options=RenderOptions(color_mode="orientation"))
    check("默认着色的整件同色渲染相同",
          np.array_equal(colored, mono), "auto 与 uniform 图像应一致")
    # 在「模型覆盖区域」比较，而不是全图平均：修复遮挡错误后模型轮廓更
    # 紧凑，全图平均会被大面积背景稀释，无法反映两种着色模式的真实差异。
    model_mask = (np.abs(colored.astype(float) - bg).sum(axis=2) > 24)
    diff = np.abs(orient.astype(int) - colored.astype(int))[model_mask].mean()
    check("法向模式与默认着色明显不同", diff > 15.0,
          f"模型区域平均像素差 {diff:.2f}")
    check("法向模式颜色丰富",
          len(np.unique(orient.reshape(-1, 3), axis=0))
          > len(np.unique(colored.reshape(-1, 3), axis=0)))

    # 遮挡回归：分批策略改成"按尺寸分桶"后，不透明路径不再按深度排序，
    # 因此必须确认 Z 缓冲仍能完全挡住背面 —— 封闭实体轮廓内不应透出背景。
    def inner_background_pixels(img):
        mask = np.abs(img.astype(float) - bg).sum(axis=2) > 24
        holes = 0
        for row in range(mask.shape[0]):
            xs = np.nonzero(mask[row])[0]
            if xs.size >= 4:
                holes += int((~mask[row, xs.min():xs.max() + 1]).sum())
        return holes

    for label, maker in (("圆柱", lambda s: s.add_cylinder(radius=20, height=70, segments=48)),
                         ("球", lambda s: s.add_sphere(radius=22, segments=40))):
        solid = Scene()
        maker(solid)
        worst = 0
        for yaw, pitch in ((0, 0), (35, 20), (250, 60)):
            c = Camera(360, 360)
            c.fit(*solid.bounds())
            c.yaw, c.pitch = yaw, pitch
            shot = renderer.render(solid, c, size=(360, 360), light=light,
                                   options=RenderOptions(supersample=1, show_axis=False,
                                                         show_grid=False))
            worst = max(worst, inner_background_pixels(shot))
        check(f"封闭{label}轮廓内无透光", worst == 0, f"背景像素 {worst}")

    # 透明背景
    alpha = renderer.render(scene, cam, options=RenderOptions(
        transparent_background=True, show_axis=False, show_grid=False))
    check("透明背景 RGBA", alpha.shape == (480, 640, 4) and alpha[:, :, 3].max() == 255)
    check("背景 alpha 为 0", alpha[0, 0, 3] == 0 and alpha[0, 0, :3].max() == 0)

    # 透明对象混合
    scene.objects[1].transparent = True
    scene.objects[1].alpha = 0.4
    blend = renderer.render(scene, cam, options=RenderOptions(show_axis=False))
    check("透明混合渲染", blend.shape == (480, 640, 3))

    # 超采样抗锯齿
    smooth = renderer.render(scene, cam, options=RenderOptions(supersample=2))
    check("超采样输出尺寸不变", smooth.shape == (480, 640, 3))

    # 剖切
    segs = scene.objects[0].mesh.section_polygons((0, 0, 0), (0, 1, 0))
    check("平面剖切", len(segs) >= 2, f"交线数={len(segs)}")

    if verbose:
        print(f"    render stats: {renderer.last_stats}")


def test_measure(tmp: str) -> None:
    section("测量工具")
    units = UnitSystem("mm")
    scene = Scene()
    obj = scene.add_cylinder(radius=10, height=40, segments=48)
    obj.mesh.weld(1e-6)
    cam = Camera(800, 600)
    cam.fit(*scene.bounds())
    renderer = SoftwareRenderer()

    mgr = MeasureManager()
    mgr.start("line3d")
    mgr.add_point((0, 0, 0))
    m = mgr.add_point((3, 4, 0))
    check("3D 直线测量完成", m is not None and abs(m.length - 5.0) < 1e-9,
          m.value_text(units) if m else "")
    check("测量结果导出", mgr.export(units)[0]["length_mm"] == 5.0)

    mgr.start("point3d")
    mgr.add_point((1, 2, 3))
    check("3D 点测量", mgr.items[-1].value_text(units) == "(1.000, 2.000, 3.000) mm",
          mgr.items[-1].value_text(units))

    mgr.start("circle3p")
    mgr.add_point((1, 0, 0))
    mgr.add_point((0, 1, 0))
    mgr.add_point((-1, 0, 0))
    c = mgr.items[-1].circle
    check("三点圆测量", c is not None and abs(c[1] - 1.0) < 1e-9)

    # 两点测圆：第一点圆心，第二点定半径
    mgr.start("circle2p")
    mgr.add_point((0, 0, 0))
    mgr.add_point((10, 0, 0), plane=(0, 0, 1))
    c2 = mgr.items[-1].circle
    check("两点圆测量（圆心+半径）",
          c2 is not None and abs(c2[1] - 10.0) < 1e-9
          and np.allclose(c2[0], [0, 0, 0]),
          f"R={c2[1]:.3f} center={np.round(c2[0], 3)}" if c2 else "")

    # 半径取「视图平面内」的分量：第二点带 5 的法向偏移时，圆仍经过它的投影，
    # 直径/半径标注与画面上看到的圆严格一致（斜视角取点必然有深度差）。
    mgr.start("circle2p")
    mgr.add_point((0, 0, 0))
    mgr.add_point((10, 0, 5), plane=(0, 0, 1))
    c2b = mgr.items[-1].circle
    check("两点圆半径取视图平面内分量",
          c2b is not None and abs(c2b[1] - 10.0) < 1e-9,
          f"R={c2b[1]:.3f}" if c2b else "None")

    mgr.start("angle2d")
    mgr.add_point((0, 0, 0))
    mgr.add_point((0, 10, 0))
    mgr.add_point((0, 0, 10))
    check("2D 角度测量", abs(mgr.items[-1].angle - 90.0) < 1e-6,
          f"{mgr.items[-1].angle:.3f}")

    mgr.start("rzpoint")
    mgr.add_point((3, 4, 7))
    text = mgr.items[-1].value_text(units)
    check("RZ 点测量", "R=5.000" in text and "Z=7.000" in text, text)

    mgr.refresh_labels(units)
    check("标注刷新", all(item.label() for item in mgr.items))

    # 绘制数据
    ok = all(len(item.polylines()) >= 1 or item.kind in ("point3d", "rzpoint")
             for item in mgr.items)
    check("绘制数据生成", ok)
    check("删除测量", mgr.delete_last() and mgr.delete_first())
    n = len(mgr.items)
    check("清空测量", mgr.delete_all() == n and not mgr.items)

    # 吸附拾取：点在圆柱侧面附近应吸附到顶点
    hit = snap_pick(scene, cam, renderer, 400, 300)
    check("吸附拾取", hit is not None and "point" in hit,
          f"snapped={hit['snapped'] if hit else None}")

    # 单位换算
    inch = UnitSystem("inch")
    check("英寸换算", abs(inch.to_display(25.4) - 1.0) < 1e-12
          and inch.format(25.4) == "1.0000 in", inch.format(25.4))

    # 带标注测量渲染
    mgr2 = MeasureManager()
    mgr2.start("line3d")
    mgr2.add_point((-10, 0, -20))
    mgr2.add_point((10, 0, -20))
    mgr2.refresh_labels(units)
    check("测量项进入渲染", len(mgr2.items) == 1)


def test_settings_and_i18n(tmp: str) -> None:
    section("设置持久化与多语言")
    path = os.path.join(tmp, "test.ini")
    s = AppSettings()
    s.language = "CH"
    s.units = "inch"
    s.view.show_grid = True
    s.view.color_mode = "component"
    s.view.point_size = 3
    s.camera = {"yaw": 12.5, "pitch": -20.0, "distance": 88.0}
    s.light = {"ambient": 0.5, "primary": "left", "shine_color": [1.0, 0.9, 0.8]}
    s.recent_files = ["a.stl", "b.stl"]
    s.save(path)
    loaded = AppSettings().load(path)
    check("INI 往返-常规", loaded.language == "CH" and loaded.units == "inch")
    check("INI 往返-视图", loaded.view.show_grid and loaded.view.color_mode == "component"
          and loaded.view.point_size == 3)
    check("INI 往返-相机", abs(loaded.camera["yaw"] - 12.5) < 1e-9
          and abs(loaded.camera["distance"] - 88.0) < 1e-9)
    check("INI 往返-光照", abs(loaded.light["ambient"] - 0.5) < 1e-9
          and loaded.light["primary"] == "left")
    check("INI 往返-最近文件", loaded.recent_files == ["a.stl", "b.stl"])

    cam = Camera(800, 600)
    scene = Scene()
    light = LightSettings()
    loaded.apply_to(cam, scene, light)
    check("设置应用到运行时", abs(cam.yaw - 12.5) < 1e-9 and light.primary == "left")

    loaded.reset_to_factory()
    check("恢复默认设置", loaded.language == "EN" and not loaded.view.show_grid)

    tr = Translator("CH")
    check("中文翻译", tr.tr("file") == "文件" and tr.tr("open") == "打开")
    tr.set_language("EN")
    check("英文翻译", tr.tr("file") == "File")
    tr.set_language("RU")
    check("俄语回退", tr.tr("open") == "Открыть" and tr.tr("zoom_fit") == "Zoom Fit")
    check("语言列表", len(Translator.language_names()) == 15)
    check("未知键回退", tr.tr("__missing__") == "__missing__")


def test_spin_and_macro() -> None:
    section("旋转动画与宏")
    cam = Camera(640, 480)
    cfg = SpinConfig(enabled=True, kind="rz", speed="low")
    engine = SpinEngine(cfg)
    engine._last -= 1.0                       # 模拟过去 1 秒
    yaw0 = cam.yaw
    engine.tick(cam)
    check("旋转动画推进", abs(cam.yaw - yaw0) > 1.0,
          f"Δyaw={cam.yaw - yaw0:.2f} (期望约 {cfg.deg_per_second})")

    cfg.enabled = False
    yaw1 = cam.yaw
    engine.tick(cam)
    check("关闭后不再旋转", abs(cam.yaw - yaw1) < 1e-12)

    rec = MacroRecorder(min_interval=0.0)
    rec.start()
    for i in range(5):
        cam.yaw = i * 30.0
        rec.capture(cam, force=True)
    n = rec.stop()
    check("宏录制", n == 5 and rec.macro.duration >= 0)
    state = rec.macro.state_at(rec.macro.duration * 0.5)
    check("宏插值", state is not None and 0 <= state["yaw"] <= 120,
          f"yaw={state['yaw']:.1f}")

    path = os.path.join(tempfile.gettempdir(), "num_macro.json")
    rec.save(path)
    rec2 = MacroRecorder()
    check("宏存取", rec2.load(path) == 5)
    rec2.start_playback()
    rec2.stop_playback()
    check("宏播放控制", not rec2.playback_active)


def test_cli(tmp: str) -> None:
    section("命令行接口")
    import contextlib
    import io

    # 准备一个焊接后可分割的模型
    scene = Scene()
    scene.add_cylinder(radius=8, height=30, segments=48)
    scene.add_box(size=(6, 6, 6), center=(20, 0, 0))
    v = np.vstack([o.mesh.vertices for o in scene.objects])
    f_list, base = [], 0
    for o in scene.objects:
        f_list.append(o.mesh.triangles + base)
        base += o.mesh.vertex_count
    src = os.path.join(tmp, "cli_model.stl")
    stl_io.write_binary_stl(src, v, np.vstack(f_list))

    def run(argv):
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            rc = cli_mod.main(argv)
        return rc, buf.getvalue()

    rc, out = run(["info", src])
    check("CLI info", rc == 0 and "三角面数" in out)

    # 输出目录不存在时应自动创建（回归测试）
    deep = os.path.join(tmp, "new", "sub", "out.stl")
    rc, out = run(["convert", src, "-o", deep, "--ascii"])
    check("CLI convert 自动建目录", rc == 0 and os.path.exists(deep),
          os.path.basename(deep))

    outdir = os.path.join(tmp, "parts")
    prefix = os.path.join(outdir, "p")
    rc, out = run(["components", src, "--weld", "--out-prefix", prefix])
    produced = sorted(os.listdir(outdir)) if os.path.isdir(outdir) else []
    check("CLI components 自动建目录", rc == 0 and len(produced) >= 2,
          f"{len(produced)} 个分量")

    rc, out = run(["measure", src, "--rz", "8,0,5"])
    check("CLI measure RZ", rc == 0 and "rzpoint" in out)

    rc, out = run(["render", src, "-o", os.path.join(tmp, "r.png"),
                   "--size", "160x120", "--component-colors"])
    check("CLI render", rc == 0 and os.path.exists(os.path.join(tmp, "r.png")))


def test_viewport_sync() -> None:
    section("渲染视口一致性（拖拽失真回归）")
    scene = Scene()
    scene.add_cylinder(radius=20, height=60, segments=32)
    base = Camera(640, 480)
    base.fit(*scene.bounds())
    st = base.state()
    bg = tuple(int(c * 255) for c in RenderOptions().background)

    def render_at(w, h, ss):
        # 故意把相机视口设成“窗口尺寸”，模拟 GUI 拖拽时以更低分辨率渲染
        cam = Camera(640, 480)
        cam.set_state(st)
        cam.set_viewport(640, 480)
        opts = RenderOptions(supersample=ss, show_grid=False, show_axis=False)
        return SoftwareRenderer().render(scene, cam, size=(w, h), options=opts)

    def centroid(img):
        m = (np.abs(img[:, :, :3].astype(int) - np.array(bg)).sum(axis=2) > 30)
        ys, xs = np.where(m)
        return (xs.mean() / img.shape[1], ys.mean() / img.shape[0]) if len(xs) else None

    full = centroid(render_at(640, 480, 2))
    half = centroid(render_at(320, 240, 1))
    check("全分辨率成像", full is not None)
    check("半分辨率成像", half is not None)
    if full and half:
        dx = abs(full[0] - half[0])
        dy = abs(full[1] - half[1])
        check("降分辨率渲染仍居中（无视口偏移）", dx < 0.03 and dy < 0.03,
              f"full=({full[0]:.3f},{full[1]:.3f}) half=({half[0]:.3f},{half[1]:.3f})")

    # 宽高比不应造成拉伸：球体在任意视口比例下都应接近正圆
    ball = Scene()
    ball.add_sphere(radius=10.0, segments=48)
    for (w, h) in ((800, 400), (400, 800), (600, 600)):
        cam = Camera(w, h)
        cam.fit(*ball.bounds())
        opts = RenderOptions(supersample=1, show_grid=False, show_axis=False)
        img = SoftwareRenderer().render(ball, cam, size=(w, h), options=opts)
        m = (np.abs(img[:, :, :3].astype(int) - np.array(bg)).sum(axis=2) > 30)
        ys, xs = np.where(m)
        bw = xs.max() - xs.min() + 1
        bh = ys.max() - ys.min() + 1
        ratio = bw / max(bh, 1)
        check(f"球体不随宽高比拉伸 {w}x{h}", abs(ratio - 1.0) < 0.03,
              f"bbox={bw}x{bh} ratio={ratio:.3f}")


def test_center_pivot_and_depth() -> None:
    """画布中心支点 / 投影一致 / 光栅化深度遮挡回归。"""
    section("画布中心支点与深度遮挡")

    def fitted(zoom=1.0):
        c = Camera(800, 600)
        c.fit((-10, -10, -10), (10, 10, 10))
        c.set_viewport(800, 600)
        c.zoom = zoom
        return c

    cam = fitted()
    scr, _, _ = cam.project(cam.target[None, :])
    check("支点投影在画布中心", np.allclose(scr[0], (400, 300), atol=1e-6),
          str(scr[0]))
    pivot = cam.center_world_point()
    cam.orbit_about_center(37, 19, 0.2)
    check("旋转不改变支点", np.allclose(cam.pivot, pivot))
    scr, _, _ = cam.project(cam.target[None, :])
    check("旋转后支点仍在画布中心", np.allclose(scr[0], (400, 300), atol=1e-6))
    cam.dolly_about_center(1.0, 0.1)
    check("缩放不改变支点", np.allclose(cam.pivot, pivot))

    p = cam.target.copy()
    before, _, _ = cam.project(p[None, :])
    cam.pan_about_center(60, 45)
    after, _, _ = cam.project(p[None, :])
    check("平移严格跟随光标", np.allclose(after - before, (60, 45), atol=1e-6),
          str((after - before)[0]))

    # zoom 必须同时进入投影矩阵与屏幕射线，否则缩放无效 / 平移与拾取偏移。
    for zoom in (1.0, 1.5, 0.7):
        c = fitted(zoom)
        q = c.target + c.right() * 7.0 + c.up() * 4.0
        s, _, _ = c.project(q[None, :])
        hit = c.ray_to_plane(s[0, 0], s[0, 1])
        check(f"zoom={zoom} 投影/拾取往返一致",
              np.linalg.norm(hit - q) < 1e-6,
              f"误差 {np.linalg.norm(hit - q):.2g}")
    c = fitted()
    q = np.array([[5.0, 0.0, 0.0]])
    s0, _, _ = c.project(q)
    c.zoom_by_factor(1.5)
    s1, _, _ = c.project(q)
    check("透视缩放真实生效",
          abs(s1[0, 0] - 400) > abs(s0[0, 0] - 400) + 1.0)

    # 反向绕序三角形（area<0）的深度不能被取反；同一批内重叠像素也必须
    # 按「远 -> 近」写入。两者都会让远处面盖住近处面，看起来像模型穿透。
    near = np.array([[10., 10.], [90., 10.], [10., 90.]], dtype=np.float32)
    far = near[[0, 2, 1]].copy()

    def shot(order):
        r = SoftwareRenderer()
        r.width = r.height = 100
        r._stats = {"batches": 0}
        r._ensure_buffers(100, 100)
        tris = np.stack([near if k == "n" else far for k in order])
        z = np.array([[0.2] * 3 if k == "n" else [0.8] * 3 for k in order],
                     dtype=np.float32)
        col = np.array([[1., 0., 0.] if k == "n" else [0., 1., 0.]
                        for k in order], dtype=np.float32)
        r._rasterize(tris, z, col)
        return r._color[30, 30], r._depth[30, 30]

    for order in (("n", "f"), ("f", "n")):
        color, depth = shot(order)
        check(f"批内遮挡正确 {''.join(order)}",
              np.allclose(color, (1, 0, 0)) and abs(depth - 0.2) < 1e-5,
              f"color={color}, depth={depth:.3f}")


def test_zoom_fit_views() -> None:
    """Zoom Fit 必须按当前朝向 + 真实画布宽高比铺满视口。

    回归用户用例 HLFΦ0.20-6.5：细长针状件（0.2 × 0.2 × 11），
    旧实现用外接球半径估算距离，切到俯视只剩十几像素。
    """
    from stl_viewer.camera import STANDARD_VIEWS  # noqa: E402
    section("Zoom Fit 视角/视口自适应")

    rng = np.random.default_rng(7)
    z = rng.uniform(-5.5, 5.5, 4000)
    ang = rng.uniform(0, 2 * np.pi, 4000)
    needle = np.stack([0.103 * np.cos(ang), 0.103 * np.sin(ang), z], axis=1)
    box = rng.uniform(-1.0, 1.0, (3000, 3))

    for label, pts in (("细长针", needle), ("立方体", box)):
        lo, hi = pts.min(axis=0), pts.max(axis=0)
        for w, h in ((1500, 820), (640, 900)):
            for view in STANDARD_VIEWS:
                c = Camera(w, h)
                c.set_standard_view(view)
                c.fit(lo, hi, points=pts)
                s, _, valid = c.project(pts)
                s = s[valid & np.isfinite(s).all(axis=1)]
                if len(s) == 0:
                    check(f"{label} {view} {w}x{h} 有可见点", False)
                    continue
                inside = ((s[:, 0] >= -1).all() and (s[:, 0] <= w + 1).all()
                          and (s[:, 1] >= -1).all() and (s[:, 1] <= h + 1).all())
                fill = max((s[:, 0].max() - s[:, 0].min()) / w,
                           (s[:, 1].max() - s[:, 1].min()) / h)
                check(f"{label} {view} {w}x{h} 完整入画", inside)
                check(f"{label} {view} {w}x{h} 铺满视口",
                      0.78 <= fill <= 0.96, f"占屏 {fill:.1%}")

    # 真实顶点比包围盒 8 角点更贴合（角点常落在空处）
    c1, c2 = Camera(1000, 800), Camera(1000, 800)
    c1.fit(box.min(axis=0), box.max(axis=0))
    c2.fit(box.min(axis=0), box.max(axis=0), points=box)
    check("顶点拟合距离不大于角点拟合", c2.distance <= c1.distance + 1e-9,
          f"{c2.distance:.3f} vs {c1.distance:.3f}")

    # 切换视角后重新 fit，距离必须随视角变化（俯视针状件明显拉近）
    c = Camera(1500, 820)
    c.set_standard_view("front")
    c.fit(needle.min(axis=0), needle.max(axis=0), points=needle)
    d_front = c.distance
    c.set_standard_view("top")
    c.fit(needle.min(axis=0), needle.max(axis=0), points=needle)
    check("切换视角重新适配距离", c.distance < d_front * 0.5,
          f"front={d_front:.2f} top={c.distance:.2f}")


def _box_arrays():
    """单位立方体（边长 1，用于几何断言；实际尺寸由调用方缩放）。"""
    v = np.array([
        [0, 0, 0], [2, 0, 0], [2, 2, 0], [0, 2, 0],
        [0, 0, 2], [2, 0, 2], [2, 2, 2], [0, 2, 2],
    ], dtype=np.float64) - 1.0
    f = np.array([
        [0, 3, 2], [0, 2, 1], [4, 5, 6], [4, 6, 7],
        [0, 1, 5], [0, 5, 4], [1, 2, 6], [1, 6, 5],
        [2, 3, 7], [2, 7, 6], [3, 0, 4], [3, 4, 7],
    ], dtype=np.int64)
    return v, f


def main() -> int:
    verbose = "-v" in sys.argv
    tmp = tempfile.mkdtemp(prefix="numstl_")
    print(f"临时目录: {tmp}")
    test_math()
    mesh = test_stl_io(tmp)
    test_mesh_ops()
    test_camera()
    test_render(verbose)
    test_viewport_sync()
    test_center_pivot_and_depth()
    test_zoom_fit_views()
    test_measure(tmp)
    test_settings_and_i18n(tmp)
    test_spin_and_macro()
    test_cli(tmp)
    print(f"\n总计: {len(PASSED)} 通过, {len(FAILED)} 失败")
    if FAILED:
        print("失败项:")
        for name in FAILED:
            print(" -", name)
        return 1
    print("全部核心测试通过。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
