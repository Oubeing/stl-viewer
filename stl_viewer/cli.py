"""命令行工具（无 GUI）。

无界面场景下的批处理能力：模型信息、图片导出、
格式转换、连通分量拆分、测量计算。

示例：
    python -m stl_viewer info part.stl
    python -m stl_viewer render part.stl -o part.png --view iso --colored --double
    python -m stl_viewer convert part.stl part_fixed.stl --repair --ascii
    python -m stl_viewer components part.stl --out-prefix samples/part
    python -m stl_viewer measure part.stl --line "0,0,0" "0,0,25" --units inch
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

from . import stl_io
from .camera import Camera
from .lighting import LightSettings
from .measure import MeasureManager
from .mesh import Mesh
from .renderer import RenderOptions, SoftwareRenderer
from .scene import Scene
from .settings import AppSettings, UnitSystem


# --------------------------------------------------------------------------
def _parse_triplet(text: str) -> np.ndarray:
    parts = [p for p in text.replace(";", ",").split(",") if p.strip()]
    if len(parts) != 3:
        raise argparse.ArgumentTypeError(f"需要三个数值: {text!r}")
    return np.array([float(p) for p in parts], dtype=np.float64)


def _parse_points(values, expected: int | None = None) -> list:
    """解析一个或多个三维点。

    既支持 ``--line "0,0,-20" "0,0,20"``，也支持
    ``--line "0,0,-20;0,0,20"``（推荐，含负号时不会被误判为选项）。
    """
    if values is None:
        return []
    if isinstance(values, str):
        values = [values]
    tokens = []
    for value in values:
        tokens.extend(t for t in value.split(";") if t.strip())
    points = [_parse_triplet(t) for t in tokens]
    if expected is not None and len(points) != expected:
        raise argparse.ArgumentTypeError(
            f"需要 {expected} 个点，收到 {len(points)} 个")
    return points


def _parse_size(text: str) -> tuple:
    for sep in ("x", "X", "*", ","):
        if sep in text:
            w, h = text.split(sep, 1)
            return int(w), int(h)
    raise argparse.ArgumentTypeError("尺寸格式应为 宽x高，例如 1280x960")


def _parse_color(text: str) -> tuple:
    parts = [float(p) for p in text.split(",")]
    if len(parts) != 3:
        raise argparse.ArgumentTypeError("颜色格式应为 R,G,B")
    if max(parts) > 1.0:
        parts = [p / 255.0 for p in parts]
    return tuple(parts)


def _load_scene(paths, args) -> tuple:
    scene = Scene()
    for path in paths:
        # 默认自动焊接重合顶点（STL 多为未焊接的三角汤），--no-weld 可关闭
        mesh = Mesh.load(path, weld=not getattr(args, "no_weld", False))
        if getattr(args, "repair", False):
            mesh.repair()
        scene.add_mesh(mesh, name=os.path.splitext(os.path.basename(path))[0],
                       source_path=path)
    return scene


def _make_camera(scene: Scene, args) -> Camera:
    cam = Camera(*getattr(args, "size", (1024, 768)))
    lo, hi = scene.bounds()
    cam.fit(lo, hi)
    view = getattr(args, "view", None)
    if view:
        cam.set_standard_view(view)
    if getattr(args, "yaw", None) is not None:
        cam.yaw = args.yaw
    if getattr(args, "pitch", None) is not None:
        cam.pitch = args.pitch
    if getattr(args, "ortho", False):
        cam.set_orthographic(True)
    return cam


def _make_options(args, view_options) -> RenderOptions:
    opts = RenderOptions.from_view_options(view_options)
    if getattr(args, "wire", False):
        opts.wire, opts.solid = True, False
    if getattr(args, "colored", False):
        opts.color_mode = "auto"
    if getattr(args, "orientation", False):
        opts.color_mode = "orientation"
    if getattr(args, "component_colors", False):
        opts.color_mode = "component"
    if getattr(args, "curvature_colors", False):
        opts.color_mode = "curvature"
    if getattr(args, "uniform", False):
        opts.color_mode = "uniform"
    if getattr(args, "smooth", False):
        opts.gouraud = True
    if getattr(args, "transparent", False):
        opts.transparent_background = True
    if getattr(args, "edges", False):
        opts.show_edges = True
    if getattr(args, "grid", False):
        opts.show_grid = True
    if getattr(args, "no_axis", False):
        opts.show_axis = False
    if getattr(args, "crosshair", False):
        opts.crosshair = True
    if getattr(args, "supersample", None):
        opts.supersample = max(int(args.supersample), 1)
    if getattr(args, "antialias", False):
        opts.supersample = max(opts.supersample, 2)
    if getattr(args, "background", None):
        opts.background = args.background
        opts.gradient_background = False
    if getattr(args, "gradient", False):
        opts.gradient_background = True
        if args.background:
            opts.background_top = tuple(min(1.0, c + 0.14) for c in args.background)
    return opts


# --------------------------------------------------------------------------
def cmd_info(args) -> int:
    rc = 0
    results = []
    for path in args.files:
        try:
            mesh = Mesh.load(path)
            st = mesh.statistics()
            report = stl_io.file_size_report(path)
            text = st.as_text()
            print(f"--- {path} ---")
            print(text)
            stored_v = report.get("vertices")
            if stored_v and stored_v != st.vertices:
                print(f"文件存储顶点: {stored_v}（载入时焊接，"
                      f"合并 {stored_v - st.vertices} 个重合顶点）")
            print(f"文件大小    : {report['human']} ({report['bytes']} 字节)")
            if report.get("header"):
                print(f"文件头      : {report['header']}")
            if "bytes_per_triangle" in report:
                print(f"每面字节数  : {report['bytes_per_triangle']:.1f}")
            results.append({"path": path, "statistics": st.__dict__,
                            "file": report})
        except Exception as exc:
            print(f"[错误] {path}: {exc}", file=sys.stderr)
            rc = 1
    if args.json:
        print(json.dumps(results, ensure_ascii=False, indent=2, default=str))
    return rc


def cmd_render(args) -> int:
    settings = AppSettings().load(args.settings) if args.settings else AppSettings()
    scene = _load_scene([args.file], args)
    cam = _make_camera(scene, args)
    opts = _make_options(args, settings.view)
    if args.transparent_object:
        for obj in scene.objects:
            obj.transparent = True
            obj.alpha = args.alpha
    renderer = SoftwareRenderer()
    out = args.output or (os.path.splitext(args.file)[0] + ".png")
    info = renderer.render_to_file(out, scene, cam, options=opts,
                                   light=LightSettings(), double_resolution=args.double)
    print(f"已输出: {info['path']}  尺寸={info['size']}  "
          f"面数={renderer.last_stats.get('triangles')}  "
          f"耗时={renderer.last_stats.get('seconds', 0):.2f}s")
    if args.json:
        print(json.dumps(info, ensure_ascii=False, indent=2, default=str))
    return 0


def cmd_convert(args) -> int:
    rc = 0
    for path in args.files:
        try:
            mesh = Mesh.load(path)
            if args.weld:
                mesh.weld(args.tolerance)
            if args.repair:
                info = mesh.repair(tolerance=args.tolerance)
                print(f"{path}: 修复 {info}")
            out = args.output if (args.output and len(args.files) == 1) else \
                os.path.splitext(path)[0] + ("_ascii.stl" if args.ascii else "_out.stl")
            _ensure_parent(out)
            if args.ascii:
                stl_io.write_ascii_stl(out, mesh.vertices, mesh.triangles, mesh.name)
            else:
                stl_io.write_binary_stl(out, mesh.vertices, mesh.triangles)
            print(f"已写出: {out}  V={mesh.vertex_count} F={mesh.triangle_count}")
        except Exception as exc:
            print(f"[错误] {path}: {exc}", file=sys.stderr)
            rc = 1
    return rc


def _ensure_parent(path: str) -> None:
    """确保输出文件的父目录存在（用户常直接给出尚不存在的目录）。"""
    parent = os.path.dirname(os.path.abspath(path))
    if parent:
        os.makedirs(parent, exist_ok=True)


def cmd_components(args) -> int:
    mesh = Mesh.load(args.file)
    if args.weld or args.repair:
        mesh.weld(1e-6)
    parts = mesh.split_components()
    prefix = args.out_prefix or os.path.splitext(args.file)[0]
    print(f"共 {len(parts)} 个连通分量")
    for i, part in enumerate(parts):
        if args.min_triangles and part.triangle_count < args.min_triangles:
            print(f"  [{i}] 跳过（面数 {part.triangle_count} < {args.min_triangles}）")
            continue
        out = f"{prefix}_part{i + 1:02d}.stl"
        _ensure_parent(out)
        stl_io.write_binary_stl(out, part.vertices, part.triangles)
        print(f"  [{i}] {out}  V={part.vertex_count} F={part.triangle_count} "
              f"体积={abs(part.volume()):.3f}")
    return 0


def cmd_measure(args) -> int:
    mesh = Mesh.load(args.file)
    units = UnitSystem(args.units)
    mgr = MeasureManager()

    def finish(kind, points, plane=None):
        mgr.start(kind)
        for p in points[:-1]:
            mgr.add_point(p)
        return mgr.add_point(points[-1], units=units, plane=plane)

    if args.point:
        finish("point3d", _parse_points(args.point, 1))
    if args.line:
        finish("line3d", _parse_points(args.line, 2))
    if args.circle:
        points = _parse_points(args.circle)
        kind = "circle2p" if len(points) == 2 else "circle3p"
        finish(kind, points, plane=np.array([0.0, 0.0, 1.0]))
    if args.rz:
        finish("rzpoint", _parse_points(args.rz, 1))
    if args.angle:
        finish("angle2d", _parse_points(args.angle, 3),
               plane=np.array([0.0, 0.0, 1.0]))
    if args.radius_at is not None:
        # 给定 Z 处的回转半径（回转体最大 R）
        z = float(args.radius_at)
        segs = mesh.section_polygons((0, 0, z), (0, 0, 1))
        rmax = 0.0
        for a, b in segs:
            for p in (a, b):
                rmax = max(rmax, float(np.hypot(p[0], p[1])))
        print(f"Z={z:.4f} 处最大回转半径 R={units.format_length(rmax)} {units.label}")
    results = mgr.export(units)
    for item in results:
        print(f"[{item['index']}] {item['kind']}: {item['text']}")
    if args.json:
        print(json.dumps(results, ensure_ascii=False, indent=2, default=str))
    return 0


# --------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="stl_viewer",
        description="STL Viewer —— 命令行模式")
    parser.add_argument("--settings", help="指定 INI 设置文件")
    sub = parser.add_subparsers(dest="command", required=True)

    p_info = sub.add_parser("info", help="统计模型与文件信息")
    p_info.add_argument("files", nargs="+")
    p_info.add_argument("--json", action="store_true")
    p_info.set_defaults(func=cmd_info)

    p_render = sub.add_parser("render", help="渲染图片")
    p_render.add_argument("file")
    p_render.add_argument("-o", "--output")
    p_render.add_argument("--size", type=_parse_size, default=(1024, 768))
    p_render.add_argument("--view", choices=["iso", "iso_back", "front", "back",
                                             "left", "right", "top", "bottom"])
    p_render.add_argument("--yaw", type=float)
    p_render.add_argument("--pitch", type=float)
    p_render.add_argument("--ortho", action="store_true")
    p_render.add_argument("--wire", action="store_true")
    p_render.add_argument("--colored", action="store_true",
                          help="整件同色（默认；与 --uniform 的区别是可叠加 --smooth）")
    p_render.add_argument("--orientation", action="store_true",
                          help="按法向着色（法向彩虹图，用颜色反映表面朝向）")
    p_render.add_argument("--component-colors", dest="component_colors",
                          action="store_true")
    p_render.add_argument("--curvature-colors", dest="curvature_colors",
                          action="store_true")
    p_render.add_argument("--uniform", action="store_true")
    p_render.add_argument("--smooth", action="store_true",
                          help="逐顶点法向插值（Gouraud）平滑着色")
    p_render.add_argument("--no-weld", dest="no_weld", action="store_true",
                          help="载入时不要自动焊接重合顶点")
    p_render.add_argument("--edges", action="store_true")
    p_render.add_argument("--grid", action="store_true")
    p_render.add_argument("--crosshair", action="store_true")
    p_render.add_argument("--no-axis", dest="no_axis", action="store_true")
    p_render.add_argument("--transparent", action="store_true",
                          help="透明背景（透明背景导出）")
    p_render.add_argument("--transparent-object", dest="transparent_object",
                          action="store_true", help="模型半透明显示")
    p_render.add_argument("--alpha", type=float, default=0.45)
    p_render.add_argument("--background", type=_parse_color)
    p_render.add_argument("--gradient", action="store_true")
    p_render.add_argument("--supersample", type=int, default=1)
    p_render.add_argument("--antialias", action="store_true")
    p_render.add_argument("--double", action="store_true", help="双倍分辨率")
    p_render.add_argument("--json", action="store_true")
    p_render.set_defaults(func=cmd_render)

    p_conv = sub.add_parser("convert", help="转换 / 修复 STL")
    p_conv.add_argument("files", nargs="+")
    p_conv.add_argument("-o", "--output")
    p_conv.add_argument("--ascii", action="store_true")
    p_conv.add_argument("--weld", action="store_true")
    p_conv.add_argument("--repair", action="store_true")
    p_conv.add_argument("--tolerance", type=float, default=1e-7)
    p_conv.set_defaults(func=cmd_convert)

    p_comp = sub.add_parser("components", help="拆分为连通分量")
    p_comp.add_argument("file")
    p_comp.add_argument("--out-prefix", dest="out_prefix")
    p_comp.add_argument("--min-triangles", dest="min_triangles", type=int, default=0)
    p_comp.add_argument("--weld", action="store_true")
    p_comp.add_argument("--repair", action="store_true")
    p_comp.set_defaults(func=cmd_components)

    p_meas = sub.add_parser("measure", help="测量计算")
    p_meas.add_argument("file")
    p_meas.add_argument("--units", choices=["mm", "inch"], default="mm")
    p_meas.add_argument("--point", nargs="+", help="一个点，如 \"1,2,3\"")
    p_meas.add_argument("--line", nargs="+", help="两点，如 \"0,0,-20;0,0,20\"")
    p_meas.add_argument("--circle", nargs="+",
                        help="两点圆（第一点圆心、第二点定半径）或三点圆，"
                             "点之间用 ; 分隔")
    p_meas.add_argument("--angle", nargs="+", help="三点（顶点为第一个点）")
    p_meas.add_argument("--rz", nargs="+", help="一个点，输出 R/Z/φ")
    p_meas.add_argument("--radius-at", dest="radius_at", type=float,
                        help="求指定 Z 高度处的最大回转半径")
    p_meas.add_argument("--json", action="store_true")
    p_meas.set_defaults(func=cmd_measure)
    return parser


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
