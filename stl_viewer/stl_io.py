"""STL / PLY / STPACK / 嵌入式 EXE 模型读取与写出。

"Open (*.stl;*.stpack;*.ply;*.exe)" 与
"Save / Export STL"、"Show STL file size" 等功能。

* STL  : 二进制与 ASCII 两种编码，均支持读写
* PLY  : ASCII 与 binary_little/big_endian，读取顶点/面/颜色
* STPACK: 打包格式（可能带压缩/自定义块），此处尽力解析
* EXE  : 可导出"自包含查看器"，模型数据附加在 exe 尾部
"""

from __future__ import annotations

import io
import os
import re
import struct
import zlib
from dataclasses import dataclass, field

import numpy as np

STL_EXTENSIONS = (".stl", ".stpack", ".ply", ".exe")
_TEXTURE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".gif")

BINARY_HEADER = b"Exported by STL Viewer"


class MeshReadError(Exception):
    """模型文件读取失败。"""


@dataclass
class RawMesh:
    """从文件解析出的原始数据。"""

    vertices: np.ndarray
    triangles: np.ndarray
    name: str = "model"
    header: str = ""
    source: str = ""
    extra: dict = field(default_factory=dict)

    @property
    def triangle_count(self) -> int:
        return int(len(self.triangles))

    @property
    def vertex_count(self) -> int:
        return int(len(self.vertices))


# --------------------------------------------------------------------------
# 编码判定
# --------------------------------------------------------------------------
def _looks_like_binary_stl(data: bytes) -> bool:
    """判定 STL 是二进制还是 ASCII。

    二进制 STL: 80 字节头 + uint32 面片数 + 50*n 字节数据。
    注意：部分二进制文件头也以 b"solid" 开头，因此优先用长度等式判定。
    """
    if len(data) < 84:
        return False
    (count,) = struct.unpack_from("<I", data, 80)
    if count == 0:
        # 0 面片：若头部不是文本 solid 则视为二进制
        return not data[:5].lower().startswith(b"solid")
    expected = 84 + 50 * count
    if expected == len(data):
        return True
    # 允许文件尾有少量附加数据（导出程序有时附加元信息）
    if expected < len(data) and len(data) - expected < 4096:
        return True
    head = data[:512]
    return not (head[:5].lower().startswith(b"solid") and b"facet" in head.lower())


# --------------------------------------------------------------------------
# STL
# --------------------------------------------------------------------------
def read_binary_stl(data: bytes) -> tuple:
    """解析二进制 STL。返回 (vertices, triangles, header, name)。"""
    if len(data) < 84:
        raise MeshReadError("二进制 STL 长度不足 84 字节")
    header = data[:80].split(b"\x00")[0].decode("latin-1", "ignore").strip()
    (count,) = struct.unpack_from("<I", data, 80)
    need = 84 + 50 * count
    if count == 0 or need > len(data) + 0:
        if need > len(data):
            # 面片数不可信：按实际长度回推
            count = max((len(data) - 84) // 50, 0)
            need = 84 + 50 * count
    if count == 0:
        raise MeshReadError("二进制 STL 中面片数为 0")
    raw = np.frombuffer(data, dtype=np.uint8, count=need - 84, offset=84).reshape(count, 50)
    rec = np.ascontiguousarray(raw[:, :48]).view("<f4").reshape(count, 12)
    verts = np.ascontiguousarray(rec[:, 3:12]).reshape(-1, 3).astype(np.float64)
    tris = np.arange(count * 3, dtype=np.int64).reshape(count, 3)
    return verts, tris, header, header or "model"


_ASCII_VERTEX_RE = re.compile(
    rb"vertex\s+([-+0-9.eE]+)\s+([-+0-9.eE]+)\s+([-+0-9.eE]+)", re.IGNORECASE
)


def read_ascii_stl(data: bytes) -> tuple:
    """解析 ASCII STL。"""
    head_txt = data[:2048].decode("utf-8", "ignore")
    m = re.search(r"solid\s+(.*)", head_txt)
    name = m.group(1).strip() if m else "model"
    triples = _ASCII_VERTEX_RE.findall(data)
    if not triples:
        raise MeshReadError("ASCII STL 中未找到 vertex 记录")
    arr = np.array(triples, dtype=np.float64)
    if arr.shape[0] % 3 != 0:
        arr = arr[: (arr.shape[0] // 3) * 3]
    verts = arr.reshape(-1, 3)
    tris = np.arange(verts.shape[0], dtype=np.int64).reshape(-1, 3)
    return verts, tris, "", name


def read_stl(path: str) -> RawMesh:
    """读取 STL（自动识别二进制/ASCII）。"""
    with open(path, "rb") as fh:
        data = fh.read()
    if not data:
        raise MeshReadError("文件为空")
    if _looks_like_binary_stl(data):
        verts, tris, header, name = read_binary_stl(data)
    else:
        verts, tris, header, name = read_ascii_stl(data)
    return RawMesh(verts, tris, name=name, header=header, source=path)


def write_binary_stl(path: str, vertices: np.ndarray, triangles: np.ndarray,
                     header: str = "Exported by STL Viewer") -> None:
    """写出二进制 STL。"""
    vertices = np.asarray(vertices, dtype=np.float64)
    triangles = np.asarray(triangles, dtype=np.int64)
    from .mathutil import triangle_normals

    normals = triangle_normals(vertices, triangles)
    n = len(triangles)
    out = np.zeros((n, 50), dtype=np.uint8)
    rec = out[:, :48].view("<f4").reshape(n, 12)
    rec[:, 0:3] = normals
    rec[:, 3:6] = vertices[triangles[:, 0]]
    rec[:, 6:9] = vertices[triangles[:, 1]]
    rec[:, 9:12] = vertices[triangles[:, 2]]
    head = header.encode("ascii", "ignore")[:79].ljust(80, b"\x00")
    with open(path, "wb") as fh:
        fh.write(head)
        fh.write(struct.pack("<I", n))
        fh.write(out.tobytes())


def write_ascii_stl(path: str, vertices: np.ndarray, triangles: np.ndarray,
                    name: str = "model") -> None:
    """写出 ASCII STL。"""
    vertices = np.asarray(vertices, dtype=np.float64)
    triangles = np.asarray(triangles, dtype=np.int64)
    from .mathutil import triangle_normals

    normals = triangle_normals(vertices, triangles)
    v0, v1, v2 = (vertices[triangles[:, k]] for k in range(3))
    with open(path, "w", encoding="ascii") as fh:
        fh.write(f"solid {name}\n")
        fmt = ("  facet normal {a[0]:.6e} {a[1]:.6e} {a[2]:.6e}\n"
               "    outer loop\n"
               "      vertex {b[0]:.6e} {b[1]:.6e} {b[2]:.6e}\n"
               "      vertex {c[0]:.6e} {c[1]:.6e} {c[2]:.6e}\n"
               "      vertex {d[0]:.6e} {d[1]:.6e} {d[2]:.6e}\n"
               "    endloop\n  endfacet\n")
        for i in range(len(triangles)):
            fh.write(fmt.format(a=normals[i], b=v0[i], c=v1[i], d=v2[i]))
        fh.write(f"endsolid {name}\n")


# --------------------------------------------------------------------------
# PLY
# --------------------------------------------------------------------------
_PLY_TYPES = {
    "char": "i1", "uchar": "u1", "int8": "i1", "uint8": "u1",
    "short": "i2", "ushort": "u2", "int16": "i2", "uint16": "u2",
    "int": "i4", "uint": "u4", "int32": "i4", "uint32": "u4",
    "float": "f4", "float32": "f4", "double": "f8", "float64": "f8",
}


def read_ply(path: str) -> RawMesh:
    """读取 PLY（ASCII / binary_little_endian / binary_big_endian）。"""
    with open(path, "rb") as fh:
        data = fh.read()
    end = data.find(b"end_header")
    if end < 0:
        raise MeshReadError("PLY 头缺少 end_header")
    nl = data.find(b"\n", end)
    header_txt = data[:nl].decode("ascii", "ignore")
    body = data[nl + 1:]

    fmt = "ascii"
    elements = []
    for line in header_txt.splitlines():
        parts = line.split()
        if not parts:
            continue
        if parts[0] == "format":
            fmt = parts[1]
        elif parts[0] == "element":
            elements.append({"name": parts[1], "count": int(parts[2]), "props": []})
        elif parts[0] == "property" and elements:
            if parts[1] == "list":
                elements[-1]["props"].append(("list", parts[2], parts[3], parts[4]))
            else:
                elements[-1]["props"].append(("scalar", parts[1], parts[2]))

    verts = None
    faces = None

    if fmt == "ascii":
        tokens = body.decode("ascii", "ignore").split()
        pos = 0
        for el in elements:
            cnt, props = el["count"], el["props"]
            if el["name"] == "vertex":
                vlist = []
                for _ in range(cnt):
                    row = []
                    for prop in props:
                        if prop[0] == "scalar":
                            row.append(float(tokens[pos])); pos += 1
                        else:
                            n = int(tokens[pos]); pos += 1 + n
                    vlist.append(row[:3])
                verts = np.array(vlist, dtype=np.float64)
            elif el["name"] == "face":
                flist = []
                for _ in range(cnt):
                    idx = []
                    for prop in props:
                        if prop[0] == "list":
                            n = int(tokens[pos]); pos += 1
                            idx.extend(int(t) for t in tokens[pos:pos + n]); pos += n
                        else:
                            pos += 1
                    flist.append(idx)
                faces = flist
            else:
                for _ in range(cnt):
                    for prop in props:
                        if prop[0] == "scalar":
                            pos += 1
                        else:
                            n = int(tokens[pos]); pos += 1 + n
    else:
        big = fmt.endswith("big_endian")
        order = ">" if big else "<"
        offset = 0
        for el in elements:
            cnt, props = el["count"], el["props"]
            sizes = []
            for prop in props:
                if prop[0] == "scalar":
                    sizes.append(np.dtype(_PLY_TYPES.get(prop[1], "f4")).itemsize)
            if el["name"] == "vertex" and all(p[0] == "scalar" for p in props):
                dtype = np.dtype([(f"f{i}", order + _PLY_TYPES.get(p[1], "f4"))
                                  for i, p in enumerate(props)])
                arr = np.frombuffer(body, dtype=dtype, count=cnt, offset=offset)
                offset += arr.itemsize * cnt
                verts = np.column_stack([arr[f"f{i}"] for i in range(3)]).astype(np.float64)
            elif el["name"] == "face":
                flist = []
                if props and props[0][0] == "list":
                    ctype = np.dtype(order + _PLY_TYPES.get(props[0][1], "u1"))
                    itype = np.dtype(order + _PLY_TYPES.get(props[0][2], "i4"))
                    for _ in range(cnt):
                        n = int(np.frombuffer(body, dtype=ctype, count=1, offset=offset)[0])
                        offset += ctype.itemsize
                        idx = np.frombuffer(body, dtype=itype, count=n, offset=offset)
                        offset += itype.itemsize * n
                        flist.append([int(v) for v in idx])
                faces = flist
            else:
                stride = sum(sizes)
                offset += stride * cnt

    if verts is None:
        raise MeshReadError("PLY 中未找到 vertex 元素")
    if faces:
        tris = []
        for f in faces:
            for k in range(1, len(f) - 1):  # 多边形扇形三角化
                tris.append((f[0], f[k], f[k + 1]))
        triangles = np.array(tris, dtype=np.int64) if tris else np.zeros((0, 3), np.int64)
    else:
        triangles = np.zeros((0, 3), np.int64)
    return RawMesh(verts, triangles, name=os.path.splitext(os.path.basename(path))[0],
                   source=path)


# --------------------------------------------------------------------------
# 打包件 / 自包含 EXE
# --------------------------------------------------------------------------
def extract_embedded_mesh(path: str) -> RawMesh:
    """从自包含导出件（.exe / .stpack）中提取模型数据。

    策略：
      1. 查找附加的二进制 STL 块（长度自洽的 50 字节记录序列）
      2. 查找附加的 ASCII STL 文本块（solid ... endsolid）
      3. 查找 zlib 压缩流并尝试解压出 STL/PLY
    """
    with open(path, "rb") as fh:
        data = fh.read()

    # 1) 附加二进制 STL：从每个可能的位置尝试解析
    for marker in (BINARY_HEADER, b"solid"):
        start = data.find(marker)
        while start != -1:
            try:
                tail = data[start:]
                if len(tail) >= 84:
                    verts, tris, header, name = read_binary_stl(tail)
                    if len(tris) > 0 and _plausible(verts):
                        return RawMesh(verts, tris, name=name or "embedded",
                                       header=header, source=path,
                                       extra={"embedded_offset": start})
            except Exception:
                pass
            start = data.find(marker, start + 1)

    # 2) 附加 ASCII STL
    m = re.search(rb"solid[\s\S]{20,}?endsolid", data)
    if m:
        try:
            verts, tris, header, name = read_ascii_stl(m.group(0))
            if len(tris) > 0 and _plausible(verts):
                return RawMesh(verts, tris, name=name or "embedded", source=path,
                               extra={"embedded_offset": m.start()})
        except Exception:
            pass

    # 3) zlib 流
    for i in range(len(data) - 2):
        if data[i] == 0x78 and data[i + 1] in (0x01, 0x5E, 0x9C, 0xDA):
            try:
                blob = zlib.decompressobj().decompress(data[i:])
            except Exception:
                continue
            if len(blob) < 84:
                continue
            try:
                if _looks_like_binary_stl(blob):
                    verts, tris, header, name = read_binary_stl(blob)
                else:
                    verts, tris, header, name = read_ascii_stl(blob)
                if len(tris) > 0 and _plausible(verts):
                    return RawMesh(verts, tris, name=name or "embedded", source=path,
                                   extra={"embedded_offset": i, "compressed": True})
            except Exception:
                continue
    raise MeshReadError(
        f"未能在 {os.path.basename(path)} 中找到可解析的模型数据；"
        "该文件可能是独立可执行文件或使用了未知压缩格式"
    )


def _plausible(verts: np.ndarray, limit: float = 1e6) -> bool:
    """粗略判断解析出的坐标是否合理（避免误命中随机字节）。"""
    if verts.size == 0:
        return False
    if not np.all(np.isfinite(verts)):
        return False
    span = float(np.abs(verts).max())
    return 1e-6 < span < limit


# --------------------------------------------------------------------------
# 统一入口
# --------------------------------------------------------------------------
def read_mesh(path: str) -> RawMesh:
    """按扩展名分派读取。"""
    ext = os.path.splitext(path)[1].lower()
    if ext == ".ply":
        return read_ply(path)
    if ext in (".stl",):
        return read_stl(path)
    if ext in (".stpack", ".exe", ".bin", ".dat", ""):
        # 先按 STL 尝试，失败再走嵌入式提取
        try:
            return read_stl(path)
        except Exception:
            return extract_embedded_mesh(path)
    return read_stl(path)


def file_size_report(path: str) -> dict:
    """File 菜单的 "Show STL file size"。"""
    size = os.path.getsize(path)
    info = {"path": path, "bytes": size, "human": human_size(size)}
    try:
        raw = read_mesh(path)
        info.update(
            triangles=raw.triangle_count,
            vertices=raw.vertex_count,
            name=raw.name,
            header=raw.header,
            bytes_per_triangle=(size / raw.triangle_count) if raw.triangle_count else 0.0,
        )
    except Exception as exc:  # 读不出来也要能报文件大小
        info["error"] = str(exc)
    return info


def human_size(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024 or unit == "GB":
            return f"{n:.0f} {unit}" if unit == "B" else f"{n:.1f} {unit}"
        n /= 1024.0
    return f"{n:.1f} GB"
