"""PNG 读写（仅依赖 zlib）—— 对应 Save Picture / Save Multi Res Picture /
Save Picture (transparent background) / Copy Picture to Clipboard。

自行实现 PNG 编解码可避免引入 Pillow 依赖，同时保证能在无 GUI 环境下导出图片。
"""

from __future__ import annotations

import struct
import zlib

import numpy as np


def write_png(path: str, image: np.ndarray) -> None:
    """写出 PNG。image 为 (H,W,3) 或 (H,W,4)，uint8。"""
    data = encode_png(image)
    with open(path, "wb") as fh:
        fh.write(data)


def encode_png(image: np.ndarray) -> bytes:
    arr = np.asarray(image, dtype=np.uint8)
    if arr.ndim == 2:
        arr = np.stack([arr] * 3, axis=-1)
    if arr.ndim != 3 or arr.shape[2] not in (3, 4):
        raise ValueError("图像必须是 (H,W,3) 或 (H,W,4) 的 uint8 数组")
    h, w, channels = arr.shape
    if channels == 3:
        color_type = 2
    else:
        color_type = 6

    # 逐行 append 的 Python 循环在 2K/4K 图上要几十毫秒；这里一次性拼出
    # 「每行一个 filter 字节 + 像素」的缓冲区，交给 numpy 连续写入。
    stride = w * channels
    raw = np.zeros((h, stride + 1), dtype=np.uint8)
    raw[:, 0] = 0                       # filter type 0 (None)
    raw[:, 1:] = arr.reshape(h, stride)

    def chunk(tag: bytes, payload: bytes) -> bytes:
        return (struct.pack(">I", len(payload)) + tag + payload +
                struct.pack(">I", zlib.crc32(tag + payload) & 0xFFFFFFFF))

    ihdr = struct.pack(">IIBBBBB", w, h, 8, color_type, 0, 0, 0)
    # 级别 3 与 6 都是无损 PNG，视觉完全一致；大图编码时间约减半。
    level = 3 if (w * h) > 1_500_000 else 6
    return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", ihdr) +
            chunk(b"IDAT", zlib.compress(raw.tobytes(), level)) +
            chunk(b"IEND", b""))


def read_png(path: str) -> np.ndarray:
    """读取 8 位非隔行 PNG（用于自检与回归测试）。"""
    with open(path, "rb") as fh:
        data = fh.read()
    if data[:8] != b"\x89PNG\r\n\x1a\n":
        raise ValueError("不是 PNG 文件")
    pos = 8
    width = height = None
    color_type = bit_depth = 0
    idat = bytearray()
    while pos < len(data):
        (length,) = struct.unpack_from(">I", data, pos)
        tag = data[pos + 4:pos + 8]
        payload = data[pos + 8:pos + 8 + length]
        pos += 12 + length
        if tag == b"IHDR":
            width, height, bit_depth, color_type, comp, filt, interlace = \
                struct.unpack(">IIBBBBB", payload)
            if bit_depth != 8 or interlace != 0:
                raise ValueError("仅支持 8 位非隔行 PNG")
        elif tag == b"IDAT":
            idat.extend(payload)
        elif tag == b"IEND":
            break
    raw = zlib.decompress(bytes(idat))
    channels = {0: 1, 2: 3, 4: 2, 6: 4}[color_type]
    stride = width * channels
    out = np.zeros((height, stride), dtype=np.uint8)
    prev = np.zeros(stride, dtype=np.uint8)
    offset = 0
    for y in range(height):
        ftype = raw[offset]
        offset += 1
        line = np.frombuffer(raw, dtype=np.uint8, count=stride, offset=offset).copy()
        offset += stride
        if ftype == 0:
            cur = line
        elif ftype == 1:
            cur = line
            for i in range(channels, stride):
                cur[i] = (int(cur[i]) + int(cur[i - channels])) & 0xFF
        elif ftype == 2:
            cur = (line.astype(np.uint16) + prev).astype(np.uint8)
        elif ftype == 3:
            cur = line
            for i in range(stride):
                left = int(cur[i - channels]) if i >= channels else 0
                cur[i] = (int(cur[i]) + ((left + int(prev[i])) >> 1)) & 0xFF
        elif ftype == 4:
            cur = line
            for i in range(stride):
                a = int(cur[i - channels]) if i >= channels else 0
                b = int(prev[i])
                c = int(prev[i - channels]) if i >= channels else 0
                p = a + b - c
                pa, pb, pc = abs(p - a), abs(p - b), abs(p - c)
                pred = a if (pa <= pb and pa <= pc) else (b if pb <= pc else c)
                cur[i] = (int(cur[i]) + pred) & 0xFF
        else:
            raise ValueError(f"不支持的 PNG 行滤波类型 {ftype}")
        out[y] = cur
        prev = cur
    arr = out.reshape(height, width, channels)
    if channels == 3:
        return arr
    if channels == 4:
        return arr
    if channels == 1:
        return np.repeat(arr, 3, axis=2)
    return np.repeat(arr[:, :, :1], 3, axis=2)
