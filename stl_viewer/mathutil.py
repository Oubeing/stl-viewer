"""向量 / 矩阵 / 几何工具（numpy 实现）。

世界坐标系约定：Z 轴为刀具（回转）轴，Z 向上。
所有矩阵为 4x4 numpy 数组，作用于列向量：p' = M @ p_h。
"""

from __future__ import annotations

import numpy as np

EPS = 1e-12


# --------------------------------------------------------------------------
# 基本向量运算
# --------------------------------------------------------------------------
def normalize(v: np.ndarray, axis: int = -1) -> np.ndarray:
    """按指定轴归一化，零向量返回零向量。"""
    v = np.asarray(v, dtype=np.float64)
    n = np.linalg.norm(v, axis=axis, keepdims=True)
    return np.divide(v, np.where(n < EPS, 1.0, n))


def rotation_matrix(axis: np.ndarray, angle_rad: float) -> np.ndarray:
    """Rodrigues 旋转矩阵（3x3）。"""
    axis = normalize(np.asarray(axis, dtype=np.float64))
    x, y, z = axis
    c, s = np.cos(angle_rad), np.sin(angle_rad)
    k = 1.0 - c
    return np.array(
        [
            [c + x * x * k, x * y * k - z * s, x * z * k + y * s],
            [y * x * k + z * s, c + y * y * k, y * z * k - x * s],
            [z * x * k - y * s, z * y * k + x * s, c + z * z * k],
        ],
        dtype=np.float64,
    )


def rotation_matrix_z(angle_rad: float) -> np.ndarray:
    """绕 Z 轴旋转（回转轴）。"""
    c, s = np.cos(angle_rad), np.sin(angle_rad)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)


# --------------------------------------------------------------------------
# 变换矩阵
# --------------------------------------------------------------------------
def translation(t: np.ndarray) -> np.ndarray:
    m = np.eye(4)
    m[:3, 3] = np.asarray(t, dtype=np.float64)
    return m


def scaling(s) -> np.ndarray:
    s = np.asarray(s, dtype=np.float64)
    if s.ndim == 0:
        s = np.repeat(s, 3)
    m = np.eye(4)
    m[0, 0], m[1, 1], m[2, 2] = s
    return m


def affine_from_3x3(r: np.ndarray, t=(0.0, 0.0, 0.0)) -> np.ndarray:
    m = np.eye(4)
    m[:3, :3] = r
    m[:3, 3] = t
    return m


def look_at(eye: np.ndarray, target: np.ndarray, up: np.ndarray) -> np.ndarray:
    """构造视图矩阵（右手系，相机沿 -Z 观察）。"""
    eye = np.asarray(eye, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    up = np.asarray(up, dtype=np.float64)
    f = normalize(target - eye)
    if np.linalg.norm(f) < EPS:  # eye == target，退化为默认朝向
        f = np.array([0.0, 0.0, -1.0])
    s = np.cross(f, up)
    if np.linalg.norm(s) < EPS:  # f 与 up 平行
        s = np.cross(f, np.array([1.0, 0.0, 0.0]))
        if np.linalg.norm(s) < EPS:
            s = np.cross(f, np.array([0.0, 1.0, 0.0]))
    s = normalize(s)
    u = np.cross(s, f)
    m = np.eye(4)
    m[0, :3] = s
    m[1, :3] = u
    m[2, :3] = -f
    m[0, 3] = -np.dot(s, eye)
    m[1, 3] = -np.dot(u, eye)
    m[2, 3] = np.dot(f, eye)
    return m


def perspective(fov_y_deg: float, aspect: float, znear: float, zfar: float) -> np.ndarray:
    """OpenGL 风格透视投影矩阵。"""
    t = znear * np.tan(np.radians(fov_y_deg) * 0.5)
    if t < EPS or aspect < EPS:
        return np.eye(4)
    m = np.zeros((4, 4))
    m[0, 0] = znear / (aspect * t)
    m[1, 1] = znear / t
    m[2, 2] = -(zfar + znear) / (zfar - znear)
    m[2, 3] = -2.0 * zfar * znear / (zfar - znear)
    m[3, 2] = -1.0
    return m


def orthographic(width: float, height: float, znear: float, zfar: float) -> np.ndarray:
    """正交投影矩阵（width/height 为视口的世界尺寸）。"""
    width = max(abs(width), EPS)
    height = max(abs(height), EPS)
    m = np.eye(4)
    m[0, 0] = 2.0 / width
    m[1, 1] = 2.0 / height
    m[2, 2] = -2.0 / (zfar - znear)
    m[2, 3] = -(zfar + znear) / (zfar - znear)
    return m


def transform_points(m: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """对 Nx3 点集应用 4x4 变换，返回 Nx3。"""
    pts = np.asarray(pts, dtype=np.float64)
    if pts.size == 0:
        return pts.reshape(0, 3)
    h = np.concatenate([pts, np.ones((pts.shape[0], 1))], axis=1)
    return (h @ m.T)[:, :3]


def transform_directions(m: np.ndarray, dirs: np.ndarray) -> np.ndarray:
    """对方向向量应用 3x3 线性部分（不平移）。"""
    dirs = np.asarray(dirs, dtype=np.float64)
    if dirs.size == 0:
        return dirs.reshape(0, 3)
    return dirs @ m[:3, :3].T


# --------------------------------------------------------------------------
# 三角形几何
# --------------------------------------------------------------------------
def triangle_normals(vertices: np.ndarray, triangles: np.ndarray,
                     normalize_result: bool = True) -> np.ndarray:
    """逐面法线（Mx3）。"""
    v0 = vertices[triangles[:, 0]]
    v1 = vertices[triangles[:, 1]]
    v2 = vertices[triangles[:, 2]]
    n = np.cross(v1 - v0, v2 - v0)
    return normalize(n) if normalize_result else n


def triangle_areas(vertices: np.ndarray, triangles: np.ndarray) -> np.ndarray:
    v0 = vertices[triangles[:, 0]]
    v1 = vertices[triangles[:, 1]]
    v2 = vertices[triangles[:, 2]]
    return 0.5 * np.linalg.norm(np.cross(v1 - v0, v2 - v0), axis=1)


def triangle_centroids(vertices: np.ndarray, triangles: np.ndarray) -> np.ndarray:
    return vertices[triangles].mean(axis=1)


def vertex_normals(vertices: np.ndarray, triangles: np.ndarray,
                   face_normals: np.ndarray | None = None) -> np.ndarray:
    """面积加权顶点法线（Gouraud 着色用）。"""
    if face_normals is None:
        face_normals = triangle_normals(vertices, triangles, normalize_result=False)
    out = np.zeros_like(vertices, dtype=np.float64)
    for k in range(3):
        np.add.at(out, triangles[:, k], face_normals)
    return normalize(out)


def signed_volume(vertices: np.ndarray, triangles: np.ndarray) -> float:
    """闭合网格的有向体积（mm^3）。"""
    if len(triangles) == 0:
        return 0.0
    v0 = vertices[triangles[:, 0]]
    v1 = vertices[triangles[:, 1]]
    v2 = vertices[triangles[:, 2]]
    return float(np.einsum("ij,ij->i", v0, np.cross(v1, v2)).sum() / 6.0)


# --------------------------------------------------------------------------
# 射线 / 平面相交
# --------------------------------------------------------------------------
def ray_triangle_intersection(origin: np.ndarray, direction: np.ndarray,
                              v0: np.ndarray, v1: np.ndarray, v2: np.ndarray):
    """Möller–Trumbore（标量版）。返回 (命中, t, 重心(u,v))。"""
    edge1 = v1 - v0
    edge2 = v2 - v0
    pvec = np.cross(direction, edge2)
    det = np.dot(edge1, pvec)
    if abs(det) < 1e-14:
        return False, 0.0, (0.0, 0.0)
    inv_det = 1.0 / det
    tvec = origin - v0
    u = np.dot(tvec, pvec) * inv_det
    if u < -1e-9 or u > 1 + 1e-9:
        return False, 0.0, (0.0, 0.0)
    qvec = np.cross(tvec, edge1)
    v = np.dot(direction, qvec) * inv_det
    if v < -1e-9 or u + v > 1 + 1e-9:
        return False, 0.0, (0.0, 0.0)
    t = np.dot(edge2, qvec) * inv_det
    if t < 1e-9:
        return False, 0.0, (0.0, 0.0)
    return True, float(t), (float(u), float(v))


def ray_plane_intersection(origin: np.ndarray, direction: np.ndarray,
                           plane_point: np.ndarray, plane_normal: np.ndarray):
    """射线与平面相交，返回交点或 None。"""
    denom = float(np.dot(plane_normal, direction))
    if abs(denom) < 1e-14:
        return None
    t = float(np.dot(plane_point - origin, plane_normal)) / denom
    return origin + direction * t


def closest_point_on_segment(p: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """点到线段最近点。"""
    ab = b - a
    denom = float(np.dot(ab, ab))
    if denom < EPS:
        return a.copy()
    t = float(np.clip(np.dot(p - a, ab) / denom, 0.0, 1.0))
    return a + ab * t


def circle_from_three_points(p1, p2, p3):
    """三点定圆。返回 (center, radius, normal) 或 None（三点共线）。"""
    p1 = np.asarray(p1, dtype=np.float64)
    p2 = np.asarray(p2, dtype=np.float64)
    p3 = np.asarray(p3, dtype=np.float64)
    a = p2 - p1
    b = p3 - p1
    n = np.cross(a, b)
    n2 = float(np.dot(n, n))
    if n2 < 1e-18:
        return None
    center = p1 + (np.cross(np.cross(a, b), a) * float(np.dot(b, b)) +
                   np.cross(np.cross(b, a), b) * float(np.dot(a, a))) / (2.0 * n2)
    radius = float(np.linalg.norm(center - p1))
    return center, radius, normalize(n)


def circle_from_two_points(center, rim, plane_normal):
    """圆心 + 圆周上一点定圆（Measure 2 Point Circle）。

    第一点为 **圆心**，第二点决定半径，圆落在以 ``plane_normal`` 为法向的
    平面内（屏幕视图测量时就是当前视图平面）。半径取第二点在**该平面内**
    的投影距离，而不是两点间的直线距离：

    取点时模型表面会让两个点有细微的深度差（尤其斜视角下），若直接用
    三维距离当半径，画出来的圆不会经过用户点的那个位置（视觉上「圆没跟上
    光标」）。投影到平面内既保证圆严格经过该点，也在两点同深度（量平面
    上的特征，最常见的情况）时与三维距离完全相等。

    返回 (center, radius, normal)；两点重合、或第二点几乎沿法向（平面内
    半径为 0）时返回 None。
    """
    center = np.asarray(center, dtype=np.float64)
    rim = np.asarray(rim, dtype=np.float64)
    n = normalize(np.asarray(plane_normal, dtype=np.float64))
    d = rim - center
    in_plane = d - n * float(np.dot(d, n))
    radius = float(np.linalg.norm(in_plane))
    if radius < 1e-9:
        return None
    return center.copy(), radius, n


def circle_points(center, radius, normal, count: int = 96) -> np.ndarray:
    """生成圆周采样点（用于绘制圆）。"""
    center = np.asarray(center, dtype=np.float64)
    n = normalize(np.asarray(normal, dtype=np.float64))
    tmp = np.array([0.0, 0.0, 1.0]) if abs(n[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
    u = normalize(np.cross(n, tmp))
    v = np.cross(n, u)
    t = np.linspace(0.0, 2.0 * np.pi, count, endpoint=False)
    return center + np.outer(np.cos(t) * radius, u) + np.outer(np.sin(t) * radius, v)


def angle_between(v1, v2) -> float:
    """两向量夹角（度）。"""
    v1 = np.asarray(v1, dtype=np.float64)
    v2 = np.asarray(v2, dtype=np.float64)
    n1 = np.linalg.norm(v1)
    n2 = np.linalg.norm(v2)
    if n1 < EPS or n2 < EPS:
        return 0.0
    c = float(np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0))
    return float(np.degrees(np.arccos(c)))


def cartesian_to_cylindrical(p) -> tuple:
    """直角坐标 -> 柱坐标 (R, phi_deg, Z)。"""
    x, y, z = (float(c) for c in np.asarray(p, dtype=np.float64)[:3])
    return float(np.hypot(x, y)), float(np.degrees(np.arctan2(y, x))), z


def cylindrical_to_cartesian(r: float, phi_deg: float, z: float) -> np.ndarray:
    a = np.radians(phi_deg)
    return np.array([r * np.cos(a), r * np.sin(a), z], dtype=np.float64)
