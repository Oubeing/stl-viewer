"""光照模型 —— Light 菜单（Light Settings / Light options /
Light Down, DownLeft, Left / Reset Light Settings / Shine: Color）。

灯光方向定义在相机空间（随视图一起旋转），与机床仿真查看器的习惯一致：
无论怎么转动模型，光源始终相对于观察者固定。
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from . import mathutil as mu

# 光源预设：名称 -> 相机空间方向（由场景指向光源），已归一化
LIGHT_PRESETS = {
    "front": (0.0, 0.0, 1.0),
    "back": (0.0, 0.0, -1.0),
    "left": (-1.0, 0.0, 0.0),
    "right": (1.0, 0.0, 0.0),
    "up": (0.0, 1.0, 0.0),
    "down": (0.0, -1.0, 0.0),
    "up_left": (-0.6, 0.75, 0.3),
    "up_right": (0.6, 0.75, 0.3),
    "down_left": (-0.6, -0.75, 0.3),
    "down_right": (0.6, -0.75, 0.3),
    "left_front": (-0.8, 0.3, 0.5),
    "right_front": (0.8, 0.3, 0.5),
}


@dataclass
class LightSettings:
    """可调光照参数。"""

    enabled: bool = True
    ambient: float = 0.30          # 环境光强度
    diffuse: float = 0.75          # 漫反射强度
    specular: float = 0.25         # 高光强度
    shininess: float = 32.0        # 高光指数
    light_color: tuple = (1.0, 1.0, 1.0)     # 光源颜色
    shine_color: tuple = (1.0, 1.0, 1.0)     # Shine: Color
    ambient_color: tuple = (1.0, 1.0, 1.0)   # 环境光颜色
    two_sided: bool = True         # Back and Front：背面也受光
    primary: str = "up_left"       # 主光预设名
    secondary: str = "down_right"  # 补光预设名
    primary_intensity: float = 1.0
    secondary_intensity: float = 0.35
    flat: bool = False             # 是否用面法线（True）还是顶点法线插值

    dirs: np.ndarray = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self.rebuild()

    def rebuild(self) -> None:
        """把预设名解析为方向向量表。"""
        d1 = mu.normalize(np.array(LIGHT_PRESETS.get(self.primary, LIGHT_PRESETS["up_left"])))
        d2 = mu.normalize(np.array(LIGHT_PRESETS.get(self.secondary, LIGHT_PRESETS["down_right"])))
        self.dirs = np.stack([d1, d2], axis=0)

    def set_primary(self, name: str) -> None:
        if name in LIGHT_PRESETS:
            self.primary = name
            self.rebuild()

    def set_secondary(self, name: str) -> None:
        if name in LIGHT_PRESETS:
            self.secondary = name
            self.rebuild()

    def reset(self) -> None:
        self.__init__()  # type: ignore[misc]

    # ---------------- 着色 ----------------
    def shade(self, normals: np.ndarray, view_direction=None,
              positions=None, flip_backface: bool = True) -> np.ndarray:
        """计算每个面/顶点的 RGB 强度系数（0..1.5 之间的浮点数组）。

        normals        : (N,3) 相机空间法线
        view_direction : 单位向量，由物体指向相机（相机空间下为 +Z 朝向观察者）
        """
        n = np.asarray(normals, dtype=np.float64)
        if n.size == 0:
            return np.zeros((0, 3))
        if not self.enabled:
            return np.ones((len(n), 3))

        if flip_backface:
            # 让法线朝向相机，使背面（内侧）也被照亮
            face_to_view = np.array([0.0, 0.0, 1.0])
            flip = (n @ face_to_view) < 0
            n = np.where(flip[:, None], -n, n)

        ambient = self.ambient * np.array(self.ambient_color, dtype=np.float64)
        total = np.tile(ambient, (len(n), 1))

        view = np.array([0.0, 0.0, 1.0], dtype=np.float64)
        for i, direction in enumerate(self.dirs):
            intensity = self.primary_intensity if i == 0 else self.secondary_intensity
            if intensity <= 0:
                continue
            ndl = np.clip(n @ direction, 0.0, 1.0)
            col = np.array(self.light_color, dtype=np.float64)
            total += (self.diffuse * intensity) * ndl[:, None] * col
            if self.specular > 0:
                half = mu.normalize(direction + view)
                ndh = np.clip(n @ half, 0.0, 1.0)
                spec = (ndh ** self.shininess) * self.specular * intensity
                total += spec[:, None] * np.array(self.shine_color, dtype=np.float64)
        return np.clip(total, 0.0, 1.5)

    def state(self) -> dict:
        return {
            "enabled": self.enabled,
            "ambient": self.ambient,
            "diffuse": self.diffuse,
            "specular": self.specular,
            "shininess": self.shininess,
            "light_color": list(self.light_color),
            "shine_color": list(self.shine_color),
            "ambient_color": list(self.ambient_color),
            "two_sided": self.two_sided,
            "primary": self.primary,
            "secondary": self.secondary,
            "primary_intensity": self.primary_intensity,
            "secondary_intensity": self.secondary_intensity,
            "flat": self.flat,
        }

    def set_state(self, state: dict) -> None:
        if not state:
            return
        for key, value in state.items():
            if key in ("light_color", "shine_color", "ambient_color"):
                setattr(self, key, tuple(float(c) for c in value))
            elif hasattr(self, key):
                setattr(self, key, value)
        self.rebuild()
