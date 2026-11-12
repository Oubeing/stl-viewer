"""旋转动画（Spin）与宏录制（Movie / Macro）—— Extras 菜单：

    miSpinEnable / miSpinType(Horizontal, Vertical, RX, RY, RZ, Roll, Random)
    miSpinDirection(Positive, Negative) / miSpinReverse
    miSpinSpeed(Low, Medium, High)
    miMovieMacroStartRecording / Execute / Open / Save / miMovieFullScreen
"""

from __future__ import annotations

import json
import math
import random
import time
from dataclasses import dataclass, field

import numpy as np

from .camera import Camera

SPIN_KINDS = ("horizontal", "vertical", "rx", "ry", "rz", "roll", "random")
SPIN_SPEEDS = {"low": 18.0, "medium": 55.0, "high": 140.0}   # 度/秒


@dataclass
class SpinConfig:
    """旋转动画参数。"""

    enabled: bool = False
    kind: str = "rz"            # horizontal | vertical | rx | ry | rz | roll | random
    direction: int = 1          # 1 正向 / -1 反向
    speed: str = "medium"       # low | medium | high
    custom_speed: float = 0.0   # >0 时覆盖 speed 档位（度/秒）

    @property
    def deg_per_second(self) -> float:
        if self.custom_speed > 0:
            return self.custom_speed
        return SPIN_SPEEDS.get(self.speed, 55.0)

    def to_dict(self) -> dict:
        return {"enabled": self.enabled, "kind": self.kind,
                "direction": self.direction, "speed": self.speed,
                "custom_speed": self.custom_speed}

    def set_state(self, state: dict) -> None:
        for key, value in (state or {}).items():
            if hasattr(self, key):
                setattr(self, key, value)


class SpinEngine:
    """把相机按配置连续旋转。"""

    def __init__(self, config: SpinConfig | None = None):
        self.config = config or SpinConfig()
        self._last = time.perf_counter()
        self._random_target = None
        self._random_until = 0.0

    def tick(self, camera: Camera, now: float | None = None) -> bool:
        """按经过时间推进动画，返回相机是否发生变化。"""
        now = now if now is not None else time.perf_counter()
        dt = max(min(now - self._last, 0.1), 0.0)
        self._last = now
        if not self.config.enabled or dt <= 0:
            return False

        delta = self.config.deg_per_second * dt * self.config.direction
        kind = self.config.kind
        if kind == "horizontal" or kind == "rz":
            # 绕世界 Z 轴回转：与 yaw 步进完全等价
            camera.rotate_about_axis(delta, "z")
        elif kind == "vertical":
            camera.pitch = float(np.clip(camera.pitch + delta, -89.9, 89.9))
            if camera.pitch in (89.9, -89.9):
                self.config.direction *= -1
        elif kind in ("rx", "ry"):
            # 真正的绕世界 X / Y 轴旋转（旧实现只是改俯仰/滚转，绕 Y 时
            # 模型看起来像在原地打转，并不是绕 Y 轴翻转）。
            camera.rotate_about_axis(delta, "x" if kind == "rx" else "y")
        elif kind == "roll":
            # 绕视线自身滚转：与「旋转轴 > 视线」共用同一个入口。
            camera.rotate_about_axis(delta, "view")
        elif kind == "random":
            if now > self._random_until or self._random_target is None:
                self._random_target = (random.uniform(-180, 180),
                                       random.uniform(-89.9, 89.9),
                                       random.uniform(0, 360))
                self._random_until = now + random.uniform(1.0, 2.5)
            camera.yaw += delta * 0.35
            ty, tp, tr = self._random_target
            camera.pitch += (tp - camera.pitch) * min(dt * 2.0, 0.5)
            camera.roll += (tr - camera.roll) * min(dt * 2.0, 0.5)
        return True

    def reset(self) -> None:
        self._last = time.perf_counter()


@dataclass
class MacroFrame:
    """宏中的一帧（相机状态 + 相对时间）。"""

    time: float
    state: dict


@dataclass
class Macro:
    """一段录制好的视图动画。"""

    frames: list = field(default_factory=list)
    duration: float = 0.0

    def add(self, camera: Camera, timestamp: float) -> None:
        self.frames.append(MacroFrame(timestamp, camera.state()))
        self.duration = timestamp

    def state_at(self, t: float) -> dict | None:
        """按时间线性插值出相机状态。"""
        if not self.frames:
            return None
        if t <= self.frames[0].time:
            return self.frames[0].state
        if t >= self.frames[-1].time:
            return self.frames[-1].state
        lo, hi = 0, len(self.frames) - 1
        while lo + 1 < hi:
            mid = (lo + hi) // 2
            if self.frames[mid].time <= t:
                lo = mid
            else:
                hi = mid
        f0, f1 = self.frames[lo], self.frames[hi]
        span = max(f1.time - f0.time, 1e-9)
        k = (t - f0.time) / span
        return _lerp_state(f0.state, f1.state, k)

    def save(self, path: str) -> None:
        payload = {
            "format": "num-stlviewer-macro",
            "version": 1,
            "duration": self.duration,
            "frames": [{"t": f.time, "camera": f.state} for f in self.frames],
        }
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=1)

    @classmethod
    def load(cls, path: str) -> "Macro":
        with open(path, "r", encoding="utf-8") as fh:
            payload = json.load(fh)
        macro = cls()
        for frame in payload.get("frames", []):
            macro.frames.append(MacroFrame(float(frame["t"]), frame["camera"]))
        macro.duration = float(payload.get("duration", macro.frames[-1].time
                                             if macro.frames else 0.0))
        return macro


class MacroRecorder:
    """录制 / 播放相机动画。"""

    def __init__(self, min_interval: float = 1.0 / 25.0):
        self.recording = False
        self.macro = Macro()
        self.min_interval = min_interval
        self._start = 0.0
        self._last_capture = -1.0
        self.playback_active = False
        self._play_start = 0.0

    # ---------------- 录制 ----------------
    def start(self) -> None:
        self.recording = True
        self.macro = Macro()
        self._start = time.perf_counter()
        self._last_capture = -1.0

    def stop(self) -> int:
        self.recording = False
        return len(self.macro.frames)

    def capture(self, camera: Camera, force: bool = False) -> bool:
        """记录一帧（受最小间隔限制，除非 force）。"""
        if not self.recording:
            return False
        now = time.perf_counter() - self._start
        if not force and (now - self._last_capture) < self.min_interval:
            return False
        self._last_capture = now
        self.macro.add(camera, now)
        return True

    # ---------------- 播放 ----------------
    def start_playback(self) -> None:
        self.playback_active = True
        self._play_start = time.perf_counter()

    def playback_state(self, camera: Camera) -> bool:
        """推进播放；返回是否仍在播放。"""
        if not self.playback_active:
            return False
        t = time.perf_counter() - self._play_start
        if t > self.macro.duration:
            self.playback_active = False
            return False
        state = self.macro.state_at(t)
        if state:
            camera.set_state(state)
        return True

    def stop_playback(self) -> None:
        self.playback_active = False

    # ---------------- 存取 ----------------
    def save(self, path: str) -> None:
        self.macro.save(path)

    def load(self, path: str) -> int:
        self.macro = Macro.load(path)
        return len(self.macro.frames)


# --------------------------------------------------------------------------
def _lerp_state(a: dict, b: dict, k: float) -> dict:
    """相机状态线性插值（角度按最短路径）。"""
    out = dict(a)
    for key, va in a.items():
        vb = b.get(key, va)
        if isinstance(va, (int, float)) and isinstance(vb, (int, float)):
            if key in ("yaw", "roll"):
                delta = ((vb - va + 180.0) % 360.0) - 180.0
                out[key] = va + delta * k
            else:
                out[key] = va + (vb - va) * k
        elif isinstance(va, list) and isinstance(vb, list) and len(va) == len(vb):
            out[key] = [x + (y - x) * k for x, y in zip(va, vb)]
        elif isinstance(va, bool):
            out[key] = va if k < 0.5 else vb
    return out


def ease_in_out(t: float) -> float:
    """平滑插值系数。"""
    return 0.5 - 0.5 * math.cos(math.pi * max(0.0, min(1.0, t)))
