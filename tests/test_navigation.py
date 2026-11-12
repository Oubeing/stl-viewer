"""CAD 风格导航回归测试（只依赖 numpy + 标准库，无需 PyQt5 / 桌面）。

运行：
    python -m unittest discover -s tests -p "test_navigation.py" -v
    pytest tests/test_navigation.py

覆盖：按键映射（含中键组合键）、旋转方向与灵敏度、自由轨迹球翻越极点、
光标锚点缩放（透视 / 正交）、抓取式平移、绕点旋转、惯性与设置归一化。
"""

from __future__ import annotations

import math
import unittest

import numpy as np

from stl_viewer import mathutil as mu
from stl_viewer.camera import Camera
from stl_viewer.navigation import (
    BTN_LEFT, BTN_MIDDLE, BTN_RIGHT, NONE, PAN, ROTATE, ROTATE_Z, ZOOM,
    NavOptions, NavigationController, limit_step, resolve_action,
)


def _camera(ortho: bool = False, size=(800, 600)) -> Camera:
    cam = Camera(*size)
    cam.fit(np.array([-10.0, -8.0, -6.0]), np.array([10.0, 8.0, 6.0]))
    if ortho:
        cam.set_orthographic(True)
    return cam


def _plane_point(cam: Camera, x: float, y: float, p0=None, n0=None):
    p0 = cam.target.copy() if p0 is None else p0
    n0 = cam.forward().copy() if n0 is None else n0
    return mu.ray_plane_intersection(*cam.ray(x, y), p0, n0)


class TestActionMapping(unittest.TestCase):
    def test_basic_buttons(self):
        for style in ("classic", "cad", "catia", "solidworks"):
            self.assertEqual(resolve_action(BTN_LEFT, BTN_LEFT, (), style), ROTATE)
            self.assertEqual(resolve_action(BTN_LEFT, BTN_LEFT, ("ctrl",), style), ROTATE_Z)
            self.assertEqual(resolve_action(BTN_LEFT, BTN_LEFT, ("alt",), style), ZOOM)
            self.assertEqual(resolve_action(BTN_LEFT, BTN_LEFT, ("shift",), style), PAN)
            self.assertEqual(resolve_action(BTN_RIGHT, BTN_RIGHT, (), style), PAN)

    def test_middle_button_differs_by_style(self):
        self.assertEqual(resolve_action(BTN_MIDDLE, BTN_MIDDLE, (), "classic"), PAN)
        self.assertEqual(resolve_action(BTN_MIDDLE, BTN_MIDDLE, (), "cad"), PAN)
        self.assertEqual(resolve_action(BTN_MIDDLE, BTN_MIDDLE, (), "catia"), PAN)
        self.assertEqual(resolve_action(BTN_MIDDLE, BTN_MIDDLE, (), "solidworks"), ROTATE)
        self.assertEqual(
            resolve_action(BTN_MIDDLE, BTN_MIDDLE, ("ctrl",), "solidworks"), PAN)

    def test_catia_combo_keys(self):
        # 中键按住后再按左键 / 右键 = 旋转；Ctrl / Shift + 中键 = 缩放
        for style in ("cad", "catia"):
            self.assertEqual(
                resolve_action(BTN_LEFT, BTN_MIDDLE | BTN_LEFT, (), style), ROTATE)
            self.assertEqual(
                resolve_action(BTN_RIGHT, BTN_MIDDLE | BTN_RIGHT, (), style), ROTATE)
            self.assertEqual(
                resolve_action(BTN_MIDDLE, BTN_MIDDLE, ("ctrl",), style), ZOOM)
            self.assertEqual(
                resolve_action(BTN_MIDDLE, BTN_MIDDLE, ("shift",), style), ZOOM)

    def test_unknown_button(self):
        self.assertEqual(resolve_action(0, 0, ()), NONE)

    def test_linear_step(self):
        self.assertEqual(limit_step(17.0, 200.0), 17.0)
        self.assertEqual(limit_step(-17.0, 200.0), -17.0)
        self.assertEqual(limit_step(500.0, 200.0), 200.0)
        self.assertEqual(limit_step(-500.0, 200.0), -200.0)
        # 严格线性：10px 与 100px 的比值必须正好 1:10
        self.assertAlmostEqual(limit_step(100, 200) / limit_step(10, 200), 10.0)


class TestCameraExtensions(unittest.TestCase):
    def test_orientation_roundtrip(self):
        for yaw, pitch, roll in ((45, 30, 0), (-120, -70, 25), (10, 89.5, -40),
                                (200, 0, 90)):
            cam = Camera(800, 600)
            cam.set_rotation_angles(yaw, pitch, roll)
            back, up = cam.offset_vector(), cam.up()
            copy = Camera(800, 600)
            copy.set_orientation(back, up)
            self.assertLess(np.linalg.norm(copy.offset_vector() - back), 1e-6,
                            f"back {yaw}/{pitch}/{roll}")
            self.assertLess(np.linalg.norm(copy.up() - up), 1e-6,
                            f"up {yaw}/{pitch}/{roll}")
            self.assertLess(np.linalg.norm(copy.right() - cam.right()), 1e-6)

    def test_arcball_matches_turntable_direction(self):
        turntable = Camera(640, 480)
        turntable.orbit_about_center(60, 0, 0.12)
        arc = Camera(640, 480)
        arc.rotate_arcball(320, 240, 380, 240, 0.12)
        self.assertLess(arc.yaw, 45.0)          # 向右拖 yaw 减小
        self.assertLess(turntable.yaw, 45.0)
        self.assertLess(abs(arc.yaw - turntable.yaw), 2.0)

        t2 = Camera(640, 480)
        t2.orbit_about_center(0, -60, 0.12)
        a2 = Camera(640, 480)
        a2.rotate_arcball(320, 240, 320, 300, 0.12)
        self.assertLess(a2.pitch, 30.0)         # 向下拖 pitch 减小
        self.assertAlmostEqual(a2.pitch, t2.pitch, delta=0.5)

    def test_arcball_is_linear_and_can_flip(self):
        a, b = Camera(640, 480), Camera(640, 480)
        a.rotate_arcball(320, 240, 370, 240, 0.2)
        b.rotate_arcball(320, 240, 370, 240, 0.4)
        self.assertAlmostEqual((b.yaw - 45) / (a.yaw - 45), 2.0, delta=0.05)

        cam = Camera(640, 480)
        # 连续向同一方向拖，越过极点不会像转台式相机那样卡在 ±89.9°
        for k in range(24):
            cam.rotate_arcball(320, 240, 320, 280, 0.5)
        self.assertTrue(np.isfinite(cam.roll) and abs(cam.roll) > 1e-6)
        self.assertGreater(abs(cam.pitch), 0.0)

    def test_dolly_at_keeps_cursor_anchor(self):
        for ortho in (False, True):
            for x, y in ((200, 150), (400, 300), (700, 500)):
                cam = _camera(ortho)
                p0, n0 = cam.target.copy(), cam.forward().copy()
                before = _plane_point(cam, x, y, p0, n0)
                nav = NavigationController(cam, speeds={"zoom": 0.07})
                self.assertTrue(nav.wheel(x, y, 1.0))
                after = _plane_point(cam, x, y, p0, n0)
                self.assertLess(np.linalg.norm(before - after), 1e-8,
                                f"ortho={ortho} anchor=({x},{y})")

    def test_dolly_at_multiple_notches(self):
        cam = _camera(False)
        x, y = 180.0, 420.0
        p0, n0 = cam.target.copy(), cam.forward().copy()
        before = _plane_point(cam, x, y, p0, n0)
        nav = NavigationController(cam, speeds={"zoom": 0.07})
        for _ in range(5):
            nav.wheel(x, y, 0.6)
        after = _plane_point(cam, x, y, p0, n0)
        self.assertLess(np.linalg.norm(before - after), 1e-7)

    def test_center_wheel_does_not_move_target(self):
        cam = _camera(False)
        t0 = cam.target.copy()
        NavigationController(cam).wheel(cam.width / 2, cam.height / 2, 1.0)
        self.assertLess(np.linalg.norm(cam.target - t0), 1e-9)

    def test_legacy_anchor_api_fixed(self):
        cam = _camera(False)
        x, y = 120.0, 90.0
        p0, n0 = cam.target.copy(), cam.forward().copy()
        before = _plane_point(cam, x, y, p0, n0)
        cam.dolly_about_center(1.0, anchor=(x, y))
        after = _plane_point(cam, x, y, p0, n0)
        self.assertLess(np.linalg.norm(before - after), 1e-8)

    def test_pan_grab_follows_cursor(self):
        for ortho in (False, True):
            cam = _camera(ortho)
            x0, y0 = 230.0, 170.0
            grabbed, p0, n0 = cam.pan_grab(x0, y0)
            cam.pan_drag_to(310.0, 250.0, (grabbed, p0, n0))
            now = _plane_point(cam, 310.0, 250.0, p0, n0)
            self.assertLess(np.linalg.norm(now - grabbed), 1e-8,
                            f"ortho={ortho}")
            scr, _, _ = cam.project(cam.target[None, :])
            self.assertTrue(np.allclose(scr[0], (cam.width / 2, cam.height / 2),
                                        atol=1e-6))

    def test_pan_grab_is_idempotent(self):
        cam = _camera(False)
        grab = cam.pan_grab(200, 150)
        cam.pan_drag_to(300, 220, grab)
        t1 = cam.target.copy()
        cam.pan_drag_to(300, 220, grab)
        self.assertLess(np.linalg.norm(cam.target - t1), 1e-9)

    def test_reframe(self):
        cam = _camera(False)
        point = np.array([3.0, -2.0, 1.0])
        cam.reframe(point)
        self.assertTrue(np.allclose(cam.target, point))
        scr, _, _ = cam.project(point[None, :])
        self.assertTrue(np.allclose(scr[0], (cam.width / 2, cam.height / 2),
                                    atol=1e-6))


class TestController(unittest.TestCase):
    def test_rotate_speed_is_linear(self):
        c1, c2 = Camera(640, 480), Camera(640, 480)
        n1 = NavigationController(c1, speeds={"rotate": 0.2})
        n2 = NavigationController(c2, speeds={"rotate": 0.4})
        n1.begin(ROTATE, 300, 200)
        n2.begin(ROTATE, 300, 200)
        n1.drag(400, 200)
        n2.drag(400, 200)
        self.assertAlmostEqual((45 - c2.yaw) / (45 - c1.yaw), 2.0, places=6)

    def test_pan_and_zoom_sessions(self):
        cam = _camera(False)
        nav = NavigationController(cam, speeds={"pan": 1.0, "zoom": 0.07})
        nav.begin(PAN, 200, 150)
        self.assertTrue(nav.drag(260, 190))
        self.assertTrue(nav.end() is False)
        d0 = cam.distance
        self.assertTrue(nav.wheel(200, 150, 1.0))
        self.assertNotAlmostEqual(cam.distance, d0)

    def test_pivot_cursor_reframes_on_begin(self):
        cam = _camera(False)
        opts = NavOptions(pivot="cursor")
        nav = NavigationController(cam, opts)
        expected = cam.ray_to_plane(180, 120)
        nav.begin(ROTATE, 180, 120)
        self.assertTrue(np.allclose(cam.target, expected))
        scr, _, _ = cam.project(expected[None, :])
        self.assertTrue(np.allclose(scr[0], (400, 300), atol=1e-6))

    def test_inertia_decays_and_stops(self):
        cam = Camera(640, 480)
        opts = NavOptions(inertia=True, inertia_halflife=0.1)
        nav = NavigationController(cam, opts, {"rotate": 0.2})
        nav.begin(ROTATE, 300, 200)
        nav.drag(380, 240)
        self.assertTrue(nav.end())
        self.assertGreater(math.hypot(*nav._velocity), 0.5)
        yaw_before = cam.yaw
        steps = 0
        while nav.advance(0.016):
            steps += 1
            self.assertLess(steps, 200)
        self.assertGreater(steps, 2)
        self.assertLess(cam.yaw, yaw_before)
        self.assertFalse(nav.advance(0.016))

    def test_no_inertia_by_default(self):
        cam = Camera(640, 480)
        nav = NavigationController(cam)
        nav.begin(ROTATE, 300, 200)
        nav.drag(380, 240)
        self.assertFalse(nav.end())
        self.assertFalse(nav.advance(0.016))

    def test_fine_modifier(self):
        normal, fine = Camera(640, 480), Camera(640, 480)
        n1 = NavigationController(normal, speeds={"rotate": 0.2})
        n2 = NavigationController(fine, speeds={"rotate": 0.2})
        n1.begin(ROTATE, 300, 200)
        n2.begin(ROTATE, 300, 200)
        n1.drag(400, 200)
        n2.drag(400, 200, ("shift",))
        self.assertAlmostEqual((45 - fine.yaw) / (45 - normal.yaw), 0.25,
                               places=5)

    def test_zoom_anchor_center_mode(self):
        cam = _camera(False)
        nav = NavigationController(cam, NavOptions(zoom_anchor="center"))
        t0 = cam.target.copy()
        nav.wheel(120, 90, 1.0)
        self.assertLess(np.linalg.norm(cam.target - t0), 1e-12)

    def test_options_normalized(self):
        o = NavOptions(style="bad", rotate_mode="?", zoom_anchor="?",
                       pivot="?", inertia_halflife=0.0).normalized()
        self.assertEqual(o.style, "cad")
        self.assertEqual(o.rotate_mode, "turntable")
        self.assertEqual(o.zoom_anchor, "cursor")
        self.assertEqual(o.pivot, "center")
        self.assertGreaterEqual(o.inertia_halflife, 0.02)

    def test_summary(self):
        self.assertIn("光标", NavigationController.summary(NavOptions()))
        self.assertIn("中心", NavigationController.summary(
            NavOptions(zoom_anchor="center")))

    def test_wheel_clamps_huge_trackpad_delta(self):
        cam = _camera(False)
        nav = NavigationController(cam)
        nav.wheel(200, 150, 1000.0)
        self.assertLess(cam.distance, 1e9)
        self.assertTrue(np.isfinite(cam.target).all())

    def test_drag_without_begin_is_noop(self):
        cam = Camera(640, 480)
        nav = NavigationController(cam)
        self.assertFalse(nav.drag(400, 300))
        self.assertEqual(cam.yaw, 45.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
