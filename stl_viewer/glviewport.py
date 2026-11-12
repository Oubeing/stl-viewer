"""OpenGL 硬件视口 —— 大模型流畅交互的关键。

纯 numpy 软件光栅化在 84 万面模型上单帧要 1 秒以上，无论怎样降分辨率、
后台异步都只能让 UI「不阻塞」，却做不到「不卡」：画面一秒才更新一次。
本模块把实时显示交给 GPU：

* VBO 一次性上传（顶点位置 / 法线 / 颜色交错存放），之后每帧只更新矩阵
* GLSL 里完成与软件渲染器一致的双光源 Blinn-Phong 着色
* 硬件深度缓冲负责遮挡，天然没有软件 Z 缓冲的分批 / 排序问题
* 线框、边线用 ``glPolygonMode(GL_LINE)``，无需额外缓冲

软件渲染器继续承担「保存图片 / 命令行导出 / 无 GL 环境回退」。
坐标轴、网格、测量标注、中心支点标记由透明叠加控件用 QPainter 画在
GL 之上（QPainter 不能直接画在 QOpenGLWidget 的 FBO 上）。
"""

from __future__ import annotations

import time

import ctypes
import numpy as np
from PyQt5 import QtCore, QtGui, QtWidgets

# OpenGL 1.1+ 常量（避免依赖 QOpenGLFunctions 的枚举导出差异）
GL_LINES = 0x0001
GL_TRIANGLES = 0x0004
GL_FLOAT = 0x1406
GL_DEPTH_TEST = 0x0B71
GL_BLEND = 0x0BE2
GL_SRC_ALPHA = 0x0302
GL_ONE_MINUS_SRC_ALPHA = 0x0303
GL_LEQUAL = 0x0203
GL_LESS = 0x0201
GL_COLOR_BUFFER_BIT = 0x00004000
GL_DEPTH_BUFFER_BIT = 0x00000100
GL_LINE = 0x1B01
GL_FILL = 0x1B02
GL_FRONT_AND_BACK = 0x0408
GL_RGBA = 0x1908
GL_UNSIGNED_BYTE = 0x1401

VERTEX_SHADER = """
attribute vec3 aPos;
attribute vec3 aNormal;
attribute vec3 aColor;
uniform mat4 uMVP;
uniform mat4 uModelView;
varying vec3 vNormal;
varying vec3 vColor;
void main() {
    gl_Position = uMVP * vec4(aPos, 1.0);
    /* GLSL 1.10 不允许 mat3(mat4)，用 w=0 的向量乘法取线性部分即可。
       对象矩阵为刚体变换（旋转/平移/等比缩放），无需逆转置。 */
    vNormal = (uModelView * vec4(aNormal, 0.0)).xyz;
    vColor = aColor;
}
"""

FRAGMENT_SHADER = """
varying vec3 vNormal;
varying vec3 vColor;
uniform vec3 uL1;
uniform vec3 uL2;
uniform vec3 uLightColor;
uniform vec3 uAmbientColor;
uniform vec3 uShineColor;
uniform vec3 uEdgeColor;
uniform vec3 uColorUniform;
uniform float uI1;
uniform float uI2;
uniform float uAmbient;
uniform float uDiffuse;
uniform float uSpecular;
uniform float uShininess;
uniform float uShade;
uniform float uAlpha;
uniform float uUseEdge;
uniform float uUseUniformColor;
void main() {
    vec3 base = (uUseEdge > 0.5) ? uEdgeColor
              : (uUseUniformColor > 0.5 ? uColorUniform : vColor);
    vec3 col = base;
    if (uShade > 0.5) {
        vec3 n = normalize(vNormal);
        if (n.z < 0.0) n = -n;          // 与软件渲染器的 flip_backface 一致
        vec3 view = vec3(0.0, 0.0, 1.0);
        col = base * uAmbient * uAmbientColor;
        col += uDiffuse * uI1 * max(dot(n, uL1), 0.0) * uLightColor * base;
        col += uDiffuse * uI2 * max(dot(n, uL2), 0.0) * uLightColor * base;
        if (uSpecular > 0.0) {
            vec3 h1 = normalize(uL1 + view);
            col += pow(max(dot(n, h1), 0.0), uShininess) * uSpecular * uI1 * uShineColor;
            vec3 h2 = normalize(uL2 + view);
            col += pow(max(dot(n, h2), 0.0), uShininess) * uSpecular * uI2 * uShineColor;
        }
        col = clamp(col, 0.0, 1.0);
    }
    gl_FragColor = vec4(col, uAlpha);
}
"""

# 拾取用的极简着色器：两种模式共用，由 uMode 切换。
#   uMode = 0：把对象编号写入 RGB（24 位，低 8 位即对象下标+1，上限 255）
#   uMode = 1：把窗口深度编码进 RGB（24 位精度）
# 两遍都只依赖 RGBA8「颜色」读回（任何驱动都稳妥支持），绝不读
# GL_DEPTH_COMPONENT——当 FBO 深度附件是 GL_DEPTH24_STENCIL8（Qt 常见默认）
# 时该格式组合非法，驱动会返回清屏值 1.0，导致「点在模型上却永远拾取不到」。
# 深度缓冲仍照常写入并参与深度测试，保证最前景片元胜出。
PICK_VERTEX_SHADER = """
attribute vec3 aPos;
uniform mat4 uMVP;
void main() {
    gl_Position = uMVP * vec4(aPos, 1.0);
}
"""

PICK_FRAGMENT_SHADER = """
uniform float uPickId;     // 物体编号：0=背景，1..255=对象下标+1
uniform float uMode;       // 0=输出对象编号，1=输出深度编码
void main() {
    float enc;
    if (uMode < 0.5) {
        enc = clamp(uPickId, 0.0, 16777215.0);
    } else {
        float d = clamp(gl_FragCoord.z, 0.0, 1.0);   // 窗口深度 [0,1]
        enc = d * 16777215.0;                        // 24 位编码（2^24-1）
    }
    float r = floor(enc / 65536.0);
    float g = floor(mod(enc, 65536.0) / 256.0);
    float b = mod(enc, 256.0);
    gl_FragColor = vec4(r / 255.0, g / 255.0, b / 255.0, 1.0);
}
"""

STRIDE = 9 * 4          # pos3 + normal3 + color3，float32


class _GLEntry:
    """一个对象对应的 GPU 顶点缓冲。"""

    __slots__ = ("vbo", "count", "signature", "has_color")

    def __init__(self, vbo, count, signature, has_color):
        self.vbo = vbo
        self.count = count
        self.signature = signature
        self.has_color = has_color


class GLViewport(QtWidgets.QOpenGLWidget):
    """实时显示用硬件视口；作为 Canvas 的子控件，鼠标事件透传给 Canvas。"""

    def __init__(self, state, parent=None, frame_callback=None):
        super().__init__(parent)
        self.state = state
        self._frame_callback = frame_callback
        self.setAttribute(QtCore.Qt.WA_TransparentForMouseEvents, True)
        self.setAttribute(QtCore.Qt.WA_NoSystemBackground, True)
        self._f = None
        self._program = None
        self._pick_program = None
        self._pick_loc = {}
        self._pick_fbo = None
        # 拾取自保护：若 GPU 拾取在本机明显偏慢（软件 GL、老旧驱动等），
        # 连续两次超过阈值就自动停用，交互永远保持即时。
        self._pick_last_ms = 0.0
        self._pick_slow_count = 0
        self._pick_disabled = False
        self._ready = False
        self._cache: dict[int, _GLEntry] = {}
        self._loc = {}
        fmt = QtGui.QSurfaceFormat()
        fmt.setDepthBufferSize(24)
        fmt.setSamples(4)                 # 硬件 MSAA，静止/交互都清晰
        fmt.setSwapInterval(1)
        self.setFormat(fmt)
        self.setUpdateBehavior(QtWidgets.QOpenGLWidget.NoPartialUpdate)
        self._forwarding = False
        # WA_TransparentForMouseEvents 在多数平台会把事件直接交给父控件；
        # 万一平台仍把事件投到本控件，这里原样转发给 Canvas（两者同原点、
        # 同尺寸，坐标无需换算），保证旋转/平移/滚轮一定生效。
        self.installEventFilter(self)

    def eventFilter(self, obj, event) -> bool:
        if obj is self and not self._forwarding and event.type() in (
                QtCore.QEvent.MouseButtonPress, QtCore.QEvent.MouseButtonRelease,
                QtCore.QEvent.MouseMove, QtCore.QEvent.MouseButtonDblClick,
                QtCore.QEvent.Wheel, QtCore.QEvent.ContextMenu):
            canvas = self.parent()
            if canvas is not None:
                self._forwarding = True
                try:
                    QtWidgets.QApplication.sendEvent(canvas, event)
                finally:
                    self._forwarding = False
            return True
        return False

    @property
    def ready(self) -> bool:
        return self._ready

    # ------------------------------------------------------------------
    # 初始化
    # ------------------------------------------------------------------
    def initializeGL(self) -> None:
        ctx = self.context()
        if ctx is None:
            return
        profile = QtGui.QOpenGLVersionProfile()
        profile.setVersion(2, 1)
        try:
            self._f = ctx.versionFunctions(profile)
        except Exception:
            self._f = None
        if self._f is None:
            return
        program = QtGui.QOpenGLShaderProgram(self)
        if not program.addShaderFromSourceCode(QtGui.QOpenGLShader.Vertex,
                                               VERTEX_SHADER):
            return
        if not program.addShaderFromSourceCode(QtGui.QOpenGLShader.Fragment,
                                               FRAGMENT_SHADER):
            return
        if not program.link():
            return
        self._program = program
        for name in ("uMVP", "uModelView", "uL1", "uL2", "uLightColor",
                     "uAmbientColor", "uShineColor", "uEdgeColor", "uColorUniform",
                     "uI1", "uI2", "uAmbient", "uDiffuse", "uSpecular",
                     "uShininess", "uShade", "uAlpha", "uUseEdge",
                     "uUseUniformColor"):
            self._loc[name] = program.uniformLocation(name)

        # 拾取着色器（GPU 深度 + 对象 ID 帧缓冲，替代 CPU 全量射线相交）
        pick = QtGui.QOpenGLShaderProgram(self)
        if pick.addShaderFromSourceCode(QtGui.QOpenGLShader.Vertex,
                                        PICK_VERTEX_SHADER) and \
           pick.addShaderFromSourceCode(QtGui.QOpenGLShader.Fragment,
                                        PICK_FRAGMENT_SHADER) and pick.link():
            self._pick_program = pick
            self._pick_loc = {
                "uMVP": pick.uniformLocation("uMVP"),
                "uPickId": pick.uniformLocation("uPickId"),
                "uMode": pick.uniformLocation("uMode"),
            }
        else:
            self._pick_program = None
            self._pick_loc = {}

        self._ready = True

    def invalidate(self) -> None:
        """场景/颜色/网格变化后丢弃缓存（下一帧按需重建）。"""
        for entry in self._cache.values():
            try:
                entry.vbo.destroy()
            except Exception:
                pass
        self._cache.clear()

    # ------------------------------------------------------------------
    # VBO
    # ------------------------------------------------------------------
    @staticmethod
    def _signature(obj, opts) -> tuple:
        mesh = obj.mesh
        has_color = not obj.flat_color(opts.color_mode)   # 逐面取色才需要逐顶点色
        return (
            id(mesh), mesh.triangle_count, id(mesh.vertices), id(mesh.triangles),
            opts.color_mode, obj.color_mode, bool(obj.auto_color),
            tuple(float(v) for v in obj.color),
            id(getattr(obj, "_color_cache", None)),
            id(getattr(obj, "_flat_cache", None)),
            bool(opts.gouraud), bool(opts.shade), has_color,
        )

    def _vertex_data(self, obj, opts) -> tuple:
        """构造交错顶点数组与「是否含逐顶点颜色」标志。

        整件同色（auto / uniform）时不需要逐顶点颜色，只传 位置 + 法线
        （6 float/顶点），由着色器里的 uColorUniform 上色 —— 显存省约 1/3，
        对超大模型更友好；逐面取色（按分量 / 曲率 / 法向）才存 9 float/顶点。
        """
        mesh = obj.mesh
        tris = mesh.triangles
        m = len(tris)
        verts32 = np.ascontiguousarray(mesh.vertices, dtype=np.float32)
        pos = verts32[tris].reshape(-1, 3)                # (3M, 3)
        has_color = not obj.flat_color(opts.color_mode)
        if opts.gouraud and obj.flat_color(opts.color_mode):
            vn = np.asarray(mesh.vertex_normals(), dtype=np.float32)
            normal = vn[tris].reshape(-1, 3)
        else:
            fn = np.asarray(mesh.face_normals(), dtype=np.float32)
            normal = np.repeat(fn, 3, axis=0)            # (3M, 3)
        if has_color:
            base = np.asarray(obj.face_colors(opts.color_mode), dtype=np.float32)
            if base.shape == (m, 3):
                col = np.repeat(base, 3, axis=0)
            else:
                col = np.tile(base.reshape(-1)[:3], (3 * m, 1))
            data = np.empty((3 * m, 9), dtype=np.float32)
            data[:, 0:3] = pos
            data[:, 3:6] = normal
            data[:, 6:9] = col
        else:
            data = np.empty((3 * m, 6), dtype=np.float32)
            data[:, 0:3] = pos
            data[:, 3:6] = normal
        return data, has_color

    def _get_entry(self, obj, opts) -> _GLEntry | None:
        key = id(obj)
        sig = self._signature(obj, opts)
        entry = self._cache.get(key)
        if entry is not None and entry.signature == sig:
            return entry
        data, has_color = self._vertex_data(obj, opts)
        # 显存占用过大时放弃上传，避免 OOM 导致崩溃（软件路径仍可用于导出）
        if data.nbytes > 15_000_000_000:
            return None
        buf = QtGui.QOpenGLBuffer()
        if not buf.create():
            return None
        buf.bind()
        try:
            from PyQt5 import sip
            buf.allocate(sip.voidptr(data.ctypes.data), data.nbytes)
        except Exception:
            raw = data.tobytes()
            buf.allocate(raw, len(raw))
        buf.release()
        del data
        if entry is not None:
            try:
                entry.vbo.destroy()
            except Exception:
                pass
        entry = _GLEntry(buf, int(3 * len(obj.mesh.triangles)), sig, has_color)
        self._cache[key] = entry
        return entry

    # ------------------------------------------------------------------
    # 绘制
    # ------------------------------------------------------------------
    def resizeGL(self, w: int, h: int) -> None:
        if self._f is not None:
            fw, fh = self._fb_size()
            self._f.glViewport(0, 0, fw, fh)

    def _fb_size(self) -> tuple:
        """帧缓冲实际像素（考虑设备像素比 / Windows 缩放）。

        不乘 DPR 会把场景画到帧缓冲的左下角（典型表现：高分屏/缩放屏下
        模型只占画布一角）。QOpenGLWidget 的内部 FBO 尺寸是 逻辑尺寸 × DPR。
        """
        dpr = max(self.devicePixelRatioF(), 1.0)
        return max(1, int(round(self.width() * dpr))), \
            max(1, int(round(self.height() * dpr)))

    def paintGL(self) -> None:
        f, prog = self._f, self._program
        if f is None or prog is None or not self._ready:
            return
        state = self.state
        cam = state.camera
        opts = state.options
        w, h = max(self.width(), 1), max(self.height(), 1)
        fw, fh = self._fb_size()
        start = time.perf_counter()

        bg = np.asarray(opts.background, dtype=np.float32)
        f.glViewport(0, 0, fw, fh)
        f.glClearColor(float(bg[0]), float(bg[1]), float(bg[2]), 1.0)
        f.glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        f.glEnable(GL_DEPTH_TEST)
        f.glDepthFunc(GL_LEQUAL)
        f.glDepthMask(True)
        f.glDisable(GL_BLEND)
        f.glPolygonMode(GL_FRONT_AND_BACK, GL_FILL)

        old_viewport = (cam.width, cam.height)
        cam.set_viewport(w, h)
        try:
            view = cam.view_matrix()
            proj = cam.projection_matrix()
        finally:
            cam.set_viewport(*old_viewport)

        objects = [o for o in state.scene.objects
                   if o.visible and o.mesh.triangle_count > 0]
        if not objects:
            self._report(time.perf_counter() - start)
            return

        # 只保留仍在场景里的对象的缓冲
        alive = {id(o) for o in objects}
        for key in [k for k in self._cache if k not in alive]:
            try:
                self._cache[key].vbo.destroy()
            except Exception:
                pass
            del self._cache[key]

        prog.bind()
        self._set_light_uniforms(prog, state)
        model_view_cache = {}
        for obj in objects:
            model = self._object_matrix(obj)
            mv = view @ model
            mvp = proj @ mv
            entry = self._get_entry(obj, opts)
            if entry is None or entry.count == 0:
                continue
            prog.setUniformValue(self._loc["uMVP"], _qmatrix(mvp))
            prog.setUniformValue(self._loc["uModelView"], _qmatrix(mv))
            model_view_cache[id(obj)] = mv
            has_color = entry.has_color
            self._bind_attributes(prog, entry.vbo, has_color)
            is_wire = (obj.wire or (opts.wire and not opts.solid)) \
                and not opts.show_edges
            transparent = bool(obj.transparent or opts.transparent)
            alpha = float(obj.alpha) if transparent else 1.0
            prog.setUniformValue(self._loc["uAlpha"], alpha)
            prog.setUniformValue(self._loc["uUseEdge"], 0.0)
            prog.setUniformValue(self._loc["uShade"],
                                 0.0 if (not opts.shade or is_wire) else 1.0)
            if has_color:
                prog.setUniformValue(self._loc["uUseUniformColor"], 0.0)
            else:
                # 整件同色：用 per-object 统一色，省去逐顶点颜色属性
                fc = obj.face_colors(opts.color_mode)
                flat = np.asarray(fc, dtype=np.float32).reshape(-1)[:3]
                prog.setUniformValue(self._loc["uUseUniformColor"], 1.0)
                prog.setUniformValue(self._loc["uColorUniform"],
                                     QtGui.QVector3D(float(flat[0]),
                                                    float(flat[1]),
                                                    float(flat[2])))

            if transparent:
                f.glEnable(GL_BLEND)
                f.glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
                f.glDepthMask(False)
            if obj.always_on_top:
                f.glDisable(GL_DEPTH_TEST)

            if is_wire:
                f.glLineWidth(max(int(opts.line_width), 1))
                f.glPolygonMode(GL_FRONT_AND_BACK, GL_LINE)
                f.glDrawArrays(GL_TRIANGLES, 0, entry.count)
                f.glPolygonMode(GL_FRONT_AND_BACK, GL_FILL)
            else:
                f.glDrawArrays(GL_TRIANGLES, 0, entry.count)
                if opts.show_edges:
                    f.glPolygonMode(GL_FRONT_AND_BACK, GL_LINE)
                    f.glLineWidth(max(int(opts.line_width), 1))
                    prog.setUniformValue(self._loc["uUseEdge"], 1.0)
                    prog.setUniformValue(self._loc["uShade"], 0.0)
                    f.glDrawArrays(GL_TRIANGLES, 0, entry.count)
                    prog.setUniformValue(self._loc["uUseEdge"], 0.0)
                    f.glPolygonMode(GL_FRONT_AND_BACK, GL_FILL)

            if obj.always_on_top:
                f.glEnable(GL_DEPTH_TEST)
            if transparent:
                f.glDepthMask(True)
                f.glDisable(GL_BLEND)
            self._release_attributes(prog, has_color)
            entry.vbo.release()

        prog.release()
        f.glPolygonMode(GL_FRONT_AND_BACK, GL_FILL)
        self._report(time.perf_counter() - start)

    def _report(self, seconds: float) -> None:
        if self._frame_callback is not None:
            self._frame_callback(seconds)

    # ------------------------------------------------------------------
    # GPU 拾取（替代 CPU 全量射线相交，超大模型也能瞬时选点）
    # ------------------------------------------------------------------
    def _ensure_pick_fbo(self, pw: int, ph: int):
        fb = self._pick_fbo
        if fb is not None and fb.width() == pw and fb.height() == ph:
            return fb
        if fb is not None:
            try:
                fb.destroy()
            except Exception:
                pass
        fmt = QtGui.QOpenGLFramebufferObjectFormat()
        fmt.setAttachment(QtGui.QOpenGLFramebufferObject.Depth)
        self._pick_fbo = QtGui.QOpenGLFramebufferObject(pw, ph, fmt)
        return self._pick_fbo

    def pick(self, x: float, y: float, scene, camera, opts) -> dict | None:
        """GPU 拾取：在缩小的帧缓冲里分两遍渲染并各自读回 RGB——
        第一遍得「对象编号」，第二遍得「表面窗口深度」（24 位编码）。
        两遍都只依赖 RGBA8 颜色读回（任何驱动都稳妥），绝不读
        GL_DEPTH_COMPONENT（在 DEPTH24_STENCIL8 附件上该组合非法、必失败）。
        最后把深度反投影回世界坐标，得到模型表面点与命中对象。

        返回 {point, object, distance}；未命中（背景）或任何异常返回 None，
        调用方据此回退到「视平面投影点」或「软件拾取」。
        """
        if not self._ready or self._f is None or self._program is None \
                or self._pick_program is None:
            return None
        if self._pick_disabled:
            return None
        w = max(self.width(), 1)
        h = max(self.height(), 1)
        scale = min(1.0, 1024.0 / max(w, h))
        pw = max(1, int(round(w * scale)))
        ph = max(1, int(round(h * scale)))
        fb = self._ensure_pick_fbo(pw, ph)
        if fb is None or not fb.isValid():
            return None
        ctx = self.context()
        if ctx is None:
            return None
        self.makeCurrent()
        try:
            f = self._f
            fx = min(max(int(round(x * scale)), 0), pw - 1)
            fy = min(max(int(round((h - y) * scale)), 0), ph - 1)

            old_vp = (camera.width, camera.height)
            camera.set_viewport(w, h)
            try:
                view = camera.view_matrix()
                proj = camera.projection_matrix()
            finally:
                camera.set_viewport(*old_vp)

            objects = [o for o in scene.objects
                       if o.visible and not getattr(o, "locked", False)
                       and o.mesh.triangle_count > 0]
            if not objects:
                return None

            pick_prog = self._pick_program
            loc = self._pick_loc

            def _render_pick_pass(mode: float) -> None:
                """渲染一遍拾取：mode=0 输出对象编号，mode=1 输出深度编码。"""
                tp0 = time.perf_counter()
                fb.bind()
                f.glViewport(0, 0, pw, ph)
                f.glClearColor(0.0, 0.0, 0.0, 0.0)
                f.glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
                f.glEnable(GL_DEPTH_TEST)
                f.glDepthFunc(GL_LEQUAL)
                f.glDepthMask(True)
                f.glDisable(GL_BLEND)
                f.glPolygonMode(GL_FRONT_AND_BACK, GL_FILL)
                pick_prog.bind()
                pick_prog.setUniformValue(loc["uMode"], float(mode))
                for idx, obj in enumerate(objects):
                    entry = self._get_entry(obj, opts)
                    if entry is None or entry.count == 0:
                        continue
                    oid = idx + 1
                    if oid > 255:          # 编号上限 255（24 位 RGB 仍够用）
                        continue
                    mvp = proj @ (view @ self._object_matrix(obj))
                    pick_prog.setUniformValue(loc["uMVP"], _qmatrix(mvp))
                    pick_prog.setUniformValue(loc["uMode"], float(mode))
                    if mode < 0.5:
                        pick_prog.setUniformValue(loc["uPickId"], float(oid))
                    stride = 9 * 4 if entry.has_color else 6 * 4
                    entry.vbo.bind()
                    pick_prog.setAttributeBuffer("aPos", GL_FLOAT, 0, 3, stride)
                    pick_prog.enableAttributeArray("aPos")
                    f.glDrawArrays(GL_TRIANGLES, 0, entry.count)
                    pick_prog.disableAttributeArray("aPos")
                    entry.vbo.release()
                pick_prog.release()
                f.glFinish()                    # 确保 GPU 完成，读回才可靠
                dt = (time.perf_counter() - tp0) * 1000.0
                self._pick_last_ms = dt
                if dt > 300.0:
                    self._pick_slow_count += 1
                    if self._pick_slow_count >= 2:
                        # 本机拾取太慢（软件 GL / 老驱动）：停用，改用视平面坐标
                        self._pick_disabled = True
                else:
                    self._pick_slow_count = 0

            def _read_px() -> tuple:
                """读回 1 像素 RGBA，返回 4 个 0..255 整数。

                PyQt5 的 QOpenGLFunctions.glReadPixels 在本机构建上只有
                GL_RGBA/GL_FLOAT 可用（直接返回浮点元组），旧式指针形式会抛
                TypeError、UNSIGNED_BYTE 会抛 ValueError。故优先用浮点元组读回，
                再把 8-bit 归一化值还原为整数；若浮点形式不可用，再退回指针字节。
                """
                try:
                    px = f.glReadPixels(fx, fy, 1, 1, GL_RGBA, GL_FLOAT)
                    if isinstance(px, (tuple, list)) and len(px) >= 4:
                        return (int(round(float(px[0]) * 255.0)),
                                int(round(float(px[1]) * 255.0)),
                                int(round(float(px[2]) * 255.0)),
                                int(round(float(px[3]) * 255.0)))
                except Exception:
                    pass
                try:
                    from PyQt5 import sip
                    buf = (ctypes.c_ubyte * 4)(0, 0, 0, 0)
                    f.glReadPixels(fx, fy, 1,1, GL_RGBA, GL_UNSIGNED_BYTE,
                                   sip.voidptr(ctypes.addressof(buf)))
                    return (int(buf[0]), int(buf[1]), int(buf[2]), int(buf[3]))
                except Exception:
                    return (0, 0, 0, 0)

            # 第一遍：对象编号（RGB）
            _render_pick_pass(0.0)
            id_r, id_g, id_b, _ = _read_px()
            oid = (id_r << 16) | (id_g << 8) | id_b
            fb.release()
            if oid == 0 or oid > len(objects):
                return None
            obj = objects[oid - 1]

            # 第二遍：表面深度（RGB，24 位编码）
            _render_pick_pass(1.0)
            d_r, d_g, d_b, _ = _read_px()
            fb.release()
            enc = (d_r << 16) | (d_g << 8) | d_b
            depth = enc / 16777215.0          # 还原窗口深度 [0,1]
            if not (0.0 < depth < 1.0):
                return None

            ndc_x = (fx + 0.5) / pw * 2.0 - 1.0
            ndc_y = (fy + 0.5) / ph * 2.0 - 1.0
            ndc_z = depth * 2.0 - 1.0
            mvp = proj @ view
            inv = np.linalg.inv(mvp)
            p = inv @ np.array([ndc_x, ndc_y, ndc_z, 1.0], dtype=np.float64)
            if abs(p[3]) < 1e-12:
                return None
            world = p[:3] / p[3]
            return {"point": world, "object": obj,
                    "distance": float(depth)}
        except Exception:
            try:
                self._pick_fbo.release()
            except Exception:
                pass
            return None

    def _set_light_uniforms(self, prog, state) -> None:
        light = state.light
        loc = self._loc
        d = np.asarray(light.dirs, dtype=np.float32)
        prog.setUniformValue(loc["uL1"], QtGui.QVector3D(*[float(x) for x in d[0]]))
        prog.setUniformValue(loc["uL2"], QtGui.QVector3D(*[float(x) for x in d[1]]))
        prog.setUniformValue(loc["uLightColor"],
                             QtGui.QVector3D(*[float(x) for x in light.light_color]))
        prog.setUniformValue(loc["uAmbientColor"],
                             QtGui.QVector3D(*[float(x) for x in light.ambient_color]))
        prog.setUniformValue(loc["uShineColor"],
                             QtGui.QVector3D(*[float(x) for x in light.shine_color]))
        edge = np.asarray(getattr(state.options, "edge_color", (0.1, 0.1, 0.1)),
                          dtype=np.float32)
        prog.setUniformValue(loc["uEdgeColor"],
                             QtGui.QVector3D(*[float(x) for x in edge]))
        for name, value in (("uI1", light.primary_intensity),
                            ("uI2", light.secondary_intensity),
                            ("uAmbient", light.ambient if light.enabled else 1.0),
                            ("uDiffuse", light.diffuse if light.enabled else 0.0),
                            ("uSpecular", light.specular if light.enabled else 0.0),
                            ("uShininess", max(float(light.shininess), 1.0))):
            prog.setUniformValue(loc[name], float(value))

    @staticmethod
    def _object_matrix(obj) -> np.ndarray:
        matrix = getattr(obj, "matrix", None)
        return np.asarray(matrix, dtype=np.float64) if matrix is not None else np.eye(4)

    def _bind_attributes(self, prog, vbo, has_color: bool) -> None:
        vbo.bind()
        stride = 9 * 4 if has_color else 6 * 4
        prog.setAttributeBuffer("aPos", GL_FLOAT, 0, 3, stride)
        prog.setAttributeBuffer("aNormal", GL_FLOAT, 3 * 4, 3, stride)
        prog.enableAttributeArray("aPos")
        prog.enableAttributeArray("aNormal")
        if has_color:
            prog.setAttributeBuffer("aColor", GL_FLOAT, 6 * 4, 3, stride)
            prog.enableAttributeArray("aColor")
        else:
            # 不绑定 aColor：着色器用 uColorUniform 上色，未启用属性取默认值
            prog.disableAttributeArray("aColor")

    @staticmethod
    def _release_attributes(prog, has_color: bool) -> None:
        prog.disableAttributeArray("aPos")
        prog.disableAttributeArray("aNormal")
        if has_color:
            prog.disableAttributeArray("aColor")


def _qmatrix(m: np.ndarray) -> QtGui.QMatrix4x4:
    """numpy 行主序 4x4 -> QMatrix4x4（实测与 GLSL 的列主序一致）。"""
    return QtGui.QMatrix4x4(*[float(v) for v in np.asarray(m, dtype=np.float64).flatten()])


class ViewportOverlay(QtWidgets.QWidget):
    """叠在 GL 视口之上的透明 2D 层，鼠标事件透传给 Canvas。"""

    def __init__(self, paint_fn, parent=None):
        super().__init__(parent)
        self._paint_fn = paint_fn
        self.setAttribute(QtCore.Qt.WA_TransparentForMouseEvents, True)
        self.setAttribute(QtCore.Qt.WA_NoSystemBackground, True)
        self.setAttribute(QtCore.Qt.WA_TranslucentBackground, True)
        self.setAttribute(QtCore.Qt.WA_OpaquePaintEvent, False)

    def paintEvent(self, event) -> None:
        painter = QtGui.QPainter(self)
        try:
            self._paint_fn(painter)
        finally:
            painter.end()
