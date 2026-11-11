"""
stl_viewer —— Python STL / PLY 查看器
====================================

一个用纯 Python 实现的 STL / PLY 三维模型查看器。本仓库重建了常见查看器的
核心能力：模型读写、轨道相机、**双渲染后端**（OpenGL 硬件视口 + numpy 软件
光栅化）、线框 / 实体 / 彩色 / 透明显示、光照设置、坐标轴与十字线、2D/3D 测量、
多对象场景、单位换算、多语言、旋转动画、宏录制与截图导出。

> 图形界面优先使用 PyQt5 自带的 `QOpenGLWidget`（VBO + GLSL，硬件深度缓冲），
> 几十万至上百万三角面也能实时旋转；没有 GL 上下文时自动回退到自研软件
> 光栅化（Z-buffer + 画家算法 + 线性增量边函数），该后端仅依赖 numpy。
> 自带 PNG 编解码器与 5×7 位图字体，**不依赖 Pillow / PyOpenGL 等额外包**；
> 命令行模式不需要 PyQt5。

依赖: numpy（必需），PyQt5（图形界面可选）。
"""

__version__ = "1.0.0"
__all__ = [
    "mathutil",
    "stl_io",
    "mesh",
    "camera",
    "lighting",
    "renderer",
    "measure",
    "scene",
    "i18n",
    "settings",
    "macro",
    "png",
]
