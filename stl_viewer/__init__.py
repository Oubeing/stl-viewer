"""
stl_viewer —— Python STL / PLY 查看器
====================================

一个用纯 Python 实现的 STL / PLY 三维模型查看器。本仓库重建了常见查看器的
核心能力：模型读写、**CAD 风格相机导航**（光标锚点缩放、抓取式平移、
自由轨迹球旋转、中键组合键、惯性，对标 CATIA / SolidWorks）、**双渲染后端**
（OpenGL 硬件视口 + numpy 软件光栅化）、线框 / 实体 / 彩色 / 透明显示、
光照设置、始终留在画布内的坐标轴与贯穿画布的十字线、2D/3D 测量（长度与
三点圆均按视图投影平面口径，另有 3D 距离模式）、X/Y/Z/视线多轴旋转（拖拽
轴与旋转动画共用同一根轴）、多对象场景、单位换算、多语言、旋转动画、宏录制；
Save Picture 直接抓当前画布帧（返回自己拥有内存的像素副本，可安全后台编码），
不再重新渲染场景。

> 图形界面优先使用 PyQt5 自带的 `QOpenGLWidget`（VBO + GLSL，硬件深度缓冲），
> 几十万至上百万三角面也能实时旋转；没有 GL 上下文时自动回退到自研软件
> 光栅化（Z-buffer + 画家算法 + 线性增量边函数），该后端仅依赖 numpy。
> 自带 PNG 编解码器与 5×7 位图字体，**不依赖 Pillow / PyOpenGL 等额外包**；
> 命令行模式不需要 PyQt5。
>
> 内置 MCP 服务供 AI 调用：在图形界面的 `Extras → MCP Service Settings` 中启用
> （内嵌 Streamable HTTP，AI 直接操作当前窗口场景，含令牌 / 沙箱 / 权限与客户端
> 配置写入）；无界面场景可用 `python -m stl_viewer --mcp --root .`（零依赖 stdio
> 版）或 `--mcp-sdk`（官方 SDK 版）。业务层 `mcp_service.py` 与传输层分离，15 个
> 工具覆盖载入 / 统计 / 视角 / 测量 / 渲染 / 导出，详见 README 第 8 节。

依赖: numpy（必需），PyQt5（图形界面可选），mcp（SDK 版可选）。
"""

__version__ = "1.0.0"
__all__ = [
    "mathutil",
    "stl_io",
    "mesh",
    "camera",
    "navigation",
    "lighting",
    "renderer",
    "measure",
    "scene",
    "i18n",
    "settings",
    "macro",
    "png",
    "mcp_config",
    "mcp_service",
    "mcp_http",
    "mcp_runtime",
    "mcp_dialog",
    "mcp_stdio",
    "mcp_fastmcp",
]
