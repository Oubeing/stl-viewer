#!/usr/bin/env python
"""STL Viewer（Python）启动脚本。

    python run_viewer.py                  # 最大化窗口启动，空场景（无 objects）
    python run_viewer.py model.stl        # 启动并打开模型
    python run_viewer.py a.stl b.stl      # 启动并同时打开多个模型

无图形环境或只需要批处理时，请使用：python -m stl_viewer --cli --help
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from stl_viewer.viewer import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main())
