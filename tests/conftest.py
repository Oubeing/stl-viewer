"""Pytest 共享 fixture。

部分核心测试（test_stl_io / test_measure / test_settings_and_i18n /
test_cli）以 ``tmp: str`` 形式接收一个临时目录路径字符串。本文件提供该
fixture，复用 pytest 内置的 ``tmp_path``，确保这些核心用例在 pytest 下也能
正常运行（手动 ``main()`` 入口则直接传字符串目录，二者一致）。
"""


import pytest


@pytest.fixture
def tmp(tmp_path) -> str:
    """返回一个临时目录的字符串路径，供需要 str 路径的用例使用。"""
    return str(tmp_path)
