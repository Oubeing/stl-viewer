"""MCP 服务设置对话框（分类导航 + 参数页 + 运行状态）。

界面形态与 PrePoMax 的自动化服务设置一致：一个总开关、服务地址与端口、
访问令牌生成/复制、权限与工作区参数、AI 客户端配置一键写入，以及实时
运行日志。所有参数持久化到应用 INI，Apply / OK 立即对正在运行的服务生效。
"""

from __future__ import annotations

import copy
import os

from PyQt5 import QtCore, QtGui, QtWidgets

from .mcp_config import (MCPSettings, client_config_preview, generate_token,
                         remove_client_config, write_client_config)
from .mcp_service import TOOLS

#: 这些字段变化需要重启服务（令牌、端口、沙箱、权限都影响连接与鉴权）
_CRITICAL_FIELDS = (
    "enabled", "host", "port", "token", "require_token", "workspace_root",
    "allow_scene_changes", "allow_file_write", "enabled_tools",
)


def _signature(cfg: MCPSettings) -> tuple:
    return tuple((k, repr(getattr(cfg, k))) for k in _CRITICAL_FIELDS)


class MCPSettingsDialog(QtWidgets.QDialog):
    """编辑 MCPSettings 副本，Apply / OK 时写回并应用到服务。"""

    def __init__(self, window, controller, parent=None):
        super().__init__(parent or window)
        self.window = window
        self.ctl = controller
        self.state = window.state
        self.tr = self.state.tr
        self.draft = copy.deepcopy(self.state.settings.mcp)
        self._loading = False
        self.setWindowTitle(self.tr.tr("mcp_settings"))
        self.resize(760, 560)
        self._build_ui()
        self._load_from_draft()
        self.ctl.logLine.connect(self._append_log, QtCore.Qt.QueuedConnection)
        self.ctl.statusChanged.connect(self._on_status,
                                       QtCore.Qt.QueuedConnection)
        self.ctl.requestCount.connect(self._on_request_count,
                                      QtCore.Qt.QueuedConnection)
        self._refresh_preview()
        self._refresh_status()

    # ==================================================================
    # 构建界面
    # ==================================================================
    def _build_ui(self) -> None:
        tr = self.tr
        root = QtWidgets.QVBoxLayout(self)

        # ---- 顶部：总开关 + 状态 + 启停 ----
        header = QtWidgets.QGroupBox(tr.tr("mcp_service"))
        hbox = QtWidgets.QHBoxLayout(header)
        self.enable_box = QtWidgets.QCheckBox(tr.tr("mcp_enable"))
        self.enable_box.stateChanged.connect(self._on_enable_toggled)
        hbox.addWidget(self.enable_box, 1)
        self.status_label = QtWidgets.QLabel("")
        hbox.addWidget(self.status_label)
        self.start_stop = QtWidgets.QPushButton(tr.tr("mcp_start"))
        self.start_stop.clicked.connect(self._toggle_service)
        hbox.addWidget(self.start_stop)
        root.addWidget(header)

        # ---- 左侧分类 + 右侧分页 ----
        middle = QtWidgets.QHBoxLayout()
        self.nav = QtWidgets.QListWidget()
        self.nav.setFixedWidth(170)
        self.nav.setSpacing(2)
        self.stack = QtWidgets.QStackedWidget()
        for key in ("mcp_page_service", "mcp_page_security",
                    "mcp_page_workspace", "mcp_page_client"):
            self.nav.addItem(tr.tr(key))
        self.nav.currentRowChanged.connect(self.stack.setCurrentIndex)
        middle.addWidget(self.nav)
        middle.addWidget(self.stack, 1)
        root.addLayout(middle, 1)

        self.stack.addWidget(self._page_service())
        self.stack.addWidget(self._page_security())
        self.stack.addWidget(self._page_workspace())
        self.stack.addWidget(self._page_client())
        self.nav.setCurrentRow(0)

        # ---- 底部按钮 ----
        buttons = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.Ok |
            QtWidgets.QDialogButtonBox.Cancel |
            QtWidgets.QDialogButtonBox.Apply |
            QtWidgets.QDialogButtonBox.RestoreDefaults)
        buttons.button(QtWidgets.QDialogButtonBox.Ok).setText(tr.tr("ok"))
        buttons.button(QtWidgets.QDialogButtonBox.Cancel).setText(tr.tr("cancel"))
        buttons.button(QtWidgets.QDialogButtonBox.Apply).setText(tr.tr("apply"))
        buttons.button(QtWidgets.QDialogButtonBox.RestoreDefaults).setText(
            tr.tr("mcp_restore"))
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        buttons.button(QtWidgets.QDialogButtonBox.Apply).clicked.connect(
            self._apply)
        buttons.button(QtWidgets.QDialogButtonBox.RestoreDefaults).clicked.connect(
            self._restore_defaults)
        root.addWidget(buttons)

    @staticmethod
    def _form() -> tuple:
        """返回 (page, form)；page 必须被 Python 持有，否则布局会被 GC 掉。"""
        page = QtWidgets.QWidget()
        form = QtWidgets.QFormLayout(page)
        form.setLabelAlignment(QtCore.Qt.AlignRight)
        return page, form

    # ---------------- 服务页 ----------------
    def _page_service(self) -> QtWidgets.QWidget:
        tr = self.tr
        page, form = self._form()

        self.autostart = QtWidgets.QCheckBox(tr.tr("mcp_autostart"))
        form.addRow(self.autostart)
        form.addRow(tr.tr("mcp_host"), QtWidgets.QLabel("127.0.0.1 (loopback)"))
        self.port = QtWidgets.QSpinBox()
        self.port.setRange(1024, 65535)
        self.port.valueChanged.connect(self._refresh_preview)
        form.addRow(tr.tr("mcp_port"), self.port)

        url_row = QtWidgets.QHBoxLayout()
        self.endpoint_edit = QtWidgets.QLineEdit()
        self.endpoint_edit.setReadOnly(True)
        copy_btn = QtWidgets.QPushButton(tr.tr("mcp_copy"))
        copy_btn.clicked.connect(lambda: self._copy(self.endpoint_edit.text()))
        url_row.addWidget(self.endpoint_edit, 1)
        url_row.addWidget(copy_btn)
        form.addRow(tr.tr("mcp_endpoint"), url_row)

        action_row = QtWidgets.QHBoxLayout()
        probe = QtWidgets.QPushButton(tr.tr("mcp_probe"))
        probe.clicked.connect(self._test_connection)
        action_row.addWidget(probe)
        self.requests_label = QtWidgets.QLabel("0")
        action_row.addWidget(QtWidgets.QLabel(tr.tr("mcp_requests")))
        action_row.addWidget(self.requests_label, 1)
        form.addRow(tr.tr("mcp_status"), action_row)
        self.last_label = QtWidgets.QLabel("-")
        form.addRow(tr.tr("mcp_last_request"), self.last_label)

        self.log_view = QtWidgets.QPlainTextEdit()
        self.log_view.setReadOnly(True)
        self.log_view.setMaximumBlockCount(300)
        self.log_view.setMinimumHeight(120)
        clear_log = QtWidgets.QPushButton(tr.tr("mcp_clear_log"))
        clear_log.clicked.connect(self._clear_log)
        log_box = QtWidgets.QVBoxLayout()
        log_box.addWidget(QtWidgets.QLabel(tr.tr("mcp_log")))
        log_box.addWidget(self.log_view)
        log_box.addWidget(clear_log, 0, QtCore.Qt.AlignRight)
        wrap = QtWidgets.QWidget()
        wrap.setLayout(log_box)
        form.addRow(wrap)
        return page

    # ---------------- 安全页 ----------------
    def _page_security(self) -> QtWidgets.QWidget:
        tr = self.tr
        page, form = self._form()

        self.require_token = QtWidgets.QCheckBox(tr.tr("mcp_require_token"))
        self.require_token.stateChanged.connect(self._refresh_preview)
        form.addRow(self.require_token)

        self.token_edit = QtWidgets.QLineEdit()
        self.token_edit.setReadOnly(True)
        self.token_edit.setEchoMode(QtWidgets.QLineEdit.Password)
        # 令牌字段右键菜单：生成 / 复制 / 显示（与 PrePoMax 设置页一致）
        self.token_edit.setContextMenuPolicy(QtCore.Qt.CustomContextMenu)
        self.token_edit.customContextMenuRequested.connect(self._token_menu)
        token_row = QtWidgets.QHBoxLayout()
        token_row.addWidget(self.token_edit, 1)
        gen = QtWidgets.QPushButton(tr.tr("mcp_generate_token"))
        gen.clicked.connect(self._generate_token)
        token_row.addWidget(gen)
        copy_t = QtWidgets.QPushButton(tr.tr("mcp_copy"))
        copy_t.clicked.connect(lambda: self._copy(self.token_edit.text()))
        token_row.addWidget(copy_t)
        form.addRow(tr.tr("mcp_token"), token_row)
        self.show_token = QtWidgets.QCheckBox(tr.tr("mcp_show_token"))
        self.show_token.stateChanged.connect(self._show_token_changed)
        hint = QtWidgets.QLabel(tr.tr("mcp_token_hint"))
        hint.setWordWrap(True)
        hint.setStyleSheet("color:#888;")
        form.addRow(self.show_token)
        form.addRow(hint)

        form.addRow(QtWidgets.QLabel(f"— {tr.tr('mcp_page_security')} —"))
        self.allow_scene = QtWidgets.QCheckBox(tr.tr("mcp_allow_scene"))
        self.allow_write = QtWidgets.QCheckBox(tr.tr("mcp_allow_write"))
        form.addRow(self.allow_scene)
        form.addRow(self.allow_write)

        form.addRow(QtWidgets.QLabel(tr.tr("mcp_tools")))
        self.tool_list = QtWidgets.QListWidget()
        self.tool_list.setMinimumHeight(150)
        for tool in TOOLS:
            item = QtWidgets.QListWidgetItem(
                f"{tool['name']}  —  {tool['description'][:34]}")
            item.setFlags(item.flags() | QtCore.Qt.ItemIsUserCheckable)
            item.setData(QtCore.Qt.UserRole, tool["name"])
            item.setCheckState(QtCore.Qt.Checked)
            self.tool_list.addItem(item)
        tool_buttons = QtWidgets.QHBoxLayout()
        for label, checked in ((tr.tr("mcp_select_all"), True),
                               (tr.tr("mcp_select_none"), False)):
            btn = QtWidgets.QPushButton(label)
            btn.clicked.connect(lambda _, c=checked: self._select_tools(c))
            tool_buttons.addWidget(btn)
        tool_buttons.addStretch(1)
        tools_wrap = QtWidgets.QVBoxLayout()
        tools_wrap.addWidget(self.tool_list)
        tools_wrap.addLayout(tool_buttons)
        tw = QtWidgets.QWidget()
        tw.setLayout(tools_wrap)
        form.addRow(tw)
        return page

    # ---------------- 工作区与默认值 ----------------
    def _page_workspace(self) -> QtWidgets.QWidget:
        tr = self.tr
        page, form = self._form()

        self.root_edit = QtWidgets.QLineEdit()
        browse = QtWidgets.QPushButton(tr.tr("mcp_browse"))
        browse.clicked.connect(self._browse_root)
        row = QtWidgets.QHBoxLayout()
        row.addWidget(self.root_edit, 1)
        row.addWidget(browse)
        form.addRow(tr.tr("mcp_workspace"), row)

        self.max_pixels = QtWidgets.QSpinBox()
        self.max_pixels.setRange(100_000, 20_000_000)
        self.max_pixels.setSingleStep(100_000)
        form.addRow(tr.tr("mcp_max_pixels"), self.max_pixels)

        self.default_units = QtWidgets.QComboBox()
        self.default_units.addItem("mm", "mm")
        self.default_units.addItem("inch", "inch")
        form.addRow(tr.tr("mcp_default_units"), self.default_units)

        self.default_mode = QtWidgets.QComboBox()
        self.default_mode.addItem(tr.tr("mcp_mode_view"), "view")
        self.default_mode.addItem(tr.tr("mcp_mode_3d"), "3d")
        form.addRow(tr.tr("mcp_default_mode"), self.default_mode)

        size_row = QtWidgets.QHBoxLayout()
        self.render_w = QtWidgets.QSpinBox()
        self.render_h = QtWidgets.QSpinBox()
        for box in (self.render_w, self.render_h):
            box.setRange(64, 2048)
            box.setSingleStep(16)
            size_row.addWidget(box)
        size_row.addStretch(1)
        form.addRow(tr.tr("mcp_render_size"), size_row)
        self.log_requests = QtWidgets.QCheckBox(tr.tr("mcp_log_requests"))
        form.addRow(self.log_requests)

        note = QtWidgets.QLabel(tr.tr("mcp_apply_hint"))
        note.setWordWrap(True)
        note.setStyleSheet("color:#888;")
        form.addRow(note)
        return page

    # ---------------- AI 客户端页 ----------------
    def _page_client(self) -> QtWidgets.QWidget:
        tr = self.tr
        page, form = self._form()

        self.config_path = QtWidgets.QLineEdit()
        self.config_path.editingFinished.connect(self._refresh_preview)
        browse = QtWidgets.QPushButton(tr.tr("mcp_browse"))
        browse.clicked.connect(self._browse_config)
        row = QtWidgets.QHBoxLayout()
        row.addWidget(self.config_path, 1)
        row.addWidget(browse)
        form.addRow(tr.tr("mcp_client_config"), row)

        self.preview = QtWidgets.QPlainTextEdit()
        self.preview.setReadOnly(True)
        self.preview.setMinimumHeight(140)
        form.addRow(tr.tr("mcp_config_preview"), self.preview)

        btns = QtWidgets.QGridLayout()
        labels = ("mcp_write_config", "mcp_copy", "mcp_remove_config",
                  "mcp_open_config")
        slots = (self._write_config,
                 lambda: self._copy(self.preview.toPlainText()),
                 self._remove_config, self._open_config)
        for i, (label, slot) in enumerate(zip(labels, slots)):
            btn = QtWidgets.QPushButton(tr.tr(label))
            btn.clicked.connect(slot)
            btns.addWidget(btn, i // 2, i % 2)
        form.addRow(btns)

        hint = QtWidgets.QLabel(tr.tr("mcp_client_hint"))
        hint.setWordWrap(True)
        hint.setStyleSheet("color:#888;")
        form.addRow(hint)
        return page

    # ==================================================================
    # 控件 <-> 配置
    # ==================================================================
    def _load_from_draft(self) -> None:
        # 逐个 setXxx 会触发 valueChanged / stateChanged，此时控件只填了一半，
        # 绝不能让预览回调反过来把 draft 覆盖掉。
        self._loading = True
        try:
            self._fill_from_draft()
        finally:
            self._loading = False

    def _fill_from_draft(self) -> None:
        d = self.draft
        self.enable_box.setChecked(bool(d.enabled))
        self.autostart.setChecked(bool(d.autostart))
        self.port.setValue(int(d.port))
        self.require_token.setChecked(bool(d.require_token))
        self.token_edit.setText(d.token)
        self.allow_scene.setChecked(bool(d.allow_scene_changes))
        self.allow_write.setChecked(bool(d.allow_file_write))
        self.root_edit.setText(d.workspace_root or d.resolved_root())
        self.max_pixels.setValue(int(d.max_image_pixels))
        idx = self.default_units.findData(d.default_units)
        self.default_units.setCurrentIndex(idx if idx >= 0 else 0)
        idx = self.default_mode.findData(d.default_measure_mode)
        self.default_mode.setCurrentIndex(idx if idx >= 0 else 0)
        self.render_w.setValue(int(d.render_width))
        self.render_h.setValue(int(d.render_height))
        self.log_requests.setChecked(bool(d.log_requests))
        self.config_path.setText(d.resolved_client_config_path())
        self.log_view.clear()
        for line in self.ctl.logs():
            self.log_view.appendPlainText(line)
        allowed = set(d.enabled_tools)
        for i in range(self.tool_list.count()):
            item = self.tool_list.item(i)
            name = item.data(QtCore.Qt.UserRole)
            item.setCheckState(QtCore.Qt.Checked if not allowed or name in allowed
                               else QtCore.Qt.Unchecked)

    def _collect(self) -> None:
        d = self.draft
        d.enabled = self.enable_box.isChecked()
        d.autostart = self.autostart.isChecked()
        d.port = self.port.value()
        d.require_token = self.require_token.isChecked()
        d.token = self.token_edit.text()
        if d.require_token and not d.token:
            # 预览 / 写入配置都会用到令牌，先在这里生成，保证服务与客户端一致
            d.token = generate_token()
            self.token_edit.setText(d.token)
        d.allow_scene_changes = self.allow_scene.isChecked()
        d.allow_file_write = self.allow_write.isChecked()
        d.workspace_root = self.root_edit.text().strip()
        d.max_image_pixels = self.max_pixels.value()
        d.default_units = self.default_units.currentData()
        d.default_measure_mode = self.default_mode.currentData()
        d.render_width = self.render_w.value()
        d.render_height = self.render_h.value()
        d.log_requests = self.log_requests.isChecked()
        d.client_config_path = self.config_path.text().strip()
        names = []
        for i in range(self.tool_list.count()):
            item = self.tool_list.item(i)
            if item.checkState() == QtCore.Qt.Checked:
                names.append(item.data(QtCore.Qt.UserRole))
        # 全选存空列表：以后新增工具默认可用
        d.enabled_tools = [] if len(names) == self.tool_list.count() else names

    # ==================================================================
    # 应用 / 启停
    # ==================================================================
    def _apply(self) -> bool:
        self._collect()
        try:
            self.draft.validate()
        except ValueError as exc:
            QtWidgets.QMessageBox.warning(self, self.tr.tr("mcp_settings"),
                                          str(exc))
            return False
        real = self.state.settings.mcp
        before = _signature(real)
        real.set_state(self.draft.state())
        try:
            self.state.save_settings()
        except OSError as exc:
            QtWidgets.QMessageBox.warning(self, self.tr.tr("mcp_settings"),
                                          str(exc))
            return False
        after = _signature(real)
        if before != after or (real.enabled != self.ctl.running):
            if real.enabled:
                ok, msg = self.ctl.restart()
                if not ok:
                    QtWidgets.QMessageBox.warning(
                        self, self.tr.tr("mcp_settings"), msg)
            else:
                self.ctl.stop()
        else:
            self.ctl.update_policy()
        self._refresh_preview()
        self._refresh_status()
        return True

    def accept(self) -> None:
        if self._apply():
            super().accept()

    def closeEvent(self, event) -> None:
        for signal, slot in ((self.ctl.logLine, self._append_log),
                             (self.ctl.statusChanged, self._on_status),
                             (self.ctl.requestCount, self._on_request_count)):
            try:
                signal.disconnect(slot)
            except (TypeError, RuntimeError):
                pass
        super().closeEvent(event)

    def _restore_defaults(self) -> None:
        fresh = MCPSettings()
        fresh.workspace_root = self.draft.workspace_root
        fresh.client_config_path = self.draft.client_config_path
        fresh.token = self.draft.token          # 令牌不在恢复默认值时清空
        self.draft = fresh
        self._load_from_draft()
        self._refresh_preview()
        self._show_status(self.tr.tr("mcp_restore"))

    def _toggle_service(self) -> None:
        self.enable_box.setChecked(not self.ctl.running)
        self._apply()

    def _on_enable_toggled(self, _state) -> None:
        if not self.enable_box.isChecked():
            self.start_stop.setText(self.tr.tr("mcp_start"))

    # ==================================================================
    # 按钮动作
    # ==================================================================
    def _generate_token(self) -> None:
        self.draft.token = generate_token()
        self.token_edit.setText(self.draft.token)
        self._refresh_preview()
        self._show_status(self.tr.tr("mcp_token_changed"))

    def _token_menu(self, pos) -> None:
        menu = QtWidgets.QMenu(self)
        acts = ((self.tr.tr("mcp_generate_token"), self._generate_token),
                (self.tr.tr("mcp_copy"),
                 lambda: self._copy(self.token_edit.text())),
                (self.tr.tr("mcp_show_token"),
                 lambda: self.show_token.setChecked(not self.show_token.isChecked())))
        for text, slot in acts:
            act = menu.addAction(text)
            act.triggered.connect(lambda _checked=False, s=slot: s())
        menu.exec_(self.token_edit.mapToGlobal(pos))

    def _show_token_changed(self, state) -> None:
        self.token_edit.setEchoMode(QtWidgets.QLineEdit.Normal if state
                                    else QtWidgets.QLineEdit.Password)

    def _select_tools(self, checked: bool) -> None:
        for i in range(self.tool_list.count()):
            self.tool_list.item(i).setCheckState(
                QtCore.Qt.Checked if checked else QtCore.Qt.Unchecked)

    def _browse_root(self) -> None:
        path = QtWidgets.QFileDialog.getExistingDirectory(
            self, self.tr.tr("mcp_workspace"),
            self.root_edit.text() or os.getcwd())
        if path:
            self.root_edit.setText(path)

    def _browse_config(self) -> None:
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, self.tr.tr("mcp_client_config"),
            self.config_path.text(), "JSON (*.json);;All files (*)")
        if path:
            self.config_path.setText(path)

    def _write_config(self) -> None:
        if not self._apply():
            return
        cfg = self.state.settings.mcp
        path = cfg.resolved_client_config_path()
        try:
            write_client_config(path, cfg)
            self.state.save_settings()
        except (ValueError, OSError) as exc:
            QtWidgets.QMessageBox.warning(self, self.tr.tr("mcp_settings"),
                                          str(exc))
            return
        self._refresh_preview()
        QtWidgets.QMessageBox.information(
            self, self.tr.tr("mcp_settings"),
            self.tr.tr("mcp_config_written", path))

    def _remove_config(self) -> None:
        path = self.config_path.text().strip() or \
            self.draft.resolved_client_config_path()
        try:
            removed = remove_client_config(os.path.abspath(
                os.path.expanduser(path)))
        except ValueError as exc:
            QtWidgets.QMessageBox.warning(self, self.tr.tr("mcp_settings"),
                                          str(exc))
            return
        self._show_status(self.tr.tr("mcp_config_removed") if removed
                          else self.tr.tr("mcp_config_absent"))

    def _open_config(self) -> None:
        path = os.path.abspath(os.path.expanduser(
            self.config_path.text().strip() or
            self.draft.resolved_client_config_path()))
        if os.path.isfile(path):
            QtGui.QDesktopServices.openUrl(QtCore.QUrl.fromLocalFile(path))
        else:
            QtWidgets.QMessageBox.information(
                self, self.tr.tr("mcp_settings"), path)

    def _test_connection(self) -> None:
        ok, text = self.ctl.probe()
        self._show_status(f"{self.tr.tr('mcp_probe')}: "
                          f"{'OK' if ok else 'FAIL'} — {text}")
        (QtWidgets.QMessageBox.information if ok else QtWidgets.QMessageBox.warning)(
            self, self.tr.tr("mcp_settings"), text)

    def _clear_log(self) -> None:
        self.ctl.clear_logs()
        self.log_view.clear()

    @staticmethod
    def _copy(text: str) -> None:
        QtWidgets.QApplication.clipboard().setText(text)

    def _show_status(self, text: str) -> None:
        self.window.statusBar().showMessage(text, 5000)

    # ==================================================================
    # 状态刷新
    # ==================================================================
    def _refresh_preview(self) -> None:
        if self._loading:
            return
        self._collect()
        self.draft.client_config_path = self.config_path.text().strip()
        self.preview.setPlainText(client_config_preview(self.draft))
        # 预览会自动补出令牌，同步回输入框
        self.token_edit.setText(self.draft.token)

    def _refresh_status(self) -> None:
        st = self.ctl.status()
        running = self.ctl.running
        label = self.tr.tr("mcp_running" if running else "mcp_stopped")
        self.status_label.setText(("● " if running else "○ ") + label)
        self.status_label.setStyleSheet(
            f"color:{'#2e9e5b' if running else '#999'};font-weight:bold;")
        self.start_stop.setText(self.tr.tr("mcp_stop" if running else "mcp_start"))
        self.endpoint_edit.setText(st["endpoint"] if running
                                   else self.draft.endpoint)
        self.requests_label.setText(str(st["requests"]))
        self.last_label.setText(st["last_request"] or "-")

    def _append_log(self, line: str) -> None:
        self.log_view.appendPlainText(line)

    def _on_status(self, _running: bool, _endpoint: str) -> None:
        self._refresh_status()

    def _on_request_count(self, count: int) -> None:
        self.requests_label.setText(str(count))
        st = self.ctl.status()
        self.last_label.setText(st["last_request"] or "-")
