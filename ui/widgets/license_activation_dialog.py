from PyQt5.QtWidgets import QDialog, QVBoxLayout, QLabel, QLineEdit, QPushButton, QHBoxLayout, QMessageBox, QApplication
from PyQt5.QtCore import Qt
import sys
import os

# 最强健的寻址方式：当被引入时，通过检查 os.getcwd() 以及 sys.path 的组合
# 寻找真实的 license_manage 所在目录
def _inject_license_manage_path():
    candidates = [
        os.getcwd(),
        os.path.abspath(os.path.dirname(sys.argv[0])),
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    ]
    for c in candidates:
        if os.path.exists(os.path.join(c, "license_manage", "__init__.py")):
            if c not in sys.path:
                sys.path.insert(0, c)
            return

_inject_license_manage_path()

from license_manage.hardware import get_cpu_fingerprint
from license_manage.license_crypto import verify_license, b64d
from license_manage.integration import get_license_paths, SECRET
import time
from src.core.ui_scale_profile import dialog_target, is_macos

class LicenseActivationDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("授权到期")
        self.setModal(True)
        self._apply_dialog_size(520, 260, 460, 240)
        layout = QVBoxLayout(self)
        layout.addWidget(QLabel("请联系开发者: Christ.paul90@gamil.com"))
        self.fingerprint = get_cpu_fingerprint()
        fp_lbl = QLabel(f"硬件指纹: {self.fingerprint}")
        fp_lbl.setTextInteractionFlags(Qt.TextSelectableByMouse)
        layout.addWidget(fp_lbl)
        layout.addWidget(QLabel("请输入激活码:"))
        self.input = QLineEdit()
        self.input.setPlaceholderText("粘贴激活码")
        layout.addWidget(self.input)
        self.info = QLabel("")
        self.info.setWordWrap(True)
        layout.addWidget(self.info)
        btn_row = QHBoxLayout()
        self.btn_ok = QPushButton("激活并继续")
        self.btn_ok.clicked.connect(self.on_activate)
        self.btn_mgr = QPushButton("打开授权管理器")
        self.btn_mgr.clicked.connect(self.open_manager)
        self.btn_cancel = QPushButton("取消")
        self.btn_cancel.clicked.connect(self.reject)
        btn_row.addStretch()
        btn_row.addWidget(self.btn_mgr)
        btn_row.addWidget(self.btn_cancel)
        btn_row.addWidget(self.btn_ok)
        layout.addLayout(btn_row)
        self.remaining_days = 0
        self.token_written = False

    def _apply_dialog_size(self, base_w: int, base_h: int, min_w: int, min_h: int):
        if not is_macos():
            self.resize(base_w, base_h)
            return
        app = QApplication.instance()
        screen = self.screen() or (app.primaryScreen() if app else None)
        if not screen:
            self.resize(base_w, base_h)
            return
        geometry = screen.availableGeometry()
        target_w, target_h, effective_min_w, effective_min_h = dialog_target(
            base_w, base_h, min_w, min_h, geometry.width(), geometry.height()
        )
        self.setMinimumSize(effective_min_w, effective_min_h)
        self.resize(target_w, target_h)

    def on_activate(self):
        token = self.input.text().strip()
        ok, payload = verify_license(token, SECRET)
        reason = ""
        try:
            data = token.split(".", 1)[0]
            pl = b64d(data)
            plj = {}
            try:
                import json
                plj = json.loads(pl)
            except Exception:
                plj = {}
            cpu_in_token = plj.get("cpu")
            exp_ts = int(plj.get("exp", 0))
            if cpu_in_token and cpu_in_token != self.fingerprint:
                reason = "硬件指纹不匹配"
            elif exp_ts and exp_ts < int(time.time()):
                reason = "授权已过期"
            if exp_ts:
                exp_str = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(exp_ts))
                remain_days = max(0, (exp_ts - int(time.time())) // 86400)
                self.info.setText(f"授权截止: {exp_str}（剩余约 {remain_days} 天）")
        except Exception:
            pass
        if not ok or reason:
            QMessageBox.warning(self, "激活失败", reason or "激活码无效或已过期")
            return
        for p in get_license_paths():
            try:
                os.makedirs(os.path.dirname(p), exist_ok=True)
                with open(p, "w", encoding="utf-8") as f:
                    f.write(token)
                self.token_written = True
            except Exception:
                pass
        exp = int(payload.get("exp", 0))
        now = int(time.time())
        self.remaining_days = max(0, (exp - now) // 86400)
        try:
            exp_str = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(exp))
            QMessageBox.information(self, "激活成功", f"授权截止: {exp_str}\n剩余约: {self.remaining_days} 天")
        except Exception:
            pass
        self.accept()

    def open_manager(self):
        try:
            import subprocess, sys, os
            # Prefer running bundled python if available
            python_exe = sys.executable
            base = os.path.dirname(python_exe)
            cmd = [python_exe, "-m", "license_manage.main"]
            subprocess.Popen(cmd, creationflags=0x08000000 if sys.platform == "win32" else 0)
        except Exception:
            pass
