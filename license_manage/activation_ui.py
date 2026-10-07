from PyQt5.QtWidgets import QDialog, QLabel, QLineEdit, QPushButton, QVBoxLayout, QHBoxLayout, QMessageBox, QApplication
from PyQt5.QtCore import Qt
import time
import sys
import sys
from .hardware import get_cpu_fingerprint
from .license_crypto import verify_license, b64d
from .integration import get_license_paths, SECRET
from src.core.ui_scale_profile import dialog_target, is_macos
from src.core.ui_scale_profile import dialog_target, is_macos

class LicenseActivationDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("产品激活 - Medical Imaging Workflow")
        self._apply_dialog_size(500, 300, 460, 280)
        self.setWindowFlags(self.windowFlags() & ~Qt.WindowContextHelpButtonHint)
        
        layout = QVBoxLayout()
        
        # 说明
        layout.addWidget(QLabel("本产品需要激活才能使用。请将下方的机器码发送给管理员获取授权码。"))
        
        # 机器码显示
        layout.addWidget(QLabel("机器码 (Machine ID):"))
        self.cpu_fingerprint = get_cpu_fingerprint()
        
        hbox_cpu = QHBoxLayout()
        self.cpu_edit = QLineEdit(self.cpu_fingerprint)
        self.cpu_edit.setReadOnly(True)
        hbox_cpu.addWidget(self.cpu_edit)
        
        copy_btn = QPushButton("复制")
        copy_btn.clicked.connect(self.copy_cpu)
        hbox_cpu.addWidget(copy_btn)
        layout.addLayout(hbox_cpu)
        
        # 授权码输入
        layout.addWidget(QLabel("请输入授权码 (License Key):"))
        self.license_input = QLineEdit()
        self.license_input.setPlaceholderText("在此粘贴授权码...")
        layout.addWidget(self.license_input)

        self.exp_hint = QLabel("")
        self.exp_hint.setWordWrap(True)
        layout.addWidget(self.exp_hint)
        self.license_input.textChanged.connect(self._update_preview)
        
        # 按钮
        btn_layout = QHBoxLayout()
        self.activate_btn = QPushButton("激活")
        self.activate_btn.clicked.connect(self.activate)
        self.cancel_btn = QPushButton("退出")
        self.cancel_btn.clicked.connect(self.reject)
        
        btn_layout.addStretch()
        btn_layout.addWidget(self.activate_btn)
        btn_layout.addWidget(self.cancel_btn)
        layout.addLayout(btn_layout)
        
        self.setLayout(layout)

    def _apply_dialog_size(self, base_w: int, base_h: int, min_w: int, min_h: int):
        if not is_macos():
            self.setFixedSize(base_w, base_h)
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

    def _apply_dialog_size(self, base_w: int, base_h: int, min_w: int, min_h: int):
        if not is_macos():
            self.setFixedSize(base_w, base_h)
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
        
    def _update_preview(self):
        token = (self.license_input.text() or "").strip()
        if not token or "." not in token:
            self.exp_hint.setText("")
            return
        try:
            data = token.split(".", 1)[0]
            payload_raw = b64d(data)
            import json
            payload = json.loads(payload_raw)
            exp_ts = int(payload.get("exp", 0))
            if exp_ts:
                exp_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(exp_ts))
                remain_days = max(0, (exp_ts - int(time.time())) // 86400)
                self.exp_hint.setText(f"授权截止: {exp_str}（剩余约 {remain_days} 天）")
            else:
                self.exp_hint.setText("")
        except Exception:
            self.exp_hint.setText("")

    def copy_cpu(self):
        self.cpu_edit.selectAll()
        self.cpu_edit.copy()
        
    def activate(self):
        token = self.license_input.text().strip()
        if not token:
            QMessageBox.warning(self, "错误", "请输入授权码")
            return
            
        # 验证
        is_valid, payload = verify_license(token, SECRET)
        if is_valid:
            # 检查机器码是否匹配 (payload中包含机器码)
            # 假设 payload 格式中包含 'cpu' 字段，这取决于 generate_license 的实现
            # 如果 verify_license 内部已经检查了 payload['cpu'] == get_cpu_fingerprint()，那就够了
            # 但通常 verify_license 只验证签名。我们需要看看 license_crypto.py
            
            # 再次确认 machine id
            if payload.get('cpu') == self.cpu_fingerprint:
                # 保存
                self.save_license(token)
                exp_ts = int(payload.get("exp", 0) or 0)
                exp_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(exp_ts)) if exp_ts else "未知"
                remain_days = max(0, (exp_ts - int(time.time())) // 86400) if exp_ts else 0
                QMessageBox.information(self, "成功", f"激活成功！\n授权截止: {exp_str}\n剩余约: {remain_days} 天")
                self.accept()
            else:
                 QMessageBox.critical(self, "失败", "授权码与当前机器不匹配！")
        else:
            QMessageBox.critical(self, "失败", "授权码无效或已过期！")
            
    def save_license(self, token):
        paths = get_license_paths()
        success = False
        for p in paths:
            try:
                p.parent.mkdir(parents=True, exist_ok=True)
                p.write_text(token, encoding='utf-8')
                success = True
            except Exception as e:
                print(f"Failed to write license to {p}: {e}")
