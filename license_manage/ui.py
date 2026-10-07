from PyQt5.QtWidgets import QWidget, QLabel, QLineEdit, QPushButton, QVBoxLayout, QHBoxLayout, QDateTimeEdit, QTextEdit, QTableWidget, QTableWidgetItem, QMessageBox, QComboBox
from PyQt5.QtCore import QDateTime
import time
from .hardware import get_cpu_fingerprint
from .license_crypto import generate_license, verify_license
from .storage import init_db, add_record, list_records
from .integration import SECRET

class LicenseManager(QWidget):
    def __init__(self):
        super().__init__()
        init_db()
        self.setWindowTitle("License Manager")
        self.cpu_label = QLabel("CPU 指纹")
        self.cpu_input = QLineEdit()
        self.cpu_input.setText(get_cpu_fingerprint())
        copy_btn = QPushButton("复制指纹")
        copy_btn.clicked.connect(lambda: self.cpu_input.selectAll() or self.cpu_input.copy())
        self.secret_label = QLabel("密钥")
        self.secret_input = QLineEdit()
        self.secret_input.setEchoMode(QLineEdit.Password)
        self.secret_input.setText(SECRET)
        self.secret_input.setReadOnly(True)

        self.duration_label = QLabel("授权时长(天)")
        self.duration_combo = QComboBox()
        self.duration_combo.addItems(["30", "90", "180", "365", "自定义"])
        self.duration_combo.setCurrentText("180")
        self.custom_days_input = QLineEdit()
        self.custom_days_input.setPlaceholderText("输入天数")
        self.custom_days_input.setEnabled(False)
        self.custom_days_input.setFixedWidth(80)

        self.exp_label = QLabel("授权截止")
        self.exp_input = QDateTimeEdit()
        self.exp_input.setCalendarPopup(True)
        self._set_exp_by_days(180)

        self.duration_combo.currentTextChanged.connect(self.on_duration_changed)
        self.custom_days_input.textChanged.connect(self.on_custom_days_changed)

        self.gen_btn = QPushButton("生成 License")
        self.gen_btn.clicked.connect(self.on_generate)
        self.output = QTextEdit()
        self.output.setReadOnly(True)
        self.table = QTableWidget()
        self.table.setColumnCount(4)
        self.table.setHorizontalHeaderLabels(["ID", "CPU", "Expires", "Created"])
        v = QVBoxLayout()
        row1 = QHBoxLayout()
        row1.addWidget(self.cpu_label)
        row1.addWidget(self.cpu_input)
        row1.addWidget(copy_btn)
        v.addLayout(row1)
        v.addWidget(self.secret_label)
        v.addWidget(self.secret_input)

        dur_row = QHBoxLayout()
        dur_row.addWidget(self.duration_label)
        dur_row.addWidget(self.duration_combo)
        dur_row.addWidget(QLabel("自定义"))
        dur_row.addWidget(self.custom_days_input)
        dur_row.addStretch()
        v.addLayout(dur_row)

        v.addWidget(self.exp_label)
        v.addWidget(self.exp_input)
        v.addWidget(self.gen_btn)
        v.addWidget(QLabel("License"))
        v.addWidget(self.output)
        v.addWidget(QLabel("授权记录"))
        v.addWidget(self.table)
        self.setLayout(v)
        self.refresh_table()

    def _set_exp_by_days(self, days: int):
        try:
            d = int(days)
        except Exception:
            return
        if d <= 0:
            return
        exp_ts = int(time.time()) + d * 86400
        self.exp_input.setDateTime(QDateTime.fromSecsSinceEpoch(exp_ts))

    def on_duration_changed(self, text: str):
        if text == "自定义":
            self.custom_days_input.setEnabled(True)
            if self.custom_days_input.text().strip().isdigit():
                self._set_exp_by_days(int(self.custom_days_input.text().strip()))
            return
        self.custom_days_input.setEnabled(False)
        self.custom_days_input.setText("")
        if text.strip().isdigit():
            self._set_exp_by_days(int(text.strip()))

    def on_custom_days_changed(self, text: str):
        if self.duration_combo.currentText() != "自定义":
            return
        s = (text or "").strip()
        if not s:
            return
        if not s.isdigit():
            return
        self._set_exp_by_days(int(s))

    def on_generate(self):
        cpu = self.cpu_input.text().strip()
        secret = self.secret_input.text().strip()
        exp_ts = int(self.exp_input.dateTime().toSecsSinceEpoch())
        if not cpu:
            QMessageBox.warning(self, "提示", "CPU 指纹为空")
            return
        if not secret:
            QMessageBox.warning(self, "提示", "请填写密钥")
            return
        if exp_ts <= int(time.time()):
            QMessageBox.warning(self, "提示", "授权截止时间必须晚于当前时间")
            return
        token = generate_license(cpu, exp_ts, secret)
        ok, payload = verify_license(token, secret)
        remaining_days = max(0, (int(payload.get("exp", 0)) - int(time.time())) // 86400) if payload else 0
        exp_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(exp_ts))
        self.output.setPlainText(f"{token}\n\n授权截止: {exp_str}\n授权剩余(估算): {remaining_days} 天")
        add_record(cpu, exp_ts, token)
        self.refresh_table()
        if ok:
            QMessageBox.information(self, "成功", f"已生成 License\n授权截止: {exp_str}")
        else:
            QMessageBox.warning(self, "提示", "签名验证失败或已过期")

    def refresh_table(self):
        rows = list_records(100)
        self.table.setRowCount(len(rows))
        for i, (rid, cpu, exp, created, lic) in enumerate(rows):
            self.table.setItem(i, 0, QTableWidgetItem(str(rid)))
            self.table.setItem(i, 1, QTableWidgetItem(cpu))
            self.table.setItem(i, 2, QTableWidgetItem(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(exp))))
            self.table.setItem(i, 3, QTableWidgetItem(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(created))))
