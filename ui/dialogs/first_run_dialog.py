#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
首次运行初始化面板 (First-run setup dialog)

在程序首次运行时弹出，向用户展示并引导完成：
  1. 模型权重下载（DeepSeek-OCR / 文本嵌入 / SAM）；
  2. 本地 Ollama 运行时检测；
  3. 外部 OpenAI 兼容 API Key 配置与连通性测试。
"""
from __future__ import annotations

from typing import List, Dict, Optional

from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtWidgets import (
    QAbstractItemView, QDialog, QFormLayout, QGroupBox, QHBoxLayout, QHeaderView,
    QLabel, QLineEdit, QMessageBox, QProgressBar, QPushButton, QTableWidget,
    QTableWidgetItem, QTextEdit, QVBoxLayout, QWidget,
)

from src.services.setup_service import SetupService, MODEL_SPECS


def _fmt_bytes(v) -> str:
    try:
        val = float(v)
        for unit in ("B", "KB", "MB", "GB", "TB"):
            if val < 1024:
                return f"{val:.1f} {unit}"
            val /= 1024
        return f"{val:.1f} PB"
    except Exception:
        return "-"


class DownloadWorker(QThread):
    progress = pyqtSignal(str, int, int)   # rid, done, total
    log = pyqtSignal(str)
    one_done = pyqtSignal(str, bool)       # rid, ok
    all_done = pyqtSignal()

    def __init__(self, service: SetupService, specs: List[Dict], parent=None):
        super().__init__(parent)
        self.service = service
        self.specs = specs
        self._cancel = False

    def cancel(self):
        self._cancel = True

    def run(self):
        for spec in self.specs:
            if self._cancel:
                self.log.emit("已取消下载")
                break
            ok = False
            try:
                ok = self.service.download_resource(
                    spec,
                    progress_cb=lambda rid, d, t: self.progress.emit(rid, d, t),
                    log_cb=lambda m: self.log.emit(m),
                    cancel=lambda: self._cancel,
                )
            except Exception as e:
                self.log.emit(f"下载异常: {e}")
            self.one_done.emit(spec["id"], bool(ok))
        self.all_done.emit()


class FirstRunDialog(QDialog):
    def __init__(self, service: Optional[SetupService] = None, parent=None):
        super().__init__(parent)
        self.service = service or SetupService()
        self.worker = None
        self._row_of = {}
        self.setWindowTitle("首次运行初始化 · 环境配置")
        self.setMinimumSize(860, 720)
        self._build_ui()
        self.refresh()

    # ------------------------------------------------------------------ UI
    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setSpacing(12)

        title = QLabel("欢迎使用 医疗数据科学结构化工作流系统")
        title.setStyleSheet("font-size:18px;font-weight:700;color:#2E86AB;")
        root.addWidget(title)
        tip = QLabel(
            "首次运行请完成以下配置。模型权重体积较大，可稍后在本面板（菜单：帮助 → 环境初始化）再次下载。"
        )
        tip.setWordWrap(True)
        tip.setStyleSheet("color:#666;")
        root.addWidget(tip)

        # ① 模型权重 ------------------------------------------------------
        gb_models = QGroupBox("① 模型权重 / Model Weights")
        ml = QVBoxLayout(gb_models)
        self.table = QTableWidget(0, 4)
        self.table.setHorizontalHeaderLabels(["资源", "状态", "大小", "操作"])
        self.table.verticalHeader().setVisible(False)
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.table.setSelectionMode(QAbstractItemView.NoSelection)
        hh = self.table.horizontalHeader()
        hh.setSectionResizeMode(0, QHeaderView.Stretch)
        hh.setSectionResizeMode(1, QHeaderView.ResizeToContents)
        hh.setSectionResizeMode(2, QHeaderView.ResizeToContents)
        hh.setSectionResizeMode(3, QHeaderView.ResizeToContents)
        self.table.setMinimumHeight(190)
        ml.addWidget(self.table)

        row_btns = QHBoxLayout()
        self.btn_dl_missing = QPushButton("一键下载缺失项")
        self.btn_dl_missing.clicked.connect(self.download_missing)
        self.btn_refresh_models = QPushButton("重新检测")
        self.btn_refresh_models.clicked.connect(self.refresh_models)
        row_btns.addWidget(self.btn_dl_missing)
        row_btns.addWidget(self.btn_refresh_models)
        row_btns.addStretch()
        ml.addLayout(row_btns)

        self.progress = QProgressBar()
        self.progress.setValue(0)
        self.progress.setFormat("%p%")
        ml.addWidget(self.progress)
        root.addWidget(gb_models)

        # ② 推理后端 ------------------------------------------------------
        gb_backend = QGroupBox("② 推理后端 / LLM Backend")
        bl = QVBoxLayout(gb_backend)

        ollama_row = QHBoxLayout()
        self.lbl_ollama = QLabel("检测中…")
        self.lbl_ollama.setWordWrap(True)
        btn_recheck = QPushButton("重新检测 Ollama")
        btn_recheck.clicked.connect(self.refresh_ollama)
        ollama_row.addWidget(self.lbl_ollama, 1)
        ollama_row.addWidget(btn_recheck)
        bl.addLayout(ollama_row)

        form = QFormLayout()
        self.in_name = QLineEdit("deepseek")
        self.in_url = QLineEdit("https://api.deepseek.com")
        self.in_key = QLineEdit()
        self.in_key.setEchoMode(QLineEdit.Password)
        self.in_model = QLineEdit("deepseek-chat")
        form.addRow("配置名称", self.in_name)
        form.addRow("Base URL", self.in_url)
        form.addRow("API Key", self.in_key)
        form.addRow("模型名称", self.in_model)
        bl.addLayout(form)

        api_row = QHBoxLayout()
        self.btn_test = QPushButton("保存并测试")
        self.btn_test.clicked.connect(self.save_and_test_api)
        self.lbl_api = QLabel("")
        self.lbl_api.setWordWrap(True)
        api_row.addWidget(self.btn_test)
        api_row.addWidget(self.lbl_api, 1)
        bl.addLayout(api_row)
        root.addWidget(gb_backend)

        # 日志 -----------------------------------------------------------
        self.log = QTextEdit()
        self.log.setReadOnly(True)
        self.log.setMinimumHeight(110)
        self.log.setStyleSheet("font-family:Consolas,monospace;font-size:11px;")
        root.addWidget(self.log)

        # 底部按钮 -------------------------------------------------------
        foot = QHBoxLayout()
        foot.addStretch()
        self.btn_done = QPushButton("完成")
        self.btn_done.setDefault(True)
        self.btn_done.clicked.connect(self.on_done)
        self.btn_later = QPushButton("稍后")
        self.btn_later.clicked.connect(self.reject)
        foot.addWidget(self.btn_later)
        foot.addWidget(self.btn_done)
        root.addLayout(foot)

    # ----------------------------------------------------------------- 刷新
    def refresh(self):
        self.refresh_models()
        self.refresh_ollama()
        self.refresh_api()

    def refresh_models(self):
        statuses = {m["id"]: m for m in self.service.all_model_statuses()}
        self.table.setRowCount(0)
        self._row_of = {}
        for spec in MODEL_SPECS:
            st = statuses.get(spec["id"], {})
            r = self.table.rowCount()
            self.table.insertRow(r)
            self._row_of[spec["id"]] = r

            name = QTableWidgetItem(f"{spec['name']}\n{spec.get('desc','')}")
            self.table.setItem(r, 0, name)
            present = bool(st.get("present"))
            sit = QTableWidgetItem("✅ 已就绪" if present else "⬇️ 缺失")
            sit.setForeground(Qt.darkGreen if present else Qt.darkYellow)
            self.table.setItem(r, 1, sit)
            self.table.setItem(r, 2, QTableWidgetItem(spec.get("size_label", "")))

            btn = QPushButton("重新下载" if present else "下载")
            btn.clicked.connect(lambda _=False, rid=spec["id"]: self.download_one(rid))
            self.table.setCellWidget(r, 3, btn)
        self.table.resizeRowsToContents()

    def refresh_ollama(self):
        st = self.service.ollama_status()
        if st["available"]:
            self.lbl_ollama.setText(
                f"✅ Ollama 运行中（{st['host']}），模型 {len(st['models'])} 个："
                + ", ".join(st["models"][:8])
            )
        elif st["running"]:
            self.lbl_ollama.setText(
                f"⚠️ Ollama 已运行（{st['host']}）但未检测到模型，请执行：ollama pull qwen2.5:7b"
            )
        elif st["installed"]:
            self.lbl_ollama.setText(
                f"⚠️ 已安装 Ollama（{st['binary']}）但服务未启动，请先运行：ollama serve"
            )
        else:
            self.lbl_ollama.setText(
                "⚠️ 未检测到 Ollama。可安装后执行 `ollama pull qwen2.5:7b`，"
                "或在下方配置外部 API。"
            )

    def refresh_api(self):
        st = self.service.api_status()
        if st["configured"]:
            names = "、".join(st["names"]) if st["names"] else "(env)"
            self.lbl_api.setText(f"✅ 已配置 {st['valid_count']} 个可用 API：{names}")
        else:
            self.lbl_api.setText("⚠️ 尚未配置可用 API Key")

    # ----------------------------------------------------------------- 下载
    def _start_worker(self, specs: List[Dict]):
        if self.worker is not None and self.worker.isRunning():
            QMessageBox.information(self, "提示", "已有下载任务在进行中。")
            return
        if not specs:
            QMessageBox.information(self, "提示", "没有需要下载的资源。")
            return
        self.progress.setValue(0)
        self.btn_dl_missing.setEnabled(False)
        self.worker = DownloadWorker(self.service, specs, self)
        self.worker.log.connect(self._append_log)
        self.worker.progress.connect(self._on_progress)
        self.worker.one_done.connect(self._on_one_done)
        self.worker.all_done.connect(self._on_all_done)
        self.worker.start()

    def download_one(self, rid: str):
        spec = self.service.spec_by_id(rid)
        if spec:
            self._start_worker([spec])

    def download_missing(self):
        specs = [
            s for s in MODEL_SPECS
            if not self.service.model_status(s)["present"]
        ]
        self._start_worker(specs)

    def _append_log(self, msg: str):
        self.log.append(msg)

    def _on_progress(self, rid: str, done: int, total: int):
        if total > 0:
            self.progress.setValue(int(done * 100 / total))
            self.progress.setFormat(f"{_fmt_bytes(done)} / {_fmt_bytes(total)}  (%p%)")
        else:
            self.progress.setFormat(f"{_fmt_bytes(done)}")

    def _on_one_done(self, rid: str, ok: bool):
        self.refresh_models()

    def _on_all_done(self):
        self.btn_dl_missing.setEnabled(True)
        self.progress.setValue(100 if self.progress.value() > 0 else 0)
        self._append_log("全部任务结束。")
        self.refresh_models()

    # ----------------------------------------------------------------- API
    def save_and_test_api(self):
        name = self.in_name.text().strip()
        base = self.in_url.text().strip()
        key = self.in_key.text().strip()
        model = self.in_model.text().strip()
        if not name or not base or not model:
            QMessageBox.warning(self, "缺少必填项", "配置名称 / Base URL / 模型名称为必填。")
            return
        profile = {
            "name": name, "base_url": base, "api_key": key, "model": model,
            "mode": "chat", "temperature": 0.7, "max_tokens": 4096,
            "top_p": 1.0, "api_type": "custom",
        }
        if not self.service.save_api_profile(profile):
            QMessageBox.critical(self, "保存失败", "写入 API 配置失败，请检查权限。")
            return
        ok, msg = self.service.test_api_profile(profile)
        self._append_log(f"API 测试：{'成功' if ok else '失败'} - {msg}")
        if ok:
            self.lbl_api.setText(f"✅ 已保存并连通：{name}（{msg}）")
        else:
            self.lbl_api.setText(f"⚠️ 已保存，但连通性测试失败：{msg}")
        self.refresh_api()

    # ----------------------------------------------------------------- 完成
    def on_done(self):
        self.service.mark_setup_completed()
        self.accept()

    def closeEvent(self, event):
        if self.worker is not None and self.worker.isRunning():
            self.worker.cancel()
            self.worker.wait(3000)
        super().closeEvent(event)
