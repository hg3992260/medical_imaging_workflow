from PyQt5.QtCore import QEvent, QTimer, Qt
from PyQt5.QtWidgets import QFrame, QLabel, QVBoxLayout

from src.utils.gpu_manager import gpu_manager


def _owner_text() -> str:
    label = gpu_manager.get_current_owner_label()
    return f"当前 GPU 持有者: {label}"


def is_loading_message(message: str) -> bool:
    text = str(message or "")
    keys = [
        "正在加载到显存",
        "协调显存资源",
        "模型加载阶段",
        "开始预热",
        "正在准备 DeepSeek-OCR 模型",
    ]
    return any(k in text for k in keys)


def is_ready_message(message: str) -> bool:
    text = str(message or "")
    keys = [
        "已加载完成并可用",
        "模型已可用",
        "开始执行 OCR",
        "开始 OCR 识别",
        "开始识别。",
        "开始识别",
    ]
    return any(k in text for k in keys)


def is_error_message(message: str) -> bool:
    text = str(message or "")
    keys = [
        "加载失败",
        "等待超时",
        "当前不可用",
        "未就绪",
        "OCR识别失败",
        "启动分析线程失败",
    ]
    return any(k in text for k in keys)


class ModelLoadingOverlay(QFrame):
    def __init__(self, parent):
        super().__init__(parent)
        self._started_at_ms = 0
        self._timeout_hint_ms = 25000
        self._hide_timer = QTimer(self)
        self._hide_timer.setSingleShot(True)
        self._hide_timer.timeout.connect(self.hide)
        self._poll_timer = QTimer(self)
        self._poll_timer.setInterval(1000)
        self._poll_timer.timeout.connect(self._on_tick)
        self._build_ui()
        self.hide()
        parent.installEventFilter(self)
        self._sync_geometry()

    def _build_ui(self):
        self.setObjectName("modelLoadingOverlay")
        self.setStyleSheet(
            """
            QFrame#modelLoadingOverlay {
                background-color: rgba(20, 20, 20, 150);
                border: 1px solid rgba(255, 255, 255, 40);
                border-radius: 10px;
            }
            QLabel {
                color: white;
                background: transparent;
            }
            """
        )
        layout = QVBoxLayout(self)
        layout.setContentsMargins(18, 14, 18, 14)
        layout.setSpacing(6)

        self.title_label = QLabel("模型正在加载")
        self.title_label.setAlignment(Qt.AlignCenter)
        self.title_label.setStyleSheet("font-size: 15px; font-weight: bold; color: #FDE68A;")
        layout.addWidget(self.title_label)

        self.message_label = QLabel("请稍候…")
        self.message_label.setWordWrap(True)
        self.message_label.setAlignment(Qt.AlignCenter)
        self.message_label.setStyleSheet("font-size: 13px;")
        layout.addWidget(self.message_label)

        self.owner_label = QLabel(_owner_text())
        self.owner_label.setAlignment(Qt.AlignCenter)
        self.owner_label.setStyleSheet("font-size: 12px; color: #BFDBFE;")
        layout.addWidget(self.owner_label)

        self.hint_label = QLabel("")
        self.hint_label.setWordWrap(True)
        self.hint_label.setAlignment(Qt.AlignCenter)
        self.hint_label.setStyleSheet("font-size: 12px; color: #FCA5A5;")
        self.hint_label.hide()
        layout.addWidget(self.hint_label)

    def eventFilter(self, watched, event):
        if watched is self.parent():
            if event.type() in (QEvent.Resize, QEvent.Show, QEvent.Move):
                self._sync_geometry()
        return super().eventFilter(watched, event)

    def _sync_geometry(self):
        parent = self.parentWidget()
        if parent is None:
            return
        w = min(max(parent.width() - 48, 280), 520)
        h = 126
        x = max(12, (parent.width() - w) // 2)
        y = max(12, (parent.height() - h) // 2)
        self.setGeometry(x, y, w, h)
        self.raise_()

    def _on_tick(self):
        self.owner_label.setText(_owner_text())
        if not self.isVisible():
            return
        if self._started_at_ms <= 0:
            return
        elapsed = max(0, self._now_ms() - self._started_at_ms)
        if elapsed >= self._timeout_hint_ms:
            self.hint_label.setText(
                "加载时间较长，建议先释放另一个模型后重试。"
                " 若仍等待，请优先关闭 DeepSeek-OCR / Ollama / SAM 中暂时不用的一项。"
            )
            self.hint_label.show()

    def _now_ms(self) -> int:
        import time

        return int(time.time() * 1000)

    def start_loading(self, message: str):
        self._hide_timer.stop()
        self._started_at_ms = self._now_ms()
        self.title_label.setText("模型正在加载")
        self.message_label.setText(str(message or "请稍候…"))
        self.owner_label.setText(_owner_text())
        self.hint_label.hide()
        self._sync_geometry()
        self.show()
        self.raise_()
        self._poll_timer.start()
        self._push_to_status_bar(str(message or "模型正在加载"))

    def update_message(self, message: str):
        if not self.isVisible():
            self.start_loading(message)
            return
        self.message_label.setText(str(message or "请稍候…"))
        self.owner_label.setText(_owner_text())
        self._push_to_status_bar(str(message or "模型正在加载"))

    def mark_ready(self, message: str):
        self.title_label.setText("模型已可用")
        self.message_label.setText(str(message or "模型已加载完成并可用。"))
        self.owner_label.setText(_owner_text())
        self.hint_label.hide()
        self.show()
        self.raise_()
        self._push_to_status_bar(str(message or "模型已可用"))
        self._hide_timer.start(1200)

    def mark_error(self, message: str):
        self.title_label.setText("模型加载异常")
        self.message_label.setText(str(message or "模型加载失败"))
        self.owner_label.setText(_owner_text())
        self.hint_label.setText(
            "建议先释放另一个模型后重试。若显存只有 10G，避免同时保留 DeepSeek-OCR、Ollama 和 SAM。"
        )
        self.hint_label.show()
        self.show()
        self.raise_()
        self._push_to_status_bar(str(message or "模型加载异常"))
        self._hide_timer.start(2600)

    def stop_loading(self):
        self._poll_timer.stop()
        self._hide_timer.stop()
        self.hide()

    def _push_to_status_bar(self, message: str):
        win = self.window()
        if hasattr(win, "set_model_loading_status"):
            try:
                win.set_model_loading_status(message)
            except Exception:
                pass


def route_loading_message(overlay: ModelLoadingOverlay, message: str):
    text = str(message or "").strip()
    if not text:
        return
    if is_loading_message(text):
        overlay.update_message(text)
    elif is_ready_message(text):
        overlay.mark_ready(text)
    elif is_error_message(text):
        overlay.mark_error(text)
