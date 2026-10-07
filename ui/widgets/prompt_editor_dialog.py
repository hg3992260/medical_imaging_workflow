"""
Prompt Editor Dialog — visual flow chart of the agent core loop stages,
with inline prompt editing, hot-reload, and one-click restore.
"""

import os
from typing import Any, Callable, Dict, List, Optional

from PyQt5.QtCore import Qt, QRectF, pyqtSignal
from PyQt5.QtGui import QBrush, QColor, QFont, QPainter, QPainterPath, QPen
from PyQt5.QtWidgets import (
    QDialog,
    QGraphicsDropShadowEffect,
    QGraphicsItem,
    QGraphicsObject,
    QGraphicsScene,
    QGraphicsView,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QSplitter,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from src.ai.prompt_registry import PromptRegistry


class FlowNodeWidget(QGraphicsObject):
    """A visual node in the flow chart representing one stage of the agent loop."""
    WIDTH = 180
    HEIGHT = 48
    RADIUS = 8
    clicked = pyqtSignal(str)

    def __init__(self, stage_id: str, label: str, is_overridden: bool, x: float, y: float):
        super().__init__()
        self.stage_id = stage_id
        self.label = label
        self._overridden = is_overridden
        self.setPos(x, y)
        self.setAcceptHoverEvents(True)
        self.setCursor(Qt.PointingHandCursor)
        self._hover = False
        self._selected = False
        shadow = QGraphicsDropShadowEffect()
        shadow.setBlurRadius(8)
        shadow.setOffset(2, 2)
        shadow.setColor(QColor(0, 0, 0, 40))
        self.setGraphicsEffect(shadow)

    def boundingRect(self) -> QRectF:
        return QRectF(-self.WIDTH / 2, -self.HEIGHT / 2, self.WIDTH, self.HEIGHT)

    def paint(self, painter: QPainter, option, widget=None):
        painter.setRenderHint(QPainter.Antialiasing)
        if self._selected:
            base = QColor("#4C72B0")
        elif self._hover:
            base = QColor("#6688CC")
        elif self._overridden:
            base = QColor("#DD8452")
        else:
            base = QColor("#7EABE5")
        path = QPainterPath()
        r = QRectF(-self.WIDTH / 2, -self.HEIGHT / 2, self.WIDTH, self.HEIGHT)
        path.addRoundedRect(r, self.RADIUS, self.RADIUS)
        painter.setBrush(QBrush(base))
        painter.setPen(QPen(base.darker(120), 2))
        painter.drawPath(path)
        painter.setPen(QPen(Qt.white, 1))
        font = painter.font()
        font.setPointSize(9)
        font.setBold(True)
        painter.setFont(font)
        painter.drawText(r, Qt.AlignCenter, self.label)

    def set_selected(self, sel: bool):
        self._selected = sel
        self.update()

    def mousePressEvent(self, event):
        super().mousePressEvent(event)
        self.clicked.emit(self.stage_id)

    def hoverEnterEvent(self, event):
        self._hover = True
        self.update()

    def hoverLeaveEvent(self, event):
        self._hover = False
        self.update()


class FlowChartView(QGraphicsView):
    """Zoomable/pannable view for the flow chart."""
    def __init__(self, scene: QGraphicsScene):
        super().__init__(scene)
        self.setRenderHint(QPainter.Antialiasing)
        self.setDragMode(QGraphicsView.ScrollHandDrag)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.AnchorUnderMouse)

    def wheelEvent(self, event):
        factor = 1.15
        if event.angleDelta().y() < 0:
            factor = 1.0 / factor
        self.scale(factor, factor)


class PromptEditorDialog(QDialog):
    """Full-screen dialog showing the agent loop flow chart with editable prompts."""

    revert_requested = pyqtSignal(str)
    revert_all_requested = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Prompt Editor — Agent Core Loop")
        self.setMinimumSize(1200, 800)
        self.resize(1400, 900)
        self._registry = PromptRegistry.instance()
        self._nodes: Dict[str, FlowNodeWidget] = {}
        self._current_stage: Optional[str] = None
        self._dirty = False
        self._init_ui()
        self._build_flow()
        # Auto-select first node so defaults are visible on open
        if self._registry.stage_order:
            self._on_node_clicked(self._registry.stage_order[0])

    def _init_ui(self):
        main_layout = QHBoxLayout(self)
        splitter = QSplitter(Qt.Horizontal)

        # ---- Left: Flow chart ----
        self.scene = QGraphicsScene()
        self.view = FlowChartView(self.scene)
        self.view.setStyleSheet("background:#F8F9FA;border:1px solid #CCC;")
        splitter.addWidget(self.view)

        # ---- Right: Editor panel ----
        right = QWidget()
        right_layout = QVBoxLayout(right)

        self.stage_label = QLabel("Select a node to edit its prompt")
        self.stage_label.setStyleSheet("font-size:14px;font-weight:bold;color:#333;")
        right_layout.addWidget(self.stage_label)

        self.override_indicator = QLabel("")
        self.override_indicator.setStyleSheet("color:#DD8452;font-weight:bold;")
        right_layout.addWidget(self.override_indicator)

        right_layout.addWidget(QLabel("Prompt Template:"))
        self.prompt_editor = QTextEdit()
        self.prompt_editor.setStyleSheet("font-family:Consolas;font-size:11px;border:1px solid #AAA;")
        self.prompt_editor.textChanged.connect(self._on_text_changed)
        right_layout.addWidget(self.prompt_editor)

        btn_layout = QHBoxLayout()
        self.save_btn = QPushButton("💾 保存并热更新")
        self.save_btn.clicked.connect(self._on_save)
        self.save_btn.setStyleSheet("background:#4C72B0;color:white;font-weight:bold;padding:6px 16px;")
        btn_layout.addWidget(self.save_btn)

        self.revert_btn = QPushButton("🔄 还原此提示词")
        self.revert_btn.clicked.connect(self._on_revert)
        self.revert_btn.setStyleSheet("padding:6px 16px;")
        btn_layout.addWidget(self.revert_btn)

        self.revert_all_btn = QPushButton("🔁 一键还原全部")
        self.revert_all_btn.clicked.connect(self._on_revert_all)
        self.revert_all_btn.setStyleSheet("background:#C44E52;color:white;padding:6px 16px;")
        btn_layout.addWidget(self.revert_all_btn)

        self.revert_all_btn.setToolTip("Restore ALL prompts from backup snapshot")

        self.factory_reset_btn = QPushButton("🏭 恢复出厂默认")
        self.factory_reset_btn.clicked.connect(self._on_factory_reset)
        self.factory_reset_btn.setStyleSheet("background:#8B0000;color:white;padding:6px 16px;")
        self.factory_reset_btn.setToolTip("Hard reset to shipping defaults (ignores backup)")
        btn_layout.addWidget(self.factory_reset_btn)

        right_layout.addLayout(btn_layout)
        right_layout.addStretch()

        splitter.addWidget(right)
        splitter.setSizes([600, 600])
        main_layout.addWidget(splitter)

    def _build_flow(self):
        stages = self._registry.stage_order
        cols = 3
        col_w = 230
        row_h = 80
        x0 = 100
        y0 = 60

        # Draw arrows between nodes
        for i in range(len(stages)):
            stage_id = stages[i]
            label = self._registry.get_label(stage_id)
            overridden = self._registry.is_overridden(stage_id)

            col = i % cols
            row = i // cols
            x = x0 + col * col_w
            y = y0 + row * row_h

            node = FlowNodeWidget(stage_id, label, overridden, x, y)
            node.clicked.connect(self._on_node_clicked)
            self.scene.addItem(node)
            self._nodes[stage_id] = node

            # Draw arrow to next node
            if i + 1 < len(stages):
                nc = (i + 1) % cols
                nr = (i + 1) // cols
                nx = x0 + nc * col_w
                ny = y0 + nr * row_h

                if col + 1 < cols and nr == row:
                    x2 = nx - col_w / 2
                    y2 = y
                else:
                    x2 = x + col_w / 2 + 20
                    y2 = y + row_h / 2

                self._draw_arrow(x + col_w / 2, y, x2, y2)

        self.scene.setSceneRect(0, 0, x0 + cols * col_w + 50, y0 + ((len(stages) - 1) // cols + 1) * row_h + 50)

    def _draw_arrow(self, x1: float, y1: float, x2: float, y2: float):
        from PyQt5.QtCore import QLineF, QPointF
        pen = QPen(QColor("#aaa"), 2)
        line = QLineF(x1, y1, x2, y2)
        self.scene.addLine(line, pen)
        angle = line.angle()
        p1 = QPointF(x2, y2)
        p2 = p1 - QPointF(10, 0)
        p3 = p1 - QPointF(0, 10)

    def _on_node_clicked(self, stage_id: str):
        self._current_stage = stage_id
        for nid, node in self._nodes.items():
            node.set_selected(nid == stage_id)
        self.stage_label.setText(f"Stage: {self._registry.get_label(stage_id)}")
        current = self._registry.get(stage_id)
        self.prompt_editor.blockSignals(True)
        self.prompt_editor.setPlainText(current)
        self.prompt_editor.blockSignals(False)
        self._dirty = False
        if self._registry.is_overridden(stage_id):
            self.override_indicator.setText("(已修改 · 自定义版本生效中)")
        else:
            self.override_indicator.setText("")

    def _on_text_changed(self):
        self._dirty = True

    def _on_save(self):
        if self._current_stage is None:
            return
        text = self.prompt_editor.toPlainText()
        self._registry.update(self._current_stage, text)
        self._dirty = False
        if self._registry.is_overridden(self._current_stage):
            self.override_indicator.setText("✅ 已保存 · 热更新已生效")
        else:
            self.override_indicator.setText("✅ 已还原为默认值")
        self._refresh_nodes()

    def _on_revert(self):
        if self._current_stage is None:
            return
        self._registry.restore(self._current_stage)
        default = self._registry.get(self._current_stage)
        self.prompt_editor.blockSignals(True)
        self.prompt_editor.setPlainText(default)
        self.prompt_editor.blockSignals(False)
        self._dirty = False
        self.override_indicator.setText("✅ 已还原为默认值")
        self._refresh_nodes()

    def _on_revert_all(self):
        reply = QMessageBox.question(
            self, "确认", "确定要一键还原所有提示词为默认版本吗？\n此操作不可撤销。",
            QMessageBox.Yes | QMessageBox.No, QMessageBox.No
        )
        if reply == QMessageBox.Yes:
            self._registry.restore_all()
            if self._current_stage:
                default = self._registry.get(self._current_stage)
                self.prompt_editor.blockSignals(True)
                self.prompt_editor.setPlainText(default)
                self.prompt_editor.blockSignals(False)
            self._dirty = False
            self.override_indicator.setText("✅ 全部已还原")
            self._refresh_nodes()

    def _on_factory_reset(self):
        reply = QMessageBox.question(
            self, "⚠️ 确认", "确定要恢复为出厂默认提示词吗？\n将覆盖备份快照和所有自定义修改。",
            QMessageBox.Yes | QMessageBox.No, QMessageBox.No
        )
        if reply == QMessageBox.Yes:
            self._registry.factory_reset()
            self._registry.snapshot_now()
            if self._current_stage:
                default = self._registry.get(self._current_stage)
                self.prompt_editor.blockSignals(True)
                self.prompt_editor.setPlainText(default)
                self.prompt_editor.blockSignals(False)
            self._dirty = False
            self.override_indicator.setText("✅ 已恢复出厂默认")
            self._refresh_nodes()

    def _refresh_nodes(self):
        for stage_id, node in self._nodes.items():
            node._overridden = self._registry.is_overridden(stage_id)
            node.update()
