#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
数据清洗及结构化界面模块

实现数据清洗及结构化界面，包括项目选择和样本数据汇总显示。
"""

import sys
import os
import json
import html
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGridLayout,
    QPushButton, QLabel, QLineEdit, QFrame,
    QMessageBox, QGroupBox, QSizePolicy, QListWidget,
    QListWidgetItem, QInputDialog, QSplitter, QScrollArea,
    QComboBox, QTextEdit, QDialog, QTreeWidget, QTreeWidgetItem,
    QHeaderView, QTableWidget, QTableWidgetItem, QTabWidget,
    QTextBrowser,
    QProgressBar
)
from PyQt5.QtCore import Qt, pyqtSignal, QSize, QTimer, QThread, pyqtSlot
from PyQt5.QtGui import QIcon, QFont, QPalette, QPixmap

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from src.services.project_service import ProjectService
from src.services.project_storage_service import ProjectStorageService
from src.services.database_service import DatabaseService
from src.services.data_cleaning_service import DataCleaningService
from src.models.project import Project
from src.core.event_bus import get_event_bus, EventNames
from src.core.ui_scale_profile import splitter_sizes
from src.utils.logger import get_logger
from src.services.ollama_local_service import OllamaLocalService, CustomLLMService
from src.services.llm_guided_sklearn_pipeline import run_llm_guided_sklearn_pipeline
from ui.widgets.ai_summary_widget import CustomAPIDialog


class Step2MiningWorker(QThread):
    progress_signal = pyqtSignal(str)
    finished_signal = pyqtSignal(dict)

    def __init__(
        self,
        *,
        project_id: str,
        project_storage_path: str,
        workspace: str,
        llm_service: Any,
        model: str,
        materials_text: str,
        out_dir: str,
        parent=None,
    ):
        super().__init__(parent)
        self.project_id = str(project_id or "").strip()
        self.project_storage_path = str(project_storage_path or "").strip()
        self.workspace = str(workspace or "").strip()
        self.llm_service = llm_service
        self.model = str(model or "").strip()
        self.materials_text = str(materials_text or "")
        self.out_dir = str(out_dir or "").strip()

    def run(self):
        try:
            self.progress_signal.emit("Step2 mining started")
            os.environ["SKLEARN_ALLOW_WORKSPACE_CSV_EXCEL"] = "1"
            result = run_llm_guided_sklearn_pipeline(
                llm_service=self.llm_service,
                model=self.model,
                materials_text=self.materials_text,
                project_storage_path=self.project_storage_path,
                workspace=self.workspace,
                out_dir=self.out_dir,
            )
            if isinstance(result, dict):
                result["project_id"] = self.project_id
                result["out_dir"] = self.out_dir
                result["service_model"] = self.model
            self.finished_signal.emit(result if isinstance(result, dict) else {"success": False, "error": "invalid_result"})
        except Exception as e:
            self.finished_signal.emit({"success": False, "error": str(e), "out_dir": self.out_dir, "project_id": self.project_id})


class ProjectListWidget(QWidget):
    """
    项目列表组件
    """
    
    # 信号定义
    project_selected = pyqtSignal(str, str)  # project_id, project_name
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.project_service = ProjectService()
        self.data_cleaning_service = DataCleaningService(DatabaseService())
        self.selected_project_id = None
        
        self.init_ui()
        self.load_projects()
    
    def init_ui(self):
        """
        初始化UI
        """
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(10)
        
        # 标题
        title_label = QLabel("项目列表")
        title_label.setStyleSheet("""
            QLabel {
                font-size: 16px;
                font-weight: bold;
                color: #2E86AB;
                padding: 5px;
            }
        """)
        layout.addWidget(title_label)
        
        # 项目列表
        self.project_list = QListWidget()
        self.project_list.setStyleSheet("""
            QListWidget {
                border: 1px solid #E0E0E0;
                border-radius: 5px;
                background-color: #FFFFFF;
                font-size: 12px;
            }
            QListWidget::item {
                padding: 10px;
                border-bottom: 1px solid #F0F0F0;
            }
            QListWidget::item:selected {
                background-color: #E3F2FD;
                color: #2E86AB;
            }
            QListWidget::item:hover {
                background-color: #F8F9FA;
            }
        """)
        self.project_list.itemClicked.connect(self.on_project_selected)
        layout.addWidget(self.project_list)
        
        # 刷新按钮
        refresh_button = QPushButton("刷新项目列表")
        refresh_button.setStyleSheet("""
            QPushButton {
                background-color: #2E86AB;
                color: white;
                border: none;
                padding: 8px 16px;
                border-radius: 4px;
                font-size: 12px;
            }
            QPushButton:hover {
                background-color: #1976D2;
            }
            QPushButton:pressed {
                background-color: #1565C0;
            }
        """)
        refresh_button.clicked.connect(self.load_projects)
        layout.addWidget(refresh_button)
    
    def load_projects(self):
        """
        加载项目列表
        """
        try:
            self.project_list.clear()
            projects = self.data_cleaning_service.get_all_projects_for_cleaning()
            
            for project in projects:
                total_sessions = sum(project['stats'].values())
                display_text = f"{project['name']} ({total_sessions}个会话)\nID: {project['project_id']}"
                
                item = QListWidgetItem()
                item.setText(display_text)
                item.setData(Qt.UserRole, project['project_id'])
                
                # 设置工具提示
                tooltip = f"""项目: {project['name']}
项目ID: {project['project_id']}
描述: {project.get('description', '无描述')}

DICOM会话: {project['stats'].get('dicom_sessions', 0)}
OCR会话: {project['stats'].get('ocr_sessions', 0)}
文本会话: {project['stats'].get('text_sessions', 0)}
总会话数: {total_sessions}"""
                
                item.setToolTip(tooltip)
                self.project_list.addItem(item)
                
        except Exception as e:
            print(f"加载项目列表失败: {e}")
    
    def on_project_selected(self, item):
        """
        项目选择事件
        """
        project_id = item.data(Qt.UserRole)
        project_name = item.text().split('\n')[0].split(' (')[0]
        self.selected_project_id = project_id
        self.project_selected.emit(project_id, project_name)


class SampleDataWidget(QWidget):
    """
    样本数据汇总组件
    """
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.db_service = DatabaseService()
        self.storage_service = ProjectStorageService(self.db_service)
        self.current_project_id = None
        
        self.init_ui()
    
    def init_ui(self):
        """
        初始化UI
        """
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(10)
        
        # 标题
        self.title_label = QLabel("样本数据汇总")
        self.title_label.setStyleSheet("""
            QLabel {
                font-size: 16px;
                font-weight: bold;
                color: #2E86AB;
                padding: 5px;
            }
        """)
        layout.addWidget(self.title_label)
        
        # 项目信息
        self.project_info_label = QLabel("请选择项目")
        self.project_info_label.setStyleSheet("""
            QLabel {
                font-size: 12px;
                color: #666666;
                padding: 5px;
                background-color: #F8F9FA;
                border-radius: 3px;
            }
        """)
        layout.addWidget(self.project_info_label)
        
        # 数据汇总表格
        self.data_table = QTableWidget()
        self.data_table.setColumnCount(4)
        self.data_table.setHorizontalHeaderLabels(["数据类型", "数量", "最新时间", "详细信息"])
        
        # 设置表格样式
        self.data_table.setStyleSheet("""
            QTableWidget {
                border: 1px solid #E0E0E0;
                border-radius: 5px;
                background-color: #FFFFFF;
                gridline-color: #F0F0F0;
            }
            QTableWidget::item {
                padding: 8px;
                border-bottom: 1px solid #F0F0F0;
            }
            QHeaderView::section {
                background-color: #F5F5F5;
                padding: 8px;
                border: none;
                font-weight: bold;
            }
        """)
        
        # 设置表格属性
        header = self.data_table.horizontalHeader()
        header.setStretchLastSection(True)
        header.setSectionResizeMode(0, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(1, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(2, QHeaderView.ResizeToContents)
        
        self.data_table.setAlternatingRowColors(True)
        self.data_table.setSelectionBehavior(QTableWidget.SelectRows)
        
        layout.addWidget(self.data_table)
        
        # 详细信息显示区域
        detail_label = QLabel("详细信息")
        detail_label.setStyleSheet("""
            QLabel {
                font-size: 14px;
                font-weight: bold;
                color: #2E86AB;
                padding: 5px;
            }
        """)
        layout.addWidget(detail_label)
        
        self.detail_text = QTextEdit()
        self.detail_text.setMaximumHeight(150)
        self.detail_text.setStyleSheet("""
            QTextEdit {
                border: 1px solid #E0E0E0;
                border-radius: 5px;
                background-color: #FAFAFA;
                font-family: 'Consolas', 'Monaco', monospace;
                font-size: 11px;
            }
        """)
        self.detail_text.setPlainText("选择表格中的数据行查看详细信息")
        layout.addWidget(self.detail_text)
        
        # 连接表格选择事件
        self.data_table.itemSelectionChanged.connect(self.on_row_selected)
    
    def set_project(self, project_id: str, project_name: str):
        """
        设置当前项目并加载数据
        """
        self.current_project_id = project_id
        self.project_info_label.setText(f"当前项目: {project_name} (ID: {project_id})")
        self.title_label.setText(f"样本数据汇总 - {project_name}")
        self.load_sample_data()
    
    def load_sample_data(self):
        """
        加载样本数据汇总
        """
        if not self.current_project_id:
            return
            
        try:
            # 清空表格
            self.data_table.setRowCount(0)
            
            # 获取DICOM数据
            dicom_data = self.get_dicom_data()
            self.add_data_row("DICOM文件", dicom_data)
            
            # 获取OCR数据
            ocr_data = self.get_ocr_data()
            self.add_data_row("OCR数据", ocr_data)
            
            # 获取文本数据
            text_data = self.get_text_data()
            self.add_data_row("文本数据", text_data)
            
        except Exception as e:
            print(f"加载样本数据失败: {e}")
            self.detail_text.setPlainText(f"加载数据时出错: {str(e)}")
    
    def get_dicom_data(self):
        """
        获取DICOM数据统计
        """
        try:
            with self.db_service.get_connection() as conn:
                cursor = conn.cursor()
                
                # 获取DICOM会话数量和最新时间
                cursor.execute("""
                    SELECT COUNT(*) as count, MAX(created_at) as latest
                    FROM dicom_sessions 
                    WHERE project_id = ?
                """, (self.current_project_id,))
                
                result = cursor.fetchone()
                count = result[0] if result else 0
                latest = result[1] if result and result[1] else "无数据"
                
                # 获取详细信息
                cursor.execute("""
                    SELECT session_id, file_path, created_at
                    FROM dicom_sessions 
                    WHERE project_id = ?
                    ORDER BY created_at DESC
                    LIMIT 5
                """, (self.current_project_id,))
                
                sessions = cursor.fetchall()
                details = []
                for session in sessions:
                    details.append(f"会话ID: {session[0]}, 文件: {session[1]}, 时间: {session[2]}")
                
                return {
                    'count': count,
                    'latest': latest,
                    'details': details
                }
                
        except Exception as e:
            print(f"获取DICOM数据失败: {e}")
            return {'count': 0, 'latest': '获取失败', 'details': []}
    
    def get_ocr_data(self):
        """
        获取OCR数据统计
        """
        try:
            with self.db_service.get_connection() as conn:
                cursor = conn.cursor()
                
                # 获取OCR会话数量和最新时间
                cursor.execute("""
                    SELECT COUNT(*) as count, MAX(created_at) as latest
                    FROM ocr_sessions 
                    WHERE project_id = ?
                """, (self.current_project_id,))
                
                result = cursor.fetchone()
                count = result[0] if result else 0
                latest = result[1] if result and result[1] else "无数据"
                
                # 获取详细信息（联结结果表以获取识别文本）
                cursor.execute("""
                    SELECT r.session_id, s.image_path, r.recognized_text, r.created_at
                    FROM ocr_results r
                    JOIN ocr_sessions s ON r.session_id = s.session_id
                    WHERE s.project_id = ?
                    ORDER BY r.created_at DESC
                    LIMIT 10
                """, (self.current_project_id,))
                
                sessions = cursor.fetchall()
                details = []
                for session in sessions:
                    text_preview = session[2] or "无文本"  # 完整显示文本，不截断
                    details.append(f"会话ID: {session[0]}, 图像: {session[1]}, 文本: {text_preview}, 时间: {session[3]}")
                
                return {
                    'count': count,
                    'latest': latest,
                    'details': details
                }
                
        except Exception as e:
            print(f"获取OCR数据失败: {e}")
            return {'count': 0, 'latest': '获取失败', 'details': []}
    
    def get_text_data(self):
        """
        获取文本数据统计
        """
        try:
            with self.db_service.get_connection() as conn:
                cursor = conn.cursor()
                
                # 获取文本会话数量和最新时间
                cursor.execute("""
                    SELECT COUNT(*) as count, MAX(created_at) as latest
                    FROM text_sessions 
                    WHERE project_id = ?
                """, (self.current_project_id,))
                
                result = cursor.fetchone()
                count = result[0] if result else 0
                latest = result[1] if result and result[1] else "无数据"
                
                # 获取详细信息
                cursor.execute("""
                    SELECT session_id, file_path, processed_text, created_at
                    FROM text_sessions 
                    WHERE project_id = ?
                    ORDER BY created_at DESC
                    LIMIT 5
                """, (self.current_project_id,))
                
                sessions = cursor.fetchall()
                details = []
                for session in sessions:
                    text_preview = session[2] or "无文本"  # 完整显示文本，不截断
                    details.append(f"会话ID: {session[0]}, 文件: {session[1]}, 文本: {text_preview}, 时间: {session[3]}")
                
                return {
                    'count': count,
                    'latest': latest,
                    'details': details
                }
                
        except Exception as e:
            print(f"获取文本数据失败: {e}")
            return {'count': 0, 'latest': '获取失败', 'details': []}
    
    def add_data_row(self, data_type: str, data_info: dict):
        """
        添加数据行到表格
        """
        row = self.data_table.rowCount()
        self.data_table.insertRow(row)
        
        # 数据类型
        self.data_table.setItem(row, 0, QTableWidgetItem(data_type))
        
        # 数量
        self.data_table.setItem(row, 1, QTableWidgetItem(str(data_info['count'])))
        
        # 最新时间
        self.data_table.setItem(row, 2, QTableWidgetItem(str(data_info['latest'])))
        
        # 详细信息（存储在隐藏数据中）
        detail_item = QTableWidgetItem(f"{len(data_info['details'])} 条记录")
        detail_item.setData(Qt.UserRole, data_info['details'])
        self.data_table.setItem(row, 3, detail_item)
    
    def on_row_selected(self):
        """
        表格行选择事件
        """
        current_row = self.data_table.currentRow()
        if current_row >= 0:
            detail_item = self.data_table.item(current_row, 3)
            if detail_item:
                details = detail_item.data(Qt.UserRole)
                if details:
                    detail_text = "\n".join(details)
                    self.detail_text.setPlainText(detail_text)
                else:
                    self.detail_text.setPlainText("该数据类型暂无详细信息")


class DataCleaningWidget(QWidget):
    """数据清洗及结构化组件
    
    提供项目选择和样本数据汇总功能
    """
    
    # 定义信号
    project_selected = pyqtSignal(str, str)  # project_id, project_name
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.logger = get_logger(__name__)
        self.project_service = ProjectService()
        self.db_service = DatabaseService()
        self.data_cleaning_service = DataCleaningService(self.db_service)
        self.storage_service = ProjectStorageService(self.db_service)
        self.ollama = OllamaLocalService()
        self._custom_llm_service: Optional[CustomLLMService] = None
        self._custom_api_profiles: List[dict] = []
        self._latest_mining_result: Dict[str, Any] = {}
        self._latest_export_dir: str = ""
        self.selected_project_id = None
        
        self._load_custom_api_profiles()
        self.init_ui()
        self.setup_connections()
        self.load_projects()
        self._refresh_llm_combo()
    
    def init_ui(self):
        """初始化用户界面"""
        layout = QHBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(10)
        
        # 创建分割器
        splitter = QSplitter(Qt.Horizontal)
        
        # 左侧项目列表
        left_widget = self.create_project_list_widget()
        splitter.addWidget(left_widget)
        
        # 右侧样本数据显示
        right_widget = self.create_sample_data_widget()
        splitter.addWidget(right_widget)
        
        # 设置分割器比例
        left_w, right_w = splitter_sizes("step2")
        splitter.setSizes([left_w, right_w])
        
        layout.addWidget(splitter)
    
    def create_project_list_widget(self):
        """创建项目列表组件"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # 标题
        title_label = QLabel("项目列表")
        title_label.setFont(QFont("Arial", 12, QFont.Bold))
        layout.addWidget(title_label)
        
        # 项目列表
        self.project_list = QListWidget()
        self.project_list.setMinimumWidth(250)
        layout.addWidget(self.project_list)
        
        return widget
    
    def create_sample_data_widget(self):
        """创建样本数据显示组件"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # 标题
        title_label = QLabel("样本数据汇总")
        title_label.setFont(QFont("Arial", 12, QFont.Bold))
        layout.addWidget(title_label)

        ctrl = QHBoxLayout()
        ctrl.addWidget(QLabel("LLM服务"))
        self.llm_combo = QComboBox()
        self.llm_combo.setMinimumWidth(240)
        ctrl.addWidget(self.llm_combo)
        self.custom_api_btn = QPushButton("+ Custom API")
        self.custom_api_btn.clicked.connect(self._on_custom_api_clicked)
        ctrl.addWidget(self.custom_api_btn)
        self.mine_btn = QPushButton("AI数据挖掘与绘图")
        self.mine_btn.clicked.connect(self.on_mine_data_clicked)
        ctrl.addWidget(self.mine_btn)
        self.open_export_btn = QPushButton("打开导出目录")
        self.open_export_btn.setEnabled(False)
        self.open_export_btn.clicked.connect(self.open_latest_export_dir)
        ctrl.addWidget(self.open_export_btn)
        ctrl.addStretch()
        layout.addLayout(ctrl)

        self.mining_status_label = QLabel("状态: 就绪")
        layout.addWidget(self.mining_status_label)
        
        # 数据表格
        self.data_table = QTableWidget()
        self.data_table.setColumnCount(3)
        self.data_table.setHorizontalHeaderLabels(["数据类型", "数量", "操作"])
        
        # 设置表格属性
        header = self.data_table.horizontalHeader()
        header.setStretchLastSection(True)
        header.setSectionResizeMode(0, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(1, QHeaderView.ResizeToContents)
        
        layout.addWidget(self.data_table)
        
        # 详细信息显示
        detail_label = QLabel("详细信息")
        detail_label.setFont(QFont("Arial", 10, QFont.Bold))
        layout.addWidget(detail_label)

        self.result_tabs = QTabWidget()

        self.detail_text = QTextEdit()
        self.detail_text.setMinimumHeight(300)
        self.detail_text.setReadOnly(True)
        self.result_tabs.addTab(self.detail_text, "项目汇总")

        self.mining_result_browser = QTextBrowser()
        self.mining_result_browser.setOpenExternalLinks(True)
        self.mining_result_browser.setPlaceholderText("AI 挖掘结果、统计摘要与图表链接将在此显示...")
        self.result_tabs.addTab(self.mining_result_browser, "AI挖掘结果")
        layout.addWidget(self.result_tabs)
        self.data_table.cellClicked.connect(self.on_summary_cell_clicked)
        
        return widget
    
    def setup_connections(self):
        """设置信号连接"""
        self.project_list.itemClicked.connect(self.on_project_selected)
        from src.core.event_bus import get_event_bus, EventNames
        self.event_bus = get_event_bus()
        self.event_bus.subscribe(EventNames.OCR_SESSION_CREATED, self.on_ocr_session_created)

    def _custom_api_config_path(self) -> str:
        try:
            from src.core.path_config import get_coco_data_dir
            base = str(get_coco_data_dir())
        except Exception:
            base = os.path.join(os.path.expanduser("~"), "AppData", "Local", "MedicalImagingWorkflow", "coco_data")
        return os.path.join(base, "custom_api_profiles.json")

    def _load_custom_api_profiles(self):
        self._custom_api_profiles = []
        try:
            p = self._custom_api_config_path()
            if os.path.exists(p):
                with open(p, "r", encoding="utf-8") as f:
                    data = json.load(f)
                if isinstance(data, list):
                    self._custom_api_profiles = data
        except Exception:
            self._custom_api_profiles = []

    def _save_custom_api_profiles(self):
        try:
            p = self._custom_api_config_path()
            os.makedirs(os.path.dirname(p), exist_ok=True)
            with open(p, "w", encoding="utf-8") as f:
                json.dump(self._custom_api_profiles, f, ensure_ascii=False, indent=2)
        except Exception as e:
            self.logger.error(f"保存 Custom API 配置失败: {e}")

    def _refresh_llm_combo(self):
        current = self.llm_combo.currentText() if hasattr(self, "llm_combo") else ""
        items: List[str] = []
        try:
            local_models = self.ollama.list_models() or []
            for m in local_models:
                sm = str(m or "").strip()
                if sm:
                    items.append(f"Ollama:{sm}")
        except Exception:
            pass
        for p in self._custom_api_profiles:
            name = str(p.get("name") or "Custom").strip()
            if name:
                items.append(f"Custom:{name}")
        if not items:
            items = ["Custom:DeepSeek"]
        self.llm_combo.clear()
        self.llm_combo.addItems(items)
        if current:
            idx = self.llm_combo.findText(current)
            if idx >= 0:
                self.llm_combo.setCurrentIndex(idx)

    def _on_custom_api_clicked(self):
        dialog = CustomAPIDialog(self)
        if dialog.exec_() == QDialog.Accepted:
            profile = dialog.profile
            name = str(profile.get("name") or "Custom").strip()
            existing = next((i for i, p in enumerate(self._custom_api_profiles) if p.get("name") == name), -1)
            if existing >= 0:
                self._custom_api_profiles[existing] = profile
            else:
                self._custom_api_profiles.append(profile)
            self._save_custom_api_profiles()
            self._refresh_llm_combo()
            idx = self.llm_combo.findText(f"Custom:{name}")
            if idx >= 0:
                self.llm_combo.setCurrentIndex(idx)

    def _selected_custom_profile(self) -> Optional[dict]:
        current = self.llm_combo.currentText() if hasattr(self, "llm_combo") else ""
        if not current.startswith("Custom:"):
            return None
        name = current.split(":", 1)[1]
        return next((p for p in self._custom_api_profiles if str(p.get("name") or "").strip() == name), None)

    def _current_model_name(self) -> str:
        prof = self._selected_custom_profile()
        if prof:
            return str(prof.get("model") or "deepseek-v4-flash")
        current = self.llm_combo.currentText() if hasattr(self, "llm_combo") else ""
        if current.startswith("Ollama:"):
            return current.split(":", 1)[1].strip()
        return "llama3"

    def _current_service_label(self) -> str:
        prof = self._selected_custom_profile()
        if prof:
            return f"Custom API: {prof.get('name', 'Custom')}"
        return "Ollama"

    def _active_llm_service(self) -> Any:
        prof = self._selected_custom_profile()
        if prof:
            self._custom_llm_service = CustomLLMService(
                base_url=prof.get("base_url", ""),
                api_key=prof.get("api_key", ""),
                model=prof.get("model", "deepseek-v4-flash"),
                mode=prof.get("mode", "chat"),
                temperature=float(prof.get("temperature", 0.7)),
                max_tokens=int(prof.get("max_tokens", 4096)),
                top_p=float(prof.get("top_p", 1.0)),
                api_type=prof.get("api_type", "deepseek"),
            )
            return self._custom_llm_service
        return self.ollama

    def _current_project_storage_path(self) -> str:
        if not self.selected_project_id:
            return ""
        try:
            return str(self.storage_service.get_project_storage_path(self.selected_project_id))
        except Exception:
            return ""

    def _build_step2_materials_text(self) -> str:
        parts = []
        try:
            cur = self.project_list.currentItem()
            if cur:
                parts.append(f"[PROJECT]\n{cur.text()}")
        except Exception:
            pass
        summary_text = self.detail_text.toPlainText().strip() if hasattr(self, "detail_text") else ""
        if summary_text:
            parts.append(f"[SUMMARY]\n{summary_text[:18000]}")
        base = self._current_project_storage_path()
        if base:
            tabular_root = Path(base) / "tabular"
            docs_root = Path(base) / "documents"
            if tabular_root.exists():
                tabular_files = [p.name for p in sorted(tabular_root.glob("**/*")) if p.is_file()][:80]
                if tabular_files:
                    parts.append("[TABULAR_FILES]\n" + "\n".join(tabular_files))
            tab = self._tabular_type_counts()
            excel_n = tab.get("excel_count", 0)
            csv_n = tab.get("csv_count", 0)
            word_n = tab.get("word_table_count", 0)
            type_info = []
            if excel_n: type_info.append(f"{excel_n} Excel表格")
            if csv_n: type_info.append(f"{csv_n} CSV表格")
            if word_n: type_info.append(f"{word_n} 文献表格")
            if type_info:
                parts.append("[TABLE_SOURCES]\n" + ", ".join(type_info))
            if docs_root.exists():
                doc_files = [p.name for p in sorted(docs_root.glob("**/*")) if p.is_file()][:40]
                if doc_files:
                    parts.append("[DOCUMENT_FILES]\n" + "\n".join(doc_files))
        return "\n\n".join(parts).strip()

    def _tabular_type_counts(self) -> Dict[str, Any]:
        base = self._current_project_storage_path()
        if not base:
            return {}
        import json
        tabular_dir = Path(base) / "tabular"
        counts = {"excel_count": 0, "csv_count": 0, "word_table_count": 0, "files": []}
        try:
            if tabular_dir.is_dir():
                for jf in sorted(tabular_dir.glob("**/*.json"))[:200]:
                    try:
                        with open(jf, "r", encoding="utf-8") as f:
                            obj = json.load(f)
                        sheet = obj.get("sheet") or {}
                        tp = sheet.get("type") or obj.get("metadata", {}).get("type", "")
                        fname = obj.get("metadata", {}).get("source_file_name", jf.name)
                        if tp in ("excel", "xlsx", "xls"):
                            counts["excel_count"] += 1
                        elif tp in ("csv",):
                            counts["csv_count"] += 1
                        elif tp in ("word_table", "word", "docx_table"):
                            counts["word_table_count"] += 1
                        else:
                            ext = Path(fname).suffix.lower() if fname else ""
                            if ext in (".xlsx", ".xls"):
                                counts["excel_count"] += 1
                            elif ext == ".csv":
                                counts["csv_count"] += 1
                            else:
                                counts["word_table_count"] += 1
                        counts["files"].append(fname)
                    except Exception:
                        pass
        except Exception:
            pass
        return counts

    def on_mine_data_clicked(self):
        if not self.selected_project_id:
            QMessageBox.warning(self, "缺少项目", "请先选择项目。")
            return
        project_storage_path = self._current_project_storage_path()
        if not project_storage_path or not os.path.isdir(project_storage_path):
            QMessageBox.warning(self, "缺少项目存储目录", "当前项目没有可用的存储目录。")
            return
        llm_service = self._active_llm_service()
        model = self._current_model_name()
        materials_text = self._build_step2_materials_text()
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_dir = os.path.join(project_storage_path, "exports", f"step2_mining_{ts}")
        os.makedirs(out_dir, exist_ok=True)

        self.mine_btn.setEnabled(False)
        self.open_export_btn.setEnabled(False)
        self.mining_status_label.setText(f"状态: 正在分析 ({self._current_service_label()} / {model})")
        self.mining_result_browser.setPlainText("正在进行 Step2 数据挖掘与绘图，请稍候...")
        self.result_tabs.setCurrentWidget(self.mining_result_browser)

        self.mining_worker = Step2MiningWorker(
            project_id=self.selected_project_id,
            project_storage_path=project_storage_path,
            workspace=project_storage_path,
            llm_service=llm_service,
            model=model,
            materials_text=materials_text,
            out_dir=out_dir,
            parent=self,
        )
        self.mining_worker.progress_signal.connect(self._on_mining_progress)
        self.mining_worker.finished_signal.connect(self._on_mining_finished)
        self.mining_worker.start()

    def _on_mining_progress(self, message: str):
        self.logger.info(f"Step2Mining: {message}")

    def _on_mining_finished(self, result: Dict[str, Any]):
        self.mine_btn.setEnabled(True)
        self._latest_mining_result = result if isinstance(result, dict) else {}
        self._latest_export_dir = str((self._latest_mining_result or {}).get("out_dir") or "").strip()
        self.open_export_btn.setEnabled(bool(self._latest_export_dir and os.path.isdir(self._latest_export_dir)))
        ok = bool((self._latest_mining_result or {}).get("success"))
        self.mining_status_label.setText("状态: 已完成" if ok else "状态: 执行失败")
        self._render_mining_result(self._latest_mining_result)

    def _render_mining_result(self, result: Dict[str, Any]):
        if not isinstance(result, dict):
            self.mining_result_browser.setPlainText("分析结果无效。")
            return
        if not result.get("success"):
            msg = str(result.get("error") or "unknown_error")
            self.mining_result_browser.setPlainText(f"Step2 数据挖掘失败:\n{msg}")
            return

        summary = str(result.get("summary") or "").strip()
        artifacts = [str(x).replace("\\", "/") for x in (result.get("artifacts") or []) if str(x).strip()]
        figure_notes = str(result.get("figure_notes") or "").strip()
        plan = result.get("plan") or {}
        profile = result.get("profile") or {}
        out_dir = str(result.get("out_dir") or "").strip()
        project_storage_path = self._current_project_storage_path()

        def _to_file_url(rel_path: str) -> str:
            if not project_storage_path:
                return rel_path
            ap = os.path.join(project_storage_path, rel_path)
            return Path(ap).resolve().as_uri()

        html_parts = []
        html_parts.append("<h3>Step2 AI 数据挖掘结果</h3>")
        html_parts.append(f"<p><b>LLM服务:</b> {html.escape(self._current_service_label())}<br/>")
        html_parts.append(f"<b>模型:</b> {html.escape(self._current_model_name())}</p>")
        if summary:
            html_parts.append("<h4>自动分析摘要</h4>")
            html_parts.append(f"<pre>{html.escape(summary)}</pre>")
        if isinstance(profile, dict):
            html_parts.append("<h4>数据概况</h4>")
            html_parts.append(
                f"<p>Records loaded: <b>{int(profile.get('n_records') or 0)}</b><br/>"
                f"Columns profiled: <b>{len(profile.get('columns') or [])}</b></p>"
            )
        if isinstance(plan, dict) and plan.get("tasks"):
            html_parts.append("<h4>分析任务</h4><ul>")
            for task in (plan.get("tasks") or [])[:12]:
                html_parts.append(f"<li>{html.escape(json.dumps(task, ensure_ascii=False))}</li>")
            html_parts.append("</ul>")
        if artifacts:
            html_parts.append("<h4>导出结果</h4><ul>")
            for rel_path in artifacts:
                url = _to_file_url(rel_path)
                label = os.path.basename(rel_path)
                html_parts.append(f'<li><a href="{html.escape(url)}">{html.escape(label)}</a> <code>{html.escape(rel_path)}</code></li>')
            html_parts.append("</ul>")
        if figure_notes:
            html_parts.append("<h4>图表解释</h4>")
            html_parts.append(f"<pre>{html.escape(figure_notes)}</pre>")
        if out_dir:
            html_parts.append(f"<p><b>导出目录:</b> <code>{html.escape(out_dir)}</code></p>")
        self.mining_result_browser.setHtml("\n".join(html_parts))
        self.result_tabs.setCurrentWidget(self.mining_result_browser)

    def open_latest_export_dir(self):
        target = str(self._latest_export_dir or "").strip()
        if not target or not os.path.isdir(target):
            QMessageBox.information(self, "无导出目录", "当前没有可打开的导出目录。")
            return
        try:
            os.startfile(target)
        except Exception as e:
            QMessageBox.warning(self, "打开失败", f"无法打开目录:\n{target}\n\n{e}")
    
    def load_projects(self):
        """加载项目列表"""
        try:
            self.project_list.clear()
            projects = self.data_cleaning_service.get_all_projects_for_cleaning()
            
            for project in projects:
                total_sessions = sum(project['stats'].values())
                display_text = f"{project['name']} ({total_sessions}个会话)\nID: {project['project_id']}"
                
                item = QListWidgetItem()
                item.setText(display_text)
                item.setData(Qt.UserRole, project['project_id'])
                
                # 设置工具提示
                tooltip = f"""项目: {project['name']}
项目ID: {project['project_id']}
描述: {project.get('description', '无描述')}

DICOM会话: {project['stats'].get('dicom_sessions', 0)}
OCR会话: {project['stats'].get('ocr_sessions', 0)}
文本会话: {project['stats'].get('text_sessions', 0)}
总会话数: {total_sessions}"""
                
                item.setToolTip(tooltip)
                self.project_list.addItem(item)
                
        except Exception as e:
            print(f"加载项目列表失败: {e}")

    def on_ocr_session_created(self, data):
        try:
            pid = data.get("project_id") if isinstance(data, dict) else None
            if pid and getattr(self, "selected_project_id", None) == pid:
                self.load_sample_data()
        except Exception:
            pass
    def on_project_selected(self, item):
        """处理项目选择事件
        
        Args:
            item: 选中的列表项
        """
        project_id = item.data(Qt.UserRole)
        project_name = item.text().split('\n')[0].split(' (')[0]
        self.selected_project_id = project_id
        self.project_selected.emit(project_id, project_name)
        
        # 加载样本数据
        self.load_sample_data()
    
    def load_sample_data(self):
        """加载选中项目的样本数据"""
        if not self.selected_project_id:
            return
            
        try:
            self._latest_mining_result = {}
            self._latest_export_dir = ""
            self.open_export_btn.setEnabled(False)
            self.mining_status_label.setText("状态: 就绪")
            self.mining_result_browser.clear()
            # 获取详细的样本数据汇总
            summary = self.data_cleaning_service.get_project_sample_summary(self.selected_project_id)
            
            # 清空表格
            self.data_table.setRowCount(0)
            
            # 添加数据行
            tab = summary.get('tabular_data', {})
            excel_n = tab.get('excel_count', 0)
            csv_n = tab.get('csv_count', 0)
            word_n = tab.get('word_table_count', 0)
            data_types = [
                ("DICOM影像文件", summary['dicom_data']['count'], summary['dicom_data']['file_count']),
                ("OCR文本提取", summary['ocr_data']['count'], len(summary['ocr_data']['images'])),
                ("文本处理分析", summary['text_data']['count'], len(summary['text_data']['files'])),
                ("Excel表格数据", excel_n, excel_n),
                ("CSV表格数据", csv_n, csv_n),
                ("文献表格内容", word_n, word_n),
                ("样本描述", summary['samples']['count'], summary['samples']['count']),
                ("Magic Seg分割", summary['magic_seg_data']['total_count'], summary['magic_seg_data']['total_count'])
            ]
            
            self.data_table.setRowCount(len(data_types))
            
            for row, (data_type, session_count, item_count) in enumerate(data_types):
                self.data_table.setItem(row, 0, QTableWidgetItem(data_type))
                self.data_table.setItem(row, 1, QTableWidgetItem(f"{session_count} 会话 / {item_count} 项目"))
                self.data_table.setItem(row, 2, QTableWidgetItem("分页浏览全部结果" if session_count > 0 else "暂无数据"))
            
            # 更新详细信息
            self.display_detailed_summary(summary)
            
        except Exception as e:
            self.detail_text.setText(f"加载数据失败: {str(e)}")
    
    def on_summary_cell_clicked(self, row: int, col: int):
        if col != 2:
            return
        op_item = self.data_table.item(row, 2)
        if not op_item:
            return
        text = op_item.text()
        if "分页浏览全部结果" not in text:
            return
        dtype_item = self.data_table.item(row, 0)
        dtype = dtype_item.text() if dtype_item else ""
        mapping = {
            "DICOM影像文件": "DICOM会话",
            "OCR文本提取": "OCR结果",
            "文本处理分析": "文档",
            "Excel表格数据": "表格数据",
            "CSV表格数据": "表格数据",
            "文献表格内容": "表格数据",
            "样本描述": "样本描述",
            "Magic Seg分割": "ROI数据",
        }
        initial_type = mapping.get(dtype, "DICOM会话")
        self.open_summary_pagination(initial_type)
    
    def display_detailed_summary(self, summary):
        """显示详细的数据汇总信息"""
        detail_info = f"""项目样本数据详细汇总
项目ID: {summary['project_id']}
更新时间: {summary['updated_at'][:19]}

=== DICOM影像数据 ===
会话数量: {summary['dicom_data'].get('session_count', 0)}
文件数量: {summary['dicom_data'].get('file_count', 0)}
患者数量: {summary['dicom_data'].get('patient_count', 0)}
"""
        
        # 添加患者信息
        if summary['dicom_data'].get('patients'):
            detail_info += "患者信息:\n"
            for i, patient in enumerate(summary['dicom_data']['patients'][:3]):
                detail_info += f"  {i+1}. {patient.get('patient_name', 'Unknown')} (ID: {patient.get('patient_id', 'N/A')})\n"
            if len(summary['dicom_data']['patients']) > 3:
                detail_info += f"  ... 还有 {len(summary['dicom_data']['patients']) - 3} 个患者\n"
        
        # 添加DICOM文件信息
        if summary['dicom_data'].get('session_files'):
            detail_info += "\n文件列表:\n"
            for i, file_info in enumerate(summary['dicom_data']['session_files'][:5]):
                detail_info += f"  {i+1}. {file_info['file_name']} ({file_info['file_size']} bytes)\n"
            if len(summary['dicom_data']['session_files']) > 5:
                detail_info += f"  ... 还有 {len(summary['dicom_data']['session_files']) - 5} 个文件\n"
        
        # ROI识别信息
        if hasattr(self.data_cleaning_service, 'get_roi_details'):
            try:
                roi_data = self.data_cleaning_service.get_roi_details(summary['project_id'])
                detail_info += f"\n=== ROI识别数据 ===\n"
                detail_info += f"ROI总数: {roi_data.get('total_count', 0)}\n"
                detail_info += f"总面积: {roi_data.get('total_area', 0):.2f}\n"
                
                if roi_data.get('roi_types'):
                    detail_info += "ROI类型分布:\n"
                    for roi_type, count in roi_data['roi_types'].items():
                        detail_info += f"  {roi_type}: {count}个\n"
                
                if roi_data.get('roi_list'):
                    detail_info += "\n最近ROI:\n"
                    for i, roi in enumerate(roi_data['roi_list'][:3]):
                        detail_info += f"  {i+1}. {roi['roi_name']} (面积: {roi['area']:.2f})\n"
            except Exception as e:
                detail_info += f"\n=== ROI识别数据 ===\n获取ROI数据失败: {str(e)}\n"
        
        # Magic Seg信息
        if summary.get('magic_seg_data'):
            ms_data = summary['magic_seg_data']
            detail_info += f"\n=== Magic Seg数据 ===\n"
            detail_info += f"分割总数: {ms_data.get('total_count', 0)}\n"
            detail_info += f"总面积: {ms_data.get('total_area', 0):.2f}\n"
            
            if ms_data.get('source_counts'):
                detail_info += "来源分布:\n"
                for source, count in ms_data['source_counts'].items():
                    src_name = "自动(SAM)" if source == 'magic_seg_sam' else "交互式"
                    detail_info += f"  {src_name}: {count}个\n"
            
            if ms_data.get('seg_list'):
                detail_info += "\n最近分割:\n"
                for i, seg in enumerate(ms_data['seg_list'][:3]):
                    detail_info += f"  {i+1}. {seg['file_name']} (面积: {seg['area']:.2f})\n"

        detail_info += f"\n=== OCR识别数据 ===\n会话数量: {summary['ocr_data'].get('session_count', 0)}\n识别结果数: {summary['ocr_data'].get('result_count', 0)}\n"
        
        # 添加OCR置信度信息
        if summary['ocr_data'].get('avg_confidence', 0) > 0:
            detail_info += f"平均置信度: {summary['ocr_data']['avg_confidence']}%\n"
            detail_info += f"置信度范围: {summary['ocr_data']['min_confidence']}% - {summary['ocr_data']['max_confidence']}%\n"
        
        # 添加OCR文本长度信息
        if summary['ocr_data'].get('total_text_length', 0) > 0:
            detail_info += f"提取文本总长度: {summary['ocr_data']['total_text_length']} 字符\n"
        
        # 添加OCR详细结果
        if summary['ocr_data'].get('ocr_details'):
            detail_info += "\n最近识别结果:\n"
            for i, ocr_result in enumerate(summary['ocr_data']['ocr_details'][:10]):
                preview = ocr_result['text_preview']  # 完整显示文本，不截断
                detail_info += f"  {i+1}. {ocr_result['file_name']}: {preview}\n"
            if len(summary['ocr_data']['ocr_details']) > 10:
                detail_info += f"  ... 还有 {len(summary['ocr_data']['ocr_details']) - 10} 条结果\n"
        
        detail_info += f"\n=== 文本处理数据 ===\n会话数量: {summary['text_data'].get('session_count', 0)}\n文档数量: {summary['text_data'].get('document_count', 0)}\n"
        
        # 添加关键字信息
        if summary['text_data'].get('keywords_count', 0) > 0:
            detail_info += f"提取关键字数: {summary['text_data']['keywords_count']}\n"
            
            if summary['text_data'].get('top_keywords'):
                detail_info += "热门关键字:\n"
                for keyword, count in summary['text_data']['top_keywords']:  # 显示所有关键字
                    detail_info += f"  {keyword}: {count}次\n"
        
        # 添加文本内容长度信息
        if summary['text_data'].get('total_content_length', 0) > 0:
            detail_info += f"文本总长度: {summary['text_data']['total_content_length']} 字符\n"
        
        # 添加文档详细信息
        if summary['text_data'].get('document_details'):
            detail_info += "\n最近文档:\n"
            for i, doc in enumerate(summary['text_data']['document_details'][:3]):
                preview = doc['content_preview']  # 完整显示文本，不截断
                detail_info += f"  {i+1}. {doc['file_name']}: {preview}\n"
                if doc.get('keywords'):
                    detail_info += f"     关键字: {', '.join(doc['keywords'])}\n"  # 显示所有关键字
        
        detail_info += f"\n=== 样本描述 ===\n数量: {summary['samples'].get('count', 0)}\n"
        if summary['samples'].get('details'):
            detail_info += "最近样本:\n"
            for i, s in enumerate(summary['samples']['details'][:5]):
                status = "激活" if s.get('is_active', 1) else "禁用"
                display_name = s.get('display_name', '')
                user_id = s.get('user_id', '')
                desc = s.get('description', '') or '无描述'
                detail_info += f"  {i+1}. {display_name} ({user_id}) [{status}] 描述: {desc}\n"

        # 添加存储信息
        storage = summary['storage_info']
        detail_info += f"\n=== 存储信息 ===\n已用空间: {storage['used_space']} bytes\n配额限制: {storage['quota_limit']} bytes\n使用率: {storage['usage_percentage']:.1f}%\n"

        detail_info += f"\n总计处理会话: {summary['total_files']} 个"

        self.detail_text.setText(detail_info)
    
    def open_db_browser(self):
        dlg = DatabaseBrowserDialog(self)
        dlg.set_project_id(self.selected_project_id or "")
        dlg.exec_()
    
    def open_summary_pagination(self, initial_type: str = None):
        name = ""
        try:
            cur = self.project_list.currentItem()
            if cur:
                name = cur.text().split('\n')[0].split(' (')[0]
        except Exception:
            name = ""
        dlg = SummaryPaginationDialog(self)
        dlg.set_project(self.selected_project_id or "", name or "")
        if initial_type:
            dlg.type_combo.setCurrentText(initial_type)
            dlg.on_type_changed(initial_type)
        dlg.exec_()
    
    def set_current_project(self, project_id: str):
        """设置当前项目
        
        Args:
            project_id: 项目ID
        """
        self.selected_project_id = project_id
        self.load_projects()
        
        # 自动选中当前项目
        for i in range(self.project_list.count()):
            item = self.project_list.item(i)
            if item.data(Qt.UserRole) == project_id:
                self.project_list.setCurrentItem(item)
                # 触发选择事件处理，加载数据
                self.on_project_selected(item)
                break
    
    def apply_styles(self):
        """
        应用样式
        """
        self.setStyleSheet("""
            QWidget {
                background-color: #FFFFFF;
                font-family: 'Microsoft YaHei', Arial, sans-serif;
            }
            QGroupBox {
                font-weight: bold;
                border: 2px solid #E0E0E0;
                border-radius: 8px;
                margin-top: 10px;
                padding-top: 10px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px 0 5px;
                color: #2E86AB;
            }
        """)

class DatabaseBrowserDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("数据库浏览器")
        self.resize(1000, 700)
        self.db_service = DatabaseService()
        self.project_id = ""
        self.init_ui()
        self.load_tables()
    
    def init_ui(self):
        layout = QHBoxLayout(self)
        left = QVBoxLayout()
        right = QVBoxLayout()
        
        self.tables_tree = QTreeWidget()
        self.tables_tree.setHeaderLabels(["表名", "记录数"])
        self.tables_tree.header().setSectionResizeMode(0, QHeaderView.ResizeToContents)
        self.tables_tree.header().setSectionResizeMode(1, QHeaderView.ResizeToContents)
        self.tables_tree.itemClicked.connect(self.on_table_selected)
        left.addWidget(QLabel("数据库表"))
        left.addWidget(self.tables_tree)
        
        self.schema_label = QLabel("表结构")
        right.addWidget(self.schema_label)
        self.schema_view = QTextEdit()
        self.schema_view.setReadOnly(True)
        self.schema_view.setMinimumHeight(120)
        right.addWidget(self.schema_view)
        
        self.data_table = QTableWidget()
        self.data_table.setAlternatingRowColors(True)
        self.data_table.horizontalHeader().setStretchLastSection(True)
        right.addWidget(QLabel("数据"))
        right.addWidget(self.data_table)
        
        wrapper_left = QWidget()
        wrapper_left.setLayout(left)
        wrapper_right = QWidget()
        wrapper_right.setLayout(right)
        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(wrapper_left)
        splitter.addWidget(wrapper_right)
        left_w, right_w = splitter_sizes("step2")
        splitter.setSizes([left_w, right_w])
        layout.addWidget(splitter)
    
    def set_project_id(self, pid: str):
        self.project_id = pid or ""
    
    def load_tables(self):
        try:
            self.tables_tree.clear()
            with self.db_service.get_connection() as conn:
                cur = conn.cursor()
                cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%' ORDER BY name")
                tables = [row[0] for row in cur.fetchall()]
                for t in tables:
                    count = 0
                    try:
                        cur.execute(f"SELECT COUNT(*) FROM {t}")
                        count = cur.fetchone()[0]
                    except Exception:
                        count = 0
                    item = QTreeWidgetItem([t, str(count)])
                    self.tables_tree.addTopLevelItem(item)
        except Exception as e:
            QMessageBox.warning(self, "错误", f"加载表失败: {e}")
    
    def on_table_selected(self, item, col):
        table = item.text(0)
        self.show_table_schema(table)
        self.show_table_data(table)
    
    def show_table_schema(self, table: str):
        try:
            with self.db_service.get_connection() as conn:
                cur = conn.cursor()
                cur.execute(f"PRAGMA table_info({table})")
                cols = cur.fetchall()
                lines = []
                for c in cols:
                    lines.append(f"{c[1]} {c[2]}")
                self.schema_view.setPlainText("\n".join(lines))
        except Exception as e:
            self.schema_view.setPlainText(f"获取表结构失败: {e}")
    
    def show_table_data(self, table: str):
        try:
            with self.db_service.get_connection() as conn:
                cur = conn.cursor()
                cur.execute(f"PRAGMA table_info({table})")
                cols = [c[1] for c in cur.fetchall()]
                has_project = "project_id" in cols
                order_col = "created_at" if "created_at" in cols else None
                base_sql = f"SELECT * FROM {table}"
                params = ()
                if has_project and self.project_id:
                    base_sql += " WHERE project_id = ?"
                    params = (self.project_id,)
                if order_col:
                    base_sql += f" ORDER BY {order_col} DESC"
                base_sql += " LIMIT 200"
                cur.execute(base_sql, params)
                rows = cur.fetchall()
                self.data_table.clear()
                self.data_table.setColumnCount(len(cols))
                self.data_table.setHorizontalHeaderLabels(cols)
                self.data_table.setRowCount(len(rows))
                for r_i, row in enumerate(rows):
                    for c_i, col in enumerate(cols):
                        val = row[c_i]
                        s = "" if val is None else str(val)
                        if len(s) > 500:
                            s = s[:500] + "..."
                        self.data_table.setItem(r_i, c_i, QTableWidgetItem(s))
        except Exception as e:
            QMessageBox.warning(self, "错误", f"加载数据失败: {e}")
            
class SummaryPaginationDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("样本数据汇总与分页浏览")
        self.resize(1100, 720)
        self.db_service = DatabaseService()
        self.project_id = ""
        self.project_name = ""
        self.current_type = "DICOM会话"
        self.page_size = 25
        self.current_page = 1
        self.total_count = 0
        self.init_ui()
    
    def init_ui(self):
        layout = QVBoxLayout(self)
        summary_box = QGroupBox("汇总")
        s_layout = QVBoxLayout(summary_box)
        self.summary_table = QTableWidget()
        self.summary_table.setColumnCount(3)
        self.summary_table.setHorizontalHeaderLabels(["项目", "数据类型", "数量"])
        h = self.summary_table.horizontalHeader()
        h.setStretchLastSection(True)
        h.setSectionResizeMode(0, QHeaderView.ResizeToContents)
        h.setSectionResizeMode(1, QHeaderView.ResizeToContents)
        s_layout.addWidget(self.summary_table)
        layout.addWidget(summary_box)
        browse_box = QGroupBox("分页浏览")
        b_layout = QVBoxLayout(browse_box)
        ctrl = QHBoxLayout()
        self.type_combo = QComboBox()
        self.type_combo.addItems(["DICOM会话", "ROI数据", "OCR会话", "OCR结果", "文本会话", "文档", "样本描述"])
        self.count_label = QLabel("数量: 0")
        self.page_size_combo = QComboBox()
        self.page_size_combo.addItems(["10", "25", "50", "100"])
        self.page_size_combo.setCurrentText("25")
        self.prev_btn = QPushButton("上一页")
        self.next_btn = QPushButton("下一页")
        self.page_label = QLabel("第 1 页")
        ctrl.addWidget(QLabel("数据类型"))
        ctrl.addWidget(self.type_combo)
        ctrl.addStretch()
        ctrl.addWidget(QLabel("每页"))
        ctrl.addWidget(self.page_size_combo)
        ctrl.addWidget(self.count_label)
        ctrl.addWidget(self.prev_btn)
        ctrl.addWidget(self.next_btn)
        ctrl.addWidget(self.page_label)
        b_layout.addLayout(ctrl)
        self.data_table = QTableWidget()
        self.data_table.setAlternatingRowColors(True)
        self.data_table.horizontalHeader().setStretchLastSection(True)
        b_layout.addWidget(self.data_table)
        layout.addWidget(browse_box)
        self.type_combo.currentTextChanged.connect(self.on_type_changed)
        self.page_size_combo.currentTextChanged.connect(self.on_page_size_changed)
        self.prev_btn.clicked.connect(self.on_prev_page)
        self.next_btn.clicked.connect(self.on_next_page)
    
    def set_project(self, pid: str, pname: str):
        self.project_id = pid
        self.project_name = pname or pid
        self.load_summary_counts()
        self.current_page = 1
        self.current_type = self.type_combo.currentText()
        self.refresh_browse()
    
    def load_summary_counts(self):
        ps = ProjectService(self.db_service)
        dicom_sessions = self.db_service.execute_query("SELECT COUNT(*) AS c FROM dicom_sessions WHERE project_id=?", (self.project_id,))
        roi_count = self.db_service.execute_query("""
            SELECT COUNT(*) AS c FROM roi_data rd 
            JOIN dicom_sessions ds ON rd.session_id=ds.session_id
            WHERE ds.project_id=?""", (self.project_id,))
        ocr_sessions = self.db_service.execute_query("SELECT COUNT(*) AS c FROM ocr_sessions WHERE project_id=?", (self.project_id,))
        ocr_results = self.db_service.execute_query("""
            SELECT COUNT(*) AS c FROM ocr_results r 
            JOIN ocr_sessions s ON r.session_id=s.session_id
            WHERE s.project_id=?""", (self.project_id,))
        text_sessions = self.db_service.execute_query("SELECT COUNT(*) AS c FROM text_sessions WHERE project_id=?", (self.project_id,))
        documents = self.db_service.execute_query("""
            SELECT COUNT(*) AS c FROM documents d 
            JOIN text_sessions ts ON d.session_id=ts.session_id
            WHERE ts.project_id=?""", (self.project_id,))
        samples = self.db_service.execute_query("SELECT COUNT(*) AS c FROM user_ids WHERE project_id=?", (self.project_id,))
        items = [
            ("DICOM会话", dicom_sessions[0]['c'] if dicom_sessions else 0),
            ("ROI数据", roi_count[0]['c'] if roi_count else 0),
            ("OCR会话", ocr_sessions[0]['c'] if ocr_sessions else 0),
            ("OCR结果", ocr_results[0]['c'] if ocr_results else 0),
            ("文本会话", text_sessions[0]['c'] if text_sessions else 0),
            ("文档", documents[0]['c'] if documents else 0),
            ("样本描述", samples[0]['c'] if samples else 0),
        ]
        self.summary_table.setRowCount(len(items))
        for r, (dtype, cnt) in enumerate(items):
            self.summary_table.setItem(r, 0, QTableWidgetItem(self.project_name))
            self.summary_table.setItem(r, 1, QTableWidgetItem(dtype))
            self.summary_table.setItem(r, 2, QTableWidgetItem(str(cnt)))
    
    def on_type_changed(self, t):
        self.current_type = t
        self.current_page = 1
        self.refresh_browse()
    
    def on_page_size_changed(self, s):
        try:
            self.page_size = int(s)
        except:
            self.page_size = 25
        self.current_page = 1
        self.refresh_browse()
    
    def on_prev_page(self):
        if self.current_page > 1:
            self.current_page -= 1
            self.refresh_browse()
    
    def on_next_page(self):
        max_page = (self.total_count + self.page_size - 1) // self.page_size if self.page_size > 0 else 1
        if self.current_page < max_page:
            self.current_page += 1
            self.refresh_browse()
    
    def refresh_browse(self):
        ps = self.page_size
        off = (self.current_page - 1) * ps
        cols, rows, count = self.query_page(self.current_type, self.project_id, ps, off)
        self.total_count = count
        self.count_label.setText(f"数量: {count}")
        self.page_label.setText(f"第 {self.current_page} 页")
        self.data_table.clear()
        self.data_table.setColumnCount(len(cols))
        self.data_table.setHorizontalHeaderLabels(cols)
        self.data_table.setRowCount(len(rows))
        for ri, row in enumerate(rows):
            for ci, col in enumerate(cols):
                val = row.get(col, "")
                s = "" if val is None else str(val)
                if len(s) > 300:
                    s = s[:300] + "..."
                self.data_table.setItem(ri, ci, QTableWidgetItem(s))
    
    def query_page(self, dtype: str, pid: str, limit: int, offset: int):
        db = self.db_service
        if dtype == "DICOM会话":
            rc = db.execute_query("SELECT COUNT(*) AS c FROM dicom_sessions WHERE project_id=?", (pid,))
            count = rc[0]['c'] if rc else 0
            rows = db.execute_query("""
                SELECT session_id, file_name, file_size, created_at
                FROM dicom_sessions
                WHERE project_id=?
                ORDER BY created_at DESC
                LIMIT ? OFFSET ?
            """, (pid, limit, offset))
            cols = ["session_id", "file_name", "file_size", "created_at"]
            return cols, rows, count
        if dtype == "ROI数据":
            rc = db.execute_query("""
                SELECT COUNT(*) AS c FROM roi_data rd 
                JOIN dicom_sessions ds ON rd.session_id=ds.session_id
                WHERE ds.project_id=?""", (pid,))
            count = rc[0]['c'] if rc else 0
            rows = db.execute_query("""
                SELECT rd.roi_id, rd.session_id, rd.roi_type, rd.roi_name, rd.area, rd.source, rd.created_at
                FROM roi_data rd JOIN dicom_sessions ds ON rd.session_id=ds.session_id
                WHERE ds.project_id=?
                ORDER BY rd.created_at DESC
                LIMIT ? OFFSET ?
            """, (pid, limit, offset))
            cols = ["roi_id", "session_id", "roi_type", "roi_name", "area", "source", "created_at"]
            return cols, rows, count
        if dtype == "OCR会话":
            rc = db.execute_query("SELECT COUNT(*) AS c FROM ocr_sessions WHERE project_id=?", (pid,))
            count = rc[0]['c'] if rc else 0
            rows = db.execute_query("""
                SELECT session_id, image_path, created_at
                FROM ocr_sessions
                WHERE project_id=?
                ORDER BY created_at DESC
                LIMIT ? OFFSET ?
            """, (pid, limit, offset))
            cols = ["session_id", "image_path", "created_at"]
            return cols, rows, count
        if dtype == "OCR结果":
            rc = db.execute_query("""
                SELECT COUNT(*) AS c FROM ocr_results r 
                JOIN ocr_sessions s ON r.session_id=s.session_id
                WHERE s.project_id=?""", (pid,))
            count = rc[0]['c'] if rc else 0
            rows = db.execute_query("""
                SELECT r.result_id, r.session_id, s.image_path, r.recognized_text, r.confidence, r.created_at
                FROM ocr_results r JOIN ocr_sessions s ON r.session_id=s.session_id
                WHERE s.project_id=?
                ORDER BY r.created_at DESC
                LIMIT ? OFFSET ?
            """, (pid, limit, offset))
            cols = ["result_id", "session_id", "image_path", "recognized_text", "confidence", "created_at"]
            return cols, rows, count
        if dtype == "文本会话":
            rc = db.execute_query("SELECT COUNT(*) AS c FROM text_sessions WHERE project_id=?", (pid,))
            count = rc[0]['c'] if rc else 0
            rows = db.execute_query("""
                SELECT session_id, operation_type, file_path, created_at
                FROM text_sessions
                WHERE project_id=?
                ORDER BY created_at DESC
                LIMIT ? OFFSET ?
            """, (pid, limit, offset))
            cols = ["session_id", "operation_type", "file_path", "created_at"]
            return cols, rows, count
        if dtype == "文档":
            rc = db.execute_query("""
                SELECT COUNT(*) AS c FROM documents d 
                JOIN text_sessions ts ON d.session_id=ts.session_id
                WHERE ts.project_id=?""", (pid,))
            count = rc[0]['c'] if rc else 0
            rows = db.execute_query("""
                SELECT d.doc_id, d.session_id, d.file_name, d.file_size, d.created_at
                FROM documents d JOIN text_sessions ts ON d.session_id=ts.session_id
                WHERE ts.project_id=?
                ORDER BY d.created_at DESC
                LIMIT ? OFFSET ?
            """, (pid, limit, offset))
            cols = ["doc_id", "session_id", "file_name", "file_size", "created_at"]
            return cols, rows, count
        if dtype == "样本描述":
            rc = db.execute_query("SELECT COUNT(*) AS c FROM user_ids WHERE project_id=?", (pid,))
            count = rc[0]['c'] if rc else 0
            rows = db.execute_query("""
                SELECT user_id, display_name, description, is_active, created_at
                FROM user_ids
                WHERE project_id=?
                ORDER BY created_at DESC
                LIMIT ? OFFSET ?
            """, (pid, limit, offset))
            cols = ["user_id", "display_name", "description", "is_active", "created_at"]
            return cols, rows, count
        return [], [], 0
