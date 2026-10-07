#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
主窗口界面模块

实现应用程序的主窗口，包含用户ID管理界面和功能导航。
"""

import sys
import os
import glob
import re
import hashlib
import urllib.parse
from pathlib import Path
from PyQt5.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
    QStackedWidget, QPushButton, QLabel, QFrame,
    QMenuBar, QStatusBar, QAction, QMessageBox, QDialog,
    QGridLayout, QSizePolicy
)
from PyQt5.QtCore import Qt, pyqtSignal, QSize, QPoint, QTimer, QThread
from PyQt5.QtGui import QIcon, QFont, QPalette, QColor

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from ui.widgets.project_management_widget import ProjectManagementWidget
from ui.widgets.dicom_roi_widget import DicomRoiWidget
from ui.widgets.text_processing_widget import TextProcessingWidget
from ui.widgets.ocr_widget import OCRWidget
from ui.widgets.magic_seg_widget import MagicSegWidget
from ui.widgets.text_processing_widget import TextProcessingWidget
from ui.widgets.ai_summary_widget import AISummaryWidget
from ui.widgets.scientific_preview_widget import ScientificPreviewWidget
from ui.widgets.feedback_optimization_widget import FeedbackOptimizationWidget
from ui.widgets.model_loading_overlay import ModelLoadingOverlay, route_loading_message
from src.core.app_config import AppConfig
from src.utils.logger import get_logger
from src.utils.gpu_manager import gpu_manager
from ui.widgets.license_activation_dialog import LicenseActivationDialog
from src.services.project_service import ProjectService


class SklearnBackfillWorker(QThread):
    finished_report = pyqtSignal(dict)

    def __init__(self, project_ids, project_root):
        super().__init__()
        self.project_ids = list(project_ids or [])
        self.project_root = project_root

    def run(self):
        report = {
            "scanned": 0,
            "eligible": 0,
            "generated": 0,
            "failed": 0,
            "fail_reasons": [],
        }
        try:
            from src.services.llm_guided_sklearn_pipeline import run_llm_guided_sklearn_pipeline
        except Exception as e:
            report["failed"] = len(self.project_ids)
            report["fail_reasons"].append(f"pipeline import error: {e}")
            self.finished_report.emit(report)
            return

        workspace = os.path.join(self.project_root, "workspace")
        for pid in self.project_ids:
            report["scanned"] += 1
            try:
                base = os.path.join(self.project_root, "project_storage", "projects", pid)
                tab_dir = os.path.join(base, "tabular")
                ex_dir = os.path.join(base, "exports")
                tab_cnt = len(glob.glob(os.path.join(tab_dir, "**", "*.json"), recursive=True))
                ex_cnt = len(glob.glob(os.path.join(ex_dir, "*.png")))
                if tab_cnt <= 0 or ex_cnt > 0:
                    continue
                report["eligible"] += 1
                res = run_llm_guided_sklearn_pipeline(
                    llm_service=None,
                    model="",
                    materials_text="",
                    project_storage_path=base,
                    workspace=workspace,
                    out_dir=ex_dir,
                )
                if bool(res.get("success")) and len(res.get("artifacts") or []) > 0:
                    report["generated"] += 1
                else:
                    report["failed"] += 1
                    report["fail_reasons"].append(f"{pid}: {res.get('error') or 'unknown'}")
            except Exception as e:
                report["failed"] += 1
                report["fail_reasons"].append(f"{pid}: {e}")
        self.finished_report.emit(report)


class MainWindow(QMainWindow):
    """
    主窗口类
    """
    
    # 信号定义
    project_switched = pyqtSignal(str)  # 项目切换信号
    function_selected = pyqtSignal(str)  # 功能选择信号
    
    def __init__(self, remaining_seconds: int = 0, license_exp_ts: int = 0):
        super().__init__()
        
        self.config = AppConfig()
        self.logger = get_logger(__name__)
        self.current_project_id = None
        self._batch_mode = False
        self._batch_project_queue = []
        self._batch_all_project_ids = []
        self._sklearn_backfill_worker = None
        self.sample_management_dialog = None  # 样本管理对话框实例
        self._drag_active = False
        self._drag_offset = QPoint(0, 0)
        self.remaining_seconds = int(remaining_seconds)
        self.license_remaining_days = max(0, self.remaining_seconds // 86400)
        self.license_exp_ts = int(license_exp_ts or 0)
        
        self.init_ui()
    
    def setup_window_geometry(self):
        """
        设置窗口几何，支持多显示器环境
        """
        try:
            from PyQt5.QtWidgets import QDesktopWidget
            desktop = QDesktopWidget()
            primary_screen = desktop.primaryScreen()
            screen_geometry = desktop.screenGeometry(primary_screen)
            
            base_w, base_h = 1920, 1080
            avail_w, avail_h = screen_geometry.width(), screen_geometry.height()
            scale = min(avail_w / base_w, avail_h / base_h)
            if scale > 1.0:
                scale = 1.0
            target_w = int(base_w * scale)
            target_h = int(base_h * scale)
            
            self.resize(target_w, target_h)
            center_x = screen_geometry.x() + (avail_w - target_w) // 2
            center_y = screen_geometry.y() + (avail_h - target_h) // 2
            self.move(center_x, center_y)
            self.logger.info(f"窗口使用1080比例加载: {target_w}x{target_h}，居中到主显示器 ({center_x}, {center_y})")
        except Exception as e:
            self.logger.error(f"设置窗口几何失败: {e}")
            # 降级处理：使用默认大小和位置
            self.resize(1200, 800)
            self.move(100, 100)
    
    def init_ui(self):
        """
        初始化用户界面
        """
        # Force title update regardless of config cache
        app_title = '医疗数据科学结构化工作流系统（Designed by Christ.paul90@gmail.com. All rights reserved.）'
        self.setWindowTitle(app_title)
        # Ensure config is updated in memory too so it saves correctly later
        self.config.set('app.name', app_title)
        
        # 设置窗口大小和位置（支持多显示器）
        self.setup_window_geometry()
        
        # 设置窗口图标
        icon_candidates = []
        try:
            if getattr(sys, "frozen", False):
                base_dir = os.path.dirname(sys.executable)
                internal_dir = getattr(sys, "_MEIPASS", os.path.join(base_dir, "_internal"))
                icon_candidates.extend(
                    [
                        os.path.join(base_dir, "_internal", "assets", "icon.ico"),
                        os.path.join(base_dir, "_internal", "assets", "medlogo.png"),
                        os.path.join(internal_dir, "assets", "icon.ico"),
                        os.path.join(internal_dir, "assets", "medlogo.png"),
                        os.path.join(base_dir, "assets", "icon.ico"),
                        os.path.join(base_dir, "assets", "medlogo.png"),
                        os.path.join(os.getcwd(), "assets", "icon.ico"),
                        os.path.join(os.getcwd(), "assets", "medlogo.png"),
                    ]
                )
        except Exception:
            pass
        icon_candidates.extend(["assets/icon.ico", "assets/medlogo.png", "assets/icons/app_icon.png"])
        for p in icon_candidates:
            if p and os.path.exists(p):
                self.setWindowIcon(QIcon(p))
                break
        
        # 创建中央部件
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # 主布局
        main_layout = QVBoxLayout(central_widget)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)
        
        # 创建标题栏
        self.create_title_bar(main_layout)
        
        # 创建内容区域
        self.create_content_area(main_layout)
        
        # 创建菜单栏
        self.create_menu_bar()
        
        # 创建状态栏
        self.create_status_bar()
        self.model_loading_overlay = ModelLoadingOverlay(self)
        
        # 应用样式
        self.apply_styles()
        
        # 设置信号连接
        self.setup_connections()
        
        # 显示项目管理界面
        self.show_project_management()
        try:
            if hasattr(self, "license_label") and self.license_label:
                exp_suffix = ""
                if self.license_exp_ts > 0:
                    import time
                    exp_suffix = f"（截止 {time.strftime('%Y-%m-%d', time.localtime(self.license_exp_ts))}）"
                if self.license_remaining_days > 0:
                    self.license_label.setText(f"授权剩余: {self.license_remaining_days} 天{exp_suffix}")
                    self.status_bar.showMessage(f'项目管理 | 授权剩余: {self.license_remaining_days} 天{exp_suffix}')
                else:
                    hrs = max(0, self.remaining_seconds // 3600)
                    self.license_label.setText(f"授权剩余: {hrs} 小时{exp_suffix}")
                    self.status_bar.showMessage(f'项目管理 | 授权剩余: {hrs} 小时{exp_suffix}')
        except Exception:
            pass
        if self.remaining_seconds <= 0:
            dlg = LicenseActivationDialog(self)
            if dlg.exec_() == QDialog.Accepted and dlg.token_written:
                self.license_remaining_days = dlg.remaining_days
                self.remaining_seconds = max(0, self.license_remaining_days * 86400)
                if hasattr(self, "license_label") and self.license_label:
                    if self.license_remaining_days > 0:
                        self.license_label.setText(f"授权剩余: {self.license_remaining_days} 天")
                        self.status_bar.showMessage(f'项目管理 | 授权剩余: {self.license_remaining_days} 天')
                    else:
                        hrs = max(0, self.remaining_seconds // 3600)
                        self.license_label.setText(f"授权剩余: {hrs} 小时")
                        self.status_bar.showMessage(f'项目管理 | 授权剩余: {hrs} 小时')
            else:
                os._exit(1)
    
    def create_title_bar(self, parent_layout):
        """
        创建标题栏
        
        Args:
            parent_layout: 父布局
        """
        title_frame = QFrame()
        title_frame.setFixedHeight(80)
        title_frame.setStyleSheet("""
            QFrame {
                background-color: #2E86AB;
                border: none;
            }
        """)
        self._title_frame = title_frame
        self._title_frame.installEventFilter(self)
        
        title_layout = QHBoxLayout(title_frame)
        title_layout.setContentsMargins(20, 10, 20, 10)
        
        # 应用程序标题
        # Force title update regardless of config cache
        app_title = '医疗数据科学结构化工作流系统（Designed by Christ.paul90@gmail.com. All rights reserved.）'
        title_label = QLabel(app_title)
        title_label.setStyleSheet("""
            QLabel {
                color: white;
                font-size: 24px;
                font-weight: bold;
                font-family: 'Microsoft YaHei';
            }
        """)
        
        # 项目信息标签
        self.project_info_label = QLabel("未选择项目")
        self.project_info_label.setStyleSheet("""
            QLabel {
                color: white;
                font-size: 14px;
                font-family: 'Microsoft YaHei';
            }
        """)
        
        title_layout.addWidget(title_label)
        title_layout.addStretch()
        title_layout.addWidget(self.project_info_label)
        
        parent_layout.addWidget(title_frame)
    
    def create_content_area(self, parent_layout):
        """
        创建内容区域
        
        Args:
            parent_layout: 父布局
        """
        # 创建堆叠部件来管理不同的界面
        self.stacked_widget = QStackedWidget()
        
        # 创建各个功能界面
        self.project_management_widget = ProjectManagementWidget()
        self.dicom_roi_widget = DicomRoiWidget(self.config)
        self.text_processing_widget = TextProcessingWidget(self.config)
        self.ocr_recognition_widget = OCRWidget(self.config)
        self.magic_seg_widget = MagicSegWidget()
        self.ai_summary_widget = AISummaryWidget()
        self.text_preview_widget = ScientificPreviewWidget(self.config)
        self.feedback_optimization_widget = FeedbackOptimizationWidget(self.config)
        
        # 导入并创建数据清洗组件
        from ui.widgets.data_cleaning_widget import DataCleaningWidget
        self.data_cleaning_widget = DataCleaningWidget()
        
        # 添加到堆叠部件
        self.stacked_widget.addWidget(self.project_management_widget)
        self.stacked_widget.addWidget(self.dicom_roi_widget)
        self.stacked_widget.addWidget(self.text_processing_widget)
        self.stacked_widget.addWidget(self.ocr_recognition_widget)
        self.stacked_widget.addWidget(self.magic_seg_widget)
        self.stacked_widget.addWidget(self.data_cleaning_widget)
        self.stacked_widget.addWidget(self.ai_summary_widget)
        self.stacked_widget.addWidget(self.text_preview_widget)
        self.stacked_widget.addWidget(self.feedback_optimization_widget)
        
        parent_layout.addWidget(self.stacked_widget)
    
    def create_menu_bar(self):
        """
        创建菜单栏
        """
        menubar = self.menuBar()
        
        # 文件菜单
        file_menu = menubar.addMenu('系统(&F)')
        
        exit_action = QAction('退出(&X)', self)
        exit_action.setShortcut('Ctrl+Q')
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)
        
        # 功能菜单
        function_menu = menubar.addMenu('第一步数据整理(&F)')
        
        project_mgmt_action = QAction('项目管理(&P)', self)
        project_mgmt_action.setShortcut('Ctrl+P')
        project_mgmt_action.triggered.connect(self.show_project_management)
        function_menu.addAction(project_mgmt_action)
        
        dicom_action = QAction('DICOM ROI识别(&D)', self)
        dicom_action.triggered.connect(self.show_dicom_roi)
        function_menu.addAction(dicom_action)
        
        text_action = QAction('文本处理(&T)', self)
        text_action.triggered.connect(self.show_text_processing)
        function_menu.addAction(text_action)
        
        ocr_action = QAction('OCR识别(&O)', self)
        ocr_action.triggered.connect(self.show_ocr_recognition)
        function_menu.addAction(ocr_action)
        
        magic_seg_action = QAction('Magic Seg(&M)', self)
        magic_seg_action.triggered.connect(self.show_magic_seg)
        function_menu.addAction(magic_seg_action)
        
        # Add Step 2 menu
        function_menu.addSeparator()
        
        sample_action = QAction('样本管理(&S)', self)
        sample_action.triggered.connect(self.show_sample_management)
        function_menu.addAction(sample_action)
        
        # 第二步数据清洗及结构化菜单
        data_cleaning_menu = menubar.addMenu('第二步数据清洗及结构化(&C)')
        
        # 添加数据清洗功能
        data_cleaning_action = QAction('数据清洗及结构化(&D)', self)
        data_cleaning_action.triggered.connect(self.show_data_cleaning)
        data_cleaning_menu.addAction(data_cleaning_action)
        
        # 第三步本地大模型数据科学总结菜单
        ai_summary_menu = menubar.addMenu('第三步本地大模型数据科学总结(&A)')
        ai_summary_action = QAction('数据科学总结(&A)', self)
        ai_summary_action.triggered.connect(self.show_ai_summary)
        ai_summary_menu.addAction(ai_summary_action)
        
        # 第四步科学文本预览生成菜单
        text_preview_menu = menubar.addMenu('第四步科学文本预览生成(&P)')
        
        # 添加功能项
        self.text_preview_action = QAction('预览生成(&P)', self)
        # self.text_preview_action.setEnabled(False) # Removed blocking
        self.text_preview_action.triggered.connect(self.show_text_preview)
        text_preview_menu.addAction(self.text_preview_action)
        
        # 第五步科学文本研究者真实反馈及迭代优化菜单
        feedback_optimization_menu = menubar.addMenu('第五步科学文本研究者真实反馈及迭代优化(&R)')
        
        # 添加功能项
        self.feedback_action = QAction('迭代优化仪表盘(&D)', self)
        # self.feedback_action.setEnabled(False) # Removed blocking for testing
        self.feedback_action.triggered.connect(self.show_feedback_optimization)
        feedback_optimization_menu.addAction(self.feedback_action)
        
        # 帮助菜单
        help_menu = menubar.addMenu('帮助(&H)')
        
        # 视图菜单 (主题切换)
        view_menu = menubar.addMenu('视图(&V)')
        self.theme_action = QAction('切换深色/浅色主题', self)
        self.theme_action.triggered.connect(self.toggle_theme)
        view_menu.addAction(self.theme_action)
        
        about_action = QAction('关于(&A)', self)
        about_action.triggered.connect(self.show_about)
        help_menu.addAction(about_action)
    
    def create_status_bar(self):
        """
        创建状态栏
        """
        self.status_bar = self.statusBar()
        self.status_bar.showMessage('就绪')
        try:
            self.gpu_owner_label = QLabel("GPU: 空闲")
            self.status_bar.addPermanentWidget(self.gpu_owner_label)
            self.license_label = QLabel("")
            self.status_bar.addPermanentWidget(self.license_label)
        except Exception:
            pass

    def set_model_loading_status(self, message: str):
        owner = gpu_manager.get_current_owner_label()
        try:
            if hasattr(self, "gpu_owner_label") and self.gpu_owner_label:
                self.gpu_owner_label.setText(f"GPU: {owner}")
        except Exception:
            pass
        base = str(message or "就绪").strip() or "就绪"
        try:
            self.status_bar.showMessage(f"{base} | 当前 GPU 持有者: {owner}")
        except Exception:
            pass

    def on_model_status_message(self, message: str):
        text = str(message or "").strip()
        if not text:
            return
        route_loading_message(self.model_loading_overlay, text)
        self.set_model_loading_status(text)
    
    def toggle_theme(self):
        """
        切换全局明暗主题
        """
        from PyQt5.QtWidgets import QApplication
        from src.utils.theme_manager import ThemeManager
        import subprocess
        import sys
        
        current = ThemeManager.get_current_theme()
        new_theme = "light" if current == "dark" else "dark"
        
        # Update config and save to disk
        self.config.set("theme", new_theme)
        if hasattr(self.config, 'save'):
            self.config.save()
        
        # Require restart to fully apply (since many widgets have hardcoded styles applied on init)
        from PyQt5.QtWidgets import QMessageBox
        reply = QMessageBox.question(
            self,
            "切换主题",
            f"您已选择切换到 {'浅色' if new_theme == 'light' else '深色'} 主题。\n\n由于大量组件样式在启动时生成，主题切换需要重启程序才能完全生效。\n\n是否立即重启程序？",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.Yes
        )
        
        if reply == QMessageBox.Yes:
            # Restart app
            import os
            self.logger.info(f"Restarting app to apply new theme: {new_theme}")
            
            # Close all windows
            QApplication.instance().quit()
            
            # Relaunch
            if getattr(sys, 'frozen', False):
                subprocess.Popen([sys.executable] + sys.argv[1:])
            else:
                subprocess.Popen([sys.executable, sys.argv[0]] + sys.argv[1:])
            
            # Force exit current process immediately
            os._exit(0)

    def apply_styles(self):
        """
        应用样式
        """
        self.setStyleSheet("""
            QMainWindow {
                background-color: #F5F5F5;
            }
            QMenuBar {
                background-color: #FFFFFF;
                border-bottom: 1px solid #E0E0E0;
                font-family: 'Microsoft YaHei';
                font-size: 12px;
            }
            QMenuBar::item {
                padding: 5px 10px;
                background-color: transparent;
            }
            QMenuBar::item:selected {
                background-color: #E3F2FD;
            }
            QStatusBar {
                background-color: #FFFFFF;
                border-top: 1px solid #E0E0E0;
                font-family: 'Microsoft YaHei';
                font-size: 12px;
            }
        """)
    
    def setup_connections(self):
        """
        设置信号连接
        """
        # 连接项目管理界面的信号
        self.project_management_widget.project_switched.connect(self.on_project_switched)
        self.project_management_widget.function_selected.connect(self.on_function_selected)
        self.project_management_widget.batch_process_requested.connect(self.on_batch_process_requested)
        if hasattr(self.data_cleaning_widget, 'project_selected'):
            self.data_cleaning_widget.project_selected.connect(self.on_data_cleaning_project_selected)
        
        # Connect AI Summary completion to Text Preview enabling
        self.ai_summary_widget.analysis_completed.connect(self.enable_text_preview)
        
        # Connect Text Preview completion to Feedback Optimization
        self.text_preview_widget.preview_completed.connect(self.on_text_preview_completed)

        try:
            self.logger.info("主窗口事件连接完成")
        except Exception:
            pass
    
    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton and event.pos().y() <= 80:
            self._drag_active = True
            self._drag_offset = event.globalPos() - self.frameGeometry().topLeft()
        super().mousePressEvent(event)
    
    def mouseMoveEvent(self, event):
        if self._drag_active:
            self.move(event.globalPos() - self._drag_offset)
        super().mouseMoveEvent(event)
    
    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton:
            self._drag_active = False
        super().mouseReleaseEvent(event)
    
    def eventFilter(self, obj, event):
        from PyQt5.QtCore import QEvent
        if obj is getattr(self, "_title_frame", None):
            if event.type() == QEvent.MouseButtonPress and event.button() == Qt.LeftButton:
                self._drag_active = True
                self._drag_offset = event.globalPos() - self.frameGeometry().topLeft()
                return True
            if event.type() == QEvent.MouseMove and self._drag_active:
                self.move(event.globalPos() - self._drag_offset)
                return True
            if event.type() == QEvent.MouseButtonRelease and event.button() == Qt.LeftButton:
                self._drag_active = False
                return True
        return super().eventFilter(obj, event)
    
    def on_project_switched(self, project_id: str):
        """
        项目切换处理
        
        Args:
            project_id: 项目ID
        """
        self.current_project_id = project_id
        
        # 获取项目信息并更新显示
        try:
            from src.services.project_service import ProjectService
            project_service = ProjectService()
            project = project_service.get_project_by_id(project_id)
            if project:
                self.project_info_label.setText(f"当前项目: {project.name}")
            else:
                self.project_info_label.setText(f"项目ID: {project_id}")
        except Exception as e:
            self.logger.error(f"获取项目信息失败: {e}")
            self.project_info_label.setText(f"项目ID: {project_id}")
        
        # 更新样本管理对话框的项目ID
        if self.sample_management_dialog and self.sample_management_dialog.isVisible():
            self.sample_management_dialog.set_project_id(project_id)
        
        self.project_switched.emit(project_id)
        self.logger.info(f"切换到项目: {project_id}")
    
    def on_function_selected(self, function_name: str):
        """
        功能选择处理
        
        Args:
            function_name: 功能名称
        """
        if function_name == 'dicom_roi':
            self.show_dicom_roi()
        elif function_name == 'text_processing':
            self.show_text_processing()
        elif function_name == 'ocr_recognition':
            self.show_ocr_recognition()
        elif function_name == 'sample_management':
            self.show_sample_management()
    
    def show_project_management(self):
        """
        显示项目管理界面
        """
        self.stacked_widget.setCurrentWidget(self.project_management_widget)
        if hasattr(self, "license_label") and self.license_label and isinstance(self.license_remaining_days, int):
            self.status_bar.showMessage(f'项目管理 | 授权剩余: {self.license_remaining_days} 天')
        else:
            self.status_bar.showMessage('项目管理')
    
    def show_dicom_roi(self):
        """
        显示DICOM ROI识别界面
        """
        if self.current_project_id is None:
            QMessageBox.warning(self, '警告', '请先选择项目！')
            return
        
        # 设置当前项目到DICOM ROI组件
        if hasattr(self.dicom_roi_widget, 'set_current_project'):
            self.dicom_roi_widget.set_current_project(self.current_project_id)
        elif hasattr(self.dicom_roi_widget, 'set_current_user'):
            # 兼容旧版本接口
            self.dicom_roi_widget.set_current_user(self.current_project_id)
        
        self.stacked_widget.setCurrentWidget(self.dicom_roi_widget)
        self.status_bar.showMessage('DICOM ROI识别')
    
    def show_text_processing(self):
        """
        显示文本处理界面
        """
        if self.current_project_id is None:
            QMessageBox.warning(self, '警告', '请先选择项目！')
            return
        
        # 设置当前项目到文本处理组件
        if hasattr(self.text_processing_widget, 'set_current_project'):
            self.text_processing_widget.set_current_project(self.current_project_id)
        elif hasattr(self.text_processing_widget, 'set_current_user'):
            # 兼容旧版本接口
            self.text_processing_widget.set_current_user(self.current_project_id)
        
        self.stacked_widget.setCurrentWidget(self.text_processing_widget)
        self.status_bar.showMessage('文本处理')
    
    def show_ocr_recognition(self):
        """
        显示OCR识别界面
        """
        if self.current_project_id is None:
            QMessageBox.warning(self, '警告', '请先选择项目！')
            return
        
        # 设置当前项目到OCR识别组件
        if hasattr(self.ocr_recognition_widget, 'set_current_project'):
            self.ocr_recognition_widget.set_current_project(self.current_project_id)
        elif hasattr(self.ocr_recognition_widget, 'set_current_user'):
            # 兼容旧版本接口
            self.ocr_recognition_widget.set_current_user(self.current_project_id)
        
        self.stacked_widget.setCurrentWidget(self.ocr_recognition_widget)
        self.set_model_loading_status('OCR识别')
    
    def show_sample_management(self):
        """
        显示样本管理对话框
        """
        if self.current_project_id is None:
            QMessageBox.warning(self, '警告', '请先选择项目！')
            return
        
        # 如果对话框不存在或已关闭，创建新的实例
        if self.sample_management_dialog is None or not self.sample_management_dialog.isVisible():
            from ui.widgets.sample_management_dialog import SampleManagementDialog
            self.sample_management_dialog = SampleManagementDialog(self.current_project_id, self)
        else:
            # 如果对话框已存在且可见，更新项目ID并激活窗口
            self.sample_management_dialog.set_project_id(self.current_project_id)
            self.sample_management_dialog.activateWindow()
            self.sample_management_dialog.raise_()
            return
        
        self.sample_management_dialog.exec_()

    def show_magic_seg(self):
        """
        显示 Magic Seg 界面
        """
        if self.current_project_id is None:
            QMessageBox.warning(self, '警告', '请先选择项目！')
            return
            
        if hasattr(self.magic_seg_widget, 'set_current_project'):
            self.magic_seg_widget.set_current_project(self.current_project_id)
            
        self.stacked_widget.setCurrentWidget(self.magic_seg_widget)
        self.set_model_loading_status('Magic Seg')
    
    def show_data_cleaning(self):
        """
        显示数据清洗及结构化界面
        """
        # 设置当前项目到数据清洗组件
        if hasattr(self.data_cleaning_widget, 'set_current_project'):
            self.data_cleaning_widget.set_current_project(self.current_project_id)
        
        self.stacked_widget.setCurrentWidget(self.data_cleaning_widget)
        self.status_bar.showMessage('数据清洗及结构化')
    
    def show_about(self):
        """
        显示关于对话框
        """
        about_text = f"""
        <h3>{self.config.get('app.name')}</h3>
        <p><b>版本 / Version:</b> {self.config.get('app.version')} &nbsp;(Release V1.0)</p>
        <p><b>作者 / Author:</b> {self.config.get('app.author')}</p>
        <p>{self.config.get('app.description')}</p>
        <hr>
        <p><b>核心功能</b></p>
        <ul>
          <li>DICOM 影像 ROI 识别与标注（SAM 交互式分割）</li>
          <li>数据清洗与结构化（表格 / 文献 / OCR）</li>
          <li>大模型数据科学总结（本地 Ollama 或外部 API 可选）</li>
          <li>科学文本预览生成（文档 / PPT / PDF 模板）</li>
          <li>研究者真实反馈与迭代优化</li>
        </ul>
        <p><b>推理后端：</b>支持本地 Ollama 服务，或任意 OpenAI 兼容 API 服务</p>
        <p><b>项目主页：</b>https://github.com/hg3992260/medical_imaging_workflow</p>
        <p>基于 Python + PyQt5 构建</p>
        """
        
        QMessageBox.about(self, '关于', about_text)
    
    def show_ai_summary(self):
        if self.current_project_id is None:
            QMessageBox.warning(self, '警告', '请先选择项目！')
            return
        try:
            from src.services.ocr_service import get_ocr_service
            svc = get_ocr_service()
            svc.release_gpu_resources()
        except Exception:
            pass
        try:
            from src.services.magic_seg_service import MagicSegService
            MagicSegService().release_gpu_resources()
        except Exception:
            pass
        if hasattr(self.data_cleaning_widget, 'selected_project_id') and self.data_cleaning_widget.selected_project_id:
            self.current_project_id = self.data_cleaning_widget.selected_project_id
            project_name = self.data_cleaning_widget.project_list.currentItem().text().split('\n')[0].split(' (')[0] if self.data_cleaning_widget.project_list.currentItem() else self.project_info_label.text().replace("当前项目: ", "")
        else:
            project_name = self.project_info_label.text().replace("当前项目: ", "")
        if hasattr(self.ai_summary_widget, 'set_current_project'):
            self.ai_summary_widget.set_current_project(self.current_project_id, project_name)
        self.stacked_widget.setCurrentWidget(self.ai_summary_widget)
        self.set_model_loading_status('本地大模型数据科学总结')

    def enable_text_preview(self, result):
        """Enable Step 4 menu and store result"""
        self.text_preview_action.setEnabled(True)
        self.logger.info("Step 4 enabled. Result received.")
        # We can store result here or just pass it when showing
        self.last_analysis_result = result
        if self._batch_mode:
            # 批处理模式：自动推进 Step4 -> Step5
            self.show_text_preview()
            self.text_preview_widget.on_finish_clicked()
            return
        
        # Optionally prompt user
        reply = QMessageBox.question(self, '分析完成', 
                                   '科学分析已完成！是否立即进入“第四步科学文本预览生成”？',
                                   QMessageBox.Yes | QMessageBox.No, QMessageBox.Yes)
        if reply == QMessageBox.Yes:
            self.show_text_preview()

    def show_text_preview(self):
        """Show Step 4 interface"""
        # Inject the currently active LLM service (Ollama or Custom API) from Step3
        try:
            active_svc = self.ai_summary_widget._active_llm_service()
            active_model = self.ai_summary_widget._current_model_name()
            active_label = self.ai_summary_widget._current_service_label()
            self.text_preview_widget.set_llm_service(active_svc, service_label=active_label, model=active_model)
        except Exception:
            pass

        # Set current project for CGR data scope
        if self.current_project_id:
            self.text_preview_widget.set_current_project(self.current_project_id)
            
        if hasattr(self, 'last_analysis_result') and self.last_analysis_result:
            self.text_preview_widget.load_data(self.last_analysis_result)
        else:
            # Show "No Data" state instead of blocking
            self.text_preview_widget.show_no_data_state()
            
        self.stacked_widget.setCurrentWidget(self.text_preview_widget)
        self.status_bar.showMessage('科学文本预览生成 (HITL)')
    
    def on_text_preview_completed(self, text, metadata):
        """
        Handle Step 4 completion -> Proceed to Step 5
        """
        self.logger.info("Received final draft from Step 4. Proceeding to Step 5.")
        
        # 1. Ensure Step 5 widget has the correct project ID set BEFORE loading draft
        # This prevents the draft from being saved under 'default' project and then disappearing
        if self.current_project_id:
            self.feedback_optimization_widget.set_current_project(self.current_project_id)
        
        # 2. Pass data to Step 5 (This saves to DB and updates UI)
        self.feedback_optimization_widget.load_draft(text, metadata)
        
        # 3. Switch view
        self.show_feedback_optimization()
        if self._batch_mode:
            # 批处理模式下，单项目完成后立即导出，避免整批未结束时看不到 Word
            try:
                if self.current_project_id:
                    one = self._export_project_drafts_to_word(self.current_project_id)
                    self.logger.info(
                        f"Immediate Word export for {self.current_project_id}: "
                        f"success={one.get('success', 0)}, failed={one.get('failed', 0)}"
                    )
            except Exception as e:
                self.logger.warning(f"Immediate Word export failed for {self.current_project_id}: {e}")
            self.logger.info(f"Batch completed one project: {self.current_project_id}")
            # 批处理项目间隔，防止模型热状态抖动
            QTimer.singleShot(800, self._start_next_batch_project)
            return

        QMessageBox.information(self, "Step 4 Completed", 
                              "Draft saved! Now entering Step 5: Iterative Optimization Dashboard.")

    def on_batch_process_requested(self, project_ids):
        if not project_ids:
            return
        self._batch_project_queue = list(project_ids)
        self._batch_all_project_ids = list(project_ids)
        self._batch_mode = True
        try:
            self.text_preview_widget.set_auto_approve_mode(True)
        except Exception:
            pass
        QMessageBox.information(self, "批处理已启动", f"将自动执行 Step2-Step5，共 {len(self._batch_project_queue)} 个项目。")
        self._start_next_batch_project()

    def _start_next_batch_project(self):
        if not self._batch_project_queue:
            self._batch_mode = False
            try:
                self.text_preview_widget.set_auto_approve_mode(False)
            except Exception:
                pass
            export_report = self._export_batch_drafts_to_word()
            self._start_sklearn_backfill_for_batch()
            QMessageBox.information(
                self,
                "批处理完成",
                (
                    "所有项目已自动完成 Step2-Step5，草稿已存储。\n"
                    f"Word导出：成功 {export_report.get('success', 0)}，失败 {export_report.get('failed', 0)}。\n"
                    "后台将自动补偿未生成的 sklearn 插图。"
                ),
            )
            return

        pid = self._batch_project_queue.pop(0)
        self.on_project_switched(pid)
        # Step2
        self.show_data_cleaning()
        # Step3
        self.show_ai_summary()
        # 默认 deepseek R1 8B
        try:
            combo = self.ai_summary_widget.model_combo
            target_idx = -1
            for i in range(combo.count()):
                tx = combo.itemText(i).lower()
                if "deepseek" in tx and "r1" in tx and "8b" in tx:
                    target_idx = i
                    break
            if target_idx >= 0:
                combo.setCurrentIndex(target_idx)
        except Exception:
            pass
        # 默认开启 docs
        try:
            self.ai_summary_widget.docs_kb_checkbox.setChecked(True)
        except Exception:
            pass
        # 模板默认采用当前已加载模板
        self.ai_summary_widget.on_analyze_clicked()

    def _start_sklearn_backfill_for_batch(self):
        if not self._batch_all_project_ids:
            return
        if self._sklearn_backfill_worker is not None and self._sklearn_backfill_worker.isRunning():
            return
        self._sklearn_backfill_worker = SklearnBackfillWorker(
            project_ids=self._batch_all_project_ids,
            project_root=os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
        )
        self._sklearn_backfill_worker.finished_report.connect(self._on_sklearn_backfill_finished)
        self._sklearn_backfill_worker.start()

    def _on_sklearn_backfill_finished(self, report: dict):
        try:
            scanned = int(report.get("scanned", 0))
            eligible = int(report.get("eligible", 0))
            generated = int(report.get("generated", 0))
            failed = int(report.get("failed", 0))
            self.logger.info(
                f"Sklearn backfill finished: scanned={scanned}, eligible={eligible}, generated={generated}, failed={failed}"
            )
            reasons = report.get("fail_reasons") or []
            if reasons:
                self.logger.warning("Sklearn backfill failures: " + " | ".join(reasons[:5]))
        except Exception:
            pass

    def _export_batch_drafts_to_word(self) -> dict:
        """批处理完成后，将每个项目的草稿按 Step3 左右布局导出为 Word 到对应二级目录。"""
        out = {"success": 0, "failed": 0}
        if not self._batch_all_project_ids:
            return out
        for pid in self._batch_all_project_ids:
            one = self._export_project_drafts_to_word(pid)
            # Fix: Extract nested dict values correctly without appending dict object
            out["success"] += int(one.get("success", 0))
            out["failed"] += int(one.get("failed", 0))
        return out

    def _export_project_drafts_to_word(self, pid: str) -> dict:
        """导出单个项目的草稿到其 batch_folder。"""
        out = {"success": 0, "failed": 0}
        try:
            from docx import Document
            from docx.shared import Inches
        except Exception as e:
            self.logger.error(f"Word 导出失败：python-docx 不可用: {e}")
            out["failed"] = 1
            return out

        project_service = ProjectService()

        def _safe_name(s: str) -> str:
            s = re.sub(r"[\\/:*?\"<>|]+", "_", str(s or "").strip())
            return s[:80] if s else "draft"
        
        def _build_save_path(base_dir: str, idx: int, title: str) -> str:
            # Windows路径过长会触发 [Errno 2]，这里主动控制总长度并附加哈希防重名。
            title_safe = _safe_name(title)
            short_hash = hashlib.md5(title_safe.encode("utf-8", errors="ignore")).hexdigest()[:8]
            candidate = os.path.join(base_dir, f"step5_{idx:02d}_{title_safe}.docx")
            if len(candidate) > 235:
                title_safe = title_safe[:32]
                candidate = os.path.join(base_dir, f"step5_{idx:02d}_{title_safe}_{short_hash}.docx")
            # 仍然过长时，退化为固定短名
            if len(candidate) > 245:
                candidate = os.path.join(base_dir, f"step5_{idx:02d}_{short_hash}.docx")
            # 最后保险：超长路径前缀
            if len(candidate) >= 248 and not candidate.startswith("\\\\?\\"):
                candidate = "\\\\?\\" + os.path.abspath(candidate)
            return candidate

        def _md_images(text: str):
            return re.findall(r"!\[[^\]]*\]\(([^)]+)\)", str(text or ""))

        def _to_local_path(url: str) -> str:
            u = str(url or "").strip()
            if u.lower().startswith("file:///"):
                p = urllib.parse.urlparse(u).path
                p = urllib.parse.unquote(p)
                if re.match(r"^/[A-Za-z]:/", p):
                    p = p[1:]
                return p.replace("/", "\\")
            # Handle relative paths: resolve against the project's export directory
            if not u.lower().startswith(("http://", "https://", "/", "\\\\")):
                candidate = os.path.join(base_dir or os.getcwd(), u).replace("/", "\\")
                if os.path.exists(candidate):
                    return candidate
            return u

        try:
            project = project_service.get_project_by_id(pid)
            if not project:
                out["failed"] += 1
                return out
            base_dir = str((project.metadata or {}).get("batch_folder") or "").strip()
            if not base_dir:
                out["failed"] += 1
                return out
            os.makedirs(base_dir, exist_ok=True)
            drafts = project_service.get_project_drafts(pid) or []
            if not drafts:
                self.logger.info(f"Word 导出跳过：project_id={pid} 暂无草稿")
                return out
            for i, d in enumerate(drafts, 1):
                doc = Document()
                title = str(d.get("title") or f"{project.name}_draft_{i}")
                
                # User requested strictly saving ONLY the draft content, without any extra formatting.
                # So we simply write the final content (and insert local images if any).
                
                content_text = str(d.get("content") or "")
                # Merge duplicate section headings for backward compatibility
                import re
                for label in ["Introduction", "Methods", "Results", "Discussion", "Conclusion", "Abstract"]:
                    cleaned = re.sub(rf'(?im)^##?\s*{re.escape(label)}\s*\n?', '', content_text, count=1)
                    if cleaned and len(cleaned) > len(content_text) * 0.3:
                        content_text = cleaned.strip()
                
                # Iterate over text and images to preserve basic inline order if possible
                # (Simple approach: just append the text, then append the images at the bottom,
                # or split by image markdown tags to insert images inline).
                
                # A simple inline parser:
                import re
                # First pass: merge [Figure N](path) + *Figure N. caption* into a single image block
                fig_pattern = re.compile(r'!\[Figure \d+\]\(([^)]+)\)\s*\n?\s*\*(Figure \d+\..+)\*')
                content_text = fig_pattern.sub(
                    lambda m: f'[[IMAGE:caption={m.group(2)}|src={m.group(1)}]]',
                    content_text
                )
                parts = re.split(r'(\[\[IMAGE:[^\]]+\]\])', content_text)
                
                for part in parts:
                    if part.startswith('[[IMAGE:'):
                        m = re.search(r'\[\[IMAGE:caption=([^|]*)\|src=([^\]]+)\]\]', part)
                        if m:
                            caption = m.group(1).strip()
                            img_src = m.group(2).strip()
                            img_path = _to_local_path(img_src)
                            if os.path.exists(img_path):
                                try:
                                    doc.add_picture(img_path, width=Inches(5.5))
                                    if caption:
                                        cap_para = doc.add_paragraph()
                                        cap_run = cap_para.add_run(caption)
                                        cap_run.italic = True
                                        cap_para.alignment = 1  # WD_ALIGN_PARAGRAPH.CENTER
                                except Exception:
                                    doc.add_paragraph(part)
                            else:
                                doc.add_paragraph(f"[IMAGE NOT FOUND: {img_src}]")
                                if caption:
                                    doc.add_paragraph(caption)
                    elif part.strip():
                        doc.add_paragraph(part)
                
                save_path = _build_save_path(base_dir, i, title)
                doc.save(save_path)
                self.logger.info(f"Word 已导出: {save_path}")
                out["success"] += 1
        except Exception as e:
            self.logger.error(f"批量 Word 导出失败 project_id={pid}: {e}")
            out["failed"] += 1
        return out

    def show_feedback_optimization(self):
        """Show Step 5 interface"""
        if self.current_project_id:
            self.feedback_optimization_widget.set_current_project(self.current_project_id)
            
        self.stacked_widget.setCurrentWidget(self.feedback_optimization_widget)
        self.status_bar.showMessage('迭代优化仪表盘')

    def on_data_cleaning_project_selected(self, project_id: str, project_name: str):
        self.current_project_id = project_id
        if project_name:
            self.project_info_label.setText(f"当前项目: {project_name}")
        self.status_bar.showMessage(f'数据清洗选择项目: {project_name or project_id}')
    
    def closeEvent(self, event):
        """
        窗口关闭事件处理
        
        Args:
            event: 关闭事件
        """
        # 保存窗口位置和大小
        self.config.set('ui.window_size.width', self.width())
        self.config.set('ui.window_size.height', self.height())
        self.config.set('ui.window_position.x', self.x())
        self.config.set('ui.window_position.y', self.y())
        self.config.save()
        
        self.logger.info("应用程序关闭")
        
        # === 关键修复：强制退出 ===
        # 在 Windows 上，PyQt5 + PyTorch + OpenCV 组合经常在程序关闭时的析构阶段
        # 导致 ntdll.dll 崩溃 (RangeChecks / 0xC0000005)。
        # 在这里直接调用 os._exit(0) 可以跳过 Qt 的析构过程，安全退出。
        import os
        os._exit(0)
        # =======================
        
        event.accept()
