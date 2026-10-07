#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
项目管理界面模块

实现项目管理界面，包括项目创建、删除、切换和样本管理。
"""

import sys
import os
import uuid
from pathlib import Path
from datetime import datetime
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGridLayout,
    QPushButton, QLabel, QLineEdit, QFrame,
    QMessageBox, QGroupBox, QSizePolicy, QListWidget,
    QListWidgetItem, QInputDialog, QSplitter, QScrollArea,
    QComboBox, QTextEdit, QDialog, QFileDialog, QProgressDialog, QCheckBox
)
from PyQt5.QtCore import Qt, pyqtSignal, QSize, QTimer
from PyQt5.QtGui import QIcon, QFont, QPalette, QPixmap

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from src.services.project_service import ProjectService
from src.services.project_storage_service import ProjectStorageService
from src.services.database_service import DatabaseService
from src.services.text_service import TextService
from src.services.ocr_service import OCRService
from src.services.dicom_service import DicomService
from src.models.project import Project
from src.core.event_bus import get_event_bus, EventNames
from src.utils.logger import get_logger


class ProjectItemWidget(QWidget):
    """
    自定义项目项组件，包含项目信息和功能模块按钮
    """
    
    # 信号定义
    project_selected = pyqtSignal(str, str)  # project_id, project_name
    function_requested = pyqtSignal(str, str)  # function_name, project_id
    
    def __init__(self, project, parent=None):
        super().__init__(parent)
        self.project = project
        self.project_id = project.project_id
        self.project_name = project.name
        self.is_selected = False
        
        self.init_ui()
        self.setup_connections()
    
    def init_ui(self):
        """
        初始化UI
        """
        self.setFixedHeight(80)
        self.setStyleSheet("""
            ProjectItemWidget {
                background-color: #FFFFFF;
                border: 1px solid #E0E0E0;
                border-radius: 5px;
                margin: 2px;
            }
            ProjectItemWidget:hover {
                border-color: #2E86AB;
                background-color: #F8F9FA;
            }
        """)
        
        # 主布局
        main_layout = QHBoxLayout(self)
        main_layout.setContentsMargins(10, 5, 10, 5)
        main_layout.setSpacing(10)
        
        # 最左侧：复选框
        self.checkbox = QCheckBox()
        self.checkbox.setStyleSheet("QCheckBox::indicator { width: 18px; height: 18px; }")
        # 默认项目不能被删除，因此禁用复选框
        if self.project_id == "default":
            self.checkbox.setEnabled(False)
            self.checkbox.setToolTip("默认项目无法被删除")
        main_layout.addWidget(self.checkbox)
        
        # 左侧：项目信息
        info_layout = QVBoxLayout()
        info_layout.setSpacing(2)
        
        # 项目名称
        self.name_label = QLabel(self.project_name)
        self.name_label.setStyleSheet("""
            QLabel {
                font-size: 14px;
                font-weight: bold;
                color: #2E86AB;
            }
        """)
        info_layout.addWidget(self.name_label)
        
        # 项目ID和描述
        info_text = f"ID: {self.project_id}"
        if self.project.description:
            info_text += f" | {self.project.description}"
        
        self.info_label = QLabel(info_text)
        self.info_label.setStyleSheet("""
            QLabel {
                font-size: 11px;
                color: #666666;
            }
        """)
        info_layout.addWidget(self.info_label)
        
        main_layout.addLayout(info_layout)
        main_layout.addStretch()
        
        # 移除功能按钮，这些功能现在通过样本管理来访问
    
    def setup_connections(self):
        """
        设置信号连接
        """
        # 按钮已移除，不再需要连接事件
        pass
    
    def set_selected_style(self, selected=True):
        """设置选中状态样式"""
        if selected:
            self.setStyleSheet("""
                QWidget {
                    background-color: #E3F2FD;
                    border: 2px solid #2E86AB;
                    border-radius: 5px;
                }
            """)
        else:
            self.setStyleSheet("""
                QWidget {
                    background-color: #FFFFFF;
                    border: 1px solid #ddd;
                    border-radius: 5px;
                }
                QWidget:hover {
                    background-color: #F5F5F5;
                }
            """)
    
    def mousePressEvent(self, event):
        """
        鼠标点击事件 - 选中项目
        """
        if event.button() == Qt.LeftButton:
            self.project_selected.emit(self.project_id, self.project_name)
        super().mousePressEvent(event)
    
    def set_selected(self, selected):
        """
        设置选中状态
        
        Args:
            selected: 是否选中
        """
        self.is_selected = selected
        if selected:
            self.setStyleSheet("""
                ProjectItemWidget {
                    background-color: #E3F2FD;
                    border: 2px solid #2E86AB;
                    border-radius: 5px;
                    margin: 2px;
                }
            """)
        else:
            self.setStyleSheet("""
                ProjectItemWidget {
                    background-color: #FFFFFF;
                    border: 1px solid #E0E0E0;
                    border-radius: 5px;
                    margin: 2px;
                }
                ProjectItemWidget:hover {
                    border-color: #2E86AB;
                    background-color: #F8F9FA;
                }
            """)


class ProjectManagementWidget(QWidget):
    """
    项目管理界面类
    """
    
    # 信号定义
    project_switched = pyqtSignal(str)  # 项目切换信号
    function_selected = pyqtSignal(str)  # 功能选择信号
    batch_process_requested = pyqtSignal(list)  # 批处理请求信号(project_id列表)
    
    def __init__(self):
        super().__init__()
        
        self.project_service = ProjectService()
        self.db_service = DatabaseService()
        self.storage_service = ProjectStorageService(self.db_service)
        self.text_service = TextService(db_service=self.db_service)
        self.ocr_service = OCRService(db_service=self.db_service)
        self.dicom_service = DicomService(db_service=self.db_service)
        self.event_bus = get_event_bus()
        self.logger = get_logger(__name__)
        self.current_project = None
        self.selected_project_id = None
        
        # 项目项列表
        self.project_items = []
        
        self.init_ui()
        self.setup_connections()
        self.load_projects()
    
    def init_ui(self):
        """
        初始化用户界面
        """
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(20, 20, 20, 20)
        main_layout.setSpacing(20)
        
        # 创建分割器
        splitter = QSplitter(Qt.Horizontal)
        
        # 左侧：项目管理区域
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)
        self.create_project_management(left_layout)
        
        # 右侧：项目切换和功能导航区域
        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)
        self.create_project_selector(right_layout)
        self.create_function_navigation(right_layout)
        
        splitter.addWidget(left_widget)
        splitter.addWidget(right_widget)
        # 设置左侧项目管理区域宽度为40%，右侧为60%
        splitter.setStretchFactor(0, 2)  # 40%
        splitter.setStretchFactor(1, 3)  # 60%
        
        main_layout.addWidget(splitter)
        
        # 应用样式
        self.apply_styles()
    
    def create_project_selector(self, parent_layout):
        """
        创建项目选择区域
        
        Args:
            parent_layout: 父布局
        """
        # 项目选择组框
        selector_group = QGroupBox("当前项目")
        selector_group.setFixedHeight(120)
        selector_layout = QVBoxLayout(selector_group)
        selector_layout.setSpacing(15)
        
        # 项目选择下拉框
        project_layout = QHBoxLayout()
        project_label = QLabel("项目:")
        self.project_combo = QComboBox()
        self.project_combo.setMinimumWidth(200)
        self.project_combo.setStyleSheet("""
            QComboBox {
                font-family: 'Microsoft YaHei';
                font-size: 12px;
                padding: 8px;
                border: 1px solid #CCCCCC;
                border-radius: 4px;
                background-color: #FFFFFF;
            }
            QComboBox:focus {
                border: 2px solid #2E86AB;
            }
            QComboBox::drop-down {
                border: none;
                width: 20px;
            }
            QComboBox::down-arrow {
                image: url(down_arrow.png);
                width: 12px;
                height: 12px;
            }
        """)
        
        project_layout.addWidget(project_label)
        project_layout.addWidget(self.project_combo)
        project_layout.addStretch()
        
        selector_layout.addLayout(project_layout)
        
        # 项目信息显示
        self.project_info_label = QLabel("请选择项目")
        self.project_info_label.setObjectName("projectInfoLabel")
        self.project_info_label.setStyleSheet("""
            QLabel {
                font-weight: bold;
                color: #2E86AB;
                padding: 8px;
                border-radius: 4px;
            }
        """)
        selector_layout.addWidget(self.project_info_label)
        
        parent_layout.addWidget(selector_group)
    
    def create_project_management(self, parent_layout):
        """
        创建项目管理区域
        
        Args:
            parent_layout: 父布局
        """
        # 项目管理组框
        project_group = QGroupBox("项目管理")
        project_layout = QVBoxLayout(project_group)
        project_layout.setSpacing(10)
        
        # 项目操作按钮
        button_layout = QHBoxLayout()
        
        self.add_project_button = QPushButton("新建项目")
        self.add_project_button.setFixedSize(80, 30)
        
        self.delete_project_button = QPushButton("删除勾选")
        self.delete_project_button.setFixedSize(80, 30)
        self.delete_project_button.setEnabled(True)
        self.delete_project_button.setToolTip("删除列表中所有已勾选的项目")
        
        self.refresh_button = QPushButton("刷新")
        self.refresh_button.setFixedSize(60, 30)
        self.add_root_button = QPushButton("添加根目录")
        self.add_root_button.setFixedSize(95, 30)
        
        button_layout.addWidget(self.add_project_button)
        button_layout.addWidget(self.delete_project_button)
        button_layout.addWidget(self.refresh_button)
        button_layout.addWidget(self.add_root_button)
        button_layout.addStretch()
        
        project_layout.addLayout(button_layout)
        
        # 项目列表标签
        list_label = QLabel("项目列表:")
        project_layout.addWidget(list_label)
        
        # 创建滚动区域来容纳自定义项目项
        self.scroll_area = QScrollArea()
        self.scroll_area.setMaximumHeight(300)
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.scroll_area.setStyleSheet("""
            QScrollArea {
                border: 1px solid #E0E0E0;
                border-radius: 5px;
                background-color: #FAFAFA;
            }
        """)
        
        # 创建容器widget来放置项目项
        self.projects_container = QWidget()
        self.projects_layout = QVBoxLayout(self.projects_container)
        self.projects_layout.setSpacing(3)
        self.projects_layout.setContentsMargins(5, 5, 5, 5)
        
        self.scroll_area.setWidget(self.projects_container)
        project_layout.addWidget(self.scroll_area)
        
        # 统计信息
        self.stats_label = QLabel("总计: 0 个项目")
        self.stats_label.setStyleSheet("color: #666666; font-size: 11px;")
        project_layout.addWidget(self.stats_label)
        
        parent_layout.addWidget(project_group)
    
    def create_function_navigation(self, parent_layout):
        """
        创建功能导航区域
        
        Args:
            parent_layout: 父布局
        """
        # 功能导航组框
        nav_group = QGroupBox("功能模块")
        nav_group.setObjectName("functionModulesGroup")
        nav_layout = QGridLayout(nav_group)
        nav_layout.setSpacing(20)
        
        # 只保留样本管理功能按钮
        functions = [
            {
                'name': '样本管理',
                'description': '样本数据管理\n项目关联映射\n增删改查操作',
                'icon': 'sample_icon.png',
                'signal': 'sample_management'
            }
        ]
        
        # 创建功能按钮
        self.function_buttons = []
        for i, func in enumerate(functions):
            button = self.create_function_button(func)
            self.function_buttons.append(button)
            
            row = i // 2
            col = i % 2
            nav_layout.addWidget(button, row, col)
        
        # 如果是奇数个按钮，最后一个居中
        if len(functions) % 2 == 1:
            nav_layout.setColumnStretch(0, 1)
            nav_layout.setColumnStretch(1, 1)
        
        parent_layout.addWidget(nav_group)
        
        # 添加样本信息显示框
        self.create_sample_info_display(parent_layout)
        
        parent_layout.addStretch()
    
    def create_function_button(self, func_data):
        """
        创建功能按钮
        
        Args:
            func_data: 功能数据字典
        
        Returns:
            QPushButton: 功能按钮
        """
        button = QPushButton()
        button.setFixedSize(300, 150)
        button.setEnabled(False)  # 初始状态禁用，选择项目后启用
        
        # 创建按钮内容布局
        button_layout = QVBoxLayout(button)
        button_layout.setSpacing(10)
        
        # 图标（如果存在）
        icon_label = QLabel()
        icon_path = f"assets/icons/{func_data['icon']}"
        if os.path.exists(icon_path):
            pixmap = QPixmap(icon_path)
            icon_label.setPixmap(pixmap.scaled(48, 48, Qt.KeepAspectRatio, Qt.SmoothTransformation))
        else:
            icon_label.setText("📊")  # 默认图标
            icon_label.setStyleSheet("font-size: 48px;")
        
        icon_label.setAlignment(Qt.AlignCenter)
        
        # 功能名称
        name_label = QLabel(func_data['name'])
        name_label.setAlignment(Qt.AlignCenter)
        name_label.setStyleSheet("""
            QLabel {
                font-size: 16px;
                font-weight: bold;
                color: #D0D0D0;
            }
        """)
        
        # 功能描述
        desc_label = QLabel(func_data['description'])
        desc_label.setAlignment(Qt.AlignCenter)
        desc_label.setStyleSheet("""
            QLabel {
                font-size: 12px;
                color: #F2F2F2;
                line-height: 1.4;
                font-weight: 600;
            }
        """)
        
        button_layout.addWidget(icon_label)
        button_layout.addWidget(name_label)
        button_layout.addWidget(desc_label)
        
        # 连接点击事件
        button.clicked.connect(lambda: self.function_selected.emit(func_data['signal']))
        
        return button
    
    def create_sample_info_display(self, parent_layout):
        """
        创建样本信息显示框
        
        Args:
            parent_layout: 父布局
        """
        # 样本信息组框
        sample_group = QGroupBox("当前项目样本信息")
        sample_group.setObjectName("sampleInfoGroup")
        sample_group.setFixedHeight(400)  # 增加高度以容纳更多按钮
        sample_layout = QVBoxLayout(sample_group)
        sample_layout.setSpacing(10)
        
        # 当前选中样本显示
        current_layout = QHBoxLayout()
        current_label = QLabel("当前样本:")
        self.current_sample_label = QLabel("未选择")
        self.current_sample_label.setObjectName("currentSampleLabel")
        self.current_sample_label.setStyleSheet("""
            QLabel {
                font-weight: bold;
                color: #2E86AB;
                padding: 5px;
                border-radius: 3px;
            }
        """)
        current_layout.addWidget(current_label)
        current_layout.addWidget(self.current_sample_label)
        current_layout.addStretch()
        sample_layout.addLayout(current_layout)
        
        # 样本统计信息
        self.sample_stats_label = QLabel("样本统计: 暂无项目选择")
        self.sample_stats_label.setObjectName("sampleStatsLabel")
        self.sample_stats_label.setStyleSheet("""
            QLabel {
                font-weight: bold;
                color: #2E86AB;
                padding: 8px;
                border-radius: 4px;
                font-size: 12px;
            }
        """)
        sample_layout.addWidget(self.sample_stats_label)
        
        # 样本列表
        self.sample_list = QListWidget()
        self.sample_list.setObjectName("sampleList")
        self.sample_list.setMaximumHeight(180)  # 调整高度
        self.sample_list.setStyleSheet("""
            QListWidget {
                border: 1px solid #707070;
                border-radius: 5px;
                background-color: #3A3A3A;
                color: #F2F2F2;
                font-size: 11px;
            }
            QListWidget::item {
                padding: 8px;
                border-bottom: 1px solid #5A5A5A;
                color: #F2F2F2;
            }
            QListWidget::item:selected {
                background-color: #5A5A5A;
                color: #FFFFFF;
                font-weight: bold;
            }
            QListWidget::item:hover {
                background-color: #4A4A4A;
            }
        """)
        sample_layout.addWidget(self.sample_list)
        
        # 样本管理按钮
        management_layout = QHBoxLayout()
        
        self.add_sample_button = QPushButton("新建样本")
        self.add_sample_button.setFixedSize(80, 25)
        self.add_sample_button.setStyleSheet("""
            QPushButton {
                background-color: #4CAF50;
                color: white;
                border: none;
                border-radius: 3px;
                font-size: 10px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
            QPushButton:disabled {
                background-color: #CCCCCC;
                color: #666666;
            }
        """)
        
        self.delete_sample_button = QPushButton("删除样本")
        self.delete_sample_button.setFixedSize(80, 25)
        self.delete_sample_button.setStyleSheet("""
            QPushButton {
                background-color: #F44336;
                color: white;
                border: none;
                border-radius: 3px;
                font-size: 10px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #D32F2F;
            }
            QPushButton:disabled {
                background-color: #CCCCCC;
                color: #666666;
            }
        """)
        
        self.refresh_samples_button = QPushButton("刷新")
        self.refresh_samples_button.setFixedSize(60, 25)
        self.refresh_samples_button.setStyleSheet("""
            QPushButton {
                background-color: #FF9800;
                color: white;
                border: none;
                border-radius: 3px;
                font-size: 10px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #F57C00;
            }
            QPushButton:disabled {
                background-color: #CCCCCC;
                color: #666666;
            }
        """)
        
        management_layout.addWidget(self.add_sample_button)
        management_layout.addWidget(self.delete_sample_button)
        management_layout.addWidget(self.refresh_samples_button)
        management_layout.addStretch()
        
        sample_layout.addLayout(management_layout)
        
        # 功能导航按钮（选中样本后显示）
        navigation_layout = QHBoxLayout()
        
        self.dicom_nav_button = QPushButton("DICOM")
        self.dicom_nav_button.setFixedSize(60, 25)
        self.dicom_nav_button.setStyleSheet("""
            QPushButton {
                background-color: #4CAF50;
                color: white;
                border: none;
                border-radius: 3px;
                font-size: 10px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
            QPushButton:disabled {
                background-color: #CCCCCC;
                color: #666666;
            }
        """)
        
        self.text_nav_button = QPushButton("文本")
        self.text_nav_button.setFixedSize(60, 25)
        self.text_nav_button.setStyleSheet("""
            QPushButton {
                background-color: #2E86AB;
                color: white;
                border: none;
                border-radius: 3px;
                font-size: 10px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #1976D2;
            }
            QPushButton:disabled {
                background-color: #CCCCCC;
                color: #666666;
            }
        """)
        
        self.ocr_nav_button = QPushButton("OCR")
        self.ocr_nav_button.setFixedSize(60, 25)
        self.ocr_nav_button.setStyleSheet("""
            QPushButton {
                background-color: #FF9800;
                color: white;
                border: none;
                border-radius: 3px;
                font-size: 10px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #F57C00;
            }
            QPushButton:disabled {
                background-color: #CCCCCC;
                color: #666666;
            }
        """)
        
        navigation_layout.addWidget(QLabel("功能导航:"))
        navigation_layout.addWidget(self.dicom_nav_button)
        navigation_layout.addWidget(self.text_nav_button)
        navigation_layout.addWidget(self.ocr_nav_button)
        navigation_layout.addStretch()
        
        sample_layout.addLayout(navigation_layout)
        
        parent_layout.addWidget(sample_group)
        
        # 初始化变量
        self.selected_sample_id = None
        
        # 初始状态禁用按钮
        self.add_sample_button.setEnabled(False)
        self.delete_sample_button.setEnabled(False)
        self.refresh_samples_button.setEnabled(False)
        self.dicom_nav_button.setEnabled(False)
        self.text_nav_button.setEnabled(False)
        self.ocr_nav_button.setEnabled(False)
    
    def setup_connections(self):
        """
        设置信号连接
        """
        self.add_project_button.clicked.connect(self.add_project)
        self.delete_project_button.clicked.connect(self.delete_project)
        self.refresh_button.clicked.connect(self.load_projects)
        self.add_root_button.clicked.connect(self.add_root_directory_batch)
        self.project_combo.currentTextChanged.connect(self.on_project_changed)
        
        # 样本管理按钮连接
        self.add_sample_button.clicked.connect(self.handle_add_sample)
        self.delete_sample_button.clicked.connect(self.handle_delete_sample)
        self.refresh_samples_button.clicked.connect(self.refresh_samples)
        
        # 样本列表选择连接
        self.sample_list.itemSelectionChanged.connect(self.on_sample_selection_changed)
        
        # 功能导航按钮连接
        self.dicom_nav_button.clicked.connect(lambda: self.navigate_to_function('dicom'))
        self.text_nav_button.clicked.connect(lambda: self.navigate_to_function('text'))
        self.ocr_nav_button.clicked.connect(lambda: self.navigate_to_function('ocr'))
    
    def load_projects(self):
        """
        加载项目列表
        """
        try:
            # 清空现有项目项（强制从布局移除并延迟销毁，避免批量删除后UI残影）
            while self.projects_layout.count():
                child = self.projects_layout.takeAt(0)
                w = child.widget()
                if w is not None:
                    w.setParent(None)
                    w.deleteLater()
            self.project_items.clear()
            
            # 清空下拉框
            self.project_combo.clear()
            
            # 获取所有活跃项目
            projects = self.project_service.get_all_projects(status='active')
            
            if not projects:
                # 如果没有项目，创建默认项目
                self.create_default_project()
                projects = self.project_service.get_all_projects(status='active')
            
            # 添加项目到界面
            for project in projects:
                self.add_project_item(project)
                self.project_combo.addItem(project.name, project.project_id)
            
            # 更新统计信息
            self.stats_label.setText(f"总计: {len(projects)} 个项目")
            
            # 选择第一个项目
            if projects:
                self.project_combo.setCurrentIndex(0)
                self.select_project(projects[0].project_id)
            
            self.logger.info(f"加载了 {len(projects)} 个项目")
            self.projects_container.update()
            self.scroll_area.viewport().update()
            
        except Exception as e:
            self.logger.error(f"加载项目失败: {e}")
            QMessageBox.warning(self, "错误", f"加载项目失败: {e}")
    
    def create_default_project(self):
        """
        创建默认项目
        """
        try:
            default_project = Project(
                project_id="default",
                name="默认项目",
                description="系统默认项目",
                status="active"
            )
            
            success = self.project_service.create_project(default_project)
            if success:
                self.logger.info("创建默认项目成功")
            else:
                self.logger.warning("默认项目可能已存在")
                
        except Exception as e:
            self.logger.error(f"创建默认项目失败: {e}")
    
    def add_project_item(self, project):
        """
        添加项目项到界面
        
        Args:
            project: 项目对象
        """
        item_widget = ProjectItemWidget(project)
        item_widget.project_selected.connect(self.on_project_item_selected)
        item_widget.function_requested.connect(self.on_function_requested)
        
        self.projects_layout.addWidget(item_widget)
        self.project_items.append(item_widget)
    
    def add_project(self):
        """
        添加新项目
        """
        dialog = ProjectCreateDialog(self)
        if dialog.exec_() == QDialog.Accepted:
            project_data = dialog.get_project_data()
            
            try:
                project = Project(
                    project_id=project_data['project_id'],
                    name=project_data['name'],
                    description=project_data['description'],
                    status='active'
                )
                
                success = self.project_service.create_project(project)
                if success:
                    self.load_projects()  # 重新加载项目列表
                    QMessageBox.information(self, "成功", "项目创建成功！")
                else:
                    QMessageBox.warning(self, "错误", "项目创建失败！")
                    
            except Exception as e:
                self.logger.error(f"创建项目失败: {e}")
                QMessageBox.warning(self, "错误", f"创建项目失败: {e}")

    def _collect_second_level_dirs(self, root_dir: str) -> list:
        root = Path(root_dir)
        candidates = []
        seen = set()

        def _add_dir(p: Path):
            try:
                sp = str(p)
                if sp not in seen and p.is_dir():
                    seen.add(sp)
                    candidates.append(sp)
            except Exception:
                pass

        try:
            for lv1 in sorted([p for p in root.iterdir() if p.is_dir()]):
                lv1_subdirs = sorted([x for x in lv1.iterdir() if x.is_dir()])
                lv1_files = sorted([x for x in lv1.iterdir() if x.is_file()])

                # 一级目录下直接有文件时，也作为可处理目录
                if lv1_files:
                    _add_dir(lv1)

                # 二级目录保持原有支持
                for p in lv1_subdirs:
                    _add_dir(p)
        except Exception:
            pass

        # 根目录本身有文件时，增加根目录作为一个可处理项
        try:
            if any(p.is_file() for p in root.iterdir()):
                _add_dir(root)
        except Exception:
            pass

        # 若仍为空，回退到一级目录
        if not candidates:
            try:
                for p in sorted([p for p in root.iterdir() if p.is_dir()]):
                    _add_dir(p)
            except Exception:
                candidates = []
        return candidates

    def _ensure_sample_user(self, project_id: str, sample_name: str) -> str:
        uid = f"sample_{uuid.uuid4().hex[:12]}"
        self.db_service.execute_update(
            "INSERT INTO user_ids (user_id, display_name, project_id) VALUES (?, ?, ?)",
            (uid, sample_name, project_id),
        )
        return uid

    def _ingest_folder_materials_to_project(self, folder_path: str, project_id: str, user_id: str) -> dict:
        text_exts = {".txt", ".md", ".markdown", ".json", ".xml", ".yaml", ".yml", ".log", ".csv", ".pdf", ".doc", ".docx", ".xls", ".xlsx", ".xlsm"}
        image_exts = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp", ".gif"}
        dicom_exts = {".dcm", ".dicom"}
        stats = {"text": 0, "ocr": 0, "dicom": 0, "errors": 0}

        def _io_path(p: str) -> str:
            # Windows 长路径兼容：避免超长路径在 exists/open 阶段被误判为不存在
            if os.name != "nt":
                return p
            try:
                ap = os.path.abspath(p)
                if ap.startswith("\\\\?\\"):
                    return ap
                if len(ap) < 240:
                    return ap
                if ap.startswith("\\\\"):
                    return "\\\\?\\UNC\\" + ap.lstrip("\\")
                return "\\\\?\\" + ap
            except Exception:
                return p

        for root, _, files in os.walk(folder_path):
            for fn in files:
                raw_fp = os.path.join(root, fn)
                fp = _io_path(raw_fp)
                ext = os.path.splitext(fn)[1].lower()
                try:
                    # DICOM
                    if ext in dicom_exts or self.dicom_service._is_dicom_file(fp):
                        ds = self.dicom_service.load_dicom_file(fp, use_cache=True, force=True)
                        if ds is not None:
                            info = self.dicom_service.extract_dicom_info(ds)
                            self.dicom_service.create_dicom_session(
                                user_id=user_id,
                                dicom_file=fp,
                                dicom_info=info,
                                roi_data=[],
                                project_id=project_id,
                            )
                            stats["dicom"] += 1
                        continue

                    # OCR images
                    if ext in image_exts:
                        r = self.ocr_service.extract_text_from_image(fp, {"engine": "deepseek_ocr", "preprocessing": True})
                        txt = str(r.get("text") or r.get("extracted_text") or "")
                        if txt.strip():
                            self.ocr_service.create_ocr_session(
                                user_id=user_id,
                                image_path=fp,
                                extracted_text=txt,
                                confidence=float(r.get("confidence", 0.85) or 0.85),
                                processing_info={"batch_import": True},
                                project_id=project_id,
                            )
                            stats["ocr"] += 1
                        continue

                    # Text-like documents
                    if ext in text_exts:
                        rr = self.text_service.read_file_structured(fp)
                        content = str(rr.get("text") or "")
                        tabular_payload = rr.get("tabular")
                        has_tabular = isinstance(tabular_payload, dict) and bool(tabular_payload.get("sheets"))
                        if content.strip() or has_tabular:
                            analysis = self.text_service.analyze_text(content)
                            keywords = self.text_service.extract_keywords(content, max_keywords=10)
                            session_id = self.text_service.create_text_session(
                                user_id=user_id,
                                file_path=fp,
                                content=content,
                                analysis_result=analysis,
                                keywords=keywords,
                                project_id=project_id,
                                tabular_payload=tabular_payload,
                            )
                            if session_id:
                                stats["text"] += 1
                            else:
                                stats["errors"] += 1
                except Exception as e:
                    self.logger.warning(f"批量入库失败: {raw_fp} | {e}")
                    stats["errors"] += 1
                    continue
        return stats

    def add_root_directory_batch(self):
        root_dir = QFileDialog.getExistingDirectory(self, "选择根目录（将识别其二级目录）")
        if not root_dir:
            return
        second_level_dirs = self._collect_second_level_dirs(root_dir)
        if not second_level_dirs:
            QMessageBox.information(self, "提示", "未发现可处理的二级目录。")
            return

        progress = QProgressDialog("正在批量新建项目并提取素材...", "取消", 0, len(second_level_dirs), self)
        progress.setWindowTitle("批量新建项目")
        progress.setMinimumDuration(0)
        progress.setValue(0)

        created_project_ids = []
        total_stats = {"text": 0, "ocr": 0, "dicom": 0, "errors": 0}
        for i, folder in enumerate(second_level_dirs, 1):
            if progress.wasCanceled():
                break
            progress.setValue(i - 1)
            progress.setLabelText(f"处理中: {folder}")

            folder_name = os.path.basename(folder.rstrip("\\/")) or f"project_{i}"
            project_id = f"batch_{datetime.now().strftime('%Y%m%d%H%M%S')}_{i:03d}_{uuid.uuid4().hex[:6]}"
            try:
                project = Project(
                    project_id=project_id,
                    name=folder_name,
                    description=f"批量导入自: {folder}",
                    status="active",
                    metadata={"batch_root": root_dir, "batch_folder": folder},
                )
                if not self.project_service.create_project(project):
                    continue
                # 批量导入场景下项目刚创建，需先确保 project_storage 行与目录存在，
                # 否则配额检查会因找不到记录而返回 False，表现为“存储配额不足”。
                self._ensure_project_storage(project_id)
                user_id = self._ensure_sample_user(project_id, folder_name)
                stats = self._ingest_folder_materials_to_project(folder, project_id, user_id)
                for k in total_stats:
                    total_stats[k] += int(stats.get(k, 0) or 0)
                created_project_ids.append(project_id)
            except Exception:
                continue

        progress.setValue(len(second_level_dirs))
        self.load_projects()

        if not created_project_ids:
            QMessageBox.warning(self, "失败", "未成功创建任何项目。")
            return

        QMessageBox.information(
            self,
            "批量项目创建完成",
            (
                f"已创建 {len(created_project_ids)} 个项目。\n"
                f"入库统计：文本 {total_stats['text']}，OCR {total_stats['ocr']}，DICOM {total_stats['dicom']}，失败 {total_stats['errors']}。\n"
                f"将自动开启批处理（Step2-Step5），无需人工审批。"
            ),
        )
        self.batch_process_requested.emit(created_project_ids)
    
    def delete_project(self):
        """
        删除勾选的项目
        """
        # 获取所有勾选的项目
        checked_projects = []
        for item in self.project_items:
            if hasattr(item, 'checkbox') and item.checkbox.isChecked() and item.project_id != "default":
                checked_projects.append(item)
                
        if not checked_projects:
            QMessageBox.warning(self, "提示", "请先在项目列表中勾选要删除的项目！\n(注：默认项目无法被删除)")
            return
            
        project_names = "\n".join([f"- {item.project_name}" for item in checked_projects])
        
        reply = QMessageBox.question(
            self, "确认删除", 
            (
                f"确定要永久删除以下 {len(checked_projects)} 个项目吗？\n\n{project_names}\n\n"
                f"注意：这会同时删除项目数据与项目存储文件，且不可恢复。"
            ),
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No
        )
        
        if reply == QMessageBox.Yes:
            try:
                success_count = 0
                fail_count = 0
                
                for item in checked_projects:
                    if self.project_service.purge_project(item.project_id):
                        success_count += 1
                    else:
                        fail_count += 1
                        
                # 显式重置选中状态与ID缓存，防止删除后幽灵选中
                self.selected_project_id = None
                self.current_project = None
                self.project_info_label.setText("请选择项目")
                
                self.load_projects()  # 重新加载项目列表
                
                if fail_count == 0:
                    QMessageBox.information(self, "成功", f"成功永久删除了 {success_count} 个项目及其数据！")
                else:
                    QMessageBox.warning(self, "完成", f"永久删除了 {success_count} 个项目，但有 {fail_count} 个项目删除失败！")
                    
            except Exception as e:
                self.logger.error(f"批量删除项目时发生错误: {e}")
                QMessageBox.warning(self, "错误", f"批量删除项目时发生错误:\n{e}")
    
    def on_project_changed(self, project_name):
        """
        项目下拉框选择变化处理
        
        Args:
            project_name: 项目名称
        """
        if not project_name:
            return
            
        # 获取项目ID
        current_index = self.project_combo.currentIndex()
        if current_index >= 0:
            project_id = self.project_combo.itemData(current_index)
            if project_id:
                self.select_project(project_id)
                # 更新样本信息显示
                self.on_project_selected_for_samples(project_id)
    
    def select_project(self, project_id):
        """
        选择项目
        
        Args:
            project_id: 项目ID
        """
        try:
            project = self.project_service.get_project_by_id(project_id)
            if project:
                self.current_project = project
                self.selected_project_id = project_id
                
                # 确保项目存储已创建
                self._ensure_project_storage(project_id)
                
                # 更新项目信息显示
                info_text = f"项目: {project.name}\nID: {project.project_id}"
                if project.description:
                    info_text += f"\n描述: {project.description}"
                self.project_info_label.setText(info_text)
                
                # 更新项目项选中状态
                for item in self.project_items:
                    item.set_selected(item.project_id == project_id)
                
                # 启用功能按钮
                for button in self.function_buttons:
                    button.setEnabled(True)
                
                # 发送项目切换信号
                self.project_switched.emit(project_id)
                
                self.logger.info(f"切换到项目: {project.name} ({project_id})")
                
        except Exception as e:
            self.logger.error(f"选择项目失败: {e}")
    
    def _ensure_project_storage(self, project_id: str):
        """
        确保项目存储已创建
        
        Args:
            project_id: 项目ID
        """
        try:
            # 检查项目存储是否已存在
            storage_info = self.storage_service.get_project_storage_info(project_id)
            if not storage_info:
                # 创建项目存储
                self.logger.info(f"为项目 {project_id} 创建存储空间")
                success = self.storage_service.create_project_storage(project_id)
                if success:
                    self.logger.info(f"项目 {project_id} 存储空间创建成功")
                else:
                    self.logger.warning(f"项目 {project_id} 存储空间创建失败")
            else:
                self.logger.debug(f"项目 {project_id} 存储空间已存在")
        except Exception as e:
            self.logger.error(f"确保项目存储失败: {e}")
    
    def on_project_item_selected(self, project_id, project_name):
        """
        项目项选择处理
        
        Args:
            project_id: 项目ID
            project_name: 项目名称
        """
        # 更新下拉框选择
        for i in range(self.project_combo.count()):
            if self.project_combo.itemData(i) == project_id:
                self.project_combo.setCurrentIndex(i)
                break
        
        self.select_project(project_id)
        
        # 更新样本信息显示
        self.on_project_selected_for_samples(project_id)
    
    def on_function_requested(self, function_name, project_id):
        """
        功能请求处理
        
        Args:
            function_name: 功能名称
            project_id: 项目ID
        """
        # 确保项目已选中
        if project_id != self.selected_project_id:
            self.select_project(project_id)
        
        # 发送功能选择信号
        self.function_selected.emit(function_name)
    
    def handle_add_sample(self):
        """
        处理添加样本
        """
        if not self.selected_project_id:
            QMessageBox.warning(self, "警告", "请先选择一个项目！")
            return
        
        # 弹出输入对话框
        sample_id, ok = QInputDialog.getText(
            self, "新建样本", "请输入样本ID:", 
            QLineEdit.Normal, ""
        )
        
        if ok and sample_id.strip():
            sample_id = sample_id.strip()
            
            # 使用 QTimer.singleShot 将耗时操作移出事件循环
            QTimer.singleShot(10, lambda: self._add_sample_worker(sample_id))

    def _add_sample_worker(self, sample_id):
        """
        后台添加样本
        """
        try:
            # 检查样本ID是否已存在
            from src.services.database_service import DatabaseService
            db_service = DatabaseService()
            
            with db_service.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    "SELECT COUNT(*) FROM user_ids WHERE user_id = ? AND project_id = ?", 
                    (sample_id, self.selected_project_id)
                )
                
                if cursor.fetchone()[0] > 0:
                    QMessageBox.warning(self, "警告", f"样本ID '{sample_id}' 已存在！")
                    return
                
                # 添加新样本
                cursor.execute("""
                    INSERT INTO user_ids (user_id, display_name, project_id, is_active, created_at)
                    VALUES (?, ?, ?, 1, CURRENT_TIMESTAMP)
                """, (sample_id, sample_id, self.selected_project_id))
                
                conn.commit()
            
            # 刷新样本列表
            self.refresh_samples()
            QMessageBox.information(self, "成功", f"样本 '{sample_id}' 创建成功！")
            
            self.logger.info(f"添加样本成功: {sample_id}")
            
        except Exception as e:
            self.logger.error(f"添加样本失败: {e}")
            QMessageBox.warning(self, "错误", f"添加样本失败: {e}")
    
    def handle_delete_sample(self):
        """
        处理删除样本
        """
        if not self.selected_sample_id:
            QMessageBox.warning(self, "警告", "请先选择要删除的样本！")
            return
        
        reply = QMessageBox.question(
            self, "确认删除", 
            f"确定要删除样本 '{self.selected_sample_id}' 吗？\n\n注意：这将删除该样本的所有相关数据！",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No
        )
        
        if reply == QMessageBox.Yes:
            try:
                from src.services.database_service import DatabaseService
                db_service = DatabaseService()
                
                with db_service.get_connection() as conn:
                    cursor = conn.cursor()
                    
                    # 删除相关的DICOM会话数据
                    cursor.execute("DELETE FROM dicom_sessions WHERE user_id = ?", (self.selected_sample_id,))
                    
                    # 删除样本
                    cursor.execute(
                        "DELETE FROM user_ids WHERE user_id = ? AND project_id = ?", 
                        (self.selected_sample_id, self.selected_project_id)
                    )
                    
                    conn.commit()
                
                # 清空选择状态
                self.selected_sample_id = None
                self.current_sample_label.setText("未选择")
                
                # 刷新样本列表
                self.refresh_samples()
                QMessageBox.information(self, "成功", "样本删除成功！")
                
                self.logger.info(f"删除样本成功: {self.selected_sample_id}")
                
            except Exception as e:
                self.logger.error(f"删除样本失败: {e}")
                QMessageBox.warning(self, "错误", f"删除样本失败: {e}")
    
    def refresh_samples(self):
        """
        刷新当前项目的样本信息
        """
        if not self.selected_project_id:
            return
        
        try:
            # 获取当前项目的样本数据
            from src.services.database_service import DatabaseService
            db_service = DatabaseService()
            
            with db_service.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT user_id, display_name, description, created_at, last_used, is_active
                    FROM user_ids 
                    WHERE project_id = ? AND is_active = 1
                    ORDER BY created_at DESC
                """, (self.selected_project_id,))
                
                samples = cursor.fetchall()
            
            # 更新样本列表
            self.update_sample_display(samples)
            
            self.logger.info(f"刷新项目 {self.selected_project_id} 的样本信息，共 {len(samples)} 个样本")
            
        except Exception as e:
            self.logger.error(f"刷新样本信息失败: {e}")
            QMessageBox.warning(self, "错误", f"刷新样本信息失败: {e}")
    
    def manage_samples(self):
        """
        打开样本管理界面
        """
        if not self.selected_project_id:
            QMessageBox.information(self, "提示", "请先选择一个项目")
            return
        
        # 发送样本管理功能选择信号
        self.function_selected.emit('sample_management')
    
    def update_sample_display(self, samples):
        """
        更新样本信息显示
        
        Args:
            samples: 样本列表（数据库查询结果）
        """
        # 清空现有列表
        self.sample_list.clear()
        
        if not samples:
            # 没有样本时显示提示信息
            self.sample_stats_label.setText("样本统计: 当前项目暂无样本")
            item = QListWidgetItem("暂无样本数据")
            item.setFlags(item.flags() & ~Qt.ItemIsSelectable)
            self.sample_list.addItem(item)
            return
        
        # 更新统计信息
        self.sample_stats_label.setText(f"样本统计: 共 {len(samples)} 个样本")
        
        # 添加样本项
        for sample in samples:
            # sample是数据库查询结果的元组: (user_id, display_name, description, created_at, last_used, is_active)
            user_id = sample[0]
            display_name = sample[1] if sample[1] else user_id
            
            # 创建样本项文本
            sample_text = f"{user_id}"
            if display_name and display_name != user_id:
                sample_text += f" ({display_name})"
            
            item = QListWidgetItem(sample_text)
            item.setData(Qt.UserRole, user_id)  # 存储样本ID
            self.sample_list.addItem(item)
    
    def on_sample_selection_changed(self):
        """
        样本选择变化处理
        """
        selected_items = self.sample_list.selectedItems()
        
        if selected_items:
            # 获取选中的样本ID
            item = selected_items[0]
            sample_id = item.data(Qt.UserRole)
            
            if sample_id and sample_id != "暂无样本数据":
                self.selected_sample_id = sample_id
                self.current_sample_label.setText(sample_id)
                
                # 发布USER_ID_SELECTED事件，将样本ID作为用户ID传递
                try:
                    self.event_bus.publish(EventNames.USER_ID_SELECTED, {
                        'user_id': sample_id,
                        'project_id': self.selected_project_id
                    })
                    self.logger.info(f"发布USER_ID_SELECTED事件: user_id={sample_id}, project_id={self.selected_project_id}")
                except Exception as e:
                    self.logger.error(f"发布USER_ID_SELECTED事件失败: {e}")
                
                # 启用删除按钮和功能导航按钮
                self.delete_sample_button.setEnabled(True)
                self.dicom_nav_button.setEnabled(True)
                self.text_nav_button.setEnabled(True)
                self.ocr_nav_button.setEnabled(True)
            else:
                self.selected_sample_id = None
                self.current_sample_label.setText("未选择")
                
                # 发布USER_ID_SELECTED事件，清空用户ID
                try:
                    self.event_bus.publish(EventNames.USER_ID_SELECTED, {
                        'user_id': None,
                        'project_id': self.selected_project_id
                    })
                    self.logger.info(f"发布USER_ID_SELECTED事件: user_id=None, project_id={self.selected_project_id}")
                except Exception as e:
                    self.logger.error(f"发布USER_ID_SELECTED事件失败: {e}")
                
                # 禁用相关按钮
                self.delete_sample_button.setEnabled(False)
                self.dicom_nav_button.setEnabled(False)
                self.text_nav_button.setEnabled(False)
                self.ocr_nav_button.setEnabled(False)
        else:
            self.selected_sample_id = None
            self.current_sample_label.setText("未选择")
            
            # 发布USER_ID_SELECTED事件，清空用户ID
            try:
                self.event_bus.publish(EventNames.USER_ID_SELECTED, {
                    'user_id': None,
                    'project_id': self.selected_project_id
                })
                self.logger.info(f"发布USER_ID_SELECTED事件: user_id=None, project_id={self.selected_project_id}")
            except Exception as e:
                self.logger.error(f"发布USER_ID_SELECTED事件失败: {e}")
            
            # 禁用相关按钮
            self.delete_sample_button.setEnabled(False)
            self.dicom_nav_button.setEnabled(False)
            self.text_nav_button.setEnabled(False)
            self.ocr_nav_button.setEnabled(False)
    
    def navigate_to_function(self, function_type):
        """
        导航到指定功能页面
        
        Args:
            function_type: 功能类型 ('dicom', 'text', 'ocr')
        """
        if not self.selected_sample_id:
            QMessageBox.warning(self, "警告", "请先选择一个样本！")
            return
        
        if not self.selected_project_id:
            QMessageBox.warning(self, "警告", "请先选择一个项目！")
            return
        
        try:
            # 发送功能选择信号，传递样本ID和项目ID
            if function_type == 'dicom':
                self.function_selected.emit('dicom_roi')
            elif function_type == 'text':
                self.function_selected.emit('text_processing')
            elif function_type == 'ocr':
                self.function_selected.emit('ocr_recognition')
            
            # 记录日志
            self.logger.info(f"导航到 {function_type} 功能，样本: {self.selected_sample_id}, 项目: {self.selected_project_id}")
            
        except Exception as e:
            self.logger.error(f"导航到功能页面失败: {e}")
            QMessageBox.warning(self, "错误", f"导航失败: {e}")
    
    def on_project_selected_for_samples(self, project_id):
        """
        项目选择时更新样本信息
        
        Args:
            project_id: 项目ID
        """
        if project_id:
            # 启用样本操作按钮
            self.add_sample_button.setEnabled(True)
            self.refresh_samples_button.setEnabled(True)
            
            # 自动刷新样本信息
            self.refresh_samples()
        else:
            # 禁用样本操作按钮
            self.add_sample_button.setEnabled(False)
            self.delete_sample_button.setEnabled(False)
            self.refresh_samples_button.setEnabled(False)
            self.dicom_nav_button.setEnabled(False)
            self.text_nav_button.setEnabled(False)
            self.ocr_nav_button.setEnabled(False)
            
            # 清空样本显示
            self.sample_list.clear()
            self.sample_stats_label.setText("样本统计: 暂无项目选择")
            self.current_sample_label.setText("未选择")
            self.selected_sample_id = None
    
    def apply_styles(self):
        """
        应用样式
        """
        self.setStyleSheet("""
            QGroupBox {
                font-size: 14px;
                font-weight: bold;
                color: #2E86AB;
                border: 2px solid #E0E0E0;
                border-radius: 8px;
                margin-top: 10px;
                padding-top: 10px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px 0 5px;
                background-color: #FFFFFF;
            }
            QLabel {
                font-family: 'Microsoft YaHei';
                font-size: 12px;
                color: #2E86AB;
            }
            QLineEdit {
                font-family: 'Microsoft YaHei';
                font-size: 12px;
                padding: 8px;
                border: 1px solid #CCCCCC;
                border-radius: 4px;
                background-color: #FFFFFF;
            }
            QLineEdit:focus {
                border: 2px solid #2E86AB;
            }
            QPushButton {
                font-family: 'Microsoft YaHei';
                font-size: 12px;
                font-weight: bold;
                color: #F5F5F5;
                background-color: #5A5A5A;
                border: 1px solid #707070;
                border-radius: 6px;
                padding: 8px 16px;
            }
            QPushButton:hover {
                background-color: #6A6A6A;
            }
            QPushButton:pressed {
                background-color: #4A4A4A;
            }
            QPushButton:disabled {
                background-color: #CCCCCC;
                color: #666666;
            }
        """)


class ProjectCreateDialog(QDialog):
    """
    项目创建对话框
    """
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("创建新项目")
        self.setFixedSize(400, 300)
        self.setModal(True)
        
        self.project_data = None
        self.init_ui()
    
    def init_ui(self):
        """
        初始化UI
        """
        layout = QVBoxLayout(self)
        layout.setSpacing(15)
        layout.setContentsMargins(20, 20, 20, 20)
        
        # 标题
        title_label = QLabel("创建新项目")
        title_label.setStyleSheet("""
            QLabel {
                font-size: 18px;
                font-weight: bold;
                color: #2E86AB;
                margin-bottom: 10px;
            }
        """)
        layout.addWidget(title_label)
        
        # 表单
        form_layout = QGridLayout()
        form_layout.setSpacing(10)
        
        # 项目名称
        name_label = QLabel("项目名称:")
        self.name_input = QLineEdit()
        self.name_input.setPlaceholderText("请输入项目名称")
        form_layout.addWidget(name_label, 0, 0)
        form_layout.addWidget(self.name_input, 0, 1)
        
        # 项目ID
        id_label = QLabel("项目ID:")
        self.id_input = QLineEdit()
        self.id_input.setPlaceholderText("自动生成或手动输入")
        form_layout.addWidget(id_label, 1, 0)
        form_layout.addWidget(self.id_input, 1, 1)
        
        # 项目描述
        desc_label = QLabel("项目描述:")
        self.desc_input = QTextEdit()
        self.desc_input.setPlaceholderText("请输入项目描述（可选）")
        self.desc_input.setMaximumHeight(80)
        form_layout.addWidget(desc_label, 2, 0)
        form_layout.addWidget(self.desc_input, 2, 1)
        
        layout.addLayout(form_layout)
        
        # 按钮
        button_layout = QHBoxLayout()
        
        self.cancel_button = QPushButton("取消")
        self.cancel_button.setFixedSize(80, 35)
        self.cancel_button.clicked.connect(self.reject)
        
        self.create_button = QPushButton("创建")
        self.create_button.setFixedSize(80, 35)
        self.create_button.clicked.connect(self.accept)
        
        button_layout.addStretch()
        button_layout.addWidget(self.cancel_button)
        button_layout.addWidget(self.create_button)
        
        layout.addLayout(button_layout)
        
        # 连接信号
        self.name_input.textChanged.connect(self.on_name_changed)
    
    def on_name_changed(self, text):
        """
        项目名称变化时自动生成ID
        
        Args:
            text: 项目名称
        """
        if text and not self.id_input.text():
            # 生成项目ID（简单的时间戳方式）
            import time
            project_id = f"project_{int(time.time())}"
            self.id_input.setText(project_id)
    
    def accept(self):
        """
        确认创建
        """
        name = self.name_input.text().strip()
        project_id = self.id_input.text().strip()
        description = self.desc_input.toPlainText().strip()
        
        if not name:
            QMessageBox.warning(self, "警告", "请输入项目名称！")
            return
        
        if not project_id:
            QMessageBox.warning(self, "警告", "请输入项目ID！")
            return
        
        self.project_data = {
            'name': name,
            'project_id': project_id,
            'description': description
        }
        
        super().accept()
    
    def reject(self):
        """
        取消创建
        """
        self.project_data = None
        super().reject()
    
    def get_project_data(self):
        """
        获取项目数据
        
        Returns:
            dict: 项目数据
        """
        return self.project_data
