#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
用户管理界面模块

实现用户ID管理界面，包括用户登录、个人信息管理和功能导航。
"""

import sys
import os
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGridLayout,
    QPushButton, QLabel, QLineEdit, QFrame,
    QMessageBox, QGroupBox, QSizePolicy, QListWidget,
    QListWidgetItem, QInputDialog, QSplitter, QScrollArea
)
from PyQt5.QtCore import Qt, pyqtSignal, QSize
from PyQt5.QtGui import QIcon, QFont, QPalette, QPixmap

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from src.services.user_service import UserService
from src.core.user_id_manager import UserIDManager
from src.core.event_bus import get_event_bus, EventNames
from src.utils.logger import get_logger


class SampleItemWidget(QWidget):
    """
    自定义样本项组件，包含样本信息和功能模块按钮
    """
    
    # 信号定义
    sample_selected = pyqtSignal(str, str)  # user_id, display_name
    function_requested = pyqtSignal(str, str)  # function_name, user_id
    
    def __init__(self, user_id_info, parent=None):
        super().__init__(parent)
        self.user_id_info = user_id_info
        self.user_id = user_id_info['user_id']
        self.display_name = user_id_info['display_name']
        self.is_selected = False
        
        self.init_ui()
        self.setup_connections()
    
    def init_ui(self):
        """
        初始化UI
        """
        self.setFixedHeight(80)
        self.setStyleSheet("""
            SampleItemWidget {
                background-color: #FFFFFF;
                border: 1px solid #E0E0E0;
                border-radius: 5px;
                margin: 2px;
            }
            SampleItemWidget:hover {
                border-color: #2E86AB;
                background-color: #F8F9FA;
            }
        """)
        
        # 主布局
        main_layout = QHBoxLayout(self)
        main_layout.setContentsMargins(10, 5, 10, 5)
        main_layout.setSpacing(10)
        
        # 左侧：样本信息
        info_layout = QVBoxLayout()
        info_layout.setSpacing(2)
        
        # 样本名称
        self.name_label = QLabel(self.display_name)
        self.name_label.setStyleSheet("""
            QLabel {
                font-size: 14px;
                font-weight: bold;
                color: #2E86AB;
            }
        """)
        info_layout.addWidget(self.name_label)
        
        # 样本ID和描述
        info_text = f"ID: {self.user_id}"
        if self.user_id_info.get('description'):
            info_text += f" | {self.user_id_info['description']}"
        
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
        
        # 右侧：功能按钮
        buttons_layout = QHBoxLayout()
        buttons_layout.setSpacing(5)
        
        # DICOM ROI按钮
        self.dicom_button = QPushButton("DICOM")
        self.dicom_button.setFixedSize(60, 25)
        self.dicom_button.setStyleSheet("""
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
            QPushButton:pressed {
                background-color: #3d8b40;
            }
        """)
        buttons_layout.addWidget(self.dicom_button)
        
        # 文本处理按钮
        self.text_button = QPushButton("文本")
        self.text_button.setFixedSize(60, 25)
        self.text_button.setStyleSheet("""
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
            QPushButton:pressed {
                background-color: #1565C0;
            }
        """)
        buttons_layout.addWidget(self.text_button)
        
        # OCR识别按钮
        self.ocr_button = QPushButton("OCR")
        self.ocr_button.setFixedSize(60, 25)
        self.ocr_button.setStyleSheet("""
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
            QPushButton:pressed {
                background-color: #E65100;
            }
        """)
        buttons_layout.addWidget(self.ocr_button)
        
        main_layout.addLayout(buttons_layout)
    
    def setup_connections(self):
        """
        设置信号连接
        """
        self.dicom_button.clicked.connect(lambda: self.on_function_clicked('dicom_roi'))
        self.text_button.clicked.connect(lambda: self.on_function_clicked('text_processing'))
        self.ocr_button.clicked.connect(lambda: self.on_function_clicked('ocr_recognition'))
    
    def on_function_clicked(self, function_name):
        """
        功能按钮点击处理
        
        Args:
            function_name: 功能名称
        """
        # 先选中当前样本
        self.sample_selected.emit(self.user_id, self.display_name)
        # 然后请求切换到对应功能模块
        self.function_requested.emit(function_name, self.user_id)
    
    def mousePressEvent(self, event):
        """
        鼠标点击事件 - 选中样本
        """
        if event.button() == Qt.LeftButton:
            self.sample_selected.emit(self.user_id, self.display_name)
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
                SampleItemWidget {
                    background-color: #E3F2FD;
                    border: 2px solid #2E86AB;
                    border-radius: 5px;
                    margin: 2px;
                }
            """)
        else:
            self.setStyleSheet("""
                SampleItemWidget {
                    background-color: #FFFFFF;
                    border: 1px solid #E0E0E0;
                    border-radius: 5px;
                    margin: 2px;
                }
                SampleItemWidget:hover {
                    border-color: #2E86AB;
                    background-color: #F8F9FA;
                }
            """)


class UserManagementWidget(QWidget):
    """
    用户管理界面类
    """
    
    # 信号定义
    user_logged_in = pyqtSignal(str)  # 用户登录信号
    function_selected = pyqtSignal(str)  # 功能选择信号
    
    def __init__(self):
        super().__init__()
        
        self.user_service = UserService()
        self.user_id_manager = UserIDManager()
        self.event_bus = get_event_bus()
        self.logger = get_logger(__name__)
        self.current_user = None
        self.selected_user_id = None
        
        # 样本项列表
        self.sample_items = []
        
        self.init_ui()
        self.setup_connections()
        self.load_user_ids()
    
    def init_ui(self):
        """
        初始化用户界面
        """
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(20, 20, 20, 20)
        main_layout.setSpacing(20)
        
        # 创建分割器
        splitter = QSplitter(Qt.Horizontal)
        
        # 左侧：用户ID管理区域
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)
        self.create_user_id_management(left_layout)
        
        # 右侧：登录和功能导航区域
        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)
        self.create_login_area(right_layout)
        self.create_function_navigation(right_layout)
        
        splitter.addWidget(left_widget)
        splitter.addWidget(right_widget)
        # 设置左侧样本管理区域宽度为40%，右侧为60%
        splitter.setStretchFactor(0, 2)  # 40%
        splitter.setStretchFactor(1, 3)  # 60%
        
        main_layout.addWidget(splitter)
        
        # 应用样式
        self.apply_styles()
    
    def create_login_area(self, parent_layout):
        """
        创建登录区域
        
        Args:
            parent_layout: 父布局
        """
        # 登录组框
        login_group = QGroupBox("项目管理")
        login_group.setFixedHeight(200)
        login_layout = QVBoxLayout(login_group)
        login_layout.setSpacing(15)
        
        # 登录表单
        form_layout = QGridLayout()
        form_layout.setSpacing(10)
        
        # 用户ID输入
        user_id_label = QLabel("项目名称:")
        self.user_id_input = QLineEdit()
        self.user_id_input.setPlaceholderText("请输入项目名称")
        self.user_id_input.setText("demo_user")  # 默认演示用户
        
        form_layout.addWidget(user_id_label, 0, 0)
        form_layout.addWidget(self.user_id_input, 0, 1)
        
        # 密码输入（可选）
        password_label = QLabel("密码:")
        self.password_input = QLineEdit()
        self.password_input.setPlaceholderText("请输入密码（可选）")
        self.password_input.setEchoMode(QLineEdit.Password)
        
        form_layout.addWidget(password_label, 1, 0)
        form_layout.addWidget(self.password_input, 1, 1)
        
        login_layout.addLayout(form_layout)
        
        # 登录按钮
        button_layout = QHBoxLayout()
        
        self.login_button = QPushButton("登录")
        self.login_button.setFixedSize(100, 35)
        
        self.create_user_button = QPushButton("创建用户")
        self.create_user_button.setFixedSize(100, 35)
        
        button_layout.addStretch()
        button_layout.addWidget(self.login_button)
        button_layout.addWidget(self.create_user_button)
        button_layout.addStretch()
        
        login_layout.addLayout(button_layout)
        
        parent_layout.addWidget(login_group)
    
    def create_user_id_management(self, parent_layout):
        """
        创建用户ID管理区域
        
        Args:
            parent_layout: 父布局
        """
        # 用户ID管理组框
        user_id_group = QGroupBox("样本管理")
        user_id_layout = QVBoxLayout(user_id_group)
        user_id_layout.setSpacing(10)
        
        # 当前选中的用户ID显示
        current_layout = QHBoxLayout()
        current_label = QLabel("当前样本:")
        self.current_user_id_label = QLabel("未选择")
        self.current_user_id_label.setStyleSheet("""
            QLabel {
                font-weight: bold;
                color: #2E86AB;
                background-color: #F8F9FA;
                padding: 5px;
                border-radius: 3px;
            }
        """)
        current_layout.addWidget(current_label)
        current_layout.addWidget(self.current_user_id_label)
        current_layout.addStretch()
        
        user_id_layout.addLayout(current_layout)
        
        # 用户ID列表
        list_label = QLabel("样本列表:")
        user_id_layout.addWidget(list_label)
        
        # 创建滚动区域来容纳自定义样本项
        from PyQt5.QtWidgets import QScrollArea, QFrame
        
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
        
        # 创建容器widget来放置样本项
        self.samples_container = QWidget()
        self.samples_layout = QVBoxLayout(self.samples_container)
        self.samples_layout.setSpacing(3)
        self.samples_layout.setContentsMargins(5, 5, 5, 5)
        
        self.scroll_area.setWidget(self.samples_container)
        user_id_layout.addWidget(self.scroll_area)
        
        # 保留原有的QListWidget用于兼容性（隐藏）
        self.user_id_list = QListWidget()
        self.user_id_list.setVisible(False)
        
        # 用户ID管理按钮
        button_layout = QHBoxLayout()
        
        self.add_user_id_button = QPushButton("添加样本")
        self.add_user_id_button.setFixedSize(80, 30)
        
        self.remove_user_id_button = QPushButton("删除样本")
        self.remove_user_id_button.setFixedSize(80, 30)
        self.remove_user_id_button.setEnabled(False)
        
        self.refresh_button = QPushButton("刷新")
        self.refresh_button.setFixedSize(60, 30)
        
        button_layout.addWidget(self.add_user_id_button)
        button_layout.addWidget(self.remove_user_id_button)
        button_layout.addWidget(self.refresh_button)
        button_layout.addStretch()
        
        user_id_layout.addLayout(button_layout)
        
        # 统计信息
        self.stats_label = QLabel("总计: 0 个样本")
        self.stats_label.setStyleSheet("color: #666666; font-size: 11px;")
        user_id_layout.addWidget(self.stats_label)
        
        parent_layout.addWidget(user_id_group)
    
    def create_function_navigation(self, parent_layout):
        """
        创建功能导航区域
        
        Args:
            parent_layout: 父布局
        """
        # 功能导航组框
        nav_group = QGroupBox("功能模块")
        nav_layout = QGridLayout(nav_group)
        nav_layout.setSpacing(20)
        
        # 功能按钮数据
        functions = [
            {
                'name': 'DICOM ROI识别',
                'description': '导入DICOM文件\n识别ROI区域\n导出分析结果',
                'icon': 'dicom_icon.png',
                'signal': 'dicom_roi'
            },
            {
                'name': '文本处理',
                'description': '医疗文本编辑\n批量处理工具\n格式转换',
                'icon': 'text_icon.png',
                'signal': 'text_processing'
            },
            {
                'name': 'OCR识别',
                'description': '图像文字识别\n多语言支持\n结果编辑导出',
                'icon': 'ocr_icon.png',
                'signal': 'ocr_recognition'
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
        button.setEnabled(False)  # 初始状态禁用，登录后启用
        
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
                color: #2E86AB;
            }
        """)
        
        # 功能描述
        desc_label = QLabel(func_data['description'])
        desc_label.setAlignment(Qt.AlignCenter)
        desc_label.setStyleSheet("""
            QLabel {
                font-size: 12px;
                color: #666666;
                line-height: 1.4;
            }
        """)
        
        button_layout.addWidget(icon_label)
        button_layout.addWidget(name_label)
        button_layout.addWidget(desc_label)
        
        # 连接点击事件
        button.clicked.connect(lambda: self.function_selected.emit(func_data['signal']))
        
        return button
    
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
                color: white;
                background-color: #2E86AB;
                border: none;
                border-radius: 6px;
                padding: 8px 16px;
            }
            QPushButton:hover {
                background-color: #1976D2;
            }
            QPushButton:pressed {
                background-color: #1565C0;
            }
            QPushButton:disabled {
                background-color: #CCCCCC;
                color: #666666;
            }
        """)
        
        # 功能按钮特殊样式
        function_button_style = """
            QPushButton {
                background-color: #FFFFFF;
                border: 2px solid #E0E0E0;
                border-radius: 8px;
                color: #2E86AB;
            }
            QPushButton:hover {
                border: 2px solid #2E86AB;
                background-color: #F8F9FA;
            }
            QPushButton:pressed {
                background-color: #E3F2FD;
            }
            QPushButton:disabled {
                border: 2px solid #F0F0F0;
                background-color: #FAFAFA;
                color: #CCCCCC;
            }
        """
        
        for button in self.function_buttons:
            button.setStyleSheet(function_button_style)
    
    def setup_connections(self):
        """
        设置信号连接
        """
        self.login_button.clicked.connect(self.on_login_clicked)
        self.create_user_button.clicked.connect(self.on_create_user_clicked)
        
        # 回车键登录
        self.user_id_input.returnPressed.connect(self.on_login_clicked)
        self.password_input.returnPressed.connect(self.on_login_clicked)
        
        # 用户ID管理连接
        self.user_id_list.itemClicked.connect(self.handle_user_id_selection)
        self.add_user_id_button.clicked.connect(self.handle_add_user_id)
        self.remove_user_id_button.clicked.connect(self.handle_remove_user_id)
        self.refresh_button.clicked.connect(self.load_user_ids)
        
        # 事件总线连接
        self.event_bus.subscribe(EventNames.USER_ID_SELECTED, self.on_user_id_selected)
        self.event_bus.subscribe(EventNames.USER_ID_LIST_UPDATED, self.on_user_id_list_updated)
    
    def on_login_clicked(self):
        """
        登录按钮点击处理
        """
        user_id = self.user_id_input.text().strip()
        password = self.password_input.text().strip()
        
        if not user_id:
            QMessageBox.warning(self, '警告', '请输入项目名称！')
            return
        
        try:
            # 验证用户
            if self.user_service.authenticate_user(user_id, password):
                self.current_user = user_id
                
                # 启用功能按钮
                for button in self.function_buttons:
                    button.setEnabled(True)
                
                # 发送登录成功信号
                self.user_logged_in.emit(user_id)
                
                QMessageBox.information(self, '成功', f'用户 {user_id} 登录成功！')
                self.logger.info(f"用户 {user_id} 登录成功")
                
            else:
                QMessageBox.warning(self, '错误', '项目名称或密码错误！')
                
        except Exception as e:
            QMessageBox.critical(self, '错误', f'登录失败: {str(e)}')
            self.logger.error(f"登录失败: {e}")
    
    def on_create_user_clicked(self):
        """
        创建用户按钮点击处理
        """
        user_id = self.user_id_input.text().strip()
        
        if not user_id:
            QMessageBox.warning(self, '警告', '请输入项目名称！')
            return
        
        try:
            # 创建用户
            user_info = {
                'name': user_id,  # 默认使用用户ID作为姓名
                'email': f'{user_id}@medical.com',
                'preferences': {
                    'theme': 'default',
                    'language': 'zh_CN'
                }
            }
            
            if self.user_service.create_user(user_id, user_info):
                QMessageBox.information(self, '成功', f'用户 {user_id} 创建成功！')
                self.logger.info(f"用户 {user_id} 创建成功")
            else:
                QMessageBox.warning(self, '错误', '用户创建失败，可能用户已存在！')
                
        except Exception as e:
            QMessageBox.critical(self, '错误', f'创建用户失败: {str(e)}')
            self.logger.error(f"创建用户失败: {e}")
    
    def clear_sample_items(self):
        """
        清空现有的样本项
        """
        for sample_item in self.sample_items:
            sample_item.setParent(None)
            sample_item.deleteLater()
        self.sample_items.clear()
    
    def on_sample_selected(self, user_id, display_name):
        """
        处理样本选择
        
        Args:
            user_id: 用户ID
            display_name: 显示名称
        """
        # 更新选中状态
        for sample_item in self.sample_items:
            sample_item.set_selected(sample_item.user_id == user_id)
        
        # 更新当前选中的用户ID
        self.selected_user_id = user_id
        self.current_user_id_label.setText(display_name)
        self.remove_user_id_button.setEnabled(True)
        
        # 选择用户ID
        self.user_id_manager.select_user_id(user_id)
        
        # 发布事件
        self.event_bus.publish(EventNames.USER_ID_SELECTED, {
            'user_id': user_id,
            'display_name': display_name
        })
        
        self.logger.info(f"已选择样本: {display_name} ({user_id})")
    
    def on_function_requested(self, function_name, user_id):
        """
        处理功能请求
        
        Args:
            function_name: 功能名称
            user_id: 用户ID
        """
        # 发送功能选择信号
        self.function_selected.emit(function_name)
        self.logger.info(f"请求切换到功能模块: {function_name} (用户: {user_id})")
    
    def load_user_ids(self):
        """
        加载用户ID列表
        """
        try:
            user_ids = self.user_id_manager.get_user_id_details_list()
            
            # 清空现有的样本项
            self.clear_sample_items()
            
            # 创建新的样本项
            for user_id_info in user_ids:
                sample_item = SampleItemWidget(user_id_info)
                
                # 连接信号
                sample_item.sample_selected.connect(self.on_sample_selected)
                sample_item.function_requested.connect(self.on_function_requested)
                
                # 添加到布局
                self.samples_layout.addWidget(sample_item)
                self.sample_items.append(sample_item)
            
            # 添加弹性空间
            self.samples_layout.addStretch()
            
            # 更新统计信息
            stats = self.user_id_manager.get_user_id_statistics()
            self.stats_label.setText(f"总计: {stats['total_count']} 个样本")
            
            # 如果有当前选中的用户ID，保持选中状态
            if self.selected_user_id:
                for sample_item in self.sample_items:
                    if sample_item.user_id == self.selected_user_id:
                        sample_item.set_selected(True)
                        break
            
            self.logger.info(f"已加载 {len(user_ids)} 个用户ID")
            
        except Exception as e:
            self.logger.error(f"加载用户ID列表失败: {str(e)}")
            self.show_error_message("加载失败", f"无法加载用户ID列表: {str(e)}")
    
    def handle_user_id_selection(self, item):
        """
        处理用户ID选择（保留兼容性，但主要逻辑已移至on_sample_selected）
        
        Args:
            item: 选中的列表项
        """
        try:
            user_id = item.data(Qt.UserRole)
            if user_id:
                display_name = user_id
                # 尝试从item文本中提取显示名称
                item_text = item.text()
                if ' (' in item_text:
                    display_name = item_text.split(' (')[0]
                
                # 调用新的样本选择方法
                self.on_sample_selected(user_id, display_name)
            else:
                # 清除选择
                self.selected_user_id = None
                self.current_user_id_label.setText("未选择")
                self.remove_user_id_button.setEnabled(False)
                
                # 清除所有样本项的选中状态
                for sample_item in self.sample_items:
                    sample_item.set_selected(False)
        except Exception as e:
            self.logger.error(f"处理用户ID选择失败: {str(e)}")
            self.show_error_message("选择失败", f"处理用户ID选择时发生错误: {str(e)}")
    
    def handle_add_user_id(self):
        """
        处理添加用户ID
        """
        try:
            # 弹出输入对话框
            user_id, ok1 = QInputDialog.getText(
                self, '添加项目', '请输入项目ID (唯一标识符):'
            )
            
            if not ok1 or not user_id.strip():
                return
            
            user_id = user_id.strip()
            
            # 输入显示名称
            display_name, ok2 = QInputDialog.getText(
                self, '添加项目', '请输入显示名称:', text=user_id
            )
            
            if not ok2 or not display_name.strip():
                return
            
            display_name = display_name.strip()
            
            # 输入描述（可选）
            description, ok3 = QInputDialog.getText(
                self, '添加项目', '请输入描述 (可选):'
            )
            
            if not ok3:
                return
            
            # 添加用户ID
            success = self.user_id_manager.add_user_id(
                user_id=user_id,
                display_name=display_name,
                description=description.strip() if description.strip() else None
            )
            
            if success:
                self.show_success_message("添加成功", f"项目 '{display_name}' 已添加")
                self.load_user_ids()
            else:
                self.show_error_message("添加失败", "该项目ID已存在")
                
        except Exception as e:
            self.logger.error(f"添加用户ID失败: {str(e)}")
            self.show_error_message("添加失败", f"添加项目时发生错误: {str(e)}")
    
    def handle_remove_user_id(self):
        """
        处理删除用户ID
        """
        try:
            # 检查是否有选中的样本
            if not self.selected_user_id:
                self.show_warning_message("删除失败", "请先选择要删除的项目")
                return
            
            user_id = self.selected_user_id
            display_name = self.current_user_id_label.text()
            
            # 如果显示名称是"未选择"，使用用户ID作为显示名称
            if display_name == "未选择":
                display_name = user_id
            
            # 确认删除
            reply = QMessageBox.question(
                self, '确认删除',
                f"确定要删除项目 '{display_name}' 吗？\n\n注意：这将删除与该项目相关的所有分析数据！",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No
            )
            
            if reply == QMessageBox.Yes:
                success = self.user_id_manager.remove_user_id(user_id)
                if success:
                    self.show_success_message("删除成功", f"项目 '{display_name}' 已删除")
                    
                    # 如果删除的是当前选中的用户ID，清除选择
                    if self.selected_user_id == user_id:
                        self.selected_user_id = None
                        self.current_user_id_label.setText("未选择")
                        self.remove_user_id_button.setEnabled(False)
                        self.user_id_manager.clear_selection()
                    
                    self.load_user_ids()
                else:
                    self.show_error_message("删除失败", "无法删除该项目")
                    
        except Exception as e:
            self.logger.error(f"删除用户ID失败: {str(e)}")
            self.show_error_message("删除失败", f"删除项目时发生错误: {str(e)}")
    
    def on_user_id_selected(self, data):
        """
        响应用户ID选择事件
        
        Args:
            data: 事件数据
        """
        user_id = data.get('user_id')
        display_name = data.get('display_name', user_id)
        
        if user_id != self.selected_user_id:
            self.selected_user_id = user_id
            self.current_user_id_label.setText(display_name)
            
            # 更新列表选择状态
            for i in range(self.user_id_list.count()):
                item = self.user_id_list.item(i)
                if item.data(Qt.UserRole) == user_id:
                    self.user_id_list.setCurrentItem(item)
                    self.remove_user_id_button.setEnabled(True)
                    break
    
    def on_user_id_list_updated(self, data):
        """
        响应用户ID列表更新事件
        
        Args:
            data: 事件数据
        """
        self.load_user_ids()
    
    def show_success_message(self, title, message):
        """
        显示成功消息
        """
        QMessageBox.information(self, title, message)
    
    def show_error_message(self, title, message):
        """
        显示错误消息
        """
        QMessageBox.critical(self, title, message)
    
    def show_warning_message(self, title, message):
        """
        显示警告消息
        """
        QMessageBox.warning(self, title, message)