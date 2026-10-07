#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
样本管理对话框模块

实现样本数据的增删改查功能，与项目进行关联映射。
"""

import sys
import os
from datetime import datetime
from typing import List, Optional, Dict, Any

from PyQt5.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QGridLayout,
    QPushButton, QLabel, QLineEdit, QTextEdit, QTableWidget,
    QTableWidgetItem, QHeaderView, QMessageBox, QGroupBox,
    QComboBox, QDateTimeEdit, QCheckBox, QSplitter,
    QFrame, QScrollArea, QWidget, QFormLayout, QApplication
)
from PyQt5.QtCore import Qt, pyqtSignal, QDateTime
from PyQt5.QtGui import QFont, QIcon, QPalette, QColor

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from src.core.app_config import AppConfig
from src.core.ui_scale_profile import dialog_target, is_macos
from src.utils.logger import get_logger
from src.services.database_service import DatabaseService


class SampleManagementDialog(QDialog):
    """
    样本管理对话框类
    """
    
    # 信号定义
    sample_updated = pyqtSignal()  # 样本数据更新信号
    
    def __init__(self, project_id: str = None, parent=None):
        """
        初始化样本管理对话框
        
        Args:
            project_id: 当前项目ID
            parent: 父窗口
        """
        super().__init__(parent)
        self.project_id = project_id or "default"
        self.config = AppConfig()
        self.logger = get_logger(__name__)
        self.db_service = DatabaseService()
        
        # 样本数据
        self.samples = []
        self.current_sample = None
        
        self.init_ui()
        self.setup_connections()
        self.load_samples()
        
    def init_ui(self):
        """
        初始化用户界面
        """
        self.setWindowTitle(f"样本管理 - 项目: {self.project_id}")
        self._apply_dialog_size(1000, 700, 900, 620)
        self.setModal(True)
        
        # 主布局
        main_layout = QHBoxLayout(self)
        main_layout.setContentsMargins(10, 10, 10, 10)
        main_layout.setSpacing(10)
        
        # 创建分割器
        splitter = QSplitter(Qt.Horizontal)
        
        # 左侧：样本列表
        left_widget = self.create_sample_list_area()
        splitter.addWidget(left_widget)
        
        # 右侧：样本详情
        right_widget = self.create_sample_detail_area()
        splitter.addWidget(right_widget)
        
        # 设置分割器比例
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 1)
        
        main_layout.addWidget(splitter)
        
        # 应用样式
        self.apply_styles()

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
        
    def create_sample_list_area(self):
        """
        创建样本列表区域
        
        Returns:
            QWidget: 样本列表区域部件
        """
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(5, 5, 5, 5)
        
        # 标题和操作按钮
        header_layout = QHBoxLayout()
        
        title_label = QLabel("样本列表")
        title_label.setStyleSheet("""
            QLabel {
                font-size: 16px;
                font-weight: bold;
                color: #2E86AB;
                padding: 5px;
            }
        """)
        
        # 操作按钮
        self.add_sample_button = QPushButton("新增样本")
        self.add_sample_button.setFixedSize(80, 30)
        
        self.delete_sample_button = QPushButton("删除样本")
        self.delete_sample_button.setFixedSize(80, 30)
        self.delete_sample_button.setEnabled(False)
        
        self.refresh_button = QPushButton("刷新")
        self.refresh_button.setFixedSize(60, 30)
        
        header_layout.addWidget(title_label)
        header_layout.addStretch()
        header_layout.addWidget(self.add_sample_button)
        header_layout.addWidget(self.delete_sample_button)
        header_layout.addWidget(self.refresh_button)
        
        layout.addLayout(header_layout)
        
        # 样本表格
        self.sample_table = QTableWidget()
        self.sample_table.setColumnCount(4)
        self.sample_table.setHorizontalHeaderLabels(["样本ID", "显示名称", "创建时间", "状态"])
        
        # 设置表格属性
        self.sample_table.setSelectionBehavior(QTableWidget.SelectRows)
        self.sample_table.setSelectionMode(QTableWidget.SingleSelection)
        self.sample_table.setAlternatingRowColors(True)
        
        # 设置列宽
        header = self.sample_table.horizontalHeader()
        header.setStretchLastSection(True)
        header.setSectionResizeMode(0, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(1, QHeaderView.Stretch)
        header.setSectionResizeMode(2, QHeaderView.ResizeToContents)
        
        layout.addWidget(self.sample_table)
        
        # 统计信息
        self.stats_label = QLabel("总计: 0 个样本")
        self.stats_label.setStyleSheet("""
            QLabel {
                font-size: 12px;
                color: #666666;
                padding: 5px;
            }
        """)
        layout.addWidget(self.stats_label)
        
        return widget
        
    def create_sample_detail_area(self):
        """
        创建样本详情区域
        
        Returns:
            QWidget: 样本详情区域部件
        """
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(5, 5, 5, 5)
        
        # 标题
        title_label = QLabel("样本详情")
        title_label.setStyleSheet("""
            QLabel {
                font-size: 16px;
                font-weight: bold;
                color: #2E86AB;
                padding: 5px;
            }
        """)
        layout.addWidget(title_label)
        
        # 滚动区域
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setFrameShape(QFrame.NoFrame)
        
        # 详情表单
        form_widget = QWidget()
        form_layout = QFormLayout(form_widget)
        form_layout.setSpacing(15)
        
        # 样本ID
        self.sample_id_edit = QLineEdit()
        self.sample_id_edit.setPlaceholderText("自动生成")
        self.sample_id_edit.setReadOnly(True)
        form_layout.addRow("样本ID:", self.sample_id_edit)
        
        # 显示名称
        self.display_name_edit = QLineEdit()
        self.display_name_edit.setPlaceholderText("请输入样本显示名称")
        form_layout.addRow("显示名称:", self.display_name_edit)
        
        # 描述
        self.description_edit = QTextEdit()
        self.description_edit.setPlaceholderText("请输入样本描述信息")
        self.description_edit.setMaximumHeight(100)
        form_layout.addRow("描述:", self.description_edit)
        
        # 项目ID
        self.project_id_label = QLabel(self.project_id)
        self.project_id_label.setStyleSheet("color: #666666;")
        form_layout.addRow("所属项目:", self.project_id_label)
        
        # 创建时间
        self.created_at_edit = QDateTimeEdit()
        self.created_at_edit.setDateTime(QDateTime.currentDateTime())
        self.created_at_edit.setReadOnly(True)
        form_layout.addRow("创建时间:", self.created_at_edit)
        
        # 最后使用时间
        self.last_used_edit = QDateTimeEdit()
        self.last_used_edit.setReadOnly(True)
        form_layout.addRow("最后使用:", self.last_used_edit)
        
        # 是否激活
        self.is_active_checkbox = QCheckBox("激活状态")
        self.is_active_checkbox.setChecked(True)
        form_layout.addRow("", self.is_active_checkbox)
        
        # 元数据
        self.metadata_edit = QTextEdit()
        self.metadata_edit.setPlaceholderText("JSON格式的元数据")
        self.metadata_edit.setMaximumHeight(80)
        form_layout.addRow("元数据:", self.metadata_edit)
        
        scroll_area.setWidget(form_widget)
        layout.addWidget(scroll_area)
        
        # 操作按钮
        button_layout = QHBoxLayout()
        
        self.save_button = QPushButton("保存")
        self.save_button.setFixedSize(80, 35)
        self.save_button.setEnabled(False)
        
        self.cancel_button = QPushButton("取消")
        self.cancel_button.setFixedSize(80, 35)
        
        self.new_button = QPushButton("新建")
        self.new_button.setFixedSize(80, 35)
        
        button_layout.addStretch()
        button_layout.addWidget(self.new_button)
        button_layout.addWidget(self.save_button)
        button_layout.addWidget(self.cancel_button)
        
        layout.addLayout(button_layout)
        
        return widget
        
    def setup_connections(self):
        """
        设置信号连接
        """
        # 表格选择变化
        self.sample_table.selectionModel().selectionChanged.connect(self.on_sample_selection_changed)
        
        # 按钮点击
        self.add_sample_button.clicked.connect(self.add_sample)
        self.delete_sample_button.clicked.connect(self.delete_sample)
        self.refresh_button.clicked.connect(self.load_samples)
        
        self.save_button.clicked.connect(self.save_sample)
        self.cancel_button.clicked.connect(self.cancel_edit)
        self.new_button.clicked.connect(self.new_sample)
        
        # 表单字段变化
        self.display_name_edit.textChanged.connect(self.on_form_changed)
        self.description_edit.textChanged.connect(self.on_form_changed)
        self.is_active_checkbox.toggled.connect(self.on_form_changed)
        self.metadata_edit.textChanged.connect(self.on_form_changed)
        
    def load_samples(self):
        """
        加载样本数据
        """
        try:
            # 查询当前项目的样本数据
            with self.db_service.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT user_id, display_name, description, project_id,
                           created_at, last_used, metadata, is_active
                    FROM user_ids
                    WHERE project_id = ? OR (project_id IS NULL AND ? = 'default')
                    ORDER BY created_at DESC
                """, (self.project_id, self.project_id))
                
                rows = cursor.fetchall()
                self.samples = [dict(row) for row in rows]
            
            # 更新表格
            self.update_sample_table()
            
            # 更新统计信息
            self.stats_label.setText(f"总计: {len(self.samples)} 个样本")
            
            self.logger.info(f"加载了 {len(self.samples)} 个样本")
            
        except Exception as e:
            self.logger.error(f"加载样本数据失败: {e}")
            QMessageBox.warning(self, "错误", f"加载样本数据失败: {e}")
            
    def update_sample_table(self):
        """
        更新样本表格
        """
        self.sample_table.setRowCount(len(self.samples))
        
        for row, sample in enumerate(self.samples):
            # 样本ID
            self.sample_table.setItem(row, 0, QTableWidgetItem(sample['user_id']))
            
            # 显示名称
            self.sample_table.setItem(row, 1, QTableWidgetItem(sample['display_name']))
            
            # 创建时间
            created_at = sample.get('created_at', '')
            if created_at:
                try:
                    dt = datetime.fromisoformat(created_at.replace('Z', '+00:00'))
                    created_at = dt.strftime('%Y-%m-%d %H:%M')
                except:
                    pass
            self.sample_table.setItem(row, 2, QTableWidgetItem(created_at))
            
            # 状态
            status = "激活" if sample.get('is_active', 1) else "禁用"
            self.sample_table.setItem(row, 3, QTableWidgetItem(status))
            
    def on_sample_selection_changed(self):
        """
        样本选择变化处理
        """
        selected_rows = self.sample_table.selectionModel().selectedRows()
        
        if selected_rows:
            row = selected_rows[0].row()
            if 0 <= row < len(self.samples):
                self.current_sample = self.samples[row]
                self.load_sample_detail(self.current_sample)
                self.delete_sample_button.setEnabled(True)
            else:
                self.current_sample = None
                self.clear_sample_detail()
                self.delete_sample_button.setEnabled(False)
        else:
            self.current_sample = None
            self.clear_sample_detail()
            self.delete_sample_button.setEnabled(False)
            
    def load_sample_detail(self, sample: Dict[str, Any]):
        """
        加载样本详情
        
        Args:
            sample: 样本数据字典
        """
        self.sample_id_edit.setText(sample.get('user_id', ''))
        self.display_name_edit.setText(sample.get('display_name', ''))
        self.description_edit.setPlainText(sample.get('description', ''))
        
        # 创建时间
        created_at = sample.get('created_at', '')
        if created_at:
            try:
                dt = datetime.fromisoformat(created_at.replace('Z', '+00:00'))
                self.created_at_edit.setDateTime(QDateTime.fromString(dt.isoformat(), Qt.ISODate))
            except:
                self.created_at_edit.setDateTime(QDateTime.currentDateTime())
        
        # 最后使用时间
        last_used = sample.get('last_used', '')
        if last_used:
            try:
                dt = datetime.fromisoformat(last_used.replace('Z', '+00:00'))
                self.last_used_edit.setDateTime(QDateTime.fromString(dt.isoformat(), Qt.ISODate))
            except:
                self.last_used_edit.setDateTime(QDateTime.currentDateTime())
        else:
            self.last_used_edit.setDateTime(QDateTime.currentDateTime())
        
        # 激活状态
        self.is_active_checkbox.setChecked(bool(sample.get('is_active', 1)))
        
        # 元数据
        metadata = sample.get('metadata', '')
        self.metadata_edit.setPlainText(metadata or '')
        
        # 重置保存按钮状态
        self.save_button.setEnabled(False)
        
    def clear_sample_detail(self):
        """
        清空样本详情
        """
        self.sample_id_edit.clear()
        self.display_name_edit.clear()
        self.description_edit.clear()
        self.created_at_edit.setDateTime(QDateTime.currentDateTime())
        self.last_used_edit.setDateTime(QDateTime.currentDateTime())
        self.is_active_checkbox.setChecked(True)
        self.metadata_edit.clear()
        self.save_button.setEnabled(False)
        
    def on_form_changed(self):
        """
        表单字段变化处理
        """
        self.save_button.setEnabled(True)
        
    def add_sample(self):
        """
        添加新样本
        """
        self.new_sample()
        
    def new_sample(self):
        """
        新建样本
        """
        self.current_sample = None
        self.clear_sample_detail()
        self.sample_table.clearSelection()
        self.delete_sample_button.setEnabled(False)
        
        # 生成新的样本ID
        import uuid
        new_id = f"sample_{uuid.uuid4().hex[:8]}"
        self.sample_id_edit.setText(new_id)
        
        # 设置焦点到显示名称
        self.display_name_edit.setFocus()
        
    def save_sample(self):
        """
        保存样本
        """
        try:
            # 验证输入
            user_id = self.sample_id_edit.text().strip()
            display_name = self.display_name_edit.text().strip()
            
            if not user_id:
                QMessageBox.warning(self, "警告", "样本ID不能为空！")
                return
                
            if not display_name:
                QMessageBox.warning(self, "警告", "显示名称不能为空！")
                return
            
            # 准备数据
            description = self.description_edit.toPlainText().strip()
            is_active = 1 if self.is_active_checkbox.isChecked() else 0
            metadata = self.metadata_edit.toPlainText().strip()
            
            # 验证元数据JSON格式
            if metadata:
                try:
                    import json
                    json.loads(metadata)
                except json.JSONDecodeError:
                    QMessageBox.warning(self, "警告", "元数据必须是有效的JSON格式！")
                    return
            
            with self.db_service.get_connection() as conn:
                cursor = conn.cursor()
                
                if self.current_sample:
                    # 更新现有样本
                    cursor.execute("""
                        UPDATE user_ids
                        SET display_name = ?, description = ?, is_active = ?,
                            metadata = ?, last_used = CURRENT_TIMESTAMP
                        WHERE user_id = ?
                    """, (display_name, description, is_active, metadata, user_id))
                else:
                    # 检查样本ID是否已存在
                    cursor.execute("SELECT COUNT(*) FROM user_ids WHERE user_id = ?", (user_id,))
                    if cursor.fetchone()[0] > 0:
                        QMessageBox.warning(self, "警告", f"样本ID '{user_id}' 已存在！")
                        return
                    
                    # 创建新样本
                    cursor.execute("""
                        INSERT INTO user_ids (user_id, display_name, description, project_id,
                                            created_at, last_used, metadata, is_active)
                        VALUES (?, ?, ?, ?, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP, ?, ?)
                    """, (user_id, display_name, description, self.project_id, metadata, is_active))
                
                conn.commit()
            
            # 重新加载数据
            self.load_samples()
            
            # 发送更新信号
            self.sample_updated.emit()
            
            # 重置保存按钮状态
            self.save_button.setEnabled(False)
            
            QMessageBox.information(self, "成功", "样本保存成功！")
            
        except Exception as e:
            self.logger.error(f"保存样本失败: {e}")
            QMessageBox.warning(self, "错误", f"保存样本失败: {e}")
            
    def cancel_edit(self):
        """
        取消编辑
        """
        if self.current_sample:
            self.load_sample_detail(self.current_sample)
        else:
            self.clear_sample_detail()
        
        self.save_button.setEnabled(False)
        
    def delete_sample(self):
        """
        删除样本
        """
        if not self.current_sample:
            return
            
        user_id = self.current_sample['user_id']
        display_name = self.current_sample['display_name']
        
        # 确认删除
        reply = QMessageBox.question(
            self, "确认删除",
            f"确定要删除样本 '{display_name}' ({user_id}) 吗？\n\n注意：删除样本将同时删除相关的所有数据！",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No
        )
        
        if reply != QMessageBox.Yes:
            return
            
        try:
            with self.db_service.get_connection() as conn:
                cursor = conn.cursor()
                
                # 删除相关数据
                cursor.execute("DELETE FROM documents WHERE session_id IN (SELECT session_id FROM text_sessions WHERE user_id = ?)", (user_id,))
                cursor.execute("DELETE FROM text_sessions WHERE user_id = ?", (user_id,))
                cursor.execute("DELETE FROM ocr_results WHERE session_id IN (SELECT session_id FROM ocr_sessions WHERE user_id = ?)", (user_id,))
                cursor.execute("DELETE FROM ocr_sessions WHERE user_id = ?", (user_id,))
                cursor.execute("DELETE FROM roi_data WHERE session_id IN (SELECT session_id FROM dicom_sessions WHERE user_id = ?)", (user_id,))
                cursor.execute("DELETE FROM dicom_sessions WHERE user_id = ?", (user_id,))
                
                # 删除样本
                cursor.execute("DELETE FROM user_ids WHERE user_id = ?", (user_id,))
                
                conn.commit()
            
            # 重新加载数据
            self.load_samples()
            
            # 清空详情
            self.clear_sample_detail()
            self.current_sample = None
            self.delete_sample_button.setEnabled(False)
            
            # 发送更新信号
            self.sample_updated.emit()
            
            QMessageBox.information(self, "成功", "样本删除成功！")
            
        except Exception as e:
            self.logger.error(f"删除样本失败: {e}")
            QMessageBox.warning(self, "错误", f"删除样本失败: {e}")
            
    def set_project_id(self, project_id: str):
        """
        设置项目ID
        
        Args:
            project_id: 项目ID
        """
        self.project_id = project_id
        self.project_id_label.setText(project_id)
        self.setWindowTitle(f"样本管理 - 项目: {project_id}")
        self.load_samples()
        
    def apply_styles(self):
        """
        应用样式
        """
        self.setStyleSheet("""
            QDialog {
                background-color: #F5F5F5;
            }
            QGroupBox {
                font-weight: bold;
                border: 2px solid #CCCCCC;
                border-radius: 5px;
                margin-top: 10px;
                padding-top: 10px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px 0 5px;
            }
            QPushButton {
                background-color: #2E86AB;
                color: white;
                border: none;
                border-radius: 4px;
                padding: 5px 10px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #1E5F7A;
            }
            QPushButton:pressed {
                background-color: #0D3B52;
            }
            QPushButton:disabled {
                background-color: #CCCCCC;
                color: #666666;
            }
            QLineEdit, QTextEdit {
                border: 1px solid #CCCCCC;
                border-radius: 4px;
                padding: 5px;
                background-color: #FFFFFF;
            }
            QLineEdit:focus, QTextEdit:focus {
                border-color: #2E86AB;
            }
            QTableWidget {
                border: 1px solid #CCCCCC;
                border-radius: 4px;
                background-color: #FFFFFF;
                gridline-color: #E0E0E0;
            }
            QTableWidget::item {
                padding: 5px;
            }
            QTableWidget::item:selected {
                background-color: #E3F2FD;
                color: #1976D2;
            }
            QHeaderView::section {
                background-color: #F0F0F0;
                padding: 5px;
                border: 1px solid #CCCCCC;
                font-weight: bold;
            }
        """)
