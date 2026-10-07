from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, 
    QFileDialog, QTextEdit, QScrollArea, QGroupBox, QSpinBox,
    QCheckBox, QProgressBar, QListWidget, QSplitter, QComboBox,
    QTableWidget, QTableWidgetItem, QTabWidget, QFrame, QMessageBox
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QUrl
from PyQt5.QtGui import QPixmap, QFont, QDragEnterEvent, QDropEvent
import glob
import os
import sys

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.services.text_service import TextService
from src.services.user_service import UserService
from src.utils.logger import get_logger
from src.core.user_id_manager import UserIDManager
from src.core.event_bus import get_event_bus, EventNames
import time
import json

class TextProcessingThread(QThread):
    """文本处理线程"""
    progress_updated = pyqtSignal(int)
    processing_finished = pyqtSignal(dict)
    error_occurred = pyqtSignal(str)
    
    def __init__(self, file_path, user_id, analysis_options=None, project_id=None):
        super().__init__()
        self.file_path = file_path
        self.user_id = user_id
        self.analysis_options = analysis_options or {}
        self.project_id = project_id
        self.text_service = TextService()
        
    def run(self):
        try:
            self.progress_updated.emit(20)
            
            # 验证文件路径
            if not self.file_path or not os.path.exists(self.file_path):
                self.error_occurred.emit(f"文件不存在或路径无效: {self.file_path}")
                return
            
            # 检查文件大小（避免内存不足）
            try:
                file_size = os.path.getsize(self.file_path)
                if file_size > 100 * 1024 * 1024:  # 100MB限制
                    self.error_occurred.emit(f"文件过大 ({file_size / 1024 / 1024:.1f}MB)，请选择较小的文件")
                    return
            except OSError as e:
                self.error_occurred.emit(f"无法访问文件: {str(e)}")
                return
            
            # 读取文件内容
            try:
                read_res = self.text_service.read_file_structured(
                    self.file_path,
                    project_id=self.project_id,
                    user_id=self.user_id,
                    persist_converted_xlsx=True,
                )
                content = read_res.get("text", "")
                tabular_payload = read_res.get("tabular")
            except FileNotFoundError as e:
                self.error_occurred.emit(f"文件不存在: {str(e)}")
                return
            except PermissionError as e:
                self.error_occurred.emit(f"文件权限错误: {str(e)}")
                return
            except ValueError as e:
                self.error_occurred.emit(f"文件内容错误: {str(e)}")
                return
            except RuntimeError as e:
                self.error_occurred.emit(f"文件读取失败: {str(e)}")
                return
            except MemoryError:
                self.error_occurred.emit("内存不足，无法处理此文件")
                return
            except Exception as e:
                self.error_occurred.emit(f"读取文件时发生未知错误: {str(e)}")
                return
            
            # 验证内容
            if not content or not content.strip():
                self.error_occurred.emit("文件内容为空或无法提取有效文本")
                return
                
            # 文本清理
            try:
                cleaning_options = self.analysis_options.get('cleaning_options', {})
                if any(cleaning_options.values()):
                    content = self.text_service.clean_text(content, cleaning_options)
            except Exception as e:
                self.error_occurred.emit(f"文本清理失败: {str(e)}")
                return
                
            self.progress_updated.emit(40)
            
            # 执行文本分析
            try:
                analysis_result = self.text_service.analyze_text(content)
            except Exception as e:
                self.error_occurred.emit(f"文本分析失败: {str(e)}")
                return
            
            self.progress_updated.emit(60)
            
            # 提取关键词
            try:
                keyword_count = self.analysis_options.get('keyword_count', 10)
                keyword_method = self.analysis_options.get('keyword_method', 'frequency')
                keywords = self.text_service.extract_keywords(content, keyword_count, keyword_method)
            except Exception as e:
                self.error_occurred.emit(f"关键词提取失败: {str(e)}")
                return
            
            self.progress_updated.emit(80)
            
            # 创建文本处理会话
            try:
                # 获取当前项目ID
                current_project_id = self.project_id
                if not current_project_id:
                    from src.services.project_service import ProjectService
                    project_service = ProjectService()
                    current_project_id = project_service.get_current_project_id() or "default"
                
                session_id = self.text_service.create_text_session(
                    user_id=self.user_id,
                    file_path=self.file_path,
                    content=content,
                    analysis_result=analysis_result,
                    keywords=keywords,
                    project_id=current_project_id,
                    tabular_payload=tabular_payload
                )
            except Exception as e:
                self.error_occurred.emit(f"创建会话失败: {str(e)}")
                return
            
            self.progress_updated.emit(100)
            
            result = {
                'session_id': session_id,
                'content': content,
                'analysis_result': analysis_result,
                'keywords': keywords
            }
            
            self.processing_finished.emit(result)
            
        except MemoryError:
            self.error_occurred.emit("内存不足，请尝试处理较小的文件或重启应用程序")
        except Exception as e:
            import traceback
            error_details = traceback.format_exc()
            self.error_occurred.emit(f"文本处理过程中发生未知错误: {str(e)}\n详细信息: {error_details}")

class TextProcessingWidget(QWidget):
    """文本处理界面组件"""
    
    def __init__(self, config, parent=None):
        super().__init__(parent)
        self.config = config
        self.logger = get_logger(__name__)
        self.text_service = TextService()
        self.user_service = UserService()
        self.user_id_manager = UserIDManager()
        self.event_bus = get_event_bus()
        self.current_user_id = None
        self.selected_user_id = None
        self.current_project_id = None # Initialize current_project_id
        self.current_session_id = None
        self.processing_thread = None
        self.current_content = ""
        self.selected_files = []  # 支持多文件选择
        self.batch_processing = False  # 批量处理标志
        self.setAcceptDrops(True)  # 启用拖拽功能
        
        self.init_ui()
        self.setup_event_connections()
        self.update_ui_state()
        
    def init_ui(self):
        """初始化用户界面"""
        layout = QVBoxLayout()
        
        # 标题
        title_label = QLabel("文本处理")
        title_label.setFont(QFont("Arial", 16, QFont.Bold))
        title_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(title_label)
        
        # 创建分割器
        splitter = QSplitter(Qt.Horizontal)
        
        # 左侧控制面板
        control_panel = self.create_control_panel()
        splitter.addWidget(control_panel)
        
        # 右侧结果显示面板
        result_panel = self.create_result_panel()
        splitter.addWidget(result_panel)
        
        # 设置分割器比例
        splitter.setSizes([300, 500])
        
        layout.addWidget(splitter)
        
        # 状态栏
        self.status_label = QLabel("就绪")
        layout.addWidget(self.status_label)
        
        # 进度条
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        layout.addWidget(self.progress_bar)
        
        self.setLayout(layout)
        
    def create_control_panel(self):
        """创建控制面板"""
        panel = QWidget()
        layout = QVBoxLayout()
        
        # 文件选择组
        file_group = QGroupBox("文件选择")
        file_layout = QVBoxLayout()
        
        # 拖拽区域
        self.drop_area = QFrame()
        self.drop_area.setFrameStyle(QFrame.StyledPanel)
        self.drop_area.setStyleSheet("""
            QFrame {
                border: 2px dashed #aaa;
                border-radius: 10px;
                background-color: #f9f9f9;
                color: #666;
                min-height: 80px;
            }
            QFrame:hover {
                border-color: #007acc;
                background-color: #e6f3ff;
            }
        """)
        
        drop_layout = QVBoxLayout()
        drop_label = QLabel("拖拽文本文件或文件夹到此处\n支持 .txt, .docx, .pdf 等格式")
        drop_label.setAlignment(Qt.AlignCenter)
        drop_label.setWordWrap(True)
        drop_layout.addWidget(drop_label)
        self.drop_area.setLayout(drop_layout)
        file_layout.addWidget(self.drop_area)
        
        # 文件信息显示
        self.file_info_label = QLabel("未选择文件")
        self.file_info_label.setWordWrap(True)
        file_layout.addWidget(self.file_info_label)
        
        # 按钮布局
        button_layout = QHBoxLayout()
        
        self.select_file_btn = QPushButton("选择单个文件")
        self.select_file_btn.clicked.connect(self.select_text_file)
        button_layout.addWidget(self.select_file_btn)
        
        self.select_folder_btn = QPushButton("选择文件夹")
        self.select_folder_btn.clicked.connect(self.select_text_folder)
        button_layout.addWidget(self.select_folder_btn)
        
        self.clear_selection_btn = QPushButton("清空选择")
        self.clear_selection_btn.clicked.connect(self.clear_file_selection)
        self.clear_selection_btn.setEnabled(False)
        button_layout.addWidget(self.clear_selection_btn)
        
        file_layout.addLayout(button_layout)
        
        file_group.setLayout(file_layout)
        layout.addWidget(file_group)
        
        # 分析选项组
        analysis_group = QGroupBox("分析选项")
        analysis_layout = QVBoxLayout()
        
        # 分析类型
        self.basic_analysis_checkbox = QCheckBox("基础统计分析")
        self.basic_analysis_checkbox.setChecked(True)
        analysis_layout.addWidget(self.basic_analysis_checkbox)
        
        self.keyword_analysis_checkbox = QCheckBox("关键词提取")
        self.keyword_analysis_checkbox.setChecked(True)
        analysis_layout.addWidget(self.keyword_analysis_checkbox)
        
        self.language_analysis_checkbox = QCheckBox("语言分析")
        self.language_analysis_checkbox.setChecked(True)
        analysis_layout.addWidget(self.language_analysis_checkbox)
        
        # 关键词提取方法
        keyword_method_layout = QHBoxLayout()
        keyword_method_layout.addWidget(QLabel("提取方法:"))
        self.keyword_method_combo = QComboBox()
        self.keyword_method_combo.addItems(["词频统计", "TF-IDF", "TextRank"])
        self.keyword_method_combo.setCurrentText("词频统计")
        keyword_method_layout.addWidget(self.keyword_method_combo)
        analysis_layout.addLayout(keyword_method_layout)
        
        # 关键词数量
        keyword_layout = QHBoxLayout()
        keyword_layout.addWidget(QLabel("关键词数量:"))
        self.keyword_count_spinbox = QSpinBox()
        self.keyword_count_spinbox.setRange(5, 50)
        self.keyword_count_spinbox.setValue(10)
        keyword_layout.addWidget(self.keyword_count_spinbox)
        analysis_layout.addLayout(keyword_layout)
        
        # 自动保存选项
        self.auto_save_checkbox = QCheckBox("自动保存结果")
        self.auto_save_checkbox.setChecked(True)
        analysis_layout.addWidget(self.auto_save_checkbox)
        
        # 文本清理选项
        cleaning_group = QGroupBox("文本清理选项")
        cleaning_layout = QVBoxLayout(cleaning_group)
        
        self.remove_extra_whitespace_check = QCheckBox("移除多余空白字符")
        self.remove_extra_whitespace_check.setChecked(True)
        cleaning_layout.addWidget(self.remove_extra_whitespace_check)
        
        self.normalize_line_breaks_check = QCheckBox("统一换行符")
        self.normalize_line_breaks_check.setChecked(True)
        cleaning_layout.addWidget(self.normalize_line_breaks_check)
        
        self.remove_empty_lines_check = QCheckBox("移除空行")
        cleaning_layout.addWidget(self.remove_empty_lines_check)
        
        self.remove_special_chars_check = QCheckBox("移除特殊字符")
        cleaning_layout.addWidget(self.remove_special_chars_check)
        
        self.to_lowercase_check = QCheckBox("转换为小写")
        cleaning_layout.addWidget(self.to_lowercase_check)
        
        self.remove_punctuation_check = QCheckBox("移除标点符号")
        cleaning_layout.addWidget(self.remove_punctuation_check)
        
        layout.addWidget(cleaning_group)
        
        analysis_group.setLayout(analysis_layout)
        layout.addWidget(analysis_group)
        
        # 处理按钮
        self.process_btn = QPushButton("开始文本分析")
        self.process_btn.clicked.connect(self.start_text_processing)
        self.process_btn.setEnabled(False)
        layout.addWidget(self.process_btn)
        
        # 导出选项组
        export_group = QGroupBox("导出选项")
        export_layout = QVBoxLayout()
        
        # 导出格式
        format_layout = QHBoxLayout()
        format_layout.addWidget(QLabel("导出格式:"))
        self.export_format_combo = QComboBox()
        self.export_format_combo.addItems(["TXT", "DOCX", "XLSX"])
        format_layout.addWidget(self.export_format_combo)
        export_layout.addLayout(format_layout)
        
        # 导出按钮
        self.export_btn = QPushButton("导出结果")
        self.export_btn.clicked.connect(self.export_results)
        self.export_btn.setEnabled(False)
        export_layout.addWidget(self.export_btn)
        
        export_group.setLayout(export_layout)
        layout.addWidget(export_group)
        
        # 历史会话列表
        history_group = QGroupBox("历史会话")
        history_layout = QVBoxLayout()
        
        self.session_list = QListWidget()
        self.session_list.itemClicked.connect(self.load_session)
        history_layout.addWidget(self.session_list)
        
        refresh_btn = QPushButton("刷新列表")
        refresh_btn.clicked.connect(self.refresh_session_list)
        history_layout.addWidget(refresh_btn)
        
        # 清除历史会话按钮
        clear_sessions_btn = QPushButton("清除历史会话")
        clear_sessions_btn.clicked.connect(self.clear_all_sessions)
        clear_sessions_btn.setStyleSheet("""
            QPushButton {
                background-color: #ff6b6b;
                color: white;
                border: none;
                padding: 5px;
                border-radius: 3px;
            }
            QPushButton:hover {
                background-color: #ff5252;
            }
        """)
        history_layout.addWidget(clear_sessions_btn)
        
        history_group.setLayout(history_layout)
        layout.addWidget(history_group)
        
        layout.addStretch()
        panel.setLayout(layout)
        return panel
        
    def create_result_panel(self):
        """创建结果显示面板"""
        panel = QWidget()
        layout = QVBoxLayout()
        
        # 创建标签页
        self.tab_widget = QTabWidget()
        
        # 文本内容标签页
        content_tab = self.create_content_tab()
        self.tab_widget.addTab(content_tab, "文本内容")
        
        # 分析结果标签页
        analysis_tab = self.create_analysis_tab()
        self.tab_widget.addTab(analysis_tab, "分析结果")
        
        # 关键词标签页
        keywords_tab = self.create_keywords_tab()
        self.tab_widget.addTab(keywords_tab, "关键词")
        
        layout.addWidget(self.tab_widget)
        
        panel.setLayout(layout)
        return panel
        
    def create_content_tab(self):
        """创建文本内容标签页"""
        tab = QWidget()
        layout = QVBoxLayout()
        
        # 文本内容显示
        self.content_text = QTextEdit()
        self.content_text.setPlaceholderText("文本内容将显示在这里...")
        self.content_text.setReadOnly(True)
        layout.addWidget(self.content_text)
        
        # 操作按钮
        button_layout = QHBoxLayout()
        
        self.copy_content_btn = QPushButton("复制内容")
        self.copy_content_btn.clicked.connect(self.copy_content_to_clipboard)
        self.copy_content_btn.setEnabled(False)
        button_layout.addWidget(self.copy_content_btn)
        
        self.save_content_btn = QPushButton("保存内容")
        self.save_content_btn.clicked.connect(self.save_content_to_file)
        self.save_content_btn.setEnabled(False)
        button_layout.addWidget(self.save_content_btn)
        
        button_layout.addStretch()
        layout.addLayout(button_layout)
        
        tab.setLayout(layout)
        return tab
        
    def create_analysis_tab(self):
        """创建分析结果标签页"""
        tab = QWidget()
        layout = QVBoxLayout()
        
        # 分析结果表格
        self.analysis_table = QTableWidget()
        self.analysis_table.setColumnCount(2)
        self.analysis_table.setHorizontalHeaderLabels(["统计项", "数值"])
        self.analysis_table.horizontalHeader().setStretchLastSection(True)
        layout.addWidget(self.analysis_table)
        
        tab.setLayout(layout)
        return tab
        
    def create_keywords_tab(self):
        """创建关键词标签页"""
        tab = QWidget()
        layout = QVBoxLayout()
        
        # 关键词列表
        self.keywords_text = QTextEdit()
        self.keywords_text.setPlaceholderText("关键词将显示在这里...")
        self.keywords_text.setReadOnly(True)
        layout.addWidget(self.keywords_text)
        
        # 操作按钮
        button_layout = QHBoxLayout()
        
        self.copy_keywords_btn = QPushButton("复制关键词")
        self.copy_keywords_btn.clicked.connect(self.copy_keywords_to_clipboard)
        self.copy_keywords_btn.setEnabled(False)
        button_layout.addWidget(self.copy_keywords_btn)
        
        button_layout.addStretch()
        layout.addLayout(button_layout)
        
        tab.setLayout(layout)
        return tab
        
    def set_current_user(self, user_id):
        """设置当前用户"""
        self.current_user_id = user_id
        self.refresh_session_list()
        
    def select_text_file(self):
        """选择文本文件"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "选择文本文件", "", 
            "文本文件 (*.txt *.docx *.pdf *.doc *.rtf *.md *.xlsx *.xls *.csv);;所有文件 (*)"
        )
        
        if file_path:
            self.selected_file_path = file_path
            self.selected_files = [file_path]
            self.batch_processing = False
            self._update_file_info_display()
            self.process_btn.setEnabled(True)
            self.clear_selection_btn.setEnabled(True)
            
    def get_analysis_options(self):
        """获取分析选项"""
        # 映射UI显示文本到方法名
        method_mapping = {
            "词频统计": "frequency",
            "TF-IDF": "tfidf",
            "TextRank": "textrank"
        }
        
        # 获取文本清理选项
        cleaning_options = {
            'remove_extra_whitespace': self.remove_extra_whitespace_check.isChecked(),
            'normalize_line_breaks': self.normalize_line_breaks_check.isChecked(),
            'remove_empty_lines': self.remove_empty_lines_check.isChecked(),
            'remove_special_chars': self.remove_special_chars_check.isChecked(),
            'to_lowercase': self.to_lowercase_check.isChecked(),
            'remove_punctuation': self.remove_punctuation_check.isChecked()
        }
        
        options = {
            'basic_analysis': self.basic_analysis_checkbox.isChecked(),
            'keyword_analysis': self.keyword_analysis_checkbox.isChecked(),
            'language_analysis': self.language_analysis_checkbox.isChecked(),
            'keyword_count': self.keyword_count_spinbox.value(),
            'keyword_method': method_mapping.get(self.keyword_method_combo.currentText(), "frequency"),
            'cleaning_options': cleaning_options
        }
        return options
        
    def start_text_processing(self):
        """开始文本处理"""
        # 检查是否选择了文件
        if not self.selected_files:
            self.status_label.setText("请先选择文件")
            return
            
        # 检查是否选择了项目
        current_project_id = self.current_project_id
        if not current_project_id:
            from src.services.project_service import ProjectService
            project_service = ProjectService()
            current_project_id = project_service.get_current_project_id()
            
        if not current_project_id:
            self.status_label.setText("请先在主界面选择一个项目")
            return
            
        # 检查是否选择了用户ID
        if not self.selected_user_id:
            self.status_label.setText("请先在主界面选择一个样本ID")
            return
            
        # 禁用处理按钮
        self.process_btn.setEnabled(False)
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)
        
        # 获取分析选项
        analysis_options = self.get_analysis_options()
        
        if self.batch_processing and len(self.selected_files) > 1:
            # 批量处理模式
            self.status_label.setText(f"开始批量处理 {len(self.selected_files)} 个文件...")
            self._process_files_batch(analysis_options)
        else:
            # 单文件处理模式
            file_path = self.selected_files[0]
            
            # 创建并启动处理线程
            self.processing_thread = TextProcessingThread(
                file_path, 
                self.selected_user_id, 
                analysis_options,
                project_id=self.current_project_id
            )
            
            self.processing_thread.progress_updated.connect(self.update_progress)
            self.processing_thread.processing_finished.connect(self.on_processing_finished)
            self.processing_thread.error_occurred.connect(self.on_processing_error)
            
            self.processing_thread.start()
            self.status_label.setText("正在处理文本文件...")
        
    def update_progress(self, value):
        """更新进度条"""
        self.progress_bar.setValue(value)
        
    def on_processing_finished(self, result):
        """处理完成回调"""
        self.progress_bar.setVisible(False)
        self.process_btn.setEnabled(True)
        
        self.current_session_id = result['session_id']
        self.current_content = result['content']
        
        # 显示文本内容
        self.content_text.setText(result['content'])
        
        # 显示分析结果
        analysis_result = result['analysis_result']
        self.display_analysis_results(analysis_result)
        
        # 显示关键词
        keywords = result['keywords']
        self.display_keywords(keywords)
        
        # 启用操作按钮
        self.copy_content_btn.setEnabled(True)
        self.save_content_btn.setEnabled(True)
        self.copy_keywords_btn.setEnabled(True)
        self.export_btn.setEnabled(True)
        
        # 刷新会话列表
        self.refresh_session_list()
    
    def set_current_project(self, project_id):
        """设置当前项目并刷新数据"""
        try:
            self.current_project_id = project_id
            self.logger.info(f"文本处理模块切换到项目: {project_id}")
            
            # 刷新会话列表
            self.refresh_session_list()
            
            # 清空当前显示内容
            self.content_text.clear()
            self.analysis_table.setRowCount(0)
            
            # 重置文件选择
            self.selected_files = []
            self.batch_processing = False
            if hasattr(self, 'selected_file_path'):
                delattr(self, 'selected_file_path')
            self.file_info_label.setText("未选择文件")
            self.clear_selection_btn.setEnabled(False)
            
            # 更新UI状态
            self.update_ui_state()
            
            # 更新状态标签
            if self.selected_user_id:
                self.status_label.setText(f"已切换到项目 {project_id}，当前样本ID: {self.selected_user_id}")
            else:
                self.status_label.setText(f"已切换到项目 {project_id}，请选择一个样本ID")
                
        except Exception as e:
            self.logger.error(f"文本处理模块设置当前项目失败: {e}")
            self.status_label.setText(f"切换项目失败: {str(e)}")

    
    def setup_event_connections(self):
        """设置事件连接"""
        try:
            # 订阅用户ID选择事件
            self.event_bus.subscribe(EventNames.USER_ID_SELECTED, self.on_user_id_selected)
            self.logger.info("文本处理组件已订阅用户ID选择事件")
        except Exception as e:
            self.logger.error(f"设置事件连接失败: {e}")
    
    def on_user_id_selected(self, data):
        """响应用户ID选择事件"""
        try:
            # 处理不同格式的事件数据
            if isinstance(data, dict):
                user_id = data.get('user_id')
                display_name = data.get('display_name', user_id)
            else:
                # 兼容旧格式（直接传递user_id字符串）
                user_id = data
                display_name = user_id
            
            self.selected_user_id = user_id
            self.logger.info(f"文本处理组件接收到用户ID选择: {user_id} ({display_name})")
            
            # 更新UI状态
            self.update_ui_state()
            
            # 刷新会话列表以显示该用户ID的历史记录
            self.refresh_session_list()
            
            # 发布数据同步事件
            self.event_bus.publish(EventNames.DATA_SYNC_REQUESTED, {
                'module': 'text_processing',
                'user_id': user_id,
                'timestamp': time.time()
            })
            
        except Exception as e:
            self.logger.error(f"处理用户ID选择事件失败: {e}")
    
    def update_ui_state(self):
        """更新UI状态"""
        try:
            # 根据是否选择了用户ID来启用/禁用相关功能
            has_user_id = bool(self.selected_user_id)
            
            # 更新状态标签
            if has_user_id:
                user_info = self.user_id_manager.get_user_id_info(self.selected_user_id)
                if user_info:
                    display_name = user_info.get('display_name', self.selected_user_id)
                    self.status_label.setText(f"当前样本: {display_name}")
                else:
                    self.status_label.setText(f"当前样本: {self.selected_user_id}")
            else:
                self.status_label.setText("请在主界面选择一个项目")
            
            # 启用/禁用处理按钮（需要同时有文件和用户ID）
            has_files = bool(self.selected_files)
            self.process_btn.setEnabled(has_files and has_user_id)
            
            self.logger.debug(f"UI状态更新: 用户ID={has_user_id}, 文件={has_files}")
            
        except Exception as e:
            self.logger.error(f"更新UI状态失败: {e}")
    
    def clear_all_sessions(self):
        """清除所有文本处理历史会话"""
        try:
            # 检查是否选择了用户ID
            if not self.selected_user_id:
                QMessageBox.warning(self, "警告", "请先在主界面选择一个项目")
                return
            
            # 确认对话框
            reply = QMessageBox.question(
                self, 
                "确认清除", 
                "确定要清除所有文本处理历史会话吗？\n此操作不可撤销！",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No
            )
            
            if reply == QMessageBox.Yes:
                # 获取当前项目ID
                current_project_id = self.current_project_id
                if not current_project_id:
                    from src.services.project_service import ProjectService
                    project_service = ProjectService()
                    current_project_id = project_service.get_current_project_id() or "default"
                
                # 使用实例的text_service而不是创建新实例
                self.text_service.clear_all_text_sessions(self.selected_user_id, project_id=current_project_id)
                
                # 刷新会话列表
                self.refresh_session_list()
                
                # 显示成功消息
                QMessageBox.information(self, "成功", "所有文本处理历史会话已清除")
                self.status_label.setText("已清除所有历史会话")
                
        except Exception as e:
            QMessageBox.critical(self, "错误", f"清除历史会话失败：{str(e)}")
            self.status_label.setText(f"清除失败：{str(e)}")
        
    def display_analysis_results(self, analysis_result):
        """显示分析结果"""
        self.analysis_table.setRowCount(len(analysis_result))
        
        row = 0
        for key, value in analysis_result.items():
            # 转换键名为中文
            key_mapping = {
                'char_count': '字符数',
                'word_count': '词数',
                'line_count': '行数',
                'paragraph_count': '段落数',
                'sentence_count': '句子数',
                'chinese_char_count': '中文字符数',
                'english_word_count': '英文词数',
                'number_count': '数字数量',
                'punctuation_count': '标点符号数量'
            }
            
            display_key = key_mapping.get(key, key)
            
            self.analysis_table.setItem(row, 0, QTableWidgetItem(display_key))
            self.analysis_table.setItem(row, 1, QTableWidgetItem(str(value)))
            row += 1
            
    def display_keywords(self, keywords):
        """显示关键词"""
        if isinstance(keywords, list):
            keywords_text = "\n".join([f"{i+1}. {keyword}" for i, keyword in enumerate(keywords)])
        else:
            keywords_text = str(keywords)
            
        self.keywords_text.setText(keywords_text)
        
    def on_processing_error(self, error_message):
        """处理错误回调"""
        self.progress_bar.setVisible(False)
        self.process_btn.setEnabled(True)
        
        # 显示用户友好的错误信息
        user_friendly_message = self._get_user_friendly_error_message(error_message)
        self.status_label.setText(f"文本处理失败: {user_friendly_message}")
        
        # 显示详细错误信息的对话框
        self._show_error_dialog("文本处理错误", error_message, user_friendly_message)
        
        # 记录错误日志
        self.logger.error(f"文本处理失败: {error_message}")
        
    def _get_user_friendly_error_message(self, error_message):
        """获取用户友好的错误信息"""
        error_lower = error_message.lower()
        
        if "file not found" in error_lower or "no such file" in error_lower:
            return "文本文件未找到，请检查文件路径是否正确"
        elif "permission denied" in error_lower:
            return "文件访问权限不足，请检查文件权限"
        elif "encoding" in error_lower or "decode" in error_lower:
            return "文件编码格式不支持，请尝试UTF-8编码的文件"
        elif "empty file" in error_lower or "no content" in error_lower:
            return "文件内容为空，请选择包含文本内容的文件"
        elif "memory" in error_lower or "out of memory" in error_lower:
            return "内存不足，请尝试处理较小的文件或关闭其他程序"
        elif "timeout" in error_lower:
            return "文本处理超时，请尝试处理较小的文件"
        elif "invalid format" in error_lower:
            return "不支持的文件格式，请选择TXT、DOCX或XLSX文件"
        elif "keyword extraction" in error_lower:
            return "关键词提取失败，请检查文本内容或尝试其他提取方法"
        else:
            return "文本处理过程中发生未知错误，请检查文件格式和内容"
            
    def _show_error_dialog(self, title, technical_error, user_friendly_error):
        """显示错误对话框"""
        from PyQt5.QtWidgets import QMessageBox
        
        msg_box = QMessageBox(self)
        msg_box.setIcon(QMessageBox.Critical)
        msg_box.setWindowTitle(title)
        msg_box.setText(user_friendly_error)
        
        # 添加详细信息
        msg_box.setDetailedText(f"技术详情:\n{technical_error}")
        
        # 添加建议解决方案
        suggestions = self._get_error_suggestions(technical_error)
        if suggestions:
            msg_box.setInformativeText(f"建议解决方案:\n{suggestions}")
        
        msg_box.setStandardButtons(QMessageBox.Ok | QMessageBox.Retry)
        msg_box.setDefaultButton(QMessageBox.Retry)
        
        result = msg_box.exec_()
        if result == QMessageBox.Retry:
            # 用户选择重试，重新启用处理按钮
            self.process_btn.setEnabled(True)
            
    def _get_error_suggestions(self, error_message):
        """获取错误解决建议"""
        error_lower = error_message.lower()
        
        if "file not found" in error_lower:
            return "• 检查文本文件路径是否正确\n• 确认文件是否存在\n• 尝试重新选择文件"
        elif "permission denied" in error_lower:
            return "• 以管理员身份运行程序\n• 检查文件是否被其他程序占用\n• 确认对文件夹有读写权限"
        elif "encoding" in error_lower:
            return "• 将文件另存为UTF-8编码格式\n• 尝试使用记事本打开并重新保存\n• 检查文件是否包含特殊字符"
        elif "empty file" in error_lower:
            return "• 确认文件包含文本内容\n• 检查文件大小是否为0\n• 尝试打开文件验证内容"
        elif "memory" in error_lower:
            return "• 关闭其他占用内存的程序\n• 尝试处理较小的文件\n• 分批处理大文件"
        elif "invalid format" in error_lower:
            return "• 确认文件格式为TXT、DOCX或XLSX\n• 检查文件扩展名是否正确\n• 尝试转换为支持的格式"
        elif "keyword extraction" in error_lower:
            return "• 检查文本内容是否包含有意义的词汇\n• 尝试不同的关键词提取方法\n• 确认文本长度足够进行分析"
        else:
            return "• 检查文件格式和内容\n• 尝试重新保存文件\n• 重新启动应用程序\n• 联系技术支持"
        
    def refresh_session_list(self):
        """刷新会话列表"""
        if not self.selected_user_id:
            self.session_list.clear()
            return
            
        try:
            # 获取当前项目ID - 优先使用组件内部状态
            current_project_id = self.current_project_id
            if not current_project_id:
                from src.services.project_service import ProjectService
                project_service = ProjectService()
                current_project_id = project_service.get_current_project_id() or "default"
            
            self.logger.info(f"正在刷新文本会话列表: user_id={self.selected_user_id}, project_id={current_project_id}")
            
            sessions = self.text_service.get_text_sessions(self.selected_user_id, project_id=current_project_id)
            self.logger.info(f"获取到 {len(sessions)} 个会话，用户ID: {self.selected_user_id}, 项目ID: {current_project_id}")
            
            self.session_list.clear()
            
            for session in sessions:
                # 格式化显示项
                created_at = session.get('created_at', '')[:19]
                file_name = session.get('file_name', '未知文件')
                session_id = session.get('session_id', '')
                
                # 显示文件名和时间，将session_id作为隐藏数据或放在后面
                item_text = f"{file_name} ({created_at}) [{session_id}]"
                self.session_list.addItem(item_text)
                self.logger.debug(f"添加会话项: {item_text}")
                
            self.logger.info(f"会话列表刷新完成，当前显示 {self.session_list.count()} 个项目")
                
        except Exception as e:
            self.logger.error(f"刷新会话列表失败: {e}")
            import traceback
            self.logger.debug(traceback.format_exc())
            
    def load_session(self, item):
        """加载选中的会话"""
        try:
            # 从item文本中提取session_id
            item_text = item.text()
            # 尝试从新格式提取 [session_id]
            import re
            match = re.search(r'\[(text_[^\]]+)\]', item_text)
            if match:
                session_id = match.group(1)
            else:
                # 尝试旧格式
                try:
                    session_id = item_text.split(' ')[1]
                except IndexError:
                    self.logger.error(f"无法从文本解析会话ID: {item_text}")
                    return
            
            self.logger.info(f"正在加载会话: {session_id}")
            
            # 获取会话详情
            session_details = self.text_service.get_text_session_details(session_id)
            if not session_details:
                self.status_label.setText("无法加载会话详情")
                self.logger.warning(f"未找到会话详情: {session_id}")
                return
                
            self.current_session_id = session_id
            self.current_content = session_details.get('content', '')
            
            # 显示文本内容
            if self.current_content:
                self.content_text.setText(self.current_content)
            else:
                self.content_text.setText("（内容为空）")
            
            # 显示分析结果
            analysis_result = session_details.get('analysis_result', {})
            if isinstance(analysis_result, str):
                try:
                    import json
                    analysis_result = json.loads(analysis_result)
                except Exception as e:
                    self.logger.error(f"解析分析结果JSON失败: {e}")
                    analysis_result = {}
            self.display_analysis_results(analysis_result)
            
            # 显示关键词
            keywords = session_details.get('keywords', [])
            if isinstance(keywords, str):
                try:
                    import json
                    keywords = json.loads(keywords)
                except Exception as e:
                    self.logger.error(f"解析关键词JSON失败: {e}")
                    keywords = []
            self.display_keywords(keywords)
            
            # 启用操作按钮
            self.copy_content_btn.setEnabled(True)
            self.save_content_btn.setEnabled(True)
            self.copy_keywords_btn.setEnabled(True)
            self.export_btn.setEnabled(True)
            
            self.status_label.setText(f"已加载会话: {session_details.get('file_name', session_id)}")
            
        except Exception as e:
            self.logger.error(f"加载会话失败: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
            self.status_label.setText(f"加载会话失败: {str(e)}")
            
    def copy_content_to_clipboard(self):
        """复制内容到剪贴板"""
        try:
            from PyQt5.QtWidgets import QApplication
            clipboard = QApplication.clipboard()
            clipboard.setText(self.content_text.toPlainText())
            self.status_label.setText("内容已复制到剪贴板")
        except Exception as e:
            self.logger.error(f"复制内容失败: {e}")
            self.status_label.setText(f"复制失败: {str(e)}")
            
    def copy_keywords_to_clipboard(self):
        """复制关键词到剪贴板"""
        try:
            from PyQt5.QtWidgets import QApplication
            clipboard = QApplication.clipboard()
            clipboard.setText(self.keywords_text.toPlainText())
            self.status_label.setText("关键词已复制到剪贴板")
        except Exception as e:
            self.logger.error(f"复制关键词失败: {e}")
            self.status_label.setText(f"复制失败: {str(e)}")
            
    def save_content_to_file(self):
        """保存内容到文件"""
        try:
            file_path, _ = QFileDialog.getSaveFileName(
                self, "保存文本内容", "", 
                "Text Files (*.txt);;All Files (*)"
            )
            
            if file_path:
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(self.content_text.toPlainText())
                self.status_label.setText(f"内容已保存到: {file_path}")
                
        except Exception as e:
            self.logger.error(f"保存内容失败: {e}")
            self.status_label.setText(f"保存失败: {str(e)}")
            
    def export_results(self):
        """导出结果"""
        if not self.current_session_id:
            self.status_label.setText("请先选择一个会话")
            return
            
        try:
            export_format = self.export_format_combo.currentText()
            
            # 选择保存路径
            if export_format == "TXT":
                file_filter = "Text Files (*.txt)"
                default_ext = ".txt"
            elif export_format == "DOCX":
                file_filter = "Word Documents (*.docx)"
                default_ext = ".docx"
            elif export_format == "XLSX":
                file_filter = "Excel Files (*.xlsx)"
                default_ext = ".xlsx"
            else:
                file_filter = "All Files (*)"
                default_ext = ""
                
            file_path, _ = QFileDialog.getSaveFileName(
                self, f"导出为{export_format}", "", 
                f"{file_filter};;All Files (*)"
            )
            
            if file_path:
                if not file_path.endswith(default_ext) and default_ext:
                    file_path += default_ext
                    
                # 调用text_service的导出方法
                success = self.text_service.export_content(
                    self.current_content,
                    file_path,
                    export_format.lower()
                )
                
                if success:
                    self.status_label.setText(f"结果已导出到: {file_path}")
                else:
                    self.status_label.setText("导出失败")
                
        except Exception as e:
            self.logger.error(f"导出结果失败: {e}")
            self.status_label.setText(f"导出失败: {str(e)}")
    
    def dragEnterEvent(self, event: QDragEnterEvent):
        """拖拽进入事件"""
        if event.mimeData().hasUrls():
            # 检查是否包含有效的文本文件
            valid_files = self._get_valid_text_files_from_urls(event.mimeData().urls())
            if valid_files:
                event.acceptProposedAction()
                # 更新拖拽区域样式
                self.drop_area.setStyleSheet("""
                    QFrame {
                        border: 2px dashed #007acc;
                        border-radius: 10px;
                        background-color: #e6f3ff;
                        color: #007acc;
                    }
                """)
            else:
                event.ignore()
        else:
            event.ignore()
    
    def dragLeaveEvent(self, event):
        """拖拽离开事件"""
        # 恢复拖拽区域样式
        self.drop_area.setStyleSheet("""
            QFrame {
                border: 2px dashed #aaa;
                border-radius: 10px;
                background-color: #f9f9f9;
                color: #666;
                min-height: 80px;
            }
            QFrame:hover {
                border-color: #007acc;
                background-color: #e6f3ff;
            }
        """)
    
    def dropEvent(self, event: QDropEvent):
        """拖拽放下事件"""
        if event.mimeData().hasUrls():
            valid_files = self._get_valid_text_files_from_urls(event.mimeData().urls())
            if valid_files:
                self.selected_files = valid_files
                self._update_file_info_display()
                event.acceptProposedAction()
                
                # 恢复拖拽区域样式
                self.dragLeaveEvent(None)
                
                # 如果只有一个文件，直接设置为当前文件
                if len(valid_files) == 1:
                    self.selected_file_path = valid_files[0]
                    self.batch_processing = False
                else:
                    self.batch_processing = True
                    
                self.process_btn.setEnabled(True)
                self.clear_selection_btn.setEnabled(True)
            else:
                event.ignore()
        else:
            event.ignore()
    
    def _get_valid_text_files_from_urls(self, urls):
        """从URL列表中获取有效的文本文件"""
        valid_files = []
        text_extensions = ['.txt', '.docx', '.pdf', '.doc', '.rtf', '.md', '.xlsx', '.xls', '.csv']
        
        for url in urls:
            file_path = url.toLocalFile()
            if os.path.isfile(file_path):
                # 检查文件扩展名
                if any(file_path.lower().endswith(ext) for ext in text_extensions):
                    valid_files.append(file_path)
            elif os.path.isdir(file_path):
                # 搜索文件夹中的文本文件
                for ext in text_extensions:
                    pattern = os.path.join(file_path, f"**/*{ext}")
                    valid_files.extend(glob.glob(pattern, recursive=True))
        
        return valid_files
    
    def select_text_folder(self):
        """选择文本文件夹"""
        folder_path = QFileDialog.getExistingDirectory(self, "选择文本文件夹")
        
        if folder_path:
            # 搜索文件夹中的文本文件
            text_extensions = ['.txt', '.docx', '.pdf', '.doc', '.rtf', '.md', '.xlsx', '.xls', '.csv']
            valid_files = []
            
            for ext in text_extensions:
                pattern = os.path.join(folder_path, f"**/*{ext}")
                valid_files.extend(glob.glob(pattern, recursive=True))
            
            if valid_files:
                self.selected_files = valid_files
                self.batch_processing = True
                self._update_file_info_display()
                self.process_btn.setEnabled(True)
                self.clear_selection_btn.setEnabled(True)
            else:
                self.status_label.setText("所选文件夹中未找到支持的文本文件")
    
    def clear_file_selection(self):
        """清空文件选择"""
        self.selected_files = []
        self.batch_processing = False
        if hasattr(self, 'selected_file_path'):
            delattr(self, 'selected_file_path')
        
        self.file_info_label.setText("未选择文件")
        self.process_btn.setEnabled(False)
        self.clear_selection_btn.setEnabled(False)
        self.status_label.setText("已清空文件选择")
    
    def _update_file_info_display(self):
        """更新文件信息显示"""
        if not self.selected_files:
            self.file_info_label.setText("未选择文件")
            return
        
        if len(self.selected_files) == 1:
            file_name = os.path.basename(self.selected_files[0])
            self.file_info_label.setText(f"已选择: {file_name}")
            self.status_label.setText(f"已选择文件: {file_name}")
        else:
            self.file_info_label.setText(f"已选择 {len(self.selected_files)} 个文本文件")
            self.status_label.setText(f"已选择 {len(self.selected_files)} 个文件，准备批量处理")
    
    def _process_files_batch(self, analysis_options):
        """批量处理文件"""
        self.batch_results = []
        self.current_batch_index = 0
        self.total_batch_files = len(self.selected_files)
        self.batch_analysis_options = analysis_options
        
        # 开始处理第一个文件
        self._process_next_batch_file()
    
    def _process_next_batch_file(self):
        """处理下一个批量文件"""
        if self.current_batch_index >= self.total_batch_files:
            # 所有文件处理完成
            self._on_batch_processing_finished()
            return
        
        current_file = self.selected_files[self.current_batch_index]
        file_name = os.path.basename(current_file)
        
        self._update_batch_progress()
        self.status_label.setText(f"正在处理文件 {self.current_batch_index + 1}/{self.total_batch_files}: {file_name}")
        
        # 创建并启动处理线程
        self.processing_thread = TextProcessingThread(
            current_file,
            self.selected_user_id,
            self.batch_analysis_options,
            project_id=self.current_project_id
        )
        
        self.processing_thread.progress_updated.connect(self.update_progress)
        self.processing_thread.processing_finished.connect(self._on_batch_file_finished)
        self.processing_thread.error_occurred.connect(self._on_batch_file_error)
        
        self.processing_thread.start()
    
    def _update_batch_progress(self):
        """更新批量处理进度"""
        if self.total_batch_files > 0:
            progress = int((self.current_batch_index / self.total_batch_files) * 100)
            self.progress_bar.setValue(progress)
    
    def _on_batch_file_finished(self, result):
        """单个批量文件处理完成"""
        current_file = self.selected_files[self.current_batch_index]
        file_name = os.path.basename(current_file)
        
        # 保存结果
        batch_result = {
            'file_path': current_file,
            'file_name': file_name,
            'result': result,
            'status': 'success'
        }
        self.batch_results.append(batch_result)
        
        # 处理下一个文件
        self.current_batch_index += 1
        self._process_next_batch_file()
    
    def _on_batch_file_error(self, error_msg):
        """单个批量文件处理错误"""
        current_file = self.selected_files[self.current_batch_index]
        file_name = os.path.basename(current_file)
        
        # 保存错误结果
        batch_result = {
            'file_path': current_file,
            'file_name': file_name,
            'error': error_msg,
            'status': 'error'
        }
        self.batch_results.append(batch_result)
        
        # 处理下一个文件
        self.current_batch_index += 1
        self._process_next_batch_file()
    
    def _on_batch_processing_finished(self):
        """批量处理完成"""
        # 统计结果
        success_count = sum(1 for r in self.batch_results if r['status'] == 'success')
        error_count = len(self.batch_results) - success_count
        
        # 更新进度条
        self.progress_bar.setValue(100)
        
        # 显示批量处理结果摘要
        summary_text = f"批量处理完成！\n成功: {success_count} 个文件\n失败: {error_count} 个文件\n\n"
        
        # 添加详细结果
        for result in self.batch_results:
            if result['status'] == 'success':
                summary_text += f"✓ {result['file_name']}\n"
            else:
                summary_text += f"✗ {result['file_name']}: {result.get('error', '未知错误')}\n"
        
        # 显示结果
        self.content_text.setPlainText(summary_text)
        # 清空分析结果表格，因为这是批量处理摘要
        self.analysis_table.setRowCount(0)
        
        # 重新启用处理按钮
        self.process_btn.setEnabled(True)
        self.status_label.setText(f"批量处理完成: {success_count} 成功, {error_count} 失败")
        
        # 刷新会话列表
        self.refresh_session_list()
