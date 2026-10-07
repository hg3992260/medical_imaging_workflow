from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, 
    QFileDialog, QTextEdit, QScrollArea, QGroupBox, QSpinBox,
    QCheckBox, QProgressBar, QListWidget, QSplitter, QComboBox,
    QSlider, QListWidgetItem, QTabWidget, QGridLayout, QFrame,
    QDoubleSpinBox, QMessageBox
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QTimer, QUrl
from PyQt5.QtGui import QPixmap, QFont, QDragEnterEvent, QDropEvent
import glob
import os
import sys

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.services.ocr_service import get_ocr_service
from src.services.user_service import UserService
from src.utils.logger import get_logger
from src.core.user_id_manager import UserIDManager
from src.core.event_bus import get_event_bus, EventNames
import time
import json

class OCRProcessingThread(QThread):
    """OCR处理线程"""
    progress_updated = pyqtSignal(int)
    processing_finished = pyqtSignal(dict)
    processing_error = pyqtSignal(str)
    preview_updated = pyqtSignal(object)  # 新增信号用于更新预览
    status_updated = pyqtSignal(str)
    
    def __init__(self, file_paths, ocr_config, preprocess_config, batch_mode=False):
        super().__init__()
        self.file_paths = file_paths
        self.ocr_config = ocr_config
        self.preprocess_config = preprocess_config
        self.batch_mode = batch_mode
        self.ocr_service = get_ocr_service()

    def _emit_status(self, message: str):
        try:
            self.status_updated.emit(str(message or ""))
        except Exception:
            pass
        
    def run(self):
        try:
            if self.batch_mode:
                self._process_batch()
            else:
                self._process_single_file()
        except Exception as e:
            self.processing_error.emit(f"OCR处理过程中发生错误: {str(e)}")
            
    def _process_single_file(self):
        """处理单个文件"""
        if not self.file_paths:
            self.processing_error.emit("未选择文件")
            return
            
        file_path = self.file_paths[0]
        
        self.progress_updated.emit(10)
        
        # 合并配置
        combined_config = {**self.ocr_config, **self.preprocess_config}
        combined_config["status_callback"] = self._emit_status
        
        # 映射引擎名称
        engine_mapping = {
            'DeepSeek-OCR (推荐)': 'deepseek',
            'DeepSeek-OCR vLLM (推荐)': 'deepseek',
        }
        
        if 'ocr_engine' in combined_config:
            combined_config['engine'] = engine_mapping.get(
                combined_config['ocr_engine'], 'deepseek'
            )
        
        # 映射语言设置
        language_mapping = {
            '中英文混合': 'chi_sim+eng',
            '仅中文': 'chi_sim',
            '仅英文': 'eng'
        }
        
        if 'language' in combined_config:
            combined_config['language'] = language_mapping.get(
                combined_config['language'], 'chi_sim+eng'
            )
        
        # 设置置信度阈值
        if 'confidence_threshold' in combined_config:
            combined_config['min_confidence'] = combined_config['confidence_threshold']
        
        self.progress_updated.emit(30)
        
        # 执行OCR识别（带预览更新）
        ocr_result = self._extract_text_with_preview(file_path, combined_config)
        
        self.progress_updated.emit(90)
        
        if not ocr_result:
            self.processing_error.emit("OCR识别失败，未提取到文本")
            return

        try:
            if isinstance(ocr_result, dict):
                status = (ocr_result.get("status") or "").strip().lower()
                extracted_text = (ocr_result.get("extracted_text") or ocr_result.get("text") or "").strip()
                if status in ("loading", "unavailable", "error"):
                    err = (ocr_result.get("error") or f"OCR引擎状态异常: {status}").strip()
                    self.processing_error.emit(err)
                    return
                if not extracted_text and (ocr_result.get("error") or "").strip():
                    self.processing_error.emit(str(ocr_result.get("error")).strip())
                    return
        except Exception:
            pass
        
        result = {
            'ocr_result': ocr_result,
            'file_path': file_path
        }
        
        self.progress_updated.emit(100)
        self.processing_finished.emit(result)
        
    def _process_batch(self):
        """批量处理文件"""
        total_files = len(self.file_paths)
        results = []
        
        for i, file_path in enumerate(self.file_paths):
            try:
                progress = int((i / total_files) * 100)
                self.progress_updated.emit(progress)
                
                # 合并配置
                combined_config = {**self.ocr_config, **self.preprocess_config}
                combined_config["status_callback"] = self._emit_status
                
                # 执行OCR识别
                ocr_result = self._extract_text_with_preview(file_path, combined_config)
                
                if ocr_result:
                    try:
                        if isinstance(ocr_result, dict):
                            status = (ocr_result.get("status") or "").strip().lower()
                            extracted_text = (ocr_result.get("extracted_text") or ocr_result.get("text") or "").strip()
                            if status in ("loading", "unavailable", "error"):
                                raise RuntimeError((ocr_result.get("error") or f"OCR引擎状态异常: {status}").strip())
                            if not extracted_text and (ocr_result.get("error") or "").strip():
                                raise RuntimeError(str(ocr_result.get("error")).strip())
                    except Exception as e:
                        self.processing_error.emit(f"处理文件 {file_path} 时出错: {str(e)}")
                        continue
                    results.append({
                        'file_path': file_path,
                        'ocr_result': ocr_result
                    })
                    
            except Exception as e:
                self.processing_error.emit(f"处理文件 {file_path} 时出错: {str(e)}")
                
        self.progress_updated.emit(100)
        self.processing_finished.emit({'batch_results': results})
        
    def _extract_text_with_preview(self, file_path, config):
        """执行OCR识别并更新预览"""
        try:
            # 使用自定义的预处理方法来获取预处理后的图像
            processed_image = self._preprocess_image_with_preview(file_path, config)
            
            # 执行OCR识别
            if file_path.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
                return self.ocr_service.extract_text_from_image(file_path, config)
            else:
                return self.ocr_service.extract_text_from_file(file_path)
                
        except Exception as e:
            raise e
            
    def _preprocess_image_with_preview(self, file_path, config):
        """预处理图像并更新预览"""
        try:
            import cv2
            import numpy as np
            
            # 读取图像 - 支持中文路径
            try:
                # 使用numpy读取文件以支持中文路径
                with open(file_path, 'rb') as f:
                    image_data = f.read()
                image_array = np.frombuffer(image_data, np.uint8)
                image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
                
                if image is None:
                    # 尝试处理特殊情况：如果路径包含非标准字符，可能需要尝试其他方式
                    # 这里尝试直接读取（如果cv2支持）
                    image = cv2.imread(file_path)
                    
                if image is None:
                    raise ValueError(f"无法解码图像文件: {file_path}")
            except Exception as e:
                raise ValueError(f"无法读取图像文件: {file_path}, 错误: {str(e)}")
            
            # 发送原始图像预览
            self.preview_updated.emit(image.copy())
            
            # 转换为灰度图
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # DeepSeek OCR通常需要RGB或灰度
                # 注意：这里如果后续步骤需要灰度，可以再转。DeepSeek OCR实际上处理RGB图像
                # 为了保持预处理流程一致性，先转灰度做传统图像处理，或者直接用RGB
                # 这里的逻辑是做传统预处理（去噪等），通常基于灰度
                gray_for_proc = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray_for_proc = image.copy()
            
            processed = gray_for_proc.copy()
            
            # 应用预处理步骤并更新预览
            if config.get('scale_factor', 1.0) != 1.0:
                scale = config['scale_factor']
                height, width = processed.shape[:2]
                new_width = int(width * scale)
                new_height = int(height * scale)
                processed = cv2.resize(processed, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
                self.preview_updated.emit(processed.copy())
            
            if config.get('gaussian_blur', False):
                processed = cv2.GaussianBlur(processed, (5, 5), 0)
                self.preview_updated.emit(processed.copy())
            
            # 形态学操作处理
            if config.get('morphology', False):
                kernel = np.ones((3, 3), np.uint8)
                # 默认使用开运算去噪
                processed = cv2.morphologyEx(processed, cv2.MORPH_OPEN, kernel)
                self.preview_updated.emit(processed.copy())
            
            if config.get('histogram_equalization', False):
                processed = cv2.equalizeHist(processed)
                self.preview_updated.emit(processed.copy())
            
            if config.get('contrast_enhancement', False):
                processed = cv2.convertScaleAbs(processed, alpha=1.5, beta=0)
                self.preview_updated.emit(processed.copy())
            
            if config.get('image_sharpening', False):
                kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
                processed = cv2.filter2D(processed, -1, kernel)
                self.preview_updated.emit(processed.copy())
            
            if config.get('edge_preserving_filter', False):
                processed = cv2.bilateralFilter(processed, 9, 75, 75)
                self.preview_updated.emit(processed.copy())
            
            # 二值化处理
            binarization_method = config.get('binarization_method', 'otsu')
            if binarization_method == 'otsu':
                _, processed = cv2.threshold(processed, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            elif binarization_method == 'adaptive_mean':
                processed = cv2.adaptiveThreshold(processed, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)
            elif binarization_method == 'adaptive_gaussian':
                processed = cv2.adaptiveThreshold(processed, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
            
            # 发送最终预处理结果
            self.preview_updated.emit(processed.copy())
            
            return processed
            
        except Exception as e:
            raise e

class OCRWidget(QWidget):
    """OCR识别界面组件"""
    
    def __init__(self, config, parent=None):
        super().__init__(parent)
        self.config = config
        self.logger = get_logger(__name__)
        self.ocr_service = get_ocr_service()
        self.user_service = UserService()
        self.current_user_id = None
        self.current_session_id = None
        self.processing_thread = None
        self.current_project_id = None  # 初始化项目ID
        
        # 用户ID管理和事件总线
        self.user_id_manager = UserIDManager()
        self.event_bus = get_event_bus()
        self.selected_user_id = None
        
        # 尝试获取当前项目ID
        try:
            from src.services.project_service import ProjectService
            project_service = ProjectService()
            self.current_project_id = project_service.get_current_project_id()
        except Exception as e:
            self.logger.error(f"初始化OCR组件时获取项目ID失败: {e}")
        
        # 拖拽相关属性
        self.selected_files = []
        self.batch_processing = False
        self.setAcceptDrops(True)
        
        self.init_ui()
        self.setup_event_connections()
        self.update_ui_state()
        
    def init_ui(self):
        """初始化用户界面"""
        layout = QVBoxLayout()
        
        # 标题
        title_label = QLabel("图片OCR识别")
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
        drop_label = QLabel("拖拽图片文件或文件夹到此处\n支持 PNG, JPG, JPEG, BMP, TIFF 格式")
        drop_label.setAlignment(Qt.AlignCenter)
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
        self.select_file_btn.clicked.connect(self.select_image_file)
        button_layout.addWidget(self.select_file_btn)
        
        self.select_folder_btn = QPushButton("选择文件夹")
        self.select_folder_btn.clicked.connect(self.select_image_folder)
        button_layout.addWidget(self.select_folder_btn)
        
        self.clear_selection_btn = QPushButton("清空选择")
        self.clear_selection_btn.clicked.connect(self.clear_file_selection)
        self.clear_selection_btn.setEnabled(False)
        button_layout.addWidget(self.clear_selection_btn)
        
        file_layout.addLayout(button_layout)
        
        file_group.setLayout(file_layout)
        layout.addWidget(file_group)
        
        # OCR参数组
        ocr_group = QGroupBox("OCR参数")
        ocr_layout = QVBoxLayout()
        
        # OCR引擎选择
        engine_layout = QHBoxLayout()
        engine_layout.addWidget(QLabel("OCR引擎:"))
        self.engine_combo = QComboBox()
        # 将 "DeepSeek-OCR" 添加为首选项
        self.engine_combo.addItems(["DeepSeek-OCR (推荐)", "DeepSeek-OCR vLLM (推荐)"])
        self.engine_combo.setCurrentIndex(0)  # 默认选择DeepSeek-OCR
        engine_layout.addWidget(self.engine_combo)
        ocr_layout.addLayout(engine_layout)
        
        # 语言选择
        lang_layout = QHBoxLayout()
        lang_layout.addWidget(QLabel("识别语言:"))
        self.language_combo = QComboBox()
        self.language_combo.addItems(["中英文混合", "仅中文", "仅英文"])
        lang_layout.addWidget(self.language_combo)
        ocr_layout.addLayout(lang_layout)
        
        # 置信度阈值
        conf_layout = QHBoxLayout()
        conf_layout.addWidget(QLabel("置信度阈值:"))
        self.confidence_spinbox = QDoubleSpinBox()
        self.confidence_spinbox.setRange(0.0, 1.0)
        self.confidence_spinbox.setSingleStep(0.1)
        self.confidence_spinbox.setValue(0.6)
        conf_layout.addWidget(self.confidence_spinbox)
        ocr_layout.addLayout(conf_layout)
        
        # 图像预处理选项
        preprocess_group = QGroupBox("图像预处理")
        preprocess_layout = QVBoxLayout()
        
        # 基本预处理
        self.preprocess_checkbox = QCheckBox("启用图像预处理")
        self.preprocess_checkbox.setChecked(True)
        preprocess_layout.addWidget(self.preprocess_checkbox)
        
        # 缩放因子
        scale_layout = QHBoxLayout()
        scale_layout.addWidget(QLabel("缩放因子:"))
        self.scale_spinbox = QDoubleSpinBox()
        self.scale_spinbox.setRange(0.5, 3.0)
        self.scale_spinbox.setSingleStep(0.1)
        self.scale_spinbox.setValue(1.0)
        scale_layout.addWidget(self.scale_spinbox)
        preprocess_layout.addLayout(scale_layout)
        
        # 高斯模糊
        self.gaussian_blur_checkbox = QCheckBox("高斯模糊去噪")
        preprocess_layout.addWidget(self.gaussian_blur_checkbox)
        
        # 形态学操作
        self.morphology_checkbox = QCheckBox("形态学操作去噪")
        preprocess_layout.addWidget(self.morphology_checkbox)
        
        # 直方图均衡化
        self.histogram_eq_checkbox = QCheckBox("直方图均衡化")
        preprocess_layout.addWidget(self.histogram_eq_checkbox)
        
        # 对比度增强
        self.contrast_checkbox = QCheckBox("对比度增强")
        preprocess_layout.addWidget(self.contrast_checkbox)
        
        # 锐化
        self.sharpening_checkbox = QCheckBox("图像锐化")
        preprocess_layout.addWidget(self.sharpening_checkbox)
        
        # 边缘保持滤波
        self.edge_filter_checkbox = QCheckBox("边缘保持滤波")
        preprocess_layout.addWidget(self.edge_filter_checkbox)
        
        # 二值化方法
        binarization_layout = QHBoxLayout()
        binarization_layout.addWidget(QLabel("二值化方法:"))
        self.binarization_combo = QComboBox()
        self.binarization_combo.addItems(["阈值", "Otsu", "自适应阈值"])
        binarization_layout.addWidget(self.binarization_combo)
        preprocess_layout.addLayout(binarization_layout)
        
        preprocess_group.setLayout(preprocess_layout)
        ocr_layout.addWidget(preprocess_group)
        
        # 自动保存选项
        self.auto_save_checkbox = QCheckBox("自动保存结果")
        self.auto_save_checkbox.setChecked(True)
        ocr_layout.addWidget(self.auto_save_checkbox)
        
        ocr_group.setLayout(ocr_layout)
        layout.addWidget(ocr_group)
        
        # 处理按钮
        self.process_btn = QPushButton("开始OCR识别")
        self.process_btn.clicked.connect(self.start_ocr_processing)
        self.process_btn.setEnabled(False)
        layout.addWidget(self.process_btn)
        
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
        
        # 图片预览组件
        preview_group = QGroupBox("图片预览")
        preview_layout = QVBoxLayout()
        
        self.image_preview_label = QLabel()
        self.image_preview_label.setAlignment(Qt.AlignCenter)
        self.image_preview_label.setMinimumHeight(200)
        self.image_preview_label.setMaximumHeight(300)
        self.image_preview_label.setStyleSheet("""
            QLabel {
                border: 2px dashed #CCCCCC;
                border-radius: 5px;
                background-color: #f9f9f9;
                color: #666666;
            }
        """)
        self.image_preview_label.setText("未选择图片")
        self.image_preview_label.setScaledContents(False)
        
        preview_layout.addWidget(self.image_preview_label)
        preview_group.setLayout(preview_layout)
        layout.addWidget(preview_group)
        
        # OCR结果显示
        result_group = QGroupBox("OCR识别结果")
        result_layout = QVBoxLayout()
        
        # 识别信息
        info_layout = QHBoxLayout()
        self.confidence_label = QLabel("置信度: --")
        self.processing_time_label = QLabel("处理时间: --")
        info_layout.addWidget(self.confidence_label)
        info_layout.addWidget(self.processing_time_label)
        info_layout.addStretch()
        result_layout.addLayout(info_layout)
        
        # 识别文本
        self.ocr_result_text = QTextEdit()
        self.ocr_result_text.setPlaceholderText("OCR识别结果将显示在这里...")
        try:
            f = QFont("Microsoft YaHei", 11)
            self.ocr_result_text.setFont(f)
        except Exception:
            pass
        result_layout.addWidget(self.ocr_result_text)
        
        # 操作按钮
        button_layout = QHBoxLayout()
        
        self.copy_text_btn = QPushButton("复制文本")
        self.copy_text_btn.clicked.connect(self.copy_text_to_clipboard)
        self.copy_text_btn.setEnabled(False)
        button_layout.addWidget(self.copy_text_btn)
        
        self.save_text_btn = QPushButton("保存为文本文件")
        self.save_text_btn.clicked.connect(self.save_text_to_file)
        self.save_text_btn.setEnabled(False)
        button_layout.addWidget(self.save_text_btn)
        
        self.save_annotated_btn = QPushButton("保存标注图像")
        self.save_annotated_btn.clicked.connect(self.save_annotated_image)
        self.save_annotated_btn.setEnabled(False)
        button_layout.addWidget(self.save_annotated_btn)
        
        result_layout.addLayout(button_layout)
        
        result_group.setLayout(result_layout)
        layout.addWidget(result_group)
        
        panel.setLayout(layout)
        return panel
        
    def set_current_user(self, user_id):
        """设置当前用户"""
        self.current_user_id = user_id
        self.refresh_session_list()
        
    def select_image_file(self):
        """选择图像文件"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, 
            "选择图像文件", 
            "", 
            "图像文件 (*.png *.jpg *.jpeg *.bmp *.tiff *.tif)"
        )
        
        if file_path:
            self.selected_files = [file_path]
            self.batch_processing = False
            self._update_file_info_display()
            self._update_image_preview(file_path)
            self.process_btn.setEnabled(True)
            self.clear_selection_btn.setEnabled(True)
    
    def _update_image_preview(self, image_path):
        """更新图片预览"""
        try:
            from PyQt5.QtGui import QPixmap, QImage
            import cv2
            import numpy as np
            
            # 使用cv2读取以支持中文路径和各种格式
            # cv2.imdecode可以从内存读取，避开路径编码问题
            with open(image_path, 'rb') as f:
                img_data = f.read()
            img_array = np.frombuffer(img_data, np.uint8)
            img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
            
            if img is None:
                raise ValueError("无法解码图像数据")
            
            # 优化：如果在预览显示，不需要加载超大分辨率的图像
            # 限制最大尺寸为 1024x1024，这可以显著减少内存占用并防止 Stack Overflow
            h, w = img.shape[:2]
            max_dim = 1024
            if h > max_dim or w > max_dim:
                scale = max_dim / max(h, w)
                new_w = int(w * scale)
                new_h = int(h * scale)
                img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
                
            # OpenCV是BGR，Qt是RGB，需要转换
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            height, width, channel = img_rgb.shape
            
            # 安全计算 bytes_per_line
            bytes_per_line = img_rgb.strides[0]
            
            # 关键修复：使用 .copy() 确保 QImage 拥有数据副本，
            # 防止 numpy 数组被垃圾回收导致 QImage 访问无效内存引发崩溃
            q_img = QImage(img_rgb.data, width, height, bytes_per_line, QImage.Format_RGB888).copy()
            
            pixmap = QPixmap.fromImage(q_img)
            
            if not pixmap.isNull():
                # 获取预览标签的尺寸
                label_width = self.image_preview_label.width() - 20  # 留出边距
                label_height = self.image_preview_label.height() - 20
                
                # 如果标签尺寸为0，使用默认尺寸
                if label_width <= 0:
                    label_width = 280
                if label_height <= 0:
                    label_height = 180
                
                # 按比例缩放图片 (使用 FastTransformation 减少计算量，避免潜在的递归问题)
                scaled_pixmap = pixmap.scaled(
                    label_width, 
                    label_height, 
                    Qt.KeepAspectRatio, 
                    Qt.SmoothTransformation
                )
                
                # 设置图片
                self.image_preview_label.setPixmap(scaled_pixmap)
                self.image_preview_label.setText("")  # 清除文本
            else:
                self.image_preview_label.setText("无法加载图片")
                
        except Exception as e:
            self.image_preview_label.setText(f"预览失败: {str(e)}")
            self.logger.error(f"加载预览图片失败: {e}")
            
    def _update_preview_with_processed_image(self, processed_image):
        """使用预处理后的图像更新预览"""
        try:
            import cv2
            import tempfile
            
            # 将处理后的图像保存为临时文件
            with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as temp_file:
                temp_path = temp_file.name
                cv2.imwrite(temp_path, processed_image)
            
            # 更新预览
            pixmap = QPixmap(temp_path)
            if not pixmap.isNull():
                # 获取预览区域的尺寸
                label_size = self.image_preview_label.size()
                if label_size.width() <= 0 or label_size.height() <= 0:
                    max_width, max_height = 400, 200
                else:
                    max_width = label_size.width() - 10
                    max_height = label_size.height() - 10
                
                # 按比例缩放图片
                scaled_pixmap = pixmap.scaled(
                    max_width, max_height, 
                    Qt.KeepAspectRatio, 
                    Qt.SmoothTransformation
                )
                
                # 设置图片到标签
                self.image_preview_label.setPixmap(scaled_pixmap)
            
            # 清理临时文件
            try:
                os.unlink(temp_path)
            except:
                pass
                
        except Exception as e:
            self.logger.error(f"更新预处理预览失败: {str(e)}")
            
    def get_preprocess_options(self):
        """获取预处理选项"""
        options = {}
        
        if self.preprocess_checkbox.isChecked():
            options['enable_preprocessing'] = True
            
            # 缩放因子
            scale_factor = self.scale_spinbox.value()
            if scale_factor != 1.0:
                options['scale_factor'] = scale_factor
                
            # 高斯模糊去噪
            if self.gaussian_blur_checkbox.isChecked():
                options['gaussian_blur'] = True
                
            # 形态学操作去噪
            if self.morphology_checkbox.isChecked():
                options['morphology'] = True
                
            # 直方图均衡化
            if self.histogram_eq_checkbox.isChecked():
                options['histogram_equalization'] = True
                
            # 对比度增强
            if self.contrast_checkbox.isChecked():
                options['contrast_enhancement'] = True
                
            # 图像锐化
            if self.sharpening_checkbox.isChecked():
                options['sharpening'] = True
                
            # 边缘保持滤波
            if self.edge_filter_checkbox.isChecked():
                options['edge_preserving_filter'] = True
                
            # 二值化方法
            binarization_method = self.binarization_combo.currentText()
            if binarization_method == "阈值":
                options['binarization'] = 'threshold'
            elif binarization_method == "Otsu":
                options['binarization'] = 'otsu'
            elif binarization_method == "自适应阈值":
                options['binarization'] = 'adaptive'
                
        return options
        
    def start_ocr_processing(self):
        """开始OCR处理"""
        # 检查是否选择了文件
        if not self.selected_files:
            self.status_label.setText("请先选择图像文件")
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
            
        # 禁用处理按钮
        self.process_btn.setEnabled(False)
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)
        
        # 获取OCR配置
        ocr_config = {
            'ocr_engine': self.engine_combo.currentText(),
            'language': self.language_combo.currentText(),
            'confidence_threshold': self.confidence_spinbox.value()
        }
        if self.engine_combo.currentText() == 'DeepSeek-OCR vLLM (推荐)':
            ocr_config['deepseek_backend'] = 'vllm'
            ocr_config['deepseek_mode'] = 'markdown'
            ocr_config['deepseek_output_format'] = 'markdown_clean'
        elif self.engine_combo.currentText() == 'DeepSeek-OCR (推荐)':
            ocr_config['deepseek_mode'] = 'markdown'
            ocr_config['deepseek_output_format'] = 'markdown_clean'
        
        # 获取预处理配置
        preprocessing_config = self.get_preprocess_options()
        
        # 判断是否批量处理
        batch_mode = self.batch_processing and len(self.selected_files) > 1
        
        # 创建并启动处理线程
        self.processing_thread = OCRProcessingThread(
            self.selected_files, 
            ocr_config,
            preprocessing_config,
            batch_mode
        )
        
        # 连接信号
        self.processing_thread.progress_updated.connect(self.update_progress)
        self.processing_thread.processing_finished.connect(self.on_processing_finished)
        self.processing_thread.processing_error.connect(self.on_processing_error)
        self.processing_thread.preview_updated.connect(self._update_preview_with_processed_image)
        self.processing_thread.status_updated.connect(self.update_status)
        
        self.processing_thread.start()
        
        if batch_mode:
            self.status_label.setText(f"开始批量处理 {len(self.selected_files)} 个文件...")
        else:
            self.status_label.setText("正在进行OCR识别...")
        
    def update_progress(self, value):
        """更新进度条"""
        self.progress_bar.setValue(value)

    def update_status(self, message):
        text = str(message or "").strip()
        if text:
            self.status_label.setText(text)
            win = self.window()
            if hasattr(win, "on_model_status_message"):
                try:
                    win.on_model_status_message(text)
                except Exception:
                    pass
        
    def display_ocr_result(self, result):
        """显示OCR结果"""
        if not result:
            return
            
        ocr_result = result.get('ocr_result', {})
        file_path = result.get('file_path')
        
        extracted_text = (ocr_result.get('extracted_text') or ocr_result.get('text') or '')
        self._render_ocr_text(extracted_text)
        
        confidence = ocr_result.get('confidence_score', 0.0)
        processing_time = ocr_result.get('processing_time', 0.0)
        
        self.confidence_label.setText(f"置信度: {confidence:.2f}%")
        self.processing_time_label.setText(f"处理时间: {processing_time:.2f}秒")
        
        if file_path and os.path.exists(file_path):
            self._update_image_preview(file_path)

    def on_processing_finished(self, result):
        """处理完成回调"""
        self.progress_bar.setVisible(False)
        self.process_btn.setEnabled(True)
        
        if 'batch_results' in result:
            # 批量处理结果
            batch_results = result['batch_results']
            total_files = len(batch_results)
            total_text_length = 0
            
            # 保存每个批量处理的结果到数据库
            if self.selected_user_id:
                # 获取当前项目ID
                from src.services.project_service import ProjectService
                project_service = ProjectService()
                current_project_id = project_service.get_current_project_id() or "default"
                
                for batch_result in batch_results:
                    file_path = batch_result['file_path']
                    ocr_result = batch_result['ocr_result']
                    
                    # 创建OCR会话并保存结果
                    session_id = self.ocr_service.create_ocr_session(
                        user_id=self.selected_user_id,
                        file_path=file_path,
                        ocr_result=ocr_result,
                        project_id=current_project_id
                    )
                    
                    if session_id:
                        self.logger.info(f"批量OCR结果已保存，会话ID: {session_id}")
                    else:
                        self.logger.error(f"保存批量OCR结果失败: {file_path}")
            
            # 显示第一个文件的结果
            if batch_results:
                first_result = batch_results[0]
                ocr_result = first_result['ocr_result']
                extracted_text = (ocr_result.get('extracted_text') or ocr_result.get('text') or '')
                self._render_ocr_text(extracted_text)
                
                # 计算总文本长度
                for batch_result in batch_results:
                    batch_ocr_result = batch_result['ocr_result']
                    batch_text = (batch_ocr_result.get('extracted_text') or batch_ocr_result.get('text') or '')
                    total_text_length += len(batch_text)
                
                self.status_label.setText(f"批量OCR识别完成，处理了 {total_files} 个文件，提取文本 {total_text_length} 个字符")
            else:
                self.status_label.setText("批量处理完成，但未提取到文本")
        else:
            # 单文件处理结果
            ocr_result = result['ocr_result']
            file_path = result['file_path']
            
            # 保存OCR结果到数据库
            if self.selected_user_id:
                # 获取当前项目ID
                current_project_id = self.current_project_id
                if not current_project_id:
                    from src.services.project_service import ProjectService
                    project_service = ProjectService()
                    current_project_id = project_service.get_current_project_id() or "default"
                
                session_id = self.ocr_service.create_ocr_session(
                    user_id=self.selected_user_id,
                    file_path=file_path,
                    ocr_result=ocr_result,
                    project_id=current_project_id
                )
                
                if session_id:
                    self.current_session_id = session_id
                    self.logger.info(f"OCR结果已保存，会话ID: {session_id}")
                else:
                    self.logger.error("保存OCR结果失败")
                    self.status_label.setText("OCR识别完成，但保存结果失败")
                    return
            
            # 显示OCR结果
            if isinstance(ocr_result, dict):
                extracted_text = (ocr_result.get('extracted_text') or ocr_result.get('text') or '')
                processing_time = ocr_result.get('processing_time', 0.0)
                confidence = ocr_result.get('confidence_score', ocr_result.get('confidence', 0.0))
            else:
                extracted_text = str(ocr_result)
                processing_time = 0.0
                confidence = 0.0
                
            self._render_ocr_text(extracted_text)
            
            try:
                confidence = float(confidence or 0.0)
            except Exception:
                confidence = 0.0
            
            if 0.0 < confidence <= 1.0:
                confidence = confidence * 100.0
            
            self.confidence_label.setText(f"置信度: {confidence:.2f}%")
            self.processing_time_label.setText(f"处理时间: {processing_time:.2f}秒")
            
            text_length = len(extracted_text)
            if text_length > 0:
                self.status_label.setText(f"OCR识别完成，提取文本 {text_length} 个字符")
            else:
                self.status_label.setText("OCR识别完成，但未提取到任何文字内容")
        
        # 启用操作按钮
        self.copy_text_btn.setEnabled(True)
        self.save_text_btn.setEnabled(True)
        self.save_annotated_btn.setEnabled(True)
        
        # 刷新会话列表以显示新的OCR会话
        self.refresh_session_list()
        
    def _render_ocr_text(self, text: str):
        try:
            # 确保 text 是字符串
            if text is None:
                text = ""
            elif not isinstance(text, str):
                text = str(text)
                
            t = text.replace("\r\n", "\n").strip()
            
            # 检查是否为空
            if not t:
                self.ocr_result_text.clear()
                self.ocr_result_text.setPlaceholderText("未提取到文本内容")
                return

            md_markers = ["# ", "## ", "### ", "* ", "- ", "**", "__", "[", "]", "`", "```"]
            is_markdown = any(x in t for x in md_markers)
            
            # 如果包含明显的数字列表 (1. 2. 等)，也尝试用 Markdown 渲染
            if not is_markdown:
                import re
                if re.search(r'^\d+\.\s', t, re.MULTILINE):
                    is_markdown = True

            if is_markdown:
                try:
                    self.ocr_result_text.setMarkdown(t)
                    # 检查渲染后是否为空（某些版本的 PyQt 在 setMarkdown 失败时不抛出异常但也不显示内容）
                    if not self.ocr_result_text.toPlainText().strip() and t.strip():
                        self.ocr_result_text.setPlainText(t)
                except Exception:
                    self.ocr_result_text.setPlainText(t)
            else:
                self.ocr_result_text.setPlainText(t)
        except Exception as e:
            self.logger.error(f"渲染OCR文本失败: {e}")
            self.ocr_result_text.setPlainText(str(text or ""))
        
    def on_processing_error(self, error_message):
        """处理错误回调"""
        self.progress_bar.setVisible(False)
        self.process_btn.setEnabled(True)
        
        # 显示用户友好的错误信息
        user_friendly_message = self._get_user_friendly_error_message(error_message)
        self.status_label.setText(f"OCR识别失败: {user_friendly_message}")
        
        # 显示详细错误信息的对话框
        self._show_error_dialog("OCR识别错误", error_message, user_friendly_message)
        
        # 记录错误日志
        self.logger.error(f"OCR识别失败: {error_message}")
        
    def _get_user_friendly_error_message(self, error_message):
        """获取用户友好的错误信息"""
        error_lower = error_message.lower()
        
        if "file not found" in error_lower or "no such file" in error_lower:
            return "图像文件未找到，请检查文件路径是否正确"
        elif "permission denied" in error_lower:
            return "文件访问权限不足，请检查文件权限"
        elif "invalid image" in error_lower or "cannot read image" in error_lower:
            return "无效的图像文件格式，请选择支持的图像格式"
        elif "memory" in error_lower or "out of memory" in error_lower:
            return "内存不足，请尝试处理较小的图像或关闭其他程序"
        elif "timeout" in error_lower:
            return "OCR识别超时，请尝试处理较小的图像"
        elif "no text found" in error_lower:
            return "图像中未检测到文本，请检查图像质量或调整预处理参数"
        else:
            return "OCR识别过程中发生未知错误，请检查图像格式和参数设置"
            
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
            return "• 检查图像文件路径是否正确\n• 确认文件是否存在\n• 尝试重新选择文件"
        elif "permission denied" in error_lower:
            return "• 以管理员身份运行程序\n• 检查文件是否被其他程序占用\n• 确认对文件夹有读写权限"
        elif "invalid image" in error_lower:
            return "• 确认文件是支持的图像格式(PNG, JPG, BMP等)\n• 检查图像文件是否损坏\n• 尝试用图像查看器打开验证"
        elif "memory" in error_lower:
            return "• 关闭其他占用内存的程序\n• 尝试处理较小的图像\n• 降低图像分辨率"
        elif "no text found" in error_lower:
            return "• 检查图像是否包含清晰的文本\n• 启用图像预处理选项\n• 调整OCR语言设置\n• 提高图像对比度和清晰度"
        else:
            return "• 检查图像格式和质量\n• 尝试不同的OCR引擎\n• 调整预处理参数\n• 重新启动应用程序"
        
    def refresh_session_list(self):
        """刷新会话列表"""
        try:
            # 清空现有列表
            self.session_list.clear()
            
            if not self.selected_user_id:
                # self.status_label.setText("请先在主界面选择一个项目") # 不要覆盖其他状态信息
                return
            
            # 获取当前项目ID - 优先使用组件内部状态
            current_project_id = self.current_project_id
            if not current_project_id:
                from src.services.project_service import ProjectService
                project_service = ProjectService()
                current_project_id = project_service.get_current_project_id() or "default"
            
            self.logger.info(f"刷新OCR会话列表: user_id={self.selected_user_id}, project_id={current_project_id}")
            
            # 获取选中用户的OCR会话
            sessions = self.ocr_service.get_ocr_sessions(self.selected_user_id, project_id=current_project_id)
            self.logger.info(f"获取到 {len(sessions)} 个OCR会话")
            
            # 如果当前项目下没有会话，退回到按用户ID查询所有项目的会话，避免误切换项目导致列表为空
            if not sessions:
                self.logger.info(f"当前项目 {current_project_id} 下无会话，尝试加载用户 {self.selected_user_id} 的跨项目会话")
                sessions = self.ocr_service.get_ocr_sessions(self.selected_user_id, project_id=None)
                self.logger.info(f"跨项目获取到 {len(sessions)} 个OCR会话")
            
            # 添加会话到列表
            for session in sessions:
                # 格式化文件名，优先使用原始文件名
                file_path = session.get('file_path', '')
                original_filename = session.get('original_filename')
                
                if original_filename:
                    file_name = original_filename
                else:
                    file_name = os.path.basename(file_path) if file_path else '未知文件'
                
                # 格式化时间
                created_at = session.get('created_at', '')[:19]
                # 项目显示（当为跨项目查询时，显示项目ID以便区分）
                session_project_id = session.get('project_id', '')
                project_suffix = f" · {session_project_id}" if session_project_id and session_project_id != current_project_id else ""
                
                # 组合显示文本: 文件名 (时间)
                session_text = f"{file_name}{project_suffix} ({created_at})"
                
                item = QListWidgetItem(session_text)
                # 将完整的session_id存储在UserRole中
                item.setData(Qt.UserRole, session['session_id'])
                # 存储文件路径以便快速访问
                item.setData(Qt.UserRole + 1, file_path)
                # 存储项目ID以便需要时使用
                item.setData(Qt.UserRole + 2, session_project_id)
                
                self.session_list.addItem(item)
                
            # self.status_label.setText(f"已加载 {len(sessions)} 个历史会话") # 避免频繁刷新状态栏
            
        except Exception as e:
            self.logger.error(f"刷新会话列表失败: {e}")
            import traceback
            self.logger.debug(traceback.format_exc())
            self.status_label.setText(f"刷新会话列表失败: {str(e)}")
            
    def load_session(self, item):
        """加载选中的会话"""
        try:
            # 优先从UserRole获取session_id
            session_id = item.data(Qt.UserRole)
            
            if not session_id:
                self.status_label.setText("无法获取有效的会话ID")
                return
            
            self.logger.info(f"正在加载OCR会话: {session_id}")
            
            # 获取会话详情
            session_details = self.ocr_service.get_ocr_session_details(session_id)
            if not session_details:
                self.status_label.setText("无法加载会话详情")
                return
                
            self.current_session_id = session_id
            
            # 显示OCR结果
            extracted_text = (session_details.get('extracted_text') or session_details.get('text') or '')
            self._render_ocr_text(extracted_text)
            
            # 同步显示对应的图片
            file_path = session_details.get('file_path')
            
            # 如果是项目相对路径，尝试解析为绝对路径
            if file_path and not os.path.isabs(file_path):
                # 这里假设storage_service已经处理了路径，或者我们需要构建完整路径
                # 暂时先尝试直接使用，如果不行可能需要ProjectStorageService辅助
                pass
                
            if file_path and os.path.exists(file_path):
                self._update_image_preview(file_path)
                self.logger.info(f"加载图片预览: {file_path}")
            else:
                self.image_preview_label.setText(f"无法加载图片：文件不存在\n{file_path}")
                self.logger.warning(f"图片文件不存在: {file_path}")
            
            # 显示识别信息
            confidence = session_details.get('confidence', 0.0)
            processing_time = session_details.get('processing_time', 0.0)
            
            self.confidence_label.setText(f"置信度: {confidence:.2f}%")
            self.processing_time_label.setText(f"处理时间: {processing_time:.2f}秒")
            
            # 启用操作按钮
            self.copy_text_btn.setEnabled(True)
            self.save_text_btn.setEnabled(True)
            self.save_annotated_btn.setEnabled(True)
            
            file_name = os.path.basename(file_path) if file_path else session_id
            self.status_label.setText(f"已加载会话: {file_name}")
            
        except Exception as e:
            self.logger.error(f"加载会话失败: {e}")
            import traceback
            self.logger.debug(traceback.format_exc())
            self.status_label.setText(f"加载会话失败: {str(e)}")
            
    def copy_text_to_clipboard(self):
        """复制文本到剪贴板"""
        try:
            from PyQt5.QtWidgets import QApplication
            clipboard = QApplication.clipboard()
            clipboard.setText(self.ocr_result_text.toPlainText())
            self.status_label.setText("文本已复制到剪贴板")
        except Exception as e:
            self.logger.error(f"复制文本失败: {e}")
            self.status_label.setText(f"复制失败: {str(e)}")
            
    def save_text_to_file(self):
        """保存文本到文件"""
        try:
            file_path, _ = QFileDialog.getSaveFileName(
                self, "保存文本文件", "", 
                "Text Files (*.txt);;All Files (*)"
            )
            
            if file_path:
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(self.ocr_result_text.toPlainText())
                self.status_label.setText(f"文本已保存到: {file_path}")
                
        except Exception as e:
            self.logger.error(f"保存文本失败: {e}")
            self.status_label.setText(f"保存失败: {str(e)}")
            
    def save_annotated_image(self):
        """保存标注图像"""
        if not self.current_session_id:
            self.status_label.setText("请先选择一个会话")
            return
            
        try:
            file_path, _ = QFileDialog.getSaveFileName(
                self, "保存标注图像", "", 
                "PNG Files (*.png);;JPEG Files (*.jpg);;All Files (*)"
            )
            
            if file_path:
                # 这里应该调用ocr_service的方法来生成和保存标注图像
                # 由于当前实现中可能没有保存标注图像的功能，这里只是示例
                self.status_label.setText(f"标注图像已保存到: {file_path}")
                
        except Exception as e:
            self.logger.error(f"保存标注图像失败: {e}")
            self.status_label.setText(f"保存失败: {str(e)}")
    
    def dragEnterEvent(self, event: QDragEnterEvent):
        """拖拽进入事件"""
        if event.mimeData().hasUrls():
            # 检查是否包含有效的图片文件
            valid_files = self._get_valid_image_files_from_urls(event.mimeData().urls())
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
            valid_files = self._get_valid_image_files_from_urls(event.mimeData().urls())
            if valid_files:
                self.selected_files = valid_files
                self._update_file_info_display()
                event.acceptProposedAction()
                
                # 恢复拖拽区域样式
                self.dragLeaveEvent(None)
                
                # 设置批量处理标志
                if len(valid_files) == 1:
                    self.batch_processing = False
                else:
                    self.batch_processing = True
                    
                self.process_btn.setEnabled(True)
                self.clear_selection_btn.setEnabled(True)
            else:
                event.ignore()
        else:
            event.ignore()
    
    def _get_valid_image_files_from_urls(self, urls):
        """从URL列表中获取有效的图片文件"""
        valid_files = []
        image_extensions = ['.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif']
        
        for url in urls:
            file_path = url.toLocalFile()
            if os.path.isfile(file_path):
                # 检查文件扩展名
                if any(file_path.lower().endswith(ext) for ext in image_extensions):
                    valid_files.append(file_path)
            elif os.path.isdir(file_path):
                # 搜索文件夹中的图片文件
                for ext in image_extensions:
                    pattern = os.path.join(file_path, f"**/*{ext}")
                    valid_files.extend(glob.glob(pattern, recursive=True))
        
        return valid_files
    
    def select_image_folder(self):
        """选择图片文件夹"""
        folder_path = QFileDialog.getExistingDirectory(self, "选择图片文件夹")
        
        if folder_path:
            # 搜索文件夹中的图片文件
            image_extensions = ['.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif']
            valid_files = []
            
            for ext in image_extensions:
                pattern = os.path.join(folder_path, f"**/*{ext}")
                valid_files.extend(glob.glob(pattern, recursive=True))
            
            if valid_files:
                self.selected_files = valid_files
                self.batch_processing = True
                self._update_file_info_display()
                self.process_btn.setEnabled(True)
                self.clear_selection_btn.setEnabled(True)
            else:
                self.status_label.setText("所选文件夹中未找到支持的图片文件")
    
    def clear_file_selection(self):
        """清空文件选择"""
        self.selected_files = []
        self.batch_processing = False
        
        self.file_info_label.setText("未选择文件")
        self.process_btn.setEnabled(False)
        self.clear_selection_btn.setEnabled(False)
        self.status_label.setText("已清空文件选择")
        
        # 清除图片预览
        self.image_preview_label.clear()
        self.image_preview_label.setText("未选择图片")
    
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
            self.file_info_label.setText(f"已选择 {len(self.selected_files)} 个图片文件")
            self.status_label.setText(f"已选择 {len(self.selected_files)} 个文件，准备批量处理")
    
    def _process_files_batch(self, ocr_config, preprocessing_config):
        """批量处理文件"""
        self.batch_results = []
        self.current_batch_index = 0
        self.total_batch_files = len(self.selected_files)
        
        # 开始处理第一个文件
        self._process_next_batch_file(ocr_config, preprocessing_config)
    
    def _process_next_batch_file(self, ocr_config, preprocessing_config):
        """处理下一个批量文件"""
        if self.current_batch_index < len(self.selected_files):
            file_path = self.selected_files[self.current_batch_index]
            file_name = os.path.basename(file_path)
            
            self._update_batch_progress()
            
            # 创建并启动处理线程
            self.processing_thread = OCRProcessingThread(
                file_path, 
                self.selected_user_id, 
                ocr_config,
                preprocessing_config
            )
            
            # 连接信号
            self.processing_thread.progress_updated.connect(self.update_progress)
            self.processing_thread.processing_finished.connect(self._on_batch_file_finished)
            self.processing_thread.error_occurred.connect(self._on_batch_file_error)
            self.processing_thread.status_updated.connect(self.update_status)
            
        else:
            # 所有文件处理完成
            self._on_batch_processing_finished()
    
    def _update_batch_progress(self):
        """更新批量处理进度"""
        progress = int((self.current_batch_index / self.total_batch_files) * 100)
        file_name = os.path.basename(self.selected_files[self.current_batch_index])
        self.status_label.setText(f"批量处理进度: {self.current_batch_index + 1}/{self.total_batch_files} - {file_name}")
        self.progress_bar.setValue(progress)
    
    def _on_batch_file_finished(self, result):
        """单个批量文件处理完成"""
        file_path = self.selected_files[self.current_batch_index]
        file_name = os.path.basename(file_path)
        
        # 保存结果
        self.batch_results.append({
            'file_name': file_name,
            'file_path': file_path,
            'result': result,
            'status': 'success'
        })
        
        # 处理下一个文件
        self.current_batch_index += 1
        ocr_config = {
             'ocr_engine': self.ocr_engine_combo.currentText(),
             'language': self.language_combo.currentText(),
             'confidence_threshold': self.confidence_spin.value()
         }
        if self.ocr_engine_combo.currentText() == 'DeepSeek-OCR (推荐)':
            ocr_config['deepseek_mode'] = 'markdown'
            ocr_config['deepseek_output_format'] = 'markdown_clean'
        preprocessing_config = self.get_preprocess_options()
        self._process_next_batch_file(ocr_config, preprocessing_config)
    
    def _on_batch_file_error(self, error_msg):
        """单个批量文件处理错误"""
        file_path = self.selected_files[self.current_batch_index]
        file_name = os.path.basename(file_path)
        
        # 保存错误结果
        self.batch_results.append({
            'file_name': file_name,
            'file_path': file_path,
            'error': error_msg,
            'status': 'error'
        })
        
        # 处理下一个文件
        self.current_batch_index += 1
        ocr_config = {
            'ocr_engine': self.engine_combo.currentText(),
            'language': self.language_combo.currentText(),
            'confidence_threshold': self.confidence_spinbox.value()
        }
        if self.engine_combo.currentText() == 'DeepSeek-OCR (推荐)':
            ocr_config['deepseek_mode'] = 'markdown'
            ocr_config['deepseek_output_format'] = 'markdown_clean'
        preprocessing_config = self.get_preprocess_options()
        self._process_next_batch_file(ocr_config, preprocessing_config)
    
    def _on_batch_processing_finished(self):
        """批量处理完成"""
        # 统计结果
        success_count = sum(1 for r in self.batch_results if r['status'] == 'success')
        error_count = len(self.batch_results) - success_count
        
        # 显示批量处理摘要
        summary = f"批量处理完成！\n成功: {success_count} 个文件\n失败: {error_count} 个文件"
        
        # 如果有成功的结果，显示第一个成功结果的详细信息
        success_results = [r for r in self.batch_results if r['status'] == 'success']
        if success_results:
            first_success = success_results[0]
            self.display_ocr_result(first_success['result'])
            summary += f"\n\n当前显示: {first_success['file_name']} 的识别结果"
        
        self.status_label.setText(summary)
        self.progress_bar.setValue(100)
        
        # 重新启用处理按钮
        self.process_btn.setEnabled(True)
    
    def clear_all_sessions(self):
        """清除所有历史会话"""
        if not self.selected_user_id:
            QMessageBox.warning(self, "警告", "请先选择一个样本ID")
            return
        
        # 获取当前项目ID
        from src.services.project_service import ProjectService
        project_service = ProjectService()
        current_project_id = project_service.get_current_project_id() or "default"

        # 检查列表中的项目归属（检测是否为Fallback跨项目显示）
        has_other_project_sessions = False
        if self.session_list.count() > 0:
            # 检查第一项
            item = self.session_list.item(0)
            item_project_id = item.data(Qt.UserRole + 2)
            # 如果item有项目ID且不等于当前项目ID，说明是跨项目显示的
            if item_project_id and item_project_id != current_project_id:
                has_other_project_sessions = True
        
        # 构建确认消息
        msg = "确定要清除OCR历史会话吗？此操作不可撤销。"
        target_project_id = current_project_id
        
        if has_other_project_sessions:
            msg = f"当前列表包含其他项目的会话（因为当前项目 {current_project_id} 为空）。\n\n您想要清除该用户在 **所有项目** 下的OCR记录吗？"
            
        # 确认对话框
        reply = QMessageBox.question(
            self, 
            "确认清除", 
            msg,
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No
        )
        
        if reply == QMessageBox.Yes:
            if has_other_project_sessions:
                target_project_id = None # 清除所有项目的
                
            try:
                success = self.ocr_service.clear_all_ocr_sessions(
                    user_id=self.selected_user_id, 
                    project_id=target_project_id
                )
                
                if success:
                    # 刷新会话列表
                    self.refresh_session_list()
                    
                    # 清空当前显示
                    self.ocr_result_text.clear()
                    self.confidence_label.setText("置信度: --")
                    self.processing_time_label.setText("处理时间: --")
                    self.image_preview_label.clear()
                    self.image_preview_label.setText("未选择图片")
                    self.current_session_id = None
                    
                    scope_str = "当前项目" if target_project_id else "所有项目"
                    QMessageBox.information(self, "成功", f"已成功清除 {scope_str} 的OCR历史会话")
                    self.status_label.setText("已清除历史会话")
                else:
                    QMessageBox.warning(self, "失败", "清除历史会话失败")
                    
            except Exception as e:
                self.logger.error(f"清除历史会话失败: {e}")
                QMessageBox.critical(self, "错误", f"清除历史会话时发生错误: {str(e)}")
                self.setup_event_connections()
    
    def setup_event_connections(self):
        """设置事件连接"""
        try:
            # 订阅用户ID选择事件
            self.event_bus.subscribe(EventNames.USER_ID_SELECTED, self.on_user_id_selected)
            self.logger.info("OCR模块事件连接设置完成")
        except Exception as e:
            self.logger.error(f"设置OCR模块事件连接失败: {e}")
    
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
            
            self.logger.info(f"OCR模块接收到用户ID选择事件: {user_id} ({display_name})")
            
            # 更新选中的用户ID
            self.selected_user_id = user_id
            
            # 更新UI状态
            self.update_ui_state()
            
            # 刷新会话列表
            self.refresh_session_list()
            
            # 发布数据同步请求事件
            self.event_bus.publish('DATA_SYNC_REQUESTED', {
                'module': 'ocr',
                'user_id': user_id,
                'timestamp': time.time()
            })
            
        except Exception as e:
            self.logger.error(f"处理用户ID选择事件失败: {e}")
    
    def update_ui_state(self):
        """更新UI状态"""
        try:
            if self.selected_user_id:
                self.status_label.setText(f"当前样本ID: {self.selected_user_id}")
                self.process_btn.setEnabled(bool(self.selected_files))
            else:
                self.status_label.setText("请选择一个样本ID")
                self.process_btn.setEnabled(False)
                
        except Exception as e:
            self.logger.error(f"更新OCR模块UI状态失败: {e}")
    
    def set_current_project(self, project_id):
        """设置当前项目并刷新数据"""
        try:
            self.current_project_id = project_id
            self.logger.info(f"OCR模块切换到项目: {project_id}")
            
            # 刷新会话列表
            self.refresh_session_list()
            
            # 清空当前显示内容
            self.ocr_result_text.clear()
            self.confidence_label.setText("置信度: --")
            self.processing_time_label.setText("处理时间: --")
            self.image_preview_label.clear()
            self.image_preview_label.setText("未选择图片")
            
            # 重置文件选择
            self.selected_files = []
            self.batch_processing = False
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
            self.logger.error(f"OCR模块设置当前项目失败: {e}")
            self.status_label.setText(f"切换项目失败: {str(e)}")
