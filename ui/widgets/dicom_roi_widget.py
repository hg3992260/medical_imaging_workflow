
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, 
    QFileDialog, QTextEdit, QScrollArea, QGroupBox, QSpinBox,
    QCheckBox, QProgressBar, QListWidget, QSplitter, QFrame, QSlider,
    QMessageBox, QDoubleSpinBox, QListWidgetItem
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QUrl, QMimeData
from PyQt5.QtGui import QPixmap, QFont, QDragEnterEvent, QDropEvent, QPalette, QImage
import os
import sys
import glob
import time
import json
import numpy as np
import cv2

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.services.dicom_service import DicomService
from src.services.user_service import UserService
from src.core.user_id_manager import UserIDManager
from src.core.event_bus import get_event_bus, EventNames
from src.utils.logger import get_logger

class DicomProcessingThread(QThread):
    """DICOM处理线程"""
    progress_updated = pyqtSignal(int)
    processing_finished = pyqtSignal(dict)
    error_occurred = pyqtSignal(str)
    
    def __init__(self, file_path, user_id, auto_save=True, debug_mode=False, project_id=None):
        super().__init__()
        self.file_path = file_path
        self.user_id = user_id
        self.auto_save = auto_save
        self.debug_mode = debug_mode
        self.project_id = project_id
        self.dicom_service = DicomService()
        self._is_running = True  # 添加运行状态标志
        
    def stop(self):
        """安全停止线程"""
        self._is_running = False
        
    def run(self):
        try:
            self.progress_updated.emit(10)
            
            # 检查是否应该停止
            if not self._is_running:
                return
                
            # 加载DICOM文件
            dicom_data = self.dicom_service.load_dicom_file(self.file_path)
            if not dicom_data:
                self.error_occurred.emit("无法加载DICOM文件")
                return
                
            self.progress_updated.emit(30)
            
            # 检查是否应该停止
            if not self._is_running:
                return
                
            # 提取DICOM信息
            dicom_info = self.dicom_service.extract_dicom_info(dicom_data)
            
            # 转换为图像数组 (Raw Image - usually clean without overlay burned in)
            image_array = self.dicom_service.convert_to_image(dicom_data)
            if image_array is None:
                self.error_occurred.emit("无法转换DICOM为图像")
                return
                
            self.progress_updated.emit(50)
            
            # 检查是否应该停止
            if not self._is_running:
                return
                
            # 提取覆盖层并OCR
            overlay_images = {}
            roi_regions = []
            overlay_entries = []
            
            import numpy as np
            import cv2
            from PIL import Image
            import tempfile
            
            # 优先查找并使用磁盘上已存在的覆盖层图片
            found_overlays_dict = {}
            
            # 生成存储路径
            storage_dir = os.path.join(os.path.dirname(self.file_path), "overlays")
            storage_dir = os.path.normpath(storage_dir)
            base_name = os.path.basename(self.file_path)
            file_stem = os.path.splitext(base_name)[0]
            
            print(f"DEBUG: Checking for overlays in {storage_dir}")
            
            # 检查是否存在已生成的覆盖层图片
            if os.path.exists(storage_dir):
                # 模式1: 带组号的命名 (overlay_filename_6000.png)
                pattern_group = os.path.join(storage_dir, f"overlay_{file_stem}_????.png")
                files_group = glob.glob(pattern_group)
                print(f"DEBUG: Found group files: {len(files_group)}")
                
                # 模式2: 旧版命名 (overlay_filename.png)
                pattern_single = os.path.join(storage_dir, f"overlay_{file_stem}.png")
                if os.path.exists(pattern_single):
                    files_group.append(pattern_single)
                    print("DEBUG: Found single file")
                
                if files_group:
                    print(f"Found {len(files_group)} existing overlay images in {storage_dir}, skipping extraction.")
                    for overlay_path in files_group:
                        try:
                            # 读取图像
                            mask = cv2.imread(overlay_path, cv2.IMREAD_GRAYSCALE)
                            if mask is None:
                                continue
                                
                            # 从文件名解析组号
                            filename = os.path.basename(overlay_path)
                            name_no_ext = os.path.splitext(filename)[0]
                            parts = name_no_ext.split('_')
                            last_part = parts[-1]
                            
                            # 尝试解析十六进制组号
                            if len(last_part) == 4 and all(c in '0123456789ABCDEFabcdef' for c in last_part):
                                group = int(last_part, 16)
                            else:
                                group = 0x6000 # 默认组号
                                
                            # 避免重复 (如果同一组有多个文件，优先保留带组号的)
                            if group not in found_overlays_dict:
                                found_overlays_dict[group] = {'mask': mask, 'path': overlay_path}
                            else:
                                # 如果已经存在，且当前文件是带组号的(更精确)，则覆盖
                                if f"{group:04X}" in filename or f"{group:04x}" in filename:
                                    found_overlays_dict[group] = {'mask': mask, 'path': overlay_path}
                                    
                        except Exception as e:
                            print(f"Error loading existing overlay {overlay_path}: {e}")
            
            found_overlays = list(found_overlays_dict.items())
            
            # 如果未找到磁盘覆盖层，尝试从DICOM中提取覆盖层
            if not found_overlays:
                for group in range(0x6000, 0x601F, 2):
                    try:
                        mask = None
                        if hasattr(dicom_data, 'overlay_array'):
                            overlays = dicom_data.overlay_array(group)
                            if overlays is not None and overlays.size > 0:
                                mask = (overlays.astype(np.uint8) * 255)
                        if mask is None:
                            data_tag = (group, 0x3000)
                            if data_tag in dicom_data:
                                rows_tag = (group, 0x0010)
                                cols_tag = (group, 0x0011)
                                overlay_rows = dicom_data[rows_tag].value if rows_tag in dicom_data else image_array.shape[0]
                                overlay_cols = dicom_data[cols_tag].value if cols_tag in dicom_data else image_array.shape[1]
                                overlay_data = dicom_data[data_tag].value
                                total_bits = overlay_rows * overlay_cols
                                required_bytes = (total_bits + 7) // 8
                                if len(overlay_data) < required_bytes:
                                    overlay_data = overlay_data + b'\x00' * (required_bytes - len(overlay_data))
                                elif len(overlay_data) > required_bytes:
                                    overlay_data = overlay_data[:required_bytes]
                                overlay_bits = np.unpackbits(np.frombuffer(overlay_data, dtype=np.uint8))
                                overlay_bits = overlay_bits[:total_bits]
                                overlay_mask_raw = overlay_bits.reshape((overlay_rows, overlay_cols))
                                mask = (overlay_mask_raw.astype(np.uint8) * 255)
                        if mask is not None:
                            # 保存到磁盘以便持久化
                            storage_dir = os.path.join(os.path.dirname(self.file_path), "overlays")
                            os.makedirs(storage_dir, exist_ok=True)
                            base_name = os.path.basename(self.file_path)
                            overlay_filename = f"overlay_{os.path.splitext(base_name)[0]}_{group:04X}.png"
                            overlay_path = os.path.join(storage_dir, overlay_filename)
                            cv2.imwrite(overlay_path, mask)
                            found_overlays.append((group, {'mask': mask, 'path': overlay_path}))
                    except Exception as e:
                        print(f"Overlay extraction error (group {group:04X}): {e}")
            
            self.progress_updated.emit(70)
            
            # 检查是否应该停止
            if not self._is_running:
                return
                
            overlay_images['raw'] = image_array
            
            if found_overlays:
                # 生成覆盖层图片存储
                storage_dir = os.path.join(os.path.dirname(self.file_path), "overlays")
                os.makedirs(storage_dir, exist_ok=True)
                base_name = os.path.basename(self.file_path)
                
                # 组合所有覆盖层用于显示
                combined_mask = np.zeros_like(found_overlays[0][1]['mask'])
                
                # 遍历所有找到的覆盖层并进行OCR
                for group, entry in found_overlays:
                    mask = entry.get('mask')
                    overlay_path = entry.get('path')
                    # 组合显示
                    combined_mask = cv2.bitwise_or(combined_mask, mask)
                    
                    # 调用OCR
                    try:
                        print(f"Starting OCR for {overlay_path} (Group {group:04X}) using DeepSeek...")
                        from src.services.ocr_service import get_ocr_service
                        ocr_service = get_ocr_service()
                        
                        # 使用DeepSeek引擎识别
                        ocr_result = ocr_service.extract_text_from_image(overlay_path, {'engine': 'deepseek'})
                        print(f"OCR Result (Group {group:04X}): {ocr_result}")
                        
                        # 读取图像数据
                        image_data = None
                        if os.path.exists(overlay_path):
                            with open(overlay_path, 'rb') as f:
                                image_data = f.read()
                        
                        if ocr_result and ocr_result.get('text'):
                            roi_regions.append({
                                'bbox': {'x': 0, 'y': 0, 'width': int(image_array.shape[1]), 'height': int(image_array.shape[0])},
                                'area': float(image_array.shape[0]*image_array.shape[1]),
                                'ocr_result': {'text': ocr_result.get('text',''), 'confidence': ocr_result.get('confidence',0.0)},
                                'image_data': image_data, # 添加图像数据用于持久化
                                'source': f'overlay_{group:04X}'
                            })
                            overlay_entries.append({
                                'group': f'{group:04X}',
                                'path': overlay_path,
                                'text': ocr_result.get('text',''),
                                'confidence': ocr_result.get('confidence',0.0)
                            })
                    except Exception as e:
                        print(f"OCR failed for group {group:04X}: {e}")
                
                # 准备显示用的覆盖层图像 (RGB)
                overlay_display = cv2.cvtColor(combined_mask, cv2.COLOR_GRAY2RGB)
                overlay_display[combined_mask > 0] = [255, 0, 0] 
                
                overlay_images['overlay'] = overlay_display
                # 使用第一个覆盖层路径作为主路径
                first_group, first_entry = found_overlays[0]
                overlay_images['overlay_path'] = first_entry.get('path')
                overlay_images['overlays'] = overlay_entries
                    
            else:
                overlay_images['overlay'] = None
            
            self.progress_updated.emit(90)
            
            # 检查是否应该停止
            if not self._is_running:
                return
                
            # 创建DICOM会话（如果启用自动保存）
            session_id = None
            if self.auto_save:
                current_project_id = self.project_id
                if not current_project_id:
                    from src.services.project_service import ProjectService
                    project_service = ProjectService()
                    current_project_id = project_service.get_current_project_id() or "default"
                
                session_id = self.dicom_service.create_dicom_session(
                    user_id=self.user_id,
                    file_path=self.file_path,
                    dicom_info=dicom_info,
                    roi_data=roi_regions,
                    project_id=current_project_id
                )
            
            self.progress_updated.emit(100)
            
            result = {
                'session_id': session_id,
                'dicom_info': dicom_info,
                'roi_regions': roi_regions,
                'image_array': image_array,
                'overlay_images': overlay_images
            }
            
            self.processing_finished.emit(result)
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            self.error_occurred.emit(f"处理过程中发生错误: {str(e)}")

class OverlayFolderProcessingThread(QThread):
    progress_updated = pyqtSignal(int)
    processing_finished = pyqtSignal(dict)
    error_occurred = pyqtSignal(str)
    def __init__(self, folder_path, user_id):
        super().__init__()
        self.folder_path = folder_path
        self.user_id = user_id
        self._is_running = True
    def stop(self):
        self._is_running = False
    def run(self):
        try:
            import glob
            import os
            import numpy as np
            import cv2
            self.progress_updated.emit(10)
            if not self._is_running:
                return
                
            # 获取所有覆盖层文件
            overlay_files = glob.glob(os.path.join(self.folder_path, "*.png"))
            
            # 按基础文件名分组（支持 overlay_{base_name}_{group}.png 和 overlay_{base_name}.png）
            grouped_overlays = {}
            for path in overlay_files:
                filename = os.path.basename(path)
                # 尝试解析文件名
                if filename.startswith('overlay_'):
                    # 去掉前缀和扩展名
                    name_part = os.path.splitext(filename)[0][8:]
                    
                    # 尝试检测末尾是否为4位十六进制组号
                    parts = name_part.split('_')
                    if len(parts) > 1:
                        last_part = parts[-1]
                        # 检查最后一部分是否为4位十六进制数（如6000）
                        if len(last_part) == 4 and all(c in '0123456789ABCDEFabcdef' for c in last_part):
                            base_name = "_".join(parts[:-1])
                        else:
                            # 如果不是组号，则整个部分都是基础文件名
                            base_name = name_part
                    else:
                        base_name = name_part
                        
                    if base_name not in grouped_overlays:
                        grouped_overlays[base_name] = []
                    grouped_overlays[base_name].append(path)
            
            # 处理每一组覆盖层
            processed_count = 0
            total_groups = len(grouped_overlays)
            
            from src.services.ocr_service import get_ocr_service
            ocr_service = get_ocr_service()
            
            for base_name, paths in grouped_overlays.items():
                if not self._is_running:
                    return
                
                # 尝试寻找对应的DICOM文件
                parent_dir = os.path.dirname(self.folder_path)
                dicom_file_path = None
                dicom_info = {'file_name': base_name} # 初始化默认值
                
                # 策略1: 直接匹配文件名（无扩展名）
                potential_path = os.path.join(parent_dir, base_name)
                if os.path.exists(potential_path) and os.path.isfile(potential_path):
                     dicom_file_path = potential_path
                
                # 策略2: 匹配带扩展名的文件
                if not dicom_file_path:
                    for ext in ['.dcm', '.dicom', '']:
                        p = os.path.join(parent_dir, base_name + ext)
                        if os.path.exists(p) and os.path.isfile(p):
                            dicom_file_path = p
                            break
                            
                # 策略3: 遍历目录寻找stem匹配的文件
                if not dicom_file_path:
                    try:
                        for f in os.listdir(parent_dir):
                            f_path = os.path.join(parent_dir, f)
                            if os.path.isfile(f_path):
                                if os.path.splitext(f)[0] == base_name:
                                    dicom_file_path = f_path
                                    break
                    except Exception:
                        pass
                
                # 如果找到了DICOM文件，提取详细信息
                if dicom_file_path:
                    try:
                        from src.services.dicom_service import DicomService
                        temp_service = DicomService()
                        dicom_data = temp_service.load_dicom_file(dicom_file_path)
                        if dicom_data:
                            full_info = temp_service.extract_dicom_info(dicom_data)
                            dicom_info.update(full_info)
                    except Exception as e:
                        print(f"Error extracting DICOM info for {dicom_file_path}: {e}")
                    
                overlay_entries = []
                roi_regions = []
                combined_mask = None
                
                for path in paths:
                    gray = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
                    if gray is None:
                        continue
                        
                    result = ocr_service.extract_text_from_image(path, {'engine': 'deepseek'})
                    text = result.get('text', '') if isinstance(result, dict) else ''
                    conf = result.get('confidence', 0.0) if isinstance(result, dict) else 0.0
                    
                    group_str = os.path.splitext(os.path.basename(path))[0].split('_')[-1]
                    overlay_entries.append({'group': group_str, 'path': path, 'text': text, 'confidence': conf})
                    
                    # 读取覆盖层图像数据作为字节流，用于持久化
                    with open(path, 'rb') as f:
                        image_data = f.read()
                        
                    roi_regions.append({
                        'bbox': {'x': 0, 'y': 0, 'width': gray.shape[1], 'height': gray.shape[0]}, 
                        'area': float(gray.shape[0]*gray.shape[1]), 
                        'ocr_result': {'text': text, 'confidence': conf},
                        'image_data': image_data, # 添加图像数据用于持久化
                        'source': f'overlay_{group_str}'
                    })
                    
                    if combined_mask is None:
                        combined_mask = np.zeros_like(gray)
                    if combined_mask.shape == gray.shape:
                        combined_mask = cv2.bitwise_or(combined_mask, (gray > 0).astype(np.uint8)*255)
                
                overlay_images = {}
                if combined_mask is not None:
                    overlay_display = cv2.cvtColor(combined_mask, cv2.COLOR_GRAY2RGB)
                    overlay_display[combined_mask > 0] = [255, 0, 0]
                    overlay_images['overlay'] = overlay_display
                else:
                    overlay_images['overlay'] = None
                overlay_images['overlays'] = overlay_entries
                
                # 构建结果
                result = {
                    'session_id': None, 
                    'dicom_info': dicom_info, # 使用提取的完整信息
                    'roi_regions': roi_regions, 
                    'image_array': None, 
                    'overlay_images': overlay_images,
                    'file_identifier': base_name, # 添加标识符
                    'dicom_file_path': dicom_file_path # 传递找到的DICOM文件路径
                }
                
                self.processing_finished.emit(result)
                
                processed_count += 1
                progress = 10 + int((processed_count / total_groups) * 90)
                self.progress_updated.emit(progress)
                
            self.progress_updated.emit(100)
            # 发送结束信号
            self.processing_finished.emit({})
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            self.error_occurred.emit(str(e))
class DicomRoiWidget(QWidget):
    """DICOM ROI识别界面组件"""
    
    def __init__(self, config, parent=None):
        super().__init__(parent)
        self.config = config
        self.logger = get_logger(__name__)
        self.dicom_service = DicomService()
        self.user_service = UserService()
        self.user_id_manager = UserIDManager()
        self.event_bus = get_event_bus()
        self.current_user_id = None
        self.selected_user_id = None  # 当前选中的样本ID
        self.current_session_id = None
        self.processing_thread = None
        self.selected_files = []  # 存储选中的文件列表
        self.batch_processing = False  # 批量处理标志
        
        # DICOM文件导航相关变量
        self.dicom_files_list = []  # 当前路径下的所有DICOM文件
        self.current_image_index = 0  # 当前显示的图像索引
        self.current_project_id = None  # 当前项目ID
        
        self.init_ui()
        self.setup_event_connections()
        
        # 初始化UI状态
        self.update_ui_state()
        
        self.logger.info("DICOM ROI组件初始化完成")
    
    def init_ui(self):
        """初始化UI"""
        # 主布局
        main_layout = QHBoxLayout()
        
        # 左侧控制面板
        left_panel = self.create_left_panel()
        main_layout.addWidget(left_panel, 1)
        
        # 中间图像显示区域
        center_panel = self.create_center_panel()
        main_layout.addWidget(center_panel, 3)
        
        # 右侧信息面板
        right_panel = self.create_right_panel()
        main_layout.addWidget(right_panel, 2)
        
        self.setLayout(main_layout)
        
        # 设置样式
        self.setStyleSheet("""
            QGroupBox {
                font-weight: bold;
                border: 1px solid #ccc;
                border-radius: 5px;
                margin-top: 10px;
                padding-top: 10px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px 0 5px;
            }
        """)
    
    def create_left_panel(self):
        """创建左侧控制面板"""
        panel = QGroupBox("控制面板")
        layout = QVBoxLayout()
        
        # 文件选择区域
        file_group = QGroupBox("文件选择")
        file_layout = QVBoxLayout()
        
        # 拖拽区域
        self.drop_area = QFrame()
        self.drop_area.setFrameStyle(QFrame.Box)
        self.drop_area.setMinimumHeight(100)
        self.drop_area.setAcceptDrops(True)
        self.drop_area.setStyleSheet("""
            QFrame {
                border: 2px dashed #aaa;
                border-radius: 10px;
                background-color: #f9f9f9;
                color: #666;
            }
            QFrame:hover {
                border-color: #007acc;
                background-color: #e6f3ff;
            }
        """)
        
        drop_label = QLabel("拖拽DICOM文件或文件夹到此处\n或点击下方按钮选择")
        drop_label.setAlignment(Qt.AlignCenter)
        drop_label.setWordWrap(True)
        self.drop_area.setLayout(QVBoxLayout())
        self.drop_area.layout().addWidget(drop_label)
        
        # 设置拖拽事件
        self.drop_area.dragEnterEvent = self.dragEnterEvent
        self.drop_area.dragLeaveEvent = self.dragLeaveEvent
        self.drop_area.dropEvent = self.dropEvent
        
        file_layout.addWidget(self.drop_area)
        
        # 文件选择按钮
        select_file_btn = QPushButton("选择DICOM文件")
        select_file_btn.clicked.connect(self.select_dicom_file)
        file_layout.addWidget(select_file_btn)
        
        select_folder_btn = QPushButton("选择DICOM文件夹")
        select_folder_btn.clicked.connect(self.select_dicom_folder)
        file_layout.addWidget(select_folder_btn)
        
        self.clear_selection_btn = QPushButton("清空选择")
        self.clear_selection_btn.clicked.connect(self.clear_file_selection)
        self.clear_selection_btn.setEnabled(False)
        file_layout.addWidget(self.clear_selection_btn)
        
        overlay_folder_group = QGroupBox("覆盖层文件夹")
        overlay_folder_layout = QVBoxLayout()
        self.overlay_folder_label = QLabel("未选择覆盖层文件夹")
        overlay_folder_layout.addWidget(self.overlay_folder_label)
        select_overlay_folder_btn = QPushButton("选择覆盖层文件夹")
        select_overlay_folder_btn.clicked.connect(self.select_overlay_folder)
        overlay_folder_layout.addWidget(select_overlay_folder_btn)
        self.process_overlay_folder_btn = QPushButton("处理覆盖层文件夹")
        self.process_overlay_folder_btn.clicked.connect(self.start_overlay_folder_processing)
        self.process_overlay_folder_btn.setEnabled(False)
        overlay_folder_layout.addWidget(self.process_overlay_folder_btn)
        overlay_folder_group.setLayout(overlay_folder_layout)
        file_layout.addWidget(overlay_folder_group)
        
        # 文件列表显示
        self.file_list_widget = QListWidget()
        self.file_list_widget.itemClicked.connect(self.on_file_list_item_clicked)
        self.file_list_widget.setStyleSheet("""
            QListWidget {
                border: 1px solid #ccc;
                background-color: #fff;
                color: #333;
                min-height: 100px;
            }
        """)
        file_layout.addWidget(self.file_list_widget)
        
        file_group.setLayout(file_layout)
        layout.addWidget(file_group)
        
        # 处理选项
        options_group = QGroupBox("处理选项")
        options_layout = QVBoxLayout()
        
        self.auto_save_checkbox = QCheckBox("自动保存结果")
        self.auto_save_checkbox.setChecked(True)
        options_layout.addWidget(self.auto_save_checkbox)
        
        self.debug_mode_checkbox = QCheckBox("调试模式")
        self.debug_mode_checkbox.setChecked(False)
        options_layout.addWidget(self.debug_mode_checkbox)
        
        options_group.setLayout(options_layout)
        layout.addWidget(options_group)
        
        # 处理按钮
        self.process_btn = QPushButton("开始处理")
        self.process_btn.clicked.connect(self.start_roi_processing)
        self.process_btn.setEnabled(False)
        self.process_btn.setStyleSheet("""
            QPushButton {
                background-color: #007acc;
                color: white;
                font-weight: bold;
                padding: 10px;
                border-radius: 5px;
            }
            QPushButton:hover {
                background-color: #005a9e;
            }
            QPushButton:disabled {
                background-color: #ccc;
                color: #666;
            }
        """)
        layout.addWidget(self.process_btn)
        
        # 进度条
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        layout.addWidget(self.progress_bar)
        
        # 状态标签
        self.status_label = QLabel("请在主界面选择一个项目")
        self.status_label.setWordWrap(True)
        self.status_label.setStyleSheet("font-size: 12px; color: #333; padding: 5px;")
        layout.addWidget(self.status_label)
        
        # 历史会话
        history_group = QGroupBox("历史会话")
        history_layout = QVBoxLayout()
        
        self.session_list = QListWidget()
        self.session_list.itemClicked.connect(self.load_session)
        history_layout.addWidget(self.session_list)
        
        # 会话操作按钮
        session_btn_layout = QHBoxLayout()
        refresh_btn = QPushButton("刷新")
        refresh_btn.clicked.connect(self.refresh_session_list)
        session_btn_layout.addWidget(refresh_btn)
        
        clear_btn = QPushButton("清除历史")
        clear_btn.clicked.connect(self.clear_all_sessions)
        session_btn_layout.addWidget(clear_btn)
        
        history_layout.addLayout(session_btn_layout)
        history_group.setLayout(history_layout)
        layout.addWidget(history_group)
        
        layout.addStretch()
        panel.setLayout(layout)
        return panel
    
    def create_center_panel(self):
        """创建中间图像显示面板"""
        panel = QGroupBox("图像显示")
        layout = QVBoxLayout()
        
        # 图像显示（1:1比例，带滚动）
        self.dicom_image_label = QLabel("请选择DICOM文件")
        self.dicom_image_label.setAlignment(Qt.AlignCenter)
        self.dicom_image_label.setStyleSheet("""
            QLabel {
                border: 1px solid #ccc;
                background-color: #F5F5F5;
                font-size: 14px;
                color: #666;
            }
        """)
        self.dicom_image_label.setScaledContents(False)
        self.dicom_scroll = QScrollArea()
        self.dicom_scroll.setWidget(self.dicom_image_label)
        self.dicom_scroll.setWidgetResizable(False)
        self.dicom_scroll.setMinimumSize(400, 400)
        layout.addWidget(self.dicom_scroll)
        
        # 图像信息显示
        self.image_info_label = QLabel("")
        self.image_info_label.setAlignment(Qt.AlignCenter)
        self.image_info_label.setStyleSheet("font-size: 12px; color: #666;")
        layout.addWidget(self.image_info_label)
        
        # 导航控件
        nav_layout = QHBoxLayout()
        
        self.prev_image_btn = QPushButton("上一张")
        self.prev_image_btn.clicked.connect(self.show_previous_image)
        self.prev_image_btn.setEnabled(False)
        nav_layout.addWidget(self.prev_image_btn)
        
        self.image_slider = QSlider(Qt.Horizontal)
        self.image_slider.setEnabled(False)
        self.image_slider.valueChanged.connect(self.on_slider_value_changed)
        nav_layout.addWidget(self.image_slider)
        
        self.next_image_btn = QPushButton("下一张")
        self.next_image_btn.clicked.connect(self.show_next_image)
        self.next_image_btn.setEnabled(False)
        nav_layout.addWidget(self.next_image_btn)
        
        layout.addLayout(nav_layout)
        
        # 覆盖层图像显示
        overlay_layout = QHBoxLayout()
        
        # 原始覆盖层
        overlay_group1 = QGroupBox("覆盖层图像")
        overlay_layout1 = QVBoxLayout()
        self.overlay_image_label = QLabel("无覆盖层")
        self.overlay_image_label.setAlignment(Qt.AlignCenter)
        self.overlay_image_label.setStyleSheet("""
            QLabel {
                border: 1px solid #ccc;
                background-color: #F5F5F5;
                font-size: 12px;
                color: #666;
            }
        """)
        self.overlay_image_label.setScaledContents(False)
        self.overlay_scroll = QScrollArea()
        self.overlay_scroll.setWidget(self.overlay_image_label)
        self.overlay_scroll.setWidgetResizable(False)
        self.overlay_scroll.setMinimumSize(200, 200)
        overlay_layout1.addWidget(self.overlay_scroll)
        self.overlay_list_widget = QListWidget()
        overlay_layout1.addWidget(self.overlay_list_widget)
        self.overlay_list_widget.currentRowChanged.connect(self.on_overlay_selected)
        overlay_nav_layout = QHBoxLayout()
        self.prev_overlay_btn = QPushButton("上一覆盖层")
        self.next_overlay_btn = QPushButton("下一覆盖层")
        self.prev_overlay_btn.clicked.connect(self.show_previous_overlay)
        self.next_overlay_btn.clicked.connect(self.show_next_overlay)
        overlay_nav_layout.addWidget(self.prev_overlay_btn)
        overlay_nav_layout.addWidget(self.next_overlay_btn)
        overlay_layout1.addLayout(overlay_nav_layout)
        overlay_group1.setLayout(overlay_layout1)
        overlay_layout.addWidget(overlay_group1)
        
        layout.addLayout(overlay_layout)
        
        panel.setLayout(layout)
        return panel
    
    def on_overlay_selected(self, index):
        try:
            if index <= 0:
                if hasattr(self, 'current_overlay_combined') and self.current_overlay_combined is not None:
                    self._display_image_to_label(self.current_overlay_combined, self.overlay_image_label)
                if hasattr(self, 'current_roi_regions'):
                    text = self._format_roi_results(self.current_roi_regions)
                    self.roi_result_text.setText(text)
                return
            entries = getattr(self, 'current_overlays', [])
            if not entries or index - 1 >= len(entries):
                return
            entry = entries[index - 1]
            path = entry.get('path','')
            if path and os.path.exists(path):
                from PyQt5.QtGui import QPixmap
                pixmap = QPixmap(path)
                self._set_pixmap_1_to_1(self.overlay_image_label, pixmap)
            text = entry.get('text','')
            confidence = entry.get('confidence',0.0)
            roi = [{
                'bbox': {'x':0,'y':0,'width':0,'height':0},
                'area': 0.0,
                'ocr_result': {'text': text, 'confidence': confidence}
            }]
            self.roi_result_text.setText(self._format_roi_results(roi))
        except Exception as e:
            self.logger.error(f"覆盖层选择更新失败: {e}")
    
    def show_previous_overlay(self):
        try:
            current = self.overlay_list_widget.currentRow()
            if current > 0:
                self.overlay_list_widget.setCurrentRow(current - 1)
        except Exception as e:
            self.logger.error(f"上一覆盖层切换失败: {e}")
    
    def show_next_overlay(self):
        try:
            current = self.overlay_list_widget.currentRow()
            count = self.overlay_list_widget.count()
            if current < count - 1 and count > 0:
                self.overlay_list_widget.setCurrentRow(current + 1)
        except Exception as e:
            self.logger.error(f"下一覆盖层切换失败: {e}")
    
    def create_right_panel(self):
        """创建右侧信息面板"""
        panel = QGroupBox("信息面板")
        layout = QVBoxLayout()
        
        # DICOM信息
        dicom_info_group = QGroupBox("DICOM信息")
        dicom_info_layout = QVBoxLayout()
        self.dicom_info_text = QTextEdit()
        self.dicom_info_text.setReadOnly(True)
        self.dicom_info_text.setMaximumHeight(200)
        self.dicom_info_text.setStyleSheet("""
            QTextEdit {
                font-family: monospace;
                font-size: 11px;
                background-color: #f9f9f9;
            }
        """)
        dicom_info_layout.addWidget(self.dicom_info_text)
        dicom_info_group.setLayout(dicom_info_layout)
        layout.addWidget(dicom_info_group)
        
        # ROI结果
        roi_result_group = QGroupBox("ROI识别结果")
        roi_result_layout = QVBoxLayout()
        self.roi_result_text = QTextEdit()
        self.roi_result_text.setReadOnly(True)
        self.roi_result_text.setMaximumHeight(300)
        self.roi_result_text.setStyleSheet("""
            QTextEdit {
                font-family: monospace;
                font-size: 11px;
                background-color: #f9f9f9;
            }
        """)
        roi_result_layout.addWidget(self.roi_result_text)
        roi_result_group.setLayout(roi_result_layout)
        layout.addWidget(roi_result_group)
        
        # 操作按钮
        action_layout = QHBoxLayout()
        
        save_btn = QPushButton("保存结果")
        save_btn.clicked.connect(self.save_roi_results)
        action_layout.addWidget(save_btn)
        
        export_btn = QPushButton("导出数据")
        export_btn.clicked.connect(self.export_roi_data)
        action_layout.addWidget(export_btn)
        
        layout.addLayout(action_layout)
        layout.addStretch()
        panel.setLayout(layout)
        return panel
    
    def start_roi_processing(self):
        """开始ROI处理"""
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
            
        # 禁用处理按钮
        self.process_btn.setEnabled(False)
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)
        
        auto_save = self.auto_save_checkbox.isChecked()
        debug_mode = self.debug_mode_checkbox.isChecked()
        
        if self.batch_processing and len(self.selected_files) > 1:
            # 批量处理模式
            self.status_label.setText(f"正在批量处理 {len(self.selected_files)} 个文件...")
            self._process_files_batch(auto_save, debug_mode)
        else:
            # 单文件处理模式
            file_path = self.selected_files[0]
            self.selected_file_path = file_path
            
            # 创建并启动处理线程
            self.processing_thread = DicomProcessingThread(
                file_path, 
                self.selected_user_id, 
                auto_save,
                debug_mode,
                project_id=self.current_project_id
            )
            
            self.processing_thread.progress_updated.connect(self.update_progress)
            self.processing_thread.processing_finished.connect(self.on_processing_finished)
            self.processing_thread.error_occurred.connect(self.on_processing_error)
            
            self.processing_thread.start()
            self.status_label.setText("正在处理DICOM文件...")
    
    def _process_files_batch(self, auto_save, debug_mode):
        """批量处理文件"""
        self.batch_results = []
        self.current_batch_index = 0
        self.total_batch_files = len(self.selected_files)
        self.batch_auto_save = auto_save
        self.batch_debug_mode = debug_mode
        self._process_next_batch_file()
    
    def _process_next_batch_file(self):
        """处理下一个批量文件"""
        if self.current_batch_index >= self.total_batch_files:
            # 所有文件处理完成
            self._on_batch_processing_finished()
            return
        
        file_path = self.selected_files[self.current_batch_index]
        file_name = os.path.basename(file_path)
        self.status_label.setText(f"正在处理 ({self.current_batch_index + 1}/{self.total_batch_files}): {file_name}")
        
        # 更新总体进度
        overall_progress = int((self.current_batch_index / self.total_batch_files) * 100)
        self.progress_bar.setValue(overall_progress)
        
        # 启动处理线程
        self.processing_thread = DicomProcessingThread(
            file_path, 
            self.selected_user_id, 
            self.batch_auto_save,
            self.batch_debug_mode,
            project_id=self.current_project_id
        )
        self.processing_thread.progress_updated.connect(self._update_batch_progress)
        self.processing_thread.processing_finished.connect(self._on_batch_file_finished)
        self.processing_thread.error_occurred.connect(self._on_batch_file_error)
        self.processing_thread.start()
    
    def _update_batch_progress(self, progress):
        """更新批量处理进度"""
        # 计算当前文件在总体进度中的贡献
        file_progress = progress / self.total_batch_files
        base_progress = (self.current_batch_index / self.total_batch_files) * 100
        total_progress = int(base_progress + file_progress)
        self.progress_bar.setValue(min(total_progress, 100))
    
    def _on_batch_file_finished(self, result):
        """单个批量文件处理完成"""
        file_path = self.selected_files[self.current_batch_index]
        self.batch_results.append({
            'file_path': file_path,
            'result': result,
            'status': 'success'
        })
        
        self.current_batch_index += 1
        
        # 更新文件列表状态
        self._update_file_info_display()
        
        # 处理下一个文件
        self._process_next_batch_file()
    
    def _on_batch_file_error(self, error_msg):
        """单个批量文件处理错误"""
        file_path = self.selected_files[self.current_batch_index]
        self.batch_results.append({
            'file_path': file_path,
            'error': error_msg,
            'status': 'error'
        })
        
        self.current_batch_index += 1
        
        # 更新文件列表状态
        self._update_file_info_display()
        
        # 继续处理下一个文件
        self._process_next_batch_file()
    
    def _on_batch_processing_finished(self):
        """批量处理完成"""
        self.progress_bar.setVisible(False)
        self.process_btn.setEnabled(True)
        
        # 统计结果
        success_count = sum(1 for r in self.batch_results if r['status'] == 'success')
        error_count = sum(1 for r in self.batch_results if r['status'] == 'error')
        
        self.status_label.setText(f"批量处理完成: 成功 {success_count} 个, 失败 {error_count} 个")
        
        # 显示第一个成功的结果
        for result in self.batch_results:
            if result['status'] == 'success':
                self.on_processing_finished(result['result'])
                break
        
        self.batch_processing = False
        self._update_file_info_display()
        self.process_btn.setEnabled(True)
        self.clear_selection_btn.setEnabled(True)
        
    def update_progress(self, value):
        """更新进度条"""
        self.progress_bar.setValue(value)
        
    def on_processing_finished(self, result):
        """处理完成回调"""
        self.progress_bar.setVisible(False)
        self.process_btn.setEnabled(True)
        
        # 设置当前会话ID
        self.current_session_id = result.get('session_id')
        
        # 显示DICOM图像
        image_array = result.get('image_array')
        if image_array is not None:
            self._display_dicom_image(image_array)
            # 保存当前图像数组
            self.current_image_array = image_array
        else:
            # 尝试从路径加载DICOM图像 (针对覆盖层文件夹处理模式)
            dicom_file_path = result.get('dicom_file_path')
            if dicom_file_path and os.path.exists(dicom_file_path):
                self._load_and_display_dicom_image(dicom_file_path)
            else:
                self.dicom_image_label.setText("未找到对应的DICOM文件")
                self.current_image_array = None
        
        overlay_images = result.get('overlay_images', {})
        if overlay_images:
            if overlay_images.get('overlay') is not None:
                self._display_image_to_label(overlay_images.get('overlay'), self.overlay_image_label)
            else:
                self.overlay_image_label.setText("无覆盖层")
            
            overlays_meta = overlay_images.get('overlays', [])
            if hasattr(self, 'overlay_list_widget'):
                self.overlay_list_widget.clear()
                if overlay_images.get('overlay') is not None:
                    self.overlay_list_widget.addItem("合并视图")
                for entry in overlays_meta:
                    name = os.path.basename(entry.get('path',''))
                    self.overlay_list_widget.addItem(f"{entry.get('group','----')} - {name}")
                if self.overlay_list_widget.count() > 0:
                    self.overlay_list_widget.setCurrentRow(0)
                self.current_overlays = overlays_meta
                self.current_overlay_combined = overlay_images.get('overlay')
            else:
                self.current_overlays = overlays_meta
                self.current_overlay_combined = overlay_images.get('overlay')
                
            # 清空第三个标签，或者如果需要可以显示其他内容
            if hasattr(self, 'no_overlay_image_label'):
                self.no_overlay_image_label.clear()
                self.no_overlay_image_label.setText("原始图像(参照左侧)")
            
        # 设置DICOM导航功能
        if hasattr(self, 'selected_file_path') and self.selected_file_path:
            self._setup_dicom_navigation(self.selected_file_path)
        
        # 保存DICOM信息用于物理测量计算
        self.current_dicom_info = result.get('dicom_info', {})
        
        # 显示DICOM信息
        dicom_info = result.get('dicom_info', {})
        info_text = self._format_dicom_info(dicom_info)
        self.dicom_info_text.setText(info_text)
        
        # 显示ROI结果
        roi_regions = result.get('roi_regions', [])
        roi_text = self._format_roi_results(roi_regions)
        self.roi_result_text.setText(roi_text)
        self.current_roi_regions = roi_regions
        
        # 更新状态
        if overlay_images.get('overlay') is not None:
            self.status_label.setText("处理完成 - 已识别覆盖层文本")
        else:
            self.status_label.setText("处理完成 - 未找到覆盖层")
        
        # 刷新会话列表
        self.refresh_session_list()
        
        # 发布处理完成事件
        self.event_bus.publish(EventNames.PROCESSING_COMPLETED, {
            'module': 'dicom_roi',
            'session_id': self.current_session_id,
            'file_path': getattr(self, 'selected_file_path', None),
            'user_id': self.selected_user_id,
            'timestamp': time.time()
        })
        
    def on_processing_error(self, error_message):
        """处理错误回调"""
        self.progress_bar.setVisible(False)
        self.process_btn.setEnabled(True)
        
        # 显示用户友好的错误信息
        user_friendly_message = self._get_user_friendly_error_message(error_message)
        self.status_label.setText(f"处理失败: {user_friendly_message}")
        
        # 显示详细错误信息的对话框
        self._show_error_dialog("DICOM ROI处理错误", error_message, user_friendly_message)
        
        # 记录错误日志
        self.logger.error(f"DICOM ROI处理失败: {error_message}")
        
    def _get_user_friendly_error_message(self, error_message):
        """获取用户友好的错误信息"""
        error_lower = error_message.lower()
        
        if "file not found" in error_lower or "no such file" in error_lower:
            return "文件未找到，请检查文件路径是否正确"
        elif "permission denied" in error_lower:
            return "文件访问权限不足，请检查文件权限"
        elif "invalid dicom" in error_lower or "not a dicom" in error_lower:
            return "无效的DICOM文件格式，请选择正确的DICOM文件"
        elif "memory" in error_lower or "out of memory" in error_lower:
            return "内存不足，请尝试处理较小的文件或关闭其他程序"
        elif "timeout" in error_lower:
            return "处理超时，请尝试处理较小的文件或检查网络连接"
        elif "roi detection failed" in error_lower:
            return "ROI检测失败，请尝试调整检测参数或选择其他检测方法"
        else:
            return "处理过程中发生未知错误，请检查文件格式和参数设置"
            
    def _show_error_dialog(self, title, technical_error, user_friendly_error):
        """显示错误对话框"""
        from PyQt5.QtWidgets import QMessageBox, QTextEdit
        
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
            return "• 检查文件路径是否正确\n• 确认文件是否存在\n• 尝试重新选择文件"
        elif "permission denied" in error_lower:
            return "• 以管理员身份运行程序\n• 检查文件是否被其他程序占用\n• 确认对文件夹有读写权限"
        elif "invalid dicom" in error_lower:
            return "• 确认文件是标准DICOM格式\n• 尝试使用DICOM查看器验证文件\n• 检查文件是否损坏"
        elif "memory" in error_lower:
            return "• 关闭其他占用内存的程序\n• 尝试处理较小的文件\n• 重启应用程序"
        elif "roi detection failed" in error_lower:
            return "• 调整检测阈值参数\n• 尝试不同的检测方法\n• 启用图像预处理选项"
        else:
            return "• 检查文件格式和完整性\n• 重新启动应用程序\n• 联系技术支持"
            
    def refresh_session_list(self):
        """刷新会话列表"""
        self.logger.info("开始刷新DICOM会话列表")
        
        if not self.selected_user_id:
            self.logger.info("未选择用户ID，清空会话列表")
            self.session_list.clear()
            return
            
        try:
            self.logger.info(f"为用户ID {self.selected_user_id} 刷新会话列表")
            
            # 获取当前项目ID - 优先使用组件内部状态
            current_project_id = self.current_project_id
            if not current_project_id:
                from src.services.project_service import ProjectService
                project_service = ProjectService()
                current_project_id = project_service.get_current_project_id() or "default"
            
            self.logger.info(f"使用项目ID: {current_project_id}")
            
            # 获取会话数据
            sessions = self.dicom_service.get_dicom_sessions(self.selected_user_id, project_id=current_project_id)
            self.logger.info(f"从数据库获取到 {len(sessions)} 个会话")
            
            # 清空现有列表
            old_count = self.session_list.count()
            self.session_list.clear()
            self.logger.info(f"清空UI列表，原有 {old_count} 个项目")
            
            # 添加新的会话项目
            for i, session in enumerate(sessions):
                item_text = f"会话 {session['session_id']} - {session['created_at'][:19]}"
                self.session_list.addItem(item_text)
                self.logger.debug(f"添加会话项目 {i+1}: {item_text}")
            
            new_count = self.session_list.count()
            self.logger.info(f"会话列表刷新完成，UI显示 {new_count} 个项目")
            
            # 验证UI更新
            if len(sessions) != new_count:
                self.logger.error(f"UI更新异常: 数据库有 {len(sessions)} 个会话，但UI显示 {new_count} 个")
            else:
                self.logger.info("UI更新正常，数据库和UI项目数量一致")
                
        except Exception as e:
            self.logger.error(f"刷新会话列表失败: {e}")
            
    def load_session(self, item):
        """加载选中的会话"""
        try:
            # 从item文本中提取session_id
            item_text = item.text()
            self.logger.debug(f"load_session: item_text = {item_text}")
            
            # 使用更健壮的解析方式
            if "会话" in item_text and "-" in item_text:
                parts = item_text.split(' ')
                if len(parts) > 1:
                    session_id = parts[1]
                else:
                    self.logger.error(f"无法解析会话ID: {item_text}")
                    return
            else:
                # 尝试直接使用整个文本作为ID，或者使用其他分隔符
                session_id = item_text.strip()
                
            self.logger.debug(f"load_session: session_id = {session_id}")
            
            # 获取会话详情
            session_details = self.dicom_service.get_dicom_session_details(session_id)
            self.logger.debug(f"load_session获取到的session_details类型: {type(session_details)}, 内容: {session_details}")
            
            if not session_details:
                self.status_label.setText("无法加载会话详情")
                return
                
            # 确保session_details是字典类型
            if not isinstance(session_details, dict):
                self.logger.error(f"会话详情数据类型错误: {type(session_details)}, 内容: {session_details}")
                self.status_label.setText("会话详情数据格式错误")
                return
                
            self.current_session_id = session_id
            
            # 获取文件路径并加载DICOM图像
            self.logger.debug(f"load_session: 准备获取file_path")
            file_path = session_details.get('file_path')
            self.logger.debug(f"load_session: file_path = {file_path}")
            if file_path and os.path.exists(file_path):
                self._load_and_display_dicom_image(file_path)
            else:
                self.dicom_image_label.setText("DICOM文件不存在")
            
            # 显示DICOM信息
            self.logger.debug(f"load_session: 准备获取dicom_info")
            dicom_info = session_details.get('dicom_info', {})
            self.logger.debug(f"load_session: dicom_info类型 = {type(dicom_info)}")
            if isinstance(dicom_info, str):
                import json
                dicom_info = json.loads(dicom_info)
            
            # 设置当前DICOM信息，这样_calculate_physical_measurements方法才能正确计算物理测量信息
            self.current_dicom_info = dicom_info
                
            info_text = self._format_dicom_info(dicom_info)
            self.dicom_info_text.setText(info_text)
            
            # 显示ROI结果
            self.logger.debug(f"load_session: 准备获取roi_data")
            roi_data = session_details.get('roi_data', [])
            self.logger.debug(f"load_session: roi_data类型 = {type(roi_data)}, 长度 = {len(roi_data)}")
            
            if isinstance(roi_data, str):
                import json
                roi_data = json.loads(roi_data)
            
            # 关键修复：设置 current_roi_regions，确保切换视图时能正确显示文本
            self.current_roi_regions = roi_data
            
            roi_text = self._format_roi_results(roi_data)
            self.roi_result_text.setText(roi_text)
            
            # 恢复可视化结果
            self._restore_visualization(roi_data)
            
            self.status_label.setText(f"已加载会话: {session_id}")
            
        except Exception as e:
            self.logger.error(f"加载会话失败: {e}")
            self.status_label.setText(f"加载会话失败: {str(e)}")
            
    def _restore_visualization(self, roi_data):
        """恢复ROI可视化"""
        if not hasattr(self, 'current_image_array') or self.current_image_array is None:
            return

        try:
            height, width = self.current_image_array.shape[:2]
            combined_mask = np.zeros((height, width), dtype=np.uint8)
            
            # 清空列表
            if hasattr(self, 'overlay_list_widget'):
                self.overlay_list_widget.clear()
                self.overlay_list_widget.addItem("合并视图")
            
            overlay_entries = []
            
            for i, roi in enumerate(roi_data):
                # 尝试从文件加载覆盖层
                properties = roi.get('properties', {})
                image_path = properties.get('image_path')
                mask = None
                
                if image_path and os.path.exists(image_path):
                    try:
                        mask = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
                        if mask is not None and mask.shape != (height, width):
                            mask = cv2.resize(mask, (width, height), interpolation=cv2.INTER_NEAREST)
                    except Exception as e:
                        self.logger.warning(f"加载ROI图像失败: {e}")
                
                # 如果无法加载图像，尝试从坐标绘制
                if mask is None and 'coordinates' in roi:
                    try:
                        coords = roi['coordinates']
                        if coords:
                            # 确保格式正确 (list of [x, y])
                            if isinstance(coords, str):
                                import json
                                coords = json.loads(coords)
                            
                            pts = np.array(coords, dtype=np.int32)
                            mask = np.zeros((height, width), dtype=np.uint8)
                            cv2.fillPoly(mask, [pts], 255)
                    except Exception as e:
                        self.logger.warning(f"绘制ROI轮廓失败: {e}")

                if mask is not None:
                    combined_mask = cv2.bitwise_or(combined_mask, mask)
                    
                    # 添加到列表
                    source = roi.get('source', 'unknown')
                    ocr_text = roi.get('ocr_text', '')
                    item_text = f"{source}"
                    if ocr_text:
                        item_text += f" - {ocr_text[:10]}..."
                    
                    if hasattr(self, 'overlay_list_widget'):
                        self.overlay_list_widget.addItem(item_text)
                    
                    overlay_entries.append({
                        'group': source,
                        'path': image_path,
                        'text': ocr_text,
                        'confidence': roi.get('ocr_confidence', 0.0)
                    })

            # 显示合并后的覆盖层
            if np.any(combined_mask):
                overlay_display = cv2.cvtColor(combined_mask, cv2.COLOR_GRAY2RGB)
                overlay_display[combined_mask > 0] = [255, 0, 0]  # Red color
                self._display_image_to_label(overlay_display, self.overlay_image_label)
                self.current_overlay_combined = overlay_display
            else:
                self.overlay_image_label.setText("无覆盖层")
                self.current_overlay_combined = None
            
            self.current_overlays = overlay_entries
            
            # 选中第一项
            if hasattr(self, 'overlay_list_widget') and self.overlay_list_widget.count() > 0:
                self.overlay_list_widget.setCurrentRow(0)
                
        except Exception as e:
            self.logger.error(f"恢复可视化失败: {e}")

    def select_dicom_file(self):
        """选择DICOM文件"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, 
            "选择DICOM文件", 
            "", 
            "DICOM Files (*.dcm *.dicom);;All Files (*.*)"
        )
        
        if file_path:
            self.selected_files = [file_path]
            self.batch_processing = False
            self._update_file_info_display()
            self.process_btn.setEnabled(True)
            self.clear_selection_btn.setEnabled(True)
            self.status_label.setText(f"已选择文件: {os.path.basename(file_path)}")
    
    def save_roi_results(self):
        """保存ROI结果"""
        if not self.current_session_id:
            self.status_label.setText("没有可保存的结果")
            return
            
        try:
            # 获取当前会话的ROI数据
            session_details = self.dicom_service.get_dicom_session_details(self.current_session_id)
            if not session_details:
                self.status_label.setText("无法获取会话数据")
                return
                
            roi_data = session_details.get('roi_data', [])
            
            # 选择保存路径
            file_path, _ = QFileDialog.getSaveFileName(
                self,
                "保存ROI结果",
                f"roi_results_{self.current_session_id}.json",
                "JSON Files (*.json)"
            )
            
            if file_path:
                import json
                with open(file_path, 'w', encoding='utf-8') as f:
                    json.dump(roi_data, f, ensure_ascii=False, indent=2)
                
                self.status_label.setText(f"ROI结果已保存到: {file_path}")
                
        except Exception as e:
            self.logger.error(f"保存ROI结果失败: {e}")
            self.status_label.setText(f"保存失败: {str(e)}")
    
    def export_roi_data(self):
        """导出ROI数据"""
        if not self.current_session_id:
            self.status_label.setText("没有可导出的数据")
            return
            
        try:
            # 获取当前会话的完整数据
            session_details = self.dicom_service.get_dicom_session_details(self.current_session_id)
            if not session_details:
                self.status_label.setText("无法获取会话数据")
                return
                
            # 选择导出路径
            file_path, _ = QFileDialog.getSaveFileName(
                self,
                "导出ROI数据",
                f"dicom_roi_export_{self.current_session_id}.json",
                "JSON Files (*.json)"
            )
            
            if file_path:
                import json
                # 构建完整的导出数据
                export_data = {
                    'session_id': self.current_session_id,
                    'file_path': session_details.get('file_path'),
                    'user_id': session_details.get('user_id'),
                    'created_at': session_details.get('created_at'),
                    'dicom_info': session_details.get('dicom_info', {}),
                    'roi_data': session_details.get('roi_data', [])
                }
                
                with open(file_path, 'w', encoding='utf-8') as f:
                    json.dump(export_data, f, ensure_ascii=False, indent=2)
                
                self.status_label.setText(f"ROI数据已导出到: {file_path}")
                
        except Exception as e:
            self.logger.error(f"导出ROI数据失败: {e}")
            self.status_label.setText(f"导出失败: {str(e)}")
                
    def dragEnterEvent(self, event: QDragEnterEvent):
        """拖拽进入事件"""
        if event.mimeData().hasUrls():
            # 检查是否包含有效的DICOM文件
            valid_files = self._get_valid_dicom_files_from_urls(event.mimeData().urls())
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
            }
            QFrame:hover {
                border-color: #007acc;
                background-color: #e6f3ff;
            }
        """)
    
    def dropEvent(self, event: QDropEvent):
        """拖拽放下事件"""
        if event.mimeData().hasUrls():
            valid_files = self._get_valid_dicom_files_from_urls(event.mimeData().urls())
            if valid_files:
                self.selected_files = valid_files
                self._update_file_info_display()
                self.process_btn.setEnabled(True)
                self.clear_selection_btn.setEnabled(True)
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
    
    def _get_valid_dicom_files_from_urls(self, urls):
        """从URL列表中获取有效的DICOM文件"""
        valid_files = []
        
        for url in urls:
            file_path = url.toLocalFile()
            # 使用DicomService的智能搜索功能
            found_files = self.dicom_service.find_dicom_files(file_path, recursive=True)
            valid_files.extend(found_files)
        
        return valid_files
    
    def select_dicom_folder(self):
        """选择DICOM文件夹"""
        folder_path = QFileDialog.getExistingDirectory(self, "选择DICOM文件夹")
        
        if folder_path:
            # 使用DicomService的智能搜索功能
            valid_files = self.dicom_service.find_dicom_files(folder_path, recursive=True)
            
            if valid_files:
                self.selected_files = valid_files
                self.batch_processing = True
                self._update_file_info_display()
                self.process_btn.setEnabled(True)
                self.clear_selection_btn.setEnabled(True)
                self.status_label.setText(f"在文件夹中找到 {len(valid_files)} 个DICOM文件")
            else:
                self.status_label.setText("所选文件夹中未找到DICOM文件")
    
    def select_overlay_folder(self):
        folder_path = QFileDialog.getExistingDirectory(self, "选择覆盖层文件夹")
        if folder_path:
            self.overlay_folder_path = folder_path
            self.overlay_folder_label.setText(f"已选择: {folder_path}")
            self.process_overlay_folder_btn.setEnabled(True)
        else:
            self.overlay_folder_label.setText("未选择覆盖层文件夹")
            self.process_overlay_folder_btn.setEnabled(False)
    
    def start_overlay_folder_processing(self):
        if not hasattr(self, 'overlay_folder_path'):
            self.status_label.setText("未选择覆盖层文件夹")
            return
        
        # 初始化批量处理状态
        self.batch_results = []
        self.selected_files = [] # 清空已选文件，因为这里是处理覆盖层文件夹
        self.file_list_widget.clear()
        self.batch_processing = True
        
        self.process_overlay_folder_btn.setEnabled(False)
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)
        
        self.processing_thread = OverlayFolderProcessingThread(self.overlay_folder_path, self.selected_user_id)
        self.processing_thread.progress_updated.connect(self.update_progress)
        self.processing_thread.processing_finished.connect(self.on_overlay_folder_processing_finished)
        self.processing_thread.error_occurred.connect(self.on_processing_error)
        self.processing_thread.start()
        self.status_label.setText("正在处理覆盖层文件夹...")
        
    def on_overlay_folder_processing_finished(self, result):
        """处理覆盖层文件夹中单个结果完成"""
        # 如果是最终完成信号（没有数据）
        if not result:
            self.progress_bar.setVisible(False)
            self.process_overlay_folder_btn.setEnabled(True)
            self.status_label.setText(f"覆盖层文件夹处理完成: 共 {len(self.batch_results)} 个项目")
            return

        # 保存结果
        file_identifier = result.get('file_identifier', 'unknown')
        self.batch_results.append({
            'file_path': file_identifier, # 使用标识符作为路径
            'result': result,
            'status': 'success'
        })
        
        # 尝试自动保存到数据库
        if self.auto_save_checkbox.isChecked():
            dicom_file_path = result.get('dicom_file_path')
            if dicom_file_path and os.path.exists(dicom_file_path):
                try:
                    current_project_id = self.current_project_id
                    if not current_project_id:
                        from src.services.project_service import ProjectService
                        project_service = ProjectService()
                        current_project_id = project_service.get_current_project_id() or "default"
                    
                    # 准备数据
                    dicom_info = result.get('dicom_info', {})
                    roi_regions = result.get('roi_regions', [])
                    
                    # 创建会话
                    session_id = self.dicom_service.create_dicom_session(
                        user_id=self.selected_user_id or 'default',
                        file_path=dicom_file_path,
                        dicom_info=dicom_info,
                        roi_data=roi_regions,
                        project_id=current_project_id
                    )
                    
                    if session_id:
                        result['session_id'] = session_id
                        self.logger.info(f"自动保存成功: {file_identifier} -> Session {session_id}")
                    else:
                        self.logger.warning(f"自动保存失败: {file_identifier}")
                        
                except Exception as e:
                    self.logger.error(f"自动保存异常: {e}")
            else:
                self.logger.warning(f"无法自动保存 {file_identifier}: 未找到对应的DICOM文件")
        
        # 添加到文件列表
        item = QListWidgetItem(f"{file_identifier} [已处理]")
        item.setData(Qt.UserRole, file_identifier)
        self.file_list_widget.addItem(item)
        
        # 如果是第一个结果，自动显示
        if len(self.batch_results) == 1:
            self.on_processing_finished(result)
            self.selected_file_path = file_identifier # 临时设置以便逻辑兼容
    
    def clear_file_selection(self):
        """清空文件选择"""
        self.selected_files = []
        self.batch_processing = False
        if hasattr(self, 'selected_file_path'):
            delattr(self, 'selected_file_path')
        
        self.file_list_widget.clear()
        self.process_btn.setEnabled(False)
        self.clear_selection_btn.setEnabled(False)
        self.status_label.setText("已清空文件选择")
    
    def _update_file_info_display(self):
        """更新文件信息显示"""
        self.file_list_widget.clear()
        
        if not self.selected_files:
            return
        
        for file_path in self.selected_files:
            file_name = os.path.basename(file_path)
            # 检查是否有处理结果状态
            status_text = ""
            if hasattr(self, 'batch_results'):
                for res in self.batch_results:
                    if res.get('file_path') == file_path:
                        if res.get('status') == 'success':
                            status_text = " [已处理]"
                        elif res.get('status') == 'error':
                            status_text = " [失败]"
                        break
            
            item = QListWidgetItem(f"{file_name}{status_text}")
            item.setData(Qt.UserRole, file_path)
            self.file_list_widget.addItem(item)
            
        if len(self.selected_files) == 1:
            file_name = os.path.basename(self.selected_files[0])
            self.status_label.setText(f"已选择文件: {file_name}")
        else:
            self.status_label.setText(f"已选择 {len(self.selected_files)} 个文件，准备批量处理")
            
    def on_file_list_item_clicked(self, item):
        """处理文件列表点击事件"""
        file_path = item.data(Qt.UserRole)
        if not file_path:
            return
            
        # 查找是否有已处理的结果
        if hasattr(self, 'batch_results'):
            for res in self.batch_results:
                if res.get('file_path') == file_path:
                    if res.get('status') == 'success':
                        self.on_processing_finished(res.get('result'))
                        return
                    elif res.get('status') == 'error':
                        self.status_label.setText(f"该文件处理失败: {res.get('error')}")
                        return
        
        # 如果没有结果，可能是还没处理或者单文件模式，尝试直接显示
        self.selected_file_path = file_path
        
        # 尝试清空之前的显示，避免误导
        self.dicom_image_label.setText("请点击处理按钮进行分析")
        self.roi_result_text.clear()
        if hasattr(self, 'overlay_list_widget'):
            self.overlay_list_widget.clear()
        self.overlay_image_label.clear()
        self.overlay_image_label.setText("覆盖层预览")
        
        # 这里可以添加逻辑去加载未处理的DICOM显示预览
        # 目前先简单更新选中状态
        self.status_label.setText(f"已选择: {os.path.basename(file_path)}")
    
    def show_previous_image(self):
        """显示上一张图像"""
        if self.dicom_files_list and self.current_image_index > 0:
            self.current_image_index -= 1
            self._update_image_display()
    
    def show_next_image(self):
        """显示下一张图像"""
        if self.dicom_files_list and self.current_image_index < len(self.dicom_files_list) - 1:
            self.current_image_index += 1
            self._update_image_display()
    
    def on_slider_value_changed(self, value):
        """滑动条值变化事件"""
        if self.dicom_files_list and 0 <= value < len(self.dicom_files_list):
            self.current_image_index = value
            self._update_image_display()
    
    def _update_image_display(self):
        """更新图像显示"""
        if not self.dicom_files_list or self.current_image_index >= len(self.dicom_files_list):
            return
        
        try:
            current_file = self.dicom_files_list[self.current_image_index]
            self._load_and_display_dicom_image(current_file)
            
            try:
                self.current_image_index = self.dicom_files_list.index(file_path)
            except ValueError:
                self.current_image_index = 0
            
            # 更新滑动条范围
            if len(self.dicom_files_list) > 1:
                self.image_slider.setMaximum(len(self.dicom_files_list) - 1)
                self.image_slider.setValue(self.current_image_index)
                self.image_slider.setEnabled(True)
                
                # 启用导航按钮
                self.prev_image_btn.setEnabled(self.current_image_index > 0)
                self.next_image_btn.setEnabled(self.current_image_index < len(self.dicom_files_list) - 1)
                
                # 更新图像信息
                file_name = os.path.basename(file_path)
                self.image_info_label.setText(f"{self.current_image_index + 1}/{len(self.dicom_files_list)}: {file_name}")
            else:
                # 只有一个文件时禁用导航
                self.image_slider.setEnabled(False)
                self.prev_image_btn.setEnabled(False)
                self.next_image_btn.setEnabled(False)
                file_name = os.path.basename(file_path)
                self.image_info_label.setText(f"1/1: {file_name}")
                
        except Exception as e:
            self.logger.error(f"设置DICOM导航失败: {e}")
            # 禁用导航控件
            self.image_slider.setEnabled(False)
            self.prev_image_btn.setEnabled(False)
            self.next_image_btn.setEnabled(False)
            self.image_info_label.setText("导航设置失败")
    
    def clear_all_sessions(self):
        """清除所有历史会话"""
        self.logger.info("开始清除所有DICOM历史会话")
        
        if not self.selected_user_id:
            self.logger.warning("清除会话失败: 未选择用户ID")
            QMessageBox.warning(self, "警告", "请先在主界面选择一个项目")
            return
        
        self.logger.info(f"当前选择的用户ID: {self.selected_user_id}")
        
        # 获取当前项目ID用于调试
        try:
            current_project_id = self.current_project_id
            if not current_project_id:
                from src.services.project_service import ProjectService
                project_service = ProjectService()
                current_project_id = project_service.get_current_project_id() or "default"
            self.logger.info(f"当前项目ID: {current_project_id}")
        except Exception as e:
            self.logger.error(f"获取项目ID失败: {e}")
            current_project_id = "default"
        
        # 清除前先检查现有会话数量
        try:
            existing_sessions = self.dicom_service.get_dicom_sessions(self.selected_user_id, project_id=current_project_id)
            self.logger.info(f"清除前现有会话数量: {len(existing_sessions)}")
            for session in existing_sessions:
                self.logger.debug(f"现有会话: {session['session_id']} - {session['created_at']}")
        except Exception as e:
            self.logger.error(f"获取现有会话失败: {e}")
        
        # 确认对话框
        reply = QMessageBox.question(
            self, 
            "确认清除", 
            "确定要清除所有DICOM历史会话吗？此操作不可撤销。",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No
        )
        
        if reply == QMessageBox.Yes:
            try:
                self.logger.info("用户确认清除，开始执行清除操作")
                
                # 调用DicomService的清除方法，传递项目ID
                self.logger.info(f"调用服务清除会话: user_id={self.selected_user_id}, project_id={current_project_id}")
                success = self.dicom_service.clear_all_dicom_sessions(self.selected_user_id, current_project_id)
                self.logger.info(f"清除操作结果: {success}")
                
                if success:
                    self.logger.info("清除成功，开始刷新会话列表")
                    self.status_label.setText("清除成功，正在刷新列表...")
                    
                    # 刷新会话列表
                    self.refresh_session_list()
                    
                    # 验证清除结果
                    try:
                        remaining_sessions = self.dicom_service.get_dicom_sessions(self.selected_user_id, project_id=current_project_id)
                        self.logger.info(f"清除后剩余会话数量: {len(remaining_sessions)}")
                        
                        if len(remaining_sessions) == 0:
                            self.logger.info("会话清除完全成功")
                        else:
                            self.logger.warning(f"清除后仍有 {len(remaining_sessions)} 个会话残留")
                            for session in remaining_sessions:
                                self.logger.warning(f"残留会话: {session['session_id']} - {session['created_at']}")
                    except Exception as e:
                        self.logger.error(f"验证清除结果失败: {e}")
                    
                    if len(remaining_sessions) == 0:
                        QMessageBox.information(self, "成功", "已成功清除所有DICOM历史会话")
                        self.status_label.setText("已清除所有历史会话")
                    else:
                        QMessageBox.warning(self, "警告", f"清除操作已执行，但仍有 {len(remaining_sessions)} 个会话未能清除。请查看日志或重试。")
                        self.status_label.setText(f"清除不完全: 剩 {len(remaining_sessions)} 个")
                else:
                    self.logger.error("清除操作返回失败")
                    QMessageBox.warning(self, "失败", "清除历史会话失败")
                    
            except Exception as e:
                self.logger.error(f"清除历史会话异常: {e}", exc_info=True)
                QMessageBox.critical(self, "错误", f"清除历史会话时发生错误: {str(e)}")
        else:
            self.logger.info("用户取消清除操作")
    
    def setup_event_connections(self):
        """设置事件总线连接"""
        try:
            # 订阅用户ID选择事件
            self.event_bus.subscribe(EventNames.USER_ID_SELECTED, self.on_user_id_selected)
            self.logger.info("DICOM ROI组件已订阅用户ID选择事件")
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
            self.logger.info(f"DICOM ROI组件接收到用户ID选择: {user_id} ({display_name})")
            
            # 更新UI状态
            self.update_ui_state()
            
            # 刷新会话列表以显示该用户ID的历史记录
            self.refresh_session_list()
            
            # 发布数据同步事件
            self.event_bus.publish(EventNames.DATA_SYNC_REQUESTED, {
                'module': 'dicom_roi',
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
    
    def set_current_project(self, project_id):
        """设置当前项目并刷新数据
        
        Args:
            project_id: 项目ID
        """
        try:
            # 如果项目ID没有变化，则不重置UI状态，保留当前的处理结果
            if hasattr(self, 'current_project_id') and self.current_project_id == project_id:
                self.logger.info(f"项目ID未变更 ({project_id})，保留当前状态")
                return

            self.current_project_id = project_id
            self.logger.info(f"DICOM ROI界面切换到项目: {project_id}")
            
            # 刷新会话列表以显示当前项目的数据
            self.refresh_session_list()
            
            # 清空当前显示的内容
            self.dicom_image_label.clear()
            self.dicom_image_label.setText("请选择DICOM文件")
            self.dicom_info_text.clear()
            self.roi_result_text.clear()
            
            # 重置文件选择
            self.selected_files = []
            self._update_file_info_display()
            
            # 清除批量处理结果
            if hasattr(self, 'batch_results'):
                self.batch_results = []
            
            # 更新UI状态
            self.update_ui_state()
            
            self.status_label.setText(f"已切换到项目: {project_id}")
            
        except Exception as e:
            self.logger.error(f"设置当前项目失败: {e}")
            self.status_label.setText(f"项目切换失败: {str(e)}")
            
    def _display_dicom_image(self, image_array):
        """显示DICOM图像"""
        try:
            # 转换图像格式
            if len(image_array.shape) == 2:
                # 灰度图像
                height, width = image_array.shape
                bytes_per_line = width
                q_image = QImage(image_array.data, width, height, bytes_per_line, QImage.Format_Grayscale8)
            else:
                # 彩色图像
                height, width, channel = image_array.shape
                bytes_per_line = channel * width
                q_image = QImage(image_array.data, width, height, bytes_per_line, QImage.Format_RGB888)
            
            # 转换为QPixmap并显示
            pixmap = QPixmap.fromImage(q_image)
            self._set_pixmap_1_to_1(self.dicom_image_label, pixmap)
            
        except Exception as e:
            self.logger.error(f"显示DICOM图像失败: {e}")
            self.dicom_image_label.setText(f"显示图像失败: {str(e)}")
    
    def _display_image_to_label(self, image_array, label):
        """将图像显示到指定标签"""
        try:
            # 转换图像格式
            if len(image_array.shape) == 2:
                # 灰度图像
                height, width = image_array.shape
                bytes_per_line = width
                q_image = QImage(image_array.data, width, height, bytes_per_line, QImage.Format_Grayscale8)
            else:
                # 彩色图像
                height, width, channel = image_array.shape
                bytes_per_line = channel * width
                q_image = QImage(image_array.data, width, height, bytes_per_line, QImage.Format_RGB888)
            
            # 转换为QPixmap并显示
            pixmap = QPixmap.fromImage(q_image)
            self._set_pixmap_1_to_1(label, pixmap)
            
        except Exception as e:
            self.logger.error(f"显示图像到标签失败: {e}")
            label.setText(f"显示图像失败: {str(e)}")
    
    def _set_pixmap_1_to_1(self, label, pixmap):
        try:
            if not pixmap.isNull():
                label.setPixmap(pixmap)
                label.resize(pixmap.size())
                label.setMinimumSize(pixmap.size())
        except Exception as e:
            self.logger.error(f"设置1:1图像失败: {e}")
    
    def _load_and_display_dicom_image(self, file_path):
        """加载并显示DICOM图像"""
        try:
            # 加载DICOM文件
            dicom_data = self.dicom_service.load_dicom_file(file_path)
            if not dicom_data:
                self.dicom_image_label.setText("无法加载DICOM文件")
                return
            
            # 转换为图像数组
            image_array = self.dicom_service.convert_to_image(dicom_data)
            if image_array is None:
                self.dicom_image_label.setText("无法转换DICOM为图像")
                return
            
            # 保存当前图像数组
            self.current_image_array = image_array
            
            # 显示图像
            self._display_dicom_image(image_array)
            
        except Exception as e:
            self.logger.error(f"加载并显示DICOM图像失败: {e}")
            self.dicom_image_label.setText(f"加载图像失败: {str(e)}")
    
    def _format_dicom_info(self, dicom_info):
        """格式化DICOM信息"""
        if not dicom_info:
            return "无DICOM信息"
        
        try:
            info_text = "DICOM文件信息:\n"
            info_text += "=" * 30 + "\n"
            
            # 基本信息
            info_text += f"患者姓名: {dicom_info.get('patient_name', '未知')}\n"
            info_text += f"患者ID: {dicom_info.get('patient_id', '未知')}\n"
            info_text += f"性别: {dicom_info.get('patient_sex', '未知')}\n" # 注意: extract_dicom_info中没有sex字段，需要添加
            
            # 检查是否有 study_info 嵌套结构 (兼容旧格式)
            if 'study_info' in dicom_info:
                study = dicom_info['study_info']
                info_text += f"检查日期: {study.get('date', '未知')}\n"
                info_text += f"检查描述: {study.get('description', '未知')}\n"
            else:
                # 新格式直接在根级别
                info_text += f"检查日期: {dicom_info.get('study_date', '未知')}\n"
                info_text += f"检查时间: {dicom_info.get('study_time', '未知')}\n"
                info_text += f"检查描述: {dicom_info.get('study_description', '未知')}\n"
                info_text += f"检查号: {dicom_info.get('accession_number', '未知')}\n"
            
            if 'series_info' in dicom_info:
                series = dicom_info['series_info']
                info_text += f"序列号: {series.get('number', '未知')}\n"
                info_text += f"序列描述: {series.get('description', '未知')}\n"
            else:
                info_text += f"序列号: {dicom_info.get('series_number', '未知')}\n"
                info_text += f"序列描述: {dicom_info.get('series_description', '未知')}\n"
                info_text += f"模态: {dicom_info.get('modality', '未知')}\n"
            
            # 图像信息
            if 'image_info' in dicom_info:
                image = dicom_info['image_info']
                width = image.get('width', 0)
                height = image.get('height', 0)
                pixel_spacing = image.get('pixel_spacing', '未知')
                slice_thickness = image.get('slice_thickness', '未知')
                window_width = image.get('window_width', '未知')
                window_center = image.get('window_center', '未知')
            else:
                width = dicom_info.get('columns', 0)
                height = dicom_info.get('rows', 0)
                pixel_spacing = dicom_info.get('pixel_spacing', '未知')
                slice_thickness = dicom_info.get('slice_thickness', '未知')
                window_width = dicom_info.get('window_width', '未知')
                window_center = dicom_info.get('window_center', '未知')
            
            info_text += f"图像尺寸: {width} x {height}\n"
            
            # 格式化像素间距
            if isinstance(pixel_spacing, list) and len(pixel_spacing) == 2:
                info_text += f"像素间距: {pixel_spacing[0]:.4f} x {pixel_spacing[1]:.4f} mm\n"
            else:
                info_text += f"像素间距: {pixel_spacing}\n"
                
            info_text += f"切片厚度: {slice_thickness} mm\n"
            info_text += f"窗宽/窗位: {window_width} / {window_center}\n"
            
            # 厂家信息
            manufacturer = dicom_info.get('manufacturer', '未知')
            model = dicom_info.get('manufacturer_model', '未知')
            if manufacturer != '未知':
                info_text += f"设备厂家: {manufacturer}\n"
            if model != '未知':
                info_text += f"设备型号: {model}\n"
            
            return info_text
            
        except Exception as e:
            self.logger.error(f"格式化DICOM信息失败: {e}")
            return f"格式化DICOM信息失败: {str(e)}"
    
    def _format_roi_results(self, roi_regions):
        """格式化ROI结果"""
        if not roi_regions:
            return "未检测到ROI区域"
        
        try:
            result_text = "ROI识别结果:\n"
            result_text += "=" * 30 + "\n"
            
            for i, roi in enumerate(roi_regions):
                result_text += f"ROI {i+1}:\n"
                
                if 'bbox' in roi:
                    bbox = roi['bbox']
                    result_text += f"  位置: ({bbox.get('x', 0)}, {bbox.get('y', 0)})"
                    result_text += f" 尺寸: {bbox.get('width', 0)} x {bbox.get('height', 0)}\n"
                
                if 'area' in roi:
                    result_text += f"  面积: {roi['area']:.2f} 像素\n"
                
                if 'ocr_result' in roi:
                    ocr_result = roi['ocr_result']
                    result_text += f"  OCR文本: {ocr_result.get('text', '无')}\n"
                    result_text += f"  OCR置信度: {ocr_result.get('confidence', 0):.2f}\n"
                
                result_text += "\n"
            
            return result_text
            
        except Exception as e:
            self.logger.error(f"格式化ROI结果失败: {e}")
            return f"格式化ROI结果失败: {str(e)}"
    
    def _setup_dicom_navigation(self, file_path):
        """设置DICOM导航功能"""
        try:
            # 这里可以添加DICOM导航相关的设置
            # 例如：加载DICOM序列、设置导航控件等
            self.logger.info(f"设置DICOM导航: {file_path}")
            
            # 如果需要，可以在这里添加更多的导航功能
            # 例如：加载同系列的DICOM文件、设置滑动条范围等
            
        except Exception as e:
            self.logger.error(f"设置DICOM导航失败: {e}")
    
    def closeEvent(self, event):
        """重写关闭事件，确保线程安全退出"""
        if self.processing_thread and self.processing_thread.isRunning():
            self.logger.info("正在停止处理线程...")
            self.processing_thread.stop()
            self.processing_thread.quit()
            self.processing_thread.wait()
            self.logger.info("处理线程已停止")
        event.accept()
